//===- JITCallToJITRegion.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/SetVector.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "jit-call-to-jit-region"

namespace mlir::enzyme {
#define GEN_PASS_DEF_JITCALLTOJITREGIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

struct JITCallToJITRegionPass
    : public mlir::enzyme::impl::JITCallToJITRegionPassBase<
          JITCallToJITRegionPass> {
  using mlir::enzyme::impl::JITCallToJITRegionPassBase<
      JITCallToJITRegionPass>::JITCallToJITRegionPassBase;

  // The type of the buffer holding `tensor` that the function argument `arg`
  // can be bound to, or null if there is none.
  static MemRefType getBufferType(Type tensor, Type arg) {
    auto T = dyn_cast<RankedTensorType>(tensor);
    if (!T)
      return nullptr;
    if (auto MT = dyn_cast<MemRefType>(arg))
      return MT.getShape() == T.getShape() &&
                     MT.getElementType() == T.getElementType()
                 ? MT
                 : nullptr;
    // A pointer is the address of a buffer in its address space.
    if (auto PT = dyn_cast<LLVM::LLVMPointerType>(arg)) {
      Attribute memorySpace;
      if (unsigned addressSpace = PT.getAddressSpace())
        memorySpace = IntegerAttr::get(IntegerType::get(arg.getContext(), 64),
                                       addressSpace);
      return MemRefType::get(T.getShape(), T.getElementType(),
                             /* layout= */ MemRefLayoutAttrInterface{},
                             memorySpace);
    }
    return nullptr;
  }

  // Whether nothing outside the module can call `fn`.
  static bool isPrivate(FunctionOpInterface fn) {
    if (auto llvmFn = dyn_cast<LLVM::LLVMFuncOp>(fn.getOperation()))
      return llvmFn.getLinkage() == LLVM::Linkage::Private;
    return fn.isPrivate();
  }

  // The function `call` calls, if the call can be replaced by a jit_region.
  static FunctionOpInterface getInlinableCallee(enzymexla::JITCallOp call,
                                                SymbolTableCollection &tables) {
    auto fn = dyn_cast_or_null<FunctionOpInterface>(
        tables.lookupNearestSymbolFrom(call, call.getFnAttr()));
    if (!fn || fn.isExternal() || !llvm::hasSingleElement(fn.getFunctionBody()))
      return nullptr;
    if (!call.getBackendConfig().empty() || !fn.getResultTypes().empty() ||
        fn.getNumArguments() != call.getInputs().size())
      return nullptr;

    Operation *terminator = fn.getFunctionBody().front().getTerminator();
    if (!terminator || !terminator->hasTrait<OpTrait::ReturnLike>() ||
        terminator->getNumOperands() != 0)
      return nullptr;

    for (auto [input, arg] :
         llvm::zip_equal(call.getInputs(), fn.getArgumentTypes()))
      if (!getBufferType(input.getType(), arg))
        return nullptr;

    // Each result is read from the buffer of a distinct input.
    ArrayAttr aliases = call.getOutputOperandAliases();
    if (aliases.size() != call.getNumResults())
      return nullptr;
    llvm::SmallDenseSet<int64_t> aliased;
    for (Attribute attr : aliases) {
      auto alias = dyn_cast<stablehlo::OutputOperandAliasAttr>(attr);
      if (!alias || !alias.getOperandTupleIndices().empty() ||
          !aliased.insert(alias.getOperandIndex()).second)
        return nullptr;
    }
    return fn;
  }

  static void inlineJITCall(enzymexla::JITCallOp call, FunctionOpInterface fn) {
    SmallVector<Type> bufferTypes;
    for (auto [input, arg] :
         llvm::zip_equal(call.getInputs(), fn.getArgumentTypes()))
      bufferTypes.push_back(getBufferType(input.getType(), arg));

    OpBuilder builder(call);
    auto region = enzymexla::JITRegionOp::create(
        builder, call.getLoc(), call.getResultTypes(), call.getInputs());
    region.setOutputOperandAliasesAttr(call.getOutputOperandAliases());
    if (Attribute layouts = call.getOperandLayoutsAttr())
      region.setOperandLayoutsAttr(layouts);
    if (ArrayAttr argAttrs = call.getArgAttrsAttr())
      region.setOperandAttrsAttr(argAttrs);
    if (ArrayAttr resAttrs = call.getResAttrsAttr())
      region.setResultAttrsAttr(resAttrs);
    if (call.getXlaSideEffectFree())
      region.setXlaSideEffectFree(true);

    Block *body = builder.createBlock(
        &region.getBody(), {}, bufferTypes,
        SmallVector<Location>(bufferTypes.size(), call.getLoc()));
    auto yield = enzymexla::JITRegionYieldOp::create(builder, call.getLoc());
    builder.setInsertionPoint(yield);

    IRMapping mapping;
    for (auto [arg, buffer] :
         llvm::zip_equal(fn.getArguments(), body->getArguments())) {
      if (isa<LLVM::LLVMPointerType>(arg.getType()))
        mapping.map(arg, enzymexla::Memref2PointerOp::create(
                             builder, arg.getLoc(), arg.getType(), buffer));
      else
        mapping.map(arg, buffer);
    }
    for (Operation &op : fn.getFunctionBody().front().without_terminator())
      builder.clone(op, mapping);

    call.replaceAllUsesWith(region.getResults());
    call.erase();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbolTables;

    SmallVector<enzymexla::JITCallOp> calls;
    module.walk([&](enzymexla::JITCallOp call) { calls.push_back(call); });

    llvm::SetVector<FunctionOpInterface> callees;
    for (enzymexla::JITCallOp call : calls) {
      FunctionOpInterface fn = getInlinableCallee(call, symbolTables);
      if (!fn)
        continue;
      inlineJITCall(call, fn);
      callees.insert(fn);
    }

    for (FunctionOpInterface fn : callees)
      if (isPrivate(fn) && SymbolTable::symbolKnownUseEmpty(fn, module))
        fn.erase();
  }
};

} // namespace
