//===- EnzymeXLAOps.cpp - EnzymeXLA dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ops.h"
#include "Dialect.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

#define DEBUG_TYPE "enzymexla"

using namespace mlir;
using namespace enzymexla;
using namespace mlir::arith;

using namespace mlir::LLVM;
using namespace mlir::stablehlo;

static std::optional<int64_t> getConstant(Operation *op) {
  if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<LLVM::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return {};
}

static std::optional<int64_t> getConstant(Value v) {
  Operation *op = v.getDefiningOp();
  if (op)
    return getConstant(op);
  return {};
}

LogicalResult
TritonCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // tt.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

void TritonCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable TritonCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

Operation::operand_range TritonCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange TritonCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

LogicalResult
GPUOccupancyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

LogicalResult
GPUKernelAddressOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

LogicalResult
KernelCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

void KernelCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable KernelCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

Operation::operand_range KernelCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange KernelCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

static void addMemoryEffectsFromAttr(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
    ArrayAttr effectsAttr) {
  for (auto attr : effectsAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    assert(strAttr &&
           "enzymexla.memory_effects must be a ArrayAttr<StringAttr>");

    StringRef kind = strAttr.getValue();
    if (kind == "allocate")
      effects.emplace_back(MemoryEffects::Allocate::get());
    else if (kind == "free")
      effects.emplace_back(MemoryEffects::Free::get());
    else if (kind == "write")
      effects.emplace_back(MemoryEffects::Write::get());
    else if (kind == "read")
      effects.emplace_back(MemoryEffects::Read::get());
    else
      assert(false && "enzymexla.memory_effects has an invalid value");
  }
}

static void
addAllMemoryEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

void TritonCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ModuleOp moduleOp = (*this)->getParentOfType<ModuleOp>();
  assert(moduleOp && "TritonCallOp must be inside a ModuleOp");

  auto callee = moduleOp.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "TritonCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
}

void KernelCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ModuleOp moduleOp = (*this)->getParentOfType<ModuleOp>();
  assert(moduleOp && "KernelCallOp must be inside a ModuleOp");

  auto callee = moduleOp.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "KernelCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
}

LogicalResult JITCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

void JITCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable JITCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

MutableOperandRange JITCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

Operation::operand_range JITCallOp::getArgOperands() { return getInputs(); }

void JITCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  ModuleOp moduleOp = (*this)->getParentOfType<ModuleOp>();
  assert(moduleOp && "JITCallOp must be inside a ModuleOp");

  auto callee = moduleOp.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "JITCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
}

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
template <typename OpTy>
class ReadOnlyArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  OpTy create(PatternRewriter &rewriter, OpTy launchOp, ArrayRef<Type> resTys,
              ArrayAttr outputAliases) const;
  LogicalResult matchAndRewrite(OpTy launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(
        ((Operation *)launchOp)->getParentOfType<ModuleOp>());
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    auto operand_aliases = launchOp.getOutputOperandAliases();
    assert(operand_aliases.size() == launchOp.getNumResults());
    bool changed = false;
    size_t outputs = launchOp.getNumResults();
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<OutputOperandAliasAttr>(alias_attr);
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {

        changed = true;
        outputs--;
      }
    }
    if (!changed)
      return failure();
    SmallVector<Attribute> outputAliases;
    SmallVector<Type> resTys;
    size_t out_idx = 0;
    for (auto en : llvm::enumerate(operand_aliases)) {
      auto idx = en.index();
      auto alias = cast<OutputOperandAliasAttr>(en.value());
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      assert(launchOp.getInputs()[operandIndex].getType() ==
             launchOp.getResultTypes()[idx]);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {
        continue;
      }
      resTys.push_back(launchOp.getResultTypes()[idx]);
      if (outputs == 1) {
        outputAliases.push_back(OutputOperandAliasAttr::get(
            launchOp->getContext(), {}, operandIndex, {}));
      } else {
        outputAliases.push_back(OutputOperandAliasAttr::get(
            launchOp->getContext(), {(long)out_idx}, operandIndex, {}));
      }
      out_idx++;
    }

    auto newOp = create(rewriter, launchOp, resTys,
                        ArrayAttr::get(launchOp->getContext(), outputAliases));

    assert(outputAliases.size() == newOp.getNumResults());
    SmallVector<Value> replacements;
    out_idx = 0;
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<OutputOperandAliasAttr>(alias_attr);
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {
        replacements.push_back(launchOp.getInputs()[operandIndex]);
        continue;
      } else {
        replacements.push_back(newOp.getResult(out_idx));
        out_idx++;
      }
    }
    rewriter.replaceOp(launchOp, replacements);
    return success();
  }
};

template <>
enzymexla::KernelCallOp ReadOnlyArg<enzymexla::KernelCallOp>::create(
    PatternRewriter &rewriter, enzymexla::KernelCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return rewriter.create<enzymexla::KernelCallOp>(
      launchOp.getLoc(), resTys, launchOp.getFn(), launchOp.getGridx(),
      launchOp.getGridy(), launchOp.getGridz(), launchOp.getBlockx(),
      launchOp.getBlocky(), launchOp.getBlockz(), launchOp.getShmem(),
      launchOp.getClusterx(), launchOp.getClustery(), launchOp.getClusterz(),
      launchOp.getInputs(), launchOp.getBackendConfigAttr(),
      launchOp.getOperandLayoutsAttr(), /*resultLayouts*/ nullptr,
      launchOp.getArgAttrsAttr(), launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

template <>
enzymexla::JITCallOp ReadOnlyArg<enzymexla::JITCallOp>::create(
    PatternRewriter &rewriter, enzymexla::JITCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return rewriter.create<enzymexla::JITCallOp>(
      launchOp.getLoc(), resTys, launchOp.getFn(), launchOp.getInputs(),
      launchOp.getBackendConfigAttr(), launchOp.getOperandLayoutsAttr(),
      /*resultLayouts*/ nullptr, launchOp.getArgAttrsAttr(),
      launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

template <>
enzymexla::TritonCallOp ReadOnlyArg<enzymexla::TritonCallOp>::create(
    PatternRewriter &rewriter, enzymexla::TritonCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return rewriter.create<enzymexla::TritonCallOp>(
      launchOp.getLoc(), resTys, launchOp.getFn(), launchOp.getGridx(),
      launchOp.getGridy(), launchOp.getGridz(), launchOp.getShmem(),
      launchOp.getInputs(), launchOp.getBackendConfigAttr(),
      launchOp.getOperandLayoutsAttr(), /*resultLayouts*/ nullptr,
      launchOp.getArgAttrsAttr(), launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

template <typename OpTy>
class ReadNoneArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  void updateOperandSegmentSizes(OpTy call, int32_t numLiveOperands,
                                 PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(OpTy launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    auto mod = ((Operation *)launchOp)->getParentOfType<ModuleOp>();
    symbolTable.getSymbolTable(mod);
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    // Early error if no arg is read none
    {
      bool potentialReadNone = false;
      for (auto arg : fn.front().getArguments()) {
        bool readnone = arg.use_empty();
        if (!readnone)
          continue;
        potentialReadNone = true;
        break;
      }
      if (!potentialReadNone)
        return failure();
    }
    bool changed = false;

    SmallVector<OpTy> calls;
    auto use_opt = symbolTable.getSymbolTable(mod).getSymbolUses(fn, mod);
    if (!use_opt)
      return failure();
    for (auto u : *use_opt) {
      auto launch2 = dyn_cast<OpTy>(u.getUser());
      if (!launch2)
        return failure();
      calls.push_back(launch2);
      auto operand_aliases2 = launchOp.getOutputOperandAliases();
      (void)operand_aliases2;
      assert(operand_aliases2.size() == launchOp.getNumResults());
    }

    BitVector deadArgs(fn.front().getNumArguments(), false);
    for (auto arg : fn.front().getArguments()) {
      auto operandIndex = arg.getArgNumber();
      bool readnone = arg.use_empty();
      if (!readnone)
        continue;

      for (auto call : calls) {
        auto operand_aliases = call.getOutputOperandAliases();
        for (auto alias_attr : operand_aliases) {
          auto alias = cast<OutputOperandAliasAttr>(alias_attr);
          auto aliasOperandIndex = alias.getOperandIndex();
          if (aliasOperandIndex == operandIndex) {
            return failure();
          }
        }
      }
      changed = true;
      deadArgs[operandIndex] = true;
    }

    if (!changed)
      return failure();

    rewriter.modifyOpInPlace(fn, [&]() {
      // fn.eraseArguments(deadArgs);
      if (auto T = dyn_cast<LLVMFunctionType>(fn.getFunctionType())) {
        SmallVector<Type> argStorage;
        mlir::filterTypesOut(fn.getArgumentTypes(), deadArgs, argStorage);
        auto fty2 =
            LLVMFunctionType::get(T.getReturnType(), argStorage, T.getVarArg());
        mlir::function_interface_impl::eraseFunctionArguments(fn, deadArgs,
                                                              fty2);
      } else {
        (void)fn.eraseArguments(deadArgs);
      }
    });

    for (auto call : calls) {
      BitVector nonLiveCallOperands(call.getNumOperands(), false);
      for (int index : deadArgs.set_bits())
        nonLiveCallOperands.set(call.getInputs().getBeginOperandIndex() +
                                index);

      int32_t numLiveOperands = 0;
      for (int32_t idx = call.getInputs().getBeginOperandIndex();
           idx < nonLiveCallOperands.size(); idx++) {
        if (nonLiveCallOperands[idx])
          continue;
        numLiveOperands++;
      }

      SmallVector<Attribute> outputAliases;
      auto operand_aliases = call.getOutputOperandAliases();

      for (auto alias_attr : operand_aliases) {
        auto alias = cast<OutputOperandAliasAttr>(alias_attr);
        auto operandIndex = alias.getOperandIndex();
        size_t nextIndex = operandIndex;
        for (int index : deadArgs.set_bits()) {
          if (index <= operandIndex)
            nextIndex--;
        }
        outputAliases.push_back(OutputOperandAliasAttr::get(
            call->getContext(), alias.getOutputTupleIndices(), nextIndex,
            alias.getOperandTupleIndices()));
      }

      rewriter.modifyOpInPlace(call, [&]() {
        call->eraseOperands(nonLiveCallOperands);
        updateOperandSegmentSizes(call, numLiveOperands, rewriter);
        call.setOutputOperandAliasesAttr(
            ArrayAttr::get(call->getContext(), outputAliases));
      });
    }
    return success();
  }
};

template <>
void ReadNoneArg<KernelCallOp>::updateOperandSegmentSizes(
    KernelCallOp call, int32_t numLiveOperands,
    PatternRewriter &rewriter) const {
  call->setAttr("operandSegmentSizes",
                rewriter.getDenseI32ArrayAttr(
                    {1, 1, 1, 1, 1, 1, 1, call.getClusterx() ? 1 : 0,
                     call.getClustery() ? 1 : 0, call.getClusterz() ? 1 : 0,
                     numLiveOperands}));
}

template <>
void ReadNoneArg<JITCallOp>::updateOperandSegmentSizes(
    JITCallOp call, int32_t numLiveOperands, PatternRewriter &rewriter) const {}

void KernelCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ReadOnlyArg<KernelCallOp>, ReadNoneArg<KernelCallOp>>(context);
}

void JITCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<ReadOnlyArg<JITCallOp>, ReadNoneArg<JITCallOp>>(context);
}

void TritonCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ReadOnlyArg<TritonCallOp>, ReadNoneArg<TritonCallOp>>(context);
}

/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
class Memref2Pointer2MemrefCast final
    : public OpRewritePattern<Pointer2MemrefOp> {
public:
  using OpRewritePattern<Pointer2MemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Pointer2MemrefOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Memref2PointerOp>();
    if (!src)
      return failure();
    auto smt = cast<MemRefType>(src.getSource().getType());
    auto omt = cast<MemRefType>(op.getType());
    if (smt.getShape().size() != omt.getShape().size())
      return failure();
    for (unsigned i = 1; i < smt.getShape().size(); i++) {
      if (smt.getShape()[i] != omt.getShape()[i])
        return failure();
    }
    if (smt.getElementType() != omt.getElementType())
      return failure();
    if (smt.getMemorySpace() != omt.getMemorySpace())
      return failure();

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(),
                                                src.getSource());
    return success();
  }
};

#if 0
/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
class Memref2PointerIndex final : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<SubIndexOp>();
    if (!src)
      return failure();

    if (cast<MemRefType>(src.getSource().getType()).getShape().size() != 1)
      return failure();

    Value idx[] = {src.getIndex()};
    auto PET = cast<LLVM::LLVMPointerType>(op.getType()).getElementType();
    auto MET = cast<MemRefType>(src.getSource().getType()).getElementType();
    if (PET != MET) {
      Value ps;
      if (PET)
        // non-opaque pointer
        ps = rewriter.create<enzymexla::TypeSizeOp>(
            op.getLoc(), rewriter.getIndexType(), mlir::TypeAttr::get(PET));
      else
        // opaque pointer
        ps = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
      auto ms = rewriter.create<enzymexla::TypeSizeOp>(
          op.getLoc(), rewriter.getIndexType(), mlir::TypeAttr::get(MET));
      idx[0] = rewriter.create<MulIOp>(op.getLoc(), idx[0], ms);
      idx[0] = rewriter.create<DivUIOp>(op.getLoc(), idx[0], ps);
    }
    idx[0] = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                 rewriter.getI64Type(), idx[0]);
    if (PET)
      // non-opaque pointer
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, op.getType(),
          rewriter.create<Memref2PointerOp>(op.getLoc(), op.getType(),
                                            src.getSource()),
          idx);
    else
      // opaque pointer
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, op.getType(), rewriter.getI8Type(),
          rewriter.create<Memref2PointerOp>(op.getLoc(), op.getType(),
                                            src.getSource()),
          idx);
    return success();
  }
};
#endif

/// Simplify memref2pointer(pointer2memref(x)) to cast(x)
class Memref2PointerBitCast final : public OpRewritePattern<LLVM::BitcastOp> {
public:
  using OpRewritePattern<LLVM::BitcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::BitcastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getOperand().getDefiningOp<Memref2PointerOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<Memref2PointerOp>(op, op.getType(),
                                                  src.getOperand());
    return success();
  }
};

/// Simplify pointer2memref(memref2pointer(x)) to cast(x)
template <typename T>
class CopySimplification final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {

    Value dstv = op.getDst();
    auto dst = dstv.getDefiningOp<enzymexla::Memref2PointerOp>();
    if (!dst)
      return failure();

    auto dstTy = cast<MemRefType>(dst.getSource().getType());

    Value srcv = op.getSrc();
    auto src = srcv.getDefiningOp<enzymexla::Memref2PointerOp>();
    if (!src)
      return failure();
    auto srcTy = cast<MemRefType>(src.getSource().getType());
    if (srcTy.getShape().size() != dstTy.getShape().size())
      return failure();

    if (dstTy.getElementType() != srcTy.getElementType())
      return failure();
    Type elTy = dstTy.getElementType();

    size_t width = 1;
    if (auto IT = dyn_cast<IntegerType>(elTy))
      width = IT.getWidth() / 8;
    else if (auto FT = dyn_cast<FloatType>(elTy))
      width = FT.getWidth() / 8;
    else {
      // TODO extend to llvm compatible type
      return failure();
    }
    bool first = true;
    SmallVector<size_t> bounds;
    for (auto pair : llvm::zip(dstTy.getShape(), srcTy.getShape())) {
      if (first) {
        first = false;
        continue;
      }
      if (std::get<0>(pair) != std::get<1>(pair))
        return failure();
      bounds.push_back(std::get<0>(pair));
      width *= std::get<0>(pair);
    }

    SmallVector<Value> todo = {op.getLen()};
    size_t factor = 1;
    while (factor % width != 0 && todo.size()) {
      Value len = todo.back();
      todo.pop_back();
      IntegerAttr constValue;
      if (auto ext = len.getDefiningOp<arith::ExtUIOp>())
        todo.push_back(ext.getIn());
      else if (auto ext = len.getDefiningOp<arith::ExtSIOp>())
        todo.push_back(ext.getIn());
      else if (auto ext = len.getDefiningOp<arith::TruncIOp>()) {
        if (APInt(64, width).isPowerOf2() &&
            ext.getType().getIntOrFloatBitWidth() >
                APInt(64, width).nearestLogBase2())
          todo.push_back(ext.getIn());
      } else if (auto ext = len.getDefiningOp<arith::IndexCastOp>())
        todo.push_back(ext.getIn());
      else if (auto mul = len.getDefiningOp<arith::MulIOp>()) {
        todo.push_back(mul.getLhs());
        todo.push_back(mul.getRhs());
      } else if (matchPattern(len, m_Constant(&constValue))) {
        factor *= constValue.getValue().getLimitedValue();
      } else
        continue;
    }

    if (factor % width != 0)
      return failure();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> idxs;
    auto forOp = rewriter.create<scf::ForOp>(
        op.getLoc(), c0,
        rewriter.create<arith::DivUIOp>(
            op.getLoc(),
            rewriter.create<arith::IndexCastOp>(
                op.getLoc(), rewriter.getIndexType(), op.getLen()),
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), width)),
        c1);

    rewriter.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
    idxs.push_back(forOp.getInductionVar());

    for (auto bound : bounds) {
      auto forOp = rewriter.create<scf::ForOp>(
          op.getLoc(), c0, rewriter.create<ConstantIndexOp>(op.getLoc(), bound),
          c1);
      rewriter.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
      idxs.push_back(forOp.getInductionVar());
    }

    rewriter.create<memref::StoreOp>(
        op.getLoc(),
        rewriter.create<memref::LoadOp>(op.getLoc(), src.getSource(), idxs),
        dst.getSource(), idxs);

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename T>
class SetSimplification final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {

    Value dstv = op.getDst();
    auto dst = dstv.getDefiningOp<enzymexla::Memref2PointerOp>();
    if (!dst)
      return failure();

    auto dstTy = cast<MemRefType>(dst.getSource().getType());
    Type elTy = dstTy.getElementType();

    if (!isa<IntegerType, FloatType>(elTy))
      return failure();

    size_t width = 1;
    if (auto IT = dyn_cast<IntegerType>(elTy))
      width = IT.getWidth() / 8;
    else if (auto FT = dyn_cast<FloatType>(elTy))
      width = FT.getWidth() / 8;
    else {
      // TODO extend to llvm compatible type
      return failure();
    }
    bool first = true;
    SmallVector<size_t> bounds;
    for (auto pair : dstTy.getShape()) {
      if (first) {
        first = false;
        continue;
      }
      bounds.push_back(pair);
      width *= pair;
    }

    SmallVector<Value> todo = {op.getLen()};
    size_t factor = 1;
    while (factor % width != 0 && todo.size()) {
      Value len = todo.back();
      todo.pop_back();
      IntegerAttr constValue;
      if (auto ext = len.getDefiningOp<arith::ExtUIOp>())
        todo.push_back(ext.getIn());
      else if (auto ext = len.getDefiningOp<arith::ExtSIOp>())
        todo.push_back(ext.getIn());
      else if (auto ext = len.getDefiningOp<arith::TruncIOp>()) {
        if (APInt(64, width).isPowerOf2() &&
            ext.getType().getIntOrFloatBitWidth() >
                APInt(64, width).nearestLogBase2())
          todo.push_back(ext.getIn());
      } else if (auto ext = len.getDefiningOp<arith::IndexCastOp>())
        todo.push_back(ext.getIn());
      else if (auto mul = len.getDefiningOp<arith::MulIOp>()) {
        todo.push_back(mul.getLhs());
        todo.push_back(mul.getRhs());
      } else if (matchPattern(len, m_Constant(&constValue))) {
        factor *= constValue.getValue().getLimitedValue();
      } else
        continue;
    }

    if (factor % width != 0)
      return failure();

    Value c0 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> idxs;
    Value val = cast<mlir::enzyme::AutoDiffTypeInterface>(elTy).createNullValue(
        rewriter, op.getLoc());

    auto forOp = rewriter.create<scf::ForOp>(
        op.getLoc(), c0,
        rewriter.create<arith::DivUIOp>(
            op.getLoc(),
            rewriter.create<arith::IndexCastOp>(
                op.getLoc(), rewriter.getIndexType(), op.getLen()),
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), width)),
        c1);

    rewriter.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
    idxs.push_back(forOp.getInductionVar());

    for (auto bound : bounds) {
      auto forOp = rewriter.create<scf::ForOp>(
          op.getLoc(), c0, rewriter.create<ConstantIndexOp>(op.getLoc(), bound),
          c1);
      rewriter.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
      idxs.push_back(forOp.getInductionVar());
    }

    rewriter.create<memref::StoreOp>(op.getLoc(), val, dst.getSource(), idxs);

    rewriter.eraseOp(op);
    return success();
  }
};

void Memref2PointerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<
      // Memref2Pointer2MemrefCast, Memref2PointerIndex,
      // Memref2PointerBitCast,
      Memref2Pointer2MemrefCast, Memref2PointerBitCast,

      SetSimplification<LLVM::MemsetOp>, CopySimplification<LLVM::MemcpyOp>,
      CopySimplification<LLVM::MemmoveOp>>(context);
}

OpFoldResult Memref2PointerOp::fold(FoldAdaptor adaptor) {
#if 0
  if (auto subindex = getSource().getDefiningOp<SubIndexOp>()) {
    if (auto cop = subindex.getIndex().getDefiningOp<ConstantIndexOp>()) {
      if (cop.getValue() == 0) {
        getSourceMutable().assign(subindex.getSource());
        return getResult();
      }
    }
  }
#endif
  /// Simplify memref2pointer(cast(x)) to memref2pointer(x)
  if (auto mc = getSource().getDefiningOp<memref::CastOp>()) {
    getSourceMutable().assign(mc.getSource());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
    if (mc.getSource().getType() == getType()) {
      return mc.getSource();
    }
  }
  return nullptr;
}

/// Simplify cast(pointer2memref(x)) to pointer2memref(x)
class Pointer2MemrefCast final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<enzymexla::Pointer2MemrefOp>(op, op.getType(),
                                                             src.getSource());
    return success();
  }
};

/// Simplify memref2pointer(pointer2memref(x)) to cast(x)
class Pointer2Memref2PointerCast final
    : public OpRewritePattern<Memref2PointerOp> {
public:
  using OpRewritePattern<Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<Pointer2MemrefOp>();
    if (!src)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getType(),
                                                 src.getSource());
    return success();
  }
};

/// Simplify load(pointer2memref(gep(...(x)))) to load(x, idx)
template <typename T>
class LoadStorePointer2MemrefGEP final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  SmallVector<Value> newIndex(T op, Value toAdd,
                              PatternRewriter &rewriter) const;

  void createNewOp(T op, Value baseMemref, SmallVector<Value> vals,
                   PatternRewriter &rewriter) const;

  Value getMemref(T op) const;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // FIXME: Only handle memref.load with single index for now
    if (op.getIndices().size() != 1)
      return failure();

    // Match pointer2memref -> load pattern
    auto src =
        getMemref(op).template getDefiningOp<enzymexla::Pointer2MemrefOp>();
    if (!src)
      return failure();

    // Get the element type and size of the final memref
    Type elementType = op.getMemRefType().getElementType();
    unsigned elementSize = elementType.isIntOrFloat()
                               ? elementType.getIntOrFloatBitWidth() / 8
                               : 0;
    if (elementSize == 0)
      return failure();

    // Collect all GEPs in the chain
    SmallVector<std::pair<LLVM::GEPOp, unsigned>> gepOps;
    Value ptr = src.getSource();

    while (auto gep = ptr.getDefiningOp<LLVM::GEPOp>()) {
      // FIXME: Only handle GEPs with single index for now
      if (gep.getIndices().size() != 1)
        return failure();

      // Get element type size in bytes
      unsigned gepElemSize = 1;
      auto elemTy = gep.getElemType();
      if (elemTy.isIntOrFloat()) {
        gepElemSize = elemTy.getIntOrFloatBitWidth() / 8;
      } else {
        // Unknown type to get size from, bail early.
        break;
      }

      gepOps.emplace_back(gep, gepElemSize);
      ptr = gep.getBase();
    }

    if (gepOps.empty())
      return failure();

    Location loc = op.getLoc();
    auto baseMemref = rewriter.create<Pointer2MemrefOp>(
        loc, cast<MemRefType>(src.getType()), ptr);

    // Start with the original load offset
    Value finalIndex = nullptr;
    // Process GEPs in reverse order
    for (auto [gep, gepElemSize] : llvm::reverse(gepOps)) {
      PointerUnion<IntegerAttr, Value> rawIdx = gep.getIndices()[0];
      Value idx = dyn_cast_if_present<Value>(rawIdx);
      if (!idx)
        idx = rewriter.create<arith::ConstantIndexOp>(
            loc, cast<IntegerAttr>(rawIdx).getValue().getSExtValue());
      // TODO: verify the total byte offset will be element-aligned for dynamic
      // indices by inserting runtime check
      if (auto constIdx = idx.getDefiningOp<arith::ConstantIndexOp>()) {
        // For constant indices, static check and reject unaligned access
        if ((constIdx.value() * gepElemSize) % elementSize != 0) {
          return failure();
        }
      }

      // Convert index to the right type if needed
      if (!idx.getType().isIndex()) {
        idx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                  idx);
      }

      // Calculate byte offset: idx * gepElemSize / elementSize
      unsigned gcd = std::gcd(gepElemSize, elementSize);
      unsigned scaledGep = gepElemSize / gcd;
      unsigned scaledElement = elementSize / gcd;

      // Multiply first if needed
      Value scaledIdx =
          (scaledGep != 1)
              ? rewriter.create<arith::MulIOp>(
                    loc, idx,
                    rewriter.create<arith::ConstantIndexOp>(loc, scaledGep))
              : idx;

      // Then divide if needed
      Value elemOffset =
          (scaledElement != 1)
              ? rewriter.create<arith::DivSIOp>(
                    loc, scaledIdx,
                    rewriter.create<arith::ConstantIndexOp>(loc, scaledElement))
              : scaledIdx;

      // Add to total offset
      if (finalIndex)
        finalIndex =
            rewriter.create<arith::AddIOp>(loc, finalIndex, elemOffset);
      else
        finalIndex = elemOffset;
    }

    // Replace the load with a direct load from the base memref
    createNewOp(op, baseMemref, newIndex(op, finalIndex, rewriter), rewriter);
    return success();
  }
};

template <>
Value LoadStorePointer2MemrefGEP<memref::LoadOp>::getMemref(
    memref::LoadOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<memref::StoreOp>::getMemref(
    memref::StoreOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::getMemref(
    affine::AffineLoadOp op) const {
  return op.getMemref();
}

template <>
Value LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::getMemref(
    affine::AffineStoreOp op) const {
  return op.getMemref();
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<memref::LoadOp>::newIndex(
    memref::LoadOp op, Value finalIndex, PatternRewriter &rewriter) const {
  auto operands = llvm::to_vector(op.getIndices());
  operands[0] =
      rewriter.create<arith::AddIOp>(op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::newIndex(
    affine::AffineLoadOp op, Value finalIndex,
    PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<affine::AffineApplyOp>(
      op.getLoc(), op.getAffineMap(), op.getMapOperands());

  SmallVector<Value> operands;
  for (auto op : apply->getResults())
    operands.push_back(op);
  operands[0] =
      rewriter.create<arith::AddIOp>(op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<memref::StoreOp>::newIndex(
    memref::StoreOp op, Value finalIndex, PatternRewriter &rewriter) const {
  auto operands = llvm::to_vector(op.getIndices());
  operands[0] =
      rewriter.create<arith::AddIOp>(op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
SmallVector<Value> LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::newIndex(
    affine::AffineStoreOp op, Value finalIndex,
    PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<affine::AffineApplyOp>(
      op.getLoc(), op.getAffineMap(), op.getMapOperands());

  SmallVector<Value> operands;
  for (auto op : apply->getResults())
    operands.push_back(op);
  operands[0] =
      rewriter.create<arith::AddIOp>(op.getLoc(), operands[0], finalIndex);
  return operands;
}

template <>
void LoadStorePointer2MemrefGEP<memref::LoadOp>::createNewOp(
    memref::LoadOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, baseMemref, idxs);
}

template <>
void LoadStorePointer2MemrefGEP<affine::AffineLoadOp>::createNewOp(
    affine::AffineLoadOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, baseMemref, idxs);
}

template <>
void LoadStorePointer2MemrefGEP<memref::StoreOp>::createNewOp(
    memref::StoreOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), baseMemref,
                                               idxs);
}

template <>
void LoadStorePointer2MemrefGEP<affine::AffineStoreOp>::createNewOp(
    affine::AffineStoreOp op, Value baseMemref, SmallVector<Value> idxs,
    PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(), baseMemref,
                                               idxs);
}

/// Simplify load (pointer2memref(x)) to llvm.load x
template <typename Op>
class MetaPointer2Memref final : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  Value computeIndex(Op op, size_t idx, PatternRewriter &rewriter) const;

  void rewriteInternal(Op op, Value ptr, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Value opPtr = op.getMemref();
    Pointer2MemrefOp src = opPtr.getDefiningOp<enzymexla::Pointer2MemrefOp>();
    if (!src)
      return failure();

    auto mt = cast<MemRefType>(src.getType());

    // Fantastic optimization, disabled for now to make a hard debug case
    // easier to find.
    if (auto before =
            src.getSource().getDefiningOp<enzymexla::Memref2PointerOp>()) {
      auto mt0 = cast<MemRefType>(before.getSource().getType());
      if (mt0.getElementType() == mt.getElementType()) {
        auto sh0 = mt0.getShape();
        auto sh = mt.getShape();
        if (sh.size() == sh0.size()) {
          bool eq = true;
          for (size_t i = 1; i < sh.size(); i++) {
            if (sh[i] != sh0[i]) {
              eq = false;
              break;
            }
          }
          if (eq) {
            op.getMemrefMutable().assign(before.getSource());
            return success();
          }
        }
      }
    }

    for (size_t i = 1; i < mt.getShape().size(); i++)
      if (mt.getShape()[i] == ShapedType::kDynamic)
        return failure();

    Value val = src.getSource();

    Value idx = nullptr;
    auto shape = mt.getShape();
    for (size_t i = 0; i < shape.size(); i++) {
      auto off = computeIndex(op, i, rewriter);
      auto cur = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), off);
      if (idx == nullptr) {
        idx = cur;
      } else {
        idx = rewriter.create<AddIOp>(
            op.getLoc(),
            rewriter.create<MulIOp>(op.getLoc(), idx,
                                    rewriter.create<arith::ConstantIntOp>(
                                        op.getLoc(), shape[i], 32)),
            cur);
      }
    }

    if (idx) {
      Value idxs[] = {idx};
      val = rewriter.create<LLVM::GEPOp>(op.getLoc(), val.getType(),
                                         mt.getElementType(), val, idxs);
    }
    rewriteInternal(op, val, rewriter);
    return success();
  }
};

template <>
Value MetaPointer2Memref<memref::LoadOp>::computeIndex(
    memref::LoadOp op, size_t i, PatternRewriter &rewriter) const {
  return op.getIndices()[i];
}

template <>
void MetaPointer2Memref<memref::LoadOp>::rewriteInternal(
    memref::LoadOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr);
}

template <>
Value MetaPointer2Memref<memref::StoreOp>::computeIndex(
    memref::StoreOp op, size_t i, PatternRewriter &rewriter) const {
  return op.getIndices()[i];
}

template <>
void MetaPointer2Memref<memref::StoreOp>::rewriteInternal(
    memref::StoreOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getValue(), ptr);
}

template <>
Value MetaPointer2Memref<affine::AffineLoadOp>::computeIndex(
    affine::AffineLoadOp op, size_t i, PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<affine::AffineApplyOp>(
      op.getLoc(), map.getSliceMap(i, 1), op.getMapOperands());
  return apply->getResult(0);
}

template <>
void MetaPointer2Memref<affine::AffineLoadOp>::rewriteInternal(
    affine::AffineLoadOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, op.getType(), ptr);
}

template <>
Value MetaPointer2Memref<affine::AffineStoreOp>::computeIndex(
    affine::AffineStoreOp op, size_t i, PatternRewriter &rewriter) const {
  auto map = op.getAffineMap();
  auto apply = rewriter.create<affine::AffineApplyOp>(
      op.getLoc(), map.getSliceMap(i, 1), op.getMapOperands());
  return apply->getResult(0);
}

template <>
void MetaPointer2Memref<affine::AffineStoreOp>::rewriteInternal(
    affine::AffineStoreOp op, Value ptr, PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, op.getValue(), ptr);
}

void Pointer2MemrefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<Pointer2MemrefCast, Pointer2Memref2PointerCast,
                 LoadStorePointer2MemrefGEP<memref::LoadOp>,
                 LoadStorePointer2MemrefGEP<affine::AffineLoadOp>,
                 LoadStorePointer2MemrefGEP<memref::StoreOp>,
                 LoadStorePointer2MemrefGEP<affine::AffineStoreOp>>(context);
  /*
  results.insert<Pointer2MemrefCast, Pointer2Memref2PointerCast,
                 MetaPointer2Memref<memref::LoadOp>,
                 MetaPointer2Memref<memref::StoreOp>,
                 MetaPointer2Memref<affine::AffineLoadOp>,
                 MetaPointer2Memref<affine::AffineStoreOp>>(context);
                 */
}

OpFoldResult Pointer2MemrefOp::fold(FoldAdaptor adaptor) {
  /// Simplify pointer2memref(cast(x)) to pointer2memref(x)
  if (auto mc = getSource().getDefiningOp<LLVM::BitcastOp>()) {
    getSourceMutable().assign(mc.getArg());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<LLVM::AddrSpaceCastOp>()) {
    getSourceMutable().assign(mc.getArg());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<LLVM::GEPOp>()) {
    for (auto idx : mc.getDynamicIndices()) {
      assert(idx);
      if (!matchPattern(idx, m_Zero()))
        return nullptr;
    }
    auto staticIndices = mc.getRawConstantIndices();
    for (auto pair : llvm::enumerate(staticIndices)) {
      if (pair.value() != LLVM::GEPOp::kDynamicIndex)
        if (pair.value() != 0)
          return nullptr;
    }

    getSourceMutable().assign(mc.getBase());
    return getResult();
  }
  if (auto mc = getSource().getDefiningOp<enzymexla::Memref2PointerOp>()) {
    if (mc.getSource().getType() == getType()) {
      return mc.getSource();
    }
  }
  return nullptr;
}

LogicalResult WrapOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  WrapOpAdaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getLhs() < 0)
    return failure();
  if (adaptor.getRhs() < 0)
    return failure();
  if (adaptor.getDimension() < 0)
    return failure();
  auto RT = cast<RankedTensorType>(adaptor.getOperand().getType());
  if (adaptor.getDimension() >= RT.getShape().size())
    return failure();

  SmallVector<int64_t> resShape = llvm::to_vector(RT.getShape());
  if (resShape[adaptor.getDimension()] != -1)
    resShape[adaptor.getDimension()] += adaptor.getLhs() + adaptor.getRhs();
  inferredReturnTypes.push_back(
      RankedTensorType::get(resShape, RT.getElementType()));
  return success();
}

LogicalResult ExtendOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  ExtendOpAdaptor adaptor(operands, attributes, properties, regions);
  if (adaptor.getLhs() < 0)
    return failure();
  if (adaptor.getRhs() < 0)
    return failure();
  if (adaptor.getDimension() < 0)
    return failure();
  auto RT = cast<RankedTensorType>(adaptor.getOperand().getType());
  if (adaptor.getDimension() >= RT.getShape().size())
    return failure();

  SmallVector<int64_t> resShape = llvm::to_vector(RT.getShape());
  if (resShape[adaptor.getDimension()] != -1)
    resShape[adaptor.getDimension()] += adaptor.getLhs() + adaptor.getRhs();
  inferredReturnTypes.push_back(
      RankedTensorType::get(resShape, RT.getElementType()));
  return success();
}

void CommRegionOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(RegionSuccessor(getResults()));
}

LogicalResult enzymexla::MemcpyOp::verify() {
  auto srcType = getSource().getType();
  auto dstType = getTarget().getType();

  if (getElementTypeOrSelf(srcType) != getElementTypeOrSelf(dstType))
    return emitOpError("arguments have incompatible element type");

  return success();
}

namespace {

/// Erases a common case of copy ops where a destination value is used only by
/// the copy op, alloc and dealloc ops.
struct EraseTrivialCopyOp : public OpRewritePattern<enzymexla::MemcpyOp> {
  using OpRewritePattern<enzymexla::MemcpyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    Value dest = op.getTarget();
    Operation *destDefOp = dest.getDefiningOp();
    // `dest` must be defined by an op having Allocate memory effect in order to
    // perform the folding.
    if (!destDefOp ||
        !hasSingleEffect<MemoryEffects::Allocate>(destDefOp, dest))
      return failure();
    // We can erase `op` iff `dest` has no other use apart from its
    // use by `op` and dealloc ops.
    if (llvm::any_of(dest.getUsers(), [op, dest](Operation *user) {
          return user != op &&
                 !hasSingleEffect<MemoryEffects::Free>(user, dest);
        }))
      return failure();
    // We can perform the folding if and only if op has a single async
    // dependency and produces an async token as result, or if it does not have
    // any async dependency and does not produce any async token result.
    if (op.getAsyncDependencies().size() > 1 ||
        ((op.getAsyncDependencies().empty() && op.getAsyncToken()) ||
         (!op.getAsyncDependencies().empty() && !op.getAsyncToken())))
      return failure();
    rewriter.replaceOp(op, op.getAsyncDependencies());
    return success();
  }
};

struct CopyWithTypes : public OpRewritePattern<enzymexla::MemcpyOp> {
  using OpRewritePattern<enzymexla::MemcpyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    Value vals[2];
    MemRefType tys[2];
    for (int i = 0; i < 2; i++) {
      auto v = op->getOperand(i);
      if (auto p2m = v.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
        if (auto m2p =
                p2m.getSource().getDefiningOp<enzymexla::Memref2PointerOp>()) {
          if (p2m.getType().getMemorySpace() ==
              m2p.getSource().getType().getMemorySpace())
            v = m2p.getSource();
        }
      }
      vals[i] = v;
      tys[i] = cast<MemRefType>(v.getType());
    }

    MemRefType finalType = tys[0];

    if (tys[0].getElementType() != tys[1].getElementType()) {
      if (tys[0].getElementType().isInteger(8)) {
        finalType = tys[1];
      } else if (tys[1].getElementType().isInteger(8)) {
        finalType = tys[0];
      } else {
        return failure();
      }
    }

    if (finalType.getElementType() == op.getTarget().getType().getElementType())
      return failure();

    DataLayoutAnalysis dataLayoutAnalysis(op);
    auto &dataLayout = dataLayoutAnalysis.getAtOrAbove(op);
    int64_t elNum =
        dataLayout.getTypeSize(op.getTarget().getType().getElementType());

    Value sz = op.getSize();
    APInt copySize;
    if (matchPattern(sz, m_ConstantInt(&copySize))) {
      elNum *= (copySize.getSExtValue());
    } else {
      size_t num = 1;
      size_t den = 1;
      Value op = sz;
      while (true) {
        if (auto icast = op.getDefiningOp<arith::IndexCastOp>()) {
          op = icast.getOperand();
          continue;
        }
        if (auto icast = op.getDefiningOp<arith::IndexCastUIOp>()) {
          op = icast.getOperand();
          continue;
        }
        if (auto shr = op.getDefiningOp<arith::ShRSIOp>()) {
          if (auto cst = getConstant(shr.getRhs())) {
            auto val = 1ULL << *cst;
            if (num % val == 0) {
              num /= val;
              op = shr.getLhs();
              continue;
            } else if (val != 0 && val % num == 0) {
              den *= (val / num);
              num = 1;
              op = shr.getLhs();
              continue;
            }
          }
        }
        if (auto shr = op.getDefiningOp<arith::ShRUIOp>()) {
          if (auto cst = getConstant(shr.getRhs())) {
            auto val = 1ULL << *cst;
            if (num % val == 0) {
              num /= val;
              op = shr.getLhs();
              continue;
            } else if (val != 0 && val % num == 0) {
              den *= (val / num);
              num = 1;
              op = shr.getLhs();
              continue;
            }
          }
        }
        if (auto shl = op.getDefiningOp<arith::ShLIOp>()) {
          if (auto cst = getConstant(shl.getRhs())) {
            auto val = 1ULL << *cst;
            if (den % val == 0) {
              den /= val;
              op = shl.getLhs();
              continue;
            } else if (val != 0 && val % den == 0) {
              num *= (val / den);
              den = 1;
              op = shl.getLhs();
              continue;
            }
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "could not deduce size of copy due to " << op
                                << " num=" << num << " den=" << den << "\n");
        break;
      }
      assert(den == 1);
      if (den == 1) {
        elNum *= num;
      } else {
        return failure();
      }
    }

    auto newElSize = dataLayout.getTypeSize(finalType.getElementType());

    int64_t newElnum = elNum / newElSize;
    if (newElSize * newElnum != elNum) {
      LLVM_DEBUG(llvm::dbgs()
                 << "non divisible size: newElSize " << newElSize << " elNum "
                 << elNum << " newElnum: " << newElnum << "\n");
      return failure();
    }

    SmallVector<int64_t, 1> sizes = {newElnum};
    for (int i = 0; i < 2; i++) {
      auto MT = cast<MemRefType>(vals[i].getType());
      if (MT.getElementType() == finalType.getElementType())
        continue;
      vals[i] = rewriter.create<enzymexla::Memref2PointerOp>(
          op.getLoc(),
          LLVM::LLVMPointerType::get(vals[i].getContext(),
                                     MT.getMemorySpaceAsInt()),
          vals[i]);
      auto shape2 = llvm::to_vector(MT.getShape());
      if (shape2.size() > 0)
        shape2[shape2.size() - 1] = ShapedType::kDynamic;
      vals[i] = rewriter.create<enzymexla::Pointer2MemrefOp>(
          op.getLoc(),
          MemRefType::get(shape2, finalType.getElementType(), MT.getLayout(),
                          MT.getMemorySpace()),
          vals[i]);
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.getTargetMutable().set(vals[0]);
      op.getSourceMutable().set(vals[1]);
    });
    return success();
  }
};

} // end anonymous namespace

void enzymexla::MemcpyOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<EraseTrivialCopyOp, CopyWithTypes>(context);
}

LogicalResult
enzymexla::MemcpyOp::fold(FoldAdaptor adaptor,
                          SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

using namespace mlir::enzyme;
llvm::cl::opt<bool> BarrierOpt("barrier-opt", llvm::cl::init(true),
                               llvm::cl::desc("Optimize barriers"));

class BarrierHoist final : public OpRewritePattern<BarrierOp> {
public:
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    if (!BarrierOpt)
      return failure();
    if (isa<scf::IfOp, affine::AffineIfOp>(barrier->getParentOp())) {

      bool below = true;
      for (Operation *it = barrier->getNextNode(); it != nullptr;
           it = it->getNextNode()) {
        if (!isReadNone(it)) {
          below = false;
          break;
        }
      }
      if (below) {
        rewriter.setInsertionPoint(barrier->getParentOp()->getNextNode());
        rewriter.create<BarrierOp>(barrier.getLoc(), barrier.getOperands());
        rewriter.eraseOp(barrier);
        return success();
      }
      bool above = true;
      for (Operation *it = barrier->getPrevNode(); it != nullptr;
           it = it->getPrevNode()) {
        if (!isReadNone(it)) {
          above = false;
          break;
        }
      }
      if (above) {
        rewriter.setInsertionPoint(barrier->getParentOp());
        rewriter.create<BarrierOp>(barrier.getLoc(), barrier.getOperands());
        rewriter.eraseOp(barrier);
        return success();
      }
    }
    // Move barrier into after region and after loop, if possible
    if (auto whileOp = dyn_cast<scf::WhileOp>(barrier->getParentOp())) {
      if (barrier->getParentRegion() == &whileOp.getBefore()) {
        auto cond = whileOp.getBefore().front().getTerminator();

        bool above = true;
        for (Operation *it = cond; it != nullptr; it = it->getPrevNode()) {
          if (it == barrier)
            break;
          if (!isReadNone(it)) {
            above = false;
            break;
          }
        }
        if (above) {
          rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
          rewriter.create<BarrierOp>(barrier.getLoc(), barrier.getOperands());
          rewriter.setInsertionPoint(whileOp->getNextNode());
          rewriter.create<BarrierOp>(barrier.getLoc(), barrier.getOperands());
          rewriter.eraseOp(barrier);
          return success();
        }
      }
    }
    return failure();
  }
};

void BarrierOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {

  // If this doesn't synchronize any values, it has no effects.
  if (llvm::all_of(getOperands(), [](Value v) {
        IntegerAttr constValue;
        return matchPattern(v, m_Constant(&constValue));
      }))
    return;

  Operation *op = getOperation();

  if (!getEffectsBefore(op, effects, /*stopAtBarrier*/ true))
    return;

  if (!getEffectsAfter(op, effects, /*stopAtBarrier*/ true))
    return;
}

void BarrierOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<BarrierHoist>(context);
  // results.insert<BarrierHoist, BarrierElim</*TopLevelOnly*/ false>>(context);
}

void GPUWrapperOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange blockSizes) {
  result.addTypes(builder.getIndexType());
  result.addOperands(blockSizes);
  OpBuilder::InsertionGuard g(builder);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  GPUWrapperOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void GPUWrapperOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIndexType());
  OpBuilder::InsertionGuard g(builder);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  GPUWrapperOp::ensureTerminator(*bodyRegion, builder, result.location);
}

LogicalResult fixupGetFunc(LLVM::CallOp op, OpBuilder &rewriter,
                           SmallVectorImpl<Value> &vals) {
  if (op.getCallee())
    return failure();

  Value pval = op.getOperand(0);

  auto FT = op.getCalleeFunctionType();

  if (FT.isVarArg())
    return failure();

  while (true) {
    if (auto bc = pval.getDefiningOp<LLVM::BitcastOp>())
      pval = bc.getOperand();
    else if (auto mt = pval.getDefiningOp<Memref2PointerOp>())
      pval = mt.getOperand();
    else if (auto mt = pval.getDefiningOp<Pointer2MemrefOp>())
      pval = mt.getOperand();
    else
      break;
  }

  return failure();
#if 0
  auto gfn = pval.getDefiningOp<GetFuncOp>();
  if (!gfn)
    return failure();

  LLVM::LLVMFunctionType FT2;
  if (auto fn =
          gfn->getParentOfType<ModuleOp>().lookupSymbol(gfn.getNameAttr())) {
    if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(fn))
      FT2 = funcOp.getFunctionType();
    else if (auto funcOp = dyn_cast<func::FuncOp>(fn))
      FT2 = LLVM::LLVMFunctionType::get(
          rewriter.getContext(),
          op.getResultTypes().empty()
              ? LLVM::LLVMVoidType::get(rewriter.getContext())
              : funcOp.getResultTypes().front(),
          funcOp.getArgumentTypes(), /*isVarArg=*/false);
    else
      return failure();
  } else {
    return failure();
  }

  if (FT2.getParams().size() != FT.getParams().size())
    return failure();

  SmallVector<Value> args(op.getArgOperands());
  for (unsigned i = 0; i < args.size(); i++) {
    if (FT2.getParams()[i] != args[i].getType()) {
      if (!FT2.getParams()[i].isa<MemRefType>() ||
          !args[i].getType().isa<LLVM::LLVMPointerType>())
        return failure();
      args[i] = rewriter.create<polygeist::Pointer2MemrefOp>(
          op.getLoc(), FT2.getParams()[i], args[i]);
    }
  }

  if (op.getResultTypes().size() &&
      (!op.getResultTypes()[0].isa<LLVM::LLVMPointerType>() ||
       !FT2.getReturnType().isa<MemRefType>()))
    return failure();

  auto res = rewriter
                 .create<func::CallOp>(op.getLoc(), gfn.getNameAttr(),
                                       op.getResultTypes(), args)
                 .getResults();
  for (Value r : res) {
    if (r.getType() != FT.getReturnType())
      r = rewriter.create<polygeist::Memref2PointerOp>(op.getLoc(),
                                                       FT.getReturnType(), r);
    vals.push_back(r);
  }
  return success();
#endif
}

struct NoopResource : public SideEffects::Resource::Base<NoopResource> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NoopResource)

  StringRef getName() final { return "<NoopResource>"; }
};

void NoopOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange indices) {
  result.addOperands(indices);
}

void NoopOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // TODO CHECK is it okay to ::get() a new resource every time?
  SideEffects::Resource *resource = NoopResource::get();
  MemoryEffects::Effect *effect =
      MemoryEffects::Effect::get<MemoryEffects::Write>();
  effects.emplace_back(effect, resource);
}

void GPUErrorOp::build(OpBuilder &builder, OperationState &result) {
  result.addTypes(builder.getIndexType());
  OpBuilder::InsertionGuard g(builder);
  Region *bodyRegion = result.addRegion();
  builder.createBlock(bodyRegion);
  GPUErrorOp::ensureTerminator(*bodyRegion, builder, result.location);
}

LogicalResult
XLAWrapperOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

void XLAWrapperOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable XLAWrapperOp::getCallableForCallee() { return getFn(); }

MutableOperandRange XLAWrapperOp::getArgOperandsMutable() {
  return getInputsMutable();
}

Operation::operand_range XLAWrapperOp::getArgOperands() { return getInputs(); }

void XLAWrapperOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
}

//===----------------------------------------------------------------------===//
// AlternativesOp
//===----------------------------------------------------------------------===//

void AlternativesOp::build(OpBuilder &builder, OperationState &result,
                           int regionNum) {
  OpBuilder::InsertionGuard g(builder);
  for (int i = 0; i < regionNum; i++) {
    Region *bodyRegion = result.addRegion();
    Block *block = builder.createBlock(bodyRegion);
    builder.setInsertionPointToEnd(block);
    builder.create<PolygeistYieldOp>(result.location);
  }
}

class HoistSingleAlternative final : public OpRewritePattern<AlternativesOp> {
public:
  using OpRewritePattern<AlternativesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AlternativesOp aop,
                                PatternRewriter &rewriter) const override {
    assert(aop->getNumRegions() > 0);
    if (aop->getNumRegions() > 1) {
      return failure();
    }
    auto block = &*aop->getRegions()[0].begin();
    rewriter.eraseOp(block->getTerminator());
    rewriter.inlineBlockBefore(block, aop);
    rewriter.eraseOp(aop);
    return success();
  }
};

class FlattenAlternatives final : public OpRewritePattern<AlternativesOp> {
public:
  using OpRewritePattern<AlternativesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AlternativesOp aop,
                                PatternRewriter &rewriter) const override {
    // Ignore nested alternatives ops
    if (aop->getParentOfType<AlternativesOp>())
      return failure();

    AlternativesOp innerAop = nullptr;
    unsigned regionId = 0;
    for (auto &region : aop->getRegions()) {
      for (auto &op : region.getOps()) {
        if (auto aop = dyn_cast<AlternativesOp>(&op)) {
          innerAop = aop;
          break;
        }
      }
      if (innerAop)
        break;
      regionId++;
    }
    if (!innerAop)
      return failure();

    // TODO use block insertion etc for better performance
    auto newAop = rewriter.create<enzymexla::AlternativesOp>(
        aop->getLoc(), innerAop->getNumRegions() + aop->getNumRegions() - 1);
    newAop->setAttrs(aop->getAttrs());
    auto outerDescs = aop->getAttrOfType<ArrayAttr>("alternatives.descs");
    auto innerDescs = innerAop->getAttrOfType<ArrayAttr>("alternatives.descs");
    std::vector<Attribute> configs;
    unsigned curRegion = 0;
    for (; curRegion < innerAop->getNumRegions(); curRegion++) {
      IRMapping mapping;
      auto block = &*newAop->getRegion(curRegion).begin();
      rewriter.setInsertionPointToStart(block);
      for (auto &op : *innerAop->getBlock()) {
        if (&op == innerAop.getOperation()) {
          for (auto &op : innerAop->getRegion(curRegion).getOps())
            if (!isa<PolygeistYieldOp>(&op))
              rewriter.clone(op, mapping);
        } else {
          if (!isa<PolygeistYieldOp>(&op))
            rewriter.clone(op, mapping);
        }
      }
      configs.push_back(rewriter.getStringAttr(
          cast<StringAttr>(outerDescs[regionId]).str() +
          cast<StringAttr>(innerDescs[curRegion]).str()));
    }

    unsigned oldRegion = 0;
    for (; oldRegion < aop->getNumRegions(); oldRegion++) {
      auto &srcRegion = aop->getRegion(oldRegion);
      if (innerAop->getBlock()->getParent() == &srcRegion) {
        assert(oldRegion == regionId);
        continue;
      }
      auto block = &*newAop->getRegion(curRegion).begin();
      rewriter.setInsertionPointToStart(block);
      IRMapping mapping;
      for (auto &op : srcRegion.getOps())
        if (!isa<PolygeistYieldOp>(&op))
          rewriter.clone(op, mapping);
      configs.push_back(rewriter.getStringAttr(
          cast<StringAttr>(outerDescs[oldRegion]).str()));
      curRegion++;
    }
    newAop->setAttr("alternatives.descs", rewriter.getArrayAttr(configs));

    rewriter.eraseOp(aop);

    return success();
  }
};

void AlternativesOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<HoistSingleAlternative, FlattenAlternatives>(context);
}

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
class CastOfSubIndex final : public OpRewritePattern<memref::CastOp> {
public:
  using OpRewritePattern<memref::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto subindexOp = castOp.getSource().getDefiningOp<SubIndexOp>();
    if (!subindexOp)
      return failure();

    if (cast<MemRefType>(castOp.getType()).getShape().size() !=
        cast<MemRefType>(subindexOp.getType()).getShape().size())
      return failure();
    if (cast<MemRefType>(castOp.getType()).getElementType() !=
        cast<MemRefType>(subindexOp.getResult().getType()).getElementType())
      return failure();

    rewriter.replaceOpWithNewOp<SubIndexOp>(castOp, castOp.getType(),
                                            subindexOp.getSource(),
                                            subindexOp.getIndex());
    return success();
  }
};

// Replace subindex(subindex(x)) with subindex(x) with appropriate
// indexing.
class SubIndex2 final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subViewOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = subViewOp.getSource().getDefiningOp<SubIndexOp>();
    if (!prevOp)
      return failure();

    auto mt0 = cast<MemRefType>(prevOp.getSource().getType());
    auto mt1 = cast<MemRefType>(prevOp.getType());
    auto mt2 = cast<MemRefType>(subViewOp.getType());
    if (mt0.getShape().size() == mt2.getShape().size() &&
        mt1.getShape().size() == mt0.getShape().size() + 1) {
      rewriter.replaceOpWithNewOp<SubIndexOp>(
          subViewOp, mt2, prevOp.getSource(), subViewOp.getIndex());
      return success();
    }
    if (mt0.getShape().size() == mt2.getShape().size() &&
        mt1.getShape().size() == mt0.getShape().size()) {
      rewriter.replaceOpWithNewOp<SubIndexOp>(
          subViewOp, mt2, prevOp.getSource(),
          rewriter.create<AddIOp>(prevOp.getLoc(), subViewOp.getIndex(),
                                  prevOp.getIndex()));
      return success();
    }
    return failure();
  }
};

// When possible, simplify subindex(x) to cast(x)
class SubToCast final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subViewOp,
                                PatternRewriter &rewriter) const override {
    auto prev = cast<MemRefType>(subViewOp.getSource().getType());
    auto post = cast<MemRefType>(subViewOp.getType());
    bool legal = prev.getShape().size() == post.getShape().size();
    if (legal) {

      auto cidx = subViewOp.getIndex().getDefiningOp<ConstantIndexOp>();
      if (!cidx)
        return failure();

      if (cidx.getValue() != 0)
        return failure();

      rewriter.replaceOpWithNewOp<memref::CastOp>(subViewOp, post,
                                                  subViewOp.getSource());
      return success();
    }

    return failure();
  }
};

// Simplify enzymexla.subindex to memref.subview.
class SubToSubView final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcMemRefType = cast<MemRefType>(op.getSource().getType());
    auto resMemRefType = cast<MemRefType>(op.getResult().getType());
    auto dims = srcMemRefType.getShape().size();

    // For now, restrict subview lowering to statically defined memref's
    if (!srcMemRefType.hasStaticShape() | !resMemRefType.hasStaticShape())
      return failure();

    // For now, restrict to simple rank-reducing indexing
    if (srcMemRefType.getShape().size() <= resMemRefType.getShape().size())
      return failure();

    // Build offset, sizes and strides
    SmallVector<OpFoldResult> sizes(dims, rewriter.getIndexAttr(0));
    sizes[0] = op.getIndex();
    SmallVector<OpFoldResult> offsets(dims);
    for (auto dim : llvm::enumerate(srcMemRefType.getShape())) {
      if (dim.index() == 0)
        offsets[0] = rewriter.getIndexAttr(1);
      else
        offsets[dim.index()] = rewriter.getIndexAttr(dim.value());
    }
    SmallVector<OpFoldResult> strides(dims, rewriter.getIndexAttr(1));

    // Generate the appropriate return type:
    auto subMemRefType = MemRefType::get(srcMemRefType.getShape().drop_front(),
                                         srcMemRefType.getElementType());

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op, subMemRefType, op.getSource(), sizes, offsets, strides);

    return success();
  }
};

// Simplify redundant dynamic subindex patterns which tries to represent
// rank-reducing indexing:
//   %3 = "enzymexla.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<?x1000xi32> %4 = "enzymexla.subindex"(%3, %c0) :
//   (memref<?x1000xi32>, index) -> memref<1000xi32>
// simplifies to:
//   %4 = "enzymexla.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<1000xi32>

class RedundantDynSubIndex final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcOp = op.getSource().getDefiningOp<SubIndexOp>();
    if (!srcOp)
      return failure();

    auto preMemRefType = cast<MemRefType>(srcOp.getSource().getType());
    auto srcMemRefType = cast<MemRefType>(op.getSource().getType());
    auto resMemRefType = cast<MemRefType>(op.getResult().getType());

    // Check that this is indeed a rank reducing operation
    if (srcMemRefType.getShape().size() !=
        (resMemRefType.getShape().size() + 1))
      return failure();

    // Check that the previous op is the same rank.
    if (srcMemRefType.getShape().size() != preMemRefType.getShape().size())
      return failure();

    // Valid optimization target; perform the substitution.
    rewriter.replaceOpWithNewOp<SubIndexOp>(
        op, op.getResult().getType(), srcOp.getSource(),
        rewriter.create<arith::AddIOp>(op.getLoc(), op.getIndex(),
                                       srcOp.getIndex()));
    return success();
  }
};
/// Simplify all uses of subindex, specifically
//    store subindex(x) = ...
//    affine.store subindex(x) = ...
//    load subindex(x)
//    affine.load subindex(x)
//    dealloc subindex(x)
struct SimplifySubIndexUsers : public OpRewritePattern<SubIndexOp> {
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp subindex,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    for (OpOperand &use : llvm::make_early_inc_range(subindex->getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        changed = true;
        rewriter.replaceOpWithNewOp<memref::DeallocOp>(dealloc,
                                                       subindex.getSource());
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
        if (loadOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = loadOp.getIndices();

          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            assert(cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                   cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size());
            indices.insert(indices.begin(), subindex.getIndex());
          }

          assert(cast<MemRefType>(subindex.getSource().getType())
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(
              loadOp, subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            assert(cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                   cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size());
            indices.insert(indices.begin(), subindex.getIndex());
          }
          assert(cast<MemRefType>(subindex.getSource().getType())
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.getValue(), subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::AtomicRMWOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            assert(cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                   cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size());
            indices.insert(indices.begin(), subindex.getIndex());
          }
          assert(cast<MemRefType>(subindex.getSource().getType())
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(
              storeOp, storeOp.getType(), storeOp.getKind(), storeOp.getValue(),
              subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.getIndex());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.getValue(), subindex.getSource(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineLoadOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.getIndex());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::LoadOp>(
                storeOp, subindex.getSource(), indices);
            changed = true;
          }
        }
      }
    }

    return success(changed);
  }
};

struct SimplifySubViewUsers : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subindex,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    int64_t offs = -1;
    for (auto tup :
         llvm::zip(subindex.getStaticOffsets(), subindex.getStaticSizes(),
                   subindex.getStaticStrides())) {
      auto sz = std::get<1>(tup);

      auto stride = std::get<2>(tup);
      if (stride != 1)
        return failure();

      if (offs == -1) {
        offs = std::get<0>(tup);
        if (sz != 1)
          return failure();
      }
    }
    Value off = rewriter.create<ConstantIndexOp>(subindex.getLoc(), offs);
    assert(off);

    for (OpOperand &use : llvm::make_early_inc_range(subindex->getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      if (auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner())) {
        changed = true;
        rewriter.replaceOpWithNewOp<memref::DeallocOp>(dealloc,
                                                       subindex.getSource());
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(use.getOwner())) {
        if (loadOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = loadOp.getIndices();
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
            if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                cast<MemRefType>(subindex.getSource().getType())
                    .getShape()
                    .size())
              indices.insert(indices.begin(), off);
            else {
              assert(indices.size() > 0);
              indices.erase(indices.begin());
            }
          }

          assert(cast<MemRefType>(subindex.getSource().getType())
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(
              loadOp, subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
            if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                cast<MemRefType>(subindex.getSource().getType())
                    .getShape()
                    .size())
              indices.insert(indices.begin(), off);
            else {
              if (indices.size() == 0) {
                llvm::errs() << " storeOp: " << storeOp
                             << " - subidx: " << subindex << "\n";
              }
              assert(indices.size() > 0);
              indices.erase(indices.begin());
            }
          }

          if (cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size() != indices.size()) {
            llvm::errs() << " storeOp: " << storeOp << " - subidx: " << subindex
                         << "\n";
          }
          assert(cast<MemRefType>(subindex.getSource().getType())
                     .getShape()
                     .size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.getValue(), subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.getValue(), subindex.getSource(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineLoadOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType())
                  .getShape()
                  .size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            assert(cast<MemRefType>(subindex.getSource().getType())
                       .getShape()
                       .size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::LoadOp>(
                storeOp, subindex.getSource(), indices);
            changed = true;
          }
        }
      }
    }

    return success(changed);
  }
};

/// Simplify select cast(x), cast(y) to cast(select x, y)
struct SelectOfCast : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cst1 = op.getTrueValue().getDefiningOp<memref::CastOp>();
    if (!cst1)
      return failure();

    auto cst2 = op.getFalseValue().getDefiningOp<memref::CastOp>();
    if (!cst2)
      return failure();

    if (cst1.getSource().getType() != cst2.getSource().getType())
      return failure();

    auto newSel = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), cst1.getSource(), cst2.getSource());

    rewriter.replaceOpWithNewOp<memref::CastOp>(op, op.getType(), newSel);
    return success();
  }
};

/// Simplify select subindex(x), subindex(y) to subindex(select x, y)
struct SelectOfSubIndex : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern<mlir::arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto cst1 = op.getTrueValue().getDefiningOp<SubIndexOp>();
    if (!cst1)
      return failure();

    auto cst2 = op.getFalseValue().getDefiningOp<SubIndexOp>();
    if (!cst2)
      return failure();

    if (cst1.getSource().getType() != cst2.getSource().getType())
      return failure();

    auto newSel = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), cst1.getSource(), cst2.getSource());
    auto newIdx = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), cst1.getIndex(), cst2.getIndex());
    rewriter.replaceOpWithNewOp<SubIndexOp>(op, op.getType(), newSel, newIdx);
    return success();
  }
};

OpFoldResult TypeAlignOp::fold(FoldAdaptor adaptor) {
  Type T = getSourceAttr().getValue();
  if (isa<IntegerType, FloatType>(T) || LLVM::isCompatibleType(T)) {
    DataLayout DLI(((Operation *)*this)->getParentOfType<ModuleOp>());
    return IntegerAttr::get(getResult().getType(),
                            APInt(64, DLI.getTypeABIAlignment(T)));
  }
  return nullptr;
}

/// Given an operation, return whether this op is guaranteed to
/// allocate an AutomaticAllocationScopeResource
static bool isGuaranteedAutomaticAllocation(Operation *op) {
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return false;
  for (auto res : op->getResults()) {
    if (auto effect =
            interface.getEffectOnValue<MemoryEffects::Allocate>(res)) {
      if (isa<SideEffects::AutomaticAllocationScopeResource>(
              effect->getResource()))
        return true;
    }
  }
  return false;
}

template <typename T>
struct AlwaysAllocaScopeHoister : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T top,
                                PatternRewriter &rewriter) const override {

    Operation *op = top;
    if (!op->getParentWithTrait<OpTrait::AutomaticAllocationScope>())
      return failure();

    Operation *lastParentWithoutScope =
        op->hasTrait<OpTrait::AutomaticAllocationScope>() ? op
                                                          : op->getParentOp();

    if (!lastParentWithoutScope)
      return failure();

    while (!lastParentWithoutScope->getParentOp()
                ->hasTrait<OpTrait::AutomaticAllocationScope>()) {
      lastParentWithoutScope = lastParentWithoutScope->getParentOp();
      if (!lastParentWithoutScope)
        return failure();
    }
    assert(lastParentWithoutScope->getParentOp()
               ->hasTrait<OpTrait::AutomaticAllocationScope>());

    Region *containingRegion = nullptr;
    if (lastParentWithoutScope == op)
      containingRegion = &op->getRegion(0);
    for (auto &r : lastParentWithoutScope->getRegions()) {
      if (r.isAncestor(op->getParentRegion())) {
        assert(containingRegion == nullptr &&
               "only one region can contain the op");
        containingRegion = &r;
      }
    }
    assert(containingRegion && "op must be contained in a region");

    SetVector<Operation *> toHoist;

    op->walk<WalkOrder::PreOrder>([&](Operation *alloc) {
      if (alloc != op && alloc->hasTrait<OpTrait::AutomaticAllocationScope>())
        return WalkResult::skip();

      if (!isGuaranteedAutomaticAllocation(alloc))
        return WalkResult::advance();

      SetVector<Operation *> subHoist;
      std::function<bool(Value)> fix = [&](Value v) -> /*legal*/ bool {
        if (!containingRegion->isAncestor(v.getParentRegion()))
          return true;
        auto *op = v.getDefiningOp();
        if (toHoist.count(op))
          return true;
        if (subHoist.count(op))
          return true;
        if (!op)
          return false;
        if (!isReadNone(op))
          return false;
        for (auto o : op->getOperands()) {
          if (!fix(o))
            return false;
        }
        subHoist.insert(op);
        return true;
      };

      // If any operand is not defined before the location of
      // lastParentWithoutScope (i.e. where we would hoist to), skip.
      if (llvm::any_of(alloc->getOperands(), [&](Value v) { return !fix(v); }))
        return WalkResult::skip();
      for (auto s : subHoist)
        toHoist.insert(s);
      toHoist.insert(alloc);
      return WalkResult::advance();
    });

    if (toHoist.empty())
      return failure();
    rewriter.setInsertionPoint(lastParentWithoutScope);
    IRMapping map;
    for (auto *op : toHoist) {
      auto *cloned = rewriter.clone(*op, map);
      rewriter.replaceOp(op, cloned->getResults());
    }
    return success();
  }
};

void TypeAlignOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<AlwaysAllocaScopeHoister<memref::AllocaScopeOp>,
                 AlwaysAllocaScopeHoister<scf::ForOp>,
                 AlwaysAllocaScopeHoister<affine::AffineForOp>>(context);
}

/// Simplify select subindex(x), subindex(y) to subindex(select x, y)
template <typename T> struct LoadSelect : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  static Value ptr(T op);
  static MutableOperandRange ptrMutable(T op);

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto mem0 = ptr(op);
    mlir::arith::SelectOp mem =
        dyn_cast_or_null<mlir::arith::SelectOp>(mem0.getDefiningOp());
    if (!mem)
      return failure();

    Type tys[] = {op.getType()};
    auto iop = rewriter.create<scf::IfOp>(mem.getLoc(), tys, mem.getCondition(),
                                          /*hasElse*/ true);

    auto vop = cast<T>(op->clone());
    iop.thenBlock()->push_front(vop);
    ptrMutable(vop).assign(mem.getTrueValue());
    rewriter.setInsertionPointToEnd(iop.thenBlock());
    rewriter.create<scf::YieldOp>(op.getLoc(), vop->getResults());

    auto eop = cast<T>(op->clone());
    iop.elseBlock()->push_front(eop);
    ptrMutable(eop).assign(mem.getFalseValue());
    rewriter.setInsertionPointToEnd(iop.elseBlock());
    rewriter.create<scf::YieldOp>(op.getLoc(), eop->getResults());

    rewriter.replaceOp(op, iop.getResults());
    return success();
  }
};

template <> Value LoadSelect<memref::LoadOp>::ptr(memref::LoadOp op) {
  return op.getMemref();
}
template <>
MutableOperandRange LoadSelect<memref::LoadOp>::ptrMutable(memref::LoadOp op) {
  return op.getMemrefMutable();
}
template <>
Value LoadSelect<affine::AffineLoadOp>::ptr(affine::AffineLoadOp op) {
  return op.getMemref();
}
template <>
MutableOperandRange
LoadSelect<affine::AffineLoadOp>::ptrMutable(affine::AffineLoadOp op) {
  return op.getMemrefMutable();
}
template <> Value LoadSelect<LLVM::LoadOp>::ptr(LLVM::LoadOp op) {
  return op.getAddr();
}
template <>
MutableOperandRange LoadSelect<LLVM::LoadOp>::ptrMutable(LLVM::LoadOp op) {
  return op.getAddrMutable();
}

OpFoldResult SubIndexOp::fold(FoldAdaptor adaptor) {
  if (getResult().getType() == getSource().getType()) {
    if (matchPattern(getIndex(), m_Zero()))
      return getSource();
  }
  /// Replace subindex(cast(x)) with subindex(x)
  if (auto castOp = getSource().getDefiningOp<memref::CastOp>()) {
    if (cast<MemRefType>(castOp.getType()).getElementType() ==
        cast<MemRefType>(getResult().getType()).getElementType()) {
      getSourceMutable().assign(castOp.getSource());
      return getResult();
    }
  }
  return nullptr;
}

void SubIndexOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<CastOfSubIndex, SubIndex2, SubToCast, SimplifySubViewUsers,
                 SimplifySubIndexUsers, SelectOfCast, SelectOfSubIndex,
                 RedundantDynSubIndex, LoadSelect<memref::LoadOp>,
                 LoadSelect<affine::AffineLoadOp>, LoadSelect<LLVM::LoadOp>>(
      context);
  // Disabled: SubToSubView
}