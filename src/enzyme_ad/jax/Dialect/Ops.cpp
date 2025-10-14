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

template <typename OpTy>
class ReadNoneArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

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
        call.setOutputOperandAliasesAttr(
            ArrayAttr::get(call->getContext(), outputAliases));
      });
    }
    return success();
  }
};

void KernelCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ReadOnlyArg<KernelCallOp>, ReadNoneArg<KernelCallOp>>(context);
}

void JITCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<ReadOnlyArg<JITCallOp>, ReadNoneArg<JITCallOp>>(context);
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

bool valueCmp_Polygeist(Cmp_Polygeist cmp, Value bval, ValueOrInt_Polygeist val) {
  if (auto icast = bval.getDefiningOp<IndexCastOp>()) {
    return valueCmp_Polygeist(cmp, icast.getIn(), val);
  }
  if (auto icast = bval.getDefiningOp<IndexCastUIOp>()) {
    return valueCmp_Polygeist(cmp, icast.getIn(), val);
  }

  IntegerAttr iattr;
  if (matchPattern(bval, m_Constant(&iattr))) {
    switch (cmp) {
    case Cmp_Polygeist::EQ:
      return val == iattr.getValue();
    case Cmp_Polygeist::LT:
      return val > iattr.getValue();
    case Cmp_Polygeist::LE:
      return val >= iattr.getValue();
    case Cmp_Polygeist::GT:
      return val < iattr.getValue();
    case Cmp_Polygeist::GE:
      return val <= iattr.getValue();
    }
  }

  if (cmp == Cmp_Polygeist::GE && !val.isValue && val.i_val == 0) {
    if (auto baval = bval.getDefiningOp<arith::AddIOp>()) {
      return valueCmp_Polygeist(cmp, baval.getLhs(), val) &&
             valueCmp_Polygeist(cmp, baval.getRhs(), val);
    }
    if (auto baval = bval.getDefiningOp<arith::MulIOp>()) {
      APInt ival;
      if (matchPattern(baval.getRhs(), m_ConstantInt(&ival))) {
        if (ival == 0)
          return true;
        if (ival.isStrictlyPositive())
          return valueCmp_Polygeist(cmp, baval.getLhs(), val);
        else
          return valueCmp_Polygeist(Cmp_Polygeist::LE, baval.getLhs(), val);
      }
    }
    if (auto baval = bval.getDefiningOp<arith::ShRUIOp>()) {
      return valueCmp_Polygeist(cmp, baval.getLhs(), val);
    }
    if (auto baval = bval.getDefiningOp<arith::ShLIOp>()) {
      return valueCmp_Polygeist(cmp, baval.getLhs(), val);
    }
    if (auto baval = bval.getDefiningOp<arith::DivUIOp>()) {
      return valueCmp_Polygeist(cmp, baval.getLhs(), val);
    }
  }

  if (auto baval = dyn_cast<BlockArgument>(bval)) {
    if (affine::AffineForOp afFor =
            dyn_cast<affine::AffineForOp>(baval.getOwner()->getParentOp())) {
      auto for_lb = afFor.getLowerBoundMap().getResults()[baval.getArgNumber()];
      auto for_ub = afFor.getUpperBoundMap().getResults()[baval.getArgNumber()];
      switch (cmp) {
      // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
      case Cmp_Polygeist::EQ: {
        if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_lb, afFor.getLowerBoundMap().getNumDims(),
                      afFor.getLowerBoundOperands(), val))
          return false;
        if (!val.isValue) {
          if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_ub, afFor.getUpperBoundMap().getNumDims(),
                        afFor.getUpperBoundOperands(), val.i_val + 1))
            return false;
          return true;
        }
        return false;
      }
      // \forall i \in [LB, UB) < k   => UB <= k
      case Cmp_Polygeist::LT: {
        return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, afFor.getUpperBoundMap().getNumDims(),
                        afFor.getUpperBoundOperands(), val);
      }
      // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
      case Cmp_Polygeist::LE: {
        if (!val.isValue) {
          return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub,
                          afFor.getUpperBoundMap().getNumDims(),
                          afFor.getUpperBoundOperands(), val.i_val + 1);
        }
        return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, afFor.getUpperBoundMap().getNumDims(),
                        afFor.getUpperBoundOperands(), val);
      }
      // \forall i \in [LB, UB) > k   => LB > k
      case Cmp_Polygeist::GT: {
        return valueCmp_Polygeist(Cmp_Polygeist::GT, for_lb, afFor.getLowerBoundMap().getNumDims(),
                        afFor.getLowerBoundOperands(), val);
      }
      // \forall i \in [LB, UB) >= k   => LB >= k
      case Cmp_Polygeist::GE: {
        return valueCmp_Polygeist(Cmp_Polygeist::GE, for_lb, afFor.getLowerBoundMap().getNumDims(),
                        afFor.getLowerBoundOperands(), val);
      }
      }
    }
    if (affine::AffineParallelOp afFor = dyn_cast<affine::AffineParallelOp>(
            baval.getOwner()->getParentOp())) {
      switch (cmp) {
      // \forall i \in [max(LB...), min(UB...)) == k   => all(LB == k) and
      // all(UB == k+1)
      case Cmp_Polygeist::EQ: {
        for (auto for_lb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_lb, afFor.getLowerBoundsMap().getNumDims(),
                        afFor.getLowerBoundsOperands(), val))
            return false;
        if (!val.isValue) {
          for (auto for_ub :
               afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
            if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_ub,
                          afFor.getUpperBoundsMap().getNumDims(),
                          afFor.getUpperBoundsOperands(), val.i_val + 1))
              return false;
          return true;
        }
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) < k   => any(UB <= k)
      case Cmp_Polygeist::LT: {
        for (auto for_ub :
             afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, afFor.getUpperBoundsMap().getNumDims(),
                       afFor.getUpperBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) <= k   => any(UB-1 <= k)  =>
      // any(UB <= k+1)
      case Cmp_Polygeist::LE: {
        if (!val.isValue) {
          for (auto for_ub :
               afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
            if (valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub,
                         afFor.getUpperBoundsMap().getNumDims(),
                         afFor.getUpperBoundsOperands(), val.i_val + 1))
              return true;
          return false;
        }

        for (auto for_ub :
             afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, afFor.getUpperBoundsMap().getNumDims(),
                       afFor.getUpperBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) > k   => any(LB > k)
      case Cmp_Polygeist::GT: {
        for (auto for_lb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp_Polygeist(Cmp_Polygeist::GT, for_lb, afFor.getLowerBoundsMap().getNumDims(),
                       afFor.getLowerBoundsOperands(), val))
            return true;
        return false;
      }
      // \forall i \in [max(LB...), min(UB...)) >= k   => any(LB >= k)
      case Cmp_Polygeist::GE: {
        for (auto for_lb :
             afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
          if (valueCmp_Polygeist(Cmp_Polygeist::GE, for_lb, afFor.getLowerBoundsMap().getNumDims(),
                       afFor.getLowerBoundsOperands(), val))
            return true;
        return false;
      }
      }
    }

    if (scf::ForOp afFor =
            dyn_cast<scf::ForOp>(baval.getOwner()->getParentOp())) {
      if (baval.getArgNumber() == 0) {
        auto for_lb = afFor.getLowerBound();
        auto for_ub = afFor.getUpperBound();
        switch (cmp) {
        // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
        case Cmp_Polygeist::EQ: {
          if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_lb, val))
            return false;
          if (!val.isValue) {
            if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_ub, val.i_val + 1))
              return false;
            return true;
          }
          return false;
        }
        // \forall i \in [LB, UB) < k   => UB <= k
        case Cmp_Polygeist::LT: {
          return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val);
        }
        // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
        case Cmp_Polygeist::LE: {
          if (!val.isValue) {
            return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val.i_val + 1);
          }
          return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val);
        }
        // \forall i \in [LB, UB) > k   => LB > k
        case Cmp_Polygeist::GT: {
          return valueCmp_Polygeist(Cmp_Polygeist::GT, for_lb, val);
        }
        // \forall i \in [LB, UB) >= k   => LB >= k
        case Cmp_Polygeist::GE: {
          return valueCmp_Polygeist(Cmp_Polygeist::GE, for_lb, val);
        }
        }
      }
    }

    if (scf::ParallelOp afFor =
            dyn_cast<scf::ParallelOp>(baval.getOwner()->getParentOp())) {
      auto for_lb = afFor.getLowerBound()[baval.getArgNumber()];
      auto for_ub = afFor.getUpperBound()[baval.getArgNumber()];
      switch (cmp) {
      // \forall i \in [LB, UB) == k   => LB == k and UB == k+1
      case Cmp_Polygeist::EQ: {
        if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_lb, val))
          return false;
        if (!val.isValue) {
          if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, for_ub, val.i_val + 1))
            return false;
          return true;
        }
        return false;
      }
      // \forall i \in [LB, UB) < k   => UB <= k
      case Cmp_Polygeist::LT: {
        return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val);
      }
      // \forall i \in [LB, UB) <= k   => UB-1 <= k  => UB <= k+1
      case Cmp_Polygeist::LE: {
        if (!val.isValue) {
          return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val.i_val + 1);
        }
        return valueCmp_Polygeist(Cmp_Polygeist::LE, for_ub, val);
      }
      // \forall i \in [LB, UB) > k   => LB > k
      case Cmp_Polygeist::GT: {
        return valueCmp_Polygeist(Cmp_Polygeist::GT, for_lb, val);
      }
      // \forall i \in [LB, UB) >= k   => LB >= k
      case Cmp_Polygeist::GE: {
        return valueCmp_Polygeist(Cmp_Polygeist::GE, for_lb, val);
      }
      }
    }
  }
  if (val.isValue && val.v_val == bval) {
    switch (cmp) {
    case Cmp_Polygeist::EQ:
      return true;
    case Cmp_Polygeist::LT:
      return false;
    case Cmp_Polygeist::LE:
      return true;
    case Cmp_Polygeist::GT:
      return false;
    case Cmp_Polygeist::GE:
      return true;
    }
  }
  return false;
}

bool valueCmp_Polygeist(Cmp_Polygeist cmp, ValueOrInt_Polygeist expr, ValueOrInt_Polygeist val) {
  if (expr.isValue)
    return valueCmp_Polygeist(cmp, expr.v_val, val);
  else {
    return valueCmp_Polygeist(cmp, expr.i_val, val);
  }
}

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, ValueOrInt_Polygeist expr, int64_t val) {
//   return valueCmp_Polygeist(cmp, expr, APInt(64, val, true));
// }

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, APInt expr, ValueOrInt_Polygeist val) {
//   switch (cmp) {
//   case Cmp_Polygeist::EQ:
//     return val == expr;
//   case Cmp_Polygeist::LT:
//     return val > expr;
//   case Cmp_Polygeist::LE:
//     return val >= expr;
//   case Cmp_Polygeist::GT:
//     return val < expr;
//   case Cmp_Polygeist::GE:
//     return val <= expr;
//   }
// }

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, AffineExpr expr, size_t numDim, ValueRange operands,
//               int64_t val) {
//   return valueCmp_Polygeist(cmp, expr, numDim, operands, APInt(64, val, true));
// }

bool valueCmp_Polygeist(Cmp_Polygeist cmp, AffineExpr expr, size_t numDim, ValueRange operands,
              ValueOrInt_Polygeist val) {

  if (auto opd = dyn_cast<AffineConstantExpr>(expr)) {
    switch (cmp) {
    case Cmp_Polygeist::EQ:
      return val == opd.getValue();
    case Cmp_Polygeist::LT:
      return val > opd.getValue();
    case Cmp_Polygeist::LE:
      return val >= opd.getValue();
    case Cmp_Polygeist::GT:
      return val < opd.getValue();
    case Cmp_Polygeist::GE:
      return val <= opd.getValue();
    }
  }
  if (auto opd = dyn_cast<AffineDimExpr>(expr)) {
    return valueCmp_Polygeist(cmp, operands[opd.getPosition()], val);
  }
  if (auto opd = dyn_cast<AffineSymbolExpr>(expr)) {
    return valueCmp_Polygeist(cmp, operands[opd.getPosition() + numDim], val);
  }

  if (auto bop = dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (bop.getKind() == AffineExprKind::Add) {
      switch (cmp) {
      case Cmp_Polygeist::EQ:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::LT:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::LE:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::GT:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::GE:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val));
      }
    }
    if (bop.getKind() == AffineExprKind::Mul && val == 0) {
      switch (cmp) {
      case Cmp_Polygeist::EQ:
        return (valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val) ||
                valueCmp_Polygeist(cmp, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::LT:
        return (valueCmp_Polygeist(Cmp_Polygeist::LT, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(Cmp_Polygeist::GT, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::GT, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(Cmp_Polygeist::LT, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::LE:
        return valueCmp_Polygeist(Cmp_Polygeist::EQ, bop.getLHS(), numDim, operands, val) ||
               valueCmp_Polygeist(Cmp_Polygeist::EQ, bop.getRHS(), numDim, operands, val) ||
               ((valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getRHS(), numDim, operands, val)) ||
                (valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getRHS(), numDim, operands, val)));
      case Cmp_Polygeist::GT:
        return (valueCmp_Polygeist(Cmp_Polygeist::LT, bop.getLHS(), numDim, operands, val) &&
                valueCmp_Polygeist(Cmp_Polygeist::LT, bop.getRHS(), numDim, operands, 0)) ||
               (valueCmp_Polygeist(Cmp_Polygeist::GT, bop.getLHS(), numDim, operands, 0) &&
                valueCmp_Polygeist(Cmp_Polygeist::GT, bop.getRHS(), numDim, operands, val));
      case Cmp_Polygeist::GE:
        return valueCmp_Polygeist(Cmp_Polygeist::EQ, bop.getLHS(), numDim, operands, val) ||
               valueCmp_Polygeist(Cmp_Polygeist::EQ, bop.getRHS(), numDim, operands, val) ||
               ((valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp_Polygeist(Cmp_Polygeist::GE, bop.getRHS(), numDim, operands, val)) ||
                (valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getLHS(), numDim, operands, 0) &&
                 valueCmp_Polygeist(Cmp_Polygeist::LE, bop.getRHS(), numDim, operands, val)));
      }
    }
    if ((bop.getKind() == AffineExprKind::Mod ||
         bop.getKind() == AffineExprKind::FloorDiv) &&
        val == 0 && isa<AffineConstantExpr>(bop.getRHS()) &&
        cast<AffineConstantExpr>(bop.getRHS()).getValue() > 0) {
      switch (cmp) {
      case Cmp_Polygeist::GE:
        return valueCmp_Polygeist(cmp, bop.getLHS(), numDim, operands, val);
      default:
        break;
      }
    }
  }
  return false;
}

Value getBase(Value v) {
  while (true) {
    if (auto s = v.getDefiningOp<SubIndexOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<Memref2PointerOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<Pointer2MemrefOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::GEPOp>()) {
      v = s.getBase();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::BitcastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<memref::CastOp>()) {
      v = s.getSource();
      continue;
    }
    break;
  }
  return v;
}

// Range is [lb, ub)
bool rangeIncludes(Value bval, ValueOrInt_Polygeist lb, ValueOrInt_Polygeist ub) {
  if (auto baval = dyn_cast<BlockArgument>(bval)) {
    if (affine::AffineForOp afFor =
            dyn_cast<affine::AffineForOp>(baval.getOwner()->getParentOp())) {
      return valueCmp_Polygeist(
                 Cmp_Polygeist::LE,
                 afFor.getLowerBoundMap().getResults()[baval.getArgNumber()],
                 afFor.getLowerBoundMap().getNumDims(),
                 afFor.getLowerBoundOperands(), lb) &&
             valueCmp_Polygeist(
                 Cmp_Polygeist::GE,
                 afFor.getUpperBoundMap().getResults()[baval.getArgNumber()],
                 afFor.getUpperBoundMap().getNumDims(),
                 afFor.getUpperBoundOperands(), ub);
    }
    //  \forall i in [max(LB...),  min(UB)...] is a superset of [lb, ub)
    if (affine::AffineParallelOp afFor = dyn_cast<affine::AffineParallelOp>(
            baval.getOwner()->getParentOp())) {
      for (auto flb : afFor.getLowerBoundMap(baval.getArgNumber()).getResults())
        if (!valueCmp_Polygeist(Cmp_Polygeist::LE, flb, afFor.getLowerBoundsMap().getNumDims(),
                      afFor.getLowerBoundsOperands(), lb))
          return false;

      for (auto ulb : afFor.getUpperBoundMap(baval.getArgNumber()).getResults())
        if (!valueCmp_Polygeist(Cmp_Polygeist::GE, ulb, afFor.getUpperBoundsMap().getNumDims(),
                      afFor.getUpperBoundsOperands(), ub))
          return false;
      return true;
    }

    if (scf::ForOp afFor =
            dyn_cast<scf::ForOp>(baval.getOwner()->getParentOp())) {
      if (baval.getArgNumber() == 0) {
        auto flb = afFor.getLowerBound();
        auto fub = afFor.getUpperBound();
        return valueCmp_Polygeist(Cmp_Polygeist::LE, flb, lb) && valueCmp_Polygeist(Cmp_Polygeist::GE, fub, ub);
      }
    }

    if (scf::ParallelOp afFor =
            dyn_cast<scf::ParallelOp>(baval.getOwner()->getParentOp())) {
      auto flb = afFor.getLowerBound()[baval.getArgNumber()];
      auto fub = afFor.getUpperBound()[baval.getArgNumber()];
      return valueCmp_Polygeist(Cmp_Polygeist::LE, flb, lb) && valueCmp_Polygeist(Cmp_Polygeist::GE, fub, ub);
    }
  }

  IntegerAttr iattr;
  if (matchPattern(bval, m_Constant(&iattr))) {
    return lb == iattr.getValue() && ub == iattr.getValue() + 1;
  }

  return false;
}

// Range is [lb, ub)
bool rangeIncludes(AffineExpr expr, size_t numDims, ValueRange operands,
                   ValueOrInt_Polygeist lb, ValueOrInt_Polygeist ub) {
  if (auto opd = dyn_cast<AffineConstantExpr>(expr)) {
    return lb == opd.getValue() && ub == opd.getValue() + 1;
  }
  if (auto opd = dyn_cast<AffineDimExpr>(expr)) {
    return rangeIncludes(operands[opd.getPosition()], lb, ub);
  }
  if (auto opd = dyn_cast<AffineSymbolExpr>(expr)) {
    return rangeIncludes(operands[opd.getPosition() + numDims], lb, ub);
  }
  return false;
}



//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//

class UndefToLLVM final : public OpRewritePattern<enzymexla::UndefOp> {
public:
  using OpRewritePattern<enzymexla::UndefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::UndefOp uop,
                                PatternRewriter &rewriter) const override {
    auto ty = uop.getResult().getType();
    if (!LLVM::isCompatibleType(ty))
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(uop, ty);
    return success();
  }
};

void enzymexla::UndefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<UndefToLLVM>(context);
}

template <typename UndefType>
class UndefCmpProp final : public OpRewritePattern<CmpIOp> {
public:
  using OpRewritePattern<CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpIOp op,
                                PatternRewriter &rewriter) const override {
    Value v = op->getOperand(0);
    Operation *undef;
    if (!(undef = v.getDefiningOp<UndefType>()))
      return failure();
    if (!op.getRhs().getDefiningOp<arith::ConstantOp>())
      return failure();
    rewriter.setInsertionPoint(undef);
    rewriter.replaceOpWithNewOp<UndefType>(op, op.getType());
    return success();
  }
};

class CmpProp final : public OpRewritePattern<CmpIOp> {
public:
  using OpRewritePattern<CmpIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpIOp op,
                                PatternRewriter &rewriter) const override {
    auto ifOp = op.getLhs().getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return failure();
    auto rhs = op.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!rhs) {
      return failure();
    }
    // auto idx = op.getLhs().cast<OpResult>().getResultNumber();
    auto idx = cast<OpResult>(op.getLhs()).getResultNumber();
    bool change = false;
    for (auto v :
         {ifOp.thenYield().getOperand(idx), ifOp.elseYield().getOperand(idx)}) {
      change |= v.getDefiningOp<arith::ConstantIntOp>() ||
                v.getDefiningOp<LLVM::UndefOp>() ||
                v.getDefiningOp<enzymexla::UndefOp>();
      if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
        if (auto it = dyn_cast<IntegerType>(extOp.getIn().getType()))
          change |= it.getWidth() == 1;
      if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
        if (auto it = dyn_cast<IntegerType>(extOp.getIn().getType()))
          change |= it.getWidth() == 1;
    }
    if (!change) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    llvm::append_range(resultTypes, ifOp.getResultTypes());
    resultTypes.push_back(op.getType());

    rewriter.setInsertionPoint(ifOp);
    auto rhs2 = rewriter.clone(*rhs)->getResult(0);
    auto nop = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), resultTypes, ifOp.getCondition(), /*hasElse*/ true);
    rewriter.eraseBlock(nop.thenBlock());
    rewriter.eraseBlock(nop.elseBlock());

    rewriter.inlineRegionBefore(ifOp.getThenRegion(), nop.getThenRegion(),
                                nop.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), nop.getElseRegion(),
                                nop.getElseRegion().begin());

    SmallVector<Value> thenYields;
    llvm::append_range(thenYields, nop.thenYield().getOperands());
    rewriter.setInsertionPoint(nop.thenYield());
    thenYields.push_back(rewriter.create<CmpIOp>(op.getLoc(), op.getPredicate(),
                                                 thenYields[idx], rhs2));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(nop.thenYield(), thenYields);

    SmallVector<Value> elseYields;
    llvm::append_range(elseYields, nop.elseYield().getOperands());
    rewriter.setInsertionPoint(nop.elseYield());
    elseYields.push_back(rewriter.create<CmpIOp>(op.getLoc(), op.getPredicate(),
                                                 elseYields[idx], rhs2));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(nop.elseYield(), elseYields);
    rewriter.replaceOp(ifOp, nop.getResults().take_front(ifOp.getNumResults()));
    rewriter.replaceOp(op, nop.getResults().take_back(1));
    return success();
  }
};

class SelectOfExt final : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = dyn_cast<IntegerType>(op.getType());
    if (!ty)
      return failure();
    IntegerAttr lhs, rhs;
    Value lhs_v = nullptr, rhs_v = nullptr;
    unsigned lhs_w = 0, rhs_w = 0;
    if (auto ext = op.getTrueValue().getDefiningOp<arith::ExtUIOp>()) {
      lhs_v = ext.getIn();
      // lhs_w = lhs_v.getType().cast<IntegerType>().getWidth();
      lhs_w = cast<IntegerType>(lhs_v.getType()).getWidth();
    } else if (matchPattern(op.getTrueValue(), m_Constant(&lhs))) {
    } else {
      return failure();
    }

    if (auto ext = op.getFalseValue().getDefiningOp<arith::ExtUIOp>()) {
      rhs_v = ext.getIn();
      // rhs_w = rhs_v.getType().cast<IntegerType>().getWidth();
      rhs_w = cast<IntegerType>(rhs_v.getType()).getWidth();
    } else if (matchPattern(op.getFalseValue(), m_Constant(&rhs))) {
    } else {
      return failure();
    }

    // No ext's
    if (!lhs_v && !rhs_v)
      return failure();

    auto fitsIn = [&](auto i, int width) {
      // if there is nothing in the bits that would be discarded
      return !((~((static_cast<decltype(i)>(1) << width) - 1)) & i);
    };

    // Truncate the other non-extended const but only if the original constant
    // fits in the new width
    if (!lhs_v && fitsIn(lhs.getInt(), rhs_w)) {
      assert(rhs_v);
      lhs_v = rewriter.create<ConstantIntOp>(op.getLoc(), lhs.getInt(), rhs_w);
    }
    if (!rhs_v && fitsIn(rhs.getInt(), lhs_w)) {
      assert(lhs_v);
      rhs_v = rewriter.create<ConstantIntOp>(op.getLoc(), rhs.getInt(), lhs_w);
    }

    // If we werent able to truncate the const
    if (!(lhs_v && rhs_v))
      return failure();

    rewriter.replaceOpWithNewOp<ExtUIOp>(
        op, op.getType(),
        rewriter.create<arith::SelectOp>(op.getLoc(), op.getCondition(), lhs_v,
                                  rhs_v));
    return success();
  }
};



template <typename T, typename UndefType>
class UndefProp final : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    Value v = op->getOperand(0);
    Operation *undef;
    if (!(undef = v.getDefiningOp<UndefType>()))
      return failure();
    rewriter.setInsertionPoint(undef);
    rewriter.replaceOpWithNewOp<UndefType>(op, op.getType());
    return success();
  }
};

class OrIExcludedMiddle final : public OpRewritePattern<arith::OrIOp> {
public:
  using OpRewritePattern<arith::OrIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<CmpIOp>();
    auto rhs = op.getRhs().getDefiningOp<CmpIOp>();
    if (!lhs || !rhs)
      return failure();
    if (lhs.getLhs() != rhs.getLhs() || lhs.getRhs() != rhs.getRhs() ||
        lhs.getPredicate() != arith::invertPredicate(rhs.getPredicate()))
      return failure();
    rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
    return success();
  }
};

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

OpFoldResult TypeAlignOp::fold(FoldAdaptor adaptor) {
  Type T = getSourceAttr().getValue();
  if (isa<IntegerType, FloatType>(T) || LLVM::isCompatibleType(T)) {
    DataLayout DLI(((Operation *)*this)->getParentOfType<ModuleOp>());
    return IntegerAttr::get(getResult().getType(),
                            APInt(64, DLI.getTypeABIAlignment(T)));
  }
  return nullptr;
}

struct TypeAlignCanonicalize : public OpRewritePattern<TypeAlignOp> {
  using OpRewritePattern<TypeAlignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TypeAlignOp op,
                                PatternRewriter &rewriter) const override {
    Type T = op.getSourceAttr().getValue();
    if (isa<IntegerType, FloatType>(T) || LLVM::isCompatibleType(T)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, DLI.getTypeABIAlignment(T));
      return success();
    }
    return failure();
  }
};

struct ConstantRankReduction : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rewriter) const override {
    mlir::Type Ty = op->getResult(0).getType();
    // MemRefType MT = Ty.cast<MemRefType>();
    MemRefType MT = cast<MemRefType>(Ty);
    if (MT.getShape().size() == 0)
      return failure();
    SmallVector<uint64_t> v;
    bool set = false;
    for (auto u : op->getResult(0).getUsers()) {
      if (auto load = dyn_cast<memref::LoadOp>(u)) {
        if (!set) {
          for (auto i : load.getIndices()) {
            IntegerAttr constValue;
            if (!matchPattern(i, m_Constant(&constValue)))
              return failure();
            v.push_back(constValue.getValue().getZExtValue());
          }
          set = true;
        } else {
          for (auto pair : llvm::zip(load.getIndices(), v)) {
            IntegerAttr constValue;
            if (!matchPattern(std::get<0>(pair), m_Constant(&constValue)))
              return failure();
            if (constValue.getValue().getZExtValue() != std::get<1>(pair))
              return failure();
          }
        }
        continue;
      }
      if (auto load = dyn_cast<affine::AffineLoadOp>(u)) {
        SmallVector<Value> indices;
        auto map = load.getAffineMapAttr().getValue();
        if (!set) {
          for (AffineExpr op : map.getResults()) {
            // auto opd = op.dyn_cast<AffineConstantExpr>();
            auto opd = dyn_cast<AffineConstantExpr>(op);
            if (!opd)
              return failure();
            v.push_back(opd.getValue());
          }
          set = true;
        } else {
          for (auto pair : llvm::zip(map.getResults(), v)) {
            // auto opd = std::get<0>(pair).dyn_cast<AffineConstantExpr>();
            auto opd = dyn_cast<AffineConstantExpr>(std::get<0>(pair));
            if (!opd)
              return failure();
            if (opd.getValue() != std::get<1>(pair))
              return failure();
          }
        }
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(u)) {
        if (store.getValue() == op)
          return failure();
        continue;
      }
      if (auto store = dyn_cast<affine::AffineStoreOp>(u)) {
        if (store.getValue() == op)
          return failure();
        continue;
      }

      return failure();
    }
    if (!set)
      return failure();

    MT = MemRefType::get({}, MT.getElementType(), MemRefLayoutAttrInterface(),
                         MT.getMemorySpace());

    auto newOp = rewriter.create<memref::AllocaOp>(op.getLoc(), MT);

    for (auto u : llvm::make_early_inc_range(op->getResult(0).getUsers())) {
      rewriter.setInsertionPoint(u);
      if (auto load = dyn_cast<memref::LoadOp>(u)) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(load, newOp,
                                                    ArrayRef<Value>());
        continue;
      }
      if (auto load = dyn_cast<affine::AffineLoadOp>(u)) {
        rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
            load, newOp, AffineMap::get(op.getContext()), ArrayRef<Value>());
        continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(u)) {
        Value cond = nullptr;
        for (auto pair : llvm::zip(store.getIndices(), v)) {
          auto val = rewriter.create<arith::CmpIOp>(
              store.getLoc(), CmpIPredicate::eq, std::get<0>(pair),
              rewriter.create<arith::ConstantIndexOp>(store.getLoc(),
                                                      std::get<1>(pair)));
          if (cond == nullptr)
            cond = val;
          else
            cond = rewriter.create<arith::AndIOp>(store.getLoc(), cond, val);
        }
        auto loc = store.getLoc();
        auto val = store.getValue();
        auto ifOp = rewriter.replaceOpWithNewOp<scf::IfOp>(
            store, TypeRange(), cond, /*hasElse*/ false);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        rewriter.create<memref::StoreOp>(loc, val, newOp, ArrayRef<Value>());
        continue;
      }
      if (auto store = dyn_cast<affine::AffineStoreOp>(u)) {
        Value cond = nullptr;
        auto map = store.getAffineMapAttr().getValue();
        for (auto pair : llvm::enumerate(v)) {
          auto apply = rewriter.create<affine::AffineApplyOp>(
              store.getLoc(), map.getSliceMap(pair.index(), 1),
              store.getMapOperands());
          auto val = rewriter.create<arith::CmpIOp>(
              store.getLoc(), CmpIPredicate::eq, apply.getResult(),
              rewriter.create<arith::ConstantIndexOp>(store.getLoc(),
                                                      pair.value()));
          if (cond == nullptr)
            cond = val;
          else
            cond = rewriter.create<arith::AndIOp>(store.getLoc(), cond, val);
        }
        auto loc = store.getLoc();
        auto val = store.getValue();
        auto ifOp = rewriter.replaceOpWithNewOp<scf::IfOp>(
            store, TypeRange(), cond, /*hasElse*/ false);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        rewriter.create<affine::AffineStoreOp>(loc, val, newOp,
                                               AffineMap::get(op.getContext()),
                                               ArrayRef<Value>());
        continue;
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AffineIfSinking : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0)
      return failure();
    if (op.hasElse())
      return failure();
    auto par = dyn_cast<affine::AffineParallelOp>(op->getParentOp());
    if (!par)
      return failure();

    bool sink = true;
    bool hoist = true;
    for (auto node = op->getNextNode(); node; node = node->getNextNode()) {
      if (!isReadNone(node)) {
        sink = false;
        break;
      }
    }
    for (auto node = op->getPrevNode(); node; node = node->getPrevNode()) {
      if (!isReadNone(node)) {
        hoist = false;
        break;
      }
    }
    if (!sink && !hoist)
      return failure();

    SmallVector<bool> doneVars(par.getSteps().size(), false);
    SmallVector<bool> doneConstraints(
        op.getIntegerSet().getConstraints().size(), false);

    SmallVector<AffineExpr> remaining;
    SmallVector<bool> isEq;
    SmallVector<Value> dimVals;
    SmallVector<Value> symVals;

    SmallVector<AffineExpr> dimReplacements;

    for (unsigned idx = 0; idx < par.getUpperBoundsMap().getNumDims(); ++idx) {
      dimReplacements.push_back(
          getAffineDimExpr(dimVals.size(), op.getContext()));
      dimVals.push_back(par.getUpperBoundsOperands()[idx]);
    }
    SmallVector<AffineExpr> symReplacements;
    for (unsigned idx = 0; idx < par.getUpperBoundsMap().getNumSymbols();
         ++idx) {
      symReplacements.push_back(
          getAffineSymbolExpr(dimVals.size(), op.getContext()));
      symVals.push_back(
          par.getUpperBoundsOperands()[idx +
                                       par.getUpperBoundsMap().getNumDims()]);
    }

    for (auto cst : llvm::enumerate(op.getIntegerSet().getConstraints())) {
      if (!op.getIntegerSet().isEq(cst.index())) {
        return failure();
      }

      // auto opd = cst.value().dyn_cast<AffineDimExpr>();
      auto opd = dyn_cast<AffineDimExpr>(cst.value());
      if (!opd) {
        return failure();
      }
      auto ival = dyn_cast<BlockArgument>(op.getOperands()[opd.getPosition()]);
      if (!ival) {
        return failure();
      }

      if (ival.getOwner()->getParentOp() != par) {
        remaining.push_back(getAffineDimExpr(dimVals.size(), op.getContext()));
        dimVals.push_back(ival);
        isEq.push_back(op.getIntegerSet().isEq(cst.index()));
        continue;
      }

      if (doneVars[ival.getArgNumber()]) {
        return failure();
      }

      if (!valueCmp_Polygeist(Cmp_Polygeist::GE, ival, 0)) {
        return failure();
      }

      // TODO make this a check in the below at runtime
      // rather than at compile time.

      for (auto ub : par.getUpperBoundMap(cst.index()).getResults()) {
        auto ub2 = ub.replaceDimsAndSymbols(dimReplacements, symReplacements);
        remaining.push_back(ub2 - 1);
        isEq.push_back(false);
      }
      doneVars[ival.getArgNumber()] = true;
      doneConstraints[cst.index()];
    }

    if (!llvm::all_of(doneVars, [](bool b) { return b; })) {
      return failure();
    }

    bool failed = false;
    SmallVector<Operation *> toSink;
    std::function<void(Value)> recur = [&](Value v) {
      if (!par.getRegion().isAncestor(v.getParentRegion()) ||
          op.getThenRegion().isAncestor(v.getParentRegion()))
        return;
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        if (ba.getOwner()->getParentOp() == par) {
          return;
        }
      }
      Operation *vo = v.getDefiningOp();
      if (!vo) {
        failed = true;
        return;
      }
      if (isReadNone(vo)) {
        toSink.push_back(vo);
        for (auto oper : vo->getOperands())
          recur(oper);
        return;
      }
      failed = true;
      return;
    };
    op->walk([&](Operation *sub) {
      if (sub != op) {
        for (auto oper : sub->getOperands())
          recur(oper);
      }
    });
    if (failed)
      return failure();

    auto iset =
        IntegerSet::get(dimVals.size(), symVals.size(), remaining, isEq);

    SmallVector<Value> newVals(dimVals);
    newVals.append(symVals);

    if (sink)
      rewriter.setInsertionPoint(par->getNextNode());
    else
      rewriter.setInsertionPoint(par);

    IRMapping map;
    auto c0 = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    for (auto i : par.getIVs()) {
      map.map(i, c0);
      for (auto &val : newVals)
        if (val == i)
          val = c0;
    }

    auto newIf = rewriter.create<affine::AffineIfOp>(
        op.getLoc(), TypeRange(), iset, newVals, /*hasElse*/ false);
    rewriter.eraseBlock(newIf.getThenBlock());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().begin());
    rewriter.eraseOp(op);
    rewriter.setInsertionPointToStart(newIf.getThenBlock());
    for (auto o : llvm::reverse(toSink)) {
      auto nop = rewriter.clone(*o, map);
      rewriter.replaceOpUsesWithinBlock(o, nop->getResults(), newIf.getThenBlock());
    }
    for (auto i : par.getIVs()) {
      i.replaceUsesWithIf(c0, [&](OpOperand &user) {
        return newIf->isAncestor(user.getOwner());
      });
    }
    return success();
  }
};

static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

struct AffineIfSimplification : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<AffineExpr> todo;
    SmallVector<bool> eqFlags;
    bool knownFalse = false;
    bool removed = false;
    for (auto cst : llvm::enumerate(op.getIntegerSet().getConstraints())) {
      // auto opd = cst.value().dyn_cast<AffineConstantExpr>();
      auto opd = dyn_cast<AffineConstantExpr>(cst.value());
      if (!opd) {
        if (op.getIntegerSet().isEq(cst.index())) {
          if (auto bop = dyn_cast<AffineBinaryOpExpr>(cst.value())) {
            if (bop.getKind() == AffineExprKind::Mul &&
                bop.getRHS().getKind() == AffineExprKind::Constant) {
              removed = true;
              if (cast<AffineConstantExpr>(bop.getRHS()).getValue() != 0) {
                todo.push_back(bop.getLHS());
                eqFlags.push_back(op.getIntegerSet().isEq(cst.index()));
              }
              continue;
            }
            if (bop.getKind() == AffineExprKind::Add &&
                valueCmp_Polygeist(Cmp_Polygeist::GE, bop, op.getIntegerSet().getNumDims(),
                         op.getOperands(), 0)) {
              todo.push_back(bop.getLHS());
              eqFlags.push_back(op.getIntegerSet().isEq(cst.index()));
              todo.push_back(bop.getRHS());
              eqFlags.push_back(op.getIntegerSet().isEq(cst.index()));
              removed = true;
              continue;
            }
          }
        }

        bool canRemove = false;
        for (auto paren = op->getParentOfType<affine::AffineIfOp>(); paren;
             paren = paren->getParentOfType<affine::AffineIfOp>()) {
          for (auto cst2 : paren.getIntegerSet().getConstraints()) {
            if (paren.getElseRegion().isAncestor(op->getParentRegion()))
              continue;
            if (cst2 == cst.value() &&
                paren.getIntegerSet().getNumDims() ==
                    op.getIntegerSet().getNumDims() &&
                paren.getIntegerSet().getNumSymbols() ==
                    op.getIntegerSet().getNumSymbols() &&
                llvm::all_of(llvm::zip(paren.getOperands(), op.getOperands()),
                             [](std::tuple<Value, Value> p) {
                               return std::get<0>(p) == std::get<1>(p);
                             })) {
              canRemove = true;
              break;
            }
          }
          if (canRemove)
            break;
        }
        //// expr -1 >= 0    => expr > 0
        if (!op.getIntegerSet().isEq(cst.index())) {
          auto expr = cst.value() + 1;
          for (auto paren = op->getParentOfType<affine::AffineParallelOp>();
               paren;
               paren = paren->getParentOfType<affine::AffineParallelOp>()) {
            if (canRemove)
              break;
            for (auto tup : llvm::enumerate(paren.getSteps())) {
              bool found = false;
              for (auto ub : paren.getUpperBoundMap(tup.index()).getResults()) {
                if (auto exprS = dyn_cast<AffineSymbolExpr>(expr)) {
                  if (auto ubS = dyn_cast<AffineSymbolExpr>(ub)) {
                    if (op.getOperands()[exprS.getPosition() +
                                         op.getIntegerSet().getNumDims()] ==
                        paren.getUpperBoundsOperands()[ubS.getPosition() +
                                                       paren.getUpperBoundsMap()
                                                           .getNumDims()]) {

                      found = true;
                      break;
                    }
                  }
                }
              }
              if (!found)
                continue;

              if (!valueCmp_Polygeist(Cmp_Polygeist::GE, paren.getIVs()[tup.index()], 0))
                continue;

              canRemove = true;
              break;
            }
          }
          if (auto bop = dyn_cast<AffineBinaryOpExpr>(cst.value())) {
            if (bop.getKind() == AffineExprKind::Add) {
            }
          }
        }
        if (canRemove) {
          removed = true;
          continue;
        }

        todo.push_back(cst.value());
        eqFlags.push_back(op.getIntegerSet().isEq(cst.index()));
        continue;
      }
      removed = true;

      if (op.getIntegerSet().isEq(cst.index())) {
        if (opd.getValue() != 0) {
          knownFalse = true;
          break;
        }
      }
      if (!(opd.getValue() >= 0)) {
        knownFalse = true;
        break;
      }
    }

    if (knownFalse) {
      todo.clear();
    }

    if (todo.size() == 0) {

      if (!knownFalse)
        replaceOpWithRegion(rewriter, op, op.getThenRegion());
      else if (!op.getElseRegion().empty())
        replaceOpWithRegion(rewriter, op, op.getElseRegion());
      else
        rewriter.eraseOp(op);

      return success();
    }

    if (!removed)
      return failure();

    auto iset =
        IntegerSet::get(op.getIntegerSet().getNumDims(),
                        op.getIntegerSet().getNumSymbols(), todo, eqFlags);

    auto newIf = rewriter.create<affine::AffineIfOp>(
        op.getLoc(), op.getResultTypes(), iset, op.getOperands(),
        /*hasElse*/ true);
    rewriter.eraseBlock(newIf.getThenBlock());
    rewriter.eraseBlock(newIf.getElseBlock());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().begin());
    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

struct CombineAffineIfs : public OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern<affine::AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp nextIf,
                                PatternRewriter &rewriter) const override {
    Block *parent = nextIf->getBlock();
    if (nextIf == &parent->front())
      return failure();

    auto prevIf = dyn_cast<affine::AffineIfOp>(nextIf->getPrevNode());
    if (!prevIf)
      return failure();

    // Determine the logical then/else blocks when prevIf's
    // condition is used. Null means the block does not exist
    // in that case (e.g. empty else). If neither of these
    // are set, the two conditions cannot be compared.
    Block *nextThen = nullptr;
    Block *nextElse = nullptr;

    if (nextIf.getIntegerSet() == prevIf.getIntegerSet() &&
        llvm::all_of(llvm::zip(nextIf.getOperands(), prevIf.getOperands()),
                     [](std::tuple<Value, Value> p) {
                       return std::get<0>(p) == std::get<1>(p);
                     })) {
      nextThen = nextIf.getThenBlock();
      if (!nextIf.getElseRegion().empty())
        nextElse = nextIf.getElseBlock();
    }

    if (!nextThen && !nextElse)
      return failure();

    SmallVector<Value> prevElseYielded;
    if (!prevIf.getElseRegion().empty())
      prevElseYielded =
          cast<affine::AffineYieldOp>(prevIf.getElseBlock()->getTerminator())
              .getOperands();
    // Replace all uses of return values of op within nextIf with the
    // corresponding yields
    for (auto it : llvm::zip(
             prevIf.getResults(),
             cast<affine::AffineYieldOp>(prevIf.getThenBlock()->getTerminator())
                 .getOperands(),
             prevElseYielded))
      for (OpOperand &use :
           llvm::make_early_inc_range(std::get<0>(it).getUses())) {
        if (nextThen && nextThen->getParent()->isAncestor(
                            use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<1>(it));
          rewriter.finalizeOpModification(use.getOwner());
        } else if (nextElse && nextElse->getParent()->isAncestor(
                                   use.getOwner()->getParentRegion())) {
          rewriter.startOpModification(use.getOwner());
          use.set(std::get<2>(it));
          rewriter.finalizeOpModification(use.getOwner());
        }
      }

    SmallVector<Type> mergedTypes(prevIf.getResultTypes());
    llvm::append_range(mergedTypes, nextIf.getResultTypes());

    affine::AffineIfOp combinedIf = rewriter.create<affine::AffineIfOp>(
        nextIf.getLoc(), mergedTypes, prevIf.getIntegerSet(),
        prevIf.getOperands(), /*hasElse=*/true);
    rewriter.eraseBlock(&combinedIf.getThenRegion().back());
    rewriter.eraseBlock(&combinedIf.getElseRegion().back());

    rewriter.inlineRegionBefore(prevIf.getThenRegion(),
                                combinedIf.getThenRegion(),
                                combinedIf.getThenRegion().begin());

    if (nextThen) {
      affine::AffineYieldOp thenYield = cast<affine::AffineYieldOp>(
          combinedIf.getThenBlock()->getTerminator());
      affine::AffineYieldOp thenYield2 =
          cast<affine::AffineYieldOp>(nextThen->getTerminator());
      rewriter.mergeBlocks(nextThen, combinedIf.getThenBlock());
      rewriter.setInsertionPointToEnd(combinedIf.getThenBlock());

      SmallVector<Value> mergedYields(thenYield.getOperands());
      llvm::append_range(mergedYields, thenYield2.getOperands());
      rewriter.create<affine::AffineYieldOp>(thenYield2.getLoc(), mergedYields);
      rewriter.eraseOp(thenYield);
      rewriter.eraseOp(thenYield2);
    }

    rewriter.inlineRegionBefore(prevIf.getElseRegion(),
                                combinedIf.getElseRegion(),
                                combinedIf.getElseRegion().begin());

    if (nextElse) {
      if (combinedIf.getElseRegion().empty()) {
        rewriter.inlineRegionBefore(*nextElse->getParent(),
                                    combinedIf.getElseRegion(),
                                    combinedIf.getElseRegion().begin());
      } else {
        affine::AffineYieldOp elseYield = cast<affine::AffineYieldOp>(
            combinedIf.getElseBlock()->getTerminator());
        affine::AffineYieldOp elseYield2 =
            cast<affine::AffineYieldOp>(nextElse->getTerminator());
        rewriter.mergeBlocks(nextElse, combinedIf.getElseBlock());

        rewriter.setInsertionPointToEnd(combinedIf.getElseBlock());

        SmallVector<Value> mergedElseYields(elseYield.getOperands());
        llvm::append_range(mergedElseYields, elseYield2.getOperands());

        rewriter.create<affine::AffineYieldOp>(elseYield2.getLoc(),
                                               mergedElseYields);
        rewriter.eraseOp(elseYield);
        rewriter.eraseOp(elseYield2);
      }
    }

    SmallVector<Value> prevValues;
    SmallVector<Value> nextValues;
    for (const auto &pair : llvm::enumerate(combinedIf.getResults())) {
      if (pair.index() < prevIf.getNumResults())
        prevValues.push_back(pair.value());
      else
        nextValues.push_back(pair.value());
    }
    rewriter.replaceOp(prevIf, prevValues);
    rewriter.replaceOp(nextIf, nextValues);
    return success();
  }
};

struct MergeNestedAffineParallelLoops
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = op.getRegion().getBlocks().front();
    if (!llvm::hasSingleElement(outerBody.without_terminator()))
      return failure();

    auto innerOp = dyn_cast<affine::AffineParallelOp>(outerBody.front());
    if (!innerOp)
      return failure();

    for (auto val : outerBody.getArguments())
      if (llvm::is_contained(innerOp.getLowerBoundsOperands(), val) ||
          llvm::is_contained(innerOp.getUpperBoundsOperands(), val))
        return failure();

    // Reductions are not supported yet.
    if (!op.getReductions().empty() || !innerOp.getReductions().empty())
      return failure();

    SmallVector<Type> newTypes(op.getResultTypes());
    for (auto T : innerOp.getResultTypes())
      newTypes.push_back(T);

    ArrayRef<Attribute> reductions;
    SmallVector<AffineExpr> lbounds;
    SmallVector<AffineExpr> ubounds;
    SmallVector<Value> lboundValues;
    SmallVector<Value> uboundValues;

    for (size_t i = 0; i < op.getLowerBoundsMap().getNumDims(); i++)
      lboundValues.push_back(op.getLowerBoundsOperands()[i]);

    for (size_t i = 0; i < op.getUpperBoundsMap().getNumDims(); i++)
      uboundValues.push_back(op.getUpperBoundsOperands()[i]);

    for (size_t i = 0; i < innerOp.getLowerBoundsMap().getNumDims(); i++)
      lboundValues.push_back(innerOp.getLowerBoundsOperands()[i]);

    for (size_t i = 0; i < innerOp.getUpperBoundsMap().getNumDims(); i++)
      uboundValues.push_back(innerOp.getUpperBoundsOperands()[i]);

    for (size_t i = 0; i < op.getLowerBoundsMap().getNumSymbols(); i++)
      lboundValues.push_back(
          op.getLowerBoundsOperands()[i + op.getLowerBoundsMap().getNumDims()]);

    for (size_t i = 0; i < op.getUpperBoundsMap().getNumSymbols(); i++)
      uboundValues.push_back(
          op.getUpperBoundsOperands()[i + op.getUpperBoundsMap().getNumDims()]);

    for (size_t i = 0; i < innerOp.getLowerBoundsMap().getNumSymbols(); i++)
      lboundValues.push_back(
          innerOp.getLowerBoundsOperands()[i + innerOp.getLowerBoundsMap()
                                                   .getNumDims()]);

    for (size_t i = 0; i < innerOp.getUpperBoundsMap().getNumSymbols(); i++)
      uboundValues.push_back(
          innerOp.getUpperBoundsOperands()[i + innerOp.getUpperBoundsMap()
                                                   .getNumDims()]);

    for (auto e : op.getLowerBoundsMap().getResults()) {
      lbounds.push_back(e);
    }

    for (auto e : op.getUpperBoundsMap().getResults()) {
      ubounds.push_back(e);
    }

    for (auto e : innerOp.getLowerBoundsMap()
                      .shiftDims(op.getLowerBoundsMap().getNumDims())
                      .shiftSymbols(op.getLowerBoundsMap().getNumSymbols())
                      .getResults()) {
      lbounds.push_back(e);
    }

    for (auto e : innerOp.getUpperBoundsMap()
                      .shiftDims(op.getUpperBoundsMap().getNumDims())
                      .shiftSymbols(op.getUpperBoundsMap().getNumSymbols())
                      .getResults()) {
      ubounds.push_back(e);
    }

    SmallVector<Value> operands = lboundValues;
    operands.append(uboundValues);

    SmallVector<int32_t> lboundGroup;
    SmallVector<int32_t> uboundGroup;
    for (auto U : op.getLowerBoundsGroups())
      lboundGroup.push_back(U.getZExtValue());
    for (auto U : innerOp.getLowerBoundsGroups())
      lboundGroup.push_back(U.getZExtValue());
    for (auto U : op.getUpperBoundsGroups())
      uboundGroup.push_back(U.getZExtValue());
    for (auto U : innerOp.getUpperBoundsGroups())
      uboundGroup.push_back(U.getZExtValue());

    SmallVector<int64_t> steps;
    for (auto U : op.getSteps())
      steps.push_back(U);
    for (auto U : innerOp.getSteps())
      steps.push_back(U);

    affine::AffineParallelOp affineLoop =
        rewriter.create<affine::AffineParallelOp>(
            op.getLoc(), newTypes, rewriter.getArrayAttr(reductions),
            AffineMapAttr::get(
                AffineMap::get(op.getLowerBoundsMap().getNumDims() +
                                   innerOp.getLowerBoundsMap().getNumDims(),
                               op.getLowerBoundsMap().getNumSymbols() +
                                   innerOp.getLowerBoundsMap().getNumSymbols(),
                               lbounds, op.getContext())),
            rewriter.getI32TensorAttr(lboundGroup),
            AffineMapAttr::get(
                AffineMap::get(op.getUpperBoundsMap().getNumDims() +
                                   innerOp.getUpperBoundsMap().getNumDims(),
                               op.getUpperBoundsMap().getNumSymbols() +
                                   innerOp.getUpperBoundsMap().getNumSymbols(),
                               ubounds, op.getContext())),
            rewriter.getI32TensorAttr(uboundGroup),
            rewriter.getI64ArrayAttr(steps), operands);

    rewriter.inlineRegionBefore(op.getRegion(), affineLoop.getRegion(),
                                affineLoop.getRegion().begin());
    auto yld = affineLoop.getBody()->getTerminator();
    rewriter.eraseOp(innerOp.getBody()->getTerminator());
    SmallVector<Value> post;
    for (auto v : innerOp.getIVs()) {
      post.push_back(
          affineLoop.getBody()->addArgument(v.getType(), v.getLoc()));
    }
    rewriter.inlineBlockBefore(innerOp.getBody(), yld, post);
    return success();
  }
};

struct PrepMergeNestedAffineParallelLoops
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp oop,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = oop.getRegion().getBlocks().front();
    affine::AffineParallelOp innerOp = nullptr;
    SmallVector<Operation *> toMove;
    for (auto &op : outerBody) {
      if (auto innerOp2 = dyn_cast<affine::AffineParallelOp>(&op)) {
        if (innerOp)
          return failure();
        if (!isa<affine::AffineYieldOp>(innerOp2->getNextNode())) {
          return failure();
        }
        innerOp = innerOp2;
        continue;
      }
      if (isMemoryEffectFree(&op)) {
        if (!isa<affine::AffineYieldOp>(&op))
          toMove.push_back(&op);
        continue;
      }

      return failure();
    }

    if (!innerOp || !toMove.size()) {
      return failure();
    }

    IRMapping map;
    rewriter.setInsertionPointToStart(innerOp.getBody());
    for (auto o : toMove) {
      rewriter.replaceOp(o, rewriter.clone(*o)->getResults());
    }
    return success();
  }
};

struct MergeNestedAffineParallelIf
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    Block &outerBody = op.getRegion().getBlocks().front();

    affine::AffineIfOp innerOp = nullptr;
    for (auto &op : outerBody) {
      if (auto innerOp2 = dyn_cast<affine::AffineIfOp>(&op)) {
        if (innerOp)
          return failure();
        if (!isa<affine::AffineYieldOp>(innerOp2->getNextNode())) {
          return failure();
        }
        innerOp = innerOp2;
        continue;
      }
      if (!isReadOnly(&op))
        return failure();
    }

    if (!innerOp)
      return failure();

    // Reductions are not supported yet.
    if (!op.getReductions().empty())
      return failure();

    if (innerOp.hasElse())
      return failure();

    SmallVector<int32_t> lboundGroup;
    SmallVector<int32_t> uboundGroup;
    for (auto U : op.getLowerBoundsGroups())
      lboundGroup.push_back(U.getZExtValue());
    for (auto U : op.getUpperBoundsGroups())
      uboundGroup.push_back(U.getZExtValue());

    SmallVector<AffineExpr> lbounds;
    SmallVector<AffineExpr> ubounds;

    for (auto e : op.getLowerBoundsMap().getResults()) {
      lbounds.push_back(e);
    }

    for (auto e : op.getUpperBoundsMap().getResults()) {
      ubounds.push_back(e);
    }

    bool changed = false;
    SmallVector<AffineExpr> remaining;
    SmallVector<bool> isEq;
    for (auto cst : llvm::enumerate(innerOp.getIntegerSet().getConstraints())) {
      if (innerOp.getIntegerSet().isEq(cst.index())) {
        remaining.push_back(cst.value());
        isEq.push_back(innerOp.getIntegerSet().isEq(cst.index()));
        continue;
      }

      auto getIndUsage = [&op](AffineExpr cst, ValueRange operands,
                               std::map<size_t, AffineExpr> &indUsage,
                               bool &legal,
                               bool *failure = nullptr) -> AffineExpr {
        AffineExpr rhs = getAffineConstantExpr(0, cst.getContext());
        SmallVector<AffineExpr> todo = {cst};
        legal = true;
        while (todo.size()) {
          auto cur = todo.back();
          todo.pop_back();
          if (isa<AffineConstantExpr, AffineSymbolExpr>(cur)) {
            rhs = rhs + cur;
            continue;
          }
          if (auto dim = dyn_cast<AffineDimExpr>(cur)) {
            auto ival = dyn_cast<BlockArgument>(operands[dim.getPosition()]);
            if (!ival || ival.getOwner()->getParentOp() != op) {
              rhs = rhs + dim;
              if (failure)
                *failure = true;
              continue;
            }
            if (indUsage.find(ival.getArgNumber()) != indUsage.end()) {
              legal = false;
              continue;
            }
            indUsage[ival.getArgNumber()] =
                getAffineConstantExpr(1, op.getContext());
            continue;
          }
          if (auto bop = dyn_cast<AffineBinaryOpExpr>(cur)) {
            if (bop.getKind() == AffineExprKind::Add) {
              todo.push_back(bop.getLHS());
              todo.push_back(bop.getRHS());
              continue;
            }
            if (bop.getKind() == AffineExprKind::Mul) {
              if (!isa<AffineConstantExpr, AffineSymbolExpr>(bop.getRHS())) {
                legal = false;
                continue;
              }

              if (auto dim = dyn_cast<AffineDimExpr>(bop.getLHS())) {
                auto ival =
                    dyn_cast<BlockArgument>(operands[dim.getPosition()]);
                if (!ival || ival.getOwner()->getParentOp() != op) {
                  rhs = rhs + bop;
                  // While legal, this may run before parallel merging
                  // and prevent parallel fusion
                  legal = false;
                  if (failure)
                    *failure = true;
                  continue;
                }
                if (indUsage.find(ival.getArgNumber()) != indUsage.end()) {
                  legal = false;
                  continue;
                }
                indUsage[ival.getArgNumber()] = bop.getRHS();
                continue;
              }
            }
          }
          if (failure)
            *failure = true;
          legal = false;
          break;
        }
        return rhs;
      };

      bool legal;
      std::map<size_t, AffineExpr> indUsage;
      bool failureV = false;
      AffineExpr rhs = getIndUsage(cst.value(), innerOp.getOperands(), indUsage,
                                   legal, &failureV);
      if (failureV)
        return failure();

      if (!legal || indUsage.size() != 1) {
        remaining.push_back(cst.value());
        isEq.push_back(innerOp.getIntegerSet().isEq(cst.index()));
        continue;
      }
      auto pair = *indUsage.begin();
      auto affCst = dyn_cast<AffineConstantExpr>(pair.second);
      if (!affCst) {
        remaining.push_back(cst.value());
        isEq.push_back(innerOp.getIntegerSet().isEq(cst.index()));
        continue;
      }

      // currently aff * idx + stuff >= 0
      // currently aff * idx >= -stuff
      //    idx >= (-stuff).floorDiv(aff)   OR   idx <= ...

      if (affCst.getValue() < 0)
        rhs = rhs.floorDiv(-affCst.getValue()) + 1;
      else {
        remaining.push_back(cst.value());
        isEq.push_back(innerOp.getIntegerSet().isEq(cst.index()));
        continue;
      }

      changed = true;

      size_t off = 0;
      for (size_t i = 0; i < pair.first; i++)
        off += uboundGroup[i];

      if (auto newCst = dyn_cast<AffineConstantExpr>(rhs)) {
        bool seen = false;
        for (size_t i = 0; i < uboundGroup[pair.first]; i++) {
          if (auto oldCst = dyn_cast<AffineConstantExpr>(ubounds[off + i])) {
            seen = true;
            if (newCst.getValue() < oldCst.getValue())
              ubounds[i] = rhs;
          }
        }
        if (seen)
          continue;
      }
      ubounds.insert(ubounds.begin() + off,
                     rhs.shiftDims(innerOp.getIntegerSet().getNumDims(),
                                   op.getUpperBoundsMap().getNumDims())
                         .shiftSymbols(innerOp.getIntegerSet().getNumSymbols(),
                                       op.getUpperBoundsMap().getNumSymbols()));

      uboundGroup[pair.first]++;
    }

    if (!changed)
      return failure();

    SmallVector<Value> lboundValues;
    SmallVector<Value> uboundValues;

    for (size_t i = 0; i < op.getLowerBoundsMap().getNumDims(); i++)
      lboundValues.push_back(op.getLowerBoundsOperands()[i]);

    for (size_t i = 0; i < op.getUpperBoundsMap().getNumDims(); i++)
      uboundValues.push_back(op.getUpperBoundsOperands()[i]);

    for (size_t i = 0; i < innerOp.getIntegerSet().getNumDims(); i++)
      uboundValues.push_back(innerOp.getOperands()[i]);

    for (size_t i = 0; i < op.getLowerBoundsMap().getNumSymbols(); i++)
      lboundValues.push_back(
          op.getLowerBoundsOperands()[i + op.getLowerBoundsMap().getNumDims()]);

    for (size_t i = 0; i < op.getUpperBoundsMap().getNumSymbols(); i++)
      uboundValues.push_back(
          op.getUpperBoundsOperands()[i + op.getUpperBoundsMap().getNumDims()]);

    for (size_t i = 0; i < innerOp.getIntegerSet().getNumSymbols(); i++)
      uboundValues.push_back(
          innerOp.getOperands()[i + innerOp.getIntegerSet().getNumDims()]);

    SmallVector<Value> operands = lboundValues;
    operands.append(uboundValues);

    ArrayRef<Attribute> reductions;

    affine::AffineParallelOp affineLoop =
        rewriter.create<affine::AffineParallelOp>(
            op.getLoc(), op.getResultTypes(), rewriter.getArrayAttr(reductions),
            AffineMapAttr::get(
                AffineMap::get(op.getLowerBoundsMap().getNumDims(),
                               op.getLowerBoundsMap().getNumSymbols(), lbounds,
                               op.getContext())),
            rewriter.getI32TensorAttr(lboundGroup),
            AffineMapAttr::get(
                AffineMap::get(op.getUpperBoundsMap().getNumDims() +
                                   innerOp.getIntegerSet().getNumDims(),
                               op.getUpperBoundsMap().getNumSymbols() +
                                   innerOp.getIntegerSet().getNumSymbols(),
                               ubounds, op.getContext())),
            rewriter.getI32TensorAttr(uboundGroup), op.getStepsAttr(),
            operands);

    rewriter.inlineRegionBefore(op.getRegion(), affineLoop.getRegion(),
                                affineLoop.getRegion().begin());

    rewriter.setInsertionPoint(innerOp);

    if (remaining.empty()) {
      auto yld =
          cast<affine::AffineYieldOp>(innerOp.getThenBlock()->getTerminator());
      SmallVector<Value> toRet(yld.getOperands());
      rewriter.eraseOp(yld);
      rewriter.inlineBlockBefore(innerOp.getThenBlock(), innerOp);
      rewriter.replaceOp(innerOp, toRet);
    } else {
      affine::AffineIfOp newIf = rewriter.create<affine::AffineIfOp>(
          innerOp.getLoc(), innerOp.getResultTypes(),
          IntegerSet::get(innerOp.getIntegerSet().getNumDims(),
                          innerOp.getIntegerSet().getNumSymbols(), remaining,
                          isEq),
          innerOp.getOperands(), /*hasElse*/ false);

      rewriter.eraseBlock(newIf.getThenBlock());

      rewriter.inlineRegionBefore(innerOp.getThenRegion(),
                                  newIf.getThenRegion(),
                                  newIf.getThenRegion().begin());
      rewriter.inlineRegionBefore(innerOp.getElseRegion(),
                                  newIf.getElseRegion(),
                                  newIf.getElseRegion().begin());

      rewriter.replaceOp(innerOp, newIf->getResults());
      rewriter.replaceOp(op, affineLoop->getResults());
    }
    return success();
  }
};

struct MergeParallelInductions
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    // Reductions are not supported yet.
    if (!op.getReductions().empty())
      return failure();

    auto getIndUsage = [&op](AffineExpr cst, ValueRange operands,
                             std::map<size_t, AffineExpr> &indUsage,
                             bool &legal) -> AffineExpr {
      AffineExpr rhs = getAffineConstantExpr(0, cst.getContext());
      SmallVector<AffineExpr> todo = {cst};
      legal = true;
      while (todo.size()) {
        auto cur = todo.back();
        todo.pop_back();
        if (isa<AffineConstantExpr, AffineSymbolExpr>(cur)) {
          rhs = rhs + cur;
          continue;
        }
        if (auto dim = dyn_cast<AffineDimExpr>(cur)) {
          auto ival = dyn_cast<BlockArgument>(operands[dim.getPosition()]);
          if (!ival || ival.getOwner()->getParentOp() != op) {
            rhs = rhs + dim;
            continue;
          }
          if (indUsage.find(ival.getArgNumber()) != indUsage.end()) {
            legal = false;
            continue;
          }
          indUsage[ival.getArgNumber()] =
              getAffineConstantExpr(1, op.getContext());
          continue;
        }
        if (auto bop = dyn_cast<AffineBinaryOpExpr>(cur)) {
          if (bop.getKind() == AffineExprKind::Add) {
            todo.push_back(bop.getLHS());
            todo.push_back(bop.getRHS());
            continue;
          }
          if (bop.getKind() == AffineExprKind::Mul) {
            if (!isa<AffineConstantExpr, AffineSymbolExpr>(bop.getRHS())) {
              legal = false;
              continue;
            }

            if (auto dim = dyn_cast<AffineDimExpr>(bop.getLHS())) {
              auto ival = dyn_cast<BlockArgument>(operands[dim.getPosition()]);
              if (!ival || ival.getOwner()->getParentOp() != op) {
                rhs = rhs + bop;
                continue;
              }
              if (indUsage.find(ival.getArgNumber()) != indUsage.end()) {
                legal = false;
                continue;
              }
              indUsage[ival.getArgNumber()] = bop.getRHS();
              continue;
            }
          }
        }
        legal = false;
        break;
      }
      return rhs;
    };

    // TODO check all users are affine sums like this.
    std::map<size_t, IndexCastOp> idxCasts;
    SetVector<Operation *> affineMapUsers;
    SmallVector<bool> legality;
    SmallVector<ValueOrInt_Polygeist> fixedUpperBounds;
    for (auto iv : op.getIVs()) {
      bool legal = true;
      IndexCastOp idxCst = nullptr;

      for (auto lb : op.getLowerBoundMap(iv.getArgNumber()).getResults()) {
        if (auto cst = dyn_cast<AffineConstantExpr>(lb)) {
          if (cst.getValue() != 0) {
            legal = false;
            break;
          }
        } else {
          legal = false;
          break;
        }
      }
      bool seenub = false;
      for (auto ub : op.getUpperBoundMap(iv.getArgNumber()).getResults()) {
        if (seenub) {
          legal = false;
          break;
        }
        seenub = true;
        if (auto cst = dyn_cast<AffineConstantExpr>(ub)) {
          fixedUpperBounds.push_back(ValueOrInt_Polygeist(cst.getValue()));
        } else if (auto dim = dyn_cast<AffineDimExpr>(ub)) {
          fixedUpperBounds.push_back(
              ValueOrInt_Polygeist(op.getUpperBoundsOperands()[dim.getPosition()]));
        } else if (auto sym = dyn_cast<AffineSymbolExpr>(ub)) {
          fixedUpperBounds.push_back(ValueOrInt_Polygeist(
              op.getUpperBoundsOperands()[op.getUpperBoundsMap().getNumDims() +
                                          sym.getPosition()]));
        } else {
          legal = false;
          fixedUpperBounds.push_back(ValueOrInt_Polygeist(0));
        }
      }

      SmallVector<Operation *, 1> affineMapUsers_t;
      for (auto U : iv.getUsers()) {
        SmallVector<AffineExpr> exprs;
        ValueRange operands;
        if (auto AL = dyn_cast<affine::AffineLoadOp>(U)) {
          for (auto E : AL.getAffineMap().getResults())
            exprs.push_back(E);
          operands = AL.getMapOperands();
          affineMapUsers_t.push_back(U);
        } else if (auto AS = dyn_cast<affine::AffineStoreOp>(U)) {
          if (AS.getValue() == iv)
            legal = false;
          for (auto E : AS.getAffineMap().getResults())
            exprs.push_back(E);
          operands = AS.getMapOperands();
          affineMapUsers_t.push_back(U);
        } else if (auto AI = dyn_cast<affine::AffineIfOp>(U)) {
          for (auto E : AI.getIntegerSet().getConstraints())
            exprs.push_back(E);
          operands = AI.getOperands();
          affineMapUsers_t.push_back(U);
        } else if (auto idx = dyn_cast<IndexCastOp>(U)) {
          if (idxCst) {
            legal = false;
            break;
          } else
            idxCst = idx;
        } else {
          legal = false;
          break;
        }
        for (auto expr : exprs) {
          bool flegal = true;
          std::map<size_t, AffineExpr> indUsage;
          getIndUsage(expr, operands, indUsage, flegal);
          if (!flegal || indUsage.size() < 2) {
            legal = false;
            break;
          }
        }
      }
      legality.push_back(legal);
      if (legal) {
        for (auto o : affineMapUsers_t) {
          affineMapUsers.insert(o);
        }
        if (idxCst)
          idxCasts[iv.getArgNumber()] = idxCst;
      }
    }
    for (auto tup : llvm::zip(op.getIVs(), legality)) {
      if (!std::get<1>(tup))
        for (auto U : std::get<0>(tup).getUsers())
          if (affineMapUsers.count(U))
            affineMapUsers.remove(U);
    }
    for (auto U : affineMapUsers) {
      SmallVector<AffineExpr> exprs;
      ValueRange operands;
      size_t numDim;
      if (auto AL = dyn_cast<affine::AffineLoadOp>(U)) {
        for (auto E : AL.getAffineMap().getResults())
          exprs.push_back(E);
        operands = AL.getMapOperands();
        numDim = AL.getAffineMap().getNumDims();
      } else if (auto AS = dyn_cast<affine::AffineStoreOp>(U)) {
        for (auto E : AS.getAffineMap().getResults())
          exprs.push_back(E);
        operands = AS.getMapOperands();
        numDim = AS.getAffineMap().getNumDims();
      } else if (auto AI = dyn_cast<affine::AffineIfOp>(U)) {
        for (auto E : AI.getIntegerSet().getConstraints())
          exprs.push_back(E);
        operands = AI.getOperands();
        numDim = AI.getIntegerSet().getNumDims();
      } else {
        llvm_unreachable("Unknown affine use type");
      }

      for (auto expr : exprs) {
        bool flegal;
        std::map<size_t, AffineExpr> indUsage;
        getIndUsage(expr, operands, indUsage, flegal);

        for (auto pair1 : indUsage) {
          for (auto pair2 : indUsage) {
            if (pair1.first == pair2.first)
              continue;
            if (auto cst = dyn_cast<AffineConstantExpr>(pair1.second)) {
              if (cst.getValue() == -1) {
                pair2.second = -pair2.second;
                pair1.second = -pair1.second;
              } else if (cst.getValue() != 1)
                continue;
            } else
              continue;

            if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, pair2.second, numDim, operands,
                          fixedUpperBounds[pair1.first]))
              continue;

            if (idxCasts.count(pair1.first) != idxCasts.count(pair2.first))
              continue;

            bool legalPair = true;
            for (auto U : affineMapUsers) {
              if (!legalPair)
                break;
              SmallVector<AffineExpr> exprs;
              ValueRange operands;
              if (auto AL = dyn_cast<affine::AffineLoadOp>(U)) {
                for (auto E : AL.getAffineMap().getResults())
                  exprs.push_back(E);
                operands = AL.getMapOperands();
              } else if (auto AS = dyn_cast<affine::AffineStoreOp>(U)) {
                for (auto E : AS.getAffineMap().getResults())
                  exprs.push_back(E);
                operands = AS.getMapOperands();
              } else if (auto AI = dyn_cast<affine::AffineIfOp>(U)) {
                for (auto E : AI.getIntegerSet().getConstraints())
                  exprs.push_back(E);
                operands = AI.getOperands();
              } else {
                llvm_unreachable("Unknown affine use type");
              }

              for (auto expr : exprs) {
                if (!legalPair)
                  break;
                bool sublegal;
                std::map<size_t, AffineExpr> subIndUsage;
                getIndUsage(expr, operands, subIndUsage, sublegal);
                auto find1 = subIndUsage.find(pair1.first);
                auto find2 = subIndUsage.find(pair2.first);

                if (find1 == subIndUsage.end() && find2 == subIndUsage.end())
                  continue;
                if (find1 == subIndUsage.end() || find2 == subIndUsage.end()) {
                  legalPair = false;
                  break;
                }
                if (find1->second * pair2.second !=
                    find2->second * pair1.second) {
                  legalPair = false;
                  break;
                }
              }
            }

            if (idxCasts.count(pair1.first)) {
              Value val = idxCasts[pair1.first];
              if (!val.hasOneUse())
                continue;
              AddIOp add = dyn_cast<AddIOp>(*val.user_begin());
              if (!add)
                continue;
              Value other = (add.getLhs() == val) ? add.getRhs() : add.getLhs();

              MulIOp mul = other.getDefiningOp<MulIOp>();
              if (mul.getLhs() == idxCasts[pair2.first]) {
                if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, mul.getRhs(),
                              fixedUpperBounds[pair1.first]))
                  continue;
              } else {
                if (mul.getRhs() != idxCasts[pair2.first])
                  continue;
                if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, mul.getLhs(),
                              fixedUpperBounds[pair1.first]))
                  continue;
              }
              if (!mul->getResult(0).hasOneUse())
                continue;
              if (!idxCasts[pair2.first]->getResult(0).hasOneUse())
                continue;
            }

            SmallVector<int32_t> uboundGroup;
            for (auto U : op.getUpperBoundsGroups())
              uboundGroup.push_back(U.getZExtValue());

            SmallVector<AffineExpr> ubounds;

            for (auto e : op.getUpperBoundsMap().getResults()) {
              ubounds.push_back(e);
            }

            size_t off1 = 0;
            for (size_t i = 0; i < pair1.first; i++)
              off1 += uboundGroup[i];
            size_t off2 = 0;
            for (size_t i = 0; i < pair2.first; i++)
              off2 += uboundGroup[i];

            ubounds[off1] = ubounds[off1] * ubounds[off2];
            ubounds[off2] = getAffineConstantExpr(1, op.getContext());

            affine::AffineParallelOp affineLoop =
                rewriter.create<affine::AffineParallelOp>(
                    op.getLoc(), op.getResultTypes(), op.getReductionsAttr(),
                    op.getLowerBoundsMapAttr(), op.getLowerBoundsGroupsAttr(),
                    AffineMapAttr::get(
                        AffineMap::get(op.getUpperBoundsMap().getNumDims(),
                                       op.getUpperBoundsMap().getNumSymbols(),
                                       ubounds, op.getContext())),
                    op.getUpperBoundsGroupsAttr(), op.getStepsAttr(),
                    op.getOperands());

            rewriter.inlineRegionBefore(op.getRegion(), affineLoop.getRegion(),
                                        affineLoop.getRegion().begin());
            return success();
          }
        }
      }
    }
    return failure();
  }
};

struct RemoveAffineParallelSingleIter
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {

    // Reductions are not supported yet.
    if (!op.getReductions().empty())
      return failure();

    ArrayRef<Attribute> reductions;
    SmallVector<AffineExpr> lbounds;
    SmallVector<AffineExpr> ubounds;

    for (auto e : op.getLowerBoundsMap().getResults()) {
      lbounds.push_back(e);
    }

    for (auto e : op.getUpperBoundsMap().getResults()) {
      ubounds.push_back(e);
    }

    SmallVector<int32_t> lboundGroup;
    SmallVector<int32_t> uboundGroup;
    for (auto U : op.getLowerBoundsGroups())
      lboundGroup.push_back(U.getZExtValue());
    for (auto U : op.getUpperBoundsGroups())
      uboundGroup.push_back(U.getZExtValue());

    SmallVector<int64_t> steps;
    for (auto U : op.getSteps())
      steps.push_back(U);

    Block *Tmp = new Block();
    SmallVector<Value> replacements;
    bool changed = false;
    for (ssize_t idx = steps.size() - 1; idx >= 0; idx--) {
      replacements.insert(replacements.begin(),
                          Tmp->insertArgument((unsigned)0,
                                              op.getIVs()[idx].getType(),
                                              op.getIVs()[idx].getLoc()));
      if (lboundGroup[idx] != 1)
        continue;
      if (uboundGroup[idx] != 1)
        continue;
      size_t loff = 0;
      for (size_t i = 0; i < idx; i++)
        loff += lboundGroup[i];

      size_t uoff = 0;
      for (size_t i = 0; i < idx; i++)
        uoff += uboundGroup[i];

      auto lb = dyn_cast<AffineConstantExpr>(lbounds[loff]);
      if (!lb)
        continue;
      auto ub = dyn_cast<AffineConstantExpr>(ubounds[uoff]);
      if (!ub)
        continue;
      if (lb.getValue() >= ub.getValue())
        continue;
      if (lb.getValue() + steps[idx] >= ub.getValue()) {
        Tmp->eraseArgument(0);
        replacements[0] =
            rewriter.create<ConstantIndexOp>(op.getLoc(), lb.getValue());
        lboundGroup.erase(lboundGroup.begin() + idx);
        uboundGroup.erase(uboundGroup.begin() + idx);
        lbounds.erase(lbounds.begin() + loff);
        ubounds.erase(ubounds.begin() + uoff);
        steps.erase(steps.begin() + idx);
        changed = true;
        continue;
      }
      continue;
    }
    if (!changed) {
      delete Tmp;
      return failure();
    }

    if (steps.size() == 0) {
      delete Tmp;

      auto yld = cast<affine::AffineYieldOp>(op.getBody()->getTerminator());
      SmallVector<Value> toRet(yld.getOperands());
      rewriter.eraseOp(yld);
      rewriter.inlineBlockBefore(op.getBody(), op, replacements);
      rewriter.replaceOp(op, toRet);
    } else {

      affine::AffineParallelOp affineLoop =
          rewriter.create<affine::AffineParallelOp>(
              op.getLoc(), op.getResultTypes(),
              rewriter.getArrayAttr(reductions),
              AffineMapAttr::get(
                  AffineMap::get(op.getLowerBoundsMap().getNumDims(),
                                 op.getLowerBoundsMap().getNumSymbols(),
                                 lbounds, op.getContext())),
              rewriter.getI32TensorAttr(lboundGroup),
              AffineMapAttr::get(
                  AffineMap::get(op.getUpperBoundsMap().getNumDims(),
                                 op.getUpperBoundsMap().getNumSymbols(),
                                 ubounds, op.getContext())),
              rewriter.getI32TensorAttr(uboundGroup),
              rewriter.getI64ArrayAttr(steps), op.getOperands());

      affineLoop.getRegion().getBlocks().push_back(Tmp);
      if (rewriter.getListener())
        rewriter.getListener()->notifyBlockInserted(Tmp);

      rewriter.mergeBlocks(op.getBody(), affineLoop.getBody(), replacements);
      rewriter.replaceOp(op, affineLoop->getResults());
    }

    return success();
  }
};

template <typename T> struct BufferElimination : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  static bool legalFor(T op, affine::AffineForOp afFor) {
    auto S = op.getType().getShape();
    if (S.size() != 1)
      return false;

    size_t opidx = 0;
    for (size_t i = 0; i < S.size(); i++) {
      if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, afFor.getLowerBoundMap().getResults()[i],
                    afFor.getLowerBoundMap().getNumDims(),
                    afFor.getLowerBoundOperands(), 0))
        return false;

      if (S[i] == -1) {
        Value ubval = op.getOperands()[i];
        opidx++;
        if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, afFor.getUpperBoundMap().getResults()[i],
                      afFor.getUpperBoundMap().getNumDims(),
                      afFor.getUpperBoundOperands(), ubval))
          return false;
      } else {
        if (!valueCmp_Polygeist(Cmp_Polygeist::EQ, afFor.getUpperBoundMap().getResults()[i],
                      afFor.getUpperBoundMap().getNumDims(),
                      afFor.getUpperBoundOperands(), S[i]))
          return false;
      }
    }

    return true;
  }

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (isCaptured(op))
      return failure();

    for (auto U : op->getResult(0).getUsers()) {
      if (auto load = dyn_cast<affine::AffineLoadOp>(U)) {
        AffineMap map = load.getAffineMapAttr().getValue();
        if (map.getNumResults() != 1)
          continue;
        auto opd = dyn_cast<AffineDimExpr>(map.getResults()[0]);
        if (!opd)
          continue;
        auto val = dyn_cast<BlockArgument>(
            ((Value)load.getMapOperands()[opd.getPosition()]));
        if (!val)
          continue;

        affine::AffineForOp copyOutOfBuffer =
            dyn_cast<affine::AffineForOp>(val.getOwner()->getParentOp());
        if (!copyOutOfBuffer)
          continue;
        if (copyOutOfBuffer.getNumResults())
          continue;

        if (!legalFor(op, copyOutOfBuffer))
          continue;

        if (load->getParentOp() != copyOutOfBuffer)
          continue;
        if (!llvm::hasNItems(*copyOutOfBuffer.getBody(), 3))
          continue;

        auto store = dyn_cast<affine::AffineStoreOp>(load->getNextNode());
        if (!store)
          continue;

        Value otherBuf = store.getMemref();

        if (load.getAffineMapAttr().getValue() !=
            store.getAffineMapAttr().getValue())
          continue;
        if (!llvm::all_of(
                llvm::zip(load.getMapOperands(), store.getMapOperands()),
                [](std::tuple<Value, Value> v) -> bool {
                  return std::get<0>(v) == std::get<1>(v);
                }))
          continue;

        // Needs to be noalias, otherwise we cannot tell if intermediate users
        // also use the other buffer.
        if (!(otherBuf.getDefiningOp<memref::AllocOp>()) &&
            !(otherBuf.getDefiningOp<memref::AllocaOp>()))
          continue;

        for (auto U2 : otherBuf.getUsers()) {
          if (auto load = dyn_cast<affine::AffineLoadOp>(U2)) {
            AffineMap map = load.getAffineMapAttr().getValue();
            if (map.getNumResults() != 1)
              continue;
            auto opd = dyn_cast<AffineDimExpr>(map.getResults()[0]);
            if (!opd)
              continue;
            auto val = dyn_cast<BlockArgument>(
                ((Value)load.getMapOperands()[opd.getPosition()]));
            if (!val)
              continue;

            affine::AffineForOp copyIntoBuffer =
                dyn_cast<affine::AffineForOp>(val.getOwner()->getParentOp());
            if (!copyIntoBuffer)
              continue;
            if (copyIntoBuffer.getNumResults())
              continue;

            if (load->getParentOp() != copyIntoBuffer)
              continue;
            if (!llvm::hasNItems(*copyIntoBuffer.getBody(), 3))
              continue;

            auto store = dyn_cast<affine::AffineStoreOp>(load->getNextNode());
            if (!store)
              continue;

            if (load.getAffineMapAttr().getValue() !=
                store.getAffineMapAttr().getValue())
              continue;
            if (!llvm::all_of(
                    llvm::zip(load.getMapOperands(), store.getMapOperands()),
                    [](std::tuple<Value, Value> v) -> bool {
                      return std::get<0>(v) == std::get<1>(v);
                    }))
              continue;

            if (store.getMemref() != op)
              continue;

            if (copyIntoBuffer->getBlock() != copyOutOfBuffer->getBlock())
              continue;

            bool legal = true;
            for (Operation *mod = copyIntoBuffer->getNextNode();
                 mod != copyOutOfBuffer; mod = mod->getNextNode()) {
              if (!mod) {
                legal = false;
                break;
              }
              for (auto U3 : otherBuf.getUsers()) {
                if (mod->isAncestor(U3)) {
                  legal = false;
                  break;
                }
              }
            }
            if (!legal)
              continue;

            if (!legalFor(op, copyIntoBuffer))
              continue;

            assert(otherBuf.getType() == op.getType());

            rewriter.replaceOpUsesWithIf(
                op, otherBuf, nullptr, [&](OpOperand &use) {
                  Operation *owner = use.getOwner();
                  while (owner &&
                         owner->getBlock() != copyIntoBuffer->getBlock()) {
                    owner = owner->getParentOp();
                  }
                  if (!owner)
                    return false;

                  return copyIntoBuffer->isBeforeInBlock(owner) &&
                         owner->isBeforeInBlock(copyOutOfBuffer);
                });

            rewriter.setInsertionPoint(copyOutOfBuffer);
            rewriter.clone(*copyIntoBuffer);
            rewriter.eraseOp(copyOutOfBuffer);
            rewriter.eraseOp(copyIntoBuffer);
            // TODO remove
            //
            //    %op = alloc
            //    stuff(op)
            //
            //    copyIntoBuffer(op, otherBuf)
            //
            //    stuffToReplace(op)
            //
            //    copyOutOfBuffer(otherBuf, op)
            //
            //    stuff2(op)
            //
            //
            //    BECOMES
            //
            //    %op = alloc
            //    stuff(op)
            //
            //
            //    stuffToReplace(otherBuf)
            //
            //    # ERASED copyOutOfBuffer(otherBuf, op)
            //    copyIntoBuffer(op, otherBuf)
            //
            //    stuff2(op)
            //
            return success();
          }
        }
      }
    }
    return failure();
  }
};

template <typename T> struct SimplifyDeadAllocV2 : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T alloc,
                                PatternRewriter &rewriter) const override {
    if (llvm::any_of(alloc->getUsers(), [&](Operation *op) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(op))
            return storeOp.getValue() == alloc;
          if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
            return storeOp.getValue() == alloc;
          if (auto storeOp = dyn_cast<LLVM::StoreOp>(op))
            return storeOp.getValue() == alloc;
          return !isa<memref::DeallocOp>(op);
        }))
      return failure();

    for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
      rewriter.eraseOp(user);

    rewriter.eraseOp(alloc);
    return success();
  }
};

struct AggressiveAllocaScopeInliner
    : public OpRewritePattern<memref::AllocaScopeOp> {
  using OpRewritePattern<memref::AllocaScopeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaScopeOp op,
                                PatternRewriter &rewriter) const override {
    auto hasPotentialAlloca = [&]() {
      return op
          ->walk<WalkOrder::PreOrder>([&](Operation *alloc) {
            if (alloc == op || isa<LLVM::CallOp>(alloc) ||
                isa<func::CallOp>(alloc) || isa<omp::BarrierOp>(alloc) ||
                isa<enzymexla::BarrierOp>(alloc))
              return WalkResult::advance();
            if (isOpItselfPotentialAutomaticAllocation(alloc))
              return WalkResult::interrupt();
            if (alloc->hasTrait<OpTrait::AutomaticAllocationScope>())
              return WalkResult::skip();
            return WalkResult::advance();
          })
          .wasInterrupted();
    };

    // If we are the not last operation in the block
    if (op->getNextNode() != op->getBlock()->getTerminator()) {
      // If this contains no potential allocation, it is always legal to
      // inline. Otherwise, consider two conditions:
      if (hasPotentialAlloca()) {
        // If the parent isn't an allocation scope, or we are not the last
        // non-terminator op in the parent, we will extend the lifetime.
        if (!op->getParentOp()->hasTrait<OpTrait::AutomaticAllocationScope>())
          return failure();
      }
    }

    Block *block = &op.getRegion().front();
    Operation *terminator = block->getTerminator();
    ValueRange results = terminator->getOperands();
    rewriter.inlineBlockBefore(block, op);
    rewriter.replaceOp(op, results);
    rewriter.eraseOp(terminator);
    return success();
  }
};

struct InductiveVarRemoval : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto tup : llvm::zip(forOp.getResults(), forOp.getRegionIterArgs(),
                              forOp.getInits())) {
      if (!std::get<0>(tup).use_empty() || std::get<1>(tup).use_empty()) {
        continue;
      }
      bool legal = true;
      SmallVector<Value> vals = {std::get<1>(tup)};
      SmallPtrSet<Value, 2> seen = {};
      while (vals.size()) {
        Value v = vals.pop_back_val();
        if (seen.count(v))
          continue;
        seen.insert(v);
        for (OpOperand &back : v.getUses()) {
          if (auto yop = dyn_cast<scf::YieldOp>(back.getOwner())) {
            if (auto ifOp = dyn_cast<scf::IfOp>(yop->getParentOp())) {
              vals.push_back(ifOp.getResult(back.getOperandNumber()));
              continue;
            }
            if (auto op = dyn_cast<scf::ForOp>(yop->getParentOp())) {
              vals.push_back(op.getResult(back.getOperandNumber()));
              vals.push_back(op.getRegionIterArgs()[back.getOperandNumber()]);
              continue;
            }
          }
          if (auto yop = dyn_cast<affine::AffineYieldOp>(back.getOwner())) {
            if (auto ifOp = dyn_cast<affine::AffineIfOp>(yop->getParentOp())) {
              vals.push_back(ifOp.getResult(back.getOperandNumber()));
              continue;
            }
            if (auto op = dyn_cast<affine::AffineForOp>(yop->getParentOp())) {
              vals.push_back(op.getResult(back.getOperandNumber()));
              vals.push_back(op.getRegionIterArgs()[back.getOperandNumber()]);
              continue;
            }
          }
          if (auto selOp = dyn_cast<arith::SelectOp>(back.getOwner())) {
            if (selOp.getCondition() != v)
              vals.push_back(selOp);
            continue;
          }
          legal = false;
          break;
        }
        if (!legal)
          break;
      }
      if (legal) {
        rewriter.modifyOpInPlace(forOp, [&] {
          std::get<1>(tup).replaceAllUsesWith(std::get<2>(tup));
        });
        changed = true;
      }
    }
    return success(changed);
  }
};

template <typename T>
struct AffineBufferElimination : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> stores;
    LLVM_DEBUG(llvm::dbgs()
               << " Attempting affine buffer elim of: " << op << "\n");
    SmallVector<Operation *> loads;
    for (auto U : op->getResult(0).getUsers()) {

      if (auto store2 = dyn_cast<affine::AffineStoreOp>(U)) {
        if (store2.getValue() == op->getResult(0)) {
          LLVM_DEBUG(llvm::dbgs() << " + stored the ptr " << *U << "\n");
          return failure();
        }
        stores.push_back(store2);
        continue;
      }

      if (auto store2 = dyn_cast<memref::StoreOp>(U)) {
        if (store2.getValue() == op->getResult(0)) {
          LLVM_DEBUG(llvm::dbgs() << " + stored the ptr " << *U << "\n");
          return failure();
        }
        stores.push_back(store2);
        continue;
      }

      if (auto load = dyn_cast<affine::AffineLoadOp>(U)) {
        loads.push_back(load);
        continue;
      }

      if (auto load = dyn_cast<memref::LoadOp>(U)) {
        loads.push_back(load);
        continue;
      }

      if (isa<memref::DeallocOp>(U)) {
        continue;
      }
      if (isa<func::ReturnOp>(U)) {
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << " + unknown user " << *U << "\n");
      return failure();
    }

    bool changed = false;

    for (auto store : stores) {

      Value storeVal = nullptr;
      SmallVector<ValueOrInt_Polygeist> storeIdxs;
      if (auto store2 = dyn_cast<affine::AffineStoreOp>(store)) {
        bool legal = true;
        for (AffineExpr ores : store2.getAffineMap().getResults()) {
          ValueOrInt_Polygeist V((Value) nullptr);
          if (auto dim = dyn_cast<AffineDimExpr>(ores)) {
            V = ValueOrInt_Polygeist(store2.getMapOperands()[dim.getPosition()]);
          } else if (auto dim = dyn_cast<AffineSymbolExpr>(ores)) {
            V = ValueOrInt_Polygeist(
                store2.getMapOperands()[dim.getPosition() +
                                        store2.getAffineMap().getNumDims()]);
          } else if (auto dim = dyn_cast<AffineConstantExpr>(ores)) {
            V = ValueOrInt_Polygeist(dim.getValue());
          } else {
            legal = false;
            break;
          }
          storeIdxs.push_back(V);
        }

        if (!legal)
          continue;

        storeVal = store2.getValue();
      } else if (auto store2 = cast<memref::StoreOp>(store)) {
        for (auto idx : store2.getIndices())
          storeIdxs.push_back(ValueOrInt_Polygeist(idx));
        storeVal = store2.getValue();
      } else {
        llvm_unreachable("Unknown store type");
      }

      if (Operation *replacedValue = storeVal.getDefiningOp())
        if (!isReadOnly(replacedValue))
          continue;

      int opn = 0;

      // Ensure the one store fully covers the space.

      // Note: this may not actually necessary, since if there's only one store,
      //   which is at the same level

      // 1) Check that each memref index expression covers all bounds, except
      // for known constants
      for (auto pair : llvm::enumerate(op.getType().getShape())) {
        ValueOrInt_Polygeist val(pair.value());
        if (pair.value() == -1) {
          val = ValueOrInt_Polygeist(op.getOperands()[opn]);
          opn++;
        }
        if (storeIdxs[pair.index()].isValue) {
          Value auval = storeIdxs[pair.index()].v_val;
          BlockArgument bval = dyn_cast<BlockArgument>(auval);
          if (!bval) {
            LLVM_DEBUG(llvm::dbgs() << " + non bval expr " << bval << "\n");
            continue;
          }
          if (!rangeIncludes(bval, 0, val)) {
            LLVM_DEBUG(llvm::dbgs() << " + non in range " << bval << "\n");
            continue;
          }
        }
      }

      // Subselect all loads that match that constant to permit forwarding
      SmallVector<Operation **> validLoads;
      for (auto &ld : loads) {
        if (ld == nullptr)
          continue;
        bool legal = true;
        for (auto idxp : llvm::enumerate(storeIdxs)) {
          auto idx = idxp.value();
          auto i = idxp.index();
          if (!idx.isValue) {
            if (auto ald = dyn_cast<affine::AffineLoadOp>(ld)) {
              if (auto ac = ald.getAffineMap()
                                .getResult(i)
                                .dyn_cast<AffineConstantExpr>()) {
                if (idx == ac.getValue())
                  continue;
              }
              legal = false;
              break;
            } else if (auto mld = cast<memref::LoadOp>(ld)) {
              if (ValueOrInt_Polygeist(mld.getIndices()[i]) == idx.i_val)
                continue;
              legal = false;
              break;
            }
            legal = false;
            break;
          }
        }
        if (legal)
          validLoads.push_back(&ld);
      }

      // Early exit if no loads match the store.
      if (validLoads.size() == 0)
        continue;

      // 2) Ensure all operands in the map are direct, guaranteed to execute
      // parents
      //  (e.g. this is not contained within if statements and thus
      //  conditionally executed) note that even an intervening for statement
      //  can act as a conditional for 0 .. cond
      SmallPtrSet<Operation *, 1> boundContainers;
      SmallPtrSet<Value, 1> storeIdxSet;
      for (auto VI : storeIdxs) {
        if (!VI.isValue)
          continue;
        auto V = VI.v_val;
        auto BA = dyn_cast<BlockArgument>(V);
        if (!BA) {
          LLVM_DEBUG(llvm::dbgs() << " + non map oper " << V << "\n");
          return failure();
        }
        // Repeated indices mean that the space is not filled, but only
        // a diagonal.
        if (storeIdxSet.count(V))
          return failure();
        storeIdxSet.insert(V);
        Operation *parent = BA.getOwner()->getParentOp();

        // Ensure this is a for dimension, not an induction var.
        //  Also check other operands which may exist are positive Ranged.
        if (auto fOp = dyn_cast<scf::ForOp>(parent)) {
          if (BA.getArgNumber() != 0)
            return failure();
        } else if (auto fOp = dyn_cast<affine::AffineForOp>(parent)) {
          if (BA.getArgNumber() != 0)
            return failure();
        } else if (auto fOp = dyn_cast<scf::ParallelOp>(parent)) {
          if (BA.getArgNumber() >= fOp.getInductionVars().size())
            return failure();
          for (auto iv : fOp.getInductionVars())
            if (!rangeIncludes(iv, 0, 1))
              return failure();
        } else if (auto fOp = dyn_cast<affine::AffineParallelOp>(parent)) {
          if (BA.getArgNumber() >= fOp.getIVs().size())
            return failure();
          for (auto iv : fOp.getIVs())
            if (!rangeIncludes(iv, 0, 1))
              return failure();

        } else {
          return failure();
        }
        boundContainers.insert(parent);
      }
      boundContainers.insert(store);

      // Check containers all the way up to allocation op are guaranteed
      // to execute.
      Operation *prevParent = nullptr;
      for (auto c : boundContainers) {
        if (c->getParentOp() == op->getParentOp()) {
          prevParent = c;
        }
        Operation *cur = c->getParentOp();
        while (true) {
          if (cur->getParentOp() == op->getParentOp()) {
            prevParent = cur;
          }
          if (cur == op->getParentOp()) {
            break;
          }
          if (boundContainers.count(cur))
            break;
          if (auto fOp = dyn_cast<scf::ForOp>(cur)) {
            if (!rangeIncludes(fOp.getInductionVar(), 0, 1))
              return failure();
          } else if (auto fOp = dyn_cast<affine::AffineForOp>(cur)) {
            if (!rangeIncludes(fOp.getInductionVar(), 0, 1))
              return failure();
          } else if (auto fOp = dyn_cast<scf::ParallelOp>(cur)) {
            for (auto iv : fOp.getInductionVars())
              if (!rangeIncludes(iv, 0, 1))
                return failure();
          } else if (auto fOp = dyn_cast<affine::AffineParallelOp>(cur)) {
            for (auto iv : fOp.getIVs())
              if (!rangeIncludes(iv, 0, 1))
                return failure();

          } else {
            return failure();
          }
          cur = cur->getParentOp();
        }
      }
      assert(prevParent);

      // Check containers all the way up to allocation op are guaranteed
      // to execute.
      Operation *innerParent = store;
      for (auto VI : storeIdxs) {
        if (!VI.isValue)
          continue;
        auto V = VI.v_val;
        auto BA = dyn_cast<BlockArgument>(V);
        Operation *c = BA.getOwner()->getParentOp();
        if (isa<affine::AffineParallelOp>(c) || isa<scf::ParallelOp>(c)) {
          Operation *tmp = store;
          bool found = false;
          while (true) {
            for (auto now = tmp->getNextNode(); now; now = now->getNextNode()) {
              if (isa<enzymexla::BarrierOp>(now) &&
                  llvm::any_of(now->getOperands(),
                               [&](Value v) { return v == V; })) {
                found = true;
                c = now;
                break;
              }
            }
            if (found)
              break;
            tmp = tmp->getParentOp();
            if (tmp == c)
              break;
          }
        }

        if (innerParent == store)
          innerParent = c;
        else if (c->isAncestor(innerParent))
          innerParent = c;
      }
      assert(innerParent);

      DominanceInfo DI(op->getParentOp());

      std::function<bool(Operation * loc, SmallVectorImpl<Operation *> &)>
          canReplace = [&](Operation *loc,
                           SmallVectorImpl<Operation *> &toReplace) {
            SmallVector<Value> todoV = {storeVal};
            while (todoV.size()) {
              auto V = todoV.back();
              todoV.pop_back();
              if (storeIdxSet.count(V))
                continue;

              if (auto BA = dyn_cast<BlockArgument>(V)) {
                Operation *parent = BA.getOwner()->getParentOp();

                if (auto sop = storeVal.getDefiningOp())
                  if (sop->isAncestor(parent))
                    continue;

                if (!DI.dominates(BA, loc)) {
                  return false;
                }

                if (isa<FunctionOpInterface>(parent)) {
                } else if (auto fOp = dyn_cast<scf::ForOp>(parent)) {
                  if (BA.getArgNumber() != 0)
                    return false;
                } else if (auto fOp = dyn_cast<affine::AffineForOp>(parent)) {
                  if (BA.getArgNumber() != 0)
                    return false;
                } else if (auto fOp = dyn_cast<scf::ParallelOp>(parent)) {
                  if (BA.getArgNumber() >= fOp.getInductionVars().size())
                    return false;
                } else if (auto fOp =
                               dyn_cast<affine::AffineParallelOp>(parent)) {
                  if (BA.getArgNumber() >= fOp.getIVs().size())
                    return false;
                } else {
                  return false;
                }

              } else {
                auto vop = V.getDefiningOp();

                if (DI.dominates((Operation *)vop, loc) &&
                    DI.dominates((Operation *)vop, prevParent)) {
                  continue;
                }

                if (vop->getRegions().size()) {
                  if (!isa<scf::IfOp, scf::ExecuteRegionOp,
                           memref::AllocaScopeOp, affine::AffineIfOp>(vop))
                    return false;
                }

                SmallVector<MemoryEffects::EffectInstance> effects;
                collectEffects(vop, effects, /*barrier*/ false);
                for (auto res : effects) {
                  if (Value v = res.getValue()) {
                    if (!isa<MemoryEffects::Read>(res.getEffect()))
                      return false;
                    unsigned addr = 0;
                    if (auto MT = dyn_cast<MemRefType>(v.getType()))
                      addr = MT.getMemorySpaceAsInt();
                    else if (auto LT =
                                 dyn_cast<LLVM::LLVMPointerType>(v.getType()))
                      addr = LT.getAddressSpace();
                    else
                      return false;
                    mlir::Type TT = op->getResult(0).getType();
                    auto MT = cast<MemRefType>(TT);
                    if (addr != MT.getMemorySpaceAsInt())
                      return false;
                  } else
                    return false;
                }

                toReplace.push_back(vop);

                for (auto &region : vop->getRegions()) {
                  for (auto &block : region) {
                    for (auto &innerOp : block)
                      for (auto o : innerOp.getOperands())
                        todoV.push_back(o);
                  }
                }
                for (auto o : vop->getOperands()) {
                  todoV.push_back(o);
                }
              }
            }
            return true;
          };

      auto end = innerParent->getBlock()->end();
      for (auto iter = innerParent->getNextNode()->getIterator();;) {
        if (iter == end) {
          auto prev = iter;
          prev--;
          if (prev->getParentOp() == op->getParentOp())
            break;
          iter = prev->getParentOp()->getNextNode()->getIterator();
          end = prev->getParentOp()->getBlock()->end();
          continue;
        }
        auto &tc = *iter;
        iter++;

        SmallVector<Operation *, 1> toRedo;
        if (!canReplace(&tc, toRedo))
          break;

        SmallVector<MemoryEffects::EffectInstance> readResources;
        for (auto r : toRedo)
          collectEffects(r, readResources, /*barrier*/ false);

        bool legal = true;
        std::function<void(Operation *)> checkOverwritingOp =
            [&](Operation *ist) {
              if (auto AS = dyn_cast<affine::AffineStoreOp>(ist)) {
                if (AS.getMemRef() == op->getResult(0)) {
                  for (auto pair :
                       llvm::enumerate(AS.getAffineMap().getResults())) {
                    auto V = storeIdxs[pair.index()];
                    if (auto c = dyn_cast<AffineConstantExpr>(pair.value())) {
                      if (!V.isValue && V.i_val != c.getValue())
                        return;
                    }
                  }
                }
              }
              if (auto AS = dyn_cast<memref::StoreOp>(ist)) {
                if (AS.getMemRef() == op->getResult(0)) {
                  for (auto pair : llvm::enumerate(AS.getIndices())) {
                    auto V = storeIdxs[pair.index()];
                    auto V2 = ValueOrInt_Polygeist(pair.value());
                    if (!V.isValue && !V2.isValue && V.i_val != V2.i_val)
                      return;
                  }
                }
              }
              if (!legal)
                return;
              if (isa<enzymexla::BarrierOp>(ist))
                return;
              if (ist->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
                for (auto &region : ist->getRegions()) {
                  for (auto &block : region) {
                    for (auto &innerOp : block)
                      checkOverwritingOp(&innerOp);
                  }
                }
                return;
              }
              if (isReadOnly(ist))
                return;
              SmallVector<MemoryEffects::EffectInstance> effects;
              collectEffects(ist, effects, /*barrier*/ false);
              SmallVector<MemoryEffects::EffectInstance> readResourcesT(
                  readResources);
              if (ist != store) {
                // Overwriting the original buffer means that the reload is not
                // valid.
                readResourcesT.emplace_back(
                    ::mlir::MemoryEffects::Read::get(), op,
                    ::mlir::SideEffects::DefaultResource::get());
              }
              for (auto res : readResourcesT) {
                bool seenUse = false;
                if (Value v = res.getValue()) {
                  v = getBase(v);
                  if (isa<LLVM::CallOp, func::CallOp>(ist) &&
                      (isStackAlloca(v) ||
                       v.getDefiningOp<memref::AllocOp>()) &&
                      !isCaptured(v)) {
                    isCaptured(v, ist, &seenUse);
                    if (!seenUse)
                      continue;
                  }
                }
                for (auto effect : effects) {
                  if (!mayAlias(effect, res))
                    continue;
                  if (isa<MemoryEffects::Write>(effect.getEffect())) {
                    LLVM_DEBUG(llvm::dbgs() << " + stopping at " << tc
                                            << " due to " << *ist << "\n");
                    legal = false;
                    return;
                  }
                }
              }
            };

        {
          Operation *start = innerParent;
          while (start->getBlock() != tc.getBlock())
            start = start->getParentOp();
          assert(start);
          while (true) {
            checkOverwritingOp(start);
            if (start == &tc)
              break;
            start = start->getNextNode();
          }
        }

        if (!legal)
          break;

        for (auto ldp : validLoads) {
          Operation *&ld = *ldp;
          if (!ld)
            continue;
          if (tc.isAncestor(ld)) {
            rewriter.setInsertionPoint(ld);
            Value repval = storeVal;
            if (toRedo.size()) {
              IRMapping map;
              if (auto ald = dyn_cast<affine::AffineLoadOp>(ld)) {
                for (size_t i = 0; i < storeIdxs.size(); ++i) {
                  if (storeIdxs[i].isValue) {
                    auto apply = rewriter.create<affine::AffineApplyOp>(
                        ald.getLoc(),
                        ald.getAffineMapAttr().getValue().getSliceMap(i, 1),
                        ald.getMapOperands());
                    map.map(storeIdxs[i].v_val, apply->getResult(0));
                  }
                }
              } else {
                auto mld = cast<memref::LoadOp>(ld);
                for (size_t i = 0; i < storeIdxs.size(); ++i) {
                  if (storeIdxs[i].isValue) {
                    map.map(storeIdxs[i].v_val, mld.getIndices()[i]);
                  }
                }
              }
              Operation *torep = nullptr;
              SmallPtrSet<Operation *, 1> seen;
              for (auto rop : llvm::reverse(toRedo)) {
                if (seen.count(rop))
                  continue;
                seen.insert(rop);
                torep = rewriter.clone(*rop, map);
              }
              repval =
                  torep->getResult(cast<OpResult>().getResultNumber(storeVal));
            }
            Value vals[] = {repval};
            rewriter.setInsertionPoint(op);
            rewriter.replaceOp(ld, ValueRange(vals));
            ld = nullptr;
            changed = true;
          }
        }
      }
    }
    return success(changed);
  }
};

static llvm::cl::opt<bool>
    BufferElim("enable-buffer-elim", llvm::cl::init(true),
               llvm::cl::desc("Enable buffer elimination"));

void TypeAlignOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<
      TypeAlignCanonicalize, OrIExcludedMiddle, SelectOfExt,
      UndefProp<ExtUIOp, LLVM::UndefOp>, UndefProp<ExtUIOp, enzymexla::UndefOp>,
      UndefProp<ExtSIOp, LLVM::UndefOp>, UndefProp<ExtSIOp, enzymexla::UndefOp>,
      UndefProp<TruncIOp, LLVM::UndefOp>,
      UndefProp<TruncIOp, enzymexla::UndefOp>, CmpProp,
      UndefCmpProp<LLVM::UndefOp>, UndefCmpProp<enzymexla::UndefOp>,
      AlwaysAllocaScopeHoister<memref::AllocaScopeOp>,
      AlwaysAllocaScopeHoister<scf::ForOp>,
      AlwaysAllocaScopeHoister<affine::AffineForOp>, ConstantRankReduction,
      AffineIfSinking, AffineIfSimplification, CombineAffineIfs,
      MergeNestedAffineParallelLoops, PrepMergeNestedAffineParallelLoops,
      MergeNestedAffineParallelIf, RemoveAffineParallelSingleIter>(context);

  if (BufferElim) {
    results.insert<BufferElimination<memref::AllocaOp>,
                   BufferElimination<memref::AllocOp>,
                   AffineBufferElimination<memref::AllocaOp>,
                   AffineBufferElimination<memref::AllocOp>>(context);
  }

  results.insert<SimplifyDeadAllocV2<memref::AllocaOp>,
                 SimplifyDeadAllocV2<memref::AllocOp>,
                 SimplifyDeadAllocV2<LLVM::AllocaOp>, MulDivMul,
                 MergeParallelInductions,
                 // RankReduction<memref::AllocaOp, scf::ParallelOp>,
                 AggressiveAllocaScopeInliner, InductiveVarRemoval>(context);
}

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

    // if (castOp.getType().cast<MemRefType>().getShape().size() !=
    //     subindexOp.getType().cast<MemRefType>().getShape().size())
    if (cast<MemRefType>(castOp.getType()).getShape().size() !=
        cast<MemRefType>(subindexOp.getType()).getShape().size())
      return failure();
    // if (castOp.getType().cast<MemRefType>().getElementType() !=
    //     subindexOp.getResult().getType().cast<MemRefType>().getElementType())
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

    // auto mt0 = prevOp.getSource().getType().cast<MemRefType>();
    // auto mt1 = prevOp.getType().cast<MemRefType>();
    // auto mt2 = subViewOp.getType().cast<MemRefType>();
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
    // auto prev = subViewOp.getSource().getType().cast<MemRefType>();
    // auto post = subViewOp.getType().cast<MemRefType>();
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

// Simplify polygeist.subindex to memref.subview.
class SubToSubView final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    // auto srcMemRefType = op.getSource().getType().cast<MemRefType>();
    // auto resMemRefType = op.getResult().getType().cast<MemRefType>();
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
//   %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<?x1000xi32> %4 = "polygeist.subindex"(%3, %c0) :
//   (memref<?x1000xi32>, index) -> memref<1000xi32>
// simplifies to:
//   %4 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) ->
//   memref<1000xi32>

class RedundantDynSubIndex final : public OpRewritePattern<SubIndexOp> {
public:
  using OpRewritePattern<SubIndexOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIndexOp op,
                                PatternRewriter &rewriter) const override {
    auto srcOp = op.getSource().getDefiningOp<SubIndexOp>();
    if (!srcOp)
      return failure();

    // auto preMemRefType = srcOp.getSource().getType().cast<MemRefType>();
    // auto srcMemRefType = op.getSource().getType().cast<MemRefType>();
    // auto resMemRefType = op.getResult().getType().cast<MemRefType>();
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
          // if (subindex.getType().cast<MemRefType>().getShape().size() ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            // assert(subindex.getType().cast<MemRefType>().getShape().size() +
            //            1 ==
            //        subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size());
            assert(cast<MemRefType>(subindex.getType()).getShape().size() +
                       1 ==
                   cast<MemRefType>(subindex.getSource().getType()).getShape().size());
            indices.insert(indices.begin(), subindex.getIndex());
          }

          // assert(subindex.getSource()
          //            .getType()
          //            .cast<MemRefType>()
          //            .getShape()
          //            .size() == indices.size());
          assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(
              loadOp, subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          // if (subindex.getType().cast<MemRefType>().getShape().size() ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            // assert(subindex.getType().cast<MemRefType>().getShape().size() +
            //            1 ==
            //        subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size());
            assert(cast<MemRefType>(subindex.getType()).getShape().size() +
                       1 ==
                   cast<MemRefType>(subindex.getSource().getType()).getShape().size());
            indices.insert(indices.begin(), subindex.getIndex());
          }
          // assert(subindex.getSource()
          //            .getType()
          //            .cast<MemRefType>()
          //            .getShape()
          //            .size() == indices.size());
          assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.getValue(), subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::AtomicRMWOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          // if (subindex.getType().cast<MemRefType>().getShape().size() ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {
            assert(indices.size() > 0);
            indices[0] = rewriter.create<AddIOp>(subindex.getLoc(), indices[0],
                                                 subindex.getIndex());
          } else {
            // assert(subindex.getType().cast<MemRefType>().getShape().size() +
            //            1 ==
            //        subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size());
            assert(cast<MemRefType>(subindex.getType()).getShape().size() +
                       1 ==
                   cast<MemRefType>(subindex.getSource().getType()).getShape().size());
            indices.insert(indices.begin(), subindex.getIndex());
          }
          // assert(subindex.getSource()
          //            .getType()
          //            .cast<MemRefType>()
          //            .getShape()
          //            .size() == indices.size());
          assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::AtomicRMWOp>(
              storeOp, storeOp.getType(), storeOp.getKind(), storeOp.getValue(),
              subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.getIndex());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }

            // assert(subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size() == indices.size());
            assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.getValue(), subindex.getSource(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineLoadOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(subindex.getIndex());
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            // assert(subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size() == indices.size());
            assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
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

/// Simplify all uses of subindex, specifically
//    store subindex(x) = ...
//    affine.store subindex(x) = ...
//    load subindex(x)
//    affine.load subindex(x)
//    dealloc subindex(x)
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
          // if (subindex.getType().cast<MemRefType>().getShape().size() ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
              // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
              //     subindex.getSource()
              //         .getType()
              //         .cast<MemRefType>()
              //         .getShape()
              //         .size())
              if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                  cast<MemRefType>(subindex.getSource().getType()).getShape().size())
              indices.insert(indices.begin(), off);
            else {
              assert(indices.size() > 0);
              indices.erase(indices.begin());
            }
          }

          // assert(subindex.getSource()
          //            .getType()
          //            .cast<MemRefType>()
          //            .getShape()
          //            .size() == indices.size());
          assert(cast<MemRefType>(subindex.getSource().getType())
                .getShape().size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::LoadOp>(
              loadOp, subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          SmallVector<Value, 4> indices = storeOp.getIndices();
          // if (subindex.getType().cast<MemRefType>().getShape().size() ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size())
          if (cast<MemRefType>(subindex.getType()).getShape().size() ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {
            assert(indices.size() > 0);
            indices[0] =
                rewriter.create<AddIOp>(subindex.getLoc(), indices[0], off);
          } else {
            // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
            //     subindex.getSource()
            //         .getType()
            //         .cast<MemRefType>()
            //         .getShape()
            //         .size())
            if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
                cast<MemRefType>(subindex.getSource().getType()).getShape().size())
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

          // if (subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size() != indices.size()) {
          if (cast<MemRefType>(subindex.getSource().getType()).getShape().size() != indices.size()) {
            llvm::errs() << " storeOp: " << storeOp << " - subidx: " << subindex
                         << "\n";
          }
          // assert(subindex.getSource()
          //            .getType()
          //            .cast<MemRefType>()
          //            .getShape()
          //            .size() == indices.size());
          assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, storeOp.getValue(), subindex.getSource(), indices);
          changed = true;
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }

            // assert(subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size() == indices.size());
            assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
            rewriter.replaceOpWithNewOp<memref::StoreOp>(
                storeOp, storeOp.getValue(), subindex.getSource(), indices);
            changed = true;
          }
        }
      } else if (auto storeOp =
                     dyn_cast<affine::AffineLoadOp>(use.getOwner())) {
        if (storeOp.getMemref() == subindex) {
          // if (subindex.getType().cast<MemRefType>().getShape().size() + 1 ==
          //     subindex.getSource()
          //         .getType()
          //         .cast<MemRefType>()
          //         .getShape()
          //         .size()) {
          if (cast<MemRefType>(subindex.getType()).getShape().size() + 1 ==
              cast<MemRefType>(subindex.getSource().getType()).getShape().size()) {

            std::vector<Value> indices;
            auto map = storeOp.getAffineMap();
            indices.push_back(off);
            for (size_t i = 0; i < map.getNumResults(); i++) {
              auto apply = rewriter.create<affine::AffineApplyOp>(
                  storeOp.getLoc(), map.getSliceMap(i, 1),
                  storeOp.getMapOperands());
              indices.push_back(apply->getResult(0));
            }
            // assert(subindex.getSource()
            //            .getType()
            //            .cast<MemRefType>()
            //            .getShape()
            //            .size() == indices.size());
            assert(cast<MemRefType>(subindex.getSource().getType()).getShape().size() == indices.size());
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

    auto newSel = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.getSource(), cst2.getSource());

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

    auto newSel = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.getSource(), cst2.getSource());
    auto newIdx = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), op.getCondition(),
                                            cst1.getIndex(), cst2.getIndex());
    rewriter.replaceOpWithNewOp<SubIndexOp>(op, op.getType(), newSel, newIdx);
    return success();
  }
};

/// Simplify select subindex(x), subindex(y) to subindex(select x, y)
template <typename T> struct LoadSelect : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  static Value ptr(T op);
  static MutableOperandRange ptrMutable(T op);

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto mem0 = ptr(op);
    mlir::arith::SelectOp mem = dyn_cast_or_null<mlir::arith::SelectOp>(mem0.getDefiningOp());
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
    // if (castOp.getType().cast<MemRefType>().getElementType() ==
    //     getResult().getType().cast<MemRefType>().getElementType()) {
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