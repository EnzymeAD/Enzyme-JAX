//===- PolygeistOps.cpp - BFV dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <cassert>
#include <cmath>
#include <iterator>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <optional>
#include <set>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::arith;

namespace mlir {
namespace enzyme {

void commonLowerUpdateWithoutCorners(enzymexla::UpdateWithoutCornersOp extend,
                                     PatternRewriter &rewriter) {

  auto extendSharding = mlir::sdy::getSharding(extend);

  auto iotaX = stablehlo::IotaOp::create(
      rewriter, extend.getLoc(),
      RankedTensorType::get(extend.getType().getShape(), rewriter.getI32Type()),
      extend.getDimensionX());
  if (extendSharding)
    sdy::setSharding(iotaX, extendSharding);

  auto iotaY = stablehlo::IotaOp::create(
      rewriter, extend.getLoc(),
      RankedTensorType::get(extend.getType().getShape(), rewriter.getI32Type()),
      extend.getDimensionY());
  if (extendSharding)
    sdy::setSharding(iotaY, extendSharding);

  Value x1 = stablehlo::ConstantOp::create(
      rewriter, extend.getLoc(),
      SplatElementsAttr::get(iotaX.getType(),
                             rewriter.getI32IntegerAttr(extend.getX1())));

  Value x2 = stablehlo::ConstantOp::create(
      rewriter, extend.getLoc(),
      SplatElementsAttr::get(iotaX.getType(),
                             rewriter.getI32IntegerAttr(extend.getX2())));

  Value y1 = stablehlo::ConstantOp::create(
      rewriter, extend.getLoc(),
      SplatElementsAttr::get(iotaY.getType(),
                             rewriter.getI32IntegerAttr(extend.getY1())));

  Value y2 = stablehlo::ConstantOp::create(
      rewriter, extend.getLoc(),
      SplatElementsAttr::get(iotaY.getType(),
                             rewriter.getI32IntegerAttr(extend.getY2())));

  auto xCmp1 = stablehlo::CompareOp::create(
      rewriter, extend.getLoc(), iotaX, x1, stablehlo::ComparisonDirection::LT);
  if (extendSharding)
    sdy::setSharding(xCmp1, extendSharding);

  auto xCmp2 = stablehlo::CompareOp::create(
      rewriter, extend.getLoc(), iotaX, x2, stablehlo::ComparisonDirection::GE);
  if (extendSharding)
    sdy::setSharding(xCmp2, extendSharding);

  auto xVals = stablehlo::OrOp::create(rewriter, extend.getLoc(), xCmp1, xCmp2);
  if (extendSharding)
    sdy::setSharding(xVals, extendSharding);

  auto yCmp1 = stablehlo::CompareOp::create(
      rewriter, extend.getLoc(), iotaY, y1, stablehlo::ComparisonDirection::LT);
  if (extendSharding)
    sdy::setSharding(yCmp1, extendSharding);

  auto yCmp2 = stablehlo::CompareOp::create(
      rewriter, extend.getLoc(), iotaY, y2, stablehlo::ComparisonDirection::GE);
  if (extendSharding)
    sdy::setSharding(yCmp2, extendSharding);

  auto yVals = stablehlo::OrOp::create(rewriter, extend.getLoc(), yCmp1, yCmp2);
  if (extendSharding)
    sdy::setSharding(yVals, extendSharding);

  auto inCorner =
      stablehlo::AndOp::create(rewriter, extend.getLoc(), xVals, yVals);
  if (extendSharding)
    sdy::setSharding(inCorner, extendSharding);

  auto result =
      stablehlo::SelectOp::create(rewriter, extend.getLoc(), inCorner,
                                  extend.getOperand(), extend.getUpdate());
  if (extendSharding)
    sdy::setSharding(result, extendSharding);

  rewriter.replaceOp(extend, result);
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' it
/// could extract the effect information from the op, otherwise returns 'false'
/// and conservatively populates the list with all possible effects.
bool collectEffects(Operation *op,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                    bool ignoreBarriers) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<enzymexla::BarrierOp>(op))
    return true;

  // Ignore CacheLoads as they are already guaranteed to not have side effects
  // in the context of a parallel op, these only exist while we are in the
  // CPUifyPass
  // if (isa<enzymexla::CacheLoadOp>(op))
  //   return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  if (auto cop = dyn_cast<LLVM::CallOp>(op)) {
    if (auto callee = cop.getCallee()) {
      if (*callee == "scanf" || *callee == "__isoc99_scanf") {
        // Global read
        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());

        bool first = true;
        for (auto &arg : cop.getArgOperandsMutable()) {
          if (first)
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg);
          else
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          first = false;
        }

        return true;
      }
      if (*callee == "fscanf" || *callee == "__isoc99_fscanf") {
        // Global read
        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());

        for (auto &&[idx, arg] : llvm::enumerate(cop.getArgOperandsMutable())) {
          if (idx == 0) {
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          } else if (idx == 1) {
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          } else
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
        }

        return true;
      }
      if (*callee == "printf") {
        // Global read
        effects.emplace_back(
            MemoryEffects::Effect::get<MemoryEffects::Write>());
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
      if (*callee == "free") {
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Free::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
      if (*callee == "strlen") {
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
    }
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
  return false;
}

// Rethrns if we are non-conservative whether we have filled with all possible
// effects.
bool getEffectsBefore(Operation *op,
                      SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                      bool stopAtBarrier) {
  if (op != &op->getBlock()->front())
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<enzymexla::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects, /* ignoreBarriers */ true)) {
        return false;
      }
    }

  bool conservative = false;

  if (isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp()))
    return true;

  // As we didn't hit another barrier, we must check the predecessors of this
  // operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier)) {
    return false;
  }
  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  if (!isa<scf::IfOp, affine::AffineIfOp, memref::AllocaScopeOp>(
          op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects, /* ignoreBarriers */ true)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  return !conservative;
}
bool getEffectsAfter(Operation *op,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                     bool stopAtBarrier) {
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<enzymexla::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects, /* ignoreBarriers */ true))
        return false;
    }

  bool conservative = false;

  if (isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp()))
    return true;

  // As we didn't hit another barrier, we must check the predecessors of this
  // operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  if (!isa<scf::IfOp, affine::AffineIfOp, memref::AllocaScopeOp>(
          op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects, /* ignoreBarriers */ true)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

bool isCaptured(Value v, Operation *potentialUser = nullptr,
                bool *seenuse = nullptr) {
  SmallVector<Value> todo = {v};
  while (todo.size()) {
    Value v = todo.pop_back_val();
    for (auto u : v.getUsers()) {
      if (seenuse && u == potentialUser)
        *seenuse = true;
      if (isa<memref::LoadOp, LLVM::LoadOp, affine::AffineLoadOp>(u))
        continue;
      // if (isa<polygeist::CacheLoad>(u)) continue
      if (auto s = dyn_cast<memref::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<affine::AffineStoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<LLVM::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto sub = dyn_cast<LLVM::GEPOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::BitcastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::AddrSpaceCastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<func::ReturnOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemsetOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemcpyOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemmoveOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<memref::CastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<memref::DeallocOp>(u)) {
        continue;
      }
      // if (auto sub = dyn_cast<polygeist::SubIndexOp>(u)) {
      //   todo.push_back(sub);
      // }
      if (auto sub = dyn_cast<enzymexla::Memref2PointerOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<enzymexla::Pointer2MemrefOp>(u)) {
        todo.push_back(sub);
      }
      if (auto cop = dyn_cast<LLVM::CallOp>(u)) {
        if (auto callee = cop.getCallee()) {
          if (getNonCapturingFunctions().count(callee->str()))
            continue;
        }
      }
      if (auto cop = dyn_cast<func::CallOp>(u)) {
        if (getNonCapturingFunctions().count(cop.getCallee().str()))
          continue;
      }
      return true;
    }
  }

  return false;
}

Value getBase(Value v) {
  while (true) {
    // if (auto s = v.getDefiningOp<SubIndexOp>()) {
    //   v = s.getSource();
    //   continue;
    // }
    if (auto s = v.getDefiningOp<enzymexla::Memref2PointerOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
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

bool isStackAlloca(Value v) {
  return v.getDefiningOp<memref::AllocaOp>() ||
         v.getDefiningOp<memref::AllocOp>() ||
         v.getDefiningOp<LLVM::AllocaOp>();
}

bool mayWriteTo(Operation *op, Value val, bool ignoreBarrier) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (mayWriteTo(&nestedOp, val, ignoreBarrier))
            return true;
      }
    }
    return false;
  }

  if (ignoreBarrier && isa<enzymexla::BarrierOp>(op))
    return false;

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    for (auto it : effects) {
      if (!isa<MemoryEffects::Write>(it.getEffect()))
        continue;
      if (mayAlias(it, val))
        return true;
    }
    return false;
  }

  // Calls which do not use a derived pointer of a known alloca, which is not
  // captured can not write to said memory.
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    auto base = getBase(val);
    bool seenuse = false;
    if (isStackAlloca(base) && !isCaptured(base, op, &seenuse) && !seenuse) {
      return false;
    }
  }
  return true;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty) {
  Ty = getElementTypeOrSelf(Ty);
  if (Ty.isInteger())
    return true;
  return allowOnFloatingPointMath;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty,
                          mlir::Operation *op, PatternRewriter &rewriter) {
  Ty = getElementTypeOrSelf(Ty);
  if (Ty.isInteger())
    return true;
  return allowOnFloatingPointMath || guaranteedNoNanResult(op, rewriter);
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy,
                          Type inTy) {
  outTy = getElementTypeOrSelf(outTy);
  inTy = getElementTypeOrSelf(inTy);
  if (outTy.isInteger() && inTy.isInteger())
    return true;
  return allowOnFloatingPointMath;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy, Type inTy,
                          mlir::Operation *op, PatternRewriter &rewriter) {
  outTy = getElementTypeOrSelf(outTy);
  inTy = getElementTypeOrSelf(inTy);
  if (outTy.isInteger() && inTy.isInteger())
    return true;
  return allowOnFloatingPointMath || guaranteedNoNanResult(op, rewriter);
}

bool canApplySymmetricPattern(mlir::Operation *op, PatternRewriter &rewriter) {
  return guaranteedSymmetricResult(op, rewriter);
}
bool canApplySymmetricPattern(Value val, PatternRewriter &rewriter) {
  return guaranteedSymmetricResult(val, rewriter);
}

SymmetricResultAnalysis initSymmetricResultAnalysis() {
  return SymmetricResultAnalysis();
}

bool checkNotEqual(APInt a, APInt b) { return a != b; }

bool checkNotEqual(APFloat a, APFloat b) {
  return a.compare(b) != llvm::APFloat::cmpEqual;
}

template <typename Ty> bool checkConstantSymmetric(DenseElementsAttr attr) {
  if (!attr)
    return false;

  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type)
    return false;

  if (type.getRank() == 0)
    return true;
  if (type.getRank() != 2)
    return false;

  auto shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  if (rows != cols)
    return false;
  if (attr.isSplat())
    return true;

  auto values = attr.getValues<Ty>();
  auto it = values.begin();

  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = i + 1; j < cols; j++) {
      auto a = *(it + i * cols + j);
      auto b = *(it + j * cols + i);
      if (checkNotEqual(a, b))
        return false;
    }
  }

  return true;
}

bool SymmetricResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  return checkConstantSymmetric<APInt>(attr);
}

bool SymmetricResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  return checkConstantSymmetric<APFloat>(attr);
}

SymmetricResultAnalysis::State SymmetricResultAnalysis::localGuaranteed(
    Value val, SmallVectorImpl<Value> &localtodo, PatternRewriter &rewriter) {
  auto valTy = cast<RankedTensorType>(val.getType());
  if (valTy.getRank() != 2)
    return State::NOTGUARANTEED; // this pass only checks for symmetric matrices
  if (valTy.getDimSize(0) != valTy.getDimSize(1))
    return State::NOTGUARANTEED; // quick check and exit

  SplatElementsAttr splatAttr;
  if (matchPattern(val, m_Constant(&splatAttr))) {
    return State::GUARANTEED;
  }

  auto op = val.getDefiningOp();
  if (!op)
    return State::NOTGUARANTEED;

  if (isa<enzymexla::SyrkOp>(op))
    return State::GUARANTEED;

  // check that transpose dimensions are [1,0]
  auto isTrueTranspose = [](stablehlo::TransposeOp tOp) -> bool {
    auto perm = tOp.getPermutation();
    return perm.size() == 2 && perm[0] == 1 && perm[1] == 0;
  };

  if (auto broadcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    auto operand = broadcastOp.getOperand();
    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto dims = broadcastOp.getBroadcastDimensions();
    if (operandTy.getRank() == 0 && dims.empty()) {
      return State::GUARANTEED;
    }
  }

  // commutative operation with A and A^T will always be symmetric
  // op(A, A^T) will also always be symmetric
  if (stablehlo::hasTraitElementwise(op) &&
      (op->hasTrait<OpTrait::IsCommutative>() ||
       op->hasTrait<hlo::OpTrait::IsCommutative>())) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    // op(A, A^T)
    if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(rhsT)) {
        if (lhs == rhsT.getOperand()) {
          return State::GUARANTEED;
        }
      }
    }

    // op(A^T, A)
    if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(lhsT)) {
        if (rhs == lhsT.getOperand()) {
          return State::GUARANTEED;
        }
      }
    }
  }

  // A x (A^T) / (A^T) x A will always be symmetric
  if (auto dotOp = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    auto dotDimNumbers = dotOp.getDotDimensionNumbers();
    auto lhs = dotOp.getLhs();
    auto rhs = dotOp.getRhs();

    auto lhsCDims = dotDimNumbers.getLhsContractingDimensions();
    auto rhsCDims = dotDimNumbers.getRhsContractingDimensions();

    if (dotDimNumbers.getLhsBatchingDimensions().size() == 0 &&
        dotDimNumbers.getRhsBatchingDimensions().size() == 0 &&
        lhsCDims.size() == 1 && rhsCDims.size() == 1) {
      if (lhs == rhs && lhsCDims[0] == rhsCDims[0]) {
        return State::GUARANTEED;
      }

      if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
        if (isTrueTranspose(lhsT) && lhsT.getOperand() == rhs &&
            lhsCDims[0] == 1 - rhsCDims[0]) {
          return State::GUARANTEED;
        }
      }

      if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
        if (isTrueTranspose(rhsT) && rhsT.getOperand() == lhs &&
            lhsCDims[0] == 1 - rhsCDims[0]) {
          return State::GUARANTEED;
        }
      }
    }
  }

  // diagonal implies symmetric
  if (auto scatterOp = dyn_cast<stablehlo::ScatterOp>(op)) {
    Value outValues;
    // TODO: technically we only need that the scatter indices touch exclusively
    // the diagonal indices
    if (detectDiagonalTensor(scatterOp, &outValues, [](auto scatterInput) {
          SplatElementsAttr attr;
          return matchPattern(scatterInput, m_Constant(&attr));
        }).ok()) {
      return State::GUARANTEED;
    }
  }

  bool recursiveCheck = false;

  // elementwise ops
  if (stablehlo::hasTraitElementwise(op)) {
    recursiveCheck = true;
  }

  /**
   * TODO
   * - check if its * 0 -> symmetric
   */

  if (recursiveCheck) {
    bool allOperandsGuaranteed = true;
    for (auto operand : op->getOperands()) {
      {
        auto found = valueCache.find(operand);
        if (found != valueCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      localtodo.push_back(operand);
      allOperandsGuaranteed = false;
    }

    if (allOperandsGuaranteed)
      return State::GUARANTEED;
    else
      return State::PENDING;
  } else {
    return State::NOTGUARANTEED;
  }
}

NoNanResultAnalysis initNoNanResultAnalysis() {
  auto finiteAnalysis = std::make_shared<FiniteResultAnalysis>();
  auto noNanAnalysis = std::make_shared<NoNanResultAnalysis>();
  noNanAnalysis->setFiniteResultAnalysis(finiteAnalysis);
  finiteAnalysis->setNoNanResultAnalysis(noNanAnalysis);
  return *noNanAnalysis;
}

bool NoNanResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  return true;
}

bool NoNanResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (elem.isNaN())
      return false;
  }
  return true;
}

NoNanResultAnalysis::State NoNanResultAnalysis::localGuaranteed(
    Value val, SmallVectorImpl<Value> &localtodo, PatternRewriter &rewriter) {
  auto op = val.getDefiningOp();
  if (!op)
    return State::NOTGUARANTEED;

  // integer ops
  if (isa<stablehlo::AndOp, stablehlo::OrOp, stablehlo::XorOp, stablehlo::NotOp,
          stablehlo::IotaOp>(op)) {
    return State::GUARANTEED;
  }

  bool recursiveCheck = false;
  SmallVector<Value> operandsToCheck;

  if (isa<stablehlo::SliceOp, stablehlo::ConcatenateOp,
          stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp>(op)) {
    // data movement ops
    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::AbsOp, stablehlo::ExpOp, stablehlo::ConvertOp,
                 stablehlo::CompareOp, stablehlo::TanhOp, stablehlo::LogisticOp,
                 stablehlo::FloorOp, stablehlo::CeilOp>(op)) {
    // elementwise ops that are no-nan if all operands are not nan
    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::AddOp, stablehlo::SubtractOp>(op)) {
    // If any one of the operands is a Inf, the result is Inf. If both are Inf,
    // the result is NaN.
    auto lhsFinite =
        finiteResultAnalysis->guaranteed(op->getOperand(0), rewriter);
    auto rhsFinite =
        finiteResultAnalysis->guaranteed(op->getOperand(1), rewriter);

    if (lhsFinite && rhsFinite) {
      return State::GUARANTEED;
    }

    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::SineOp, stablehlo::CosineOp>(op)) {

    if (!finiteResultAnalysis->guaranteed(op->getOperand(0), rewriter)) {
      return State::NOTGUARANTEED;
    }

    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    // if lhs is Inf & rhs is 0 or the other way around, mul is going to be NaN

    // TODO: If one is inf check if the other is zero. We can significantly
    // relax this check if we can prove that the other is not zero.
    if (!finiteResultAnalysis->guaranteed(mulOp.getLhs(), rewriter) ||
        !finiteResultAnalysis->guaranteed(mulOp.getRhs(), rewriter)) {
      return State::NOTGUARANTEED;
    }

    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(1));
    operandsToCheck.push_back(op->getOperand(2));
  } else if (isa<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp>(
                 op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(0));
  }

  if (recursiveCheck) {
    return recursivelyCheckOperands(localtodo, operandsToCheck, true);
  } else {
    return State::NOTGUARANTEED;
  }
}

FiniteResultAnalysis initFiniteResultAnalysis() {
  auto noNanAnalysis = std::make_shared<NoNanResultAnalysis>();
  auto finiteAnalysis = std::make_shared<FiniteResultAnalysis>();
  noNanAnalysis->setFiniteResultAnalysis(finiteAnalysis);
  finiteAnalysis->setNoNanResultAnalysis(noNanAnalysis);
  return *finiteAnalysis;
}

bool FiniteResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (!elem.isFinite())
      return false;
  }
  return true;
}

bool FiniteResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  return true;
}

FiniteResultAnalysis::State FiniteResultAnalysis::localGuaranteed(
    Value val, SmallVectorImpl<Value> &localtodo, PatternRewriter &rewriter) {
  auto op = val.getDefiningOp();
  if (!op)
    return State::NOTGUARANTEED;

  // integer ops
  if (isa<stablehlo::AndOp, stablehlo::OrOp, stablehlo::XorOp, stablehlo::NotOp,
          stablehlo::IotaOp>(op)) {
    return State::GUARANTEED;
  }

  bool recursiveCheck = false;
  SmallVector<Value> operandsToCheck;

  if (isa<stablehlo::SliceOp, stablehlo::ConcatenateOp,
          stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp>(op)) {
    // data movement ops
    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp,
                 stablehlo::AbsOp, stablehlo::ExpOp, stablehlo::ConvertOp,
                 stablehlo::CompareOp>(op)) {
    // if both finite [but possibly nan], the result is finite, or nan
    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::TanhOp, stablehlo::LogisticOp, stablehlo::SineOp,
                 stablehlo::CosineOp>(op)) {
    // guaranteed finite or nan result, always
    return State::GUARANTEED;
  } else if (isa<stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(1));
    operandsToCheck.push_back(op->getOperand(2));
  } else if (isa<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp>(
                 op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(0));
  }

  if (recursiveCheck) {
    return recursivelyCheckOperands(localtodo, operandsToCheck, true);
  } else {
    return State::NOTGUARANTEED;
  }
}

bool NonNegativeResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APInt>()) {
    if (elem.isNegative())
      return false;
  }
  return true;
}

bool NonNegativeResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (elem.isNegative())
      return false;
  }
  return true;
}

NonNegativeResultAnalysis::State NonNegativeResultAnalysis::localGuaranteed(
    Value val, SmallVectorImpl<Value> &localtodo, PatternRewriter &rewriter) {
  auto op = val.getDefiningOp();
  if (!op)
    return State::NOTGUARANTEED;

  // integer ops
  if (isa<stablehlo::AbsOp, stablehlo::SqrtOp, stablehlo::ExpOp,
          stablehlo::IotaOp, stablehlo::AndOp, stablehlo::OrOp,
          stablehlo::XorOp, stablehlo::NotOp>(op)) {
    return State::GUARANTEED;
  }

  if (isa<chlo::ErfInvOp>(op)) {
    return State::NOTGUARANTEED;
  }

  // Any non-negative operation that produces a non-negative result
  // Here we recur on the rhs, as that is more likely to be a constant.
  if (isa<stablehlo::MaxOp>(op)) {
    if (guaranteed(op->getOperand(1), rewriter)) {
      return State::GUARANTEED;
    }

    auto operand = op->getOperand(0);

    {
      auto found = valueCache.find(operand);
      if (found != valueCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    localtodo.push_back(operand);
    return State::PENDING;
  }

  // (mul a a) is always non-negative
  if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    auto lhsOp = mulOp.getLhs().getDefiningOp();
    auto rhsOp = mulOp.getRhs().getDefiningOp();

    if (lhsOp == rhsOp) {
      return State::GUARANTEED;
    }
  }

  if (auto clampOp = dyn_cast<stablehlo::ClampOp>(op)) {
    // Clamp is non-negative if the min operand is non-negative
    auto operand = clampOp.getMin();

    {
      auto found = valueCache.find(operand);
      if (found != valueCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    localtodo.push_back(operand);
    return State::PENDING;
  }

  bool recursiveCheck = false;
  SmallVector<Value> operandsToCheck;

  if (isa<stablehlo::MinOp, stablehlo::AddOp, stablehlo::MulOp,
          stablehlo::ConcatenateOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp, stablehlo::SliceOp,
          stablehlo::BroadcastInDimOp>(op)) {
    // All non-negative operations that produce a non-negative result
    recursiveCheck = true;
    operandsToCheck.append(op->getOperands().begin(), op->getOperands().end());
  } else if (isa<stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(1));
    operandsToCheck.push_back(op->getOperand(2));
  } else if (isa<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp>(
                 op)) {
    recursiveCheck = true;
    operandsToCheck.push_back(op->getOperand(0));
  }

  if (recursiveCheck) {
    return recursivelyCheckOperands(localtodo, operandsToCheck, false);
  } else {
    return State::NOTGUARANTEED;
  }
}

bool anyOperandIsConstant(mlir::Operation *op) {
  DenseElementsAttr attr;
  for (auto operand : op->getOperands()) {
    if (matchPattern(operand, m_Constant(&attr)))
      return true;
  }
  return false;
}

bool allOperandsAreConstant(mlir::Operation *op) {
  DenseElementsAttr attr;
  for (auto operand : op->getOperands()) {
    if (!matchPattern(operand, m_Constant(&attr)))
      return false;
  }
  return true;
}

SmallVector<int64_t> findReshapeInsertionDims(RankedTensorType inputType,
                                              RankedTensorType outputType) {
  return findReshapeInsertionDims(inputType.getShape(), outputType.getShape());
}

SmallVector<int64_t> findReshapeInsertionDims(ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> outputShape) {
  if (inputShape.size() >= outputShape.size())
    return {}; // trivial no insertion case

  SmallVector<int64_t> insertionDims;
  size_t inputDimIndex = 0;

  for (size_t i = 0; i < outputShape.size(); ++i) {
    auto dim = outputShape[i];
    if (inputDimIndex < inputShape.size() && dim == inputShape[inputDimIndex]) {
      ++inputDimIndex;
    } else if (dim == 1 && (inputDimIndex >= inputShape.size() ||
                            dim != inputShape[inputDimIndex])) {
      // Singleton dimension inserted by reshape.
      insertionDims.push_back(i);
    } else {
      // Reshape modifies existing dimensions, which we don't handle here.
      return {};
    }
  }

  // If we haven't seen all of the input dimensions, we don't have a valid
  // insertion point.
  if (inputDimIndex != inputShape.size())
    return {};

  return insertionDims;
}

bool isInsertDimOp(stablehlo::ReshapeOp reshapeOp) {
  RankedTensorType inputTy = reshapeOp.getOperand().getType();
  RankedTensorType outputTy = reshapeOp.getType();
  auto insertDims = findReshapeInsertionDims(inputTy, outputTy);
  return !insertDims.empty();
}

bool isDeleteDimOp(stablehlo::ReshapeOp reshapeOp) {
  RankedTensorType inputTy = reshapeOp.getOperand().getType();
  RankedTensorType outputTy = reshapeOp.getType();
  auto deleteDims = findReshapeInsertionDims(outputTy, inputTy);
  return !deleteDims.empty();
}

void getSingletonInsertionDims(stablehlo::BroadcastInDimOp bcastOp,
                               SmallVectorImpl<int64_t> &insertionDims) {
  RankedTensorType outputTy = bcastOp.getType();

  for (size_t i = 0; i < outputTy.getRank(); ++i) {
    if (llvm::is_contained(bcastOp.getBroadcastDimensions(), i)) {
      continue;
    }
    if (outputTy.getDimSize(i) == 1) {
      insertionDims.push_back(i);
    }
  }
}

bool areValidInsertionDims(RankedTensorType inputType,
                           RankedTensorType outputType,
                           SmallVector<int64_t> insertionDims) {
  if (insertionDims.size() != outputType.getRank() - inputType.getRank())
    return false;

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  for (auto dim : insertionDims) {
    if (dim >= outputType.getRank() || dim < 0 || outputShape[dim] != 1)
      return false;
  }

  SmallVector<int64_t> outShapeAfterDeletion;
  outShapeAfterDeletion.reserve(outputShape.size() - insertionDims.size());
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (llvm::is_contained(insertionDims, i))
      continue;
    outShapeAfterDeletion.push_back(outputShape[i]);
  }

  for (auto [inDim, outDim] : llvm::zip(inputShape, outShapeAfterDeletion)) {
    if (inDim != outDim)
      return false;
  }

  return true;
}

bool getCollapsingMapping(ArrayRef<int64_t> oldShape,
                          ArrayRef<int64_t> newShape,
                          DenseMap<int64_t, SmallVector<int64_t, 2>> &mapping) {
  if (newShape.size() >= oldShape.size()) {
    return false;
  }
  if (newShape.empty()) {
    return llvm::product_of(oldShape) == 1;
  }

  size_t oldIdx = 0;
  for (size_t newIdx = 0; newIdx < newShape.size(); ++newIdx) {
    int64_t target = newShape[newIdx];
    int64_t current = 1;
    bool consumed = false;
    SmallVector<int64_t, 2> group;

    while (oldIdx < oldShape.size()) {
      if (consumed && current == target) {
        if (newIdx == newShape.size() - 1) {
          if (oldShape[oldIdx] == 1) {
            // keep going to consume trailing dims
          } else {
            break;
          }
        } else {
          break;
        }
      }

      current *= oldShape[oldIdx];
      group.push_back(oldIdx);
      consumed = true;
      oldIdx++;

      if (current > target) {
        return false;
      }
    }

    if (current != target || !consumed) {
      return false;
    }
    mapping[newIdx] = group;
  }

  return oldIdx == oldShape.size();
}

bool isOnlyUsedInOperation(Operation *operation, Operation *parentOp) {
  if (!operation || !parentOp)
    return false;

  for (Operation *user : operation->getUsers()) {
    if (user != parentOp)
      return false;
  }
  return true;
}

bool isValueOnlyUsedInOperation(Value value, Operation *parentOp) {
  if (!parentOp)
    return false;

  for (Operation *user : value.getUsers()) {
    if (user != parentOp)
      return false;
  }
  return true;
}

bool mayReadFrom(Operation *op, Value val) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (mayReadFrom(&nestedOp, val))
            return true;
      }
    }
    return false;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    for (auto it : effects) {
      if (!isa<MemoryEffects::Read>(it.getEffect()))
        continue;
      if (mayAlias(it, val))
        return true;
    }
    return false;
  }
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    auto base = getBase(val);
    bool seenuse = false;
    if (isStackAlloca(base) && !isCaptured(base, op, &seenuse) && !seenuse) {
      return false;
    }
  }
  return true;
}

RankedTensorType removeBatchedDims(RankedTensorType Ty,
                                   ArrayRef<int64_t> dims) {
  SmallVector<int64_t> newShape;
  for (auto i : llvm::enumerate(Ty.getShape())) {
    if (!llvm::is_contained(dims, i.index())) {
      newShape.push_back(i.value());
    }
  }
  return RankedTensorType::get(newShape, Ty.getElementType());
}

enzymexla::LapackTranspose
transposeLapackTranspose(enzymexla::LapackTranspose trans, bool canBeComplex) {
  switch (trans) {
  case enzymexla::LapackTranspose::none:
    return enzymexla::LapackTranspose::transpose;
  case enzymexla::LapackTranspose::transpose:
    return enzymexla::LapackTranspose::none;
  case enzymexla::LapackTranspose::adjoint:
    assert(!canBeComplex &&
           "cannot trivially tranpose adjoint of complex numbers");
    return enzymexla::LapackTranspose::none;
  }
  llvm_unreachable("Unknown LapackTranspose");
}

enzymexla::LapackUplo transposeLapackUplo(enzymexla::LapackUplo uplo) {
  switch (uplo) {
  case enzymexla::LapackUplo::F:
    return enzymexla::LapackUplo::F;
  case enzymexla::LapackUplo::L:
    return enzymexla::LapackUplo::U;
  case enzymexla::LapackUplo::U:
    return enzymexla::LapackUplo::L;
  }
  llvm_unreachable("Unknown LapackUplo");
}

enzymexla::LapackUplo standardizeUplo(enzymexla::LapackUplo uplo) {
  switch (uplo) {
  case enzymexla::LapackUplo::F:
    return enzymexla::LapackUplo::U;
  default:
    return uplo;
  }
}

absl::Status detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                             bool allowedMultipleUses,
                                             InputValidatorFn inputValidator) {
  SplatElementsAttr constSetIndexValue = nullptr;
  return detectConstantSetindexScatterOp(scatterOp, allowedMultipleUses,
                                         inputValidator, constSetIndexValue);
}

absl::Status detectConstantSetindexScatterOp(
    stablehlo::ScatterOp scatterOp, bool allowedMultipleUses,
    InputValidatorFn inputValidator, SplatElementsAttr &constSetIndexValue) {
  if (scatterOp.getInputs().size() != 1) {
    return absl::UnimplementedError(
        "Detection not implemented for scatter op with >1 input.");
  }

  if (!scatterOp.getResult(0).hasOneUse() && !allowedMultipleUses) {
    return absl::InvalidArgumentError(
        "ScatterOp has multiple uses, not supported.");
  }

  auto checkCommonScatterOp = mlir::stablehlo::CheckCommonScatterOp(scatterOp);

  using ScatterOpKind = mlir::stablehlo::ScatterOpKind;
  if (checkCommonScatterOp.kind != ScatterOpKind::Setindex &&
      checkCommonScatterOp.kind != ScatterOpKind::ConstantSetindex) {
    return absl::InvalidArgumentError("ScatterOp is not a setindex op.");
  }

  if (checkCommonScatterOp.kind == ScatterOpKind::ConstantSetindex) {
    constSetIndexValue = checkCommonScatterOp.constant;
  }

  auto input = scatterOp.getInputs()[0];
  if (inputValidator(input)) {
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "Scatter Op is not a constant setindex op.");
}

// TODO: detect batched diagonal tensors
absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates,
                                  InputValidatorFn inputValidator) {
  auto status =
      detectConstantSetindexScatterOp(scatterOp, true, inputValidator);
  if (!status.ok())
    return status;

  auto input = scatterOp.getInputs()[0];
  auto inputShape = cast<ShapedType>(input.getType()).getShape();
  // TODO: support the non-square case
  if (inputShape.size() != 2 || inputShape[0] != inputShape[1])
    return absl::InvalidArgumentError("Input is not a diagonal tensor.");

  auto indices = scatterOp.getScatterIndices();
  auto indicesShape = cast<ShapedType>(indices.getType()).getShape();
  if (indicesShape.size() != 2 || indicesShape[0] != inputShape[0] ||
      indicesShape[1] != 2)
    return absl::InvalidArgumentError("Indices are not for a diagonal tensor.");

  auto updates = scatterOp.getUpdates()[0];
  if (cast<RankedTensorType>(updates.getType()).getRank() != 1)
    return absl::InvalidArgumentError("Updates are not a vector.");

  auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();
  auto validScatterDimNumbers = stablehlo::ScatterDimensionNumbersAttr::get(
      scatterOp.getContext(), ArrayRef<int64_t>(), ArrayRef<int64_t>({0, 1}),
      ArrayRef<int64_t>(), ArrayRef<int64_t>(), ArrayRef<int64_t>({0, 1}), 1);
  if (scatterDimNumbers != validScatterDimNumbers)
    return absl::InvalidArgumentError(
        "Scatter dimension numbers are not valid for a diagonal tensor.");

  auto isIotaLikeTensor = detectIotaLikeTensor(indices);
  if (isIotaLikeTensor) {
    auto iotaLikeTensor = isIotaLikeTensor.value();
    if (iotaLikeTensor.dimension == 0 && isZeroAttr(iotaLikeTensor.start) &&
        isOneAttr(iotaLikeTensor.scale)) {
      *outUpdates = updates;
      return absl::OkStatus();
    }
  }

  return absl::InvalidArgumentError("Not a diagonal tensor.");
}

absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp) {
  mlir::Value updates;
  return detectDiagonalTensor(scatterOp, &updates);
}

absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates) {
  return detectDiagonalTensor(scatterOp, outUpdates, [](mlir::Value input) {
    return matchPattern(input, m_AnyZeroFloat()) ||
           matchPattern(input, m_Zero());
  });
}

namespace {

// Helper to get a "one" value for APFloat/APInt
template <typename T> T getOneValue(T example);

template <> APFloat getOneValue<APFloat>(APFloat example) {
  return APFloat(example.getSemantics(), 1);
}

template <> APInt getOneValue<APInt>(APInt example) {
  return APInt(example.getBitWidth(), (int64_t)1, /*isSigned=*/true);
}

// Helper to multiply T by an int64_t index
template <typename T> T multiplyByIndex(T value, int64_t index);

template <> APFloat multiplyByIndex<APFloat>(APFloat value, int64_t index) {
  APFloat indexFloat(value.getSemantics());
  indexFloat.convertFromAPInt(APInt(64, index, /*isSigned=*/true),
                              /*isSigned=*/true, APFloat::rmNearestTiesToEven);
  value.multiply(indexFloat, APFloat::rmNearestTiesToEven);
  return value;
}

template <> APInt multiplyByIndex<APInt>(APInt value, int64_t index) {
  APInt indexInt(value.getBitWidth(), index, /*isSigned=*/true);
  return value * indexInt;
}

// Helper for exact comparison
template <typename T> bool valuesAreEqual(T lhs, T rhs);

template <> bool valuesAreEqual<APFloat>(APFloat lhs, APFloat rhs) {
  return lhs.bitwiseIsEqual(rhs);
}

template <> bool valuesAreEqual<APInt>(APInt lhs, APInt rhs) {
  return lhs == rhs;
}

// Helper to create TypedAttr from APFloat/APInt
template <typename T>
TypedAttr createAttrFromValue(MLIRContext *ctx, Type elemType, T value);

template <>
TypedAttr createAttrFromValue<APFloat>(MLIRContext *ctx, Type elemType,
                                       APFloat value) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    // Convert APFloat to the target semantics if needed
    bool losesInfo = false;
    value.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
    return FloatAttr::get(floatType, value);
  }
  return nullptr;
}

template <>
TypedAttr createAttrFromValue<APInt>(MLIRContext *ctx, Type elemType,
                                     APInt value) {
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    // Extend or truncate to match the target bitwidth
    if (value.getBitWidth() != intType.getWidth()) {
      value = value.sextOrTrunc(intType.getWidth());
    }
    return IntegerAttr::get(intType, value);
  }
  return nullptr;
}

template <typename T>
std::optional<IotaLikeTensor>
detectIotaLikeTensorImpl(DenseElementsAttr denseAttr) {
  auto constType = dyn_cast<RankedTensorType>(denseAttr.getType());
  if (!constType) {
    return std::nullopt;
  }
  auto shape = constType.getShape();
  auto elemType = constType.getElementType();
  MLIRContext *ctx = denseAttr.getContext();

  auto strides = computeStrides(shape);

  auto values = denseAttr.getValues<T>();
  int64_t numElements = constType.getNumElements();

  for (int64_t dim = 0; dim < constType.getRank(); dim++) {
    bool isIotaAlongDim = true;
    std::optional<T> detectedStart;
    std::optional<T> detectedScale;

    SmallVector<int64_t> indices;

    for (int64_t idx = 0; idx < numElements && isIotaAlongDim; idx++) {
      linearToMultiIndex(idx, strides, indices);

      T actualValue = values[idx];

      if (!detectedStart) {
        detectedStart = actualValue;
      } else if (!detectedScale && indices[dim] == 1) {
        // Detect scale from the second element along this dimension
        detectedScale = actualValue - detectedStart.value();
        if (detectedScale.value().isZero()) {
          // Scale of 0 means all values are the same, not an iota
          isIotaAlongDim = false;
          break;
        }
      }

      // Compute expected value: start + indices[dim] * scale
      T scale = detectedScale.value_or(getOneValue(actualValue));
      T expectedValue =
          detectedStart.value() + multiplyByIndex(scale, indices[dim]);

      if (!valuesAreEqual(actualValue, expectedValue)) {
        isIotaAlongDim = false;
        break;
      }
    }

    if (isIotaAlongDim && detectedStart) {
      TypedAttr startAttr =
          createAttrFromValue(ctx, elemType, detectedStart.value());
      TypedAttr scaleAttr = createAttrFromValue(
          ctx, elemType,
          detectedScale.value_or(getOneValue(detectedStart.value())));
      return IotaLikeTensor{startAttr, dim, scaleAttr, constType};
    }
  }

  return std::nullopt;
}

} // namespace

std::optional<IotaLikeTensor>
detectIotaLikeTensor(DenseElementsAttr denseAttr) {
  if (!denseAttr || denseAttr.isSplat()) {
    return std::nullopt;
  }

  auto elemType = denseAttr.getType().getElementType();
  if (isa<FloatType, IntegerType>(elemType)) {
    if (elemType.getIntOrFloatBitWidth() == 1) {
      return std::nullopt;
    }
  }

  if (isa<FloatType>(elemType)) {
    return detectIotaLikeTensorImpl<APFloat>(denseAttr);
  } else if (isa<IntegerType>(elemType)) {
    return detectIotaLikeTensorImpl<APInt>(denseAttr);
  }

  return std::nullopt;
}

std::optional<IotaLikeTensor> detectIotaLikeTensor(mlir::Value tensor) {
  if (!tensor) {
    return std::nullopt;
  }

  struct ChainItem {
    mlir::Operation *op;
    TypedAttr offset; // only populated for AddOp/SubtractOp (nullptr otherwise)
    TypedAttr scale;  // only populated for MulOp (nullptr otherwise)
  };

  // Build a chain of operations from startOp to the base case
  SmallVector<ChainItem> chain;
  llvm::DenseSet<mlir::Operation *> visited;
  mlir::Operation *currentOp = tensor.getDefiningOp();

  // Traverse to find base case
  while (currentOp && !visited.contains(currentOp)) {
    visited.insert(currentOp);

    // check if we found a base case
    if (isa<stablehlo::IotaOp, stablehlo::ConstantOp>(currentOp)) {
      chain.push_back({currentOp, nullptr, nullptr});
      break;
    }

    // navigate to the next op. If any unsupported intermediate op is found,
    // then return std::nullopt
    // TODO: we might want to support broadcast_in_dim / insert_dims / drop_dims
    // as well
    auto nextOp =
        llvm::TypeSwitch<Operation *, Operation *>(currentOp)
            .Case<stablehlo::TransposeOp>([&](auto transposeOp) {
              chain.push_back({currentOp, nullptr, nullptr});
              return transposeOp.getOperand().getDefiningOp();
            })
            .Case<stablehlo::ConvertOp>([&](auto convertOp) -> Operation * {
              auto elemType = cast<TensorType>(convertOp.getOperand().getType())
                                  .getElementType();
              if (!isa<mlir::IntegerType, mlir::FloatType>(elemType)) {
                return nullptr;
              }
              chain.push_back({currentOp, nullptr, nullptr});
              return convertOp.getOperand().getDefiningOp();
            })
            .Case<stablehlo::AddOp>([&](auto addOp) -> Operation * {
              SplatElementsAttr rhsAttr, lhsAttr;
              if (matchPattern(addOp.getRhs(), m_Constant(&rhsAttr))) {
                chain.push_back(
                    {currentOp, rhsAttr.getSplatValue<TypedAttr>(), nullptr});
                return addOp.getLhs().getDefiningOp();
              } else if (matchPattern(addOp.getLhs(), m_Constant(&lhsAttr))) {
                chain.push_back(
                    {currentOp, lhsAttr.getSplatValue<TypedAttr>(), nullptr});
                return addOp.getRhs().getDefiningOp();
              }
              return nullptr;
            })
            .Case<stablehlo::SubtractOp>([&](auto subOp) -> Operation * {
              SplatElementsAttr rhsAttr;
              if (matchPattern(subOp.getRhs(), m_Constant(&rhsAttr))) {
                chain.push_back(
                    {currentOp, rhsAttr.getSplatValue<TypedAttr>(), nullptr});
                return subOp.getLhs().getDefiningOp();
              }
              return nullptr;
            })
            .Case<stablehlo::MulOp>([&](auto mulOp) -> Operation * {
              SplatElementsAttr rhsAttr, lhsAttr;
              if (matchPattern(mulOp.getRhs(), m_Constant(&rhsAttr))) {
                chain.push_back(
                    {currentOp, nullptr, rhsAttr.getSplatValue<TypedAttr>()});
                return mulOp.getLhs().getDefiningOp();
              } else if (matchPattern(mulOp.getLhs(), m_Constant(&lhsAttr))) {
                chain.push_back(
                    {currentOp, nullptr, lhsAttr.getSplatValue<TypedAttr>()});
                return mulOp.getRhs().getDefiningOp();
              }
              return nullptr;
            })
            .Default([](Operation *) -> Operation * { return nullptr; });

    if (!nextOp)
      return std::nullopt;

    currentOp = nextOp;
  }

  if (chain.empty()) {
    return std::nullopt;
  }

  // process the base case
  IotaLikeTensor result;
  MLIRContext *ctx = tensor.getContext();

  auto baseCaseResult =
      llvm::TypeSwitch<Operation *, std::optional<IotaLikeTensor>>(
          chain.back().op)
          .Case<stablehlo::IotaOp>(
              [&](auto iotaOp) -> std::optional<IotaLikeTensor> {
                auto iotaType =
                    cast<RankedTensorType>(iotaOp.getResult().getType());
                auto iotaDim = static_cast<int64_t>(iotaOp.getIotaDimension());
                auto elemType = iotaType.getElementType();
                TypedAttr zeroAttr = createAttrFromDouble(ctx, elemType, 0.0);
                TypedAttr oneAttr = createAttrFromDouble(ctx, elemType, 1.0);
                return IotaLikeTensor{zeroAttr, iotaDim, oneAttr, iotaType};
              })
          .Case<stablehlo::ConstantOp>(
              [&](auto constantOp) -> std::optional<IotaLikeTensor> {
                auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constantOp.getValue());
                return detectIotaLikeTensor(denseAttr);
              })
          .Default([](Operation *) -> std::optional<IotaLikeTensor> {
            return std::nullopt;
          });

  if (!baseCaseResult)
    return std::nullopt;
  result = *baseCaseResult;

  // traverse the chain in reverse order
  for (int64_t i = chain.size() - 2; i >= 0; i--) {
    auto item = chain[i];

    auto success =
        llvm::TypeSwitch<Operation *, bool>(item.op)
            .Case<stablehlo::ConvertOp>([&](auto) {
              auto newElemType =
                  cast<TensorType>(item.op->getResult(0).getType())
                      .getElementType();
              auto startVal = getDoubleFromAttr(result.start);
              auto scaleVal = getDoubleFromAttr(result.scale);
              if (!startVal || !scaleVal)
                return false;
              result.start = createAttrFromDouble(ctx, newElemType, *startVal);
              result.scale = createAttrFromDouble(ctx, newElemType, *scaleVal);
              return true;
            })
            .Case<stablehlo::TransposeOp>([&](auto transposeOp) {
              auto permutation = transposeOp.getPermutation();
              for (int64_t idx = 0;
                   idx < static_cast<int64_t>(permutation.size()); idx++) {
                if (permutation[idx] == result.dimension) {
                  result.dimension = idx;
                  break;
                }
              }
              return true;
            })
            .Case<stablehlo::AddOp>([&](auto) {
              auto startVal = getDoubleFromAttr(result.start);
              auto offsetVal = getDoubleFromAttr(item.offset);
              if (!startVal || !offsetVal)
                return false;
              auto elemType = result.start.getType();
              result.start =
                  createAttrFromDouble(ctx, elemType, *startVal + *offsetVal);
              return true;
            })
            .Case<stablehlo::SubtractOp>([&](auto) {
              auto startVal = getDoubleFromAttr(result.start);
              auto offsetVal = getDoubleFromAttr(item.offset);
              if (!startVal || !offsetVal)
                return false;
              auto elemType = result.start.getType();
              result.start =
                  createAttrFromDouble(ctx, elemType, *startVal - *offsetVal);
              return true;
            })
            .Case<stablehlo::MulOp>([&](auto) {
              auto startVal = getDoubleFromAttr(result.start);
              auto scaleVal = getDoubleFromAttr(result.scale);
              auto mulVal = getDoubleFromAttr(item.scale);
              if (!startVal || !scaleVal || !mulVal)
                return false;
              auto elemType = result.start.getType();
              result.start =
                  createAttrFromDouble(ctx, elemType, *startVal * *mulVal);
              result.scale =
                  createAttrFromDouble(ctx, elemType, *scaleVal * *mulVal);
              return true;
            })
            .Default([](Operation *) {
              assert(false && "reached unreachable case...");
              return false;
            });

    if (!success)
      return std::nullopt;
  }

  result.tensorType = cast<RankedTensorType>(tensor.getType());
  return result;
}

namespace {

struct PaddingResult {
  SmallVector<int64_t> lowPadding;
  SmallVector<int64_t> highPadding;
  Attribute paddingValueAttr;
  int64_t totalPaddingElements = 0;
};

template <typename T>
std::optional<PaddedTensor> detectPaddedTensorImpl(DenseElementsAttr attr) {
  auto tensorType = cast<RankedTensorType>(attr.getType());
  auto shape = tensorType.getShape();
  int64_t rank = tensorType.getRank();

  auto elemType = tensorType.getElementType();
  auto strides = computeStrides(shape);
  int64_t numElements = tensorType.getNumElements();

  auto values = attr.getValues<T>();

  // Generic helper to detect padding using a specific padding value
  // Optimized O(numElements * rank) algorithm: single pass to find
  // min/max non-padding indices per dimension
  auto detectPaddingWithValue = [&](T paddingVal) -> PaddingResult {
    PaddingResult result;
    result.lowPadding.resize(rank, 0);
    result.highPadding.resize(rank, 0);
    result.paddingValueAttr = makeAttr(elemType, paddingVal);

    // Track the min and max non-padding index for each dimension
    SmallVector<int64_t> minNonPadIdx(rank);
    SmallVector<int64_t> maxNonPadIdx(rank);
    for (int64_t dim = 0; dim < rank; dim++) {
      minNonPadIdx[dim] = shape[dim]; // Initialize to beyond max valid index
      maxNonPadIdx[dim] = -1;         // Initialize to before min valid index
    }

    // Single pass through all elements
    SmallVector<int64_t> indices;
    for (int64_t linearIdx = 0; linearIdx < numElements; linearIdx++) {
      if (values[linearIdx] != paddingVal) {
        // This element is NOT padding - update min/max for all dimensions
        linearToMultiIndex(linearIdx, strides, indices);
        for (int64_t dim = 0; dim < rank; dim++) {
          minNonPadIdx[dim] = std::min(minNonPadIdx[dim], indices[dim]);
          maxNonPadIdx[dim] = std::max(maxNonPadIdx[dim], indices[dim]);
        }
      }
    }

    // Convert min/max non-padding indices to low/high padding amounts
    for (int64_t dim = 0; dim < rank; dim++) {
      if (maxNonPadIdx[dim] < 0) {
        // All elements are padding in this dimension (entire tensor is padding)
        result.lowPadding[dim] = shape[dim];
        result.highPadding[dim] = 0;
      } else {
        result.lowPadding[dim] = minNonPadIdx[dim];
        result.highPadding[dim] = shape[dim] - 1 - maxNonPadIdx[dim];
      }
    }

    // Calculate total padding elements saved
    int64_t innerElements = 1;
    for (int64_t dim = 0; dim < rank; dim++) {
      innerElements *=
          shape[dim] - result.lowPadding[dim] - result.highPadding[dim];
    }
    result.totalPaddingElements = numElements - innerElements;

    return result;
  };

  // Try both first element and last element as potential padding values
  // and pick the one that gives maximum padding
  auto resultFirst = detectPaddingWithValue(values[0]);
  auto resultLast = detectPaddingWithValue(values[numElements - 1]);
  PaddingResult bestResult =
      (resultLast.totalPaddingElements > resultFirst.totalPaddingElements)
          ? std::move(resultLast)
          : std::move(resultFirst);

  auto &lowPadding = bestResult.lowPadding;
  auto &highPadding = bestResult.highPadding;
  auto &paddingValueAttr = bestResult.paddingValueAttr;

  // Check if we found any padding at all
  bool hasPadding = false;
  for (int64_t dim = 0; dim < rank; dim++) {
    if (lowPadding[dim] > 0 || highPadding[dim] > 0) {
      hasPadding = true;
      break;
    }
  }

  if (!hasPadding) {
    return std::nullopt;
  }

  // Calculate the inner tensor shape
  SmallVector<int64_t> innerShape(rank);
  for (int64_t dim = 0; dim < rank; dim++) {
    innerShape[dim] = shape[dim] - lowPadding[dim] - highPadding[dim];
    if (innerShape[dim] <= 0) {
      return std::nullopt;
    }
  }

  // Calculate number of elements in inner tensor
  int64_t innerNumElements = 1;
  for (int64_t dim = 0; dim < rank; dim++) {
    innerNumElements *= innerShape[dim];
  }

  // Ensure we're actually saving something substantial
  if (innerNumElements * 2 >= numElements) {
    return std::nullopt;
  }

  auto innerStrides = computeStrides(innerShape);

  auto innerTensorType = RankedTensorType::get(innerShape, elemType);

  SmallVector<T> innerValues;
  innerValues.reserve(innerNumElements);

  for (int64_t innerLinear = 0; innerLinear < innerNumElements; innerLinear++) {
    SmallVector<int64_t> innerIndices;
    linearToMultiIndex(innerLinear, innerStrides, innerIndices);

    SmallVector<int64_t> origIndices(rank);
    for (int64_t d = 0; d < rank; d++) {
      origIndices[d] = innerIndices[d] + lowPadding[d];
    }

    int64_t origLinear = multiToLinearIndex(origIndices, strides);
    innerValues.push_back(values[origLinear]);
  }

  auto innerTensorAttr =
      DenseElementsAttr::get(innerTensorType, ArrayRef<T>(innerValues));

  PaddedTensor result;
  result.innerTensorAttr = innerTensorAttr;
  result.paddingValue = paddingValueAttr;
  result.lowPadding = std::move(lowPadding);
  result.highPadding = std::move(highPadding);
  result.resultType = cast<RankedTensorType>(tensorType);
  return result;
}

} // namespace

std::optional<PaddedTensor> detectPaddedTensor(DenseElementsAttr attr) {
  if (!attr || attr.isSplat()) {
    return std::nullopt;
  }

  auto tensorType = attr.getType();
  int64_t rank = tensorType.getRank();

  if (rank == 0) {
    return std::nullopt;
  }

  auto elemType = tensorType.getElementType();
  if (isa<FloatType>(elemType)) {
    return detectPaddedTensorImpl<APFloat>(attr);
  } else if (isa<IntegerType>(elemType)) {
    return detectPaddedTensorImpl<APInt>(attr);
  }

  return std::nullopt;
}

bool allAccessesAreOnMainDiagonalPostReshape(stablehlo::ReshapeOp op,
                                             stablehlo::SliceOp sliceOp) {
  auto reshapeInTy = cast<RankedTensorType>(op.getOperand().getType());
  auto reshapeOutTy = cast<RankedTensorType>(op.getType());

  if (reshapeOutTy.getRank() != 1 ||
      reshapeInTy.getRank() != 2) // [M, N] -> [M * N] vector
    return false;

  auto M = reshapeInTy.getDimSize(0);
  auto N = reshapeInTy.getDimSize(1);
  auto diagLen = std::min(M, N);
  auto diagStride = N + 1;

  int64_t start = sliceOp.getStartIndices()[0];
  int64_t limit = sliceOp.getLimitIndices()[0];
  int64_t stride = sliceOp.getStrides()[0];

  if (stride % diagStride != 0)
    return false;

  // start can be on any of the diagonal elements
  if (start % diagStride != 0)
    return false;

  if (limit > M * N)
    return false; // technically this is illegal

  // sanity check
  int64_t count = (limit - start + stride - 1) / stride;
  if (count <= 0 || count > diagLen)
    return false;

  return true;
}

bool allAccessesAreOnMainDiagonalPostReshape(
    stablehlo::ReshapeOp op, Operation *user,
    llvm::SetVector<Operation *> &opsToReplace) {
  if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(user)) {
    if (allAccessesAreOnMainDiagonalPostReshape(op, sliceOp)) {
      opsToReplace.insert(sliceOp);
      return true;
    }
    return false;
  }
  return false;
}

bool allAccessesAreOnMainDiagonal(Operation *op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
    return allAccessesAreOnMainDiagonal(reshapeOp, opsToReplace);
  } else if (auto gatherOp = dyn_cast<stablehlo::GatherOp>(op)) {
    return allAccessesAreOnMainDiagonal(gatherOp, opsToReplace);
  }
  return false;
}

bool allAccessesAreOnMainDiagonal(stablehlo::ReshapeOp op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  auto reshapeInTy = cast<RankedTensorType>(op.getOperand().getType());
  if (reshapeInTy.getRank() != 2) // [M, N] matrix
    return false;                 // quick exit

  llvm::SmallPtrSet<Operation *, 4> seenOps;
  for (auto user : op->getUsers()) {
    if (seenOps.count(user))
      continue;

    if (!allAccessesAreOnMainDiagonalPostReshape(op, user, opsToReplace))
      return false;

    seenOps.insert(user);
  }

  return true;
}

bool allAccessesAreOnMainDiagonal(stablehlo::GatherOp op,
                                  llvm::SetVector<Operation *> &opsToReplace) {
  return false; // TODO: implement this where we are doing gather with iota
}

} // namespace enzyme

namespace stablehlo {

// Templated helper (default returns nullptr).
template <typename OpTy>
Value getIdentityValueForOp(OpBuilder &builder, Location loc, Type elemType) {
  return nullptr;
}

// Specializations for identity values of specific binary ops.
template <>
Value getIdentityValueForOp<stablehlo::AddOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  TypedAttr zeroVal;
  if (auto complexType = dyn_cast<ComplexType>(elemType)) {
    auto floatType = cast<FloatType>(complexType.getElementType());
    auto zero = APFloat::getZero(floatType.getFloatSemantics());
    auto complexZero = std::complex<APFloat>(zero, zero);
    zeroVal = DenseElementsAttr::get(RankedTensorType::get({}, elemType),
                                     complexZero);
  } else {
    zeroVal = builder.getZeroAttr(elemType);
  }
  return stablehlo::ConstantOp::create(builder, loc, zeroVal);
}

template <>
Value getIdentityValueForOp<stablehlo::MulOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  TypedAttr identityVal;
  if (auto complexType = dyn_cast<ComplexType>(elemType)) {
    auto floatType = cast<FloatType>(complexType.getElementType());
    auto one = APFloat(floatType.getFloatSemantics(), 1);
    auto zero = APFloat::getZero(floatType.getFloatSemantics());
    auto complexOne = std::complex<APFloat>(one, zero);
    identityVal =
        DenseElementsAttr::get(RankedTensorType::get({}, elemType), complexOne);
  } else {
    identityVal = builder.getOneAttr(elemType);
  }
  return stablehlo::ConstantOp::create(builder, loc, identityVal);
}

template <>
Value getIdentityValueForOp<stablehlo::MinOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    auto negInf =
        APFloat::getInf(floatType.getFloatSemantics(), /*negative=*/false);
    return stablehlo::ConstantOp::create(
        builder, loc, builder.getFloatAttr(elemType, negInf));
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    auto minVal = APInt::getSignedMaxValue(intType.getWidth());
    return stablehlo::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(elemType, minVal));
  }
  return nullptr;
}

template <>
Value getIdentityValueForOp<stablehlo::MaxOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    auto inf =
        APFloat::getInf(floatType.getFloatSemantics(), /*negative=*/true);
    return stablehlo::ConstantOp::create(builder, loc,
                                         builder.getFloatAttr(elemType, inf));
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    auto maxVal = APInt::getSignedMinValue(intType.getWidth());
    return stablehlo::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(elemType, maxVal));
  }
  return nullptr;
}

// Identity values for bitwise logical ops.
// OR/XOR: identity = 0
template <>
Value getIdentityValueForOp<stablehlo::OrOp>(OpBuilder &builder, Location loc,
                                             Type elemType) {
  // Zero is a valid identity for OR and XOR across integer and boolean types.
  return stablehlo::ConstantOp::create(builder, loc,
                                       builder.getZeroAttr(elemType));
}

template <>
Value getIdentityValueForOp<stablehlo::XorOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  // Zero is a valid identity for XOR as well.
  return stablehlo::ConstantOp::create(builder, loc,
                                       builder.getZeroAttr(elemType));
}

// AND: identity is all-ones bitpattern for integer types (applies to booleans
// too).
template <>
Value getIdentityValueForOp<stablehlo::AndOp>(OpBuilder &builder, Location loc,
                                              Type elemType) {
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    // All ones value for the integer width (e.g., 0xFFFF...); this yields
    // the 'all bits set' value which acts as identity for bitwise AND.
    auto ones = APInt::getAllOnes(intType.getWidth());
    return stablehlo::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(elemType, ones));
  }
  return nullptr;
}

Value getIdentityValue(OpBuilder &builder, Location loc, Type elemType,
                       Operation *op) {
  return TypeSwitch<Operation *, Value>(op)
      .Case<stablehlo::AddOp, stablehlo::MulOp, stablehlo::MinOp,
            stablehlo::MaxOp, stablehlo::OrOp, stablehlo::XorOp,
            stablehlo::AndOp>([&](auto binOp) {
        return getIdentityValueForOp<decltype(binOp)>(builder, loc, elemType);
      })
      .Default([&](Operation *op) -> Value { return nullptr; });
}

bool canFuseIntoReduce(Operation *op) {
  return isa<stablehlo::AddOp, stablehlo::MulOp, stablehlo::MinOp,
             stablehlo::MaxOp, stablehlo::OrOp, stablehlo::XorOp,
             stablehlo::AndOp>(op);
}

stablehlo::GatherDimensionNumbersAttr
getGatherDims(mlir::MLIRContext *ctx,
              stablehlo::ScatterDimensionNumbersAttr scatterDimNumbers) {
  return stablehlo::GatherDimensionNumbersAttr::get(
      ctx, scatterDimNumbers.getUpdateWindowDims(),
      scatterDimNumbers.getInsertedWindowDims(),
      scatterDimNumbers.getInputBatchingDims(),
      scatterDimNumbers.getScatterIndicesBatchingDims(),
      scatterDimNumbers.getScatterDimsToOperandDims(),
      scatterDimNumbers.getIndexVectorDim());
}

bool isSetindexBlock(mlir::Block *block,
                     std::function<bool(stablehlo::ReturnOp retOp)> fn) {
  if (block->getNumArguments() != 2) {
    return false;
  }

  // The block should have exactly one operation (the return)
  if (block->getOperations().size() != 1) {
    return false;
  }

  auto &returnOp = block->front();
  auto stablehloReturnOp = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturnOp) {
    return false;
  }

  if (stablehloReturnOp.getNumOperands() != 1) {
    return false;
  }

  return fn(stablehloReturnOp);
}

bool isSetindexBlock(mlir::Block *block) {
  return isSetindexBlock(block, [&](stablehlo::ReturnOp retOp) {
    return retOp.getOperand(0) == block->getArgument(1);
  });
}

bool isSetindexBlock(mlir::Block *block, mlir::Value &val) {
  return isSetindexBlock(block, [&val, block](stablehlo::ReturnOp retOp) {
    val = retOp.getOperand(0);
    // Check that val is defined outside the block
    if (auto *definingOp = val.getDefiningOp()) {
      if (definingOp->getBlock() == block) {
        return false;
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      if (blockArg.getOwner() == block) {
        return false;
      }
    }
    return true;
  });
}

bool isConstantSetindexBlock(mlir::Block *block,
                             mlir::SplatElementsAttr &constant) {
  return isSetindexBlock(block, [&constant](stablehlo::ReturnOp retOp) {
    return matchPattern(retOp.getOperand(0), m_Constant(&constant));
  });
}

SmallVector<int64_t> computeGatherSliceSizes(stablehlo::ScatterOp &scatterOp) {
  auto inputType = cast<RankedTensorType>(scatterOp.getInputs()[0].getType());
  auto updateType = cast<RankedTensorType>(scatterOp.getUpdates()[0].getType());
  auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();

  auto updateShape = updateType.getShape();

  SmallVector<int64_t> sliceSizes;
  sliceSizes.reserve(inputType.getRank());

  auto updateWindowDims = scatterDimNumbers.getUpdateWindowDims();
  auto insertedWindowDims = scatterDimNumbers.getInsertedWindowDims();
  auto operandBatchingDims = scatterDimNumbers.getInputBatchingDims();

  // https://github.com/jax-ml/jax/blob/6d41fa0c5a9400f40a4c8fb53c32b45e460903cd/jax/_src/lax/slicing.py#L2799
  int64_t pos = 0;
  for (size_t i = 0; i < inputType.getRank(); ++i) {
    if (llvm::is_contained(insertedWindowDims, i) ||
        llvm::is_contained(operandBatchingDims, i)) {
      sliceSizes.push_back(1);
    } else {
      sliceSizes.push_back(updateShape[updateWindowDims[pos]]);
      pos++;
    }
  }

  return sliceSizes;
}

stablehlo::ComparisonDirection
reversedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GE;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GT;
  default:
    llvm_unreachable("Cannot perform reverse comparison");
  }
}

stablehlo::ComparisonDirection
negatedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GT;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GE;
  default:
    llvm_unreachable("Cannot negate comparison");
  }
}

bool reshapeIsTranspose(stablehlo::ReshapeOp reshapeOp) {
  auto input = reshapeOp.getOperand();
  auto output = reshapeOp.getResult();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

  if (!inputType || !outputType)
    return false;

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  // If the number of dimensions is different, it cannot be a simple transpose
  if (inputShape.size() != outputShape.size())
    return false;

  // for each input dim find the next output dim that is of same size.
  // if there are any non-1 dims, it cannot be a simple transpose
  size_t j = 0, i = 0;
  for (; i < inputShape.size(); ++i) {
    auto insz = inputShape[i];
    if (insz == 1)
      continue;

    bool found = false;
    while (j < outputShape.size()) {
      auto outsz = outputShape[j];
      ++j;
      if (outsz == 1)
        continue;
      if (insz == outsz) {
        found = true;
        break;
      }
      if (insz != outsz)
        return false;
    }

    if (!found)
      return false;
  }

  for (; j < outputShape.size(); ++j) {
    if (outputShape[j] != 1) {
      return false;
    }
  }

  return true;
}

Value reshapeAxisInto(OpBuilder &builder, Value input,
                      ArrayRef<int64_t> &batchSizes, int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  SmallVector<int64_t> permutation(inputShape.size());
  for (size_t i = 0; i < dim; i++)
    permutation[i] = i + batchSizes.size(); // left shift
  for (size_t i = 0; i < batchSizes.size(); i++)
    permutation[dim + i] = i; // move the batch dims
  for (size_t i = batchSizes.size() + dim; i < permutation.size(); i++)
    permutation[i] = i; // keep the rest

  auto transposedInput =
      stablehlo::TransposeOp::create(builder, input.getLoc(), input,
                                     builder.getDenseI64ArrayAttr(permutation));

  SmallVector<int64_t> newShape(inputShape.begin() + batchSizes.size(),
                                inputShape.end());
  newShape[dim] =
      newShape[dim] * std::accumulate(batchSizes.begin(), batchSizes.end(), 1,
                                      std::multiplies<int64_t>());
  return stablehlo::ReshapeOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(newShape, inputType.getElementType()),
      transposedInput);
}

Value reshapeAxisOutOf(OpBuilder &builder, Value input,
                       ArrayRef<int64_t> &batchSizes, int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = llvm::to_vector(inputType.getShape());
  auto batchSize = std::accumulate(batchSizes.begin(), batchSizes.end(), 1,
                                   std::multiplies<int64_t>());
  inputShape[dim] = inputShape[dim] / batchSize;
  for (size_t i = 0; i < batchSizes.size(); i++)
    inputShape.insert(inputShape.begin() + dim + i, batchSizes[i]);

  auto reshapedInput = stablehlo::ReshapeOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(inputShape, inputType.getElementType()), input);

  SmallVector<int64_t> permutation(inputShape.size());
  for (size_t i = 0; i < batchSizes.size(); i++)
    permutation[i] = dim + i;
  for (size_t i = 0; i < dim; i++)
    permutation[batchSizes.size() + i] = i;
  for (size_t i = batchSizes.size() + dim; i < permutation.size(); i++)
    permutation[i] = i;

  return stablehlo::TransposeOp::create(
      builder, input.getLoc(), reshapedInput,
      builder.getDenseI64ArrayAttr(permutation));
}

bool hasTraitElementwise(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;

  if (op->hasTrait<hlo::OpTrait::BroadcastingElementwise>()) {
    // Check sizes (shapes) match across operands, not the exact types.
    auto refShapedTy = dyn_cast<ShapedType>(op->getOperand(0).getType());
    if (!refShapedTy)
      return false;

    for (auto operand : op->getOperands()) {
      auto curShapedTy = dyn_cast<ShapedType>(operand.getType());
      if (!curShapedTy)
        return false;

      if (curShapedTy.getRank() != refShapedTy.getRank())
        return false;

      for (int64_t i = 0; i < curShapedTy.getRank(); ++i) {
        int64_t a = curShapedTy.getDimSize(i);
        int64_t b = refShapedTy.getDimSize(i);
        // If both are static and different, sizes don't match.
        if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
          return false;
      }
    }
    return true;
  }

  return false;
}

bool isAssociativeOp(Operation *op) {
  return isa<stablehlo::AddOp, stablehlo::MulOp, stablehlo::MinOp,
             stablehlo::MaxOp, stablehlo::AndOp, stablehlo::OrOp,
             stablehlo::XorOp>(op);
}

bool extractMultiplicationFactor(Value v, Value &other, Operation *op,
                                 OpBuilder &builder) {
  auto mulOp = v.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp) {
    other = v;
    return true;
  }

  if (!isOnlyUsedInOperation(mulOp, op))
    return false;

  Value mLhs = mulOp.getLhs(), mRhs = mulOp.getRhs();

  if (isScalarValue(mLhs)) {
    other = mRhs;
  } else if (isScalarValue(mRhs)) {
    other = mLhs;
  } else {
    return false;
  }
  return true;
}

void extractMultiplicationFactor(Value v, Value &scalar, Value &other,
                                 Operation *op, OpBuilder &builder) {
  auto mulOp = v.getDefiningOp<stablehlo::MulOp>();
  if (!mulOp) {
    scalar = nullptr;
    other = v;
    return;
  }

  Value mLhs = mulOp.getLhs(), mRhs = mulOp.getRhs();

  auto lhsScalarValue = stablehlo::getScalarValue(mLhs, builder);
  if (lhsScalarValue) {
    scalar = lhsScalarValue;
    other = mRhs;
  } else {
    auto rhsScalarValue = stablehlo::getScalarValue(mRhs, builder);
    if (rhsScalarValue) {
      scalar = rhsScalarValue;
      other = mLhs;
    } else {
      scalar = nullptr;
      other = v;
    }
  }
}

Value getScalarValue(Value val, OpBuilder &builder) {
  return getScalarValue(val.getDefiningOp(), builder);
}

Value getScalarValue(Operation *op, OpBuilder &builder) {
  if (!op)
    return nullptr;

  // Splatted Constant
  SplatElementsAttr elems;
  if (matchPattern(op, m_Constant(&elems))) {
    auto scalarElemType = RankedTensorType::get(
        {}, cast<TensorType>(op->getResult(0).getType()).getElementType());
    auto constInit = ConstantOp::create(builder, op->getLoc(), scalarElemType,
                                        elems.resizeSplat(scalarElemType));
    return constInit;
  }

  // BroadcastInDim / Reshape
  if (isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp>(op)) {
    if (cast<RankedTensorType>(op->getOperand(0).getType()).getRank() == 0) {
      return op->getOperand(0);
    }
  }

  // Convert
  if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(op)) {
    auto scalar =
        getScalarValue(convertOp.getOperand().getDefiningOp(), builder);
    if (scalar) {
      auto convertOutElemType =
          cast<RankedTensorType>(convertOp.getResult().getType())
              .getElementType();
      return stablehlo::ConvertOp::create(
          builder, op->getLoc(), RankedTensorType::get({}, convertOutElemType),
          scalar);
    }
  }

  return nullptr;
}

bool isScalarValue(Value val) { return isScalarValue(val.getDefiningOp()); }

bool isScalarValue(Operation *op) {
  if (!op)
    return false;

  SplatElementsAttr splatAttr;
  if (matchPattern(op, m_Constant(&splatAttr)))
    return true;

  if (isa<stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp>(op) &&
      cast<RankedTensorType>(op->getOperand(0).getType()).getRank() == 0)
    return true;

  if (isa<stablehlo::ConvertOp>(op))
    return isScalarValue(op->getOperand(0));

  return false;
}

Value copyTriangularPart(OpBuilder &builder, Value input,
                         enzymexla::LapackUplo uplo) {
  if (uplo == enzymexla::LapackUplo::F)
    return input;

  // TODO: run a backward propagation to check if input potentially originates
  // from a Op that create a partially filled output

  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 2 && "only 2D matrices supported");
  assert(inputType.getDimSize(0) != ShapedType::kDynamic &&
         inputType.getDimSize(1) != ShapedType::kDynamic &&
         "only statically sized matrices supported");
  auto inputShape = inputType.getShape();

  Value rowIdxs = stablehlo::IotaOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(inputShape, builder.getI32Type()), 0);
  Value colIdxs = stablehlo::IotaOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(inputShape, builder.getI32Type()), 1);

  Value indicator = stablehlo::CompareOp::create(
      builder, input.getLoc(), rowIdxs, colIdxs,
      uplo == enzymexla::LapackUplo::U ? ComparisonDirection::LT
                                       : ComparisonDirection::GT);

  Value transposedInput = stablehlo::TransposeOp::create(
      builder, input.getLoc(), input, builder.getDenseI64ArrayAttr({1, 0}));

  return stablehlo::SelectOp::create(builder, input.getLoc(), indicator, input,
                                     transposedInput);
}

bool OpIsReshapeLike(BroadcastInDimOp op) {
  auto input = op.getOperand();
  auto outputType = op.getType();
  auto inputType = input.getType();
  auto broadcastDims = op.getBroadcastDimensions();

  // reshape cannot expand number of elements
  if (inputType.getNumElements() != outputType.getNumElements()) {
    return false;
  }

  // singleton dims can be freely moved around
  SmallVector<int64_t> nonSingletonDims;
  for (auto [dim, inputSize] : llvm::zip(broadcastDims, inputType.getShape())) {
    if (inputSize != 1) {
      nonSingletonDims.push_back(dim);
    }
  }

  // transposed input is not reshape
  if (!llvm::is_sorted(nonSingletonDims)) {
    return false;
  }

  return true;
}

bool OpIsReshapeLike(stablehlo::TransposeOp op) {
  return OpIsReshapeLike(
      op, cast<RankedTensorType>(op.getOperand().getType()).getShape());
}

bool OpIsReshapeLike(stablehlo::TransposeOp op, ArrayRef<int64_t> inShape) {
  SmallVector<int64_t> nonZeroPerms;
  llvm::copy_if(op.getPermutation(), std::back_inserter(nonZeroPerms),
                [&](int64_t i) { return inShape[i] != 1; });
  return llvm::is_sorted(nonZeroPerms);
}

bool canMergeSlicesAlongAxis(int dimension, ArrayRef<int64_t> sliceStarts,
                             ArrayRef<int64_t> otherSliceStarts,
                             ArrayRef<int64_t> sliceLimits,
                             ArrayRef<int64_t> otherSliceLimits,
                             ArrayRef<int64_t> sliceStrides,
                             ArrayRef<int64_t> otherSliceStrides) {
  bool canMerge = true;

  for (int d = 0, ndims = sliceStarts.size(); d < ndims; ++d) {
    if (d == dimension) {
      canMerge &= sliceLimits[d] == otherSliceStarts[d] &&
                  sliceStrides[d] == otherSliceStrides[d];
    } else {
      canMerge &= sliceStarts[d] == otherSliceStarts[d] &&
                  sliceLimits[d] == otherSliceLimits[d] &&
                  sliceStrides[d] == otherSliceStrides[d];
    }
  }
  return canMerge;
}

bool canMergeSlicesAlongAxis(int dimension, stablehlo::SliceOp slice,
                             stablehlo::SliceOp otherSlice) {
  if (otherSlice.getOperand() != slice.getOperand()) {
    return false;
  }

  // Check that both slices are contiguous only in dim
  ArrayRef<int64_t> sliceStarts = slice.getStartIndices(),
                    otherSliceStarts = otherSlice.getStartIndices(),
                    sliceLimits = slice.getLimitIndices(),
                    otherSliceLimits = otherSlice.getLimitIndices(),
                    sliceStrides = slice.getStrides(),
                    otherSliceStrides = otherSlice.getStrides();

  return canMergeSlicesAlongAxis(dimension, sliceStarts, otherSliceStarts,
                                 sliceLimits, otherSliceLimits, sliceStrides,
                                 otherSliceStrides);
}

stablehlo::ConcatenateOp lowerWrap(enzymexla::WrapOp wrap,
                                   PatternRewriter &rewriter, bool replace) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(wrap);

  // sl0[end-lhs:end], mid, sl1[0:rhs]
  auto wrapOpT = cast<RankedTensorType>(wrap.getOperand().getType());
  SmallVector<int64_t> strides(wrapOpT.getShape().size(), 1);

  SmallVector<Value> args;

  auto shard = sdy::getShardingPerValue(wrap);

  if (wrap.getLhs() != 0) {
    SmallVector<int64_t> sl0_starts(wrapOpT.getShape().size(), 0);
    SmallVector<int64_t> sl0_ends(wrapOpT.getShape());

    sl0_starts[wrap.getDimension()] =
        wrapOpT.getShape()[wrap.getDimension()] - wrap.getLhs();

    auto sl0 =
        stablehlo::SliceOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                   sl0_starts, sl0_ends, strides);
    if (shard)
      sdy::setShardings(sl0, shard);

    args.push_back(sl0);
  }

  args.push_back(wrap.getOperand());

  if (wrap.getRhs() != 0) {
    SmallVector<int64_t> sl1_starts(wrapOpT.getShape().size(), 0);
    SmallVector<int64_t> sl1_ends(wrapOpT.getShape());

    sl1_ends[wrap.getDimension()] = wrap.getRhs();
    auto sl1 =
        stablehlo::SliceOp::create(rewriter, wrap.getLoc(), wrap.getOperand(),
                                   sl1_starts, sl1_ends, strides);
    if (shard)
      sdy::setShardings(sl1, shard);

    args.push_back(sl1);
  }

  auto newConcat = stablehlo::ConcatenateOp::create(rewriter, wrap.getLoc(),
                                                    args, wrap.getDimension());
  if (replace)
    rewriter.replaceOp(wrap, newConcat);
  if (shard)
    sdy::setShardings(newConcat, shard);
  return newConcat;
}

LogicalResult concatReshapeSliceSimplify(PatternRewriter &rewriter,
                                         SmallVectorImpl<Value> &operands,
                                         int64_t dim,
                                         SmallVectorImpl<Value> &newOperands) {
  bool changed = false;

  auto getShapeWithoutDims = [](RankedTensorType type,
                                int64_t dim) -> SmallVector<int64_t> {
    SmallVector<int64_t> shape;
    for (size_t i = 0; i < type.getRank(); ++i) {
      if (i == dim)
        continue;
      shape.push_back(type.getDimSize(i));
    }
    return shape;
  };

  for (size_t i = 0, e = operands.size(); i < e; ++i) {
    auto operand = operands[i];
    stablehlo::SliceOp slice;
    stablehlo::ReshapeOp reshape =
        operand.getDefiningOp<stablehlo::ReshapeOp>();

    bool insertions = false, deletions = false;
    int64_t totalMergeSize;
    SmallVector<int64_t> insertionDims, deletionDims;

    int64_t sliceDim = -1;
    if (reshape) {
      slice = reshape.getOperand().getDefiningOp<stablehlo::SliceOp>();
      if (!slice) {
        newOperands.push_back(operand);
        continue;
      } else {
        auto sliceInTy = cast<ShapedType>(slice.getOperand().getType());
        auto sliceTy = cast<ShapedType>(slice.getType());
        for (size_t idx = 0; idx < slice.getStartIndices().size(); ++idx) {
          if (sliceTy.getDimSize(idx) == 1 &&
              !(slice.getStartIndices()[idx] == 0 &&
                slice.getLimitIndices()[idx] == sliceInTy.getDimSize(idx) &&
                slice.getStrides()[idx] == 1)) {
            if (sliceDim != -1) {
              sliceDim = -1; // TODO: support multiple dims here
              break;
            }
            sliceDim = idx;
          }
        }

        if (sliceDim == -1) {
          newOperands.push_back(operand);
          continue;
        }

        auto srcWithoutSliceDim = getShapeWithoutDims(
            cast<RankedTensorType>(slice.getType()), sliceDim);
        auto dstWithoutConcatDim =
            getShapeWithoutDims(cast<RankedTensorType>(reshape.getType()), dim);

        if (srcWithoutSliceDim != dstWithoutConcatDim) {
          auto curInsertionDims =
              findReshapeInsertionDims(srcWithoutSliceDim, dstWithoutConcatDim);
          auto curDeletionDims =
              findReshapeInsertionDims(dstWithoutConcatDim, srcWithoutSliceDim);

          if (curInsertionDims.empty() && curDeletionDims.empty()) {
            newOperands.push_back(operand);
            continue;
          } else if (!curInsertionDims.empty()) {
            insertions = true;
            insertionDims = std::move(curInsertionDims);
          } else {
            deletions = true;
            deletionDims = std::move(curDeletionDims);
          }
        }

        totalMergeSize =
            cast<RankedTensorType>(reshape.getType()).getDimSize(dim);
      }
    } else {
      newOperands.push_back(operand);
      continue;
    }

    int64_t ndims = cast<RankedTensorType>(reshape.getType()).getRank();
    int64_t ndimsCorrected = ndims;
    if (insertions) {
      ndimsCorrected -= insertionDims.size();
    }
    if (deletions) {
      ndimsCorrected += deletionDims.size();
    }

    if (ndimsCorrected <= dim) {
      newOperands.push_back(operand);
      continue;
    }

    Value newOperand = operand;
    bool needsPerm = false;
    while (i + 1 < e) {
      if (auto otherReshape =
              operands[i + 1].getDefiningOp<stablehlo::ReshapeOp>()) {
        if (auto otherSlice =
                otherReshape.getOperand().getDefiningOp<stablehlo::SliceOp>()) {
          // Check if reshapes are the same
          if (!OperationEquivalence::isEquivalentTo(
                  reshape, otherReshape,
                  OperationEquivalence::ignoreValueEquivalence, nullptr,
                  OperationEquivalence::IgnoreLocations, nullptr)) {
            break;
          }

          // Check if slices can be merged
          if (canMergeSlicesAlongAxis(sliceDim, slice, otherSlice)) {
            totalMergeSize +=
                cast<RankedTensorType>(otherReshape.getType()).getDimSize(dim);

            slice = stablehlo::SliceOp::create(
                rewriter, slice->getLoc(), slice.getOperand(),
                slice.getStartIndices(), otherSlice.getLimitIndices(),
                slice.getStrides());
            newOperand = slice.getResult();
            changed = true;
            needsPerm = true;
            i++;
            continue;
          }
        }
      }

      break;
    }

    if (needsPerm) {
      SmallVector<int64_t> mapping(ndimsCorrected);
      std::iota(mapping.begin(), mapping.end(), 0);
      mapping[sliceDim] = dim;
      if (sliceDim > dim) {
        for (int64_t i = dim; i < sliceDim; i++) { // shift right
          mapping[i]++;
        }
      } else {
        for (int64_t i = sliceDim + 1; i <= dim; i++) { // shift left
          mapping[i]--;
        }
      }

      SmallVector<int64_t> permutation(mapping.size(), 0);
      for (int64_t i = 0; i < mapping.size(); i++) {
        permutation[mapping[i]] = i;
      }

      auto transposeOp = stablehlo::TransposeOp::create(
          rewriter, reshape.getLoc(), newOperand, permutation);
      if (!insertions && !deletions) {
        newOperand = transposeOp.getResult();
      } else {
        auto reshapeOrigType = cast<RankedTensorType>(reshape.getType());
        SmallVector<int64_t> newShape =
            llvm::to_vector(reshapeOrigType.getShape());
        newShape[dim] = totalMergeSize;
        newOperand = stablehlo::ReshapeOp::create(
            rewriter, reshape.getLoc(),
            RankedTensorType::get(newShape, reshapeOrigType.getElementType()),
            transposeOp.getResult());
      }
    }

    newOperands.push_back(newOperand);
  }
  return changed ? success() : failure();
}

LogicalResult concatSliceSimplify(PatternRewriter &rewriter,
                                  SmallVectorImpl<Value> &operands, int64_t dim,
                                  SmallVectorImpl<Value> &newOperands) {
  bool changed = false;
  for (size_t i = 0, e = operands.size(); i < e; ++i) {
    auto operand = operands[i];
    auto slice = operand.getDefiningOp<stablehlo::SliceOp>();

    if (!slice) {
      newOperands.push_back(operand);
      continue;
    }

    while (i + 1 < e) {
      if (auto otherSlice =
              operands[i + 1].getDefiningOp<stablehlo::SliceOp>()) {
        if (canMergeSlicesAlongAxis(dim, slice, otherSlice)) {
          slice = stablehlo::SliceOp::create(
              rewriter, slice->getLoc(), slice.getOperand(),
              slice.getStartIndices(), otherSlice.getLimitIndices(),
              slice.getStrides());
          changed = true;
          i++;
          continue;
        } else {
          break;
        }
      }

      if (auto otherWrap = operands[i + 1].getDefiningOp<enzymexla::WrapOp>()) {
        auto wrapSlice =
            otherWrap.getOperand().getDefiningOp<stablehlo::SliceOp>();
        if (wrapSlice && wrapSlice.getOperand() == slice.getOperand() &&
            otherWrap.getLhs() != 0) {
          SmallVector<int64_t> wrapStarts =
              llvm::to_vector(wrapSlice.getStartIndices());
          SmallVector<int64_t> wrapLimits =
              llvm::to_vector(wrapSlice.getLimitIndices());
          if (wrapSlice.getStrides()[dim] == 1) {
            wrapStarts[dim] = wrapLimits[dim] - otherWrap.getLhs();
          }
          if (canMergeSlicesAlongAxis(dim, slice.getStartIndices(), wrapStarts,
                                      slice.getLimitIndices(), wrapLimits,
                                      slice.getStrides(),
                                      wrapSlice.getStrides())) {

            changed = true;
            auto c2 = lowerWrap(otherWrap, rewriter, /*replace*/ false);
            auto newSlice = stablehlo::SliceOp::create(
                rewriter, slice->getLoc(), slice.getOperand(),
                slice.getStartIndices(), wrapLimits, slice.getStrides());
            newOperands.push_back(newSlice);
            for (int i = 1; i < c2.getOperands().size(); i++) {
              newOperands.push_back(c2.getOperands()[i]);
            }
            i++;
            slice = nullptr;
            break;
          } else {
            break;
          }
        }
      }
      break;
    }

    if (slice) {
      newOperands.push_back(slice.getResult());
    }
  }

  return success(changed);
}

Value ConcatenateOpCreate(
    OpBuilder &builder, Location loc, ArrayRef<Value> inputs, int64_t dimension,
    std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  assert(inputs.size() >= 1);

  if (inputs.size() == 1) {
    if (auto defSliceOp = inputs[0].getDefiningOp<stablehlo::SliceOp>()) {
      // special case if we are using full slices
      if (defSliceOp.getOperand().getType() == defSliceOp.getType()) {
        return defSliceOp.getOperand();
      }
    }
    return inputs[0];
  }

  auto concatOp =
      stablehlo::ConcatenateOp::create(builder, loc, inputs, dimension);
  if (sharding.has_value()) {
    sdy::setShardings(concatOp, *sharding);
  }
  return concatOp.getResult();
}

Value ReshapeOpCreate(OpBuilder &builder, Location loc, Value input,
                      ArrayRef<int64_t> shape,
                      std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  auto inputTy = cast<RankedTensorType>(input.getType());
  if (inputTy.getShape() == shape) {
    return input;
  }

  auto reshapeOp = stablehlo::ReshapeOp::create(
      builder, loc, RankedTensorType::get(shape, inputTy.getElementType()),
      input);
  if (sharding.has_value()) {
    sdy::setShardings(reshapeOp, *sharding);
  }
  return reshapeOp.getResult();
}

Value TransposeOpCreate(
    OpBuilder &builder, Location loc, Value input,
    ArrayRef<int64_t> permutation,
    std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  if (llvm::is_sorted(permutation)) {
    return input;
  }

  auto transposeOp = stablehlo::TransposeOp::create(
      builder, loc, input, builder.getDenseI64ArrayAttr(permutation));
  if (sharding.has_value()) {
    sdy::setShardings(transposeOp, *sharding);
  }
  return transposeOp.getResult();
}

Value SliceOpCreate(OpBuilder &builder, Location loc, Value input,
                    ArrayRef<int64_t> sliceStarts,
                    ArrayRef<int64_t> sliceLimits,
                    ArrayRef<int64_t> sliceStrides,
                    std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
  bool sliceNeeded = false;
  for (auto [start, limit, stride, shape] :
       llvm::zip_equal(sliceStarts, sliceLimits, sliceStrides, inputShape)) {
    if (start != 0 || limit != shape || stride != 1) {
      sliceNeeded = true;
      break;
    }
  }

  if (sliceNeeded) {
    auto sliceOp = stablehlo::SliceOp::create(builder, loc, input, sliceStarts,
                                              sliceLimits, sliceStrides);
    if (sharding.has_value()) {
      sdy::setShardings(sliceOp, *sharding);
    }
    return sliceOp.getResult();
  }
  return input;
}

Value DynamicSliceOpCreate(
    OpBuilder &builder, Location loc, Value input, ArrayRef<Value> sliceStarts,
    ArrayRef<int64_t> sliceSizes,
    std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
  SmallVector<int64_t> constStarts, constLimits, constStrides;
  bool emitSliceOp = true, warnOutOfBoundsAccess = false;
  for (auto [start, size, shape] :
       llvm::zip_equal(sliceStarts, sliceSizes, inputShape)) {
    APInt constStart;
    if (matchPattern(start, m_ConstantInt(&constStart))) {
      auto constStartInt = constStart.getSExtValue();
      auto limit = constStartInt + size;
      if (limit > shape) {
        emitSliceOp = false;
        warnOutOfBoundsAccess = true;
        break;
      }
      constStarts.push_back(constStartInt);
      constLimits.push_back(limit);
      constStrides.push_back(1);
    } else {
      emitSliceOp = false;
      break;
    }
  }

  if (emitSliceOp) {
    return SliceOpCreate(builder, loc, input, constStarts, constLimits,
                         constStrides);
  }

  auto dsOp = stablehlo::DynamicSliceOp::create(builder, loc, input,
                                                sliceStarts, sliceSizes);
  if (warnOutOfBoundsAccess) {
    dsOp.emitWarning("potential out of bounds indexing detected");
  }
  if (sharding.has_value()) {
    sdy::setShardings(dsOp, *sharding);
  }
  return dsOp.getResult();
}

Type GetDotGeneralResultType(Value lhs, Value rhs, Type resElemType,
                             stablehlo::DotDimensionNumbersAttr dotDims) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());

  SmallVector<int64_t> resultShape;
  for (auto dim : dotDims.getLhsBatchingDimensions()) {
    resultShape.push_back(lhsType.getDimSize(dim));
  }
  for (size_t i = 0; i < lhsType.getRank(); ++i) {
    if (llvm::is_contained(dotDims.getLhsBatchingDimensions(), i) ||
        llvm::is_contained(dotDims.getLhsContractingDimensions(), i)) {
      continue;
    }
    resultShape.push_back(lhsType.getDimSize(i));
  }
  for (size_t i = 0; i < rhsType.getRank(); ++i) {
    if (llvm::is_contained(dotDims.getRhsBatchingDimensions(), i) ||
        llvm::is_contained(dotDims.getRhsContractingDimensions(), i)) {
      continue;
    }
    resultShape.push_back(rhsType.getDimSize(i));
  }

  return RankedTensorType::get(resultShape, resElemType);
}

SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> res(perm.size(), 0);
  for (auto en : llvm::enumerate(perm)) {
    res[en.value()] = en.index();
  }
  return res;
}

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, stablehlo::SliceOp op) {
  return transposeSliceHelper(transpose, rewriter, op.getStartIndices(),
                              op.getLimitIndices(), op.getStrides());
}

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicSliceOp op) {
  return transposeSliceHelper(transpose, rewriter,
                              llvm::to_vector(op.getStartIndices()),
                              op.getSliceSizes());
}

Value transposeLikeSliceHelper(stablehlo::BroadcastInDimOp transpose,
                           PatternRewriter &rewriter, stablehlo::SliceOp op) {
  return transposeLikeSliceHelper(transpose, rewriter, op.getStartIndices(),
                              op.getLimitIndices(), op.getStrides());
}

Value transposeLikeSliceHelper(stablehlo::BroadcastInDimOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicSliceOp op) {
  return transposeLikeSliceHelper(transpose, rewriter,
                              llvm::to_vector(op.getStartIndices()),
                              op.getSliceSizes());
}

Value transposeLikeSliceHelper(stablehlo::BroadcastInDimOp transpose,
                           PatternRewriter &rewriter, ArrayRef<int64_t> starts,
                           ArrayRef<int64_t> limits,
                           ArrayRef<int64_t> strides) {
  auto permutation = transpose.getPermutation();
  SmallVector<int64_t> permutedLimit(transpose.getType().getShape().size(), 1);
  SmallVector<int64_t> permutedStrides(transpose.getType().getShape().size(), 1);
  SmallVector<int64_t> permutedStart(transpose.getType().getShape().size(), 0);
  for (auto [permIndex, i] : llvm::enumerate(transpose.getBroadcastDimensions())) {
    permutedStart[i] = starts[permIndex];
    permutedLimit[i] = limits[permIndex];
    permutedStrides[i] = strides[permIndex];
  }
  return SliceOpCreate(rewriter, transpose.getLoc(), transpose.getResult(),
                       permutedStart, permutedLimit, permutedStrides);
}

Value transposeLikeSliceHelper(stablehlo::BroadcastInDimOp transpose,
                           PatternRewriter &rewriter,
                           ArrayRef<Value> sliceStarts,
                           ArrayRef<int64_t> sliceSizes) {
  auto permutation = transpose.getPermutation();
  SmallVector<int64_t> sizes(transpose.getType().getShape().size(), 1);
  Type eT = RankedTensorType::get({}, rewriter.getInt32Type());
  if (sliceStarts.size()) eT = sliceStarts[0].getType();
  Value zero = stablehlo::ConstantOp::create(rewriter, transpose.getLoc(), eT, cast<ElementsAttr>(makeAttr(0, eT)));
  SmallVector<Value> starts(transpose.getType().getShape().size(), zero);
  for (auto [i, permIndex] : llvm::enumerate(transpose.getBroadcastDimensions())) {
    sizes[permIndex] = sliceSizes[i];
    starts[permIndex] = sliceStarts[i];
  }
  return DynamicSliceOpCreate(rewriter, transpose.getLoc(),
                              transpose.getResult(), starts, sizes);
}

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, ArrayRef<int64_t> starts,
                           ArrayRef<int64_t> limits,
                           ArrayRef<int64_t> strides) {
  auto permutation = transpose.getPermutation();
  SmallVector<int64_t> permutedStart(permutation.size()),
      permutedLimit(permutation.size()), permutedStrides(permutation.size());
  for (auto [i, permIndex] : llvm::enumerate(permutation)) {
    permutedStart[i] = starts[permIndex];
    permutedLimit[i] = limits[permIndex];
    permutedStrides[i] = strides[permIndex];
  }
  return SliceOpCreate(rewriter, transpose.getLoc(), transpose.getResult(),
                       permutedStart, permutedLimit, permutedStrides);
}

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           ArrayRef<Value> sliceStarts,
                           ArrayRef<int64_t> sliceSizes) {
  auto permutation = transpose.getPermutation();
  SmallVector<int64_t> sizes(permutation.size());
  SmallVector<Value> starts(permutation.size());
  for (auto [i, permIndex] : llvm::enumerate(permutation)) {
    sizes[i] = sliceSizes[permIndex];
    starts[i] = sliceStarts[permIndex];
  }
  return DynamicSliceOpCreate(rewriter, transpose.getLoc(),
                              transpose.getResult(), starts, sizes);
}


Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, stablehlo::SliceOp op) {
  return sliceTransposeHelper(transpose, rewriter, op.getStartIndices(),
                              op.getLimitIndices(), op.getStrides());
}

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicSliceOp op) {
  return sliceTransposeHelper(transpose, rewriter,
                              llvm::to_vector(op.getStartIndices()),
                              op.getSliceSizes());
}

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicUpdateSliceOp op) {
  auto newUpdate =
      TransposeOpCreate(rewriter, transpose->getLoc(), op.getUpdate(),
                        transpose.getPermutation());
  SmallVector<Value> starts;
  for (auto ind : getInversePermutation(transpose.getPermutation())) {
    starts.push_back(op.getStartIndices()[ind]);
  }
  return stablehlo::DynamicUpdateSliceOp::create(
      rewriter, transpose->getLoc(), transpose.getOperand(), newUpdate, starts);
}

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, ArrayRef<int64_t> starts,
                           ArrayRef<int64_t> limits,
                           ArrayRef<int64_t> strides) {
  SmallVector<int64_t> start, end, step;
  for (auto ind : getInversePermutation(transpose.getPermutation())) {
    start.push_back(starts[ind]);
    end.push_back(limits[ind]);
    step.push_back(strides[ind]);
  }
  return SliceOpCreate(rewriter, transpose.getLoc(), transpose.getOperand(),
                       start, end, step);
}

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           ArrayRef<Value> sliceStarts,
                           ArrayRef<int64_t> sliceSizes) {
  SmallVector<int64_t> sizes;
  SmallVector<Value> starts;
  for (auto ind : getInversePermutation(transpose.getPermutation())) {
    sizes.push_back(sliceSizes[ind]);
    starts.push_back(sliceStarts[ind]);
  }
  return DynamicSliceOpCreate(rewriter, transpose.getLoc(),
                              transpose.getOperand(), starts, sizes);
}

bool isFusible(stablehlo::TransposeOp transpose, Operation *op) {
  if (isa<stablehlo::TransposeOp, stablehlo::BroadcastInDimOp,
          stablehlo::DotGeneralOp>(op)) {
    return true;
  }

  SplatElementsAttr splat;
  if (matchPattern(op, m_Constant(&splat))) {
    return true;
  }

  if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
    auto inputType = cast<RankedTensorType>(reshapeOp.getOperand().getType());
    auto outputType = cast<RankedTensorType>(reshapeOp.getResult().getType());

    auto insertionDims = findReshapeInsertionDims(inputType, outputType);
    if (!insertionDims.empty()) { // fused to a broadcast_in_dim
      return true;
    }

    if (reshapeIsTranspose(reshapeOp)) { // transpose_tranpose elimination
      return true;
    }
  }

  return false;
}

bool isFusible(stablehlo::BroadcastInDimOp transpose, Operation *op) {
  if (isa<stablehlo::TransposeOp, stablehlo::BroadcastInDimOp,
          stablehlo::DotGeneralOp>(op)) {
    return true;
  }

  SplatElementsAttr splat;
  if (matchPattern(op, m_Constant(&splat))) {
    return true;
  }

  if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op)) {
    auto inputType = cast<RankedTensorType>(reshapeOp.getOperand().getType());
    auto outputType = cast<RankedTensorType>(reshapeOp.getResult().getType());

    auto insertionDims = findReshapeInsertionDims(inputType, outputType);
    if (!insertionDims.empty()) { // fused to a broadcast_in_dim
      return true;
    }

    if (reshapeIsTranspose(reshapeOp)) { // transpose_tranpose elimination
      return true;
    }
  }

  return false;
}

bool isFusible(Operation *op, stablehlo::ReshapeOp reshape) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<stablehlo::ReshapeOp>([](auto prevOp) { return true; })
      .Case<stablehlo::BroadcastInDimOp>([&](auto prevOp) {
        if (isInsertDimOp(reshape)) {
          return true;
        }
        auto deletionDims = findReshapeInsertionDims(
            reshape.getType(), reshape.getOperand().getType());
        if (!deletionDims.empty()) {
          SmallVector<int64_t> bcastInsertionDims;
          getSingletonInsertionDims(prevOp, bcastInsertionDims);
          // if all of the dims deleted we part of the insertion dims, we can
          // do the fusion
          if (!bcastInsertionDims.empty()) {
            return llvm::all_of(deletionDims, [&](auto delDim) {
              return llvm::is_contained(bcastInsertionDims, delDim);
            });
          }
        }
        return false;
      })
      .Case<stablehlo::ReduceOp>(
          [&](auto redOp) { return isDeleteDimOp(reshape); })
      .Default([](auto other) { return matchPattern(other, m_Constant()); });
}

bool isFusible(Operation *op, stablehlo::BroadcastInDimOp bcast) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<stablehlo::TransposeOp, stablehlo::BroadcastInDimOp>(
          [](auto prevOp) { return true; })
      .Case<stablehlo::ReshapeOp>(
          [](auto reshape) { return isInsertDimOp(reshape); })
      .Default([](auto other) { return matchPattern(other, m_Constant()); });
}

bool IsTensorFilled(Value input) {
  // Use a worklist-based approach to traverse the SSA def-use chain
  // and determine if the value is known to be a dense (fully-populated) matrix.
  //
  // A value is considered dense if it comes from:
  // - stablehlo ops (except custom_call) - they produce dense outputs if inputs
  // are dense
  // - Block arguments (conservatively assume not dense)
  // - CallOpInterface - check the function body's return values
  SymbolTableCollection symbolTable;

  std::deque<Value> worklist;
  llvm::DenseSet<Value> visited;

  worklist.push_back(input);

  while (!worklist.empty()) {
    Value current = worklist.front();
    worklist.pop_front();

    if (visited.contains(current)) {
      continue;
    }
    visited.insert(current);

    if (matchPattern(current, m_Constant())) { // constants are always filled
      continue;
    }

    // Block arguments are considered dense (inputs to functions)
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      return false;
    }

    Operation *op = current.getDefiningOp();
    if (!op) {
      return false;
    }

    // Handle enzymexla dialect ops with special triangular semantics
    if (auto syrkOp = dyn_cast<enzymexla::SyrkOp>(op)) {
      // syrk produces a dense output only if output_uplo is F (full)
      if (syrkOp.getOutputUplo() != enzymexla::LapackUplo::F) {
        return false;
      }
      continue;
    }

    // Handle CallOpInterface - check the function body
    if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      mlir::ModuleOp modOp = op->getParentOfType<ModuleOp>();
      if (!modOp) {
        return false;
      }
      symbolTable.getSymbolTable(modOp);

      auto callable = callOp.resolveCallableInTable(&symbolTable);
      if (auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callable)) {
        // Find which result index corresponds to our value
        size_t resultIdx = cast<OpResult>(current).getResultNumber();

        // Check function body for return values
        if (!funcOp.isExternal() && !funcOp.getBlocks().empty()) {
          for (Block &block : funcOp.getBlocks()) {
            auto term = block.getTerminator();
            if (!term || resultIdx >= term->getNumResults()) {
              return false;
            }
            worklist.push_back(term->getOperand(resultIdx));
          }
          continue;
        }
      }

      // If we can't resolve the call, conservatively assume not dense
      return false;
    }

    // Handle stablehlo custom_call - conservatively not dense
    if (isa<stablehlo::CustomCallOp>(op)) {
      return false;
    }

    // All other stablehlo ops produce dense outputs
    if (op->getDialect()->getNamespace() == "stablehlo") {
      // Add operands to worklist to continue the traversal
      // Most stablehlo ops preserve density, so we just need to check
      // that all inputs are also dense
      for (auto operand : op->getOperands()) {
        worklist.push_back(operand);
      }
      continue;
    }

    // For other dialects/ops, conservatively not dense
    return false;
  }

  return true;
}

void ExtractBlockIntoFunction(Block *block, ModuleOp modOp, func::FuncOp &func,
                              llvm::SetVector<Value> &capturedValues,
                              llvm::SmallVectorImpl<Type> &resultTypes,
                              OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(modOp.getBody());

  auto retOp = block->getTerminator();
  assert(retOp && "expected block to have a terminator");

  mlir::SymbolTable symbolTable(modOp);

  // traverse the block. we fail if the value is defined outside the block and
  // is not a constant
  llvm::MapVector<Value, DenseElementsAttr> constValMap;
  int64_t argNumCounter = block->getNumArguments();
  SmallVector<int64_t> additionalInputNum;
  for (auto &innerOp : block->getOperations()) {
    for (auto operand : innerOp.getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() == block) {
          continue;
        }
      } else if (auto defOp = operand.getDefiningOp()) {
        if (defOp->getBlock() == block) {
          continue;
        }
      } else {
        llvm_unreachable("unknown operand kind");
      }

      // Operand is a constant - that's fine, we'll clone it
      DenseElementsAttr attr;
      if (matchPattern(operand, m_Constant(&attr))) {
        constValMap[operand] = attr;
        continue;
      }

      if (!capturedValues.contains(operand)) {
        capturedValues.insert(operand);
        additionalInputNum.push_back(argNumCounter++);
      }
    }
  }

  auto funcArgTypes = llvm::to_vector(block->getArgumentTypes());
  for (auto addInput : capturedValues) {
    funcArgTypes.push_back(addInput.getType());
  }

  FunctionType calleeType =
      builder.getFunctionType(funcArgTypes, retOp->getOperandTypes());

  func = func::FuncOp::create(builder, retOp->getLoc(),
                              "extract_block_to_function", calleeType);
  func.setPrivate();
  symbolTable.insert(func);

  // Create the function body by cloning the block
  Block *entryBlock = func.addEntryBlock();
  IRMapping mapping;
  for (int64_t i = 0; i < block->getNumArguments(); i++) {
    mapping.map(block->getArgument(i), entryBlock->getArgument(i));
  }

  for (auto [idx, operand] :
       llvm::zip_equal(additionalInputNum, capturedValues)) {
    mapping.map(operand, entryBlock->getArgument(idx));
  }

  builder.setInsertionPointToEnd(entryBlock);
  for (auto &&[operand, attr] : constValMap) {
    auto newConstOp = stablehlo::ConstantOp::create(builder, retOp->getLoc(),
                                                    cast<ElementsAttr>(attr));
    mapping.map(operand, newConstOp);
  }

  for (auto &innerOp : block->without_terminator()) {
    builder.clone(innerOp, mapping);
  }

  // Clone the return op as func.return
  SmallVector<Value> returnValues;
  for (auto op : retOp->getOperands()) {
    returnValues.push_back(mapping.lookup(op));
    resultTypes.push_back(op.getType());
  }
  func::ReturnOp::create(builder, retOp->getLoc(), returnValues);
}

} // namespace stablehlo

} // namespace mlir
