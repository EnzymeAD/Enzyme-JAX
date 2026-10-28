//===- WhileLoopInfo.h - While Op range analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

namespace enzyme {

struct WhileLoopInfo {
  struct AffineIndexInfo {
    llvm::APInt scale;
    llvm::APInt offset;
  };

  struct Bounds {
    llvm::APInt min;
    llvm::APInt max;
  };

  WhileLoopInfo(stablehlo::WhileOp op_) : op(op_) {}

  LogicalResult computeInfo();

  stablehlo::WhileOp getOp() { return op; }

  bool isValid() { return start && limit && foundStep; }
  bool isConstantStart() { return constStart.has_value(); }
  bool isConstantLimit() { return constLimit.has_value(); }
  bool isConstantStep() { return constStep.has_value(); }
  bool isConstant() {
    return isConstantStart() && isConstantLimit() && isConstantStep();
  }

  std::optional<int64_t> getConstantStep() { return constStep; }
  std::optional<int64_t> getConstantStart() { return constStart; }
  std::optional<int64_t> getConstantLimit() { return constLimit; }

  bool isStepOne();

  mlir::Value getStart() { return start; }

  mlir::Value getLimit() { return limit; }

  mlir::Value getStep(OpBuilder &builder, const IRMapping &mapping = {});

  // returns the arg number of the iv. assumes computeInfo() has been called and
  // was successful
  size_t getArgNumber() {
    auto &condBlk = op.getCond().front();
    auto condTerm = cast<stablehlo::ReturnOp>(condBlk.getTerminator());
    auto condV = condTerm->getOperand(0);
    auto cond = condV.getDefiningOp<stablehlo::CompareOp>();
    if (!cond ||
        cond.getComparisonDirection() != stablehlo::ComparisonDirection::LT) {
      return -1;
    }
    auto induct = dyn_cast<BlockArgument>(cond.getOperand(0));
    size_t blockArgNum = induct.getArgNumber();
    return blockArgNum;
  }

  // assumes computeInfo() has been called and was successful
  // returns the induction variable in the body of the while op
  Value getInductionVariable() {
    auto blockArgNum = getArgNumber();
    return blockArgNum == -1 ? nullptr
                             : op.getBody().front().getArgument(blockArgNum);
  }

  int64_t getConstantNumIters();
  Value getNumIters(OpBuilder &builder, const IRMapping &mapping = {});

  void propagateAffineIndexInfo();
  void propagateAffineIndexInfo(Value v, AffineIndexInfo curInfo,
                                SmallVectorImpl<Value> &newPropagated);

  void propagateBounds();
  void propagateBounds(Value v, Bounds curBounds,
                       SmallVectorImpl<Value> &newPropagated);

  std::optional<Bounds> getBounds(Value v);

  // The map has an entry per value that is affine in the induction variable;
  // in a long loop body that is many entries, so it is handed out by
  // reference. Bind it with `auto &`.
  const llvm::MapVector<Value, AffineIndexInfo> &getAffineIndexInfo() const {
    return affineIndexInfo;
  }

  llvm::MapVector<Value, Bounds> &getBoundsMap() { return boundsMap; }

  unsigned getBoundsBitWidth() const { return boundsBitWidth; }

  // Whether `v` has the same value on every iteration of the loop. On
  // success `outerValue` is the value outside the loop it equals, when there
  // is one (defined outside, or a carried argument passed straight through);
  // otherwise it is null and `canBeHoisted` lists the ops that recompute `v`
  // outside the loop, each once, in no particular order. With
  // `checkOperands` false only the two direct cases are tried.
  //
  // Answers are remembered across calls, so asking about every value of a
  // loop body costs one walk of the body rather than one per value. The
  // memo describes the IR as it was when it was filled: call
  // `invalidateInvarianceCache` after rewriting the loop, before asking
  // again. (`computeInfo`'s affine-index map has the same contract.)
  bool isConstantAcrossIterations(Value v, bool checkOperands = true);
  bool isConstantAcrossIterations(Value v, Value &outerValue,
                                  SmallVector<Operation *> &canBeHoisted,
                                  bool checkOperands = true);

  // Forget every remembered invariance answer. Needed after the loop body
  // has been rewritten: an erased op's storage can be reused by a new one,
  // and the memo is keyed by value identity.
  void invalidateInvarianceCache() { invarianceCache.clear(); }

private:
  // Whether a value computed in the body is the same on every iteration,
  // i.e. whether every operand of its defining op is. The walk stops at the
  // first operand found to vary, so a varying value costs the walk down to
  // its first varying leaf; without the memo a chain of n such values would
  // cost n^2 across n queries, and a shared invariant subgraph would be
  // re-proved once per path reaching it.
  bool computeInvariance(Value v);

  // The ops that recompute an invariant `v` outside the loop: the defining
  // ops of its operand DAG up to the values defined outside or passed
  // straight through, each listed once.
  void collectHoistableChain(Value v, SmallVectorImpl<Operation *> &chain);

  // Values computed in the body only; the two direct cases are decided
  // without it.
  DenseMap<Value, bool> invarianceCache;

public:
  bool canHoistOperationFromLoop(mlir::stablehlo::DynamicSliceOp sliceOp,
                                 SmallVectorImpl<int64_t> &dimensions);
  // Whether hoisting `sliceOp` over `dimensions` out of the loop can succeed,
  // including that the region it sweeps fits `operand`. Pure: callers decide
  // with it before creating any op, so a refused hoist leaves nothing behind.
  bool canHoistOperationFromLoop(Value operand,
                                 mlir::stablehlo::DynamicSliceOp sliceOp,
                                 SmallVectorImpl<int64_t> &dimensions);
  bool slicedRegionFitsOperand(Value operand,
                               mlir::stablehlo::DynamicSliceOp sliceOp,
                               int64_t sliceIndex);
  bool hoistOperationFromLoop(OpBuilder &builder, Value operand,
                              mlir::stablehlo::DynamicSliceOp sliceOp,
                              int64_t sliceIndex, Value &result);
  bool hoistOperationFromLoop(OpBuilder &builder, Value operand,
                              mlir::stablehlo::DynamicSliceOp sliceOp,
                              SmallVectorImpl<int64_t> &dimensions,
                              Value &result);

  bool canHoistOperationFromLoop(mlir::stablehlo::DynamicUpdateSliceOp dusOp,
                                 SmallVectorImpl<int64_t> &dimensions);
  bool hoistOperationFromLoop(OpBuilder &builder, Value operand, Value update,
                              mlir::stablehlo::DynamicUpdateSliceOp dusOp,
                              int64_t dusIndex, Value &result);
  bool hoistOperationFromLoop(OpBuilder &builder, Value operand, Value update,
                              mlir::stablehlo::DynamicUpdateSliceOp dusOp,
                              SmallVectorImpl<int64_t> &dimensions,
                              Value &result);

private:
  stablehlo::WhileOp op;

  mlir::Value start; // guaranteed to dominate the while op
  std::optional<int64_t> constStart;

  mlir::Value limit; // not guaranteed to dominate the while op
  std::optional<int64_t> constLimit;

  mlir::Value step; // not guaranteed to dominate the while op
  APInt stepInt;
  bool foundStep;
  std::optional<int64_t> constStep;

  llvm::MapVector<Value, AffineIndexInfo> affineIndexInfo;
  DenseSet<Value> affineIndexPropagationVisited;

  llvm::MapVector<Value, Bounds> boundsMap;
  unsigned int boundsBitWidth;

  std::optional<Bounds> computeBounds(Operation *op);

  void computeConstantValues();

  bool isConstantValue(Value v, llvm::APInt &constVal);

  std::optional<int64_t> getConstantStepCalculate();
  std::optional<int64_t> getConstantStartCalculate();
  std::optional<int64_t> getConstantLimitCalculate();

  AffineIndexInfo updateAffineIndexInfo(AffineIndexInfo curInfo,
                                        llvm::APInt scale, llvm::APInt offset);
};

template <typename OpTy>
void hoistStartIndicesOutsideLoop(OpTy op, OpBuilder &builder,
                                  SmallVectorImpl<Value> &newStartIndices,
                                  SmallVectorImpl<int64_t> &dimensions,
                                  WhileLoopInfo &whileLoopInfo);

void hoistChainOfOps(DenseMap<Value, SmallVector<Operation *>> &hoistMap,
                     OpBuilder &builder, stablehlo::WhileOp whileOp,
                     WhileLoopInfo &info,
                     DenseMap<Value, Value> &hoistedValues);

} // end namespace enzyme

} // namespace mlir
