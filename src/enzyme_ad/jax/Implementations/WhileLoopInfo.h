//===- WhileLoopInfo.h - While Op range analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/MapVector.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

namespace enzyme {

struct WhileLoopInfo {
  struct AffineIndexInfo {
    llvm::APInt scale;
    llvm::APInt offset;
  };

  WhileLoopInfo(stablehlo::WhileOp op_) : op(op_) {}

  LogicalResult computeInfo();

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

  mlir::Value getStep(OpBuilder &builder);

  // assumes computeInfo() has been called and was successful
  // returns the induction variable in the body of the while op
  Value getInductionVariable() {
    auto &condBlk = op.getCond().front();
    auto condTerm = cast<stablehlo::ReturnOp>(condBlk.getTerminator());
    auto condV = condTerm->getOperand(0);
    auto cond = condV.getDefiningOp<stablehlo::CompareOp>();
    auto induct = dyn_cast<BlockArgument>(cond.getOperand(0));
    auto blockArgNum = induct.getArgNumber();
    return op.getBody().front().getArgument(blockArgNum);
  }

  int64_t getConstantNumIters();
  Value getNumIters(OpBuilder &builder);

  void propagateAffineIndexInfo();
  llvm::MapVector<Value, AffineIndexInfo> getAffineIndexInfo() {
    return affineIndexInfo;
  }

  bool isConstantAcrossIterations(Value v);
  bool isConstantAcrossIterations(Value v, Value &outerValue);

  bool canHoistOperationFromLoop(mlir::stablehlo::DynamicSliceOp sliceOp,
                                 SmallVectorImpl<int64_t> &dimensions);
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

  void computeConstantValues();

  bool isConstantValue(Value v, llvm::APInt &constVal);

  std::optional<int64_t> getConstantStepCalculate();
  std::optional<int64_t> getConstantStartCalculate();
  std::optional<int64_t> getConstantLimitCalculate();

  AffineIndexInfo updateAffineIndexInfo(AffineIndexInfo curInfo,
                                        llvm::APInt scale, llvm::APInt offset);
};

} // end namespace enzyme

} // namespace mlir
