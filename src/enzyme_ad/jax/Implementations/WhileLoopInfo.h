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

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir {

namespace enzyme {

struct WhileLoopInfo {
  WhileOp op;
  llvm::MapVector<Value, APInt> inductionVarOffsets;
  mlir::Value start; // guaranteed to dominate the while op
  mlir::Value limit; // not guaranteed to dominate the while op
  mlir::Value step;  // not guaranteed to dominate the while op

  WhileLoopInfo(WhileOp op_) : op(op_) {}

  LogicalResult computeInfo();

  bool isValid() { return start && limit && step; }
  bool isConstant() {
    return getConstantStep().has_value() && getConstantStart().has_value() &&
           getConstantLimit().has_value();
  }

  std::optional<int64_t> getConstantStep();
  std::optional<int64_t> getConstantStart();
  std::optional<int64_t> getConstantLimit();

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

  void propagateInductionVarOffsets();
  llvm::MapVector<Value, APInt> getInductionVarOffsets() {
    return inductionVarOffsets;
  }

private:
  APInt updateOffset(APInt curOffset, APInt update) {
    if (curOffset.getBitWidth() != update.getBitWidth())
      update = update.sextOrTrunc(curOffset.getBitWidth());
    return curOffset + update;
  }
};

} // end namespace enzyme

} // namespace mlir
