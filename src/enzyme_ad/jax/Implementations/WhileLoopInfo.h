//===- WhileLoopInfo.h - While Op range analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/DenseMap.h"

#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir {

namespace enzyme {

struct WhileLoopInfo {
  WhileOp op;
  DenseMap<Value, APInt> inductionVarOffsets;
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

  Value getInductionVariable() { return op.getBody().front().getArgument(0); }

  int64_t getConstantNumIters();
  Value getNumIters(OpBuilder &builder);

  void propagateInductionVarOffsets();
  DenseMap<Value, APInt> getInductionVarOffsets() {
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
