//===- WhileLoopInfo.h - While Op range analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace mlir {

namespace enzyme {

struct WhileLoopInfo {
  WhileOp op;

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

  int64_t getConstantNumIters();
  Value getNumIters(OpBuilder &builder);
};

} // end namespace enzyme

} // namespace mlir
