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

  mlir::Value start; // garanteed to dominate the while op
  mlir::Value limit; // not garanteed to dominate the while op
  mlir::Value step;  // not garanteed to dominate the while op
  bool inclusive = false;

  WhileLoopInfo(WhileOp op_) : op(op_) {}

  LogicalResult computeInfo();

  bool isValid() { return start && limit && step; }
  bool isConstant() {
    return getConstantStep().has_value() && getConstantStart().has_value() &&
           getConstantLimit().has_value();
  }

  std::optional<DenseIntOrFPElementsAttr> getConstantStep();
  std::optional<DenseIntOrFPElementsAttr> getConstantStart();
  std::optional<DenseIntOrFPElementsAttr> getConstantLimit();

  int64_t getConstantNumIters();
  Value getNumIters(OpBuilder &builder);
};

} // end namespace enzyme

} // namespace mlir
