//===- LoopUndistribute.h - Fuse & Interchange GPU Block Loops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TRANSFORMS_LOOPUNDISTRIBUTE_H
#define POLYMER_TRANSFORMS_LOOPUNDISTRIBUTE_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace polymer {

static constexpr char kGpuGridParAttr[] = "gpu.par.grid";
static constexpr char kGpuBlockParAttr[] = "gpu.par.block";

bool isGridParallel(mlir::Operation *op);
bool isBlockParallel(mlir::Operation *op);
mlir::affine::AffineParallelOp getAffineGridParallel(mlir::Operation *op);
mlir::affine::AffineParallelOp getAffineBlockParallel(mlir::Operation *op);

/// Undistributes `gpu.par.block` `affine.parallel` loops inside every
/// `gpu.par.grid` region of `root`: interchanges enclosing `affine.for` /
/// `scf.for` / `affine.if` / `scf.if` operations inside a single thread-block
/// parallel loop and fuses adjacent `gpu.par.block` loops with minimal
/// `enzymexla.barrier` synchronizations.
mlir::LogicalResult undistributeFunctions(mlir::Operation *root);

} // namespace polymer

#endif // POLYMER_TRANSFORMS_LOOPUNDISTRIBUTE_H
