//===- SharedMemoryPromotion.h - Polyhedral GPU Shared Mem ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Polyhedral shared-memory and thread-private register opportunity analysis and
// code generation adapted from PPCG (gpu_group.c / gpu.c) for MLIR Affine GPU
// kernels.
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_TRANSFORMS_SHAREDMEMORYPROMOTION_H
#define POLYMER_TRANSFORMS_SHAREDMEMORYPROMOTION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include <cstdint>

namespace polymer {

struct SharedMemoryPromotionOptions {
  int64_t maxSharedMemBytes = 49152;
  bool enableSharedMem = true;
  bool enablePrivateMem = true;
};

/// Analyzes `gpu.par.grid` `affine.parallel` regions inside `root` for data
/// reuse and uncoalesced global memory accesses, promoting tiles to
/// block-shared memory (`memref.alloca` at the top of `gpu.par.grid`) and
/// per-thread register accumulators (`memref.alloca {polymer.private}`).
mlir::LogicalResult
promoteSharedMemory(mlir::Operation *root,
                    const SharedMemoryPromotionOptions &options = {});

/// Reduces thread-indexed `{polymer.private}` allocas inside `gpu.par.grid` to
/// per-thread scalar `memref.alloca`s at the top of the fused `gpu.par.block`
/// after loop undistribution.
mlir::LogicalResult reducePrivateAllocas(mlir::Operation *root);

} // namespace polymer

#endif // POLYMER_TRANSFORMS_SHAREDMEMORYPROMOTION_H
