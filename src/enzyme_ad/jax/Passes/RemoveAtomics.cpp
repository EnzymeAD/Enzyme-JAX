//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "../Polymer/Support/IslScop.h"
#include "../Polymer/Target/ISL.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/Support/DebugLog.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "remove-atomics"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_REMOVEATOMICSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Operation *getClosedWorldScope(enzyme::AffineAtomicRMWOp rmw) {
  return rmw->getParentOfType<enzymexla::GPUWrapperOp>();
}

bool atomicCanBeRemoved(enzyme::AffineAtomicRMWOp rmw, Operation *scope) {
  TypedValue<MemRefType> memref = rmw.getMemref();

  SmallVector<MemoryEffectOpInterface> memEffectOps;
  auto res = scope->walk([&](Operation *op) {
    if (op == scope)
      return WalkResult::advance();
    if (auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (memEffectOp.getEffectOnValue<MemoryEffects::Write>(memref)) {
        LDBG() << "Found write: " << *memEffectOp;
        memEffectOps.push_back(memEffectOp);
      }
      assert(!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>());
      return WalkResult::skip();
    } else if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return WalkResult::advance();
    } else {
      LDBG() << "Found non-pure op: " << *op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted())
    return false;

  // We should have `rmw` in here
  assert(memEffectOps.size() >= 1);

  return true;
}

struct RemoveAtomicsPass
    : public enzyme::impl::RemoveAtomicsPassBase<RemoveAtomicsPass> {
  using RemoveAtomicsPassBase::RemoveAtomicsPassBase;
  void runOnOperation() override {

    #if 0
    getOperation()->walk([](enzyme::AffineAtomicRMWOp rmw) {
      LDBG() << "Processing " << rmw << " to " << rmw.getMemref();

      Operation *scope = getClosedWorldScope(rmw);
      if (!scope)
        return;

      if (!atomicCanBeRemoved(rmw, scope))
        return;

      LDBG() << "Success";
    });
    #endif
    getOperation()->walk([](enzymexla::GPUWrapperOp gwo) {
      LDBG() << "Processing " << gwo;
      std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(gwo);
    });
  }
};
} // namespace
