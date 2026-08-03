//===- HoistAllocas.cpp - Move allocations to where they are freed -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass that moves stack allocations to the start of the
// region whose exit frees them -- the nearest enclosing
// AutomaticAllocationScope
// -- which is where everything downstream expects to find them.
//
// An allocation is freed when the scope holding it is left, so where it sits
// within that scope says nothing about how long it lives. Emitted in place it
// commonly ends up in a block that is only conditionally reached: lowering a
// parallel loop to a kernel, for instance, wraps the whole body in a bounds
// check, and everything the body allocated is then inside that check.
//
// Nothing downstream expects that. LLVM's SROA only ever considers allocations
// in the entry block of a function, so one left in a guarded block is never
// promoted however simple its uses are, and the stack it takes is spent for the
// whole call. Clang emits every fixed-size allocation in the entry block for
// this reason.
//
// The scope this hoists to is the nearest one, so an allocation inside a loop
// stays inside it: scf.for, scf.forall and scf.parallel free what their bodies
// allocate on every iteration, and only what sits in something like an scf.if,
// which frees nothing, travels any distance.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_HOISTALLOCASPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// Whether the allocation takes the same amount of stack however it is reached,
// which is what makes taking it unconditionally the same as taking it here.
static bool isFixedSize(Operation *op) {
  if (auto alloca = dyn_cast<LLVM::AllocaOp>(op)) {
    APInt count;
    return matchPattern(alloca.getArraySize(), m_ConstantInt(&count));
  }
  if (auto alloca = dyn_cast<memref::AllocaOp>(op))
    return alloca.getType().hasStaticShape() &&
           alloca.getDynamicSizes().empty();
  return false;
}

// The block whose exit frees what is allocated in it, and where an allocation
// therefore belongs.
static Block *allocationBlockOf(Operation *op) {
  Operation *scope =
      op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  if (!scope || scope->getNumRegions() == 0)
    return nullptr;
  Region &region = scope->getRegion(0);
  if (region.empty())
    return nullptr;
  return &region.front();
}

static void hoistAllocas(Operation *root, DominanceInfo &domInfo) {
  SmallVector<Operation *> allocas;
  root->walk([&](Operation *op) {
    if (isa<LLVM::AllocaOp, memref::AllocaOp>(op))
      allocas.push_back(op);
  });

  for (Operation *alloca : allocas) {
    if (!isFixedSize(alloca))
      continue;
    Block *block = allocationBlockOf(alloca);
    if (!block || alloca->getBlock() == block)
      continue;

    // As early in the block as everything the allocation is built from allows,
    // since what uses it lies after where it was.
    Operation *after = nullptr;
    for (Value operand : alloca->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || def->getBlock() != block)
        continue;
      if (!after || after->isBeforeInBlock(def))
        after = def;
    }
    Operation *before = after ? after->getNextNode() : &block->front();
    if (!before)
      continue;
    if (!llvm::all_of(alloca->getOperands(), [&](Value operand) {
          return domInfo.properlyDominates(operand, before);
        }))
      continue;

    alloca->moveBefore(before);
  }
}

struct HoistAllocasPass
    : public enzyme::impl::HoistAllocasPassBase<HoistAllocasPass> {
  using HoistAllocasPassBase::HoistAllocasPassBase;

  void runOnOperation() override {
    hoistAllocas(getOperation(), getAnalysis<DominanceInfo>());
  }
};

} // end anonymous namespace
