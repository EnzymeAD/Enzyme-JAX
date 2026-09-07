//===- RemoveDuplicateFuncDef.cpp - Remove duplicate fund def -------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to remove duplicate function definitions.
//===---------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_REMOVEDUPLICATEFUNCDEFPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct RemoveDuplicateFuncDefPass
    : public enzyme::impl::RemoveDuplicateFuncDefPassBase<
          RemoveDuplicateFuncDefPass> {
  using RemoveDuplicateFuncDefPassBase::RemoveDuplicateFuncDefPassBase;

  static bool areEquivalent(LLVM::LLVMFuncOp funcOp1,
                            LLVM::LLVMFuncOp funcOp2) {
    // Same function.
    if (funcOp1 == funcOp2)
      return true;

    // Both of the functions must be declarations.
    if (funcOp1.isDeclaration() || funcOp2.isDeclaration())
      return false;

    // Check arguments.
    if (funcOp1.getNumArguments() != funcOp2.getNumArguments())
      return false;
    if (funcOp1.getArgumentTypes() != funcOp2.getArgumentTypes())
      return false;

    // Check return trypes.
    if (funcOp1.getResultTypes() != funcOp2.getResultTypes())
      return false;

    // Discardable attributes equivalence.
    if (funcOp1->getDiscardableAttrDictionary() !=
        funcOp2->getDiscardableAttrDictionary())
      return false;

    Region *body1 = funcOp1.getCallableRegion();
    Region *body2 = funcOp2.getCallableRegion();
    if (!body1 || !body2)
      return false;
    return OperationEquivalence::isRegionEquivalentTo(
        body1, body2, OperationEquivalence::IgnoreLocations);
  }

  // Outlining emits one gpu.module per launch site, so a kernel launched from
  // several places (e.g. a ping-pong loop, or the augmented forward and
  // gradient of one kernel) yields byte-identical modules that each get their
  // own binary, stub and registration ctor.
  static bool areEquivalent(gpu::GPUModuleOp moduleOp1,
                            gpu::GPUModuleOp moduleOp2) {
    if (moduleOp1 == moduleOp2)
      return true;

    // Everything but the name -- targets, offloading handler, ... -- has to
    // agree for the modules to compile to the same binary.
    auto attrsWithoutName = [](gpu::GPUModuleOp moduleOp) {
      NamedAttrList attrs(moduleOp->getAttrDictionary());
      attrs.erase(SymbolTable::getSymbolAttrName());
      return attrs.getDictionary(moduleOp.getContext());
    };
    if (attrsWithoutName(moduleOp1) != attrsWithoutName(moduleOp2))
      return false;

    return OperationEquivalence::isRegionEquivalentTo(
        &moduleOp1.getBodyRegion(), &moduleOp2.getBodyRegion(),
        OperationEquivalence::IgnoreLocations);
  }

  // Point every use of `duplicate` at `canonicalName` and mark it for removal.
  // Bails out if the uses cannot all be rewritten, since erasing it would then
  // leave a dangling symbol reference.
  template <typename OpT>
  static void mergeInto(OpT &duplicate, StringAttr canonicalName,
                        ModuleOp moduleOp,
                        SmallVectorImpl<Operation *> &toRemove) {
    if (failed(SymbolTable::replaceAllSymbolUses(duplicate, canonicalName,
                                                 moduleOp)))
      return;
    toRemove.push_back(duplicate);
    duplicate = nullptr;
  }

  // Deduplicate `ops`, which must all live directly in `moduleOp`.
  template <typename OpT>
  static void deduplicate(SmallVectorImpl<OpT> &ops, ModuleOp moduleOp,
                          SmallVectorImpl<Operation *> &toRemove) {
    for (size_t i = 0, e = ops.size(); i < e; ++i) {
      if (!ops[i])
        continue;
      for (size_t j = i + 1; j < e; ++j) {
        if (!ops[j] || !areEquivalent(ops[i], ops[j]))
          continue;
        mergeInto(ops[j], ops[i].getSymNameAttr(), moduleOp, toRemove);
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<Operation *> toRemove;

    // Only symbols owned by this module's symbol table are considered: names
    // are unique within a symbol table, but a gpu.module is one of its own, so
    // kernels outlined from the same source share a name across modules.
    auto funcOps = llvm::to_vector(moduleOp.getOps<LLVM::LLVMFuncOp>());
    deduplicate(funcOps, moduleOp, toRemove);

    auto gpuModuleOps = llvm::to_vector(moduleOp.getOps<gpu::GPUModuleOp>());
    deduplicate(gpuModuleOps, moduleOp, toRemove);

    // At this point it should be safe to remove the duplicates.
    for (Operation *op : toRemove)
      op->erase();
  }
};
} // end anonymous namespace