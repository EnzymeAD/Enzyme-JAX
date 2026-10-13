//===- DiscardUnreferencedLinkonce.cpp - GlobalDCE for linkonce functions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// symbol-dce keeps every symbol that is not private, so a linkonce
// definition the inliner has folded into all its callers survives it. Each
// translation unit that uses such a function defines its own copy, so an
// unreferenced definition is discardable, as in LLVM's GlobalDCE. A weak
// definition is not: an explicit template instantiation is weak_odr and the
// only definition its `extern template` users will find.
//
//===---------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DISCARDUNREFERENCEDLINKONCEPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {
static bool isDiscardableLinkage(LLVM::Linkage linkage) {
  switch (linkage) {
  case LLVM::Linkage::Linkonce:
  case LLVM::Linkage::LinkonceODR:
    return true;
  default:
    return false;
  }
}

struct DiscardUnreferencedLinkoncePass
    : public enzyme::impl::DiscardUnreferencedLinkoncePassBase<
          DiscardUnreferencedLinkoncePass> {
  using DiscardUnreferencedLinkoncePassBase::
      DiscardUnreferencedLinkoncePassBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    // Erasing one definition can leave another unreferenced.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<LLVM::LLVMFuncOp> dead;
      root->walk([&](LLVM::LLVMFuncOp fn) {
        if (fn.isExternal() || !isDiscardableLinkage(fn.getLinkage()))
          return;
        if (SymbolTable::symbolKnownUseEmpty(fn, root))
          dead.push_back(fn);
      });
      for (auto fn : dead) {
        fn.erase();
        changed = true;
      }
    }
  }
};
} // namespace
