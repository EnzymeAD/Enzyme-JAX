//===- StripDeadPersonality.cpp - Drop personalities without landing pads -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// The LLVM dialect inliner refuses a callee with a personality. A personality
// is consulted only through landing pads, so a function without one has no
// use for it: a throwing callee unwinds through the function as through any
// function that never had a personality, and with no landing pad to unwind
// to, every call in it is a plain call. Dropping the attribute lets such
// functions inline.
//
//===---------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_STRIPDEADPERSONALITYPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {
struct StripDeadPersonalityPass
    : public enzyme::impl::StripDeadPersonalityPassBase<
          StripDeadPersonalityPass> {
  using StripDeadPersonalityPassBase::StripDeadPersonalityPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp fn) {
      if (!fn.getPersonality())
        return;
      bool hasLandingPad = false;
      fn.walk([&](LLVM::LandingpadOp) { hasLandingPad = true; });
      if (!hasLandingPad)
        fn->removeAttr(fn.getPersonalityAttrName());
    });
  }
};
} // namespace
