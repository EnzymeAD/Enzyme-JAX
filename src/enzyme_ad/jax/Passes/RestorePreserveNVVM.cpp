//===- RestorePreserveNVVM.cpp - Undo PreserveNVVM's begin fixups ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Enzyme's PreserveNVVM(Begin) marks the functions enzyme has a stake in --
// most of them libdevice math definitions -- so they reach the raising alive
// and recognizable: noinline so the calls survive, and a promotion to
// external linkage so nothing internalizes or drops the definitions. It
// records what it changed in prev_* attributes so PreserveNVVM(End) can put
// things back.
//
// In the raising pipeline the marks are only needed up to
// libdevice-funcs-raise: past it the math calls have become dialect ops, and
// what the marks now do is keep every dead libdevice definition alive --
// externally visible, so symbol-dce must assume someone wants it -- through
// the whole rest of the pipeline. This pass is PreserveNVVM(End) said in
// MLIR: restore the recorded linkage and inlining, drop the records, and let
// the next symbol-dce reap what nothing calls anymore.
//
//===---------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RESTOREPRESERVENVVMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// llvm::GlobalValue::LinkageTypes, as PreserveNVVM(Begin) recorded it.
static std::optional<LLVM::Linkage> linkageFromLLVMOrdinal(int64_t v) {
  switch (v) {
  case 0:
    return LLVM::Linkage::External;
  case 1:
    return LLVM::Linkage::AvailableExternally;
  case 2:
    return LLVM::Linkage::Linkonce;
  case 3:
    return LLVM::Linkage::LinkonceODR;
  case 4:
    return LLVM::Linkage::Weak;
  case 5:
    return LLVM::Linkage::WeakODR;
  case 6:
    return LLVM::Linkage::Appending;
  case 7:
    return LLVM::Linkage::Internal;
  case 8:
    return LLVM::Linkage::Private;
  case 9:
    return LLVM::Linkage::ExternWeak;
  case 10:
    return LLVM::Linkage::Common;
  default:
    return std::nullopt;
  }
}

static bool isPrevEntry(Attribute a, StringRef &linkageValue, bool &fixup,
                        bool &alwaysInline, bool &noInline) {
  if (auto s = dyn_cast<StringAttr>(a)) {
    if (s.getValue() == "prev_fixup") {
      fixup = true;
      return true;
    }
    if (s.getValue() == "prev_always_inline") {
      alwaysInline = true;
      return true;
    }
    if (s.getValue() == "prev_no_inline") {
      noInline = true;
      return true;
    }
    return false;
  }
  if (auto arr = dyn_cast<ArrayAttr>(a)) {
    if (arr.size() == 2)
      if (auto key = dyn_cast<StringAttr>(arr[0]))
        if (key.getValue() == "prev_linkage")
          if (auto val = dyn_cast<StringAttr>(arr[1])) {
            linkageValue = val.getValue();
            return true;
          }
    return false;
  }
  return false;
}

struct RestorePreserveNVVMPass
    : public enzyme::impl::RestorePreserveNVVMPassBase<
          RestorePreserveNVVMPass> {
  using RestorePreserveNVVMPassBase::RestorePreserveNVVMPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp fn) {
      auto pass = fn.getPassthrough();
      if (!pass)
        return;

      bool fixup = false, alwaysInline = false, noInline = false;
      StringRef linkageValue;
      SmallVector<Attribute> kept;
      for (Attribute a : *pass)
        if (!isPrevEntry(a, linkageValue, fixup, alwaysInline, noInline))
          kept.push_back(a);
      if (!fixup)
        return;

      if (!linkageValue.empty()) {
        int64_t v;
        if (!linkageValue.getAsInteger(10, v))
          if (auto linkage = linkageFromLLVMOrdinal(v))
            fn.setLinkage(*linkage);
      }

      if (alwaysInline) {
        fn.setAlwaysInline(true);
        fn.setNoInline(false);
      } else if (!noInline) {
        fn.setNoInline(false);
      }

      if (kept.empty())
        fn.removePassthroughAttr();
      else
        fn.setPassthroughAttr(ArrayAttr::get(fn.getContext(), kept));
    });
  }
};

} // namespace
