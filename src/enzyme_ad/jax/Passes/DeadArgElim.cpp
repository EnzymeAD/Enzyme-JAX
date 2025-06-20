//===- DeadArgElim.h - Dead argument elimination -----------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remove trivially dead function arguments.
//
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CFGToSCF.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DEADARGELIMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct DeadArgElimPass
    : public enzyme::impl::DeadArgElimPassBase<DeadArgElimPass> {
  using DeadArgElimPassBase::DeadArgElimPassBase;

  void runOnOperation() override {
    using llvm::errs;
    SymbolTableCollection symbolTable;
    auto funcName = FlatSymbolRefAttr::get(getOperation()->getContext(), func);
    auto funcOp = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        getOperation(), funcName);
    if (!funcOp) {
      getOperation()->emitError() << "Failed to find function '" << func << "'";
      return signalPassFailure();
    }

    llvm::BitVector toErase(funcOp.getNumArguments(), false);
    for (auto &&[idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg.getUsers().empty())
        toErase[idx] = true;
    }

    if (failed(funcOp.eraseArguments(toErase))) {
      return signalPassFailure();
    }
  }
};
}; // namespace
