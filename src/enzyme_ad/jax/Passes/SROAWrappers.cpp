//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "sroa-julia-wrappers"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct SROAJuliaWrappersPass
    : public SROAJuliaWrappersPassBase<SROAJuliaWrappersPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<LLVM::LLVMFuncOp> fs;
    auto st = SymbolTable::getNearestSymbolTable(m);
    auto res = m->walk([&](LLVM::LLVMFuncOp f) {
      if (f.getCConv() == LLVM::CConv::PTX_Kernel) {
        fs.push_back(f);
        if (f.walk([&](LLVM::CallOp call) {
               auto callee = call.getCallee();
               if (callee) {
                 auto f = dyn_cast<LLVM::LLVMFuncOp>(m.lookupSymbol(*callee));
                 if (!f) {
                   llvm::errs() << "Callee not llvm.func\n";
                   return WalkResult::interrupt();
                 }
                 fs.push_back(f);
               }
               return WalkResult::advance();
             }).wasInterrupted())
          return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });
    if (res.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    llvm::dbgs() << fs.size() << "\n";
    for (auto f : fs)
      llvm::dbgs() << f.getName() << "\n";
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createSROAJuliaWrappersPass() {
  return std::make_unique<SROAJuliaWrappersPass>();
}
} // namespace enzyme
} // namespace mlir
