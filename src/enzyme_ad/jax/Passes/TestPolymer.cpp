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

#include "../polymer/mlir/include/mlir/Conversion/Polymer/Support/IslScop.h"
#include "../polymer/mlir/include/mlir/Conversion/Polymer/Target/ISL.h"
#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "remove-atomics"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TESTPOLYMERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct TestPolymerPass
    : public enzyme::impl::TestPolymerPassBase<TestPolymerPass> {
  using TestPolymerPassBase::TestPolymerPassBase;
  void runOnOperation() override {
    getOperation()->walk([](func::FuncOp gwo) {
      LDBG() << "Processing " << gwo;
      std::unique_ptr<polymer::IslScop> scop =
          polymer::createIslFromFuncOp(gwo);
      scop->buildSchedule();
      scop->dumpSchedule(llvm::errs());
      llvm::outs() << "Schedule:\n";
      isl_schedule_dump(scop->getScheduleTree().get());
      llvm::outs() << "Accesses:\n";
      scop->dumpAccesses(llvm::dbgs());
    });
    getOperation()->walk([](enzymexla::GPUWrapperOp gwo) {
      LDBG() << "Processing " << gwo;
      std::unique_ptr<polymer::IslScop> scop =
          polymer::createIslFromFuncOp(gwo);
      scop->buildSchedule();
      scop->dumpSchedule(llvm::errs());
      llvm::outs() << "Schedule:\n";
      isl_schedule_dump(scop->getScheduleTree().get());
      llvm::outs() << "Accesses:\n";
      scop->dumpAccesses(llvm::dbgs());
    });
  }
};
} // namespace
