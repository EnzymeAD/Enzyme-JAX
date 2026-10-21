#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_PERFIFYTOLEANPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {
struct PerfifyToLeanPass
    : public enzyme::perfify::impl::PerfifyToLeanPassBase<PerfifyToLeanPass> {
  using PerfifyToLeanPassBase::PerfifyToLeanPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) { visitOperation(op); });
  }

  void visitOperation(Operation *op) {
    llvm::outs() << "Logging the operation " << op->getName() << "\n";
  }
};
} // namespace
