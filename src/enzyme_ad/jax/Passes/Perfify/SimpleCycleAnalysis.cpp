#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include <iostream>
namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_SIMPLECYCLEANALYSISPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;
std::unordered_map<std::string, int> cost_map;
namespace {

struct SimpleCycleAnalysisPass
    : public enzyme::perfify::impl::SimpleCycleAnalysisPassBase<SimpleCycleAnalysisPass> {
  using SimpleCycleAnalysisPassBase::SimpleCycleAnalysisPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    visitOperation(op);
  }
  void visitOperation(Operation *op) {
    // Print the operation itself and some of its properties
    if (auto costOp = dyn_cast<CostOp>(op)) {
      if (!op->getAttrs().empty()) {
        for (NamedAttribute attr : op->getAttrs()) {
          llvm::outs() << attr.getName().getValue() << " : " << attr.getValue() << "\n";

        }
      }
    }
    for (Region &region : op->getRegions())
      visitRegion(region);
  }

  void visitRegion(Region &region) {
    for (Block &block : region.getBlocks())
      visitBlock(block);
  }

  void visitBlock(Block &block) {
    for (Operation &op : block.getOperations())
      visitOperation(&op);
  }  
};

} // namespace
