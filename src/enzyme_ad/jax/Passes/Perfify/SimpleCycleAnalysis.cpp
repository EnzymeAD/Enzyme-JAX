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
std::unordered_map<std::string, int64_t> cost_map;
namespace {

struct SimpleCycleAnalysisPass
    : public enzyme::perfify::impl::SimpleCycleAnalysisPassBase<
          SimpleCycleAnalysisPass> {
  using SimpleCycleAnalysisPassBase::SimpleCycleAnalysisPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    visitOperation(op);
  }
  void visitOperation(Operation *op) {
    if (auto costOp = dyn_cast<CostOp>(op)) {
      if (!op->getAttrs().empty()) {
          NamedAttribute cost_attr = op->getAttrs()[0];
          NamedAttribute op_attr = op->getAttrs()[1];
          // llvm::outs() << cost_attr.getName().getValue() << " : " << cost_attr.getValue()
          //              << "\n";
          // llvm::outs() << op_attr.getName().getValue() << " : " << op_attr.getValue()
          //              << "\n";
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cost_attr.getValue())) {
            if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(op_attr.getValue())) {
              int64_t value = intAttr.getInt();
              if (cost_map.find(strAttr.getValue().str()) == cost_map.end()) {
                cost_map[strAttr.getValue().str()] = value;  
                llvm::outs() << "Op:[" << strAttr.getValue().str() << "] Cost:[" << cost_map[strAttr.getValue().str()] << "]\n";
              }
            } 
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
