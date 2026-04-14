
#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/CheckedRewrite.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Types.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "Interfaces/AutoDiffTypeInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include "llvm/ADT/MapVector.h"
#include <iterator>
#include <numeric>
#define DEBUG_TYPE "propagateshapes"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROPAGATESHAPESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;


namespace {

llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> parseArg(const std::string &s) {
  llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> result;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t sep = s.find(':', pos);
    if (sep == std::string::npos) break;
    int key = std::stoi(s.substr(pos, sep - pos));
    pos = sep + 1;

    size_t next = s.find(';', pos);
    std::string vecStr = s.substr(pos, next - pos);

    llvm::SmallVector<int, 3> vec;
    size_t vpos = 0;
    while (vpos < vecStr.size()) {
      size_t comma = vecStr.find(',', vpos);
      if (comma == std::string::npos) comma = vecStr.size();
      vec.push_back(std::stoi(vecStr.substr(vpos, comma - vpos)));
      vpos = comma + 1;
    }

    result[key] = vec;
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  return result;
}
  
struct PropagateShapesPass
    : public enzyme::impl::PropagateShapesPassBase<PropagateShapesPass> {
  using PropagateShapesPassBase::PropagateShapesPassBase;

  llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> shapeMap;

  mlir::func::FuncOp setFuncOperandTypes(mlir::func::FuncOp func) {
    // auto oldType = func.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputs;
    int i = 0;
    for (auto arg : func.getArguments()) {
      auto type = arg.getType();

      if (shapeMap.count(i) > 0) {
        llvm::errs() << "shape found at " << i << "\n";
        // Tensors are passed flattened, so we need to get the size as such
        int64_t totalSize = 1;
        // shapeDims.push_back(shapeMap[i][0]);
        totalSize *= shapeMap[i][0];
        for (int j = 2; j < shapeMap[i].size(); j++) {
          totalSize *= shapeMap[i][j];
        }
        type = mlir::RankedTensorType::get({totalSize},
            cast<mlir::RankedTensorType>(type).getElementType());
      }
      type.dump();
      newInputs.push_back(type);
      i++;
    }
    auto newType = mlir::FunctionType::get(
      func.getContext(), newInputs, newInputs
    );

    auto newFunc = mlir::func::FuncOp::create(func.getLoc(), func.getName(), newType);
    newFunc->dump();

    llvm::errs() << "\nafter set attrs:\n";
    // newFunc->setAttrs(func->getAttrs());
    newFunc->dump();

    func->getBlock()->getOperations().insert(
        mlir::Block::iterator(func), newFunc);
    // Clone body
    auto &oldBlock = func.getBody().front();
    auto &newBlock = *newFunc.addEntryBlock();

    mlir::IRMapping mapping;
    for (auto [oldArg, newArg] :
        llvm::zip(oldBlock.getArguments(), newBlock.getArguments())) {
      mapping.map(oldArg, newArg);
    }

    for (auto &op : oldBlock.without_terminator()) {
      newBlock.push_back(op.clone(mapping));
    }
    newBlock.push_back(oldBlock.getTerminator()->clone(mapping));

    // Replace + erase
    func->replaceAllUsesWith(newFunc);
    func.erase();
    return newFunc;
  }

  void runOnOperation() override {
    shapeMap = parseArg(shapes);

    // debug print shapeMap
    // for (const auto &it : shapeMap) {
    //   int key = it.first;
    //   const auto &vec = it.second;

    //   llvm::outs() << "key " << key << ": [";
    //   for (size_t i = 0; i < vec.size(); ++i) {
    //     llvm::outs() << vec[i];
    //     if (i + 1 < vec.size())
    //       llvm::outs() << ", ";
    //   }
    //   llvm::outs() << "]\n";
    // }

    auto root = getOperation();
    llvm::errs() << "\n=============Initial Root==============\n";
    root->dump();
    // auto context = getOperation()->getContext();

    root->walk([&](mlir::func::FuncOp func) {
      func = setFuncOperandTypes(func);

      llvm::errs() << "\nprinting metadata:\n";
      func.walk([](mlir::Operation *op) {
        llvm::outs() << "Operation: " << op->getName() << "\n";
        for (auto namedAttr : op->getAttrs()) {
          llvm::outs() << "  "
                      << namedAttr.getName() << " = "
                      << namedAttr.getValue() << "\n";
        }
      });
    });

    // auto func = M->lookupSymbol<mlir::func::FuncOp>("main");

    // // Old types
    // auto oldType = func.getFunctionType();

    // // Build new input types
    // llvm::SmallVector<mlir::Type> newInputs;
    // for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
    //   auto type = arg.getType();

    //   if (shapeMap.count[i] > 0) {
    //     llvm::SmallVector<int, 4> shapeDims;
    //     shapeDims.push_back(shapeMap[i][0]);
    //     for (int j = 2; j < shapeMap[i].size; j++) {
    //       shapeDims.push_back(shapeMap[i][j]);
    //     }
    //     type = mlir::RankedTensorType::get(shapeDims,
    //         type.cast<mlir::RankedTensorType>().getElementType());
    //   }

    //   newInputs.push_back(type);
    // }

    // // Keep result types the same
    // auto newFuncType = mlir::FunctionType::get(
    //     func.getContext(),
    //     newInputs,
    //     newInputs);

    // // Update function type
    // func.setType(newFuncType);

    // if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
    //                                         config))) {
    //   signalPassFailure();
    // }
    llvm::errs() << "===========Final function==========\n";
    root->dump();
  }
};

} // end anonymous namespace