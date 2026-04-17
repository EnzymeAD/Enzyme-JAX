
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
#include "mlir/Interfaces/InferTypeOpInterface.h"
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
#include <cstdint>
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

llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>>
parseArg(const std::string &s) {
  llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> result;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t sep = s.find(':', pos);
    if (sep == std::string::npos)
      break;
    int key = std::stoi(s.substr(pos, sep - pos));
    pos = sep + 1;

    size_t next = s.find(';', pos);
    std::string vecStr = s.substr(pos, next - pos);

    llvm::SmallVector<int, 3> vec;
    size_t vpos = 0;
    while (vpos < vecStr.size()) {
      size_t comma = vecStr.find(',', vpos);
      if (comma == std::string::npos)
        comma = vecStr.size();
      vec.push_back(std::stoi(vecStr.substr(vpos, comma - vpos)));
      vpos = comma + 1;
    }

    result[key] = vec;
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  return result;
}

struct ShapeInfoState {
  llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> shapeMap;
};

SmallVector<int64_t> getShapeFromOp(mlir::Operation *op,
                                    ShapeInfoState &state) {
  int srcIdx = -1;
  int dim0 = -1;
  int dim1 = -1;
  if (auto srcIdxAttr = op->getAttrOfType<IntegerAttr>("sourceArgIdx")) {
    srcIdx = srcIdxAttr.getInt();
  } else {
    return SmallVector<int64_t>();
  }

  if (auto dim0Attr = op->getAttrOfType<StringAttr>("dim.0")) {
    if (dim0Attr.getValue() == "ldim") {
      dim0 = state.shapeMap[srcIdx][1];
    } else if (dim0Attr.getValue() == "row") {
      dim0 = state.shapeMap[srcIdx][2];
    } else if (dim0Attr.getValue() == "col") {
      dim0 = state.shapeMap[srcIdx][3];
    } else if (dim0Attr.getValue() == "ldim.col") {
      dim0 = state.shapeMap[srcIdx][1] * state.shapeMap[srcIdx][3];
    } else if (dim0Attr.getValue() == "ldim.row") {
      dim0 = state.shapeMap[srcIdx][1] * state.shapeMap[srcIdx][2];
    }
  }
  if (auto dim1Attr = op->getAttrOfType<StringAttr>("dim.1")) {
    if (dim1Attr.getValue() == "ldim") {
      dim1 = state.shapeMap[srcIdx][1];
    } else if (dim1Attr.getValue() == "row") {
      dim1 = state.shapeMap[srcIdx][2];
    } else if (dim1Attr.getValue() == "col") {
      dim1 = state.shapeMap[srcIdx][3];
    } else if (dim1Attr.getValue() == "ldim.col") {
      dim1 = state.shapeMap[srcIdx][1] * state.shapeMap[srcIdx][3];
    }
  }
  SmallVector<int64_t> newShape;

  if (dim1 == -1) {
    if (dim0 == -1) {
      return newShape;
    }
    newShape = {dim0};
  } else {
    newShape = {dim0, dim1};
  }
  return newShape;
}

struct RemoveDynamicReshapePattern
    : OpRewritePattern<stablehlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("replaceWithStaticReshape")) {
      return success();
    }
    Value src = op.getOperand();
    rewriter.replaceOp(op, src);
    return success();
  }
};

struct RefineReshapePattern : OpRewritePattern<stablehlo::ReshapeOp> {

  RefineReshapePattern(MLIRContext *ctx, ShapeInfoState &state)
      : OpRewritePattern(ctx), state(state) {}

  LogicalResult matchAndRewrite(stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = cast<RankedTensorType>(op.getType());
    auto elemTy = oldTy.getElementType();

    SmallVector<int64_t> newShape = getShapeFromOp(op, state);
    auto newTy = RankedTensorType::get(newShape, elemTy);

    if (newTy == oldTy || newShape.size() <= 0)
      return failure();

    auto newOp = rewriter.clone(*op);
    newOp->getResult(0).setType(newTy);
    rewriter.replaceOp(op, newOp);
    return success();
  }

private:
  ShapeInfoState &state;
};

struct RefineSlicePattern : OpRewritePattern<stablehlo::SliceOp> {

  RefineSlicePattern(MLIRContext *ctx, ShapeInfoState &state)
      : OpRewritePattern(ctx), state(state) {}

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = cast<RankedTensorType>(op.getType());
    auto elemTy = oldTy.getElementType();
    auto newShape = getShapeFromOp(op, state);
    auto newTy = RankedTensorType::get(newShape, elemTy);
    if (newTy == oldTy || newShape.size() <= 0)
      return failure();

    // auto limit = op.getLimitIndices();
    auto newLimitAttr = rewriter.getDenseI64ArrayAttr(newShape);
    llvm::errs() << "rewriting slice\n";

    stablehlo::SliceOp newOp = cast<stablehlo::SliceOp>(rewriter.clone(*op));
    newOp.getResult().setType(newTy);
    newOp.setLimitIndices(newLimitAttr);
    rewriter.replaceOp(op, newOp);

    // rewriter.modifyOpInPlace(op, [&] {
    //   op.getResult().setType(newTy);
    //   op.setLimitIndices(newLimitAttr);
    // });
    return success();
  }

private:
  ShapeInfoState &state;
};

struct RefineBroadcastPattern : OpRewritePattern<stablehlo::BroadcastInDimOp> {

  RefineBroadcastPattern(MLIRContext *ctx, ShapeInfoState &state)
      : OpRewritePattern(ctx), state(state) {}

  LogicalResult matchAndRewrite(stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = cast<RankedTensorType>(op.getType());
    auto elemTy = oldTy.getElementType();
    auto newShape = getShapeFromOp(op, state);
    auto newTy = RankedTensorType::get(newShape, elemTy);
    if (newTy == oldTy || newShape.size() <= 0)
      return failure();

    // auto limit = op.getLimitIndices();
    // auto newLimitAttr = rewriter.getDenseI64ArrayAttr(newShape);

    stablehlo::BroadcastInDimOp newOp =
        cast<stablehlo::BroadcastInDimOp>(rewriter.clone(*op));
    newOp.getResult().setType(newTy);
    rewriter.replaceOp(op, newOp);

    // rewriter.modifyOpInPlace(op, [&] {
    //   op.getResult().setType(newTy);
    //   op.setLimitIndices(newLimitAttr);
    // });
    return success();
  }

private:
  ShapeInfoState &state;
};

struct MergeTransposeSelectPattern : OpRewritePattern<stablehlo::SelectOp> {

  MergeTransposeSelectPattern(MLIRContext *ctx, ShapeInfoState &state)
      : OpRewritePattern(ctx), state(state) {}

  LogicalResult matchAndRewrite(stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto isTransposeSelect =
            op->getAttrOfType<BoolAttr>("isTransposeSelect")) {
      if (auto srcIdxAttr = op->getAttrOfType<IntegerAttr>("sourceArgIdx")) {
        if (state.shapeMap[srcIdxAttr.getInt()][0]) {
          rewriter.replaceOp(op, op.getOnTrue());
        } else {
          rewriter.replaceOp(op, op.getOnFalse());
        }
        return success();
      }
    }

    return failure();
  }

private:
  ShapeInfoState &state;
};

struct DotGeneralTypePropagationPattern
    : OpRewritePattern<stablehlo::DotGeneralOp> {
  DotGeneralTypePropagationPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = cast<RankedTensorType>(op.getType());
    auto elemTy = oldTy.getElementType();

    auto newTy =
        cast<mlir::RankedTensorType>(stablehlo::GetDotGeneralResultType(
            op.getLhs(), op.getRhs(), elemTy, op.getDotDimensionNumbers()));
    if (newTy == oldTy)
      return failure();

    // auto limit = op.getLimitIndices();
    // auto newLimitAttr = rewriter.getDenseI64ArrayAttr(newShape);
    stablehlo::DotGeneralOp newOp =
        cast<stablehlo::DotGeneralOp>(rewriter.clone(*op));
    newOp.getResult().setType(newTy);
    rewriter.replaceOp(op, newOp);

    // rewriter.modifyOpInPlace(op, [&] {
    //   op.getResult().setType(newTy);
    //   op.setLimitIndices(newLimitAttr);
    // });
    return success();
  }
};

struct TypePropagationPattern : RewritePattern {
  TypePropagationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto iface = dyn_cast<InferTypeOpInterface>(op);
    if (!iface)
      return failure(); // not a shape-inference-capable op

    SmallVector<Type> inferredTypes;

    LogicalResult res = iface.inferReturnTypes(
        op->getContext(), op->getLoc(), op->getOperands(),
        op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
        inferredTypes);

    if (failed(res))
      return failure();

    // Check if anything actually changes
    bool changed = false;
    for (auto [oldTy, newTy] : llvm::zip(op->getResultTypes(), inferredTypes)) {
      if (oldTy != newTy) {
        changed = true;
        break;
      }
    }

    if (!changed)
      return failure();

    llvm::errs() << "trying to replace\n";
    // Recreate op with corrected types (required in MLIR)
    auto newOp = rewriter.clone(*op);
    for (int idx = 0; idx < inferredTypes.size(); idx++) {
      newOp->getResult(idx).setType(inferredTypes[idx]);
    }
    rewriter.replaceOp(op, newOp);

    // rewriter.modifyOpInPlace(op, [&] {
    //   for (int idx = 0; idx < inferredTypes.size(); idx++) {
    //     op->getResult(idx).setType(inferredTypes[idx]);
    //   }
    // });

    return success();
  }
};

struct DeleteWrongTransposePattern : RewritePattern {
  DeleteWrongTransposePattern(MLIRContext *context, ShapeInfoState &state)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        state(state) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    int srcIdx = -1;
    if (auto srcIdxAttr = op->getAttrOfType<IntegerAttr>("sourceArgIdx")) {
      srcIdx = srcIdxAttr.getInt();
    } else {
      return failure();
    }

    // Delete reshapes that are part of the wrong transpose
    if (auto isTransposed = op->getAttrOfType<BoolAttr>("transposed")) {
      if (isTransposed.getValue() != state.shapeMap[srcIdx][0]) {
        llvm::SmallPtrSet<Operation *, 16> visited;
        llvm::SmallVector<Operation *, 16> postOrder;

        // DFS to collect users (forward slice)
        std::function<void(Operation *)> dfs = [&](Operation *op) {
          if (!visited.insert(op).second)
            return;

          for (Value result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
              dfs(user);
            }
          }

          postOrder.push_back(op); // post-order = safe deletion order
        };

        dfs(op);

        // Erase in post-order (users first)
        for (Operation *op : postOrder) {
          if (op->use_empty()) {
            rewriter.eraseOp(op);
          }
        }
        return success();
      }
    }
    return failure();
  }

private:
  ShapeInfoState &state;
};

struct PropagateShapesPass
    : public enzyme::impl::PropagateShapesPassBase<PropagateShapesPass> {
  using PropagateShapesPassBase::PropagateShapesPassBase;

  llvm::SmallDenseMap<int, llvm::SmallVector<int, 3>> shapeMap;

  void setFuncOperandTypes(mlir::func::FuncOp func) {
    // auto oldType = func.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputs;
    int i = 0;
    for (auto arg : func.getArguments()) {
      auto type = arg.getType();

      if (shapeMap.count(i) > 0) {
        // llvm::errs() << "shape found at " << i << "";
        // Tensors are passed flattened, so we need to get the size as such
        int64_t totalSize = shapeMap[i].back();
        type = mlir::RankedTensorType::get(
            {totalSize}, cast<mlir::RankedTensorType>(type).getElementType());
      }
      newInputs.push_back(type);
      i++;
    }
    auto newType =
        mlir::FunctionType::get(func.getContext(), newInputs, newInputs);

    func.setType(newType);

    Block &entry = func.front();
    if (entry.getNumArguments() == newInputs.size()) {
      for (auto [arg, newTy] : llvm::zip(entry.getArguments(), newInputs)) {
        arg.setType(newTy);
      }
    }
  }

  void runOnOperation() override {
    shapeMap = parseArg(shapes);

    auto root = getOperation();
    llvm::errs() << "\n=============Initial Root==============\n";
    root->dump();

    root->walk([&](mlir::func::FuncOp func) { setFuncOperandTypes(func); });

    // Update function type
    auto context = getOperation()->getContext();
    ShapeInfoState state{shapeMap};
    RewritePatternSet patterns(context);
    patterns.add<RemoveDynamicReshapePattern>(context);
    patterns.add<RefineReshapePattern>(context, state);
    patterns.add<RefineSlicePattern>(context, state);
    patterns.add<RefineBroadcastPattern>(context, state);

    patterns.add<DotGeneralTypePropagationPattern>(context);
    patterns.add<TypePropagationPattern>(context);

    RewritePatternSet deletePatterns(context);
    deletePatterns.add<MergeTransposeSelectPattern>(context, state);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(deletePatterns),
                                     config))) {
      signalPassFailure();
    }
    llvm::errs() << "===========After deletion==========\n";
    getOperation()->dump();
    // config.setMaxIterations(max_iterations);
    // config.setUseTopDownTraversal(top_down);
    // config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
    llvm::errs() << "===========Final function==========\n";
    getOperation()->dump();
  }
};

} // end anonymous namespace