
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

struct ShapeInfo {
  int64_t ldim = -1;
  int64_t totalSize = -1;
  int64_t transposed = -1;
  llvm::SmallVector<int64_t, 2> shape;
};

llvm::SmallDenseMap<int, ShapeInfo> decodeShapeInfoStruct(llvm::StringRef input) {
  llvm::SmallDenseMap<int, ShapeInfo> map;

  while (!input.empty()) {
    auto [entry, rest] = input.split(';');
    input = rest;

    if (entry.empty())
      continue;

    // Split key and payload
    auto [keyStr, payload] = entry.split(':');

    int key;
    if (keyStr.getAsInteger(10, key))
      continue; // or handle error

    // Split payload fields: a|b|flag|vec
    llvm::SmallVector<llvm::StringRef, 4> parts;
    payload.split(parts, '|');

    if (parts.size() < 4)
      continue; // malformed

    ShapeInfo s;

    parts[0].getAsInteger(10, s.ldim);
    parts[1].getAsInteger(10, s.totalSize);
    parts[2].getAsInteger(10, s.transposed);

    // Parse vector: v0,v1,v2
    llvm::StringRef vecStr = parts[3];
    while (!vecStr.empty()) {
      auto [elem, restVec] = vecStr.split(',');
      vecStr = restVec;

      if (elem.empty())
        continue;

      int v;
      if (!elem.getAsInteger(10, v))
        s.shape.push_back(v);
    }

    map.try_emplace(key, std::move(s));
  }

  return map;
}

struct ShapeInfoState {
  llvm::SmallDenseMap<int, ShapeInfo> shapeMap;
};

SmallVector<int64_t> getShapeFromOp(Operation *op,
                                    ShapeInfoState &state) {
  // Get source index
  auto srcIdxAttr = op->getAttrOfType<IntegerAttr>("sourceArgIdx");
  if (!srcIdxAttr)
    return {};

  int srcIdx = srcIdxAttr.getInt();
  const auto &info = state.shapeMap[srcIdx];

  // Helper: decode a dimension string
  auto decodeDim = [&](StringAttr attr) -> int64_t {
    if (!attr)
      return -1;

    StringRef v = attr.getValue();

    if (v == "ldim") return info.ldim;
    if (v == "row") return info.shape[0];
    if (v == "col") return info.shape[1];
    if (v == "ldim.col") return info.ldim * info.shape[1];
    if (v == "ldim.row") return info.ldim * info.shape[0];

    return -1;
  };

  int64_t dim0 = decodeDim(op->getAttrOfType<StringAttr>("dim.0"));
  int64_t dim1 = decodeDim(op->getAttrOfType<StringAttr>("dim.1"));

  if (dim0 == -1)
    return {};

  if (dim1 == -1)
    return {dim0};

  return {dim0, dim1};
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
        if (state.shapeMap[srcIdxAttr.getInt()].transposed) {
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
      if (isTransposed.getValue() != state.shapeMap[srcIdx].transposed) {
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

  llvm::SmallDenseMap<int, ShapeInfo> shapeMap;

  void setFuncOperandTypes(mlir::func::FuncOp func) {
    // auto oldType = func.getFunctionType();
    llvm::SmallVector<mlir::Type> newInputs;
    int i = 0;
    for (auto arg : func.getArguments()) {
      auto type = arg.getType();

      if (shapeMap.count(i) > 0) {
        // llvm::errs() << "shape found at " << i << "";
        // Tensors are passed flattened, so we need to get the size as such
        int64_t totalSize = shapeMap[i].totalSize;
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
    shapeMap = decodeShapeInfoStruct(shapes);

    auto root = getOperation();
    // llvm::errs() << "\n=============Initial Root==============\n";
    // root->dump();
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
    // llvm::errs() << "===========After deletion==========\n";
    // getOperation()->dump();
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