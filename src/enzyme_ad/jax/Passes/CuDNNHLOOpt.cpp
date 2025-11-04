#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "enzymexla-cudnn-hlo-opt"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CUDNNHLOOPT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

constexpr llvm::StringLiteral kCuDNNFusionFuncPrefix =
    "__cudnn_fused_elementwise_dot_";

// TODO: We can generalize this to capture convert ops as well
template <typename ElementwiseOpTy>
struct DotGeneralElementwiseToCuDNNFusion
    : public OpRewritePattern<ElementwiseOpTy> {
  using OpRewritePattern<ElementwiseOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseOpTy elemOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = elemOp.getOperand(0);
    auto rhs = elemOp.getOperand(1);

    func::FuncOp parentFunc = elemOp->template getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return rewriter.notifyMatchFailure(elemOp, "No parent func found");

    if (parentFunc.getSymName().starts_with(kCuDNNFusionFuncPrefix))
      return failure();

    auto lhsDotGeneral = lhs.template getDefiningOp<stablehlo::DotGeneralOp>();
    auto rhsDotGeneral = rhs.template getDefiningOp<stablehlo::DotGeneralOp>();

    if (!lhsDotGeneral && !rhsDotGeneral)
      return failure();

    if (lhsDotGeneral && rhsDotGeneral)
      return failure();

    stablehlo::DotGeneralOp dotGeneral;
    bool dotGeneralIsLhs;
    if (lhsDotGeneral) {
      dotGeneral = lhsDotGeneral;
      dotGeneralIsLhs = true;
    } else {
      dotGeneral = rhsDotGeneral;
      dotGeneralIsLhs = false;
    }

    // TODO: does cudnn allow returning intermediate results?
    if (!dotGeneral->hasOneUse())
      return failure();

    ModuleOp mod = elemOp->template getParentOfType<ModuleOp>();
    if (!mod)
      return rewriter.notifyMatchFailure(elemOp, "No module found");

    static int fusionCounter = 0;
    std::string fnName =
        (kCuDNNFusionFuncPrefix + std::to_string(fusionCounter)).str();
    auto fnSym = rewriter.getStringAttr(fnName);
    fusionCounter++;

    // Input Types
    auto dotGeneralLhsTy = dotGeneral.getLhs().getType();
    auto dotGeneralRhsTy = dotGeneral.getRhs().getType();
    Value elemOther;
    Type elemOtherTy;
    if (dotGeneralIsLhs) {
      elemOther = rhs;
      elemOtherTy = cast<RankedTensorType>(rhs.getType());
    } else {
      elemOther = lhs;
      elemOtherTy = cast<RankedTensorType>(lhs.getType());
    }

    auto resultTy = elemOp.getType();
    auto elemLoc = elemOp.getLoc();

    // Replace with a custom call
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        elemOp, TypeRange{resultTy},
        ValueRange{dotGeneral.getLhs(), dotGeneral.getRhs(), elemOther},
        rewriter.getStringAttr("__cudnn$fusion"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/
        ArrayAttr::get(rewriter.getContext(), {FlatSymbolRefAttr::get(fnSym)}),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr);

    auto funcTy = rewriter.getFunctionType(
        {dotGeneralLhsTy, dotGeneralRhsTy, elemOtherTy}, {resultTy});

    // Construct the fusion function
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());

      auto funcOp = func::FuncOp::create(rewriter, elemLoc, fnSym, funcTy);
      funcOp.setVisibility(SymbolTable::Visibility::Private);
      funcOp.setNoInlineAttr(rewriter.getUnitAttr());
      auto &entryBlock = *funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);

      auto arg0 = entryBlock.getArgument(0);
      auto arg1 = entryBlock.getArgument(1);
      auto arg2 = entryBlock.getArgument(2);

      auto newDotGeneral = stablehlo::DotGeneralOp::create(
          rewriter, elemLoc, dotGeneral.getType(), arg0, arg1,
          dotGeneral.getDotDimensionNumbersAttr(),
          dotGeneral.getPrecisionConfigAttr(), dotGeneral.getAlgorithmAttr());
      Value newElementwise;
      if (dotGeneralIsLhs) {
        newElementwise =
            ElementwiseOpTy::create(rewriter, elemLoc, newDotGeneral, arg2);
      } else {
        newElementwise =
            ElementwiseOpTy::create(rewriter, elemLoc, arg2, newDotGeneral);
      }
      func::ReturnOp::create(rewriter, elemLoc, newElementwise);
    }

    return success();
  }
};

struct CuDNNHLOOptPass : public enzyme::impl::CuDNNHLOOptBase<CuDNNHLOOptPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<DotGeneralElementwiseToCuDNNFusion<stablehlo::AddOp>,
                 DotGeneralElementwiseToCuDNNFusion<stablehlo::MulOp>,
                 DotGeneralElementwiseToCuDNNFusion<stablehlo::SubtractOp>>(
        context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
