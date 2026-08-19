//===- LowerSparseCSR.cpp - Lower CSR sparse products to custom calls -----===//
//
// Rewrites `stablehlo.dot_general` ops whose lhs is a CSR-encoded
// `sparse_tensor.assemble` result into a
// `stablehlo.custom_call @reactant_csr_matmul` on the raw CSR buffers, so
// that no sparse-encoded tensor types survive. The custom call is
// implemented via cuSPARSE ("CUDA") / hipSPARSE ("ROCM").
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "lower-sparse-csr"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERSPARSECSRPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct CSRDotGeneralToCustomCall
    : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto assemble = op.getLhs().getDefiningOp<sparse_tensor::AssembleOp>();
    if (!assemble)
      return failure();

    auto spTy = cast<RankedTensorType>(op.getLhs().getType());
    auto enc = sparse_tensor::getSparseTensorEncoding(spTy);
    if (!enc || enc.getLvlRank() != 2 || !enc.isIdentity() ||
        !enc.isDenseLvl(0) || !enc.isCompressedLvl(1))
      return rewriter.notifyMatchFailure(op, "lhs is not CSR-encoded");

    if (assemble.getLevels().size() != 2)
      return rewriter.notifyMatchFailure(
          op, "expected positions + coordinates level buffers");

    if (sparse_tensor::getSparseTensorEncoding(op.getRhs().getType()) ||
        sparse_tensor::getSparseTensorEncoding(op.getType()))
      return rewriter.notifyMatchFailure(op, "rhs and result must be dense");

    auto rhsTy = cast<RankedTensorType>(op.getRhs().getType());
    int64_t rhsRank = rhsTy.getRank();
    if (rhsRank != 1 && rhsRank != 2)
      return rewriter.notifyMatchFailure(op, "rhs must be a vector or matrix");
    if (!rhsTy.hasStaticShape() || !spTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes");

    auto dims = op.getDotDimensionNumbers();
    if (!dims.getLhsBatchingDimensions().empty() ||
        !dims.getRhsBatchingDimensions().empty() ||
        dims.getLhsContractingDimensions() != ArrayRef<int64_t>{1} ||
        dims.getRhsContractingDimensions() != ArrayRef<int64_t>{0})
      return rewriter.notifyMatchFailure(
          op, "only plain A * x / A * B contractions are supported");

    Value rowptr = assemble.getLevels()[0];
    Value colind = assemble.getLevels()[1];
    Value nzval = assemble.getValues();

    // `sparse_tensor` coordinates are 0-based.
    SmallVector<NamedAttribute> configAttrs = {
        rewriter.getNamedAttr("m", rewriter.getI64IntegerAttr(
                                       spTy.getDimSize(0))),
        rewriter.getNamedAttr("n", rewriter.getI64IntegerAttr(
                                       spTy.getDimSize(1))),
        rewriter.getNamedAttr("transpose", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("index_base", rewriter.getI64IntegerAttr(0)),
    };

    auto resTy = cast<RankedTensorType>(op.getType());
    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{resTy},
        ValueRange{rowptr, colind, nzval, op.getRhs()},
        rewriter.getStringAttr("reactant_csr_matmul"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ rewriter.getDictionaryAttr(configAttrs),
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*called_computations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, {1, 1, 1, rhsRank}, {true, true, true, true},
                      2),
        /*result_layouts*/
        getSHLOLayout(rewriter, {resTy.getRank()}, {true}, 2),
        /*output_operand_aliases*/ rewriter.getArrayAttr({}),
        /*result_tilings*/ nullptr);

    rewriter.replaceOp(op, customCall.getResults());
    return success();
  }
};

struct LowerSparseCSRPass
    : public enzyme::impl::LowerSparseCSRPassBase<LowerSparseCSRPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<CSRDotGeneralToCustomCall>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // The greedy driver DCEs `sparse_tensor.assemble` ops it visited; sweep
    // any remaining unused ones so the check below only reports genuinely
    // unsupported usage.
    SmallVector<Operation *> deadAssembles;
    getOperation()->walk([&](sparse_tensor::AssembleOp assemble) {
      if (assemble->use_empty())
        deadAssembles.push_back(assemble);
    });
    for (Operation *op : deadAssembles)
      op->erase();

    // XLA cannot consume sparse-encoded tensor types, so any surviving
    // sparse_tensor op is an error.
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa_and_nonnull<sparse_tensor::SparseTensorDialect>(
              op->getDialect())) {
        op->emitError()
            << "unsupported use of the sparse_tensor dialect: only CSR "
               "`A * x` / `A * B` products consumed by `stablehlo.dot_general` "
               "can be lowered";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
