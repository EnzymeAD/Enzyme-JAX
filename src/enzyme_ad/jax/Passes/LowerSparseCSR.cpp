//===- LowerSparseCSR.cpp - Lower CSR sparse products to custom calls -----===//
//
// Lowers CSR sparse matrix products to library custom calls in two steps:
//
//  1. `stablehlo.dot_general` ops whose lhs is a CSR-encoded
//     `sparse_tensor.assemble` result are raised to the semantic
//     `enzymexla.sparse.spmm` op (with alpha = 1, beta = 0).
//  2. `enzymexla.sparse.spmm` ops are lowered to
//     `stablehlo.custom_call @reactant_csr_matmul` (beta == 0) or
//     `stablehlo.custom_call @reactant_csr_matmul_acc` (fused
//     alpha*A*B + beta*C, with C aliased to the output) on the raw CSR
//     buffers, so that no sparse-encoded tensor types survive. When alpha or
//     beta are not compile-time constants the scaling/accumulation is emitted
//     as explicit stablehlo ops around the plain product instead.
//
// The custom calls are implemented via cuSPARSE ("CUDA") / hipSPARSE ("ROCM").
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
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

// Returns the defining `sparse_tensor.assemble` of `A` if `A` is a
// statically-shaped CSR-encoded matrix assembled from (positions,
// coordinates), values buffers.
static sparse_tensor::AssembleOp getCSRAssemble(Value A) {
  auto assemble = A.getDefiningOp<sparse_tensor::AssembleOp>();
  if (!assemble)
    return nullptr;

  auto spTy = cast<RankedTensorType>(A.getType());
  auto enc = sparse_tensor::getSparseTensorEncoding(spTy);
  if (!enc || enc.getLvlRank() != 2 || !enc.isIdentity() ||
      !enc.isDenseLvl(0) || !enc.isCompressedLvl(1))
    return nullptr;
  if (assemble.getLevels().size() != 2)
    return nullptr;
  if (!spTy.hasStaticShape())
    return nullptr;

  return assemble;
}

static std::optional<double> extractConstantScalar(Value val) {
  DenseElementsAttr attr;
  if (!matchPattern(val, m_Constant(&attr)))
    return std::nullopt;
  if (!isa<FloatType>(cast<RankedTensorType>(val.getType()).getElementType()))
    return std::nullopt;
  return attr.getSplatValue<APFloat>().convertToDouble();
}

// Raise `dot_general(assemble(...), B)` to `enzymexla.sparse.spmm` with
// alpha = 1, beta = 0.
struct RaiseCSRDotGeneral : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    if (!getCSRAssemble(op.getLhs()))
      return failure();

    if (sparse_tensor::getSparseTensorEncoding(op.getRhs().getType()) ||
        sparse_tensor::getSparseTensorEncoding(op.getType()))
      return rewriter.notifyMatchFailure(op, "rhs and result must be dense");

    auto rhsTy = cast<RankedTensorType>(op.getRhs().getType());
    int64_t rhsRank = rhsTy.getRank();
    if (rhsRank != 1 && rhsRank != 2)
      return rewriter.notifyMatchFailure(op, "rhs must be a vector or matrix");
    if (!rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes");
    if (!rhsTy.getElementType().isF32() && !rhsTy.getElementType().isF64())
      return rewriter.notifyMatchFailure(op, "only f32/f64 are supported");

    auto spTy = cast<RankedTensorType>(op.getLhs().getType());
    auto resTy = cast<RankedTensorType>(op.getType());
    if (spTy.getElementType() != rhsTy.getElementType() ||
        resTy.getElementType() != rhsTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "mixed element types are not supported");

    auto dims = op.getDotDimensionNumbers();
    if (!dims.getLhsBatchingDimensions().empty() ||
        !dims.getRhsBatchingDimensions().empty() ||
        dims.getLhsContractingDimensions() != ArrayRef<int64_t>{1} ||
        dims.getRhsContractingDimensions() != ArrayRef<int64_t>{0})
      return rewriter.notifyMatchFailure(
          op, "only plain A * x / A * B contractions are supported");

    rewriter.replaceOpWithNewOp<enzymexla::SparseSpMMOp>(op, op.getLhs(),
                                                         op.getRhs());
    return success();
  }
};

// Lower `enzymexla.sparse.spmm` on a CSR-assembled matrix to custom calls.
struct LowerSparseSpMM : public OpRewritePattern<enzymexla::SparseSpMMOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::SparseSpMMOp op,
                                PatternRewriter &rewriter) const override {
    auto assemble = getCSRAssemble(op.getA());
    if (!assemble)
      return rewriter.notifyMatchFailure(
          op, "A is not a CSR-encoded sparse_tensor.assemble result");

    auto spTy = cast<RankedTensorType>(op.getA().getType());
    auto resTy = cast<RankedTensorType>(op.getType());
    auto rhsTy = cast<RankedTensorType>(op.getB().getType());
    int64_t rhsRank = rhsTy.getRank();
    if (!rhsTy.hasStaticShape() || !resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static shapes");
    if (!resTy.getElementType().isF32() && !resTy.getElementType().isF64())
      return rewriter.notifyMatchFailure(op, "only f32/f64 are supported");

    Value rowptr = assemble.getLevels()[0];
    Value colind = assemble.getLevels()[1];
    Value nzval = assemble.getValues();

    auto alphaConst = extractConstantScalar(op.getAlpha());
    auto betaConst = extractConstantScalar(op.getBeta());
    bool betaIsZero = betaConst && *betaConst == 0.0;

    // `sparse_tensor` coordinates are 0-based.
    SmallVector<NamedAttribute> configAttrs = {
        rewriter.getNamedAttr("m",
                              rewriter.getI64IntegerAttr(spTy.getDimSize(0))),
        rewriter.getNamedAttr("n",
                              rewriter.getI64IntegerAttr(spTy.getDimSize(1))),
        rewriter.getNamedAttr("transpose", rewriter.getI64IntegerAttr(0)),
        rewriter.getNamedAttr("index_base", rewriter.getI64IntegerAttr(0)),
    };

    auto apiVersion = stablehlo::CustomCallApiVersionAttr::get(
        rewriter.getContext(),
        stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI);

    if (alphaConst && betaConst && !betaIsZero) {
      // Fully fused alpha*A*B + beta*C in a single library call, with C
      // aliased to the output buffer.
      configAttrs.push_back(rewriter.getNamedAttr(
          "alpha", rewriter.getF64FloatAttr(*alphaConst)));
      configAttrs.push_back(
          rewriter.getNamedAttr("beta", rewriter.getF64FloatAttr(*betaConst)));

      auto customCall = stablehlo::CustomCallOp::create(
          rewriter, op.getLoc(), TypeRange{resTy},
          ValueRange{rowptr, colind, nzval, op.getB(), op.getC()},
          rewriter.getStringAttr("reactant_csr_matmul_acc"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ rewriter.getDictionaryAttr(configAttrs),
          /*api_version*/ apiVersion,
          /*called_computations*/ nullptr,
          /*operand_layouts*/
          getSHLOLayout(rewriter, {1, 1, 1, rhsRank, resTy.getRank()},
                        {true, true, true, true, true}, 2),
          /*result_layouts*/
          getSHLOLayout(rewriter, {resTy.getRank()}, SmallVector<bool>{true},
                        2),
          /*output_operand_aliases*/
          rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
              op.getContext(), {}, 4, {})}),
          /*result_tilings*/ nullptr);

      rewriter.replaceOp(op, customCall.getResults());
      return success();
    }

    // Plain product, with a constant alpha folded into the call. Runtime
    // alpha/beta scaling and accumulation are emitted as explicit stablehlo
    // ops around it.
    configAttrs.push_back(rewriter.getNamedAttr(
        "alpha", rewriter.getF64FloatAttr(alphaConst.value_or(1.0))));

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{resTy},
        ValueRange{rowptr, colind, nzval, op.getB()},
        rewriter.getStringAttr("reactant_csr_matmul"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ rewriter.getDictionaryAttr(configAttrs),
        /*api_version*/ apiVersion,
        /*called_computations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, {1, 1, 1, rhsRank}, {true, true, true, true},
                      2),
        /*result_layouts*/
        getSHLOLayout(rewriter, {resTy.getRank()}, SmallVector<bool>{true}, 2),
        /*output_operand_aliases*/ rewriter.getArrayAttr({}),
        /*result_tilings*/ nullptr);

    Value out = customCall.getResult(0);
    auto broadcastScalar = [&](Value scalar) {
      return stablehlo::BroadcastInDimOp::create(
          rewriter, op.getLoc(), resTy, scalar,
          rewriter.getDenseI64ArrayAttr({}));
    };
    if (!alphaConst) {
      out = stablehlo::MulOp::create(rewriter, op.getLoc(), out,
                                     broadcastScalar(op.getAlpha()));
    }
    if (!betaIsZero) {
      Value scaledC = stablehlo::MulOp::create(rewriter, op.getLoc(), op.getC(),
                                               broadcastScalar(op.getBeta()));
      out = stablehlo::AddOp::create(rewriter, op.getLoc(), out, scaledC);
    }

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct LowerSparseCSRPass
    : public enzyme::impl::LowerSparseCSRPassBase<LowerSparseCSRPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RaiseCSRDotGeneral, LowerSparseSpMM>(context);

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
    // sparse_tensor (or unlowerable enzymexla.sparse) op is an error.
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa_and_nonnull<sparse_tensor::SparseTensorDialect>(
              op->getDialect()) ||
          isa<enzymexla::SparseSpMMOp>(op)) {
        op->emitError()
            << "unsupported use of sparse tensors: only CSR "
               "`alpha * A * B + beta * C` products (`enzymexla.sparse.spmm` "
               "or `stablehlo.dot_general` on a `sparse_tensor.assemble` "
               "result) can be lowered";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
