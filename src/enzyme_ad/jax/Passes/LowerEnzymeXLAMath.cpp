// This must come first for windows builds
#define _USE_MATH_DEFINES

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "lower-enzymexla-math"

#include <functional>
#include <string>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMATHPASS
#define GEN_PASS_DEF_LOWERENZYMEXLAMLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

template <typename T>
static stablehlo::ConstantOp
createConstantOpFromScalar(PatternRewriter &rewriter, Location loc, Type type,
                           T value) {
  return stablehlo::ConstantOp::create(
      rewriter, loc, type,
      cast<ElementsAttr>(mlir::enzyme::makeAttr(type, value)));
}

namespace {
#include "src/enzyme_ad/jax/Passes/LowerEnzymeXLAMathPatterns.cpp.inc"

// Lower a tensor-typed enzyme.binomial_progress(n, s) to the Revolve advance
// distance. Enzyme's own lower-enzyme-binomial-progress handles the scalar form
// on scf/arith and skips tensors, leaving them to this pattern; the two mirror
// each other op for op and must agree with BinomialProgressConstProp in
// EnzymeHLOOpt.cpp:
//
//   %r = if (n <= 1) or (s <= 1) {
//     // n <= 1 yields n (0 or 1); otherwise s <= 1 advances the whole
//     // remainder, which is also n. Both degenerate cases are just n.
//     return n
//   } else {
//     // smallest t with beta = C(s+t, t) >= n
//     %w:2 = while (%t = 0, %beta = 1) cond(%beta < n) {
//       %t2 = %t + 1; %beta2 = %beta * (s + %t2) / %t2
//     }
//     // window [n - beta(s-1,t), beta(s,t-1)], clamped, then the midpoint
//     %lo = max(n - %beta * s / (s + %t), 1)
//     %hi = min(%beta * %t / (s + %t), n - 1)
//     return max(min((%lo + %hi) / 2, n - (s - 1)), 1)
//   }
//
// The guard must be a branch, not a select: for s <= 1 the update leaves %beta
// at 1 and the loop would spin forever.
struct LowerBinomialProgressOpToStableHLO
    : public OpRewritePattern<enzyme::BinomialProgressOp> {
  using OpRewritePattern<enzyme::BinomialProgressOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::BinomialProgressOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value n = op.getNumSteps();
    Value s = op.getBudget();

    // Scalar operands are Enzyme's to lower, onto scf/arith.
    if (!isa<TensorType>(op.getType()))
      return failure();

    auto constOfType = [&](int64_t v) -> Value {
      return stablehlo::ConstantOp::create(
          rewriter, loc, cast<ElementsAttr>(makeAttr(op.getType(), v)));
    };

    Value zero = constOfType(0);
    Value one = constOfType(1);

    // Guard both degenerate cases.
    Value nSmall = stablehlo::CompareOp::create(
        rewriter, loc, n, one, stablehlo::ComparisonDirection::LE);
    Value sSmall = stablehlo::CompareOp::create(
        rewriter, loc, s, one, stablehlo::ComparisonDirection::LE);
    Value cond = stablehlo::OrOp::create(rewriter, loc, nSmall, sSmall);

    auto ifOp =
        stablehlo::IfOp::create(rewriter, loc, TypeRange{op.getType()}, cond);

    { // true block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ifOp.getTrueBranch(), {}, {}, {});
      stablehlo::ReturnOp::create(rewriter, loc, n);
    }

    { // false block
      OpBuilder::InsertionGuard guard2(rewriter);
      rewriter.createBlock(&ifOp.getFalseBranch(), {}, {}, {});

      // Smallest t with beta = C(s + t, t) >= n, carrying (t, beta).
      auto whileOp = stablehlo::WhileOp::create(rewriter, loc, {zero, one});
      SmallVector<Location> locs{zero.getLoc(), one.getLoc()};
      { // continue while beta < n
        OpBuilder::InsertionGuard guard(rewriter);
        Block *cond = rewriter.createBlock(&whileOp.getCond(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(cond);

        Value cmp =
            stablehlo::CompareOp::create(rewriter, loc, cond->getArgument(1), n,
                                         stablehlo::ComparisonDirection::LT);
        stablehlo::ReturnOp::create(rewriter, loc, cmp);
      }

      // t' = t + 1; beta' = beta * (s + t') / t'. That steps C(s+t-1, t-1) to
      // C(s+t, t), so the division is exact.
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Block *body = rewriter.createBlock(&whileOp.getBody(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(body);

        Value t2 =
            stablehlo::AddOp::create(rewriter, loc, body->getArgument(0), one);
        Value beta2 = stablehlo::DivOp::create(
            rewriter, loc,
            stablehlo::MulOp::create(
                rewriter, loc, body->getArgument(1),
                stablehlo::AddOp::create(rewriter, loc, s, t2)),
            t2);

        stablehlo::ReturnOp::create(rewriter, loc, ValueRange{t2, beta2});
      }

      Value t = whileOp->getResult(0), beta = whileOp->getResult(1);

      // beta(s-1, t) = beta * s / (s + t) and beta(s, t-1) = beta * t / (s +
      // t), both exact in integers. Every advance between n - beta(s-1,t) and
      // beta(s, t-1) attains the optimal repetition count; clamp that window to
      // [1, n-1] -- so the caller always progresses and always leaves a tail --
      // and take its midpoint, since either edge can collapse onto the clamp
      // and waste a checkpoint on a one-step advance.
      Value sPlusT = stablehlo::AddOp::create(rewriter, loc, s, t);
      Value loRaw = stablehlo::SubtractOp::create(
          rewriter, loc, n,
          stablehlo::DivOp::create(
              rewriter, loc, stablehlo::MulOp::create(rewriter, loc, beta, s),
              sPlusT));
      Value hiRaw = stablehlo::DivOp::create(
          rewriter, loc, stablehlo::MulOp::create(rewriter, loc, beta, t),
          sPlusT);

      Value lo = stablehlo::MaxOp::create(rewriter, loc, loRaw, one);
      Value hi = stablehlo::MinOp::create(
          rewriter, loc, hiRaw,
          stablehlo::SubtractOp::create(rewriter, loc, n, one));
      Value two = constOfType(2);
      Value sum = stablehlo::AddOp::create(rewriter, loc, lo, hi);
      Value mid = stablehlo::DivOp::create(rewriter, loc, sum, two);

      // Leave a step for each of the s-1 checkpoints still to be placed.
      // Without this the advances can exhaust the interval before the slots run
      // out, and a caller walking one slot per iteration then records slots at
      // a step past the end, holding the final state rather than a checkpoint.
      Value cap = stablehlo::SubtractOp::create(
          rewriter, loc, n,
          stablehlo::SubtractOp::create(rewriter, loc, s, one));
      Value result = stablehlo::MaxOp::create(
          rewriter, loc, stablehlo::MinOp::create(rewriter, loc, mid, cap),
          one);

      stablehlo::ReturnOp::create(rewriter, loc, result);
    }

    rewriter.replaceOp(op, ifOp.getResult(0));

    return success();
  }
};

static Value createScalarConst(PatternRewriter &rewriter, Location loc,
                               Type elTy, double val) {
  auto scalarTy = RankedTensorType::get({}, elTy);
  return stablehlo::ConstantOp::create(
      rewriter, loc, scalarTy, cast<ElementsAttr>(makeAttr(scalarTy, val)));
}

static Value createSplatConst(PatternRewriter &rewriter, Location loc,
                              RankedTensorType ty, double val) {
  auto scalarTy = RankedTensorType::get({}, ty.getElementType());
  auto attr = SplatElementsAttr::get(
      ty,
      cast<ElementsAttr>(makeAttr(scalarTy, val)).getValues<Attribute>()[0]);
  return stablehlo::ConstantOp::create(rewriter, loc, ty, attr);
}

static Value createNegInfConst(PatternRewriter &rewriter, Location loc,
                               RankedTensorType ty) {
  auto fltTy = cast<FloatType>(ty.getElementType());
  APFloat negInf =
      APFloat::getInf(fltTy.getFloatSemantics(), /*Negative=*/true);
  auto attr = SplatElementsAttr::get(ty, rewriter.getFloatAttr(fltTy, negInf));
  return stablehlo::ConstantOp::create(rewriter, loc, ty, attr);
}

static Value convertElementType(PatternRewriter &rewriter, Location loc,
                                Value v, Type targetElTy) {
  auto ty = cast<RankedTensorType>(v.getType());
  if (ty.getElementType() == targetElTy)
    return v;
  return stablehlo::ConvertOp::create(
      rewriter, loc, RankedTensorType::get(ty.getShape(), targetElTy), v);
}

static Value expandKVHeads(PatternRewriter &rewriter, Location loc, Value kv,
                           int64_t N_q) {
  auto ty = cast<RankedTensorType>(kv.getType());
  int64_t B = ty.getDimSize(0);
  int64_t S = ty.getDimSize(1);
  int64_t N_kv = ty.getDimSize(2);
  int64_t D = ty.getDimSize(3);
  if (N_q == N_kv)
    return kv;
  int64_t G = N_q / N_kv;
  auto ty5D = RankedTensorType::get({B, S, N_kv, G, D}, ty.getElementType());
  Value bcast = stablehlo::BroadcastInDimOp::create(
      rewriter, loc, ty5D, kv, rewriter.getDenseI64ArrayAttr({0, 1, 2, 4}));
  return stablehlo::ReshapeOp::create(
      rewriter, loc, RankedTensorType::get({B, S, N_q, D}, ty.getElementType()),
      bcast);
}

static Value reduceSumAlongDims(PatternRewriter &rewriter, Location loc,
                                Value input, ArrayRef<int64_t> dims) {
  if (dims.empty())
    return input;
  auto inTy = cast<RankedTensorType>(input.getType());
  Type elTy = inTy.getElementType();
  SmallVector<int64_t> sortedDims(dims.begin(), dims.end());
  llvm::sort(sortedDims);
  SmallVector<int64_t> outShape;
  for (int64_t d = 0; d < inTy.getRank(); ++d) {
    if (!llvm::is_contained(sortedDims, d))
      outShape.push_back(inTy.getDimSize(d));
  }
  auto outTy = RankedTensorType::get(outShape, elTy);
  Value initVal = createScalarConst(rewriter, loc, elTy, 0.0);
  auto reduceOp = stablehlo::ReduceOp::create(
      rewriter, loc, TypeRange{outTy}, ValueRange{input}, ValueRange{initVal},
      rewriter.getDenseI64ArrayAttr(sortedDims));
  Block *block = new Block();
  reduceOp.getBody().push_back(block);
  auto argTy = RankedTensorType::get({}, elTy);
  block->addArgument(argTy, loc);
  block->addArgument(argTy, loc);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    Value sum = stablehlo::AddOp::create(rewriter, loc, block->getArgument(0),
                                         block->getArgument(1));
    stablehlo::ReturnOp::create(rewriter, loc, ValueRange{sum});
  }
  return reduceOp.getResult(0);
}

static Value buildScoreValidityMask(PatternRewriter &rewriter, Location loc,
                                    int64_t B, int64_t N_q, int64_t T,
                                    int64_t S, bool isCausal, Value mask,
                                    Value kvLengths) {
  auto bntsI64Ty = RankedTensorType::get({B, N_q, T, S}, rewriter.getI64Type());
  auto bntsI1Ty = RankedTensorType::get({B, N_q, T, S}, rewriter.getI1Type());
  Value validMask;
  auto combineMask = [&](Value m) {
    validMask =
        validMask
            ? stablehlo::AndOp::create(rewriter, loc, validMask, m).getResult()
            : m;
  };

  if (isCausal) {
    Value tIdx = stablehlo::IotaOp::create(rewriter, loc, bntsI64Ty,
                                           rewriter.getI64IntegerAttr(2));
    Value sIdx = stablehlo::IotaOp::create(rewriter, loc, bntsI64Ty,
                                           rewriter.getI64IntegerAttr(3));
    Value causal = stablehlo::CompareOp::create(
        rewriter, loc, tIdx, sIdx, stablehlo::ComparisonDirection::GE);
    combineMask(causal);
  }

  if (mask) {
    Value m = mask;
    if (m.getType() != bntsI1Ty) {
      m = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, bntsI1Ty, m,
          rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
    }
    combineMask(m);
  }

  if (kvLengths) {
    Value kvLenI64 =
        convertElementType(rewriter, loc, kvLengths, rewriter.getI64Type());
    Value kvLenBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsI64Ty, kvLenI64, rewriter.getDenseI64ArrayAttr({0}));
    Value sIdx = stablehlo::IotaOp::create(rewriter, loc, bntsI64Ty,
                                           rewriter.getI64IntegerAttr(3));
    Value kvMask = stablehlo::CompareOp::create(
        rewriter, loc, sIdx, kvLenBcast, stablehlo::ComparisonDirection::LT);
    combineMask(kvMask);
  }

  return validMask;
}

static bool canUseCuDNNFMHA(StringRef backend, RankedTensorType qTy,
                            RankedTensorType vTy, RankedTensorType lseTy,
                            Value bias, Value mask) {
  if (backend != "cuda")
    return false;
  Type elTy = qTy.getElementType();
  if (!elTy.isF16() && !elTy.isBF16())
    return false;
  if (!lseTy.getElementType().isF32())
    return false;

  int64_t B = qTy.getDimSize(0);
  int64_t T = qTy.getDimSize(1);
  int64_t N_q = qTy.getDimSize(2);
  int64_t D_qk = qTy.getDimSize(3);
  int64_t S = vTy.getDimSize(1);
  int64_t D_v = vTy.getDimSize(3);

  if (D_qk != D_v || D_qk <= 0 || (D_qk % 8 != 0) || D_qk > 128)
    return false;
  if (mask)
    return false;
  if (bias) {
    auto biasTy = cast<RankedTensorType>(bias.getType());
    auto bShape = biasTy.getShape();
    if ((bShape[0] != 1 && bShape[0] != B) ||
        (bShape[1] != 1 && bShape[1] != N_q) || bShape[2] != T ||
        bShape[3] != S)
      return false;
    if ((T % 2 != 0) || (S % 2 != 0))
      return false;
  }
  return true;
}

static std::string buildCuDNNFMHABackendConfig(int64_t B, int64_t N_q,
                                               int64_t T, int64_t S, Type elTy,
                                               double scale, bool isCausal,
                                               bool hasPadding, bool isBwd) {
  StringRef dtypeStr = elTy.isF16() ? "F16" : "BF16";
  StringRef maskStr = isCausal ? (hasPadding ? "PADDING_CAUSAL" : "CAUSAL")
                               : (hasPadding ? "PADDING" : "NO_MASK");
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "{\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],"
     << "\"cudnn_fmha_backend_config\":{"
     << "\"algorithm\":{\"algo_id\":\"0\",\"math_type\":\"TENSOR_OP_MATH\","
     << "\"tuning_knobs\":{\"17\":\"1\",\"24\":\"0\"},"
     << "\"is_cudnn_frontend\":true,\"workspace_size\":\"0\"},"
     << "\"fmha_scale\":" << scale << ","
     << "\"dropout_rate\":0.0,"
     << "\"intermediate_tensor_shape\":{\"element_type\":\"" << dtypeStr
     << "\",\"dimensions\":[\"" << B << "\",\"" << N_q << "\",\"" << T
     << "\",\"" << S << "\"],\"tuple_shapes\":[],\"layout\":{"
     << "\"dim_level_types\":[],\"dim_unique\":[],\"dim_ordered\":[],"
     << "\"minor_to_major\":[\"3\",\"2\",\"1\",\"0\"],\"tiles\":[],"
     << "\"element_size_in_bits\":\"0\",\"memory_space\":\"0\","
     << "\"index_primitive_type\":\"PRIMITIVE_TYPE_INVALID\","
     << "\"pointer_primitive_type\":\"PRIMITIVE_TYPE_INVALID\","
     << "\"dynamic_shape_metadata_prefix_bytes\":\"0\"},"
     << "\"is_dynamic_dimension\":[false,false,false,false]},"
     << "\"is_flash_attention\":true,"
     << "\"mask_type\":\"" << maskStr << "\","
     << "\"seed\":\"42\","
     << "\"sliding_window_length\":0,"
     << "\"max_seg_per_batch\":1,"
     << "\"is_paged_attention\":false,";
  if (!isBwd) {
    os << "\"bmm1_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"3\"],"
       << "\"rhs_contracting_dimensions\":[\"3\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"2\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]},"
       << "\"bmm2_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"3\"],"
       << "\"rhs_contracting_dimensions\":[\"1\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"1\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]}";
  } else {
    os << "\"bmm1_grad_gemm1_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"2\"],"
       << "\"rhs_contracting_dimensions\":[\"1\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"1\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]},"
       << "\"bmm1_grad_gemm2_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"3\"],"
       << "\"rhs_contracting_dimensions\":[\"1\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"1\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]},"
       << "\"bmm2_grad_gemm1_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"2\"],"
       << "\"rhs_contracting_dimensions\":[\"1\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"1\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]},"
       << "\"bmm2_grad_gemm2_dot_dimension_numbers\":{"
       << "\"lhs_contracting_dimensions\":[\"3\"],"
       << "\"rhs_contracting_dimensions\":[\"3\"],"
       << "\"lhs_batch_dimensions\":[\"0\",\"2\"],"
       << "\"rhs_batch_dimensions\":[\"0\",\"2\"]}";
  }
  os << "}}";
  return os.str();
}

static Value getOrCreateSeqLenI32(PatternRewriter &rewriter, Location loc,
                                  Value seqLen, int64_t B, int64_t defaultLen) {
  auto targetTy = RankedTensorType::get({B}, rewriter.getI32Type());
  if (!seqLen) {
    auto attr = SplatElementsAttr::get(
        targetTy, rewriter.getI32IntegerAttr(static_cast<int32_t>(defaultLen)));
    return stablehlo::ConstantOp::create(rewriter, loc, targetTy, attr);
  }
  Value v = convertElementType(rewriter, loc, seqLen, rewriter.getI32Type());
  if (v.getType() != targetTy) {
    v = stablehlo::BroadcastInDimOp::create(rewriter, loc, targetTy, v,
                                            rewriter.getDenseI64ArrayAttr({0}));
  }
  return v;
}

static Attribute getDefaultLayoutAttr(PatternRewriter &rewriter, int64_t rank) {
  SmallVector<int64_t> layout;
  for (int64_t i = rank - 1; i >= 0; --i)
    layout.push_back(i);
  return rewriter.getIndexTensorAttr(layout);
}

struct AttentionOpLowering : public OpRewritePattern<enzymexla::AttentionOp> {
  std::string backend;

  AttentionOpLowering(std::string backend, MLIRContext *context)
      : OpRewritePattern(context), backend(std::move(backend)) {}

  LogicalResult matchAndRewrite(enzymexla::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value q = op.getQuery();
    Value k = op.getKey();
    Value v = op.getValue();

    auto qTy = cast<RankedTensorType>(q.getType());
    auto kTy = cast<RankedTensorType>(k.getType());
    auto vTy = cast<RankedTensorType>(v.getType());
    auto lseTy = cast<RankedTensorType>(op.getLse().getType());

    int64_t B = qTy.getDimSize(0);
    int64_t T = qTy.getDimSize(1);
    int64_t N_q = qTy.getDimSize(2);
    int64_t S = kTy.getDimSize(1);
    int64_t D_v = vTy.getDimSize(3);
    Type elTy = qTy.getElementType();
    double scaleVal = op.getScale().convertToDouble();

    if (canUseCuDNNFMHA(backend, qTy, vTy, lseTy, op.getBias(), op.getMask())) {
      bool hasBias = (bool)op.getBias();
      bool hasPadding = (bool)op.getQLengths() || (bool)op.getKvLengths();
      bool needLse = !op.getLse().use_empty();

      SmallVector<Value> operands = {q, k, v};
      if (hasBias)
        operands.push_back(op.getBias());
      if (hasPadding) {
        operands.push_back(
            getOrCreateSeqLenI32(rewriter, loc, op.getQLengths(), B, T));
        operands.push_back(
            getOrCreateSeqLenI32(rewriter, loc, op.getKvLengths(), B, S));
      }

      SmallVector<Attribute> operandLayouts;
      for (Value operand : operands) {
        auto rTy = cast<RankedTensorType>(operand.getType());
        operandLayouts.push_back(getDefaultLayoutAttr(rewriter, rTy.getRank()));
      }

      auto outBnthTy = RankedTensorType::get({B, N_q, T, D_v}, elTy);
      auto scratchTy =
          RankedTensorType::get({0}, IntegerType::get(rewriter.getContext(), 8,
                                                      IntegerType::Unsigned));

      SmallVector<Type> resultTypes = {outBnthTy};
      SmallVector<Attribute> resultLayouts = {
          rewriter.getIndexTensorAttr({3, 1, 2, 0})};
      if (needLse) {
        resultTypes.push_back(lseTy);
        resultLayouts.push_back(getDefaultLayoutAttr(rewriter, 3));
      }
      resultTypes.push_back(scratchTy);
      resultLayouts.push_back(getDefaultLayoutAttr(rewriter, 1));

      StringRef callTarget =
          hasBias ? "__cudnn$fmhaScaleBiasSoftmax" : "__cudnn$fmhaSoftmax";
      std::string configStr = buildCuDNNFMHABackendConfig(
          B, N_q, T, S, elTy, scaleVal, op.getIsCausal(), hasPadding,
          /*isBwd=*/false);

      auto callOp = stablehlo::CustomCallOp::create(
          rewriter, loc, resultTypes, operands,
          rewriter.getStringAttr(callTarget),
          /*has_side_effect=*/rewriter.getBoolAttr(false),
          /*backend_config=*/rewriter.getStringAttr(configStr),
          /*api_version=*/
          stablehlo::CustomCallApiVersionAttr::get(
              rewriter.getContext(),
              stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
          /*called_computations=*/nullptr,
          /*operand_layouts=*/rewriter.getArrayAttr(operandLayouts),
          /*result_layouts=*/rewriter.getArrayAttr(resultLayouts),
          /*output_operand_aliases=*/nullptr,
          /*result_tilings=*/nullptr);

      Value outBtnh = stablehlo::TransposeOp::create(
          rewriter, loc, callOp.getResult(0),
          rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));
      Value lseVal = needLse ? callOp.getResult(1)
                             : createSplatConst(rewriter, loc, lseTy, 0.0);
      rewriter.replaceOp(op, ValueRange{outBtnh, lseVal});
      return success();
    }

    Type computeTy =
        (elTy.isF16() || elTy.isBF16()) ? (Type)rewriter.getF32Type() : elTy;

    Value qC = convertElementType(rewriter, loc, q, computeTy);
    Value kC = expandKVHeads(
        rewriter, loc, convertElementType(rewriter, loc, k, computeTy), N_q);
    Value vC = expandKVHeads(
        rewriter, loc, convertElementType(rewriter, loc, v, computeTy), N_q);

    auto bntsTy = RankedTensorType::get({B, N_q, T, S}, computeTy);
    auto bntTy = RankedTensorType::get({B, N_q, T}, computeTy);

    auto qkDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 2}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{3});
    Value scores = stablehlo::DotGeneralOp::create(
        rewriter, loc, bntsTy, qC, kC, qkDims, nullptr, nullptr);

    if (scaleVal != 1.0) {
      Value scaleConst = createSplatConst(rewriter, loc, bntsTy, scaleVal);
      scores = stablehlo::MulOp::create(rewriter, loc, scores, scaleConst);
    }

    if (Value bias = op.getBias()) {
      Value biasC = convertElementType(rewriter, loc, bias, computeTy);
      if (biasC.getType() != bntsTy) {
        biasC = stablehlo::BroadcastInDimOp::create(
            rewriter, loc, bntsTy, biasC,
            rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
      }
      scores = stablehlo::AddOp::create(rewriter, loc, scores, biasC);
    }

    Value validMask =
        buildScoreValidityMask(rewriter, loc, B, N_q, T, S, op.getIsCausal(),
                               op.getMask(), op.getKvLengths());
    if (validMask) {
      Value negInf = createNegInfConst(rewriter, loc, bntsTy);
      scores =
          stablehlo::SelectOp::create(rewriter, loc, validMask, scores, negInf);
    }

    // Row-wise max along dim 3 (S)
    Value negInfScalar =
        createNegInfConst(rewriter, loc, RankedTensorType::get({}, computeTy));
    auto maxReduceOp = stablehlo::ReduceOp::create(
        rewriter, loc, TypeRange{bntTy}, ValueRange{scores},
        ValueRange{negInfScalar}, rewriter.getDenseI64ArrayAttr({3}));
    {
      Block *block = new Block();
      maxReduceOp.getBody().push_back(block);
      auto argTy = RankedTensorType::get({}, computeTy);
      block->addArgument(argTy, loc);
      block->addArgument(argTy, loc);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      Value m = stablehlo::MaxOp::create(rewriter, loc, block->getArgument(0),
                                         block->getArgument(1));
      stablehlo::ReturnOp::create(rewriter, loc, ValueRange{m});
    }
    Value maxScores = maxReduceOp.getResult(0);

    Value isMaxFinite = stablehlo::IsFiniteOp::create(rewriter, loc, maxScores);
    Value zeroBnt = createSplatConst(rewriter, loc, bntTy, 0.0);
    Value safeMax = stablehlo::SelectOp::create(rewriter, loc, isMaxFinite,
                                                maxScores, zeroBnt);
    Value safeMaxBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsTy, safeMax,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));
    Value expScores = stablehlo::ExpOp::create(
        rewriter, loc,
        stablehlo::SubtractOp::create(rewriter, loc, scores, safeMaxBcast));

    Value sumExp = reduceSumAlongDims(rewriter, loc, expScores, {3});
    Value lseVal = stablehlo::AddOp::create(
        rewriter, loc, safeMax,
        stablehlo::LogOp::create(rewriter, loc, sumExp));

    if (Value qLengths = op.getQLengths()) {
      auto bntI64Ty = RankedTensorType::get({B, N_q, T}, rewriter.getI64Type());
      Value qLenI64 =
          convertElementType(rewriter, loc, qLengths, rewriter.getI64Type());
      Value qLenBcast = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, bntI64Ty, qLenI64, rewriter.getDenseI64ArrayAttr({0}));
      Value tIdx = stablehlo::IotaOp::create(rewriter, loc, bntI64Ty,
                                             rewriter.getI64IntegerAttr(2));
      Value qValidBnt = stablehlo::CompareOp::create(
          rewriter, loc, tIdx, qLenBcast, stablehlo::ComparisonDirection::LT);
      Value negInfBnt = createNegInfConst(rewriter, loc, bntTy);
      lseVal = stablehlo::SelectOp::create(rewriter, loc, qValidBnt, lseVal,
                                           negInfBnt);
    }

    Value oneBnt = createSplatConst(rewriter, loc, bntTy, 1.0);
    Value safeSumExp =
        stablehlo::SelectOp::create(rewriter, loc, isMaxFinite, sumExp, oneBnt);
    Value sumExpBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsTy, safeSumExp,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));
    Value probs =
        stablehlo::DivOp::create(rewriter, loc, expScores, sumExpBcast);

    auto pvDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{1});
    auto outBntdTy = RankedTensorType::get({B, N_q, T, D_v}, computeTy);
    Value outBntd = stablehlo::DotGeneralOp::create(
        rewriter, loc, outBntdTy, probs, vC, pvDims, nullptr, nullptr);
    Value outBtnd = stablehlo::TransposeOp::create(
        rewriter, loc, outBntd, rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));

    if (Value qLengths = op.getQLengths()) {
      auto btndI64Ty =
          RankedTensorType::get({B, T, N_q, D_v}, rewriter.getI64Type());
      Value qLenI64 =
          convertElementType(rewriter, loc, qLengths, rewriter.getI64Type());
      Value qLenBcast = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, btndI64Ty, qLenI64,
          rewriter.getDenseI64ArrayAttr({0}));
      Value tIdx = stablehlo::IotaOp::create(rewriter, loc, btndI64Ty,
                                             rewriter.getI64IntegerAttr(1));
      Value qValidBtnd = stablehlo::CompareOp::create(
          rewriter, loc, tIdx, qLenBcast, stablehlo::ComparisonDirection::LT);
      Value zeroBtnd = createSplatConst(
          rewriter, loc, RankedTensorType::get({B, T, N_q, D_v}, computeTy),
          0.0);
      outBtnd = stablehlo::SelectOp::create(rewriter, loc, qValidBtnd, outBtnd,
                                            zeroBtnd);
    }

    Value finalOut = convertElementType(rewriter, loc, outBtnd, elTy);
    Value finalLse =
        convertElementType(rewriter, loc, lseVal, lseTy.getElementType());
    rewriter.replaceOp(op, ValueRange{finalOut, finalLse});
    return success();
  }
};

struct AttentionGradOpLowering
    : public OpRewritePattern<enzymexla::AttentionGradOp> {
  std::string backend;

  AttentionGradOpLowering(std::string backend, MLIRContext *context)
      : OpRewritePattern(context), backend(std::move(backend)) {}

  LogicalResult matchAndRewrite(enzymexla::AttentionGradOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value q = op.getQuery();
    Value k = op.getKey();
    Value v = op.getValue();
    Value output = op.getOutput();
    Value lse = op.getLse();
    Value dOutput = op.getDOutput();

    auto qTy = cast<RankedTensorType>(q.getType());
    auto kTy = cast<RankedTensorType>(k.getType());
    auto vTy = cast<RankedTensorType>(v.getType());
    auto lseTy = cast<RankedTensorType>(lse.getType());

    int64_t B = qTy.getDimSize(0);
    int64_t T = qTy.getDimSize(1);
    int64_t N_q = qTy.getDimSize(2);
    int64_t D_qk = qTy.getDimSize(3);
    int64_t S = kTy.getDimSize(1);
    int64_t N_kv = kTy.getDimSize(2);
    int64_t D_v = vTy.getDimSize(3);
    int64_t G = N_q / N_kv;
    Type elTy = qTy.getElementType();
    double scaleVal = op.getScale().convertToDouble();

    if (canUseCuDNNFMHA(backend, qTy, vTy, lseTy, op.getBias(), op.getMask())) {
      bool hasBias = (bool)op.getBias();
      bool hasDBias = (bool)op.getDBias();
      bool hasPadding = (bool)op.getQLengths() || (bool)op.getKvLengths();

      // {Q, K, V, activation, dO, bias*, O, q_seqlen*, kv_seqlen*}
      SmallVector<Value> operands = {q, k, v, lse, dOutput};
      if (hasBias)
        operands.push_back(op.getBias());
      operands.push_back(output);
      if (hasPadding) {
        operands.push_back(
            getOrCreateSeqLenI32(rewriter, loc, op.getQLengths(), B, T));
        operands.push_back(
            getOrCreateSeqLenI32(rewriter, loc, op.getKvLengths(), B, S));
      }

      SmallVector<Attribute> operandLayouts;
      for (Value operand : operands) {
        auto rTy = cast<RankedTensorType>(operand.getType());
        operandLayouts.push_back(getDefaultLayoutAttr(rewriter, rTy.getRank()));
      }

      auto dqBnthTy = RankedTensorType::get({B, N_q, T, D_qk}, elTy);
      auto dkBnthTy = RankedTensorType::get({B, N_kv, S, D_qk}, elTy);
      auto dvBnthTy = RankedTensorType::get({B, N_kv, S, D_v}, elTy);
      auto scratchTy =
          RankedTensorType::get({0}, IntegerType::get(rewriter.getContext(), 8,
                                                      IntegerType::Unsigned));

      Attribute gradLayout = rewriter.getIndexTensorAttr({3, 1, 2, 0});
      SmallVector<Type> resultTypes = {dqBnthTy, dkBnthTy, dvBnthTy};
      SmallVector<Attribute> resultLayouts = {gradLayout, gradLayout,
                                              gradLayout};
      if (hasDBias) {
        resultTypes.push_back(op.getDBias().getType());
        resultLayouts.push_back(getDefaultLayoutAttr(rewriter, 4));
      }
      resultTypes.push_back(scratchTy);
      resultLayouts.push_back(getDefaultLayoutAttr(rewriter, 1));

      StringRef callTarget = hasBias ? "__cudnn$fmhaScaleBiasSoftmaxBackward"
                                     : "__cudnn$fmhaSoftmaxBackward";
      std::string configStr = buildCuDNNFMHABackendConfig(
          B, N_q, T, S, elTy, scaleVal, op.getIsCausal(), hasPadding,
          /*isBwd=*/true);

      auto callOp = stablehlo::CustomCallOp::create(
          rewriter, loc, resultTypes, operands,
          rewriter.getStringAttr(callTarget),
          /*has_side_effect=*/rewriter.getBoolAttr(false),
          /*backend_config=*/rewriter.getStringAttr(configStr),
          /*api_version=*/
          stablehlo::CustomCallApiVersionAttr::get(
              rewriter.getContext(),
              stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
          /*called_computations=*/nullptr,
          /*operand_layouts=*/rewriter.getArrayAttr(operandLayouts),
          /*result_layouts=*/rewriter.getArrayAttr(resultLayouts),
          /*output_operand_aliases=*/nullptr,
          /*result_tilings=*/nullptr);

      auto permAttr = rewriter.getDenseI64ArrayAttr({0, 2, 1, 3});
      Value dqBtnh = stablehlo::TransposeOp::create(
          rewriter, loc, callOp.getResult(0), permAttr);
      Value dkBtnh = stablehlo::TransposeOp::create(
          rewriter, loc, callOp.getResult(1), permAttr);
      Value dvBtnh = stablehlo::TransposeOp::create(
          rewriter, loc, callOp.getResult(2), permAttr);

      SmallVector<Value> replacements = {dqBtnh, dkBtnh, dvBtnh};
      if (hasDBias)
        replacements.push_back(callOp.getResult(3));
      rewriter.replaceOp(op, replacements);
      return success();
    }

    Type computeTy =
        (elTy.isF16() || elTy.isBF16()) ? (Type)rewriter.getF32Type() : elTy;

    Value qC = convertElementType(rewriter, loc, q, computeTy);
    Value kC = expandKVHeads(
        rewriter, loc, convertElementType(rewriter, loc, k, computeTy), N_q);
    Value vC = expandKVHeads(
        rewriter, loc, convertElementType(rewriter, loc, v, computeTy), N_q);
    Value outC = convertElementType(rewriter, loc, output, computeTy);
    Value dOutC = convertElementType(rewriter, loc, dOutput, computeTy);
    Value lseC = convertElementType(rewriter, loc, lse, computeTy);

    auto bntsTy = RankedTensorType::get({B, N_q, T, S}, computeTy);
    auto bntTy = RankedTensorType::get({B, N_q, T}, computeTy);

    auto qkDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 2}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{3});
    Value scores = stablehlo::DotGeneralOp::create(
        rewriter, loc, bntsTy, qC, kC, qkDims, nullptr, nullptr);

    if (scaleVal != 1.0) {
      Value scaleConst = createSplatConst(rewriter, loc, bntsTy, scaleVal);
      scores = stablehlo::MulOp::create(rewriter, loc, scores, scaleConst);
    }

    if (Value bias = op.getBias()) {
      Value biasC = convertElementType(rewriter, loc, bias, computeTy);
      if (biasC.getType() != bntsTy) {
        biasC = stablehlo::BroadcastInDimOp::create(
            rewriter, loc, bntsTy, biasC,
            rewriter.getDenseI64ArrayAttr({0, 1, 2, 3}));
      }
      scores = stablehlo::AddOp::create(rewriter, loc, scores, biasC);
    }

    Value validMask =
        buildScoreValidityMask(rewriter, loc, B, N_q, T, S, op.getIsCausal(),
                               op.getMask(), op.getKvLengths());

    Value qValidBnt = stablehlo::IsFiniteOp::create(rewriter, loc, lseC);
    if (Value qLengths = op.getQLengths()) {
      auto bntI64Ty = RankedTensorType::get({B, N_q, T}, rewriter.getI64Type());
      Value qLenI64 =
          convertElementType(rewriter, loc, qLengths, rewriter.getI64Type());
      Value qLenBcast = stablehlo::BroadcastInDimOp::create(
          rewriter, loc, bntI64Ty, qLenI64, rewriter.getDenseI64ArrayAttr({0}));
      Value tIdx = stablehlo::IotaOp::create(rewriter, loc, bntI64Ty,
                                             rewriter.getI64IntegerAttr(2));
      Value qMask = stablehlo::CompareOp::create(
          rewriter, loc, tIdx, qLenBcast, stablehlo::ComparisonDirection::LT);
      qValidBnt = stablehlo::AndOp::create(rewriter, loc, qValidBnt, qMask);
    }

    Value zeroBnt = createSplatConst(rewriter, loc, bntTy, 0.0);
    Value safeLse =
        stablehlo::SelectOp::create(rewriter, loc, qValidBnt, lseC, zeroBnt);
    Value safeLseBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsTy, safeLse,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));
    Value pRaw = stablehlo::ExpOp::create(
        rewriter, loc,
        stablehlo::SubtractOp::create(rewriter, loc, scores, safeLseBcast));

    auto bntsI1Ty = RankedTensorType::get({B, N_q, T, S}, rewriter.getI1Type());
    Value validAll = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsI1Ty, qValidBnt,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));
    if (validMask)
      validAll = stablehlo::AndOp::create(rewriter, loc, validAll, validMask);

    Value zeroBnts = createSplatConst(rewriter, loc, bntsTy, 0.0);
    Value probs =
        stablehlo::SelectOp::create(rewriter, loc, validAll, pRaw, zeroBnts);

    // Mask out output and dOutput on invalid/all-masked query rows to avoid NaN
    Value qValidBtn = stablehlo::TransposeOp::create(
        rewriter, loc, qValidBnt, rewriter.getDenseI64ArrayAttr({0, 2, 1}));
    auto btndI1Ty =
        RankedTensorType::get({B, T, N_q, D_v}, rewriter.getI1Type());
    Value qValidBtnd = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, btndI1Ty, qValidBtn,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));
    auto btndTy = RankedTensorType::get({B, T, N_q, D_v}, computeTy);
    Value zeroBtnd = createSplatConst(rewriter, loc, btndTy, 0.0);
    Value safeOut =
        stablehlo::SelectOp::create(rewriter, loc, qValidBtnd, outC, zeroBtnd);
    Value safeDOut =
        stablehlo::SelectOp::create(rewriter, loc, qValidBtnd, dOutC, zeroBtnd);

    // dV = P^T @ safeDOut
    auto dvDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{2}, /*rhsContract=*/{1});
    auto dvBnsdTy = RankedTensorType::get({B, N_q, S, D_v}, computeTy);
    Value dvBnsd = stablehlo::DotGeneralOp::create(
        rewriter, loc, dvBnsdTy, probs, safeDOut, dvDims, nullptr, nullptr);
    Value dvBsnd = stablehlo::TransposeOp::create(
        rewriter, loc, dvBnsd, rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));
    if (G > 1) {
      Value dv5D = stablehlo::ReshapeOp::create(
          rewriter, loc, RankedTensorType::get({B, S, N_kv, G, D_v}, computeTy),
          dvBsnd);
      dvBsnd = reduceSumAlongDims(rewriter, loc, dv5D, {3});
    }

    // dP = safeDOut @ V^T -> (B, N_q, T, S)
    auto dpDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 2}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{3});
    Value dP = stablehlo::DotGeneralOp::create(rewriter, loc, bntsTy, safeDOut,
                                               vC, dpDims, nullptr, nullptr);

    // delta = sum_d(safeDOut * safeOut) -> (B, T, N_q) -> transpose (B, N_q, T)
    Value dOutMulOut =
        stablehlo::MulOp::create(rewriter, loc, safeDOut, safeOut);
    Value deltaBtn = reduceSumAlongDims(rewriter, loc, dOutMulOut, {3});
    Value deltaBnt = stablehlo::TransposeOp::create(
        rewriter, loc, deltaBtn, rewriter.getDenseI64ArrayAttr({0, 2, 1}));
    Value deltaBcast = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, bntsTy, deltaBnt,
        rewriter.getDenseI64ArrayAttr({0, 1, 2}));

    // dS = P * (dP - delta)
    Value dS = stablehlo::MulOp::create(
        rewriter, loc, probs,
        stablehlo::SubtractOp::create(rewriter, loc, dP, deltaBcast));

    Value dBiasFinal;
    if (op.getDBias()) {
      auto biasTy = cast<RankedTensorType>(op.getDBias().getType());
      auto bShape = biasTy.getShape();
      SmallVector<int64_t> reduceDims;
      for (int64_t d = 0; d < 4; ++d) {
        if (bShape[d] == 1 && bntsTy.getDimSize(d) > 1)
          reduceDims.push_back(d);
      }
      Value dbC = reduceSumAlongDims(rewriter, loc, dS, reduceDims);
      auto targetBiasCTy = RankedTensorType::get(bShape, computeTy);
      if (dbC.getType() != targetBiasCTy)
        dbC = stablehlo::ReshapeOp::create(rewriter, loc, targetBiasCTy, dbC);
      dBiasFinal = convertElementType(rewriter, loc, dbC, elTy);
    }

    Value dSScaled = dS;
    if (scaleVal != 1.0) {
      Value scaleConst = createSplatConst(rewriter, loc, bntsTy, scaleVal);
      dSScaled = stablehlo::MulOp::create(rewriter, loc, dS, scaleConst);
    }

    // dQ = dSScaled @ K
    auto dqDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{3}, /*rhsContract=*/{1});
    auto dqBntdTy = RankedTensorType::get({B, N_q, T, D_qk}, computeTy);
    Value dqBntd = stablehlo::DotGeneralOp::create(
        rewriter, loc, dqBntdTy, dSScaled, kC, dqDims, nullptr, nullptr);
    Value dqBtnd = stablehlo::TransposeOp::create(
        rewriter, loc, dqBntd, rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));

    // dK = dSScaled^T @ Q
    auto dkDims = stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatch=*/{0, 1}, /*rhsBatch=*/{0, 2},
        /*lhsContract=*/{2}, /*rhsContract=*/{1});
    auto dkBnsdTy = RankedTensorType::get({B, N_q, S, D_qk}, computeTy);
    Value dkBnsd = stablehlo::DotGeneralOp::create(
        rewriter, loc, dkBnsdTy, dSScaled, qC, dkDims, nullptr, nullptr);
    Value dkBsnd = stablehlo::TransposeOp::create(
        rewriter, loc, dkBnsd, rewriter.getDenseI64ArrayAttr({0, 2, 1, 3}));
    if (G > 1) {
      Value dk5D = stablehlo::ReshapeOp::create(
          rewriter, loc,
          RankedTensorType::get({B, S, N_kv, G, D_qk}, computeTy), dkBsnd);
      dkBsnd = reduceSumAlongDims(rewriter, loc, dk5D, {3});
    }

    Value finalDQ = convertElementType(rewriter, loc, dqBtnd, elTy);
    Value finalDK = convertElementType(rewriter, loc, dkBsnd, elTy);
    Value finalDV = convertElementType(rewriter, loc, dvBsnd, elTy);

    SmallVector<Value> replacements = {finalDQ, finalDK, finalDV};
    if (op.getDBias())
      replacements.push_back(dBiasFinal);
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

void lowerEnzymeXLAMath(Operation *op, std::string backend,
                        std::function<void()> signalPassFailure) {
  auto context = op->getContext();
  RewritePatternSet patterns(context);

  populateWithGenerated(patterns);

  patterns.add<LowerBinomialProgressOpToStableHLO>(context);
  patterns.add<AttentionOpLowering, AttentionGradOpLowering>(backend, context);

  GreedyRewriteConfig config;
  config.enableFolding();
  if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
    signalPassFailure();
  }

  // Verify that all illegal ops have been lowered. A tensor-typed
  // enzyme.binomial_progress counts: nothing downstream of here lowers it,
  // since Enzyme's own pass only handles the scalar form.
  auto walkResult = op->walk([&](Operation *local_op) {
    if (auto bp = dyn_cast<enzyme::BinomialProgressOp>(local_op)) {
      if (!isa<TensorType>(bp.getType()))
        return WalkResult::advance();
      bp->emitError("Failed to lower enzyme.binomial_progress");
      return WalkResult::interrupt();
    }
    if ((local_op->getName().getStringRef().starts_with("enzymexla.math.") ||
         local_op->getName().getStringRef().starts_with("enzymexla.ml.")) &&
        (local_op->getNumResults() > 0 &&
         llvm::any_of(local_op->getResultTypes(),
                      [](Type ty) { return isa<TensorType>(ty); })) &&
        (local_op->getNumOperands() > 0 &&
         llvm::any_of(local_op->getOperandTypes(),
                      [](Type ty) { return isa<TensorType>(ty); }))) {
      local_op->emitError("Failed to lower enzymexla math operation");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    signalPassFailure();
  }
}
} // namespace

struct LowerEnzymeXLAMathPass
    : public enzyme::impl::LowerEnzymeXLAMathPassBase<LowerEnzymeXLAMathPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), backend,
                       [this]() { signalPassFailure(); });
  }
};

// TODO: delete this once Reactant uses `lower-enzymexla-math` instead of
// `lower-enzymexla-ml`
struct LowerEnzymeXLAMLPass
    : public enzyme::impl::LowerEnzymeXLAMLPassBase<LowerEnzymeXLAMLPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), backend,
                       [this]() { signalPassFailure(); });
  }
};
