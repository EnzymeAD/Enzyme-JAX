#include "src/enzyme_ad/jax/Passes/Julia/Passes.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

namespace mlir::enzymexla::julia {

#define GEN_PASS_DEF_LOWERSTABLEHLOTOJULIAPASS
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

namespace {

class StableHLOToJuliaLowerer {
public:
  StableHLOToJuliaLowerer(bool reverseDims, bool preferBlas)
      : reverseDims(reverseDims), preferBlas(preferBlas) {}

  LogicalResult lowerModule(ModuleOp module);

private:
  bool reverseDims;
  bool preferBlas;

  int64_t mapDim(int64_t dim, int64_t rank) const {
    return reverseDims ? (rank - 1 - dim) : dim;
  }

  SmallVector<int64_t> mapShape(ArrayRef<int64_t> shape) const {
    SmallVector<int64_t> out(shape.begin(), shape.end());
    if (reverseDims)
      std::reverse(out.begin(), out.end());
    return out;
  }

  Type convertType(Type ty) const {
    if (auto rtt = dyn_cast<RankedTensorType>(ty)) {
      if (rtt.getRank() == 0)
        return rtt.getElementType();
      return ArrayType::get(mapShape(rtt.getShape()), rtt.getElementType());
    }
    return ty;
  }

  FunctionType convertFunctionType(FunctionType fnTy) const {
    SmallVector<Type> ins, outs;
    for (Type t : fnTy.getInputs())
      ins.push_back(convertType(t));
    for (Type t : fnTy.getResults())
      outs.push_back(convertType(t));
    return FunctionType::get(fnTy.getContext(), ins, outs);
  }

  Value createIndexConstant(OpBuilder &b, Location loc, int64_t v) const {
    return ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(v));
  }

  Value castToIndex(OpBuilder &b, Location loc, Value v) const {
    if (v.getType().isIndex())
      return v;
    return CastOp::create(b, loc, b.getIndexType(), v);
  }

  Value emitScalarConstantAttr(OpBuilder &b, Location loc, Type elemTy,
                               Attribute attr) const {
    if (auto cplxTy = dyn_cast<ComplexType>(elemTy)) {
      Type partTy = cplxTy.getElementType();
      if (auto arr = dyn_cast<ArrayAttr>(attr); arr && arr.size() == 2) {
        Value re = emitScalarConstantAttr(b, loc, partTy, arr[0]);
        Value im = emitScalarConstantAttr(b, loc, partTy, arr[1]);
        return BinaryOp::create(b, loc, cplxTy, BinaryOpKind::complex_make, re,
                                im);
      }
      TypedAttr zeroPart = b.getZeroAttr(partTy);
      Value z = ConstantOp::create(b, loc, partTy, zeroPart);
      return BinaryOp::create(b, loc, cplxTy, BinaryOpKind::complex_make, z, z);
    }
    if (auto typed = dyn_cast<TypedAttr>(attr))
      return ConstantOp::create(b, loc, elemTy, typed);
    return ConstantOp::create(b, loc, elemTy, b.getZeroAttr(elemTy));
  }

  /// Build dynamic dimension operands matching dynamic entries in `arrTy`.
  SmallVector<Value>
  buildDynamicDimsFromOperand(OpBuilder &b, Location loc, ArrayType arrTy,
                              Value sameShapeArrayOperand) const {
    SmallVector<Value> dynDims;
    for (int64_t d = 0; d < arrTy.getRank(); ++d) {
      if (arrTy.isDynamicDim(d)) {
        dynDims.push_back(SizeOp::create(b, loc, sameShapeArrayOperand,
                                         b.getI64IntegerAttr(d)));
      }
    }
    return dynDims;
  }

  /// Wrap a scalar computation `(OpBuilder&, Location, ValueRange scalars) -> Value`
  /// into either a direct scalar op (when `resTensorTy.getRank() == 0`) or a
  /// `julia.kernel` over `1:N_d` (supporting implicit CHLO/scalar broadcasts).
  Value emitElementwise(
      OpBuilder &b, Location loc, RankedTensorType resTensorTy,
      ArrayRef<Value> mappedInputs,
      ArrayRef<RankedTensorType> origInputTensorTypes,
      llvm::function_ref<Value(OpBuilder &, Location, ArrayRef<Value>)>
          buildScalar) const;

  LogicalResult lowerRegion(Region &srcRegion, Region &dstRegion, OpBuilder &b,
                            IRMapping &mapping, bool isFuncTop = false,
                            bool isWhileCond = false);

  LogicalResult lowerOperation(Operation *op, OpBuilder &b, IRMapping &mapping);
  LogicalResult lowerConstant(Operation *op, ElementsAttr valAttr, OpBuilder &b,
                              IRMapping &mapping);
  LogicalResult lowerDotGeneral(stablehlo::DotGeneralOp op, OpBuilder &b,
                                IRMapping &mapping);
  LogicalResult lowerGemm(enzymexla::GemmOp op, OpBuilder &b,
                          IRMapping &mapping);
  LogicalResult lowerReduce(stablehlo::ReduceOp op, OpBuilder &b,
                            IRMapping &mapping);
  LogicalResult lowerReduceWindow(stablehlo::ReduceWindowOp op, OpBuilder &b,
                                  IRMapping &mapping);
  LogicalResult lowerGather(stablehlo::GatherOp op, OpBuilder &b,
                            IRMapping &mapping);
  LogicalResult lowerScatter(stablehlo::ScatterOp op, OpBuilder &b,
                             IRMapping &mapping);
  LogicalResult lowerSort(stablehlo::SortOp op, OpBuilder &b,
                          IRMapping &mapping);
  LogicalResult lowerConvolution(stablehlo::ConvolutionOp op, OpBuilder &b,
                                 IRMapping &mapping);
};

Value StableHLOToJuliaLowerer::emitElementwise(
    OpBuilder &b, Location loc, RankedTensorType resTensorTy,
    ArrayRef<Value> mappedInputs,
    ArrayRef<RankedTensorType> origInputTensorTypes,
    llvm::function_ref<Value(OpBuilder &, Location, ArrayRef<Value>)>
        buildScalar) const {
  (void)origInputTensorTypes;
  if (resTensorTy.getRank() == 0) {
    return buildScalar(b, loc, mappedInputs);
  }

  auto resArrTy = cast<ArrayType>(convertType(resTensorTy));
  int64_t rank = resArrTy.getRank();

  // Collect dynamic dimensions from any rank-matching array input.
  Value shapeDonor;
  for (Value v : mappedInputs) {
    if (auto st = dyn_cast<ShapedType>(v.getType());
        st && st.getRank() == rank) {
      shapeDonor = v;
      break;
    }
  }
  SmallVector<Value> dynDims;
  if (shapeDonor)
    dynDims = buildDynamicDimsFromOperand(b, loc, resArrTy, shapeDonor);

  auto kernel = KernelOp::create(b, loc, TypeRange{resArrTy}, dynDims);
  Block *body = &kernel.getBody().emplaceBlock();
  SmallVector<Value> ivs;
  for (int64_t d = 0; d < rank; ++d)
    ivs.push_back(body->addArgument(b.getIndexType(), loc));

  OpBuilder kb(body, body->end());
  SmallVector<Value> loadedArgs;
  loadedArgs.reserve(mappedInputs.size());
  Value oneIdx;

  for (size_t i = 0; i < mappedInputs.size(); ++i) {
    Value inVal = mappedInputs[i];
    auto inShaped = dyn_cast<ShapedType>(inVal.getType());
    if (!inShaped) {
      // Unboxed 0-D scalar operand!
      loadedArgs.push_back(inVal);
      continue;
    }
    int64_t inRank = inShaped.getRank();
    SmallVector<Value> loadIdx(inRank);
    // Right-aligned NumPy/CHLO broadcast in StableHLO corresponds to:
    // under `reverseDims = true`, leading-aligned dims `0 .. inRank-1`;
    // under `reverseDims = false`, trailing-aligned dims `rank - inRank .. rank-1`.
    for (int64_t d = 0; d < inRank; ++d) {
      int64_t outDim = reverseDims ? d : (rank - inRank + d);
      if (inShaped.getDimSize(d) == 1 && resArrTy.getDimSize(outDim) != 1) {
        if (!oneIdx)
          oneIdx = createIndexConstant(kb, loc, 1);
        loadIdx[d] = oneIdx;
      } else {
        loadIdx[d] = ivs[outDim];
      }
    }
    loadedArgs.push_back(
        LoadOp::create(kb, loc, inShaped.getElementType(), inVal, loadIdx));
  }

  Value resScalar = buildScalar(kb, loc, loadedArgs);
  YieldOp::create(kb, loc, ValueRange{resScalar});
  return kernel.getResult(0);
}

/// Permute row-major `DenseElementsAttr` elements into column-major order when
/// `reverseDims == false` and `rank >= 2` so `ArrayConstantOp` always holds
/// column-major elements matching `reshape(T[...], dims...)` in Julia.
static ElementsAttr normalizeDenseElementsToColumnMajor(DenseElementsAttr dense,
                                                        bool reverseDims) {
  auto st = dyn_cast<RankedTensorType>(dense.getType());
  if (!st || reverseDims || st.getRank() <= 1 || dense.isSplat() ||
      st.getNumElements() <= 1) {
    return dense;
  }

  int64_t rank = st.getRank();
  ArrayRef<int64_t> shape = st.getShape();
  int64_t total = st.getNumElements();

  // Compute row-major strides for `shape`.
  SmallVector<int64_t> rowStrides(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d)
    rowStrides[d] = rowStrides[d + 1] * shape[d + 1];

  auto colLinearToRowLinear = [&](int64_t colLin) -> int64_t {
    int64_t rem = colLin;
    int64_t rowLin = 0;
    for (int64_t d = 0; d < rank; ++d) {
      int64_t coord = rem % shape[d];
      rem /= shape[d];
      rowLin += coord * rowStrides[d];
    }
    return rowLin;
  };

  if (st.getElementType().isInteger(1)) {
    SmallVector<bool> vals(total);
    auto srcVals = dense.getValues<bool>();
    SmallVector<bool> srcVec(srcVals.begin(), srcVals.end());
    for (int64_t c = 0; c < total; ++c)
      vals[c] = srcVec[colLinearToRowLinear(c)];
    return DenseElementsAttr::get(st, ArrayRef<bool>(vals));
  }

  ArrayRef<char> raw = dense.getRawData();
  if (raw.size() % static_cast<size_t>(total) == 0) {
    size_t elemBytes = raw.size() / static_cast<size_t>(total);
    SmallVector<char> outRaw(raw.size());
    for (int64_t c = 0; c < total; ++c) {
      int64_t r = colLinearToRowLinear(c);
      std::copy_n(raw.data() + static_cast<size_t>(r) * elemBytes, elemBytes,
                  outRaw.data() + static_cast<size_t>(c) * elemBytes);
    }
    return DenseElementsAttr::getFromRawBuffer(st, outRaw);
  }
  return dense;
}

LogicalResult StableHLOToJuliaLowerer::lowerConstant(Operation *op,
                                                     ElementsAttr valAttr,
                                                     OpBuilder &b,
                                                     IRMapping &mapping) {
  Location loc = op->getLoc();
  auto rtt = cast<RankedTensorType>(op->getResult(0).getType());
  Type elemTy = rtt.getElementType();
  auto dense = dyn_cast<DenseElementsAttr>(valAttr);
  if (!dense)
    return op->emitError("unsupported non-dense constant in Julia lowering");

  if (rtt.getRank() == 0) {
    Attribute splat = dense.getValues<Attribute>()[0];
    Value c = emitScalarConstantAttr(b, loc, elemTy, splat);
    mapping.map(op->getResult(0), c);
    return success();
  }

  auto arrTy = cast<ArrayType>(convertType(rtt));
  if (dense.isSplat()) {
    Attribute splat = dense.getSplatValue<Attribute>();
    Value c = emitScalarConstantAttr(b, loc, elemTy, splat);
    Value filled = FillOp::create(b, loc, arrTy, c, ValueRange{});
    mapping.map(op->getResult(0), filled);
    return success();
  }

  ElementsAttr normAttr = normalizeDenseElementsToColumnMajor(dense, reverseDims);
  Value arrCst = ArrayConstantOp::create(b, loc, arrTy, normAttr);
  mapping.map(op->getResult(0), arrCst);
  return success();
}

static std::optional<UnaryOpKind> mapStableHLOUnary(Operation *op) {
  if (isa<stablehlo::NegOp>(op))
    return UnaryOpKind::neg;
  if (isa<stablehlo::NotOp>(op))
    return UnaryOpKind::not_op;
  if (isa<stablehlo::AbsOp>(op))
    return UnaryOpKind::abs;
  if (isa<stablehlo::ExpOp>(op))
    return UnaryOpKind::exp;
  if (isa<stablehlo::Expm1Op>(op))
    return UnaryOpKind::expm1;
  if (isa<stablehlo::Exp2Op>(op))
    return UnaryOpKind::exp2;
  if (isa<stablehlo::LogOp>(op))
    return UnaryOpKind::log;
  if (isa<stablehlo::Log1pOp>(op))
    return UnaryOpKind::log1p;
  if (isa<stablehlo::Log2Op>(op))
    return UnaryOpKind::log2;
  if (isa<stablehlo::SqrtOp>(op))
    return UnaryOpKind::sqrt;
  if (isa<stablehlo::RsqrtOp>(op))
    return UnaryOpKind::rsqrt;
  if (isa<stablehlo::CbrtOp>(op))
    return UnaryOpKind::cbrt;
  if (isa<stablehlo::SineOp>(op))
    return UnaryOpKind::sin;
  if (isa<stablehlo::CosineOp>(op))
    return UnaryOpKind::cos;
  if (isa<stablehlo::TanOp, chlo::TanOp>(op))
    return UnaryOpKind::tan;
  if (isa<chlo::AsinOp>(op))
    return UnaryOpKind::asin;
  if (isa<chlo::AcosOp>(op))
    return UnaryOpKind::acos;
  if (isa<chlo::AtanOp>(op))
    return UnaryOpKind::atan;
  if (isa<chlo::SinhOp>(op))
    return UnaryOpKind::sinh;
  if (isa<chlo::CoshOp>(op))
    return UnaryOpKind::cosh;
  if (isa<stablehlo::TanhOp>(op))
    return UnaryOpKind::tanh;
  if (isa<chlo::AsinhOp>(op))
    return UnaryOpKind::asinh;
  if (isa<chlo::AcoshOp>(op))
    return UnaryOpKind::acosh;
  if (isa<chlo::AtanhOp>(op))
    return UnaryOpKind::atanh;
  if (isa<stablehlo::FloorOp>(op))
    return UnaryOpKind::floor;
  if (isa<stablehlo::CeilOp>(op))
    return UnaryOpKind::ceil;
  if (isa<stablehlo::RoundNearestEvenOp>(op))
    return UnaryOpKind::round_nearest_even;
  if (isa<stablehlo::RoundOp>(op))
    return UnaryOpKind::round_nearest_afz;
  if (isa<stablehlo::SignOp>(op))
    return UnaryOpKind::sign;
  if (isa<stablehlo::IsFiniteOp>(op))
    return UnaryOpKind::is_finite;
  if (isa<stablehlo::LogisticOp>(op))
    return UnaryOpKind::logistic;
  if (isa<stablehlo::RealOp>(op))
    return UnaryOpKind::real;
  if (isa<stablehlo::ImagOp>(op))
    return UnaryOpKind::imag;
  if (isa<chlo::ConjOp>(op))
    return UnaryOpKind::conj;
  if (isa<stablehlo::ClzOp>(op))
    return UnaryOpKind::leading_zeros;
  if (isa<stablehlo::PopulationCountOp>(op))
    return UnaryOpKind::popcnt;
  return std::nullopt;
}

static std::optional<BinaryOpKind> mapStableHLOBinary(Operation *op) {
  if (isa<stablehlo::AddOp, chlo::BroadcastAddOp>(op))
    return BinaryOpKind::add;
  if (isa<stablehlo::SubtractOp, chlo::BroadcastSubOp>(op))
    return BinaryOpKind::sub;
  if (isa<stablehlo::MulOp, chlo::BroadcastMulOp>(op))
    return BinaryOpKind::mul;
  if (isa<stablehlo::DivOp, chlo::BroadcastDivOp>(op))
    return BinaryOpKind::div;
  if (isa<stablehlo::RemOp, chlo::BroadcastRemOp>(op))
    return BinaryOpKind::rem;
  if (isa<stablehlo::PowOp, chlo::BroadcastPowOp>(op))
    return BinaryOpKind::pow;
  if (isa<stablehlo::Atan2Op, chlo::BroadcastAtan2Op>(op))
    return BinaryOpKind::atan2;
  if (isa<stablehlo::MaxOp, chlo::BroadcastMaxOp>(op))
    return BinaryOpKind::max;
  if (isa<stablehlo::MinOp, chlo::BroadcastMinOp>(op))
    return BinaryOpKind::min;
  if (isa<stablehlo::AndOp, chlo::BroadcastAndOp>(op))
    return BinaryOpKind::and_op;
  if (isa<stablehlo::OrOp, chlo::BroadcastOrOp>(op))
    return BinaryOpKind::or_op;
  if (isa<stablehlo::XorOp, chlo::BroadcastXorOp>(op))
    return BinaryOpKind::xor_op;
  if (isa<stablehlo::ShiftLeftOp, chlo::BroadcastShiftLeftOp>(op))
    return BinaryOpKind::shl;
  if (isa<stablehlo::ShiftRightArithmeticOp,
          chlo::BroadcastShiftRightArithmeticOp>(op))
    return BinaryOpKind::shr_arith;
  if (isa<stablehlo::ShiftRightLogicalOp, chlo::BroadcastShiftRightLogicalOp>(
          op))
    return BinaryOpKind::shr_logical;
  if (isa<stablehlo::ComplexOp, chlo::BroadcastComplexOp>(op))
    return BinaryOpKind::complex_make;
  return std::nullopt;
}

static CmpPred mapComparisonDirection(stablehlo::ComparisonDirection dir) {
  switch (dir) {
  case stablehlo::ComparisonDirection::EQ:
    return CmpPred::eq;
  case stablehlo::ComparisonDirection::NE:
    return CmpPred::ne;
  case stablehlo::ComparisonDirection::LT:
    return CmpPred::lt;
  case stablehlo::ComparisonDirection::LE:
    return CmpPred::le;
  case stablehlo::ComparisonDirection::GT:
    return CmpPred::gt;
  case stablehlo::ComparisonDirection::GE:
    return CmpPred::ge;
  }
  return CmpPred::eq;
}

static CmpPred mapChloComparisonDirection(chlo::ComparisonDirection dir) {
  switch (dir) {
  case chlo::ComparisonDirection::EQ:
    return CmpPred::eq;
  case chlo::ComparisonDirection::NE:
    return CmpPred::ne;
  case chlo::ComparisonDirection::LT:
    return CmpPred::lt;
  case chlo::ComparisonDirection::LE:
    return CmpPred::le;
  case chlo::ComparisonDirection::GT:
    return CmpPred::gt;
  case chlo::ComparisonDirection::GE:
    return CmpPred::ge;
  }
  return CmpPred::eq;
}

LogicalResult StableHLOToJuliaLowerer::lowerDotGeneral(
    stablehlo::DotGeneralOp op, OpBuilder &b, IRMapping &mapping) {
  Location loc = op.getLoc();
  Value jLhs = mapping.lookup(op.getLhs());
  Value jRhs = mapping.lookup(op.getRhs());
  auto lhsRtt = cast<RankedTensorType>(op.getLhs().getType());
  auto rhsRtt = cast<RankedTensorType>(op.getRhs().getType());
  auto resRtt = cast<RankedTensorType>(op.getResult().getType());
  Type resTy = convertType(resRtt);

  int64_t lhsRank = lhsRtt.getRank();
  int64_t rhsRank = rhsRtt.getRank();
  int64_t outRank = resRtt.getRank();

  auto dnums = op.getDotDimensionNumbers();
  ArrayRef<int64_t> sLhsBatch = dnums.getLhsBatchingDimensions();
  ArrayRef<int64_t> sRhsBatch = dnums.getRhsBatchingDimensions();
  ArrayRef<int64_t> sLhsContract = dnums.getLhsContractingDimensions();
  ArrayRef<int64_t> sRhsContract = dnums.getRhsContractingDimensions();
  size_t numBatch = sLhsBatch.size();

  SmallVector<int64_t> jLhsBatch, jRhsBatch, jLhsContract, jRhsContract;
  for (int64_t d : sLhsBatch)
    jLhsBatch.push_back(mapDim(d, lhsRank));
  for (int64_t d : sRhsBatch)
    jRhsBatch.push_back(mapDim(d, rhsRank));
  for (int64_t d : sLhsContract)
    jLhsContract.push_back(mapDim(d, lhsRank));
  for (int64_t d : sRhsContract)
    jRhsContract.push_back(mapDim(d, rhsRank));

  llvm::DenseSet<int64_t> sLhsBatchSet(sLhsBatch.begin(), sLhsBatch.end());
  llvm::DenseSet<int64_t> sLhsConSet(sLhsContract.begin(), sLhsContract.end());
  llvm::DenseSet<int64_t> sRhsBatchSet(sRhsBatch.begin(), sRhsBatch.end());
  llvm::DenseSet<int64_t> sRhsConSet(sRhsContract.begin(), sRhsContract.end());

  SmallVector<int64_t> sLhsFree, sRhsFree;
  for (int64_t d = 0; d < lhsRank; ++d)
    if (!sLhsBatchSet.contains(d) && !sLhsConSet.contains(d))
      sLhsFree.push_back(d);
  for (int64_t d = 0; d < rhsRank; ++d)
    if (!sRhsBatchSet.contains(d) && !sRhsConSet.contains(d))
      sRhsFree.push_back(d);

  llvm::DenseSet<int64_t> jLhsBatchSet(jLhsBatch.begin(), jLhsBatch.end());
  llvm::DenseSet<int64_t> jLhsConSet(jLhsContract.begin(), jLhsContract.end());
  llvm::DenseSet<int64_t> jRhsBatchSet(jRhsBatch.begin(), jRhsBatch.end());
  llvm::DenseSet<int64_t> jRhsConSet(jRhsContract.begin(), jRhsContract.end());

  SmallVector<int64_t> jLhsFree, jRhsFree;
  for (int64_t d = 0; d < lhsRank; ++d)
    if (!jLhsBatchSet.contains(d) && !jLhsConSet.contains(d))
      jLhsFree.push_back(d);
  for (int64_t d = 0; d < rhsRank; ++d)
    if (!jRhsBatchSet.contains(d) && !jRhsConSet.contains(d))
      jRhsFree.push_back(d);

  // Try lowering to `julia.matmul` when `preferBlas` is enabled and there is
  // exactly 1 contracting dimension and <= 1 free dimension per operand.
  if (preferBlas && jLhsContract.size() == 1) {
    int64_t kL = jLhsContract[0];
    int64_t kR = jRhsContract[0];

    // 1D inner product: `(K,) . (K,) -> T`
    if (numBatch == 0 && lhsRank == 1 && rhsRank == 1 && outRank == 0) {
      Value res = MatmulOp::create(
          b, loc, resTy, jLhs, jRhs, /*acc=*/Value(), /*alpha=*/Value(),
          /*beta=*/Value(), TransposeMode::none, TransposeMode::none,
          ArrayRef<int64_t>{}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{});
      mapping.map(op.getResult(), res);
      return success();
    }

    // 2D or batched 2D matrix-matrix product: `(..., M, K) * (..., K, N)`
    if (jLhsFree.size() == 1 && jRhsFree.size() == 1) {
      int64_t fL = jLhsFree[0];
      int64_t fR = jRhsFree[0];
      int64_t jOutLhsFree = mapDim(static_cast<int64_t>(numBatch), outRank);
      int64_t jOutRhsFree =
          mapDim(static_cast<int64_t>(numBatch) + 1, outRank);
      SmallVector<int64_t> jOutBatch;
      for (size_t i = 0; i < numBatch; ++i)
        jOutBatch.push_back(mapDim(static_cast<int64_t>(i), outRank));

      if (jOutLhsFree < jOutRhsFree) {
        TransposeMode tA =
            (fL < kL) ? TransposeMode::none : TransposeMode::transpose;
        TransposeMode tB =
            (kR < fR) ? TransposeMode::none : TransposeMode::transpose;
        Value res = MatmulOp::create(b, loc, resTy, jLhs, jRhs, Value(),
                                     Value(), Value(), tA, tB, jLhsBatch,
                                     jRhsBatch, jOutBatch);
        mapping.map(op.getResult(), res);
        return success();
      } else {
        TransposeMode tA =
            (fR < kR) ? TransposeMode::none : TransposeMode::transpose;
        TransposeMode tB =
            (kL < fL) ? TransposeMode::none : TransposeMode::transpose;
        Value res = MatmulOp::create(b, loc, resTy, jRhs, jLhs, Value(),
                                     Value(), Value(), tA, tB, jRhsBatch,
                                     jLhsBatch, jOutBatch);
        mapping.map(op.getResult(), res);
        return success();
      }
    }

    // Matrix-vector `(M, K) * (K,) -> (M,)`
    if (numBatch == 0 && jLhsFree.size() == 1 && jRhsFree.empty() &&
        lhsRank == 2 && rhsRank == 1 && outRank == 1) {
      int64_t fL = jLhsFree[0];
      TransposeMode tA =
          (fL < kL) ? TransposeMode::none : TransposeMode::transpose;
      Value res = MatmulOp::create(
          b, loc, resTy, jLhs, jRhs, Value(), Value(), Value(), tA,
          TransposeMode::none, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
          ArrayRef<int64_t>{});
      mapping.map(op.getResult(), res);
      return success();
    }

    // Vector-matrix `(K,) * (K, N) -> (N,)` -> normalized to `op(B) * x`!
    if (numBatch == 0 && jLhsFree.empty() && jRhsFree.size() == 1 &&
        lhsRank == 1 && rhsRank == 2 && outRank == 1) {
      int64_t fR = jRhsFree[0];
      TransposeMode tA =
          (fR < kR) ? TransposeMode::none : TransposeMode::transpose;
      Value res = MatmulOp::create(
          b, loc, resTy, jRhs, jLhs, Value(), Value(), Value(), tA,
          TransposeMode::none, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
          ArrayRef<int64_t>{});
      mapping.map(op.getResult(), res);
      return success();
    }
  }

  // Scalar-output multi-dim contraction (e.g. `contract = [[0, 1], [0, 1]]`):
  // permute `jRhs` if needed so contracting dims align with `0 .. rank-1`,
  // multiply elementwise via `KernelOp`, and sum-reduce to scalar `resTy`!
  if (outRank == 0) {
    if (lhsRank == 0 && rhsRank == 0) {
      Value prod = BinaryOp::create(b, loc, resTy, BinaryOpKind::mul, jLhs,
                                    jRhs);
      mapping.map(op.getResult(), prod);
      return success();
    }
    // Build `perm` on `jRhs` so `perm[jLhsContract[c]] = jRhsContract[c]`.
    SmallVector<int64_t> perm(rhsRank, 0);
    for (size_t c = 0; c < jLhsContract.size(); ++c)
      perm[jLhsContract[c]] = jRhsContract[c];
    bool isId = true;
    for (int64_t d = 0; d < rhsRank; ++d)
      if (perm[d] != d)
        isId = false;
    Value alignedRhs = jRhs;
    if (!isId) {
      alignedRhs =
          TransposeOp::create(b, loc, jLhs.getType(), jRhs, perm).getResult();
    }
    auto lhsArrTy = cast<ArrayType>(jLhs.getType());
    SmallVector<Value> dynDims =
        buildDynamicDimsFromOperand(b, loc, lhsArrTy, jLhs);
    auto ker = KernelOp::create(b, loc, TypeRange{lhsArrTy}, dynDims);
    Block *kbBlk = &ker.getBody().emplaceBlock();
    SmallVector<Value> ivs;
    for (int64_t d = 0; d < lhsRank; ++d)
      ivs.push_back(kbBlk->addArgument(b.getIndexType(), loc));
    OpBuilder kb(kbBlk, kbBlk->end());
    Value lVal = LoadOp::create(kb, loc, resTy, jLhs, ivs);
    Value rVal = LoadOp::create(kb, loc, resTy, alignedRhs, ivs);
    Value prod = BinaryOp::create(kb, loc, resTy, BinaryOpKind::mul, lVal, rVal);
    YieldOp::create(kb, loc, ValueRange{prod});

    Value zeroInit = emitScalarConstantAttr(b, loc, resTy, b.getZeroAttr(resTy));
    SmallVector<int64_t> allDims;
    for (int64_t d = 0; d < lhsRank; ++d)
      allDims.push_back(d);
    auto red = ReduceOp::create(b, loc, TypeRange{resTy},
                                ValueRange{ker.getResult(0)},
                                ValueRange{zeroInit}, allDims);
    Block *rbBlk = &red.getCombiner().emplaceBlock();
    Value accArg = rbBlk->addArgument(resTy, loc);
    Value elemArg = rbBlk->addArgument(resTy, loc);
    OpBuilder rb(rbBlk, rbBlk->end());
    Value sum =
        BinaryOp::create(rb, loc, resTy, BinaryOpKind::add, accArg, elemArg);
    YieldOp::create(rb, loc, ValueRange{sum});
    mapping.map(op.getResult(), red.getResult(0));
    return success();
  }

  // General rank >= 1 tensor contraction: compute `resultDimOrder`.
  SmallVector<int64_t> resultDimOrder(outRank, 0);
  for (size_t i = 0; i < numBatch; ++i) {
    int64_t jOut = mapDim(static_cast<int64_t>(i), outRank);
    resultDimOrder[jOut] = static_cast<int64_t>(i);
  }
  for (size_t f = 0; f < jLhsFree.size(); ++f) {
    int64_t sDim = mapDim(jLhsFree[f], lhsRank);
    auto it = llvm::find(sLhsFree, sDim);
    size_t pos = std::distance(sLhsFree.begin(), it);
    int64_t jOut = mapDim(static_cast<int64_t>(numBatch + pos), outRank);
    resultDimOrder[jOut] = static_cast<int64_t>(numBatch + f);
  }
  for (size_t g = 0; g < jRhsFree.size(); ++g) {
    int64_t sDim = mapDim(jRhsFree[g], rhsRank);
    auto it = llvm::find(sRhsFree, sDim);
    size_t pos = std::distance(sRhsFree.begin(), it);
    int64_t jOut = mapDim(
        static_cast<int64_t>(numBatch + sLhsFree.size() + pos), outRank);
    resultDimOrder[jOut] =
        static_cast<int64_t>(numBatch + jLhsFree.size() + g);
  }

  Value res = DotGeneralOp::create(b, loc, resTy, jLhs, jRhs, jLhsBatch,
                                   jRhsBatch, jLhsContract, jRhsContract,
                                   resultDimOrder);
  mapping.map(op.getResult(), res);
  return success();
}

static TransposeMode mapLapackTranspose(enzymexla::LapackTranspose t) {
  switch (t) {
  case enzymexla::LapackTranspose::none:
    return TransposeMode::none;
  case enzymexla::LapackTranspose::transpose:
    return TransposeMode::transpose;
  case enzymexla::LapackTranspose::adjoint:
    return TransposeMode::adjoint;
  }
  return TransposeMode::none;
}

LogicalResult StableHLOToJuliaLowerer::lowerGemm(enzymexla::GemmOp op,
                                                 OpBuilder &b,
                                                 IRMapping &mapping) {
  Location loc = op.getLoc();
  Value alpha = mapping.lookup(op.getAlpha());
  Value jA = mapping.lookup(op.getA());
  Value jB = mapping.lookup(op.getB());
  Value beta = mapping.lookup(op.getBeta());
  Value jC = mapping.lookup(op.getC());
  auto outRtt = cast<RankedTensorType>(op.getOutput().getType());
  int64_t rank = outRtt.getRank();
  Type resTy = convertType(outRtt);

  TransposeMode tA = mapLapackTranspose(op.getTransa());
  TransposeMode tB = mapLapackTranspose(op.getTransb());

  SmallVector<int64_t> batchDims;
  for (int64_t s = 0; s < rank - 2; ++s)
    batchDims.push_back(mapDim(s, rank));

  Value res;
  if (reverseDims) {
    // `(alpha * opA(A) * opB(B) + beta * C)ᵀ = alpha * opB(B)ᵀ * opA(A)ᵀ + beta * Cᵀ`
    res = MatmulOp::create(b, loc, resTy, jB, jA, jC, alpha, beta, tB, tA,
                           batchDims, batchDims, batchDims);
  } else {
    res = MatmulOp::create(b, loc, resTy, jA, jB, jC, alpha, beta, tA, tB,
                           batchDims, batchDims, batchDims);
  }
  mapping.map(op.getOutput(), res);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerReduce(stablehlo::ReduceOp op,
                                                   OpBuilder &b,
                                                   IRMapping &mapping) {
  Location loc = op.getLoc();
  SmallVector<Value> jInputs, jInits;
  for (Value v : op.getInputs())
    jInputs.push_back(mapping.lookup(v));
  for (Value v : op.getInitValues())
    jInits.push_back(mapping.lookup(v));

  auto inRtt = cast<RankedTensorType>(op.getInputs()[0].getType());
  int64_t inRank = inRtt.getRank();

  SmallVector<int64_t> jDims;
  for (int64_t d : op.getDimensions())
    jDims.push_back(mapDim(d, inRank));
  llvm::sort(jDims);

  SmallVector<Type> resTypes;
  for (Type t : op.getResultTypes())
    resTypes.push_back(convertType(t));

  auto redOp = ReduceOp::create(b, loc, resTypes, jInputs, jInits, jDims);
  IRMapping combMap = mapping;
  if (failed(lowerRegion(op.getBody(), redOp.getCombiner(), b, combMap)))
    return failure();

  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), redOp.getResults()))
    mapping.map(oldRes, newRes);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerReduceWindow(
    stablehlo::ReduceWindowOp op, OpBuilder &b, IRMapping &mapping) {
  Location loc = op.getLoc();
  SmallVector<Value> jInputs, jInits;
  for (Value v : op.getInputs())
    jInputs.push_back(mapping.lookup(v));
  for (Value v : op.getInitValues())
    jInits.push_back(mapping.lookup(v));

  auto inRtt = cast<RankedTensorType>(op.getInputs()[0].getType());
  int64_t rank = inRtt.getRank();

  SmallVector<int64_t> winDims = mapShape(op.getWindowDimensions());
  SmallVector<int64_t> strides(rank, 1), baseDil(rank, 1), winDil(rank, 1);
  if (auto s = op.getWindowStrides())
    strides = mapShape(*s);
  if (auto bd = op.getBaseDilations())
    baseDil = mapShape(*bd);
  if (auto wd = op.getWindowDilations())
    winDil = mapShape(*wd);

  SmallVector<int64_t> pad(2 * rank, 0);
  if (auto padAttr = op.getPadding()) {
    auto vals = padAttr->getValues<int64_t>();
    SmallVector<int64_t> rawPad(vals.begin(), vals.end());
    for (int64_t s = 0; s < rank; ++s) {
      int64_t jd = mapDim(s, rank);
      pad[2 * jd] = rawPad[2 * s];
      pad[2 * jd + 1] = rawPad[2 * s + 1];
    }
  }

  SmallVector<Type> resTypes;
  for (Type t : op.getResultTypes())
    resTypes.push_back(convertType(t));

  // If output is 0-D scalar (e.g. rank-0 reduce_window), handle degenerate case:
  auto outRtt = cast<RankedTensorType>(op.getResult(0).getType());
  if (outRtt.getRank() == 0) {
    for (auto [oldRes, jIn] : llvm::zip(op.getResults(), jInputs))
      mapping.map(oldRes, jIn);
    return success();
  }

  auto rwOp = ReduceWindowOp::create(b, loc, resTypes, jInputs, jInits, winDims,
                                     strides, baseDil, winDil, pad);
  IRMapping combMap = mapping;
  if (failed(lowerRegion(op.getBody(), rwOp.getCombiner(), b, combMap)))
    return failure();

  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), rwOp.getResults()))
    mapping.map(oldRes, newRes);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerGather(stablehlo::GatherOp op,
                                                   OpBuilder &b,
                                                   IRMapping &mapping) {
  Location loc = op.getLoc();
  Value jOperand = mapping.lookup(op.getOperand());
  Value jIndices = mapping.lookup(op.getStartIndices());
  auto opRtt = cast<RankedTensorType>(op.getOperand().getType());
  auto idxRtt = cast<RankedTensorType>(op.getStartIndices().getType());
  auto resRtt = cast<RankedTensorType>(op.getResult().getType());

  int64_t opRank = opRtt.getRank();
  int64_t idxRank = idxRtt.getRank();
  int64_t outRank = resRtt.getRank();

  auto dnums = op.getDimensionNumbers();
  SmallVector<int64_t> jOffsetDims;
  for (int64_t d : dnums.getOffsetDims())
    jOffsetDims.push_back(mapDim(d, outRank));
  if (reverseDims)
    std::reverse(jOffsetDims.begin(), jOffsetDims.end());

  SmallVector<int64_t> jCollapsed;
  for (int64_t d : dnums.getCollapsedSliceDims())
    jCollapsed.push_back(mapDim(d, opRank));

  SmallVector<int64_t> jOpBatch, jIdxBatch;
  for (int64_t d : dnums.getOperandBatchingDims())
    jOpBatch.push_back(mapDim(d, opRank));
  for (int64_t d : dnums.getStartIndicesBatchingDims())
    jIdxBatch.push_back(mapDim(d, idxRank));

  SmallVector<int64_t> jStartMap;
  for (int64_t d : dnums.getStartIndexMap())
    jStartMap.push_back(mapDim(d, opRank));

  int64_t sIdxVecDim = dnums.getIndexVectorDim();
  int64_t jIdxVecDim =
      (sIdxVecDim == idxRank) ? idxRank : mapDim(sIdxVecDim, idxRank);

  SmallVector<int64_t> jSliceSizes = mapShape(op.getSliceSizes());

  Type resTy = convertType(resRtt);
  Value res = GatherOp::create(b, loc, resTy, jOperand, jIndices, jOffsetDims,
                               jCollapsed, jOpBatch, jIdxBatch, jStartMap,
                               jIdxVecDim, jSliceSizes);
  mapping.map(op.getResult(), res);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerScatter(stablehlo::ScatterOp op,
                                                    OpBuilder &b,
                                                    IRMapping &mapping) {
  Location loc = op.getLoc();
  auto inRtt = cast<RankedTensorType>(op.getInputs()[0].getType());
  if (inRtt.getRank() == 0) {
    // 0-D scatter applies `update_computation(input, update)` once!
    IRMapping compMap = mapping;
    Block &srcBlk = op.getUpdateComputation().front();
    size_t n = op.getInputs().size();
    for (size_t i = 0; i < n; ++i) {
      compMap.map(srcBlk.getArgument(i), mapping.lookup(op.getInputs()[i]));
      compMap.map(srcBlk.getArgument(n + i),
                  mapping.lookup(op.getUpdates()[i]));
    }
    for (Operation &inner : srcBlk.without_terminator())
      if (failed(lowerOperation(&inner, b, compMap)))
        return failure();
    Operation *term = srcBlk.getTerminator();
    for (size_t i = 0; i < n; ++i)
      mapping.map(op.getResult(i), compMap.lookup(term->getOperand(i)));
    return success();
  }

  SmallVector<Value> jInputs, jUpdates;
  for (Value v : op.getInputs())
    jInputs.push_back(mapping.lookup(v));
  for (Value v : op.getUpdates())
    jUpdates.push_back(mapping.lookup(v));
  Value jIndices = mapping.lookup(op.getScatterIndices());

  auto idxRtt = cast<RankedTensorType>(op.getScatterIndices().getType());
  auto updRtt = cast<RankedTensorType>(op.getUpdates()[0].getType());
  int64_t opRank = inRtt.getRank();
  int64_t idxRank = idxRtt.getRank();
  int64_t updRank = updRtt.getRank();

  auto dnums = op.getScatterDimensionNumbers();
  SmallVector<int64_t> jUpdWinDims;
  for (int64_t d : dnums.getUpdateWindowDims())
    jUpdWinDims.push_back(mapDim(d, updRank));
  if (reverseDims)
    std::reverse(jUpdWinDims.begin(), jUpdWinDims.end());

  SmallVector<int64_t> jInsertedDims;
  for (int64_t d : dnums.getInsertedWindowDims())
    jInsertedDims.push_back(mapDim(d, opRank));

  SmallVector<int64_t> jInBatch, jIdxBatch;
  for (int64_t d : dnums.getInputBatchingDims())
    jInBatch.push_back(mapDim(d, opRank));
  for (int64_t d : dnums.getScatterIndicesBatchingDims())
    jIdxBatch.push_back(mapDim(d, idxRank));

  SmallVector<int64_t> jScatterMap;
  for (int64_t d : dnums.getScatterDimsToOperandDims())
    jScatterMap.push_back(mapDim(d, opRank));

  int64_t sIdxVecDim = dnums.getIndexVectorDim();
  int64_t jIdxVecDim =
      (sIdxVecDim == idxRank) ? idxRank : mapDim(sIdxVecDim, idxRank);

  SmallVector<Type> resTypes;
  for (Type t : op.getResultTypes())
    resTypes.push_back(convertType(t));

  auto scOp = ScatterOp::create(b, loc, resTypes, jInputs, jIndices, jUpdates,
                                jUpdWinDims, jInsertedDims, jInBatch, jIdxBatch,
                                jScatterMap, jIdxVecDim);
  IRMapping compMap = mapping;
  if (failed(
          lowerRegion(op.getUpdateComputation(), scOp.getUpdateComputation(), b,
                      compMap)))
    return failure();

  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), scOp.getResults()))
    mapping.map(oldRes, newRes);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerSort(stablehlo::SortOp op,
                                                 OpBuilder &b,
                                                 IRMapping &mapping) {
  Location loc = op.getLoc();
  auto inRtt = cast<RankedTensorType>(op.getInputs()[0].getType());
  int64_t rank = inRtt.getRank();
  if (rank == 0) {
    for (auto [oldRes, inVal] : llvm::zip(op.getResults(), op.getInputs()))
      mapping.map(oldRes, mapping.lookup(inVal));
    return success();
  }

  SmallVector<Value> jInputs;
  for (Value v : op.getInputs())
    jInputs.push_back(mapping.lookup(v));
  SmallVector<Type> resTypes;
  for (Type t : op.getResultTypes())
    resTypes.push_back(convertType(t));

  int64_t sDim = op.getDimension();
  if (sDim < 0)
    sDim += rank;
  int64_t jDim = mapDim(sDim, rank);

  auto sortOp =
      SortOp::create(b, loc, resTypes, jInputs, jDim, op.getIsStable());
  IRMapping compMap = mapping;
  if (failed(lowerRegion(op.getComparator(), sortOp.getComparator(), b, compMap)))
    return failure();

  for (auto [oldRes, newRes] : llvm::zip(op.getResults(), sortOp.getResults()))
    mapping.map(oldRes, newRes);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerConvolution(
    stablehlo::ConvolutionOp op, OpBuilder &b, IRMapping &mapping) {
  Location loc = op.getLoc();
  Value jLhs = mapping.lookup(op.getLhs());
  Value jRhs = mapping.lookup(op.getRhs());
  auto lhsRtt = cast<RankedTensorType>(op.getLhs().getType());
  auto rhsRtt = cast<RankedTensorType>(op.getRhs().getType());
  auto resRtt = cast<RankedTensorType>(op.getResult().getType());
  int64_t lhsRank = lhsRtt.getRank();
  int64_t rhsRank = rhsRtt.getRank();
  int64_t outRank = resRtt.getRank();

  auto dnums = op.getDimensionNumbers();
  size_t numSpatial = dnums.getInputSpatialDimensions().size();

  SmallVector<int64_t> strides(numSpatial, 1);
  if (auto s = op.getWindowStrides())
    strides.assign(s->begin(), s->end());

  SmallVector<int64_t> lhsDil(numSpatial, 1);
  if (auto ld = op.getLhsDilation())
    lhsDil.assign(ld->begin(), ld->end());

  SmallVector<int64_t> rhsDil(numSpatial, 1);
  if (auto rd = op.getRhsDilation())
    rhsDil.assign(rd->begin(), rd->end());

  SmallVector<bool> winRev(numSpatial, false);
  if (auto wr = op.getWindowReversal())
    winRev.assign(wr->begin(), wr->end());

  SmallVector<int64_t> pad(2 * numSpatial, 0);
  if (auto padAttr = op.getPadding()) {
    auto vals = padAttr->getValues<int64_t>();
    pad.assign(vals.begin(), vals.end());
  }

  int64_t jInBatch = mapDim(dnums.getInputBatchDimension(), lhsRank);
  int64_t jInFeat = mapDim(dnums.getInputFeatureDimension(), lhsRank);
  SmallVector<int64_t> jInSpatial;
  for (int64_t d : dnums.getInputSpatialDimensions())
    jInSpatial.push_back(mapDim(d, lhsRank));

  int64_t jKerInFeat = mapDim(dnums.getKernelInputFeatureDimension(), rhsRank);
  int64_t jKerOutFeat =
      mapDim(dnums.getKernelOutputFeatureDimension(), rhsRank);
  SmallVector<int64_t> jKerSpatial;
  for (int64_t d : dnums.getKernelSpatialDimensions())
    jKerSpatial.push_back(mapDim(d, rhsRank));

  int64_t jOutBatch = mapDim(dnums.getOutputBatchDimension(), outRank);
  int64_t jOutFeat = mapDim(dnums.getOutputFeatureDimension(), outRank);
  SmallVector<int64_t> jOutSpatial;
  for (int64_t d : dnums.getOutputSpatialDimensions())
    jOutSpatial.push_back(mapDim(d, outRank));

  Type resTy = convertType(resRtt);
  Value res = ConvOp::create(
      b, loc, resTy, jLhs, jRhs, strides, pad, lhsDil, rhsDil, winRev, jInBatch,
      jInFeat, jInSpatial, jKerInFeat, jKerOutFeat, jKerSpatial, jOutBatch,
      jOutFeat, jOutSpatial, op.getFeatureGroupCount(),
      op.getBatchGroupCount());
  mapping.map(op.getResult(), res);
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerOperation(Operation *op,
                                                      OpBuilder &b,
                                                      IRMapping &mapping) {
  Location loc = op->getLoc();

  // Terminators: handled in `lowerRegion`.
  if (isa<func::ReturnOp, stablehlo::ReturnOp>(op))
    return success();

  // Constants
  if (auto cOp = dyn_cast<stablehlo::ConstantOp>(op))
    return lowerConstant(op, cOp.getValue(), b, mapping);
  if (auto cOp = dyn_cast<chlo::ConstantOp>(op))
    return lowerConstant(op, cOp.getValue(), b, mapping);
  if (auto aCst = dyn_cast<arith::ConstantOp>(op)) {
    if (auto elemAttr = dyn_cast<ElementsAttr>(aCst.getValue()))
      return lowerConstant(op, elemAttr, b, mapping);
    Value c = emitScalarConstantAttr(b, loc, aCst.getType(), aCst.getValue());
    mapping.map(aCst.getResult(), c);
    return success();
  }

  // OptimizationBarrier / No-op identity pass-through
  if (auto bar = dyn_cast<stablehlo::OptimizationBarrierOp>(op)) {
    for (auto [oldRes, inVal] : llvm::zip(bar.getResults(), bar.getOperands()))
      mapping.map(oldRes, mapping.lookup(inVal));
    return success();
  }

  // Function call
  if (auto callOp = dyn_cast<func::CallOp>(op)) {
    SmallVector<Value> args;
    for (Value v : callOp.getOperands())
      args.push_back(mapping.lookup(v));
    SmallVector<Type> resTypes;
    for (Type t : callOp.getResultTypes())
      resTypes.push_back(convertType(t));
    auto jCall = CallOp::create(b, loc, resTypes, callOp.getCalleeAttr(), args,
                                callOp.getArgAttrsAttr(),
                                callOp.getResAttrsAttr());
    for (auto [oldRes, newRes] :
         llvm::zip(callOp.getResults(), jCall.getResults()))
      mapping.map(oldRes, newRes);
    return success();
  }

  // Unary elementwise ops
  if (auto uKind = mapStableHLOUnary(op)) {
    auto resRtt = cast<RankedTensorType>(op->getResult(0).getType());
    auto inRtt = cast<RankedTensorType>(op->getOperand(0).getType());
    Value jIn = mapping.lookup(op->getOperand(0));
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jIn}, {inRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return UnaryOp::create(kb, kl, outScalarTy, *uKind, s[0]);
        });
    mapping.map(op->getResult(0), res);
    return success();
  }

  // CHLO special libm functions: erf, erfc, lgamma, digamma
  if (isa<chlo::ErfOp, chlo::ErfcOp, chlo::LgammaOp>(op)) {
    LibmFn fn = isa<chlo::ErfOp>(op)
                    ? LibmFn::erf
                    : (isa<chlo::ErfcOp>(op) ? LibmFn::erfc : LibmFn::lgamma);
    auto resRtt = cast<RankedTensorType>(op->getResult(0).getType());
    auto inRtt = cast<RankedTensorType>(op->getOperand(0).getType());
    Value jIn = mapping.lookup(op->getOperand(0));
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jIn}, {inRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return MathLibmOp::create(kb, kl, outScalarTy, fn, s[0]);
        });
    mapping.map(op->getResult(0), res);
    return success();
  }

  // Binary elementwise ops
  if (auto bKind = mapStableHLOBinary(op)) {
    auto resRtt = cast<RankedTensorType>(op->getResult(0).getType());
    auto lhsRtt = cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsRtt = cast<RankedTensorType>(op->getOperand(1).getType());
    Value jLhs = mapping.lookup(op->getOperand(0));
    Value jRhs = mapping.lookup(op->getOperand(1));
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jLhs, jRhs}, {lhsRtt, rhsRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return BinaryOp::create(kb, kl, outScalarTy, *bKind, s[0], s[1]);
        });
    mapping.map(op->getResult(0), res);
    return success();
  }

  // Compare
  if (auto cmpOp = dyn_cast<stablehlo::CompareOp>(op)) {
    CmpPred pred = mapComparisonDirection(cmpOp.getComparisonDirection());
    auto resRtt = cast<RankedTensorType>(cmpOp.getResult().getType());
    auto lhsRtt = cast<RankedTensorType>(cmpOp.getLhs().getType());
    auto rhsRtt = cast<RankedTensorType>(cmpOp.getRhs().getType());
    Value jLhs = mapping.lookup(cmpOp.getLhs());
    Value jRhs = mapping.lookup(cmpOp.getRhs());
    Value res = emitElementwise(
        b, loc, resRtt, {jLhs, jRhs}, {lhsRtt, rhsRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return CmpOp::create(kb, kl, pred, s[0], s[1]);
        });
    mapping.map(cmpOp.getResult(), res);
    return success();
  }
  if (auto bCmp = dyn_cast<chlo::BroadcastCompareOp>(op)) {
    CmpPred pred = mapChloComparisonDirection(bCmp.getComparisonDirection());
    auto resRtt = cast<RankedTensorType>(bCmp.getResult().getType());
    auto lhsRtt = cast<RankedTensorType>(bCmp.getLhs().getType());
    auto rhsRtt = cast<RankedTensorType>(bCmp.getRhs().getType());
    Value jLhs = mapping.lookup(bCmp.getLhs());
    Value jRhs = mapping.lookup(bCmp.getRhs());
    Value res = emitElementwise(
        b, loc, resRtt, {jLhs, jRhs}, {lhsRtt, rhsRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return CmpOp::create(kb, kl, pred, s[0], s[1]);
        });
    mapping.map(bCmp.getResult(), res);
    return success();
  }

  // Select
  if (isa<stablehlo::SelectOp, chlo::BroadcastSelectOp>(op)) {
    auto resRtt = cast<RankedTensorType>(op->getResult(0).getType());
    auto pRtt = cast<RankedTensorType>(op->getOperand(0).getType());
    auto tRtt = cast<RankedTensorType>(op->getOperand(1).getType());
    auto fRtt = cast<RankedTensorType>(op->getOperand(2).getType());
    Value jP = mapping.lookup(op->getOperand(0));
    Value jT = mapping.lookup(op->getOperand(1));
    Value jF = mapping.lookup(op->getOperand(2));
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jP, jT, jF}, {pRtt, tRtt, fRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return SelectOp::create(kb, kl, outScalarTy, s[0], s[1], s[2]);
        });
    mapping.map(op->getResult(0), res);
    return success();
  }

  // Clamp
  if (auto clOp = dyn_cast<stablehlo::ClampOp>(op)) {
    auto resRtt = cast<RankedTensorType>(clOp.getResult().getType());
    auto minRtt = cast<RankedTensorType>(clOp.getMin().getType());
    auto opRtt = cast<RankedTensorType>(clOp.getOperand().getType());
    auto maxRtt = cast<RankedTensorType>(clOp.getMax().getType());
    Value jMin = mapping.lookup(clOp.getMin());
    Value jOp = mapping.lookup(clOp.getOperand());
    Value jMax = mapping.lookup(clOp.getMax());
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jMin, jOp, jMax}, {minRtt, opRtt, maxRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return ClampOp::create(kb, kl, outScalarTy, s[0], s[1], s[2]);
        });
    mapping.map(clOp.getResult(), res);
    return success();
  }

  // Convert
  if (auto cvt = dyn_cast<stablehlo::ConvertOp>(op)) {
    auto resRtt = cast<RankedTensorType>(cvt.getResult().getType());
    auto inRtt = cast<RankedTensorType>(cvt.getOperand().getType());
    Value jIn = mapping.lookup(cvt.getOperand());
    Type outScalarTy = resRtt.getElementType();
    if (inRtt.getElementType() == outScalarTy) {
      mapping.map(cvt.getResult(), jIn);
      return success();
    }
    Value res = emitElementwise(
        b, loc, resRtt, {jIn}, {inRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return CastOp::create(kb, kl, outScalarTy, s[0]);
        });
    mapping.map(cvt.getResult(), res);
    return success();
  }

  // BitcastConvert (same rank / same bitwidth elementwise)
  if (auto bc = dyn_cast<stablehlo::BitcastConvertOp>(op)) {
    auto resRtt = cast<RankedTensorType>(bc.getResult().getType());
    auto inRtt = cast<RankedTensorType>(bc.getOperand().getType());
    if (resRtt.getRank() != inRtt.getRank())
      return op->emitError(
          "rank-changing bitcast_convert not yet lowered to Julia");
    Value jIn = mapping.lookup(bc.getOperand());
    Type outScalarTy = resRtt.getElementType();
    Value res = emitElementwise(
        b, loc, resRtt, {jIn}, {inRtt},
        [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
          return BitcastOp::create(kb, kl, outScalarTy, s[0]);
        });
    mapping.map(bc.getResult(), res);
    return success();
  }

  // Iota / DynamicIota
  if (auto iota = dyn_cast<stablehlo::IotaOp>(op)) {
    auto resRtt = cast<RankedTensorType>(iota.getResult().getType());
    auto resArrTy = cast<ArrayType>(convertType(resRtt));
    int64_t jDim = mapDim(iota.getIotaDimension(), resRtt.getRank());
    Type elemTy = resRtt.getElementType();

    auto ker = KernelOp::create(b, loc, TypeRange{resArrTy}, ValueRange{});
    Block *blk = &ker.getBody().emplaceBlock();
    SmallVector<Value> ivs;
    for (int64_t d = 0; d < resArrTy.getRank(); ++d)
      ivs.push_back(blk->addArgument(b.getIndexType(), loc));
    OpBuilder kb(blk, blk->end());
    Value one = createIndexConstant(kb, loc, 1);
    Value zeroBased = BinaryOp::create(kb, loc, b.getIndexType(),
                                       BinaryOpKind::sub, ivs[jDim], one);
    Value casted = CastOp::create(kb, loc, elemTy, zeroBased);
    YieldOp::create(kb, loc, ValueRange{casted});
    mapping.map(iota.getResult(), ker.getResult(0));
    return success();
  }

  // GetDimensionSizeOp / tensor::DimOp
  if (auto gds = dyn_cast<stablehlo::GetDimensionSizeOp>(op)) {
    auto inRtt = cast<RankedTensorType>(gds.getOperand().getType());
    Value jSrc = mapping.lookup(gds.getOperand());
    int64_t jDim = mapDim(gds.getDimension(), inRtt.getRank());
    Value szIdx = SizeOp::create(b, loc, jSrc, b.getI64IntegerAttr(jDim));
    Type resScalarTy =
        cast<RankedTensorType>(gds.getResult().getType()).getElementType();
    Value szCast = CastOp::create(b, loc, resScalarTy, szIdx);
    mapping.map(gds.getResult(), szCast);
    return success();
  }
  if (auto dimOp = dyn_cast<tensor::DimOp>(op)) {
    auto constIdx = dimOp.getConstantIndex();
    if (!constIdx)
      return op->emitError("non-constant tensor.dim index in Julia lowering");
    auto inRtt = cast<RankedTensorType>(dimOp.getSource().getType());
    Value jSrc = mapping.lookup(dimOp.getSource());
    int64_t jDim = mapDim(*constIdx, inRtt.getRank());
    Value szIdx = SizeOp::create(b, loc, jSrc, b.getI64IntegerAttr(jDim));
    mapping.map(dimOp.getResult(), szIdx);
    return success();
  }

  // BroadcastInDimOp & DynamicBroadcastInDimOp
  if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    auto inRtt = cast<RankedTensorType>(bcast.getOperand().getType());
    auto resRtt = cast<RankedTensorType>(bcast.getResult().getType());
    Value jIn = mapping.lookup(bcast.getOperand());
    if (resRtt.getRank() == 0) {
      mapping.map(bcast.getResult(), jIn);
      return success();
    }
    auto resArrTy = cast<ArrayType>(convertType(resRtt));
    if (inRtt.getRank() == 0) {
      Value filled = FillOp::create(b, loc, resArrTy, jIn, ValueRange{});
      mapping.map(bcast.getResult(), filled);
      return success();
    }
    SmallVector<int64_t> jBcastDims(inRtt.getRank());
    ArrayRef<int64_t> sBcastDims = bcast.getBroadcastDimensions();
    for (int64_t d = 0; d < inRtt.getRank(); ++d) {
      int64_t sInDim = mapDim(d, inRtt.getRank());
      jBcastDims[d] = mapDim(sBcastDims[sInDim], resRtt.getRank());
    }
    Value res =
        BroadcastInDimOp::create(b, loc, resArrTy, jIn, jBcastDims, ValueRange{});
    mapping.map(bcast.getResult(), res);
    return success();
  }

  if (auto dbcast = dyn_cast<stablehlo::DynamicBroadcastInDimOp>(op)) {
    auto inRtt = cast<RankedTensorType>(dbcast.getOperand().getType());
    auto resRtt = cast<RankedTensorType>(dbcast.getResult().getType());
    Value jIn = mapping.lookup(dbcast.getOperand());
    Value jOutDims = mapping.lookup(dbcast.getOutputDimensions());
    auto resArrTy = cast<ArrayType>(convertType(resRtt));
    SmallVector<Value> dynDims;
    for (int64_t d = 0; d < resArrTy.getRank(); ++d) {
      if (resArrTy.isDynamicDim(d)) {
        int64_t sDim = mapDim(d, resRtt.getRank());
        Value oneBased = createIndexConstant(b, loc, sDim + 1);
        Value raw = LoadOp::create(
            b, loc, cast<ShapedType>(jOutDims.getType()).getElementType(),
            jOutDims, ValueRange{oneBased});
        dynDims.push_back(castToIndex(b, loc, raw));
      }
    }
    if (inRtt.getRank() == 0) {
      Value filled = FillOp::create(b, loc, resArrTy, jIn, dynDims);
      mapping.map(dbcast.getResult(), filled);
      return success();
    }
    SmallVector<int64_t> jBcastDims(inRtt.getRank());
    ArrayRef<int64_t> sBcastDims = dbcast.getBroadcastDimensions();
    for (int64_t d = 0; d < inRtt.getRank(); ++d) {
      int64_t sInDim = mapDim(d, inRtt.getRank());
      jBcastDims[d] = mapDim(sBcastDims[sInDim], resRtt.getRank());
    }
    Value res =
        BroadcastInDimOp::create(b, loc, resArrTy, jIn, jBcastDims, dynDims);
    mapping.map(dbcast.getResult(), res);
    return success();
  }

  // TransposeOp
  if (auto tr = dyn_cast<stablehlo::TransposeOp>(op)) {
    auto resRtt = cast<RankedTensorType>(tr.getResult().getType());
    int64_t rank = resRtt.getRank();
    Value jIn = mapping.lookup(tr.getOperand());
    if (rank <= 1) {
      mapping.map(tr.getResult(), jIn);
      return success();
    }
    ArrayRef<int64_t> sPerm = tr.getPermutation();
    SmallVector<int64_t> jPerm(rank);
    bool isIdentity = true;
    for (int64_t jd = 0; jd < rank; ++jd) {
      int64_t sd = mapDim(jd, rank);
      jPerm[jd] = mapDim(sPerm[sd], rank);
      if (jPerm[jd] != jd)
        isIdentity = false;
    }
    if (isIdentity) {
      mapping.map(tr.getResult(), jIn);
      return success();
    }
    Value res = TransposeOp::create(b, loc, convertType(resRtt), jIn, jPerm);
    mapping.map(tr.getResult(), res);
    return success();
  }

  // ReverseOp
  if (auto rev = dyn_cast<stablehlo::ReverseOp>(op)) {
    auto resRtt = cast<RankedTensorType>(rev.getResult().getType());
    int64_t rank = resRtt.getRank();
    Value jIn = mapping.lookup(rev.getOperand());
    if (rank == 0 || rev.getDimensions().empty()) {
      mapping.map(rev.getResult(), jIn);
      return success();
    }
    SmallVector<int64_t> jDims;
    for (int64_t sd : rev.getDimensions())
      jDims.push_back(mapDim(sd, rank));
    Value res = ReverseOp::create(b, loc, convertType(resRtt), jIn, jDims);
    mapping.map(rev.getResult(), res);
    return success();
  }

  // ReshapeOp
  if (auto resh = dyn_cast<stablehlo::ReshapeOp>(op)) {
    auto inRtt = cast<RankedTensorType>(resh.getOperand().getType());
    auto resRtt = cast<RankedTensorType>(resh.getResult().getType());
    Value jIn = mapping.lookup(resh.getOperand());
    int64_t inRank = inRtt.getRank();
    int64_t outRank = resRtt.getRank();

    if (inRank == 0 && outRank == 0) {
      mapping.map(resh.getResult(), jIn);
      return success();
    }
    if (inRank == 0 && outRank >= 1) {
      auto resArrTy = cast<ArrayType>(convertType(resRtt));
      Value filled = FillOp::create(b, loc, resArrTy, jIn, ValueRange{});
      mapping.map(resh.getResult(), filled);
      return success();
    }
    if (inRank >= 1 && outRank == 0) {
      SmallVector<Value> ones(inRank, createIndexConstant(b, loc, 1));
      Value scalar =
          LoadOp::create(b, loc, resRtt.getElementType(), jIn, ones);
      mapping.map(resh.getResult(), scalar);
      return success();
    }

    auto resArrTy = cast<ArrayType>(convertType(resRtt));
    if (jIn.getType() == resArrTy) {
      mapping.map(resh.getResult(), jIn);
      return success();
    }

    // Check if reshape only inserts/deletes unit (`size == 1`) dimensions; if
    // so, linear order is identical in both row-major and column-major!
    auto stripOnes = [](ArrayRef<int64_t> sh) {
      SmallVector<int64_t> res;
      for (int64_t s : sh)
        if (s != 1)
          res.push_back(s);
      return res;
    };
    bool isUnitDimOnly =
        inRtt.hasStaticShape() && resRtt.hasStaticShape() &&
        stripOnes(inRtt.getShape()) == stripOnes(resRtt.getShape());

    if (reverseDims || isUnitDimOnly || (inRank <= 1 && outRank <= 1)) {
      Value res = ReshapeOp::create(b, loc, resArrTy, jIn, ValueRange{});
      mapping.map(resh.getResult(), res);
      return success();
    }

    // When `reverseDims == false` and reshape mixes non-unit multi-dimensional
    // axes, StableHLO reshapes in row-major order whereas Julia reshapes in
    // column-major order. Reversing dims -> reshaping -> reversing dims makes
    // `reverseDims = false` 100% semantically exact too!
    Value src = jIn;
    if (inRank >= 2) {
      SmallVector<int64_t> revInShape(inRtt.getShape().rbegin(),
                                      inRtt.getShape().rend());
      SmallVector<int64_t> revPerm(inRank);
      for (int64_t d = 0; d < inRank; ++d)
        revPerm[d] = inRank - 1 - d;
      src = TransposeOp::create(
          b, loc, ArrayType::get(revInShape, inRtt.getElementType()), src,
          revPerm);
    }
    SmallVector<int64_t> revOutShape(resRtt.getShape().rbegin(),
                                     resRtt.getShape().rend());
    Value mid = ReshapeOp::create(
        b, loc, ArrayType::get(revOutShape, resRtt.getElementType()), src,
        ValueRange{});
    if (outRank >= 2) {
      SmallVector<int64_t> revPerm(outRank);
      for (int64_t d = 0; d < outRank; ++d)
        revPerm[d] = outRank - 1 - d;
      mid = TransposeOp::create(b, loc, resArrTy, mid, revPerm);
    }
    mapping.map(resh.getResult(), mid);
    return success();
  }

  // DynamicReshapeOp
  if (auto dresh = dyn_cast<stablehlo::DynamicReshapeOp>(op)) {
    auto resRtt = cast<RankedTensorType>(dresh.getResult().getType());
    auto resArrTy = cast<ArrayType>(convertType(resRtt));
    Value jIn = mapping.lookup(dresh.getOperand());
    Value jShape = mapping.lookup(dresh.getOutputShape());
    SmallVector<Value> dynDims;
    for (int64_t d = 0; d < resArrTy.getRank(); ++d) {
      if (resArrTy.isDynamicDim(d)) {
        int64_t sDim = mapDim(d, resRtt.getRank());
        Value oneBased = createIndexConstant(b, loc, sDim + 1);
        Value raw = LoadOp::create(
            b, loc, cast<ShapedType>(jShape.getType()).getElementType(), jShape,
            ValueRange{oneBased});
        dynDims.push_back(castToIndex(b, loc, raw));
      }
    }
    Value res = ReshapeOp::create(b, loc, resArrTy, jIn, dynDims);
    mapping.map(dresh.getResult(), res);
    return success();
  }

  // SliceOp
  if (auto sl = dyn_cast<stablehlo::SliceOp>(op)) {
    auto resRtt = cast<RankedTensorType>(sl.getResult().getType());
    Value jIn = mapping.lookup(sl.getOperand());
    if (resRtt.getRank() == 0) {
      mapping.map(sl.getResult(), jIn);
      return success();
    }
    SmallVector<int64_t> jStarts = mapShape(sl.getStartIndices());
    SmallVector<int64_t> jLimits = mapShape(sl.getLimitIndices());
    SmallVector<int64_t> jStrides = mapShape(sl.getStrides());
    Value res = SliceOp::create(b, loc, convertType(resRtt), jIn, jStarts,
                                jLimits, jStrides);
    mapping.map(sl.getResult(), res);
    return success();
  }

  // DynamicSliceOp
  if (auto dsl = dyn_cast<stablehlo::DynamicSliceOp>(op)) {
    auto resRtt = cast<RankedTensorType>(dsl.getResult().getType());
    Value jIn = mapping.lookup(dsl.getOperand());
    if (resRtt.getRank() == 0) {
      mapping.map(dsl.getResult(), jIn);
      return success();
    }
    int64_t rank = resRtt.getRank();
    SmallVector<Value> jStarts(rank);
    for (int64_t s = 0; s < rank; ++s)
      jStarts[mapDim(s, rank)] = mapping.lookup(dsl.getStartIndices()[s]);
    SmallVector<int64_t> jSizes = mapShape(dsl.getSliceSizes());
    Value res = DynamicSliceOp::create(b, loc, convertType(resRtt), jIn,
                                       jStarts, jSizes);
    mapping.map(dsl.getResult(), res);
    return success();
  }

  // DynamicUpdateSliceOp
  if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op)) {
    auto resRtt = cast<RankedTensorType>(dus.getResult().getType());
    Value jOp = mapping.lookup(dus.getOperand());
    Value jUpd = mapping.lookup(dus.getUpdate());
    if (resRtt.getRank() == 0) {
      mapping.map(dus.getResult(), jUpd);
      return success();
    }
    int64_t rank = resRtt.getRank();
    SmallVector<Value> jStarts(rank);
    for (int64_t s = 0; s < rank; ++s)
      jStarts[mapDim(s, rank)] = mapping.lookup(dus.getStartIndices()[s]);
    Value res = DynamicUpdateSliceOp::create(b, loc, convertType(resRtt), jOp,
                                             jUpd, jStarts);
    mapping.map(dus.getResult(), res);
    return success();
  }

  // ConcatenateOp
  if (auto cat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
    auto resRtt = cast<RankedTensorType>(cat.getResult().getType());
    int64_t jDim = mapDim(cat.getDimension(), resRtt.getRank());
    SmallVector<Value> jIns;
    for (Value v : cat.getInputs())
      jIns.push_back(mapping.lookup(v));
    Value res = ConcatOp::create(b, loc, convertType(resRtt), jIns, jDim);
    mapping.map(cat.getResult(), res);
    return success();
  }

  // PadOp
  if (auto padOp = dyn_cast<stablehlo::PadOp>(op)) {
    auto resRtt = cast<RankedTensorType>(padOp.getResult().getType());
    Value jOp = mapping.lookup(padOp.getOperand());
    Value jPadVal = mapping.lookup(padOp.getPaddingValue());
    if (resRtt.getRank() == 0) {
      mapping.map(padOp.getResult(), jOp);
      return success();
    }
    SmallVector<int64_t> jLow = mapShape(padOp.getEdgePaddingLow());
    SmallVector<int64_t> jHigh = mapShape(padOp.getEdgePaddingHigh());
    SmallVector<int64_t> jInt = mapShape(padOp.getInteriorPadding());
    Value res = PadOp::create(b, loc, convertType(resRtt), jOp, jPadVal, jLow,
                              jHigh, jInt);
    mapping.map(padOp.getResult(), res);
    return success();
  }

  // Contractions & Linear Algebra
  if (auto dg = dyn_cast<stablehlo::DotGeneralOp>(op))
    return lowerDotGeneral(dg, b, mapping);
  if (auto gemm = dyn_cast<enzymexla::GemmOp>(op))
    return lowerGemm(gemm, b, mapping);

  if (auto chol = dyn_cast<stablehlo::CholeskyOp>(op)) {
    auto resRtt = cast<RankedTensorType>(chol.getResult().getType());
    Value jIn = mapping.lookup(chol.getA());
    bool isComplex = isa<ComplexType>(resRtt.getElementType());
    // Note: for real matrices under `reverseDims`, `Aᵀ = (Lᵀ)ᵀ * (Lᵀ)`, so
    // flipping `lower` computes `Lᵀ` directly in-place! For complex matrices
    // under `reverseDims`, conjugate before & after so `(Aᴴ)ᵀ = Ā = L̄ * Lᵀ`.
    if (reverseDims && isComplex) {
      jIn = emitElementwise(
          b, loc, resRtt, {jIn}, {resRtt},
          [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
            return UnaryOp::create(kb, kl, resRtt.getElementType(),
                                   UnaryOpKind::conj, s[0]);
          });
    }
    bool jLower = reverseDims ? !chol.getLower() : chol.getLower();
    Value res = CholeskyOp::create(b, loc, convertType(resRtt), jIn, jLower);
    if (reverseDims && isComplex) {
      res = emitElementwise(
          b, loc, resRtt, {res}, {resRtt},
          [&](OpBuilder &kb, Location kl, ArrayRef<Value> s) -> Value {
            return UnaryOp::create(kb, kl, resRtt.getElementType(),
                                   UnaryOpKind::conj, s[0]);
          });
    }
    mapping.map(chol.getResult(), res);
    return success();
  }

  if (auto trsm = dyn_cast<stablehlo::TriangularSolveOp>(op)) {
    auto resRtt = cast<RankedTensorType>(trsm.getResult().getType());
    Value jA = mapping.lookup(trsm.getA());
    Value jB = mapping.lookup(trsm.getB());
    bool jLeft = reverseDims ? !trsm.getLeftSide() : trsm.getLeftSide();
    bool jLower = reverseDims ? !trsm.getLower() : trsm.getLower();
    TransposeMode jTrans = TransposeMode::none;
    switch (trsm.getTransposeA()) {
    case stablehlo::Transpose::NO_TRANSPOSE:
    case stablehlo::Transpose::TRANSPOSE_INVALID:
      jTrans = TransposeMode::none;
      break;
    case stablehlo::Transpose::TRANSPOSE:
      jTrans = TransposeMode::transpose;
      break;
    case stablehlo::Transpose::ADJOINT:
      jTrans = TransposeMode::adjoint;
      break;
    }
    Value res = TriangularSolveOp::create(b, loc, convertType(resRtt), jA, jB,
                                          jLeft, jLower, trsm.getUnitDiagonal(),
                                          jTrans);
    mapping.map(trsm.getResult(), res);
    return success();
  }

  // Reductions, Indexing, Sorting, Convolution
  if (auto red = dyn_cast<stablehlo::ReduceOp>(op))
    return lowerReduce(red, b, mapping);
  if (auto rw = dyn_cast<stablehlo::ReduceWindowOp>(op))
    return lowerReduceWindow(rw, b, mapping);
  if (auto g = dyn_cast<stablehlo::GatherOp>(op))
    return lowerGather(g, b, mapping);
  if (auto s = dyn_cast<stablehlo::ScatterOp>(op))
    return lowerScatter(s, b, mapping);
  if (auto srt = dyn_cast<stablehlo::SortOp>(op))
    return lowerSort(srt, b, mapping);
  if (auto conv = dyn_cast<stablehlo::ConvolutionOp>(op))
    return lowerConvolution(conv, b, mapping);

  // Control flow: IfOp, CaseOp, WhileOp
  if (auto ifOp = dyn_cast<stablehlo::IfOp>(op)) {
    Value pred = mapping.lookup(ifOp.getPred());
    SmallVector<Type> resTypes;
    for (Type t : ifOp.getResultTypes())
      resTypes.push_back(convertType(t));
    auto jIf = IfOp::create(b, loc, resTypes, pred);
    IRMapping tMap = mapping, fMap = mapping;
    if (failed(
            lowerRegion(ifOp.getTrueBranch(), jIf.getThenRegion(), b, tMap)) ||
        failed(lowerRegion(ifOp.getFalseBranch(), jIf.getElseRegion(), b, fMap)))
      return failure();
    for (auto [oldRes, newRes] : llvm::zip(ifOp.getResults(), jIf.getResults()))
      mapping.map(oldRes, newRes);
    return success();
  }

  if (auto caseOp = dyn_cast<stablehlo::CaseOp>(op)) {
    Value idx = mapping.lookup(caseOp.getIndex());
    SmallVector<Type> resTypes;
    for (Type t : caseOp.getResultTypes())
      resTypes.push_back(convertType(t));
    size_t numBranches = caseOp.getBranches().size();

    // Build nested `julia.if` chain from `0 .. numBranches - 2`, with the last
    // branch `numBranches - 1` serving as the default/out-of-bounds branch per
    // StableHLO `case` semantics (`index < 0 || index >= N-1` runs branch N-1).
    std::function<LogicalResult(size_t, OpBuilder &, SmallVectorImpl<Value> &)>
        emitBranchChain = [&](size_t brIdx, OpBuilder &cb,
                              SmallVectorImpl<Value> &outVals) -> LogicalResult {
      if (brIdx + 1 >= numBranches) {
        Region &lastSrc = caseOp.getBranches()[numBranches - 1];
        IRMapping brMap = mapping;
        Block &srcBlk = lastSrc.front();
        for (Operation &inner : srcBlk.without_terminator())
          if (failed(lowerOperation(&inner, cb, brMap)))
            return failure();
        for (Value v : srcBlk.getTerminator()->getOperands())
          outVals.push_back(brMap.lookup(v));
        return success();
      }
      Value brConst = emitScalarConstantAttr(
          cb, loc, idx.getType(), cb.getIntegerAttr(idx.getType(), brIdx));
      Value cond = CmpOp::create(cb, loc, CmpPred::eq, idx, brConst);
      auto jIf = IfOp::create(cb, loc, resTypes, cond);
      IRMapping thenMap = mapping;
      if (failed(lowerRegion(caseOp.getBranches()[brIdx], jIf.getThenRegion(),
                            cb, thenMap)))
        return failure();

      Block *elseBlk = &jIf.getElseRegion().emplaceBlock();
      OpBuilder eb(elseBlk, elseBlk->end());
      SmallVector<Value> elseVals;
      if (failed(emitBranchChain(brIdx + 1, eb, elseVals)))
        return failure();
      YieldOp::create(eb, loc, elseVals);

      for (Value r : jIf.getResults())
        outVals.push_back(r);
      return success();
    };

    SmallVector<Value> finalResults;
    if (failed(emitBranchChain(0, b, finalResults)))
      return failure();
    for (auto [oldRes, newRes] : llvm::zip(caseOp.getResults(), finalResults))
      mapping.map(oldRes, newRes);
    return success();
  }

  if (auto whileOp = dyn_cast<stablehlo::WhileOp>(op)) {
    SmallVector<Value> jInits;
    SmallVector<Type> resTypes;
    for (Value v : whileOp.getOperand()) {
      Value mv = mapping.lookup(v);
      jInits.push_back(mv);
      resTypes.push_back(mv.getType());
    }
    auto jWhile = WhileOp::create(b, loc, resTypes, jInits);
    IRMapping condMap = mapping, bodyMap = mapping;
    if (failed(lowerRegion(whileOp.getCond(), jWhile.getCond(), b, condMap,
                           /*isFuncTop=*/false, /*isWhileCond=*/true)) ||
        failed(lowerRegion(whileOp.getBody(), jWhile.getBody(), b, bodyMap)))
      return failure();
    for (auto [oldRes, newRes] :
         llvm::zip(whileOp.getResults(), jWhile.getResults()))
      mapping.map(oldRes, newRes);
    return success();
  }

  return op->emitError("unsupported operation in LowerStableHLOToJuliaPass: ")
         << op->getName();
}

LogicalResult StableHLOToJuliaLowerer::lowerRegion(Region &srcRegion,
                                                   Region &dstRegion,
                                                   OpBuilder &b,
                                                   IRMapping &mapping,
                                                   bool isFuncTop,
                                                   bool isWhileCond) {
  if (srcRegion.empty())
    return success();
  if (!llvm::hasSingleElement(srcRegion))
    return srcRegion.getParentOp()->emitError(
        "multi-block regions are not supported in Julia lowering");

  Block &srcBlk = srcRegion.front();
  OpBuilder::InsertionGuard guard(b);
  Block *dstBlk = b.createBlock(&dstRegion);

  for (BlockArgument arg : srcBlk.getArguments()) {
    BlockArgument newArg =
        dstBlk->addArgument(convertType(arg.getType()), arg.getLoc());
    mapping.map(arg, newArg);
  }

  b.setInsertionPointToEnd(dstBlk);
  for (Operation &op : srcBlk.without_terminator()) {
    if (failed(lowerOperation(&op, b, mapping)))
      return failure();
  }

  Operation *term = srcBlk.getTerminator();
  SmallVector<Value> termOps;
  for (Value v : term->getOperands())
    termOps.push_back(mapping.lookup(v));

  if (isFuncTop) {
    ReturnOp::create(b, term->getLoc(), termOps);
  } else if (isWhileCond) {
    CondOp::create(b, term->getLoc(), termOps[0]);
  } else {
    YieldOp::create(b, term->getLoc(), termOps);
  }
  return success();
}

LogicalResult StableHLOToJuliaLowerer::lowerModule(ModuleOp module) {
  SmallVector<func::FuncOp> funcs;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto f = dyn_cast<func::FuncOp>(&op))
      funcs.push_back(f);
  }

  for (func::FuncOp fn : funcs) {
    OpBuilder b(fn);
    FunctionType newFnTy = convertFunctionType(fn.getFunctionType());
    auto jFn = FuncOp::create(b, fn.getLoc(), fn.getName(), newFnTy);
    if (auto vis = fn.getSymVisibilityAttr())
      jFn.setSymVisibilityAttr(vis);
    if (auto argAttrs = fn.getArgAttrsAttr())
      jFn.setArgAttrsAttr(argAttrs);
    if (auto resAttrs = fn.getResAttrsAttr())
      jFn.setResAttrsAttr(resAttrs);

    // Preserve custom function attributes (such as memory_effects / aliasing).
    for (NamedAttribute attr : fn->getAttrs()) {
      StringRef name = attr.getName().getValue();
      if (name == fn.getFunctionTypeAttrName() ||
          name == fn.getSymNameAttrName() ||
          name == fn.getSymVisibilityAttrName() ||
          name == fn.getArgAttrsAttrName() || name == fn.getResAttrsAttrName())
        continue;
      jFn->setAttr(attr.getName(), attr.getValue());
    }

    if (!fn.isExternal()) {
      IRMapping mapping;
      if (failed(lowerRegion(fn.getBody(), jFn.getBody(), b, mapping,
                             /*isFuncTop=*/true)))
        return failure();
    }
    fn.erase();
  }
  return success();
}

struct LowerStableHLOToJuliaPass
    : public impl::LowerStableHLOToJuliaPassBase<LowerStableHLOToJuliaPass> {
  using Base::Base;

  void runOnOperation() override {
    StableHLOToJuliaLowerer lowerer(reverse_dims, prefer_blas);
    if (failed(lowerer.lowerModule(getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::enzymexla::julia
