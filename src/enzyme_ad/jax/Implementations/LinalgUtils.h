#pragma once

#include "mlir/IR/Builders.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <numeric>
#include <utility>

// Helpers for the derivatives of linear algebra ops. The last two dimensions
// are the matrix dimensions, the others are batch dimensions.

namespace mlir {
namespace enzyme {

inline bool isComplexTensor(Value v) {
  return isa<ComplexType>(cast<ShapedType>(v.getType()).getElementType());
}

inline Value conjIfComplex(OpBuilder &builder, Location loc, Value v) {
  if (!isComplexTensor(v))
    return v;
  return chlo::ConjOp::create(builder, loc, v);
}

// add the leading dimension of the shadows to a primal value
inline Value broadcastToWidth(OpBuilder &builder, Location loc, Value v,
                              int64_t width) {
  if (width == 1)
    return v;
  auto ty = cast<RankedTensorType>(v.getType());
  SmallVector<int64_t> shape{width};
  shape.append(ty.getShape().begin(), ty.getShape().end());
  SmallVector<int64_t> dims(ty.getRank());
  std::iota(dims.begin(), dims.end(), 1);
  return stablehlo::BroadcastInDimOp::create(
      builder, loc, RankedTensorType::get(shape, ty.getElementType()), v, dims);
}

// conjugate transpose
inline Value adjointMatrix(OpBuilder &builder, Location loc, Value v) {
  int64_t rank = cast<RankedTensorType>(v.getType()).getRank();
  SmallVector<int64_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[rank - 1], perm[rank - 2]);
  Value t = stablehlo::TransposeOp::create(builder, loc, v, perm);
  return conjIfComplex(builder, loc, t);
}

// batched matrix product, optionally of the transposed operands
inline Value batchedMatmul(OpBuilder &builder, Location loc, Value lhs,
                           Value rhs, bool transposeLhs = false,
                           bool transposeRhs = false) {
  auto lhsTy = cast<RankedTensorType>(lhs.getType());
  auto rhsTy = cast<RankedTensorType>(rhs.getType());
  int64_t rank = lhsTy.getRank();
  int64_t lhsContract = transposeLhs ? rank - 2 : rank - 1;
  int64_t rhsContract = transposeRhs ? rank - 1 : rank - 2;
  SmallVector<int64_t> batch(rank - 2);
  std::iota(batch.begin(), batch.end(), 0);
  SmallVector<int64_t> shape(lhsTy.getShape().begin(),
                             lhsTy.getShape().begin() + (rank - 2));
  shape.push_back(lhsTy.getDimSize(transposeLhs ? rank - 1 : rank - 2));
  shape.push_back(rhsTy.getDimSize(transposeRhs ? rank - 2 : rank - 1));
  auto dims = stablehlo::DotDimensionNumbersAttr::get(
      builder.getContext(), batch, batch, {lhsContract}, {rhsContract});
  auto highest = stablehlo::PrecisionAttr::get(builder.getContext(),
                                               stablehlo::Precision::HIGHEST);
  return stablehlo::DotGeneralOp::create(
      builder, loc, RankedTensorType::get(shape, lhsTy.getElementType()), lhs,
      rhs, dims, builder.getArrayAttr({highest, highest}),
      stablehlo::DotAlgorithmAttr());
}

inline Value splatLike(OpBuilder &builder, Location loc, Value like,
                       double val) {
  auto ty = cast<RankedTensorType>(like.getType());
  return stablehlo::ConstantOp::create(builder, loc, ty,
                                       cast<ElementsAttr>(makeAttr(ty, val)));
}

// mask of the entries of v with (row index) dir (column index)
inline Value matrixIndexPredicate(OpBuilder &builder, Location loc, Value v,
                                  stablehlo::ComparisonDirection dir) {
  auto ty = cast<RankedTensorType>(v.getType());
  int64_t rank = ty.getRank();
  auto indexTy = RankedTensorType::get(ty.getShape(), builder.getI32Type());
  Value row = stablehlo::IotaOp::create(builder, loc, indexTy, rank - 2);
  Value col = stablehlo::IotaOp::create(builder, loc, indexTy, rank - 1);
  return stablehlo::CompareOp::create(builder, loc, row, col, dir);
}

inline Value keepTriangle(OpBuilder &builder, Location loc, Value v,
                          stablehlo::ComparisonDirection dir) {
  return stablehlo::SelectOp::create(builder, loc,
                                     matrixIndexPredicate(builder, loc, v, dir),
                                     v, splatLike(builder, loc, v, 0.0));
}

// the entries of a triangular operand that are read
inline stablehlo::ComparisonDirection readTriangle(bool lower,
                                                   bool unitDiagonal) {
  if (lower)
    return unitDiagonal ? stablehlo::ComparisonDirection::GT
                        : stablehlo::ComparisonDirection::GE;
  return unitDiagonal ? stablehlo::ComparisonDirection::LT
                      : stablehlo::ComparisonDirection::LE;
}

inline Value triangularSolve(OpBuilder &builder, Location loc, Value a, Value b,
                             bool leftSide, bool lower, bool unitDiagonal,
                             stablehlo::Transpose t) {
  return stablehlo::TriangularSolveOp::create(builder, loc, b.getType(), a, b,
                                              leftSide, lower, unitDiagonal, t);
}

inline stablehlo::Transpose adjointTranspose(Value a) {
  return isComplexTensor(a) ? stablehlo::Transpose::ADJOINT
                            : stablehlo::Transpose::TRANSPOSE;
}

} // namespace enzyme
} // namespace mlir
