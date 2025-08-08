#include "LinalgUtils.h"

using namespace mlir;

// https://github.com/jax-ml/jax/blob/48001a24cb74f311b51d8bcf0891437069db6b95/jax/_src/lax/linalg.py#L2792
SmallVector<int64_t> columnMajorMatrixLayout(int64_t ndim) {
  SmallVector<int64_t> layout = {ndim - 2, ndim - 1};
  for (int64_t i = ndim - 3; i >= 0; i--) {
    layout.push_back(i);
  }
  return layout;
}

SmallVector<int64_t> rowMajorMatrixLayout(int64_t ndim) {
  SmallVector<int64_t> layout;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    layout.push_back(i);
  }
  return layout;
}

mlir::Attribute getSHLOLayout(PatternRewriter &rewriter, int64_t ndim,
                              bool isColMajor, int64_t maxNumDims) {
  if (isColMajor && ndim == maxNumDims) {
    return rewriter.getIndexTensorAttr(columnMajorMatrixLayout(ndim));
  }
  return rewriter.getIndexTensorAttr(rowMajorMatrixLayout(ndim));
}

mlir::ArrayAttr getSHLOLayout(PatternRewriter &rewriter,
                              SmallVector<int64_t> ndims,
                              SmallVector<bool> isColMajorArr,
                              int64_t maxNumDims) {
  SmallVector<mlir::Attribute> attrs;
  for (auto [ndim, isColMajor] : llvm::zip(ndims, isColMajorArr)) {
    attrs.push_back(getSHLOLayout(rewriter, ndim, isColMajor, maxNumDims));
  }
  return rewriter.getArrayAttr(attrs);
}

std::optional<std::string> lapack_precision_prefix(Type elementType) {

  // single-precision float
  if (elementType.isF32()) {
    return "s";

    // double-precision float
  } else if (elementType.isF64()) {
    return "d";

  } else if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    auto elem = complexType.getElementType();

    // single-precision complex
    if (elem.isF32()) {
      return "c";

      // double-precision complex
    } else if (elem.isF64()) {
      return "z";

    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
}
