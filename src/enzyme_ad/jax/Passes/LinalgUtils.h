#ifndef ENZYMEXLA_LINALGUTILS_H
#define ENZYMEXLA_LINALGUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Attributes.h"

llvm::SmallVector<int64_t> columnMajorMatrixLayout(int64_t ndim);

llvm::SmallVector<int64_t> rowMajorMatrixLayout(int64_t ndim);

mlir::Attribute getSHLOLayout(mlir::PatternRewriter &rewriter, int64_t ndim,
                              bool isColMajor, int64_t maxNumDims);

mlir::ArrayAttr getSHLOLayout(mlir::PatternRewriter &rewriter,
                              llvm::SmallVector<int64_t> ndims,
                              llvm::SmallVector<bool> isColMajorArr,
                              int64_t maxNumDims);

std::optional<std::string> lapack_precision_prefix(mlir::Type elementType);

#endif // ENZYMEXLA_LINALGUTILS_H
