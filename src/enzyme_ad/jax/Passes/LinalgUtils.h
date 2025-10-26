#ifndef ENZYMEXLA_LINALGUTILS_H
#define ENZYMEXLA_LINALGUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

llvm::SmallVector<int64_t> columnMajorMatrixLayout(int64_t ndim);

llvm::SmallVector<int64_t> rowMajorMatrixLayout(int64_t ndim);

mlir::Attribute getSHLOLayout(mlir::PatternRewriter &rewriter, int64_t ndim,
                              bool isColMajor, int64_t maxNumDims);

mlir::ArrayAttr getSHLOLayout(mlir::PatternRewriter &rewriter,
                              llvm::SmallVector<int64_t> ndims,
                              llvm::SmallVector<bool> isColMajorArr,
                              int64_t maxNumDims);

std::optional<std::string> lapackPrecisionPrefix(mlir::Type elementType);

#endif // ENZYMEXLA_LINALGUTILS_H
