#ifndef ENZYME_AD_JAX_PASSES_BARRIER_UTILS_H
#define ENZYME_AD_JAX_PASSES_BARRIER_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzyme {

inline llvm::SmallVector<mlir::Value>
emitIterationCounts(mlir::OpBuilder &rewriter, mlir::scf::ParallelOp op) {
  using namespace mlir;
  SmallVector<Value> iterationCounts;
  for (auto bounds :
       llvm::zip(op.getLowerBound(), op.getUpperBound(), op.getStep())) {
    Value lowerBound = std::get<0>(bounds);
    Value upperBound = std::get<1>(bounds);
    Value step = std::get<2>(bounds);
    Value diff =
        rewriter.create<arith::SubIOp>(op.getLoc(), upperBound, lowerBound);
    Value count = rewriter.create<arith::CeilDivSIOp>(op.getLoc(), diff, step);
    iterationCounts.push_back(count);
  }
  return iterationCounts;
}

template <typename T>
static mlir::Value
allocateTemporaryBuffer(mlir::OpBuilder &rewriter, mlir::Value value,
                        mlir::ValueRange iterationCounts, bool alloca = true,
                        mlir::DataLayout *DLI = nullptr) {
  using namespace mlir;
  SmallVector<int64_t> bufferSize(iterationCounts.size(), ShapedType::kDynamic);
  mlir::Type ty = value.getType();
  if (alloca)
    if (auto allocaOp = value.getDefiningOp<memref::AllocaOp>()) {
      auto mt = allocaOp.getType();
      bool hasDynamicSize = false;
      for (auto s : mt.getShape()) {
        if (s == ShapedType::kDynamic) {
          hasDynamicSize = true;
          break;
        }
      }
      if (!hasDynamicSize) {
        for (auto s : mt.getShape()) {
          bufferSize.push_back(s);
        }
        ty = mt.getElementType();
      }
    }
  auto type = MemRefType::get(bufferSize, ty);
  return rewriter.create<T>(value.getLoc(), type, iterationCounts);
}

}
}

#endif