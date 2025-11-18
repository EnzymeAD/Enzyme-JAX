#include "src/enzyme_ad/jax/Analysis/StructuredMatrixAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace structure_analysis {

//===----------------------------------------------------------------------===//
// Structured Sparsity Pattern Implementation
//===----------------------------------------------------------------------===//

void StructuredSparsityPattern::initializeBandwidths() {
  switch (kind) {
  case StructuredSparsityKind::Diagonal:
    lowerBandwidth = 0;
    upperBandwidth = 0;
    break;
  case StructuredSparsityKind::Bidiagonal:
    lowerBandwidth = 0;
    upperBandwidth = 1;
    break;
  case StructuredSparsityKind::Tridiagonal:
    lowerBandwidth = 1;
    upperBandwidth = 1;
    break;
  case StructuredSparsityKind::UpperTriangular:
    lowerBandwidth = 0;
    upperBandwidth = std::numeric_limits<int64_t>::max();
    break;
  case StructuredSparsityKind::LowerTriangular:
    lowerBandwidth = std::numeric_limits<int64_t>::max();
    upperBandwidth = 0;
    break;
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// Value Properties Implementation
//===----------------------------------------------------------------------===//

} // namespace structure_analysis
} // namespace mlir
