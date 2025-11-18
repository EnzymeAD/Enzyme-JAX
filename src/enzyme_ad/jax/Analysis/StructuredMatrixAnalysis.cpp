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
  case StructuredSparsityKind::Unknown:
    break; // leave as is
  case StructuredSparsityKind::Dense:
    lowerBandwidth = std::numeric_limits<int64_t>::max();
    upperBandwidth = std::numeric_limits<int64_t>::max();
  case StructuredSparsityKind::Band:
    llvm_unreachable("constructing band with no bandwidths");
  case StructuredSparsityKind::UpperTriangular:
    lowerBandwidth = 0;
    upperBandwidth = std::numeric_limits<int64_t>::max();
    break;
  case StructuredSparsityKind::UpperBidiagonal:
    lowerBandwidth = 0;
    upperBandwidth = 1;
    break;
  case StructuredSparsityKind::LowerTriangular:
    lowerBandwidth = std::numeric_limits<int64_t>::max();
    upperBandwidth = 0;
    break;
  case StructuredSparsityKind::LowerBidiagonal:
    lowerBandwidth = 1;
    upperBandwidth = 0;
    break;
  case StructuredSparsityKind::Tridiagonal:
    lowerBandwidth = 1;
    upperBandwidth = 1;
    break;
  case StructuredSparsityKind::Diagonal:
    lowerBandwidth = 0;
    upperBandwidth = 0;
    break;
  case StructuredSparsityKind::Empty:
    break;
  }
}

void StructuredSparsityPattern::refineKind() {
  if (kind != StructuredSparsityKind::Band)
    return;

  if (lowerBandwidth == 0) {
    if (upperBandwidth == 0) {
      kind = StructuredSparsityKind::Diagonal;
      return;
    }
    if (upperBandwidth == 1) {
      kind = StructuredSparsityKind::UpperBidiagonal;
      return;
    }
    if (upperBandwidth == std::numeric_limits<int64_t>::max()) {
      kind = StructuredSparsityKind::UpperTriangular;
      return;
    }
  }

  // lowerBandwidth != 0
  if (upperBandwidth == 0) {
    if (lowerBandwidth == 1) {
      kind = StructuredSparsityKind::LowerBidiagonal;
      return;
    }
    if (lowerBandwidth == std::numeric_limits<int64_t>::max()) {
      kind = StructuredSparsityKind::LowerTriangular;
      return;
    }
  }

  if (lowerBandwidth == 1 && upperBandwidth == 1) {
    kind = StructuredSparsityKind::Tridiagonal;
    return;
  }

  if (lowerBandwidth == std::numeric_limits<int64_t>::max() &&
      upperBandwidth == std::numeric_limits<int64_t>::max()) {
    kind = StructuredSparsityKind::Dense;
    return;
  }
}

//===----------------------------------------------------------------------===//
// Value Properties Implementation
//===----------------------------------------------------------------------===//

} // namespace structure_analysis
} // namespace mlir
