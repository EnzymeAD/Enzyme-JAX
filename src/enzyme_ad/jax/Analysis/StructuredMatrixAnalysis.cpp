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

// most specific pattern
StructuredSparsityPattern
StructuredSparsityPattern::meet(const StructuredSparsityPattern &lhs,
                                const StructuredSparsityPattern &rhs) {
  if (lhs.kind == StructuredSparsityKind::Empty ||
      rhs.kind == StructuredSparsityKind::Empty)
    return StructuredSparsityPattern(StructuredSparsityKind::Empty);

  if (lhs.kind == StructuredSparsityKind::Unknown)
    return rhs;
  if (rhs.kind == StructuredSparsityKind::Unknown)
    return lhs;

  // for all other cases, we take the min of the bandwidths and refine
  auto lb = std::min(lhs.lowerBandwidth, rhs.lowerBandwidth);
  auto ub = std::min(lhs.upperBandwidth, rhs.upperBandwidth);
  auto newPattern = StructuredSparsityPattern(lb, ub);
  newPattern.refineKind();
  return newPattern;
}

// least specific structure containing both
StructuredSparsityPattern
StructuredSparsityPattern::join(const StructuredSparsityPattern &lhs,
                                const StructuredSparsityPattern &rhs) {
  if (lhs.kind == StructuredSparsityKind::Empty)
    return rhs;
  if (rhs.kind == StructuredSparsityKind::Empty)
    return lhs;

  if (lhs.kind == StructuredSparsityKind::Unknown ||
      rhs.kind == StructuredSparsityKind::Unknown)
    return StructuredSparsityPattern(StructuredSparsityKind::Unknown);

  auto lb = std::max(lhs.lowerBandwidth, rhs.lowerBandwidth);
  auto ub = std::max(lhs.upperBandwidth, rhs.upperBandwidth);
  auto newPattern = StructuredSparsityPattern(lb, ub);
  newPattern.refineKind();
  return newPattern;
}

void StructuredSparsityPattern::print(raw_ostream &os) const {
  switch (kind) {
  case StructuredSparsityKind::Unknown:
    os << "Unknown";
    break;
  case StructuredSparsityKind::Dense:
    os << "Dense";
    break;
  case StructuredSparsityKind::Band:
    os << "Band(" << lowerBandwidth << ", " << upperBandwidth << ")";
    break;
  case StructuredSparsityKind::UpperTriangular:
    os << "UpperTriangular";
    break;
  case StructuredSparsityKind::UpperBidiagonal:
    os << "UpperBidiagonal";
    break;
  case StructuredSparsityKind::LowerTriangular:
    os << "LowerTriangular";
    break;
  case StructuredSparsityKind::LowerBidiagonal:
    os << "LowerBidiagonal";
    break;
  case StructuredSparsityKind::Tridiagonal:
    os << "Tridiagonal";
    break;
  case StructuredSparsityKind::Diagonal:
    os << "Diagonal";
    break;
  case StructuredSparsityKind::Empty:
    os << "Empty";
    break;
  }
}

//===----------------------------------------------------------------------===//
// Value Properties Implementation
//===----------------------------------------------------------------------===//

ValueProperties ValueProperties::meet(const ValueProperties &lhs,
                                      const ValueProperties &rhs) {
  return ValueProperties(lhs.flags & rhs.flags);
}

ValueProperties ValueProperties::join(const ValueProperties &lhs,
                                      const ValueProperties &rhs) {
  return ValueProperties(lhs.flags | rhs.flags);
}

void ValueProperties::print(raw_ostream &os) const {
  os << "{";
  bool first = true;
  auto add = [&](const char *s) {
    if (!first)
      os << ", ";
    os << s;
    first = false;
  };

  if (hasUnitDiagonal())
    add("UnitDiagonal");
  if (isSymmetric())
    add("Symmetric");
  if (isHermitian())
    add("Hermitian");
  if (isBroadcastedScalar())
    add("BroadcastedScalar");

  os << "}";
}

//===----------------------------------------------------------------------===//
// Structured Matrix Type
//===----------------------------------------------------------------------===//

StructuredMatrixType
StructuredMatrixType::meet(const StructuredMatrixType &lhs,
                           const StructuredMatrixType &rhs) {
  return StructuredMatrixType(
      StructuredSparsityPattern::meet(lhs.sparsityPattern, rhs.sparsityPattern),
      ValueProperties::meet(lhs.valueProperties, rhs.valueProperties));
}

StructuredMatrixType
StructuredMatrixType::join(const StructuredMatrixType &lhs,
                           const StructuredMatrixType &rhs) {
  return StructuredMatrixType(
      StructuredSparsityPattern::join(lhs.sparsityPattern, rhs.sparsityPattern),
      ValueProperties::join(lhs.valueProperties, rhs.valueProperties));
}

void StructuredMatrixType::print(raw_ostream &os) const {
  os << "StructuredMatrixType(";
  sparsityPattern.print(os);
  os << " ";
  valueProperties.print(os);
  os << ")";
}

//===----------------------------------------------------------------------===//
// Lattice Element
//===----------------------------------------------------------------------===//

void StructuredMatrixLattice::print(raw_ostream &os) const {
  os << "StructuredMatrixLattice(";
  value.print(os);
  os << ")";
}

//===----------------------------------------------------------------------===//
// Dataflow Analysis
//===----------------------------------------------------------------------===//

void StructuredMatrixAnalysis::setToEntryState(
    StructuredMatrixLattice *lattice) {
  lattice->setValue(StructuredMatrixType());
}

LogicalResult StructuredMatrixAnalysis::visitOperation(
    Operation *op, ArrayRef<const StructuredMatrixLattice *> operands,
    ArrayRef<StructuredMatrixLattice *> results) {
  llvm::errs() << "Visiting operation " << *op << "\n";

  return success();
}

//===----------------------------------------------------------------------===//
// Structure Originators
//===----------------------------------------------------------------------===//

} // namespace structure_analysis
} // namespace mlir
