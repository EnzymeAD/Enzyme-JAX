#include "src/enzyme_ad/jax/Analysis/StructuredMatrixAnalysis.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace structure_analysis {

//===----------------------------------------------------------------------===//
// Structured Sparsity Pattern Implementation
//===----------------------------------------------------------------------===//

StructuredSparsityPattern::StructuredSparsityPattern(Value v) {
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    // TODO: If block arg is annotated with a pattern we should parse that
    setUnknown(); // be pessimistic by default
    return;
  }

  llvm::errs() << "TODO: structured sparsity pattern not implemented for " << v
               << "\n";
  setUnknown();
  return;
}

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

ValueProperties::ValueProperties(Value v) {
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    // TODO: If block arg is annotated with a pattern we should parse that
    setFlags(0); // be pessimistic by default
    return;
  }

  auto vTy = cast<RankedTensorType>(v.getType());
  if (!vTy.hasStaticShape() || vTy.getRank() != 2)
    return;
  auto vShape = vTy.getShape();
  if (vShape[0] != vShape[1]) // TODO: should we allow rectangular matrices?
    return;

  DenseElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr))) {
    auto props = getPropertiesFromDenseAttr(denseAttr);
    setFlags(props.getFlags());
    return;
  }

  auto defOp = v.getDefiningOp();
  if (!defOp)
    return;

  // check that transpose dimensions are [1,0]
  auto isTrueTranspose = [](stablehlo::TransposeOp tOp) -> bool {
    auto perm = tOp.getPermutation();
    return perm.size() == 2 && perm[0] == 1 && perm[1] == 0;
  };

  // comm_op(A, A^T) will always be symmetric
  if (stablehlo::hasTraitElementwise(defOp) &&
      (defOp->hasTrait<OpTrait::IsCommutative>() ||
       defOp->hasTrait<hlo::OpTrait::IsCommutative>())) {
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);

    if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(rhsT) && lhs == rhsT.getOperand()) {
        set(ValueProperty::Symmetric);
      }
    }

    if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(lhsT) && rhs == lhsT.getOperand()) {
        set(ValueProperty::Symmetric);
      }
    }
  }

  // TODO: A x A^T will always be symmetric

  if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(defOp)) {
    auto operand = bcastOp.getOperand();
    if (cast<RankedTensorType>(operand.getType()).getRank() == 0) { // bcast(scalar)
      if (matchPattern(operand, m_One())) // bcast(1)
        set(ValueProperty::UnitDiagonal);
      set(ValueProperty::BroadcastedScalar);
      set(ValueProperty::Symmetric);
      return;
    }
  }

  // TODO: unit diagonal
  //       - iota scatter with constant

  return;
}

ValueProperties
ValueProperties::getPropertiesFromDenseAttr(DenseElementsAttr attr) {
  ValueProperties props;

  if (attr.isSplat()) {
    auto val = attr.getSplatValue<Attribute>();
    if (utils::isOne(val))
      props.set(ValueProperty::UnitDiagonal);

    props.set(ValueProperty::BroadcastedScalar);
    props.set(ValueProperty::Symmetric);
    props.set(ValueProperty::Hermitian);
    return props;
  }

  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type)
    return props;

  auto shape = type.getShape();
  int64_t nrows = shape[0];
  int64_t ncols = shape[1];
  if (nrows != ncols)
    return props;

  if (isUnitDiagonal(attr, nrows, ncols))
    props.set(ValueProperty::UnitDiagonal);

  auto [isSymmetric, isHermitian] = isSymmetricOrHermitian(attr, nrows, ncols);
  if (isSymmetric)
    props.set(ValueProperty::Symmetric);
  if (isHermitian)
    props.set(ValueProperty::Hermitian);

  return props;
}

template <typename T>
bool isUnitDiagonalImpl(DenseElementsAttr attr, int64_t nrows, int64_t ncols) {
  auto values = attr.getValues<T>().begin();
  for (int64_t i = 0; i < std::min(nrows, ncols); i++) {
    if (!utils::isOne(values[i]))
      return false;
  }
  return true;
}

bool ValueProperties::isUnitDiagonal(DenseElementsAttr attr, int64_t nrows,
                                     int64_t ncols) {
  if (isa<IntegerType>(attr.getElementType())) {
    return isUnitDiagonalImpl<APInt>(attr, nrows, ncols);
  } else if (isa<FloatType>(attr.getElementType())) {
    return isUnitDiagonalImpl<APFloat>(attr, nrows, ncols);
  }
  return false;
}

template <typename T>
std::tuple<int64_t, int64_t> isSymmetricOrHermitianImpl(DenseElementsAttr attr,
                                                        int64_t nrows,
                                                        int64_t ncols) {
  auto values = attr.getValues<T>().begin();
  for (int64_t i = 0; i < nrows; i++) {
    for (int64_t j = i + 1; j < ncols; j++) {
      auto a = *(values + i * ncols + j);
      auto b = *(values + j * ncols + i);
      if (!utils::areEqual(a, b)) {
        return {false, false}; // TODO: check for hermitian
      }
    }
  }

  return {true, false}; // TODO: check for hermitian
}

std::tuple<int64_t, int64_t>
ValueProperties::isSymmetricOrHermitian(DenseElementsAttr attr, int64_t nrows,
                                        int64_t ncols) {
  if (isa<IntegerType>(attr.getElementType())) {
    return isSymmetricOrHermitianImpl<APInt>(attr, nrows, ncols);
  } else if (isa<FloatType>(attr.getElementType())) {
    return isSymmetricOrHermitianImpl<APFloat>(attr, nrows, ncols);
  }
  return {false, false};
}

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

ChangeResult StructuredMatrixLattice::meet(const AbstractSparseLattice &rhs) {
  const auto *rhsStruct =
      reinterpret_cast<const StructuredMatrixLattice *>(&rhs);
  return meet(*rhsStruct);
}

ChangeResult StructuredMatrixLattice::meet(StructuredMatrixLattice rhs) {
  auto newValue = StructuredMatrixType::meet(getValue(), rhs.getValue());
  if (getValue() == newValue)
    return ChangeResult::NoChange;

  setValue(newValue);
  return ChangeResult::Change;
}

ChangeResult StructuredMatrixLattice::join(const AbstractSparseLattice &rhs) {
  const auto *rhsStruct =
      reinterpret_cast<const StructuredMatrixLattice *>(&rhs);
  return join(*rhsStruct);
}

ChangeResult StructuredMatrixLattice::join(StructuredMatrixLattice rhs) {
  auto newValue = StructuredMatrixType::join(getValue(), rhs.getValue());
  if (getValue() == newValue)
    return ChangeResult::NoChange;

  setValue(newValue);
  return ChangeResult::Change;
}

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
  for (auto operand : operands) {
    llvm::errs() << "    operand: ";
    operand->getValue().print(llvm::errs());
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";

  return success();
}

} // namespace structure_analysis
} // namespace mlir
