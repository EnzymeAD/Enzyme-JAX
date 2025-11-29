#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace enzyme {

// Represents the partial symmetry of a tensor as a partition of its dimensions.
class PartialSymmetryAnnotation {
public:
  PartialSymmetryAnnotation() : dimensionSetIDs() {}

  explicit PartialSymmetryAnnotation(ArrayRef<int64_t> dimensionSetIDs);

  static PartialSymmetryAnnotation createUninitialized(int64_t rank);
  static PartialSymmetryAnnotation createNotSymmetric(int64_t rank);
  static PartialSymmetryAnnotation createFullySymmetric(int64_t rank);

  bool isSymmetric(int64_t i, int64_t j) const;

  int64_t getSetId(int64_t i) const { return dimensionSetIDs[i]; }

  int64_t getRank() const { return dimensionSetIDs.size(); }

  static PartialSymmetryAnnotation join(const PartialSymmetryAnnotation &lhs,
                                        const PartialSymmetryAnnotation &rhs);
  static PartialSymmetryAnnotation meet(const PartialSymmetryAnnotation &lhs,
                                        const PartialSymmetryAnnotation &rhs);

  static PartialSymmetryAnnotation
  propagateTranspose(const PartialSymmetryAnnotation &annotation,
                     ArrayRef<int64_t> permutation);

  static PartialSymmetryAnnotation
  propagateBroadcastInDim(const PartialSymmetryAnnotation &annotation,
                          int64_t outputRank,
                          ArrayRef<int64_t> broadcastDimensions);

  static PartialSymmetryAnnotation
  propagateDotGeneral(const PartialSymmetryAnnotation &lhsAnnotation,
                      const PartialSymmetryAnnotation &rhsAnnotation,
                      int64_t resultRank, ArrayRef<int64_t> lhsBatchingDims,
                      ArrayRef<int64_t> rhsBatchingDims,
                      ArrayRef<int64_t> lhsContractingDims,
                      ArrayRef<int64_t> rhsContractingDims, bool rhsAliasesLhs,
                      ArrayRef<int64_t> rhsDimToLhs);

  static PartialSymmetryAnnotation checkConstant(DenseElementsAttr attr);

  static PartialSymmetryAnnotation
  propagateElementwiseBinary(const PartialSymmetryAnnotation &lhsAnnotation,
                             const PartialSymmetryAnnotation &rhsAnnotation,
                             int64_t resultRank, bool rhsAliasesLhs,
                             ArrayRef<int64_t> rhsDimToLhs);

  bool operator==(const PartialSymmetryAnnotation &other) const {
    return dimensionSetIDs == other.dimensionSetIDs;
  }

  SmallVector<SmallVector<int64_t>> getDimensionSets() const;

  void print(raw_ostream &os) const;

private:
  SmallVector<int64_t> dimensionSetIDs;

  void canonicalize();
  void uniteDimensionSets(int64_t rank, int64_t i, int64_t j);
};

class PartialSymmetryLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  PartialSymmetryLattice(Value v) : AbstractSparseLattice(v) {
    if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
      value = PartialSymmetryAnnotation::createFullySymmetric(type.getRank());
    }
  }

  ChangeResult join(const AbstractSparseLattice &rhs) override;
  ChangeResult join(const PartialSymmetryLattice &rhs);

  void print(raw_ostream &os) const override;

  const PartialSymmetryAnnotation &getValue() const { return value; }
  void setValue(const PartialSymmetryAnnotation &v) { value = v; }

private:
  PartialSymmetryAnnotation value;
};

class PartialSymmetryAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<PartialSymmetryLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(PartialSymmetryLattice *lattice) override;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const PartialSymmetryLattice *> operands,
                 ArrayRef<PartialSymmetryLattice *> results) override;
};

} // namespace enzyme
} // namespace mlir
