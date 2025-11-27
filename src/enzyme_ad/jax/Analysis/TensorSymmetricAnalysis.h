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

// Represents the symmetry of a tensor as a partition of its dimensions.
// Dimensions in the same set are symmetric (invariant under permutation).
class SymmetryGroup {
public:
  SymmetryGroup() = default;

  // Initialize with 'rank' dimensions, each in its own set (no symmetry).
  explicit SymmetryGroup(int64_t rank);

  // Initialize with raw storage (Set IDs).
  explicit SymmetryGroup(ArrayRef<int> storage);

  // Initialize with 'rank' dimensions, all in the same set (full symmetry).
  static SymmetryGroup getFullySymmetric(int64_t rank);

  // Returns true if dimensions i and j are in the same symmetry set.
  bool isSymmetric(int64_t i, int64_t j) const;

  // Returns the Set ID for dimension i.
  int getSetId(int64_t i) const { return storage[i]; }

  int64_t getRank() const { return storage.size(); }

  // Intersection of two partitions.
  static SymmetryGroup meet(const SymmetryGroup &lhs, const SymmetryGroup &rhs);

  // Apply a permutation to the dimensions.
  // permutation[i] = old_index means the new i-th dimension comes from
  // old_index.
  static SymmetryGroup propagateTranspose(const SymmetryGroup &group,
                                          ArrayRef<int64_t> permutation);

  bool operator==(const SymmetryGroup &other) const {
    return storage == other.storage;
  }

  void print(raw_ostream &os) const;

private:
  // storage[i] is the Set ID for dimension i.
  // Canonical form: Set IDs are 0..k-1.
  SmallVector<int, 4> storage;

  void canonicalize();
};

class TensorSymmetricLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  TensorSymmetricLattice(Value v) : AbstractSparseLattice(v) {
    if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
      value = SymmetryGroup(type.getRank());
    }
  }

  ChangeResult meet(const AbstractSparseLattice &rhs) override;
  ChangeResult meet(const TensorSymmetricLattice &rhs);

  void print(raw_ostream &os) const override;

  const SymmetryGroup &getValue() const { return value; }
  void setValue(const SymmetryGroup &v) { value = v; }

private:
  SymmetryGroup value;
};

class TensorSymmetricAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TensorSymmetricLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void setToEntryState(TensorSymmetricLattice *lattice) override;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const TensorSymmetricLattice *> operands,
                 ArrayRef<TensorSymmetricLattice *> results) override;
};

} // namespace enzyme
} // namespace mlir
