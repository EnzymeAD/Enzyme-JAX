#include "src/enzyme_ad/jax/Analysis/TensorSymmetricAnalysis.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace enzyme {

//===----------------------------------------------------------------------===//
// SymmetryGroup Implementation
//===----------------------------------------------------------------------===//

SymmetryGroup::SymmetryGroup(int64_t rank) {
  for (int64_t i = 0; i < rank; ++i) {
    storage.push_back(i);
  }
}

SymmetryGroup::SymmetryGroup(ArrayRef<int> s) {
  storage.assign(s.begin(), s.end());
  canonicalize();
}

SymmetryGroup SymmetryGroup::getFullySymmetric(int64_t rank) {
  SymmetryGroup group;
  for (int64_t i = 0; i < rank; ++i) {
    group.storage.push_back(0);
  }
  return group;
}

bool SymmetryGroup::isSymmetric(int64_t i, int64_t j) const {
  if (i < 0 || i >= (int64_t)storage.size() || j < 0 ||
      j >= (int64_t)storage.size())
    return false;
  return storage[i] == storage[j];
}

void SymmetryGroup::canonicalize() {
  llvm::SmallDenseMap<int, int> map;
  int nextId = 0;
  for (auto &id : storage) {
    if (map.find(id) == map.end()) {
      map[id] = nextId++;
    }
    id = map[id];
  }
}

SymmetryGroup SymmetryGroup::meet(const SymmetryGroup &lhs,
                                  const SymmetryGroup &rhs) {
  if (lhs.getRank() != rhs.getRank()) {
    return lhs;
  }

  SymmetryGroup result;
  result.storage.resize(lhs.getRank());

  // Pair (lhs_id, rhs_id) -> new_id
  llvm::SmallDenseMap<std::pair<int, int>, int> map;
  int nextId = 0;

  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    std::pair<int, int> key = {lhs.getSetId(i), rhs.getSetId(i)};
    if (map.find(key) == map.end()) {
      map[key] = nextId++;
    }
    result.storage[i] = map[key];
  }

  result.canonicalize();
  return result;
}

SymmetryGroup SymmetryGroup::propagateTranspose(const SymmetryGroup &group,
                                                ArrayRef<int64_t> permutation) {
  if ((int64_t)permutation.size() != group.getRank())
    return group;

  SymmetryGroup result;
  result.storage.resize(group.getRank());

  for (int64_t i = 0; i < group.getRank(); ++i) {
    result.storage[i] = group.getSetId(permutation[i]);
  }

  result.canonicalize();
  return result;
}

void SymmetryGroup::print(raw_ostream &os) const {
  os << "{";
  llvm::SmallDenseMap<int, SmallVector<int>> sets;
  for (int i = 0; i < (int)storage.size(); ++i) {
    sets[storage[i]].push_back(i);
  }

  SmallVector<int> sortedKeys;
  for (auto &kv : sets)
    sortedKeys.push_back(kv.first);
  std::sort(sortedKeys.begin(), sortedKeys.end(),
            [&](int a, int b) { return sets[a][0] < sets[b][0]; });

  bool firstSet = true;
  for (int key : sortedKeys) {
    if (!firstSet)
      os << ", ";
    os << "{";
    bool firstElem = true;
    for (int dim : sets[key]) {
      if (!firstElem)
        os << ",";
      os << dim;
      firstElem = false;
    }
    os << "}";
    firstSet = false;
  }
  os << "}";
}

//===----------------------------------------------------------------------===//
// TensorSymmetricLattice Implementation
//===----------------------------------------------------------------------===//

ChangeResult TensorSymmetricLattice::meet(const AbstractSparseLattice &rhs) {
  const auto *rhsStruct =
      reinterpret_cast<const TensorSymmetricLattice *>(&rhs);
  return meet(*rhsStruct);
}

ChangeResult TensorSymmetricLattice::meet(const TensorSymmetricLattice &rhs) {
  auto newValue = SymmetryGroup::meet(getValue(), rhs.getValue());
  if (getValue() == newValue)
    return ChangeResult::NoChange;

  setValue(newValue);
  return ChangeResult::Change;
}

void TensorSymmetricLattice::print(raw_ostream &os) const { value.print(os); }

//===----------------------------------------------------------------------===//
// TensorSymmetricAnalysis Implementation
//===----------------------------------------------------------------------===//

void TensorSymmetricAnalysis::setToEntryState(TensorSymmetricLattice *lattice) {
  // Already initialized in constructor
}

static bool checkPairwiseSymmetry(DenseElementsAttr attr, int64_t dimA,
                                  int64_t dimB) {
  auto type = cast<RankedTensorType>(attr.getType());
  auto shape = type.getShape();
  int64_t rank = type.getRank();

  if (shape[dimA] != shape[dimB])
    return false;

  int64_t numElements = type.getNumElements();
  if (numElements > 10000)
    return false;

  if (auto intAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    auto values = intAttr.getValues<APInt>();
    SmallVector<int64_t> strides(rank);
    int64_t currentStride = 1;
    for (int i = rank - 1; i >= 0; --i) {
      strides[i] = currentStride;
      currentStride *= shape[i];
    }

    for (int64_t i = 0; i < numElements; ++i) {
      SmallVector<int64_t, 4> coords(rank);
      int64_t temp = i;
      for (int d = 0; d < rank; ++d) {
        coords[d] = temp / strides[d];
        temp %= strides[d];
      }

      std::swap(coords[dimA], coords[dimB]);

      int64_t swappedIdx = 0;
      for (int d = 0; d < rank; ++d) {
        swappedIdx += coords[d] * strides[d];
      }

      if (values[i] != values[swappedIdx])
        return false;
    }
    return true;
  } else if (auto floatAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
    auto values = floatAttr.getValues<APFloat>();
    SmallVector<int64_t> strides(rank);
    int64_t currentStride = 1;
    for (int i = rank - 1; i >= 0; --i) {
      strides[i] = currentStride;
      currentStride *= shape[i];
    }

    for (int64_t i = 0; i < numElements; ++i) {
      SmallVector<int64_t, 4> coords(rank);
      int64_t temp = i;
      for (int d = 0; d < rank; ++d) {
        coords[d] = temp / strides[d];
        temp %= strides[d];
      }

      std::swap(coords[dimA], coords[dimB]);

      int64_t swappedIdx = 0;
      for (int d = 0; d < rank; ++d) {
        swappedIdx += coords[d] * strides[d];
      }

      if (values[i].compare(values[swappedIdx]) != APFloat::cmpEqual)
        return false;
    }
    return true;
  }
  return false;
}

LogicalResult TensorSymmetricAnalysis::visitOperation(
    Operation *op, ArrayRef<const TensorSymmetricLattice *> operands,
    ArrayRef<TensorSymmetricLattice *> results) {

  SmallVector<SymmetryGroup> resultGroups;
  for (size_t i = 0; i < results.size(); ++i) {
    if (auto type = dyn_cast<RankedTensorType>(op->getResult(i).getType())) {
      resultGroups.push_back(SymmetryGroup(type.getRank()));
    } else {
      resultGroups.push_back(SymmetryGroup());
    }
  }

  bool handled = false;

  if (auto transposeOp = dyn_cast<stablehlo::TransposeOp>(op)) {
    if (operands.size() > 0) {
      resultGroups[0] = SymmetryGroup::propagateTranspose(
          operands[0]->getValue(), transposeOp.getPermutation());
      handled = true;
    }
  } else if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    auto outputType = cast<RankedTensorType>(bcastOp.getType());
    int64_t outRank = outputType.getRank();
    auto bcastDims = bcastOp.getBroadcastDimensions();

    SmallVector<int> newStorage(outRank);

    int maxId = -1;
    if (operands.size() > 0) {
      const auto &inputGroup = operands[0]->getValue();
      for (int64_t i = 0; i < inputGroup.getRank(); ++i) {
        maxId = std::max(maxId, inputGroup.getSetId(i));
      }
    }
    int newSetId = maxId + 1;

    if (operands.size() > 0) {
      const auto &inputGroup = operands[0]->getValue();
      for (int64_t i = 0; i < outRank; ++i) {
        bool isMapped = false;
        int mappedInputDim = -1;
        for (size_t j = 0; j < bcastDims.size(); ++j) {
          if (bcastDims[j] == i) {
            isMapped = true;
            mappedInputDim = j;
            break;
          }
        }

        if (isMapped) {
          newStorage[i] = inputGroup.getSetId(mappedInputDim);
        } else {
          newStorage[i] = newSetId;
        }
      }
    } else {
      // Scalar broadcast
      for (int64_t i = 0; i < outRank; ++i)
        newStorage[i] = 0;
    }

    resultGroups[0] = SymmetryGroup(newStorage);
    handled = true;
  } else if (stablehlo::hasTraitElementwise(op)) {
    if (results.size() == 1 && operands.size() > 0) {
      SymmetryGroup res = operands[0]->getValue();
      for (size_t i = 1; i < operands.size(); ++i) {
        res = SymmetryGroup::meet(res, operands[i]->getValue());
      }
      resultGroups[0] = res;
      handled = true;
    }
  } else if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op)) {
    auto attr = constOp.getValue();
    if (auto type = dyn_cast<RankedTensorType>(attr.getType())) {
      int64_t rank = type.getRank();
      SmallVector<int> storage(rank);
      for (int i = 0; i < rank; ++i)
        storage[i] = i;

      for (int i = 0; i < rank; ++i) {
        for (int j = i + 1; j < rank; ++j) {
          if (storage[i] == storage[j])
            continue;

          if (checkPairwiseSymmetry(cast<DenseElementsAttr>(attr), i, j)) {
            int oldId = storage[j];
            int newId = storage[i];
            for (int k = 0; k < rank; ++k) {
              if (storage[k] == oldId)
                storage[k] = newId;
            }
          }
        }
      }
      resultGroups[0] = SymmetryGroup(storage);
      handled = true;
    }
  }

  for (size_t i = 0; i < results.size(); i++) {
    if (handled) {
      auto resultOrig = results[i]->getValue();
      auto resultNew = SymmetryGroup::meet(resultOrig, resultGroups[i]);
      // Wait, meet moves down (intersection).
      // If we computed a precise result (e.g. Transpose), that IS the result.
      // But we need to join with previous state?
      // SparseForwardDataFlowAnalysis logic:
      // visitOperation computes the effect of the op.
      // We should set the result to the computed value.
      // The framework handles the join at block merges.
      // But wait, if we visit the op multiple times?
      // The lattice value should monotonically decrease (or increase depending
      // on direction). Here, we start at Uninitialized. First visit: Set to
      // Computed. Second visit: Set to Computed. If Computed changes, we
      // propagate.

      // Actually, we should just set it.
      // But wait, `meet` in lattice is used for control flow merges.
      // Inside `visitOperation`, we calculate the output based on inputs.
      // If inputs changed, output might change.
      // We just set the new value.

      // However, if we are refining, we should ensure monotonicity?
      // If inputs become LESS symmetric (more known), output becomes LESS
      // symmetric. Yes.

      if (resultOrig == resultGroups[i]) {
        propagateIfChanged(results[i], ChangeResult::NoChange);
      } else {
        results[i]->setValue(resultGroups[i]);
        propagateIfChanged(results[i], ChangeResult::Change);
      }
    }
  }

  return success();
}

} // namespace enzyme
} // namespace mlir
