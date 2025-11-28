#include "src/enzyme_ad/jax/Analysis/PartialSymmetryAnalysis.h"
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
// PartialSymmetryAnnotation Implementation
//===----------------------------------------------------------------------===//

PartialSymmetryAnnotation::PartialSymmetryAnnotation(ArrayRef<int> s)
    : known(true) {
  storage.assign(s.begin(), s.end());
  canonicalize();
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createFullySymmetric(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  annotation.known = true;
  for (int64_t i = 0; i < rank; ++i) {
    annotation.storage.push_back(0);
  }
  return annotation;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createNotSymmetric(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  annotation.known = true;
  for (int64_t i = 0; i < rank; ++i) {
    annotation.storage.push_back(i);
  }
  return annotation;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createKnownUninitialized(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  annotation.known = true;
  annotation.storage.resize(rank);
  return annotation;
}

bool PartialSymmetryAnnotation::isSymmetric(int64_t i, int64_t j) const {
  if (i < 0 || i >= (int64_t)storage.size() || j < 0 ||
      j >= (int64_t)storage.size())
    return false;
  return storage[i] == storage[j];
}

void PartialSymmetryAnnotation::canonicalize() {
  llvm::SmallDenseMap<int, int> map;
  int nextId = 0;
  for (auto &id : storage) {
    if (map.find(id) == map.end()) {
      map[id] = nextId++;
    }
    id = map[id];
  }
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::join(const PartialSymmetryAnnotation &lhs,
                                const PartialSymmetryAnnotation &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result = createKnownUninitialized(lhs.getRank());
  int nextId = 0;

  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    bool found = false;
    for (int64_t j = 0; j < i; ++j) {
      if (lhs.getSetId(i) == lhs.getSetId(j) &&
          rhs.getSetId(i) == rhs.getSetId(j)) {
        result.storage[i] = result.storage[j];
        found = true;
        break;
      }
    }
    if (!found) {
      result.storage[i] = nextId++;
    }
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::meet(const PartialSymmetryAnnotation &lhs,
                                const PartialSymmetryAnnotation &rhs) {
  if (lhs.isUnknown())
    return rhs;
  if (rhs.isUnknown())
    return lhs;

  PartialSymmetryAnnotation result = createKnownUninitialized(lhs.getRank());
  int nextId = 0;

  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    bool found = false;
    for (int64_t j = 0; j < i; ++j) {
      if (lhs.getSetId(i) == lhs.getSetId(j) ||
          rhs.getSetId(i) == rhs.getSetId(j)) {
        result.storage[i] = result.storage[j];
        found = true;
        break;
      }
    }
    if (!found) {
      result.storage[i] = nextId++;
    }
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateTranspose(
    const PartialSymmetryAnnotation &annotation,
    ArrayRef<int64_t> permutation) {
  if (annotation.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result =
      createKnownUninitialized(annotation.getRank());

  for (int64_t i = 0; i < annotation.getRank(); ++i) {
    result.storage[i] = annotation.getSetId(permutation[i]);
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateBroadcastInDim(
    const PartialSymmetryAnnotation &annotation, int64_t outputRank,
    ArrayRef<int64_t> broadcastDimensions) {

  if (annotation.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result = createKnownUninitialized(outputRank);

  llvm::SmallDenseMap<int64_t, int64_t> outputToInput;
  for (size_t i = 0; i < broadcastDimensions.size(); ++i) {
    outputToInput[broadcastDimensions[i]] = i;
  }

  int maxSetId = -1;
  for (int64_t i = 0; i < annotation.getRank(); ++i) {
    maxSetId = std::max(maxSetId, annotation.getSetId(i));
  }

  int nextNewSetId = maxSetId + 1;
  for (int64_t outputDim = 0; outputDim < outputRank; ++outputDim) {
    if (outputToInput.find(outputDim) != outputToInput.end()) {
      // dimension is preserved => use old ID
      int64_t inputDim = outputToInput[outputDim];
      result.storage[outputDim] = annotation.getSetId(inputDim);
    } else {
      // broadcasted dimension => new ID
      result.storage[outputDim] = nextNewSetId++;
    }
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::generateSymmetryFromBilinearTranspose(
    const PartialSymmetryAnnotation &annotation,
    ArrayRef<int64_t> permutation) {
  int64_t rank = permutation.size();

  // Each pair (i, j) where perm[i] = j and perm[j] = i is symmetric
  PartialSymmetryAnnotation transposeSymmetry = createKnownUninitialized(rank);
  SmallVector<bool> assigned(rank, false);
  int nextId = 0;

  for (int64_t i = 0; i < rank; ++i) {
    if (assigned[i])
      continue;

    int64_t j = permutation[i];
    if (j != i && permutation[j] == i) {
      // i and j are swapped, so assign them the same ID
      transposeSymmetry.storage[i] = nextId;
      transposeSymmetry.storage[j] = nextId;
      assigned[i] = true;
      assigned[j] = true;
    } else {
      // dimension i is not swapped, so assign it a new ID
      transposeSymmetry.storage[i] = nextId;
      assigned[i] = true;
    }
    nextId++;
  }

  transposeSymmetry.canonicalize();

  // Meet the existing annotation with the transpose symmetry
  return meet(annotation, transposeSymmetry);
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateDotGeneral(
    const PartialSymmetryAnnotation &lhsAnnotation,
    const PartialSymmetryAnnotation &rhsAnnotation, int64_t resultRank,
    ArrayRef<int64_t> lhsBatchingDims, ArrayRef<int64_t> rhsBatchingDims,
    ArrayRef<int64_t> lhsContractingDims, ArrayRef<int64_t> rhsContractingDims,
    bool lhsEqualsRhs) {

  if (lhsAnnotation.isUnknown() || rhsAnnotation.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result = createNotSymmetric(resultRank);

  for (int i = 0; i < lhsBatchingDims.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      if (lhsAnnotation.getSetId(lhsBatchingDims[i]) ==
              lhsAnnotation.getSetId(lhsBatchingDims[j]) &&
          rhsAnnotation.getSetId(rhsBatchingDims[i]) ==
              rhsAnnotation.getSetId(rhsBatchingDims[j])) {
        result.storage[i] = result.storage[j];
      }
    }
  }

  if (lhsEqualsRhs && lhsBatchingDims == rhsBatchingDims &&
      lhsContractingDims == rhsContractingDims) {
    // Also preserve symmetry in non-contracting, non-batching dimensions
    // TODO
  }

  result.canonicalize();
  return result;
}

static bool checkPairwiseSymmetry(DenseElementsAttr attr, int64_t dimA,
                                  int64_t dimB) {
  auto type = cast<RankedTensorType>(attr.getType());
  auto shape = type.getShape();
  int64_t rank = type.getRank();

  if (shape[dimA] != shape[dimB])
    return false;

  int64_t numElements = type.getNumElements();

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

PartialSymmetryAnnotation
PartialSymmetryAnnotation::checkConstant(DenseElementsAttr attr) {
  if (auto type = dyn_cast<RankedTensorType>(attr.getType())) {
    int64_t rank = type.getRank();
    SmallVector<int> storage(rank);
    for (int i = 0; i < rank; ++i)
      storage[i] = i;

    for (int i = 0; i < rank; ++i) {
      for (int j = i + 1; j < rank; ++j) {
        if (storage[i] == storage[j])
          continue;

        if (checkPairwiseSymmetry(attr, i, j)) {
          int oldId = storage[j];
          int newId = storage[i];
          for (int k = 0; k < rank; ++k) {
            if (storage[k] == oldId)
              storage[k] = newId;
          }
        }
      }
    }
    return PartialSymmetryAnnotation(storage);
  }
  return PartialSymmetryAnnotation();
}

SmallVector<SmallVector<int64_t>>
PartialSymmetryAnnotation::getDimensionSets() const {
  llvm::SmallDenseMap<int, SmallVector<int64_t>> sets;
  for (int64_t i = 0; i < (int64_t)storage.size(); ++i) {
    sets[storage[i]].push_back(i);
  }

  SmallVector<int> sortedKeys;
  for (auto &kv : sets)
    sortedKeys.push_back(kv.first);
  std::sort(sortedKeys.begin(), sortedKeys.end(),
            [&](int a, int b) { return sets[a][0] < sets[b][0]; });

  SmallVector<SmallVector<int64_t>> result;
  for (int key : sortedKeys) {
    result.push_back(sets[key]);
  }
  return result;
}

void PartialSymmetryAnnotation::print(raw_ostream &os) const {
  auto dimensionSets = getDimensionSets();
  os << "{";
  bool firstSet = true;
  for (const auto &set : dimensionSets) {
    if (!firstSet)
      os << ", ";
    os << "{";
    bool firstElem = true;
    for (int64_t dim : set) {
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
// PartialSymmetryLattice Implementation
//===----------------------------------------------------------------------===//

ChangeResult PartialSymmetryLattice::join(const AbstractSparseLattice &rhs) {
  const auto *rhsStruct =
      reinterpret_cast<const PartialSymmetryLattice *>(&rhs);
  return join(*rhsStruct);
}

ChangeResult PartialSymmetryLattice::join(const PartialSymmetryLattice &rhs) {
  auto newValue = PartialSymmetryAnnotation::join(getValue(), rhs.getValue());
  if (getValue() == newValue)
    return ChangeResult::NoChange;

  setValue(newValue);
  return ChangeResult::Change;
}

void PartialSymmetryLattice::print(raw_ostream &os) const { value.print(os); }

//===----------------------------------------------------------------------===//
// PartialSymmetryAnalysis Implementation
//===----------------------------------------------------------------------===//

void PartialSymmetryAnalysis::setToEntryState(PartialSymmetryLattice *lattice) {
  lattice->setValue(PartialSymmetryAnnotation());
}

LogicalResult PartialSymmetryAnalysis::visitOperation(
    Operation *op, ArrayRef<const PartialSymmetryLattice *> operands,
    ArrayRef<PartialSymmetryLattice *> results) {

  SmallVector<bool> updatedAnnotation(results.size(), false);
  SmallVector<PartialSymmetryAnnotation> propagatedAnnotation(results.size());

  SmallVector<PartialSymmetryAnnotation> operandAnnotations(operands.size());
  for (size_t i = 0; i < operands.size(); i++) {
    operandAnnotations[i] = operands[i]->getValue();
  }

  if (auto transposeOp = dyn_cast<stablehlo::TransposeOp>(op)) {
    updatedAnnotation[0] = true;
    propagatedAnnotation[0] = PartialSymmetryAnnotation::propagateTranspose(
        operandAnnotations[0], transposeOp.getPermutation());
  }

  if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    if (results.size() > 0) {
      if (auto resultType =
              dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        updatedAnnotation[0] = true;
        propagatedAnnotation[0] =
            PartialSymmetryAnnotation::propagateBroadcastInDim(
                operandAnnotations[0], resultType.getRank(),
                bcastOp.getBroadcastDimensions());
      }
    }
  }

  if (auto dotGeneralOp = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    if (results.size() > 0) {
      if (auto resultType =
              dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        auto dotDimNumbers = dotGeneralOp.getDotDimensionNumbers();
        auto lhs = dotGeneralOp.getLhs();
        auto rhs = dotGeneralOp.getRhs();
        bool lhsEqualsRhs = (lhs == rhs);

        // Check for transpose pattern: A x A^T or A^T x A
        bool transposePatternDetected = false;
        ArrayRef<int64_t> transposePermutation;

        if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (rhs == lhsT.getOperand()) {
            transposePatternDetected = true;
            transposePermutation = lhsT.getPermutation();
          }
        }
        if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (lhs == rhsT.getOperand()) {
            transposePatternDetected = true;
            transposePermutation = rhsT.getPermutation();
          }
        }

        // Propagate symmetry through dotGeneral
        propagatedAnnotation[0] =
            PartialSymmetryAnnotation::propagateDotGeneral(
                operandAnnotations[0], operandAnnotations[1],
                resultType.getRank(), dotDimNumbers.getLhsBatchingDimensions(),
                dotDimNumbers.getRhsBatchingDimensions(),
                dotDimNumbers.getLhsContractingDimensions(),
                dotDimNumbers.getRhsContractingDimensions(), lhsEqualsRhs);

        // If transpose pattern detected, add symmetry from transpose
        if (transposePatternDetected) {
          // TODO
        }

        updatedAnnotation[0] = true;
      }
    }
  }

  if (stablehlo::hasTraitElementwise(op)) {
    if (results.size() == 1 && operands.size() > 0) {
      propagatedAnnotation[0] = operandAnnotations[0];
      for (size_t i = 1; i < operands.size(); ++i) {
        propagatedAnnotation[0] = PartialSymmetryAnnotation::join(
            propagatedAnnotation[0], operandAnnotations[i]);
      }
      updatedAnnotation[0] = true;

      // Generate symmetry from commutative operation with transpose argument
      if (op->hasTrait<OpTrait::IsCommutative>() ||
          op->hasTrait<hlo::OpTrait::IsCommutative>() &&
              op->getNumOperands() == 2) {
        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);

        bool transposePatternDetected = false;
        ArrayRef<int64_t> transposePermutation;

        if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (rhs == lhsT.getOperand()) {
            transposePatternDetected = true;
            transposePermutation = lhsT.getPermutation();
          }
        }
        if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (lhs == rhsT.getOperand()) {
            transposePatternDetected = true;
            transposePermutation = rhsT.getPermutation();
          }
        }

        if (transposePatternDetected) {
          propagatedAnnotation[0] =
              PartialSymmetryAnnotation::generateSymmetryFromBilinearTranspose(
                  propagatedAnnotation[0], transposePermutation);
        }
      }
    }
  }

  DenseElementsAttr denseAttr;
  if (matchPattern(op->getResult(0), m_Constant(&denseAttr))) {
    updatedAnnotation[0] = true;
    propagatedAnnotation[0] =
        PartialSymmetryAnnotation::checkConstant(denseAttr);
  }

  for (size_t i = 0; i < results.size(); i++) {
    if (updatedAnnotation[i]) {
      auto resultOrig = results[i]->getValue();
      auto resultNew =
          PartialSymmetryAnnotation::join(resultOrig, propagatedAnnotation[i]);
      results[i]->setValue(resultNew);
      propagateIfChanged(results[i], resultNew == resultOrig
                                         ? ChangeResult::NoChange
                                         : ChangeResult::Change);
    }
  }

  return success();
}

} // namespace enzyme
} // namespace mlir
