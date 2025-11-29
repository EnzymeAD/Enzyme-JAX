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

void PartialSymmetryAnnotation::uniteDimensionSets(int64_t rank, int i, int j) {
  if (isUnknown()) {
    *this = createNotSymmetric(rank);
  }
  
  if (storage[i] == storage[j])
    return;
  
  int oldId = storage[i];
  int newId = storage[j];
  for (size_t k = 0; k < storage.size(); ++k) {
    if (storage[k] == oldId) {
      storage[k] = newId;
    }
  }
  
  canonicalize();
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::join(const PartialSymmetryAnnotation &lhs,
                                const PartialSymmetryAnnotation &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result = createNotSymmetric(lhs.getRank());

  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    bool found = false;
    for (int64_t j = 0; j < i; ++j) {
      if (lhs.getSetId(i) == lhs.getSetId(j) &&
          rhs.getSetId(i) == rhs.getSetId(j)) {
        result.uniteDimensionSets(lhs.getRank(), i, j);
      }
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

  PartialSymmetryAnnotation result = createNotSymmetric(lhs.getRank());

  for (int64_t i = 0; i < lhs.getRank(); ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (lhs.getSetId(i) == lhs.getSetId(j) ||
          rhs.getSetId(i) == rhs.getSetId(j)) {
        result.uniteDimensionSets(lhs.getRank(), i, j);
      }
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

  PartialSymmetryAnnotation result = createKnownUninitialized(annotation.getRank());

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
PartialSymmetryAnnotation::propagateElementwiseBinary(
    const PartialSymmetryAnnotation &lhsAnnotation,
    const PartialSymmetryAnnotation &rhsAnnotation,
    int64_t resultRank,
    bool rhsAliasesLhs,
    ArrayRef<int64_t> rhsDimToLhs) {
  
  PartialSymmetryAnnotation result = join(lhsAnnotation, rhsAnnotation);
  
  if (rhsAliasesLhs) {
    int64_t rank = resultRank;
    
    PartialSymmetryAnnotation transposeSymmetry = createKnownUninitialized(rank);
    
    for (int64_t i = 0; i < rank; ++i) {
      int64_t j = rhsDimToLhs[i];
      if (rhsDimToLhs[j] == i) {
        result.uniteDimensionSets(rank, i, j);
      }       
    }
    
    result.canonicalize();
  }
  
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateDotGeneral(
    const PartialSymmetryAnnotation &lhsAnnotation,
    const PartialSymmetryAnnotation &rhsAnnotation, int64_t resultRank,
    ArrayRef<int64_t> lhsBatchingDims, ArrayRef<int64_t> rhsBatchingDims,
    ArrayRef<int64_t> lhsContractingDims, ArrayRef<int64_t> rhsContractingDims,
    bool rhsAliasesLhs, ArrayRef<int64_t> rhsDimToLhs) {

  if (lhsAnnotation.isUnknown() || rhsAnnotation.isUnknown())
    return PartialSymmetryAnnotation();

  PartialSymmetryAnnotation result = createNotSymmetric(resultRank);

  // Preserve symmetry in batching dimensions
  for (int i = 0; i < lhsBatchingDims.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      if (lhsAnnotation.getSetId(lhsBatchingDims[i]) ==
              lhsAnnotation.getSetId(lhsBatchingDims[j]) &&
          rhsAnnotation.getSetId(rhsBatchingDims[i]) ==
              rhsAnnotation.getSetId(rhsBatchingDims[j])) {
        result.uniteDimensionSets(resultRank, i, j);
      }
    }
  }

  // Preserve symmetry in free (non-contracting, non-batching) dimensions
  if (rhsAliasesLhs) {

    bool exchange_valid = true;
    
    // check that each batching dimension has same ID for LHS and RHS
    for (int i = 0; i < lhsBatchingDims.size(); ++i) {
      if (lhsAnnotation.getSetId(lhsBatchingDims[i]) != lhsAnnotation.getSetId(rhsDimToLhs[rhsBatchingDims[i]])) {
        exchange_valid = false;
      }
    }
    
    // check that the multiset of IDs for contracting dimensions are equal for LHS and RHS
    SmallVector<int> lhsContractingIds, rhsContractingIds;
    for (int64_t dim : lhsContractingDims) {
      lhsContractingIds.push_back(lhsAnnotation.getSetId(dim));
    }
    for (int64_t dim : rhsContractingDims) {
      rhsContractingIds.push_back(lhsAnnotation.getSetId(rhsDimToLhs[dim]));
    }
    llvm::sort(lhsContractingIds);
    llvm::sort(rhsContractingIds);
    if (lhsContractingIds != rhsContractingIds) {
      exchange_valid = false;
    }
        
    if (exchange_valid) {
      SmallVector<int64_t> lhsResultDims;
      for (int64_t i = 0; i < lhsAnnotation.getRank(); ++i) {
        if (!llvm::is_contained(lhsBatchingDims, i) && !llvm::is_contained(lhsContractingDims, i)) {
          lhsResultDims.push_back(i);
        }
      }
      
      SmallVector<int64_t> rhsResultDims;
      for (int64_t i = 0; i < rhsAnnotation.getRank(); ++i) {
        if (!llvm::is_contained(rhsBatchingDims, i) && !llvm::is_contained(rhsContractingDims, i)) {
          rhsResultDims.push_back(i);
        }
      }

      // Symmetry within free dimensions of LHS
      for (int i = 0; i < lhsResultDims.size(); ++i) {
        for (int j = 0; j < i; ++j) {
          if (lhsAnnotation.getSetId(lhsResultDims[i]) == lhsAnnotation.getSetId(lhsResultDims[j])) {
            result.uniteDimensionSets(resultRank, lhsBatchingDims.size() + i, lhsBatchingDims.size() + j);
          }
        }
      }
      
      // Symmetry between free dimensions of RHS
      for (int i = 0; i < rhsResultDims.size(); ++i) {
        for (int j = 0; j < i; ++j) {
          if (rhsAnnotation.getSetId(rhsResultDims[i]) == rhsAnnotation.getSetId(rhsResultDims[j])) {
            result.uniteDimensionSets(resultRank, lhsBatchingDims.size() + lhsResultDims.size() + i, lhsBatchingDims.size() + lhsResultDims.size() + j);
          }
        }
      }
      
      // Symmetry between free dimensions of LHS and RHS
      for (int i = 0; i < lhsResultDims.size(); ++i) {
        for (int j = 0; j < rhsResultDims.size(); ++j) {
          if (lhsAnnotation.getSetId(lhsResultDims[i]) == lhsAnnotation.getSetId(rhsDimToLhs[rhsResultDims[j]])) {
            result.uniteDimensionSets(resultRank, lhsBatchingDims.size() + i, lhsBatchingDims.size() + lhsResultDims.size() + j);
          }
        }
      }
    }
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
    PartialSymmetryAnnotation result = createNotSymmetric(rank);
    
    for (int i = 0; i < rank; ++i) {
      for (int j = i + 1; j < rank; ++j) {
        if (result.getSetId(i) == result.getSetId(j))
          continue;

        if (checkPairwiseSymmetry(attr, i, j)) {
          result.uniteDimensionSets(rank, i, j);
        }
      }
    }
    return result;
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
    if (auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      updatedAnnotation[0] = true;
      propagatedAnnotation[0] =
          PartialSymmetryAnnotation::propagateBroadcastInDim(
              operandAnnotations[0], resultType.getRank(),
              bcastOp.getBroadcastDimensions());
    }
  }

  if (auto dotGeneralOp = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    if (auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      auto dotDimNumbers = dotGeneralOp.getDotDimensionNumbers();
      auto lhs = dotGeneralOp.getLhs();
      auto rhs = dotGeneralOp.getRhs();

      // Check for aliasing between LHS and RHS (up to transpose)
      bool rhsAliasesLhs = false;
      SmallVector<int64_t> rhsDimToLhs;
      if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
        if (rhs == lhsT.getOperand()) {
          rhsDimToLhs.assign(lhsT.getPermutation().begin(), lhsT.getPermutation().end());
          rhsAliasesLhs = true;
        }
      }
      if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
        if (lhs == rhsT.getOperand()) {
          rhsDimToLhs.resize(rhsT.getPermutation().size());
          for (size_t i = 0; i < rhsT.getPermutation().size(); ++i)
            rhsDimToLhs[rhsT.getPermutation()[i]] = i;
          rhsAliasesLhs = true;
        }
      }

      // Propagate symmetry through dotGeneral
      propagatedAnnotation[0] =
          PartialSymmetryAnnotation::propagateDotGeneral(
              operandAnnotations[0], operandAnnotations[1],
              resultType.getRank(), dotDimNumbers.getLhsBatchingDimensions(),
              dotDimNumbers.getRhsBatchingDimensions(),
              dotDimNumbers.getLhsContractingDimensions(),
              dotDimNumbers.getRhsContractingDimensions(), rhsAliasesLhs, rhsDimToLhs);

      updatedAnnotation[0] = true;
    }
  }

  if (stablehlo::hasTraitElementwise(op)) {
    if (auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      if (operands.size() == 1) {
        propagatedAnnotation[0] = operandAnnotations[0];
        updatedAnnotation[0] = true;
      } else if (operands.size() == 2 &&
                  (op->hasTrait<OpTrait::IsCommutative>() ||
                  op->hasTrait<hlo::OpTrait::IsCommutative>())) {
        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);
        
        bool rhsAliasesLhs = false;
        SmallVector<int64_t> rhsDimToLhs;
        
        if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (rhs == lhsT.getOperand()) {
            rhsDimToLhs.assign(lhsT.getPermutation().begin(), lhsT.getPermutation().end());
            rhsAliasesLhs = true;
          }
        }
        if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
          if (lhs == rhsT.getOperand()) {
            rhsDimToLhs.resize(rhsT.getPermutation().size());
            for (size_t i = 0; i < rhsT.getPermutation().size(); ++i)
              rhsDimToLhs[rhsT.getPermutation()[i]] = i;
            rhsAliasesLhs = true;
          }
        }
        
        llvm::errs() << "handling elementwise op" << "\n";
        
        propagatedAnnotation[0] = PartialSymmetryAnnotation::propagateElementwiseBinary(
            operandAnnotations[0], operandAnnotations[1], resultType.getRank(), rhsAliasesLhs, rhsDimToLhs);
        updatedAnnotation[0] = true;
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
