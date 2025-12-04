#include "src/enzyme_ad/jax/Analysis/PartialSymmetryAnalysis.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir {
namespace enzyme {

//===----------------------------------------------------------------------===//
// PartialSymmetryAnnotation Implementation
//===----------------------------------------------------------------------===//

PartialSymmetryAnnotation::PartialSymmetryAnnotation(
    ArrayRef<int64_t> dimensionSetIDs) {
  this->dimensionSetIDs.assign(dimensionSetIDs.begin(), dimensionSetIDs.end());
  canonicalize();
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createFullySymmetric(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  for (int64_t i = 0; i < rank; ++i) {
    annotation.dimensionSetIDs.push_back(0);
  }
  return annotation;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createNotSymmetric(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  for (int64_t i = 0; i < rank; ++i) {
    annotation.dimensionSetIDs.push_back(i);
  }
  return annotation;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::createUninitialized(int64_t rank) {
  PartialSymmetryAnnotation annotation;
  annotation.dimensionSetIDs.resize(rank);
  return annotation;
}

bool PartialSymmetryAnnotation::isSymmetric(int64_t i, int64_t j) const {
  return dimensionSetIDs[i] == dimensionSetIDs[j];
}

void PartialSymmetryAnnotation::canonicalize() {
  llvm::SmallDenseMap<int64_t, int64_t> map;
  int64_t nextId = 0;
  for (auto &id : dimensionSetIDs) {
    if (map.find(id) == map.end()) {
      map[id] = nextId++;
    }
    id = map[id];
  }
}

void PartialSymmetryAnnotation::uniteDimensionSets(int64_t rank, int64_t i,
                                                   int64_t j) {
  if (dimensionSetIDs[i] == dimensionSetIDs[j])
    return;

  int64_t oldId = dimensionSetIDs[i];
  int64_t newId = dimensionSetIDs[j];
  for (int64_t k = 0; k < (int64_t)dimensionSetIDs.size(); ++k) {
    if (dimensionSetIDs[k] == oldId) {
      dimensionSetIDs[k] = newId;
    }
  }

  canonicalize();
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::meet(const PartialSymmetryAnnotation &lhs,
                                const PartialSymmetryAnnotation &rhs) {
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
PartialSymmetryAnnotation::join(const PartialSymmetryAnnotation &lhs,
                                const PartialSymmetryAnnotation &rhs) {
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

  PartialSymmetryAnnotation result = createUninitialized(annotation.getRank());

  for (int64_t i = 0; i < annotation.getRank(); ++i) {
    result.dimensionSetIDs[i] = annotation.getSetId(permutation[i]);
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateBroadcastInDim(
    const PartialSymmetryAnnotation &annotation, int64_t outputRank,
    ArrayRef<int64_t> broadcastDimensions) {

  PartialSymmetryAnnotation result = createUninitialized(outputRank);

  llvm::SmallDenseMap<int64_t, int64_t> outputToInput;
  for (int64_t i = 0; i < (int64_t)broadcastDimensions.size(); ++i) {
    outputToInput[broadcastDimensions[i]] = i;
  }

  int64_t maxSetId = -1;
  for (int64_t i = 0; i < annotation.getRank(); ++i) {
    maxSetId = std::max(maxSetId, (int64_t)annotation.getSetId(i));
  }

  int64_t nextSetId = maxSetId + 1;
  for (int64_t outputDim = 0; outputDim < outputRank; ++outputDim) {
    if (outputToInput.find(outputDim) != outputToInput.end()) {
      // dimension is preserved => use old ID
      int64_t inputDim = outputToInput[outputDim];
      result.dimensionSetIDs[outputDim] = annotation.getSetId(inputDim);
    } else {
      // result is constant in each broadcasted dimension,
      // so they are partially symmetric with each other
      result.dimensionSetIDs[outputDim] = nextSetId;
    }
  }

  result.canonicalize();
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::propagateElementwiseBinary(
    const PartialSymmetryAnnotation &lhsAnnotation,
    const PartialSymmetryAnnotation &rhsAnnotation, int64_t resultRank,
    bool rhsAliasesLhs, ArrayRef<int64_t> rhsDimToLhs) {

  PartialSymmetryAnnotation result = meet(lhsAnnotation, rhsAnnotation);

  if (rhsAliasesLhs) {
    int64_t changed_dim = -1;
    int changed_dims = 0;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (rhsDimToLhs[i] != i) {
        changed_dim = i;
        changed_dims++;
      }
    }
    if (changed_dims == 2) {
      result.uniteDimensionSets(resultRank, changed_dim, rhsDimToLhs[changed_dim]);
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

  PartialSymmetryAnnotation result = createNotSymmetric(resultRank);

  // Symmetry between batching dimensions
  for (int64_t i = 0; i < (int64_t)lhsBatchingDims.size(); ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (lhsAnnotation.getSetId(lhsBatchingDims[i]) ==
              lhsAnnotation.getSetId(lhsBatchingDims[j]) &&
          rhsAnnotation.getSetId(rhsBatchingDims[i]) ==
              rhsAnnotation.getSetId(rhsBatchingDims[j])) {
        result.uniteDimensionSets(resultRank, i, j);
      }
    }
  }

  // Calculate free (non-contracting, non-batching) dimensions
  SmallVector<int64_t> lhsFreeDims;
  for (int64_t i = 0; i < lhsAnnotation.getRank(); ++i) {
    if (!llvm::is_contained(lhsBatchingDims, i) &&
        !llvm::is_contained(lhsContractingDims, i)) {
      lhsFreeDims.push_back(i);
    }
  }

  SmallVector<int64_t> rhsFreeDims;
  for (int64_t i = 0; i < rhsAnnotation.getRank(); ++i) {
    if (!llvm::is_contained(rhsBatchingDims, i) &&
        !llvm::is_contained(rhsContractingDims, i)) {
      rhsFreeDims.push_back(i);
    }
  }

  // Symmetry between free dimensions from LHS
  for (int64_t i = 0; i < (int64_t)lhsFreeDims.size(); ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (lhsAnnotation.getSetId(lhsFreeDims[i]) ==
          lhsAnnotation.getSetId(lhsFreeDims[j])) {
        result.uniteDimensionSets(resultRank, lhsBatchingDims.size() + i,
                                  lhsBatchingDims.size() + j);
      }
    }
  }

  // Symmetry between free dimensions from RHS
  for (int64_t i = 0; i < (int64_t)rhsFreeDims.size(); ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (rhsAnnotation.getSetId(rhsFreeDims[i]) ==
          rhsAnnotation.getSetId(rhsFreeDims[j])) {
        result.uniteDimensionSets(
            resultRank, lhsBatchingDims.size() + lhsFreeDims.size() + i,
            lhsBatchingDims.size() + lhsFreeDims.size() + j);
      }
    }
  }

  // Symmetry between free dimensions of LHS and free dimensions of RHS
  if (rhsAliasesLhs) {
    bool exchange_valid = true;

    // check that each batching dimension has same ID for LHS and RHS
    for (int64_t i = 0; i < (int64_t)lhsBatchingDims.size(); ++i) {
      if (lhsAnnotation.getSetId(lhsBatchingDims[i]) !=
          lhsAnnotation.getSetId(rhsDimToLhs[rhsBatchingDims[i]])) {
        exchange_valid = false;
      }
    }

    // check that the multiset of IDs for contracting dimensions are equal for
    // LHS and RHS
    SmallVector<int64_t> lhsContractingIds, rhsContractingIds;
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
      for (int64_t i = 0; i < (int64_t)lhsFreeDims.size(); ++i) {
        for (int64_t j = 0; j < (int64_t)rhsFreeDims.size(); ++j) {
          if (lhsAnnotation.getSetId(lhsFreeDims[i]) ==
              lhsAnnotation.getSetId(rhsDimToLhs[rhsFreeDims[j]])) {
            result.uniteDimensionSets(resultRank, lhsBatchingDims.size() + i,
                                      lhsBatchingDims.size() +
                                          lhsFreeDims.size() + j);
          }
        }
      }
    }
  }

  result.canonicalize();
  return result;
}

template <typename Ty>
static bool checkPairwiseSymmetry(DenseElementsAttr attr, int64_t dimA,
                                  int64_t dimB) {
  auto type = cast<RankedTensorType>(attr.getType());
  auto shape = type.getShape();
  int64_t rank = type.getRank();

  if (shape[dimA] != shape[dimB])
    return false;

  if (attr.isSplat())
    return true;

  auto values = attr.getValues<Ty>();
  auto it = values.begin();

  SmallVector<int64_t> strides(rank);
  int64_t currentStride = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    strides[i] = currentStride;
    currentStride *= shape[i];
  }

  int64_t numElements = 1;
  for (int64_t s : shape)
    numElements *= s;

  for (int64_t i = 0; i < numElements; ++i) {
    SmallVector<int64_t> coords(rank);
    int64_t temp = i;
    for (int64_t d = 0; d < rank; ++d) {
      coords[d] = temp / strides[d];
      temp %= strides[d];
    }

    std::swap(coords[dimA], coords[dimB]);

    int64_t swappedIdx = 0;
    for (int64_t d = 0; d < rank; ++d) {
      swappedIdx += coords[d] * strides[d];
    }

    auto a = *(it + i);
    auto b = *(it + swappedIdx);
    if (checkNotEqual(a, b))
      return false;
  }
  return true;
}

PartialSymmetryAnnotation
PartialSymmetryAnnotation::checkConstant(DenseElementsAttr attr) {
  if (auto type = dyn_cast<RankedTensorType>(attr.getType())) {
    int64_t rank = type.getRank();
    PartialSymmetryAnnotation result = createNotSymmetric(rank);

    for (int64_t i = 0; i < rank; ++i) {
      for (int64_t j = 0; j < i; ++j) {
        bool isSymmetric = false;
        if (isa<FloatType>(attr.getElementType())) {
          isSymmetric = checkPairwiseSymmetry<APFloat>(attr, i, j);
        } else if (isa<IntegerType>(attr.getElementType())) {
          isSymmetric = checkPairwiseSymmetry<APInt>(attr, i, j);
        }

        if (isSymmetric) {
          result.uniteDimensionSets(rank, i, j);
          continue;
        }
      }
    }
    return result;
  }
  return PartialSymmetryAnnotation();
}

SmallVector<SmallVector<int64_t>>
PartialSymmetryAnnotation::getDimensionSets() const {
  llvm::SmallDenseMap<int64_t, SmallVector<int64_t>> sets;
  for (int64_t i = 0; i < (int64_t)dimensionSetIDs.size(); ++i) {
    sets[dimensionSetIDs[i]].push_back(i);
  }

  SmallVector<int64_t> sortedKeys;
  for (auto &kv : sets)
    sortedKeys.push_back(kv.first);
  std::sort(sortedKeys.begin(), sortedKeys.end(),
            [&](int64_t a, int64_t b) { return sets[a][0] < sets[b][0]; });

  SmallVector<SmallVector<int64_t>> result;
  for (int64_t key : sortedKeys) {
    result.push_back(sets[key]);
  }
  return result;
}

PartialSymmetryAnnotation PartialSymmetryAnnotation::createFromDimensionSets(
    int64_t rank, ArrayRef<ArrayRef<int64_t>> dimensionSets) {
  SmallVector<int64_t> dimensionSetIDs(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dimensionSetIDs[i] = i;
  }

  // Note that dimensionSets is not assumed to be a complete partition.
  // Missing dimensions are treated as separate sets.
  for (auto dims : dimensionSets) {
    for (int64_t i = 1; i < (int64_t)dims.size(); ++i) {
      dimensionSetIDs[dims[i]] = dimensionSetIDs[dims[0]];
    }
  }

  return PartialSymmetryAnnotation(dimensionSetIDs);
}

std::optional<PartialSymmetryAnnotation>
PartialSymmetryAnnotation::createFromIR(Value val) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (blockArg) {
    auto op = blockArg.getOwner()->getParentOp();
    auto funcOpInterface = dyn_cast<FunctionOpInterface>(op);
    if (!funcOpInterface) {
      return std::nullopt;
    }

    auto argAttrs = funcOpInterface.getArgAttrs(blockArg.getArgNumber());
    for (auto attr : argAttrs) {
      if (attr.getName() == "enzymexla.partial_symmetry") {
        auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
        if (!arrayAttr || arrayAttr.empty()) {
          continue;
        }

        auto partialSymmetryAttr =
            dyn_cast<enzymexla::PartialSymmetryAnalysisResultAttr>(
                arrayAttr[0]);

        if (!partialSymmetryAttr) {
          continue;
        }

        auto dimensionSetAttrs = partialSymmetryAttr.getValues();
        auto rank = cast<RankedTensorType>(val.getType()).getRank();

        SmallVector<ArrayRef<int64_t>> dimensionSets;
        for (auto dimensionSetAttr : dimensionSetAttrs) {
          auto dims = dimensionSetAttr.getDimensions().asArrayRef();
          dimensionSets.push_back(dims);
        }

        return PartialSymmetryAnnotation::createFromDimensionSets(
            rank, dimensionSets);
      }
    }
    return std::nullopt;
  }

  auto op = val.getDefiningOp();
  if (!op)
    return std::nullopt;

  auto arrayAttr = op->getAttrOfType<ArrayAttr>("enzymexla.partial_symmetry");
  if (!arrayAttr || arrayAttr.empty())
    return std::nullopt;

  auto opResult = dyn_cast<OpResult>(val);
  if (!opResult)
    return std::nullopt;

  auto resultNumber = opResult.getResultNumber();

  auto partialSymmetryAttr =
      dyn_cast<enzymexla::PartialSymmetryAnalysisResultAttr>(
          arrayAttr[resultNumber]);
  if (!partialSymmetryAttr)
    return std::nullopt;

  auto dimensionSetAttrs = partialSymmetryAttr.getValues();
  auto rank = cast<RankedTensorType>(val.getType()).getRank();

  SmallVector<ArrayRef<int64_t>> dimensionSets;
  for (auto dimensionSetAttr : dimensionSetAttrs) {
    auto dims = dimensionSetAttr.getDimensions().asArrayRef();
    dimensionSets.push_back(dims);
  }

  return PartialSymmetryAnnotation::createFromDimensionSets(rank,
                                                            dimensionSets);
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

PartialSymmetryLattice::PartialSymmetryLattice(Value v)
    : AbstractSparseLattice(v) {
  if (auto type = dyn_cast<RankedTensorType>(v.getType())) {
    // Trust existing IR annotations if present.
    auto annotation = PartialSymmetryAnnotation::createFromIR(v);
    if (annotation.has_value()) {
      value = annotation.value();
      return;
    }

    value = PartialSymmetryAnnotation::createFullySymmetric(type.getRank());
  }
}

ChangeResult PartialSymmetryLattice::meet(const AbstractSparseLattice &rhs) {
  const auto *rhsStruct =
      reinterpret_cast<const PartialSymmetryLattice *>(&rhs);
  return meet(*rhsStruct);
}

ChangeResult PartialSymmetryLattice::meet(const PartialSymmetryLattice &rhs) {
  auto newValue = PartialSymmetryAnnotation::meet(getValue(), rhs.getValue());
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
  auto annotation =
      PartialSymmetryAnnotation::createFromIR(lattice->getAnchor());
  if (annotation.has_value()) {
    lattice->setValue(annotation.value());
    return;
  }

  lattice->setValue(PartialSymmetryAnnotation::createNotSymmetric(
      lattice->getValue().getRank()));
}

LogicalResult PartialSymmetryAnalysis::visitOperation(
    Operation *op, ArrayRef<const PartialSymmetryLattice *> operands,
    ArrayRef<PartialSymmetryLattice *> results) {

  SmallVector<bool> updatedAnnotation(results.size(), false);
  SmallVector<PartialSymmetryAnnotation> propagatedAnnotation(results.size());

  SmallVector<PartialSymmetryAnnotation> operandAnnotations(operands.size());
  for (int64_t i = 0; i < (int64_t)operands.size(); i++) {
    operandAnnotations[i] = operands[i]->getValue();
  }

  if (auto transposeOp = dyn_cast<stablehlo::TransposeOp>(op)) {
    updatedAnnotation[0] = true;
    propagatedAnnotation[0] = PartialSymmetryAnnotation::propagateTranspose(
        operandAnnotations[0], transposeOp.getPermutation());
  }

  if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    if (auto resultType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      updatedAnnotation[0] = true;
      propagatedAnnotation[0] =
          PartialSymmetryAnnotation::propagateBroadcastInDim(
              operandAnnotations[0], resultType.getRank(),
              bcastOp.getBroadcastDimensions());
    }
  }

  if (auto dotGeneralOp = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    if (auto resultType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      auto dotDimNumbers = dotGeneralOp.getDotDimensionNumbers();
      auto lhs = dotGeneralOp.getLhs();
      auto rhs = dotGeneralOp.getRhs();

      // Check for aliasing between LHS and RHS (up to transpose)
      bool rhsAliasesLhs = false;
      SmallVector<int64_t> rhsDimToLhs;
      if (lhs == rhs) {
        auto lhsType = cast<RankedTensorType>(lhs.getType());
        rhsDimToLhs.resize(lhsType.getRank());
        for (int64_t i = 0; i < lhsType.getRank(); ++i)
          rhsDimToLhs[i] = i;  // Identity mapping
        rhsAliasesLhs = true;
      } else if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>();
                 lhsT && rhs == lhsT.getOperand()) {
        rhsDimToLhs.assign(lhsT.getPermutation().begin(),
                           lhsT.getPermutation().end());
        rhsAliasesLhs = true;
      } else if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>();
                 rhsT && lhs == rhsT.getOperand()) {
        rhsDimToLhs.resize(rhsT.getPermutation().size());
        for (int64_t i = 0; i < (int64_t)rhsT.getPermutation().size(); ++i)
          rhsDimToLhs[rhsT.getPermutation()[i]] = i;
        rhsAliasesLhs = true;
      }

      // Propagate symmetry through dotGeneral
      propagatedAnnotation[0] = PartialSymmetryAnnotation::propagateDotGeneral(
          operandAnnotations[0], operandAnnotations[1], resultType.getRank(),
          dotDimNumbers.getLhsBatchingDimensions(),
          dotDimNumbers.getRhsBatchingDimensions(),
          dotDimNumbers.getLhsContractingDimensions(),
          dotDimNumbers.getRhsContractingDimensions(), rhsAliasesLhs,
          rhsDimToLhs);

      updatedAnnotation[0] = true;
    }
  }

  if (stablehlo::hasTraitElementwise(op)) {
    if (auto resultType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      if (operands.size() == 1) {
        propagatedAnnotation[0] = operandAnnotations[0];
        updatedAnnotation[0] = true;
      } else if (operands.size() == 2 &&
                 (op->hasTrait<OpTrait::IsCommutative>() ||
                  op->hasTrait<hlo::OpTrait::IsCommutative>())) {
        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);

        // Check for aliasing between LHS and RHS (up to transpose)
        bool rhsAliasesLhs = false;
        SmallVector<int64_t> rhsDimToLhs;
        if (lhs == rhs) {
          auto lhsType = cast<RankedTensorType>(lhs.getType());
          rhsDimToLhs.resize(lhsType.getRank());
          for (int64_t i = 0; i < lhsType.getRank(); ++i)
            rhsDimToLhs[i] = i;  // Identity mapping
          rhsAliasesLhs = true;
        } else if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>();
                  lhsT && rhs == lhsT.getOperand()) {
          rhsDimToLhs.assign(lhsT.getPermutation().begin(),
                            lhsT.getPermutation().end());
          rhsAliasesLhs = true;
        } else if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>();
                  rhsT && lhs == rhsT.getOperand()) {
          rhsDimToLhs.resize(rhsT.getPermutation().size());
          for (int64_t i = 0; i < (int64_t)rhsT.getPermutation().size(); ++i)
            rhsDimToLhs[rhsT.getPermutation()[i]] = i;
          rhsAliasesLhs = true;
        }

        propagatedAnnotation[0] =
            PartialSymmetryAnnotation::propagateElementwiseBinary(
                operandAnnotations[0], operandAnnotations[1],
                resultType.getRank(), rhsAliasesLhs, rhsDimToLhs);
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

  for (int64_t i = 0; i < (int64_t)results.size(); i++) {
    if (updatedAnnotation[i]) {
      auto resultOrig = results[i]->getValue();
      auto resultNew =
          PartialSymmetryAnnotation::meet(resultOrig, propagatedAnnotation[i]);
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
