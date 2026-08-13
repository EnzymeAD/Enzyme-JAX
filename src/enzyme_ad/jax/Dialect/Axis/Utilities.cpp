#include "Utilities.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "enzyme-axis-utilities"

namespace mlir::enzyme::axis {

// Refreshes one operation in-place and reports whether any result type changed.
// Uses InferTypeOpInterface so result types stay driven by op attributes.
static FailureOr<bool> refreshResultTypesInPlace(Operation *op) {
  auto inferIface = dyn_cast<InferTypeOpInterface>(op);
  if (!inferIface) {
    return false;
  }

  SmallVector<Type> inferredResultTypes;
  if (failed(inferIface.inferReturnTypes(
          op->getContext(), op->getLoc(), op->getOperands(),
          op->getAttrDictionary(), op->getPropertiesStorage(), op->getRegions(),
          inferredResultTypes))) {
    return op->emitOpError()
           << "failed to infer return types during type propagation";
  }

  bool changed = false;
  for (auto [result, inferredType] :
       llvm::zip_equal(op->getResults(), inferredResultTypes)) {
    if (result.getType() == inferredType) {
      continue;
    }
    result.setType(inferredType);
    changed = true;
  }
  return changed;
}

// Dispatches alias checks for canonical axes. Canonical axes are
// either equivalent or wholly disjoint.
bool areAxesEquivalent(TypedValue<AxisTypeInterface> lhs,
                       TypedValue<AxisTypeInterface> rhs) {
  if (lhs.getType().getTypeID() != rhs.getType().getTypeID()) {
    return false;
  }
  auto lhsAxisIface = lhs.getType();
  return lhsAxisIface.aliases(lhs, rhs);
}

// Tests if two axis factors are disjoint members of some valid factorization
// of a shared source axis.
bool arePairwiseFactorsDisjoint(
    TypedValue<AxisFactorType> lhsFactor, TypedValue<AxisFactorType> rhsFactor,
    TypedValue<AxisTypeInterface> lhsProvenanceAxis,
    TypedValue<AxisTypeInterface> rhsProvenanceAxis) {
  auto lhsType = lhsFactor.getType();
  auto rhsType = rhsFactor.getType();

  TypedValue<AxisTypeInterface> lhsAxis;
  if (lhsProvenanceAxis) {
    lhsAxis = lhsProvenanceAxis;
  } else {
    auto lhsProvenance = getFactorProvenanceAxis(lhsFactor);
    assert(succeeded(lhsProvenance) && "factor must have a provenance axis");
    if (failed(lhsProvenance)) {
      return false;
    }
    lhsAxis = *lhsProvenance;
  }

  TypedValue<AxisTypeInterface> rhsAxis;
  if (rhsProvenanceAxis) {
    rhsAxis = rhsProvenanceAxis;
  } else {
    auto rhsProvenance = getFactorProvenanceAxis(rhsFactor);
    assert(succeeded(rhsProvenance) && "factor must have a provenance axis");
    if (failed(rhsProvenance)) {
      return false;
    }
    rhsAxis = *rhsProvenance;
  }

  // Factors from different canonical axes are disjoint by definition.
  if (!areAxesEquivalent(lhsAxis, rhsAxis)) {
    return true;
  }

  unsigned majorStride = lhsType.getStride();
  unsigned majorExtent = lhsType.getExtent();
  unsigned minorStride = rhsType.getStride();
  unsigned minorExtent = rhsType.getExtent();
  if (majorStride < minorStride) {
    std::swap(majorStride, minorStride);
    std::swap(majorExtent, minorExtent);
  }

  (void)majorExtent;
  unsigned minorSpan = minorStride * minorExtent;
  if (majorStride < minorSpan) {
    return false;
  }
  if ((majorStride % minorSpan) != 0) {
    return false;
  }
  return true;
}

// Asserts an axis (not factor) type and gets the extent.
int getAxisExtent(TypedValue<AxisTypeInterface> axis) {
  return static_cast<int>(axis.getType().extent());
}

int getAxisDimIndex(TypedValue<ShapeAxisType> axis) {
  return static_cast<int>(axis.getType().getAxisIndex());
}

// Asserts a factor type and gets the extent.
int getFactorExtent(TypedValue<AxisFactorType> factor) {
  return static_cast<int>(factor.getType().getExtent());
}

// Asserts a factor type and gets the stride.
int getFactorStride(TypedValue<AxisFactorType> factor) {
  return static_cast<int>(factor.getType().getStride());
}

// Asserts a segment type and gets the extent.
// Returns the defining op for a canonical axis SSA value.
FailureOr<Operation *> getAxisProvenanceOp(Value axis) {
  auto result = dyn_cast<OpResult>(axis);
  if (!result) {
    return failure();
  }
  return result.getOwner();
}

// Returns the defining source axis for a factor value.
FailureOr<TypedValue<AxisTypeInterface>>
getFactorProvenanceAxis(TypedValue<AxisFactorType> factor) {
  if (auto axisFactor = factor.getDefiningOp<AxisFactorOp>()) {
    auto axisValue = axisFactor.getAxis();
    return castTypedValue<AxisTypeInterface>(axisValue, "AxisTypeInterface");
  }
  return failure();
}

// Returns the factor list used to build a factor-product SSA value.
FailureOr<llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>>
getProductProvenanceFactors(TypedValue<FactorGroupType> factorProduct) {
  auto productOp = factorProduct.getDefiningOp<AxisProductOp>();
  if (!productOp) {
    return failure();
  }
  return castTypedValueList<AxisFactorType>(ValueRange(productOp.getFactors()),
                                            "AxisFactorType");
}

// Returns the product of extents for a factor-product SSA value.
FailureOr<uint64_t>
getFactorGroupExtent(TypedValue<FactorGroupType> factorProduct) {
  auto factors = getProductProvenanceFactors(factorProduct);
  if (failed(factors)) {
    return failure();
  }

  uint64_t extent = 1;
  for (Value factor : *factors) {
    auto factorType = dyn_cast<AxisFactorType>(factor.getType());
    if (!factorType) {
      return failure();
    }
    extent *= static_cast<uint64_t>(factorType.getExtent());
  }

  return extent;
}

// Checks factor compatibility and pairwise non-overlap metadata.
bool areFactorsDisjoint(
    llvm::ArrayRef<::mlir::TypedValue<AxisFactorType>> factors) {
  if (factors.empty()) {
    return true;
  }

  assert(factors.size() < 100 &&
         "factor disjointness uses quadratic pairwise checks");

  // Cache provenance once so pairwise checks remain pure and cheap.
  SmallVector<TypedValue<AxisTypeInterface>> cachedProvenances;
  cachedProvenances.reserve(factors.size());
  for (auto factor : factors) {
    assert(getFactorExtent(factor) > 0 && "factor extent must be positive");
    assert(getFactorStride(factor) > 0 && "factor stride must be positive");

    auto provenance = getFactorProvenanceAxis(factor);
    assert(succeeded(provenance) && "factor must have a provenance axis");
    cachedProvenances.push_back(*provenance);
  }

  for (size_t i = 0; i < cachedProvenances.size(); ++i) {
    for (size_t j = i + 1; j < cachedProvenances.size(); ++j) {
      if (!areAxesEquivalent(cachedProvenances[i], cachedProvenances[j])) {
        continue;
      }
      if (!arePairwiseFactorsDisjoint(factors[i], factors[j],
                                      cachedProvenances[i],
                                      cachedProvenances[j])) {
        return false;
      }
    }
  }

  return true;
}

// From a list of factors known to be from the same axis,
// creates a list of pairs indicating the maximum factor ranges.
// Ranges are gauranteed to be return in major-first order.
llvm::SmallVector<std::pair<int, int>>
build_max_factors(TypedValueArrayRef<AxisFactorType> factors) {
  if (factors.empty()) {
    return {};
  }
  // convert into intervals
  llvm::SmallVector<std::pair<int, int>> factor_pairs;
  for (TypedValue<AxisFactorType> factor : factors) {
    auto factorType = factor.getType();
    int extent = static_cast<int>(factorType.getExtent());
    int stride = static_cast<int>(factorType.getStride());
    factor_pairs.push_back({extent, stride});
  }
  // sort intervals by stride
  std::sort(
      factor_pairs.begin(), factor_pairs.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; });

  llvm::SmallVector<std::pair<int, int>> max_factors;
  std::pair<int, int> current_factor = factor_pairs[0];
  for (size_t i = 1; i < factor_pairs.size(); ++i) {
    // if the stride of the current factor = stride * extent of the next factor,
    // they can be combined.
    const auto &next_factor = factor_pairs[i];
    if (current_factor.second == next_factor.first * next_factor.second) {
      current_factor.first *= next_factor.first;
      current_factor.second = next_factor.second;
    } else {
      max_factors.push_back(current_factor);
      current_factor = next_factor;
    }
  }
  max_factors.push_back(current_factor);
  return max_factors;
}

llvm::SmallVector<std::pair<int, int>> build_max_factors(ValueRange factors) {
  if (factors.empty()) {
    return {};
  }
  llvm::SmallVector<TypedValue<AxisFactorType>> typedFactors;
  typedFactors.reserve(factors.size());
  for (Value factor : factors) {
    typedFactors.push_back(
        castTypedValue<AxisFactorType>(factor, "AxisFactorType"));
  }
  return build_max_factors(TypedValueArrayRef<AxisFactorType>(typedFactors));
}

// Compares two factor lists as index-space descriptors, ignoring ordering.
// This is multiset equality over (extent, stride, provenance-axis
// equivalence) and is intentionally permutation-invariant.
bool areFactorIndexSpacesEqual(TypedValueArrayRef<AxisFactorType> lhsFactors,
                               TypedValueArrayRef<AxisFactorType> rhsFactors) {
  struct AxisFactors {
    TypedValue<AxisTypeInterface> provenance;
    SmallVector<Value> lhsFactors;
    SmallVector<Value> rhsFactors;
  };

  auto addFactorsToBuckets = [](TypedValueArrayRef<AxisFactorType> factors,
                                bool isLhs,
                                SmallVectorImpl<AxisFactors> &grouped) {
    for (TypedValue<AxisFactorType> factor : factors) {
      auto provenance = getFactorProvenanceAxis(factor);
      if (failed(provenance)) {
        return false;
      }

      bool inserted = false;
      for (AxisFactors &bucket : grouped) {
        if (areAxesEquivalent(bucket.provenance, *provenance)) {
          if (isLhs) {
            bucket.lhsFactors.push_back(factor);
          } else {
            bucket.rhsFactors.push_back(factor);
          }
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        AxisFactors bucket;
        bucket.provenance = *provenance;
        if (isLhs) {
          bucket.lhsFactors.push_back(factor);
        } else {
          bucket.rhsFactors.push_back(factor);
        }
        grouped.push_back(std::move(bucket));
      }
    }
    return true;
  };

  SmallVector<AxisFactors> grouped;
  if (!addFactorsToBuckets(lhsFactors, /*isLhs=*/true, grouped) ||
      !addFactorsToBuckets(rhsFactors, /*isLhs=*/false, grouped)) {
    return false;
  }

  for (AxisFactors &bucket : grouped) {
    auto lhsMaxFactors = build_max_factors(ValueRange(bucket.lhsFactors));
    auto rhsMaxFactors = build_max_factors(ValueRange(bucket.rhsFactors));
    if (lhsMaxFactors != rhsMaxFactors) {
      return false;
    }
  }

  return true;
}

// Compares two factor lists structurally, preserving list order.
// This requires matching length and pairwise equality of
// (provenance-axis equivalence, extent, stride).
bool areFactorListsStructurallyEqual(
    TypedValueArrayRef<AxisFactorType> lhsFactors,
    TypedValueArrayRef<AxisFactorType> rhsFactors, bool respectShapeTypes) {
  if (lhsFactors.size() != rhsFactors.size()) {
    return false;
  }

  for (auto [lhsFactor, rhsFactor] : llvm::zip(lhsFactors, rhsFactors)) {
    auto lhsProvenance = getFactorProvenanceAxis(lhsFactor);
    auto rhsProvenance = getFactorProvenanceAxis(rhsFactor);
    if (failed(lhsProvenance) || failed(rhsProvenance)) {
      return false;
    }
    bool axes_ok = areAxesEquivalent(*lhsProvenance, *rhsProvenance);

    if (!respectShapeTypes) {
      // if we are two shape axes, then they are OK if they have the same
      // axis index
      ShapeAxisType lhsShape =
          dyn_cast<ShapeAxisType>((*lhsProvenance).getType());
      ShapeAxisType rhsShape =
          dyn_cast<ShapeAxisType>((*rhsProvenance).getType());
      if (lhsShape && rhsShape) {
        axes_ok = lhsShape.getAxisIndex() == rhsShape.getAxisIndex();
      }
    }
    if (!axes_ok) {
      return false;
    }

    if (getFactorExtent(lhsFactor) != getFactorExtent(rhsFactor)) {
      return false;
    }
    if (getFactorStride(lhsFactor) != getFactorStride(rhsFactor)) {
      return false;
    }
  }

  return true;
}

// Checks that factors reconstruct the full source axis extent.
bool areFactorsComplete(TypedValue<AxisTypeInterface> axis,
                        TypedValueArrayRef<AxisFactorType> factors) {
  if (factors.empty() || !areFactorsDisjoint(factors)) {
    return false;
  }

  // Given disjointness, we are complete iff all factors belong to the target
  // axis and their extents cover the whole source-axis extent.
  uint64_t product = 1;
  for (TypedValue<AxisFactorType> factor : factors) {
    auto provenance = getFactorProvenanceAxis(factor);
    assert(succeeded(provenance) && "factor must have a provenance axis");
    assert(*provenance == axis && "factor must belong to the target axis");
    if (*provenance != axis)
      return false; // for non-debug builds

    product *= static_cast<uint64_t>(getFactorExtent(factor));
  }

  return product == static_cast<uint64_t>(getAxisExtent(axis));
}

llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
flattenGroupsToFactors(TypedValueArrayRef<FactorGroupType> factorGroups) {
  llvm::SmallVector<::mlir::TypedValue<AxisFactorType>> flattenedFactors;
  for (auto group : factorGroups) {
    auto factors = getProductProvenanceFactors(group);
    if (failed(factors)) {
      llvm_unreachable(
          "flattenGroupsToFactors failed to get factors from FactorGroupType");
    }
    flattenedFactors.append(factors->begin(), factors->end());
  }
  return flattenedFactors;
}

bool areFactorGroupsDisjoint(TypedValueArrayRef<FactorGroupType> factorGroups) {
  auto flattenedFactors = flattenGroupsToFactors(factorGroups);
  return areFactorsDisjoint(flattenedFactors);
}

llvm::SmallVector<::mlir::TypedValue<AxisTypeInterface>>
createAxesForRankedShape(::mlir::Type shapeType, ::mlir::OpBuilder &builder,
                         ::mlir::Location loc) {
  auto rankedShapeType = cast<ShapedType>(shapeType);
  auto type_attr = TypeAttr::get(rankedShapeType);
  int rank = rankedShapeType.getRank();
  llvm::SmallVector<::mlir::TypedValue<AxisTypeInterface>> axes;
  axes.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto rank_attr = builder.getI32IntegerAttr(i);
    auto axis = builder.create<AxisGetAxisOp>(loc, type_attr, rank_attr);
    axes.push_back(castTypedValue<AxisTypeInterface>(axis.getResult(),
                                                     "AxisTypeInterface"));
  }
  return axes;
}

llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(::mlir::ValueRange axes, ::mlir::OpBuilder &builder,
                  ::mlir::Location loc) {
  auto typedAxes =
      castTypedValueList<AxisTypeInterface>(axes, "AxisTypeInterface");
  return viewAxesAsFactors(typedAxes, builder, loc);
}

llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(TypedValueArrayRef<AxisTypeInterface> axes,
                  ::mlir::OpBuilder &builder, ::mlir::Location loc) {
  llvm::SmallVector<::mlir::TypedValue<AxisFactorType>> factors;
  factors.reserve(axes.size());
  for (auto axis : axes) {
    int extent = getAxisExtent(axis);
    auto factor = builder.create<AxisFactorOp>(loc, axis, extent, 1);
    factors.push_back(
        castTypedValue<AxisFactorType>(factor.getResult(), "AxisFactorType"));
  }
  return factors;
}

::mlir::TypedValue<FactorGroupType>
viewFactorsAsProduct(::mlir::ValueRange factors, ::mlir::OpBuilder &builder,
                     ::mlir::Location loc) {
  auto typedFactors =
      castTypedValueList<AxisFactorType>(factors, "AxisFactorType");
  return viewFactorsAsProduct(TypedValueArrayRef<AxisFactorType>(typedFactors),
                              builder, loc);
}

::mlir::TypedValue<FactorGroupType>
viewFactorsAsProduct(TypedValueArrayRef<AxisFactorType> factors,
                     ::mlir::OpBuilder &builder, ::mlir::Location loc) {
  SmallVector<Value> factorValues;
  factorValues.reserve(factors.size());
  for (TypedValue<AxisFactorType> factor : factors) {
    factorValues.push_back(factor);
  }
  auto product = builder.create<AxisProductOp>(loc, ValueRange(factorValues));
  return product.getProduct();
}

llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
factorAxisByExtents(::mlir::Value axis, llvm::ArrayRef<int32_t> extents,
                    ::mlir::OpBuilder &builder, ::mlir::Location loc) {
  auto typedAxis = castTypedValue<AxisTypeInterface>(axis, "AxisTypeInterface");
  (void)typedAxis;

  llvm::SmallVector<unsigned> strides(extents.size());
  unsigned runningStride = 1;
  for (int idx = static_cast<int>(extents.size()) - 1; idx >= 0; --idx) {
    assert(extents[idx] > 0 && "factor extent must be positive");
    strides[idx] = runningStride;
    runningStride *= static_cast<unsigned>(extents[idx]);
  }

  llvm::SmallVector<::mlir::TypedValue<AxisFactorType>> factors;
  factors.reserve(extents.size());
  for (auto [extent, stride] : llvm::zip_equal(extents, strides)) {
    auto factor = builder.create<AxisFactorOp>(loc, axis, extent,
                                               static_cast<int32_t>(stride));
    factors.push_back(
        castTypedValue<AxisFactorType>(factor.getResult(), "AxisFactorType"));
  }
  return factors;
}

llvm::SmallVector<uint64_t> computeSplits(ArrayRef<uint64_t> lhsExtents,
                                          ArrayRef<uint64_t> rhsExtents) {
  llvm::SmallVector<uint64_t> splits;
  if (lhsExtents.empty() || rhsExtents.empty()) {
    assert(lhsExtents.empty() == rhsExtents.empty() &&
           "split inputs must both be empty or both be non-empty");
    return splits;
  }

#ifndef NDEBUG
  auto assertValidExtents = [](ArrayRef<uint64_t> extents) {
    for (uint64_t extent : extents) {
      assert(extent > 1 && "split extents must be greater than 1");
    }
  };

  assertValidExtents(lhsExtents);
  assertValidExtents(rhsExtents);
#endif

  struct cursor {
    size_t pos;         // current subfactor we are working on.
    uint64_t subfactor; // extent already taken from the current factor
  };
  cursor leftCursor = {0, 1};
  cursor rightCursor = {0, 1};
  // when we "take" a subfactor of given extent, this steps the cursor to
  // the next
  auto advanceCursor = [](cursor &c, ArrayRef<uint64_t> extents,
                          uint64_t size) {
    c.subfactor *= size;
    uint64_t factorSize = extents[c.pos];
    assert(!(c.subfactor > factorSize) && "Subfactor exceeds factor size");
    assert(factorSize % c.subfactor == 0 &&
           "Subfactor does not divide factor size");
    if (c.subfactor == factorSize) {
      c.pos++;
      c.subfactor = 1;
    }
  };
  auto getNextExtent = [](cursor &c, ArrayRef<uint64_t> extents) {
    uint64_t factorSize = extents[c.pos];
    uint64_t remaining = factorSize / c.subfactor;
    assert(remaining > 1 && "Remaining extent must be greater than 1");
    return remaining;
  };

  uint64_t lhsResidual = 1;
  uint64_t rhsResidual = 1;
  while (leftCursor.pos < lhsExtents.size() &&
         rightCursor.pos < rhsExtents.size()) {
    uint64_t newRhs = getNextExtent(rightCursor, rhsExtents);
    uint64_t newLhs = getNextExtent(leftCursor, lhsExtents);

    if (lhsResidual == 1 && rhsResidual == 1) {
      // No residual axis parts from previously,
      // so we are aiming for the maximal one-to-one
      // split
      uint64_t common = std::gcd(newRhs * rhsResidual, newLhs * lhsResidual);
      if (common != 1) {
        splits.push_back(common);
        advanceCursor(leftCursor, lhsExtents, common);
        advanceCursor(rightCursor, rhsExtents, common);
      } else {
        lhsResidual = newLhs;
        rhsResidual = newRhs;
        advanceCursor(leftCursor, lhsExtents, newLhs);
        advanceCursor(rightCursor, rhsExtents, newRhs);
      }
    } else {
      // residual axis parts from previously,
      // so we are aiming for the smallest correct split
      uint64_t lcm = std::lcm(rhsResidual, lhsResidual);
      uint64_t needFromLhs = lcm / lhsResidual;
      uint64_t needFromRhs = lcm / rhsResidual;
      if (newLhs % needFromLhs == 0 && newRhs % needFromRhs == 0) {
        splits.push_back(lcm);
        lhsResidual = 1;
        rhsResidual = 1;
        advanceCursor(leftCursor, lhsExtents, needFromLhs);
        advanceCursor(rightCursor, rhsExtents, needFromRhs);
      } else {
        // Still cannot find a factor, need to add whole axis
        // and move on
        lhsResidual *= newLhs;
        rhsResidual *= newRhs;
        advanceCursor(leftCursor, lhsExtents, newLhs);
        advanceCursor(rightCursor, rhsExtents, newRhs);
      }
    }
  }

  assert(leftCursor.pos == lhsExtents.size() &&
         "Left cursor did not reach end");
  assert(rightCursor.pos == rhsExtents.size() &&
         "Right cursor did not reach end");
  assert(lhsResidual == 1 && "Left residual not fully reduced");
  assert(rhsResidual == 1 && "Right residual not fully reduced");
  return splits;
}

llvm::SmallVector<llvm::SmallVector<SplitExtentSlice>>
computeSplitExtentSlices(ArrayRef<uint64_t> extents, ArrayRef<uint64_t> cuts) {
  llvm::SmallVector<llvm::SmallVector<SplitExtentSlice>> splitSlices;
  splitSlices.reserve(cuts.size());
  if (extents.empty() || cuts.empty()) {
    assert(extents.empty() == cuts.empty() &&
           "split slices require extents and cuts to be simultaneously empty");
    return splitSlices;
  }

  size_t extentIdx = 0;
  uint64_t extentTaken = 1;
  for (uint64_t cut : cuts) {
    assert(cut > 0 && "split cuts must be positive");
    llvm::SmallVector<SplitExtentSlice> currentCut;
    uint64_t cutTaken = 1;
    while (cutTaken < cut) {
      assert(extentIdx < extents.size() &&
             "split slices exhausted extents before cuts");
      uint64_t extent = extents[extentIdx];
      uint64_t extentRemaining = extent / extentTaken;
      uint64_t cutRemaining = cut / cutTaken;
      assert(extentRemaining % cutRemaining == 0 ||
             cutRemaining % extentRemaining == 0);

      uint64_t take = std::min(extentRemaining, cutRemaining);
      currentCut.push_back({extentIdx, take, extentRemaining / take});

      cutTaken *= take;
      extentTaken *= take;
      if (extentTaken == extent) {
        ++extentIdx;
        extentTaken = 1;
      }
    }
    assert(cutTaken == cut && "split cut was not fully materialized");
    splitSlices.push_back(std::move(currentCut));
  }

  assert(extentIdx == extents.size() && "Did not finish slicing extents");
  assert(extentTaken == 1 && "Extent slicing left a trailing partial extent");
  return splitSlices;
}

static llvm::SmallVector<uint64_t>
computeSplits(ArrayRef<TypedValue<AxisFactorType>> lhs,
              ArrayRef<TypedValue<AxisFactorType>> rhs) {
  llvm::SmallVector<uint64_t> lhsExtents;
  lhsExtents.reserve(lhs.size());
  for (TypedValue<AxisFactorType> factor : lhs) {
    lhsExtents.push_back(static_cast<uint64_t>(getFactorExtent(factor)));
  }

  llvm::SmallVector<uint64_t> rhsExtents;
  rhsExtents.reserve(rhs.size());
  for (TypedValue<AxisFactorType> factor : rhs) {
    rhsExtents.push_back(static_cast<uint64_t>(getFactorExtent(factor)));
  }

  return computeSplits(ArrayRef<uint64_t>(lhsExtents),
                       ArrayRef<uint64_t>(rhsExtents));
}

// Attempts to split a mapping of factor products into one-to-one
// factor mappings. For instance, (8) -> (2 * 4) will be split into
// 2->2 and 4->4. This may not always be possible, as in (3 * 2) --> (2 * 3).
// In this case this function will split as much as possible,
// such as (3 * 6) -> (2 * 9) will go to (3 * 2) -> (2 * 3) and (3) -> (3).
// Returns true if split was possible, false if at least one mapping
// could not be split.
// Will attempt to find "maximal" splits but will not merge any factors
// kept separate in the input.
// As always, use the recursive insert strategy if any returned factor
// products are added to the IR.
bool split_divisible(ArrayRef<TypedValue<FactorGroupType>> lhs,
                     ArrayRef<TypedValue<FactorGroupType>> rhs,
                     llvm::SmallVector<TypedValue<FactorGroupType>> &lhs_out,
                     llvm::SmallVector<TypedValue<FactorGroupType>> &rhs_out,
                     mlir::OpBuilder &builder) {
  lhs_out.clear();
  rhs_out.clear();

  bool success = true;
  for (auto [g1, g2] : llvm::zip_equal(lhs, rhs)) {
    auto g1_factors = getProductProvenanceFactors(g1);
    assert(succeeded(g1_factors));
    auto g2_factors = getProductProvenanceFactors(g2);
    assert(succeeded(g2_factors));
    Location groupLoc = g1.getLoc();
    if (g1_factors->size() == 1 && g2_factors->size() == 1) {
      lhs_out.push_back(g1);
      rhs_out.push_back(g2);
      continue;
    }

    // Nonatomic product group
    auto splits = computeSplits(*g1_factors, *g2_factors);
    auto construct_splits =
        [&](ArrayRef<TypedValue<AxisFactorType>> factors,
            llvm::ArrayRef<uint64_t> factorExtents,
            llvm::SmallVector<TypedValue<FactorGroupType>> &out) {
          auto slicedFactors = computeSplitExtentSlices(factorExtents, splits);
          for (const llvm::SmallVector<SplitExtentSlice> &groupSlices :
               slicedFactors) {
            llvm::SmallVector<Value> currentGroup;
            for (const SplitExtentSlice &slice : groupSlices) {
              TypedValue<AxisFactorType> sourceFactor =
                  factors[slice.extentIdx];
              auto factor_axis = getFactorProvenanceAxis(sourceFactor);
              assert(succeeded(factor_axis) &&
                     "factor must have a provenance axis");
              int32_t new_factor_extent = static_cast<int32_t>(slice.subExtent);
              int32_t new_factor_stride = static_cast<int32_t>(
                  static_cast<uint64_t>(getFactorStride(sourceFactor)) *
                  slice.stride);
              auto splitFactor = builder.create<AxisFactorOp>(
                  groupLoc, *factor_axis, new_factor_extent, new_factor_stride);
              currentGroup.push_back(splitFactor.getResult());
            }

            auto product = builder.create<AxisProductOp>(
                groupLoc, ValueRange(currentGroup));
            out.push_back(castTypedValue<FactorGroupType>(product.getResult(),
                                                          "FactorGroupType"));
            success = success && (currentGroup.size() == 1);
          }
        };

    llvm::SmallVector<uint64_t> g1Extents;
    g1Extents.reserve(g1_factors->size());
    for (TypedValue<AxisFactorType> factor : *g1_factors) {
      g1Extents.push_back(static_cast<uint64_t>(getFactorExtent(factor)));
    }
    llvm::SmallVector<uint64_t> g2Extents;
    g2Extents.reserve(g2_factors->size());
    for (TypedValue<AxisFactorType> factor : *g2_factors) {
      g2Extents.push_back(static_cast<uint64_t>(getFactorExtent(factor)));
    }

    construct_splits(*g1_factors, g1Extents, lhs_out);
    construct_splits(*g2_factors, g2Extents, rhs_out);
  }

  return success;
}

// Subtract one factor from one factor. Returns major-first remainder factors.
static FailureOr<llvm::SmallVector<TypedValue<AxisFactorType>>>
subtractFactorFromFactor(TypedValue<AxisFactorType> minuend,
                         TypedValue<AxisFactorType> subtrahend,
                         OpBuilder &builder, Location loc) {
  if (arePairwiseFactorsDisjoint(minuend, subtrahend)) {
    return llvm::SmallVector<TypedValue<AxisFactorType>>{minuend};
  }

  auto minuendAxis = getFactorProvenanceAxis(minuend);
  auto subAxis = getFactorProvenanceAxis(subtrahend);
  if (failed(minuendAxis) || failed(subAxis)) {
    return failure();
  }
  if (!areAxesEquivalent(*minuendAxis, *subAxis)) {
    return failure();
  }

  int aExtent = getFactorExtent(minuend);
  int aStride = getFactorStride(minuend);
  int bExtent = getFactorExtent(subtrahend);
  int bStride = getFactorStride(subtrahend);
  if (aExtent <= 1 || aStride <= 1 || bExtent <= 1 || bStride <= 1) {
    return failure();
  }

  int64_t aSpan = static_cast<int64_t>(aExtent) * static_cast<int64_t>(aStride);
  int64_t bSpan = static_cast<int64_t>(bExtent) * static_cast<int64_t>(bStride);

  llvm::SmallVector<TypedValue<AxisFactorType>> remainder;

  // Upper remainder: larger covered range than the removed factor.
  if (aSpan > bSpan) {
    if ((aSpan % bSpan) != 0) {
      return failure();
    }
    int64_t upperExtent = aSpan / bSpan;
    if (upperExtent > 1) {
      if (upperExtent > std::numeric_limits<int32_t>::max()) {
        return failure();
      }
      auto upperFactor = builder.create<AxisFactorOp>(
          loc, *minuendAxis, static_cast<int32_t>(upperExtent),
          static_cast<int32_t>(bSpan));
      remainder.push_back(castTypedValue<AxisFactorType>(
          upperFactor.getResult(), "AxisFactorType"));
    }
  }

  // Lower remainder: retained minor regions below the removed factor stride.
  if (aStride < bStride) {
    if ((bStride % aStride) != 0) {
      return failure();
    }
    int64_t lowerExtent = bStride / aStride;
    if (lowerExtent > 1) {
      if (lowerExtent > std::numeric_limits<int32_t>::max()) {
        return failure();
      }
      auto lowerFactor = builder.create<AxisFactorOp>(
          loc, *minuendAxis, static_cast<int32_t>(lowerExtent),
          static_cast<int32_t>(aStride));
      remainder.push_back(castTypedValue<AxisFactorType>(
          lowerFactor.getResult(), "AxisFactorType"));
    }
  }

  // Neither condition means minuend is eclipsed by subtrahend under this
  // factor-space subtraction model.
  return remainder;
}

FailureOr<llvm::SmallVector<TypedValue<AxisFactorType>>>
subtractFactorsFromFactorGroup(
    TypedValue<FactorGroupType> minuend,
    llvm::ArrayRef<TypedValue<AxisFactorType>> subtrahend, OpBuilder &builder) {
  auto remainder = getProductProvenanceFactors(minuend);
  if (failed(remainder)) {
    return failure();
  }
  auto loc = minuend.getLoc();

  for (TypedValue<AxisFactorType> removedFactor : subtrahend) {
    llvm::SmallVector<TypedValue<AxisFactorType>> nextRemainder;
    nextRemainder.reserve(remainder->size());

    bool hadAliasOverlap = false;
    for (TypedValue<AxisFactorType> candidate : *remainder) {
      auto candidateAxis = getFactorProvenanceAxis(candidate);
      auto removedAxis = getFactorProvenanceAxis(removedFactor);
      if (failed(candidateAxis) || failed(removedAxis)) {
        return failure();
      }
      hadAliasOverlap =
          hadAliasOverlap || areAxesEquivalent(*candidateAxis, *removedAxis);

      auto partialRemainder =
          subtractFactorFromFactor(candidate, removedFactor, builder, loc);
      if (failed(partialRemainder)) {
        return failure();
      }
      nextRemainder.append(partialRemainder->begin(), partialRemainder->end());
    }

    // If there is no aliasing factor in the current remainder, subtraction is
    // undefined for this removed factor.
    if (!hadAliasOverlap) {
      return failure();
    }

    *remainder = std::move(nextRemainder);
  }

  return *remainder;
}

struct _global_factor {
  int extent;
  int global_stride;
};

// Projects one factor defined in a virtual factor-group index space onto
// factors of the real underlying axes.
static FailureOr<llvm::SmallVector<TypedValue<AxisFactorType>>>
projectVirtualFactorToRealFactors(TypedValue<FactorGroupType> virtualAxis,
                                  int virtualStride, int virtualExtent,
                                  OpBuilder &builder, Location loc) {
  LLVM_DEBUG(llvm::dbgs() << "[axis-infer-map] project start stride="
                          << virtualStride << " extent=" << virtualExtent
                          << "\n");
  if (virtualStride <= 0 || virtualExtent <= 0) {
    return failure();
  }

  auto virtualFactors = getProductProvenanceFactors(virtualAxis);
  if (failed(virtualFactors) || virtualFactors->empty()) {
    return failure();
  }

  // Remove complete minor-most virtual factors from the virtual stride,
  // then split the first partially-covered factor as needed.
  int pivot = static_cast<int>(virtualFactors->size()) - 1;
  int localStrideInPivot = virtualStride;
  while (pivot >= 0 &&
         localStrideInPivot >= getFactorExtent((*virtualFactors)[pivot])) {
    if (localStrideInPivot % getFactorExtent((*virtualFactors)[pivot]) != 0) {
      return failure();
    }
    localStrideInPivot /= getFactorExtent((*virtualFactors)[pivot]);
    --pivot;
  }
  LLVM_DEBUG(llvm::dbgs() << "[axis-infer-map] project pivot=" << pivot
                          << " localStrideInPivot=" << localStrideInPivot
                          << "\n");
  assert(pivot >= 0 && "Virtual factor must fit within product group extent");

  int remainingExtent = virtualExtent;
  llvm::SmallVector<TypedValue<AxisFactorType>> projectedMinorToMajor;

  for (int i = pivot; i >= 0 && remainingExtent > 1; --i) {
    auto sourceFactor = (*virtualFactors)[i];
    int sourceExtent = getFactorExtent(sourceFactor);
    int sourceStride = getFactorStride(sourceFactor);
    int sourcePieceExtent = sourceExtent;
    int sourcePieceStride = sourceStride;

    if (i == pivot) {
      sourcePieceExtent = sourceExtent / localStrideInPivot;
      sourcePieceStride = sourceStride * localStrideInPivot;
    }

    int takeExtent = 0;
    if (remainingExtent >= sourcePieceExtent) {
      if (remainingExtent % sourcePieceExtent != 0) {
        return failure();
      }
      takeExtent = sourcePieceExtent;
    } else {
      if (sourcePieceExtent % remainingExtent != 0) {
        return failure();
      }
      takeExtent = remainingExtent;
    }

    // For partial picks, take the minor-most subpiece of the available source
    // piece so disjoint virtual factors project to disjoint real factors.
    int projectedStride = sourcePieceStride;
    if (takeExtent <= 1) {
      return failure();
    }

    auto provenanceAxis = getFactorProvenanceAxis(sourceFactor);
    if (failed(provenanceAxis)) {
      return failure();
    }

    auto projected = builder.create<AxisFactorOp>(loc, *provenanceAxis,
                                                  takeExtent, projectedStride);
    LLVM_DEBUG(llvm::dbgs()
               << "[axis-infer-map]   project factor i=" << i
               << " src(ext=" << sourceExtent << ", stride=" << sourceStride
               << ") piece(ext=" << sourcePieceExtent
               << ", stride=" << sourcePieceStride << ") take=" << takeExtent
               << " -> projected stride=" << projectedStride << "\n");
    projectedMinorToMajor.push_back(castTypedValue<AxisFactorType>(
        projected.getResult(), "AxisFactorType"));
    remainingExtent /= takeExtent;
  }

  if (remainingExtent != 1) {
    return failure();
  }

  std::reverse(projectedMinorToMajor.begin(), projectedMinorToMajor.end());
  return projectedMinorToMajor;
}

// against convention takes MINORMOST FIRST
llvm::SmallVector<int>
_globalFactorsToRHSIndices(ArrayRef<_global_factor> factors) {
  llvm::SmallVector<int> rhs_indices;
  rhs_indices.push_back(0);
  for (const auto &factor : factors) {
    int existing = rhs_indices.size();
    for (int i = 1; i < factor.extent; ++i) {
      for (int j = 0; j < existing; ++j) {
        rhs_indices.push_back(rhs_indices[j] + i * factor.global_stride);
      }
    }
  }
  return rhs_indices;
}

// rhs_indices are in the same index space,
// and are in-order according to their LHS indices
// (rhs_indices[i] = j means i->j, with i the i'th
// element within the index space regardless of how the
// index space is actually laid out.)
FailureOr<TypedValue<AxisMapType>>
inferMapFromIndices(TypedValue<FactorGroupType> index_space,
                    llvm::ArrayRef<int> rhs_indices, OpBuilder &builder) {
  LLVM_DEBUG(llvm::dbgs() << "[axis-infer-map] inferMapFromIndices rhs size="
                          << rhs_indices.size() << "\n");
  auto indexSpaceExtent = getFactorGroupExtent(index_space);
  if (failed(indexSpaceExtent)) {
    return failure();
  }
  assert(*indexSpaceExtent == rhs_indices.size() &&
         "index-space extent must match rhs index count");
  if (*indexSpaceExtent != rhs_indices.size()) {
    return failure();
  }
  if (rhs_indices.size() <= 1) {
    return failure();
  }
  if (rhs_indices[0] != 0) {
    // no axis map moves zero
    return failure();
  }

  auto loc = index_space.getLoc();

  // minormost first, against convention, since it is in this
  // case easiest to construct the global factors in this order.
  llvm::SmallVector<_global_factor> factors;
  int group_stride = 1;
  int factor_working_extent = 1;
  int factor_working_stride = rhs_indices[1] - rhs_indices[0];
  while (group_stride < rhs_indices.size()) {
    int i1 = group_stride * (factor_working_extent - 1);
    int i2 = group_stride * factor_working_extent;
    if (i2 >= rhs_indices.size()) {
      // finished all of our runs
      break;
    }
    int diff = rhs_indices[i2] - rhs_indices[i1];
    if (diff != factor_working_stride) {
      // we've found the end of the run!
      factors.push_back({factor_working_extent, factor_working_stride});
      group_stride *= factor_working_extent;
      if (rhs_indices.size() % group_stride != 0) {
        // indices don't implement a regular axis mapping
        return failure();
      }
      factor_working_extent = 1;
      factor_working_stride = rhs_indices[group_stride] - rhs_indices[0];
    } else {
      // extend the current run
      factor_working_extent++;
    }
  }
  // push the last run if it exists
  if (factor_working_extent > 1) {
    factors.push_back({factor_working_extent, factor_working_stride});
  }

  // verify that the reconstructed RHS indices match the original
  // (we didn't check every index, so this is a final verification)
  if (_globalFactorsToRHSIndices(factors) != rhs_indices) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[axis-infer-map] global factors (minor->major):";
    for (const auto &factor : factors) {
      llvm::dbgs() << " (ext=" << factor.extent
                   << ", stride=" << factor.global_stride << ")";
    }
    llvm::dbgs() << "\n";
  });

  // reverse order of global factors now to meet the
  // major-most convention used elswhere
  std::reverse(factors.begin(), factors.end());

  LLVM_DEBUG({
    llvm::dbgs() << "[axis-infer-map] global factors (major->minor):";
    for (const auto &factor : factors) {
      llvm::dbgs() << " (ext=" << factor.extent
                   << ", stride=" << factor.global_stride << ")";
    }
    llvm::dbgs() << "\n";
  });

  llvm::SmallVector<TypedValue<AxisFactorType>> rhsFactors;
  for (const auto &globalFactor : factors) {
    LLVM_DEBUG(llvm::dbgs() << "[axis-infer-map] project global factor ext="
                            << globalFactor.extent << " stride="
                            << globalFactor.global_stride << "\n");
    auto projected = projectVirtualFactorToRealFactors(
        index_space, globalFactor.global_stride, globalFactor.extent, builder,
        loc);
    if (failed(projected)) {
      return failure();
    }
    rhsFactors.append(projected->begin(), projected->end());
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[axis-infer-map] rhs factors:";
    for (TypedValue<AxisFactorType> factor : rhsFactors) {
      llvm::dbgs() << " (ext=" << getFactorExtent(factor)
                   << ", stride=" << getFactorStride(factor) << ")";
    }
    llvm::dbgs() << "\n";
  });

  llvm::SmallVector<Value> rhsValues;
  rhsValues.reserve(rhsFactors.size());
  for (TypedValue<AxisFactorType> factor : rhsFactors) {
    rhsValues.push_back(factor);
  }

  auto rhsGroup =
      builder.create<AxisProductOp>(loc, ValueRange(rhsValues)).getProduct();
  llvm::SmallVector<Value> lhsGroups;
  lhsGroups.push_back(index_space);
  llvm::SmallVector<Value> rhsGroups;
  rhsGroups.push_back(rhsGroup);
  auto mapOp = builder.create<AxisMapOp>(loc, ValueRange(lhsGroups),
                                         ValueRange(rhsGroups));
  return castTypedValue<AxisMapType>(mapOp.getMap(), "AxisMapType");
}

LogicalResult propagateResultTypeChanges(ArrayRef<Operation *> initialUsers) {
  // Use a set-backed worklist so each op is refreshed at most once per wave.
  llvm::SmallSetVector<Operation *, 32> worklist;
  for (Operation *user : initialUsers) {
    if (user) {
      worklist.insert(user);
    }
  }

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!op || !op->getBlock()) {
      continue;
    }

    auto changed = refreshResultTypesInPlace(op);
    if (failed(changed)) {
      return failure();
    }
    if (!*changed) {
      continue;
    }

    // A result-type change can require all downstream users to refresh too.
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        worklist.insert(user);
      }
    }
  }

  return success();
}

LogicalResult replaceAndTypePropagate(Value from, Value to) {
  if (from == to) {
    return success();
  }

  // Snapshot current users because RAUW invalidates use iteration.
  SmallVector<Operation *> affectedUsers;
  for (Operation *user : from.getUsers()) {
    affectedUsers.push_back(user);
  }

  from.replaceAllUsesWith(to);
  if (from.getType() == to.getType()) {
    return success();
  }
  return propagateResultTypeChanges(affectedUsers);
}

Predicate<std::pair<::mlir::TypedValue<FactorGroupType>,
                    ::mlir::TypedValue<FactorGroupType>>>
predGroupPairIsIdentity(bool respectShapeTypes) {
  return [respectShapeTypes](std::pair<::mlir::TypedValue<FactorGroupType>,
                                       ::mlir::TypedValue<FactorGroupType>>
                                 groupPair) {
    auto lhsFactors = getProductProvenanceFactors(groupPair.first);
    auto rhsFactors = getProductProvenanceFactors(groupPair.second);
    if (failed(lhsFactors) || failed(rhsFactors)) {
      llvm_unreachable(
          "predGroupPairIsIdentity failed to get factors from FactorGroupType");
    }

    return areFactorListsStructurallyEqual(*lhsFactors, *rhsFactors,
                                           respectShapeTypes);
  };
}

} // namespace mlir::enzyme::axis
