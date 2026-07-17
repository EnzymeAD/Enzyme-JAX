#include "Utilities.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "axis-infer-map"

namespace mlir::enzyme::axis {

// Dispatches alias checks for canonical axes. Canonical axes are
// either equivalent or wholly disjoint.
static bool areAxesEquivalent(Value lhs, Value rhs) {
  if (!isa<AxisTypeInterface>(lhs.getType()) ||
      !isa<AxisTypeInterface>(rhs.getType())) {
    return false;
  }
  if (lhs.getType().getTypeID() != rhs.getType().getTypeID()) {
    return false;
  }
  auto lhsAxisIface = dyn_cast<AxisTypeInterface>(lhs.getType());
  assert(lhsAxisIface && "axis value type must implement AxisTypeInterface");
  if (!lhsAxisIface) {
    return false;
  }
  return lhsAxisIface.aliases(lhs, rhs);
}

// Tests if two axis factors are disjoint members of some valid factorization
// of a shared source axis.
bool arePairwiseFactorsDisjoint(Value lhsFactor, Value rhsFactor,
                                Value lhsProvenanceAxis,
                                Value rhsProvenanceAxis) {
  auto lhsTyped = castTypedValue<AxisFactorType>(lhsFactor, "AxisFactorType");
  auto rhsTyped = castTypedValue<AxisFactorType>(rhsFactor, "AxisFactorType");
  auto lhsType = lhsTyped.getType();
  auto rhsType = rhsTyped.getType();

  Value lhsAxis = lhsProvenanceAxis;
  if (!lhsAxis) {
    auto lhsProvenance = getFactorProvenanceAxis(lhsTyped);
    assert(succeeded(lhsProvenance) && "factor must have a provenance axis");
    if (failed(lhsProvenance)) {
      return false;
    }
    lhsAxis = *lhsProvenance;
  }

  Value rhsAxis = rhsProvenanceAxis;
  if (!rhsAxis) {
    auto rhsProvenance = getFactorProvenanceAxis(rhsTyped);
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

// Asserts a factor type and gets the extent.
int getFactorExtent(TypedValue<AxisFactorType> factor) {
  return static_cast<int>(factor.getType().getExtent());
}

// Asserts a factor type and gets the stride.
int getFactorStride(TypedValue<AxisFactorType> factor) {
  return static_cast<int>(factor.getType().getStride());
}

// Asserts a segment type and gets the extent.
int getSegmentExtent(TypedValue<AxisSegmentType> segment) {
  return static_cast<int>(segment.getType().getExtent());
}

// Returns the defining op for a canonical axis SSA value.
FailureOr<Operation *> getAxisProvenanceOp(Value axis) {
  auto result = dyn_cast<OpResult>(axis);
  if (!result) {
    return failure();
  }
  return result.getOwner();
}

// Returns the defining source axis for a factor value.
FailureOr<Value> getFactorProvenanceAxis(TypedValue<AxisFactorType> factor) {
  if (auto axisFactor = factor.getDefiningOp<AxisFactorOp>()) {
    return axisFactor.getAxis();
  }

  return failure();
}

// Returns the defining source axis for a segment value.
FailureOr<Value> getSegmentProvenanceAxis(TypedValue<AxisSegmentType> segment) {
  if (auto axisSegment = segment.getDefiningOp<AxisSegmentOp>()) {
    return axisSegment.getAxis();
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

  struct FactorInfo {
    Value provenance;
  };

  // Cache provenance once so pairwise checks remain pure and cheap.
  SmallVector<FactorInfo> cachedFactors;
  cachedFactors.reserve(factors.size());
  for (auto factor : factors) {
    assert(getFactorExtent(factor) > 0 && "factor extent must be positive");
    assert(getFactorStride(factor) > 0 && "factor stride must be positive");

    auto provenance = getFactorProvenanceAxis(factor);
    assert(succeeded(provenance) && "factor must have a provenance axis");
    cachedFactors.push_back({*provenance});
  }

  for (size_t i = 0; i < cachedFactors.size(); ++i) {
    for (size_t j = i + 1; j < cachedFactors.size(); ++j) {
      Value lhsAxis = cachedFactors[i].provenance;
      Value rhsAxis = cachedFactors[j].provenance;

      if (!areAxesEquivalent(lhsAxis, rhsAxis)) {
        continue;
      }
      if (!arePairwiseFactorsDisjoint(factors[i], factors[j], lhsAxis,
                                      rhsAxis)) {
        return false;
      }
    }
  }

  return true;
}

// From a list of factors known to be from the same axis,
// creates a list of pairs indicating the maximum factor ranges.
// Ranges are gauranteed to be return in major-first order.
llvm::SmallVector<std::pair<int, int>> build_max_factors(ValueRange factors) {
  if (factors.empty()) {
    return {};
  }
  // convert into intervals
  llvm::SmallVector<std::pair<int, int>> factor_pairs;
  for (Value factor : factors) {
    auto factorTyped = castTypedValue<AxisFactorType>(factor, "AxisFactorType");
    auto factorType = factorTyped.getType();
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

// Compares two factor lists as index-space descriptors, ignoring ordering.
// This is multiset equality over (extent, stride, provenance-axis
// equivalence) and is intentionally permutation-invariant.
bool areFactorIndexSpacesEqual(TypedValueArrayRef<AxisFactorType> lhsFactors,
                               TypedValueArrayRef<AxisFactorType> rhsFactors) {
  struct AxisFactors {
    Value provenance;
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

// Checks segment pairwise non-overlap metadata.
bool arePairwiseSegmentsDisjoint(Value lhsSegment, Value rhsSegment,
                                 Value lhsProvenanceAxis,
                                 Value rhsProvenanceAxis) {
  auto lhsTyped =
      castTypedValue<AxisSegmentType>(lhsSegment, "AxisSegmentType");
  auto rhsTyped =
      castTypedValue<AxisSegmentType>(rhsSegment, "AxisSegmentType");
  auto lhsType = lhsTyped.getType();
  auto rhsType = rhsTyped.getType();

  Value lhsAxis = lhsProvenanceAxis;
  if (!lhsAxis) {
    auto lhsProvenance = getSegmentProvenanceAxis(lhsTyped);
    assert(succeeded(lhsProvenance) && "segment must have a provenance axis");
    if (failed(lhsProvenance)) {
      return false;
    }
    lhsAxis = *lhsProvenance;
  }

  Value rhsAxis = rhsProvenanceAxis;
  if (!rhsAxis) {
    auto rhsProvenance = getSegmentProvenanceAxis(rhsTyped);
    assert(succeeded(rhsProvenance) && "segment must have a provenance axis");
    if (failed(rhsProvenance)) {
      return false;
    }
    rhsAxis = *rhsProvenance;
  }

  // Segments from different canonical axes are disjoint by definition.
  if (!areAxesEquivalent(lhsAxis, rhsAxis)) {
    return true;
  }

  uint64_t lhsStart = static_cast<uint64_t>(lhsType.getOffset());
  uint64_t lhsEnd = lhsStart + static_cast<uint64_t>(lhsType.getExtent());
  uint64_t rhsStart = static_cast<uint64_t>(rhsType.getOffset());
  uint64_t rhsEnd = rhsStart + static_cast<uint64_t>(rhsType.getExtent());
  return lhsEnd <= rhsStart || rhsEnd <= lhsStart;
}

// Checks segment group pairwise non-overlap metadata.
bool areSegmentsDisjoint(ValueRange segments) {
  if (segments.empty()) {
    return true;
  }

  assert(segments.size() < 100 &&
         "segment disjointness uses quadratic pairwise checks");

  SmallVector<Value> provenanceAxes;
  provenanceAxes.reserve(segments.size());
  for (Value segment : segments) {
    auto segmentTyped =
        castTypedValue<AxisSegmentType>(segment, "AxisSegmentType");
    assert(getSegmentExtent(segmentTyped) > 0 &&
           "segment extent must be positive");

    auto provenance = getSegmentProvenanceAxis(segmentTyped);
    assert(succeeded(provenance) && "segment must have a provenance axis");
    if (failed(provenance)) {
      return false;
    }
    provenanceAxes.push_back(*provenance);
  }

  for (size_t i = 0; i < segments.size(); ++i) {
    for (size_t j = i + 1; j < segments.size(); ++j) {
      if (!areAxesEquivalent(provenanceAxes[i], provenanceAxes[j])) {
        continue;
      }
      if (!arePairwiseSegmentsDisjoint(segments[i], segments[j],
                                       provenanceAxes[i], provenanceAxes[j])) {
        return false;
      }
    }
  }

  return true;
}

// Checks that factors reconstruct the full source axis extent.
bool areFactorsComplete(Value axis,
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

  return product ==
         static_cast<uint64_t>(getAxisExtent(
             castTypedValue<AxisTypeInterface>(axis, "AxisTypeInterface")));
}

// Checks that segments reconstruct the full source axis interval [0, extent).
bool areSegmentsComplete(Value axis, ValueRange segments) {
  if (segments.empty() || !areSegmentsDisjoint(segments)) {
    return false;
  }

  SmallVector<std::pair<uint64_t, uint64_t>> intervals;
  intervals.reserve(segments.size());

  for (Value segment : segments) {
    auto segmentTyped =
        castTypedValue<AxisSegmentType>(segment, "AxisSegmentType");
    auto segmentType = segmentTyped.getType();

    auto provenance = getSegmentProvenanceAxis(segmentTyped);
    assert(succeeded(provenance) && "segment must have a provenance axis");
    if (failed(provenance) || *provenance != axis) {
      return false;
    }

    uint64_t start = static_cast<uint64_t>(segmentType.getOffset());
    uint64_t end = start + static_cast<uint64_t>(segmentType.getExtent());
    intervals.emplace_back(start, end);
  }

  std::sort(intervals.begin(), intervals.end());
  if (intervals.front().first != 0) {
    return false;
  }

  uint64_t cursor = 0;
  for (auto [start, end] : intervals) {
    if (start != cursor) {
      return false;
    }
    cursor = end;
  }

  return cursor ==
         static_cast<uint64_t>(getAxisExtent(
             castTypedValue<AxisTypeInterface>(axis, "AxisTypeInterface")));
}

llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
flattenGroupsToFactors(TypedValueArrayRef<FactorGroupType> factorGroups) {
  llvm::SmallVector<::mlir::TypedValue<AxisFactorType>> flattenedFactors;
  for (auto group : factorGroups) {
    auto factors = getProductProvenanceFactors(group);
    if (failed(factors)) {
      llvm::report_fatal_error(
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

llvm::SmallVector<int>
compute_splits(ArrayRef<TypedValue<AxisFactorType>> lhs,
               ArrayRef<TypedValue<AxisFactorType>> rhs) {
  struct cursor {
    int pos;       // current subfactor we are working on.
    int subfactor; // "stride" of the factors we haven't yet taken
  };
  cursor left_cursor = {0, 1};
  cursor right_cursor = {0, 1};
  // when we "take" a subfactor of given extent, this steps the cursor to
  // the next
  auto advance_cursor =
      [](cursor &c, ArrayRef<TypedValue<AxisFactorType>> factors, int size) {
        c.subfactor *= size;
        int factor_size = getFactorExtent(factors[c.pos]);
        assert(!(c.subfactor > factor_size) && "Subfactor exceeds factor size");
        assert(factor_size % c.subfactor == 0 &&
               "Subfactor does not divide factor size");
        if (c.subfactor == factor_size) {
          c.pos++;
          c.subfactor = 1;
        }
      };
  auto get_next_extent = [](cursor &c,
                            ArrayRef<TypedValue<AxisFactorType>> factors) {
    int factor_size = getFactorExtent(factors[c.pos]);
    int remaining = factor_size / c.subfactor;
    assert(remaining > 1 && "Remaining extent must be greater than 1");
    return remaining;
  };
  llvm::SmallVector<int> splits;
  int lhs_residual = 1;
  int rhs_residual = 1;
  while (left_cursor.pos < lhs.size() && right_cursor.pos < rhs.size()) {
    int new_rhs = get_next_extent(right_cursor, rhs);
    int new_lhs = get_next_extent(left_cursor, lhs);

    if (lhs_residual == 1 && rhs_residual == 1) {
      // No residual axis parts from previously,
      // so we are aiming for the maximal one-to-one
      // split
      int common = std::gcd(new_rhs * rhs_residual, new_lhs * lhs_residual);
      if (common != 1) {
        splits.push_back(common);
        advance_cursor(left_cursor, lhs, common);
        advance_cursor(right_cursor, rhs, common);
      } else {
        lhs_residual = new_lhs;
        rhs_residual = new_rhs;
        advance_cursor(left_cursor, lhs, new_lhs);
        advance_cursor(right_cursor, rhs, new_rhs);
      }
    } else {
      // residual axis parts from previously,
      // so we are aiming for the smallest correct split
      int lcm = std::lcm(rhs_residual, lhs_residual);
      int need_from_lhs = lcm / lhs_residual;
      int need_from_rhs = lcm / rhs_residual;
      if (new_lhs % need_from_lhs == 0 && new_rhs % need_from_rhs == 0) {
        splits.push_back(lcm);
        lhs_residual = 1;
        rhs_residual = 1;
        advance_cursor(left_cursor, lhs, need_from_lhs);
        advance_cursor(right_cursor, rhs, need_from_rhs);
      } else {
        // Still cannot find a factor, need to add whole axis
        // and move on
        lhs_residual *= new_lhs;
        rhs_residual *= new_rhs;
        advance_cursor(left_cursor, lhs, new_lhs);
        advance_cursor(right_cursor, rhs, new_rhs);
      }
    }
  }
  // expect both cursors to have reached end
  assert(left_cursor.pos == lhs.size() && "Left cursor did not reach end");
  assert(right_cursor.pos == rhs.size() && "Right cursor did not reach end");
  assert(lhs_residual == 1 && "Left residual not fully reduced");
  assert(rhs_residual == 1 && "Right residual not fully reduced");
  return splits;
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
    if (g1_factors->size() == 1 && g2_factors->size() == 1) {
      lhs_out.push_back(g1);
      rhs_out.push_back(g2);
      continue;
    }

    // Nonatomic product group
    auto splits = compute_splits(*g1_factors, *g2_factors);
    auto construct_splits =
        [&](ArrayRef<TypedValue<AxisFactorType>> factors,
            llvm::SmallVector<TypedValue<FactorGroupType>> &out) {
          llvm::SmallVector<Value> currentGroup;
          auto factor_it = factors.begin();
          int factor_taken = 1;
          auto split_it = splits.begin();
          int split_taken = 1;

          while (split_it != splits.end()) {
            // because of our iteration order we are visiting high-order first
            int factor_remaining = getFactorExtent(*factor_it) / factor_taken;
            int split_remaining = *split_it / split_taken;
            // expect split to divide factor or vice versa, or both if equal.
            assert(factor_remaining % split_remaining == 0 ||
                   split_remaining % factor_remaining == 0);
            int take = std::min(factor_remaining, split_remaining);
            int new_factor_extent = take;
            int new_factor_stride = getFactorStride(*factor_it) *
                                    (factor_remaining / new_factor_extent);
            auto factor_axis = getFactorProvenanceAxis(*factor_it);
            assert(succeeded(factor_axis) &&
                   "factor must have a provenance axis");
            auto loc = g1.getLoc();
            auto splitFactor = builder.create<AxisFactorOp>(
                loc, *factor_axis, new_factor_extent, new_factor_stride);
            currentGroup.push_back(splitFactor.getResult());
            split_taken *= take;
            factor_taken *= take;
            bool splitFullyConsumed = (split_taken == *split_it);
            bool factorFullyConsumed =
                (factor_taken == getFactorExtent(*factor_it));

            if (splitFullyConsumed) {
              // Push our current group to out
              auto product = builder.create<AxisProductOp>(
                  g1.getLoc(), ValueRange(currentGroup));
              out.push_back(castTypedValue<FactorGroupType>(product.getResult(),
                                                            "FactorGroupType"));
              success = success && (currentGroup.size() ==
                                    1); // record if atomic factors found
              currentGroup.clear();

              // reset the split
              split_it++;
              split_taken = 1;
            }
            if (factorFullyConsumed) {
              // Move to the next factor
              factor_it++;
              factor_taken = 1;
            }
          }
        };
    construct_splits(*g1_factors, lhs_out);
    construct_splits(*g2_factors, rhs_out);
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
  if (failed(minuendAxis) || failed(subAxis) ||
      !areAxesEquivalent(*minuendAxis, *subAxis)) {
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

} // namespace mlir::enzyme::axis
