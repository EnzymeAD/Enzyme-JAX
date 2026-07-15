#include "Utilities.h"

#include <algorithm>

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
  auto lhsType = dyn_cast<AxisFactorType>(lhsFactor.getType());
  auto rhsType = dyn_cast<AxisFactorType>(rhsFactor.getType());
  assert(lhsType && "factor value must have AxisFactorType");
  assert(rhsType && "factor value must have AxisFactorType");
  if (!lhsType || !rhsType) {
    return false;
  }

  Value lhsAxis = lhsProvenanceAxis;
  if (!lhsAxis) {
    auto lhsProvenance =
        getFactorProvenanceAxis(cast<TypedValue<AxisFactorType>>(lhsFactor));
    assert(succeeded(lhsProvenance) && "factor must have a provenance axis");
    if (failed(lhsProvenance)) {
      return false;
    }
    lhsAxis = *lhsProvenance;
  }

  Value rhsAxis = rhsProvenanceAxis;
  if (!rhsAxis) {
    auto rhsProvenance =
        getFactorProvenanceAxis(cast<TypedValue<AxisFactorType>>(rhsFactor));
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
int getAxisExtent(Value axis) {
  auto axisInterface = dyn_cast<AxisTypeInterface>(axis.getType());
  assert(axisInterface && "axis type must implement AxisTypeInterface");
  return static_cast<int>(axisInterface.extent());
}

// Asserts a factor type and gets the extent.
int getFactorExtent(Value factor) {
  auto factorType = dyn_cast<AxisFactorType>(factor.getType());
  assert(factorType && "factor type must be AxisFactorType");
  return static_cast<int>(factorType.getExtent());
}

// Asserts a segment type and gets the extent.
int getSegmentExtent(Value segment) {
  auto segmentType = dyn_cast<AxisSegmentType>(segment.getType());
  assert(segmentType && "segment type must be AxisSegmentType");
  return static_cast<int>(segmentType.getExtent());
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
FailureOr<ValueRange>
getProductProvenanceFactors(TypedValue<FactorGroupType> factorProduct) {
  auto productOp = factorProduct.getDefiningOp<AxisProductOp>();
  if (!productOp) {
    return failure();
  }
  return productOp.getFactors();
}

// Checks factor compatibility and pairwise non-overlap metadata.
bool areFactorsDisjoint(ValueRange factors) {
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
  for (Value factor : factors) {
    auto factorType = dyn_cast<AxisFactorType>(factor.getType());
    assert(factorType && "factor value must have AxisFactorType");
    if (!factorType) {
      return false;
    }
    assert(factorType.getExtent() > 0 && "factor extent must be positive");
    assert(factorType.getStride() > 0 && "factor stride must be positive");

    auto provenance =
        getFactorProvenanceAxis(cast<TypedValue<AxisFactorType>>(factor));
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

// Checks segment pairwise non-overlap metadata.
bool arePairwiseSegmentsDisjoint(Value lhsSegment, Value rhsSegment,
                                 Value lhsProvenanceAxis,
                                 Value rhsProvenanceAxis) {
  auto lhsType = dyn_cast<AxisSegmentType>(lhsSegment.getType());
  auto rhsType = dyn_cast<AxisSegmentType>(rhsSegment.getType());
  assert(lhsType && "segment value must have AxisSegmentType");
  assert(rhsType && "segment value must have AxisSegmentType");
  if (!lhsType || !rhsType) {
    return false;
  }

  Value lhsAxis = lhsProvenanceAxis;
  if (!lhsAxis) {
    auto lhsProvenance =
        getSegmentProvenanceAxis(cast<TypedValue<AxisSegmentType>>(lhsSegment));
    assert(succeeded(lhsProvenance) && "segment must have a provenance axis");
    if (failed(lhsProvenance)) {
      return false;
    }
    lhsAxis = *lhsProvenance;
  }

  Value rhsAxis = rhsProvenanceAxis;
  if (!rhsAxis) {
    auto rhsProvenance =
        getSegmentProvenanceAxis(cast<TypedValue<AxisSegmentType>>(rhsSegment));
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
    auto segmentType = dyn_cast<AxisSegmentType>(segment.getType());
    assert(segmentType && "segment value must have AxisSegmentType");
    if (!segmentType) {
      return false;
    }
    assert(segmentType.getExtent() > 0 && "segment extent must be positive");

    auto provenance =
        getSegmentProvenanceAxis(cast<TypedValue<AxisSegmentType>>(segment));
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
bool areFactorsComplete(Value axis, ValueRange factors) {
  if (factors.empty() || !areFactorsDisjoint(factors)) {
    return false;
  }

  // Given disjointness, we are complete iff all factors belong to the target
  // axis and their extents cover the whole source-axis extent.
  uint64_t product = 1;
  for (Value factor : factors) {
    auto factorType = dyn_cast<AxisFactorType>(factor.getType());
    assert(factorType && "factor value must have AxisFactorType");
    if (!factorType) {
      return false;
    }

    auto provenance =
        getFactorProvenanceAxis(cast<TypedValue<AxisFactorType>>(factor));
    assert(succeeded(provenance) && "factor must have a provenance axis");
    if (failed(provenance) || *provenance != axis) {
      return false;
    }

    product *= static_cast<uint64_t>(getFactorExtent(factor));
  }

  return product == static_cast<uint64_t>(getAxisExtent(axis));
}

// Checks that segments reconstruct the full source axis interval [0, extent).
bool areSegmentsComplete(Value axis, ValueRange segments) {
  if (segments.empty() || !areSegmentsDisjoint(segments)) {
    return false;
  }

  SmallVector<std::pair<uint64_t, uint64_t>> intervals;
  intervals.reserve(segments.size());

  for (Value segment : segments) {
    auto segmentType = dyn_cast<AxisSegmentType>(segment.getType());
    assert(segmentType && "segment value must have AxisSegmentType");
    if (!segmentType) {
      return false;
    }

    auto provenance =
        getSegmentProvenanceAxis(cast<TypedValue<AxisSegmentType>>(segment));
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

  return cursor == static_cast<uint64_t>(getAxisExtent(axis));
}

} // namespace mlir::enzyme::axis
