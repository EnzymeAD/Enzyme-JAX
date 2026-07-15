#include "Utilities.h"

#include <algorithm>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::axis {

template <typename T>
static TypedValue<T> castTypedValue(Value value, llvm::StringRef expectedType) {
  if (auto typed = dyn_cast<TypedValue<T>>(value)) {
    return typed;
  }

  std::string typeString;
  llvm::raw_string_ostream os(typeString);
  value.getType().print(os);
  os.flush();
  llvm::errs() << "castTypedValue failed: expected " << expectedType
               << ", got value type " << typeString << "\n";
  llvm::report_fatal_error("invalid typed value cast");
}

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
FailureOr<ValueRange>
getProductProvenanceFactors(TypedValue<FactorGroupType> factorProduct) {
  auto productOp = factorProduct.getDefiningOp<AxisProductOp>();
  if (!productOp) {
    return failure();
  }
  return productOp.getFactors();
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
    auto factorTyped = castTypedValue<AxisFactorType>(factor, "AxisFactorType");
    auto factorType = factorTyped.getType();
    assert(factorType.getExtent() > 0 && "factor extent must be positive");
    assert(factorType.getStride() > 0 && "factor stride must be positive");

    auto provenance = getFactorProvenanceAxis(factorTyped);
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

// Compares two factor lists as index-space descriptors, ignoring ordering.
// This is multiset equality over (extent, stride, provenance-axis equivalence)
// and is intentionally permutation-invariant.
bool areFactorIndexSpacesEqual(ValueRange lhsFactors, ValueRange rhsFactors) {
  if (lhsFactors.size() != rhsFactors.size()) {
    return false;
  }

  struct FactorInfo {
    Value provenance;
    unsigned extent;
    unsigned stride;
  };

  auto buildFactorInfo = [](ValueRange factors,
                            SmallVectorImpl<FactorInfo> &out) -> bool {
    out.clear();
    out.reserve(factors.size());
    for (Value factor : factors) {
      auto factorTyped =
          castTypedValue<AxisFactorType>(factor, "AxisFactorType");
      auto factorType = factorTyped.getType();
      auto provenance = getFactorProvenanceAxis(factorTyped);
      if (failed(provenance)) {
        return false;
      }
      out.push_back(
          {*provenance, factorType.getExtent(), factorType.getStride()});
    }
    return true;
  };

  SmallVector<FactorInfo> lhsInfo;
  SmallVector<FactorInfo> rhsInfo;
  if (!buildFactorInfo(lhsFactors, lhsInfo) ||
      !buildFactorInfo(rhsFactors, rhsInfo)) {
    return false;
  }

  SmallVector<bool> rhsMatched(rhsInfo.size(), false);
  for (const FactorInfo &lhs : lhsInfo) {
    bool foundMatch = false;
    for (auto [rhsIndex, rhs] : llvm::enumerate(rhsInfo)) {
      if (rhsMatched[rhsIndex]) {
        continue;
      }
      if (lhs.extent != rhs.extent || lhs.stride != rhs.stride) {
        continue;
      }
      if (!areAxesEquivalent(lhs.provenance, rhs.provenance)) {
        continue;
      }
      rhsMatched[rhsIndex] = true;
      foundMatch = true;
      break;
    }
    if (!foundMatch) {
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
    auto segmentType = segmentTyped.getType();
    assert(segmentType.getExtent() > 0 && "segment extent must be positive");

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
bool areFactorsComplete(Value axis, ValueRange factors) {
  if (factors.empty() || !areFactorsDisjoint(factors)) {
    return false;
  }

  // Given disjointness, we are complete iff all factors belong to the target
  // axis and their extents cover the whole source-axis extent.
  uint64_t product = 1;
  for (Value factor : factors) {
    auto factorTyped = castTypedValue<AxisFactorType>(factor, "AxisFactorType");

    auto provenance = getFactorProvenanceAxis(factorTyped);
    assert(succeeded(provenance) && "factor must have a provenance axis");
    if (failed(provenance) || *provenance != axis) {
      return false;
    }

    product *= static_cast<uint64_t>(getFactorExtent(factorTyped));
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

llvm::SmallVector<::mlir::Value>
flattenGroupsToFactors(::mlir::ValueRange factorGroups) {
  llvm::SmallVector<::mlir::Value> flattenedFactors;
  for (auto group : factorGroups) {
    auto typedGroup =
        cast<::mlir::TypedValue<::mlir::enzyme::axis::FactorGroupType>>(group);
    auto factors = getProductProvenanceFactors(typedGroup);
    if (failed(factors)) {
      llvm::report_fatal_error(
          "flattenGroupsToFactors failed to get factors from FactorGroupType");
    }
    flattenedFactors.append(factors->begin(), factors->end());
  }
  return flattenedFactors;
}

bool areFactorGroupsDisjoint(::mlir::ValueRange factorGroups) {
  auto flattenedFactors = flattenGroupsToFactors(factorGroups);
  return areFactorsDisjoint(ValueRange(flattenedFactors));
}

llvm::SmallVector<::mlir::Value>
createAxesForRankedShape(::mlir::Type shapeType, ::mlir::OpBuilder &builder,
                         ::mlir::Location loc) {
  auto rankedShapeType = cast<ShapedType>(shapeType);
  auto type_attr = TypeAttr::get(rankedShapeType);
  int rank = rankedShapeType.getRank();
  llvm::SmallVector<::mlir::Value> axes;
  axes.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto rank_attr = builder.getI32IntegerAttr(i);
    auto axis = builder.create<AxisGetAxisOp>(loc, type_attr, rank_attr);
    axes.push_back(axis);
  }
  return axes;
}

llvm::SmallVector<::mlir::Value> viewAxesAsFactors(::mlir::ValueRange axes,
                                                   ::mlir::OpBuilder &builder,
                                                   ::mlir::Location loc) {
  llvm::SmallVector<::mlir::Value> factors;
  factors.reserve(axes.size());
  for (auto axis : axes) {
    auto axis_typed =
        castTypedValue<AxisTypeInterface>(axis, "AxisTypeInterface");
    int extent = getAxisExtent(axis_typed);
    auto factor =
        builder.create<AxisFactorOp>(loc, axis, ArrayRef<int32_t>{extent});
    factors.push_back(factor.getResult(0));
  }
  return factors;
}
} // namespace mlir::enzyme::axis
