#include "Utilities.h"

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
// of a shared source axis. Assumes both factors are derived from the same
// source axis.
static bool arePairwiseFactorsDisjoint(const AxisFactorType &f1,
                                       const AxisFactorType &f2) {
  unsigned majorStride = f1.getStride();
  unsigned majorExtent = f1.getExtent();
  unsigned minorStride = f2.getStride();
  unsigned minorExtent = f2.getExtent();
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

// Returns the factor list used to build a factor-group SSA value.
FailureOr<ValueRange>
getGroupProvenanceFactors(TypedValue<FactorGroupType> factorGroup) {
  auto groupOp = factorGroup.getDefiningOp<AxisGroupOp>();
  if (!groupOp) {
    return failure();
  }
  return groupOp.getFactors();
}

// Checks factor compatibility and pairwise non-overlap metadata.
bool areFactorsDisjoint(ValueRange factors) {
  if (factors.empty()) {
    return true;
  }

  assert(factors.size() < 100 &&
         "factor disjointness uses quadratic pairwise checks");

  struct FactorInfo {
    AxisFactorType factorType;
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
    cachedFactors.push_back({factorType, *provenance});
  }

  for (size_t i = 0; i < cachedFactors.size(); ++i) {
    for (size_t j = i + 1; j < cachedFactors.size(); ++j) {
      const auto &[lhsType, lhsAxis] = cachedFactors[i];
      const auto &[rhsType, rhsAxis] = cachedFactors[j];

      if (!areAxesEquivalent(lhsAxis, rhsAxis)) {
        continue;
      }
      if (!arePairwiseFactorsDisjoint(lhsType, rhsType)) {
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

} // namespace mlir::enzyme::axis
