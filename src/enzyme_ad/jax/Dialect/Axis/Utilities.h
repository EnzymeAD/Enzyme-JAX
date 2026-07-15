#ifndef ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H

#include "Dialect.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme::axis {
template <typename T> using TypedValueArrayRef = llvm::ArrayRef<TypedValue<T>>;

// Utility for casting with better error reporting
template <typename T>
TypedValue<T> castTypedValue(::mlir::Value value, llvm::StringRef expectedType);

// Utility for casting a list of values with better error reporting
template <typename T>
llvm::SmallVector<TypedValue<T>>
castTypedValueList(ValueRange values, llvm::StringRef expectedType);

template <typename T>
TypedValue<T> castTypedValue(::mlir::Value value,
                             llvm::StringRef expectedType) {
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

template <typename T>
llvm::SmallVector<TypedValue<T>>
castTypedValueList(ValueRange values, llvm::StringRef expectedType) {
  llvm::SmallVector<TypedValue<T>> typedValues;
  typedValues.reserve(values.size());
  for (Value value : values) {
    typedValues.push_back(castTypedValue<T>(value, expectedType));
  }
  return typedValues;
}

// Returns the static extent for any canonical axis SSA value.
int getAxisExtent(::mlir::TypedValue<AxisTypeInterface> axis);

// Returns the static extent for any factor SSA value.
int getFactorExtent(::mlir::TypedValue<AxisFactorType> factor);

// Returns the static extent for any segment SSA value.
int getSegmentExtent(::mlir::TypedValue<AxisSegmentType> segment);

// Returns the defining op for a canonical axis SSA value.
::mlir::FailureOr<::mlir::Operation *> getAxisProvenanceOp(::mlir::Value axis);

// Resolves the source canonical axis used to produce a factor value.
::mlir::FailureOr<::mlir::Value>
getFactorProvenanceAxis(::mlir::TypedValue<AxisFactorType> factor);

// Resolves the source canonical axis used to produce a segment value.
::mlir::FailureOr<::mlir::Value>
getSegmentProvenanceAxis(::mlir::TypedValue<AxisSegmentType> segment);

// Returns the factor list used to build a factor-product SSA value.
::mlir::FailureOr<llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>>
getProductProvenanceFactors(::mlir::TypedValue<FactorGroupType> factorProduct);

// Returns the product of extents for a factor-product SSA value.
::mlir::FailureOr<uint64_t>
getFactorGroupExtent(::mlir::TypedValue<FactorGroupType> factorProduct);

// Checks that factors are pairwise non-overlapping for one source axis.
bool arePairwiseFactorsDisjoint(::mlir::Value lhsFactor,
                                ::mlir::Value rhsFactor,
                                ::mlir::Value lhsProvenanceAxis = nullptr,
                                ::mlir::Value rhsProvenanceAxis = nullptr);

// Checks that factors are pairwise non-overlapping for one source axis.
bool areFactorsDisjoint(TypedValueArrayRef<AxisFactorType> factors);

// Checks that segment intervals are pairwise non-overlapping per source axis.
bool arePairwiseSegmentsDisjoint(::mlir::Value lhsSegment,
                                 ::mlir::Value rhsSegment,
                                 ::mlir::Value lhsProvenanceAxis = nullptr,
                                 ::mlir::Value rhsProvenanceAxis = nullptr);

// Checks that segment intervals are pairwise non-overlapping per source axis.
bool areSegmentsDisjoint(::mlir::ValueRange segments);

// Returns true when two factor lists cover the same index space modulo
// permutation order. This checks multiset equality over factor metadata and
// provenance-axis equivalence, but does not require matching list order.
bool areFactorIndexSpacesEqual(TypedValueArrayRef<AxisFactorType> lhsFactors,
                               TypedValueArrayRef<AxisFactorType> rhsFactors);

// Checks that factors cover an axis exactly and therefore are disjoint.
bool areFactorsComplete(::mlir::Value axis,
                        TypedValueArrayRef<AxisFactorType> factors);

// Checks that segments exactly cover [0, axis extent) with no overlap or gaps.
bool areSegmentsComplete(::mlir::Value axis, ::mlir::ValueRange segments);

// Small utlity for extracting all factors from one or more factor groups.
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
flattenGroupsToFactors(TypedValueArrayRef<FactorGroupType> factorGroups);

// Shortcut for calling distjoint on a flattened factor list
bool areFactorGroupsDisjoint(TypedValueArrayRef<FactorGroupType> factorGroups);

// Given a shapeType with a known rank, returns a list of canonical axes for
// each dimension of that shape. Use a builder without an insertion point and
// an unknown location for ephemeral axes.
llvm::SmallVector<::mlir::TypedValue<AxisTypeInterface>>
createAxesForRankedShape(::mlir::Type shapeType, ::mlir::OpBuilder &builder,
                         ::mlir::Location loc);
// Creates a single factor for each axis, with full extent and stride 1.
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(::mlir::ValueRange axes, ::mlir::OpBuilder &builder,
                  ::mlir::Location loc);
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
viewAxesAsFactors(TypedValueArrayRef<AxisTypeInterface> axes,
                  ::mlir::OpBuilder &builder, ::mlir::Location loc);

// Creates a full major-first factorization for one axis from major-first
// extents. Strides are inferred
llvm::SmallVector<::mlir::TypedValue<AxisFactorType>>
factorAxisByExtents(::mlir::Value axis, llvm::ArrayRef<int32_t> extents,
                    ::mlir::OpBuilder &builder, ::mlir::Location loc);
} // namespace mlir::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
