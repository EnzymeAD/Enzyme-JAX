#ifndef ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H

#include "Dialect.h"

namespace mlir::enzyme::axis {

// Returns the static extent for any canonical axis SSA value.
int getAxisExtent(::mlir::Value axis);

// Returns the static extent for any factor SSA value.
int getFactorExtent(::mlir::Value factor);

// Returns the static extent for any segment SSA value.
int getSegmentExtent(::mlir::Value segment);

// Returns the defining op for a canonical axis SSA value.
::mlir::FailureOr<::mlir::Operation *> getAxisProvenanceOp(::mlir::Value axis);

// Resolves the source canonical axis used to produce a factor value.
::mlir::FailureOr<::mlir::Value>
getFactorProvenanceAxis(::mlir::TypedValue<AxisFactorType> factor);

// Resolves the source canonical axis used to produce a segment value.
::mlir::FailureOr<::mlir::Value>
getSegmentProvenanceAxis(::mlir::TypedValue<AxisSegmentType> segment);

// Returns the factor list used to build a factor-product SSA value.
::mlir::FailureOr<::mlir::ValueRange>
getProductProvenanceFactors(::mlir::TypedValue<FactorGroupType> factorProduct);

// Checks that factors are pairwise non-overlapping for one source axis.
bool arePairwiseFactorsDisjoint(::mlir::Value lhsFactor,
                                ::mlir::Value rhsFactor,
                                ::mlir::Value lhsProvenanceAxis = nullptr,
                                ::mlir::Value rhsProvenanceAxis = nullptr);

// Checks that factors are pairwise non-overlapping for one source axis.
bool areFactorsDisjoint(::mlir::ValueRange factors);

// Checks that segment intervals are pairwise non-overlapping per source axis.
bool arePairwiseSegmentsDisjoint(::mlir::Value lhsSegment,
                                 ::mlir::Value rhsSegment,
                                 ::mlir::Value lhsProvenanceAxis = nullptr,
                                 ::mlir::Value rhsProvenanceAxis = nullptr);

// Checks that segment intervals are pairwise non-overlapping per source axis.
bool areSegmentsDisjoint(::mlir::ValueRange segments);

// Checks that factors cover an axis exactly and therefore are disjoint.
bool areFactorsComplete(::mlir::Value axis, ::mlir::ValueRange factors);

// Checks that segments exactly cover [0, axis extent) with no overlap or gaps.
bool areSegmentsComplete(::mlir::Value axis, ::mlir::ValueRange segments);

} // namespace mlir::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
