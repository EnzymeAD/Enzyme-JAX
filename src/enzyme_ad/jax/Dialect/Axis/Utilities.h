#ifndef ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
#define ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H

#include "Dialect.h"

namespace mlir::enzyme::axis {

// Returns the static extent for any canonical axis SSA value.
int getAxisExtent(::mlir::Value axis);

// Returns the static extent for any factor SSA value.
int getFactorExtent(::mlir::Value factor);

// Returns the defining op for a canonical axis SSA value.
::mlir::FailureOr<::mlir::Operation *> getAxisProvenanceOp(::mlir::Value axis);

// Resolves the source canonical axis used to produce a factor value.
::mlir::FailureOr<::mlir::Value>
getFactorProvenanceAxis(::mlir::TypedValue<AxisFactorType> factor);

// Returns the factor list used to build a factor-group SSA value.
::mlir::FailureOr<::mlir::ValueRange>
getGroupProvenanceFactors(::mlir::TypedValue<FactorGroupType> factorGroup);

// Checks that factors are pairwise non-overlapping for one source axis.
bool areFactorsDisjoint(::mlir::ValueRange factors);

// Checks that factors cover an axis exactly and therefore are disjoint.
bool areFactorsComplete(::mlir::Value axis, ::mlir::ValueRange factors);

} // namespace mlir::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_UTILITIES_H
