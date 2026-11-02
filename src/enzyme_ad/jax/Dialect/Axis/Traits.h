#ifndef ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
#define ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::enzyme::axis {

// Marks axis-algebra ops: compile-time metadata about axes, independent of any
// runtime value. It places no constraint on where an op lives:
// - In pipeline IR, axis ops sit in the module body, ahead of any scheduled
//   op. Pipeline passes rely on it: ReorderSchedule's
//   reorderBlockByAdmission keeps the metadata before the first anchor op as
//   a fixed prefix. Nothing verifies this placement, so pipeline code must
//   keep creating axis ops through ModuleScopeGuard.
// - In derivation IR, they sit inside the derivation's func.func container.
template <typename ConcreteType>
class MetadataTrait : public OpTrait::TraitBase<ConcreteType, MetadataTrait> {};

} // namespace mlir::OpTrait::enzyme::axis

#endif // ENZYME_AD_JAX_DIALECT_AXIS_TRAITS_H
