#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H

#include "Dialect.h"
#include "mlir/IR/Builders.h"

namespace mlir::enzyme::distributed {

using AxisFactor = TypedValue<axis::AxisFactorType>;
using OperatorAxis = TypedValue<axis::FactorGroupType>;
using DerivationTensor = TypedValue<DerivationTensorType>;

// The factors a tensor position reads through `roles`: the factors of its role
// operator axes, in role order. `axisFactors` gives each operator axis's
// factors (GrammarTermOp::getOperatorAxisFactors on a built term), as lists of
// Values or of AxisFactors.
template <typename AxisFactorLists>
auto getRoleFactors(OperatorAxesAttr roles, const AxisFactorLists &axisFactors,
                    bool isOutput, int64_t tensorPosition) {
  SmallVector<typename AxisFactorLists::value_type::value_type> factors;
  for (sdy::DimMappingAttr dim :
       roles.getMappings(isOutput)[tensorPosition].getDimMappings())
    for (int64_t operatorAxis : dim.getFactorIndices())
      llvm::append_range(factors, axisFactors[operatorAxis]);
  return factors;
}

// Builds derivation IR: boundary input tensors, and grammar terms together
// with their output tensor nodes.
//
// Axes go through the axis dialect: a fresh axis is a LogicalMeshAxes root
// used through its whole factor, digits are factors composed onto the root,
// and an operator axis is an axis.product of digits. Every op, axis ops
// included, is created at the builder's insertion point.
//
// An output tensor holds its role digits and the batch digits of the inputs
// it depends on (getDependence), so a term is batched by giving its inputs
// the batch digits.
class DerivationBuilder {
public:
  DerivationBuilder(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  // A fresh axis of `extent`, as its whole factor.
  AxisFactor newAxis(int64_t extent);
  // The digits of `factor` of the given extents, major-first, composed onto
  // its root. The extents must multiply to the factor's extent.
  SmallVector<AxisFactor> split(AxisFactor factor, ArrayRef<int64_t> extents);
  // The operator axis reading `factors`, major-first.
  OperatorAxis product(ArrayRef<AxisFactor> factors);

  // A boundary input tensor over `factors`.
  DerivationTensor input(Type elementType, ArrayRef<AxisFactor> factors);

  // A term over `inputs` and the operator axes `axes`, read through `roles`,
  // with one output tensor per output element type.
  SmallVector<DerivationTensor> term(StringRef kind, ValueRange inputs,
                                     ArrayRef<OperatorAxis> axes,
                                     OperatorAxesAttr roles,
                                     DictionaryAttr params,
                                     TypeRange outputElementTypes);

private:
  OpBuilder &builder;
  Location loc;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H
