#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H

#include "Dialect.h"
#include "mlir/IR/Builders.h"

namespace mlir::enzyme::distributed {

using AxisFactor = TypedValue<axis::AxisFactorType>;
using OperatorAxis = TypedValue<axis::FactorGroupType>;
using DerivationTensor = TypedValue<DerivationTensorType>;

// Builds derivation IR: boundary input tensors, and grammar terms together
// with their output tensor nodes.
//
// Axes go through the axis dialect: a fresh axis is a LogicalMeshAxes root
// used through its whole factor, digits are factors composed onto the root,
// and an operator axis is an axis.product of digits. Every op, axis ops
// included, is created at the builder's insertion point.
//
// An output tensor holds its role digits and the batch digits of every
// enclosing BatchScope. An output invariant in some batch digit is built
// outside that digit's scope.
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

  // A boundary input tensor over `factors`. Batch scopes don't apply: an
  // input carries a batch digit only when it is given one.
  DerivationTensor input(Type elementType, ArrayRef<AxisFactor> factors);

  // A term over `inputs` and the operator axes `axes`, read through `roles`,
  // with one output tensor per output element type.
  SmallVector<DerivationTensor> term(StringRef kind, ValueRange inputs,
                                     ArrayRef<OperatorAxis> axes,
                                     OperatorAxesAttr roles,
                                     DictionaryAttr params,
                                     TypeRange outputElementTypes);

private:
  friend class BatchScope;

  OpBuilder &builder;
  Location loc;
  // The batch factors of each enclosing BatchScope, outermost first.
  SmallVector<SmallVector<AxisFactor>> batchScopes;
};

// Adds batch factors to the outputs of every term built through `builder`
// during its lifetime. Scopes nest, and their factors add up. A batch factor
// must not be a digit of a term's operator axes.
class BatchScope {
public:
  BatchScope(DerivationBuilder &builder, ArrayRef<AxisFactor> factors);
  ~BatchScope();
  BatchScope(const BatchScope &) = delete;
  BatchScope &operator=(const BatchScope &) = delete;

private:
  DerivationBuilder &builder;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_IR_H
