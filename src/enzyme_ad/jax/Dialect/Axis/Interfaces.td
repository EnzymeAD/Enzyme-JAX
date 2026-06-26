#ifndef ENZYME_AD_JAX_DIALECT_AXIS_INTERFACES_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_INTERFACES_TD

include "mlir/IR/OpBase.td"

def AxisTypeInterface : TypeInterface<"AxisTypeInterface"> {
  let cppNamespace = "::mlir::enzyme::axis";
  let description = [{
    Interface for canonical axis types with statically decidable equivalence.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the compile-time extent of this canonical axis.}],
      "unsigned", "extent", (ins), "",
      [{
        return $_type.getExtent();
      }]>,
    InterfaceMethod<
      [{Returns true when two SSA values of this axis type are equivalent.}],
      "bool", "aliases",
      (ins "::mlir::Value":$ax1, "::mlir::Value":$ax2)
    >
  ];
}

#endif // ENZYME_AD_JAX_DIALECT_AXIS_INTERFACES_TD
