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
      [{Returns true when two  SSA values of this axis type are equivalent.
}],
      "bool", "aliases",
      (ins "::mlir::Value":$ax1, "::mlir::Value":$ax2)
    >
  ];
}

def MaybeTemporaryInterface : OpInterface<"MaybeTemporaryInterface"> {
  let cppNamespace = "::mlir::enzyme::axis";
  let description = [{
    Interface for operations that may produce temporary values. For examples,
    some axis logic creates equivalent formulations / factors that may or may
    not ever be added to the actual IR. The main purpose of this interface is
    to mark such operations as participants to a recursive insertion sweep
    that places such temporaries into a provide MLIR graph region when
    it is decided they should be materialized into the IR.
  }];
  // We provide a default recursion that looks at all SSA operands,
  // but provide this as an op interface in case there are edge cases
  // with e.g. symbol dependencies requiring special handling.
  let methods = [
    InterfaceMethod<
      [{isManifested == present in an MLIR block}], "bool", "isManifested", (ins), "",
      [{
        return $_op->getBlock() != nullptr;
      }]>,
    InterfaceMethod<
      [{Recursively places this op and its operands into the provided graph block.}],
      "void", "materialize", (ins "::mlir::Block &":$block), "",
      [{
        ::mlir::enzyme::axis::materializeMaybeTemporaryOp($_op, block);
      }]>
  ];
}

#endif // ENZYME_AD_JAX_DIALECT_AXIS_INTERFACES_TD
