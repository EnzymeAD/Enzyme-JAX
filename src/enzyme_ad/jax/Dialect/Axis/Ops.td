#ifndef ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect.td"
include "Interfaces.td"
include "Types.td"

def AxisGetAxisOp : AxisOp<"getaxis", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let description = [{
    Materializes a canonical ShapeAxis value for one ranked, static shape index.
  }];
  let arguments = (ins
    TypeAttr:$shape_type,
    I32Attr:$axis_index
  );
  let results = (outs ShapeAxisType:$axis);
  let assemblyFormat = "$shape_type $axis_index attr-dict";
  let hasVerifier = 1;
}

def AxisFactorOp : AxisOp<"factor", [MetadataOpTrait, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
  let description = [{
    Defines one factor of a canonical axis with static extent and stride, encoded by the type
  }];
let arguments = (ins AnyType : $axis);
let results = (outs AxisFactorType : $axis_factor);
let assemblyFormat = "$axis attr-dict `:` functional-type(operands, results)";
  let builders = [
    OpBuilder<(ins "Value":$axis, "int32_t":$extent, "int32_t":$stride), [{
      auto factorType = AxisFactorType::get(
          $_builder.getContext(), axis.getType(),
          static_cast<unsigned>(extent), static_cast<unsigned>(stride));
      build($_builder, $_state, factorType, axis);
    }]>
  ];
  let hasVerifier = 1;
}

def AxisSegmentOp : AxisOp<"segment", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
  let description = [{
    Splits one canonical axis into left-to-right contiguous linear segments.
    Segment result index is value-ordered: lower result indices denote lower
    axis values. Concretely, result #0 starts at offset 0 and each following
    result starts at the cumulative extent of all prior segments.
  }];
  let arguments = (ins
    AnyType:$axis,
    DenseI32ArrayAttr:$segment_extents
  );
  let results = (outs Variadic<AxisSegmentType>:$axis_segments);
  let assemblyFormat = "$axis $segment_extents attr-dict `:` type($axis)";
  let hasVerifier = 1;
}

def AxisProductOp : AxisOp<"product", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
  let description = [{
    Recombines a leftmost-major factor list into one virtual factor product.
  }];
  let arguments = (ins Variadic<AxisFactorType>:$factors);
  let results = (outs FactorGroupType:$product);
  let assemblyFormat = "$factors attr-dict `:` type($factors)";
  let hasVerifier = 1;
  }

  def AxisMapOp : AxisOp<"map", [
    MetadataOpTrait, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MaybeTemporaryInterface>
  ]> {
    let description = [{
    Declares a mapping between groups of axis products.
    This op is intended to replace per-collective embedded mapping lists in
    distributed collectives.
  }];
  let arguments = (ins
    Variadic<FactorGroupType>:$mapping_lhs,
    Variadic<FactorGroupType>:$mapping_rhs
  );
  let results = (outs AxisMapType:$map);
  let assemblyFormat = [{
    $mapping_lhs `to` $mapping_rhs
    attr-dict
    `:` `[` type($mapping_lhs) `]` `[` type($mapping_rhs) `]`
  }];
  let hasVerifier = 1;
}
#endif // ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD
