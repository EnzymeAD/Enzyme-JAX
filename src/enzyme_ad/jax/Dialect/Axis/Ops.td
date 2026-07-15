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

def AxisFactorOp : AxisOp<"factor", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let description = [{
    Factors one canonical axis into leftmost-major factors with static extents.
  }];
  let arguments = (ins
    AnyType:$axis,
    DenseI32ArrayAttr:$factor_extents
  );
  let results = (outs Variadic<AxisFactorType>:$axis_factors);
  let assemblyFormat = "$axis $factor_extents attr-dict `:` type($axis)";
  let hasVerifier = 1;
}

def AxisSegmentOp : AxisOp<"segment", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

def AxisProductOp : AxisOp<"product", [MetadataOpTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let description = [{
    Recombines a leftmost-major factor list into one virtual factor product.
  }];
  let arguments = (ins Variadic<AxisFactorType>:$factors);
  let results = (outs FactorGroupType:$product);
  let assemblyFormat = "$factors attr-dict `:` type($factors)";
  let hasVerifier = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD
