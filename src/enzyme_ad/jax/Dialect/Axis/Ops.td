#ifndef ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect.td"
include "Interfaces.td"
include "Types.td"

def AxisGetAxisOp : AxisOp<"getaxis", [PureMetadataTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

def AxisFactorOp : AxisOp<"factor", [PureMetadataTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
  let description = [{
    Defines one factor of a canonical axis with static extent and stride, encoded by the type.
  }];
  let arguments = (ins AnyType:$axis, I32Attr:$extent, I32Attr:$stride);
  let results = (outs AxisFactorType:$axis_factor);
  let assemblyFormat = "$axis attr-dict `:` type($axis)`<`$extent`,`$stride`>`";
  let hasVerifier = 1;
}

def AxisSegmentOp : AxisOp<"segment", [PureMetadataTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
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

def AxisProductOp : AxisOp<"product", [PureMetadataTrait, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<MaybeTemporaryInterface>]> {
  let description = [{
    Recombines a leftmost-major factor list into one virtual factor product.
    An empty factor list denotes a fully-sharded axis (extent 1) and still
    parses/prints (as "()"), so it can be kept as a placeholder operand.
  }];
  let arguments = (ins Variadic<AxisFactorType>:$factors);
  let results = (outs FactorGroupType:$product);
  let assemblyFormat = "custom<VariadicWithTypes>($factors, type($factors)) attr-dict";
  let hasVerifier = 1;
}

def AxisMapOp : AxisOp<"map", [
  PureMetadataTrait, AttrSizedOperandSegments,
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
  let extraClassDeclaration = [{
    llvm::SmallVector<TypedValue<FactorGroupType>> getTypedMappingLhs();
    llvm::SmallVector<TypedValue<FactorGroupType>> getTypedMappingRhs();
    llvm::SmallVector<std::pair<TypedValue<FactorGroupType>,
                                TypedValue<FactorGroupType>>>
    getTypedMappingPairs();
  }];
  let hasVerifier = 1;
}
#endif // ENZYME_AD_JAX_DIALECT_AXIS_OPS_TD
