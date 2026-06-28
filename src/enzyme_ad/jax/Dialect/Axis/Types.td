#ifndef ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD

include "Dialect.td"
include "Interfaces.td"

class AxisType<string name, string type_mnemonic, list<Trait> traits = []>
    : AxisDialectType<name, type_mnemonic,
                      [DeclareTypeInterfaceMethods<AxisTypeInterface>] #
                          traits>;

// Canonical axis for ranked shape dimensions.
def ShapeAxisType : AxisType<"ShapeAxis", "shape_axis"> {
  let parameters = (ins "::mlir::Type":$shapeType, "unsigned":$axisIndex);
  let assemblyFormat = "`<` $shapeType `,` $axisIndex `>`";
  let extraClassDeclaration = [{
    unsigned getExtent() const;
  }];
}

// Factor of a canonical axis with static extent and stride metadata.
def AxisFactorType : AxisDialectType<"AxisFactor", "axis_factor"> {
  let parameters =
      (ins "::mlir::Type":$axisType, "unsigned":$extent, "unsigned":$stride);
  let assemblyFormat = "`<` $axisType `,` $extent `,` $stride `>`";
}

// Contiguous linear segment of a canonical axis with static extent and offset.
def AxisSegmentType : AxisDialectType<"AxisSegment", "axis_segment"> {
  let parameters =
      (ins "::mlir::Type":$axisType, "unsigned":$extent, "unsigned":$offset);
  let assemblyFormat = "`<` $axisType `,` $extent `,` $offset `>`";
}

// Virtual axis reconstructed from a list of factors.
def FactorGroupType : AxisDialectType<"FactorGroup", "factor_group"> {
  let parameters = (ins "unsigned":$extent);
  let assemblyFormat = "`<` $extent `>`";
}

#endif // ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD
