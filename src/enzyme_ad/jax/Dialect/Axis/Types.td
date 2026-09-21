#ifndef ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD
#define ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD

include "Dialect.td"
include "Interfaces.td"

class AxisType<string name, string type_mnemonic, list<Trait> traits = []>
    : AxisDialectType<name, type_mnemonic,
                      [DeclareTypeInterfaceMethods<AxisTypeInterface>] #traits>;

// Canonical axis for ranked shape dimensions.
def ShapeAxisType : AxisType<"ShapeAxis", "shape_axis"> {
  let parameters = (ins "::mlir::Type" : $shapeType, "unsigned" : $axisIndex);
  let assemblyFormat = "`<` $shapeType `,` $axisIndex `>`";
  let extraClassDeclaration = [{ AxisExtentT getExtent() const; }];
}

// Factor of a canonical axis with static extent and stride metadata.
def AxisFactorType : AxisDialectType<"AxisFactor", "axis_factor"> {
  let parameters = (ins "::mlir::Type"
                    : $axisType, "::mlir::enzyme::axis::AxisExtentT"
                    : $extent, "::mlir::enzyme::axis::AxisStrideT"
                    : $stride);
  let assemblyFormat = "`<` $axisType `,` $extent `,` $stride `>`";
}

// Contiguous linear segment of a canonical axis with static extent and
// offset. The offset is a position within the source axis, so it uses the
// extent width.
def AxisSegmentType : AxisDialectType<"AxisSegment", "axis_segment"> {
  let parameters = (ins "::mlir::Type"
                    : $axisType, "::mlir::enzyme::axis::AxisExtentT"
                    : $extent, "::mlir::enzyme::axis::AxisExtentT"
                    : $offset);
  let assemblyFormat = "`<` $axisType `,` $extent `,` $offset `>`";
}

// Virtual axis reconstructed from a list of factors. Its extent is the
// product of those factors' extents (see AxisProductOp).
def FactorGroupType : AxisDialectType<"FactorGroup", "factor_group"> {
  let parameters = (ins "::mlir::enzyme::axis::AxisExtentT" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// Opaque SSA value representing an axis.map declaration.
def AxisMapType : AxisDialectType<"AxisMap", "map">;

#endif // ENZYME_AD_JAX_DIALECT_AXIS_TYPES_TD
