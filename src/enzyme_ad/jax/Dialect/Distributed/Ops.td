#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"
include "src/enzyme_ad/jax/Dialect/Axis/Dialect.td"

defvar TensorType = HLO_Tensor;
defvar ShardedTensorType =
    HLO_Tensor; // Subject to change, for now shardy just adds attributes on top
                // of a regular tensor.

// Marker for distributed metadata ops. Includes builtin Pure.
def PureDistributedMetadata : TraitList<[MetadataOpTrait]>;

def GetPhysicalAxisOp
    : DistributedOp<"GetPhysicalAxis", [PureDistributedMetadata]> {
  let description = [{Resolves a module -
                      level physical axis symbol into an SSA axis value.}];
  let arguments = (ins FlatSymbolRefAttr : $physical_axis);
  let results = (outs PhysicalCommAxisType : $axis);
  let assemblyFormat = "$physical_axis attr-dict `:` type($axis)";
  let hasVerifier = 1;
}

def MeshComputationOp
    : DistributedOp<"MeshComputation",
                    [HasOnlyGraphRegion, NoTerminator, Symbol, SymbolTable]> {
  let description = [{
        Represents a closed distributed program. Contains a mix non-computational SSA values
        (operations that represent a static computation, like axis algebra) and distributed
        functions. Distributed functions can call into normal single-device MLIR functions
        outside of this region, but all functions with internal distributed logic (communication)
        must be fully contained in this region, allowing us to reason about the whole call graph
        for optimization decisions.
  }];
  let arguments = (ins SymbolNameAttr
                   : $computation_name, FlatSymbolRefAttr
                   : $physical_mesh);
  let regions = (region SizedRegion<1> : $body);
  let assemblyFormat =
      "$computation_name `mesh` $physical_mesh $body attr-dict";
  let hasVerifier = 1;
}

def DistributedYieldOp : DistributedOp<"DistributedYield", [Terminator]> {
  let description = [{
        Terminator for distributed functions. Yields local values back to the caller.
  }];
  let arguments = (ins Variadic<ShardedTensorType> : $returns);
  let assemblyFormat = "$returns type($returns) attr-dict";
}
#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD