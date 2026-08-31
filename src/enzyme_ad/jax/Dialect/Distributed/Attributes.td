#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD

include "mlir/IR/BuiltinAttributes.td"

include "Dialect.td"

// Per-tensor sharding metadata keyed by partitioning-axis indices.
// - dim_partitioning_axes: one DenseI64ArrayAttr per tensor dim, where each
//   entry stores zero or more indices into the owning op's partitioning_axes.
// - unreduced_axes: tensor-dim indices that remain unreduced.
def IndexedTensorShardingAttr : AttrDef<DistributedDialect, "IndexedTensorSharding"> {
  let summary = "Tensor sharding with integer partitioning-axis indices";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding";

  let parameters = (ins
    ArrayRefParameter<"DenseI64ArrayAttr">:$dim_partitioning_axes,
    "DenseI64ArrayAttr":$unreduced_axes
  );

  let assemblyFormat = [{
    `<` `dim_partitioning_axes` `=` `[` $dim_partitioning_axes `]` `:`
    `unreduced_axes` `=` $unreduced_axes `>`
  }];
}

// One IndexedTensorShardingAttr per operand/result value.
def IndexedTensorShardingPerValueAttr : AttrDef<DistributedDialect, "IndexedTensorShardingPerValue"> {
  let summary = "Indexed tensor sharding for each value";
  let cppNamespace = "::mlir::enzyme::distributed";
  let mnemonic = "indexed_tensor_sharding_per_value";

  let parameters = (ins
    ArrayRefParameter<"IndexedTensorShardingAttr">:$shardings
  );

  let assemblyFormat = [{
    `<` `[` $shardings `]` `>`
  }];
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ATTRIBUTES_TD