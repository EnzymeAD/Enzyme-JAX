#ifndef ENZYME_DISTRIBUTED_DIALECT_TYPES_H
#define ENZYME_DISTRIBUTED_DIALECT_TYPES_H

include "Dialect.td"
include "Interfaces.td"
include "src/enzyme_ad/jax/Dialect/Axis/Interfaces.td"

class DistributedAxisType<string name, string type_mnemonic,
                          list<Trait> traits = []>
    : DistributedType<name, type_mnemonic,
                      [DeclareTypeInterfaceMethods<AxisTypeInterface>] #traits>;

// A natural axis defined by the hardware topology.
def PhysicalCommAxisType
    : DistributedAxisType<"PhysicalCommAxis", "physical_comm_axis"> {
  let parameters = (ins "unsigned" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// Axis of a logical mesh where hardware details are abstracted away.
def LogicalMeshAxisType
    : DistributedAxisType<"LogicalMeshAxis", "logical_mesh_axis"> {
  let parameters = (ins "unsigned" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// A standalone axis type used to represent replication.
def ReplicationAxisType
    : DistributedAxisType<"ReplicationAxis", "replication_axis"> {
  let parameters = (ins "unsigned" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// Wraps a value type that may be synchronized at a later point.
def AsynchHandleType : DistributedType<"AsynchHandle", "asynch_handle"> {
  let summary = "Handle to a value that can be synchronized later";
  let parameters = (ins "::mlir::Type" : $valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H