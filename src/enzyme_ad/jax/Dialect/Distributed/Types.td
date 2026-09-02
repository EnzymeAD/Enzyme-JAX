#ifndef ENZYME_DISTRIBUTED_DIALECT_TYPES_H
#define ENZYME_DISTRIBUTED_DIALECT_TYPES_H

include "Dialect.td"
include "Interfaces.td"
include "src/enzyme_ad/jax/Dialect/Axis/Interfaces.td"

class DistributedAxisType<string name, string type_mnemonic,
                          list<Trait> traits = []>
    : DistributedType<name, type_mnemonic,
                      [DeclareTypeInterfaceMethods<AxisTypeInterface>] #traits>;

// Common shape for axis types that are parameterized solely by an extent.
class DistributedBasicAxisType<string name, string type_mnemonic,
                               list<Trait> traits = []>
    : DistributedAxisType<name, type_mnemonic, traits> {
  let parameters = (ins "unsigned" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// A natural axis defined by the hardware topology.
// ID stride is for conversion between device IDs to
// axis coordinates.
def PhysicalCommAxisType
    : DistributedAxisType<"PhysicalCommAxis", "physical_comm_axis"> {
  let parameters = (ins "unsigned" : $extent, "unsigned" : $idStride);
  let assemblyFormat = "`<` $extent `,` $idStride `>`";
}

// Axis of a logical mesh where hardware details are abstracted away.
def LogicalMeshAxisType
    : DistributedBasicAxisType<"LogicalMeshAxis", "logical_mesh_axis">;

// A standalone axis type used to represent replication.
def ReplicationAxisType
    : DistributedBasicAxisType<"ReplicationAxis", "replication_axis">;

// An axis type used to indicate that part of the "parallelism"
// should instead be placed entirely within a local device.
def DeviceLocalAxisType
    : DistributedBasicAxisType<"DeviceLocalAxis", "device_local_axis">;

// Wraps a value type that may be synchronized at a later point.
def AsynchHandleType : DistributedType<"AsynchHandle", "asynch_handle"> {
  let summary = "Handle to a value that can be synchronized later";
  let parameters = (ins "::mlir::Type" : $valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H