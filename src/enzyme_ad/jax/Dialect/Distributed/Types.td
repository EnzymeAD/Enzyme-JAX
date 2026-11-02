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
  let parameters = (ins "::mlir::enzyme::axis::AxisExtentT" : $extent);
  let assemblyFormat = "`<` $extent `>`";
}

// A natural axis defined by the hardware topology.
// ID stride is for conversion between device IDs to axis coordinates. It is
// bounded by the product of the other physical axes' extents in the mesh
// (a device count), so it stays 32-bit.
def PhysicalCommAxisType
    : DistributedAxisType<"PhysicalCommAxis", "physical_comm_axis"> {
  let parameters = (ins "::mlir::enzyme::axis::AxisExtentT" : $extent, "unsigned" : $idStride);
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

// A tensor of derivation IR: an element type and, on its distributed.Tensor
// node, a flat set of axis factors. It has no rank or layout; how an operator
// reads it is given by the roles of the GrammarTerm that takes it.
def DerivationTensorType : DistributedType<"DerivationTensor", "tensor"> {
  let summary = "Layout-free tensor of derivation IR";
  let parameters = (ins "::mlir::Type" : $elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// One output of a GrammarTerm, before a distributed.Tensor node gives it its
// axes.
def TermOutputType : DistributedType<"TermOutput", "term_output"> {
  let summary = "Output of a grammar term, consumed by its tensor node";
}

// Wraps a value type that may be synchronized at a later point.
def AsynchHandleType : DistributedType<"AsynchHandle", "asynch_handle"> {
  let summary = "Handle to a value that can be synchronized later";
  let parameters = (ins "::mlir::Type" : $valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H