#ifndef ENZYME_DISTRIBUTED_DIALECT_TYPES_H
#define ENZYME_DISTRIBUTED_DIALECT_TYPES_H

include "Dialect.td"

// A natural axis defined by the hardware topology.
def PhysicalCommAxisType : DistributedType<"PhysicalCommAxis", "physical_comm_axis">;
// Includes physical axis factors and products of other logical axis.
def LogicalCommAxisType : DistributedType<"LogicalCommAxis", "logical_comm_axis">;
// Small (unecessary?) type wrapping products of logical axis and "reduce" for sharding annotations.
def ShardingAxisType : DistributedType<"Sharding", "sharding">;

def PhysicalMeshType : DistributedType<"PhysicalMesh", "physical_mesh">;
def LogicalMeshType : DistributedType<"LogicalMesh", "logical_mesh">;

def CollectiveTokenType : DistributedType<"CollectiveToken", "collective_token">;
// TODO: split this into send and recieve token types?
def MessageTokenType : DistributedType<"MessageToken", "message_token">;

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H