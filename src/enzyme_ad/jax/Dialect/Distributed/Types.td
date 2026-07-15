#ifndef ENZYME_DISTRIBUTED_DIALECT_TYPES_H
#define ENZYME_DISTRIBUTED_DIALECT_TYPES_H

include "Dialect.td"
include "Interfaces.td"

class AxisType<string name, string type_mnemonic, list<Trait> traits = []>
		: DistributedType<name, type_mnemonic,
											[DeclareTypeInterfaceMethods<AxisTypeInterface>] #
													traits>;

// A natural axis defined by the hardware topology.
def PhysicalCommAxisType : AxisType<"PhysicalCommAxis", "physical_comm_axis"> {
	let parameters = (ins "unsigned":$extent);
	let assemblyFormat = "`<` $extent `>`";
}

// Axis of a logical mesh where hardware details are abstracted away.
def LogicalMeshAxisType : AxisType<"LogicalMeshAxis", "logical_mesh_axis"> {
	let parameters = (ins "unsigned":$extent);
	let assemblyFormat = "`<` $extent `>`";
}

// A standalone axis type used to represent replication.
def ReplicationAxisType : AxisType<"ReplicationAxis", "replication_axis"> {
	let parameters = (ins "unsigned":$extent);
	let assemblyFormat = "`<` $extent `>`";
}

// Tensor axis type with extent inferred from the tensor type.
def TensorAxisType : AxisType<"TensorAxis", "tensor_axis"> {
	let parameters = (ins "::mlir::Type":$tensorType, "unsigned":$axisIndex);
	let assemblyFormat = "`<` $tensorType `,` $axisIndex `>`";
	let extraClassDeclaration = [{
		unsigned getExtent() const {
			auto shaped = ::mlir::dyn_cast<::mlir::ShapedType>(getTensorType());
			assert(shaped && "TensorAxisType expects a shaped tensor type");
			// TODO: support non-static/dynamic dimensions when AxisTypeInterface
			// allows non-constant extents.
			assert(shaped.hasRank() && "TensorAxisType expects a ranked tensor type");
			assert(getAxisIndex() < shaped.getRank() && "axis index out of range");
			assert(!shaped.isDynamicDim(getAxisIndex()) &&
						 "dynamic tensor dimensions are not yet supported");
			return static_cast<unsigned>(shaped.getDimSize(getAxisIndex()));
		}
	}];
}

// Generic factor wrapper for axis algebra.
def AxisFactorType : DistributedType<"AxisFactor", "axis_factor"> {
	let parameters =
			(ins "::mlir::Type":$axisType, "unsigned":$extent, "unsigned":$stride);
	let assemblyFormat = "`<` $axisType `,` $extent `,` $stride `>`";
}

// Small (unecessary?) type wrapping products of logical axis and "reduce" for sharding annotations.
def ShardingAxisType : DistributedType<"Sharding", "sharding">;

def PhysicalMeshType : DistributedType<"PhysicalMesh", "physical_mesh">;
def LogicalMeshType : DistributedType<"LogicalMesh", "logical_mesh">;

// Wraps a value type that may be synchronized at a later point.
def AsynchHandleType : DistributedType<"AsynchHandle", "asynch_handle"> {
  let summary = "Handle to a value that can be synchronized later";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

// Single token type used for collective declaration and message events.
def MessageTokenType : DistributedType<"MessageToken", "message_token">;

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H