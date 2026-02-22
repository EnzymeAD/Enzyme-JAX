#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"

// Mesh definition ops
def PhysicalMeshOp : DistributedOp<"PhysicalMesh"> {
    let description = [{Defines a physical mesh of devices based on the hardware topology. A physical mesh is a Cartesian product
    of zero or more physical axes. Axes should be distinct! Two copies of the same axis is permitted (but may better be represented
    as a larger axis), but passing the same axis twice is an error.}];
    let arguments = (ins
        Variadic<PhysicalCommAxisType>:$axes
    );
    let results = (outs PhysicalMeshType:$mesh);
    let assemblyFormat = "$axes attr-dict";
}

def LogicalMeshOp : DistributedOp<"LogicalMesh"> {
    let description = [{Defines a logical mesh of devices, intended to represent the mesh used by the program. Defined in terms of its projection
    onto a physical mesh via a product of logical axes.}];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$axes
    );
    let results = (outs LogicalMeshType:$mesh);
    let assemblyFormat = "$axes attr-dict";
}

def AxisAllToAllOp : DistributedOp<"AxisAllToAll"> {
    let description = [{
       A communication channel where any device can communicate to any other device without
       bandwidth conflict or latency scaling with index distance. 
     }];
    let arguments = (ins
        I32ArrayAttr:$axis_size,
        // TODO should these be floats?
        UI64Attr:$bandwidth, // in bytes per picosecond
        UI64Attr:$latency // in picoseconds.
    );
    let results = (outs PhysicalCommAxisType:$axis);
    let assemblyFormat = "$axis_size $bandwidth $latency attr-dict";
}

def AxisFactorOp : DistributedOp<"AxisFactor", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let description = [{
       Factors a physical axis into multiple logical axes, according to a static array of factor sizes.
       The lowest index of factors corresponds to the lowest-significance / most contiguous new axis.
       Factors must multiply into the original physical axis size.
       Returns a logical axis per factor.
     }];
    let arguments = (ins
        PhysicalCommAxisType:$physical_axis,
        I32ArrayAttr:$factors
    );
    let results = (outs Variadic<LogicalCommAxisType>:$logical_axes);
    let assemblyFormat = "$physical_axis $factors attr-dict";
    let hasVerifier = 1; // TODO: verify factors multiply to physical axis size, and correct number of outputs.
}


def AxisProductOp : DistributedOp<"AxisProduct"> {
    let description = [{
       Combines multiple logical axes into a single logical axis. Logical axis must be disjoint,
       but may come from any mix of physical axes, including different strides over the same physical
       axis. The order encodes the significance, with the lowest-index/first logical axis being
       index as the least-significant/most contiguous dimension. 
       For simplicity for implementation, we require any factors of the same physical axis to be
       defined from the same factorization op. (For instance, this makes disjointness easier)
     }];
    let arguments = (ins
        Variadic<LogicalCommAxisType>:$logical_axes
    );
    let results = (outs LogicalCommAxisType:$logical_axis);
    let assemblyFormat = "$logical_axes attr-dict";
    let hasVerifier = 1; // TODO: axis disjointness.
}


// Device definition ops

def ChannelOp : DistributedOp<"Channel", [Symbol, ChannelDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        // a variadic list of devices connected by this channel
        ArrayAttr:$sending_devices,
        ArrayAttr:$receiving_devices
        // TODO: channel type, bandwidth, latency, etc
    );
    let assemblyFormat = "$sym_name $sending_devices $receiving_devices attr-dict";
}

def LeafDeviceOp : DistributedOp<"LeafDevice", [Symbol, DeviceDefTrait]>{
    let arguments = (ins 
        SymbolNameAttr:$sym_name
        // TODO: device type, e.g. TPU, GPU, CPU, and other attributes
    );
    let assemblyFormat = "$sym_name attr-dict";
}

def DeviceGroupOp : DistributedOp<"DeviceGroup", [Symbol, DeviceDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        // a variadic list of devices in the group
        ArrayAttr:$devices, // TODO verify array of SymbolRefAttr of devices
        // Interconnects between devices
        ArrayAttr:$channels // TODO verify array of SymbolRefAttr of channels
    );
    let assemblyFormat = "$sym_name $devices $channels attr-dict";
}
def DeviceMeshOp : DistributedOp<"DeviceMesh", [Symbol, DeviceDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$device_type,
        ArrayAttr:$shape
        // TODO attributes of mesh interconnect
    );
    let assemblyFormat = "$sym_name $device_type $shape attr-dict";
}

// Ops for breaking down computation across the device hierarchy

def MeshForOp : DistributedOp<"MeshFor", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoTerminator, SingleBlock]>{
    let arguments = (ins SymbolRefAttr:$mesh); // TODO: verify it's a mesh
    let regions = (region MaxSizedRegion<1>:$body); // TODO: body's block args are device type and mesh index
    let results = (outs ); // TODO
    // let hasVerifier = 1; // TODO: verify body's block args take mesh index
    let assemblyFormat = "$mesh $body attr-dict";
}

def GroupSplitOp : DistributedOp<"GroupSplit", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoTerminator, SingleBlock]>{
    let arguments = (ins
        SymbolRefAttr:$device_group // TODO: verify it's a group
    );
    let regions = (region SizedRegion<1>:$declarations); // Takes as args the devices and channels in the group
    let results = (outs ); // TODO
    // let hasVerifier = 1; // TODO
    // let hasCanonicalizer = 1; // TODO: token declarations up front, followed by device and channel branches in order of listing in the group
    let assemblyFormat = "$device_group $declarations  attr-dict";
}

def SplitBranchOp : DistributedOp<"SplitBranch", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoTerminator, SingleBlock]>{
    let arguments = (ins
        SymbolRefAttr:$device_or_channel // TODO: verify it's a device or channel
        );
    let regions = (region MaxSizedRegion<1>:$body); // Takes as args the device or channel
    let results = (outs ); // TODO
    // let hasVerifier = 1; // TODO: parent is a groupsplitop
    let assemblyFormat = "$device_or_channel $body attr-dict";
}

def DefineTokenOp : DistributedOp<"DefineToken", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
            SymbolRefAttr:$channel
        );
    let results = (outs TokenType:$token_out);
    // let hasVerifier = 1; // TODO: verify writers and readers are connected to the channel
    let assemblyFormat = "$channel attr-dict";
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD