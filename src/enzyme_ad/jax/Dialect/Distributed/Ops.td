#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"

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