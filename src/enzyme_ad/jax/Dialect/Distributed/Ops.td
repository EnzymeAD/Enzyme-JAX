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

def GroupSplitOp : DistributedOp<"GroupSplit", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoTerminator]>{
    let arguments = (ins
        SymbolRefAttr:$device_group,
        ArrayAttr:$branch_assignments // Symbols mapping to each branch region
    );
    // TODO check that declarations only declares tokens.
    let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
    let results = (outs ); // TODO
    // let hasVerifier = 1; // TODO
    let assemblyFormat = "$device_group custom<SplitBranches>($branch_assignments, $branches) attr-dict";
}

def DefineTokenOp : DistributedOp<"DefineToken", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
            SymbolRefAttr:$channel
        );
    let results = (outs ReadTokenType:$read_token, WriteTokenType:$write_token);
    // let hasVerifier = 1; // TODO: verify writers and readers are connected to the channel
    let assemblyFormat = "$channel attr-dict";
}

def SendOp : DistributedOp<"Send", [DeclareOpInterfaceMethods<TokenWriterOpInterface>]>{
    let arguments = (ins
        WriteTokenType:$token,
        // value to send
        AnyType:$value);
    let assemblyFormat = "$token type($value) $value attr-dict";
}

def RecvOp : DistributedOp<"Recv", [DeclareOpInterfaceMethods<TokenReaderOpInterface>]>{
    let arguments = (ins
        ReadTokenType:$token);
    let results = (outs AnyType:$value);
    let assemblyFormat = "$token type($value) attr-dict";
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD