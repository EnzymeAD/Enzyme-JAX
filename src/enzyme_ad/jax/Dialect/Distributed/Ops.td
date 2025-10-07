#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"

include "Dialect.td"
include "Types.td"
include "Interfaces.td"

// Device definition ops

def ChannelOp : DistributedOp<"channel", [Symbol, ChannelDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins 
        SymbolNameAttr:$sym_name,
        // a variadic list of devices connected by this channel
        ArrayAttr:$sending_devices,
        ArrayAttr:$receiving_devices
        // TODO: channel type, bandwidth, latency, etc
    );
    let assemblyFormat = "$sym_name $sending_devices $receiving_devices attr-dict";
}

def LeafDeviceOp : DistributedOp<"leaf_device", [Symbol, DeviceDefTrait]>{
    let arguments = (ins 
        SymbolNameAttr:$sym_name
        // TODO: device type, e.g. TPU, GPU, CPU, and other attributes
    );
    let assemblyFormat = "$sym_name attr-dict";
}

def DeviceGroupOp : DistributedOp<"device_group", [Symbol, DeviceDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        // a variadic list of devices in the group
        ArrayAttr:$devices, // TODO verify array of SymbolRefAttr of devices
        // Interconnects between devices
        ArrayAttr:$channels // TODO verify array of SymbolRefAttr of channels
    );
    let assemblyFormat = "$sym_name $devices $channels attr-dict";
}
def DeviceMeshOp : DistributedOp<"device_mesh", [Symbol, DeviceDefTrait, DeclareOpInterfaceMethods<SymbolUserOpInterface>]>{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$device_type,
        ArrayAttr:$shape
        // TODO attributes of mesh interconnect
    );
    let assemblyFormat = "$sym_name $device_type $shape attr-dict";
}

// def ContinueOp : DistributedOp<"continue", [Terminator]> {
//     let description = [{
//         A terminator for DeviceParallelOp regions. Takes as arguments the tokens to be passed to the
//         continuation of the DeviceParallelOp. These values can then be used in a subsequent DeviceParallelOp
//         that is a sibling to the original DeviceParallelOp by referencing the returned tokens.
//     }];
//     let arguments = (ins Variadic<AnyType>:$operands);
//     let results = (outs ); // No outputs for terminators, the token is output by the parent DeviceParallelOp.
//     let assemblyFormat = "$operands type($operands) attr-dict";
// }

def DeviceParallelOp : DistributedOp<"device_parallel", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, NoTerminator]>{
    let description = [{
        An op for mapping computations to subdevices. Serves both for homogenous device meshes as well
        as explicitly enumerated device groups. In the case of device meshes, this op should contain
        a single region to be executed in parallel on each device. In the case of device groups, this
        op should contain one region per device and channel in the group.

        In either case, regions must take as argument one device index within the parent device followed
        by a number of token arguments. Tokens are matched by positionally between different branches,
        and all branches must have the same number and type of token arguments (though they may be unused).
    }];
    
    let arguments = (ins
        SymbolRefAttr:$enclosing_device,
        ArrayAttr:$branch_assignments // the device components for each region (device-specific branch)
    );
    let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
    // let results = (outs Variadic<TokenType>:$continuation_tokens);
    let results = (outs );
    let hasVerifier = 1; // TODO
    let assemblyFormat = "$enclosing_device `{` custom<DeviceBranches>($branch_assignments, $branches) `}` attr-dict";
    let extraClassDeclaration = [{
        Operation* getEnclosingDeviceOp();
    }];
}

def SendOp : DistributedOp<"send", [DeclareOpInterfaceMethods<TokenWriterOpInterface>]>{
    let arguments = (ins
        TokenType:$token,
        // value to send
        AnyType:$value);
    let assemblyFormat = "$token type($value) $value attr-dict";
}

def RecvOp : DistributedOp<"recv", [DeclareOpInterfaceMethods<TokenReaderOpInterface>]>{
    let arguments = (ins
        TokenType:$token);
    let results = (outs AnyType:$value);
    let assemblyFormat = "$token type($value) attr-dict";
}

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_OPS_TD