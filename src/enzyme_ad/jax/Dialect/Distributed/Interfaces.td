#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES

include "mlir/IR/OpBase.td"

def DeviceDefTrait : NativeOpTrait<"enzyme::distributed::DeviceDefTrait">;
def ChannelDefTrait : NativeOpTrait<"enzyme::distributed::ChannelDefTrait">;

def TokenReaderOpInterface : OpInterface<"TokenReaderOpInterface"> {
    let cppNamespace = "::mlir::enzyme::distributed";
    let description = [{
        An interface to determine which ops can read from a channel and what type they expect.
        Ops may read from multiple channels.
    }];
    let methods = [
        InterfaceMethod<"Returns the SSA values of tokens read from this op.", "::llvm::ArrayRef<::mlir::TypedValue<::mlir::enzyme::distributed::TokenType>>", "getReadTokens">,
        InterfaceMethod<"Returns the types of tokens read from this op, parallel to getReadTokens.", "::llvm::ArrayRef<::mlir::Type>", "getReadTokenTypes">
    ];
}

def TokenWriterOpInterface : OpInterface<"TokenWriterOpInterface"> {
    let cppNamespace = "::mlir::enzyme::distributed";
    let description = [{
        An interface to determine which ops can write to a channel and what type they provide.
        Ops may write to multiple channels.
    }];
    let methods = [
        InterfaceMethod<"Returns the SSA values of tokens written from this op.", "::llvm::ArrayRef<::mlir::TypedValue<::mlir::enzyme::distributed::TokenType>>", "getWriteTokens">,
        InterfaceMethod<"Returns the types of tokens written from this op, parallel to getWriteTokens.", "::llvm::ArrayRef<::mlir::Type>", "getWriteTokenTypes">
    ];
}


#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_INTERFACES