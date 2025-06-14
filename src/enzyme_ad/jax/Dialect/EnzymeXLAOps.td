//===- EnzymeXLAOps.td - EnzymeXLA dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_OPS
#define ENZYMEXLA_OPS

include "Enzyme/MLIR/Dialect/Dialect.td"
include "Dialect.td"


include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"

def TensorI64 : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && ::llvm::cast<::mlir::TensorType>($_self).getShape().size() == 0 && ::llvm::cast<::mlir::TensorType>($_self).getElementType().isSignlessInteger(64)">, "tensor<i64>",
                 "::mlir::TensorType">,
            BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(64))">;

def KernelCallOp: EnzymeXLA_Op<"kernel_call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<CallOpInterface>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Kernel Call operation";

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    TensorI64:$gridx,
    TensorI64:$gridy,
    TensorI64:$gridz,
    TensorI64:$blockx,
    TensorI64:$blocky,
    TensorI64:$blockz,
    TensorI64:$shmem,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        ArrayAttr, "{}">:$output_operand_aliases,
    OptionalAttr<UnitAttr>:$xla_side_effect_free
  );

  let results = (outs Variadic<AnyType>);


  let assemblyFormat = [{
    $fn ` ` `blocks` `in` `(` $gridx `,` $gridy `,` $gridz `)` ` ` `threads` `in` `(` $blockx `,` $blocky `,` $blockz `)` ` ` `shmem` `=` $shmem ` ` `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;
}

def MemcpyOp : EnzymeXLA_Op<"memcpy", [CopyOpInterface]> {

  let summary = "GPU memcpy operation";

  let description = [{
    The `gpu.memcpy` operation copies the content of one memref to another.

    The op does not execute before all async dependencies have finished
    executing.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token.

    Example:

    ```mlir
    %token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$target,
                   Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$source,
		   Index:$size
		);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $target`,` $source `,` $size `:` type($target)`,` type($source) attr-dict
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def JITCallOp: EnzymeXLA_Op<"jit_call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<CallOpInterface>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "JIT Call operation";

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        ArrayAttr, "{}">:$output_operand_aliases,
    OptionalAttr<UnitAttr>:$xla_side_effect_free
  );

  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $fn ` ` `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;
}

def GetStreamOp : EnzymeXLA_Op<"get_stream", [Pure]> {
  let summary = "Get current execution stream within a jit_call operation";
  let description = [{
  }];
  let results = (outs AnyType:$result);
}


def GPUWrapperOp : EnzymeXLA_Op<"gpu_wrapper", [
  RecursiveMemoryEffects,
  AutomaticAllocationScope,
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]> {
  let arguments = (ins Variadic<Index>:$blockDims);
  let summary = "Indicates the region contained must be executed on the GPU";
  let description = [{
    The optional arguments to this operation are suggestions about what block
    dimensions this gpu kernel should have - usually taken from kernel launch
    params
  }];
  let results = (outs Index : $result);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins "ValueRange":$blockSizes)>,
      OpBuilder<(ins)>];
}

def XLAWrapperOp: EnzymeXLA_Op<"xla_wrapper", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<CallOpInterface>, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "XLA Call operation";

  let arguments = (ins
    SymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs
  );

  let assemblyFormat = [{
    $fn ` ` `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

}

def GPUErrorOp : EnzymeXLA_Op<"gpu_error", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]>,
  Arguments<(ins)> {
  let summary = "Gets the error returned by the gpu operation inside";
  // TODO should be i32, not index
  let results = (outs Index : $result);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];

}

def NoopOp
    : EnzymeXLA_Op<"noop",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Noop for preventing folding or transformations";
  let arguments = (ins Variadic<Index>:$blockDims);
  let skipDefaultBuilders = 1;
  let builders = [
      OpBuilder<(ins "ValueRange":$indices)>];
  let description = [{}];
}


def GPUBlockOp : EnzymeXLA_Op<"gpu_block", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]>,
  Arguments<(ins Index:$blockIndexX, Index:$blockIndexY, Index:$blockIndexZ)> {
  let summary = "Wraps a GPU kernel block to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
      "Value":$blockIndexX, "Value":$blockIndexY, "Value":$blockIndexZ)>];
}

def GPUThreadOp : EnzymeXLA_Op<"gpu_thread", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]>,
  Arguments<(ins Index:$threadIndexX, Index:$threadIndexY, Index:$threadIndexZ)> {
  let summary = "Wraps a GPU kernel thread to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
      "Value":$threadIndexX, "Value":$threadIndexY, "Value":$threadIndexZ)>];
}

def BarrierOp
    : EnzymeXLA_Op<"barrier",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins Variadic<Index>:$indices);
  let summary = "barrier for parallel loops";
  let description = [{}];
  let hasCanonicalizer = true;
}

def PolygeistYieldOp : EnzymeXLA_Op<"polygeist_yield", [Pure, ReturnLike, Terminator]> {
    //ParentOneOf<["AlternativesOp", "GPUWrapperOp", "GPUErrorOp", "GPUBlockOp", "GPUThreadOp"]>]> {
  let summary = "Polygeist ops terminator";
}

def StreamToTokenOp : EnzymeXLA_Op<"stream2token", [
  Pure
]> {
  let summary = "Extract an async stream from a cuda stream";

  let arguments = (ins AnyType : $source);
  let results = (outs AnyType : $result);
}

def Memref2PointerOp : EnzymeXLA_Op<"memref2pointer", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "Extract and LLVM pointer from a MemRef";

  let arguments = (ins AnyMemRef : $source);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def Pointer2MemrefOp : EnzymeXLA_Op<"pointer2memref", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "Upgrade a pointer to a memref";

  let arguments = (ins AnyType:$source);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def AffineScopeOp : EnzymeXLA_Op<"scope", [
      AffineScope,
      AutomaticAllocationScope,
      RecursiveMemoryEffects,
    ]>,
    Arguments<(ins Variadic<AnyType>:$operands)>,
    Results<(outs Variadic<AnyType>:$results)> {
  let summary = "Inline affine scope";
  let regions = (region SizedRegion<1>:$region);
}

def RotateOp : EnzymeXLA_Op<"rotate", [Pure, SameOperandsAndResultType]> {
  let summary = "Takes `amount` from the start of the tensor in `dimension` and appends it to the end";
  let arguments = (ins
    HLO_Tensor:$operand,
    SI32Attr:$amount,
    SI32Attr:$dimension
  );
  let results = (outs HLO_Tensor:$result);
}

def WrapOp: EnzymeXLA_Op<
      "wrap",
      [Pure, SameOperandsAndResultElementType,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Wrap operation";

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$lhs,
    I64Attr:$rhs,
    I64Attr:$dimension
  );

  let results = (outs HLO_Tensor:$result);
}

def ExtendOp: EnzymeXLA_Op<
      "extend",
      [Pure, SameOperandsAndResultElementType,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extend operation";

  let arguments = (ins
    HLO_Tensor:$operand,
    I64Attr:$lhs,
    I64Attr:$rhs,
    I64Attr:$dimension
  );

  let results = (outs HLO_Tensor:$result);
}

def CommRegionOp : EnzymeXLA_Op<"comm_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    RecursiveMemoryEffects, RecursivelySpeculatable]> {
  let summary = "container op for grouping communication";

  let results = (outs Variadic<AnyType>);

  let regions = (region
    SizedRegion<1>:$body /*while_i3*/
  );
}

def LUFactorizationOp: EnzymeXLA_Op<"linalg.lu", [Pure]> {
  let summary = "LU factorization operation with RowMaximum pivoting.";

  let arguments = (ins
    HLO_Tensor:$input
  );

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$pivots,
    HLO_Tensor:$permutation,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

#endif // ENZYMEXLA_OPS
