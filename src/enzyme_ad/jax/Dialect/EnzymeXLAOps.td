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
include "EnzymeXLAAttrs.td"

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
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

def TensorI64 : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && ::llvm::cast<::mlir::TensorType>($_self).getShape().size() == 0 && ::llvm::cast<::mlir::TensorType>($_self).getElementType().isSignlessInteger(64)">, "tensor<i64>",
                 "::mlir::TensorType">,
            BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(64))">;

def TensorFloat : Type<
  CPred<"::llvm::isa<::mlir::TensorType>($_self) && "
        "::llvm::cast<::mlir::TensorType>($_self).getShape().size() == 0 && "
        "(::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::TensorType>($_self).getElementType()) || "
        " (::llvm::isa<::mlir::ComplexType>(::llvm::cast<::mlir::TensorType>($_self).getElementType()) && "
        "  ::llvm::isa<::mlir::FloatType>(::llvm::cast<::mlir::ComplexType>(::llvm::cast<::mlir::TensorType>($_self).getElementType()).getElementType())))">,
  "tensor<float or complex<float> scalar>",
  "::mlir::TensorType">;

def KernelCallOp: EnzymeXLA_Op<"kernel_call", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Kernel Call operation";

  let arguments = (ins
    SymbolRefAttr:$fn,
    TensorI64:$gridx,
    TensorI64:$gridy,
    TensorI64:$gridz,
    TensorI64:$blockx,
    TensorI64:$blocky,
    TensorI64:$blockz,
    TensorI64:$shmem,
    Optional<TensorI64>:$clusterx,
    Optional<TensorI64>:$clustery,
    Optional<TensorI64>:$clusterz,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    DefaultValuedOptionalAttr<
        ArrayAttr, "{}">:$output_operand_aliases,
    OptionalAttr<UnitAttr>:$xla_side_effect_free
  );

  let results = (outs Variadic<AnyType>);


  let assemblyFormat = [{
    $fn
    ( `clusters` `in` `(` $clusterx^ `,` $clustery `,` $clusterz `)` )?
    `blocks` `in` `(` $gridx `,` $gridy `,` $gridz `)`
    `threads` `in` `(` $blockx `,` $blocky `,` $blockz `)`
    `shmem` `=` $shmem ` `
    `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;
}

def MemcpyOp : EnzymeXLA_Op<"memcpy"> {

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
    SymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
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


def GPUOccupancyOp : EnzymeXLA_Op<"gpu_occupancy", [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
	SymbolRefAttr:$fn,
	AnyType:$blockSize,
	AnyType:$dynamicSMemSize,
	AnyType:$flags
);
  let results = (outs AnyType : $result);
}

def GPUKernelAddressOp : EnzymeXLA_Op<"gpu_kernel_address", [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
	SymbolRefAttr:$fn
  );
  let results = (outs AnyType : $result);
}

def GPUWrapperOp : EnzymeXLA_Op<"gpu_wrapper", [
  RecursiveMemoryEffects,
  AffineScope,
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
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
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
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]> {
  let arguments = (ins Index:$blockIndexX, Index:$blockIndexY, Index:$blockIndexZ);
  let summary = "Wraps a GPU kernel block to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
}

def GPUThreadOp : EnzymeXLA_Op<"gpu_thread", [
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"enzymexla::PolygeistYieldOp">]> {
  let arguments = (ins Index:$threadIndexX, Index:$threadIndexY, Index:$threadIndexZ);
  let summary = "Wraps a GPU kernel thread to prevent restructuring";
  let regions = (region SizedRegion<1>:$region);
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
  let results = (outs LLVM_AnyPointer:$result);

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

  let arguments = (ins LLVM_AnyPointer:$source);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  
  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def AlternativesOp : EnzymeXLA_Op<"alternatives", [
  RecursiveMemoryEffects]> {
  let summary = "Provides several alternatives kernels for gpu code";
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "int":$regionNum)>];
  let hasCanonicalizer = 1;
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

// Linear Algebra Ops

def SymmOp: EnzymeXLA_Op<"lapack.symm", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Multiplication involving a symmetric matrix";

  let description = [{
    C := alpha*A*B + beta*C, or C := alpha*B*A + beta*C, where alpha and beta are scalars,  A is a symmetric matrix"
  }];

  let arguments = (ins
    HLO_Tensor:$A,
    HLO_Tensor:$B,
    HLO_Tensor:$C,
    TensorFloat:$alpha,
    TensorFloat:$beta,
    EnzymeXLA_LapackSideAttr:$side,
    EnzymeXLA_LapackUploAttr:$uplo
  );

  let results = (outs
    HLO_Tensor: $output
  );

  let assemblyFormat = [{
    $A `,` $B `,` $C `,` $alpha `,` $beta attr-dict `:` functional-type(operands, results)
  }];
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

def QRFactorizationOp: EnzymeXLA_Op<"linalg.qr", [Pure]> {
  let summary = "QR factorization operation (high-level op)";

  let description = [{
    This operation computes the QR factorization of a matrix using Householder 
    reflections. Mathematically, it decomposes A into the product of an 
    orthogonal (unitary if complex) matrix Q and an upper triangular matrix R, 
    such that A = QR.

    If A has size m x n and m > n, Q is an m x n isometric matrix. If m < n, R
    will be a m x n trapezoidal matrix.

    This operation is modeled after the mathematical formulation of the QR 
    factorization, and not after LAPACK's compact formats.
  }];

  let arguments = (ins
    HLO_Tensor:$input,
    DefaultValuedAttr<EnzymeXLA_QrAlgorithmAttr, "mlir::enzymexla::QrAlgorithm::geqrf">:$algorithm
  );

  let results = (outs
    HLO_Tensor:$Q,
    HLO_Tensor:$R
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GeqrfOp : EnzymeXLA_Op<"lapack.geqrf", [Pure]> {
  let summary = "QR factorization operation (low-level op)";

  let description = [{
    This operation computes the QR factorization of a matrix using Householder 
    reflections. Mathematically, it decomposes A into the product of an 
    orthogonal matrix Q and an upper triangular matrix R, such that A = QR.

    This operation is modeled after LAPACK's *GEQRF routines, which returns the 
    result in the QR packed format.
  }];

  let arguments = (ins
    HLO_Tensor:$input
  );

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$tau,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def GeqrtOp : EnzymeXLA_Op<"lapack.geqrt", [Pure]> {
  let summary = "QR factorization operation (low-level op)";

  let description = [{
    This operation computes the QR factorization of a matrix using Householder 
    reflections. Mathematically, it decomposes A into the product of an 
    orthogonal matrix Q and an upper triangular matrix R, such that A = QR.

    This operation is modeled after LAPACK's *GEQRT routines, which returns the 
    result in the QR CompactWY format.
  }];

  let arguments = (ins
    HLO_Tensor:$input,
    OptionalAttr<I64Attr>:$blocksize // nb argument, min(m,n) >= nb >= 1
  );

  let results = (outs
    HLO_Tensor:$output,
    HLO_Tensor:$T,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def OrgqrOp : EnzymeXLA_Op<"lapack.orgqr", [Pure]> {
  let summary = [{
    Generate orthonormal/unitary matrix from a product of
    elementary Householder reflectors
  }];

  let description = [{
    This operation is modeled after LAPACK's *ORGQR/*UNGQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$input,
    HLO_Tensor:$tau
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $input `,` $tau attr-dict `:` functional-type(operands, results)
  }];
}

def OrmqrOp : EnzymeXLA_Op<"lapack.ormqr", [Pure]> {
  let summary = [{
    Perform a matrix multiplication with the Q matrix from a QR
    decomposition, given only the householder reflections.
  }];

  let description = [{
    This operation is modeled after LAPACK's *ORMQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$A,
    HLO_Tensor:$tau,
    HLO_Tensor:$C,
    EnzymeXLA_LapackSideAttr:$side,
    DefaultValuedAttr<EnzymeXLA_LapackTransposeAttr, "::mlir::enzymexla::LapackTranspose::none">:$transpose // true if apply Q^T
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $A `,` $tau `,` $C attr-dict `:` functional-type(operands, results)
  }];
}

def GemqrtOp : EnzymeXLA_Op<"lapack.gemqrt", [Pure]> {
  let summary = [{
    Perform a matrix multiplication with the Q matrix from a QR
    decomposition, given the WY decomposition (i.e. geqrt).
  }];

  let description = [{
    This operation is modeled after LAPACK's *GEMQR routines.
  }];

  let arguments = (ins
    HLO_Tensor:$V,
    HLO_Tensor:$T,
    HLO_Tensor:$C,
    EnzymeXLA_LapackSideAttr:$side,
    DefaultValuedAttr<EnzymeXLA_LapackTransposeAttr, "::mlir::enzymexla::LapackTranspose::none">:$transpose // true if apply Q^T
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let assemblyFormat = [{
    $V `,` $T `,` $C attr-dict `:` functional-type(operands, results)
  }];
}

def SVDFactorizationOp : EnzymeXLA_Op<"linalg.svd", [Pure]> {
  let summary = "Singular Value Decomposition (SVD) factorization operation.";

  let arguments = (ins
    HLO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$full
  );

  let results = (outs
    HLO_Tensor:$U,
    HLO_Tensor:$S,
    HLO_Tensor:$Vt,
    HLO_Tensor:$info
  );

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

// Machine Learning Ops

def GeluOp: EnzymeXLA_Op<"ml.gelu", [Pure, SameOperandsAndResultType, Elementwise]> {
  let summary = "Computes the GELU activation function";

  let arguments = (ins
    HLO_Tensor:$input,
    EnzymeXLA_GeluApproximationAttr:$gelu_approximation
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $input `,` `approximation``=`$gelu_approximation attr-dict `:` functional-type($input, results)
  }];
}

def ReluOp: EnzymeXLA_Op<"ml.relu", [Pure, SameOperandsAndResultType, Elementwise]> {
  let summary = "Computes the RELU activation function";

  let arguments = (ins
    HLO_Tensor:$input
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, results)
  }];
}

def CacheLoad : EnzymeXLA_Op<"cacheload"> {
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = llvm::cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];
  let summary = "load from cross barier cache";
  let description = [{}];
}


def SubIndexOp : EnzymeXLA_Op<"subindex", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source, Index : $index);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

def TypeAlignOp : EnzymeXLA_Op<"typeAlign", [Pure]> {
  let summary = "Get alignment of type";
  let arguments = (ins TypeAttr : $source);
  let results = (outs AnyType : $result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif // ENZYMEXLA_OPS
