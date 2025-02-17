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

def TensorI64 : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && ::llvm::cast<::mlir::TensorType>($_self).getShape().size() == 0 && ::llvm::cast<::mlir::TensorType>($_self).getElementType().isSignlessInteger(64)">, "tensor<i64>",
                 "::mlir::TensorType">,
            BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(64))">;

def KernelCallOp: EnzymeXLA_Op<"kernel_call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<CallOpInterface>, Pure]> {
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
        ArrayAttr, "{}">:$output_operand_aliases
  );

  let results = (outs Variadic<AnyType>);


  let assemblyFormat = [{
    $fn ` ` `blocks` `in` `(` $gridx `,` $gridy `,` $gridz `)` ` ` `threads` `in` `(` $blockx `,` $blocky `,` $blockz `)` ` ` `shmem` `=` $shmem ` ` `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;
}

def JITCallOp: EnzymeXLA_Op<"jit_call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<CallOpInterface>, Pure]> {
  let summary = "JIT Call operation";

  let arguments = (ins
    FlatSymbolRefAttr:$fn, 
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$backend_config,
    OptionalAttr<AnyAttr>:$operand_layouts,
    OptionalAttr<AnyAttr>:$result_layouts,
    DefaultValuedOptionalAttr<
        ArrayAttr, "{}">:$output_operand_aliases
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

#endif // ENZYMEXLA_OPS
