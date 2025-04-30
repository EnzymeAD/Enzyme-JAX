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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "stablehlo/dialect/Base.td"

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

#endif // ENZYMEXLA_OPS
