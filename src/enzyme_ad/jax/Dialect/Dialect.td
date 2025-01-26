//===- EnzymeXLA.td - EnzymeXLA dialect --------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_DIALECT
#define ENZYMEXLA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Enzyme dialect definition.
//===----------------------------------------------------------------------===//

def EnzymeXLA_Dialect : Dialect {
  let name = "enzymexla";
  let description = [{}];
  let cppNamespace = "::mlir::enzymexla";
  // let useDefaultAttributePrinterParser = 1;
  // let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Enzyme operation definition.
//===----------------------------------------------------------------------===//

class EnzymeXLA_Op<string mnemonic, list<Trait> traits = []>
    : Op<EnzymeXLA_Dialect, mnemonic, traits>;

class EnzymeXLA_Type<string name> : TypeDef<EnzymeXLA_Dialect, name>;

//===----------------------------------------------------------------------===//
// AtAddrOp
//===----------------------------------------------------------------------===//

def AtAddrOp : EnzymeXLA_Op<"ataddr", [Pure]> {
  let summary =
      "Construct a c-style memref at an addr";
  let description = [{
  }];
  let arguments = (ins AnyType:$addr);
  let results = (outs AnyRankedOrUnrankedMemRef:$dest);
  let builders = [
    OpBuilder<(ins "Value":$source), [{
      return build($_builder, $_state,
        TypeRange(MemRefType::get({ShapedType::kDynamic}, $_builder.getI8Type())),
        ValueRange(source));
    }]>,
    OpBuilder<(ins "Value":$source, "Type":$dstType), [{
      return build($_builder, $_state,
        TypeRange(dstType),
        ValueRange(source));
    }]>
  ];
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

#endif // ENZYMEXLA_DIALECT
