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

#endif // ENZYMEXLA_DIALECT
