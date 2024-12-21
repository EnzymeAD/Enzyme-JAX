//===- EnzymeXLADialect.cpp - EnzymeXLA dialect -----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "Ops.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Dialect.h"

// #include "Dialect/EnzymeEnums.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/EnzymeXLADialect.cpp.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "Dialect/EnzymeXLAOpsTypes.cpp.inc"
//  #include "Dialect/EnzymeTypes.cpp.inc"

using namespace mlir;
using namespace mlir::enzymexla;

//===----------------------------------------------------------------------===//
// Enzyme dialect.
//===----------------------------------------------------------------------===//

void EnzymeXLADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.cpp.inc"
      >();
  //  addAttributes<
  // #define GET_ATTRDEF_LIST
  // #include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttributes.cpp.inc"
  //      >();
  //  addTypes<
  // #define GET_TYPEDEF_LIST
  // #include "src/enzyme_ad/jax/Dialect/EnzymeXLAOpsTypes.cpp.inc"
  //      >();
}

// #define GET_ATTRDEF_CLASSES
// #include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttributes.cpp.inc"
