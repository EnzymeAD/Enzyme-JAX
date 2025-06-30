//===- Dialect.h - EnzymeXLA dialect -------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_DIALECT_H
#define ENZYMEXLA_DIALECT_H

#include "mlir/IR/Dialect.h"

// #define GET_ATTRDEF_CLASSES
// #include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrEnums.h.inc"

// #define GET_ATTRDEF_CLASSES
// #include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLADialect.h.inc"

#endif // ENZYME_DIALECT_H
