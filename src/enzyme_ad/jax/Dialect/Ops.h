//===- EnzymeXLAOps.h - EnzymeXLA dialect ops -------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLAOPS_H
#define ENZYMEXLAOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"

// #include "Dialect/EnzymeXLAEnums.h.inc"

// #define GET_ATTRDEF_CLASSES
// #include "Dialect/EnzymeXLAAttributes.h.inc"

// #define GET_TYPEDEF_CLASSES
// #include "Dialect/EnzymeXLAOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.h.inc"

// #include "Dialect/EnzymeXLATypes.h.inc"

#endif // ENZYMEXLAOPS_H
