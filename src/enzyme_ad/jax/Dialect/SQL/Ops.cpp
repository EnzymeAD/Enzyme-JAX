//===- Ops.cpp - SQL dialect operations ----------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "src/enzyme_ad/jax/Dialect/SQL/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/SQL/SQLOps.cpp.inc"
