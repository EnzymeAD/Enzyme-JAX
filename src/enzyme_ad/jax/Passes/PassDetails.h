//===- PassDetails.h - Enzyme pass class details ----------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different polygeist passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_ENZYMEXLA_TRANSFORMS_PASSDETAILS_H
#define DIALECT_ENZYMEXLA_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
class FunctionOpInterface;
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);
namespace enzyme {

class EnzymeDialect;

#define GEN_PASS_CLASSES
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

} // namespace enzyme
} // namespace mlir

#endif // DIALECT_ENZYME_TRANSFORMS_PASSDETAILS_H
