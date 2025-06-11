//===- Passes.h - Enzyme pass include header  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ENZYMEXLA_COMM_PASSES_H
#define ENZYMEXLA_COMM_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

namespace enzyme {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Comm/CommPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Comm/CommPasses.h.inc"

} // namespace enzyme
} // namespace mlir
#endif // ENZYMEXLA_COMM_PASSES_H