//===- Passes.h - Enzyme pass include header  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ENZYMEXLA_PASSES_H
#define ENZYMEXLA_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace enzyme {

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

void populateLibDeviceFuncsToOpsPatterns(MLIRContext *context,
                                         RewritePatternSet &patterns);

} // namespace enzyme

namespace cf {
void populateLLVMToControlFlowConversionPatterns(RewritePatternSet &patterns);
} // namespace cf

} // end namespace mlir

#endif // ENZYMEXLA_PASSES_H
