//===- EnzymeHLOPatterns.h - functions to register patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

namespace mlir {
class MLIRContext;
class PatternBenefit;
class RewritePatternSet;
} // namespace mlir

#include "src/enzyme_ad/jax/Passes/StablehloOptPopulatePatterns.h.inc"
