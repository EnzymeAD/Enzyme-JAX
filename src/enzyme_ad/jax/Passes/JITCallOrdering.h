#pragma once

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

/// Unit attribute marking a JIT call that `lower-jit` must order with respect
/// to the other marked calls in the module.
inline constexpr llvm::StringLiteral kRequireJITOrderingAttrName =
    "enzymexla.require_jit_ordering";

/// Adds token dependencies to JIT calls and promotes their token slots.
llvm::LogicalResult
orderJITCalls(llvm::ArrayRef<mlir::enzymexla::JITCallOp> callOps);
