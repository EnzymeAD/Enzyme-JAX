#pragma once

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

/// Adds token dependencies to JIT calls and promotes their token slots.
llvm::LogicalResult
orderJITCalls(llvm::ArrayRef<mlir::enzymexla::JITCallOp> callOps);
