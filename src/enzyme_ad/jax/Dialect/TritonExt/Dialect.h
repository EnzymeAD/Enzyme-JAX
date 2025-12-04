#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/TritonExt/TritonExtDialect.h.inc"

// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/TritonExt/TritonExtOps.h.inc"
