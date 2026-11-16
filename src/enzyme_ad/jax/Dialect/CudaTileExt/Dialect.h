#pragma once

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/CudaTileExt/CudaTileExtDialect.h.inc"

// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/CudaTileExt/CudaTileExtOps.h.inc"
