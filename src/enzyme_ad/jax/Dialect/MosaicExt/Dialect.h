#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/MosaicExt/MosaicExtDialect.h.inc"

// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/MosaicExt/MosaicExtOps.h.inc"
