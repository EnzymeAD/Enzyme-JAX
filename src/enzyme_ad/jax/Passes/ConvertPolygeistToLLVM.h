//===- ConvertPolygeistToLLVM.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_PASSES_CONVERTPOLYGEISTTOLLVM_H
#define ENZYMEXLA_PASSES_CONVERTPOLYGEISTTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringRef.h"

#include <string>

void populatePolygeistToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateCStyleGPUFuncLoweringPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    std::string gpuTarget, bool func);

// Computes the element type used for the bare-pointer (C-style) lowering of
// a memref: the leading dimension is dropped (it becomes pointer arithmetic
// at each access), and any remaining static dimensions are wrapped in
// llvm.array. Returns a null Type if the memref isn't eligible (non-identity
// layout, or a dynamic dimension other than the leading one).
mlir::Type convertMemrefElementTypeForLLVMPointer(
    mlir::MemRefType type, const mlir::LLVMTypeConverter &converter);

// Appends patterns that lower memref ops (load/store/alloca/alloc/dealloc/
// atomics/memspace casts) directly to LLVM pointer arithmetic, assuming
// memrefs have already been converted to bare llvm.ptr via
// convertMemrefElementTypeForLLVMPointer (no descriptor struct).
void populateCStyleMemRefLoweringPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::LLVMTypeConverter &typeConverter,
                                          llvm::StringRef backend);

#endif // ENZYMEXLA_PASSES_CONVERTPOLYGEISTTOLLVM_H
