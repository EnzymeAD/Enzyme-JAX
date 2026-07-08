//===- ConversionUtils.h - Utilities for conversions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_PASSES_CONVERSIONUTILS_H
#define ENZYMEXLA_PASSES_CONVERSIONUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {

Value extractLimb(Value tensor, int limbIndex, OpBuilder &builder, Location loc,
                  llvm::StringRef concatDimension);

Value packLimbs(llvm::ArrayRef<Value> limbs, OpBuilder &builder, Location loc,
                llvm::StringRef concatDimension);

Value packLimbs(Value high, Value low, OpBuilder &builder, Location loc,
                llvm::StringRef concatDimension);

} // namespace enzyme
} // namespace mlir

#endif // ENZYMEXLA_PASSES_CONVERSIONUTILS_H
