//===- ISLUtils.h - ISL Polyhedral & Affine Utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_ISLUTILS_H
#define POLYMER_SUPPORT_ISLUTILS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"

#include "isl/aff.h"
#include "isl/isl-noexceptions.h"
#include "isl/set.h"
#include "isl/union_map.h"

namespace polymer {

/// Contract intermediate vertices in a tagged dependence graph to compute
/// maximal live-range paths for ISL live-range span scheduling.
isl::union_map get_maximal_paths(isl::union_map umap);

/// Convert an `isl_aff` (over `isl_dim_in` dimensions and `isl_dim_param`
/// symbols) into an `mlir::AffineExpr`. Returns null `AffineExpr` if `aff` is
/// null or contains unsupported non-integer/NaN values.
mlir::AffineExpr islAffToAffineExpr(__isl_keep isl_aff *aff,
                                    mlir::MLIRContext *ctx);

/// Convert an `isl_multi_aff` into an `mlir::AffineMap`.
mlir::AffineMap islMultiAffToAffineMap(__isl_keep isl_multi_aff *maff,
                                       mlir::MLIRContext *ctx);

/// Convert a single `isl_basic_set` (over set dimensions as dims and ISL
/// parameters as symbols) into an `mlir::IntegerSet`. Returns null `IntegerSet`
/// if the basic set has no constraints (universe) or cannot be represented.
mlir::IntegerSet islBasicSetToIntegerSet(__isl_keep isl_basic_set *bset,
                                         mlir::MLIRContext *ctx);

} // namespace polymer

#endif // POLYMER_SUPPORT_ISLUTILS_H
