//===- ISLUtils.h - ISL Polyhedral & Affine Utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_ISLUTILS_H
#define POLYMER_SUPPORT_ISLUTILS_H

#include "isl/isl-noexceptions.h"
#include "isl/set.h"
#include "isl/union_map.h"

namespace polymer {

/// Contract intermediate vertices in a tagged dependence graph to compute
/// maximal live-range paths for ISL live-range span scheduling.
isl::union_map get_maximal_paths(isl::union_map umap);

} // namespace polymer

#endif // POLYMER_SUPPORT_ISLUTILS_H
