//===- PDLLTransformHelpers.h - Configurable PDLL helpers --------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Core C++ logic for the configurable PDLL patterns defined in
// `Passes/PDLL/patterns.pdll`. The functions here are intentionally
// decoupled from PDLL/PatternMatcher boilerplate so they can be reused
// from native C++ patterns or bound to PDLL externals at different
// configuration levels.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_PDLL_TRANSFORM_HELPERS_H
#define ENZYME_AD_JAX_PASSES_PDLL_TRANSFORM_HELPERS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace stablehlo {
class SliceOp;
} // namespace stablehlo
namespace enzymexla {
class ExtendOp;
} // namespace enzymexla

namespace enzyme {

/// Controls how aggressively configurable PDLL patterns rewrite the IR.
///
/// `PureLocal`     -- skip every "is this profitable?" check and run the
///                    minimal local rewrite. Patterns fire whenever the
///                    structural match succeeds.
/// `CheckedLocal`  -- run the profitability check, but only perform the
///                    minimal local rewrite once it succeeds.
/// `CheckedGlobal` -- run the profitability check and perform the larger
///                    rewrite that also deduplicates matching sibling
///                    users in addition to the trigger match.
enum class RewriteExtent { PureLocal, CheckedLocal, CheckedGlobal };

/// Parse a CLI-friendly string ("pure-local", "checked-local",
/// "checked-global") into a `RewriteExtent`. Returns `std::nullopt` if
/// `name` does not match a known mode.
std::optional<RewriteExtent> parseRewriteExtent(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// SliceExtend commute helpers
//===----------------------------------------------------------------------===//

/// Returns success if `sliceOp.getOperand()` has a static shape and
/// `sliceOp` does not modify the dimension being extended by `extendOp`
/// (i.e. start = 0, limit = base dim size, stride = 1 along that dim).
LogicalResult isValidSliceForExtend(::mlir::stablehlo::SliceOp sliceOp,
                                    ::mlir::enzymexla::ExtendOp extendOp);

/// Returns success when at least one OTHER user of `sliceOp.getOperand()`
/// matches the extend characteristics of `extendOp` (either a direct
/// `enzymexla.extend` user or a `stablehlo.slice` whose single user is a
/// matching `enzymexla.extend`).
LogicalResult
hasMultipleMatchingExtendUsers(::mlir::stablehlo::SliceOp sliceOp,
                               ::mlir::enzymexla::ExtendOp extendOp);

/// Performs the SliceExtend commute transformation. Always rewrites the
/// triggering `extendOp` into a new `enzymexla.extend` on the slice base
/// followed by an adjusted `stablehlo.slice`. When
/// `doGlobalDeduplication` is true, also rewrites every other matching
/// sibling user (Direct Extend or Extend-of-Slice) so that they share the
/// new base extend op.
///
/// Returns the newly created replacement op for the trigger extend.
Operation *commuteExtendAndSlice(PatternRewriter &rewriter,
                                 ::mlir::enzymexla::ExtendOp extendOp,
                                 ::mlir::stablehlo::SliceOp sliceOp,
                                 bool doGlobalDeduplication);

/// Register the dynamic PDLL constraint/rewrite bindings for the
/// configurable SliceExtend commute pattern. Must be called after the
/// generated PDLL patterns have been added to `patterns` and before the
/// pattern set is frozen / used.
void registerSliceExtendDynamicPDLLBindings(RewritePatternSet &patterns,
                                            RewriteExtent extent);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_PDLL_TRANSFORM_HELPERS_H
