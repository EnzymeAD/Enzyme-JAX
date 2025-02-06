//===---- RaisingTransformOps.h - Declarations of Transform extension  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.h.inc"
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformPatterns.h.inc"

namespace mlir {
namespace enzyme {
void registerRaisingTransformExtension(mlir::DialectRegistry &registry);

} // namespace enzyme
} // namespace mlir

namespace mlir {
namespace transform {
struct RemoveIVs : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ForOp forOp,
                                mlir::PatternRewriter &rewriter) const override;
};

struct NormalizeLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace transform
} // namespace mlir
