//===- ImpulseTransformOps.cpp - Definition of impulse transform extension ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.cpp.inc"
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOpsImpl.cpp.inc"

using namespace mlir;

LogicalResult transform::ApplySicmPDLPatternsOp::verify() {
  for (Operation &op : *getBody())
    if (!isa<pdl::PatternOp>(op))
      return emitOpError() << "expects only pdl.pattern ops in its body";
  return success();
}

void transform::ApplySicmPDLPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  // The PDL bytecode compiles a module of patterns; copy ours into one.
  OwningOpRef<ModuleOp> pdlModule = ModuleOp::create(getLoc());
  OpBuilder builder = OpBuilder::atBlockEnd(pdlModule->getBody());
  for (Operation &pattern : *getBody())
    builder.clone(pattern);
  populateSicmPDLPatterns(patterns, std::move(pdlModule));
}

namespace {
class ImpulseTransformExtension
    : public transform::TransformDialectExtension<ImpulseTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImpulseTransformExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/ImpulseTransformOps.cpp.inc"
        >();
    // SICM rules emit enzymexla; sicm_pdl compiles to pdl_interp when applied.
    declareGeneratedDialect<enzymexla::EnzymeXLADialect>();
    declareGeneratedDialect<pdl_interp::PDLInterpDialect>();
  }
};
} // namespace

void mlir::enzyme::registerImpulseTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ImpulseTransformExtension>();
  registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
}
