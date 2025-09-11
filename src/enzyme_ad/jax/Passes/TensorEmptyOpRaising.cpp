//===- TensorEmptyOpRaising.cpp - Raise tensor::EmptyOp dialect
//----------------//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise tensor empty ops to the stablehlo.
// dialect.
//
//===---------------------------------------------------------------------===//

#include "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TENSOREMPTYOPRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace enzyme;

namespace {
struct TensorEmptyOpRaisingPass
    : public enzyme::impl::TensorEmptyOpRaisingPassBase<
          TensorEmptyOpRaisingPass> {
  using TensorEmptyOpRaisingPassBase::TensorEmptyOpRaisingPassBase;

  void runOnOperation() override {
    auto op = getOperation();

    op->walk([](tensor::EmptyOp eop) {
      OpBuilder builder(eop);

      // Dynamic shape
      if (eop->getNumOperands() > 0)
        return;

      auto ty = cast<RankedTensorType>(eop.getResult().getType());
      auto ety = ty.getElementType();

      Attribute attr = makeAttr(ety, 0);

      Value replacementValue = builder.create<stablehlo::ConstantOp>(
          eop.getLoc(), ty, SplatElementsAttr::get(ty, attr));

      eop.replaceAllUsesWith(replacementValue);
      eop.erase();
    });
  }
};
} // end anonymous namespace
