//===- ConsumingInterpreterPass.cpp - Interpret and remove transforms -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Pass/Pass.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONSUMINGINTERPRETERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct ConsumingInterpreterPass
    : public enzyme::impl::ConsumingInterpreterPassBase<
          ConsumingInterpreterPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    Operation *entryPoint =
        transform::detail::findTransformEntryPoint(op, nullptr);
    if (!entryPoint)
      return signalPassFailure();

    auto transformModule = dyn_cast<ModuleOp>(entryPoint->getParentOp());
    if (!transformModule) {
      emitError(entryPoint->getLoc())
          << "expected the transform entry point to be located in a module";
      return signalPassFailure();
    }

    transformModule->remove();
    OwningOpRef<ModuleOp> owningTransformModule(transformModule);

    RaggedArray<transform::MappedValue> bindings;
    bindings.push_back(ArrayRef<Operation *>{op});
    if (failed(transform::applyTransformNamedSequence(
            bindings, cast<transform::TransformOpInterface>(entryPoint),
            *owningTransformModule,
            transform::TransformOptions().enableExpensiveChecks(true))))
      return signalPassFailure();
  }
};

} // namespace