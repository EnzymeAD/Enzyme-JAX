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
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
class ConsumingInterpreterPass
    : public ConsumingInterpreterPassBase<ConsumingInterpreterPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConsumingInterpreterPass)

  StringRef getArgument() const override {
    return "enzyme-consuming-transform-interpreter";
  }

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

std::unique_ptr<Pass> mlir::enzyme::createConsumingInterpreterPass() {
  return std::make_unique<ConsumingInterpreterPass>();
}
