//===- GenerateApplyPatterns.cpp - Generate transform scripts --------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"

using namespace mlir;

struct OpConfig {
  OperationName name;
  DictionaryAttr attrs;
};

void generatePatternGroup(OpBuilder &builder, Location loc, Value root,
                          ArrayRef<OpConfig> configurations,
                          llvm::APInt selectionBitmask) {
  OpBuilder::InsertionGuard guard(builder);
  auto apply = builder.create<transform::ApplyPatternsOp>(
      loc, root, [](OpBuilder &builder, Location loc) {});
  builder.setInsertionPointToStart(apply.getBody());
  for (auto &&[i, opConfig] : llvm::enumerate(configurations)) {
    if (selectionBitmask.extractBits(/*numBits=*/1, /*bitPosition=*/i).isZero())
      continue;
    OperationState state(loc, opConfig.name);
    state.addAttributes(opConfig.attrs.getValue());
    builder.create(state);
  }
}

LogicalResult generateTransform(OpBuilder &builder, llvm::APInt version) {
  auto loc = builder.getUnknownLoc();
  auto namedSequence = builder.create<transform::NamedSequenceOp>(
      loc, "__transform_main", builder.getType<transform::AnyOpType>(),
      TypeRange(), [](OpBuilder &builder, Location loc, BlockArgument) {
        builder.create<transform::YieldOp>(loc);
      });

  SmallVector<OpConfig> opConfigurations;
  for (StringRef name : mlir::enzyme::getTransformOperationNames()) {
    std::optional<RegisteredOperationName> opName =
        RegisteredOperationName::lookup(name, builder.getContext());
    if (!opName) {
      return namedSequence->emitError() << "unregistered pattern op '" << name
                                        << "' listed for construction";
    }
    auto *conceptV =
        opName->getInterface<SearchablePatternDescriptorOpInterface>();
    for (DictionaryAttr attrs :
         conceptV->getPossibleAttrCombinations(builder)) {
      opConfigurations.push_back(OpConfig{*opName, attrs});
    }
  }

  builder.setInsertionPointToStart(&namedSequence.getBody().front());
  auto match = builder.create<transform::MatchOp>(
      loc, namedSequence.getBody().front().getArgument(0),
      ArrayRef<StringRef>{func::FuncOp::getOperationName()});

  auto configPow = llvm::APInt::getOneBitSet(opConfigurations.size() + 1,
                                             opConfigurations.size());
  do {
    llvm::APInt configuration = version.srem(configPow);
    generatePatternGroup(builder, loc, match, opConfigurations, configuration);
    version = version.sdiv(configPow);
  } while (!version.isZero());
  return success();
}

namespace {
class GenerateApplyPatternsPass
    : public PassWrapper<GenerateApplyPatternsPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateApplyPatternsPass)

  GenerateApplyPatternsPass() = default;
  GenerateApplyPatternsPass(const GenerateApplyPatternsPass &other)
      : PassWrapper<GenerateApplyPatternsPass, OperationPass<>>(other) {}

  StringRef getArgument() const override { return "enzyme-hlo-generate-td"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (op->getNumRegions() != 1 || !llvm::hasSingleElement(op->getRegion(0))) {
      op->emitError()
          << "can only run on a single-region single-block operation";
      return signalPassFailure();
    }

    llvm::APInt version(
        llvm::APInt::getSufficientBitsNeeded(flags.getValue(), radix),
        flags.getValue(), radix);

    OpBuilder builder(&getContext());
    op->setAttr(transform::TransformDialect::kWithNamedSequenceAttrName,
                builder.getUnitAttr());

    builder.setInsertionPointToStart(&op->getRegion(0).front());
    if (failed(generateTransform(builder, version)))
      return signalPassFailure();
  }

  Option<std::string> flags{*this, "flags", llvm::cl::init("")};
  Option<int> radix{*this, "radix", llvm::cl::init(10)};
};

class RemoveTransform : public PassWrapper<RemoveTransform, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveTransform)

  StringRef getArgument() const override {
    return "enzyme-hlo-remove-transform";
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<transform::TransformOpInterface>(op)) {
        op->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

void mlir::enzyme::registerGenerateApplyPatternsPass() {
  PassRegistration<GenerateApplyPatternsPass>();
}

void mlir::enzyme::registerRemoveTransformPass() {
  PassRegistration<RemoveTransform>();
}
