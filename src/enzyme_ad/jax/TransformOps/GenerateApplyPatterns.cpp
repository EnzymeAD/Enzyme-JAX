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

Value generateTransformMain(OpBuilder &builder, Location loc) {
  auto namedSequence = builder.create<transform::NamedSequenceOp>(
      loc, "__transform_main", builder.getType<transform::AnyOpType>(),
      TypeRange(), [](OpBuilder &builder, Location loc, BlockArgument) {
        builder.create<transform::YieldOp>(loc);
      });
  builder.setInsertionPointToStart(&namedSequence.getBody().front());
  auto match = builder.create<transform::MatchOp>(
      loc, namedSequence.getBody().front().getArgument(0),
      ArrayRef<StringRef>{func::FuncOp::getOperationName()});
  return match;
}

LogicalResult generateTransform(OpBuilder &builder, llvm::APInt version) {
  auto loc = builder.getUnknownLoc();
  Value match = generateTransformMain(builder, loc);

  SmallVector<OpConfig> opConfigurations;
  for (StringRef name : mlir::enzyme::getTransformOperationNames()) {
    std::optional<RegisteredOperationName> opName =
        RegisteredOperationName::lookup(name, builder.getContext());
    if (!opName) {
      return emitError(loc) << "unregistered pattern op '" << name
                            << "' listed for construction";
    }
    auto *conceptV =
        opName->getInterface<SearchablePatternDescriptorOpInterface>();
    for (DictionaryAttr attrs :
         conceptV->getPossibleAttrCombinations(builder)) {
      opConfigurations.push_back(OpConfig{*opName, attrs});
    }
  }

  if (version.getBitWidth() < opConfigurations.size() + 1)
    version = version.zext(opConfigurations.size() + 1);

  auto configPow =
      llvm::APInt::getOneBitSet(version.getBitWidth(), opConfigurations.size());
  do {
    llvm::APInt configuration = version.srem(configPow);
    generatePatternGroup(builder, loc, match, opConfigurations, configuration);
    version = version.sdiv(configPow);
  } while (!version.isZero());
  return success();
}

LogicalResult parseTransform(OpBuilder &builder, Location loc,
                             StringRef patterns) {
  Value root = generateTransformMain(builder, loc);
  auto apply = builder.create<transform::ApplyPatternsOp>(
      loc, root, [](OpBuilder &builder, Location loc) {});
  builder.setInsertionPointToStart(apply.getBody());

  SmallVector<StringRef> singlePatterns;
  patterns.split(singlePatterns, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef pattern : singlePatterns) {
    pattern = pattern.trim();
    size_t pos = pattern.find_first_of("<(");
    StringRef opName =
        pos == std::string::npos ? pattern : pattern.take_front(pos).trim();
    StringRef remainder =
        pos == std::string::npos ? "" : pattern.drop_front(pos);

    int64_t benefit = 1;
    if (remainder.starts_with("<")) {
      size_t closing = remainder.find('>');
      if (closing == std::string::npos) {
        return ::emitError(loc)
               << "couldn't find matching '>' in " << remainder;
      }
      StringRef benefitStr = remainder.drop_front().take_front(closing - 1);
      if (benefitStr.getAsInteger(0, benefit)) {
        return ::emitError(loc) << "couldn't parse benefit: " << benefitStr;
      }
      remainder = remainder.drop_front(closing + 1).trim();
    }

    int64_t parameter = -1;
    if (remainder.starts_with("(")) {
      if (!remainder.ends_with(")")) {
        return ::emitError(loc)
               << "couldn't find the closing ')' in " << remainder;
      }
      StringRef parameterStr = remainder.drop_front().drop_back();
      if (parameterStr.getAsInteger(0, parameter)) {
        return ::emitError(loc) << "couldn't parse parameter: " << parameterStr;
      }
    }

    std::string potentialOpName =
        "transform.apply_patterns.enzyme_hlo." + opName.str();
    if (!RegisteredOperationName::lookup(potentialOpName,
                                         builder.getContext())) {
      potentialOpName = "transform.apply_patterns." + opName.str();
      if (!RegisteredOperationName::lookup(potentialOpName,
                                           builder.getContext())) {
        return ::emitError(loc)
               << "couldn't find a pattern operation corresponding to "
               << opName;
      }
    }

    OperationState state(loc, potentialOpName);
    if (benefit != 1)
      state.addAttribute("benefit", builder.getI64IntegerAttr(benefit));
    if (parameter != -1) {
      if (opName == "no_nan_add_sub_simplify")
        state.addAttribute("parameter", builder.getBoolAttr(parameter));
      else
        state.addAttribute("parameter", builder.getI64IntegerAttr(parameter));
    }
    builder.create(state);
  }
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
    if (!flags.getValue().empty() && !patterns.getValue().empty()) {
      op->emitError() << "flags and patterns are mutually exclusive";
      return signalPassFailure();
    }
    if (op->getNumRegions() != 1 || !llvm::hasSingleElement(op->getRegion(0))) {
      op->emitError()
          << "can only run on a single-region single-block operation";
      return signalPassFailure();
    }

    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&op->getRegion(0).front());
    if (createModule) {
      auto transformModule = builder.create<ModuleOp>(op->getLoc());
      op = transformModule;
      builder.setInsertionPointToStart(&op->getRegion(0).front());
    }
    op->setAttr(transform::TransformDialect::kWithNamedSequenceAttrName,
                builder.getUnitAttr());

    if (!flags.empty()) {
      llvm::APInt version(
          llvm::APInt::getSufficientBitsNeeded(flags.getValue(), radix) + 1,
          flags.getValue(), radix);
      if (failed(generateTransform(builder, version)))
        return signalPassFailure();
    } else {
      if (failed(parseTransform(builder, op->getLoc(), patterns)))
        return signalPassFailure();
    }
  }

  Option<std::string> flags{*this, "flags", llvm::cl::init("")};
  Option<int> radix{*this, "radix", llvm::cl::init(10)};
  Option<std::string> patterns{*this, "patterns", llvm::cl::init("")};
  Option<bool> createModule{*this, "create-module", llvm::cl::init(false)};
};

class RemoveTransform : public PassWrapper<RemoveTransform, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveTransform)

  StringRef getArgument() const override {
    return "enzyme-hlo-remove-transform";
  }

  void runOnOperation() override {
    auto op = getOperation();
    if (op->hasAttr(transform::TransformDialect::kWithNamedSequenceAttrName))
      op->removeAttr(transform::TransformDialect::kWithNamedSequenceAttrName);
    op->walk<WalkOrder::PreOrder>([&](Operation *op) {
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
