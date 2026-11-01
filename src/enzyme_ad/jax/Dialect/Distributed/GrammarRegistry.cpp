#include "GrammarRegistry.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

namespace {

// Shardy factor types map onto the base case: reductions combine partial
// results, need-replication and permutation factors can't be sharded, and
// pass-through factors are carried through.
AxisSignature shardyFactorSignature(const OpShardingRuleAndReductionKind &rule,
                                    int64_t factor) {
  switch (rule.rule.getFactorType(factor)) {
  case sdy::FactorType::kPassThrough:
    return AxisSignature::parallel();
  case sdy::FactorType::kReduction:
    return AxisSignature::reduce(rule.reductionKind);
  case sdy::FactorType::kNeedReplication:
  case sdy::FactorType::kPermutation:
    return AxisSignature::local();
  }
  llvm_unreachable("unknown Shardy factor type");
}

BaseSignature shardyRuleSignature(const OpShardingRuleAndReductionKind &rule) {
  BaseSignature signature;
  for (int64_t factor = 0; factor < rule.rule.getNumFactors(); ++factor)
    signature.axes.push_back(shardyFactorSignature(rule, factor));
  return signature;
}

// A GrammarTerm already carries its kind and operator axes; its signature
// comes from the kind's entry.
FailureOr<GrammarDescription>
describeGrammarTerm(const GrammarRegistry &registry, GrammarTermOp term) {
  const GrammarKindEntry *entry = registry.lookup(term.getKind());
  if (!entry) {
    term.emitWarning() << "no grammar registry entry for kind \""
                       << term.getKind() << "\"";
    return failure();
  }
  return GrammarDescription{
      term.getKind(), term.getAxes(), term.getParamsAttr(),
      entry->getBaseSignature(term.getAxes(), term.getParamsAttr())};
}

// The adapter that recognize() falls back on. See its header comment for the
// one-sided dimensions.
FailureOr<GrammarDescription> describeByShardyRule(Operation *op) {
  OpShardingRuleAndReductionKind rule = getOrSynthesizeOpShardingRule(op);
  if (!rule.rule)
    return failure();
  // Ops reach this from arbitrary input IR, where Shardy rules can exist for
  // dynamic shapes that no operator axis can describe.
  auto isStatic = [](Type type) {
    auto tensorType = dyn_cast<RankedTensorType>(type);
    return tensorType && tensorType.hasStaticShape();
  };
  if (!llvm::all_of(op->getOperandTypes(), isStatic) ||
      !llvm::all_of(op->getResultTypes(), isStatic))
    return failure();

  MLIRContext *context = op->getContext();
  ArrayRef<int64_t> factorSizes = rule.rule.getFactorSizes();
  SmallVector<int64_t> extents(factorSizes);
  BaseSignature signature = shardyRuleSignature(rule);
  auto fitToShapes = [&](ArrayRef<sdy::TensorMappingAttr> mappings,
                         TypeRange types) {
    SmallVector<sdy::TensorMappingAttr> fitted;
    for (auto [mapping, type] : llvm::zip_equal(mappings, types)) {
      ArrayRef<int64_t> shape = cast<RankedTensorType>(type).getShape();
      SmallVector<sdy::DimMappingAttr> dims;
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        int64_t product = 1;
        for (int64_t factor : dimMapping.getFactorIndices())
          product *= factorSizes[factor];
        if (product == shape[dim]) {
          dims.push_back(dimMapping);
          continue;
        }
        dims.push_back(
            sdy::DimMappingAttr::get(context, {(int64_t)extents.size()}));
        extents.push_back(shape[dim]);
        signature.axes.push_back(AxisSignature::local());
      }
      fitted.push_back(sdy::TensorMappingAttr::get(context, dims));
    }
    return fitted;
  };
  // Every factor Shardy builds is sized from some tensor's dimension, so each
  // keeps at least one use and the attribute stays valid.
  SmallVector<sdy::TensorMappingAttr> operandMappings =
      fitToShapes(rule.rule.getOperandMappings(), op->getOperandTypes());
  SmallVector<sdy::TensorMappingAttr> resultMappings =
      fitToShapes(rule.rule.getResultMappings(), op->getResultTypes());
  auto axes =
      OperatorAxesAttr::get(context, extents, operandMappings, resultMappings);
  return GrammarDescription{op->getName().getStringRef(), axes,
                            /*params=*/nullptr, std::move(signature)};
}

StringRef reduceKindName(stablehlo::ReduceOpKind kind) {
  switch (kind) {
  case stablehlo::ReduceOpKind::Unknown:
    return "unknown";
  case stablehlo::ReduceOpKind::Add:
    return "add";
  case stablehlo::ReduceOpKind::Min:
    return "min";
  case stablehlo::ReduceOpKind::Max:
    return "max";
  case stablehlo::ReduceOpKind::Mul:
    return "mul";
  case stablehlo::ReduceOpKind::And:
    return "and";
  case stablehlo::ReduceOpKind::Or:
    return "or";
  case stablehlo::ReduceOpKind::Xor:
    return "xor";
  }
  llvm_unreachable("covered switch");
}

} // namespace

GrammarRegistry &getGrammarRegistry(MLIRContext *context) {
  auto *dialect = context->getLoadedDialect<DistributedDialect>();
  assert(dialect && "the distributed dialect must be loaded");
  return dialect->getGrammarRegistry();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const AxisSignature &signature) {
  switch (signature.getKind()) {
  case AxisSignature::Kind::Local:
    return os << "local";
  case AxisSignature::Kind::Parallel:
    return os << "parallel";
  case AxisSignature::Kind::Reduce:
    return os << "reduce(" << reduceKindName(signature.getReduceKind()) << ")";
  case AxisSignature::Kind::Custom:
    return os << "custom";
  case AxisSignature::Kind::CustomTerminal:
    return os << "custom-terminal";
  }
  llvm_unreachable("covered switch");
}

void GrammarRegistry::registerEntry(StringRef kind, GrammarKindEntry entry) {
  [[maybe_unused]] bool inserted =
      entries.try_emplace(kind, std::move(entry)).second;
  assert(inserted && "grammar kind registered twice");
}

const GrammarKindEntry *GrammarRegistry::lookup(StringRef kind) const {
  auto it = entries.find(kind);
  return it == entries.end() ? nullptr : &it->second;
}

void GrammarRegistry::registerRecognizer(GrammarRecognizer recognizer) {
  recognizers.push_back(std::move(recognizer));
}

FailureOr<GrammarDescription> recognize(Operation *op) {
  const GrammarRegistry &registry = getGrammarRegistry(op->getContext());
  if (auto term = dyn_cast<GrammarTermOp>(op))
    return describeGrammarTerm(registry, term);

  for (const GrammarRecognizer &recognizer : registry.getRecognizers()) {
    std::optional<GrammarDescription> description = recognizer(op);
    if (!description)
      continue;
    const GrammarKindEntry *entry = registry.lookup(description->kind);
    assert(entry && "recognizer produced an unregistered grammar kind");
    description->signature =
        entry->getBaseSignature(description->axes, description->params);
    return *description;
  }
  return describeByShardyRule(op);
}

GrammarTermOp materializeGrammarTerm(OpBuilder &builder, Location loc,
                                     const GrammarDescription &description,
                                     ValueRange operands,
                                     TypeRange resultTypes) {
  assert(getGrammarRegistry(builder.getContext()).lookup(description.kind) &&
         "only registered grammar kinds are materialized");
  return builder.create<GrammarTermOp>(loc, resultTypes, operands,
                                       description.kind, description.axes,
                                       description.params);
}

} // namespace mlir::enzyme::distributed
