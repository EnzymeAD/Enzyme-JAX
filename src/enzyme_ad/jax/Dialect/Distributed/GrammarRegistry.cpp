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

} // namespace

void GrammarRegistry::registerEntry(StringRef kind, GrammarKindEntry entry) {
  [[maybe_unused]] bool inserted =
      entries.try_emplace(kind, std::move(entry)).second;
  assert(inserted && "grammar kind registered twice");
}

const GrammarKindEntry *GrammarRegistry::lookup(StringRef kind) const {
  auto it = entries.find(kind);
  return it == entries.end() ? nullptr : &it->second;
}

FailureOr<BaseSignature> getBaseSignature(Operation *op) {
  if (auto term = dyn_cast<GrammarTermOp>(op)) {
    const GrammarRegistry &registry =
        cast<DistributedDialect>(term->getDialect())->getGrammarRegistry();
    const GrammarKindEntry *entry = registry.lookup(term.getKind());
    if (!entry) {
      term.emitWarning() << "no grammar registry entry for kind \""
                         << term.getKind() << "\"";
      return failure();
    }
    return entry->getBaseSignature(term.getAxes(), term.getParamsAttr());
  }

  OpShardingRuleAndReductionKind rule = getOrSynthesizeOpShardingRule(op);
  if (!rule.rule)
    return failure();
  BaseSignature signature;
  for (int64_t factor = 0; factor < rule.rule.getNumFactors(); ++factor)
    signature.axes.push_back(shardyFactorSignature(rule, factor));
  return signature;
}

} // namespace mlir::enzyme::distributed
