#include "GrammarRegistry.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

namespace {

// Ops described through their Shardy rule get kinds in the Shardy dialect's
// namespace: `sdy.<op name>`.
StringAttr getShardyKind(Operation *op) {
  return StringAttr::get(op->getContext(),
                         sdy::SdyDialect::getDialectNamespace() + "." +
                             op->getName().getStringRef());
}

bool isShardyKind(StringRef kind) {
  return kind.consume_front(sdy::SdyDialect::getDialectNamespace()) &&
         kind.starts_with(".");
}

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
      term.getKindAttr(), term.getAxes(), term.getParamsAttr(),
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
  return GrammarDescription{getShardyKind(op), axes, /*params=*/nullptr,
                            std::move(signature)};
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

// `description` with each operator axis `axis` replaced by the axes
// `replacement[axis]` (major-first), of extents `newExtents`. Each new axis
// takes the signature of the axis it replaces.
GrammarDescription replaceAxes(const GrammarDescription &description,
                               ArrayRef<int64_t> newExtents,
                               ArrayRef<SmallVector<int64_t>> replacement) {
  OperatorAxesAttr axes = description.axes;
  using TensorDigits = SmallVector<SmallVector<int64_t>>;
  auto replaceDigits = [&](bool isResult) {
    SmallVector<TensorDigits> tensors;
    for (sdy::TensorMappingAttr mapping : axes.getMappings(isResult)) {
      TensorDigits &dims = tensors.emplace_back();
      for (sdy::DimMappingAttr dimMapping : mapping.getDimMappings()) {
        SmallVector<int64_t> &digits = dims.emplace_back();
        for (int64_t axis : dimMapping.getFactorIndices())
          llvm::append_range(digits, replacement[axis]);
      }
    }
    return tensors;
  };

  // Every new axis appears in some replacement, so the placeholder is always
  // overwritten.
  BaseSignature signature{
      SmallVector<AxisSignature>(newExtents.size(), AxisSignature::local())};
  for (auto [axis, newAxes] : llvm::enumerate(replacement))
    for (int64_t newAxis : newAxes)
      signature.axes[newAxis] = description.signature.axes[axis];

  auto newOperatorAxes = OperatorAxesAttr::get(
      axes.getContext(), newExtents, replaceDigits(/*isResult=*/false),
      replaceDigits(/*isResult=*/true));
  return GrammarDescription{description.kind, newOperatorAxes,
                            description.params, std::move(signature)};
}

// Adds to `boundaries` where the digits `digitExtents` (major-first) cut the
// axes `dimAxes` of one dimension. A cut inside an axis is recorded as a
// stride within that axis. Fails if a cut doesn't divide the axis.
LogicalResult
collectDimBoundaries(OperatorAxesAttr axes, ArrayRef<int64_t> dimAxes,
                     ArrayRef<int64_t> digitExtents,
                     MutableArrayRef<SmallVector<int64_t>> boundaries) {
  // Cuts as strides within the dimension, i.e. the product of the digits minor
  // to each cut.
  SmallVector<int64_t> cuts;
  int64_t stride = 1;
  for (int64_t extent : llvm::reverse(digitExtents.drop_front())) {
    stride *= extent;
    cuts.push_back(stride);
  }

  int64_t axisStride = 1;
  for (int64_t axis : llvm::reverse(dimAxes)) {
    int64_t axisEnd = axisStride * axes.getAxisExtent(axis);
    for (int64_t cut : cuts) {
      if (cut <= axisStride || cut >= axisEnd)
        continue;
      if (cut % axisStride != 0 || axisEnd % cut != 0)
        return failure();
      boundaries[axis].push_back(cut / axisStride);
    }
    axisStride = axisEnd;
  }
  assert((digitExtents.empty() ? 1 : stride * digitExtents.front()) ==
             axisStride &&
         "digit extents must multiply to the dimension's size");
  return success();
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
  assert(!isShardyKind(kind) &&
         "the sdy. kind namespace is reserved for Shardy-described ops");
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
    const GrammarKindEntry *entry =
        registry.lookup(description->kind.getValue());
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
  assert(getGrammarRegistry(builder.getContext())
             .lookup(description.kind.getValue()) &&
         "only registered grammar kinds are materialized");
  return builder.create<GrammarTermOp>(loc, resultTypes, operands,
                                       description.kind, description.axes,
                                       description.params);
}

CanonicalizedAxes canonicalizeAxes(const GrammarDescription &description) {
  OperatorAxesAttr axes = description.axes;
  constexpr int64_t kUnnumbered = -1;
  SmallVector<int64_t> perm(axes.getNumAxes(), kUnnumbered);
  int64_t next = 0;
  for (bool isResult : {false, true})
    for (sdy::TensorMappingAttr mapping : axes.getMappings(isResult))
      for (sdy::DimMappingAttr dimMapping : mapping.getDimMappings())
        for (int64_t axis : dimMapping.getFactorIndices())
          if (perm[axis] == kUnnumbered)
            perm[axis] = next++;
  assert(next == axes.getNumAxes() &&
         "OperatorAxesAttr maps every axis on some tensor");

  SmallVector<int64_t> extents(axes.getNumAxes());
  SmallVector<SmallVector<int64_t>> replacement;
  for (auto [axis, canonical] : llvm::enumerate(perm)) {
    extents[canonical] = axes.getAxisExtent(axis);
    replacement.push_back({canonical});
  }
  return {replaceAxes(description, extents, replacement), std::move(perm)};
}

FailureOr<RefinedAxes>
refineAxes(const GrammarDescription &description,
           ArrayRef<SmallVector<SmallVector<int64_t>>> operandDigits,
           ArrayRef<SmallVector<SmallVector<int64_t>>> resultDigits) {
  OperatorAxesAttr axes = description.axes;
  // Per axis, the strides within it at which some dimension cuts it.
  SmallVector<SmallVector<int64_t>> boundaries(axes.getNumAxes());
  for (bool isResult : {false, true}) {
    ArrayRef<SmallVector<SmallVector<int64_t>>> digits =
        isResult ? resultDigits : operandDigits;
    ArrayRef<sdy::TensorMappingAttr> mappings = axes.getMappings(isResult);
    assert(digits.size() == mappings.size() &&
           "digits are given for every tensor");
    for (auto [tensorIdx, mapping] : llvm::enumerate(mappings)) {
      assert(digits[tensorIdx].size() == mapping.getRank() &&
             "digits are given for every dimension");
      for (int64_t dim = 0; dim < mapping.getRank(); ++dim)
        if (failed(collectDimBoundaries(
                axes, axes.getDigits(isResult, tensorIdx, dim),
                digits[tensorIdx][dim], boundaries)))
          return failure();
    }
  }

  // Each axis becomes the pieces between its sorted boundaries, which must
  // nest: a cut at 4 and one at 6 in an axis of 12 have no common refinement.
  SmallVector<int64_t> extents;
  SmallVector<SmallVector<int64_t>> split;
  for (auto [axis, axisBoundaries] : llvm::enumerate(boundaries)) {
    llvm::sort(axisBoundaries);
    axisBoundaries.erase(llvm::unique(axisBoundaries), axisBoundaries.end());
    SmallVector<int64_t> cuts = {1};
    llvm::append_range(cuts, axisBoundaries);
    cuts.push_back(axes.getAxisExtent(axis));

    SmallVector<int64_t> &pieces = split.emplace_back();
    for (int64_t i = cuts.size() - 1; i > 0; --i) {
      if (cuts[i] % cuts[i - 1] != 0)
        return failure();
      pieces.push_back(extents.size());
      extents.push_back(cuts[i] / cuts[i - 1]);
    }
  }
  return RefinedAxes{replaceAxes(description, extents, split),
                     std::move(split)};
}

} // namespace mlir::enzyme::distributed
