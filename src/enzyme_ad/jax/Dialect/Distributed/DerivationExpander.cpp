#include "DerivationExpander.h"

#include "llvm/ADT/BitVector.h"

namespace mlir::enzyme::distributed {

namespace {

// Whether each of `peeledAxes` lies on exactly the outputs that depend on an
// input carrying it, as a term's batch does (GrammarKindEntry::peelBatch).
[[maybe_unused]] bool peelsOntoDependentOutputs(StringRef kind,
                                                OperatorAxesAttr roles,
                                                DictionaryAttr params,
                                                ArrayRef<int64_t> peeledAxes) {
  SmallVector<SmallVector<int64_t>> axisIndices;
  for (int64_t axis = 0; axis < roles.getNumAxes(); ++axis)
    axisIndices.push_back({axis});
  auto carries = [&](bool isResult, int64_t position, int64_t axis) {
    return llvm::is_contained(
        getRoleFactors(roles, axisIndices, isResult, position), axis);
  };
  SmallVector<SmallVector<int64_t>> dependence =
      getDependence(roles.getContext(), kind, roles, params);
  for (int64_t axis : peeledAxes)
    for (auto [output, inputs] : llvm::enumerate(dependence)) {
      bool fromInput = llvm::any_of(inputs, [&](int64_t input) {
        return carries(/*isResult=*/false, input, axis);
      });
      if (carries(/*isResult=*/true, output, axis) != fromInput)
        return false;
    }
  return true;
}

} // namespace

KeyMap PropagationKey::fromRoles(StringAttr kind, const GrammarKindEntry *entry,
                                 OperatorAxesAttr roles, DictionaryAttr params,
                                 const BaseSignature *descriptionSignature,
                                 SmallVector<Type> inputElementTypes,
                                 SmallVector<Type> outputElementTypes) {
  PeeledBatch peeled;
  if (entry && entry->peelBatch) {
    peeled = entry->peelBatch(roles, params);
  } else {
    peeled.innerRoles = roles;
    peeled.innerParams = params;
    peeled.innerAxes =
        llvm::to_vector(llvm::seq<int64_t>(0, roles.getNumAxes()));
  }
  assert((entry || descriptionSignature) &&
         "a kind without a registry entry needs its signature given");
  BaseSignature signature =
      entry ? entry->getBaseSignature(peeled.innerRoles, peeled.innerParams)
            : *descriptionSignature;
  // A registered kind's signature on the peeled roles must agree with the
  // signature its description was given on the full roles.
  assert((!entry || !descriptionSignature ||
          llvm::all_of(
              llvm::enumerate(peeled.innerAxes),
              [&](auto innerAndAxis) {
                auto [innerAxis, axis] = innerAndAxis;
                return descriptionSignature->axes[axis].isBaseShardable() ==
                       signature.axes[innerAxis].isBaseShardable();
              })) &&
         "the description's signature disagrees with its kind's");
  assert(peelsOntoDependentOutputs(kind.getValue(), roles, params,
                                   peeled.batchAxes) &&
         "a kind's peeled axis lies on exactly the outputs that depend on an "
         "input carrying it");
  CanonicalizedAxes canonical = canonicalizeAxes(GrammarDescription{
      kind, peeled.innerRoles, peeled.innerParams, std::move(signature)});
  assert(distributed::getDependence(kind.getContext(), kind.getValue(), roles,
                                    params) ==
             distributed::getDependence(kind.getContext(), kind.getValue(),
                                        canonical.description.axes,
                                        canonical.description.params) &&
         "a kind's dependence doesn't change under peeling and renumbering");

  KeyMap result;
  PropagationKey &key = result.key;
  key.baseShardable = llvm::map_to_vector(
      canonical.description.signature.axes,
      [](const AxisSignature &axis) { return axis.isBaseShardable(); });
  // A registered kind's signature follows from its roles and params. An
  // unregistered kind has no steps, so its spec keeps no signature, and keys
  // differing only in reduce kinds stay equal.
  if (!entry)
    canonical.description.signature = BaseSignature();
  key.spec =
      OperatorSpec{std::move(canonical.description),
                   std::move(inputElementTypes), std::move(outputElementTypes)};

  result.axisMap.resize(roles.getNumAxes(), KeyAxisRef::keyAxis(0));
  llvm::BitVector mapped(roles.getNumAxes());
  for (auto [innerAxis, axis] : llvm::enumerate(peeled.innerAxes)) {
    result.axisMap[axis] = KeyAxisRef::keyAxis(canonical.perm[innerAxis]);
    mapped.set(axis);
  }
  int64_t numPeeled = 0;
  for (bool isResult : {false, true})
    for (sdy::TensorMappingAttr mapping : roles.getMappings(isResult))
      for (sdy::DimMappingAttr dim : mapping.getDimMappings())
        for (int64_t axis : dim.getFactorIndices())
          if (!mapped.test(axis)) {
            result.axisMap[axis] = KeyAxisRef::peeledAxis(numPeeled++);
            mapped.set(axis);
          }
  return result;
}

FailureOr<KeyMap> PropagationKey::from(GrammarTermOp term) {
  if (!getGrammarRegistry(term.getContext()).lookup(term.getKind())) {
    term.emitWarning() << "grammar kind '" << term.getKind()
                       << "' is not registered, so the term has no key";
    return failure();
  }
  return from(specOf(term, /*refined=*/false));
}

KeyMap PropagationKey::from(const TypedDescription &td) {
  auto elementTypes = [](ArrayRef<Type> types) {
    return llvm::map_to_vector(types, [](Type type) {
      return cast<ShapedType>(type).getElementType();
    });
  };
  const GrammarDescription &description = td.description;
  const GrammarKindEntry *entry =
      getGrammarRegistry(description.kind.getContext())
          .lookup(description.kind.getValue());
  return fromRoles(description.kind, entry, description.axes,
                   description.params, &description.signature,
                   elementTypes(td.operandTypes), elementTypes(td.resultTypes));
}

KeyMap PropagationKey::from(const OperatorSpec &spec) {
  const GrammarKindEntry *entry =
      getGrammarRegistry(spec.getKind().getContext())
          .lookup(spec.getKind().getValue());
  assert(entry && "a spec is of a registered kind");
  return fromRoles(spec.getKind(), entry, spec.getRoles(),
                   spec.description.params, /*descriptionSignature=*/nullptr,
                   spec.inputElementTypes, spec.outputElementTypes);
}

const OperatorSpec &PropagationKey::getSpec() const {
  assert(
      getGrammarRegistry(getKind().getContext()).lookup(getKind().getValue()) &&
      "only keys of registered kinds have a spec");
  return spec;
}

bool PropagationKey::operator==(const PropagationKey &other) const {
  return spec == other.spec && baseShardable == other.baseShardable;
}

// The signature is left out: it follows from the rest of the spec.
llvm::hash_code hash_value(const PropagationKey &key) {
  return llvm::hash_combine(
      key.getKind(), key.getParams(), key.getRoles(),
      llvm::hash_combine_range(key.baseShardable.begin(),
                               key.baseShardable.end()),
      llvm::hash_combine_range(key.getInputElementTypes().begin(),
                               key.getInputElementTypes().end()),
      llvm::hash_combine_range(key.getOutputElementTypes().begin(),
                               key.getOutputElementTypes().end()));
}

// Prints e.g. `dft roles <([i])->([j]) {i=8, j=8}> base-shardable [] :
// (complex<f32>) -> (complex<f32>)`, listing the base-shardable axes by
// number.
void PropagationKey::print(raw_ostream &os) const {
  os << getKind().getValue();
  if (getParams())
    os << " params " << getParams();
  os << " roles " << getRoles() << " base-shardable [";
  llvm::interleaveComma(llvm::make_filter_range(
                            llvm::seq<int64_t>(0, baseShardable.size()),
                            [&](int64_t axis) { return baseShardable[axis]; }),
                        os);
  os << "] : (";
  llvm::interleaveComma(getInputElementTypes(), os);
  os << ") -> (";
  llvm::interleaveComma(getOutputElementTypes(), os);
  os << ")";
}

axis::AtomSet replicatedDigits(const StepGraph &graph,
                               const StepGraph::Occurrence &occurrence,
                               int64_t index) {
  // The term verifier keeps every other position's role digits off this
  // position's tensor, so removing all of the occurrence's key-axis digits
  // leaves just the tensor's batch.
  axis::AtomSet keyDigits;
  for (ArrayRef<DigitId> digits : occurrence.axes)
    llvm::append_range(keyDigits, digits);
  llvm::sort(keyDigits);
  return axis::atomSetMinus(graph.tensors[occurrence.tensorIndices[index]],
                            keyDigits);
}

BottomUpExpander::BottomUpExpander(MLIRContext *context) : context(context) {
  assert(context->getLoadedDialect<func::FuncDialect>() &&
         context->getLoadedDialect<axis::AxisDialect>() &&
         context->getLoadedDialect<DistributedDialect>() &&
         "the expander builds func, axis and distributed IR");
}

KeyId BottomUpExpander::internKey(PropagationKey key) {
  auto [found, inserted] = ids.try_emplace(key, keys.size());
  if (inserted) {
    keys.push_back(std::move(key));
    states.emplace_back();
  }
  return found->second;
}

FailureOr<KeyId> BottomUpExpander::intern(GrammarTermOp term) {
  FailureOr<KeyMap> keyMap = PropagationKey::from(term);
  if (failed(keyMap))
    return failure();
  return internKey(std::move(keyMap->key));
}

std::pair<KeyId, SmallVector<KeyAxisRef>>
BottomUpExpander::intern(const TypedDescription &td) {
  KeyMap keyMap = PropagationKey::from(td);
  return {internKey(std::move(keyMap.key)), std::move(keyMap.axisMap)};
}

ArrayRef<Step> BottomUpExpander::steps(KeyId id) {
  KeyState &state = states[id];
  if (state.listed)
    return state.steps;
  state.listed = true;
  state.steps.push_back(Step{std::nullopt, nullptr});
  // Kinds without rules, Shardy-described ones among them, never need an
  // instance.
  if (!getGrammarRegistry(context)
           .getRules(keys[id].getKind().getValue())
           .empty())
    for (auto [rule, choice] : choices(buildInstance(keys[id].getSpec())))
      state.steps.push_back(Step{rule, choice});
  state.graphs.resize(state.steps.size());
  return state.steps;
}

const StepGraph &BottomUpExpander::graph(StepRef ref) {
  // Listing the key's steps, if they aren't yet, makes room for its graphs.
  [[maybe_unused]] const Step &step = this->step(ref);
  assert(!step.isBase() && "a base step is a terminal and has no graph");
  if (const std::optional<StepGraph> &built = states[ref.key].graphs[ref.index])
    return *built;
  return *graphs(ref.key, {ref.index}).front();
}

SmallVector<const StepGraph *>
BottomUpExpander::graphs(KeyId id, ArrayRef<unsigned> stepIndexes) {
  ArrayRef<Step> keySteps = steps(id);
  KeyState &state = states[id];
  std::optional<Instance> instance;
  SmallVector<const StepGraph *> result;
  for (unsigned index : stepIndexes) {
    const Step &step = keySteps[index];
    assert(!step.isBase() && "a base step is a terminal and has no graph");
    std::optional<StepGraph> &graph = state.graphs[index];
    if (!graph) {
      if (!instance)
        instance = buildInstance(keys[id].getSpec());
      graph =
          internOccurrences(indexChoice(*instance, *step.rule, step.choice));
    }
    result.push_back(&*graph);
  }
  return result;
}

RebuiltStep BottomUpExpander::rebuild(StepRef ref) {
  const Step &step = this->step(ref);
  assert(!step.isBase() && "a base step is its key's instance, not a rule's");
  // The graph is built first in every build mode, so that the debug check
  // interns no key that a release build wouldn't.
  [[maybe_unused]] const StepGraph &expected = graph(ref);
  RebuiltStep rebuilt = buildChoice(buildInstance(keys[ref.key].getSpec()),
                                    *step.rule, step.choice);
  assert(index(*rebuilt.func, rebuilt.roots) == expected &&
         "rules are deterministic: a rebuilt step indexes to its graph");
  return rebuilt;
}

StepGraph BottomUpExpander::index(func::FuncOp func,
                                  ArrayRef<AxisFactor> externalAxes) {
  return internOccurrences(indexStep(func, externalAxes));
}

StepGraph BottomUpExpander::internOccurrences(StepIndex index) {
  StepGraph graph;
  graph.digitExtents = std::move(index.digitExtents);
  graph.external = std::move(index.external);
  graph.tensors = std::move(index.tensors);
  graph.inputs = std::move(index.inputs);
  graph.outputs = std::move(index.outputs);
  for (StepIndex::Occurrence &occurrence : index.occurrences) {
    KeyMap keyMap = PropagationKey::from(occurrence.spec);
    StepGraph::Occurrence &keyed = graph.occurrences.emplace_back();
    keyed.axes.resize(llvm::count_if(
        keyMap.axisMap, [](const KeyAxisRef &ref) { return !ref.peeled; }));
    for (auto [axis, ref] : llvm::enumerate(keyMap.axisMap))
      if (!ref.peeled)
        keyed.axes[ref.index] = std::move(occurrence.axes[axis]);
    keyed.tensorIndices = std::move(occurrence.tensorIndices);
    keyed.key = internKey(std::move(keyMap.key));
  }
  return graph;
}

namespace {

// Splices `derivation` into `term` and recurses into the children. Unlike
// materialize, it does not check the whole func.
void materializeRecursively(BottomUpExpander &expander,
                            const Derivation &derivation, GrammarTermOp term) {
  assert([&] {
    FailureOr<KeyMap> keyMap = PropagationKey::from(term);
    return succeeded(keyMap) &&
           keyMap->key == expander.key(derivation.step.key);
  }() && "a derivation is materialized into a term of its key");
  if (expander.step(derivation.step).isBase())
    return;
  // The step's IR is dropped before recursing into its children.
  SmallVector<GrammarTermOp> spliced;
  {
    SmallVector<KeyAxisRef> axisMap = PropagationKey::from(term)->axisMap;
    RebuiltStep step = expander.rebuild(derivation.step);
    assert(derivation.children.size() == step.occurrences.size() &&
           "a derivation has one child per occurrence of its step");
    SmallVector<SmallVector<Value>> operatorAxisFactors =
        term.getOperatorAxisFactors();
    SmallVector<SmallVector<AxisFactor>> bound(step.roots.size());
    for (auto [operatorAxis, ref] : llvm::enumerate(axisMap))
      if (!ref.peeled)
        bound[ref.index] = llvm::map_to_vector(
            operatorAxisFactors[operatorAxis],
            [](Value factor) { return cast<AxisFactor>(factor); });
    assert(checkSplice(step, bound).empty() &&
           "a step's cuts of a key axis nest with the factors the axis stands "
           "for; binding them otherwise needs a relayout");
    spliced = splice(step, term, bound);
  }
  for (auto [child, splicedTerm] :
       llvm::zip_equal(derivation.children, spliced))
    materializeRecursively(expander, child, splicedTerm);
}

} // namespace

void materialize(BottomUpExpander &expander, const Derivation &derivation,
                 GrammarTermOp term) {
#ifndef NDEBUG
  auto func = term->getParentOfType<func::FuncOp>();
  assert(func && "derivation IR lives in a func.func");
  bool nestedBefore = rootCutsNest(func);
#endif
  materializeRecursively(expander, derivation, term);
  assert((!nestedBefore || rootCutsNest(func)) &&
         "the spliced terms cut each root at nesting boundaries");
}

} // namespace mlir::enzyme::distributed
