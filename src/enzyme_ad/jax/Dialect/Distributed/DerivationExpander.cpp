#include "DerivationExpander.h"

#include "mlir/IR/Verifier.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallBitVector.h"

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

KeyMap<PropagationKey> PropagationKey::fromRoles(
    StringAttr kind, const GrammarKindEntry *entry, OperatorAxesAttr roles,
    DictionaryAttr params, const BaseSignature *descriptionSignature,
    SmallVector<Type> inputElementTypes, SmallVector<Type> outputElementTypes) {
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

  KeyMap<PropagationKey> result;
  PropagationKey &key = result.key;
  key.kind = kind;
  key.params = canonical.description.params;
  key.roles = canonical.description.axes;
  key.baseShardable = llvm::map_to_vector(
      canonical.description.signature.axes,
      [](const AxisSignature &axis) { return axis.isBaseShardable(); });
  key.inputElementTypes = std::move(inputElementTypes);
  key.outputElementTypes = std::move(outputElementTypes);

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

FailureOr<KeyMap<PropagationKey>> PropagationKey::from(GrammarTermOp term) {
  StringAttr kind = term.getKindAttr();
  const GrammarKindEntry *entry =
      getGrammarRegistry(term.getContext()).lookup(kind.getValue());
  if (!entry) {
    term.emitWarning() << "grammar kind '" << kind.getValue()
                       << "' is not registered, so the term has no key";
    return failure();
  }
  auto elementType = [](Value tensor) {
    return cast<DerivationTensorType>(tensor.getType()).getElementType();
  };
  SmallVector<Type> inputElementTypes =
      llvm::map_to_vector(term.getInputs(), elementType);
  SmallVector<Type> outputElementTypes;
  for (int64_t position = 0;
       position < static_cast<int64_t>(term.getOutputs().size()); ++position)
    outputElementTypes.push_back(
        elementType(term.getTensor(/*isOutput=*/true, position).getTensor()));
  return fromRoles(kind, entry, term.getRoles(), term.getParamsAttr(),
                   /*descriptionSignature=*/nullptr,
                   std::move(inputElementTypes), std::move(outputElementTypes));
}

KeyMap<PropagationKey> PropagationKey::from(const TypedDescription &td) {
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

KeyInstance PropagationKey::buildInstance(DerivationBuilder &builder) const {
  assert(getGrammarRegistry(kind.getContext()).lookup(kind.getValue()) &&
         "only keys of registered kinds have an instance");
  KeyInstance instance;
  SmallVector<OperatorAxis> products;
  SmallVector<SmallVector<AxisFactor>> axisFactors;
  for (int64_t extent : roles.getAxisExtents()) {
    AxisFactor axis = builder.newAxis(extent);
    instance.axes.push_back(axis);
    products.push_back(builder.product({axis}));
    axisFactors.push_back({axis});
  }
  SmallVector<Value> inputs;
  for (auto [position, elementType] : llvm::enumerate(inputElementTypes)) {
    inputs.push_back(builder.input(
        elementType,
        getRoleFactors(roles, axisFactors, /*isOutput=*/false, position)));
  }
  instance.outputs = builder.term(kind.getValue(), inputs, products, roles,
                                  params, outputElementTypes);
  return instance;
}

bool PropagationKey::operator==(const PropagationKey &other) const {
  return kind == other.kind && params == other.params && roles == other.roles &&
         baseShardable == other.baseShardable &&
         inputElementTypes == other.inputElementTypes &&
         outputElementTypes == other.outputElementTypes;
}

llvm::hash_code hash_value(const PropagationKey &key) {
  return llvm::hash_combine(
      key.kind, key.params, key.roles,
      llvm::hash_combine_range(key.baseShardable.begin(),
                               key.baseShardable.end()),
      llvm::hash_combine_range(key.inputElementTypes.begin(),
                               key.inputElementTypes.end()),
      llvm::hash_combine_range(key.outputElementTypes.begin(),
                               key.outputElementTypes.end()));
}

// Prints e.g. `dft roles <([i])->([j]) {i=8, j=8}> base-shardable [] :
// (complex<f32>) -> (complex<f32>)`, listing the base-shardable axes by
// number.
void PropagationKey::print(raw_ostream &os) const {
  os << kind.getValue();
  if (params)
    os << " params " << params;
  os << " roles " << roles << " base-shardable [";
  llvm::interleaveComma(llvm::make_filter_range(
                            llvm::seq<int64_t>(0, baseShardable.size()),
                            [&](int64_t axis) { return baseShardable[axis]; }),
                        os);
  os << "] : (";
  llvm::interleaveComma(inputElementTypes, os);
  os << ") -> (";
  llvm::interleaveComma(outputElementTypes, os);
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

template <typename K> Expander<K>::Expander(MLIRContext *context) {
  assert(context->getLoadedDialect<func::FuncDialect>() &&
         context->getLoadedDialect<axis::AxisDialect>() &&
         context->getLoadedDialect<DistributedDialect>() &&
         "the expander builds func, axis and distributed IR");
  scratch = ModuleOp::create(UnknownLoc::get(context));
}

template <typename K> KeyId Expander<K>::internKey(K key) {
  auto [it, inserted] = ids.try_emplace(key, keys.size());
  if (inserted) {
    keys.push_back(std::move(key));
    states.emplace_back();
  }
  return it->second;
}

template <typename K> FailureOr<KeyId> Expander<K>::intern(GrammarTermOp term) {
  FailureOr<KeyMap<K>> keyMap = K::from(term);
  if (failed(keyMap))
    return failure();
  return internKey(std::move(keyMap->key));
}

template <typename K>
std::pair<KeyId, SmallVector<KeyAxisRef>>
Expander<K>::intern(const TypedDescription &td) {
  KeyMap<K> keyMap = K::from(td);
  return {internKey(std::move(keyMap.key)), std::move(keyMap.axisMap)};
}

template <typename K>
typename Expander<K>::ScratchInstance
Expander<K>::buildScratchInstance(KeyId id) {
  MLIRContext *context = scratch->getContext();
  Location loc = UnknownLoc::get(context);
  OpBuilder builder = OpBuilder::atBlockEnd(scratch->getBody());
  auto func = builder.create<func::FuncOp>(loc, "instance",
                                           builder.getFunctionType({}, {}));
  builder.setInsertionPointToStart(func.addEntryBlock());
  DerivationBuilder derivationBuilder(builder, loc);
  KeyInstance instance = keys[id].buildInstance(derivationBuilder);
  SmallVector<Value> outputs(instance.outputs.begin(), instance.outputs.end());
  builder.create<func::ReturnOp>(loc, outputs);
  func.setFunctionType(
      builder.getFunctionType({}, ValueRange(outputs).getTypes()));
  GrammarTermOp term = *func.getBody().getOps<GrammarTermOp>().begin();
  return {func, term, std::move(instance.axes)};
}

template <typename K> ArrayRef<Step> Expander<K>::steps(KeyId id) {
  KeyState &state = states[id];
  if (state.listed)
    return state.steps;
  state.listed = true;
  state.steps.push_back(Step{std::nullopt, nullptr});
  const GrammarRegistry &registry = getGrammarRegistry(scratch->getContext());
  ArrayRef<RuleId> rules = registry.getRules(keys[id].getKind().getValue());
  // Kinds without rules, Shardy-described ones among them, never need an
  // instance.
  if (!rules.empty()) {
    ScratchInstance instance = buildScratchInstance(id);
    for (RuleId rule : rules)
      for (Attribute choice : registry.getRule(rule).splits(instance.term))
        state.steps.push_back(Step{rule, choice});
    instance.func.erase();
  }
  state.graphs.resize(state.steps.size());
  return state.steps;
}

template <typename K> const StepGraph &Expander<K>::graph(StepRef ref) {
  const Step &step = this->step(ref);
  assert(!step.isBase() && "a base step is a terminal and has no graph");
  KeyState &state = states[ref.key];
  if (state.graphs[ref.index])
    return *state.graphs[ref.index];

  ScratchInstance instance = buildScratchInstance(ref.key);
  GrammarTermOp term = instance.term;
  OpBuilder builder(term);
  DerivationBuilder derivationBuilder(builder, term.getLoc());
  SmallVector<DerivationTensor> outputs =
      getGrammarRegistry(scratch->getContext())
          .getRule(*step.rule)
          .apply(derivationBuilder, term, step.choice);
  assert(outputs.size() == term.getOutputs().size() &&
         "a rule returns one tensor per output tensor position");

  // The rule's outputs take the place of the term's output tensors.
  SmallVector<DerivationTensorOp> replaced;
  for (auto [position, output] : llvm::enumerate(outputs)) {
    DerivationTensorOp old = term.getTensor(/*isOutput=*/true, position);
#ifndef NDEBUG
    FailureOr<axis::CollectedAtoms> atoms = axis::collectAtoms(
        {ValueRange(old.getAxes()),
         ValueRange(
             cast<DerivationTensorOp>(output.getDefiningOp()).getAxes())});
    assert(succeeded(atoms) && atoms->sets[0] == atoms->sets[1] &&
           output.getType() == old.getTensor().getType() &&
           "a rule's output holds the digits and type of the output it "
           "replaces");
#endif
    old.getTensor().replaceAllUsesWith(output);
    replaced.push_back(old);
  }
  for (DerivationTensorOp old : replaced)
    old.erase();
  term.erase();
  assert(succeeded(verify(instance.func)) &&
         "a rule builds valid derivation IR");

  StepGraph built = index(instance.func, instance.axes);
  assert(preservesDependence(ref.key, built) &&
         "a rule's outputs depend on exactly the inputs its kind declares");
  instance.func.erase();
  state.graphs[ref.index] = std::move(built);
  return *state.graphs[ref.index];
}

template <typename K>
StepGraph Expander<K>::index(func::FuncOp func,
                             ArrayRef<AxisFactor> externalAxes) {
  Block &body = func.getBody().front();
  SmallVector<DerivationTensorOp> tensors =
      llvm::to_vector(body.getOps<DerivationTensorOp>());
  SmallVector<GrammarTermOp> terms =
      llvm::to_vector(body.getOps<GrammarTermOp>());

  // One factor list per external axis, per tensor, and per factor of each
  // term's operator axes. A single factor's atom set lists its atoms
  // major-first, so an operator axis's digits are its factors' sets in order.
  SmallVector<SmallVector<Value>> lists;
  for (AxisFactor axis : externalAxes)
    lists.push_back({axis});
  for (DerivationTensorOp tensor : tensors)
    lists.emplace_back(tensor.getAxes());
  SmallVector<SmallVector<SmallVector<Value>>> termAxisFactors;
  for (GrammarTermOp term : terms) {
    termAxisFactors.push_back(term.getOperatorAxisFactors());
    for (ArrayRef<Value> factors : termAxisFactors.back())
      for (Value factor : factors)
        lists.push_back({factor});
  }
  SmallVector<ValueRange> ranges(lists.begin(), lists.end());
  FailureOr<axis::CollectedAtoms> atoms = axis::collectAtoms(ranges);
  assert(succeeded(atoms) && "a step's cuts on each root must nest");

  StepGraph graph;
  graph.digitExtents = atoms->extents;
  auto nextSet = atoms->sets.begin();
  for (size_t axis = 0; axis < externalAxes.size(); ++axis)
    graph.external.push_back(*nextSet++);
  DenseMap<Operation *, unsigned> tensorIndex;
  for (DerivationTensorOp tensor : tensors) {
    tensorIndex[tensor] = graph.tensors.size();
    graph.tensors.push_back(*nextSet++);
  }
  for (auto [term, axisFactors] : llvm::zip_equal(terms, termAxisFactors)) {
    SmallVector<SmallVector<DigitId>> axisDigits;
    for (ArrayRef<Value> factors : axisFactors) {
      SmallVector<DigitId> &digits = axisDigits.emplace_back();
      for (size_t factor = 0; factor < factors.size(); ++factor)
        llvm::append_range(digits, *nextSet++);
    }

    FailureOr<KeyMap<K>> keyMap = K::from(term);
    assert(succeeded(keyMap) && "rules build terms of registered kinds only");
    StepGraph::Occurrence occurrence;
    occurrence.axes.resize(llvm::count_if(
        keyMap->axisMap, [](const KeyAxisRef &ref) { return !ref.peeled; }));
    for (auto [axis, ref] : llvm::enumerate(keyMap->axisMap))
      if (!ref.peeled)
        occurrence.axes[ref.index] = std::move(axisDigits[axis]);
    for (bool isOutput : {false, true}) {
      int64_t numPositions =
          isOutput ? term.getOutputs().size() : term.getInputs().size();
      for (int64_t position = 0; position < numPositions; ++position)
        occurrence.tensorIndices.push_back(
            tensorIndex.lookup(term.getTensor(isOutput, position)));
    }
    occurrence.key = internKey(std::move(keyMap->key));
    graph.occurrences.push_back(std::move(occurrence));
  }

  for (DerivationTensorOp tensor : tensors)
    if (!tensor.getDefinition())
      graph.inputs.push_back(tensorIndex.lookup(tensor));
  for (Value output : body.getTerminator()->getOperands()) {
    auto found = tensorIndex.find(output.getDefiningOp());
    assert(found != tensorIndex.end() && "a step returns tensor nodes");
    graph.outputs.push_back(found->second);
  }
  return graph;
}

template <typename K>
bool Expander<K>::preservesDependence(KeyId id, const StepGraph &graph) const {
  // Per tensor, the step's inputs it depends on, through each occurrence's
  // dependence in IR order.
  size_t numInputs = graph.inputs.size();
  SmallVector<llvm::SmallBitVector> dependsOn(graph.tensors.size(),
                                              llvm::SmallBitVector(numInputs));
  for (auto [position, tensor] : llvm::enumerate(graph.inputs))
    dependsOn[tensor].set(position);
  for (const StepGraph::Occurrence &occurrence : graph.occurrences) {
    SmallVector<SmallVector<int64_t>> dependence =
        keys[occurrence.key].getDependence();
    size_t firstOutput = occurrence.tensorIndices.size() - dependence.size();
    for (auto [output, inputs] : llvm::enumerate(dependence)) {
      llvm::SmallBitVector &reached =
          dependsOn[occurrence.tensorIndices[firstOutput + output]];
      for (int64_t input : inputs)
        reached |= dependsOn[occurrence.tensorIndices[input]];
    }
  }

  for (auto [output, inputs] : llvm::enumerate(keys[id].getDependence())) {
    llvm::SmallBitVector declared(numInputs);
    for (int64_t input : inputs)
      declared.set(input);
    if (dependsOn[graph.outputs[output]] != declared)
      return false;
  }
  return true;
}

template class Expander<PropagationKey>;

} // namespace mlir::enzyme::distributed
