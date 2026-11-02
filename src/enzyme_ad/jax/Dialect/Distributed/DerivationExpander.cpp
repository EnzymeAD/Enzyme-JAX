#include "DerivationExpander.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::enzyme::distributed {

namespace {

// Whether two derivation IR tensors hold the same digits.
[[maybe_unused]] bool holdSameDigits(Value lhs, Value rhs) {
  auto axesOf = [](Value tensor) -> ValueRange {
    return cast<DerivationTensorOp>(tensor.getDefiningOp()).getAxes();
  };
  FailureOr<axis::CollectedAtoms> atoms =
      axis::collectAtoms({axesOf(lhs), axesOf(rhs)});
  return succeeded(atoms) && atoms->sets[0] == atoms->sets[1];
}

// The element type of the tensor at each of `term`'s output tensor positions.
SmallVector<Type> getTermOutputElementTypes(GrammarTermOp term) {
  SmallVector<Type> types;
  for (int64_t position = 0;
       position < static_cast<int64_t>(term.getOutputs().size()); ++position)
    types.push_back(term.getTensor(/*isOutput=*/true, position)
                        .getTensor()
                        .getType()
                        .getElementType());
  return types;
}

// The boundary input tensors of a derivation IR container's body, in IR
// order, which is the order of its input tensor positions.
SmallVector<DerivationTensorOp> getBoundaryInputs(Block &body) {
  return llvm::to_vector(llvm::make_filter_range(
      body.getOps<DerivationTensorOp>(),
      [](DerivationTensorOp tensor) { return !tensor.getDefinition(); }));
}

// Puts `outputs`, one per output tensor position, in place of `term`'s output
// tensors, and erases the term with them. Each output holds the digits and
// type of the tensor it replaces.
void replaceTerm(GrammarTermOp term, ValueRange outputs) {
  assert(outputs.size() == term.getOutputs().size() &&
         "one tensor replaces each output tensor position");
  SmallVector<DerivationTensorOp> replaced;
  for (int64_t position = 0; position < static_cast<int64_t>(outputs.size());
       ++position)
    replaced.push_back(term.getTensor(/*isOutput=*/true, position));
  for (auto [old, output] : llvm::zip_equal(replaced, outputs)) {
    assert(holdSameDigits(old.getTensor(), output) &&
           output.getType() == old.getTensor().getType() &&
           "a tensor replacing an output holds its digits and type");
    old.getTensor().replaceAllUsesWith(output);
    old.erase();
  }
  term.erase();
}

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
  SmallVector<Type> inputElementTypes =
      llvm::map_to_vector(term.getInputs(), [](Value tensor) {
        return cast<DerivationTensorType>(tensor.getType()).getElementType();
      });
  return fromRoles(kind, entry, term.getRoles(), term.getParamsAttr(),
                   /*descriptionSignature=*/nullptr,
                   std::move(inputElementTypes),
                   getTermOutputElementTypes(term));
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
  assert(!step(ref).isBase() && "a base step is a terminal and has no graph");
  KeyState &state = states[ref.key];
  if (state.graphs[ref.index])
    return *state.graphs[ref.index];

  RebuiltStep built = buildStep(ref);
  StepGraph graph = index(built.func, built.axes);
  assert(preservesDependence(ref.key, graph) &&
         "a rule's outputs depend on exactly the inputs its kind declares");
  built.func.erase();
  state.graphs[ref.index] = std::move(graph);
  return *state.graphs[ref.index];
}

template <typename K> RebuiltStep Expander<K>::rebuild(StepRef ref) {
  // The graph is built first in every build mode, so that the debug check
  // interns no key that a release build wouldn't.
  [[maybe_unused]] const StepGraph &expected = graph(ref);
  RebuiltStep rebuilt = buildStep(ref);
  assert(index(rebuilt.func, rebuilt.axes) == expected &&
         "rules are deterministic: a rebuilt step indexes to its graph");
  return rebuilt;
}

template <typename K> RebuiltStep Expander<K>::buildStep(StepRef ref) {
  const Step &step = this->step(ref);
  assert(!step.isBase() && "a base step is its key's instance, not a rule's");
  ScratchInstance instance = buildScratchInstance(ref.key);
  GrammarTermOp term = instance.term;
  OpBuilder builder(term);
  DerivationBuilder derivationBuilder(builder, term.getLoc());
  SmallVector<DerivationTensor> outputs =
      getGrammarRegistry(scratch->getContext())
          .getRule(*step.rule)
          .apply(derivationBuilder, term, step.choice);
  replaceTerm(term, SmallVector<Value>(outputs.begin(), outputs.end()));
  assert(succeeded(verify(instance.func)) &&
         "a rule builds valid derivation IR");
  Block &body = instance.func.getBody().front();
  return {instance.func, std::move(instance.axes),
          llvm::to_vector(body.getOps<GrammarTermOp>())};
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

  for (DerivationTensorOp input : getBoundaryInputs(body))
    graph.inputs.push_back(tensorIndex.lookup(input));
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

namespace {

// The factors that `factor` stands for when its root stands for `bound`, an
// operator axis's factors major-first: one per factor of `bound` that its
// digits touch, major-first, cut to the digits it holds. A factor of `bound`
// it holds whole is that factor itself.
SmallVector<AxisFactor> bindFactor(AxisFactor factor,
                                   ArrayRef<AxisFactor> bound,
                                   OpBuilder &builder, Location loc) {
  SmallVector<uint64_t> boundExtents =
      llvm::map_to_vector(bound, [](AxisFactor boundFactor) -> uint64_t {
        return axis::getFactorExtent(boundFactor);
      });
  uint64_t extent = axis::getFactorExtent(factor);
  uint64_t stride = axis::getFactorStride(factor);
  uint64_t rootExtent = llvm::product_of(boundExtents);
  assert(rootExtent % (extent * stride) == 0 &&
         "a factor lies within its root, which has the bound extent");
  // The root as the digits major to the factor, the factor's, and those
  // minor to it; the pieces of the middle one are the bound factors' parts.
  SmallVector<AxisFactor> factors;
  for (const axis::ExtentPiece &piece : axis::alignExtents(
           boundExtents, {rootExtent / (extent * stride), extent, stride})) {
    if (llvm::none_of(piece.rhs, [](const axis::SplitExtentSlice &slice) {
          return slice.extentIdx == 1;
        }))
      continue;
    assert(piece.aligned() &&
           "a step's cuts of a key axis nest with the factors the axis stands "
           "for; binding them otherwise needs a relayout");
    const axis::SplitExtentSlice &slice = piece.lhs.front();
    AxisFactor boundFactor = bound[slice.extentIdx];
    factors.push_back(slice.subExtent == boundExtents[slice.extentIdx]
                          ? boundFactor
                          : axis::createSubfactor(boundFactor, slice.subExtent,
                                                  slice.stride, builder, loc));
  }
  return factors;
}

// Builds `step` in place of `term` and returns the spliced terms in IR order.
// The step's key axis `axisMap[a]` stands for the term's operator axis `a`.
// The spliced IR goes right before the term, and the term is erased.
SmallVector<GrammarTermOp> splice(RebuiltStep &step, GrammarTermOp term,
                                  ArrayRef<KeyAxisRef> axisMap) {
  OpBuilder builder(term);
  Location loc = term.getLoc();
  DerivationBuilder derivationBuilder(builder, loc);
  auto rootOf = [](Value factor) -> Value {
    return *axis::getFactorProvenanceAxis(cast<AxisFactor>(factor));
  };

  // What each root of the step stands for: a key axis's root for the term's
  // operator axis it keys, and a fresh root for a new root.
  DenseMap<Value, SmallVector<AxisFactor>> roots;
  SmallVector<SmallVector<Value>> operatorAxisFactors =
      term.getOperatorAxisFactors();
  for (auto [operatorAxis, ref] : llvm::enumerate(axisMap))
    if (!ref.peeled)
      roots[rootOf(step.axes[ref.index])] = llvm::map_to_vector(
          operatorAxisFactors[operatorAxis],
          [](Value factor) { return cast<AxisFactor>(factor); });
  DenseMap<Value, SmallVector<AxisFactor>> factorBindings;
  auto bind = [&](Value factor) -> SmallVector<AxisFactor> {
    auto [it, inserted] = factorBindings.try_emplace(factor);
    if (inserted) {
      Value root = rootOf(factor);
      auto [rootIt, fresh] = roots.try_emplace(root);
      if (fresh)
        rootIt->second.push_back(derivationBuilder.newAxis(axis::getAxisExtent(
            cast<TypedValue<axis::AxisTypeInterface>>(root))));
      it->second =
          bindFactor(cast<AxisFactor>(factor), rootIt->second, builder, loc);
    }
    return it->second;
  };

  IRMapping tensors;
  Block &body = step.func.getBody().front();
  for (auto [input, termInput] :
       llvm::zip_equal(getBoundaryInputs(body), term.getInputs()))
    tensors.map(input.getTensor(), termInput);

  SmallVector<GrammarTermOp> spliced;
  for (GrammarTermOp occurrence : step.occurrences) {
    SmallVector<OperatorAxis> axes;
    for (ArrayRef<Value> axisFactors : occurrence.getOperatorAxisFactors()) {
      SmallVector<AxisFactor> boundFactors;
      for (Value factor : axisFactors)
        llvm::append_range(boundFactors, bind(factor));
      axes.push_back(derivationBuilder.product(boundFactors));
    }
    SmallVector<Value> inputs =
        llvm::map_to_vector(occurrence.getInputs(),
                            [&](Value input) { return tensors.lookup(input); });
    SmallVector<DerivationTensor> outputs = derivationBuilder.term(
        occurrence.getKind(), inputs, axes, occurrence.getRoles(),
        occurrence.getParamsAttr(), getTermOutputElementTypes(occurrence));
    assert(!outputs.empty() && "a term has an output");
    for (auto [position, output] : llvm::enumerate(outputs))
      tensors.map(occurrence.getTensor(/*isOutput=*/true, position).getTensor(),
                  output);
    spliced.push_back(cast<DerivationTensorOp>(outputs.front().getDefiningOp())
                          .getDefinition()
                          .getDefiningOp<GrammarTermOp>());
  }

  // The term's axis ops that only it and its outputs use go with it.
  llvm::SmallSetVector<Operation *, 8> products, factors;
  for (Value product : term.getAxes())
    products.insert(product.getDefiningOp());
  for (ArrayRef<Value> axisFactors : operatorAxisFactors)
    for (Value factor : axisFactors)
      factors.insert(factor.getDefiningOp());
  for (Value output : term.getOutputs())
    for (Value factor :
         cast<DerivationTensorOp>(*output.user_begin()).getAxes())
      factors.insert(factor.getDefiningOp());
  replaceTerm(term, llvm::map_to_vector(
                        body.getTerminator()->getOperands(),
                        [&](Value output) { return tensors.lookup(output); }));
  for (auto *axisOps : {&products, &factors})
    for (Operation *op : *axisOps)
      if (op->use_empty())
        op->erase();
  return spliced;
}

} // namespace

template <typename K>
void materialize(Expander<K> &expander, const Derivation &derivation,
                 GrammarTermOp term) {
  assert([&] {
    FailureOr<KeyMap<K>> keyMap = K::from(term);
    return succeeded(keyMap) &&
           keyMap->key == expander.key(derivation.step.key);
  }() && "a derivation is materialized into a term of its key");
  if (expander.step(derivation.step).isBase())
    return;
  RebuiltStep step = expander.rebuild(derivation.step);
  assert(derivation.children.size() == step.occurrences.size() &&
         "a derivation has one child per occurrence of its step");
  SmallVector<GrammarTermOp> spliced =
      splice(step, term, K::from(term)->axisMap);
  step.func.erase();
  for (auto [child, splicedTerm] :
       llvm::zip_equal(derivation.children, spliced))
    materialize(expander, child, splicedTerm);
}

template void materialize<PropagationKey>(Expander<PropagationKey> &,
                                          const Derivation &, GrammarTermOp);

} // namespace mlir::enzyme::distributed
