#include "DerivationSteps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::enzyme::distributed {

namespace {

const GrammarKindEntry &getRegisteredEntry(StringAttr kind) {
  const GrammarKindEntry *entry =
      getGrammarRegistry(kind.getContext()).lookup(kind.getValue());
  assert(entry && "only operators of registered kinds have steps");
  return *entry;
}

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

Value rootOf(Value factor) {
  return *axis::getFactorProvenanceAxis(cast<AxisFactor>(factor));
}

// The pieces of a root of `boundExtents`, major-first, that hold `factor`'s
// digits. Each nests when it is one part of one bound factor.
SmallVector<axis::ExtentPiece>
getBindingPieces(AxisFactor factor, ArrayRef<uint64_t> boundExtents) {
  uint64_t extent = axis::getFactorExtent(factor);
  uint64_t stride = axis::getFactorStride(factor);
  uint64_t rootExtent = llvm::product_of(boundExtents);
  assert(rootExtent % (extent * stride) == 0 &&
         "a factor lies within its root, which has the bound extent");
  // The root as the digits major to the factor, the factor's, and those
  // minor to it; the pieces of the middle one are the bound factors' parts.
  return llvm::to_vector(llvm::make_filter_range(
      axis::alignExtents(boundExtents,
                         {rootExtent / (extent * stride), extent, stride}),
      [](const axis::ExtentPiece &piece) {
        return llvm::any_of(piece.rhs, [](const axis::SplitExtentSlice &slice) {
          return slice.extentIdx == 1;
        });
      }));
}

SmallVector<uint64_t> getExtents(ArrayRef<AxisFactor> factors) {
  return llvm::map_to_vector(factors, [](AxisFactor factor) -> uint64_t {
    return axis::getFactorExtent(factor);
  });
}

// The factors that `factor` stands for when its root stands for `bound`, an
// operator axis's factors major-first: one per factor of `bound` that its
// digits touch, major-first, cut to the digits it holds. A factor of `bound`
// it holds whole is that factor itself.
SmallVector<AxisFactor> bindFactor(AxisFactor factor,
                                   ArrayRef<AxisFactor> bound,
                                   OpBuilder &builder, Location loc) {
  SmallVector<uint64_t> boundExtents = getExtents(bound);
  SmallVector<AxisFactor> factors;
  for (const axis::ExtentPiece &piece :
       getBindingPieces(factor, boundExtents)) {
    assert(piece.aligned() &&
           "a step's cuts of a root nest with the factors the root stands "
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

} // namespace

SmallVector<SmallVector<int64_t>> OperatorSpec::getDependence() const {
  return distributed::getDependence(getKind().getContext(),
                                    getKind().getValue(), getRoles(),
                                    description.params);
}

bool OperatorSpec::operator==(const OperatorSpec &other) const {
  return description.kind == other.description.kind &&
         description.axes == other.description.axes &&
         description.params == other.description.params &&
         description.signature.axes == other.description.signature.axes &&
         inputElementTypes == other.inputElementTypes &&
         outputElementTypes == other.outputElementTypes;
}

OperatorSpec getOperatorSpec(StringAttr kind, OperatorAxesAttr roles,
                             DictionaryAttr params,
                             SmallVector<Type> inputElementTypes,
                             SmallVector<Type> outputElementTypes) {
  return OperatorSpec{
      GrammarDescription{
          kind, roles, params,
          getRegisteredEntry(kind).getBaseSignature(roles, params)},
      std::move(inputElementTypes), std::move(outputElementTypes)};
}

OperatorSpec specOf(GrammarTermOp term, bool refined) {
  OperatorAxesAttr roles = term.getRoles();
  OperatorSpec spec = getOperatorSpec(
      term.getKindAttr(), roles, term.getParamsAttr(),
      llvm::map_to_vector(term.getInputs(),
                          [](Value tensor) {
                            return cast<DerivationTensorType>(tensor.getType())
                                .getElementType();
                          }),
      getTermOutputElementTypes(term));
  if (!refined)
    return spec;

  // Each dimension's digits are the factors of its operator axes in order.
  SmallVector<SmallVector<int64_t>> factorExtents = llvm::map_to_vector(
      term.getOperatorAxisFactors(), [](ArrayRef<Value> factors) {
        return llvm::map_to_vector(factors, [](Value factor) {
          return axis::getFactorExtent(cast<AxisFactor>(factor));
        });
      });
  auto digitsOf = [&](bool isOutput) {
    SmallVector<SmallVector<SmallVector<int64_t>>> tensorDigits;
    for (sdy::TensorMappingAttr mapping : roles.getMappings(isOutput)) {
      SmallVector<SmallVector<int64_t>> &dims = tensorDigits.emplace_back();
      for (sdy::DimMappingAttr dim : mapping.getDimMappings()) {
        SmallVector<int64_t> &digits = dims.emplace_back();
        for (int64_t operatorAxis : dim.getFactorIndices())
          llvm::append_range(digits, factorExtents[operatorAxis]);
      }
    }
    return tensorDigits;
  };
  FailureOr<RefinedAxes> refinedAxes = refineAxes(
      spec.description, digitsOf(/*isOutput=*/false), digitsOf(true));
  assert(succeeded(refinedAxes) && "the borders of one product's factors nest");
  spec.description = std::move(refinedAxes->description);
  return spec;
}

Instance buildInstance(const OperatorSpec &spec) {
  getRegisteredEntry(spec.getKind());
  MLIRContext *context = spec.getKind().getContext();
  Location loc = UnknownLoc::get(context);
  OpBuilder builder(context);
  Instance instance;
  instance.func = builder.create<func::FuncOp>(loc, "instance",
                                               builder.getFunctionType({}, {}));
  builder.setInsertionPointToStart(instance.func->addEntryBlock());
  DerivationBuilder derivationBuilder(builder, loc);

  OperatorAxesAttr roles = spec.getRoles();
  SmallVector<OperatorAxis> products;
  SmallVector<SmallVector<AxisFactor>> axisFactors;
  for (int64_t extent : roles.getAxisExtents()) {
    AxisFactor root = derivationBuilder.newAxis(extent);
    instance.roots.push_back(root);
    products.push_back(derivationBuilder.product({root}));
    axisFactors.push_back({root});
  }
  SmallVector<Value> inputs;
  for (auto [position, elementType] : llvm::enumerate(spec.inputElementTypes))
    inputs.push_back(derivationBuilder.input(
        elementType,
        getRoleFactors(roles, axisFactors, /*isOutput=*/false, position)));
  SmallVector<DerivationTensor> outputs =
      derivationBuilder.term(spec.getKind().getValue(), inputs, products, roles,
                             spec.description.params, spec.outputElementTypes);
  SmallVector<Value> returned(outputs.begin(), outputs.end());
  builder.create<func::ReturnOp>(loc, returned);
  instance.func->setFunctionType(
      builder.getFunctionType({}, ValueRange(returned).getTypes()));
  instance.term = cast<DerivationTensorOp>(outputs.front().getDefiningOp())
                      .getDefinition()
                      .getDefiningOp<GrammarTermOp>();
  return instance;
}

SmallVector<std::pair<RuleId, Attribute>> choices(const Instance &instance) {
  GrammarTermOp term = instance.term;
  const GrammarRegistry &registry = getGrammarRegistry(term.getContext());
  SmallVector<std::pair<RuleId, Attribute>> result;
  for (RuleId rule : registry.getRules(term.getKind()))
    for (Attribute choice : registry.getRule(rule).splits(term))
      result.push_back({rule, choice});
  return result;
}

StepIndex indexStep(func::FuncOp func, ArrayRef<AxisFactor> roots) {
  Block &body = func.getBody().front();
  SmallVector<DerivationTensorOp> tensors =
      llvm::to_vector(body.getOps<DerivationTensorOp>());
  SmallVector<GrammarTermOp> terms =
      llvm::to_vector(body.getOps<GrammarTermOp>());

  // One factor list per root, per tensor, and per factor of each term's
  // operator axes. A single factor's atom set lists its atoms major-first, so
  // an operator axis's digits are its factors' sets in order.
  SmallVector<SmallVector<Value>> lists;
  for (AxisFactor root : roots)
    lists.push_back({root});
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

  StepIndex index;
  index.digitExtents = atoms->extents;
  auto nextSet = atoms->sets.begin();
  for (size_t root = 0; root < roots.size(); ++root)
    index.external.push_back(*nextSet++);
  DenseMap<Operation *, unsigned> tensorIndex;
  for (DerivationTensorOp tensor : tensors) {
    tensorIndex[tensor] = index.tensors.size();
    index.tensors.push_back(*nextSet++);
  }
  for (auto [term, axisFactors] : llvm::zip_equal(terms, termAxisFactors)) {
    StepIndex::Occurrence occurrence;
    occurrence.spec = specOf(term, /*refined=*/false);
    for (ArrayRef<Value> factors : axisFactors) {
      SmallVector<DigitId> &digits = occurrence.axes.emplace_back();
      for (size_t factor = 0; factor < factors.size(); ++factor)
        llvm::append_range(digits, *nextSet++);
    }
    for (bool isOutput : {false, true}) {
      int64_t numPositions =
          isOutput ? term.getOutputs().size() : term.getInputs().size();
      for (int64_t position = 0; position < numPositions; ++position)
        occurrence.tensorIndices.push_back(
            tensorIndex.lookup(term.getTensor(isOutput, position)));
    }
    index.occurrences.push_back(std::move(occurrence));
  }

  for (DerivationTensorOp input : getBoundaryInputs(body))
    index.inputs.push_back(tensorIndex.lookup(input));
  for (Value output : body.getTerminator()->getOperands()) {
    auto found = tensorIndex.find(output.getDefiningOp());
    assert(found != tensorIndex.end() && "a step returns tensor nodes");
    index.outputs.push_back(found->second);
  }
  return index;
}

RebuiltStep buildChoice(const Instance &instance, RuleId rule,
                        Attribute choice) {
  IRMapping mapping;
  RebuiltStep step;
  step.func = cast<func::FuncOp>(instance.func.get()->clone(mapping));
  step.roots = llvm::map_to_vector(instance.roots, [&](AxisFactor root) {
    return cast<AxisFactor>(mapping.lookup(Value(root)));
  });
  GrammarTermOp instanceTerm = instance.term;
  auto term = cast<GrammarTermOp>(mapping.lookup(instanceTerm.getOperation()));

  OpBuilder builder(term);
  DerivationBuilder derivationBuilder(builder, term.getLoc());
  SmallVector<DerivationTensor> outputs =
      getGrammarRegistry(term.getContext())
          .getRule(rule)
          .apply(derivationBuilder, term, choice);
  replaceTerm(term, SmallVector<Value>(outputs.begin(), outputs.end()));
  assert(succeeded(verify(*step.func)) && "a rule builds valid derivation IR");
  step.occurrences =
      llvm::to_vector(step.func->getBody().front().getOps<GrammarTermOp>());
  return step;
}

StepIndex indexChoice(const Instance &instance, RuleId rule, Attribute choice) {
  RebuiltStep step = buildChoice(instance, rule, choice);
  StepIndex index = indexStep(*step.func, step.roots);
  assert(preservesDependence(specOf(instance.term, /*refined=*/false), index) &&
         "a rule's outputs depend on exactly the inputs its kind declares");
  return index;
}

bool preservesDependence(const OperatorSpec &spec, const StepIndex &index) {
  // Per tensor, the step's inputs it depends on, through each occurrence's
  // dependence in IR order.
  size_t numInputs = index.inputs.size();
  SmallVector<llvm::SmallBitVector> dependsOn(index.tensors.size(),
                                              llvm::SmallBitVector(numInputs));
  for (auto [position, tensor] : llvm::enumerate(index.inputs))
    dependsOn[tensor].set(position);
  for (const StepIndex::Occurrence &occurrence : index.occurrences) {
    SmallVector<SmallVector<int64_t>> dependence =
        occurrence.spec.getDependence();
    size_t firstOutput = occurrence.tensorIndices.size() - dependence.size();
    for (auto [output, inputs] : llvm::enumerate(dependence)) {
      llvm::SmallBitVector &reached =
          dependsOn[occurrence.tensorIndices[firstOutput + output]];
      for (int64_t input : inputs)
        reached |= dependsOn[occurrence.tensorIndices[input]];
    }
  }

  for (auto [output, inputs] : llvm::enumerate(spec.getDependence())) {
    llvm::SmallBitVector declared(numInputs);
    for (int64_t input : inputs)
      declared.set(input);
    if (dependsOn[index.outputs[output]] != declared)
      return false;
  }
  return true;
}

RebuiltStep rebuild(const OperatorSpec &spec, RuleId rule, Attribute choice,
                    [[maybe_unused]] const StepIndex &expected) {
  RebuiltStep step = buildChoice(buildInstance(spec), rule, choice);
  assert(indexStep(*step.func, step.roots) == expected &&
         "rules are deterministic: a rebuilt step indexes to its index");
  return step;
}

SmallVector<AxisFactor> checkSplice(const RebuiltStep &step,
                                    ArrayRef<SmallVector<AxisFactor>> bound) {
  DenseMap<Value, SmallVector<uint64_t>> boundExtents;
  for (auto [root, factors] : llvm::zip_equal(step.roots, bound))
    boundExtents[rootOf(root)] = getExtents(factors);
  llvm::SmallSetVector<AxisFactor, 8> unnested;
  for (GrammarTermOp occurrence : step.occurrences)
    for (ArrayRef<Value> factors : occurrence.getOperatorAxisFactors())
      for (Value factor : factors) {
        auto found = boundExtents.find(rootOf(factor));
        if (found != boundExtents.end() &&
            !llvm::all_of(
                getBindingPieces(cast<AxisFactor>(factor), found->second),
                [](const axis::ExtentPiece &piece) { return piece.aligned(); }))
          unnested.insert(cast<AxisFactor>(factor));
      }
  return unnested.takeVector();
}

SmallVector<GrammarTermOp> splice(RebuiltStep &step, GrammarTermOp term,
                                  ArrayRef<SmallVector<AxisFactor>> bound) {
  OpBuilder builder(term);
  Location loc = term.getLoc();
  DerivationBuilder derivationBuilder(builder, loc);

  // What each root of the step stands for: a bound root for its term
  // factors, and a fresh root for a new root.
  DenseMap<Value, SmallVector<AxisFactor>> roots;
  for (auto [root, factors] : llvm::zip_equal(step.roots, bound))
    roots[rootOf(root)] = factors;
  DenseMap<Value, SmallVector<AxisFactor>> factorBindings;
  auto bind = [&](Value factor) -> SmallVector<AxisFactor> {
    auto [binding, inserted] = factorBindings.try_emplace(factor);
    if (inserted) {
      Value root = rootOf(factor);
      auto [rootBinding, fresh] = roots.try_emplace(root);
      if (fresh)
        rootBinding->second.push_back(
            derivationBuilder.newAxis(axis::getAxisExtent(
                cast<TypedValue<axis::AxisTypeInterface>>(root))));
      binding->second = bindFactor(cast<AxisFactor>(factor),
                                   rootBinding->second, builder, loc);
    }
    return binding->second;
  };

  IRMapping tensors;
  Block &body = step.func->getBody().front();
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
  for (ArrayRef<Value> axisFactors : term.getOperatorAxisFactors())
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

} // namespace mlir::enzyme::distributed
