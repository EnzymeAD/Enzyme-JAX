#include "DerivationExpander.h"

#include "llvm/ADT/BitVector.h"

namespace mlir::enzyme::distributed {

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
  CanonicalizedAxes canonical = canonicalizeAxes(GrammarDescription{
      kind, peeled.innerRoles, peeled.innerParams, std::move(signature)});

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
  for (int64_t slot = 0; slot < static_cast<int64_t>(term.getOutputs().size());
       ++slot)
    outputElementTypes.push_back(
        elementType(term.getSlotTensor(/*isOutput=*/true, slot).getTensor()));
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
  for (auto [slot, elementType] : llvm::enumerate(inputElementTypes)) {
    inputs.push_back(builder.input(
        elementType,
        getSlotRoleFactors(roles, axisFactors, /*isOutput=*/false, slot)));
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

} // namespace mlir::enzyme::distributed
