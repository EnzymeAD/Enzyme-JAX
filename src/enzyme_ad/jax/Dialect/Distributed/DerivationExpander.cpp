#include "DerivationExpander.h"

namespace mlir::enzyme::distributed {

std::pair<PropagationKey, SmallVector<int64_t>>
PropagationKey::from(const TypedDescription &td) {
  CanonicalizedAxes canonical = canonicalizeAxes(td.description);
  const GrammarDescription &description = canonical.description;
  PropagationKey key;
  key.kind = description.kind;
  key.params = description.params;
  key.axes = description.axes;
  key.baseShardable = llvm::map_to_vector(
      description.signature.axes,
      [](const AxisSignature &axis) { return axis.isBaseShardable(); });
  key.operandTypes = td.operandTypes;
  key.resultTypes = td.resultTypes;
  return {std::move(key), std::move(canonical.perm)};
}

TypedDescription PropagationKey::typedDescription() const {
  const GrammarKindEntry *entry =
      getGrammarRegistry(kind.getContext()).lookup(kind.getValue());
  assert(entry && "only keys of registered kinds have a typed description");
  BaseSignature signature = entry->getBaseSignature(axes, params);
  assert(llvm::equal(baseShardable,
                     llvm::map_range(signature.axes,
                                     [](const AxisSignature &axis) {
                                       return axis.isBaseShardable();
                                     })) &&
         "the kind's signature disagrees with the key's base-shardable axes");
  return TypedDescription{GrammarDescription{kind, axes, params, signature},
                          operandTypes, resultTypes};
}

bool PropagationKey::operator==(const PropagationKey &other) const {
  return kind == other.kind && params == other.params && axes == other.axes &&
         baseShardable == other.baseShardable &&
         operandTypes == other.operandTypes && resultTypes == other.resultTypes;
}

llvm::hash_code hash_value(const PropagationKey &key) {
  return llvm::hash_combine(
      key.kind, key.params, key.axes,
      llvm::hash_combine_range(key.baseShardable.begin(),
                               key.baseShardable.end()),
      llvm::hash_combine_range(key.operandTypes.begin(),
                               key.operandTypes.end()),
      llvm::hash_combine_range(key.resultTypes.begin(), key.resultTypes.end()));
}

// Prints e.g. `dft axes <([i])->([j]) {i=8, j=8}> base-shardable [] :
// (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>`, listing the
// base-shardable axes by number.
void PropagationKey::print(raw_ostream &os) const {
  os << kind.getValue();
  if (params)
    os << " params " << params;
  os << " axes " << axes << " base-shardable [";
  llvm::interleaveComma(llvm::make_filter_range(
                            llvm::seq<int64_t>(0, baseShardable.size()),
                            [&](int64_t axis) { return baseShardable[axis]; }),
                        os);
  os << "] : (";
  llvm::interleaveComma(operandTypes, os);
  os << ") -> (";
  llvm::interleaveComma(resultTypes, os);
  os << ")";
}

} // namespace mlir::enzyme::distributed
