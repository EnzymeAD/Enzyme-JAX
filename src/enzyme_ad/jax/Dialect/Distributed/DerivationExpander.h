#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H

#include "GrammarRegistry.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::enzyme::distributed {

// Derivation keys.
//
// A key identifies an operator up to everything a derivation can depend on, so
// that operators with equal keys share their derivations. What a key keeps
// from a description depends on its client, so the key is a type parameter K
// of the engine. A key type provides:
// - `static std::pair<K, SmallVector<int64_t>> from(const TypedDescription &)`:
//   the key, and for each description axis the key axis it became;
// - `TypedDescription typedDescription() const`: the canonical description the
//   key stands for, which scratch IR and replay are built from;
// - `operator==`, `hash_value(const K &)` and `print(raw_ostream &)`.

// The key used by propagation. It keeps the canonical operator axes (at the
// digit factoring of the description it was made from), which axes the base
// case shards by itself, and the operand and result types.
//
// Reduce kinds are dropped, since building IR never needs them, so a sum and a
// max over the same axes share a key. Element types are kept, since scratch IR
// is built from the key.
class PropagationKey {
public:
  static std::pair<PropagationKey, SmallVector<int64_t>>
  from(const TypedDescription &td);

  // Only keys of registered kinds have one: the signature comes from the
  // kind's registry entry. Rules exist only on registered kinds, so nothing
  // builds IR for any other key.
  TypedDescription typedDescription() const;

  StringAttr getKind() const { return kind; }
  OperatorAxesAttr getAxes() const { return axes; }
  bool isBaseShardable(int64_t axis) const { return baseShardable[axis]; }

  bool operator==(const PropagationKey &other) const;
  friend llvm::hash_code hash_value(const PropagationKey &key);
  void print(raw_ostream &os) const;

private:
  StringAttr kind;
  DictionaryAttr params;
  OperatorAxesAttr axes;
  SmallVector<bool> baseShardable;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
