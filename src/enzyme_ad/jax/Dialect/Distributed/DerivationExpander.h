#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H

#include "DerivationIR.h"
#include "GrammarRegistry.h"
#include "llvm/ADT/Hashing.h"

namespace mlir::enzyme::distributed {

// Where one operator axis of a term or description went in its key: a key
// axis, or a peeled batch axis that the key doesn't see. Peeled axes are
// numbered by first appearance over the slots, so one peeled number stands
// for the same replicated digit on every tensor the axis is on.
struct KeyAxisRef {
  bool peeled;
  int64_t index;

  static KeyAxisRef keyAxis(int64_t axis) { return {false, axis}; }
  static KeyAxisRef peeledAxis(int64_t number) { return {true, number}; }

  bool operator==(const KeyAxisRef &other) const {
    return peeled == other.peeled && index == other.index;
  }
};

// A key, and for each operator axis of what it was made from, where that axis
// went.
template <typename K> struct KeyMap {
  K key;
  SmallVector<KeyAxisRef> axisMap;
};

// A key's operator built as derivation IR.
struct KeyInstance {
  // The whole factor of each key axis's fresh root, in key-axis order.
  SmallVector<AxisFactor> axes;
  SmallVector<DerivationTensor> outputs;
};

// Derivation keys.
//
// A key identifies an operator up to everything a derivation can depend on, so
// that operators with equal keys share their derivations. What a key keeps
// depends on its client, so the key is a type parameter K of the engine. A key
// type provides:
// - `static FailureOr<KeyMap<K>> from(GrammarTermOp)`: the key of a term in
//   derivation IR. Fails, warning on the term, when its kind isn't registered.
// - `static KeyMap<K> from(const TypedDescription &)`: the key of a program
//   op, at the boundary between program IR and derivations.
// - `KeyInstance buildInstance(DerivationBuilder &) const`: the key's operator
//   as one term over fresh axes and boundary input tensors. Only keys of
//   registered kinds have one.
// - `operator==`, `hash_value(const K &)` and `print(raw_ostream &)`.

// The key used by propagation: the operator with its batch peeled. It keeps
// the kind, params and roles left after the kind's peelBatch (the roles
// numbered canonically, see canonicalizeAxes), which of those axes the base
// case shards by itself, and each slot's element type.
//
// A term's batch, its tensors' digits beyond its roles, never reaches the key,
// so one key serves every batch size and every broadcast pattern. Reduce kinds
// are dropped too, since building IR doesn't need them, so a sum and a max
// over the same axes share a key.
class PropagationKey {
public:
  static FailureOr<KeyMap<PropagationKey>> from(GrammarTermOp term);
  // `td` keeps the digit factoring it comes with; propagation refines it by
  // the incoming strings first (refineAxes). A kind without a registry entry,
  // such as a Shardy-described op's, keeps td's signature and peels nothing;
  // a registered kind's signature must agree with td's.
  static KeyMap<PropagationKey> from(const TypedDescription &td);

  KeyInstance buildInstance(DerivationBuilder &builder) const;

  StringAttr getKind() const { return kind; }
  DictionaryAttr getParams() const { return params; }
  OperatorAxesAttr getRoles() const { return roles; }
  bool isBaseShardable(int64_t axis) const { return baseShardable[axis]; }
  ArrayRef<Type> getInputElementTypes() const { return inputElementTypes; }
  ArrayRef<Type> getOutputElementTypes() const { return outputElementTypes; }

  bool operator==(const PropagationKey &other) const;
  friend llvm::hash_code hash_value(const PropagationKey &key);
  void print(raw_ostream &os) const;

private:
  // Both `from`s: peel, take the kind's base signature on what remains (or
  // the description's signature for a kind without a registry `entry`), and
  // number the axes canonically.
  static KeyMap<PropagationKey>
  fromRoles(StringAttr kind, const GrammarKindEntry *entry,
            OperatorAxesAttr roles, DictionaryAttr params,
            const BaseSignature *descriptionSignature,
            SmallVector<Type> inputElementTypes,
            SmallVector<Type> outputElementTypes);

  StringAttr kind;
  DictionaryAttr params;
  OperatorAxesAttr roles;
  SmallVector<bool> baseShardable;
  SmallVector<Type> inputElementTypes;
  SmallVector<Type> outputElementTypes;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
