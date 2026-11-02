#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H

#include "DerivationIR.h"
#include "GrammarRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "llvm/ADT/Hashing.h"

#include <deque>
#include <optional>
#include <unordered_map>

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
// - `StringAttr getKind() const`: the grammar kind, whose rules expand it.
// - `SmallVector<SmallVector<int64_t>> getDependence() const`: per output
//   slot, the input slots it depends on (see GrammarKindEntry::dependence).
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
  SmallVector<SmallVector<int64_t>> getDependence() const {
    return distributed::getDependence(kind.getContext(), kind.getValue(), roles,
                                      params);
  }

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

// Interned keys and their steps are referred to by these handles, which are
// only meaningful within the Expander that made them.
using KeyId = unsigned;

// One way to derive a key: its base case, or one choice of one rule.
struct Step {
  // None for the base step.
  std::optional<RuleId> rule;
  // The rule's choice; null for the base step.
  Attribute choice;

  bool isBase() const { return !rule; }
};

struct StepRef {
  KeyId key;
  unsigned index;
};

// A digit of a step's IR: one atom of the factors in it (see collectAtoms).
using DigitId = size_t;

// The index of one rule step: what a traversal needs from the IR the rule
// built, so that the IR itself can be erased.
//
// Digits identify: two tensors or occurrences mentioning the same digit
// share it, which is how the rule's intermediate tensors connect its
// occurrences. Nothing else links them; a relayout's sides are distinct
// tensors with distinct digits.
struct StepGraph {
  // One term the rule built, as an occurrence of its key.
  struct Occurrence {
    KeyId key;
    // Per key axis of the occurrence's key, the digits of the operator axis
    // it came from, major-first.
    SmallVector<SmallVector<DigitId>> axes;
    // Per slot, inputs then outputs, the index of its tensor in `tensors`.
    SmallVector<unsigned> slots;
  };

  SmallVector<uint64_t> digitExtents;
  // Per key axis of the expanded key, its digits major-first.
  SmallVector<SmallVector<DigitId>> external;
  // In IR order.
  SmallVector<Occurrence> occurrences;
  // Every tensor node's digits, in IR order.
  SmallVector<axis::AtomSet> tensors;
  // Per slot of the expanded key, the index in `tensors` of its boundary
  // input tensor, and of the tensor the step returns for it.
  SmallVector<unsigned> inputs;
  SmallVector<unsigned> outputs;
};

// The batch digits an occurrence's slot carries: its tensor's digits that
// aren't digits of the occurrence's key axes. These include the axes the
// kind peels.
axis::AtomSet replicatedDigits(const StepGraph &graph,
                               const StepGraph::Occurrence &occurrence,
                               int64_t slot);

// A derivation of a key: a step, and for each of the step's occurrences a
// derivation of the occurrence's key. A base step has no children.
struct Derivation {
  StepRef step;
  std::vector<Derivation> children;
};

// Expands keys into their steps. Keys are interned, and each key's steps are
// listed when first asked for: the base step, then every choice of every rule
// of the key's kind, in registration order and each rule's order. A rule
// step's graph is built when first asked for, by applying the rule to a fresh
// instance of the key in scratch IR, indexing the result and erasing it. The
// expander never expands the sub-keys it interns along the way.
template <typename K> class Expander {
public:
  // The func, axis and distributed dialects must be loaded in `context`; a
  // pass using the expander declares them as dependent dialects.
  explicit Expander(MLIRContext *context);

  // Fails, with the key's warning, for a term of an unregistered kind.
  FailureOr<KeyId> intern(GrammarTermOp term);
  // The key of a program op at the boundary, and where each of its operator
  // axes went (see KeyMap).
  std::pair<KeyId, SmallVector<KeyAxisRef>> intern(const TypedDescription &td);

  const K &key(KeyId id) const { return keys[id]; }
  ArrayRef<Step> steps(KeyId id);
  const Step &step(StepRef ref) { return steps(ref.key)[ref.index]; }
  // The graph of a rule step. A base step has none: it is a terminal.
  const StepGraph &graph(StepRef ref);

  // Whether each output of `graph`, a step of `id`, depends on exactly the
  // inputs the key's kind declares, transitively through the step's
  // occurrences. A rule that adds or drops a dependence would give its
  // rewritten outputs the wrong batch.
  bool preservesDependence(KeyId id, const StepGraph &graph) const;

  // Indexes `func`, a derivation IR container whose boundary axes are
  // `externalAxes` (one whole factor per key axis), interning every term in
  // it. Each term must be of a registered kind, and the cuts on each root
  // must nest across the whole func, which the op verifiers check only per
  // op. Its boundary input tensors, in IR order, are the key's inputs, and
  // it returns the key's outputs.
  StepGraph index(func::FuncOp func, ArrayRef<AxisFactor> externalAxes);

private:
  struct KeyHash {
    size_t operator()(const K &key) const { return hash_value(key); }
  };
  struct KeyState {
    bool listed = false;
    SmallVector<Step> steps;
    // Per step; built on request.
    std::vector<std::optional<StepGraph>> graphs;
  };

  // A key's instance in a func.func of the scratch module, which its user
  // erases.
  struct ScratchInstance {
    func::FuncOp func;
    GrammarTermOp term;
    // The whole factor of each key axis's root.
    SmallVector<AxisFactor> axes;
  };

  KeyId internKey(K key);
  ScratchInstance buildScratchInstance(KeyId id);

  OwningOpRef<ModuleOp> scratch;
  // Deques, so that references handed out stay valid while expansion interns
  // more keys.
  std::deque<K> keys;
  std::deque<KeyState> states;
  std::unordered_map<K, KeyId, KeyHash> ids;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
