#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H

#include "DerivationSteps.h"
#include "llvm/ADT/Hashing.h"

#include <deque>
#include <optional>
#include <unordered_map>

namespace mlir::enzyme::distributed {

// Where one operator axis of a term or description went in its key: a key
// axis, or a peeled batch axis that the key doesn't see. Peeled axes are
// numbered by first appearance over the tensor positions, so one peeled number
// stands for the same replicated digit on every tensor the axis is on.
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
// - `static KeyMap<K> from(const OperatorSpec &)`: the key of a spec of a
//   registered kind, such as a step's occurrence.
// - `OperatorSpec getSpec() const`: the key's operator, whose instance (one
//   root per key axis) its steps are built on. Only keys of registered kinds
//   have one.
// - `StringAttr getKind() const`: the grammar kind, whose rules expand it.
// - `operator==`, `hash_value(const K &)` and `print(raw_ostream &)`.

// The key used by propagation: the operator with its batch peeled. It keeps
// the kind, params and roles left after the kind's peelBatch (the roles
// numbered canonically, see canonicalizeAxes), which of those axes the base
// case shards by itself, and each tensor position's element type.
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
  static KeyMap<PropagationKey> from(const OperatorSpec &spec);

  OperatorSpec getSpec() const;

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

// The index of one rule step (StepIndex) with its occurrences interned: what
// a traversal needs from the IR the rule built, so that the IR itself can be
// dropped.
struct StepGraph {
  // One term the rule built, as an occurrence of its key.
  struct Occurrence {
    KeyId key;
    // Per key axis of the occurrence's key, the digits of the operator axis
    // it came from, major-first.
    SmallVector<SmallVector<DigitId>> axes;
    // The index in StepGraph::tensors of the tensor at each tensor position,
    // inputs then outputs.
    SmallVector<unsigned> tensorIndices;

    bool operator==(const Occurrence &other) const {
      return key == other.key && axes == other.axes &&
             tensorIndices == other.tensorIndices;
    }
  };

  SmallVector<uint64_t> digitExtents;
  // Per key axis of the expanded key, its digits major-first.
  SmallVector<SmallVector<DigitId>> external;
  // In IR order.
  SmallVector<Occurrence> occurrences;
  // Every tensor node's digits, in IR order.
  SmallVector<axis::AtomSet> tensors;
  // Per input tensor position of the expanded key, the index in `tensors` of
  // its boundary input tensor; per output tensor position, of the tensor the
  // step returns there.
  SmallVector<unsigned> inputs;
  SmallVector<unsigned> outputs;

  bool operator==(const StepGraph &other) const {
    return digitExtents == other.digitExtents && external == other.external &&
           occurrences == other.occurrences && tensors == other.tensors &&
           inputs == other.inputs && outputs == other.outputs;
  }
};

// The batch digits the occurrence's tensor `occurrence.tensorIndices[index]`
// carries (inputs, then outputs): its digits that aren't digits of the
// occurrence's key axes. These include the axes the kind peels.
axis::AtomSet replicatedDigits(const StepGraph &graph,
                               const StepGraph::Occurrence &occurrence,
                               int64_t index);

// A derivation of a key: a step, and for each of the step's occurrences a
// derivation of the occurrence's key. A base step has no children.
struct Derivation {
  StepRef step;
  std::vector<Derivation> children;
};

// Expands keys into their steps. Keys are interned, and each key's steps are
// listed when first asked for: the base step, then every choice of every rule
// of the key's kind, in registration order and each rule's order (choices). A
// rule step's graph is built when first asked for, by indexing the rule's IR
// on an instance of the key (indexChoice) and interning its occurrences. The
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
  // The IR of a rule step, built the way graph() builds it, on an instance
  // of the key (one root per key axis). Rules are deterministic, which debug
  // builds check: the rebuilt IR indexes to graph(ref).
  RebuiltStep rebuild(StepRef ref);

  // indexStep(func, externalAxes) with every term interned; `externalAxes`
  // are the boundary roots, one per key axis.
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

  KeyId internKey(K key);
  // `index` with each occurrence's spec interned, its axes in key-axis order.
  StepGraph internOccurrences(StepIndex index);

  MLIRContext *context;
  // Deques, so that references handed out stay valid while expansion interns
  // more keys.
  std::deque<K> keys;
  std::deque<KeyState> states;
  std::unordered_map<K, KeyId, KeyHash> ids;
};

// Replaces `term` by `derivation`'s base-step terms, in place. The term's key
// must be the key of the derivation's step. A base derivation leaves the term
// as it is. Otherwise the step is rebuilt and spliced in place of the term
// (splice), each key axis's root standing for the factors of the term's
// operator axis it keys, and each spliced term is then materialized by its
// child derivation. The term's batch and the axes its kind peels reach the
// spliced terms by dependence.
//
// Each step's cuts of a key axis must nest with the factors the axis stands
// for. Under coarse keys and mixed radices a derivation can need a relayout to
// bind instead (its digits there are Local); that isn't supported.
//
// A derivation of a key alone is materialized into the term of the key's
// instance (buildInstance(key.getSpec())).
template <typename K>
void materialize(Expander<K> &expander, const Derivation &derivation,
                 GrammarTermOp term);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
