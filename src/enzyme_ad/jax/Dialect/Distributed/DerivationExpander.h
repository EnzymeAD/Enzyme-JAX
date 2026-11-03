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

struct KeyMap;

// The key used by propagation, identifying an operator up to everything its
// derivations depend on, so that operators with equal keys share them. It is
// the operator's coarse spec with its batch peeled: the kind, params and
// roles left after the kind's peelBatch (the roles numbered canonically, see
// canonicalizeAxes) and each tensor position's element type, together with
// which of its axes the base case shards by itself.
//
// A term's batch, its tensors' digits beyond its roles, never reaches the key,
// so one key serves every batch size and every broadcast pattern. Reduce kinds
// are dropped too, since building IR doesn't need them, so a sum and a max
// over the same axes share a key: a registered kind's spec takes its
// signature from the kind, and an unregistered kind's spec keeps none.
class PropagationKey {
public:
  // Fails, warning on the term, when its kind isn't registered.
  static FailureOr<KeyMap> from(GrammarTermOp term);
  // `td` keeps the digit factoring it comes with; propagation refines it by
  // the incoming strings first (refineAxes). A kind without a registry entry,
  // such as a Shardy-described op's, keeps td's signature and peels nothing;
  // a registered kind's signature must agree with td's.
  static KeyMap from(const TypedDescription &td);
  // A spec of a registered kind, such as a step's occurrence.
  static KeyMap from(const OperatorSpec &spec);

  // The spec the key's steps are built on, with one instance root per key
  // axis. Only keys of registered kinds have one.
  const OperatorSpec &getSpec() const;

  StringAttr getKind() const { return spec.getKind(); }
  DictionaryAttr getParams() const { return spec.description.params; }
  OperatorAxesAttr getRoles() const { return spec.getRoles(); }
  bool isBaseShardable(int64_t axis) const { return baseShardable[axis]; }
  ArrayRef<Type> getInputElementTypes() const { return spec.inputElementTypes; }
  ArrayRef<Type> getOutputElementTypes() const {
    return spec.outputElementTypes;
  }

  bool operator==(const PropagationKey &other) const;
  friend llvm::hash_code hash_value(const PropagationKey &key);
  void print(raw_ostream &os) const;

private:
  // Both `from`s: peel, take the kind's base signature on what remains (or
  // the description's signature for a kind without a registry `entry`), and
  // number the axes canonically.
  static KeyMap fromRoles(StringAttr kind, const GrammarKindEntry *entry,
                          OperatorAxesAttr roles, DictionaryAttr params,
                          const BaseSignature *descriptionSignature,
                          SmallVector<Type> inputElementTypes,
                          SmallVector<Type> outputElementTypes);

  OperatorSpec spec;
  SmallVector<bool> baseShardable;
};

// A key, and for each operator axis of what it was made from, where that axis
// went.
struct KeyMap {
  PropagationKey key;
  SmallVector<KeyAxisRef> axisMap;
};

// Interned keys and their steps are referred to by these handles, which are
// only meaningful within the BottomUpExpander that made them.
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

// The index of one rule step (StepIndex) with each occurrence's spec
// interned as its key: what a traversal needs from the IR the rule built, so
// that the IR itself can be dropped.
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

// Expands propagation keys into their steps, bottom-up: a key's steps and
// graphs don't depend on where the key occurs, so they are memoized per key.
// Keys are interned, and each key's steps are listed when first asked for:
// the base step, then every choice of every rule of the key's kind, in
// registration order and each rule's order (choices). A rule step's graph is
// built when first asked for, by indexing the rule's IR on an instance of the
// key (indexChoice) and interning its occurrences. The expander never expands
// the sub-keys it interns along the way.
class BottomUpExpander {
public:
  // The func, axis and distributed dialects must be loaded in `context`; a
  // pass using the expander declares them as dependent dialects.
  explicit BottomUpExpander(MLIRContext *context);

  // Fails, with the key's warning, for a term of an unregistered kind.
  FailureOr<KeyId> intern(GrammarTermOp term);
  // The key of a program op at the boundary, and where each of its operator
  // axes went (see KeyMap).
  std::pair<KeyId, SmallVector<KeyAxisRef>> intern(const TypedDescription &td);

  const PropagationKey &key(KeyId id) const { return keys[id]; }
  ArrayRef<Step> steps(KeyId id);
  const Step &step(StepRef ref) { return steps(ref.key)[ref.index]; }
  // The graph of a rule step. A base step has none: it is a terminal.
  const StepGraph &graph(StepRef ref);
  // The graphs of several rule steps of `id`, by step index; those not built
  // yet are built on one instance of the key.
  SmallVector<const StepGraph *> graphs(KeyId id,
                                        ArrayRef<unsigned> stepIndexes);
  // The IR of a rule step, built the way graph() builds it, on an instance
  // of the key (one root per key axis). Rules are deterministic, which debug
  // builds check: the rebuilt IR indexes to graph(ref).
  RebuiltStep rebuild(StepRef ref);

  // indexStep(func, externalAxes) with every term interned; `externalAxes`
  // are the boundary roots, one per key axis.
  StepGraph index(func::FuncOp func, ArrayRef<AxisFactor> externalAxes);

private:
  struct KeyHash {
    size_t operator()(const PropagationKey &key) const {
      return hash_value(key);
    }
  };
  struct KeyState {
    bool listed = false;
    SmallVector<Step> steps;
    // Per step; built on request.
    std::vector<std::optional<StepGraph>> graphs;
  };

  KeyId internKey(PropagationKey key);
  // `index` with each occurrence's spec interned, its axes in key-axis order.
  StepGraph internOccurrences(StepIndex index);

  MLIRContext *context;
  // Deques, so that references handed out stay valid while expansion interns
  // more keys.
  std::deque<PropagationKey> keys;
  std::deque<KeyState> states;
  std::unordered_map<PropagationKey, KeyId, KeyHash> ids;
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
// for (checkSplice), which materialize asserts. Under coarse keys and mixed
// radices a derivation can need a relayout to bind instead (its digits there
// are Local); that isn't supported. Debug builds also check that a func
// whose cuts on each root nested before still does afterwards (rootCutsNest),
// since splicing checks each step on its own.
//
// A derivation of a key alone is materialized into the term of the key's
// instance (buildInstance(key.getSpec())).
void materialize(BottomUpExpander &expander, const Derivation &derivation,
                 GrammarTermOp term);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_EXPANDER_H
