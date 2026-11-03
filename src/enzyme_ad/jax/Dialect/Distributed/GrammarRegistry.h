#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H

#include "DerivationIR.h"
#include "Dialect.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "llvm/ADT/StringMap.h"

#include <functional>

namespace mlir::enzyme::distributed {

// How an op may be sharded along one of its operator axes in its base case,
// before any rewrite rule applies.
//
// Propagation and derivation only ask whether an axis is shardable. The
// shardable variants differ in what lowering has to do:
// - Parallel (`⊗ I`): each device computes its own block, with no
//   communication.
// - Reduce: each device computes a partial result, and a reduction collective
//   combines them.
// - Custom: shardable only through derivation, i.e. some rewrite rule.
// - CustomTerminal: the kind's lowering entry handles communication over this
//   axis, or over any combination of the op's CustomTerminal axes. Rules may
//   still rewrite the op further.
// Parallel and Reduce let lowering skip derivation altogether.
class AxisSignature {
public:
  enum class Kind { Local, Parallel, Reduce, Custom, CustomTerminal };

  static AxisSignature local() { return AxisSignature(Kind::Local); }
  static AxisSignature parallel() { return AxisSignature(Kind::Parallel); }
  static AxisSignature reduce(stablehlo::ReduceOpKind reduceKind) {
    return AxisSignature(Kind::Reduce, reduceKind);
  }
  static AxisSignature custom() { return AxisSignature(Kind::Custom); }
  static AxisSignature customTerminal() {
    return AxisSignature(Kind::CustomTerminal);
  }

  Kind getKind() const { return kind; }
  bool isLocal() const { return kind == Kind::Local; }
  bool isShardable() const { return !isLocal(); }
  // Shardable by the base case itself, without any rewrite rule.
  bool isBaseShardable() const { return isShardable() && kind != Kind::Custom; }

  // How partial results combine. Unknown for a reduction whose body doesn't
  // classify; its body then comes from the op itself at lowering time.
  stablehlo::ReduceOpKind getReduceKind() const {
    assert(kind == Kind::Reduce && "only a Reduce axis has a reduce kind");
    return reduceKind;
  }

  bool operator==(const AxisSignature &other) const {
    return kind == other.kind && reduceKind == other.reduceKind;
  }

private:
  explicit AxisSignature(Kind kind, stablehlo::ReduceOpKind reduceKind =
                                        stablehlo::ReduceOpKind::Unknown)
      : kind(kind), reduceKind(reduceKind) {}

  Kind kind;
  stablehlo::ReduceOpKind reduceKind;
};

// An op's base case: one AxisSignature per operator axis.
struct BaseSignature {
  SmallVector<AxisSignature> axes;

  bool isShardable(int64_t axis) const { return axes[axis].isShardable(); }
};

// Prints `parallel`, `local`, `reduce(add)`, `custom` or `custom-terminal`.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const AxisSignature &signature);

// An op described as a grammar operator, without rewriting it.
struct GrammarDescription {
  // A registered grammar kind, or for an op described through its Shardy rule,
  // `sdy.<op name>`. The `sdy.` namespace is reserved for those, so the two
  // never collide.
  StringAttr kind;
  OperatorAxesAttr axes;
  DictionaryAttr params;
  // One AxisSignature per operator axis of `axes`.
  BaseSignature signature;
};

// A description together with the operand and result types of the op it
// describes: what a key is made from at the boundary between program IR and
// derivations.
struct TypedDescription {
  GrammarDescription description;
  SmallVector<Type> operandTypes;
  SmallVector<Type> resultTypes;
};

// The utilities below produce a new description with different operator axes.
// They do not modify the described op or any IR. Anything outside the
// description that refers to its axes (per-axis analysis state, a result mapped
// back onto the op) must go through the returned axis map, and `params` must
// never name an axis by index.

// A description with its axes renumbered by first appearance: operands, then
// results, dimensions in order, digits major-first. Tensors, dimensions, digit
// order, extents, kind and params are unchanged; the signature is permuted
// with the axes. Descriptions that differ only in axis numbering canonicalize
// alike.
struct CanonicalizedAxes {
  GrammarDescription description;
  // perm[axis] is the canonical number of `axis`.
  SmallVector<int64_t> perm;
};
CanonicalizedAxes canonicalizeAxes(const GrammarDescription &description);

// A description with its axes split at given digit boundaries, which is how a
// factoring imposed from outside (e.g. propagation's incoming strings) reaches
// the description.
struct RefinedAxes {
  GrammarDescription description;
  // split[axis] lists the axes `axis` became, major-first. Pieces of one axis
  // are numbered consecutively, in the order of the axes they came from.
  SmallVector<SmallVector<int64_t>> split;
};
// `operandDigits` and `resultDigits` give, per tensor and dimension, the digit
// extents to split at, major-first; they must multiply to the dimension's size.
// An axis is split everywhere it occurs, at the union of the boundaries that
// fall inside it on any of its dimensions, and each piece keeps its signature.
// Fails if those boundaries don't nest within the axis: each must divide the
// next larger one and the axis's extent.
FailureOr<RefinedAxes>
refineAxes(const GrammarDescription &description,
           ArrayRef<SmallVector<SmallVector<int64_t>>> operandDigits,
           ArrayRef<SmallVector<SmallVector<int64_t>>> resultDigits);

// Describes ops of some pattern as a registered grammar kind, or returns
// nullopt. A recognizer leaves the description's signature empty; recognize()
// fills it in from the kind's entry.
using GrammarRecognizer =
    std::function<std::optional<GrammarDescription>(Operation *op)>;

// A rewrite rule on one grammar kind: an ordinary C++ pattern on a term of
// that kind in derivation IR, which builds the rewritten operator through a
// DerivationBuilder.
//
// A rule only enumerates its choices (e.g. Cooley-Tukey's factor pairs); it
// never picks one. The derivation engine makes one step per choice and a
// traversal decides which steps to take. No step's IR is kept: whatever needs
// it applies the rule again, so `apply` must be deterministic: the same term,
// choice and builder state give the same ops in the same order.
//
// Rules see terms of keys, whose batch is peeled. Every term a rule builds
// must be of a registered kind, and the factors a rule uses must cut each
// root at nesting boundaries; changing a tensor's factoring takes a relayout
// term onto fresh axes.
struct DerivationRule {
  std::string name;
  // Every admissible choice for `term`, most preferred first. Choices come
  // from the rule's own conditions and the boundaries between the term's
  // operator axes, never from the mesh. Rules only see a key's instance, so
  // these are the key's axis boundaries.
  std::function<SmallVector<Attribute>(GrammarTermOp term)> splits;
  // Builds `term` for one of its choices from its input tensors, with the
  // builder's insertion point before the term. Returns one tensor per output
  // position, holding the same digits as the term's tensor there; the caller
  // replaces and erases the term, which `apply` leaves alone.
  std::function<SmallVector<DerivationTensor>(
      DerivationBuilder &builder, GrammarTermOp term, Attribute choice)>
      apply;
};

// Identifies a rule within its GrammarRegistry.
using RuleId = unsigned;

// A kind's roles with its pure batch (`⊗ I`) operator axes taken out: the
// operator that remains once those axes are treated like a tensor's batch.
struct PeeledBatch {
  // The outer operator axes that are pure batch.
  SmallVector<int64_t> batchAxes;
  // The roles without the batch axes, and without the logical dimensions left
  // with no axis; the remaining axes keep their order.
  OperatorAxesAttr innerRoles;
  DictionaryAttr innerParams;
  // innerAxes[inner axis] is the outer operator axis it came from.
  SmallVector<int64_t> innerAxes;
};

// One grammar operator kind, covering both compute and communication
// operators. Its hooks name inputs and outputs by tensor position (see
// OperatorAxesAttr), whether or not a tensor is bound there.
struct GrammarKindEntry {
  std::function<BaseSignature(OperatorAxesAttr axes, DictionaryAttr params)>
      getBaseSignature;
  // Which of a term's operator axes are pure batch for this kind. Null when
  // none ever are. A peeled axis must lie on exactly the outputs that depend
  // on an input carrying it, as a term's batch does.
  std::function<PeeledBatch(OperatorAxesAttr roles, DictionaryAttr params)>
      peelBatch;
  // Kind-specific checks, run by GrammarTermOp's verifier after its own. Null
  // when the kind has none. Reports errors on the term, never asserts: parsed
  // IR reaches it. Diagnostics name tensor positions with
  // getTensorPositionName.
  std::function<LogicalResult(GrammarTermOp term)> verify;
  // Per output tensor position, the input tensor positions it depends on. Null
  // when every output depends on every input.
  // - It must over-approximate the kind's true dependence: an output's batch
  //   is exactly the batch of the inputs it depends on, so under-declaring
  //   would mix batch slices.
  // - Each of the kind's rules must give its outputs exactly this dependence
  //   (preservesDependence), or the rewritten outputs' batch would
  //   differ from the term's.
  // - It must be total on any roles and params, since the term verifier
  //   calls it before the kind's `verify`, and must not change when peelBatch
  //   peels or the axes are renumbered, since keys ask it about their inner
  //   roles.
  std::function<SmallVector<SmallVector<int64_t>>(OperatorAxesAttr roles,
                                                  DictionaryAttr params)>
      dependence;
  // Placeholder for how the kind lowers to local HLO and collectives. Its real
  // signature, which needs the term's full tensor assignments, is designed
  // with the first lowering that uses it. Null for every kind until then.
  void (*lower)() = nullptr;
};

// The grammar kinds known in one MLIRContext, keyed by GrammarTerm kind. The
// DistributedDialect owns it and registers the built-in kinds; any other code
// may register more through the dialect.
class GrammarRegistry {
public:
  // Registering a kind twice, or a kind in the `sdy.` namespace (reserved for
  // ops described through their Shardy rule), is a programming error.
  void registerEntry(StringRef kind, GrammarKindEntry entry);

  // The entry for `kind`, or null if no such kind is registered.
  const GrammarKindEntry *lookup(StringRef kind) const;

  // Recognizers are tried in registration order; the first match wins.
  void registerRecognizer(GrammarRecognizer recognizer);
  ArrayRef<GrammarRecognizer> getRecognizers() const { return recognizers; }

  // Rules attach to registered kinds only, so registering one on any other
  // kind (including a `sdy.` kind) is a programming error.
  RuleId registerRule(StringRef kind, DerivationRule rule);

  // The rules registered on `kind`, in registration order.
  ArrayRef<RuleId> getRules(StringRef kind) const;
  const DerivationRule &getRule(RuleId rule) const { return rules[rule]; }

private:
  llvm::StringMap<GrammarKindEntry> entries;
  SmallVector<GrammarRecognizer> recognizers;
  // Indexed by RuleId.
  SmallVector<DerivationRule> rules;
  llvm::StringMap<SmallVector<RuleId>> rulesByKind;
};

// Per output tensor position of an operator of `kind` with `roles`, the input
// tensor positions it depends on: the kind's declared dependence, or every
// input for a kind without one or without a registry entry.
SmallVector<SmallVector<int64_t>> getDependence(MLIRContext *context,
                                                StringRef kind,
                                                OperatorAxesAttr roles,
                                                DictionaryAttr params);

// Names a tensor position in diagnostics, `input 0` or `output 1`, for the
// term's verifier and the kinds' `verify`.
std::string getTensorPositionName(bool isOutput, int64_t tensorPosition);

// Registers the kinds and recognizers the distributed compiler itself defines.
void registerBuiltinGrammarEntries(GrammarRegistry &registry);

// The registry of the DistributedDialect, which must be loaded in `context`.
GrammarRegistry &getGrammarRegistry(MLIRContext *context);

// Describes a program (HLO) op as a grammar operator, the one place an op's
// operator axes and base signature come from:
// - the first registered recognizer that matches;
// - otherwise the op's Shardy sharding rule, as operator axes of kind
//   `sdy.<op name>`.
// Fails for an op with neither, or with a tensor that isn't statically shaped.
// Requires the DistributedDialect to be loaded.
//
// A Shardy rule may size a factor from one side only (slice, pad, concat), so
// some dimension's extents multiply to something other than its size. Such a
// dimension gets its own `local` axis, appended after the rule's factors so
// that the first axes still line up with the rule. Being local, the op is
// realized by resharding around it.
FailureOr<GrammarDescription> recognize(Operation *op);

// recognize(op) together with the op's operand and result types.
FailureOr<TypedDescription> describe(Operation *op);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H
