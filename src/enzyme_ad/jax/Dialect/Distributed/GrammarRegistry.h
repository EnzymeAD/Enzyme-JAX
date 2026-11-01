#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H

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
  // the op's name. Not owned: a recognizer returns a string that outlives the
  // description, such as a literal or a string owned by the MLIRContext.
  StringRef kind;
  OperatorAxesAttr axes;
  DictionaryAttr params;
  // One AxisSignature per operator axis of `axes`.
  BaseSignature signature;
};

// Describes ops of some pattern as a registered grammar kind, or returns
// nullopt. A recognizer leaves the description's signature empty; recognize()
// fills it in from the kind's entry.
using GrammarRecognizer =
    std::function<std::optional<GrammarDescription>(Operation *op)>;

// One grammar operator kind, covering both compute and communication
// operators.
struct GrammarKindEntry {
  std::function<BaseSignature(OperatorAxesAttr axes, DictionaryAttr params)>
      getBaseSignature;
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
  // Registering a kind twice is a programming error.
  void registerEntry(StringRef kind, GrammarKindEntry entry);

  // The entry for `kind`, or null if no such kind is registered.
  const GrammarKindEntry *lookup(StringRef kind) const;

  // Recognizers are tried in registration order; the first match wins.
  void registerRecognizer(GrammarRecognizer recognizer);
  ArrayRef<GrammarRecognizer> getRecognizers() const { return recognizers; }

private:
  llvm::StringMap<GrammarKindEntry> entries;
  SmallVector<GrammarRecognizer> recognizers;
};

// Registers the kinds and recognizers the distributed compiler itself defines.
void registerBuiltinGrammarEntries(GrammarRegistry &registry);

// The registry of the DistributedDialect, which must be loaded in `context`.
GrammarRegistry &getGrammarRegistry(MLIRContext *context);

// Describes `op` as a grammar operator, the one place an op's operator axes
// and base signature come from:
// - a GrammarTerm describes itself, with its kind's signature; an unregistered
//   kind gets a warning on the op and fails;
// - otherwise the first registered recognizer that matches;
// - otherwise the op's Shardy sharding rule, as operator axes named after the
//   op.
// Fails for an op with none of these, or with a tensor that isn't statically
// shaped. Requires the DistributedDialect to be loaded. May warn, so it must
// not be called from an op verifier, whose diagnostics re-verify the op.
//
// A Shardy rule may size a factor from one side only (slice, pad, concat), so
// some dimension's extents multiply to something other than its size. Such a
// dimension gets its own `local` axis, appended after the rule's factors so
// that the first axes still line up with the rule. Being local, the op is
// realized by resharding around it.
FailureOr<GrammarDescription> recognize(Operation *op);

// Builds a GrammarTerm for a description of a registered kind. Ops described
// only through their Shardy rule stay as they are, so materializing one is a
// programming error.
GrammarTermOp materializeGrammarTerm(OpBuilder &builder, Location loc,
                                     const GrammarDescription &description,
                                     ValueRange operands,
                                     TypeRange resultTypes);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H
