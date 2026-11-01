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

// An op's base case: one AxisSignature per operator axis of a GrammarTerm, or
// per factor of another op's Shardy sharding rule.
struct BaseSignature {
  SmallVector<AxisSignature> axes;

  bool isShardable(int64_t axis) const { return axes[axis].isShardable(); }
};

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

private:
  llvm::StringMap<GrammarKindEntry> entries;
};

// Registers the kinds the distributed compiler itself defines.
void registerBuiltinGrammarEntries(GrammarRegistry &registry);

// The base signature of any op: a GrammarTerm's from its kind's entry, any
// other op's from its Shardy sharding rule. Fails, with a warning on the op,
// for a GrammarTerm of an unregistered kind, and fails silently for an op with
// no sharding rule.
FailureOr<BaseSignature> getBaseSignature(Operation *op);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_GRAMMAR_REGISTRY_H
