#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_STEPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_STEPS_H

#include "DerivationIR.h"
#include "GrammarRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include <vector>

namespace mlir::enzyme::distributed {

// One operator's derivation steps as derivation IR: its instance, its rules'
// choices, each choice's IR and that IR's index, and splicing a step in place
// of a term. Traversals key and memoize operators differently but build steps
// the same way, so they share this layer. Each traversal chooses the factoring
// of the specs it builds on: the expander (propagation) builds instances at
// coarse specs, one root per operator-axis product, while a spec refined to its
// term's factors (specOf) has one root per factor.
//
// Step IR is owned by whoever built it and lives only as long as they need
// it. What persists is a step's rule and choice and its StepIndex; anything
// that needs the IR again builds it again, which rules being deterministic
// makes the same IR.

// An operator to build steps of: its description, and the element type of
// each input and output tensor position. Its factoring is its roles: an
// instance has one root per operator axis, so rules can cut nothing finer
// than the roles give them.
struct OperatorSpec {
  GrammarDescription description;
  SmallVector<Type> inputElementTypes;
  SmallVector<Type> outputElementTypes;

  StringAttr getKind() const { return description.kind; }
  OperatorAxesAttr getRoles() const { return description.axes; }
  // Per output tensor position, the input tensor positions it depends on
  // (getDependence).
  SmallVector<SmallVector<int64_t>> getDependence() const;

  bool operator==(const OperatorSpec &other) const;
};

// The spec of an operator of a registered kind, with the base signature the
// kind gives `roles`.
OperatorSpec getOperatorSpec(StringAttr kind, OperatorAxesAttr roles,
                             DictionaryAttr params,
                             SmallVector<Type> inputElementTypes,
                             SmallVector<Type> outputElementTypes);

// The spec of `term`, whose kind must be registered, with its operator axes
// numbered as the term numbers them and nothing peeled:
// - coarse, with the term's roles, one operator axis per product;
// - refined, with each product split at its factors' borders (refineAxes),
//   so that each operator axis stands for one factor of the term.
OperatorSpec specOf(GrammarTermOp term, bool refined);

// A spec's operator as derivation IR: a func.func, outside any module, over
// one fresh root per operator axis, with a boundary input tensor holding its
// role axes per input tensor position, one term, and the term's outputs
// returned. Splicing into its term (splice, materialize) uses the instance up:
// it no longer serves choices.
struct Instance {
  OwningOpRef<func::FuncOp> func;
  GrammarTermOp term;
  // The whole factor of each operator axis's root, in operator-axis order.
  SmallVector<AxisFactor> roots;
};
// Only specs of registered kinds have an instance.
Instance buildInstance(const OperatorSpec &spec);

// Every choice of every rule of the instance's kind: rules in registration
// order, each rule's choices in its own order.
SmallVector<std::pair<RuleId, Attribute>> choices(const Instance &instance);

// A digit of a step's IR: one atom of the factors in it (see collectAtoms).
using DigitId = size_t;

// The index of one step's IR: what a traversal needs from it, so that the IR
// itself can be dropped.
//
// Digits identify: two tensors or occurrences mentioning the same digit
// share it, which is how a rule's intermediate tensors connect its
// occurrences. Nothing else links them; a relayout's sides are distinct
// tensors with distinct digits.
struct StepIndex {
  // One term of the step.
  struct Occurrence {
    // Coarse, numbered as the term.
    OperatorSpec spec;
    // Per operator axis of `spec`, its digits major-first.
    SmallVector<SmallVector<DigitId>> axes;
    // The index in StepIndex::tensors of the tensor at each tensor position,
    // inputs then outputs.
    SmallVector<unsigned> tensorIndices;

    bool operator==(const Occurrence &other) const {
      return spec == other.spec && axes == other.axes &&
             tensorIndices == other.tensorIndices;
    }
  };

  SmallVector<uint64_t> digitExtents;
  // Per root of the instance, its digits major-first.
  SmallVector<SmallVector<DigitId>> external;
  // In IR order.
  std::vector<Occurrence> occurrences;
  // Every tensor node's digits, in IR order.
  SmallVector<axis::AtomSet> tensors;
  // Per input tensor position of the instance's spec, the index in `tensors`
  // of its boundary input tensor; per output tensor position, of the tensor
  // the step returns there.
  SmallVector<unsigned> inputs;
  SmallVector<unsigned> outputs;

  bool operator==(const StepIndex &other) const {
    return digitExtents == other.digitExtents && external == other.external &&
           occurrences == other.occurrences && tensors == other.tensors &&
           inputs == other.inputs && outputs == other.outputs;
  }
};

// Indexes `func`, a derivation IR container whose boundary roots are `roots`
// (one whole factor each). Each term must be of a registered kind, and the
// cuts on each root must nest across the whole func, which the op verifiers
// check only per op. Its boundary input tensors, in IR order, are the inputs,
// and it returns the outputs.
StepIndex indexStep(func::FuncOp func, ArrayRef<AxisFactor> roots);

// A step's IR: a rule applied to a copy of an instance, its outputs in place
// of the instance term's.
struct RebuiltStep {
  OwningOpRef<func::FuncOp> func;
  // The whole factor of each of the instance's roots, in its order.
  SmallVector<AxisFactor> roots;
  // The terms the rule built, in IR order.
  SmallVector<GrammarTermOp> occurrences;
};

// The IR of `choice` of `rule` on a copy of `instance`, which stays as it is,
// so one instance serves every choice. Debug builds verify the IR.
RebuiltStep buildChoice(const Instance &instance, RuleId rule,
                        Attribute choice);

// The index of buildChoice's IR, which is dropped. Debug builds also check
// that the rule preserves its kind's dependence (preservesDependence).
StepIndex indexChoice(const Instance &instance, RuleId rule, Attribute choice);

// Whether each output of `index`, a step of `spec`, depends on exactly the
// inputs the spec's kind declares, transitively through the step's
// occurrences. A rule that adds or drops a dependence would give its
// rewritten outputs the wrong batch.
bool preservesDependence(const OperatorSpec &spec, const StepIndex &index);

// The IR of `choice` of `rule` on a fresh instance of `spec`. Rules are
// deterministic, which debug builds check: the IR indexes to `expected`.
RebuiltStep rebuild(const OperatorSpec &spec, RuleId rule, Attribute choice,
                    const StepIndex &expected);

// The factors of `step`'s terms whose cuts don't nest with what their root
// is bound to. `bound` has one entry per root of the step: `bound[r]` is the
// term factors, major-first, that root `r` stands for. The roots the rule
// created are fresh, and their factors always bind. Computed from
// extents alone, so no IR is created.
//
// A cut that doesn't nest, such as a root bound to factors (3, 2) cut as
// 2 · 3, has no factor form: binding it would take a relayout.
SmallVector<AxisFactor> checkSplice(const RebuiltStep &step,
                                    ArrayRef<SmallVector<AxisFactor>> bound);

// Builds `step` in place of `term`, with root `r` of the step standing for
// the term factors `bound[r]` (one entry per root, as for checkSplice), and
// returns the spliced terms in IR order.
// checkSplice must find nothing.
// - A factor of a bound root becomes the bound factors its digits touch, cut
//   where it cuts them, so every axis stays a factor of a root of the term's
//   IR. A fresh root of the step becomes a new root.
// - The step's boundary inputs are the term's inputs, and its outputs replace
//   the term's outputs. The term's axis ops that nothing else uses are erased
//   with it.
// - Spliced terms are built through DerivationBuilder, so their batch flows
//   from the term's inputs by dependence. Since rules preserve dependence,
//   each spliced output holds exactly the digits of the output it replaces.
SmallVector<GrammarTermOp> splice(RebuiltStep &step, GrammarTermOp term,
                                  ArrayRef<SmallVector<AxisFactor>> bound);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DERIVATION_STEPS_H
