//===----------------------------------------------------------------------===//
//
// Properties of values, such as `SPD` or `symmetric`, and how Tessera knows
// them.
//
// DECLARATIONS are what a library expert writes on functions:
//
//   [[tessera::guarantees("SPD")]]            the return value is SPD
//   [[tessera::guarantees("SPD(M)")]]         M is SPD when the call returns
//   [[tessera::assumes("SPD(M)")]]            M is SPD when the call starts
//   [[tessera::preserves("SPD", "A", "B")]]   the output is SPD if A and B are
//
// lift-tessera-annotations turns them into attributes on the function:
//
//   tessera.assumes = ["SPD"]      on a parameter
//   tessera.guarantees = ["SPD"]   on the result, the sret parameter, or a
//                                  parameter the function writes
//   tessera.preserves = [{property = "SPD", inputs = [0, 1]}]
//                                  on the function. Inputs are parameter
//                                  numbers, sret included. The output is the
//                                  return value, or else the one argument the
//                                  function writes.
//
// FACTS are what tessera-propagate-properties derives from declarations: "the
// value passed here is SPD". Each is recorded as `tessera.property` on a
// call's argument, where rule conditions read it.
//
// VALUES VS. HANDLES. A value can't change, so a fact about it holds at every
// use, and survives rewrites. A handle, such as a PETSc `Mat`, is different:
// the pointer stays the same while the object behind it changes. So a
// guarantee on a pointer parameter the function doesn't lift becomes
//
//   tessera.establishes = ["SPD"]  the object is SPD once the call returns
//
// and a function that leaves the object alone is marked
//
//   tessera.readonly
//
// For a call taking a handle, the pass walks back through earlier calls. The
// fact holds if the walk reaches a call that establishes it, passing only
// calls that don't take the handle or take it readonly. This trusts that only
// calls given the handle can change its object, as is true of PETSc.
//
// An assumption on a pointer parameter is likewise about the object on entry,
// not about the pointer: the walk can also end at the function's start, where
// the parameter's object has what is assumed of it. Other pointers, such as
// another parameter or a global, may reach that object too, so for it any call
// or store that might write memory other than the function's own stack
// variables counts as a change.
//
// FACTS ON STATEMENTS. A user can also state a fact after a statement:
//
//   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
//   __attribute__((tessera_guarantees("SPD(A)")));
//
// The plugin turns this into a call, `__tessera_fact_0(&A)`, to an empty
// function marked `tessera.fact_marker` whose parameter establishes the fact
// (or assumes it, for tessera_assumes). It's given `&A` rather than the loaded
// handle because after a branch the loaded value may no longer be
// recognizable as A. tessera-to-llvm deletes the markers once facts are used.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_TESSERA_PROPERTY_PROPAGATION_H
#define ENZYME_AD_JAX_PASSES_TESSERA_PROPERTY_PROPAGATION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {

constexpr llvm::StringLiteral kAssumesAttr = "tessera.assumes";
constexpr llvm::StringLiteral kGuaranteesAttr = "tessera.guarantees";
constexpr llvm::StringLiteral kPropertyAttr = "tessera.property";
constexpr llvm::StringLiteral kPreservesAttr = "tessera.preserves";
constexpr llvm::StringLiteral kEstablishesAttr = "tessera.establishes";
constexpr llvm::StringLiteral kReadonlyAttr = "tessera.readonly";
constexpr llvm::StringLiteral kFactMarkerAttr = "tessera.fact_marker";

// Derives facts from declarations. Results are cached, so don't keep using an
// instance after changing the IR: a new value can reuse an erased one's slot.
class PropertyAnalysis {
public:
  // Records on each argument of `call` the facts known about it, and about
  // the object if it's a handle. Facts already recorded are kept.
  void recordFactsOnCall(CallOp call);

private:
  // The facts known about `value`. `depth` counts the calls followed so far.
  // Implied properties (SPD implies symmetric) are not added.
  llvm::SmallVector<StringAttr> cachedValueFacts(Value value, unsigned depth);
  llvm::SmallVector<StringAttr> computeValueFacts(Value value, unsigned depth);

  llvm::DenseMap<Value, llvm::SmallVector<StringAttr>> valueFactsCache;
};

// Whether a fact recorded where `value` is passed to a call says it has
// `property`, directly or by implication. Rule conditions use this.
bool hasRecordedProperty(Value value, llvm::StringRef property);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PROPERTY_PROPAGATION_H
