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
// and a guarantee on a returned pointer, `double *form_gram_matrix(...)`, or
// on a struct holding pointers returned by value, is likewise about the object
// it points to. A function that changes no
// property of an object it is given -- it may still write it, as
// MatSetOption sets a flag -- is marked
//
//   tessera.unchanged
//
// which also covers anything the object holds: a call that leaves a KSP
// unchanged leaves the matrices inside it unchanged too.
//
// For a call taking a handle, the pass walks back through earlier calls. The
// fact holds if the walk reaches the call that established it (or, for a
// returned pointer, the call that returned it), passing only calls that don't
// reach the object or leave it unchanged. A handle loaded from its variable
// before a store into it may name another object, so it has no facts there,
// and a call given it establishes none.
//
// HOLDERS. A call may also store a pointer to an object it is given, as
// KSPSetOperators(ksp, A, P) stores A inside ksp. A later call given ksp, but
// not A, can then change A. So the pass also counts as reaching the object a
// call given one of its holders: a pointer passed, along with the object or
// another holder, to an earlier call that may have stored it there, or what
// such a call returned.
//
// Any call, annotated or not, may store any pointer it is given, except one
// to a parameter marked
//
//   tessera.noescape    (from C's `__attribute__((noescape))`)
//   llvm.nocapture
//
// or one given to a fact marker. Tessera reads noescape about the object: the
// call holds on to no pointer to it, nor into it, once it returns. So a
// getter such as `v.data()` or KSPGetOperators(ksp, ...) is not noescape on
// what it reads from. Leaving noescape out is always safe; it only loses
// facts.
//
// Likewise, a returned pointer may point to an object any of the call's
// pointer arguments not marked noescape holds. So the returned object's facts
// end at a later call given one of those, unless it leaves that one
// unchanged.
//
// A store the function makes, or a copy such as llvm.intr.memcpy, makes a
// holder the same way: of the memory it writes a pointer to the object, or
// into it, as `w.data = v.data` makes w one of v.
//
// A holder can only be memory that can hold a pointer: a stack variable whose
// type has none, such as `PetscInt idx[4]`, holds nothing. A pointer loaded
// from a global, such as PETSC_COMM_WORLD, isn't counted either (see below).
//
// A pointer to the object, or into it, passed on as another value -- a
// select's result, a struct it's put in, a successor block's argument, an
// `scf.if`'s result or a loop's carried value -- is followed as that value,
// and a write through it counts as a write to the object.
//
// ESCAPES. A pointer to the object, or into it, can also go where the search
// can't follow it: the function stores it in a global, or turns it into an
// integer. From then on the object is treated as a parameter's (see below):
// any op that might write memory other than the function's own stack
// variables ends its facts.
//
// THE ENCAPSULATION ASSUMPTION. What the pass can't see -- where an opaque
// callee stores a pointer -- it trusts: an object changes only through calls
// given it, given one of its holders, or (for a parameter, or an object a
// pointer to escaped) reaching it some other way. In particular, a library
// doesn't store a pointer to an object in a global registry and change the
// object through it later.
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
// CONDITIONAL GUARANTEES. A guarantee about a handle parameter's object may
// hold only under a condition on the call's arguments, as for a product:
//
//   tessera_guarantees("SPD(C) if FullColumnRank(A) && A == B && "
//                      "transa == Trans")
//
// lift-tessera-annotations records it on the function as
//
//   tessera.conditional_establishes = [{arg, property, condition, names,
//                                       constants}]
//
// and the call establishes the property only where the facts at the call
// prove the condition: a property of an argument, from its value's facts and
// its object's before the call; `A == B`, the same value, or loads of one
// variable nothing stored into between; and comparisons of constants. A
// condition that isn't proven establishes nothing, and the call is then like
// any other given the object.
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
constexpr llvm::StringLiteral kUnchangedAttr = "tessera.unchanged";
constexpr llvm::StringLiteral kNoEscapeAttr = "tessera.noescape";
constexpr llvm::StringLiteral kFactMarkerAttr = "tessera.fact_marker";
constexpr llvm::StringLiteral kConditionalEstablishesAttr =
    "tessera.conditional_establishes";

// Derives facts from declarations. Results are cached, so don't keep using an
// instance after changing the IR: a new value can reuse an erased one's slot.
class PropertyAnalysis {
public:
  // Records on each argument of `call` the facts known about it, and about
  // the object if it's a handle. Facts already recorded are kept. `replaced`
  // are ops a rewrite is about to remove: their uses don't count, and they
  // neither change an object nor store it anywhere.
  void recordFactsOnCall(CallOp call,
                         llvm::ArrayRef<Operation *> replaced = {});

private:
  // The facts known about `value`. `depth` counts the calls followed so far.
  // Implied properties (SPD implies symmetric) are not added.
  llvm::SmallVector<StringAttr> cachedValueFacts(Value value, unsigned depth);
  llvm::SmallVector<StringAttr> computeValueFacts(Value value, unsigned depth);
  // What the callee that produced `result` says of it: its guarantees, and
  // what it preserves from inputs that have the property. For an input that
  // is a handle, the property is of its object at the call.
  llvm::SmallVector<StringAttr> callResultFacts(OpResult result,
                                                unsigned depth);
  // The properties `call`'s conditional guarantees give the object it is
  // given as argument `index`: those whose condition the facts at the call
  // prove.
  llvm::SmallVector<StringAttr>
  conditionalFacts(Operation *call, unsigned index, unsigned depth);

  llvm::DenseMap<Value, llvm::SmallVector<StringAttr>> valueFactsCache;
};

// Whether a fact recorded where `value` is passed to a call says it has
// `property`, directly or by implication. Rule conditions use this.
bool hasRecordedProperty(Value value, llvm::StringRef property);

// Whether a guard around `op` has checked that `value` has `property`, directly
// or by implication: `op` is in the guard's first region, which runs only when
// the guard's condition, requiring the property through `&&`, holds.
bool guardChecksProperty(Operation *op, Value value, llvm::StringRef property);

// Whether nothing in `region`, an op such as a loop, may change the object
// `handle` names, apart from the `ignored` ops, the first of which is one
// `handle` is given to or the call that returns it. Calls to pure tessera ops
// only read. False if the handle can't be tracked.
bool objectUnchangedIn(Value handle, llvm::ArrayRef<Operation *> ignored,
                       Operation *region);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PROPERTY_PROPAGATION_H
