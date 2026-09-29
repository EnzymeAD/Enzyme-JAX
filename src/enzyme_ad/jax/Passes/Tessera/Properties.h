//===----------------------------------------------------------------------===//
//
// Properties of values, such as `SPD` or `symmetric`, and how they are known.
//
// Two layers. Declarations are what a domain scientist or library expert
// states about functions, through the plugin:
//
//   [[tessera::guarantees("SPD")]]            the return value is always SPD
//   [[tessera::guarantees("SPD(M)")]]         M is always SPD on exit
//   [[tessera::assumes("SPD(M)")]]            M is always SPD on entry
//   [[tessera::preserves("SPD", "A", "B")]]   the output is SPD if A and B are
//
// Facts are what follows at a particular call: this value, passed here, is
// SPD. tessera-propagate-properties derives facts from declarations and
// records them on call arguments, and a rule's condition reads them there.
//
// Both use one attribute, `tessera.property`, and where it sits says which it
// is:
//
//   - on a definition's input parameter: true of it on entry to every call
//     (assumes);
//   - on a definition's result, sret parameter or write-only parameter: true
//     of it on exit from every call (guarantees);
//   - on an operation with a single result: true of that result;
//   - on a call's argument: true of the value passed there.
//
// A preserves declaration is never unconditionally true of anything, so it
// stays on the function as a rule:
//
//   tessera.preserves = [{property = "SPD", inputs = [0, 1]}]
//
// where inputs are the function's own parameter numbers, sret included. It
// applies to the function's return value, or, if it returns nothing, to the
// one argument it writes.
//
// Facts belong to SSA values rather than to memory, which is what makes them
// safe to carry: a matrix a tessera op takes by `val=in` arrives as a loaded
// value, and nothing can change a value after the fact was established. For
// the same reason a fact survives a rewrite: a correct rewrite produces the
// same value, so what was true of the old one is true of the new.
//
// Handles are the exception. A PETSc `Mat` is a pointer to an object the
// library owns, and a function like `FillCOO(Mat A, ...)` fills that object
// in place: the pointer is the same before and after, but the object is only
// SPD from then on, and only until something changes it. So a guarantee on a
// pointer parameter that is not lifted is a fact about the object at a point
// in the program, recorded on the parameter as
//
//   tessera.establishes = ["SPD"]    the object is SPD once the call returns
//
// and never read as true of the pointer value. A function can also declare
// that it leaves an object alone:
//
//   tessera.readonly                 the call does not change the object
//
// At a tessera.call taking a handle, tessera-propagate-properties walks back
// through the calls before it. The handle's properties hold if it reaches a
// call establishing them, passing only calls that take the handle read-only
// or not at all. This assumes that only a call given the handle, or the
// variable holding it, can change the object it names: what a handle API's
// annotations claim, and true of PETSc objects. See handleFactsAt in
// Properties.cpp for how far the walk follows.
//
// A fact can also be stated on a statement, of a variable, for a handle that
// no function builds:
//
//   PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
//   __attribute__((tessera_guarantees("SPD(A)")));
//
// The plugin makes that a call, right after the statement, to a generated
// function that does nothing and whose parameter establishes the fact. The
// function is marked
//
//   tessera.fact_marker
//
// and tessera-to-llvm removes it, and every call to it, once the facts have
// been used.
//
// The marker is given the variable's address, `__tessera_fact_0(&A)`, not the
// handle loaded from it: the fact is of whatever handle the variable holds
// there, and a loaded value need not be recognizably a load of the variable
// -- where A may have been replaced in a branch, LLVM passes the merged
// value. A marker given the slot's address is the one call the walk does not
// take to change what the slot holds.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_TESSERA_PROPERTIES_H
#define ENZYME_AD_JAX_PASSES_TESSERA_PROPERTIES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {

constexpr llvm::StringLiteral kPropertyAttr = "tessera.property";
constexpr llvm::StringLiteral kPreservesAttr = "tessera.preserves";
constexpr llvm::StringLiteral kEstablishesAttr = "tessera.establishes";
constexpr llvm::StringLiteral kReadonlyAttr = "tessera.readonly";
constexpr llvm::StringLiteral kFactMarkerAttr = "tessera.fact_marker";

/// Does having `have` imply having `want`? Reflexive and transitive: SPD
/// implies symmetric, so a rule asking for `symmetric(A)` is satisfied by a
/// matrix known to be SPD.
bool impliesProperty(llvm::StringRef have, llvm::StringRef want);

/// Works out what is known of values from the declarations in the module.
///
/// Answers are cached, so an instance must not outlive changes to the IR it
/// has looked at: an erased value's slot can be reused by a new one.
class PropertyFacts {
public:
  /// The properties known of `value`, as declared or derived. Implications
  /// are not expanded; use impliesProperty to ask about one.
  llvm::SmallVector<StringAttr> of(Value value);

  /// Record, on each argument of `call`, what is known of the value passed
  /// there, and of the object a handle passed there names. Facts already
  /// recorded are kept.
  void annotate(CallOp call);

private:
  llvm::SmallVector<StringAttr> derive(Value value, unsigned depth);
  llvm::SmallVector<StringAttr> get(Value value, unsigned depth);

  llvm::DenseMap<Value, llvm::SmallVector<StringAttr>> cache;
};

/// Whether a fact recorded on a call argument says `value` has `property`,
/// directly or by implication. This is what a rule's condition reads.
bool hasRecordedProperty(Value value, llvm::StringRef property);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PROPERTIES_H
