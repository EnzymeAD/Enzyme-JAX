//===----------------------------------------------------------------------===//
//
// The predicates a tessera optimization rule can test, and the machinery for
// synthesizing the code that tests them.
//
// Nothing here asks the user for a checking function. Given a matrix operand
// and its layout, the compiler emits the comparisons itself; the whole point
// is that a domain scientist writes the rule and nothing else.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H
#define ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {

//===----------------------------------------------------------------------===//
// Layout
//===----------------------------------------------------------------------===//

/// How a matrix operand is laid out, which is what makes `elem(i, j)`
/// expressible. Only fixed-size dense matrices are described; strided views and
/// dynamic extents are not.
struct MatrixLayout {
  Type elemType;
  int64_t rows = 0;
  int64_t cols = 0;
  bool rowMajor = true;

  /// True when `rowMajor` was assumed rather than declared. Reading
  /// column-major storage as row-major is exactly a transpose, so this only
  /// matters to a predicate that is not transpose-invariant; those refuse to
  /// emit a check unless the order was declared.
  bool orderInferred = false;

  bool isValid() const { return elemType && rows > 0 && cols > 0; }
  bool isSquare() const { return rows == cols; }
  int64_t numElements() const { return rows * cols; }

  /// Position of element (r, c) in the flat storage.
  int64_t linearIndex(int64_t r, int64_t c) const {
    return rowMajor ? r * cols + c : c * rows + r;
  }
};

/// Work out the layout of a value the guard carries.
///
/// The value itself says nothing: a by-reference matrix operand is an
/// `!llvm.ptr`. The declaration of the callee does say, so the lookup goes
/// through the calls the guard kept in its else region -- the matched root and,
/// for a chain rule, the producers sunk in with it -- to whichever takes the
/// value:
///
///   1. a `tessera.layout` dictionary on the callee's argument, or
///   2. the pointee type the callee takes that operand through (its lifted
///      `argModes` type, or a byval type), walked down to its element array.
///      That gives the element type and count exactly, but neither the
///      rows/cols split nor the storage order, so both are assumed (square,
///      row-major) and `orderInferred` is set. It emits a remark either way.
///
/// Returns an invalid layout when neither applies.
MatrixLayout resolveMatrixLayout(Value value, GuardOp guard);

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Everything a predicate needs to emit its check.
struct CheckContext {
  OpBuilder &builder;
  Location loc;
  GuardOp guard;
  /// Above this many elements the comparisons are not unrolled, and the
  /// predicate declines rather than emitting an unreasonable amount of code.
  int64_t maxUnrolledElems;
};

/// Report why the check for `guard`'s condition cannot be built. The guard
/// then keeps the original computation, which is always correct, so this is a
/// warning: a rule that cannot be checked is not applied, and nothing else
/// changes.
InFlightDiagnostic emitCheckWarning(GuardOp guard);

/// What the IR alone says about a condition.
///
/// This is an internal three-way answer, not a third outcome: only `True`
/// changes what happens, by letting the check be skipped. `False` and
/// `Unknown` both still produce a guard -- a provably false condition simply
/// folds to a constant test that the LLVM optimizer drops. The extra state
/// exists so that negation can be reasoned through.
enum class Proof { True, False, Unknown };

/// Flip a proof, for `!cond`. Unknown stays unknown.
inline Proof invertProof(Proof proof) {
  switch (proof) {
  case Proof::True:
    return Proof::False;
  case Proof::False:
    return Proof::True;
  case Proof::Unknown:
    return Proof::Unknown;
  }
  return Proof::Unknown;
}

struct TesseraPredicate {
  llvm::StringRef name;
  unsigned arity;

  /// Decide the property from the IR alone, without emitting anything.
  /// Returning Unknown is the ordinary case and just means the check is
  /// emitted; returning True must be certain, since it removes the check.
  Proof (*prove)(llvm::StringRef name, ArrayRef<Value> args);

  /// Emit an i1 that is true when the property holds, or a null Value if it
  /// cannot be emitted (with a diagnostic already reported).
  ///
  /// A check must be SOUND: a true result always means the property holds. It
  /// need not be complete -- `positive_definite` deliberately answers false for
  /// matrices that do qualify, because the exact test costs as much as the
  /// operation being optimized.
  Value (*emitCheck)(ArrayRef<Value> args, CheckContext &ctx);
};

/// What is left of a condition once everything the IR settles is folded away.
struct ResidualCondition {
  /// Set when the IR settles the whole condition: true, the rewrite applies
  /// outright; false, it does not apply at all.
  std::optional<bool> settled;

  /// Otherwise, what remains to test at run time. Every predicate left in it
  /// has a runtime check.
  std::optional<Cond> cond;

  /// The first declared property that could not be shown, as the rule writes
  /// it (`SPD(A)`), for saying why a rule did not apply.
  std::string unshown;
};

/// Decide as much of a condition as the IR allows, given the values its
/// variables are bound to.
///
/// A predicate with a runtime check that cannot be proven stays in the residual
/// condition, to be tested by the guard. A predicate with no runtime check --
/// any name not in the registry, like `SPD` -- is a property that holds only by
/// declaration (see provePropertyOfValue). When it cannot be shown it counts as
/// whichever of true or false makes the whole condition false, so an unshown
/// property never selects the rewrite, under a negation or not.
ResidualCondition residualizeCondition(const Cond &cond,
                                       const llvm::StringMap<Value> &boundVars);

/// Whether the IR shows that `value` has the named property: see
/// Predicates.cpp. Answers True or Unknown, never False.
Proof provePropertyOfValue(Value value, llvm::StringRef property);

/// Look a predicate up by the name a rule used, or null if there is none.
const TesseraPredicate *lookupPredicate(llvm::StringRef name);

/// The names of every known predicate, for diagnostics.
llvm::SmallVector<llvm::StringRef> getKnownPredicateNames();

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H
