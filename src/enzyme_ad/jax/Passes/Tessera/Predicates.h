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
/// through the call the guard kept in its else region:
///
///   1. a `tessera.layout` dictionary on the callee's argument, or
///   2. the argument's `byRefTypes` entry, walked down to its element array.
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

/// Decide a whole condition from the IR, given the values its variables are
/// bound to. Used before a guard is built, to skip building one at all.
Proof proveCondition(const Cond &cond, const llvm::StringMap<Value> &boundVars);

/// Look a predicate up by the name a rule used, or null if there is none.
const TesseraPredicate *lookupPredicate(llvm::StringRef name);

/// The names of every known predicate, for diagnostics.
llvm::SmallVector<llvm::StringRef> getKnownPredicateNames();

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H
