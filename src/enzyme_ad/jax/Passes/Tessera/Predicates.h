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
/// The value itself rarely says enough -- after llvm-to-tessera a matrix is
/// typically a flat integer such as i512, which records the size but not the
/// element type or the shape. The declaration of the callee does say, so the
/// lookup goes through the call the guard kept in its else region:
///
///   1. a `tessera.layout` dictionary on the callee's argument, or
///   2. the argument's by-reference type, walked down to its element array and
///      assumed square and row-major -- a guess, so it emits a remark.
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

struct TesseraPredicate {
  llvm::StringRef name;
  unsigned arity;

  /// Emit an i1 that is true when the property holds, or a null Value if it
  /// cannot be emitted (with a diagnostic already reported).
  ///
  /// A check must be SOUND: a true result always means the property holds. It
  /// need not be complete -- `positive_definite` deliberately answers false for
  /// matrices that do qualify, because the exact test costs as much as the
  /// operation being optimized.
  Value (*emitCheck)(ArrayRef<Value> args, CheckContext &ctx);
};

/// Look a predicate up by the name a rule used, or null if there is none.
const TesseraPredicate *lookupPredicate(llvm::StringRef name);

/// The names of every known predicate, for diagnostics.
llvm::SmallVector<llvm::StringRef> getKnownPredicateNames();

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H
