//===----------------------------------------------------------------------===//
//
// The predicates a rule can test, how to prove them at compile time, and how
// to generate their runtime checks. The user writes no checking code.
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

// The layout of a fixed-size dense matrix operand.
struct MatrixLayout {
  Type elemType;
  int64_t rows = 0;
  int64_t cols = 0;
  bool rowMajor = true;

  // `rowMajor` was assumed, not declared. Only matters to predicates that
  // aren't transpose-invariant, which then refuse to emit a check.
  bool orderInferred = false;

  bool isValid() const { return elemType && rows > 0 && cols > 0; }
  int64_t numElements() const { return rows * cols; }
};

// The layout of a value the guard carries, from the callee in its else region
// that takes it: a `tessera.layout` on that argument, or else the argument's
// lifted or byval pointee type, assumed square and row-major (setting
// `orderInferred`). Invalid if neither applies.
MatrixLayout resolveMatrixLayout(Value value, GuardOp guard);

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

// Everything a predicate needs to emit its check.
struct CheckContext {
  OpBuilder &builder;
  Location loc;
  GuardOp guard;
  // A predicate declines on matrices with more elements than this.
  int64_t maxUnrolledElems;
};

// Warn that `guard`'s check can't be built. The guard keeps the original code.
InFlightDiagnostic emitCheckWarning(GuardOp guard);

// What the IR alone says about a condition. Only `True` changes anything (the
// check is skipped); `False` exists so negation can be reasoned through.
enum class Proof { True, False, Unknown };

// Flip a proof, for `!cond`. Unknown stays unknown.
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

  // Decide the property from the IR alone. True must be certain, since it
  // removes the check.
  Proof (*prove)(llvm::StringRef name, ArrayRef<Value> args);

  // Emit an i1 that is true when the property holds, or null (after warning).
  // True must mean the property holds, but false may be wrong:
  // `positive_definite` uses a cheaper, conservative test.
  Value (*emitCheck)(ArrayRef<Value> args, CheckContext &ctx);
};

// What is left of a condition after folding what the IR settles.
struct ResidualCondition {
  // Set when the IR settles the whole condition.
  std::optional<bool> settled;

  // Otherwise, what to test at run time. Every predicate in it has a check.
  std::optional<Cond> cond;

  // The first declared property that couldn't be shown, like `SPD(A)`, for
  // saying why a rule didn't apply.
  std::string unshown;
};

// Decide as much of a condition as the IR allows. An unproven predicate with
// a runtime check stays in the residual. One without (any name not in the
// registry, like `SPD`) holds only by declaration; if it can't be shown, it
// counts as whatever makes the condition false, even under a negation.
ResidualCondition residualizeCondition(const Cond &cond,
                                       const llvm::StringMap<Value> &boundVars);

// Whether the IR shows `value` has the property. True or Unknown, never False.
Proof provePropertyOfValue(Value value, llvm::StringRef property);

// The predicate a rule names, or null.
const TesseraPredicate *lookupPredicate(llvm::StringRef name);

// The names of every known predicate, for diagnostics.
llvm::SmallVector<llvm::StringRef> getKnownPredicateNames();

// `lhs op rhs` in the LLVM dialect. Callers check that both operands have the
// same integer or float type.
Value buildArith(OpBuilder &builder, Location loc, ArithOp op, Value lhs,
                 Value rhs);

} // namespace tessera
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_AD_JAX_PASSES_TESSERA_PREDICATES_H
