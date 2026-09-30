//===----------------------------------------------------------------------===//
//
// Layout resolution, element access, and the synthesized property checks.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Tessera/Predicates.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Properties.h"
#include "llvm/ADT/ArrayRef.h"
#include <variant>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

//===----------------------------------------------------------------------===//
// Layout resolution
//===----------------------------------------------------------------------===//

/// A call in a guard's else region that takes the value being resolved, and
/// the operand position it takes it at.
struct OriginalUse {
  CallOp call;
  unsigned index;
};

/// The else region of a guard holds a clone of what was matched: the root call
/// and, for a chain rule, the nested producers sunk in ahead of it. Those calls
/// are the only thing tying a value the guard carries back to the argument
/// position it occupies, which is where the layout is declared. A variable in
/// the condition may be taken by any call in the chain, not just the root, so
/// all of them are searched.
SmallVector<OriginalUse> findOriginalUses(Value value, GuardOp guard) {
  SmallVector<OriginalUse> uses;
  // The region is still intact when a predicate runs, because the check is
  // synthesized before the guard is taken apart -- but do not rely on it.
  if (guard.getElseRegion().empty())
    return uses;
  for (auto call : guard.getElseRegion().front().getOps<CallOp>())
    for (auto [position, operand] : llvm::enumerate(call.getArgOperands()))
      if (operand == value)
        uses.push_back({call, static_cast<unsigned>(position)});
  return uses;
}

MatrixLayout parseLayoutAttr(DictionaryAttr dict) {
  MatrixLayout layout;
  if (auto elem = dict.getAs<TypeAttr>("elem"))
    layout.elemType = elem.getValue();
  if (auto rows = dict.getAs<IntegerAttr>("rows"))
    layout.rows = rows.getInt();
  if (auto cols = dict.getAs<IntegerAttr>("cols"))
    layout.cols = cols.getInt();
  if (auto rowMajor = dict.getAs<BoolAttr>("row_major"))
    layout.rowMajor = rowMajor.getValue();
  return layout;
}

/// Walk a nested aggregate down to the array that actually holds the elements.
/// An Eigen fixed-size matrix arrives as a chain of single-member structs
/// wrapping one array, so peeling single-member structs gets there.
LLVM::LLVMArrayType findElementArray(Type type) {
  while (auto structType = dyn_cast_or_null<LLVM::LLVMStructType>(type)) {
    ArrayRef<Type> body = structType.getBody();
    if (body.size() != 1)
      return nullptr;
    type = body[0];
  }
  return dyn_cast_or_null<LLVM::LLVMArrayType>(type);
}

int64_t integerSquareRoot(int64_t n) {
  int64_t root = 0;
  while (root * root < n)
    ++root;
  return root * root == n ? root : 0;
}

} // namespace

namespace mlir {
namespace enzyme {
namespace tessera {

InFlightDiagnostic emitCheckWarning(GuardOp guard) {
  InFlightDiagnostic diagnostic = guard.emitWarning();
  diagnostic << "optimization rule not applied here, the original call is "
                "kept: ";
  return diagnostic;
}

MatrixLayout resolveMatrixLayout(Value value, GuardOp guard) {
  SmallVector<OriginalUse> uses = findOriginalUses(value, guard);

  auto lookupDefine = [&](CallOp call) {
    return SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        guard, call.getCalleeAttr().getAttr());
  };

  // An explicit declaration wins, wherever in the chain it is made: it is the
  // only one that can describe a matrix whose shape is not recoverable from its
  // type.
  for (auto [call, index] : uses) {
    DefineOp define = lookupDefine(call);
    if (!define)
      continue;
    auto dict = dyn_cast_or_null<DictionaryAttr>(
        define.getArgAttr(index, "tessera.layout"));
    if (!dict)
      continue;
    MatrixLayout layout = parseLayoutAttr(dict);
    if (layout.isValid())
      return layout;
    // A malformed declaration is left out, as though it were not there.
    define.emitWarning() << "tessera.layout on argument " << index
                         << " is incomplete: it needs elem, rows and cols; it "
                            "is ignored";
  }

  // Otherwise infer from the by-reference type. The element type and count are
  // exact; the rows/cols split and the storage order are not in the type at
  // all, so both are assumed.
  //
  // getCallOperandPointeeType is indexed by call operand, like `index`. The
  // argModes accessors are not: they count write-only arguments, which a call
  // has no operand for, so with one of those ahead of the matrix they would
  // read some other argument's type.
  for (auto [call, index] : uses) {
    DefineOp define = lookupDefine(call);
    if (!define)
      continue;
    LLVM::LLVMArrayType array =
        findElementArray(define.getCallOperandPointeeType(index));
    if (!array)
      continue;

    int64_t side = integerSquareRoot(array.getNumElements());
    if (side == 0)
      continue;

    MatrixLayout layout;
    layout.elemType = array.getElementType();
    layout.rows = layout.cols = side;
    layout.rowMajor = true;
    layout.orderInferred = true;

    // This is a guess, and a wrong guess miscompiles quietly rather than
    // failing, so say so.
    guard.emitRemark() << "assuming argument " << index << " of '"
                       << call.getCallee() << "' is a " << side << "x" << side
                       << " row-major matrix of " << layout.elemType
                       << "; declare tessera.layout on the tessera.define to "
                          "be certain";
    return layout;
  }
  return {};
}

} // namespace tessera
} // namespace enzyme
} // namespace mlir

//===----------------------------------------------------------------------===//
// Static proof
//===----------------------------------------------------------------------===//

/// Is `value` known to have the named property?
///
/// Only what is recorded counts: a fact tessera-propagate-properties wrote on
/// a call argument the value is passed as, or one written there by hand. The
/// declarations those facts come from are described in Properties.h. A fact
/// implies others, so one of SPD answers a question about symmetry.
///
/// This answers True or Unknown and never False: nothing can declare that a
/// value *lacks* a property.
Proof mlir::enzyme::tessera::provePropertyOfValue(Value value,
                                                  llvm::StringRef property) {
  return hasRecordedProperty(value, property) ? Proof::True : Proof::Unknown;
}

namespace {

/// The matrix predicates each test one matrix, so they all prove the same way.
Proof proveMatrixProperty(llvm::StringRef name, ArrayRef<Value> args) {
  return provePropertyOfValue(args[0], name);
}

/// Constant value of a comparison operand, if it has one. A literal in the rule
/// is constant by construction; a variable is only constant when the value it
/// matched is.
std::optional<llvm::APInt>
constantIntOperand(const Expr &expr, const llvm::StringMap<Value> &boundVars) {
  if (auto *lit = std::get_if<IntLit>(&expr.data))
    return llvm::APInt(64, lit->value, /*isSigned=*/true);
  if (auto *var = std::get_if<Var>(&expr.data)) {
    auto it = boundVars.find(var->name);
    if (it == boundVars.end())
      return std::nullopt;
    llvm::APInt value;
    if (matchPattern(it->second, m_ConstantInt(&value)))
      return value;
  }
  return std::nullopt;
}

std::optional<llvm::APFloat>
constantFloatOperand(const Expr &expr,
                     const llvm::StringMap<Value> &boundVars) {
  if (auto *lit = std::get_if<FloatLit>(&expr.data))
    return llvm::APFloat(lit->value);
  if (auto *var = std::get_if<Var>(&expr.data)) {
    auto it = boundVars.find(var->name);
    if (it == boundVars.end())
      return std::nullopt;
    llvm::APFloat value(0.0);
    if (matchPattern(it->second, m_ConstantFloat(&value)))
      return value;
  }
  return std::nullopt;
}

Proof fromBool(bool value) { return value ? Proof::True : Proof::False; }

/// Fold a comparison whose operands are both known at compile time. This is
/// what makes `if n > 64, ...` free when n is a constant.
Proof proveCompare(const Compare &cmp,
                   const llvm::StringMap<Value> &boundVars) {
  if (auto lhs = constantIntOperand(cmp.lhs, boundVars))
    if (auto rhs = constantIntOperand(cmp.rhs, boundVars)) {
      // Widen to a common width before comparing; the literal is 64-bit while
      // the matched constant can be any width.
      unsigned width = std::max(lhs->getBitWidth(), rhs->getBitWidth());
      llvm::APInt a = lhs->sext(width);
      llvm::APInt b = rhs->sext(width);
      switch (cmp.op) {
      case CmpOp::Eq:
        return fromBool(a.eq(b));
      case CmpOp::Ne:
        return fromBool(a.ne(b));
      case CmpOp::Lt:
        return fromBool(a.slt(b));
      case CmpOp::Le:
        return fromBool(a.sle(b));
      case CmpOp::Gt:
        return fromBool(a.sgt(b));
      case CmpOp::Ge:
        return fromBool(a.sge(b));
      }
    }

  if (auto lhs = constantFloatOperand(cmp.lhs, boundVars))
    if (auto rhs = constantFloatOperand(cmp.rhs, boundVars)) {
      // Compare in double, which both sides convert to exactly for every
      // format the rule language can produce.
      bool lost = false;
      llvm::APFloat a = *lhs, b = *rhs;
      a.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lost);
      b.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lost);
      llvm::APFloat::cmpResult order = a.compare(b);
      if (order == llvm::APFloat::cmpUnordered)
        // A NaN makes every ordered comparison false and '!=' true, matching
        // what the emitted fcmp would do.
        return cmp.op == CmpOp::Ne ? Proof::True : Proof::False;
      switch (cmp.op) {
      case CmpOp::Eq:
        return fromBool(order == llvm::APFloat::cmpEqual);
      case CmpOp::Ne:
        return fromBool(order != llvm::APFloat::cmpEqual);
      case CmpOp::Lt:
        return fromBool(order == llvm::APFloat::cmpLessThan);
      case CmpOp::Le:
        return fromBool(order != llvm::APFloat::cmpGreaterThan);
      case CmpOp::Gt:
        return fromBool(order == llvm::APFloat::cmpGreaterThan);
      case CmpOp::Ge:
        return fromBool(order != llvm::APFloat::cmpLessThan);
      }
    }

  return Proof::Unknown;
}

//===----------------------------------------------------------------------===//
// Element access
//===----------------------------------------------------------------------===//

/// Load element (r, c) of `matrix`.
///
/// Which form is used follows from what the callee itself will read, so that
/// the check and the call can never disagree:
///
///   - a pointer operand is read by the callee through that pointer, so the
///     check reads the same memory;
///   - an integer or aggregate operand carries the matrix by value, so the
///     check takes the elements apart rather than going back to whatever memory
///     it came from, which may since have been written to.
///
/// On failure this reports why and returns a null Value, so the first element
/// a predicate cannot read is also the only error it produces.
Value emitElement(Value matrix, const MatrixLayout &layout, int64_t r,
                  int64_t c, CheckContext &ctx) {
  OpBuilder &b = ctx.builder;
  Location loc = ctx.loc;
  int64_t index = layout.linearIndex(r, c);
  Type matrixType = matrix.getType();

  auto unreadable = [&]() {
    emitCheckWarning(ctx.guard)
        << "cannot read the elements of a matrix operand of "
           "type "
        << matrixType << " as " << layout.rows << "x" << layout.cols << " "
        << layout.elemType;
    return Value();
  };

  if (isa<LLVM::LLVMPointerType>(matrixType)) {
    Value offset = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                            b.getI64IntegerAttr(index));
    Value address = LLVM::GEPOp::create(b, loc, matrixType, layout.elemType,
                                        matrix, ValueRange{offset});
    return LLVM::LoadOp::create(b, loc, layout.elemType, address);
  }

  // Aggregate form. A lifted argument arrives as the value loaded from its
  // pointer, which for a fixed-size matrix is the same single-member-struct
  // chain around one array that layout inference walks.
  if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(matrixType)) {
    SmallVector<int64_t> position;
    Type type = matrixType;
    while (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
      if (structType.getBody().size() != 1)
        return unreadable();
      position.push_back(0);
      type = structType.getBody()[0];
    }
    auto array = dyn_cast<LLVM::LLVMArrayType>(type);
    if (!array || array.getElementType() != layout.elemType ||
        static_cast<uint64_t>(index) >= array.getNumElements())
      return unreadable();
    position.push_back(index);
    return LLVM::ExtractValueOp::create(b, loc, matrix, position);
  }

  // Packed-integer form. Element 0 occupies the low bits, which is how a
  // little-endian target lays an array out when it is loaded as one integer.
  auto intType = dyn_cast<IntegerType>(matrixType);
  if (!intType)
    return unreadable();

  unsigned elemBits = layout.elemType.getIntOrFloatBitWidth();
  Value shiftAmount = LLVM::ConstantOp::create(
      b, loc, intType, b.getIntegerAttr(intType, index * elemBits));
  Value shifted = LLVM::LShrOp::create(b, loc, matrix, shiftAmount);
  Value narrowed = LLVM::TruncOp::create(
      b, loc, IntegerType::get(b.getContext(), elemBits), shifted);
  if (isa<FloatType>(layout.elemType))
    return LLVM::BitcastOp::create(b, loc, layout.elemType, narrowed);
  return narrowed;
}

Value emitScalarConstant(Type type, int64_t value, CheckContext &ctx) {
  if (isa<FloatType>(type))
    return LLVM::ConstantOp::create(
        ctx.builder, ctx.loc, type,
        ctx.builder.getFloatAttr(type, static_cast<double>(value)));
  return LLVM::ConstantOp::create(ctx.builder, ctx.loc, type,
                                  ctx.builder.getIntegerAttr(type, value));
}

Value emitEqual(Value lhs, Value rhs, CheckContext &ctx) {
  if (isa<FloatType>(lhs.getType()))
    return LLVM::FCmpOp::create(ctx.builder, ctx.loc, LLVM::FCmpPredicate::oeq,
                                lhs, rhs);
  return LLVM::ICmpOp::create(ctx.builder, ctx.loc, LLVM::ICmpPredicate::eq,
                              lhs, rhs);
}

/// Conjoin the terms. An empty list is vacuously true -- a 1x1 matrix really
/// is symmetric.
Value emitConjunction(ArrayRef<Value> terms, CheckContext &ctx) {
  if (terms.empty())
    return LLVM::ConstantOp::create(ctx.builder, ctx.loc,
                                    ctx.builder.getI1Type(),
                                    ctx.builder.getBoolAttr(true));
  Value result = terms.front();
  for (Value term : terms.drop_front())
    result = LLVM::AndOp::create(ctx.builder, ctx.loc, result, term);
  return result;
}

/// Shared entry checks for the matrix predicates.
///
/// `orderMatters` says the property is not preserved by transposition, so an
/// assumed storage order could make the check answer true for a matrix that
/// does not have the property. Only the triangular pair is in that position:
/// reading column-major storage as row-major turns an upper triangle into a
/// lower one. Everything else here is transpose-invariant -- symmetry,
/// diagonality and identity obviously so, and diagonal dominance because
/// column dominance implies nonsingularity just as row dominance does -- so an
/// inferred order cannot make those unsound.
bool prepareMatrix(llvm::StringRef name, Value matrix, CheckContext &ctx,
                   bool requireSquare, bool orderMatters,
                   MatrixLayout &layout) {
  layout = resolveMatrixLayout(matrix, ctx.guard);
  if (!layout.isValid()) {
    emitCheckWarning(ctx.guard)
        << "cannot determine the layout of the operand of '" << name
        << "'; declare tessera.layout on the corresponding tessera.define "
           "argument";
    return false;
  }
  if (orderMatters && layout.orderInferred) {
    emitCheckWarning(ctx.guard)
        << "'" << name
        << "' depends on the storage order, which was assumed "
           "rather than declared; declare tessera.layout with row_major on the "
           "corresponding tessera.define argument";
    return false;
  }
  if (requireSquare && !layout.isSquare()) {
    emitCheckWarning(ctx.guard)
        << "'" << name << "' needs a square matrix, but the " << "operand is "
        << layout.rows << "x" << layout.cols;
    return false;
  }
  if (layout.numElements() > ctx.maxUnrolledElems) {
    // The comparisons are emitted straight-line, so a large matrix would turn
    // into an unreasonable amount of code. A loop form would lift this.
    emitCheckWarning(ctx.guard)
        << "'" << name << "' needs " << layout.numElements()
        << " elements checked, above the max-unrolled-elems limit of "
        << ctx.maxUnrolledElems << "; a loop form is not implemented yet";
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// The predicates
//===----------------------------------------------------------------------===//

Value emitAbsolute(Value value, CheckContext &ctx) {
  Type type = value.getType();
  if (isa<FloatType>(type))
    return LLVM::FAbsOp::create(ctx.builder, ctx.loc, value);
  Value zero = emitScalarConstant(type, 0, ctx);
  Value negated = LLVM::SubOp::create(ctx.builder, ctx.loc, zero, value);
  Value isNegative = LLVM::ICmpOp::create(
      ctx.builder, ctx.loc, LLVM::ICmpPredicate::slt, value, zero);
  return LLVM::SelectOp::create(ctx.builder, ctx.loc, isNegative, negated,
                                value);
}

Value emitAdd(Value lhs, Value rhs, CheckContext &ctx) {
  if (isa<FloatType>(lhs.getType()))
    return LLVM::FAddOp::create(ctx.builder, ctx.loc, lhs, rhs);
  return LLVM::AddOp::create(ctx.builder, ctx.loc, lhs, rhs);
}

Value emitGreater(Value lhs, Value rhs, CheckContext &ctx) {
  if (isa<FloatType>(lhs.getType()))
    return LLVM::FCmpOp::create(ctx.builder, ctx.loc, LLVM::FCmpPredicate::ogt,
                                lhs, rhs);
  return LLVM::ICmpOp::create(ctx.builder, ctx.loc, LLVM::ICmpPredicate::sgt,
                              lhs, rhs);
}

// The pieces below take an already-resolved layout so that a predicate built
// out of several of them resolves it once. Resolving twice would also repeat
// the remark that layout inference emits.

/// A[i][j] == A[j][i] for every i < j.
Value symmetryOf(Value matrix, const MatrixLayout &layout, CheckContext &ctx) {
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r)
    for (int64_t c = r + 1; c < layout.cols; ++c) {
      Value upper = emitElement(matrix, layout, r, c, ctx);
      Value lower = emitElement(matrix, layout, c, r, ctx);
      if (!upper || !lower)
        return Value();
      terms.push_back(emitEqual(upper, lower, ctx));
    }
  return emitConjunction(terms, ctx);
}

/// |A[i][i]| > sum over j != i of |A[i][j]|, for every row.
///
/// Strict diagonal dominance is the cheap sufficient condition the two
/// conservative predicates are built from: it costs O(n^2), where an exact
/// test of either property costs a factorization.
Value diagonalDominanceOf(Value matrix, const MatrixLayout &layout,
                          CheckContext &ctx) {
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r) {
    Value diagonal = emitElement(matrix, layout, r, r, ctx);
    if (!diagonal)
      return Value();
    diagonal = emitAbsolute(diagonal, ctx);

    Value total = emitScalarConstant(layout.elemType, 0, ctx);
    for (int64_t c = 0; c < layout.cols; ++c) {
      if (c == r)
        continue;
      Value element = emitElement(matrix, layout, r, c, ctx);
      if (!element)
        return Value();
      total = emitAdd(total, emitAbsolute(element, ctx), ctx);
    }
    terms.push_back(emitGreater(diagonal, total, ctx));
  }
  return emitConjunction(terms, ctx);
}

/// Every diagonal entry is strictly positive.
Value positiveDiagonalOf(Value matrix, const MatrixLayout &layout,
                         CheckContext &ctx) {
  Value zero = emitScalarConstant(layout.elemType, 0, ctx);
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r) {
    Value diagonal = emitElement(matrix, layout, r, r, ctx);
    if (!diagonal)
      return Value();
    terms.push_back(emitGreater(diagonal, zero, ctx));
  }
  return emitConjunction(terms, ctx);
}

/// A[i][j] == A[j][i] for every i < j. Exact.
Value emitSymmetric(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("symmetric", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();
  return symmetryOf(args[0], layout, ctx);
}

/// Strictly diagonally dominant, which implies nonsingular by the
/// Levy-Desplanques theorem.
///
/// CONSERVATIVE. The exact test is an LU factorization with pivoting, which
/// costs as much as the operation being optimized, so it is not worth emitting
/// no matter who writes it. This answers true only for matrices it is certain
/// about; an invertible matrix that is not diagonally dominant simply tests
/// false and takes the general path, costing only the check. A true answer is
/// never wrong, which is the property that matters.
Value emitInvertible(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("invertible", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();
  return diagonalDominanceOf(args[0], layout, ctx);
}

/// Symmetric, with a positive diagonal, and strictly diagonally dominant --
/// which together imply positive definiteness.
///
/// CONSERVATIVE, for the same reason as `invertible`: the exact test is a
/// Cholesky factorization. Sound but incomplete.
Value emitPositiveDefinite(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("positive_definite", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();

  Value symmetric = symmetryOf(args[0], layout, ctx);
  Value positive = positiveDiagonalOf(args[0], layout, ctx);
  Value dominant = diagonalDominanceOf(args[0], layout, ctx);
  if (!symmetric || !positive || !dominant)
    return Value();
  return emitConjunction({symmetric, positive, dominant}, ctx);
}

/// Every off-diagonal entry is zero. Exact, and meaningful for a non-square
/// matrix too.
Value emitDiagonal(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("diagonal", args[0], ctx, /*requireSquare=*/false,
                     /*orderMatters=*/false, layout))
    return Value();

  Value zero = emitScalarConstant(layout.elemType, 0, ctx);
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r)
    for (int64_t c = 0; c < layout.cols; ++c) {
      if (r == c)
        continue;
      Value element = emitElement(args[0], layout, r, c, ctx);
      if (!element)
        return Value();
      terms.push_back(emitEqual(element, zero, ctx));
    }
  return emitConjunction(terms, ctx);
}

/// Everything strictly below the diagonal is zero. Exact.
Value emitTriangularUpper(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("triangular_upper", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/true, layout))
    return Value();

  Value zero = emitScalarConstant(layout.elemType, 0, ctx);
  SmallVector<Value> terms;
  for (int64_t r = 1; r < layout.rows; ++r)
    for (int64_t c = 0; c < r; ++c) {
      Value element = emitElement(args[0], layout, r, c, ctx);
      if (!element)
        return Value();
      terms.push_back(emitEqual(element, zero, ctx));
    }
  return emitConjunction(terms, ctx);
}

/// Everything strictly above the diagonal is zero. Exact.
Value emitTriangularLower(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("triangular_lower", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/true, layout))
    return Value();

  Value zero = emitScalarConstant(layout.elemType, 0, ctx);
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r)
    for (int64_t c = r + 1; c < layout.cols; ++c) {
      Value element = emitElement(args[0], layout, r, c, ctx);
      if (!element)
        return Value();
      terms.push_back(emitEqual(element, zero, ctx));
    }
  return emitConjunction(terms, ctx);
}

/// Ones on the diagonal, zeros everywhere else. Exact.
Value emitIdentity(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("identity", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();

  Value zero = emitScalarConstant(layout.elemType, 0, ctx);
  Value one = emitScalarConstant(layout.elemType, 1, ctx);
  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r)
    for (int64_t c = 0; c < layout.cols; ++c) {
      Value element = emitElement(args[0], layout, r, c, ctx);
      if (!element)
        return Value();
      terms.push_back(emitEqual(element, r == c ? one : zero, ctx));
    }
  return emitConjunction(terms, ctx);
}

//===----------------------------------------------------------------------===//
// Integer predicates
//===----------------------------------------------------------------------===//

/// A constant settles `power_of_two` at compile time, so a call that passes a
/// literal, like `div_ui(x, 2)`, is rewritten with no check at all.
Proof provePowerOfTwo(llvm::StringRef, ArrayRef<Value> args) {
  llvm::APInt value;
  if (matchPattern(args[0], m_ConstantInt(&value)))
    return fromBool(value.isPowerOf2());
  return Proof::Unknown;
}

/// Exactly one bit set: `n != 0 && (n & (n - 1)) == 0`. Exact.
///
/// The bits are read as unsigned, as for an `unsigned long` count, so the
/// most negative value of a signed type passes; a rule on a signed operand
/// says `n > 0` as well if that matters.
Value emitPowerOfTwo(ArrayRef<Value> args, CheckContext &ctx) {
  Value n = args[0];
  auto type = dyn_cast<IntegerType>(n.getType());
  if (!type) {
    emitCheckWarning(ctx.guard)
        << "predicate 'power_of_two' takes an integer, but got "
        << n.getType();
    return Value();
  }
  Value zero = emitScalarConstant(type, 0, ctx);
  Value one = emitScalarConstant(type, 1, ctx);
  Value below = LLVM::SubOp::create(ctx.builder, ctx.loc, n, one);
  Value shared = LLVM::AndOp::create(ctx.builder, ctx.loc, n, below);
  Value nonzero = LLVM::ICmpOp::create(ctx.builder, ctx.loc,
                                       LLVM::ICmpPredicate::ne, n, zero);
  return emitConjunction({nonzero, emitEqual(shared, zero, ctx)}, ctx);
}

constexpr TesseraPredicate kPredicates[] = {
    {"power_of_two", 1, provePowerOfTwo, emitPowerOfTwo},
    {"symmetric", 1, proveMatrixProperty, emitSymmetric},
    {"diagonal", 1, proveMatrixProperty, emitDiagonal},
    {"triangular_upper", 1, proveMatrixProperty, emitTriangularUpper},
    {"triangular_lower", 1, proveMatrixProperty, emitTriangularLower},
    {"identity", 1, proveMatrixProperty, emitIdentity},
    {"invertible", 1, proveMatrixProperty, emitInvertible},
    {"positive_definite", 1, proveMatrixProperty, emitPositiveDefinite},
};

} // namespace

namespace mlir {
namespace enzyme {
namespace tessera {

namespace {

/// Whether a predicate holds of what its variables are bound to, from the IR
/// alone. A predicate with a runtime check proves the way it says; any other
/// name is a declared property of one value, and proves only by declaration.
Proof provePredicate(const Pred &p, const llvm::StringMap<Value> &boundVars) {
  SmallVector<Value> args;
  for (const Expr &arg : p.args) {
    auto *var = std::get_if<Var>(&arg.data);
    if (!var)
      return Proof::Unknown;
    auto it = boundVars.find(var->name);
    if (it == boundVars.end())
      return Proof::Unknown;
    args.push_back(it->second);
  }
  if (const TesseraPredicate *predicate = lookupPredicate(p.name)) {
    if (args.size() != predicate->arity)
      return Proof::Unknown;
    return predicate->prove(p.name, args);
  }
  if (args.size() != 1)
    return Proof::Unknown;
  return provePropertyOfValue(args[0], p.name);
}

ResidualCondition settledAs(bool holds) {
  ResidualCondition result;
  result.settled = holds;
  return result;
}

ResidualCondition remaining(Cond cond) {
  ResidualCondition result;
  result.cond = std::move(cond);
  return result;
}

/// `positive` is false beneath an odd number of negations. It decides what a
/// property that cannot be shown stands for: whichever value makes the
/// condition as a whole false, so that not knowing never takes the rewrite.
ResidualCondition residualize(const Cond &cond,
                              const llvm::StringMap<Value> &boundVars,
                              bool positive, std::string &unshown) {
  return std::visit(
      overloaded{
          [&](const Pred &p) -> ResidualCondition {
            // Only a predicate with a compile-time answer, like
            // power_of_two(6), is ever proven false; a declared property is
            // either shown or not known.
            Proof proof = provePredicate(p, boundVars);
            if (proof != Proof::Unknown)
              return settledAs(proof == Proof::True);
            if (lookupPredicate(p.name))
              return remaining(Cond(Pred(p)));
            if (unshown.empty())
              unshown = renderCond(Cond(Pred(p)));
            return settledAs(!positive);
          },
          [&](const Compare &c) -> ResidualCondition {
            Proof proof = proveCompare(c, boundVars);
            if (proof != Proof::Unknown)
              return settledAs(proof == Proof::True);
            return remaining(Cond(Compare(c)));
          },
          [&](const NotCond &c) -> ResidualCondition {
            ResidualCondition inner =
                residualize(*c.operand, boundVars, !positive, unshown);
            if (inner.settled)
              return settledAs(!*inner.settled);
            return remaining(Cond(NotCond{box(std::move(*inner.cond))}));
          },
          [&](const AndCond &c) -> ResidualCondition {
            ResidualCondition lhs =
                residualize(*c.lhs, boundVars, positive, unshown);
            ResidualCondition rhs =
                residualize(*c.rhs, boundVars, positive, unshown);
            if (lhs.settled == false || rhs.settled == false)
              return settledAs(false);
            if (lhs.settled)
              return rhs;
            if (rhs.settled)
              return lhs;
            return remaining(Cond(
                AndCond{box(std::move(*lhs.cond)), box(std::move(*rhs.cond))}));
          },
          [&](const OrCond &c) -> ResidualCondition {
            ResidualCondition lhs =
                residualize(*c.lhs, boundVars, positive, unshown);
            ResidualCondition rhs =
                residualize(*c.rhs, boundVars, positive, unshown);
            if (lhs.settled == true || rhs.settled == true)
              return settledAs(true);
            if (lhs.settled)
              return rhs;
            if (rhs.settled)
              return lhs;
            return remaining(Cond(
                OrCond{box(std::move(*lhs.cond)), box(std::move(*rhs.cond))}));
          },
      },
      cond.data);
}

} // namespace

ResidualCondition
residualizeCondition(const Cond &cond,
                     const llvm::StringMap<Value> &boundVars) {
  std::string unshown;
  ResidualCondition result =
      residualize(cond, boundVars, /*positive=*/true, unshown);
  result.unshown = std::move(unshown);
  return result;
}

const TesseraPredicate *lookupPredicate(llvm::StringRef name) {
  for (const TesseraPredicate &predicate : kPredicates)
    if (predicate.name == name)
      return &predicate;
  return nullptr;
}

llvm::SmallVector<llvm::StringRef> getKnownPredicateNames() {
  llvm::SmallVector<llvm::StringRef> names;
  for (const TesseraPredicate &predicate : kPredicates)
    names.push_back(predicate.name);
  return names;
}

} // namespace tessera
} // namespace enzyme
} // namespace mlir
