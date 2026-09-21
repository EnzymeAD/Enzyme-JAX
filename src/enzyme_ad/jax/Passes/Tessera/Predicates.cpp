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

/// The else region of a guard holds a clone of the call that was matched. It is
/// the only thing that ties a value the guard carries back to the argument
/// position it occupies, which is where the layout is declared.
CallOp findOriginalCall(GuardOp guard) {
  // The region is still intact when a predicate runs, because the check is
  // synthesized before the guard is taken apart -- but do not rely on it.
  if (guard.getElseRegion().empty())
    return nullptr;
  for (Operation &op : guard.getElseRegion().front())
    if (auto call = dyn_cast<CallOp>(&op))
      return call;
  return nullptr;
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

MatrixLayout resolveMatrixLayout(Value value, GuardOp guard) {
  CallOp call = findOriginalCall(guard);
  if (!call)
    return {};

  int64_t index = -1;
  for (auto [position, operand] : llvm::enumerate(call.getArgOperands()))
    if (operand == value) {
      index = position;
      break;
    }
  if (index < 0)
    return {};

  auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      guard, call.getCalleeAttr().getAttr());
  if (!define)
    return {};

  // An explicit declaration wins: it is the only one that can describe a
  // matrix whose shape is not recoverable from its type.
  if (auto dict = dyn_cast_or_null<DictionaryAttr>(
          define.getArgAttr(index, "tessera.layout"))) {
    MatrixLayout layout = parseLayoutAttr(dict);
    if (layout.isValid())
      return layout;
    define.emitError() << "tessera.layout on argument " << index
                       << " is incomplete: it needs elem, rows and cols";
    return {};
  }

  // Otherwise infer from the by-reference type. The element type and count are
  // exact; the rows/cols split and the storage order are not in the type at
  // all, so both are assumed.
  LLVM::LLVMArrayType array = findElementArray(define.getByRefType(index));
  if (!array)
    return {};

  int64_t side = integerSquareRoot(array.getNumElements());
  if (side == 0)
    return {};

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
                     << "; declare tessera.layout on the tessera.define to be "
                        "certain";
  return layout;
}

} // namespace tessera
} // namespace enzyme
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Static proof
//===----------------------------------------------------------------------===//

/// Step through operations that pass a value along unchanged, so a guarantee
/// made about the original still applies to what the rule matched.
Value lookThroughForwarding(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (isa<LLVM::FreezeOp, LLVM::BitcastOp>(op)) {
      value = op->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

bool listContains(ArrayAttr list, llvm::StringRef name) {
  if (!list)
    return false;
  for (Attribute entry : list)
    if (auto str = dyn_cast<StringAttr>(entry))
      if (str.getValue() == name)
        return true;
  return false;
}

/// Is `value` known to have the named property?
///
/// Two sources, both declarations rather than inferences:
///
///   - the value came out of a tessera.call whose callee declares the property
///     on that result (`tessera.guarantees`), which is the useful one: written
///     once on a producing function, it holds at every call site;
///   - the defining operation carries `tessera.property.<name>` directly.
///
/// This answers True or Unknown and never False: nothing can declare that a
/// value *lacks* a property. Callers in TesseraApplyPDL.cpp rely on that -- a
/// provably false condition can therefore only come from constant-folding a
/// comparison, which always lowers to a constant test. Adding a way to
/// disprove a property means revisiting how a false condition is handled
/// there; the comment on the elision path says what has to change.
Proof provePropertyOfValue(Value value, llvm::StringRef property) {
  value = lookThroughForwarding(value);
  Operation *op = value.getDefiningOp();
  if (!op)
    return Proof::Unknown;

  std::string attrName = ("tessera.property." + property).str();
  if (op->hasAttr(attrName))
    return Proof::True;

  auto call = dyn_cast<CallOp>(op);
  if (!call)
    return Proof::Unknown;

  auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      op, call.getCalleeAttr().getAttr());
  if (!define)
    return Proof::Unknown;

  ArrayAttr resAttrs = define.getResAttrsAttr();
  unsigned resultNumber = cast<OpResult>(value).getResultNumber();
  if (!resAttrs || resultNumber >= resAttrs.size())
    return Proof::Unknown;

  auto dict = dyn_cast<DictionaryAttr>(resAttrs[resultNumber]);
  if (!dict)
    return Proof::Unknown;

  if (listContains(dict.getAs<ArrayAttr>("tessera.guarantees"), property))
    return Proof::True;
  return Proof::Unknown;
}

/// Every predicate currently tests one matrix, so they all prove the same way.
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
///   - an integer operand carries the matrix by value, so the check takes the
///     elements apart rather than going back to whatever memory it came from,
///     which may since have been written to.
Value emitElement(Value matrix, const MatrixLayout &layout, int64_t r,
                  int64_t c, CheckContext &ctx) {
  OpBuilder &b = ctx.builder;
  Location loc = ctx.loc;
  int64_t index = layout.linearIndex(r, c);

  if (isa<LLVM::LLVMPointerType>(matrix.getType())) {
    Value offset = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                            b.getI64IntegerAttr(index));
    Value address = LLVM::GEPOp::create(
        b, loc, matrix.getType(), layout.elemType, matrix, ValueRange{offset});
    return LLVM::LoadOp::create(b, loc, layout.elemType, address);
  }

  // Packed-integer form. Element 0 occupies the low bits, which is how a
  // little-endian target lays an array out when it is loaded as one integer.
  auto intType = dyn_cast<IntegerType>(matrix.getType());
  if (!intType)
    return Value();

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
    ctx.guard.emitError()
        << "cannot determine the layout of the operand of '" << name
        << "'; declare tessera.layout on the corresponding tessera.define "
           "argument";
    return false;
  }
  if (orderMatters && layout.orderInferred) {
    ctx.guard.emitError()
        << "'" << name
        << "' depends on the storage order, which was assumed "
           "rather than declared; declare tessera.layout with row_major on the "
           "corresponding tessera.define argument";
    return false;
  }
  if (requireSquare && !layout.isSquare()) {
    ctx.guard.emitError() << "'" << name << "' needs a square matrix, but the "
                          << "operand is " << layout.rows << "x" << layout.cols;
    return false;
  }
  if (layout.numElements() > ctx.maxUnrolledElems) {
    // The comparisons are emitted straight-line, so a large matrix would turn
    // into an unreasonable amount of code. A loop form would lift this.
    ctx.guard.emitError()
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

constexpr TesseraPredicate kPredicates[] = {
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

Proof proveCondition(const Cond &cond,
                     const llvm::StringMap<Value> &boundVars) {
  return std::visit(
      overloaded{
          [&](const Pred &p) -> Proof {
            const TesseraPredicate *predicate = lookupPredicate(p.name);
            if (!predicate || p.args.size() != predicate->arity)
              return Proof::Unknown;
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
            return predicate->prove(p.name, args);
          },
          [&](const Compare &c) -> Proof { return proveCompare(c, boundVars); },
          [&](const NotCond &c) -> Proof {
            return invertProof(proveCondition(*c.operand, boundVars));
          },
          [&](const AndCond &c) -> Proof {
            Proof lhs = proveCondition(*c.lhs, boundVars);
            Proof rhs = proveCondition(*c.rhs, boundVars);
            // One false side settles it even if the other is unknown.
            if (lhs == Proof::False || rhs == Proof::False)
              return Proof::False;
            if (lhs == Proof::True && rhs == Proof::True)
              return Proof::True;
            return Proof::Unknown;
          },
          [&](const OrCond &c) -> Proof {
            Proof lhs = proveCondition(*c.lhs, boundVars);
            Proof rhs = proveCondition(*c.rhs, boundVars);
            if (lhs == Proof::True || rhs == Proof::True)
              return Proof::True;
            if (lhs == Proof::False && rhs == Proof::False)
              return Proof::False;
            return Proof::Unknown;
          },
      },
      cond.data);
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
