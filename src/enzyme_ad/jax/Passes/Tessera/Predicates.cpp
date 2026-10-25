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
#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"
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

// A call in a guard's else region that takes the value, and at which operand.
struct OriginalUse {
  CallOp call;
  unsigned index;
};

// The calls in the guard's else region (the matched root and any producers
// sunk with it) that take `value`. Their callees' arguments are where the
// layout is declared.
SmallVector<OriginalUse> findOriginalUses(Value value, GuardOp guard) {
  SmallVector<OriginalUse> uses;
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

// The array inside a chain of single-member structs, which is how an Eigen
// fixed-size matrix arrives.
LLVM::LLVMArrayType findElementArray(Type type) {
  while (auto structType = dyn_cast_or_null<LLVM::LLVMStructType>(type)) {
    ArrayRef<Type> body = structType.getBody();
    if (body.size() != 1)
      return nullptr;
    type = body[0];
  }
  return dyn_cast_or_null<LLVM::LLVMArrayType>(type);
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

  // A declared tessera.layout on any of the calls wins.
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
    define.emitWarning() << "tessera.layout on argument " << index
                         << " is incomplete: it needs elem, rows and cols; it "
                            "is ignored";
  }

  // Otherwise infer from the pointee type, which gives the element type and
  // count but not the shape or order, so assume square and row-major.
  // getCallOperandPointeeType is indexed by call operand, like `index`; the
  // argModes accessors also count "out" arguments, so they'd be off.
  for (auto [call, index] : uses) {
    DefineOp define = lookupDefine(call);
    if (!define)
      continue;
    LLVM::LLVMArrayType array =
        findElementArray(define.getCallOperandPointeeType(index));
    if (!array)
      continue;

    // The count must be a nonzero perfect square.
    int64_t count = array.getNumElements();
    int64_t side = 0;
    while (side * side < count)
      ++side;
    if (side == 0 || side * side != count)
      continue;

    MatrixLayout layout;
    layout.elemType = array.getElementType();
    layout.rows = layout.cols = side;
    layout.rowMajor = true;
    layout.orderInferred = true;

    // A wrong guess miscompiles silently, so say so.
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

// True if a fact recorded on a call argument the value is passed as (see
// PropertyPropagation.h) gives or implies the property. Never False, since
// nothing declares that a value lacks a property.
Proof mlir::enzyme::tessera::provePropertyOfValue(Value value,
                                                  llvm::StringRef property) {
  return hasRecordedProperty(value, property) ? Proof::True : Proof::Unknown;
}

namespace {

Proof proveMatrixProperty(llvm::StringRef name, ArrayRef<Value> args) {
  return provePropertyOfValue(args[0], name);
}

// The constant value of a comparison operand, if it has one.
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
  // Folded at the wider operand width (64 bits for a literal).
  if (auto *arith = std::get_if<Arith>(&expr.data)) {
    auto lhs = constantIntOperand(*arith->lhs, boundVars);
    auto rhs = constantIntOperand(*arith->rhs, boundVars);
    if (!lhs || !rhs)
      return std::nullopt;
    unsigned width = std::max(lhs->getBitWidth(), rhs->getBitWidth());
    llvm::APInt a = lhs->sext(width), b = rhs->sext(width);
    switch (arith->op) {
    case ArithOp::Add:
      return a + b;
    case ArithOp::Sub:
      return a - b;
    case ArithOp::Mul:
      return a * b;
    case ArithOp::Div:
      if (b.isZero())
        return std::nullopt;
      return a.sdiv(b);
    case ArithOp::Rem:
      if (b.isZero())
        return std::nullopt;
      return a.srem(b);
    }
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
  if (auto *arith = std::get_if<Arith>(&expr.data)) {
    auto lhs = constantFloatOperand(*arith->lhs, boundVars);
    auto rhs = constantFloatOperand(*arith->rhs, boundVars);
    if (!lhs || !rhs)
      return std::nullopt;
    // In double, as the comparison that uses the result is.
    bool lost = false;
    llvm::APFloat a = *lhs, b = *rhs;
    a.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    b.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
              &lost);
    const auto rounding = llvm::APFloat::rmNearestTiesToEven;
    switch (arith->op) {
    case ArithOp::Add:
      a.add(b, rounding);
      break;
    case ArithOp::Sub:
      a.subtract(b, rounding);
      break;
    case ArithOp::Mul:
      a.multiply(b, rounding);
      break;
    case ArithOp::Div:
      a.divide(b, rounding);
      break;
    case ArithOp::Rem:
      a.mod(b);
      break;
    }
    return a;
  }
  return std::nullopt;
}

Proof fromBool(bool value) { return value ? Proof::True : Proof::False; }

// Fold a comparison of two constants, so `if n > 64` costs nothing when n is
// constant.
Proof proveCompare(const Compare &cmp,
                   const llvm::StringMap<Value> &boundVars) {
  if (auto lhs = constantIntOperand(cmp.lhs, boundVars))
    if (auto rhs = constantIntOperand(cmp.rhs, boundVars)) {
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
      // In double, which every format here converts to exactly.
      bool lost = false;
      llvm::APFloat a = *lhs, b = *rhs;
      a.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lost);
      b.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lost);
      llvm::APFloat::cmpResult order = a.compare(b);
      if (order == llvm::APFloat::cmpUnordered)
        // NaN: only '!=' is true, as with the emitted fcmp.
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

// Element (r, c) of `matrix`, read the way the callee reads it: through a
// pointer operand, or extracted from a by-value aggregate or integer. Warns
// and returns null on failure.
Value emitElement(Value matrix, const MatrixLayout &layout, int64_t r,
                  int64_t c, CheckContext &ctx) {
  OpBuilder &b = ctx.builder;
  Location loc = ctx.loc;
  int64_t index =
      layout.rowMajor ? r * layout.cols + c : c * layout.rows + r;
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

  // A lifted argument: single-member structs around one array.
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

  // Packed into an integer, element 0 in the low bits (little-endian).
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

// The AND of the terms; true if there are none.
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

// Resolve the layout and check it suits the predicate. `orderMatters` is set
// for properties that change under transposition (only the triangular ones),
// so an assumed storage order isn't enough for them.
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
  if (requireSquare && layout.rows != layout.cols) {
    emitCheckWarning(ctx.guard)
        << "'" << name << "' needs a square matrix, but the " << "operand is "
        << layout.rows << "x" << layout.cols;
    return false;
  }
  if (layout.numElements() > ctx.maxUnrolledElems) {
    // Checks are unrolled, so a big matrix would mean too much code.
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

Value emitGreater(Value lhs, Value rhs, CheckContext &ctx) {
  if (isa<FloatType>(lhs.getType()))
    return LLVM::FCmpOp::create(ctx.builder, ctx.loc, LLVM::FCmpPredicate::ogt,
                                lhs, rhs);
  return LLVM::ICmpOp::create(ctx.builder, ctx.loc, LLVM::ICmpPredicate::sgt,
                              lhs, rhs);
}

// The helpers below take a resolved layout, so a predicate built from several
// resolves it (and remarks) once.

// A[i][j] == A[j][i] for every i < j.
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

// |A[i][i]| > sum over j != i of |A[i][j]|, for every row. An O(n^2)
// sufficient condition for the two conservative predicates, whose exact tests
// need a factorization.
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
      Value absolute = emitAbsolute(element, ctx);
      if (isa<FloatType>(total.getType()))
        total = LLVM::FAddOp::create(ctx.builder, ctx.loc, total, absolute);
      else
        total = LLVM::AddOp::create(ctx.builder, ctx.loc, total, absolute);
    }
    terms.push_back(emitGreater(diagonal, total, ctx));
  }
  return emitConjunction(terms, ctx);
}

// Every diagonal entry is strictly positive.
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

// A[i][j] == A[j][i] for every i < j. Exact.
Value emitSymmetric(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("symmetric", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();
  return symmetryOf(args[0], layout, ctx);
}

// Strictly diagonally dominant, which implies invertible (Levy-Desplanques).
// Conservative: the exact test, an LU factorization, costs as much as the
// operation being optimized. Other invertible matrices just take the original
// path.
Value emitInvertible(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("invertible", args[0], ctx, /*requireSquare=*/true,
                     /*orderMatters=*/false, layout))
    return Value();
  return diagonalDominanceOf(args[0], layout, ctx);
}

// Symmetric, positive diagonal and strictly diagonally dominant, which
// together imply positive definite. Conservative, like `invertible`: the exact
// test is a Cholesky factorization.
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

// Every off-diagonal entry is zero. Exact; works for non-square matrices.
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

// Everything strictly below the diagonal is zero. Exact.
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

// Everything strictly above the diagonal is zero. Exact.
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

// Ones on the diagonal, zeros everywhere else. Exact.
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

// A constant settles it, so `div_ui(x, 2)` needs no runtime check.
Proof provePowerOfTwo(llvm::StringRef, ArrayRef<Value> args) {
  llvm::APInt value;
  if (matchPattern(args[0], m_ConstantInt(&value)))
    return fromBool(value.isPowerOf2());
  return Proof::Unknown;
}

// Exactly one bit set: `n != 0 && (n & (n - 1)) == 0`. Exact.
//
// The bits are read as unsigned, as for an `unsigned long` count, so the
// most negative value of a signed type passes; a rule on a signed operand
// says `n > 0` as well if that matters.
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

// Whether the value is a compile-time constant, like __builtin_constant_p.
// Always settled at compile time.
Proof proveConstant(llvm::StringRef, ArrayRef<Value> args) {
  return fromBool(matchPattern(args[0], m_Constant()));
}

// Never reached, since proveConstant always settles.
Value emitConstant(ArrayRef<Value>, CheckContext &ctx) {
  return LLVM::ConstantOp::create(ctx.builder, ctx.loc,
                                  ctx.builder.getI1Type(),
                                  ctx.builder.getBoolAttr(false));
}

constexpr TesseraPredicate kPredicates[] = {
    {"constant", 1, proveConstant, emitConstant},
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

// Whether the IR shows a predicate holds. A name not in kPredicates is a
// declared property of one value.
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

// `positive` is false under an odd number of negations. An unshown property
// takes whichever value makes the whole condition false.
ResidualCondition residualize(const Cond &cond,
                              const llvm::StringMap<Value> &boundVars,
                              bool positive, std::string &unshown) {
  return std::visit(
      overloaded{
          [&](const Pred &p) -> ResidualCondition {
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

Value buildArith(OpBuilder &builder, Location loc, ArithOp op, Value lhs,
                 Value rhs) {
  if (isa<FloatType>(lhs.getType())) {
    switch (op) {
    case ArithOp::Add:
      return LLVM::FAddOp::create(builder, loc, lhs, rhs);
    case ArithOp::Sub:
      return LLVM::FSubOp::create(builder, loc, lhs, rhs);
    case ArithOp::Mul:
      return LLVM::FMulOp::create(builder, loc, lhs, rhs);
    case ArithOp::Div:
      return LLVM::FDivOp::create(builder, loc, lhs, rhs);
    case ArithOp::Rem:
      return LLVM::FRemOp::create(builder, loc, lhs, rhs);
    }
  }
  switch (op) {
  case ArithOp::Add:
    return LLVM::AddOp::create(builder, loc, lhs, rhs);
  case ArithOp::Sub:
    return LLVM::SubOp::create(builder, loc, lhs, rhs);
  case ArithOp::Mul:
    return LLVM::MulOp::create(builder, loc, lhs, rhs);
  case ArithOp::Div:
    return LLVM::SDivOp::create(builder, loc, lhs, rhs);
  case ArithOp::Rem:
    return LLVM::SRemOp::create(builder, loc, lhs, rhs);
  }
  llvm_unreachable("unknown arithmetic operator");
}

} // namespace tessera
} // namespace enzyme
} // namespace mlir
