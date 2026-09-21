//===----------------------------------------------------------------------===//
//
// Layout resolution, element access, and the synthesized property checks.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Tessera/Predicates.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

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

  // Otherwise infer from the by-reference type. Only the element count is
  // really known, so squareness and row-major order are assumed.
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
bool prepareMatrix(llvm::StringRef name, Value matrix, CheckContext &ctx,
                   bool requireSquare, MatrixLayout &layout) {
  layout = resolveMatrixLayout(matrix, ctx.guard);
  if (!layout.isValid()) {
    ctx.guard.emitError()
        << "cannot determine the layout of the operand of '" << name
        << "'; declare tessera.layout on the corresponding tessera.define "
           "argument";
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

/// A[i][j] == A[j][i] for every i < j. Exact.
Value emitSymmetric(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("symmetric", args[0], ctx, /*requireSquare=*/true, layout))
    return Value();

  SmallVector<Value> terms;
  for (int64_t r = 0; r < layout.rows; ++r)
    for (int64_t c = r + 1; c < layout.cols; ++c) {
      Value upper = emitElement(args[0], layout, r, c, ctx);
      Value lower = emitElement(args[0], layout, c, r, ctx);
      if (!upper || !lower)
        return Value();
      terms.push_back(emitEqual(upper, lower, ctx));
    }
  return emitConjunction(terms, ctx);
}

/// Every off-diagonal entry is zero. Exact, and meaningful for a non-square
/// matrix too.
Value emitDiagonal(ArrayRef<Value> args, CheckContext &ctx) {
  MatrixLayout layout;
  if (!prepareMatrix("diagonal", args[0], ctx, /*requireSquare=*/false, layout))
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
                     layout))
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
                     layout))
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
  if (!prepareMatrix("identity", args[0], ctx, /*requireSquare=*/true, layout))
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
    {"symmetric", 1, emitSymmetric},
    {"diagonal", 1, emitDiagonal},
    {"triangular_upper", 1, emitTriangularUpper},
    {"triangular_lower", 1, emitTriangularLower},
    {"identity", 1, emitIdentity},
};

} // namespace

namespace mlir {
namespace enzyme {
namespace tessera {

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
