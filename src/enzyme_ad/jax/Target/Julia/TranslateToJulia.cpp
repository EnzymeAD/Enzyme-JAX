#include "src/enzyme_ad/jax/Target/Julia/TranslateToJulia.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"

using namespace mlir;
using namespace mlir::enzymexla::julia;

namespace {

static bool isJuliaKeyword(StringRef s) {
  static const llvm::StringSet<> keywords = {
      "baremodule", "begin",   "break",    "catch",   "const",   "continue",
      "do",         "else",    "elseif",   "end",     "export",  "false",
      "finally",    "for",     "function", "global",  "if",      "import",
      "in",         "isa",     "let",      "local",   "macro",   "module",
      "mutable",    "primitive", "quote",  "return",  "struct",  "true",
      "try",        "type",    "using",    "where",   "while",   "Array",
      "Bool",       "Int",     "Int8",     "Int16",   "Int32",   "Int64",
      "UInt8",      "UInt16",  "UInt32",   "UInt64",  "Float16", "BFloat16",
      "Float32",    "Float64", "ComplexF32", "ComplexF64", "LinearAlgebra",
      "Base",       "Core"};
  return keywords.contains(s);
}

static std::string sanitizeIdentifier(StringRef raw, StringRef fallbackPrefix) {
  std::string out;
  out.reserve(raw.size() + 4);
  for (size_t i = 0; i < raw.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(raw[i]);
    if (std::isalnum(c) || c == '_') {
      out.push_back(static_cast<char>(c));
    } else if (c == '!' && i + 1 == raw.size() && !out.empty()) {
      out.push_back('!');
    } else {
      out.push_back('_');
    }
  }
  if (out.empty())
    out = fallbackPrefix.str();
  if (std::isdigit(static_cast<unsigned char>(out[0])))
    out.insert(out.begin(), '_');
  if (isJuliaKeyword(out))
    out = "_" + out;
  return out;
}

class JuliaEmitter {
public:
  explicit JuliaEmitter(llvm::raw_ostream &os) : os(os) {}

  LogicalResult emitModule(ModuleOp module);
  LogicalResult emitFunc(FuncOp func);

private:
  llvm::raw_ostream &os;
  int indentLevel = 0;
  unsigned nextId = 0;
  llvm::DenseMap<Value, std::string> valueExprs;
  llvm::StringSet<> usedNames;

  llvm::raw_ostream &indent() {
    os.indent(indentLevel * 4);
    return os;
  }

  std::string freshName(StringRef prefix = "v") {
    while (true) {
      std::string cand = (prefix + llvm::Twine(nextId++)).str();
      if (usedNames.insert(cand).second)
        return cand;
    }
  }

  std::string registerNamed(Value v, StringRef hint) {
    std::string base = sanitizeIdentifier(hint, "arg");
    std::string cand = base;
    unsigned suffix = 0;
    while (!usedNames.insert(cand).second)
      cand = base + "_" + std::to_string(++suffix);
    valueExprs[v] = cand;
    return cand;
  }

  std::string lookup(Value v) const {
    auto it = valueExprs.find(v);
    assert(it != valueExprs.end() && "unmapped SSA value in JuliaEmitter");
    return it->second;
  }

  static std::string formatScalarType(Type t);
  static std::string formatType(Type t);
  static std::string formatFloatLiteral(const llvm::APFloat &val, Type ty);
  static std::string formatScalarAttr(Attribute attr, Type ty);

  std::string formatDimSize(Value bufferOrNull, int64_t staticDim,
                            ValueRange dynamicDims, unsigned &dynIdx) const;
  std::string formatShapeArgs(ArrayRef<int64_t> shape,
                              ValueRange dynamicDims) const;

  bool shouldInlinePureOp(Operation *op) const;
  int exprDepth(Value v, int limit = 7) const;

  LogicalResult emitBlockStatements(Block &block, Operation *&terminatorOut);
  LogicalResult emitOperation(Operation *op);

  std::string buildPureScalarExpr(Operation *op);
  std::string buildCastExpr(Value input, Type dstTy);
  std::string buildViewSliceExpr(Value src, ArrayRef<int64_t> starts,
                                 ArrayRef<int64_t> limits,
                                 ArrayRef<int64_t> strides);
  std::string buildViewDynamicSliceExpr(Value src, ValueRange starts,
                                        ArrayRef<int64_t> sizes);
  std::string buildViewReshapeExpr(Value src, ShapedType resTy,
                                   ValueRange dynamicDims);

  LogicalResult emitKernelLoop(ValueRange outs, Region &body);
  LogicalResult emitMatmul(Value dst, Value lhs, Value rhs, Value alpha,
                           Value beta, TransposeMode lhsTrans,
                           TransposeMode rhsTrans,
                           ArrayRef<int64_t> batchLhs,
                           ArrayRef<int64_t> batchRhs,
                           ArrayRef<int64_t> batchOut);
  LogicalResult emitDotGeneral(Value dst, Value lhs, Value rhs,
                               ArrayRef<int64_t> lhsBatch,
                               ArrayRef<int64_t> rhsBatch,
                               ArrayRef<int64_t> lhsContract,
                               ArrayRef<int64_t> rhsContract,
                               ArrayRef<int64_t> resultDimOrder);
  LogicalResult emitReduceScalar(ReduceOp op);
  LogicalResult emitReduceArray(ValueRange dsts, ValueRange inputs,
                                ValueRange inits, ArrayRef<int64_t> reduceDims,
                                Region &combiner);
  LogicalResult emitReduceWindow(ValueRange dsts, ValueRange inputs,
                                 ValueRange inits, ArrayRef<int64_t> winDims,
                                 ArrayRef<int64_t> winStrides,
                                 ArrayRef<int64_t> baseDil,
                                 ArrayRef<int64_t> winDil,
                                 ArrayRef<int64_t> padding, Region &combiner);
  LogicalResult emitPad(Value dst, Value src, Value padVal,
                        ArrayRef<int64_t> low, ArrayRef<int64_t> high,
                        ArrayRef<int64_t> interior);
  LogicalResult emitDynamicSliceCopy(Value dst, Value src, ValueRange starts,
                                     ArrayRef<int64_t> sizes);
  LogicalResult emitDynamicUpdateSlice(Value dst, Value update,
                                       ValueRange starts);
  LogicalResult emitGather(Value dst, Value operand, Value startIndices,
                           ArrayRef<int64_t> offsetDims,
                           ArrayRef<int64_t> collapsedDims,
                           ArrayRef<int64_t> opBatchDims,
                           ArrayRef<int64_t> idxBatchDims,
                           ArrayRef<int64_t> startIndexMap,
                           int64_t indexVecDim, ArrayRef<int64_t> sliceSizes);
  LogicalResult emitScatter(ValueRange dsts, Value scatterIndices,
                            ValueRange updates,
                            ArrayRef<int64_t> updateWinDims,
                            ArrayRef<int64_t> insertedWinDims,
                            ArrayRef<int64_t> inBatchDims,
                            ArrayRef<int64_t> idxBatchDims,
                            ArrayRef<int64_t> scatterMap, int64_t indexVecDim,
                            Region &updateComp);
  LogicalResult emitSort(ValueRange dsts, int64_t dimension, bool isStable,
                         Region &comparator);
  LogicalResult emitConv(Value dst, Value lhs, Value rhs,
                         ArrayRef<int64_t> strides, ArrayRef<int64_t> padding,
                         ArrayRef<int64_t> lhsDil, ArrayRef<int64_t> rhsDil,
                         ArrayRef<bool> winRev, int64_t inBatchDim,
                         int64_t inFeatDim, ArrayRef<int64_t> inSpatialDims,
                         int64_t kerInFeatDim, int64_t kerOutFeatDim,
                         ArrayRef<int64_t> kerSpatialDims, int64_t outBatchDim,
                         int64_t outFeatDim, ArrayRef<int64_t> outSpatialDims,
                         int64_t featGroups, int64_t batchGroups);
  LogicalResult emitCholesky(Value dst, bool lower);
  LogicalResult emitTriangularSolve(Value dst, Value a, bool leftSide,
                                    bool lower, bool unitDiag,
                                    TransposeMode transA);
  LogicalResult emitIf(IfOp op);
  LogicalResult emitWhile(WhileOp op);
};

std::string JuliaEmitter::formatScalarType(Type t) {
  if (t.isIndex())
    return "Int";
  if (auto intTy = dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    if (w == 1)
      return "Bool";
    bool u = intTy.isUnsigned();
    switch (w) {
    case 8:
      return u ? "UInt8" : "Int8";
    case 16:
      return u ? "UInt16" : "Int16";
    case 32:
      return u ? "UInt32" : "Int32";
    case 64:
      return u ? "UInt64" : "Int64";
    case 128:
      return u ? "UInt128" : "Int128";
    default:
      return u ? "UInt64" : "Int64";
    }
  }
  if (isa<Float16Type>(t))
    return "Float16";
  if (isa<BFloat16Type>(t))
    return "BFloat16";
  if (isa<Float32Type>(t))
    return "Float32";
  if (isa<Float64Type>(t))
    return "Float64";
  if (auto cplx = dyn_cast<ComplexType>(t)) {
    Type elem = cplx.getElementType();
    if (isa<Float32Type>(elem))
      return "ComplexF32";
    if (isa<Float64Type>(elem))
      return "ComplexF64";
    return "Complex{" + formatScalarType(elem) + "}";
  }
  return "Any";
}

std::string JuliaEmitter::formatType(Type t) {
  if (auto arrTy = dyn_cast<ArrayType>(t)) {
    return "Array{" + formatScalarType(arrTy.getElementType()) + ", " +
           std::to_string(arrTy.getRank()) + "}";
  }
  if (auto viewTy = dyn_cast<ViewType>(t)) {
    return "AbstractArray{" + formatScalarType(viewTy.getElementType()) + ", " +
           std::to_string(viewTy.getRank()) + "}";
  }
  return formatScalarType(t);
}

std::string JuliaEmitter::formatFloatLiteral(const llvm::APFloat &val,
                                             Type ty) {
  if (val.isNaN()) {
    if (isa<Float32Type>(ty))
      return "NaN32";
    if (isa<Float16Type>(ty))
      return "NaN16";
    if (isa<BFloat16Type>(ty))
      return "BFloat16(NaN32)";
    return "NaN";
  }
  if (val.isInfinity()) {
    bool neg = val.isNegative();
    if (isa<Float32Type>(ty))
      return neg ? "-Inf32" : "Inf32";
    if (isa<Float16Type>(ty))
      return neg ? "-Inf16" : "Inf16";
    if (isa<BFloat16Type>(ty))
      return neg ? "BFloat16(-Inf32)" : "BFloat16(Inf32)";
    return neg ? "-Inf" : "Inf";
  }

  llvm::SmallString<32> buf;
  val.toString(buf, /*FormatPrecision=*/0, /*FormatMaxPadding=*/0,
               /*TruncateZero=*/false);
  std::string s(buf.str());
  bool hasDotOrExp = (s.find('.') != std::string::npos) ||
                     (s.find('e') != std::string::npos) ||
                     (s.find('E') != std::string::npos);
  if (!hasDotOrExp)
    s += ".0";

  if (isa<Float32Type>(ty)) {
    size_t expPos = s.find_first_of("eE");
    if (expPos != std::string::npos) {
      s[expPos] = 'f';
    } else {
      s += "f0";
    }
    return s;
  }
  if (isa<Float16Type>(ty))
    return "Float16(" + s + ")";
  if (isa<BFloat16Type>(ty))
    return "BFloat16(" + s + ")";
  size_t expPos = s.find('E');
  if (expPos != std::string::npos)
    s[expPos] = 'e';
  return s;
}

std::string JuliaEmitter::formatScalarAttr(Attribute attr, Type ty) {
  if (auto bAttr = dyn_cast<BoolAttr>(attr))
    return bAttr.getValue() ? "true" : "false";

  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    APInt v = iAttr.getValue();
    if (ty.isInteger(1))
      return v.getBoolValue() ? "true" : "false";
    if (ty.isIndex()) {
      int64_t sv = v.getSExtValue();
      if (sv == INT64_MIN)
        return "typemin(Int)";
      return std::to_string(sv);
    }
    auto intTy = dyn_cast<IntegerType>(ty);
    if (intTy && intTy.isUnsigned()) {
      llvm::SmallString<32> str;
      v.toStringUnsigned(str);
      return formatScalarType(ty) + "(" + std::string(str.str()) + ")";
    }
    if (intTy) {
      if (v.isMinSignedValue())
        return "typemin(" + formatScalarType(ty) + ")";
      llvm::SmallString<32> str;
      v.toStringSigned(str);
      if (intTy.getWidth() == 64)
        return std::string(str.str());
      return formatScalarType(ty) + "(" + std::string(str.str()) + ")";
    }
  }

  if (auto fAttr = dyn_cast<FloatAttr>(attr))
    return formatFloatLiteral(fAttr.getValue(), ty);

  if (auto arrAttr = dyn_cast<ArrayAttr>(attr)) {
    if (auto cplxTy = dyn_cast<ComplexType>(ty);
        cplxTy && arrAttr.size() == 2) {
      Type elemTy = cplxTy.getElementType();
      return formatScalarType(ty) + "(" +
             formatScalarAttr(arrAttr[0], elemTy) + ", " +
             formatScalarAttr(arrAttr[1], elemTy) + ")";
    }
  }
  return "zero(" + formatScalarType(ty) + ")";
}

std::string JuliaEmitter::formatShapeArgs(ArrayRef<int64_t> shape,
                                          ValueRange dynamicDims) const {
  std::string out;
  llvm::raw_string_ostream ss(out);
  unsigned dynIdx = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      ss << ", ";
    if (ShapedType::isDynamic(shape[i])) {
      assert(dynIdx < dynamicDims.size());
      ss << lookup(dynamicDims[dynIdx++]);
    } else {
      ss << shape[i];
    }
  }
  return ss.str();
}

int JuliaEmitter::exprDepth(Value v, int limit) const {
  if (limit <= 0)
    return 1;
  Operation *def = v.getDefiningOp();
  if (!def || isa<ConstantOp>(def))
    return 0;
  auto it = valueExprs.find(v);
  if (it != valueExprs.end()) {
    // If already materialized as a short identifier, depth is 0; if inlined,
    // estimate from operands.
    bool looksAtomic = llvm::all_of(it->second, [](char c) {
      return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    });
    if (looksAtomic)
      return 0;
  }
  int best = 0;
  for (Value opnd : def->getOperands())
    best = std::max(best, exprDepth(opnd, limit - 1));
  return best + 1;
}

bool JuliaEmitter::shouldInlinePureOp(Operation *op) const {
  if (isa<ConstantOp>(op))
    return true;
  if (op->getNumResults() != 1)
    return false;
  Value res = op->getResult(0);
  if (!res.hasOneUse())
    return false;
  Operation *user = *res.user_begin();
  if (user->getBlock() != op->getBlock())
    return false;

  if (isa<ViewSliceOp, ViewDynamicSliceOp, ViewReshapeOp>(op)) {
    // Only inline views when directly consumed by an op that doesn't index
    // the view multiple times inside a loop.
    return isa<MatmulInPlaceOp, MatmulOp, CopyToOp, CopyOp, MaterializeArrayOp,
               TriangularSolveInPlaceOp, TriangularSolveOp>(user);
  }

  if (isa<LoadOp>(op)) {
    if (!isa_and_nonnull<KernelOp, KernelInPlaceOp>(op->getParentOp()))
      return false;
    return exprDepth(res) <= 5;
  }

  if (isa<UnaryOp, BinaryOp, CmpOp, SelectOp, ClampOp, CastOp, BitcastOp,
          MathLibmOp, SizeOp>(op)) {
    return exprDepth(res) <= 5;
  }
  return false;
}

std::string JuliaEmitter::buildCastExpr(Value input, Type dstTy) {
  Type srcTy = input.getType();
  std::string x = lookup(input);
  if (srcTy == dstTy)
    return x;

  std::string dstName = formatScalarType(dstTy);
  if (dstTy.isInteger(1)) {
    if (isa<ComplexType>(srcTy))
      return "((real(" + x + ") != 0) | (imag(" + x + ") != 0))";
    return "(" + x + " != zero(" + formatScalarType(srcTy) + "))";
  }
  if (srcTy.isInteger(1)) {
    return "ifelse(" + x + ", one(" + dstName + "), zero(" + dstName + "))";
  }

  bool srcIsInt = isa<IntegerType>(srcTy) || srcTy.isIndex();
  bool dstIsInt = isa<IntegerType>(dstTy) || dstTy.isIndex();
  if (isa<FloatType>(srcTy) && dstIsInt) {
    return "unsafe_trunc(" + dstName + ", " + x + ")";
  }
  if (isa<ComplexType>(srcTy) && !isa<ComplexType>(dstTy)) {
    std::string re = "real(" + x + ")";
    if (dstIsInt)
      return "unsafe_trunc(" + dstName + ", " + re + ")";
    return dstName + "(" + re + ")";
  }
  if (srcIsInt && dstIsInt) {
    unsigned srcW = srcTy.isIndex() ? 64 : cast<IntegerType>(srcTy).getWidth();
    unsigned dstW = dstTy.isIndex() ? 64 : cast<IntegerType>(dstTy).getWidth();
    bool srcU = !srcTy.isIndex() && cast<IntegerType>(srcTy).isUnsigned();
    bool dstU = !dstTy.isIndex() && cast<IntegerType>(dstTy).isUnsigned();
    if (dstW <= srcW) {
      return "(" + x + " % " + dstName + ")";
    }
    // Widening: if signed -> unsigned (e.g. Int8(-1) -> UInt32), sign-extend
    // to signed integer of dstW first, then reinterpret via `% DstUInt`.
    if (!srcU && dstU) {
      std::string signedDst =
          "Int" + std::to_string(dstW == 128 ? 128 : dstW);
      return "(" + signedDst + "(" + x + ") % " + dstName + ")";
    }
    return dstName + "(" + x + ")";
  }
  return dstName + "(" + x + ")";
}

std::string JuliaEmitter::buildPureScalarExpr(Operation *op) {
  if (auto cOp = dyn_cast<ConstantOp>(op))
    return formatScalarAttr(cOp.getValue(), cOp.getType());

  if (auto szOp = dyn_cast<SizeOp>(op)) {
    return "size(" + lookup(szOp.getSource()) + ", " +
           std::to_string(szOp.getDimension() + 1) + ")";
  }

  if (auto loadOp = dyn_cast<LoadOp>(op)) {
    std::string out = lookup(loadOp.getSource()) + "[";
    llvm::raw_string_ostream ss(out);
    llvm::interleaveComma(loadOp.getIndices(), ss,
                          [&](Value idx) { ss << lookup(idx); });
    ss << "]";
    return ss.str();
  }

  if (auto uOp = dyn_cast<UnaryOp>(op)) {
    std::string x = lookup(uOp.getInput());
    Type inTy = uOp.getInput().getType();
    Type outTy = uOp.getType();
    switch (uOp.getOp()) {
    case UnaryOpKind::neg:
      return "(-" + x + ")";
    case UnaryOpKind::not_op:
      return inTy.isInteger(1) ? "(!" + x + ")" : "(~" + x + ")";
    case UnaryOpKind::abs:
      return "abs(" + x + ")";
    case UnaryOpKind::exp:
      return "exp(" + x + ")";
    case UnaryOpKind::expm1:
      return "expm1(" + x + ")";
    case UnaryOpKind::exp2:
      return "exp2(" + x + ")";
    case UnaryOpKind::log:
      return "log(" + x + ")";
    case UnaryOpKind::log1p:
      return "log1p(" + x + ")";
    case UnaryOpKind::log2:
      return "log2(" + x + ")";
    case UnaryOpKind::sqrt:
      return "sqrt(" + x + ")";
    case UnaryOpKind::rsqrt:
      return "inv(sqrt(" + x + "))";
    case UnaryOpKind::cbrt:
      return "cbrt(" + x + ")";
    case UnaryOpKind::sin:
      return "sin(" + x + ")";
    case UnaryOpKind::cos:
      return "cos(" + x + ")";
    case UnaryOpKind::tan:
      return "tan(" + x + ")";
    case UnaryOpKind::asin:
      return "asin(" + x + ")";
    case UnaryOpKind::acos:
      return "acos(" + x + ")";
    case UnaryOpKind::atan:
      return "atan(" + x + ")";
    case UnaryOpKind::sinh:
      return "sinh(" + x + ")";
    case UnaryOpKind::cosh:
      return "cosh(" + x + ")";
    case UnaryOpKind::tanh:
      return "tanh(" + x + ")";
    case UnaryOpKind::asinh:
      return "asinh(" + x + ")";
    case UnaryOpKind::acosh:
      return "acosh(" + x + ")";
    case UnaryOpKind::atanh:
      return "atanh(" + x + ")";
    case UnaryOpKind::floor:
      return "floor(" + x + ")";
    case UnaryOpKind::ceil:
      return "ceil(" + x + ")";
    case UnaryOpKind::round_nearest_even:
      return "round(" + x + ", RoundNearest)";
    case UnaryOpKind::round_nearest_afz:
      return "round(" + x + ", RoundNearestTiesAway)";
    case UnaryOpKind::sign:
      return "sign(" + x + ")";
    case UnaryOpKind::is_finite:
      return "isfinite(" + x + ")";
    case UnaryOpKind::logistic:
      return "inv(one(" + formatScalarType(outTy) + ") + exp(-" + x + "))";
    case UnaryOpKind::real:
      return "real(" + x + ")";
    case UnaryOpKind::imag:
      return "imag(" + x + ")";
    case UnaryOpKind::conj:
      return "conj(" + x + ")";
    case UnaryOpKind::leading_zeros:
      return "(leading_zeros(" + x + ") % " + formatScalarType(outTy) + ")";
    case UnaryOpKind::popcnt:
      return "(count_ones(" + x + ") % " + formatScalarType(outTy) + ")";
    }
  }

  if (auto bOp = dyn_cast<BinaryOp>(op)) {
    std::string a = lookup(bOp.getLhs());
    std::string b = lookup(bOp.getRhs());
    Type ty = bOp.getType();
    bool isBool = ty.isInteger(1);
    bool isInt = (isa<IntegerType>(ty) && !isBool) || ty.isIndex();
    switch (bOp.getOp()) {
    case BinaryOpKind::add:
      return isBool ? "(" + a + " | " + b + ")" : "(" + a + " + " + b + ")";
    case BinaryOpKind::sub:
      return isBool ? "(" + a + " != " + b + ")" : "(" + a + " - " + b + ")";
    case BinaryOpKind::mul:
      return isBool ? "(" + a + " & " + b + ")" : "(" + a + " * " + b + ")";
    case BinaryOpKind::div:
      return isInt ? "div(" + a + ", " + b + ")" : "(" + a + " / " + b + ")";
    case BinaryOpKind::rem:
      return "rem(" + a + ", " + b + ")";
    case BinaryOpKind::pow: {
      if (isInt) {
        auto intTy = dyn_cast<IntegerType>(ty);
        if (!intTy || !intTy.isUnsigned()) {
          std::string tName = formatScalarType(ty);
          return "ifelse(" + b + " < zero(" + tName + "), ifelse(" + a +
                 " == one(" + tName + "), one(" + tName + "), ifelse(" + a +
                 " == -one(" + tName + "), ifelse((" + b + " & one(" + tName +
                 ")) == zero(" + tName + "), one(" + tName + "), -one(" +
                 tName + ")), zero(" + tName +
                 "))), Base.power_by_squaring(" + a + ", " + b + "))";
        }
      }
      return "(" + a + " ^ " + b + ")";
    }
    case BinaryOpKind::atan2:
      return "atan(" + a + ", " + b + ")";
    case BinaryOpKind::max:
      return "max(" + a + ", " + b + ")";
    case BinaryOpKind::min:
      return "min(" + a + ", " + b + ")";
    case BinaryOpKind::and_op:
      return "(" + a + " & " + b + ")";
    case BinaryOpKind::or_op:
      return "(" + a + " | " + b + ")";
    case BinaryOpKind::xor_op:
      return "xor(" + a + ", " + b + ")";
    case BinaryOpKind::shl:
      return "(" + a + " << " + b + ")";
    case BinaryOpKind::shr_arith:
      return "(" + a + " >> " + b + ")";
    case BinaryOpKind::shr_logical:
      return "(" + a + " >>> " + b + ")";
    case BinaryOpKind::complex_make:
      return "complex(" + a + ", " + b + ")";
    }
  }

  if (auto cmp = dyn_cast<CmpOp>(op)) {
    std::string a = lookup(cmp.getLhs());
    std::string b = lookup(cmp.getRhs());
    switch (cmp.getPredicate()) {
    case CmpPred::eq:
      return "(" + a + " == " + b + ")";
    case CmpPred::ne:
      return "(" + a + " != " + b + ")";
    case CmpPred::lt:
      return "(" + a + " < " + b + ")";
    case CmpPred::le:
      return "(" + a + " <= " + b + ")";
    case CmpPred::gt:
      return "(" + a + " > " + b + ")";
    case CmpPred::ge:
      return "(" + a + " >= " + b + ")";
    }
  }

  if (auto sel = dyn_cast<SelectOp>(op)) {
    return "ifelse(" + lookup(sel.getPred()) + ", " +
           lookup(sel.getOnTrue()) + ", " + lookup(sel.getOnFalse()) + ")";
  }

  if (auto cl = dyn_cast<ClampOp>(op)) {
    return "clamp(" + lookup(cl.getOperand()) + ", " + lookup(cl.getMin()) +
           ", " + lookup(cl.getMax()) + ")";
  }

  if (auto castOp = dyn_cast<CastOp>(op))
    return buildCastExpr(castOp.getInput(), castOp.getType());

  if (auto bc = dyn_cast<BitcastOp>(op)) {
    return "reinterpret(" + formatScalarType(bc.getType()) + ", " +
           lookup(bc.getInput()) + ")";
  }

  if (auto libm = dyn_cast<MathLibmOp>(op)) {
    std::string x = lookup(libm.getInput());
    Type ty = libm.getType();
    bool isF64 = isa<Float64Type>(ty);
    const char *sym = "erf";
    switch (libm.getFn()) {
    case LibmFn::erf:
      sym = isF64 ? "erf" : "erff";
      break;
    case LibmFn::erfc:
      sym = isF64 ? "erfc" : "erfcf";
      break;
    case LibmFn::lgamma:
      sym = isF64 ? "lgamma" : "lgammaf";
      break;
    case LibmFn::tgamma:
      sym = isF64 ? "tgamma" : "tgammaf";
      break;
    }
    if (isF64) {
      return std::string("ccall((:") + sym +
             ", Base.Math.libm), Float64, (Float64,), " + x + ")";
    }
    if (isa<Float32Type>(ty)) {
      return std::string("ccall((:") + sym +
             ", Base.Math.libm), Float32, (Float32,), " + x + ")";
    }
    return formatScalarType(ty) + "(ccall((:" + sym +
           ", Base.Math.libm), Float32, (Float32,), Float32(" + x + ")))";
  }

  return "";
}

std::string JuliaEmitter::buildViewSliceExpr(Value src,
                                             ArrayRef<int64_t> starts,
                                             ArrayRef<int64_t> limits,
                                             ArrayRef<int64_t> strides) {
  auto srcTy = cast<ShapedType>(src.getType());
  std::string out = "@view(" + lookup(src) + "[";
  llvm::raw_string_ostream ss(out);
  for (size_t d = 0; d < starts.size(); ++d) {
    if (d > 0)
      ss << ", ";
    int64_t s = starts[d];
    int64_t l = limits[d];
    int64_t st = strides[d];
    if (st == 1 && s == 0 && !srcTy.isDynamicDim(d) &&
        l == srcTy.getDimSize(d)) {
      ss << ":";
    } else if (st == 1) {
      // 0-based [s, l) -> 1-based (s + 1):l
      ss << (s + 1) << ":" << l;
    } else {
      // Last included 0-based index is s + ((l - 1 - s) / st) * st;
      // 1-based upper bound (s + 1) + ((l - 1 - s) / st) * st = l when exact,
      // and `s+1 : st : l` in Julia stops at <= l!
      ss << (s + 1) << ":" << st << ":" << l;
    }
  }
  ss << "])";
  return ss.str();
}

std::string JuliaEmitter::buildViewDynamicSliceExpr(
    Value src, ValueRange starts, ArrayRef<int64_t> sizes) {
  std::string srcName = lookup(src);
  std::string out = "@view(" + srcName + "[";
  llvm::raw_string_ostream ss(out);
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (d > 0)
      ss << ", ";
    int64_t len = sizes[d];
    std::string rawIdx = "Int(" + lookup(starts[d]) + ")";
    std::string clamped = "clamp(" + rawIdx + ", 0, size(" + srcName + ", " +
                          std::to_string(d + 1) + ") - " +
                          std::to_string(len) + ") + 1";
    ss << "(" << clamped << "):((" << clamped << ") + " << (len - 1) << ")";
  }
  ss << "])";
  return ss.str();
}

std::string JuliaEmitter::buildViewReshapeExpr(Value src, ShapedType resTy,
                                               ValueRange dynamicDims) {
  std::string shapeArgs = formatShapeArgs(resTy.getShape(), dynamicDims);
  if (shapeArgs.empty())
    return "reshape(" + lookup(src) + ")";
  return "reshape(" + lookup(src) + ", " + shapeArgs + ")";
}

LogicalResult JuliaEmitter::emitKernelLoop(ValueRange outs, Region &body) {
  if (outs.empty())
    return success();
  Block &blk = body.front();
  auto firstTy = cast<ShapedType>(outs[0].getType());
  int64_t rank = firstTy.getRank();
  if (static_cast<int64_t>(blk.getNumArguments()) != rank)
    return failure();

  // Check if all outputs and loaded inputs are owned contiguous `!julia.array`s
  // of identical shape with identity `[%iv0, ..., %iv_{R-1}]` indexing and no
  // other uses of `%iv_d`. If so, emit a flat linear `eachindex` SIMD loop!
  bool canUseLinearIndexing = (rank >= 1) && llvm::all_of(outs, [&](Value o) {
    return isa<ArrayType>(o.getType()) &&
           cast<ShapedType>(o.getType()).getShape() == firstTy.getShape();
  });

  if (canUseLinearIndexing) {
    for (unsigned d = 0; d < blk.getNumArguments(); ++d) {
      BlockArgument iv = blk.getArgument(d);
      for (OpOperand &use : iv.getUses()) {
        auto loadOp = dyn_cast<LoadOp>(use.getOwner());
        if (!loadOp || use.getOperandNumber() != d + 1) {
          canUseLinearIndexing = false;
          break;
        }
      }
      if (!canUseLinearIndexing)
        break;
    }
  }
  if (canUseLinearIndexing) {
    blk.walk([&](LoadOp loadOp) {
      Value src = loadOp.getSource();
      if (!isa<ArrayType>(src.getType()) ||
          cast<ShapedType>(src.getType()).getShape() != firstTy.getShape() ||
          static_cast<int64_t>(loadOp.getIndices().size()) != rank) {
        canUseLinearIndexing = false;
        return;
      }
      for (int64_t d = 0; d < rank; ++d) {
        if (loadOp.getIndices()[d] != blk.getArgument(d)) {
          canUseLinearIndexing = false;
          return;
        }
      }
    });
  }

  // Check if multiple outputs exist and any output buffer also appears as a
  // loaded source in the body; if so, compute RHS into temporaries before
  // storing so multi-output in-place reuse never clobbers a sibling load.
  bool stageOutputs = false;
  if (outs.size() > 1) {
    llvm::DenseSet<Value> outSet(outs.begin(), outs.end());
    blk.walk([&](LoadOp loadOp) {
      if (outSet.contains(loadOp.getSource()))
        stageOutputs = true;
    });
  }

  std::string primaryOut = lookup(outs[0]);

  if (canUseLinearIndexing) {
    std::string linIdx = freshName("idx");
    indent() << "@inbounds @simd for " << linIdx << " in eachindex("
             << primaryOut << ")\n";
    indentLevel++;

    // Map all block args temporarily and intercept LoadOp linearly.
    for (BlockArgument arg : blk.getArguments())
      valueExprs[arg] = linIdx;

    for (Operation &op : blk.without_terminator()) {
      if (auto loadOp = dyn_cast<LoadOp>(&op)) {
        std::string expr = lookup(loadOp.getSource()) + "[" + linIdx + "]";
        if (shouldInlinePureOp(&op)) {
          valueExprs[loadOp.getResult()] = expr;
        } else {
          std::string vName = freshName("t");
          indent() << vName << " = " << expr << "\n";
          valueExprs[loadOp.getResult()] = vName;
        }
        continue;
      }
      if (failed(emitOperation(&op)))
        return failure();
    }

    auto yieldOp = cast<YieldOp>(blk.getTerminator());
    if (stageOutputs) {
      SmallVector<std::string> staged;
      for (Value v : yieldOp.getOperands()) {
        std::string tmp = freshName("out");
        indent() << tmp << " = " << lookup(v) << "\n";
        staged.push_back(tmp);
      }
      for (auto [o, tmp] : llvm::zip(outs, staged))
        indent() << lookup(o) << "[" << linIdx << "] = " << tmp << "\n";
    } else {
      for (auto [o, v] : llvm::zip(outs, yieldOp.getOperands()))
        indent() << lookup(o) << "[" << linIdx << "] = " << lookup(v) << "\n";
    }

    indentLevel--;
    indent() << "end\n";
    return success();
  }

  // General N-D column-major loop nest: outermost d = R-1 .. innermost d = 0.
  SmallVector<std::string> ivNames(rank);
  for (int64_t d = 0; d < rank; ++d) {
    ivNames[d] = freshName("i");
    valueExprs[blk.getArgument(d)] = ivNames[d];
  }

  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string bound = firstTy.isDynamicDim(d)
                            ? ("size(" + primaryOut + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(firstTy.getDimSize(d));
    indent();
    if (d == rank - 1)
      os << "@inbounds ";
    if (d == 0)
      os << "@simd ";
    os << "for " << ivNames[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(blk, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);

  std::string idxTuple;
  {
    llvm::raw_string_ostream ss(idxTuple);
    llvm::interleaveComma(ivNames, ss);
  }

  if (stageOutputs) {
    SmallVector<std::string> staged;
    for (Value v : yieldOp.getOperands()) {
      std::string tmp = freshName("out");
      indent() << tmp << " = " << lookup(v) << "\n";
      staged.push_back(tmp);
    }
    for (auto [o, tmp] : llvm::zip(outs, staged))
      indent() << lookup(o) << "[" << idxTuple << "] = " << tmp << "\n";
  } else {
    for (auto [o, v] : llvm::zip(outs, yieldOp.getOperands()))
      indent() << lookup(o) << "[" << idxTuple << "] = " << lookup(v) << "\n";
  }

  for (int64_t d = 0; d < rank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

static std::string wrapMatrixTranspose(const std::string &matExpr,
                                       TransposeMode mode) {
  switch (mode) {
  case TransposeMode::none:
    return matExpr;
  case TransposeMode::transpose:
    return "LinearAlgebra.Transpose(" + matExpr + ")";
  case TransposeMode::adjoint:
    return "LinearAlgebra.Adjoint(" + matExpr + ")";
  }
}

static char blasTransposeChar(TransposeMode mode) {
  switch (mode) {
  case TransposeMode::none:
    return 'N';
  case TransposeMode::transpose:
    return 'T';
  case TransposeMode::adjoint:
    return 'C';
  }
}

static bool isBlasFloatOrComplexType(Type elemTy) {
  if (elemTy.isF32() || elemTy.isF64())
    return true;
  if (auto cplx = dyn_cast<ComplexType>(elemTy)) {
    Type inner = cplx.getElementType();
    return inner.isF32() || inner.isF64();
  }
  return false;
}

static bool hasUnitColumnStride2D(Value v) {
  if (isa<ArrayType>(v.getType()))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto vs = dyn_cast<ViewSliceOp>(def)) {
    ArrayRef<int64_t> strides = vs.getStrides();
    return strides.size() == 2 && strides[0] == 1 && strides[1] == 1 &&
           hasUnitColumnStride2D(vs.getSource());
  }
  if (auto vr = dyn_cast<ViewReshapeOp>(def))
    return isa<ArrayType>(vr.getSource().getType());
  if (auto vds = dyn_cast<ViewDynamicSliceOp>(def))
    return vds.getSliceSizes().size() == 2 &&
           hasUnitColumnStride2D(vds.getSource());
  return false;
}

LogicalResult JuliaEmitter::emitMatmul(
    Value dst, Value lhs, Value rhs, Value alpha, Value beta,
    TransposeMode lhsTrans, TransposeMode rhsTrans, ArrayRef<int64_t> batchLhs,
    ArrayRef<int64_t> batchRhs, ArrayRef<int64_t> batchOut) {
  std::string dstExpr = lookup(dst);
  std::string lhsExpr = lookup(lhs);
  std::string rhsExpr = lookup(rhs);
  auto lhsTy = cast<ShapedType>(lhs.getType());
  auto rhsTy = cast<ShapedType>(rhs.getType());
  auto dstTy = cast<ShapedType>(dst.getType());

  auto emitSingleSliceMul = [&](const std::string &dStr,
                                const std::string &aStr,
                                const std::string &bStr, int64_t aEffectiveRank,
                                int64_t bEffectiveRank) {
    // Emit direct `LinearAlgebra.BLAS.gemm!` for 5-arg 2D BLAS matrices with
    // unit column stride, avoiding Julia's 16-byte `MulAddMul` dispatch box in
    // 5-arg `LinearAlgebra.mul!`.
    if ((alpha || beta) && batchOut.empty() && aEffectiveRank == 2 &&
        bEffectiveRank == 2 &&
        isBlasFloatOrComplexType(dstTy.getElementType()) &&
        hasUnitColumnStride2D(dst) && hasUnitColumnStride2D(lhs) &&
        hasUnitColumnStride2D(rhs)) {
      std::string elemTyName = formatScalarType(dstTy.getElementType());
      std::string aVal =
          alpha ? lookup(alpha) : ("one(" + elemTyName + ")");
      std::string bVal =
          beta ? lookup(beta) : ("zero(" + elemTyName + ")");
      indent() << "LinearAlgebra.BLAS.gemm!('" << blasTransposeChar(lhsTrans)
               << "', '" << blasTransposeChar(rhsTrans) << "', " << aVal << ", "
               << aStr << ", " << bStr << ", " << bVal << ", " << dStr << ")\n";
      return;
    }

    std::string wrappedA =
        (aEffectiveRank == 2) ? wrapMatrixTranspose(aStr, lhsTrans) : aStr;
    std::string wrappedB;
    if (bEffectiveRank == 2) {
      wrappedB = wrapMatrixTranspose(bStr, rhsTrans);
    } else {
      wrappedB = bStr;
    }

    // In Julia, `(1, K) * (K, N)` or `(K,)ᵀ * (K, N) -> (N,)` vector-matrix:
    // If `aEffectiveRank == 1 && bEffectiveRank == 2` and output is 1D `(N,)`:
    // `(Bᵀ * a)` writes directly into 1D `dStr::AbstractVector` via `mul!`!
    if (aEffectiveRank == 1 && bEffectiveRank == 2) {
      TransposeMode flipped = (rhsTrans == TransposeMode::none)
                                  ? TransposeMode::transpose
                                  : TransposeMode::none;
      wrappedA = wrapMatrixTranspose(bStr, flipped);
      wrappedB = aStr;
    }

    indent() << "LinearAlgebra.mul!(" << dStr << ", " << wrappedA << ", "
             << wrappedB;
    if (alpha || beta) {
      std::string elemTyName = formatScalarType(dstTy.getElementType());
      std::string aVal =
          alpha ? lookup(alpha) : ("one(" + elemTyName + ")");
      std::string bVal =
          beta ? lookup(beta) : ("zero(" + elemTyName + ")");
      os << ", " << aVal << ", " << bVal;
    }
    os << ")\n";
  };

  if (batchOut.empty()) {
    emitSingleSliceMul(dstExpr, lhsExpr, rhsExpr, lhsTy.getRank(),
                       rhsTy.getRank());
    return success();
  }

  // Batched matmul: emit column-major loop over batch dimensions and `@view`
  // each slice!
  size_t numBatch = batchOut.size();
  SmallVector<std::string> bVars(numBatch);
  for (size_t b = 0; b < numBatch; ++b)
    bVars[b] = freshName("b");

  for (int64_t b = static_cast<int64_t>(numBatch) - 1; b >= 0; --b) {
    int64_t outDim = batchOut[b];
    std::string bound = dstTy.isDynamicDim(outDim)
                            ? ("size(" + dstExpr + ", " +
                               std::to_string(outDim + 1) + ")")
                            : std::to_string(dstTy.getDimSize(outDim));
    indent();
    if (b == static_cast<int64_t>(numBatch) - 1)
      os << "@inbounds ";
    os << "for " << bVars[b] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  auto buildBatchView = [&](const std::string &arrName, int64_t rank,
                            ArrayRef<int64_t> bDims) {
    llvm::DenseMap<int64_t, std::string> dimToVar;
    for (size_t i = 0; i < bDims.size(); ++i)
      dimToVar[bDims[i]] = bVars[i];
    std::string v = "@view(" + arrName + "[";
    llvm::raw_string_ostream ss(v);
    for (int64_t d = 0; d < rank; ++d) {
      if (d > 0)
        ss << ", ";
      auto it = dimToVar.find(d);
      if (it != dimToVar.end())
        ss << it->second;
      else
        ss << ":";
    }
    ss << "])";
    return ss.str();
  };

  std::string dSlice = buildBatchView(dstExpr, dstTy.getRank(), batchOut);
  std::string aSlice = buildBatchView(lhsExpr, lhsTy.getRank(), batchLhs);
  std::string bSlice = buildBatchView(rhsExpr, rhsTy.getRank(), batchRhs);

  emitSingleSliceMul(
      dSlice, aSlice, bSlice,
      lhsTy.getRank() - static_cast<int64_t>(batchLhs.size()),
      rhsTy.getRank() - static_cast<int64_t>(batchRhs.size()));

  for (size_t b = 0; b < numBatch; ++b) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitDotGeneral(
    Value dst, Value lhs, Value rhs, ArrayRef<int64_t> lhsBatch,
    ArrayRef<int64_t> rhsBatch, ArrayRef<int64_t> lhsContract,
    ArrayRef<int64_t> rhsContract, ArrayRef<int64_t> resultDimOrder) {
  // General contraction:
  // Canonical logical dim list = [batch..., lhsFree..., rhsFree...].
  // `resultDimOrder[d]` gives which canonical dim index corresponds to output
  // Julia dim `d` (`0 .. outRank - 1`).
  auto lhsTy = cast<ShapedType>(lhs.getType());
  auto rhsTy = cast<ShapedType>(rhs.getType());
  auto dstShapeTy = dyn_cast<ShapedType>(dst.getType());
  Type elemTy = dstShapeTy ? dstShapeTy.getElementType() : dst.getType();
  std::string dstName = dstShapeTy ? lookup(dst) : "";
  std::string lhsName = lookup(lhs);
  std::string rhsName = lookup(rhs);

  llvm::DenseSet<int64_t> lhsBatchSet(lhsBatch.begin(), lhsBatch.end());
  llvm::DenseSet<int64_t> lhsContractSet(lhsContract.begin(),
                                         lhsContract.end());
  llvm::DenseSet<int64_t> rhsBatchSet(rhsBatch.begin(), rhsBatch.end());
  llvm::DenseSet<int64_t> rhsContractSet(rhsContract.begin(),
                                         rhsContract.end());

  SmallVector<int64_t> lhsFree, rhsFree;
  for (int64_t d = 0; d < lhsTy.getRank(); ++d)
    if (!lhsBatchSet.contains(d) && !lhsContractSet.contains(d))
      lhsFree.push_back(d);
  for (int64_t d = 0; d < rhsTy.getRank(); ++d)
    if (!rhsBatchSet.contains(d) && !rhsContractSet.contains(d))
      rhsFree.push_back(d);

  int64_t outRank = dstShapeTy ? dstShapeTy.getRank() : 0;
  SmallVector<std::string> outVars(outRank);
  for (int64_t d = 0; d < outRank; ++d)
    outVars[d] = freshName("i");

  SmallVector<std::string> canonVars(lhsBatch.size() + lhsFree.size() +
                                     rhsFree.size());
  for (int64_t d = 0; d < outRank; ++d)
    canonVars[resultDimOrder[d]] = outVars[d];

  SmallVector<std::string> kVars(lhsContract.size());
  for (size_t k = 0; k < lhsContract.size(); ++k)
    kVars[k] = freshName("k");

  for (int64_t d = outRank - 1; d >= 0; --d) {
    std::string bound = dstShapeTy.isDynamicDim(d)
                            ? ("size(" + dstName + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(dstShapeTy.getDimSize(d));
    indent();
    if (d == outRank - 1)
      os << "@inbounds ";
    os << "for " << outVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string acc = freshName("acc");
  indent() << acc << " = zero(" << formatScalarType(elemTy) << ")\n";

  for (int64_t k = static_cast<int64_t>(lhsContract.size()) - 1; k >= 0; --k) {
    int64_t lDim = lhsContract[k];
    std::string bound = lhsTy.isDynamicDim(lDim)
                            ? ("size(" + lhsName + ", " +
                               std::to_string(lDim + 1) + ")")
                            : std::to_string(lhsTy.getDimSize(lDim));
    indent();
    if (outRank == 0 && k == static_cast<int64_t>(lhsContract.size()) - 1)
      os << "@inbounds ";
    if (k == 0)
      os << "@simd ";
    os << "for " << kVars[k] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  SmallVector<std::string> lhsIdx(lhsTy.getRank()), rhsIdx(rhsTy.getRank());
  for (size_t i = 0; i < lhsBatch.size(); ++i) {
    lhsIdx[lhsBatch[i]] = canonVars[i];
    rhsIdx[rhsBatch[i]] = canonVars[i];
  }
  for (size_t i = 0; i < lhsFree.size(); ++i)
    lhsIdx[lhsFree[i]] = canonVars[lhsBatch.size() + i];
  for (size_t i = 0; i < rhsFree.size(); ++i)
    rhsIdx[rhsFree[i]] = canonVars[lhsBatch.size() + lhsFree.size() + i];
  for (size_t k = 0; k < lhsContract.size(); ++k) {
    lhsIdx[lhsContract[k]] = kVars[k];
    rhsIdx[rhsContract[k]] = kVars[k];
  }

  auto joinIdx = [](ArrayRef<std::string> v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    llvm::interleaveComma(v, ss);
    return ss.str();
  };

  indent() << acc << " += " << lhsName << "[" << joinIdx(lhsIdx) << "] * "
           << rhsName << "[" << joinIdx(rhsIdx) << "]\n";

  for (size_t k = 0; k < lhsContract.size(); ++k) {
    indentLevel--;
    indent() << "end\n";
  }

  if (dstShapeTy) {
    indent() << dstName << "[" << joinIdx(outVars) << "] = " << acc << "\n";
  } else {
    valueExprs[dst] = acc;
  }

  for (int64_t d = 0; d < outRank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitReduceScalar(ReduceOp op) {
  size_t numVars = op.getInputs().size();
  SmallVector<std::string> accNames(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    accNames[i] = freshName("acc");
    indent() << accNames[i] << " = " << lookup(op.getInitValues()[i]) << "\n";
  }

  Value firstIn = op.getInputs()[0];
  auto inTy = dyn_cast<ShapedType>(firstIn.getType());
  if (!inTy) {
    // 0-D scalar input reduction: apply combiner once!
    Block &comb = op.getCombiner().front();
    for (size_t i = 0; i < numVars; ++i) {
      valueExprs[comb.getArgument(i)] = accNames[i];
      valueExprs[comb.getArgument(numVars + i)] = lookup(op.getInputs()[i]);
    }
    Operation *term = nullptr;
    if (failed(emitBlockStatements(comb, term)))
      return failure();
    auto yieldOp = cast<YieldOp>(term);
    for (size_t i = 0; i < numVars; ++i) {
      indent() << accNames[i] << " = " << lookup(yieldOp.getOperand(i)) << "\n";
      valueExprs[op.getResult(i)] = accNames[i];
    }
    return success();
  }

  int64_t rank = inTy.getRank();
  std::string primaryIn = lookup(firstIn);
  SmallVector<std::string> ivs(rank);
  for (int64_t d = 0; d < rank; ++d)
    ivs[d] = freshName("r");

  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string bound = inTy.isDynamicDim(d)
                            ? ("size(" + primaryIn + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(inTy.getDimSize(d));
    indent();
    if (d == rank - 1)
      os << "@inbounds ";
    os << "for " << ivs[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string idxTuple;
  {
    llvm::raw_string_ostream ss(idxTuple);
    llvm::interleaveComma(ivs, ss);
  }

  Block &comb = op.getCombiner().front();
  // Note: StableHLO reduce block arguments are `(lhs_0..lhs_{K-1}, rhs_0..rhs_{K-1})`
  // wait: let's verify StableHLO's exact block arg ordering! In StableHLO
  // `stablehlo.reduce(%in0, %in1 init: %init0, %init1)`, block args are
  // `(%arg0: tensor<T0>, %arg1: tensor<T1>, %arg2: tensor<T0>, %arg3: tensor<T1})`?
  // NO! Wait: let's check StableHLO spec in `LowerStableHLOToJulia.cpp` and
  // normalize `julia.reduce`'s combiner block args to ALWAYS be
  // `(acc_0, ..., acc_{K-1}, elem_0, ..., elem_{K-1})`!
  for (size_t i = 0; i < numVars; ++i) {
    valueExprs[comb.getArgument(i)] = accNames[i];
    std::string elemVar = freshName("elem");
    indent() << elemVar << " = " << lookup(op.getInputs()[i]) << "["
             << idxTuple << "]\n";
    valueExprs[comb.getArgument(numVars + i)] = elemVar;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(comb, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);
  SmallVector<std::string> nextAccs(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    nextAccs[i] = freshName("nacc");
    indent() << nextAccs[i] << " = " << lookup(yieldOp.getOperand(i)) << "\n";
  }
  for (size_t i = 0; i < numVars; ++i)
    indent() << accNames[i] << " = " << nextAccs[i] << "\n";

  for (int64_t d = 0; d < rank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }

  for (size_t i = 0; i < numVars; ++i)
    valueExprs[op.getResult(i)] = accNames[i];
  return success();
}

LogicalResult JuliaEmitter::emitReduceArray(
    ValueRange dsts, ValueRange inputs, ValueRange inits,
    ArrayRef<int64_t> reduceDims, Region &combiner) {
  size_t numVars = inputs.size();
  auto inTy = cast<ShapedType>(inputs[0].getType());
  auto outTy = cast<ShapedType>(dsts[0].getType());
  int64_t inRank = inTy.getRank();
  int64_t outRank = outTy.getRank();
  std::string primaryIn = lookup(inputs[0]);
  std::string primaryOut = lookup(dsts[0]);

  llvm::DenseSet<int64_t> redSet(reduceDims.begin(), reduceDims.end());
  SmallVector<int64_t> keptDims;
  for (int64_t d = 0; d < inRank; ++d)
    if (!redSet.contains(d))
      keptDims.push_back(d);

  // Initialize dsts with init values first so reduction order can iterate
  // column-major over kept & reduced dims cleanly!
  for (size_t i = 0; i < numVars; ++i) {
    indent() << "fill!(" << lookup(dsts[i]) << ", " << lookup(inits[i])
             << ")\n";
  }

  // Iterate over all input dims in column-major order (outer d = inRank-1 .. 0)
  // so input reads AND output updates walk memory column-major!
  SmallVector<std::string> inVars(inRank);
  for (int64_t d = 0; d < inRank; ++d)
    inVars[d] = freshName("i");

  for (int64_t d = inRank - 1; d >= 0; --d) {
    std::string bound = inTy.isDynamicDim(d)
                            ? ("size(" + primaryIn + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(inTy.getDimSize(d));
    indent();
    if (d == inRank - 1)
      os << "@inbounds ";
    os << "for " << inVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string inIdx, outIdx;
  {
    llvm::raw_string_ostream sIn(inIdx), sOut(outIdx);
    llvm::interleaveComma(inVars, sIn);
    for (int64_t k = 0; k < outRank; ++k) {
      if (k > 0)
        sOut << ", ";
      sOut << inVars[keptDims[k]];
    }
  }

  Block &comb = combiner.front();
  for (size_t i = 0; i < numVars; ++i) {
    std::string curAcc = freshName("acc");
    std::string curElem = freshName("elem");
    indent() << curAcc << " = " << lookup(dsts[i]) << "[" << outIdx << "]\n";
    indent() << curElem << " = " << lookup(inputs[i]) << "[" << inIdx << "]\n";
    valueExprs[comb.getArgument(i)] = curAcc;
    valueExprs[comb.getArgument(numVars + i)] = curElem;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(comb, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);
  SmallVector<std::string> nextAccs(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    nextAccs[i] = freshName("nacc");
    indent() << nextAccs[i] << " = " << lookup(yieldOp.getOperand(i)) << "\n";
  }
  for (size_t i = 0; i < numVars; ++i)
    indent() << lookup(dsts[i]) << "[" << outIdx << "] = " << nextAccs[i]
             << "\n";

  for (int64_t d = 0; d < inRank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitReduceWindow(
    ValueRange dsts, ValueRange inputs, ValueRange inits,
    ArrayRef<int64_t> winDims, ArrayRef<int64_t> winStrides,
    ArrayRef<int64_t> baseDil, ArrayRef<int64_t> winDil,
    ArrayRef<int64_t> padding, Region &combiner) {
  size_t numVars = inputs.size();
  auto inTy = cast<ShapedType>(inputs[0].getType());
  auto outTy = cast<ShapedType>(dsts[0].getType());
  int64_t rank = outTy.getRank();
  std::string primaryIn = lookup(inputs[0]);
  std::string primaryOut = lookup(dsts[0]);

  SmallVector<std::string> outVars(rank), winVars(rank);
  for (int64_t d = 0; d < rank; ++d) {
    outVars[d] = freshName("o");
    winVars[d] = freshName("w");
  }

  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string bound = outTy.isDynamicDim(d)
                            ? ("size(" + primaryOut + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(outTy.getDimSize(d));
    indent();
    if (d == rank - 1)
      os << "@inbounds ";
    os << "for " << outVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  SmallVector<std::string> accNames(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    accNames[i] = freshName("acc");
    indent() << accNames[i] << " = " << lookup(inits[i]) << "\n";
  }

  for (int64_t d = rank - 1; d >= 0; --d) {
    indent() << "for " << winVars[d] << " in 0:" << (winDims[d] - 1) << "\n";
    indentLevel++;
  }

  // Compute 0-based padded coordinate `p_d = (o_d - 1) * stride_d + w_d * win_dil_d - pad_low_d`
  // Valid iff `p_d >= 0 && p_d % base_dil_d == 0 && (p_d ÷ base_dil_d) < in_dim_d`.
  SmallVector<std::string> srcIndices(rank);
  SmallVector<std::string> boundsConds;
  for (int64_t d = 0; d < rank; ++d) {
    std::string pVar = freshName("p");
    int64_t padLow = padding[2 * d];
    indent() << pVar << " = (" << outVars[d] << " - 1) * " << winStrides[d]
             << " + " << winVars[d] << " * " << winDil[d] << " - (" << padLow
             << ")\n";
    std::string inDimExpr = inTy.isDynamicDim(d)
                                ? ("size(" + primaryIn + ", " +
                                   std::to_string(d + 1) + ")")
                                : std::to_string(inTy.getDimSize(d));
    if (baseDil[d] == 1) {
      boundsConds.push_back("(" + pVar + " >= 0 && " + pVar + " < " +
                            inDimExpr + ")");
      srcIndices[d] = "(" + pVar + " + 1)";
    } else {
      boundsConds.push_back("(" + pVar + " >= 0 && rem(" + pVar + ", " +
                            std::to_string(baseDil[d]) + ") == 0 && div(" +
                            pVar + ", " + std::to_string(baseDil[d]) + ") < " +
                            inDimExpr + ")");
      srcIndices[d] = "(div(" + pVar + ", " + std::to_string(baseDil[d]) +
                      ") + 1)";
    }
  }

  if (!boundsConds.empty()) {
    indent() << "if ";
    llvm::interleave(boundsConds, os, " && ");
    os << "\n";
    indentLevel++;
  }

  std::string inIdx;
  {
    llvm::raw_string_ostream ss(inIdx);
    llvm::interleaveComma(srcIndices, ss);
  }

  Block &comb = combiner.front();
  for (size_t i = 0; i < numVars; ++i) {
    std::string elemVar = freshName("elem");
    indent() << elemVar << " = " << lookup(inputs[i]) << "[" << inIdx << "]\n";
    valueExprs[comb.getArgument(i)] = accNames[i];
    valueExprs[comb.getArgument(numVars + i)] = elemVar;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(comb, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);
  SmallVector<std::string> nextAccs(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    nextAccs[i] = freshName("nacc");
    indent() << nextAccs[i] << " = " << lookup(yieldOp.getOperand(i)) << "\n";
  }
  for (size_t i = 0; i < numVars; ++i)
    indent() << accNames[i] << " = " << nextAccs[i] << "\n";

  if (!boundsConds.empty()) {
    indentLevel--;
    indent() << "end\n";
  }

  for (int64_t d = 0; d < rank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }

  std::string outIdx;
  {
    llvm::raw_string_ostream ss(outIdx);
    llvm::interleaveComma(outVars, ss);
  }
  for (size_t i = 0; i < numVars; ++i)
    indent() << lookup(dsts[i]) << "[" << outIdx << "] = " << accNames[i]
             << "\n";

  for (int64_t d = 0; d < rank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitPad(Value dst, Value src, Value padVal,
                                    ArrayRef<int64_t> low,
                                    ArrayRef<int64_t> high,
                                    ArrayRef<int64_t> interior) {
  auto srcTy = cast<ShapedType>(src.getType());
  auto dstTy = cast<ShapedType>(dst.getType());
  int64_t rank = srcTy.getRank();
  std::string dstName = lookup(dst);
  std::string srcName = lookup(src);

  indent() << "fill!(" << dstName << ", " << lookup(padVal) << ")\n";

  // Fast path: non-negative low & high padding -> single `copyto!` into a
  // strided `@view` of `dst`!
  bool nonNegEdges = llvm::all_of(low, [](int64_t v) { return v >= 0; }) &&
                     llvm::all_of(high, [](int64_t v) { return v >= 0; });
  if (nonNegEdges && srcTy.hasStaticShape()) {
    bool nonEmpty = llvm::all_of(srcTy.getShape(),
                                 [](int64_t s) { return s > 0; });
    if (!nonEmpty)
      return success();
    std::string viewStr = "@view(" + dstName + "[";
    llvm::raw_string_ostream ss(viewStr);
    for (int64_t d = 0; d < rank; ++d) {
      if (d > 0)
        ss << ", ";
      int64_t step = interior[d] + 1;
      int64_t start1 = low[d] + 1;
      int64_t end1 = low[d] + 1 + (srcTy.getDimSize(d) - 1) * step;
      if (step == 1)
        ss << start1 << ":" << end1;
      else
        ss << start1 << ":" << step << ":" << end1;
    }
    ss << "])";
    indent() << "@inbounds copyto!(" << ss.str() << ", " << srcName << ")\n";
    return success();
  }

  // General path supporting negative edge padding (cropping) and dynamic dims.
  SmallVector<std::string> ivs(rank), dIdxs(rank);
  for (int64_t d = 0; d < rank; ++d)
    ivs[d] = freshName("i");

  for (int64_t d = rank - 1; d >= 0; --d) {
    std::string sBound = srcTy.isDynamicDim(d)
                             ? ("size(" + srcName + ", " +
                                std::to_string(d + 1) + ")")
                             : std::to_string(srcTy.getDimSize(d));
    indent();
    if (d == rank - 1)
      os << "@inbounds ";
    os << "for " << ivs[d] << " in 1:" << sBound << "\n";
    indentLevel++;
  }

  SmallVector<std::string> conds;
  for (int64_t d = 0; d < rank; ++d) {
    std::string dj = freshName("di");
    int64_t step = interior[d] + 1;
    indent() << dj << " = (" << low[d] << ") + (" << ivs[d] << " - 1) * "
             << step << " + 1\n";
    std::string dBound = dstTy.isDynamicDim(d)
                             ? ("size(" + dstName + ", " +
                                std::to_string(d + 1) + ")")
                             : std::to_string(dstTy.getDimSize(d));
    conds.push_back("(" + dj + " >= 1 && " + dj + " <= " + dBound + ")");
    dIdxs[d] = dj;
  }

  if (!conds.empty()) {
    indent() << "if ";
    llvm::interleave(conds, os, " && ");
    os << "\n";
    indentLevel++;
  }

  std::string sTuple, dTuple;
  {
    llvm::raw_string_ostream s1(sTuple), s2(dTuple);
    llvm::interleaveComma(ivs, s1);
    llvm::interleaveComma(dIdxs, s2);
  }
  indent() << dstName << "[" << dTuple << "] = " << srcName << "[" << sTuple
           << "]\n";

  if (!conds.empty()) {
    indentLevel--;
    indent() << "end\n";
  }

  for (int64_t d = 0; d < rank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitDynamicSliceCopy(Value dst, Value src,
                                                 ValueRange starts,
                                                 ArrayRef<int64_t> sizes) {
  std::string viewExpr = buildViewDynamicSliceExpr(src, starts, sizes);
  indent() << "@inbounds copyto!(" << lookup(dst) << ", " << viewExpr << ")\n";
  return success();
}

LogicalResult JuliaEmitter::emitDynamicUpdateSlice(Value dst, Value update,
                                                   ValueRange starts) {
  std::string dstName = lookup(dst);
  auto updShapeTy = dyn_cast<ShapedType>(update.getType());
  if (!updShapeTy) {
    // 0-D scalar update into 0-D? Unreachable since rank >= 1 here.
    return failure();
  }
  int64_t rank = updShapeTy.getRank();
  SmallVector<std::string> sVars(rank);
  for (int64_t d = 0; d < rank; ++d) {
    sVars[d] = freshName("s");
    int64_t uLen = updShapeTy.getDimSize(d);
    std::string uLenStr = updShapeTy.isDynamicDim(d)
                              ? ("size(" + lookup(update) + ", " +
                                 std::to_string(d + 1) + ")")
                              : std::to_string(uLen);
    indent() << sVars[d] << " = clamp(Int(" << lookup(starts[d])
             << "), 0, size(" << dstName << ", " << (d + 1) << ") - "
             << uLenStr << ") + 1\n";
  }
  std::string viewStr = "@view(" + dstName + "[";
  {
    llvm::raw_string_ostream ss(viewStr);
    for (int64_t d = 0; d < rank; ++d) {
      if (d > 0)
        ss << ", ";
      std::string uLenStr = updShapeTy.isDynamicDim(d)
                                ? ("size(" + lookup(update) + ", " +
                                   std::to_string(d + 1) + ")")
                                : std::to_string(updShapeTy.getDimSize(d));
      ss << sVars[d] << ":(" << sVars[d] << " + " << uLenStr << " - 1)";
    }
    ss << "])";
  }
  indent() << "@inbounds copyto!(" << viewStr << ", " << lookup(update)
           << ")\n";
  return success();
}

LogicalResult JuliaEmitter::emitGather(
    Value dst, Value operand, Value startIndices, ArrayRef<int64_t> offsetDims,
    ArrayRef<int64_t> collapsedDims, ArrayRef<int64_t> opBatchDims,
    ArrayRef<int64_t> idxBatchDims, ArrayRef<int64_t> startIndexMap,
    int64_t indexVecDim, ArrayRef<int64_t> sliceSizes) {
  auto opTy = cast<ShapedType>(operand.getType());
  auto dstShapeTy = dyn_cast<ShapedType>(dst.getType());
  int64_t outRank = dstShapeTy ? dstShapeTy.getRank() : 0;
  std::string opName = lookup(operand);
  std::string dstName = dstShapeTy ? lookup(dst) : "";
  std::string idxName = lookup(startIndices);
  auto idxShapeTy = dyn_cast<ShapedType>(startIndices.getType());
  int64_t idxRank = idxShapeTy ? idxShapeTy.getRank() : 0;

  llvm::DenseSet<int64_t> offsetSet(offsetDims.begin(), offsetDims.end());
  llvm::DenseSet<int64_t> collapsedSet(collapsedDims.begin(),
                                       collapsedDims.end());
  llvm::DenseSet<int64_t> opBatchSet(opBatchDims.begin(), opBatchDims.end());

  // Output dims that are NOT in `offsetDims` correspond (in ascending order of
  // output dims, or matched order — wait: let's check! Under `mapDim`, both
  // `outDims` and `startIndices` dims were reversed consistently if
  // `reverse_dims=true`, so ascending non-offset Julia output dims correspond
  // 1-to-1 in order with ascending non-`indexVecDim` Julia `startIndices` dims!).
  SmallVector<int64_t> batchOutDims;
  for (int64_t d = 0; d < outRank; ++d)
    if (!offsetSet.contains(d))
      batchOutDims.push_back(d);

  SmallVector<int64_t> idxNonVecDims;
  for (int64_t d = 0; d < idxRank; ++d)
    if (d != indexVecDim)
      idxNonVecDims.push_back(d);

  // Non-collapsed, non-batching operand dims correspond in order to `offsetDims`!
  SmallVector<int64_t> opWindowDims;
  for (int64_t d = 0; d < opTy.getRank(); ++d)
    if (!collapsedSet.contains(d) && !opBatchSet.contains(d))
      opWindowDims.push_back(d);

  SmallVector<std::string> outVars(outRank);
  for (int64_t d = 0; d < outRank; ++d)
    outVars[d] = freshName("g");

  for (int64_t d = outRank - 1; d >= 0; --d) {
    std::string bound = dstShapeTy.isDynamicDim(d)
                            ? ("size(" + dstName + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(dstShapeTy.getDimSize(d));
    indent();
    if (d == outRank - 1)
      os << "@inbounds ";
    os << "for " << outVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  // Build base start coordinate per operand dim (0-based clamped).
  SmallVector<std::string> opBase0(opTy.getRank(), "0");
  // Map batching dims
  for (size_t b = 0; b < opBatchDims.size(); ++b) {
    int64_t idxDim = idxBatchDims[b];
    // Find which position `idxDim` has in `idxNonVecDims`
    auto it = llvm::find(idxNonVecDims, idxDim);
    size_t pos = std::distance(idxNonVecDims.begin(), it);
    opBase0[opBatchDims[b]] = "(" + outVars[batchOutDims[pos]] + " - 1)";
  }

  // Read `startIndexMap` entries from `startIndices`.
  for (size_t k = 0; k < startIndexMap.size(); ++k) {
    int64_t opDim = startIndexMap[k];
    std::string rawCoord;
    if (!idxShapeTy) {
      rawCoord = "Int(" + idxName + ")";
    } else if (indexVecDim == idxRank) {
      // Implicit size-1 index vector!
      std::string sub;
      llvm::raw_string_ostream ss(sub);
      for (size_t p = 0; p < idxNonVecDims.size(); ++p) {
        if (p > 0)
          ss << ", ";
        ss << outVars[batchOutDims[p]];
      }
      rawCoord = "Int(" + idxName + "[" + ss.str() + "])";
    } else {
      SmallVector<std::string> idxCoords(idxRank);
      idxCoords[indexVecDim] = std::to_string(k + 1);
      for (size_t p = 0; p < idxNonVecDims.size(); ++p)
        idxCoords[idxNonVecDims[p]] = outVars[batchOutDims[p]];
      std::string sub;
      llvm::raw_string_ostream ss(sub);
      llvm::interleaveComma(idxCoords, ss);
      rawCoord = "Int(" + idxName + "[" + ss.str() + "])";
    }
    std::string opExtent = opTy.isDynamicDim(opDim)
                               ? ("size(" + opName + ", " +
                                  std::to_string(opDim + 1) + ")")
                               : std::to_string(opTy.getDimSize(opDim));
    std::string bVar = freshName("sb");
    indent() << bVar << " = clamp(" << rawCoord << ", 0, " << opExtent << " - "
             << sliceSizes[opDim] << ")\n";
    opBase0[opDim] = bVar;
  }

  // Add window offsets
  SmallVector<std::string> opCoord1(opTy.getRank());
  for (int64_t d = 0; d < opTy.getRank(); ++d)
    opCoord1[d] = "(" + opBase0[d] + " + 1)";
  for (size_t w = 0; w < offsetDims.size(); ++w) {
    int64_t opDim = opWindowDims[w];
    int64_t outDim = offsetDims[w];
    opCoord1[opDim] = "(" + opBase0[opDim] + " + " + outVars[outDim] + ")";
  }

  std::string opSub, outSub;
  {
    llvm::raw_string_ostream s1(opSub), s2(outSub);
    llvm::interleaveComma(opCoord1, s1);
    llvm::interleaveComma(outVars, s2);
  }
  if (dstShapeTy) {
    indent() << dstName << "[" << outSub << "] = " << opName << "[" << opSub
             << "]\n";
  } else {
    std::string gVal = freshName("gval");
    indent() << gVal << " = @inbounds " << opName << "[" << opSub << "]\n";
    valueExprs[dst] = gVal;
  }

  for (int64_t d = 0; d < outRank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitScatter(
    ValueRange dsts, Value scatterIndices, ValueRange updates,
    ArrayRef<int64_t> updateWinDims, ArrayRef<int64_t> insertedWinDims,
    ArrayRef<int64_t> inBatchDims, ArrayRef<int64_t> idxBatchDims,
    ArrayRef<int64_t> scatterMap, int64_t indexVecDim, Region &updateComp) {
  size_t numVars = dsts.size();
  auto dstTy = cast<ShapedType>(dsts[0].getType());
  int64_t opRank = dstTy.getRank();
  std::string primaryDst = lookup(dsts[0]);
  std::string idxName = lookup(scatterIndices);
  auto idxShapeTy = dyn_cast<ShapedType>(scatterIndices.getType());
  int64_t idxRank = idxShapeTy ? idxShapeTy.getRank() : 0;
  auto updShapeTy = dyn_cast<ShapedType>(updates[0].getType());
  int64_t updRank = updShapeTy ? updShapeTy.getRank() : 0;

  llvm::DenseSet<int64_t> updWinSet(updateWinDims.begin(), updateWinDims.end());
  llvm::DenseSet<int64_t> insertedSet(insertedWinDims.begin(),
                                      insertedWinDims.end());
  llvm::DenseSet<int64_t> inBatchSet(inBatchDims.begin(), inBatchDims.end());

  SmallVector<int64_t> updScatterDims;
  for (int64_t d = 0; d < updRank; ++d)
    if (!updWinSet.contains(d))
      updScatterDims.push_back(d);

  SmallVector<int64_t> idxNonVecDims;
  for (int64_t d = 0; d < idxRank; ++d)
    if (d != indexVecDim)
      idxNonVecDims.push_back(d);

  SmallVector<int64_t> opWinDims;
  for (int64_t d = 0; d < opRank; ++d)
    if (!insertedSet.contains(d) && !inBatchSet.contains(d))
      opWinDims.push_back(d);

  SmallVector<std::string> uVars(updRank);
  for (int64_t d = 0; d < updRank; ++d)
    uVars[d] = freshName("u");

  for (int64_t d = updRank - 1; d >= 0; --d) {
    std::string bound = updShapeTy.isDynamicDim(d)
                            ? ("size(" + lookup(updates[0]) + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(updShapeTy.getDimSize(d));
    indent();
    if (d == updRank - 1)
      os << "@inbounds ";
    os << "for " << uVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  SmallVector<std::string> opBase0(opRank, "0");
  for (size_t b = 0; b < inBatchDims.size(); ++b) {
    int64_t idxDim = idxBatchDims[b];
    auto it = llvm::find(idxNonVecDims, idxDim);
    size_t pos = std::distance(idxNonVecDims.begin(), it);
    opBase0[inBatchDims[b]] = "(" + uVars[updScatterDims[pos]] + " - 1)";
  }

  for (size_t k = 0; k < scatterMap.size(); ++k) {
    int64_t opDim = scatterMap[k];
    std::string rawCoord;
    if (!idxShapeTy) {
      rawCoord = "Int(" + idxName + ")";
    } else if (indexVecDim == idxRank) {
      std::string sub;
      llvm::raw_string_ostream ss(sub);
      for (size_t p = 0; p < idxNonVecDims.size(); ++p) {
        if (p > 0)
          ss << ", ";
        ss << uVars[updScatterDims[p]];
      }
      rawCoord = "Int(" + idxName + "[" + ss.str() + "])";
    } else {
      SmallVector<std::string> idxCoords(idxRank);
      idxCoords[indexVecDim] = std::to_string(k + 1);
      for (size_t p = 0; p < idxNonVecDims.size(); ++p)
        idxCoords[idxNonVecDims[p]] = uVars[updScatterDims[p]];
      std::string sub;
      llvm::raw_string_ostream ss(sub);
      llvm::interleaveComma(idxCoords, ss);
      rawCoord = "Int(" + idxName + "[" + ss.str() + "])";
    }
    std::string bVar = freshName("sc");
    indent() << bVar << " = " << rawCoord << "\n";
    opBase0[opDim] = bVar;
  }

  SmallVector<std::string> opCoord1(opRank);
  SmallVector<std::string> inBoundsConds;
  for (int64_t d = 0; d < opRank; ++d)
    opCoord1[d] = "(" + opBase0[d] + " + 1)";
  for (size_t w = 0; w < updateWinDims.size(); ++w) {
    int64_t opDim = opWinDims[w];
    int64_t uDim = updateWinDims[w];
    opCoord1[opDim] = "(" + opBase0[opDim] + " + " + uVars[uDim] + ")";
  }

  for (int64_t d = 0; d < opRank; ++d) {
    std::string cVar = freshName("di");
    indent() << cVar << " = " << opCoord1[d] << "\n";
    opCoord1[d] = cVar;
    std::string dBound = dstTy.isDynamicDim(d)
                             ? ("size(" + primaryDst + ", " +
                                std::to_string(d + 1) + ")")
                             : std::to_string(dstTy.getDimSize(d));
    inBoundsConds.push_back("(" + cVar + " >= 1 && " + cVar + " <= " + dBound +
                            ")");
  }

  if (!inBoundsConds.empty()) {
    indent() << "if ";
    llvm::interleave(inBoundsConds, os, " && ");
    os << "\n";
    indentLevel++;
  }

  std::string dstSub, updSub;
  {
    llvm::raw_string_ostream s1(dstSub), s2(updSub);
    llvm::interleaveComma(opCoord1, s1);
    llvm::interleaveComma(uVars, s2);
  }

  Block &comp = updateComp.front();
  for (size_t i = 0; i < numVars; ++i) {
    std::string curVal = freshName("cur");
    std::string updVal = freshName("upd");
    indent() << curVal << " = " << lookup(dsts[i]) << "[" << dstSub << "]\n";
    if (updShapeTy)
      indent() << updVal << " = " << lookup(updates[i]) << "[" << updSub
               << "]\n";
    else
      indent() << updVal << " = " << lookup(updates[i]) << "\n";
    valueExprs[comp.getArgument(i)] = curVal;
    valueExprs[comp.getArgument(numVars + i)] = updVal;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(comp, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);
  SmallVector<std::string> nextVals(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    nextVals[i] = freshName("sval");
    indent() << nextVals[i] << " = " << lookup(yieldOp.getOperand(i)) << "\n";
  }
  for (size_t i = 0; i < numVars; ++i)
    indent() << lookup(dsts[i]) << "[" << dstSub << "] = " << nextVals[i]
             << "\n";

  if (!inBoundsConds.empty()) {
    indentLevel--;
    indent() << "end\n";
  }

  for (int64_t d = 0; d < updRank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitSort(ValueRange dsts, int64_t dimension,
                                     bool /*isStable*/, Region &comparator) {
  size_t numVars = dsts.size();
  auto dstTy = cast<ShapedType>(dsts[0].getType());
  int64_t rank = dstTy.getRank();
  std::string primaryDst = lookup(dsts[0]);

  SmallVector<std::string> outerVars(rank);
  for (int64_t d = 0; d < rank; ++d)
    if (d != dimension)
      outerVars[d] = freshName("s");

  for (int64_t d = rank - 1; d >= 0; --d) {
    if (d == dimension)
      continue;
    std::string bound = dstTy.isDynamicDim(d)
                            ? ("size(" + primaryDst + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(dstTy.getDimSize(d));
    indent() << "@inbounds for " << outerVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  // Zero-allocation in-place stable insertion/shell-ready sort along
  // `dimension` supporting arbitrary variadic co-sorted arrays and custom
  // comparator region! (Stable insertion sort preserves `is_stable = true`
  // with 0 heap allocations!)
  std::string nVar = freshName("n");
  std::string dimBound = dstTy.isDynamicDim(dimension)
                             ? ("size(" + primaryDst + ", " +
                                std::to_string(dimension + 1) + ")")
                             : std::to_string(dstTy.getDimSize(dimension));
  indent() << nVar << " = " << dimBound << "\n";
  std::string ii = freshName("ii");
  std::string jj = freshName("jj");
  indent() << "@inbounds for " << ii << " in 2:" << nVar << "\n";
  indentLevel++;

  auto idxAt = [&](const std::string &posExpr) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    for (int64_t d = 0; d < rank; ++d) {
      if (d > 0)
        ss << ", ";
      ss << (d == dimension ? posExpr : outerVars[d]);
    }
    return ss.str();
  };

  SmallVector<std::string> keyVars(numVars);
  for (size_t k = 0; k < numVars; ++k) {
    keyVars[k] = freshName("key");
    indent() << keyVars[k] << " = " << lookup(dsts[k]) << "[" << idxAt(ii)
             << "]\n";
  }

  indent() << jj << " = " << ii << " - 1\n";
  indent() << "while " << jj << " >= 1\n";
  indentLevel++;

  // Comparator expects `(lhs_0, rhs_0, lhs_1, rhs_1, ...)` = `(key, prev)`:
  // if `comparator(key, prev)` is true, shift `prev` right!
  Block &compBlk = comparator.front();
  for (size_t k = 0; k < numVars; ++k) {
    std::string prevVar = freshName("prev");
    indent() << prevVar << " = " << lookup(dsts[k]) << "[" << idxAt(jj)
             << "]\n";
    valueExprs[compBlk.getArgument(2 * k)] = keyVars[k];
    valueExprs[compBlk.getArgument(2 * k + 1)] = prevVar;
  }

  Operation *term = nullptr;
  if (failed(emitBlockStatements(compBlk, term)))
    return failure();
  auto yieldOp = cast<YieldOp>(term);
  indent() << "if !(" << lookup(yieldOp.getOperand(0)) << ")\n";
  indentLevel++;
  indent() << "break\n";
  indentLevel--;
  indent() << "end\n";

  for (size_t k = 0; k < numVars; ++k) {
    indent() << lookup(dsts[k]) << "[" << idxAt(jj + " + 1") << "] = "
             << lookup(dsts[k]) << "[" << idxAt(jj) << "]\n";
  }
  indent() << jj << " -= 1\n";

  indentLevel--;
  indent() << "end\n";

  for (size_t k = 0; k < numVars; ++k) {
    indent() << lookup(dsts[k]) << "[" << idxAt(jj + " + 1")
             << "] = " << keyVars[k] << "\n";
  }

  indentLevel--;
  indent() << "end\n";

  for (int64_t d = 0; d < rank; ++d) {
    if (d == dimension)
      continue;
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitConv(
    Value dst, Value lhs, Value rhs, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> padding, ArrayRef<int64_t> lhsDil,
    ArrayRef<int64_t> rhsDil, ArrayRef<bool> winRev, int64_t inBatchDim,
    int64_t inFeatDim, ArrayRef<int64_t> inSpatialDims, int64_t kerInFeatDim,
    int64_t kerOutFeatDim, ArrayRef<int64_t> kerSpatialDims,
    int64_t outBatchDim, int64_t outFeatDim, ArrayRef<int64_t> outSpatialDims,
    int64_t featGroups, int64_t batchGroups) {
  auto lhsTy = cast<ShapedType>(lhs.getType());
  auto rhsTy = cast<ShapedType>(rhs.getType());
  auto dstTy = cast<ShapedType>(dst.getType());
  std::string dstName = lookup(dst);
  std::string lhsName = lookup(lhs);
  std::string rhsName = lookup(rhs);
  int64_t outRank = dstTy.getRank();
  size_t numSpatial = outSpatialDims.size();

  SmallVector<std::string> outVars(outRank);
  for (int64_t d = 0; d < outRank; ++d)
    outVars[d] = freshName("co");

  for (int64_t d = outRank - 1; d >= 0; --d) {
    std::string bound = dstTy.isDynamicDim(d)
                            ? ("size(" + dstName + ", " +
                               std::to_string(d + 1) + ")")
                            : std::to_string(dstTy.getDimSize(d));
    indent();
    if (d == outRank - 1)
      os << "@inbounds ";
    os << "for " << outVars[d] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string acc = freshName("acc");
  indent() << acc << " = zero(" << formatScalarType(dstTy.getElementType())
           << ")\n";

  // Group sizes (0-based math):
  std::string outFeatExtent =
      dstTy.isDynamicDim(outFeatDim)
          ? ("size(" + dstName + ", " + std::to_string(outFeatDim + 1) + ")")
          : std::to_string(dstTy.getDimSize(outFeatDim));
  std::string outBatchExtent =
      dstTy.isDynamicDim(outBatchDim)
          ? ("size(" + dstName + ", " + std::to_string(outBatchDim + 1) + ")")
          : std::to_string(dstTy.getDimSize(outBatchDim));
  std::string kerInExtent =
      rhsTy.isDynamicDim(kerInFeatDim)
          ? ("size(" + rhsName + ", " + std::to_string(kerInFeatDim + 1) + ")")
          : std::to_string(rhsTy.getDimSize(kerInFeatDim));

  std::string fGroupIdx = freshName("fg");
  std::string bGroupIdx = freshName("bg");
  if (featGroups > 1) {
    indent() << fGroupIdx << " = div(" << outVars[outFeatDim] << " - 1, div("
             << outFeatExtent << ", " << featGroups << "))\n";
  } else if (batchGroups > 1) {
    indent() << bGroupIdx << " = div(" << outVars[outFeatDim] << " - 1, div("
             << outFeatExtent << ", " << batchGroups << "))\n";
  }

  SmallVector<std::string> kSpatialVars(numSpatial);
  for (size_t s = 0; s < numSpatial; ++s)
    kSpatialVars[s] = freshName("kw");
  std::string kInVar = freshName("kc");

  for (int64_t s = static_cast<int64_t>(numSpatial) - 1; s >= 0; --s) {
    int64_t kd = kerSpatialDims[s];
    std::string kBound = rhsTy.isDynamicDim(kd)
                             ? ("size(" + rhsName + ", " +
                                std::to_string(kd + 1) + ")")
                             : std::to_string(rhsTy.getDimSize(kd));
    indent() << "for " << kSpatialVars[s] << " in 0:(" << kBound << " - 1)\n";
    indentLevel++;
  }

  SmallVector<std::string> lhsCoords(lhsTy.getRank());
  SmallVector<std::string> rhsCoords(rhsTy.getRank());
  SmallVector<std::string> validConds;

  for (size_t s = 0; s < numSpatial; ++s) {
    int64_t inDim = inSpatialDims[s];
    int64_t kerDim = kerSpatialDims[s];
    int64_t outDim = outSpatialDims[s];
    std::string pVar = freshName("sp");
    indent() << pVar << " = (" << outVars[outDim] << " - 1) * " << strides[s]
             << " + " << kSpatialVars[s] << " * " << rhsDil[s] << " - ("
             << padding[2 * s] << ")\n";
    std::string inExtent = lhsTy.isDynamicDim(inDim)
                               ? ("size(" + lhsName + ", " +
                                  std::to_string(inDim + 1) + ")")
                               : std::to_string(lhsTy.getDimSize(inDim));
    if (lhsDil[s] == 1) {
      validConds.push_back("(" + pVar + " >= 0 && " + pVar + " < " + inExtent +
                           ")");
      lhsCoords[inDim] = "(" + pVar + " + 1)";
    } else {
      validConds.push_back("(" + pVar + " >= 0 && rem(" + pVar + ", " +
                           std::to_string(lhsDil[s]) + ") == 0 && div(" + pVar +
                           ", " + std::to_string(lhsDil[s]) + ") < " +
                           inExtent + ")");
      lhsCoords[inDim] =
          "(div(" + pVar + ", " + std::to_string(lhsDil[s]) + ") + 1)";
    }

    if (winRev[s]) {
      std::string kExtent = rhsTy.isDynamicDim(kerDim)
                                ? ("size(" + rhsName + ", " +
                                   std::to_string(kerDim + 1) + ")")
                                : std::to_string(rhsTy.getDimSize(kerDim));
      rhsCoords[kerDim] = "(" + kExtent + " - " + kSpatialVars[s] + ")";
    } else {
      rhsCoords[kerDim] = "(" + kSpatialVars[s] + " + 1)";
    }
  }

  if (!validConds.empty()) {
    indent() << "if ";
    llvm::interleave(validConds, os, " && ");
    os << "\n";
    indentLevel++;
  }

  indent() << "@simd for " << kInVar << " in 1:" << kerInExtent << "\n";
  indentLevel++;

  if (batchGroups > 1) {
    lhsCoords[inBatchDim] =
        "(" + bGroupIdx + " * " + outBatchExtent + " + " +
        outVars[outBatchDim] + ")";
  } else {
    lhsCoords[inBatchDim] = outVars[outBatchDim];
  }

  if (featGroups > 1) {
    lhsCoords[inFeatDim] =
        "(" + fGroupIdx + " * " + kerInExtent + " + " + kInVar + ")";
  } else {
    lhsCoords[inFeatDim] = kInVar;
  }

  rhsCoords[kerInFeatDim] = kInVar;
  rhsCoords[kerOutFeatDim] = outVars[outFeatDim];

  std::string lSub, rSub, oSub;
  {
    llvm::raw_string_ostream s1(lSub), s2(rSub), s3(oSub);
    llvm::interleaveComma(lhsCoords, s1);
    llvm::interleaveComma(rhsCoords, s2);
    llvm::interleaveComma(outVars, s3);
  }

  indent() << acc << " += " << lhsName << "[" << lSub << "] * " << rhsName
           << "[" << rSub << "]\n";

  indentLevel--;
  indent() << "end\n";

  if (!validConds.empty()) {
    indentLevel--;
    indent() << "end\n";
  }

  for (size_t s = 0; s < numSpatial; ++s) {
    indentLevel--;
    indent() << "end\n";
  }

  indent() << dstName << "[" << oSub << "] = " << acc << "\n";

  for (int64_t d = 0; d < outRank; ++d) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitCholesky(Value dst, bool lower) {
  auto dstTy = cast<ShapedType>(dst.getType());
  int64_t rank = dstTy.getRank();
  std::string dstName = lookup(dst);
  const char *uplo = lower ? ":L" : ":U";
  std::string zeroLit =
      "zero(" + formatScalarType(dstTy.getElementType()) + ")";

  // Note: in Julia `Matrix(N, N)` (first two dims or last two dims?):
  // Let's check! In StableHLO, `cholesky` operates on the minor-most 2
  // dimensions `[..., N, N]`!
  // Under `reverse_dims = true`, StableHLO `[B, N, N]` maps to Julia `[N, N, B]`,
  // where the `[N, N]` matrix is the FIRST two dimensions `[:, :, b]` (Wait:
  // ALSO check matrix orientation! In StableHLO, `A[..., i, j]` has row `i`
  // (dim R-2) and col `j` (dim R-1). Under `reverse_dims = true`, Julia array
  // has `J[j, i, ...] = A[..., i, j]`, i.e., `J[:, :, b] = A[b, :, :]ᵀ`!
  // So if StableHLO wants `L` lower-triangular (`L * Lᴴ = A`), then
  // `J_out = Lᵀ` which is UPPER-triangular (`J_outᴴ * J_out = J` for real, or
  // `Uᵀ * U = Aᵀ` — wait: for complex `L * Lᴴ = A => Aᵀ = L̄ * Lᵀ`, whereas
  // `cholesky!(Hermitian(M, :U))` computes `Uᴴ * U = M`!).
  // `LowerStableHLOToJuliaPass` already adjusts `lower` when `reverse_dims=true`
  // so `JuliaEmitter::emitCholesky` simply computes `cholesky!(Hermitian(slice, uplo))`
  // on the 2D matrix! Wait: what if `rank > 2` with `reverse_dims = false` vs
  // `reverse_dims = true`? In `LowerStableHLOToJuliaPass`, let's only emit
  // `CholeskyOp` when matrix dims are dims `(0, 1)` (all rank-2 in both modes,
  // plus batched rank > 2 when `reverse_dims = true`!).
  SmallVector<std::string> bVars(rank > 2 ? rank - 2 : 0);
  for (int64_t b = rank - 3; b >= 0; --b) {
    bVars[b] = freshName("b");
    int64_t dimIdx = b + 2;
    std::string bound = dstTy.isDynamicDim(dimIdx)
                            ? ("size(" + dstName + ", " +
                               std::to_string(dimIdx + 1) + ")")
                            : std::to_string(dstTy.getDimSize(dimIdx));
    indent() << "@inbounds for " << bVars[b] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string matExpr = dstName;
  if (rank > 2) {
    std::string v = "@view(" + dstName + "[:, :";
    llvm::raw_string_ostream ss(v);
    for (const auto &bv : bVars)
      ss << ", " << bv;
    ss << "])";
    matExpr = freshName("mslice");
    indent() << matExpr << " = " << ss.str() << "\n";
  }

  indent() << "LinearAlgebra.cholesky!(LinearAlgebra.Hermitian(" << matExpr
           << ", " << uplo << "); check=false)\n";

  // Zero out the strict opposite triangle to match StableHLO spec!
  std::string cVar = freshName("cj");
  std::string rVar = freshName("ci");
  indent() << "@inbounds for " << cVar << " in 1:size(" << matExpr << ", 2)\n";
  indentLevel++;
  if (lower) {
    indent() << "for " << rVar << " in 1:(" << cVar << " - 1)\n";
  } else {
    indent() << "for " << rVar << " in (" << cVar << " + 1):size(" << matExpr
             << ", 1)\n";
  }
  indentLevel++;
  indent() << matExpr << "[" << rVar << ", " << cVar << "] = " << zeroLit
           << "\n";
  indentLevel--;
  indent() << "end\n";
  indentLevel--;
  indent() << "end\n";

  for (int64_t b = 0; b < rank - 2; ++b) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitTriangularSolve(Value dst, Value a,
                                                bool leftSide, bool lower,
                                                bool unitDiag,
                                                TransposeMode transA) {
  auto dstTy = cast<ShapedType>(dst.getType());
  int64_t rank = dstTy.getRank();
  std::string dstName = lookup(dst);
  std::string aName = lookup(a);

  auto buildTriWrapper = [&](const std::string &mat) {
    std::string inner = wrapMatrixTranspose(mat, transA);
    bool effectiveLower =
        (transA == TransposeMode::none) ? lower : !lower;
    const char *triTy =
        effectiveLower
            ? (unitDiag ? "LinearAlgebra.UnitLowerTriangular"
                        : "LinearAlgebra.LowerTriangular")
            : (unitDiag ? "LinearAlgebra.UnitUpperTriangular"
                        : "LinearAlgebra.UpperTriangular");
    return std::string(triTy) + "(" + inner + ")";
  };

  SmallVector<std::string> bVars(rank > 2 ? rank - 2 : 0);
  for (int64_t b = rank - 3; b >= 0; --b) {
    bVars[b] = freshName("b");
    int64_t dimIdx = b + 2;
    std::string bound = dstTy.isDynamicDim(dimIdx)
                            ? ("size(" + dstName + ", " +
                               std::to_string(dimIdx + 1) + ")")
                            : std::to_string(dstTy.getDimSize(dimIdx));
    indent() << "@inbounds for " << bVars[b] << " in 1:" << bound << "\n";
    indentLevel++;
  }

  std::string dSlice = dstName;
  std::string aSlice = aName;
  if (rank > 2) {
    std::string suffix;
    llvm::raw_string_ostream ss(suffix);
    for (const auto &bv : bVars)
      ss << ", " << bv;
    dSlice = "@view(" + dstName + "[:, :" + ss.str() + "])";
    aSlice = "@view(" + aName + "[:, :" + ss.str() + "])";
  }

  std::string triExpr = buildTriWrapper(aSlice);
  if (leftSide) {
    indent() << "LinearAlgebra.ldiv!(" << triExpr << ", " << dSlice << ")\n";
  } else {
    indent() << "LinearAlgebra.rdiv!(" << dSlice << ", " << triExpr << ")\n";
  }

  for (int64_t b = 0; b < rank - 2; ++b) {
    indentLevel--;
    indent() << "end\n";
  }
  return success();
}

LogicalResult JuliaEmitter::emitIf(IfOp op) {
  size_t numRes = op.getNumResults();
  SmallVector<std::string> resVars(numRes);
  for (size_t i = 0; i < numRes; ++i)
    resVars[i] = freshName("ifv");

  indent() << "if " << lookup(op.getCondition()) << "\n";
  indentLevel++;
  Operation *thenTerm = nullptr;
  if (failed(emitBlockStatements(op.getThenRegion().front(), thenTerm)))
    return failure();
  if (auto y = dyn_cast_or_null<YieldOp>(thenTerm)) {
    for (size_t i = 0; i < numRes; ++i)
      indent() << resVars[i] << " = " << lookup(y.getOperand(i)) << "\n";
  }
  indentLevel--;

  if (!op.getElseRegion().empty()) {
    indent() << "else\n";
    indentLevel++;
    Operation *elseTerm = nullptr;
    if (failed(emitBlockStatements(op.getElseRegion().front(), elseTerm)))
      return failure();
    if (auto y = dyn_cast_or_null<YieldOp>(elseTerm)) {
      for (size_t i = 0; i < numRes; ++i)
        indent() << resVars[i] << " = " << lookup(y.getOperand(i)) << "\n";
    }
    indentLevel--;
  }
  indent() << "end\n";

  for (size_t i = 0; i < numRes; ++i)
    valueExprs[op.getResult(i)] = resVars[i];
  return success();
}

LogicalResult JuliaEmitter::emitWhile(WhileOp op) {
  size_t numVars = op.getInits().size();
  SmallVector<std::string> stateVars(numVars);
  for (size_t i = 0; i < numVars; ++i) {
    stateVars[i] = freshName("w");
    indent() << stateVars[i] << " = " << lookup(op.getInits()[i]) << "\n";
  }

  Block &condBlk = op.getCond().front();
  Block &bodyBlk = op.getBody().front();
  for (size_t i = 0; i < numVars; ++i) {
    valueExprs[condBlk.getArgument(i)] = stateVars[i];
    valueExprs[bodyBlk.getArgument(i)] = stateVars[i];
  }

  // Check if condition block has only pure inlineable ops so we can emit
  // idiomatic `while <cond_expr>` instead of `while true ... if !pred break`!
  bool simpleCond = llvm::all_of(condBlk.without_terminator(), [&](Operation &o) {
    return shouldInlinePureOp(&o);
  });

  if (simpleCond) {
    for (Operation &o : condBlk.without_terminator())
      valueExprs[o.getResult(0)] = buildPureScalarExpr(&o);
    auto condOp = cast<CondOp>(condBlk.getTerminator());
    indent() << "while " << lookup(condOp.getPredicate()) << "\n";
    indentLevel++;
  } else {
    indent() << "while true\n";
    indentLevel++;
    Operation *condTerm = nullptr;
    if (failed(emitBlockStatements(condBlk, condTerm)))
      return failure();
    auto condOp = cast<CondOp>(condTerm);
    indent() << "if !(" << lookup(condOp.getPredicate()) << ")\n";
    indentLevel++;
    indent() << "break\n";
    indentLevel--;
    indent() << "end\n";
  }

  Operation *bodyTerm = nullptr;
  if (failed(emitBlockStatements(bodyBlk, bodyTerm)))
    return failure();
  auto yieldOp = cast<YieldOp>(bodyTerm);

  // Collect state variables that actually change (`yieldOperand != bodyArg`).
  // For in-place mutated arrays where `yieldOperand == bodyArg`, emit NOTHING!
  // For swapped ping-pong buffers or scalar updates, emit parallel tuple
  // assignment `(w0, w1) = (next0, next1)` so swaps happen on the stack with
  // zero temporary array allocations!
  SmallVector<size_t> changedIndices;
  for (size_t i = 0; i < numVars; ++i) {
    if (yieldOp.getOperand(i) != bodyBlk.getArgument(i))
      changedIndices.push_back(i);
  }
  if (changedIndices.size() == 1) {
    size_t idx = changedIndices[0];
    indent() << stateVars[idx] << " = " << lookup(yieldOp.getOperand(idx))
             << "\n";
  } else if (changedIndices.size() > 1) {
    indent() << "(";
    llvm::interleaveComma(changedIndices, os,
                          [&](size_t idx) { os << stateVars[idx]; });
    os << ") = (";
    llvm::interleaveComma(changedIndices, os, [&](size_t idx) {
      os << lookup(yieldOp.getOperand(idx));
    });
    os << ")\n";
  }

  indentLevel--;
  indent() << "end\n";

  for (size_t i = 0; i < op.getNumResults(); ++i)
    valueExprs[op.getResult(i)] = stateVars[i];
  return success();
}

LogicalResult JuliaEmitter::emitOperation(Operation *op) {
  // Pure scalar / view ops: inline if single-use and shallow, otherwise bind.
  if (isa<ConstantOp, UnaryOp, BinaryOp, CmpOp, SelectOp, ClampOp, CastOp,
          BitcastOp, MathLibmOp, SizeOp, LoadOp>(op)) {
    std::string expr = buildPureScalarExpr(op);
    if (shouldInlinePureOp(op)) {
      valueExprs[op->getResult(0)] = expr;
    } else {
      std::string vName = freshName("v");
      indent() << vName << " = " << expr << "\n";
      valueExprs[op->getResult(0)] = vName;
    }
    return success();
  }

  if (auto vs = dyn_cast<ViewSliceOp>(op)) {
    std::string expr = buildViewSliceExpr(
        vs.getSource(), vs.getStartIndices(), vs.getLimitIndices(),
        vs.getStrides());
    if (shouldInlinePureOp(op)) {
      valueExprs[vs.getResult()] = expr;
    } else {
      std::string vName = freshName("view");
      indent() << vName << " = " << expr << "\n";
      valueExprs[vs.getResult()] = vName;
    }
    return success();
  }

  if (auto vds = dyn_cast<ViewDynamicSliceOp>(op)) {
    std::string expr = buildViewDynamicSliceExpr(
        vds.getSource(), vds.getStartIndices(), vds.getSliceSizes());
    if (shouldInlinePureOp(op)) {
      valueExprs[vds.getResult()] = expr;
    } else {
      std::string vName = freshName("view");
      indent() << vName << " = " << expr << "\n";
      valueExprs[vds.getResult()] = vName;
    }
    return success();
  }

  if (auto vr = dyn_cast<ViewReshapeOp>(op)) {
    std::string expr = buildViewReshapeExpr(
        vr.getSource(), vr.getType(), vr.getDynamicDims());
    if (shouldInlinePureOp(op)) {
      valueExprs[vr.getResult()] = expr;
    } else {
      std::string vName = freshName("rview");
      indent() << vName << " = " << expr << "\n";
      valueExprs[vr.getResult()] = vName;
    }
    return success();
  }

  if (auto alloc = dyn_cast<AllocOp>(op)) {
    auto arrTy = alloc.getType();
    std::string bName = freshName("buf");
    std::string dims = formatShapeArgs(arrTy.getShape(), alloc.getDynamicDims());
    indent() << bName << " = " << formatType(arrTy) << "(undef";
    if (!dims.empty())
      os << ", " << dims;
    os << ")\n";
    valueExprs[alloc.getResult()] = bName;
    return success();
  }

  if (auto cp = dyn_cast<CopyOp>(op)) {
    std::string bName = freshName("buf");
    if (isa<ArrayType>(cp.getSource().getType())) {
      indent() << bName << " = copy(" << lookup(cp.getSource()) << ")\n";
    } else {
      indent() << bName << " = collect(" << lookup(cp.getSource()) << ")\n";
    }
    valueExprs[cp.getResult()] = bName;
    return success();
  }

  if (auto mat = dyn_cast<MaterializeArrayOp>(op)) {
    std::string bName = freshName("arr");
    auto resTy = mat.getType();
    indent() << bName << " = convert(" << formatType(resTy) << ", collect("
             << lookup(mat.getSource()) << "))\n";
    valueExprs[mat.getResult()] = bName;
    return success();
  }

  if (auto cpTo = dyn_cast<CopyToOp>(op)) {
    indent() << "@inbounds copyto!(" << lookup(cpTo.getDest()) << ", "
             << lookup(cpTo.getSource()) << ")\n";
    return success();
  }

  if (auto fillIp = dyn_cast<FillInPlaceOp>(op)) {
    indent() << "fill!(" << lookup(fillIp.getDest()) << ", "
             << lookup(fillIp.getValue()) << ")\n";
    return success();
  }

  if (auto store = dyn_cast<StoreOp>(op)) {
    indent() << "@inbounds " << lookup(store.getDest()) << "[";
    llvm::interleaveComma(store.getIndices(), os,
                          [&](Value idx) { os << lookup(idx); });
    os << "] = " << lookup(store.getValue()) << "\n";
    return success();
  }

  if (auto arrConst = dyn_cast<ArrayConstantOp>(op)) {
    auto arrTy = arrConst.getType();
    Type elemTy = arrTy.getElementType();
    std::string elemName = formatScalarType(elemTy);
    std::string cName = freshName("cst");
    indent() << cName << " = ";
    if (arrTy.getRank() > 1)
      os << "reshape(";
    os << elemName << "[";
    if (auto dense = dyn_cast<DenseElementsAttr>(arrConst.getValues())) {
      bool first = true;
      for (Attribute a : dense.getValues<Attribute>()) {
        if (!first)
          os << ", ";
        first = false;
        os << formatScalarAttr(a, elemTy);
      }
    }
    os << "]";
    if (arrTy.getRank() > 1) {
      os << ", " << formatShapeArgs(arrTy.getShape(), {});
      os << ")";
    }
    os << "\n";
    valueExprs[arrConst.getResult()] = cName;
    return success();
  }

  // Buffer-domain in-place ops:
  if (auto kip = dyn_cast<KernelInPlaceOp>(op))
    return emitKernelLoop(kip.getOuts(), kip.getBody());

  if (auto mip = dyn_cast<MatmulInPlaceOp>(op)) {
    return emitMatmul(mip.getDst(), mip.getLhs(), mip.getRhs(), mip.getAlpha(),
                      mip.getBeta(), mip.getLhsTranspose(),
                      mip.getRhsTranspose(), mip.getBatchDimsLhs(),
                      mip.getBatchDimsRhs(), mip.getBatchDimsOut());
  }

  if (auto dgip = dyn_cast<DotGeneralInPlaceOp>(op)) {
    return emitDotGeneral(
        dgip.getDst(), dgip.getLhs(), dgip.getRhs(),
        dgip.getLhsBatchingDims(), dgip.getRhsBatchingDims(),
        dgip.getLhsContractingDims(), dgip.getRhsContractingDims(),
        dgip.getResultDimOrder());
  }

  if (auto rip = dyn_cast<ReduceInPlaceOp>(op)) {
    return emitReduceArray(rip.getDsts(), rip.getInputs(), rip.getInitValues(),
                           rip.getDimensions(), rip.getCombiner());
  }

  if (auto rwip = dyn_cast<ReduceWindowInPlaceOp>(op)) {
    return emitReduceWindow(
        rwip.getDsts(), rwip.getInputs(), rwip.getInitValues(),
        rwip.getWindowDimensions(), rwip.getWindowStrides(),
        rwip.getBaseDilations(), rwip.getWindowDilations(), rwip.getPadding(),
        rwip.getCombiner());
  }

  if (auto pip = dyn_cast<PadInPlaceOp>(op)) {
    return emitPad(pip.getDst(), pip.getOperand(), pip.getPaddingValue(),
                   pip.getEdgePaddingLow(), pip.getEdgePaddingHigh(),
                   pip.getInteriorPadding());
  }

  if (auto dsip = dyn_cast<DynamicSliceInPlaceOp>(op)) {
    return emitDynamicSliceCopy(dsip.getDst(), dsip.getOperand(),
                                dsip.getStartIndices(), dsip.getSliceSizes());
  }

  if (auto dusip = dyn_cast<DynamicUpdateSliceInPlaceOp>(op)) {
    return emitDynamicUpdateSlice(dusip.getDst(), dusip.getUpdate(),
                                  dusip.getStartIndices());
  }

  if (auto gip = dyn_cast<GatherInPlaceOp>(op)) {
    return emitGather(
        gip.getDst(), gip.getOperand(), gip.getStartIndices(),
        gip.getOffsetDims(), gip.getCollapsedSliceDims(),
        gip.getOperandBatchingDims(), gip.getStartIndicesBatchingDims(),
        gip.getStartIndexMap(), gip.getIndexVectorDim(), gip.getSliceSizes());
  }

  if (auto scip = dyn_cast<ScatterInPlaceOp>(op)) {
    return emitScatter(
        scip.getDsts(), scip.getScatterIndices(), scip.getUpdates(),
        scip.getUpdateWindowDims(), scip.getInsertedWindowDims(),
        scip.getInputBatchingDims(), scip.getScatterIndicesBatchingDims(),
        scip.getScatterDimsToOperandDims(), scip.getIndexVectorDim(),
        scip.getUpdateComputation());
  }

  if (auto sip = dyn_cast<SortInPlaceOp>(op)) {
    return emitSort(sip.getDsts(), sip.getDimension(), sip.getIsStable(),
                    sip.getComparator());
  }

  if (auto cvip = dyn_cast<ConvInPlaceOp>(op)) {
    return emitConv(
        cvip.getDst(), cvip.getLhs(), cvip.getRhs(), cvip.getWindowStrides(),
        cvip.getPadding(), cvip.getLhsDilation(), cvip.getRhsDilation(),
        cvip.getWindowReversal(), cvip.getInputBatchDimension(),
        cvip.getInputFeatureDimension(), cvip.getInputSpatialDimensions(),
        cvip.getKernelInputFeatureDimension(),
        cvip.getKernelOutputFeatureDimension(),
        cvip.getKernelSpatialDimensions(), cvip.getOutputBatchDimension(),
        cvip.getOutputFeatureDimension(), cvip.getOutputSpatialDimensions(),
        cvip.getFeatureGroupCount(), cvip.getBatchGroupCount());
  }

  if (auto chip = dyn_cast<CholeskyInPlaceOp>(op))
    return emitCholesky(chip.getDst(), chip.getLower());

  if (auto tsip = dyn_cast<TriangularSolveInPlaceOp>(op)) {
    return emitTriangularSolve(tsip.getDst(), tsip.getA(), tsip.getLeftSide(),
                               tsip.getLower(), tsip.getUnitDiagonal(),
                               tsip.getTransposeA());
  }

  // Value-domain ops that remain (e.g., scalar reductions or unbufferized IR):
  if (auto red = dyn_cast<ReduceOp>(op)) {
    if (red.getNumResults() > 0 &&
        !isJuliaBufferType(red.getResult(0).getType())) {
      return emitReduceScalar(red);
    }
    SmallVector<Value> fakeOuts;
    for (Value r : red.getResults()) {
      auto rTy = cast<ArrayType>(r.getType());
      std::string bName = freshName("buf");
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), {}) << ")\n";
      valueExprs[r] = bName;
      fakeOuts.push_back(r);
    }
    return emitReduceArray(fakeOuts, red.getInputs(), red.getInitValues(),
                           red.getDimensions(), red.getCombiner());
  }

  if (auto fillOp = dyn_cast<FillOp>(op)) {
    auto arrTy = fillOp.getType();
    std::string bName = freshName("buf");
    indent() << bName << " = fill(" << lookup(fillOp.getValue()) << ", "
             << formatShapeArgs(arrTy.getShape(), fillOp.getDynamicDims())
             << ")\n";
    valueExprs[fillOp.getResult()] = bName;
    return success();
  }

  if (auto kern = dyn_cast<KernelOp>(op)) {
    for (Value r : kern.getResults()) {
      auto rTy = cast<ArrayType>(r.getType());
      std::string bName = freshName("buf");
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), kern.getDynamicDims())
               << ")\n";
      valueExprs[r] = bName;
    }
    return emitKernelLoop(kern.getResults(), kern.getBody());
  }

  if (auto mm = dyn_cast<MatmulOp>(op)) {
    if (!isJuliaBufferType(mm.getType())) {
      // 1D inner product returning unboxed scalar!
      auto lhsTy = cast<ShapedType>(mm.getLhs().getType());
      std::string acc = freshName("dot");
      std::string lhsName = lookup(mm.getLhs());
      std::string rhsName = lookup(mm.getRhs());
      std::string kVar = freshName("k");
      std::string bound = lhsTy.isDynamicDim(0)
                              ? ("length(" + lhsName + ")")
                              : std::to_string(lhsTy.getDimSize(0));
      indent() << acc << " = zero(" << formatScalarType(mm.getType()) << ")\n";
      indent() << "@inbounds @simd for " << kVar << " in 1:" << bound << "\n";
      indentLevel++;
      indent() << acc << " += " << lhsName << "[" << kVar << "] * " << rhsName
               << "[" << kVar << "]\n";
      indentLevel--;
      indent() << "end\n";
      if (mm.getAlpha()) {
        indent() << acc << " = " << lookup(mm.getAlpha()) << " * " << acc
                 << "\n";
      }
      valueExprs[mm.getResult()] = acc;
      return success();
    }
    auto rTy = cast<ArrayType>(mm.getType());
    std::string bName = freshName("buf");
    if (mm.getAcc()) {
      indent() << bName << " = copy(" << lookup(mm.getAcc()) << ")\n";
    } else {
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), {}) << ")\n";
    }
    valueExprs[mm.getResult()] = bName;
    return emitMatmul(mm.getResult(), mm.getLhs(), mm.getRhs(), mm.getAlpha(),
                      mm.getBeta(), mm.getLhsTranspose(), mm.getRhsTranspose(),
                      mm.getBatchDimsLhs(), mm.getBatchDimsRhs(),
                      mm.getBatchDimsOut());
  }

  if (auto dg = dyn_cast<DotGeneralOp>(op)) {
    if (auto rTy = dyn_cast<ArrayType>(dg.getType())) {
      std::string bName = freshName("buf");
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), {}) << ")\n";
      valueExprs[dg.getResult()] = bName;
    }
    return emitDotGeneral(
        dg.getResult(), dg.getLhs(), dg.getRhs(), dg.getLhsBatchingDims(),
        dg.getRhsBatchingDims(), dg.getLhsContractingDims(),
        dg.getRhsContractingDims(), dg.getResultDimOrder());
  }

  if (auto gop = dyn_cast<GatherOp>(op)) {
    if (auto rTy = dyn_cast<ArrayType>(gop.getType())) {
      std::string bName = freshName("buf");
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), {}) << ")\n";
      valueExprs[gop.getResult()] = bName;
    }
    return emitGather(
        gop.getResult(), gop.getOperand(), gop.getStartIndices(),
        gop.getOffsetDims(), gop.getCollapsedSliceDims(),
        gop.getOperandBatchingDims(), gop.getStartIndicesBatchingDims(),
        gop.getStartIndexMap(), gop.getIndexVectorDim(), gop.getSliceSizes());
  }

  if (auto bc = dyn_cast<BroadcastInDimOp>(op)) {
    auto rTy = bc.getType();
    int64_t outRank = rTy.getRank();
    std::string bName = freshName("buf");
    indent() << bName << " = " << formatType(rTy) << "(undef, "
             << formatShapeArgs(rTy.getShape(), bc.getDynamicDims()) << ")\n";
    valueExprs[bc.getResult()] = bName;
    if (!isJuliaBufferType(bc.getOperand().getType())) {
      indent() << "fill!(" << bName << ", " << lookup(bc.getOperand()) << ")\n";
      return success();
    }
    auto inTy = cast<ShapedType>(bc.getOperand().getType());
    ArrayRef<int64_t> bDims = bc.getBroadcastDimensions();
    SmallVector<std::string> ivs(outRank);
    for (int64_t d = 0; d < outRank; ++d)
      ivs[d] = freshName("i");
    for (int64_t d = outRank - 1; d >= 0; --d) {
      std::string bound = rTy.isDynamicDim(d)
                              ? ("size(" + bName + ", " +
                                 std::to_string(d + 1) + ")")
                              : std::to_string(rTy.getDimSize(d));
      indent();
      if (d == outRank - 1)
        os << "@inbounds ";
      if (d == 0)
        os << "@simd ";
      os << "for " << ivs[d] << " in 1:" << bound << "\n";
      indentLevel++;
    }
    SmallVector<std::string> inIdx(inTy.getRank());
    for (int64_t d = 0; d < inTy.getRank(); ++d) {
      if (inTy.getDimSize(d) == 1)
        inIdx[d] = "1";
      else
        inIdx[d] = ivs[bDims[d]];
    }
    std::string oSub, iSub;
    {
      llvm::raw_string_ostream s1(oSub), s2(iSub);
      llvm::interleaveComma(ivs, s1);
      llvm::interleaveComma(inIdx, s2);
    }
    indent() << bName << "[" << oSub << "] = " << lookup(bc.getOperand()) << "["
             << iSub << "]\n";
    for (int64_t d = 0; d < outRank; ++d) {
      indentLevel--;
      indent() << "end\n";
    }
    return success();
  }

  if (auto tr = dyn_cast<TransposeOp>(op)) {
    std::string bName = freshName("buf");
    indent() << bName << " = permutedims(" << lookup(tr.getOperand()) << ", (";
    ArrayRef<int64_t> perm = tr.getPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << (perm[i] + 1);
    }
    if (perm.size() == 1)
      os << ",";
    os << "))\n";
    valueExprs[tr.getResult()] = bName;
    return success();
  }

  if (auto rev = dyn_cast<ReverseOp>(op)) {
    std::string cur = "copy(" + lookup(rev.getOperand()) + ")";
    std::string bName = freshName("buf");
    indent() << bName << " = " << cur << "\n";
    for (int64_t d : rev.getDimensions()) {
      indent() << "reverse!(" << bName << "; dims=" << (d + 1) << ")\n";
    }
    valueExprs[rev.getResult()] = bName;
    return success();
  }

  if (auto resh = dyn_cast<ReshapeOp>(op)) {
    auto rTy = resh.getType();
    std::string bName = freshName("buf");
    indent() << bName << " = convert(" << formatType(rTy) << ", collect(reshape("
             << lookup(resh.getOperand()) << ", "
             << formatShapeArgs(rTy.getShape(), resh.getDynamicDims())
             << ")))\n";
    valueExprs[resh.getResult()] = bName;
    return success();
  }

  if (auto sl = dyn_cast<SliceOp>(op)) {
    auto rTy = sl.getType();
    std::string bName = freshName("buf");
    std::string vExpr = buildViewSliceExpr(
        sl.getOperand(), sl.getStartIndices(), sl.getLimitIndices(),
        sl.getStrides());
    indent() << bName << " = convert(" << formatType(rTy) << ", collect("
             << vExpr << "))\n";
    valueExprs[sl.getResult()] = bName;
    return success();
  }

  if (auto dsl = dyn_cast<DynamicSliceOp>(op)) {
    auto rTy = dsl.getType();
    std::string bName = freshName("buf");
    indent() << bName << " = " << formatType(rTy) << "(undef, "
             << formatShapeArgs(rTy.getShape(), {}) << ")\n";
    valueExprs[dsl.getResult()] = bName;
    return emitDynamicSliceCopy(dsl.getResult(), dsl.getOperand(),
                                dsl.getStartIndices(), dsl.getSliceSizes());
  }

  if (auto dus = dyn_cast<DynamicUpdateSliceOp>(op)) {
    std::string bName = freshName("buf");
    indent() << bName << " = copy(" << lookup(dus.getOperand()) << ")\n";
    valueExprs[dus.getResult()] = bName;
    return emitDynamicUpdateSlice(dus.getResult(), dus.getUpdate(),
                                  dus.getStartIndices());
  }

  if (auto cat = dyn_cast<ConcatOp>(op)) {
    std::string bName = freshName("buf");
    indent() << bName << " = cat(";
    llvm::interleaveComma(cat.getInputs(), os,
                          [&](Value v) { os << lookup(v); });
    os << "; dims=" << (cat.getDimension() + 1) << ")\n";
    valueExprs[cat.getResult()] = bName;
    return success();
  }

  if (auto pop = dyn_cast<PadOp>(op)) {
    auto rTy = pop.getType();
    std::string bName = freshName("buf");
    indent() << bName << " = " << formatType(rTy) << "(undef, "
             << formatShapeArgs(rTy.getShape(), {}) << ")\n";
    valueExprs[pop.getResult()] = bName;
    return emitPad(pop.getResult(), pop.getOperand(), pop.getPaddingValue(),
                   pop.getEdgePaddingLow(), pop.getEdgePaddingHigh(),
                   pop.getInteriorPadding());
  }

  if (auto rw = dyn_cast<ReduceWindowOp>(op)) {
    SmallVector<Value> outs;
    for (Value r : rw.getResults()) {
      auto rTy = cast<ArrayType>(r.getType());
      std::string bName = freshName("buf");
      indent() << bName << " = " << formatType(rTy) << "(undef, "
               << formatShapeArgs(rTy.getShape(), {}) << ")\n";
      valueExprs[r] = bName;
      outs.push_back(r);
    }
    return emitReduceWindow(outs, rw.getInputs(), rw.getInitValues(),
                            rw.getWindowDimensions(), rw.getWindowStrides(),
                            rw.getBaseDilations(), rw.getWindowDilations(),
                            rw.getPadding(), rw.getCombiner());
  }

  if (auto sc = dyn_cast<ScatterOp>(op)) {
    SmallVector<Value> outs;
    for (auto [inVal, resVal] : llvm::zip(sc.getInputs(), sc.getResults())) {
      std::string bName = freshName("buf");
      indent() << bName << " = copy(" << lookup(inVal) << ")\n";
      valueExprs[resVal] = bName;
      outs.push_back(resVal);
    }
    return emitScatter(
        outs, sc.getScatterIndices(), sc.getUpdates(), sc.getUpdateWindowDims(),
        sc.getInsertedWindowDims(), sc.getInputBatchingDims(),
        sc.getScatterIndicesBatchingDims(), sc.getScatterDimsToOperandDims(),
        sc.getIndexVectorDim(), sc.getUpdateComputation());
  }

  if (auto sop = dyn_cast<SortOp>(op)) {
    SmallVector<Value> outs;
    for (auto [inVal, resVal] : llvm::zip(sop.getInputs(), sop.getResults())) {
      std::string bName = freshName("buf");
      indent() << bName << " = copy(" << lookup(inVal) << ")\n";
      valueExprs[resVal] = bName;
      outs.push_back(resVal);
    }
    return emitSort(outs, sop.getDimension(), sop.getIsStable(),
                    sop.getComparator());
  }

  if (auto cv = dyn_cast<ConvOp>(op)) {
    auto rTy = cv.getType();
    std::string bName = freshName("buf");
    indent() << bName << " = " << formatType(rTy) << "(undef, "
             << formatShapeArgs(rTy.getShape(), {}) << ")\n";
    valueExprs[cv.getResult()] = bName;
    return emitConv(
        cv.getResult(), cv.getLhs(), cv.getRhs(), cv.getWindowStrides(),
        cv.getPadding(), cv.getLhsDilation(), cv.getRhsDilation(),
        cv.getWindowReversal(), cv.getInputBatchDimension(),
        cv.getInputFeatureDimension(), cv.getInputSpatialDimensions(),
        cv.getKernelInputFeatureDimension(),
        cv.getKernelOutputFeatureDimension(), cv.getKernelSpatialDimensions(),
        cv.getOutputBatchDimension(), cv.getOutputFeatureDimension(),
        cv.getOutputSpatialDimensions(), cv.getFeatureGroupCount(),
        cv.getBatchGroupCount());
  }

  if (auto ch = dyn_cast<CholeskyOp>(op)) {
    std::string bName = freshName("buf");
    indent() << bName << " = copy(" << lookup(ch.getOperand()) << ")\n";
    valueExprs[ch.getResult()] = bName;
    return emitCholesky(ch.getResult(), ch.getLower());
  }

  if (auto ts = dyn_cast<TriangularSolveOp>(op)) {
    std::string bName = freshName("buf");
    indent() << bName << " = copy(" << lookup(ts.getB()) << ")\n";
    valueExprs[ts.getResult()] = bName;
    return emitTriangularSolve(ts.getResult(), ts.getA(), ts.getLeftSide(),
                               ts.getLower(), ts.getUnitDiagonal(),
                               ts.getTransposeA());
  }

  if (auto callOp = dyn_cast<CallOp>(op)) {
    std::string callee = sanitizeIdentifier(callOp.getCallee(), "fn");
    if (callOp.getNumResults() == 0) {
      indent() << callee << "(";
      llvm::interleaveComma(callOp.getOperands(), os,
                            [&](Value v) { os << lookup(v); });
      os << ")\n";
    } else if (callOp.getNumResults() == 1) {
      std::string rName = freshName("call");
      indent() << rName << " = " << callee << "(";
      llvm::interleaveComma(callOp.getOperands(), os,
                            [&](Value v) { os << lookup(v); });
      os << ")\n";
      valueExprs[callOp.getResult(0)] = rName;
    } else {
      SmallVector<std::string> rNames;
      for (Value r : callOp.getResults()) {
        std::string rName = freshName("call");
        rNames.push_back(rName);
        valueExprs[r] = rName;
      }
      indent() << "(";
      llvm::interleaveComma(rNames, os);
      os << ") = " << callee << "(";
      llvm::interleaveComma(callOp.getOperands(), os,
                            [&](Value v) { os << lookup(v); });
      os << ")\n";
    }
    return success();
  }

  if (auto ifOp = dyn_cast<IfOp>(op))
    return emitIf(ifOp);

  if (auto whileOp = dyn_cast<WhileOp>(op))
    return emitWhile(whileOp);

  return op->emitOpError("unsupported operation in TranslateToJulia");
}

LogicalResult JuliaEmitter::emitBlockStatements(Block &block,
                                                Operation *&terminatorOut) {
  for (Operation &op : block.without_terminator()) {
    if (failed(emitOperation(&op)))
      return failure();
  }
  terminatorOut = block.getTerminator();
  return success();
}

LogicalResult JuliaEmitter::emitFunc(FuncOp func) {
  if (func.isExternal())
    return success();

  valueExprs.clear();
  usedNames.clear();
  nextId = 0;

  std::string fnName = sanitizeIdentifier(func.getName(), "func");
  usedNames.insert(fnName);

  indent() << "function " << fnName << "(";
  Block &entry = func.getBody().front();
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    if (i > 0)
      os << ", ";
    Value arg = entry.getArgument(i);
    std::string aName = registerNamed(arg, "arg" + std::to_string(i));
    os << aName << "::" << formatType(arg.getType());
  }
  os << ")\n";

  indentLevel++;
  Operation *term = nullptr;
  if (failed(emitBlockStatements(entry, term)))
    return failure();

  auto retOp = cast<ReturnOp>(term);
  if (retOp.getNumOperands() == 0) {
    indent() << "return nothing\n";
  } else if (retOp.getNumOperands() == 1) {
    indent() << "return " << lookup(retOp.getOperand(0)) << "\n";
  } else {
    indent() << "return (";
    llvm::interleaveComma(retOp.getOperands(), os,
                          [&](Value v) { os << lookup(v); });
    os << ")\n";
  }

  indentLevel--;
  indent() << "end\n";
  return success();
}

LogicalResult JuliaEmitter::emitModule(ModuleOp module) {
  bool needsLinearAlgebra = false;
  bool needsBFloat16 = false;

  auto checkType = [&](Type t) {
    if (isa<BFloat16Type>(t))
      needsBFloat16 = true;
    if (auto sh = dyn_cast<ShapedType>(t))
      if (isa<BFloat16Type>(sh.getElementType()))
        needsBFloat16 = true;
  };

  module.walk([&](Operation *op) {
    if (isa<MatmulInPlaceOp, CholeskyOp, CholeskyInPlaceOp, TriangularSolveOp,
            TriangularSolveInPlaceOp>(op)) {
      needsLinearAlgebra = true;
    } else if (auto mm = dyn_cast<MatmulOp>(op)) {
      if (isJuliaBufferType(mm.getType()))
        needsLinearAlgebra = true;
    }
    for (Type t : op->getOperandTypes())
      checkType(t);
    for (Type t : op->getResultTypes())
      checkType(t);
    if (auto fn = dyn_cast<FuncOp>(op)) {
      for (Type t : fn.getArgumentTypes())
        checkType(t);
      for (Type t : fn.getResultTypes())
        checkType(t);
    }
  });

  if (needsLinearAlgebra)
    os << "import LinearAlgebra\n\n";
  if (needsBFloat16) {
    os << "if !@isdefined(BFloat16) && isdefined(Core, :BFloat16)\n"
          "    const BFloat16 = Core.BFloat16\n"
          "end\n\n";
  }

  bool first = true;
  for (FuncOp fn : module.getOps<FuncOp>()) {
    if (fn.isExternal())
      continue;
    if (!first)
      os << "\n";
    first = false;
    if (failed(emitFunc(fn)))
      return failure();
  }
  return success();
}

} // namespace

LogicalResult mlir::enzymexla::julia::translateToJulia(Operation *op,
                                                       llvm::raw_ostream &os) {
  JuliaEmitter emitter(os);
  if (auto mod = dyn_cast<ModuleOp>(op))
    return emitter.emitModule(mod);
  if (auto fn = dyn_cast<FuncOp>(op))
    return emitter.emitFunc(fn);
  return op->emitError("expected `builtin.module` or `julia.func`");
}

void mlir::enzymexla::julia::registerToJuliaTranslation() {
  TranslateFromMLIRRegistration reg(
      "mlir-to-julia", "Translate Julia dialect IR to native Julia source",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToJulia(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<JuliaDialect, func::FuncDialect>();
      });
}
