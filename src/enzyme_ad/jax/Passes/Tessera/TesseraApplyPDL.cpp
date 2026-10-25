//===----------------------------------------------------------------------===//
//
// Applies the PDL patterns built from the rewrite rules, replacing each match
// outright or with a tessera.guard when part of its condition is unproven.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Predicates.h"
#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/xxhash.h"
#include <memory>
#include <variant>

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAAPPLYPDLPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

static LogicalResult isConstantEqualTo(PatternRewriter &rewriter,
                                       PDLResultList &results,
                                       ArrayRef<PDLValue> args) {
  // args: the matched op, and the expected IntegerAttr. The pattern doesn't
  // pin the op name, so this matches llvm.mlir.constant and arith.constant.
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APInt actual;
  if (!matchPattern(constOp->getResult(0), m_ConstantInt(&actual)))
    return failure();

  auto expectedIntAttr = dyn_cast<IntegerAttr>(expectedAttr);
  if (!expectedIntAttr)
    return failure();

  // Compare values, ignoring bit width.
  if (actual.getSExtValue() != expectedIntAttr.getValue().getSExtValue())
    return failure();

  return success();
}

static LogicalResult isFloatConstantEqualTo(PatternRewriter &rewriter,
                                            PDLResultList &results,
                                            ArrayRef<PDLValue> args) {
  // args: the matched op, and the expected f64 FloatAttr. As above.
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APFloat actual(0.0);
  if (!matchPattern(constOp->getResult(0), m_ConstantFloat(&actual)))
    return failure();

  auto expectedFloatAttr = dyn_cast<FloatAttr>(expectedAttr);
  if (!expectedFloatAttr)
    return failure();

  // Round to the constant's float type, then compare exactly.
  llvm::APFloat expected = expectedFloatAttr.getValue();
  bool losesInfo = false;
  expected.convert(actual.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                   &losesInfo);

  // Bitwise, so -0.0 != +0.0 and NaN matches NaN.
  if (!actual.bitwiseIsEqual(expected))
    return failure();

  return success();
}

// The rules already applied to an op, so the call a guard keeps in its else
// region isn't matched by the same rule again, forever. Other rules can still
// match it.
static constexpr llvm::StringLiteral kAppliedRulesAttr =
    "tessera.applied_rules";

// On a function no rule may rewrite inside, from the tessera_no_rewrite
// annotation.
static constexpr llvm::StringLiteral kNoRewriteAttr = "tessera.no_rewrite";

static void markRuleApplied(Operation *op, StringAttr rule) {
  SmallVector<Attribute> applied;
  if (auto existing = op->getAttrOfType<ArrayAttr>(kAppliedRulesAttr))
    applied.assign(existing.getValue().begin(), existing.getValue().end());
  applied.push_back(rule);
  op->setAttr(kAppliedRulesAttr, ArrayAttr::get(op->getContext(), applied));
}

static LogicalResult tesseraRuleNotApplied(PatternRewriter &rewriter,
                                           PDLResultList &results,
                                           ArrayRef<PDLValue> args) {
  Operation *op = args[0].cast<Operation *>();
  auto rule = dyn_cast<StringAttr>(args[1].cast<Attribute>());
  if (!rule)
    return failure();
  auto applied = op->getAttrOfType<ArrayAttr>(kAppliedRulesAttr);
  return success(!applied ||
                 !llvm::is_contained(applied.getValue(), Attribute(rule)));
}

static std::string calleeName(const Call &call) {
  return call.dialect + "." + call.opname;
}

static DefineOp lookupDefine(Operation *anchor, llvm::StringRef name) {
  return SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      anchor, StringAttr::get(anchor->getContext(), name));
}

// How a matched op is named in a diagnostic: by callee for a call.
static std::string describeOp(Operation *op) {
  if (auto call = dyn_cast<CallOp>(op))
    return call.getCallee().str();
  return op->getName().getStringRef().str();
}

static void collectCallees(const Expr &expr, llvm::StringSet<> &callees) {
  if (auto *call = std::get_if<Call>(&expr.data)) {
    if (!call->isBuiltin())
      callees.insert(calleeName(*call));
    for (const Expr &arg : call->args)
      collectCallees(arg, callees);
  }
  if (auto *arith = std::get_if<Arith>(&expr.data)) {
    collectCallees(*arith->lhs, callees);
    collectCallees(*arith->rhs, callees);
  }
}

//===----------------------------------------------------------------------===//
// What the match bound
//===----------------------------------------------------------------------===//

// Argument i of a left-hand-side call is operand i of the op it matched, so
// the match can be recovered from the root and the AST.

// Whether `op` matches the left-hand side, binding each variable. The pattern
// checks this too, but PDL may run tesseraRuleApplicable before its own
// checks, and that mustn't warn about an op the pattern will reject.
static bool matchLhs(const Expr &expr, Operation *op,
                     llvm::StringMap<Value> &bound) {
  auto *call = std::get_if<Call>(&expr.data);
  auto callOp = dyn_cast_or_null<CallOp>(op);
  if (!call || !callOp || callOp.getCallee() != calleeName(*call) ||
      op->getNumOperands() != call->args.size())
    return false;
  for (auto [index, arg] : llvm::enumerate(call->args)) {
    Value operand = op->getOperand(index);
    bool matched = std::visit(
        overloaded{
            [&](const Var &v) {
              auto [it, inserted] = bound.try_emplace(v.name, operand);
              return inserted || it->second == operand;
            },
            [&](const IntLit &n) {
              llvm::APInt value;
              return matchPattern(operand, m_ConstantInt(&value)) &&
                     value.getSExtValue() == n.value;
            },
            [&](const FloatLit &n) {
              llvm::APFloat value(0.0);
              if (!matchPattern(operand, m_ConstantFloat(&value)))
                return false;
              llvm::APFloat expected(n.value);
              bool losesInfo = false;
              expected.convert(value.getSemantics(),
                               llvm::APFloat::rmNearestTiesToEven, &losesInfo);
              return value.bitwiseIsEqual(expected);
            },
            [&](const StrLit &) { return false; },
            [&](const GlobalRef &) { return false; },
            [&](const Call &) {
              return matchLhs(arg, operand.getDefiningOp(), bound);
            },
            [&](const Arith &) { return false; },
        },
        arg.data);
    if (!matched)
      return false;
  }
  return true;
}

// The calls the left-hand side matched beneath its root, deepest first.
static void collectProducers(const Expr &expr, Operation *op,
                             llvm::SetVector<Operation *> &producers) {
  auto *call = std::get_if<Call>(&expr.data);
  if (!call || !op || op->getNumOperands() != call->args.size())
    return;
  for (auto [index, arg] : llvm::enumerate(call->args)) {
    if (!std::holds_alternative<Call>(arg.data))
      continue;
    Operation *producer = op->getOperand(index).getDefiningOp();
    collectProducers(arg, producer, producers);
    if (producer)
      producers.insert(producer);
  }
}

//===----------------------------------------------------------------------===//
// Moving a matched chain
//===----------------------------------------------------------------------===//

// In `f(g(x)) -> f'(g'(x))`, the producer g must go too, or a side-effecting
// g (like an FEM kernel accumulating into a buffer) would still run alongside
// g'. So producers move with the root: they're erased after it's replaced,
// and a guard's else region recomputes them.

struct ChainPlan {
  // Producers that move with the root, in block order.
  SmallVector<Operation *> moved;
};

// Decide which producers move with `root`: those only the chain uses, in the
// root's block. A pure one that can't move just stays; a side-effecting one
// blocks the rewrite. Returns why, if it's blocked.
static std::optional<std::string> planChain(Operation *root, const Rule &rule,
                                            ChainPlan &plan) {
  llvm::SetVector<Operation *> producers;
  collectProducers(rule.lhs, root, producers);

  llvm::SmallPtrSet<Operation *, 8> moving;
  moving.insert(root);

  // Users before producers.
  for (Operation *producer : llvm::reverse(producers)) {
    bool onlyChainUses = llvm::all_of(
        producer->getUsers(), [&](Operation *u) { return moving.contains(u); });
    if (onlyChainUses && producer->getBlock() == root->getBlock()) {
      moving.insert(producer);
      plan.moved.push_back(producer);
      continue;
    }
    if (isMemoryEffectFree(producer))
      continue;

    std::string why;
    llvm::raw_string_ostream os(why);
    os << "the matched call to '" << describeOp(producer)
       << "' has side effects and ";
    if (!onlyChainUses)
      os << "its results are also used outside the matched expression";
    else
      os << "is not in the same block as the call it feeds";
    return why;
  }

  llvm::sort(plan.moved,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  // A side-effecting producer can't move past other side effects.
  for (Operation *producer : plan.moved) {
    if (isMemoryEffectFree(producer))
      continue;
    for (Operation *op = producer->getNextNode(); op && op != root;
         op = op->getNextNode()) {
      if (moving.contains(op) || isMemoryEffectFree(op))
        continue;
      std::string why;
      llvm::raw_string_ostream os(why);
      os << "the matched call to '" << describeOp(producer)
         << "' has side effects and would have to move past '" << describeOp(op)
         << "', which has side effects of its own";
      return why;
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Checking and building the right-hand side
//===----------------------------------------------------------------------===//

// A literal's type: `expected` (the parameter or result type) if it fits the
// kind, else picked from the value.
static Type literalType(const Expr &literal, Type expected, OpBuilder &b) {
  if (auto *n = std::get_if<IntLit>(&literal.data)) {
    if (isa_and_nonnull<IntegerType>(expected))
      return expected;
    return getIntegerAttrForLiteral(b, n->value).getType();
  }
  auto *n = std::get_if<FloatLit>(&literal.data);
  if (isa_and_nonnull<FloatType>(expected))
    return expected;
  // f32 if the value survives the round trip (0.5 does), else f64 (pi doesn't).
  if (static_cast<double>(static_cast<float>(n->value)) == n->value)
    return b.getF32Type();
  return b.getF64Type();
}

// The type a tessera.call to `define` expects at operand `index`: the pointee
// of a lifted argument, else the parameter type.
static Type callOperandValueType(DefineOp define, unsigned index) {
  if (Type pointee = define.getCallOperandPointeeType(index))
    return pointee;
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  return raw ? define.getFunctionType().getInput(*raw) : Type();
}

// Whether a tessera.call to `define` takes type `actual` at operand `index`:
// the parameter type, or a lifted argument's exact pointee type. (Stricter
// than CallOp::verifySymbolUses.)
static bool callOperandAccepts(DefineOp define, unsigned index, Type actual) {
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  if (!raw)
    return false;
  Type formal = define.getFunctionType().getInput(*raw);
  if (actual == formal)
    return true;
  Type pointee = define.getCallOperandPointeeType(index);
  return pointee && isa<LLVM::LLVMPointerType>(formal) && actual == pointee;
}

// The global a rule names, or null.
static LLVM::GlobalOp lookupGlobal(const GlobalRef &global, Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  return module ? module.lookupSymbol<LLVM::GlobalOp>(global.symbol)
                : LLVM::GlobalOp();
}

static bool literalFits(const Expr &literal, Type type) {
  auto *n = std::get_if<IntLit>(&literal.data);
  auto intType = dyn_cast<IntegerType>(type);
  if (!n || !intType)
    return true;
  unsigned width = intType.getWidth();
  return width >= 64 || llvm::isIntN(width, n->value) ||
         (n->value >= 0 && llvm::isUIntN(width, n->value));
}

// The types the right-hand side would produce, without building it. Fails,
// setting `why`, if it wouldn't verify (a missing callee, a wrong type).
static LogicalResult checkRhsCall(const Call &call, Operation *anchor,
                                  const llvm::StringMap<Value> &bound,
                                  SmallVectorImpl<Type> &resultTypes,
                                  std::string &why);

static Type checkRhsValue(const Expr &expr, Operation *anchor,
                          const llvm::StringMap<Value> &bound, Type expected,
                          std::string &why) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> Type {
            if (Value value = bound.lookup(v.name))
              return value.getType();
            why = "'" + v.name + "' is not bound by the left-hand side";
            return Type();
          },
          [&](const IntLit &) -> Type {
            OpBuilder b(anchor->getContext());
            return literalType(expr, expected, b);
          },
          [&](const FloatLit &) -> Type {
            OpBuilder b(anchor->getContext());
            return literalType(expr, expected, b);
          },
          [&](const StrLit &) -> Type {
            return LLVM::LLVMPointerType::get(anchor->getContext());
          },
          [&](const GlobalRef &g) -> Type {
            if (!lookupGlobal(g, anchor)) {
              why = "global '@" + g.symbol + "' is not in this module";
              return Type();
            }
            return LLVM::LLVMPointerType::get(anchor->getContext());
          },
          [&](const Call &c) -> Type {
            // log2 gives a count of the same width as its argument.
            if (c.isBuiltin()) {
              Type type =
                  checkRhsValue(c.args.front(), anchor, bound, expected, why);
              if (type && !isa<IntegerType>(type)) {
                llvm::raw_string_ostream os(why);
                os << "log2 takes an integer, but is given a value of type "
                   << type;
                return Type();
              }
              return type;
            }
            SmallVector<Type> results;
            if (failed(checkRhsCall(c, anchor, bound, results, why)))
              return Type();
            if (results.empty()) {
              why = "'" + calleeName(c) +
                    "' is used as an argument, but produces no result";
              return Type();
            }
            // A nested call denotes its first result.
            return results.front();
          },
          [&](const Arith &a) -> Type {
            // A literal takes the other operand's type (`2 * p` is in p's
            // type), or `expected` if both are literals. buildExprIR uses the
            // same order.
            Type lhsType, rhsType;
            if (isLiteralOnly(*a.lhs) && !isLiteralOnly(*a.rhs)) {
              rhsType = checkRhsValue(*a.rhs, anchor, bound, expected, why);
              if (!rhsType)
                return Type();
              lhsType = checkRhsValue(*a.lhs, anchor, bound, rhsType, why);
            } else {
              lhsType = checkRhsValue(*a.lhs, anchor, bound, expected, why);
              if (!lhsType)
                return Type();
              rhsType = checkRhsValue(*a.rhs, anchor, bound,
                                      isLiteralOnly(*a.lhs) ? expected
                                                            : lhsType,
                                      why);
            }
            if (!lhsType || !rhsType)
              return Type();
            std::string op(getArithOpSpelling(a.op));
            if (lhsType != rhsType) {
              llvm::raw_string_ostream os(why);
              os << "operands of '" << op << "' have different types ("
                 << lhsType << " and " << rhsType << ")";
              return Type();
            }
            if (!isa<IntegerType, FloatType>(lhsType)) {
              llvm::raw_string_ostream os(why);
              os << "'" << op << "' is given values of type " << lhsType
                 << ", but arithmetic is on integers and floats";
              return Type();
            }
            for (const Expr *operand : {a.lhs.get(), a.rhs.get()})
              if (!literalFits(*operand, lhsType)) {
                llvm::raw_string_ostream os(why);
                os << "literal " << renderExpr(*operand) << " does not fit in "
                   << lhsType;
                return Type();
              }
            auto *divisor = std::get_if<IntLit>(&a.rhs->data);
            if ((a.op == ArithOp::Div || a.op == ArithOp::Rem) && divisor &&
                divisor->value == 0) {
              why = "'" + renderExpr(expr) + "' divides by zero";
              return Type();
            }
            return lhsType;
          },
      },
      expr.data);
}

static LogicalResult checkRhsCall(const Call &call, Operation *anchor,
                                  const llvm::StringMap<Value> &bound,
                                  SmallVectorImpl<Type> &resultTypes,
                                  std::string &why) {
  std::string name = calleeName(call);
  DefineOp define = lookupDefine(anchor, name);
  if (!define) {
    why = "its right-hand side calls '" + name +
          "', which has no tessera.define in this module (if it is an inline "
          "helper, nothing in this translation unit caused it to be emitted)";
    return failure();
  }
  if (call.args.size() != define.getNumCallOperands()) {
    why = "its right-hand side calls '" + name + "' with " +
          std::to_string(call.args.size()) + " argument(s), but it takes " +
          std::to_string(define.getNumCallOperands());
    return failure();
  }
  for (auto [index, arg] : llvm::enumerate(call.args)) {
    Type expected = callOperandValueType(define, index);
    Type actual = checkRhsValue(arg, anchor, bound, expected, why);
    if (!actual)
      return failure();
    if (!literalFits(arg, actual)) {
      why = "literal " + renderExpr(arg) + " passed as argument " +
            std::to_string(index) + " of '" + name + "' does not fit in its " +
            "parameter type";
      return failure();
    }
    if (!callOperandAccepts(define, index, actual)) {
      llvm::raw_string_ostream os(why);
      os << "argument " << index << " of '" << name << "' expects "
         << define.getFunctionType().getInput(
                *define.getArgIndexForCallOperand(index))
         << ", but the rule passes it a value of type " << actual;
      return failure();
    }
  }
  resultTypes.clear();
  llvm::append_range(resultTypes, define.getCallResultTypes());
  return success();
}

// Check that the right-hand side can replace `root` as a whole.
static LogicalResult checkRhs(const Rule &rule, Operation *root,
                              const llvm::StringMap<Value> &bound,
                              std::string &why) {
  // Calls made for their effect only need to be buildable.
  for (const Expr &expr : llvm::drop_end(rule.rhs)) {
    SmallVector<Type> results;
    if (failed(
            checkRhsCall(std::get<Call>(expr.data), root, bound, results, why)))
      return failure();
  }

  const Expr &result = rule.result();
  auto *call = std::get_if<Call>(&result.data);
  if (call && !call->isBuiltin()) {
    SmallVector<Type> results;
    if (failed(checkRhsCall(*call, root, bound, results, why)))
      return failure();
    if (!llvm::equal(results, root->getResultTypes())) {
      llvm::raw_string_ostream os(why);
      os << "'" << calleeName(*call) << "' produces (";
      llvm::interleaveComma(results, os);
      os << "), but the call it replaces produces (";
      llvm::interleaveComma(root->getResultTypes(), os);
      os << ")";
      return failure();
    }
    return success();
  }

  if (root->getNumResults() != 1) {
    why = "its right-hand side is a single value, but the call it replaces "
          "produces " +
          std::to_string(root->getNumResults()) + " results";
    return failure();
  }
  Type expected = root->getResult(0).getType();
  Type actual = checkRhsValue(result, root, bound, expected, why);
  if (!actual)
    return failure();
  if (actual != expected) {
    llvm::raw_string_ostream os(why);
    os << "its right-hand side is a value of type " << actual
       << ", but the call it replaces produces " << expected;
    return failure();
  }
  return success();
}

// The address of a NUL-terminated constant holding `text`, shared by every
// rewrite that passes the same string.
static Value buildStringLiteral(llvm::StringRef text, OpBuilder &builder,
                                Location loc, Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  std::string name =
      "__tessera_str_" + llvm::utohexstr(llvm::xxh3_64bits(text));
  auto global = module.lookupSymbol<LLVM::GlobalOp>(name);
  if (!global) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    std::string data = text.str() + '\0';
    auto type = LLVM::LLVMArrayType::get(builder.getI8Type(), data.size());
    global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                    LLVM::Linkage::Private, name,
                                    builder.getStringAttr(data));
  }
  return LLVM::AddressOfOp::create(builder, loc, global);
}

// Build the IR for a right-hand-side expression, which checkRhs has already
// shown will verify.
static Value buildExprIR(const Expr &expr, OpBuilder &builder, Location loc,
                         const llvm::StringMap<Value> &boundVars,
                         Operation *symbolAnchor, Type expected,
                         Operation **builtOp = nullptr) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> Value { return boundVars.lookup(v.name); },
          [&](const IntLit &n) -> Value {
            Type type = literalType(expr, expected, builder);
            return LLVM::ConstantOp::create(
                builder, loc, type, builder.getIntegerAttr(type, n.value));
          },
          [&](const FloatLit &n) -> Value {
            Type type = literalType(expr, expected, builder);
            return LLVM::ConstantOp::create(
                builder, loc, type, builder.getFloatAttr(type, n.value));
          },
          [&](const StrLit &s) -> Value {
            return buildStringLiteral(s.value, builder, loc, symbolAnchor);
          },
          [&](const GlobalRef &g) -> Value {
            return LLVM::AddressOfOp::create(builder, loc,
                                             lookupGlobal(g, symbolAnchor));
          },
          [&](const Call &c) -> Value {
            // log2 of a power of two is its trailing zero count. Constants
            // are folded.
            if (c.isBuiltin()) {
              Value n = buildExprIR(c.args.front(), builder, loc, boundVars,
                                    symbolAnchor, expected);
              llvm::APInt value;
              if (matchPattern(n, m_ConstantInt(&value)))
                return LLVM::ConstantOp::create(
                    builder, loc, n.getType(),
                    builder.getIntegerAttr(n.getType(), value.countr_zero()));
              return LLVM::CountTrailingZerosOp::create(
                  builder, loc, n.getType(), n, /*is_zero_poison=*/false);
            }
            DefineOp define = lookupDefine(symbolAnchor, calleeName(c));
            SmallVector<Value> argValues;
            for (auto [index, arg] : llvm::enumerate(c.args))
              argValues.push_back(
                  buildExprIR(arg, builder, loc, boundVars, symbolAnchor,
                              callOperandValueType(define, index)));

            // Including the leading results for written arguments.
            auto call = CallOp::create(builder, loc, calleeName(c),
                                       define.getCallResultTypes(), argValues);
            if (builtOp)
              *builtOp = call;
            return call.getNumResults() ? call.getResult(0) : Value();
          },
          [&](const Arith &a) -> Value {
            // Same order as checkRhsValue.
            Value lhs, rhs;
            if (isLiteralOnly(*a.lhs) && !isLiteralOnly(*a.rhs)) {
              rhs = buildExprIR(*a.rhs, builder, loc, boundVars, symbolAnchor,
                                expected);
              lhs = buildExprIR(*a.lhs, builder, loc, boundVars, symbolAnchor,
                                rhs.getType());
            } else {
              lhs = buildExprIR(*a.lhs, builder, loc, boundVars, symbolAnchor,
                                expected);
              rhs = buildExprIR(*a.rhs, builder, loc, boundVars, symbolAnchor,
                                isLiteralOnly(*a.lhs) ? expected
                                                      : lhs.getType());
            }
            return buildArith(builder, loc, a.op, lhs, rhs);
          },
      },
      expr.data);
}

// Build the right-hand side and return the root's replacement. A call that
// repeats the match, as in `f(x) -> g(x); f(x)`, is marked so the rule doesn't
// match it again forever.
static SmallVector<Value> buildReplacement(const Rule &rule,
                                           StringAttr ruleAttr,
                                           OpBuilder &builder, Location loc,
                                           const llvm::StringMap<Value> &bound,
                                           Operation *root) {
  std::string lhsText = renderExpr(rule.lhs);
  auto build = [&](const Expr &expr, Type expected) {
    Operation *op = nullptr;
    Value value = buildExprIR(expr, builder, loc, bound, root, expected, &op);
    if (op && renderExpr(expr) == lhsText)
      markRuleApplied(op, ruleAttr);
    return std::make_pair(value, op);
  };

  for (const Expr &expr : llvm::drop_end(rule.rhs))
    build(expr, Type());

  const Expr &result = rule.result();
  Type expected =
      root->getNumResults() == 1 ? root->getResult(0).getType() : Type();
  auto [value, op] = build(result, expected);
  auto *call = std::get_if<Call>(&result.data);
  if (call && !call->isBuiltin())
    return SmallVector<Value>(op->getResults().begin(), op->getResults().end());
  return {value};
}

//===----------------------------------------------------------------------===//
// State shared by the native functions
//===----------------------------------------------------------------------===//

struct ApplyState {
  // Each rule parsed once; the text is what the patterns carry.
  llvm::DenseMap<Attribute, std::unique_ptr<Rule>> rules;

  // Ops called on some rule's right-hand side. No rule applies inside their
  // bodies: a replacement's fallback often calls the op it replaces, and
  // rewriting that would make it recurse forever.
  llvm::StringSet<> rhsCallees;

  // Warnings already given, so each is reported once.
  llvm::DenseSet<Attribute> warnedRules;
  llvm::DenseSet<std::pair<Attribute, Operation *>> warnedSites;

  const Rule *getRule(StringAttr text, Location loc) {
    auto &slot = rules[text];
    if (!slot) {
      Parser parser(text.getValue().str(), loc);
      auto rule = parser.parseRule();
      if (!rule)
        return nullptr;
      slot = std::make_unique<Rule>(std::move(*rule));
    }
    return slot.get();
  }

  bool isExcluded(Operation *op) {
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (parent->hasAttr(kNoRewriteAttr))
        return true;
      if (auto define = dyn_cast<DefineOp>(parent))
        if (rhsCallees.contains(define.getSymName()))
          return true;
    }
    return false;
  }
};

// Match-time check for every pattern. Declines, without touching the IR, if
// the match is in excluded code, the right-hand side wouldn't verify (warns),
// the condition is false here (remarks), or the chain can't move (warns).
static LogicalResult tesseraRuleApplicable(ApplyState &state, Operation *root,
                                           StringAttr ruleAttr) {
  const Rule *rule = state.getRule(ruleAttr, root->getLoc());
  if (!rule)
    return failure();

  if (state.isExcluded(root))
    return failure();

  llvm::StringMap<Value> bound;
  if (!matchLhs(rule->lhs, root, bound))
    return failure();

  std::string why;
  if (failed(checkRhs(*rule, root, bound, why))) {
    if (state.warnedRules.insert(ruleAttr).second)
      root->emitWarning() << "optimization rule '" << ruleAttr.getValue()
                          << "' cannot be applied: " << why;
    return failure();
  }

  // Declining here, not in the rewrite, keeps the driver from looping.
  if (rule->cond) {
    ResidualCondition residual = residualizeCondition(*rule->cond, bound);
    if (residual.settled == false) {
      if (state.warnedSites.insert({ruleAttr, root}).second)
        root->emitRemark() << "optimization rule '" << ruleAttr.getValue()
                           << "' was not applied here: "
                           << (residual.unshown.empty()
                                   ? "its condition is false here"
                                   : "could not show " + residual.unshown);
      return failure();
    }
  }

  ChainPlan plan;
  if (auto reason = planChain(root, *rule, plan)) {
    if (state.warnedSites.insert({ruleAttr, root}).second)
      root->emitWarning() << "optimization rule '" << ruleAttr.getValue()
                          << "' was not applied here: " << *reason;
    return failure();
  }
  return success();
}

// Record facts on the calls a rewrite just built, so later rules can use them.
// Cloned calls keep theirs.
template <typename Range> static void recordFactsOnNewCalls(Range &&ops) {
  PropertyAnalysis facts;
  for (Operation &op : ops)
    op.walk([&](CallOp call) { facts.recordFactsOnCall(call); });
}

// Rewrite for every pattern. args: the root, the rule text, the variable
// names, then their matched values. With no unproven condition left, replaces
// the root; otherwise builds a tessera.guard with the rewrite and the original
// code, for tessera-lower-guards. Never fails (the driver would abort), since
// tesseraRuleApplicable already ruled failures out.
static LogicalResult tesseraRewrite(ApplyState &state,
                                    PatternRewriter &rewriter,
                                    ArrayRef<PDLValue> args) {
  Operation *root = args[0].cast<Operation *>();
  auto ruleAttr = cast<StringAttr>(args[1].cast<Attribute>());
  auto namesAttr = cast<ArrayAttr>(args[2].cast<Attribute>());

  SmallVector<Value> values;
  for (size_t i = 3; i < args.size(); ++i)
    values.push_back(args[i].cast<Value>());

  Location loc = root->getLoc();
  const Rule *rule = state.getRule(ruleAttr, loc);
  assert(rule && "a pattern carries a rule that already parsed once");

  ChainPlan plan;
  [[maybe_unused]] auto reason = planChain(root, *rule, plan);
  assert(!reason && "tesseraRuleApplicable admitted a chain it cannot move");

  llvm::StringMap<Value> boundVars;
  for (auto [nameAttr, value] : llvm::zip(namesAttr.getValue(), values))
    boundVars[cast<StringAttr>(nameAttr).getValue()] = value;

  rewriter.setInsertionPoint(root);

  // Replace the root, then erase the moved producers, latest first.
  auto replaceRoot = [&](ValueRange replacement) {
    rewriter.replaceOp(root, replacement);
    for (Operation *producer : llvm::reverse(plan.moved))
      if (producer->use_empty())
        rewriter.eraseOp(producer);
  };

  // Fold what the IR settles. If nothing is left, apply the rewrite outright;
  // otherwise the guard tests only the rest, which never includes a property
  // without a runtime check like `SPD`.
  std::optional<Cond> residual;
  if (rule->cond) {
    ResidualCondition folded = residualizeCondition(*rule->cond, boundVars);
    assert(folded.settled != false &&
           "tesseraRuleApplicable admitted a condition that is false here");
    residual = std::move(folded.cond);
  }
  if (!residual) {
    Operation *before = root->getPrevNode();
    SmallVector<Value> replacement =
        buildReplacement(*rule, ruleAttr, rewriter, loc, boundVars, root);
    Block::iterator first =
        before ? std::next(before->getIterator()) : root->getBlock()->begin();
    recordFactsOnNewCalls(llvm::make_range(first, root->getIterator()));
    replaceRoot(replacement);
    return success();
  }

  auto guard = GuardOp::create(rewriter, loc, root->getResultTypes(),
                               rewriter.getStringAttr(renderCond(*residual)),
                               namesAttr, values);

  // Then: the rule's right-hand side.
  {
    Block *block = rewriter.createBlock(&guard.getThenRegion());
    rewriter.setInsertionPointToStart(block);
    YieldOp::create(
        rewriter, loc,
        buildReplacement(*rule, ruleAttr, rewriter, loc, boundVars, root));
    recordFactsOnNewCalls(*block);
  }

  // Else: the original call and its moved producers.
  {
    Block *block = rewriter.createBlock(&guard.getElseRegion());
    rewriter.setInsertionPointToStart(block);
    IRMapping mapping;
    for (Operation *producer : plan.moved)
      rewriter.clone(*producer, mapping);
    Operation *clone = rewriter.clone(*root, mapping);
    markRuleApplied(clone, ruleAttr);
    YieldOp::create(rewriter, loc, clone->getResults());
  }

  replaceRoot(guard.getResults());
  return success();
}

struct TesseraApplyPDLPass
    : public enzyme::tessera::impl::TesseraApplyPDLPassBase<
          TesseraApplyPDLPass> {
  using TesseraApplyPDLPassBase::TesseraApplyPDLPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleOp patternModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "patterns"));

    if (!patternModule)
      return;

    if (patternModule.getBody()->getOperations().empty()) {
      patternModule.getOperation()->erase();
      return;
    }

    // Collect every rule's right-hand-side callees first (see rhsCallees).
    ApplyState state;
    patternModule.walk([&](pdl::ApplyNativeConstraintOp constraint) {
      if (constraint.getName() != "tesseraRuleApplicable" ||
          constraint.getArgs().size() < 2)
        return;
      if (auto attrOp =
              constraint.getArgs()[1].getDefiningOp<pdl::AttributeOp>())
        if (auto text = dyn_cast_or_null<StringAttr>(attrOp.getValueAttr()))
          if (const Rule *rule = state.getRule(text, constraint.getLoc()))
            for (const Expr &expr : rule->rhs)
              collectCallees(expr, state.rhsCallees);
    });

    RewritePatternSet patternList(module->getContext());

    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    pdlPattern.registerConstraintFunction("isConstantEqualTo",
                                          isConstantEqualTo);
    pdlPattern.registerConstraintFunction("isFloatConstantEqualTo",
                                          isFloatConstantEqualTo);
    pdlPattern.registerConstraintFunction("tesseraRuleNotApplied",
                                          tesseraRuleNotApplied);
    pdlPattern.registerConstraintFunction(
        "tesseraRuleApplicable",
        [&state](PatternRewriter &rewriter, PDLResultList &results,
                 ArrayRef<PDLValue> args) -> LogicalResult {
          auto rule = dyn_cast<StringAttr>(args[1].cast<Attribute>());
          if (!rule)
            return failure();
          return tesseraRuleApplicable(state, args[0].cast<Operation *>(),
                                       rule);
        });

    pdlPattern.registerRewriteFunction(
        "tesseraRewrite",
        [&state](PatternRewriter &rewriter, PDLResultList &results,
                 ArrayRef<PDLValue> args) -> LogicalResult {
          return tesseraRewrite(state, rewriter, args);
        });

    patternList.add(std::move(pdlPattern));

    if (failed(applyPatternsGreedily(
            module, std::move(patternList),
            GreedyRewriteConfig().setRegionSimplificationLevel(
                GreedySimplifyRegionLevel::Normal)))) {
      llvm::errs() << "Failed to apply PDL patterns\n";
      signalPassFailure();
    }
  }
};

} // namespace
