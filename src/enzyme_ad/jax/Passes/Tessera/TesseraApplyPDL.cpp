//===----------------------------------------------------------------------===//
//
// This file implements a pass to apply the PDL patterns created from the
// tessera optimization rewrite rules to the IR.
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
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
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
  // args[0]: the matched Operation* that should be producing a constant
  // args[1]: the expected value, passed as a PDL attribute (IntegerAttr)
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  // The pattern side deliberately does not pin the op name, so this is where
  // "is it a constant?" is actually decided. Match through the constant
  // interface rather than a concrete op class: the same rule has to fire both
  // on llvm.mlir.constant and on arith.constant.
  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APInt actual;
  if (!matchPattern(constOp->getResult(0), m_ConstantInt(&actual)))
    return failure();

  auto expectedIntAttr = dyn_cast<IntegerAttr>(expectedAttr);
  if (!expectedIntAttr)
    return failure();

  // Compare numeric value only, ignoring bit-width, so this stays robust
  // if the constant's width ever differs from what the annotation assumed.
  if (actual.getSExtValue() != expectedIntAttr.getValue().getSExtValue())
    return failure();

  return success();
}

static LogicalResult isFloatConstantEqualTo(PatternRewriter &rewriter,
                                            PDLResultList &results,
                                            ArrayRef<PDLValue> args) {
  // args[0]: the matched Operation* that should be producing a constant
  // args[1]: the expected value, passed as a PDL attribute (FloatAttr, always
  //          emitted as f64 by the rule parser)
  Operation *constOp = args[0].cast<Operation *>();
  auto expectedAttr = args[1].cast<Attribute>();

  // As in the integer case, the op name is not pinned on the pattern side, so
  // this is where "is it a float constant?" is decided — via the constant
  // interface, so the same rule fires on llvm.mlir.constant and arith.constant.
  if (constOp->getNumResults() != 1)
    return failure();

  llvm::APFloat actual(0.0);
  if (!matchPattern(constOp->getResult(0), m_ConstantFloat(&actual)))
    return failure();

  auto expectedFloatAttr = dyn_cast<FloatAttr>(expectedAttr);
  if (!expectedFloatAttr)
    return failure();

  // Round the expected value of the float into the semantics the matched
  // constant actually uses, then compare exactly.
  llvm::APFloat expected = expectedFloatAttr.getValue();
  bool losesInfo = false;
  expected.convert(actual.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                   &losesInfo);

  // bitwiseIsEqual rather than ==: it keeps +0.0 distinct from -0.0 and lets a
  // NaN literal match a NaN constant, where == would do the opposite on both.
  if (!actual.bitwiseIsEqual(expected))
    return failure();

  return success();
}

// Names the rules that have already been applied to an operation. The original
// call a guard keeps in its else region is a clone of the matched op, so
// without this the very same pattern would match it again and wrap it in
// another guard, forever. Recording the rule rather than marking the op
// off-limits outright keeps a *different* conditional rule free to specialize
// that same call, and keeps rules applying inside a guard's regions.
static constexpr llvm::StringLiteral kAppliedRulesAttr =
    "tessera.applied_rules";

// Marks a function whose body no rule may rewrite. Set by
// lift-tessera-annotations from
// `__attribute__((annotate("tessera_no_rewrite")))`.
static constexpr llvm::StringLiteral kNoRewriteAttr = "tessera.no_rewrite";

static bool hasRuleBeenApplied(Operation *op, StringAttr rule) {
  auto applied = op->getAttrOfType<ArrayAttr>(kAppliedRulesAttr);
  return applied && llvm::is_contained(applied.getValue(), Attribute(rule));
}

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
  return success(!hasRuleBeenApplied(op, rule));
}

static std::string calleeName(const Call &call) {
  return call.dialect + "." + call.opname;
}

static DefineOp lookupDefine(Operation *anchor, llvm::StringRef name) {
  return SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      anchor, StringAttr::get(anchor->getContext(), name));
}

/// How a matched op is named in a diagnostic: by callee for a call.
static std::string describeOp(Operation *op) {
  if (auto call = dyn_cast<CallOp>(op))
    return call.getCallee().str();
  return op->getName().getStringRef().str();
}

static void collectCallees(const Expr &expr, llvm::StringSet<> &callees) {
  if (auto *call = std::get_if<Call>(&expr.data)) {
    callees.insert(calleeName(*call));
    for (const Expr &arg : call->args)
      collectCallees(arg, callees);
  }
}

//===----------------------------------------------------------------------===//
// What the match bound
//===----------------------------------------------------------------------===//

// PDL matches a tessera.call's operands positionally against the arguments the
// rule wrote, so argument i of a Call node in the left-hand side is operand i
// of the op it matched. That is what lets the matched structure be recovered
// from the root and the AST alone.

/// Whether `op` really is what the left-hand side describes, binding every
/// variable to the value it matched.
///
/// The pattern checks all of this too, but PDL is free to evaluate a native
/// constraint before the pattern's own checks -- the callee, a literal, a
/// repeated variable -- so tesseraRuleApplicable can be handed an op the
/// pattern is about to reject. It must not report on one of those, and a
/// warning given once per rule must not be spent on one either.
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
            [&](const Call &) {
              return matchLhs(arg, operand.getDefiningOp(), bound);
            },
        },
        arg.data);
    if (!matched)
      return false;
  }
  return true;
}

/// The calls the left-hand side matched beneath its root, deepest first.
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

// A rule whose left-hand side nests calls, `f(g(x)) -> f'(g'(x))`, matches the
// producer g as well as the root f. Replacing only the root would leave g
// behind, still running unconditionally -- and if g has side effects, say it
// accumulates into an output buffer as every FEM element kernel does, it then
// runs on top of g' and the result is wrong. So the producers move with the
// root: they are erased once the root is replaced, and a guard's else region
// recomputes them alongside its copy of the root.
//
// Moving a producer down to the root is only sound when nothing observes the
// difference, which is what the plan below decides.

struct ChainPlan {
  /// Producers that move with the root, in block order.
  SmallVector<Operation *> moved;
};

/// Decide which matched producers move with `root`. Returns an explanation if
/// the rule cannot be applied here without changing what the program does.
///
/// A producer moves when the chain is its only user and it sits in the root's
/// block. One that cannot move may simply stay where it is if it is pure: the
/// rewritten code refers to it, and it computes the same thing either way. A
/// producer with side effects that cannot move blocks the rewrite, since the
/// new code would run in addition to it.
static std::optional<std::string> planChain(Operation *root, const Rule &rule,
                                            ChainPlan &plan) {
  llvm::SetVector<Operation *> producers;
  collectProducers(rule.lhs, root, producers);

  llvm::SmallPtrSet<Operation *, 8> moving;
  moving.insert(root);

  // Users before producers, so a producer is only judged once everything in
  // the chain that uses it has been.
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

  // A producer with side effects is being moved down to the root, so nothing
  // between the two may have side effects of its own: it could observe or
  // change what the producer does.
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

/// Erase the producers that moved with a root that has just been replaced.
/// Latest first, since each may be the last user of an earlier one.
static void eraseMovedProducers(PatternRewriter &rewriter,
                                const ChainPlan &plan) {
  for (Operation *producer : llvm::reverse(plan.moved))
    if (producer->use_empty())
      rewriter.eraseOp(producer);
}

//===----------------------------------------------------------------------===//
// Checking and building the right-hand side
//===----------------------------------------------------------------------===//

/// The type a literal is built at. A literal passed to a callee takes the type
/// of the parameter it is passed to, and one replacing the matched call takes
/// that call's result type; only without either is the width picked from the
/// literal's own magnitude.
static Type literalType(const Expr &literal, Type expected, OpBuilder &b) {
  if (auto *n = std::get_if<IntLit>(&literal.data)) {
    if (isa_and_nonnull<IntegerType>(expected))
      return expected;
    return getIntegerAttrForLiteral(b, n->value).getType();
  }
  auto *n = std::get_if<FloatLit>(&literal.data);
  if (isa_and_nonnull<FloatType>(expected))
    return expected;
  return getFloatAttrForLiteral(b, n->value).getType();
}

/// The type a tessera.call to `define` expects at operand `index`: the pointee
/// for an argument the call site loads, otherwise the parameter itself.
static Type callOperandValueType(DefineOp define, unsigned index) {
  if (Type pointee = define.getCallOperandPointeeType(index))
    return pointee;
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  return raw ? define.getFunctionType().getInput(*raw) : Type();
}

/// Whether a tessera.call to `define` can take a value of type `actual` at
/// operand `index`. This follows CallOp::verifySymbolUses, a little more
/// strictly: an argument the call site loads must be given either the pointer
/// itself or exactly the pointee type, where the verifier accepts anything.
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

static bool literalFits(const Expr &literal, Type type) {
  auto *n = std::get_if<IntLit>(&literal.data);
  auto intType = dyn_cast<IntegerType>(type);
  if (!n || !intType)
    return true;
  unsigned width = intType.getWidth();
  return width >= 64 || llvm::isIntN(width, n->value) ||
         (n->value >= 0 && llvm::isUIntN(width, n->value));
}

/// Work out, without building anything, what the right-hand side would
/// produce. Sets `why` and returns failure if it could not be built validly --
/// a callee with no tessera.define, or an argument of the wrong type -- which
/// would otherwise only surface as a verifier error on the whole module.
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
          [&](const Call &c) -> Type {
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

/// Check that the right-hand side can replace `root` as a whole.
static LogicalResult checkRhs(const Rule &rule, Operation *root,
                              const llvm::StringMap<Value> &bound,
                              std::string &why) {
  if (auto *call = std::get_if<Call>(&rule.rhs.data)) {
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
  Type actual = checkRhsValue(rule.rhs, root, bound, expected, why);
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

// Build the IR for the right-hand side of a rule. The matched values exist by
// the time a rule is applied, so the replacement is constructed directly, and
// checkRhs has already established that it will verify.
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
          [&](const Call &c) -> Value {
            DefineOp define = lookupDefine(symbolAnchor, calleeName(c));
            SmallVector<Value> argValues;
            for (auto [index, arg] : llvm::enumerate(c.args))
              argValues.push_back(
                  buildExprIR(arg, builder, loc, boundVars, symbolAnchor,
                              callOperandValueType(define, index)));

            // Result types come from the callee's declaration, including the
            // leading results its written arguments contribute, which is also
            // what makes a nested call on the right-hand side work.
            auto call = CallOp::create(builder, loc, calleeName(c),
                                       define.getCallResultTypes(), argValues);
            if (builtOp)
              *builtOp = call;
            return call.getNumResults() ? call.getResult(0) : Value();
          },
      },
      expr.data);
}

/// Build the right-hand side and return what replaces the root with it.
static SmallVector<Value> buildReplacement(const Rule &rule, OpBuilder &builder,
                                           Location loc,
                                           const llvm::StringMap<Value> &bound,
                                           Operation *root) {
  Type expected =
      root->getNumResults() == 1 ? root->getResult(0).getType() : Type();
  Operation *op = nullptr;
  Value value = buildExprIR(rule.rhs, builder, loc, bound, root, expected, &op);
  if (std::holds_alternative<Call>(rule.rhs.data))
    return SmallVector<Value>(op->getResults().begin(), op->getResults().end());
  return {value};
}

//===----------------------------------------------------------------------===//
// State shared by the native functions
//===----------------------------------------------------------------------===//

struct ApplyState {
  /// Each rule parsed once; the text is what the patterns carry.
  llvm::DenseMap<Attribute, std::unique_ptr<Rule>> rules;

  /// Every tessera op named on the right-hand side of some rule. No rule is
  /// applied inside the body of one of these: a replacement's fallback path
  /// commonly calls the very op it replaces, and rewriting that call would
  /// turn it into a call to the replacement itself, recursing forever on
  /// exactly the inputs the fallback exists for.
  llvm::StringSet<> rhsCallees;

  /// Warnings already given, so a rule the greedy driver tries at many sites,
  /// or at one site many times, is reported once.
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

  void addRule(StringAttr text, Location loc) {
    if (const Rule *rule = getRule(text, loc))
      collectCallees(rule->rhs, rhsCallees);
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

/// Match-time gate for every generated pattern. Declines, without touching
/// the IR, when:
///
///   - the match sits in code no rule may rewrite (see ApplyState);
///   - the right-hand side cannot be built into valid IR here, which is
///     reported, since the rule then silently never fires;
///   - the condition is settled false here, which is reported as a remark,
///     since it is the ordinary outcome for a call whose operand is not known
///     to have a property the rule needs;
///   - the matched chain cannot move with its root (see planChain), which is
///     reported as a warning.
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

  // Deciding this here rather than in the rewrite is what lets a rule decline
  // without touching the IR, so the driver cannot loop on it.
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

// Rewrite function for every generated pattern. PDL passes the matched root
// first, then the external arguments the pattern listed: the rule text, the
// names the rule uses, and the matched values those names refer to.
//
// An unconditional rule, or one whose condition is proven, replaces the root
// outright. Otherwise what remains of the condition is not evaluated here: it
// is recorded on a tessera.guard, holding the specialized rewrite and the
// original computation in its two regions, and -tessera-lower-guards later
// synthesizes the check and turns the guard into a branch.
//
// This never reports failure, though the signature PDL requires allows it: the
// greedy driver has no way to recover from a failed native rewrite and aborts
// the process instead. Everything that could make it fail was ruled out by
// tesseraRuleApplicable when the pattern matched.
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

  auto replaceRoot = [&](ValueRange replacement) {
    rewriter.replaceOp(root, replacement);
    eraseMovedProducers(rewriter, plan);
  };

  // Whatever the IR settles about the condition is folded away first. When
  // that settles it as true, there is nothing to test at run time: apply the
  // rewrite outright. This is a saving over the guarded form, never a
  // precondition for it -- a predicate with a runtime check that cannot be
  // proven still works, it just costs a check. A condition settled as false
  // never gets here: tesseraRuleApplicable declined it.
  //
  // Otherwise the guard tests only what is left. That is what keeps a
  // property with no runtime check, such as `SPD`, from ever reaching
  // -tessera-lower-guards: it is either shown here and folded to true, or
  // counted as whichever value makes the condition false.
  std::optional<Cond> residual;
  if (rule->cond) {
    ResidualCondition folded = residualizeCondition(*rule->cond, boundVars);
    assert(folded.settled != false &&
           "tesseraRuleApplicable admitted a condition that is false here");
    residual = std::move(folded.cond);
  }
  if (!residual) {
    replaceRoot(buildReplacement(*rule, rewriter, loc, boundVars, root));
    return success();
  }

  auto guard = GuardOp::create(rewriter, loc, root->getResultTypes(),
                               rewriter.getStringAttr(renderCond(*residual)),
                               namesAttr, values);

  // Specialized path: the rule's right-hand side.
  {
    Block *block = rewriter.createBlock(&guard.getThenRegion());
    rewriter.setInsertionPointToStart(block);
    YieldOp::create(rewriter, loc,
                    buildReplacement(*rule, rewriter, loc, boundVars, root));
  }

  // Original path: the matched call, unchanged, preceded by the producers that
  // move with it, so they run on this path only.
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

    // Collect every generated rule up front: which ops no rule may rewrite
    // inside depends on all of them, not only the one being matched.
    ApplyState state;
    patternModule.walk([&](pdl::ApplyNativeConstraintOp constraint) {
      if (constraint.getName() != "tesseraRuleApplicable" ||
          constraint.getArgs().size() < 2)
        return;
      if (auto attrOp =
              constraint.getArgs()[1].getDefiningOp<pdl::AttributeOp>())
        if (auto text = dyn_cast_or_null<StringAttr>(attrOp.getValueAttr()))
          state.addRule(text, constraint.getLoc());
    });

    RewritePatternSet patternList(module->getContext());

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    // Register native constraints referenced by generated PDL patterns.
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

    // Every generated rule builds its right-hand side through this.
    pdlPattern.registerRewriteFunction(
        "tesseraRewrite",
        [&state](PatternRewriter &rewriter, PDLResultList &results,
                 ArrayRef<PDLValue> args) -> LogicalResult {
          return tesseraRewrite(state, rewriter, args);
        });

    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
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
