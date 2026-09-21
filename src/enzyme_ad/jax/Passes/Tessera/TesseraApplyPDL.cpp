//===----------------------------------------------------------------------===//
//
// This file implements a pass to apply the PDL patterns created from the
// tessera optimization rewrite rules to the IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
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

// Build the IR for the right-hand side of a rule. This mirrors emitRewritePDL
// in ParseOptimizationRules.cpp, except that it constructs real operations
// rather than the PDL that would construct them: by the time a conditional
// rule is applied, the matched values exist, so the replacement can simply be
// built.
//
// Returns the produced operation for a call, or a null operation and the bound
// value for a bare variable on the right-hand side (the `f(f(x)) -> x` shape).
struct BuiltExpr {
  Value value;
  Operation *op = nullptr;
};

static BuiltExpr buildExprIR(const Expr &expr, OpBuilder &builder, Location loc,
                             const llvm::StringMap<Value> &boundVars,
                             Operation *symbolAnchor, TypeRange fallbackTypes) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> BuiltExpr {
            auto it = boundVars.find(v.name);
            if (it == boundVars.end()) {
              emitError(loc) << "optimization rule uses '" << v.name
                             << "' on the right-hand side, but it is not bound "
                                "on the left";
              return {};
            }
            return {it->second, nullptr};
          },
          [&](const IntLit &n) -> BuiltExpr {
            auto attr = getIntegerAttrForLiteral(builder, n.value);
            auto op =
                LLVM::ConstantOp::create(builder, loc, attr.getType(), attr);
            return {op.getResult(), op};
          },
          [&](const FloatLit &n) -> BuiltExpr {
            auto attr = getFloatAttrForLiteral(builder, n.value);
            auto op =
                LLVM::ConstantOp::create(builder, loc, attr.getType(), attr);
            return {op.getResult(), op};
          },
          [&](const Call &c) -> BuiltExpr {
            SmallVector<Value> argValues;
            for (const Expr &arg : c.args) {
              BuiltExpr built = buildExprIR(arg, builder, loc, boundVars,
                                            symbolAnchor, fallbackTypes);
              if (!built.value)
                return {};
              argValues.push_back(built.value);
            }

            // Result types come from the callee's own declaration rather than
            // being guessed, which is also what makes a nested call on the
            // right-hand side work. When the callee is not declared, fall back
            // to what the matched op produced and let the verifier report the
            // unknown symbol -- it does so more precisely than this could.
            std::string callee = c.dialect + "." + c.opname;
            auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
                symbolAnchor, StringAttr::get(builder.getContext(), callee));
            TypeRange resultTypes =
                define ? define.getFunctionType().getResults() : fallbackTypes;

            auto call =
                CallOp::create(builder, loc, callee, resultTypes, argValues);
            return {call.getNumResults() ? call.getResult(0) : Value(), call};
          },
      },
      expr.data);
}

// Rewrite function for a conditional rule. PDL passes the matched root first,
// then the external arguments the pattern listed: the rule text, the names the
// condition uses, and the matched values those names refer to.
//
// The condition is not evaluated here. It is recorded on a tessera.guard,
// holding the specialized rewrite and the original computation in its two
// regions, and -tessera-lower-guards later synthesizes the check and turns the
// guard into a branch.
//
// This never reports failure, though the signature PDL requires allows it: the
// greedy driver has no way to recover from a failed native rewrite and aborts
// the process instead. Everything this needs was fixed when the pattern was
// generated, and the one thing that is not -- whether the right-hand side
// names a real tessera.define -- is left to the verifier, which rejects a call
// to an unknown callee with a better message than anything available here.
static LogicalResult tesseraConditionalRewrite(PatternRewriter &rewriter,
                                               PDLResultList &results,
                                               ArrayRef<PDLValue> args) {
  Operation *root = args[0].cast<Operation *>();
  auto ruleAttr = cast<StringAttr>(args[1].cast<Attribute>());
  auto namesAttr = cast<ArrayAttr>(args[2].cast<Attribute>());

  SmallVector<Value> values;
  for (size_t i = 3; i < args.size(); ++i)
    values.push_back(args[i].cast<Value>());

  Location loc = root->getLoc();
  Parser parser(ruleAttr.getValue().str(), loc);
  auto rule = parser.parseRule();
  assert(rule && rule->cond &&
         "a conditional pattern carries a rule that already parsed once");

  llvm::StringMap<Value> boundVars;
  for (auto [nameAttr, value] : llvm::zip(namesAttr.getValue(), values))
    boundVars[cast<StringAttr>(nameAttr).getValue()] = value;

  // TODO: when the condition can be proven from the IR, build the right-hand
  // side directly here instead of a guard, so the check costs nothing.

  rewriter.setInsertionPoint(root);
  auto guard = GuardOp::create(rewriter, loc, root->getResultTypes(),
                               rewriter.getStringAttr(renderCond(*rule->cond)),
                               namesAttr, values);

  // Specialized path: the rule's right-hand side.
  {
    Block *block = rewriter.createBlock(&guard.getThenRegion());
    rewriter.setInsertionPointToStart(block);
    BuiltExpr built = buildExprIR(rule->rhs, rewriter, loc, boundVars, root,
                                  root->getResultTypes());
    SmallVector<Value> yielded;
    if (built.op && built.op->getNumResults())
      yielded.assign(built.op->getResults().begin(),
                     built.op->getResults().end());
    else if (built.value)
      yielded.push_back(built.value);
    YieldOp::create(rewriter, loc, yielded);
  }

  // Original path: the matched call, unchanged.
  {
    Block *block = rewriter.createBlock(&guard.getElseRegion());
    rewriter.setInsertionPointToStart(block);
    Operation *clone = rewriter.clone(*root);
    markRuleApplied(clone, ruleAttr);
    YieldOp::create(rewriter, loc, clone->getResults());
  }

  rewriter.replaceOp(root, guard.getResults());
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

    RewritePatternSet patternList(module->getContext());

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    // Register native constraints referenced by generated PDL patterns.
    pdlPattern.registerConstraintFunction("isConstantEqualTo",
                                          isConstantEqualTo);
    pdlPattern.registerConstraintFunction("isFloatConstantEqualTo",
                                          isFloatConstantEqualTo);

    // Conditional rules are applied by this, rather than declaratively.
    pdlPattern.registerConstraintFunction("tesseraRuleNotApplied",
                                          tesseraRuleNotApplied);
    pdlPattern.registerRewriteFunction("tesseraConditionalRewrite",
                                       tesseraConditionalRewrite);

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