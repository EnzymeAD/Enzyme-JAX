//===----------------------------------------------------------------------===//
//
// Turns each tessera rewrite rule into a PDL pattern. The lexer and parser
// are in RuleParser.cpp.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include <limits>
#include <optional>
#include <utility>
#include <variant>

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_PARSEOPTIMIZATIONRULESPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

std::pair<mlir::Value, mlir::Value>
emitMatchPDL(const Expr &expr, OpBuilder &builder, Location loc,
             llvm::StringMap<mlir::Value> &boundVars,
             SmallVectorImpl<std::string> &orderedVars) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> std::pair<mlir::Value, mlir::Value> {
            if (!boundVars.count(v.name)) {
              boundVars[v.name] = pdl::OperandOp::create(
                  builder, loc, builder.getType<pdl::ValueType>(),
                  /*type=*/mlir::Value());
              // Also record first-appearance order: the rewrite gets these
              // values by position, alongside a list of their names.
              orderedVars.push_back(v.name);
            }
            return {boundVars[v.name], mlir::Value()};
          },
          [&](const IntLit &n) -> std::pair<mlir::Value, mlir::Value> {
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto constOp =
                pdl::OperationOp::create(builder, loc, /*name=*/std::nullopt,
                                         /*operands=*/ValueRange{},
                                         /*attrNames=*/ArrayRef<StringRef>{},
                                         /*attrs=*/ValueRange{},
                                         /*types=*/ValueRange{resultTypeOp});

            auto expectedAttr = pdl::AttributeOp::create(
                builder, loc, getIntegerAttrForLiteral(builder, n.value));

            pdl::ApplyNativeConstraintOp::create(
                builder, loc, TypeRange{}, "isConstantEqualTo",
                ValueRange{constOp, expectedAttr});

            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const FloatLit &n) -> std::pair<mlir::Value, mlir::Value> {
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());

            auto constOp =
                pdl::OperationOp::create(builder, loc, /*name=*/std::nullopt,
                                         /*operands=*/ValueRange{},
                                         /*attrNames=*/ArrayRef<StringRef>{},
                                         /*attrs=*/ValueRange{},
                                         /*types=*/ValueRange{resultTypeOp});

            auto expectedAttr = pdl::AttributeOp::create(
                builder, loc, builder.getF64FloatAttr(n.value));

            pdl::ApplyNativeConstraintOp::create(
                builder, loc, TypeRange{}, "isFloatConstantEqualTo",
                ValueRange{constOp, expectedAttr});

            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const StrLit &) -> std::pair<mlir::Value, mlir::Value> {
            llvm_unreachable("the parser only allows a string on the "
                             "right-hand side");
          },
          [&](const GlobalRef &) -> std::pair<mlir::Value, mlir::Value> {
            llvm_unreachable("the parser only allows a global on the "
                             "right-hand side");
          },
          [&](const Arith &) -> std::pair<mlir::Value, mlir::Value> {
            llvm_unreachable("the parser does not allow arithmetic in what a "
                             "rule matches");
          },
          [&](const Call &c) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Value> argValues;
            for (int i = 0; i < c.args.size(); i++) {
              auto argPDL =
                  emitMatchPDL(c.args[i], builder, loc, boundVars, orderedVars);
              argValues.push_back(argPDL.first);
            }
            auto calleeAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(),
                                       c.dialect + "." + c.opname));
            // A pdl.types range rather than one pdl.type, so any result count
            // matches (1, or 1 + N for a callee with N output arguments).
            auto resultTypesOp = pdl::TypesOp::create(
                builder, loc,
                pdl::RangeType::get(builder.getType<pdl::TypeType>()),
                /*constantTypes=*/ArrayAttr());
            auto callOp = pdl::OperationOp::create(
                builder, loc, "tessera.call",
                /*operands=*/argValues,
                /*attrNames=*/ArrayRef<StringRef>{"callee"},
                /*attrs=*/ValueRange{calleeAttr},
                /*types=*/ValueRange{resultTypesOp});
            return {pdl::ResultOp::create(builder, loc,
                                          builder.getType<pdl::ValueType>(),
                                          callOp, builder.getI32IntegerAttr(0)),
                    callOp};
          },
      },
      expr.data);
}

// The first variable in `expr` that the left-hand side doesn't bind, if any.
// Checked here because the native rewrite that builds the right-hand side
// can't fail.
std::optional<std::string>
findUnboundVar(const Expr &expr, const llvm::StringMap<mlir::Value> &bound) {
  if (auto *var = std::get_if<Var>(&expr.data))
    return bound.count(var->name) ? std::nullopt
                                  : std::optional<std::string>(var->name);
  if (auto *call = std::get_if<Call>(&expr.data))
    for (const Expr &arg : call->args)
      if (auto name = findUnboundVar(arg, bound))
        return name;
  if (auto *arith = std::get_if<Arith>(&expr.data)) {
    if (auto name = findUnboundVar(*arith->lhs, bound))
      return name;
    return findUnboundVar(*arith->rhs, bound);
  }
  return std::nullopt;
}

struct ParseOptimizationRulesPass
    : public enzyme::tessera::impl::ParseOptimizationRulesPassBase<
          ParseOptimizationRulesPass> {
  using ParseOptimizationRulesPassBase::ParseOptimizationRulesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());

    // Create nested module to store PDL patterns
    Location location = builder.getUnknownLoc();
    ModuleOp patternsModule =
        mlir::ModuleOp::create(builder, location, "patterns");

    // Parse each rule. One that can't be used is skipped with a warning; the
    // others still apply.
    for (auto optimizations_op : module.getOps<tessera::OptimizationsOp>()) {
      for (auto optimization_op :
           optimizations_op.getBody().getOps<tessera::OptimizationOp>()) {
        Location loc = optimization_op.getLoc();
        Parser parser = Parser(optimization_op.getRule().str(), loc);
        auto rule = parser.parseRule();
        // The parser has already said what is wrong with it.
        if (!rule)
          continue;

        // Create pdl.pattern op that will store PDL for parsed rewrite rule
        builder.setInsertionPointToStart(patternsModule.getBody());
        auto pattern = pdl::PatternOp::create(builder, loc, /*benefit=*/1,
                                              /*sym_name=*/std::nullopt);
        // PatternOp::build already adds the body block; use it.
        Block *patternBlock = &pattern.getBodyRegion().front();
        builder.setInsertionPointToStart(patternBlock);
        llvm::StringMap<mlir::Value> boundVars;
        SmallVector<std::string> orderedVars;

        // Emit PDL for the left-hand side (the pattern to match).
        auto root =
            emitMatchPDL(rule->lhs, builder, loc, boundVars, orderedVars);
        if (!root.second) {
          emitWarning(loc) << "optimization rule ignored: its left-hand side "
                              "must be a call";
          pattern.erase();
          continue;
        }

        std::optional<std::string> unbound;
        for (const Expr &expr : rule->rhs)
          if ((unbound = findUnboundVar(expr, boundVars)))
            break;
        if (unbound) {
          emitWarning(loc) << "optimization rule ignored: it uses '" << *unbound
                           << "' on the right-hand side, but it is not bound "
                              "on the left";
          pattern.erase();
          continue;
        }

        // The right-hand side isn't written in PDL. tessera-apply-pdl builds it
        // with a native rewrite, because it needs things only known after a
        // match: result types from the callees' tessera.defines (which don't
        // exist yet here), whether a condition is proven (rewrite or guard),
        // and which producers a chain rule can move. So the rule is passed as
        // text, with the matched values and their names.
        auto ruleAttr = pdl::AttributeOp::create(
            builder, loc, builder.getStringAttr(optimization_op.getRule()));
        SmallVector<Attribute> nameAttrs;
        for (const std::string &name : orderedVars)
          nameAttrs.push_back(builder.getStringAttr(name));
        auto namesAttr = pdl::AttributeOp::create(
            builder, loc, builder.getArrayAttr(nameAttrs));

        // A guard keeps a copy of the matched call, and `f(x) -> g(x); f(x)`
        // re-creates it, so either would let the pattern match its own output
        // forever. This cheap check, done first, prevents that.
        std::string lhsText = renderExpr(rule->lhs);
        if (rule->cond || llvm::any_of(rule->rhs, [&](const Expr &expr) {
              return renderExpr(expr) == lhsText;
            }))
          pdl::ApplyNativeConstraintOp::create(
              builder, loc, TypeRange{}, "tesseraRuleNotApplied",
              ValueRange{root.second, ruleAttr});

        // Whether the rewrite can be built and is safe here. Declining in the
        // match, not the rewrite, keeps the greedy driver from retrying the
        // same call forever.
        pdl::ApplyNativeConstraintOp::create(builder, loc, TypeRange{},
                                             "tesseraRuleApplicable",
                                             ValueRange{root.second, ruleAttr});

        // PDL passes the matched root to the rewrite first, so don't list it.
        SmallVector<mlir::Value> externalArgs{ruleAttr, namesAttr};
        for (const std::string &name : orderedVars)
          externalArgs.push_back(boundVars[name]);

        pdl::RewriteOp::create(builder, loc, root.second,
                               builder.getStringAttr("tesseraRewrite"),
                               externalArgs);
      }
    }
  }
};
} // namespace
