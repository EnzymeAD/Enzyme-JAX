//===----------------------------------------------------------------------===//
//
// This file implements a custom lexer and parser to parse the tessera
// optimization rewrite rules defined by the user and then creates a PDL
// pattern from each rule.
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
              // Track first-appearance order too: the rewrite function is
              // handed these values positionally, paired with a list of the
              // names the rule knows them by.
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
            // Use an unbound pdl.types range (rather than a single pdl.type) so
            // the match doesn't constrain the result count of the tessera.call
            // op since it may vary (1 for a plain call, 1 + N for a callee with
            // N output-param arguments).
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

/// The first variable `expr` uses that the left-hand side does not bind, if
/// any. The right-hand side is built by a native rewrite that is not allowed
/// to fail, so an unbound name has to be caught here, where it can be reported
/// against the rule.
std::optional<std::string>
findUnboundVar(const Expr &expr, const llvm::StringMap<mlir::Value> &bound) {
  if (auto *var = std::get_if<Var>(&expr.data))
    return bound.count(var->name) ? std::nullopt
                                  : std::optional<std::string>(var->name);
  if (auto *call = std::get_if<Call>(&expr.data))
    for (const Expr &arg : call->args)
      if (auto name = findUnboundVar(arg, bound))
        return name;
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

    // Find optimization ops and parse rewrite rules. A rule that cannot be
    // used is reported as a warning and left out; the others still apply, and
    // the program compiles as it would have without it.
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
        // PatternOp::build emplaces the body block itself, so take that one
        // rather than adding a second.
        Block *patternBlock = &pattern.getBodyRegion().front();
        builder.setInsertionPointToStart(patternBlock);
        llvm::StringMap<mlir::Value> boundVars;
        SmallVector<std::string> orderedVars;

        // Emit PDL for the left hand side of the rewrite rule (the pattern to
        // match)
        auto root =
            emitMatchPDL(rule->lhs, builder, loc, boundVars, orderedVars);
        if (!root.second) {
          emitWarning(loc) << "optimization rule ignored: its left-hand side "
                              "must be a call";
          pattern.erase();
          continue;
        }

        if (auto name = findUnboundVar(rule->rhs, boundVars)) {
          emitWarning(loc) << "optimization rule ignored: it uses '" << *name
                           << "' on the right-hand side, but it is not bound "
                              "on the left";
          pattern.erase();
          continue;
        }

        // The right-hand side is not spelled out in PDL. It is built by a
        // native rewrite instead, for every rule, because the things it needs
        // to know are only knowable once the match exists:
        //
        //   - the result types of each call it builds, which come from the
        //     callee's tessera.define. Those do not exist yet when this pass
        //     runs, and PDL refuses to build a nested op whose types it cannot
        //     infer;
        //   - for a conditional rule, whether the condition can be proven,
        //     which decides between rewriting outright and building a guard;
        //   - for a chain rule, which matched producers can move along with
        //     the call they feed.
        //
        // So the whole rule travels as text, along with the matched values and
        // the names the rule calls them by, and tessera-apply-pdl re-parses it.
        auto ruleAttr = pdl::AttributeOp::create(
            builder, loc, builder.getStringAttr(optimization_op.getRule()));
        SmallVector<Attribute> nameAttrs;
        for (const std::string &name : orderedVars)
          nameAttrs.push_back(builder.getStringAttr(name));
        auto namesAttr = pdl::AttributeOp::create(
            builder, loc, builder.getArrayAttr(nameAttrs));

        // A guard keeps a clone of the matched call in its else region, so
        // without this a conditional pattern would match that clone and nest
        // guards without end. It goes first because it is the cheap one.
        if (rule->cond)
          pdl::ApplyNativeConstraintOp::create(
              builder, loc, TypeRange{}, "tesseraRuleNotApplied",
              ValueRange{root.second, ruleAttr});

        // Decides, from the matched IR, whether the rewrite can be built at
        // all and is safe to apply here. Declining in the match rather than
        // the rewrite is what keeps the greedy driver from looping on a call
        // the rule will never change.
        pdl::ApplyNativeConstraintOp::create(builder, loc, TypeRange{},
                                             "tesseraRuleApplicable",
                                             ValueRange{root.second, ruleAttr});

        // PDL passes the matched root to the rewrite function itself, ahead
        // of these, so it must not be listed here as well.
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
