//===----------------------------------------------------------------------===//
//
// Turns each tessera rewrite rule into a PDL pattern. The lexer and parser
// are in RuleParser.cpp.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

// A rule passes a call's C arguments, but a tessera.call has no operand for
// one the function only writes (val=out). Without a define, the callee is
// never called, so every argument is taken as an operand.
bool isOutputArg(tessera::DefineOp define, unsigned index) {
  return define && !define.getCallOperandForArg(index);
}

std::pair<mlir::Value, mlir::Value>
emitMatchPDL(const Expr &expr, OpBuilder &builder, Location loc,
             llvm::StringMap<mlir::Value> &boundVars,
             SmallVectorImpl<std::string> &orderedVars, ModuleOp module) {
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
          [&](const StrLit &s) -> std::pair<mlir::Value, mlir::Value> {
            // C passes a string literal as the address of a constant holding
            // it; isStringConstantEqualTo reads the text.
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto addressOp = pdl::OperationOp::create(
                builder, loc, LLVM::AddressOfOp::getOperationName(),
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{},
                /*attrs=*/ValueRange{},
                /*types=*/ValueRange{resultTypeOp});

            auto expectedAttr = pdl::AttributeOp::create(
                builder, loc, builder.getStringAttr(s.value));

            pdl::ApplyNativeConstraintOp::create(
                builder, loc, TypeRange{}, "isStringConstantEqualTo",
                ValueRange{addressOp, expectedAttr});

            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        addressOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const GlobalRef &g) -> std::pair<mlir::Value, mlir::Value> {
            // The global's address, named by the op's attribute.
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto nameAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(), g.symbol));
            auto addressOp = pdl::OperationOp::create(
                builder, loc, LLVM::AddressOfOp::getOperationName(),
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{"global_name"},
                /*attrs=*/ValueRange{nameAttr},
                /*types=*/ValueRange{resultTypeOp});
            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        addressOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const Arith &) -> std::pair<mlir::Value, mlir::Value> {
            llvm_unreachable("the parser does not allow arithmetic in what a "
                             "rule matches");
          },
          [&](const Neg &) -> std::pair<mlir::Value, mlir::Value> {
            llvm_unreachable("the parser does not allow arithmetic in what a "
                             "rule matches");
          },
          [&](const Call &c) -> std::pair<mlir::Value, mlir::Value> {
            auto define = module.lookupSymbol<tessera::DefineOp>(
                c.dialect + "." + c.opname);
            SmallVector<mlir::Value> argValues;
            for (auto [index, arg] : llvm::enumerate(c.args))
              if (!isOutputArg(define, index))
                argValues.push_back(emitMatchPDL(arg, builder, loc, boundVars,
                                                 orderedVars, module)
                                        .first);
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

// The first variable in `expr` that `bound` lacks, if any.
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
  if (auto *neg = std::get_if<Neg>(&expr.data))
    return findUnboundVar(*neg->operand, bound);
  return std::nullopt;
}

// Check each call in a left-hand side against its define. Adds to `outputs`
// each variable passed where a call writes its result, and to `matched` each
// other variable. Returns what is wrong, if anything.
std::optional<std::string> checkLhsArgs(const Expr &expr, ModuleOp module,
                                        llvm::StringSet<> &outputs,
                                        llvm::StringSet<> &matched) {
  if (auto *var = std::get_if<Var>(&expr.data))
    matched.insert(var->name);
  auto *call = std::get_if<Call>(&expr.data);
  if (!call)
    return std::nullopt;
  std::string name = call->dialect + "." + call->opname;
  auto define = module.lookupSymbol<tessera::DefineOp>(name);
  if (define && call->args.size() != define.getArgModeEntries().size())
    return "'" + name + "' takes " +
           std::to_string(define.getArgModeEntries().size()) +
           " argument(s), but the rule passes it " +
           std::to_string(call->args.size());
  for (auto [index, arg] : llvm::enumerate(call->args)) {
    if (!isOutputArg(define, index)) {
      if (auto why = checkLhsArgs(arg, module, outputs, matched))
        return why;
      continue;
    }
    auto *var = std::get_if<Var>(&arg.data);
    if (!var)
      return "argument " + std::to_string(index) + " of '" + name +
             "' is where it writes its result, so the rule must pass a "
             "variable";
    if (!outputs.insert(var->name).second)
      return "'" + var->name + "' is where two calls write their results";
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

        // An output names where the matched call writes, so it has no value
        // to match or test.
        llvm::StringSet<> outputs, matched;
        std::optional<std::string> lhsProblem =
            checkLhsArgs(rule->lhs, module, outputs, matched);
        if (rule->cond)
          collectVars(*rule->cond, matched);
        for (const auto &output : outputs)
          if (!lhsProblem && matched.contains(output.getKey()))
            lhsProblem = "'" + output.getKey().str() +
                         "' is where a call writes its result, so it can't "
                         "also be matched or tested";
        if (lhsProblem) {
          emitWarning(loc) << "optimization rule ignored: " << *lhsProblem;
          continue;
        }

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
        auto root = emitMatchPDL(rule->lhs, builder, loc, boundVars,
                                 orderedVars, module);
        if (!root.second) {
          emitWarning(loc) << "optimization rule ignored: its left-hand side "
                              "must be a call";
          pattern.erase();
          continue;
        }

        // An output, or a name bound on the right-hand side, can be used
        // after it is written (checked by tessera-apply-pdl).
        llvm::StringMap<mlir::Value> visible = boundVars;
        for (const auto &output : outputs)
          visible.try_emplace(output.getKey());
        std::string problem;
        for (auto [expr, name] : llvm::zip(rule->rhs, rule->names)) {
          if (auto unbound = findUnboundVar(expr, visible)) {
            problem = "it uses '" + *unbound +
                      "' on the right-hand side, but it is not bound on the "
                      "left";
            break;
          }
          if (!name.empty() && !visible.try_emplace(name).second) {
            problem = "its right-hand side binds '" + name +
                      "', which is already bound";
            break;
          }
        }
        if (!problem.empty()) {
          emitWarning(loc) << "optimization rule ignored: " << problem;
          pattern.erase();
          continue;
        }

        // The right-hand side isn't written in PDL. tessera-apply-pdl builds it
        // with a native rewrite, because it needs things only known after a
        // match: whether a condition is proven (rewrite or guard), and which
        // producers a chain rule can move. So the rule is passed as text, with
        // the matched values and their names.
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

    // The rules are now patterns.
    for (auto optimizations_op :
         llvm::make_early_inc_range(module.getOps<tessera::OptimizationsOp>()))
      optimizations_op.erase();
  }
};
} // namespace
