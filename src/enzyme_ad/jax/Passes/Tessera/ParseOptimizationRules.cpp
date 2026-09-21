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

// Pick the narrowest standard integer width that can hold a literal from a
// rule annotation. Literals are parsed as int64_t, so anything that does not
// round-trip through int32_t needs an i64 attribute; asking for an i32
// attribute in that case would silently truncate the value.
IntegerAttr getIntegerAttrForLiteral(OpBuilder &builder, int64_t value) {
  if (value >= std::numeric_limits<int32_t>::min() &&
      value <= std::numeric_limits<int32_t>::max())
    return builder.getI32IntegerAttr(static_cast<int32_t>(value));
  return builder.getI64IntegerAttr(value);
}

// Same idea for float literals: f32 when the value survives the round trip
// through it, otherwise f64. Values like 0.5 are exact in both, so they
// narrow to f32; floats like pi and e are not, so they stay f64.
FloatAttr getFloatAttrForLiteral(OpBuilder &builder, double value) {
  if (static_cast<double>(static_cast<float>(value)) == value)
    return builder.getF32FloatAttr(static_cast<float>(value));
  return builder.getF64FloatAttr(value);
}

std::pair<mlir::Value, mlir::Value>
emitMatchPDL(const Expr &expr, OpBuilder &builder, Location loc,
             llvm::StringMap<mlir::Value> &boundVars) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> std::pair<mlir::Value, mlir::Value> {
            if (!boundVars.count(v.name)) {
              boundVars[v.name] = pdl::OperandOp::create(
                  builder, loc, builder.getType<pdl::ValueType>(),
                  /*type=*/mlir::Value());
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
              auto argPDL = emitMatchPDL(c.args[i], builder, loc, boundVars);
              argValues.push_back(argPDL.first);
            }
            auto calleeAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(),
                                       c.dialect + "." + c.opname));
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto callOp = pdl::OperationOp::create(
                builder, loc, "tessera.call",
                /*operands=*/argValues,
                /*attrNames=*/ArrayRef<StringRef>{"callee"},
                /*attrs=*/ValueRange{calleeAttr},
                /*types=*/ValueRange{resultTypeOp});
            return {pdl::ResultOp::create(builder, loc,
                                          builder.getType<pdl::ValueType>(),
                                          callOp, builder.getI32IntegerAttr(0)),
                    callOp};
          },
      },
      expr.data);
}

std::pair<mlir::Value, mlir::Value>
emitRewritePDL(const Expr &expr, OpBuilder &builder, Location loc,
               llvm::StringMap<mlir::Value> &boundVars) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> std::pair<mlir::Value, mlir::Value> {
            return {boundVars[v.name], mlir::Value()};
          },
          [&](const IntLit &n) -> std::pair<mlir::Value, mlir::Value> {
            // NOTE: The width is chosen from the literal's magnitude
            // (i32 when it fits, otherwise i64), which guarantees the value
            // is representable, but it still takes no account of what the
            // surrounding IR expects. To fix, we should eventually derive the
            // type from context (the replaced value, or a type bound while
            // matching) and fall back to the magnitude-based choice only
            // when there is no context to read.
            auto attrVal = pdl::AttributeOp::create(
                builder, loc, getIntegerAttrForLiteral(builder, n.value));
            auto constOp = pdl::OperationOp::create(
                builder, loc, "llvm.mlir.constant",
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{"value"},
                /*attrs=*/ValueRange{attrVal}, /*types=*/ValueRange{});
            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const FloatLit &n) -> std::pair<mlir::Value, mlir::Value> {
            // The same caveat as the integer case above applies, where
            // the width is picked from the literal alone, so a value
            // that is not exact in f32 (pi, e) always materializes as f64 even
            // when the surrounding IR is f32, and f16/bf16 are unreachable. A
            // literal on the right-hand side needs its type derived from
            // context before this is dependable.
            auto attrVal = pdl::AttributeOp::create(
                builder, loc, getFloatAttrForLiteral(builder, n.value));
            auto constOp = pdl::OperationOp::create(
                builder, loc, "llvm.mlir.constant",
                /*operands=*/ValueRange{},
                /*attrNames=*/ArrayRef<StringRef>{"value"},
                /*attrs=*/ValueRange{attrVal}, /*types=*/ValueRange{});
            return {pdl::ResultOp::create(
                        builder, loc, builder.getType<pdl::ValueType>(),
                        constOp, builder.getI32IntegerAttr(0)),
                    mlir::Value()};
          },
          [&](const Call &c) -> std::pair<mlir::Value, mlir::Value> {
            SmallVector<mlir::Value> argValues;
            for (int i = 0; i < c.args.size(); i++) {
              auto argPDL = emitRewritePDL(c.args[i], builder, loc, boundVars);
              argValues.push_back(argPDL.first);
            }
            auto calleeAttr = pdl::AttributeOp::create(
                builder, loc,
                FlatSymbolRefAttr::get(builder.getContext(),
                                       c.dialect + "." + c.opname));
            auto resultTypeOp = pdl::TypeOp::create(
                builder, loc, builder.getType<pdl::TypeType>(),
                /*type=*/TypeAttr());
            auto callOp = pdl::OperationOp::create(
                builder, loc, "tessera.call",
                /*operands=*/argValues,
                /*attrNames=*/ArrayRef<StringRef>{"callee"},
                /*attrs=*/ValueRange{calleeAttr},
                /*types=*/ValueRange{resultTypeOp});
            return {pdl::ResultOp::create(builder, loc,
                                          builder.getType<pdl::ValueType>(),
                                          callOp, builder.getI32IntegerAttr(0)),
                    callOp};
          },
      },
      expr.data);
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

    // Find optimization ops and parse rewrite rules
    for (auto optimizations_op : module.getOps<tessera::OptimizationsOp>()) {
      for (auto optimization_op :
           optimizations_op.getBody().getOps<tessera::OptimizationOp>()) {
        Location loc = optimization_op.getLoc();
        Parser parser = Parser(optimization_op.getRule().str(), loc);
        auto rule = parser.parseRule();
        if (!rule) {
          signalPassFailure();
          llvm::errs() << "Pass failure\n";
          return;
        }

        // The condition is parsed but not yet turned into a guarded rewrite.
        // Reject it rather than dropping it: emitting the pattern without the
        // condition would apply a conditional rewrite unconditionally, which
        // miscompiles silently.
        if (rule->cond) {
          optimization_op.emitError(
              "conditional optimization rules are not supported yet");
          signalPassFailure();
          return;
        }

        // Create pdl.pattern op that will store PDL for parsed rewrite rule
        builder.setInsertionPointToStart(patternsModule.getBody());
        auto pattern = pdl::PatternOp::create(builder, loc, /*benefit=*/1,
                                              /*sym_name=*/nullptr);
        Block *patternBlock = builder.createBlock(&pattern.getBodyRegion());
        builder.setInsertionPointToStart(patternBlock);
        llvm::StringMap<mlir::Value> boundVars;

        // Emit PDL for the left hand side of the rewrite rule (the pattern to
        // match)
        auto root = emitMatchPDL(rule->lhs, builder, loc, boundVars);
        if (!root.second) {
          signalPassFailure();
          llvm::errs()
              << "Left hand side of optimization rule must be a call\n";
          return;
        }
        auto rewrite = pdl::RewriteOp::create(builder, loc, root.second,
                                              /*name=*/StringAttr(),
                                              /*externalArgs=*/ValueRange{});
        Block *rewriteBlock = builder.createBlock(&rewrite.getBodyRegion());
        builder.setInsertionPointToStart(rewriteBlock);

        // Emit PDL for the right hand side of the rewrite rule (the
        // replacement)
        auto replacement = emitRewritePDL(rule->rhs, builder, loc, boundVars);
        if (replacement.second) {
          pdl::ReplaceOp::create(builder, loc, root.second, replacement.second,
                                 ValueRange{});
        } else if (replacement.first) {
          pdl::ReplaceOp::create(builder, loc, root.second, mlir::Value(),
                                 ValueRange{replacement.first});
        } else {
          signalPassFailure();
          llvm::errs()
              << "Left hand side of optimization rule must be a call\n";
          return;
        }
      }
    }
  }
};
} // namespace
