//===----------------------------------------------------------------------===//
//
// This file implements the pass that turns a tessera.guard into a real branch.
//
// A tessera.guard records a condition that could not be proven at compile
// time, together with the specialized rewrite and the original computation.
// This pass synthesizes the code that tests the condition and replaces the
// guard with an ordinary conditional branch between the two, so nothing
// tessera-specific survives.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include <variant>

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_LOWERTESSERAGUARDSPASS
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

/// Map a rule-level comparison onto an LLVM integer predicate. Integer
/// comparisons are signed: the rule language has signed literals and no way to
/// say otherwise.
LLVM::ICmpPredicate toICmpPredicate(CmpOp op) {
  switch (op) {
  case CmpOp::Eq:
    return LLVM::ICmpPredicate::eq;
  case CmpOp::Ne:
    return LLVM::ICmpPredicate::ne;
  case CmpOp::Lt:
    return LLVM::ICmpPredicate::slt;
  case CmpOp::Le:
    return LLVM::ICmpPredicate::sle;
  case CmpOp::Gt:
    return LLVM::ICmpPredicate::sgt;
  case CmpOp::Ge:
    return LLVM::ICmpPredicate::sge;
  }
  llvm_unreachable("unhandled comparison operator");
}

/// The float predicates follow C: everything is an ordered comparison except
/// '!=', which is true when the operands are unordered.
LLVM::FCmpPredicate toFCmpPredicate(CmpOp op) {
  switch (op) {
  case CmpOp::Eq:
    return LLVM::FCmpPredicate::oeq;
  case CmpOp::Ne:
    return LLVM::FCmpPredicate::une;
  case CmpOp::Lt:
    return LLVM::FCmpPredicate::olt;
  case CmpOp::Le:
    return LLVM::FCmpPredicate::ole;
  case CmpOp::Gt:
    return LLVM::FCmpPredicate::ogt;
  case CmpOp::Ge:
    return LLVM::FCmpPredicate::oge;
  }
  llvm_unreachable("unhandled comparison operator");
}

/// If `expr` is a variable, the type of the value the guard carries for it.
/// Used to give a literal on the other side of a comparison the right type.
Type typeHintFor(const Expr &expr, GuardOp guard) {
  if (auto *var = std::get_if<Var>(&expr.data))
    if (Value value = guard.getArgForName(var->name))
      return value.getType();
  return Type();
}

/// Materialize one side of a comparison. `hint` is the type the other side
/// settled on, which is what a literal is built at.
Value resolveOperand(const Expr &expr, OpBuilder &builder, Location loc,
                     GuardOp guard, Type hint) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> Value {
            Value value = guard.getArgForName(v.name);
            if (!value)
              guard.emitError()
                  << "condition refers to '" << v.name
                  << "', which this guard does not carry a value for";
            return value;
          },
          [&](const IntLit &n) -> Value {
            if (!hint || !isa<IntegerType>(hint)) {
              guard.emitError() << "integer literal " << n.value
                                << " in a condition has no integer operand to "
                                   "take its type from";
              return Value();
            }
            return LLVM::ConstantOp::create(
                builder, loc, hint, builder.getIntegerAttr(hint, n.value));
          },
          [&](const FloatLit &n) -> Value {
            if (!hint || !isa<FloatType>(hint)) {
              guard.emitError() << "float literal " << n.value
                                << " in a condition has no float operand to "
                                   "take its type from";
              return Value();
            }
            return LLVM::ConstantOp::create(
                builder, loc, hint, builder.getFloatAttr(hint, n.value));
          },
          [&](const Call &c) -> Value {
            guard.emitError() << "a call is not supported as a comparison "
                                 "operand in a condition yet";
            return Value();
          },
      },
      expr.data);
}

Value emitCompare(const Compare &cmp, OpBuilder &builder, Location loc,
                  GuardOp guard) {
  // Resolve whichever side names a value first, so a literal on the other side
  // is built at a matching type rather than a guessed one.
  Type hint = typeHintFor(cmp.lhs, guard);
  if (!hint)
    hint = typeHintFor(cmp.rhs, guard);

  Value lhs = resolveOperand(cmp.lhs, builder, loc, guard, hint);
  Value rhs = resolveOperand(cmp.rhs, builder, loc, guard, hint);
  if (!lhs || !rhs)
    return Value();

  if (lhs.getType() != rhs.getType()) {
    guard.emitError() << "operands of '" << getCmpOpSpelling(cmp.op)
                      << "' in a condition have different types ("
                      << lhs.getType() << " and " << rhs.getType() << ")";
    return Value();
  }

  if (isa<IntegerType>(lhs.getType()))
    return LLVM::ICmpOp::create(builder, loc, toICmpPredicate(cmp.op), lhs,
                                rhs);
  if (isa<FloatType>(lhs.getType()))
    return LLVM::FCmpOp::create(builder, loc, toFCmpPredicate(cmp.op), lhs,
                                rhs);

  guard.emitError() << "cannot compare values of type " << lhs.getType()
                    << " in a condition";
  return Value();
}

/// Synthesize an i1 testing `cond`.
///
/// The connectives are bitwise on i1 rather than short-circuiting. Every check
/// this emits reads only memory the guarded call would read anyway, so
/// evaluating both sides is safe; that stops being true the moment a predicate
/// needs a pointer another part of the condition establishes, and this will
/// need real control flow then.
Value emitCond(const Cond &cond, OpBuilder &builder, Location loc,
               GuardOp guard) {
  return std::visit(overloaded{
                        [&](const Pred &p) -> Value {
                          guard.emitError() << "predicate '" << p.name
                                            << "' is not supported yet";
                          return Value();
                        },
                        [&](const Compare &c) -> Value {
                          return emitCompare(c, builder, loc, guard);
                        },
                        [&](const NotCond &c) -> Value {
                          Value inner =
                              emitCond(*c.operand, builder, loc, guard);
                          if (!inner)
                            return Value();
                          Value one = LLVM::ConstantOp::create(
                              builder, loc, builder.getI1Type(),
                              builder.getIntegerAttr(builder.getI1Type(), 1));
                          return LLVM::XOrOp::create(builder, loc, inner, one);
                        },
                        [&](const AndCond &c) -> Value {
                          Value lhs = emitCond(*c.lhs, builder, loc, guard);
                          Value rhs = emitCond(*c.rhs, builder, loc, guard);
                          if (!lhs || !rhs)
                            return Value();
                          return LLVM::AndOp::create(builder, loc, lhs, rhs);
                        },
                        [&](const OrCond &c) -> Value {
                          Value lhs = emitCond(*c.lhs, builder, loc, guard);
                          Value rhs = emitCond(*c.rhs, builder, loc, guard);
                          if (!lhs || !rhs)
                            return Value();
                          return LLVM::OrOp::create(builder, loc, lhs, rhs);
                        },
                    },
                    cond.data);
}

LogicalResult lowerGuard(GuardOp guard) {
  Location loc = guard.getLoc();

  auto cond = parseConditionText(guard.getCond(), loc);
  if (!cond)
    return failure();

  Block *entry = guard->getBlock();
  Region *region = entry->getParent();
  Block *thenBlock = &guard.getThenRegion().front();
  Block *elseBlock = &guard.getElseRegion().front();

  // Split so that everything after the guard becomes the continuation. The
  // guard itself leads the continuation block for now and is erased last.
  Block *tail = entry->splitBlock(guard);

  // In the LLVM dialect the phi nodes are block arguments, so the guard's
  // results become arguments of the continuation.
  SmallVector<Location> argLocs(guard.getNumResults(), loc);
  tail->addArguments(guard.getResultTypes(), argLocs);
  for (auto [result, arg] : llvm::zip(guard.getResults(), tail->getArguments()))
    result.replaceAllUsesWith(arg);

  // Each branch ends by feeding the continuation the values it yielded.
  for (Block *block : {thenBlock, elseBlock}) {
    auto yield = cast<YieldOp>(block->getTerminator());
    OpBuilder builder(yield);
    LLVM::BrOp::create(builder, loc, yield.getOperands(), tail);
    yield.erase();
  }

  // Move both bodies out of the guard's regions and into the surrounding one,
  // between the entry block and the continuation.
  region->getBlocks().splice(tail->getIterator(),
                             guard.getThenRegion().getBlocks());
  region->getBlocks().splice(tail->getIterator(),
                             guard.getElseRegion().getBlocks());

  // Everything the condition reads is a guard operand, defined before the
  // guard, so the check can be built at the end of the entry block.
  OpBuilder builder(entry, entry->end());
  Value check = emitCond(*cond, builder, loc, guard);
  if (!check)
    return failure();
  LLVM::CondBrOp::create(builder, loc, check, thenBlock, ValueRange{},
                         elseBlock, ValueRange{});

  guard.erase();
  return success();
}

struct LowerTesseraGuardsPass
    : public enzyme::tessera::impl::LowerTesseraGuardsPassBase<
          LowerTesseraGuardsPass> {
  using LowerTesseraGuardsPassBase::LowerTesseraGuardsPassBase;

  void runOnOperation() override {
    // Collect first: lowering a guard rewires the blocks around it, which a
    // walk in progress must not be looking at.
    SmallVector<GuardOp> guards;
    getOperation()->walk([&](GuardOp guard) { guards.push_back(guard); });

    for (GuardOp guard : guards)
      if (failed(lowerGuard(guard))) {
        signalPassFailure();
        return;
      }
  }
};

} // namespace
