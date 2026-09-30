//===----------------------------------------------------------------------===//
//
// This file implements the pass that turns a tessera.guard into a real branch.
//
// A tessera.guard records a condition that could not be proven at compile
// time, together with the specialized rewrite and the original computation.
// This pass synthesizes the code that tests the condition and replaces the
// guard with an ordinary conditional branch between the two, so nothing
// tessera-specific survives. Where the guard sits in a region that must stay
// one block, such as a loop body, the branch is an scf.if instead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Predicates.h"
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

std::string calleeName(const Call &call) {
  return call.dialect + "." + call.opname;
}

/// The tessera.define of the op a condition calls, or null if the module has
/// none: the function is not annotated tessera_op.
DefineOp lookupDefine(const Call &call, GuardOp guard) {
  return SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      guard, StringAttr::get(guard.getContext(), calleeName(call)));
}

/// The type a tessera.call to `define` expects at operand `index`: the pointee
/// for an argument the call site loads, otherwise the parameter itself. As in
/// tessera-apply-pdl, which builds the calls on a rule's right-hand side.
Type callOperandValueType(DefineOp define, unsigned index) {
  if (Type pointee = define.getCallOperandPointeeType(index))
    return pointee;
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  return raw ? define.getFunctionType().getInput(*raw) : Type();
}

/// Whether a tessera.call to `define` can take a value of type `actual` at
/// operand `index`: the parameter's own type, or the pointee of one the call
/// site loads. As in tessera-apply-pdl.
bool callOperandAccepts(DefineOp define, unsigned index, Type actual) {
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  if (!raw)
    return false;
  Type formal = define.getFunctionType().getInput(*raw);
  if (actual == formal)
    return true;
  Type pointee = define.getCallOperandPointeeType(index);
  return pointee && isa<LLVM::LLVMPointerType>(formal) && actual == pointee;
}

/// The type of what `expr` denotes, where that is known without building it:
/// the value the guard carries for a variable, or what a call returns. Used to
/// give a literal on the other side of a comparison the right type.
Type typeHintFor(const Expr &expr, GuardOp guard) {
  if (auto *var = std::get_if<Var>(&expr.data))
    if (Value value = guard.getArgForName(var->name))
      return value.getType();
  if (auto *call = std::get_if<Call>(&expr.data))
    if (DefineOp define = lookupDefine(*call, guard)) {
      SmallVector<Type> results = define.getCallResultTypes();
      if (results.size() == 1)
        return results.front();
    }
  return Type();
}

Value emitCall(const Call &call, OpBuilder &builder, Location loc,
               GuardOp guard);

/// Materialize one side of a comparison, or an argument of a call in the
/// condition. `hint` is the type the other side settled on, or the type the
/// callee takes, which is what a literal is built at.
Value resolveOperand(const Expr &expr, OpBuilder &builder, Location loc,
                     GuardOp guard, Type hint) {
  return std::visit(
      overloaded{
          [&](const Var &v) -> Value {
            Value value = guard.getArgForName(v.name);
            if (!value)
              emitCheckWarning(guard)
                  << "condition refers to '" << v.name
                  << "', which this guard does not carry a value for";
            return value;
          },
          [&](const IntLit &n) -> Value {
            if (!hint || !isa<IntegerType>(hint)) {
              emitCheckWarning(guard)
                  << "integer literal " << n.value
                  << " in a condition has no integer operand to "
                     "take its type from";
              return Value();
            }
            return LLVM::ConstantOp::create(
                builder, loc, hint, builder.getIntegerAttr(hint, n.value));
          },
          [&](const FloatLit &n) -> Value {
            if (!hint || !isa<FloatType>(hint)) {
              emitCheckWarning(guard)
                  << "float literal " << n.value
                  << " in a condition has no float operand to "
                     "take its type from";
              return Value();
            }
            return LLVM::ConstantOp::create(
                builder, loc, hint, builder.getFloatAttr(hint, n.value));
          },
          [&](const Call &c) -> Value {
            return emitCall(c, builder, loc, guard);
          },
      },
      expr.data);
}

/// Call a tessera op the condition names, e.g. `mpfr.cmp_d(y, 0.5)`, and
/// return its result. It becomes a tessera.call ahead of the branch, which
/// tessera-to-llvm lowers along with every other.
///
/// The callee has to be a query: one result, which the condition compares,
/// and no argument it writes, since the call runs whichever way the guard then
/// goes. It is given values the rule matched, literals, and other calls.
Value emitCall(const Call &call, OpBuilder &builder, Location loc,
               GuardOp guard) {
  std::string name = calleeName(call);
  DefineOp define = lookupDefine(call, guard);
  if (!define) {
    emitCheckWarning(guard)
        << "condition calls '" << name
        << "', which has no tessera.define; is it annotated tessera_op?";
    return Value();
  }
  if (define.getNumWrittenArgs() != 0) {
    emitCheckWarning(guard) << "condition calls '" << name
                            << "', which writes one of its arguments";
    return Value();
  }
  SmallVector<Type> resultTypes = define.getCallResultTypes();
  if (resultTypes.size() != 1) {
    emitCheckWarning(guard) << "condition calls '" << name
                            << "', which does not return exactly one value";
    return Value();
  }
  if (call.args.size() != define.getNumCallOperands()) {
    emitCheckWarning(guard)
        << "'" << name << "' takes " << define.getNumCallOperands()
        << " argument(s), but the condition passes " << call.args.size();
    return Value();
  }

  SmallVector<Value> operands;
  for (auto [index, arg] : llvm::enumerate(call.args)) {
    Value value = resolveOperand(arg, builder, loc, guard,
                                 callOperandValueType(define, index));
    if (!value)
      return Value();
    if (!callOperandAccepts(define, index, value.getType())) {
      emitCheckWarning(guard)
          << "argument " << index << " of '" << name << "' in a condition is "
          << value.getType() << ", which it does not take";
      return Value();
    }
    operands.push_back(value);
  }
  return CallOp::create(builder, loc, name, resultTypes, operands).getResult(0);
}

bool containsCall(const Expr &expr) {
  return std::holds_alternative<Call>(expr.data);
}

bool containsCall(const Cond &cond) {
  return std::visit(
      overloaded{
          [](const Pred &p) {
            return llvm::any_of(p.args,
                                [](const Expr &e) { return containsCall(e); });
          },
          [](const Compare &c) {
            return containsCall(c.lhs) || containsCall(c.rhs);
          },
          [](const NotCond &c) { return containsCall(*c.operand); },
          [](const AndCond &c) {
            return containsCall(*c.lhs) || containsCall(*c.rhs);
          },
          [](const OrCond &c) {
            return containsCall(*c.lhs) || containsCall(*c.rhs);
          },
      },
      cond.data);
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
    emitCheckWarning(guard) << "operands of '" << getCmpOpSpelling(cmp.op)
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

  emitCheckWarning(guard) << "cannot compare values of type " << lhs.getType()
                          << " in a condition";
  return Value();
}

Value emitCond(const Cond &cond, OpBuilder &builder, Location loc,
               GuardOp guard, int64_t maxUnrolledElems, bool structured);

/// Resolve a named predicate and let it synthesize its own check. Its operands
/// are looked up on the guard: a predicate tests values the rule matched, not
/// expressions computed here.
Value emitPredicate(const Pred &pred, OpBuilder &builder, Location loc,
                    GuardOp guard, int64_t maxUnrolledElems) {
  const TesseraPredicate *predicate = lookupPredicate(pred.name);
  if (!predicate) {
    auto diagnostic = emitCheckWarning(guard)
                      << "unknown predicate '" << pred.name << "'";
    diagnostic << "; known predicates are ";
    llvm::interleaveComma(getKnownPredicateNames(), diagnostic);
    return Value();
  }

  if (pred.args.size() != predicate->arity) {
    emitCheckWarning(guard)
        << "predicate '" << pred.name << "' takes " << predicate->arity
        << " argument(s), but got " << pred.args.size();
    return Value();
  }

  SmallVector<Value> args;
  for (const Expr &arg : pred.args) {
    auto *var = std::get_if<Var>(&arg.data);
    if (!var) {
      emitCheckWarning(guard) << "predicate '" << pred.name
                              << "' takes matched values, not expressions";
      return Value();
    }
    Value value = guard.getArgForName(var->name);
    if (!value) {
      emitCheckWarning(guard)
          << "condition refers to '" << var->name
          << "', which this guard does not carry a value for";
      return Value();
    }
    args.push_back(value);
  }

  CheckContext ctx{builder, loc, guard, maxUnrolledElems};
  return predicate->emitCheck(args, ctx);
}

/// `lhs && rhs` (`isAnd`) or `lhs || rhs`, evaluating rhs only when lhs does
/// not already decide the result, as C does. A call on the right is then only
/// made when what comes before it holds, which is how a rule makes a call safe:
/// `mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, 0.5) == 0` never gives mpfr_cmp_d a
/// NaN, on which it would raise MPFR's erange flag.
///
/// This needs blocks: lhs branches to one evaluating rhs or straight to a join
/// block, whose argument is the result. Both are inserted after the current
/// block, and the builder is left at the end of the join.
///
/// When `structured`, the check has to stay one block, and an scf.if on lhs
/// yields the result instead, evaluating rhs in the branch where lhs does not
/// decide it. The builder is left after the scf.if.
Value emitShortCircuit(const Cond &lhsCond, const Cond &rhsCond, bool isAnd,
                       OpBuilder &builder, Location loc, GuardOp guard,
                       int64_t maxUnrolledElems, bool structured) {
  Value lhs =
      emitCond(lhsCond, builder, loc, guard, maxUnrolledElems, structured);
  if (!lhs)
    return Value();

  Type i1 = builder.getI1Type();
  if (structured) {
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{i1}, lhs,
                                  /*addThenBlock=*/true, /*addElseBlock=*/true);
    OpBuilder::InsertionGuard afterIf(builder);
    Block *rhsBlock = isAnd ? ifOp.thenBlock() : ifOp.elseBlock();
    Block *decidedBlock = isAnd ? ifOp.elseBlock() : ifOp.thenBlock();

    builder.setInsertionPointToEnd(decidedBlock);
    Value decided = LLVM::ConstantOp::create(
        builder, loc, i1, builder.getIntegerAttr(i1, isAnd ? 0 : 1));
    scf::YieldOp::create(builder, loc, decided);

    builder.setInsertionPointToEnd(rhsBlock);
    Value rhs =
        emitCond(rhsCond, builder, loc, guard, maxUnrolledElems, structured);
    if (!rhs)
      return Value();
    scf::YieldOp::create(builder, loc, rhs);
    return ifOp.getResult(0);
  }

  Block *current = builder.getInsertionBlock();
  Region::BlockListType &blocks = current->getParent()->getBlocks();
  auto *join = new Block();
  join->addArgument(i1, loc);
  blocks.insert(std::next(current->getIterator()), join);
  auto *rhsBlock = new Block();
  blocks.insert(join->getIterator(), rhsBlock);

  // What the result is when lhs decides it: false for &&, true for ||.
  Value decided = LLVM::ConstantOp::create(
      builder, loc, i1, builder.getIntegerAttr(i1, isAnd ? 0 : 1));
  if (isAnd)
    LLVM::CondBrOp::create(builder, loc, lhs, rhsBlock, ValueRange{}, join,
                           ValueRange{decided});
  else
    LLVM::CondBrOp::create(builder, loc, lhs, join, ValueRange{decided},
                           rhsBlock, ValueRange{});

  builder.setInsertionPointToEnd(rhsBlock);
  Value rhs =
      emitCond(rhsCond, builder, loc, guard, maxUnrolledElems, structured);
  if (!rhs)
    return Value();
  LLVM::BrOp::create(builder, loc, ValueRange{rhs}, join);

  builder.setInsertionPointToEnd(join);
  return join->getArgument(0);
}

/// Synthesize an i1 testing `cond`.
///
/// A connective whose right side makes a call short-circuits (see
/// emitShortCircuit). Any other is bitwise on i1: a predicate or comparison
/// reads only memory the guarded call would read anyway, so evaluating both
/// sides is safe, and keeps the check straight-line code.
Value emitCond(const Cond &cond, OpBuilder &builder, Location loc,
               GuardOp guard, int64_t maxUnrolledElems, bool structured) {
  return std::visit(
      overloaded{
          [&](const Pred &p) -> Value {
            return emitPredicate(p, builder, loc, guard, maxUnrolledElems);
          },
          [&](const Compare &c) -> Value {
            return emitCompare(c, builder, loc, guard);
          },
          [&](const NotCond &c) -> Value {
            Value inner = emitCond(*c.operand, builder, loc, guard,
                                   maxUnrolledElems, structured);
            if (!inner)
              return Value();
            Value one = LLVM::ConstantOp::create(
                builder, loc, builder.getI1Type(),
                builder.getIntegerAttr(builder.getI1Type(), 1));
            return LLVM::XOrOp::create(builder, loc, inner, one);
          },
          [&](const AndCond &c) -> Value {
            if (containsCall(*c.rhs))
              return emitShortCircuit(*c.lhs, *c.rhs, /*isAnd=*/true, builder,
                                      loc, guard, maxUnrolledElems, structured);
            Value lhs = emitCond(*c.lhs, builder, loc, guard, maxUnrolledElems,
                                 structured);
            Value rhs = emitCond(*c.rhs, builder, loc, guard, maxUnrolledElems,
                                 structured);
            if (!lhs || !rhs)
              return Value();
            return LLVM::AndOp::create(builder, loc, lhs, rhs);
          },
          [&](const OrCond &c) -> Value {
            if (containsCall(*c.rhs))
              return emitShortCircuit(*c.lhs, *c.rhs, /*isAnd=*/false, builder,
                                      loc, guard, maxUnrolledElems, structured);
            Value lhs = emitCond(*c.lhs, builder, loc, guard, maxUnrolledElems,
                                 structured);
            Value rhs = emitCond(*c.rhs, builder, loc, guard, maxUnrolledElems,
                                 structured);
            if (!lhs || !rhs)
              return Value();
            return LLVM::OrOp::create(builder, loc, lhs, rhs);
          },
      },
      cond.data);
}

/// Replace `guard` with its else region, the computation the rule would have
/// replaced. That is always correct, so it is what a guard whose condition
/// cannot be checked becomes: the rule is not applied here, and nothing else
/// changes.
void keepOriginal(GuardOp guard) {
  Block &elseBlock = guard.getElseRegion().front();
  auto yield = cast<YieldOp>(elseBlock.getTerminator());
  guard->replaceAllUsesWith(yield.getOperands());
  guard->getBlock()->getOperations().splice(
      guard->getIterator(), elseBlock.getOperations(), elseBlock.begin(),
      yield->getIterator());
  guard.erase();
}

/// Replace `guard` with an scf.if on `check`, which `start` computes. This is
/// the form for a guard in a region that must stay one block: the check goes
/// in ahead of the guard and the guard's two regions become the scf.if's.
void replaceWithIf(GuardOp guard, Block *start, Value check) {
  Location loc = guard.getLoc();
  guard->getBlock()->getOperations().splice(guard->getIterator(),
                                            start->getOperations());

  OpBuilder builder(guard);
  auto ifOp = scf::IfOp::create(builder, loc, guard.getResultTypes(), check,
                                /*addThenBlock=*/false, /*addElseBlock=*/false);
  ifOp.getThenRegion().takeBody(guard.getThenRegion());
  ifOp.getElseRegion().takeBody(guard.getElseRegion());
  for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    auto yield = cast<YieldOp>(region->front().getTerminator());
    OpBuilder yieldBuilder(yield);
    scf::YieldOp::create(yieldBuilder, loc, yield.getOperands());
    yield.erase();
  }

  guard->replaceAllUsesWith(ifOp.getResults());
  guard.erase();
}

void lowerGuard(GuardOp guard, int64_t maxUnrolledElems) {
  Location loc = guard.getLoc();

  // The parser reports why, if it does not parse.
  auto cond = parseConditionText(guard.getCond(), loc);
  if (!cond) {
    keepOriginal(guard);
    return;
  }

  // A branch splits the guard's block, which the body of an scf.for, scf.if or
  // affine.for must not be: those regions are one block each. A guard there
  // becomes an scf.if, and its check stays one block too.
  bool structured = guard->getParentOp()->hasTrait<OpTrait::SingleBlock>();

  Block *entry = guard->getBlock();
  Region *region = entry->getParent();
  Block *thenBlock = &guard.getThenRegion().front();
  Block *elseBlock = &guard.getElseRegion().front();

  // Synthesize the check first, while the guard is still intact. A predicate
  // reads the call kept in the else region to find out how its operand is laid
  // out, so this has to happen before that region is moved away.
  //
  // It is built in a scratch region and moved into place once complete: a
  // check that short-circuits needs blocks of its own (unless structured), and
  // one that gives up part way must leave nothing behind.
  Region scratch;
  Block *start = new Block();
  scratch.push_back(start);
  // The scratch region belongs to no op, so it has no context to offer.
  OpBuilder builder(guard.getContext());
  builder.setInsertionPointToEnd(start);
  Value check =
      emitCond(*cond, builder, loc, guard, maxUnrolledElems, structured);
  if (!check) {
    scratch.dropAllReferences();
    keepOriginal(guard);
    return;
  }
  if (structured) {
    replaceWithIf(guard, start, check);
    return;
  }
  // The block the check ends in, which branches on it.
  Block *decide = builder.getInsertionBlock();

  // Split so that everything after the guard becomes the continuation. The
  // guard itself leads the continuation block for now and is erased last.
  Block *tail = entry->splitBlock(guard);

  // The check starts in the entry block; any blocks it needs follow it.
  entry->getOperations().splice(entry->end(), start->getOperations());
  if (decide == start)
    decide = entry;
  start->erase();
  region->getBlocks().splice(tail->getIterator(), scratch.getBlocks());

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

  OpBuilder branchBuilder(decide, decide->end());
  LLVM::CondBrOp::create(branchBuilder, loc, check, thenBlock, ValueRange{},
                         elseBlock, ValueRange{});

  guard.erase();
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
      lowerGuard(guard, maxUnrolledElems);
  }
};

} // namespace
