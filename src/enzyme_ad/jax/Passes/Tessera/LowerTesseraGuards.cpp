//===----------------------------------------------------------------------===//
//
// Turns each tessera.guard (a condition not proven at compile time, with the
// rewrite and the original code) into a runtime check and a branch between
// the two. In a single-block region, such as a loop body, the branch is an
// scf.if.
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

// Integer comparisons are signed, like the rule language's literals.
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

// As in C: ordered comparisons, except '!=', which is true for NaN.
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

// The tessera.define a condition calls, or null if there is none.
DefineOp lookupDefine(const Call &call, GuardOp guard) {
  return SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      guard, StringAttr::get(guard.getContext(), calleeName(call)));
}

// The type a tessera.call to `define` expects at operand `index`: the pointee
// of a lifted argument, else the parameter type. Same as tessera-apply-pdl.
Type callOperandValueType(DefineOp define, unsigned index) {
  if (Type pointee = define.getCallOperandPointeeType(index))
    return pointee;
  std::optional<unsigned> raw = define.getArgIndexForCallOperand(index);
  return raw ? define.getFunctionType().getInput(*raw) : Type();
}

// Whether a tessera.call to `define` takes type `actual` at operand `index`.
// Same as tessera-apply-pdl.
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

// The type of `expr`, if known without building it, to give a literal on the
// other side of a comparison.
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
  // Arithmetic has its operands' type.
  if (auto *arith = std::get_if<Arith>(&expr.data)) {
    if (Type type = typeHintFor(*arith->lhs, guard))
      return type;
    return typeHintFor(*arith->rhs, guard);
  }
  return Type();
}

Value emitCall(const Call &call, OpBuilder &builder, Location loc,
               GuardOp guard);

// Build a comparison operand or call argument. A literal gets type `hint`.
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
          [&](const StrLit &) -> Value {
            llvm_unreachable("the parser only allows a string on the "
                             "right-hand side of a rule");
          },
          [&](const GlobalRef &) -> Value {
            llvm_unreachable("the parser only allows a global on the "
                             "right-hand side of a rule");
          },
          [&](const Call &c) -> Value {
            return emitCall(c, builder, loc, guard);
          },
          [&](const Arith &a) -> Value {
            // From the other operand, or `hint` if both are literals.
            Type type = typeHintFor(expr, guard);
            if (!type)
              type = hint;
            Value lhs = resolveOperand(*a.lhs, builder, loc, guard, type);
            Value rhs = resolveOperand(*a.rhs, builder, loc, guard, type);
            if (!lhs || !rhs)
              return Value();
            std::string op(getArithOpSpelling(a.op));
            if (lhs.getType() != rhs.getType()) {
              emitCheckWarning(guard)
                  << "operands of '" << op << "' in a condition have "
                  << "different types (" << lhs.getType() << " and "
                  << rhs.getType() << ")";
              return Value();
            }
            if (!isa<IntegerType, FloatType>(lhs.getType())) {
              emitCheckWarning(guard)
                  << "'" << op << "' in a condition is given values of type "
                  << lhs.getType() << ", but arithmetic is on integers and "
                  << "floats";
              return Value();
            }
            auto *divisor = std::get_if<IntLit>(&a.rhs->data);
            if ((a.op == ArithOp::Div || a.op == ArithOp::Rem) && divisor &&
                divisor->value == 0) {
              emitCheckWarning(guard)
                  << "'" << renderExpr(expr) << "' in a condition divides by "
                  << "zero";
              return Value();
            }
            return buildArith(builder, loc, a.op, lhs, rhs);
          },
      },
      expr.data);
}

// A tessera.call for a call in the condition, e.g. `mpfr.cmp_d(y, 0.5)`. It
// runs before the branch either way, so the callee must return one value and
// write none of its arguments.
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
  if (auto *arith = std::get_if<Arith>(&expr.data))
    return containsCall(*arith->lhs) || containsCall(*arith->rhs);
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
  // A literal takes the type of the other side.
  Type hint = typeHintFor(cmp.lhs, guard);
  if (!hint)
    hint = typeHintFor(cmp.rhs, guard);

  // Stop at the first failure, which has already warned.
  Value lhs = resolveOperand(cmp.lhs, builder, loc, guard, hint);
  if (!lhs)
    return Value();
  Value rhs = resolveOperand(cmp.rhs, builder, loc, guard, hint);
  if (!rhs)
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

// The runtime check for a named predicate. Its arguments must be values the
// rule matched, which the guard carries.
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

// `lhs && rhs` (`isAnd`) or `lhs || rhs`, short-circuiting as in C, so a call
// on the right can rely on the left. For example, in
//   mpfr.nan_p(y) == 0 && mpfr.cmp_d(y, 0.5) == 0
// mpfr_cmp_d never gets a NaN. Branches to new blocks, leaving the builder in
// the join block, or, when `structured`, uses an scf.if.
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

  // The result when lhs decides it: false for &&, true for ||.
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

// An i1 testing `cond`. && and || short-circuit only when the right side makes
// a call; otherwise both sides are safe to evaluate, so they're combined
// bitwise to keep the check straight-line.
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

// Replace `guard` with its else region, the original code. Used when the
// condition can't be checked.
void keepOriginal(GuardOp guard) {
  Block &elseBlock = guard.getElseRegion().front();
  auto yield = cast<YieldOp>(elseBlock.getTerminator());
  guard->replaceAllUsesWith(yield.getOperands());
  guard->getBlock()->getOperations().splice(
      guard->getIterator(), elseBlock.getOperations(), elseBlock.begin(),
      yield->getIterator());
  guard.erase();
}

// Replace `guard` with an scf.if on `check`, placing `start`, which computes
// it, ahead of the guard.
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

  // The parser warns if it doesn't parse.
  auto cond = parseConditionText(guard.getCond(), loc);
  if (!cond) {
    keepOriginal(guard);
    return;
  }

  // Regions like scf.for bodies must stay one block, so no branching there.
  bool structured = guard->getParentOp()->hasTrait<OpTrait::SingleBlock>();

  Block *entry = guard->getBlock();
  Region *region = entry->getParent();
  Block *thenBlock = &guard.getThenRegion().front();
  Block *elseBlock = &guard.getElseRegion().front();

  // Build the check first, since predicates read the else region's call to
  // find operand layouts. Build it in a scratch region so a check that fails
  // part way leaves nothing behind.
  Region scratch;
  Block *start = new Block();
  scratch.push_back(start);
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

  // Everything from the guard on becomes the continuation block.
  Block *tail = entry->splitBlock(guard);

  // The check starts in the entry block; any blocks it needs follow it.
  entry->getOperations().splice(entry->end(), start->getOperations());
  if (decide == start)
    decide = entry;
  start->erase();
  region->getBlocks().splice(tail->getIterator(), scratch.getBlocks());

  // The guard's results become block arguments (phis) of the continuation.
  SmallVector<Location> argLocs(guard.getNumResults(), loc);
  tail->addArguments(guard.getResultTypes(), argLocs);
  for (auto [result, arg] : llvm::zip(guard.getResults(), tail->getArguments()))
    result.replaceAllUsesWith(arg);

  for (Block *block : {thenBlock, elseBlock}) {
    auto yield = cast<YieldOp>(block->getTerminator());
    OpBuilder builder(yield);
    LLVM::BrOp::create(builder, loc, yield.getOperands(), tail);
    yield.erase();
  }

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
    // Collect first, since lowering rewires blocks under the walk.
    SmallVector<GuardOp> guards;
    getOperation()->walk([&](GuardOp guard) { guards.push_back(guard); });

    for (GuardOp guard : guards)
      lowerGuard(guard, maxUnrolledElems);
  }
};

} // namespace
