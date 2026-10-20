#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_LIFTPERFIFYANNOTATIONSPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

static bool matchPerfifyAnnotation(StringRef str, StringRef &payload) {
  for (StringRef prefix : {"perfify_op=", "perfify_cond="}) {
    if (str.starts_with(prefix)) {
      payload = str.drop_front(prefix.size());
      while (payload.ends_with('\0'))
        payload = payload.drop_back(1);
      return true;
    }
  }
  return false;
}

struct CostTerm {
  int64_t coeff = 1;
  SmallVector<std::string> syms;
};
using CostExpr = SmallVector<CostTerm>;

// Parses a cost expression. Terms are separated by '+', factors within a term
// by '*', and a factor is an integer, a symbol, or an integer immediately
// followed by a symbol ("3X" == "3*X").
static std::optional<CostExpr> parseCostExpr(StringRef str) {
  CostExpr expr;
  SmallVector<StringRef> terms, factors;
  str.split(terms, '+');
  for (StringRef termStr : terms) {
    CostTerm &term = expr.emplace_back();
    factors.clear();
    termStr.split(factors, '*');
    for (StringRef factor : factors) {
      factor = factor.trim();
      if (factor.empty())
        return std::nullopt;
      StringRef digits = factor.take_while(llvm::isDigit);
      StringRef sym = factor.drop_front(digits.size());
      if (!digits.empty()) {
        int64_t value;
        if (digits.getAsInteger(10, value))
          return std::nullopt;
        term.coeff *= value;
      }
      if (!sym.empty()) {
        if (!llvm::all_of(sym,
                          [](char c) { return llvm::isAlnum(c) || c == '_'; }))
          return std::nullopt;
        term.syms.push_back(sym.str());
      }
    }
  }
  return expr;
}

// A cost hypothesis on a function, optionally conditioned on one of its
// arguments. The bound is either a cost expression or an interval of them:
//   "le 9"                  -> { true } f { fn_cost <= 9 }
//   "arg0 eq 0 -> le 9"     -> { arg0 == 0 } f { fn_cost <= 9 }
//   "eq [0, 8]"             -> { true } f { fn_cost in [0, 8] }
//   "eq [X, 3X]"            -> { true } f { fn_cost in [X, 3X] }
// Either side of an interval may be left empty to leave it unbounded.
struct CostHypothesis {
  CmpPredicate pred;
  bool isInterval = false;
  std::optional<CostExpr> lb, ub; // scalar goes in lb for bound generation
  std::optional<uint64_t> argIndex;
  CmpPredicate argPred;
  int64_t argValue;

  SetVector<StringRef> symbols() const {
    SetVector<StringRef> syms;
    for (const std::optional<CostExpr> *expr : {&lb, &ub})
      if (*expr)
        for (const CostTerm &term : **expr)
          syms.insert(term.syms.begin(), term.syms.end());
    return syms;
  }
};

// Parses a hypothesis of the form "<predicate> <bound>", optionally preceded
// by an argument precondition "arg<N> <predicate> <value> ->", where <bound>
// is a cost expression or "[<lb>, <ub>]".
static std::optional<CostHypothesis> parseCostHypothesis(StringRef payload) {
  CostHypothesis hyp;
  StringRef pre, post = payload;
  if (payload.contains("->"))
    std::tie(pre, post) = payload.split("->");

  auto [predStr, boundStr] = post.trim().split(' ');
  auto pred = symbolizeCmpPredicate(predStr);
  if (!pred)
    return std::nullopt;
  hyp.pred = *pred;
  boundStr = boundStr.trim();
  hyp.isInterval = boundStr.consume_front("[");
  if (hyp.isInterval) {
    if (!boundStr.consume_back("]") || boundStr.count(',') != 1)
      return std::nullopt;
    auto [lbStr, ubStr] = boundStr.split(',');
    // an empty side is left unbounded
    auto parseSide = [](StringRef str, std::optional<CostExpr> &side) {
      str = str.trim();
      return str.empty() || (side = parseCostExpr(str));
    };
    if (!parseSide(lbStr, hyp.lb) || !parseSide(ubStr, hyp.ub) ||
        (!hyp.lb && !hyp.ub))
      return std::nullopt;
  } else if (!(hyp.lb = parseCostExpr(boundStr))) {
    return std::nullopt;
  }

  pre = pre.trim();
  if (!pre.empty()) {
    SmallVector<StringRef, 3> tokens;
    pre.split(tokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    uint64_t argIndex;
    if (tokens.size() != 3 || !tokens[0].consume_front("arg") ||
        tokens[0].getAsInteger(10, argIndex) ||
        tokens[2].getAsInteger(10, hyp.argValue))
      return std::nullopt;
    auto argPred = symbolizeCmpPredicate(tokens[1]);
    if (!argPred)
      return std::nullopt;
    hyp.argIndex = argIndex;
    hyp.argPred = *argPred;
  }
  return hyp;
}

struct LiftPerfifyAnnotationsPass
    : public enzyme::perfify::impl::LiftPerfifyAnnotationsPassBase<
          LiftPerfifyAnnotationsPass> {
  using LiftPerfifyAnnotationsPassBase::LiftPerfifyAnnotationsPassBase;

  // Builds a perfify.constant_cost whose body computes `expr` out of
  // arith.constant, perfify.sym_cost, arith.muli and arith.addi.
  Value buildConstantCost(ImplicitLocOpBuilder &b, const CostExpr &expr) {
    OpBuilder::InsertionGuard guard(b);
    auto constCost = ConstantCostOp::create(b, CostType::get(b.getContext()));
    b.createBlock(&constCost.getBody());
    Value sum;
    for (const CostTerm &term : expr) {
      Value product;
      if (term.coeff != 1 || term.syms.empty())
        product = arith::ConstantOp::create(b, b.getI64IntegerAttr(term.coeff));
      for (const std::string &sym : term.syms) {
        Value symCost = SymCostOp::create(
            b, b.getI64Type(), FlatSymbolRefAttr::get(b.getContext(), sym));
        product =
            product ? arith::MulIOp::create(b, product, symCost) : symCost;
      }
      sum = sum ? arith::AddIOp::create(b, sum, product) : product;
    }
    YieldOp::create(b, sum);
    return constCost.getResult();
  }

  // Declares `perfify.define_symbol @name : i64` at the top of the module for
  // each symbol `hyp` uses that isn't declared yet. Fails if a name is taken
  // by some other symbol.
  LogicalResult declareSymbols(ModuleOp module, ImplicitLocOpBuilder &b,
                               const CostHypothesis &hyp) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    for (StringRef name : hyp.symbols()) {
      Operation *existing = module.lookupSymbol(name);
      if (existing && !isa<DefSymbolOp>(existing))
        return emitError(b.getLoc(), "perfify cost symbol '")
               << name << "' conflicts with an existing symbol";
      if (!existing)
        DefSymbolOp::create(b, b.getStringAttr(name),
                            TypeAttr::get(b.getI64Type()), StringAttr());
    }
    return success();
  }

  // Fills `region` with `perfify.assume (perfify.cmp pred, lhs, rhs)`, where
  // `operands` builds lhs and rhs inside the region.
  void buildAssume(ImplicitLocOpBuilder &b, Region &region, CmpPredicate pred,
                   function_ref<std::pair<Value, Value>()> operands) {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&region);
    auto [lhs, rhs] = operands();
    auto cmp = CompareOp::create(b, b.getI1Type(), pred, lhs, rhs);
    AssumeOp::create(b, cmp.getResult(), BoolAttr());
  }

  // Lowers `hyp` for @funcName into a perfify.conditions Hoare triple,
  // appended to the module's perfify.assumptions op (created if absent).
  // Since perfify.conditions is a terminator, each hypothesis gets its own
  // block when the previous one is already terminated.
  void buildConditions(ModuleOp module, ImplicitLocOpBuilder &b,
                       StringRef funcName, const CostHypothesis &hyp) {
    MLIRContext *ctx = b.getContext();
    OpBuilder::InsertionGuard guard(b);
    AssumptionsOp assumptions = nullptr;
    for (auto op : module.getOps<AssumptionsOp>())
      assumptions = op;
    if (!assumptions) {
      b.setInsertionPointToEnd(module.getBody());
      assumptions = AssumptionsOp::create(b);
    }

    Region &body = assumptions.getBody();
    if (body.empty() || body.back().getTerminator())
      b.createBlock(&body, body.end());
    else
      b.setInsertionPointToEnd(&body.back());
    auto conditions = ConditionsOp::create(b, funcName, /*verify_huh=*/true);

    // The precondition the hypothesis is tested under: an assumption on the
    // annotated function's argument if one was given, trivially true
    // otherwise.
    CmpPredicate prePred = hyp.argIndex ? hyp.argPred : CmpPredicate::eq;
    buildAssume(b, conditions.getPrecondition(), prePred,
                [&]() -> std::pair<Value, Value> {
                  if (!hyp.argIndex) {
                    Value zero = buildConstantCost(b, {CostTerm{0, {}}});
                    return {zero, zero};
                  }
                  return {ArgOp::create(b, CostType::get(ctx), *hyp.argIndex),
                          buildConstantCost(b, {CostTerm{hyp.argValue, {}}})};
                });

    buildAssume(b, conditions.getPostcondition(), hyp.pred,
                [&]() -> std::pair<Value, Value> {
                  Value fnCost = FnCostOp::create(b, CostType::get(ctx));
                  if (!hyp.isInterval)
                    return {fnCost, buildConstantCost(b, *hyp.lb)};
                  Value lb = hyp.lb ? buildConstantCost(b, *hyp.lb) : Value();
                  Value ub = hyp.ub ? buildConstantCost(b, *hyp.ub) : Value();
                  return {fnCost, IntervalOp::create(b, IntervalType::get(ctx),
                                                     lb, ub)};
                });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM::GlobalOp annotationGlobal = nullptr;
    DenseMap<StringRef, std::string> stringGlobals;
    SmallVector<std::string> hypotheses;

    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationGlobal = global;
      }
      if (global.getSection() && *global.getSection() == "llvm.metadata") {
        if (auto strAttr =
                dyn_cast_or_null<StringAttr>(global.getValueAttr())) {
          StringRef str = strAttr.getValue();
          stringGlobals[global.getSymName()] = str.str();
          StringRef payload;
          if (matchPerfifyAnnotation(str, payload))
            hypotheses.push_back(payload.str());
        }
      }
    }

    if (!annotationGlobal)
      return;

    Region &region = annotationGlobal.getInitializerRegion();
    if (region.empty())
      return;

    DenseMap<Value, StringRef> valueToFunction;
    DenseMap<Value, StringRef> valueToAnnotation;

    // Find addressof operations
    for (Operation &op : region.front()) {
      if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        StringRef globalName = addrOf.getGlobalName();
        Value result = addrOf.getResult();

        if (module.lookupSymbol<LLVM::LLVMFuncOp>(globalName)) {
          valueToFunction[result] = globalName;
        } else if (stringGlobals.count(globalName)) {
          valueToAnnotation[result] = stringGlobals[globalName];
        }
      }
    }
    DenseMap<Value, StringRef> structToFunction;
    DenseMap<Value, StringRef> structToAnnotation;

    // Follow insertvalue chains to match functions with annotations
    for (Operation &op : region.front()) {
      if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(&op)) {
        Value inserted = insertValue.getValue();
        Value container = insertValue.getContainer();
        Value result = insertValue.getResult();
        auto position = insertValue.getPosition();

        if (position.size() == 1) {
          if (position[0] == 0 && valueToFunction.count(inserted)) {
            structToFunction[result] = valueToFunction[inserted];
          } else if (position[0] == 1 && valueToAnnotation.count(inserted)) {
            structToAnnotation[result] = valueToAnnotation[inserted];
          }

          // Propagate information through the chain
          if (structToFunction.count(container)) {
            structToFunction[result] = structToFunction[container];
          }
          if (structToAnnotation.count(container)) {
            structToAnnotation[result] = structToAnnotation[container];
          }
        }
      }
    }
    DenseMap<StringRef, StringRef> functionToAnnotation;

    for (auto [structValue, funcName] : structToFunction) {
      if (structToAnnotation.count(structValue)) {
        StringRef annotStr = structToAnnotation[structValue];
        functionToAnnotation[funcName] = annotStr;
      }
    }

    // Apply annotations as attributes to functions and lower cost
    // hypotheses to perfify IR.
    for (auto [funcName, annotStr] : functionToAnnotation) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      if (!func)
        continue;

      StringRef payload;
      if (!matchPerfifyAnnotation(annotStr, payload))
        continue;

      func->setAttr("perfify_cond",
                    StringAttr::get(func->getContext(), payload));

      std::optional<CostHypothesis> hyp = parseCostHypothesis(payload);
      if (!hyp) {
        func.emitWarning("ignoring malformed perfify cost hypothesis '")
            << payload
            << R"(' (expected "[arg<N> <predicate> <value> ->] <predicate> <bound>", )"
            << R"(where <bound> is a cost expression or "[<lb>, <ub>]", )"
            << R"(e.g. "arg0 eq 0 -> le 9" or "eq [X, 3X]"))";
        continue;
      }
      ImplicitLocOpBuilder b(func.getLoc(), module.getContext());
      if (failed(declareSymbols(module, b, *hyp)))
        return signalPassFailure();
      buildConditions(module, b, funcName, *hyp);
    }
  }
};
} // namespace
