#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/MathExtras.h"

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

static bool matchPerfifyAnnotation(StringRef str, StringRef &payload,
                                   ArrayRef<StringRef> prefixes = {
                                       "perfify_op=", "perfify_cond="}) {
  for (StringRef prefix : prefixes) {
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

  bool operator==(const CostTerm &other) const {
    return coeff == other.coeff && syms == other.syms;
  }
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
        if (digits.getAsInteger(10, value) ||
            llvm::MulOverflow(term.coeff, value, term.coeff))
          return std::nullopt;
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
      if (*expr) {
        for (const CostTerm &term : **expr) {
          syms.insert(term.syms.begin(), term.syms.end());
        }
      }
    return syms;
  }
};

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

struct OpCost {
  StringRef target;
  CostExpr cost;
};

static std::optional<OpCost> parseOpCost(StringRef payload) {
  auto [target, costStr] = payload.split('=');
  target = target.trim();
  auto [dialect, opName] = target.split('.');
  std::optional<CostExpr> cost = parseCostExpr(costStr);
  if (!cost || dialect.empty() || opName.empty() ||
      llvm::any_of(target, llvm::isSpace)) {
    return std::nullopt;
  }
  return OpCost{target, std::move(*cost)};
}

struct LiftPerfifyAnnotationsPass
    : public enzyme::perfify::impl::LiftPerfifyAnnotationsPassBase<
          LiftPerfifyAnnotationsPass> {
  using LiftPerfifyAnnotationsPassBase::LiftPerfifyAnnotationsPassBase;

  // Fills `region` with a body computing `expr` out of arith.constant,
  // perfify.sym_cost, arith.muli and arith.addi.
  void buildCostBody(ImplicitLocOpBuilder &b, Region &region,
                     const CostExpr &expr) {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&region);
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
  }

  // Builds a perfify.estimated_cost @name whose body computes `expr`.
  FlatSymbolRefAttr buildEstimatedCost(ImplicitLocOpBuilder &b, StringRef name,
                                       const CostExpr &expr) {
    auto estCost = EstimatedCostOp::create(b, b.getStringAttr(name));
    buildCostBody(b, estCost.getBody(), expr);
    return FlatSymbolRefAttr::get(estCost.getNameAttr());
  }

  // Declares `perfify.define_symbol @name : i64` at the top of the module for
  // each of `symbols` that isn't declared yet. Fails if a name is taken by some
  // other symbol.
  LogicalResult declareSymbols(ModuleOp module, ImplicitLocOpBuilder &b,
                               const SetVector<StringRef> &symbols) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    for (StringRef name : symbols) {
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

  // Builds the bound `hyp` compares against: perfify.estimated_cost
  // @<prefix>_cost, or perfify.interval @<prefix>_range over
  // @<prefix>_cost_min/_max.
  FlatSymbolRefAttr buildBound(ImplicitLocOpBuilder &b, StringRef prefix,
                               const CostHypothesis &hyp) {
    if (!hyp.isInterval)
      return buildEstimatedCost(b, (prefix + "_cost").str(), *hyp.lb);
    FlatSymbolRefAttr lb =
        hyp.lb ? buildEstimatedCost(b, (prefix + "_cost_min").str(), *hyp.lb)
               : FlatSymbolRefAttr();
    FlatSymbolRefAttr ub =
        hyp.ub ? buildEstimatedCost(b, (prefix + "_cost_max").str(), *hyp.ub)
               : FlatSymbolRefAttr();
    auto interval =
        IntervalOp::create(b, b.getStringAttr(prefix + "_range"), lb, ub);
    return FlatSymbolRefAttr::get(interval.getNameAttr());
  }

  // Builds `perfify.hypothesis (perfify.cmp pred, cost, @bound) prePost`.
  // Loop hypotheses bound the trip count and have no Hoare state (null
  // prePost).
  HypothesisOp buildHypothesis(ImplicitLocOpBuilder &b, CmpPredicate pred,
                               Value cost, FlatSymbolRefAttr bound,
                               HoareStatesAttr prePost) {
    auto cmp =
        CompareOp::create(b, b.getI1Type(), pred, cost, bound.getValue());
    return HypothesisOp::create(b, b.getI1Type(), cmp.getResult(), prePost);
  }

  // Fills `region` with
  // `perfify.assume (perfify.hypothesis (perfify.cmp pred, cost, @bound)
  // prePost)`, where `operands` builds cost and @bound inside the region.
  void
  buildAssume(ImplicitLocOpBuilder &b, Region &region, HoareStates prePost,
              CmpPredicate pred,
              function_ref<std::pair<Value, FlatSymbolRefAttr>()> operands) {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&region);
    auto [cost, bound] = operands();
    auto hyp = buildHypothesis(b, pred, cost, bound,
                               HoareStatesAttr::get(b.getContext(), prePost));
    AssumeOp::create(b, hyp.getResult(), BoolAttr());
  }

  void setInsertionPointToAssumptions(ModuleOp module,
                                      ImplicitLocOpBuilder &b) {
    AssumptionsOp assumptions = nullptr;
    for (auto op : module.getOps<AssumptionsOp>())
      assumptions = op;
    if (!assumptions) {
      b.setInsertionPointToEnd(module.getBody());
      assumptions = AssumptionsOp::create(b);
    }

    Region &body = assumptions.getBody();
    if (body.empty())
      b.createBlock(&body);
    else
      b.setInsertionPointToEnd(&body.back());
  }

  void buildConditions(ModuleOp module, ImplicitLocOpBuilder &b,
                       StringRef funcName, const CostHypothesis &hyp) {
    MLIRContext *ctx = b.getContext();
    OpBuilder::InsertionGuard guard(b);
    setInsertionPointToAssumptions(module, b);
    auto conditions = ConditionsOp::create(b, funcName, /*verify_huh=*/true);

    // The precondition the hypothesis is tested under: an assumption on the
    // starting cost, which is zero if none was given.
    CmpPredicate prePred = hyp.argIndex ? hyp.argPred : CmpPredicate::eq;
    int64_t startCost = hyp.argIndex ? hyp.argValue : 0;
    buildAssume(
        b, conditions.getPrecondition(), HoareStates::Pre, prePred,
        [&]() -> std::pair<Value, FlatSymbolRefAttr> {
          return {
              ArgOp::create(b, CostType::get(ctx), hyp.argIndex.value_or(0)),
              buildEstimatedCost(b, "pre_cost", {CostTerm{startCost, {}}})};
        });

    buildAssume(b, conditions.getPostcondition(), HoareStates::Post, hyp.pred,
                [&]() -> std::pair<Value, FlatSymbolRefAttr> {
                  Value fnCost = FnCostOp::create(b, CostType::get(ctx));
                  return {fnCost, buildBound(b, "post", hyp)};
                });
  }

  // Lowers a perfify_func_cost payload into a perfify.func_bound for `func`, at
  // the end of the module.
  LogicalResult lowerFuncCost(ModuleOp module, LLVM::LLVMFuncOp func,
                              StringRef payload) {
    std::optional<CostHypothesis> hyp;
    if (std::optional<CostExpr> expr = parseCostExpr(payload)) {
      hyp.emplace();
      hyp->pred = CmpPredicate::eq;
      hyp->lb = expr;
    } else {
      hyp = parseCostHypothesis(payload);
    }
    if (!hyp || hyp->argIndex) {
      func.emitWarning("ignoring malformed perfify function cost '")
          << payload
          << R"(' (expected a cost expression or "<predicate> <bound>", )"
          << R"(e.g. "3X + 1", "eq [N, 3N]" or "le 10"))";
      return success();
    }
    ImplicitLocOpBuilder b(func.getLoc(), module.getContext());
    if (failed(declareSymbols(module, b, hyp->symbols())))
      return failure();
    b.setInsertionPointToEnd(module.getBody());
    auto bound = FuncBoundOp::create(b, func.getSymName(), hyp->pred);
    b.createBlock(&bound.getBody());
    buildBound(b, "func", *hyp);
    return success();
  }

  LogicalResult lowerOpCost(ModuleOp module, LLVM::GlobalOp global,
                            StringRef payload,
                            llvm::StringMap<StringRef> &lowered) {
    std::optional<OpCost> opCost = parseOpCost(payload);
    if (!opCost) {
      global.emitWarning("ignoring malformed op cost for Perfify");
      return success();
    }
    auto [it, inserted] = lowered.try_emplace(opCost->target, payload);
    if (!inserted) {
      if (parseOpCost(it->second)->cost == opCost->cost) {
        return success();
      } else {
        return global.emitError("conflicting perfify costs '")
               << it->second << "' and '" << payload << "'";
      }
    }
    ImplicitLocOpBuilder b(global.getLoc(), module.getContext());
    CostHypothesis hyp;
    hyp.lb = opCost->cost;
    if (failed(declareSymbols(module, b, hyp.symbols()))) {
      return failure();
    }
    setInsertionPointToAssumptions(module, b);
    auto cost = CostOp::create(b, b.getStringAttr(opCost->target));
    buildCostBody(b, cost.getBody(), opCost->cost);
    return success();
  }

  // Replaces each call to the perfify_loop marker `marker` with a trip count
  // hypothesis at the top of the enclosing loop's body.
  LogicalResult lowerLoopMarker(ModuleOp module, LLVM::LLVMFuncOp marker,
                                StringRef payload) {
    SmallVector<LLVM::CallOp> calls;
    module.walk([&](LLVM::CallOp call) {
      if (call.getCallee() == marker.getSymName())
        calls.push_back(call);
    });
    std::optional<CostHypothesis> hyp = parseCostHypothesis(payload);
    for (LLVM::CallOp call : calls) {
      // the loop body runs once per iteration (the second region of a while)
      Operation *loop = call->getParentOp();
      bool inBody = isa<scf::ForOp, affine::AffineForOp>(loop) ||
                    (isa<scf::WhileOp, stablehlo::WhileOp>(loop) &&
                     call->getParentRegion()->getRegionNumber() == 1);
      if (!hyp || hyp->argIndex) {
        call.emitWarning("ignoring malformed perfify loop trip count '")
            << payload << R"(' (expected "<predicate> <bound>", e.g. "le N"))";
      } else if (!inBody) {
        call.emitWarning("ignoring perfify loop trip count outside the body of "
                         "an scf.for, affine.for, scf.while or "
                         "stablehlo.while");
      } else if (!call->getBlock()->getOps<HypothesisOp>().empty()) {
        call.emitWarning("ignoring extra perfify loop trip count on a loop "
                         "that already has one");
      } else {
        ImplicitLocOpBuilder b(call.getLoc(), call);
        if (failed(declareSymbols(module, b, hyp->symbols())))
          return failure();
        b.setInsertionPointToStart(call->getBlock());
        Value trips = ArgOp::create(b, CostType::get(b.getContext()), 0);
        buildHypothesis(b, hyp->pred, trips, buildBound(b, "trip", *hyp),
                        HoareStatesAttr());
      }
      call.erase();
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM::GlobalOp annotationGlobal = nullptr;
    DenseMap<StringRef, std::string> stringGlobals;
    SmallVector<std::pair<LLVM::GlobalOp, StringRef>> opCosts;
    SmallVector<std::pair<StringRef, std::string>> namedFuncCosts;

    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationGlobal = global;
      }
      if (global.getSection() && *global.getSection() == "llvm.metadata") {
        if (auto strAttr =
                dyn_cast_or_null<StringAttr>(global.getValueAttr())) {
          stringGlobals[global.getSymName()] = strAttr.getValue().str();
          StringRef payload;
          if (matchPerfifyAnnotation(strAttr.getValue(), payload,
                                     {"perfify_cost="})) {
            opCosts.push_back({global, payload});
          }
          if (matchPerfifyAnnotation(strAttr.getValue(), payload,
                                     {"perfify_named_func_cost="})) {
            auto [symbol, cost] = payload.split('=');
            symbol = symbol.trim();
            if (symbol.empty() || !payload.contains('=')) {
              global.emitWarning(
                  "ignoring malformed perfify function cost definition '")
                  << payload << "'";
            } else {
              namedFuncCosts.push_back(
                  {symbol, ("perfify_func_cost=" + cost.trim()).str()});
            }
          }
        }
      }
    }

    if (!annotationGlobal)
      return;

    Region &region = annotationGlobal.getInitializerRegion();
    if (region.empty())
      return;
    llvm::StringMap<StringRef> lowered;
    for (auto [global, payload] : opCosts) {
      if (failed(lowerOpCost(module, global, payload, lowered))) {
        return signalPassFailure();
      }
    }

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
    auto addAnnotation = [&](StringRef funcName,
                             StringRef annotStr) -> LogicalResult {
      auto [it, inserted] =
          functionToAnnotation.try_emplace(funcName, annotStr);
      if (inserted || it->second.rtrim('\0') == annotStr.rtrim('\0')) {
        return success();
      }
      return module.lookupSymbol<LLVM::LLVMFuncOp>(funcName).emitError(
                 "conflicting perfify annotations '")
             << it->second.rtrim('\0') << "' and '" << annotStr.rtrim('\0')
             << "'";
    };

    for (auto [structValue, funcName] : structToFunction) {
      auto annotation = structToAnnotation.find(structValue);
      if (annotation == structToAnnotation.end() ||
          !annotation->second.starts_with("perfify_")) {
        continue;
      }
      if (failed(addAnnotation(funcName, annotation->second))) {
        return signalPassFailure();
      }
    }
    for (auto &[symbol, annotStr] : namedFuncCosts) {
      if (!module.lookupSymbol<LLVM::LLVMFuncOp>(symbol)) {
        continue;
      }
      if (failed(addAnnotation(symbol, annotStr))) {
        return signalPassFailure();
      }
    }

    // Apply annotations as attributes to functions and lower cost
    // hypotheses to perfify IR.
    for (auto [funcName, annotStr] : functionToAnnotation) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      if (!func)
        continue;

      StringRef payload;
      if (matchPerfifyAnnotation(annotStr, payload, {"perfify_loop="})) {
        if (failed(lowerLoopMarker(module, func, payload)))
          return signalPassFailure();
        continue;
      }
      if (matchPerfifyAnnotation(annotStr, payload, {"perfify_func_cost="})) {
        if (failed(lowerFuncCost(module, func, payload)))
          return signalPassFailure();
        continue;
      }
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
      if (failed(declareSymbols(module, b, hyp->symbols())))
        return signalPassFailure();
      buildConditions(module, b, funcName, *hyp);
    }
  }
};
} // namespace
