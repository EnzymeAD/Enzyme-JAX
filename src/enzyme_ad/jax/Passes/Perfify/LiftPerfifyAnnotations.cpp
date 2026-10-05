#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
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

// Matches the annotation strings emitted by the Reactant perfify_op
// attribute handler ("perfify_op=..."), or the "perfify_cond=..." spelling,
// and strips any trailing NUL bytes from the payload.
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

// A cost hypothesis on a function, optionally conditioned on one of its
// arguments:
//   "le 9"                  -> { true } f { fn_cost <= 9 }
//   "arg0 eq 0 -> le 9"     -> { arg0 == 0 } f { fn_cost <= 9 }
struct CostHypothesis {
  CmpPredicate pred;
  int64_t bound;
  std::optional<uint64_t> argIndex;
  CmpPredicate argPred;
  int64_t argValue;
};

// Parses a hypothesis of the form "<predicate> <bound>", optionally preceded
// by an argument precondition "arg<N> <predicate> <value> ->".
static bool parseCostHypothesis(StringRef payload, CostHypothesis &hyp) {
  payload = payload.trim();
  StringRef post = payload;
  StringRef pre;
  size_t arrow = payload.find("->");
  if (arrow != StringRef::npos) {
    pre = payload.take_front(arrow).trim();
    post = payload.drop_front(arrow + 2).trim();
  }

  SmallVector<StringRef, 3> postTokens;
  post.split(postTokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (postTokens.size() != 2)
    return false;
  auto predOpt = symbolizeCmpPredicate(postTokens[0]);
  if (!predOpt)
    return false;
  if (postTokens[1].getAsInteger(10, hyp.bound))
    return false;
  hyp.pred = *predOpt;

  hyp.argIndex.reset();
  if (!pre.empty()) {
    SmallVector<StringRef, 3> preTokens;
    pre.split(preTokens, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (preTokens.size() != 3)
      return false;
    StringRef argTok = preTokens[0];
    if (!argTok.consume_front("arg"))
      return false;
    uint64_t argIndex;
    if (argTok.getAsInteger(10, argIndex))
      return false;
    auto argPredOpt = symbolizeCmpPredicate(preTokens[1]);
    if (!argPredOpt)
      return false;
    if (preTokens[2].getAsInteger(10, hyp.argValue))
      return false;
    hyp.argIndex = argIndex;
    hyp.argPred = *argPredOpt;
  }
  return true;
}

struct LiftPerfifyAnnotationsPass
    : public enzyme::perfify::impl::LiftPerfifyAnnotationsPassBase<
          LiftPerfifyAnnotationsPass> {
  using LiftPerfifyAnnotationsPassBase::LiftPerfifyAnnotationsPassBase;

  // Builds `perfify.constant_cost { arith.constant value; perfify.yield }`.
  Value buildConstantCost(OpBuilder &builder, Location loc, int64_t value) {
    OpBuilder::InsertionGuard guard(builder);
    auto constCost = ConstantCostOp::create(builder, loc,
                                            CostType::get(builder.getContext()));
    Block *body = builder.createBlock(&constCost.getBody());
    builder.setInsertionPointToStart(body);
    arith::ConstantOp::create(builder, loc, builder.getI64IntegerAttr(value));
    YieldOp::create(builder, loc, Value());
    return constCost.getResult();
  }

  // Lowers `hyp` for @funcName into a perfify.conditions Hoare triple,
  // appended to the module's perfify.assumptions op (created if absent).
  // Since perfify.conditions is a terminator, each hypothesis gets its own
  // block when the previous one is already terminated.
  void buildConditions(ModuleOp module, OpBuilder &builder, Location loc,
                       StringRef funcName, const CostHypothesis &hyp) {
    MLIRContext *ctx = builder.getContext();
    AssumptionsOp assumptions = nullptr;
    for (auto op : module.getOps<AssumptionsOp>())
      assumptions = op;

    if (!assumptions) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(module.getBody());
      assumptions = AssumptionsOp::create(builder, loc);
    }

    Region &body = assumptions.getBody();
    Block *block;
    if (!body.empty() && !body.back().getTerminator()) {
      block = &body.back();
    } else {
      block = builder.createBlock(&body, body.end());
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    auto conditions =
        ConditionsOp::create(builder, loc, funcName, /*verify_huh=*/true);

    {
      OpBuilder::InsertionGuard regionGuard(builder);
      Block *preBlock = builder.createBlock(&conditions.getPrecondition());
      builder.setInsertionPointToStart(preBlock);
      // The precondition the hypothesis is tested under: an assumption on
      // the annotated function's argument if one was given, trivially true
      // otherwise.
      Value preLhs, preRhs;
      CmpPredicate prePred;
      if (hyp.argIndex) {
        preLhs = ArgOp::create(builder, loc, CostType::get(ctx), *hyp.argIndex)
                     .getResult();
        preRhs = buildConstantCost(builder, loc, hyp.argValue);
        prePred = hyp.argPred;
      } else {
        preLhs = buildConstantCost(builder, loc, 0);
        preRhs = preLhs;
        prePred = CmpPredicate::eq;
      }
      auto preCmp = CompareOp::create(builder, loc, builder.getI1Type(),
                                      prePred, preLhs, preRhs);
      AssumeOp::create(builder, loc, preCmp.getResult(), BoolAttr());
    }

    {
      OpBuilder::InsertionGuard regionGuard(builder);
      Block *postBlock = builder.createBlock(&conditions.getPostcondition());
      builder.setInsertionPointToStart(postBlock);
      Value fnCost =
          FnCostOp::create(builder, loc, CostType::get(ctx)).getResult();
      Value boundCost = buildConstantCost(builder, loc, hyp.bound);
      auto postCmp = CompareOp::create(builder, loc, builder.getI1Type(),
                                       hyp.pred, fnCost, boundCost);
      AssumeOp::create(builder, loc, postCmp.getResult(), BoolAttr());
    }
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

    OpBuilder builder(module.getContext());

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

      CostHypothesis hyp;
      if (!parseCostHypothesis(payload, hyp)) {
        func.emitWarning("ignoring malformed perfify cost hypothesis '")
            << payload
            << R"(' (expected "[arg<N> <predicate> <value> ->] <predicate> <bound>", e.g. "arg0 eq 0 -> le 9"))";
        continue;
      }
      buildConditions(module, builder, func.getLoc(), funcName, hyp);
    }
  }
};
} // namespace
