#include "src/enzyme_ad/jax/Passes/Perfify/PerfifyLeanEmitter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/CostLookup.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Demangle/Demangle.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

static bool isUnproven(ConditionsOp cond) {
  return cond
      .walk([](AssumeOp assume) {
        std::optional<bool> satres = assume.getSatres();
        return satres && !*satres ? WalkResult::interrupt()
                                  : WalkResult::advance();
      })
      .wasInterrupted();
}

static std::string funcLeanName(StringRef symbol) {
  return "«" + llvm::demangle(symbol) + "»";
}

static std::string costRegionToLean(Region &body,
                                    SmallVector<std::string> &params) {
  llvm::DenseMap<Value, std::string> exprs;
  std::optional<std::string> result;

  for (Operation &op : body.getOps()) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr || intAttr.getValue().isNegative())
        return "";
      llvm::SmallString<20> valueStr;
      intAttr.getValue().toString(valueStr, 10, /*Signed=*/true);
      exprs[constOp.getResult()] = valueStr.str().str();
    } else if (auto symOp = dyn_cast<SymCostOp>(op)) {
      std::string param = ("sym_" + symOp.getSymAttr().getValue()).str();
      if (!llvm::is_contained(params, param))
        params.push_back(param);
      exprs[symOp.getResult()] = param;
    } else if (isa<arith::AddIOp, arith::MulIOp>(op)) {
      auto lhs = exprs.find(op.getOperand(0));
      auto rhs = exprs.find(op.getOperand(1));
      if (lhs == exprs.end() || rhs == exprs.end())
        return "";
      StringRef sep = isa<arith::AddIOp>(op) ? " + " : " * ";
      exprs[op.getResult(0)] =
          "(" + lhs->second + sep.str() + rhs->second + ")";
    } else if (auto yieldOp = dyn_cast<YieldOp>(op)) {
      auto it = exprs.find(yieldOp.getValue());
      if (it == exprs.end())
        return "";
      result = it->second;
    } else {
      return "";
    }
  }
  return result.value_or("");
}

static std::string binders(ArrayRef<std::string> params) {
  std::string result;
  for (const std::string &param : params) {
    result += " (" + param + " : ";
    StringRef psref = StringRef(param);
    if (psref.starts_with("branch")) {
      result += "List Nat -> Bool";
    } else if (psref.starts_with("call") || psref.starts_with("iter_")) {
      result += "List Nat -> Nat";
    } else {
      result += "Nat";
    }
    result += ")";
  }
  return result;
}

struct IterationCount {
  std::string param;
  std::string expr;
  std::string premiseCost;
};

static std::optional<IterationCount>
iterationCountToLean(Operation *loop, ArrayRef<std::string> indices,
                     SmallVector<std::string> &params) {
  if (std::optional<APInt> count =
          cast<LoopLikeOpInterface>(loop).getStaticTripCount()) {
    std::string n = llvm::toString(*count, 10, /*Signed=*/false);
    return IterationCount{"", n, n};
  }
  auto tripSym = loop->getAttrOfType<FlatSymbolRefAttr>("perfify.loop_trip");
  if (!tripSym)
    return std::nullopt;
  IterationCount iterations;
  if (indices.empty()) {
    iterations.param = ("sym_" + tripSym.getValue()).str();
    iterations.expr = iterations.premiseCost = iterations.param;
  } else {
    iterations.param = ("iter_" + tripSym.getValue()).str();
    iterations.expr =
        "(" + iterations.param + " [" + llvm::join(indices, ", ") + "])";
    iterations.premiseCost = "(" + iterations.param + " l)";
  }
  if (!llvm::is_contained(params, iterations.param))
    params.push_back(iterations.param);
  return iterations;
}

static StringRef leanRelation(CmpPredicate pred) {
  switch (pred) {
  case CmpPredicate::eq:
    return "=";
  case CmpPredicate::ne:
    return "≠";
  case CmpPredicate::lt:
    return "<";
  case CmpPredicate::le:
    return "≤";
  case CmpPredicate::gt:
    return ">";
  case CmpPredicate::ge:
    return "≥";
  }
  llvm_unreachable("unknown perfify.cmp predicate");
}

static std::optional<std::string>
compareToProp(CmpPredicate pred, Operation *bound, const std::string &cost,
              SmallVector<std::string> &params) {
  std::string prop;
  if (auto estCost = dyn_cast_or_null<EstimatedCostOp>(bound)) {
    std::string expr = costRegionToLean(estCost.getBody(), params);
    if (expr.empty())
      return std::nullopt;
    prop = cost + " " + leanRelation(pred).str() + " " + expr;
  } else if (auto interval = dyn_cast_or_null<IntervalOp>(bound)) {
    std::string lb, ub;
    if (EstimatedCostOp minOp = interval.getMinCostOp())
      if ((lb = costRegionToLean(minOp.getBody(), params)).empty())
        return std::nullopt;
    if (EstimatedCostOp maxOp = interval.getMaxCostOp())
      if ((ub = costRegionToLean(maxOp.getBody(), params)).empty())
        return std::nullopt;
    std::string inInterval;
    if (!lb.empty())
      inInterval = lb + " ≤ " + cost;
    if (!ub.empty())
      inInterval += (inInterval.empty() ? "" : " ∧ ") + cost + " ≤ " + ub;
    switch (pred) {
    case CmpPredicate::eq:
      prop = inInterval;
      break;
    case CmpPredicate::ne:
      prop = "¬(" + inInterval + ")";
      break;
    case CmpPredicate::lt:
    case CmpPredicate::le:
      prop = lb.empty() ? "False"
                        : cost + " " + leanRelation(pred).str() + " " + lb;
      break;
    case CmpPredicate::gt:
    case CmpPredicate::ge:
      prop = ub.empty() ? "False"
                        : cost + " " + leanRelation(pred).str() + " " + ub;
      break;
    }
  } else {
    return std::nullopt;
  }

  return prop;
}

static std::optional<std::string>
hypothesisToProp(HypothesisOp hyp, StringRef funcName,
                 ArrayRef<std::string> funcParams,
                 SmallVector<std::string> &params) {
  auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
  if (!cmpOp)
    return std::nullopt;

  std::string name = funcLeanName(funcName);
  std::string cost;
  Operation *costDef = cmpOp.getCost().getDefiningOp();
  if (isa_and_nonnull<FnCostOp>(costDef)) {
    for (const std::string &param : funcParams) {
      if (!llvm::is_contained(params, param)) {
        params.push_back(param);
      }
    }
    if (funcParams.empty()) {
      cost = name + ".time";
    } else {
      cost = "(" + name;
      for (const std::string &param : funcParams)
        cost += " " + param;
      cost += ").time";
    }
  } else if (auto argOp = dyn_cast_or_null<ArgOp>(costDef)) {
    cost = "arg" + std::to_string(argOp.getVal());
    if (!llvm::is_contained(params, cost))
      params.push_back(cost);
  } else {
    return std::nullopt;
  }

  return compareToProp(cmpOp.getPredicate(), cmpOp.getBoundOp(), cost, params);
}

static void conditionsToLean(ConditionsOp cond,
                             ArrayRef<std::string> funcParams,
                             ArrayRef<std::string> bodyParams,
                             ArrayRef<std::string> bodyPremises, LeanFile &file,
                             size_t index,
                             const llvm::StringMap<std::string> &proofs) {
  StringRef funcName = cond.getFuncHandle();
  SmallVector<std::string> params;
  std::string premises;
  size_t numPremises = 0;
  cond.walk([&](HypothesisOp hyp) {
    if (hyp.getPrePost() != HoareStates::Pre)
      return;
    if (std::optional<std::string> prop =
            hypothesisToProp(hyp, funcName, funcParams, params))
      premises +=
          " (hpre_" + std::to_string(numPremises++) + " : " + *prop + ")";
  });
  for (const std::string &param : bodyParams)
    if (!llvm::is_contained(params, param))
      params.push_back(param);
  for (auto [k, prop] : llvm::enumerate(bodyPremises))
    premises += " (hbody_" + std::to_string(k) + " : " + prop + ")";
  if (std::optional<StringRef> region = cond.getInvalidRegion())
    file.text += ("-- Z3 counterexample: " + *region + "\n").str();

  cond.walk([&](HypothesisOp hyp) {
    std::optional<HoareStates> prePost = hyp.getPrePost();
    if (prePost == HoareStates::Pre)
      return;
    SmallVector<std::string> thmParams = params;
    std::optional<std::string> prop =
        hypothesisToProp(hyp, funcName, funcParams, thmParams);
    if (!prop)
      return;
    std::string name =
        funcLeanName(funcName) + "." +
        (prePost ? stringifyHoareStates(*prePost) : "hyp").str() + "_" +
        std::to_string(index);
    std::string statement =
        "theorem " + name + binders(thmParams) + premises + " : " + *prop;
    auto proof = proofs.find(name);
    file.text +=
        proof != proofs.end() ? proof->second : statement + " := by\n  sorry\n";
    file.theorems.push_back(LeanTheorem{cond, hyp, name, statement, *prop});
  });
}

struct TimeM {
  CostOp costOp;

  explicit TimeM(CostOp costOp) : costOp(costOp) {}

  std::string extract() {
    SmallVector<std::string> params;
    std::string result = costRegionToLean(costOp.getBody(), params);
    if (result.empty())
      return "";

    std::string name = ("«" + costOp.getTargetOp() + "»").str();
    std::string paramBinders = binders(params), args;
    for (const std::string &param : params)
      args += " " + param;
    std::string cost = params.empty() ? name : "(" + name + args + ")";

    return "def " + name + paramBinders + " : Nat := " + result + "\n" +
           "def " + name + ".tick" + paramBinders +
           " : TimeM Nat PUnit := TimeM.tick " + cost + "\n";
  }
};

struct LeanEmitter {
  ArrayRef<std::string> imports;
  const llvm::StringMap<std::string> &proofs;
  llvm::MapVector<StringAttr, SmallVector<ConditionsOp>> unprovenConds;
  llvm::StringMap<CostOp> costOps;
  SmallVector<std::string> bodyParams;
  SmallVector<std::string> bodyPremises;
  SmallVector<std::string> loopIndices;
  llvm::DenseMap<StringAttr, FuncBoundOp> funcBounds;
  unsigned int numCalls = 0;
  unsigned int numBranches = 0;
  StringAttr self;

  bool isSelfCall(CallOpInterface call) {
    auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    return callee && callee.getRootReference() == self;
  }

  LeanEmitter(ArrayRef<std::string> imports,
              const llvm::StringMap<std::string> &proofs)
      : imports(imports), proofs(proofs) {}

  bool visitCall(CallOpInterface call, StringRef indent, std::string &lines,
                 SmallVector<std::string> &params) {
    auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
    if (!callee)
      return false;
    if (isSelfCall(call))
      return true;
    auto bound = funcBounds.find(callee.getRootReference());
    if (bound == funcBounds.end())
      return false;
    FuncBoundOp boundOp = bound->second;
    std::string cost = "call" + std::to_string(numCalls++);
    params.push_back(cost);
    if (!llvm::is_contained(bodyParams, cost))
      bodyParams.push_back(cost);
    std::optional<std::string> prop =
        compareToProp(boundOp.getPredicate(), boundOp.getBoundOp(),
                      "(" + cost + " l)", bodyParams);
    if (!prop)
      return false;
    bodyPremises.push_back("∀ l, " + *prop);
    lines += indent;
    lines +=
        "TimeM.tick (" + cost + " [" + llvm::join(loopIndices, ", ") + "])\n";
    return true;
  }

  bool visitOp(Operation &op, StringRef indent, std::string &lines,
               SmallVector<std::string> &params) {
    if (isa_and_nonnull<PerfifyDialect>(op.getDialect()))
      return true;
    if (auto call = dyn_cast<CallOpInterface>(op))
      return visitCall(call, indent, lines, params);
    auto it = findCost(costOps, &op);
    SmallVector<std::string> costParams;
    bool isLoop = isa<scf::ForOp, affine::AffineForOp>(op);
    bool isIf = isa<scf::IfOp, affine::AffineIfOp>(op);
    if ((op.getNumRegions() != 0 && !isLoop && !isIf) || it == costOps.end() ||
        costRegionToLean(CostOp(it->second).getBody(), costParams).empty())
      return false;
    lines += indent;
    if (!costParams.empty())
      lines += "(";
    lines += ("«" + it->getKey() + "».tick").str();
    for (const std::string &param : costParams) {
      lines += " " + param;
      if (!llvm::is_contained(params, param))
        params.push_back(param);
    }
    if (!costParams.empty())
      lines += ")";
    lines += "\n";
    std::string inner = (indent + "  ").str();
    if (isIf) {
      std::string branch = "branch" + std::to_string(numBranches++);
      params.push_back(branch);
      lines += indent;
      lines +=
          "if " + branch + " [" + llvm::join(loopIndices, ", ") + "] then do\n";
      if (!visitArm(op.getRegion(0), inner, lines, params))
        return false;
      lines += indent;
      lines += "else do\n";
      return visitArm(op.getRegion(1), inner, lines, params);
    }
    if (!isLoop)
      return true;
    std::optional<IterationCount> iterations =
        iterationCountToLean(&op, loopIndices, params);
    if (!iterations)
      return false;
    auto hyps = op.getRegion(0).getOps<HypothesisOp>();
    if (!iterations->param.empty() && !hyps.empty())
      if (auto cmpOp = (*hyps.begin()).getCmp().getDefiningOp<CompareOp>()) {
        if (!llvm::is_contained(bodyParams, iterations->param))
          bodyParams.push_back(iterations->param);
        if (std::optional<std::string> prop =
                compareToProp(cmpOp.getPredicate(), cmpOp.getBoundOp(),
                              iterations->premiseCost, bodyParams))
          bodyPremises.push_back(loopIndices.empty() ? *prop : "∀ l, " + *prop);
      }
    std::string index = "i" + std::to_string(loopIndices.size());
    lines += indent;
    lines +=
        "(List.range " + iterations->expr + ").forM fun " + index + " => do\n";
    loopIndices.push_back(index);
    bool visited = visitArm(op.getRegion(0), inner, lines, params);
    loopIndices.pop_back();
    return visited;
  }

  bool visitArm(Region &region, StringRef indent, std::string &lines,
                SmallVector<std::string> &params) {
    std::string body;
    if (!visitRegion(region, indent, body, params))
      return false;
    lines += body.empty() ? (indent + "pure ()\n").str() : body;
    return true;
  }

  bool visitRegion(Region &region, StringRef indent, std::string &lines,
                   SmallVector<std::string> &params) {
    if (region.empty())
      return true;
    if (region.hasOneBlock())
      return visitBlock(region.front(), indent, lines, params);
    for (auto scc = llvm::scc_begin(&region); !scc.isAtEnd(); ++scc)
      if (scc.hasCycle())
        return false;
    if (!visitBlock(region.front(), indent, lines, params))
      return false;
    std::string inner = (indent + "  ").str();
    for (Block &block : llvm::drop_begin(region)) {
      std::string branch = "branch" + std::to_string(numBranches++);
      params.push_back(branch);
      std::string body;
      if (!visitBlock(block, inner, body, params))
        return false;
      lines += indent;
      lines +=
          "if " + branch + " [" + llvm::join(loopIndices, ", ") + "] then do\n";
      lines += body.empty() ? inner + "pure ()\n" : body;
      lines += indent;
      lines += "else do\n" + inner + "pure ()\n";
    }
    return true;
  }

  bool visitBlock(Block &block, StringRef indent, std::string &lines,
                  SmallVector<std::string> &params) {
    for (Operation &op : block)
      if (!visitOp(op, indent, lines, params))
        return false;
    return true;
  }

  /// Appends the parameters of the emitted program to `params`.
  std::string visitFunction(FunctionOpInterface func,
                            SmallVector<std::string> &params) {
    bodyParams.clear();
    bodyPremises.clear();
    loopIndices.clear();
    numCalls = 0;
    numBranches = 0;
    self = func.getNameAttr();
    std::string name = funcLeanName(func.getName());
    Region &body = func.getFunctionBody();
    bool recursive = func->walk([&](CallOpInterface call) {
                           return isSelfCall(call) ? WalkResult::interrupt()
                                                   : WalkResult::advance();
                         })
                         .wasInterrupted();
    if (recursive) {
      params.push_back("activations");
      loopIndices.push_back("i0");
    }
    std::string ticks;
    bool hasLeanForm =
        !body.empty() &&
        visitRegion(body, recursive ? "    " : "  ", ticks, params);

    std::string def =
        "def " + name + binders(params) + " : TimeM Nat PUnit := ";
    if (!hasLeanForm) {
      func.emitWarning("function body has no Lean form, emitting sorry");
      return def + "sorry\n";
    }
    if (ticks.empty())
      return def + "pure ()\n";
    if (recursive)
      ticks = "  (List.range activations).forM fun i0 => do\n" + ticks;
    return def + "do\n" + ticks;
  }

  LeanFile emit(ModuleOp module) {
    module.walk([&](ConditionsOp cond) {
      if (cond.getVerifyHuh() && isUnproven(cond))
        unprovenConds[cond.getFuncHandleAttr().getAttr()].push_back(cond);
    });

    module.walk([&](FuncBoundOp bound) {
      funcBounds.try_emplace(bound.getFuncHandleAttr().getAttr(), bound);
    });

    static const std::string defaultImports[] = {"Mathlib",
                                                 "Cslib.Algorithms.Lean.TimeM"};
    LeanFile file;
    std::string &lean = file.text;
    for (const std::string &leanModule :
         imports.empty() ? ArrayRef(defaultImports) : imports)
      lean += "import " + leanModule + "\n";
    lean += "\n-- Unproven conditions:";
    for (auto &[func, conds] : unprovenConds)
      lean += (" @" + func.getValue()).str();
    lean += "\n\nopen Cslib.Algorithms.Lean\n\nnamespace Perfify\n\n";

    module.walk([&](CostOp costOp) {
      costOps.try_emplace(costOp.getTargetOp(), costOp);
      lean += TimeM(costOp).extract();
    });

    for (auto &[funcName, conds] : unprovenConds) {
      ConditionsOp first = conds.front();
      auto func = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
          first, first.getFuncHandleAttr());
      if (!func) {
        first.emitError("no function named ") << first.getFuncHandleAttr();
        continue;
      }
      SmallVector<std::string> funcParams;
      lean += "\n" + visitFunction(func, funcParams);
      for (auto [index, cond] : llvm::enumerate(conds))
        conditionsToLean(cond, funcParams, bodyParams, bodyPremises, file,
                         index, proofs);
    }
    lean += "\nend Perfify\n";
    return file;
  }
};

} // namespace

LeanFile
mlir::enzyme::perfify::emitLean(ModuleOp module, ArrayRef<std::string> imports,
                                const llvm::StringMap<std::string> &proofs) {
  return LeanEmitter(imports, proofs).emit(module);
}
