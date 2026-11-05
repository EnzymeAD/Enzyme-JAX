#include "src/enzyme_ad/jax/Passes/Perfify/PerfifyLeanEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"

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

static std::string natBinders(ArrayRef<std::string> params) {
  std::string binders;
  for (const std::string &param : params)
    binders += " (" + param + " : Nat)";
  return binders;
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

static std::optional<LeanTheorem>
hypothesisToLean(ConditionsOp cond, HypothesisOp hyp, StringRef funcName,
                 ArrayRef<std::string> funcParams, size_t index) {
  auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
  if (!cmpOp)
    return std::nullopt;

  std::string name = ("«" + funcName + "»").str();
  SmallVector<std::string> params;
  std::string cost;
  Operation *costDef = cmpOp.getCost().getDefiningOp();
  if (isa_and_nonnull<FnCostOp>(costDef)) {
    params.append(funcParams.begin(), funcParams.end());
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
    params.push_back(cost);
  } else {
    return std::nullopt;
  }

  CmpPredicate pred = cmpOp.getPredicate();
  std::string prop;
  Operation *bound = cmpOp.getBoundOp();
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

  std::optional<HoareStates> prePost = hyp.getPrePost();
  std::string thmName =
      name + "." + (prePost ? stringifyHoareStates(*prePost).str() : "hyp") + "_" + std::to_string(index);
  std::string statement =
      "theorem " + thmName + natBinders(params) + " : " + prop;
  return LeanTheorem{cond, hyp, thmName, statement};
}

static void conditionsToLean(ConditionsOp cond,
                             ArrayRef<std::string> funcParams,
                             LeanFile &file,
                             size_t index,
                             const llvm::StringMap<std::string> &proofs) {
  cond.walk([&](HypothesisOp hyp) {
    std::optional<LeanTheorem> thm =
        hypothesisToLean(cond, hyp, cond.getFuncHandle(), funcParams, index);
    if (!thm)
      return;
    auto proof = proofs.find(thm->name);
    file.text += thm->statement + " := by\n" +
                 (proof != proofs.end() ? proof->second : "  sorry\n");
    file.theorems.push_back(std::move(*thm));
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
    std::string binders = natBinders(params), args;
    for (const std::string &param : params)
      args += " " + param;
    std::string cost = params.empty() ? name : "(" + name + args + ")";

    return "def " + name + binders + " : Nat := " + result + "\n" + "def " +
           name + ".tick" + binders +
           " : TimeM Nat PUnit := TimeM.tick " + cost + "\n";
  }
};

struct LeanEmitter {
  ArrayRef<std::string> imports;
  const llvm::StringMap<std::string> &proofs;
  llvm::MapVector<StringAttr, SmallVector<ConditionsOp>> unprovenConds;
  llvm::StringMap<CostOp> costOps;

  LeanEmitter(ArrayRef<std::string> imports,
              const llvm::StringMap<std::string> &proofs)
      : imports(imports), proofs(proofs) {}

  /// Appends the parameters of the emitted program to `params`.
  std::string visitFunction(FunctionOpInterface func,
                            SmallVector<std::string> &params) {
    std::string name = ("«" + func.getName() + "»").str();
    Region &body = func.getFunctionBody();
    bool hasLeanForm = body.hasOneBlock();
    std::string ticks;
    for (Operation &op : body.getOps()) {
      if (op.getNumRegions() != 0) {
        hasLeanForm = false;
        break;
      }
      auto it = costOps.find(op.getName().getStringRef());
      if (it == costOps.end())
        continue;
      SmallVector<std::string> costParams;
      if (costRegionToLean(it->second.getBody(), costParams).empty())
        continue;
      ticks += "  ";
      if (!costParams.empty())
        ticks += "(";
      ticks += ("«" + op.getName().getStringRef() + "».tick").str();
      for (const std::string &param : costParams) {
        ticks += " " + param;
        if (!llvm::is_contained(params, param))
          params.push_back(param);
      }
      if (!costParams.empty())
        ticks += ")";
      ticks += "\n";
    }

    std::string def =
        "def " + name + natBinders(params) + " : TimeM Nat PUnit := ";
    if (!hasLeanForm) {
      func.emitWarning("function body has no Lean form, emitting sorry");
      return def + "sorry\n";
    }
    if (ticks.empty())
      return def + "pure ()\n";
    return def + "do\n" + ticks;
  }

  LeanFile emit(ModuleOp module) {
    module.walk([&](ConditionsOp cond) {
      if (cond.getVerifyHuh() && isUnproven(cond))
        unprovenConds[cond.getFuncHandleAttr().getAttr()].push_back(cond);
    });

    static const std::string defaultImports[] = {
        "Mathlib", "Cslib.Algorithms.Lean.TimeM"};
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
        conditionsToLean(cond, funcParams, file, index, proofs);
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
