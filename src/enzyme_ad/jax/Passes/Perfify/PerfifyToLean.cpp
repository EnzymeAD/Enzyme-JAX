#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_PERFIFYTOLEANPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

/// A conditions op is unproven if any of its assumptions failed to check.
/// An assumption without a satres has not been checked, so it does not count.
static bool isUnproven(ConditionsOp cond) {
  return cond
      .walk([](AssumeOp assume) {
        std::optional<bool> satres = assume.getSatres();
        return satres && !*satres ? WalkResult::interrupt()
                                  : WalkResult::advance();
      })
      .wasInterrupted();
}
struct TimeM {
  CostOp costOp;

  explicit TimeM(CostOp costOp) : costOp(costOp) {}

  std::string extract() {
    llvm::DenseMap<Value, std::string> exprs;
    SmallVector<std::string> params;
    std::optional<std::string> result;

    for (Operation &op : costOp.getBody().getOps()) {
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
    if (!result)
      return "";

    std::string name = ("«" + costOp.getTargetOp() + "»").str();
    std::string binders, args;
    for (const std::string &param : params) {
      binders += " (" + param + " : Nat)";
      args += " " + param;
    }
    std::string cost = params.empty() ? name : "(" + name + args + ")";

    return "def " + name + binders + " : Nat := " + *result + "\n" + "def " +
           name + ".tick" + binders + " : TimeM PUnit := TimeM.tick " + cost +
           "\n";
  }
};

struct PerfifyToLeanPass
    : public enzyme::perfify::impl::PerfifyToLeanPassBase<PerfifyToLeanPass> {
  using PerfifyToLeanPassBase::PerfifyToLeanPassBase;

  // Unproven conditions, keyed by the function they constrain, in the order
  // they appear in the module.
  llvm::MapVector<StringAttr, SmallVector<ConditionsOp>> unprovenConds;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](ConditionsOp cond) {
      if (cond.getVerifyHuh() && isUnproven(cond))
        unprovenConds[cond.getFuncHandleAttr().getAttr()].push_back(cond);
    });

    llvm::outs() << "-- Unproven conditions:";
    for (auto &[func, conds] : unprovenConds)
      llvm::outs() << " @" << func.getValue();
    llvm::outs() << "\n";

    module.walk([&](CostOp costOp) { llvm::outs() << TimeM(costOp).extract(); });
  }
};

} // namespace
