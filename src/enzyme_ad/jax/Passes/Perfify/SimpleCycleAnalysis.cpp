#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "z3++.h"
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_SIMPLECYCLEANALYSISPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;
const std::string cost_str = "cost";

namespace {

struct SimpleCycleAnalysisPass
    : public enzyme::perfify::impl::SimpleCycleAnalysisPassBase<
          SimpleCycleAnalysisPass> {
  using SimpleCycleAnalysisPassBase::SimpleCycleAnalysisPassBase;
  mlir::Region *analysis_func = NULL;
  std::unordered_map<std::string, mlir::Region *> cost_map;
  std::unordered_map<std::string, z3::expr> sorry_map;
  llvm::DenseMap<mlir::Value, int64_t> args;
  struct HypothesisArgs {
    mlir::Region *lb = nullptr, *ub = nullptr;
  };
  std::unordered_map<HoareStates, HypothesisArgs> hypotheses;
  llvm::DenseMap<mlir::StringAttr, mlir::Region *> funcMap;

  void runOnOperation() override {
    Operation *op = getOperation();
    mlir::AsmState state(op);
    z3::context ctx;
    z3::solver solver(ctx);
    std::vector<z3::expr> cost_var;
    cost_var.push_back(solver.ctx().int_const("cost0"));

    visitOperation(op, &state, solver, cost_var);
    // z3::exprs must be released before their context is destroyed
    sorry_map.clear();
  }

  void visitOperation(Operation *op, AsmState *state, z3::solver &solver,
                      std::vector<z3::expr> &cost_var) {
    if (auto costOp = dyn_cast<CostOp>(op)) {
      std::string target_op = costOp.getTargetOp().str();
      if (cost_map.find(target_op) == cost_map.end()) {
        cost_map[target_op] =
            &costOp.getBody(); // store the body of the region for eval
      }
    } else if (auto arg = dyn_cast<ArgOp>(op)) {
      NamedAttribute arg_attr = op->getAttrs()[0];
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(arg_attr.getValue())) {
        args[op->getResult(0)] = intAttr.getInt();
      }
    } else if (auto cond = dyn_cast<ConditionsOp>(op)) {
      analysis_func = NULL;
      Attribute cond_attr = op->getAttrs()[0].getValue();
      auto cond_ref = mlir::cast<mlir::FlatSymbolRefAttr>(cond_attr);

      bool verify_huh =
          mlir::cast<mlir::BoolAttr>(op->getAttrs()[1].getValue()).getValue();
      if (verify_huh) {
        auto res = funcMap.find(cond_ref.getAttr());
        if (res != funcMap.end()) {
          analysis_func = res->second;
        } else {
          llvm::outs() << "Could not find " << cond_ref.getAttr()
                       << " in the function map\n";
        }
      } else {
        llvm::outs() << "conditions attr[1] was not true "
                     << op->getAttrs()[1].getName().getValue() << "\n";
      }
      // Clear the solver state between functions
      hypotheses.clear();
      cost_var.erase(cost_var.begin() + 1, cost_var.end());
      solver.push();
      for (Region &region : op->getRegions())
        visitRegion(region, state, solver, cost_var);
      solver.pop();
      return;
    } else if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      funcMap[func.getNameAttr()] = &func.getFunctionBody();
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      llvm::SmallVector<mlir::Operation *> allOps;
      if (analysis_func != NULL) {
        analysis_func->walk(
            [&allOps](mlir::Operation *op) { allOps.push_back(op); });
      }
      for (Operation *region_op : allOps) {
        std::string op_name = region_op->getName().getStringRef().str();
        std::optional<z3::expr> op_cost;
        auto cost_map_res = cost_map.find(op_name);
        auto sorry_map_res = sorry_map.find(op_name);
        if (cost_map_res != cost_map.end()) {
          op_cost = evaluateCostRegion(cost_map_res->second, solver);
        } else if (sorry_map_res != sorry_map.end()) {
          op_cost = sorry_map_res->second; // symbolic cost for sorried op
        } else {
          llvm::outs() << "unknown cost for operation " << op_name << "\n";
          continue;
        }
        std::string cost_v = cost_str + std::to_string(cost_var.size());
        cost_var.push_back(
            solver.ctx().int_const(cost_v.c_str())); // create a new cost var
        z3::expr post_cost = cost_var[cost_var.size() - 1];
        z3::expr pre_cost = cost_var[cost_var.size() - 2];
        z3::expr cost_expr =
            (post_cost == pre_cost + *op_cost); // increment per op cost

        solver.add(cost_expr); // add to solver
      }
    } else if (auto hyp = dyn_cast<HypothesisOp>(op)) {
      HoareStates pre_post = hyp.getPrePost();
      auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
      if (captureHypothesisArgs(cmpOp, pre_post)) {
        z3::expr holds =
            buildHypothesis(cmpOp.getPredicate(), pre_post, solver, cost_var);
        solver.add(pre_post == HoareStates::Pre ? holds : !holds);
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      // where the z3 checks actually happen
      bool met = false;
      if (auto hyp = assumeOp.getCondition().getDefiningOp<HypothesisOp>()) {
        auto check_res = solver.check();
        if (hyp.getPrePost() == HoareStates::Post && check_res == z3::sat &&
            !sorry_map.empty()) {
          // means that there was a counterexample, and now we want to compute
          // the set of values for the sorried op make this formula satisfiable
          llvm::outs() << "Condition can be proven if sorried op is resolved\n";
        }
        met = hyp.getPrePost() == HoareStates::Pre ? check_res == z3::sat
                                                   : check_res == z3::unsat;
#ifdef Z3_DEBUG_PERFIFY
        std::cout << solver << std::endl;
        llvm::outs() << (met ? "Met perf check!" : "Did not meet perf check")
                     << "\n";
#endif
      } else {
        llvm::outs()
            << "perfify.assume condition must be a perfify.hypothesis\n";
      }
      assumeOp.setSatresAttr(BoolAttr::get(&getContext(), met));
    } else if (auto sorryOp = dyn_cast<SorryOp>(op)) {
      std::string target_op = sorryOp.getTargetOp().str();
      z3::expr a = solver.ctx().int_const(("sorry_" + target_op).c_str());
      solver.add(a >= 0); // symbolic costs are never negative
      sorry_map.insert({target_op, a});
    }
    for (Region &region : op->getRegions())
      visitRegion(region, state, solver, cost_var);
  }

  bool captureHypothesisArgs(CompareOp cmpOp, HoareStates pre_post) {
    for (mlir::Value operand : cmpOp->getOperands()) {
      if (auto con_cost = operand.getDefiningOp<EstimatedCostOp>()) {
        hypotheses[pre_post] = {&con_cost.getBody(), &con_cost.getBody()};
        return true;
      }
      if (auto interval = operand.getDefiningOp<IntervalOp>()) {
        auto lb = interval.getLb()
                      ? interval.getLb().getDefiningOp<EstimatedCostOp>()
                      : nullptr;
        auto ub = interval.getUb()
                      ? interval.getUb().getDefiningOp<EstimatedCostOp>()
                      : nullptr;
        if ((interval.getLb() && !lb) || (interval.getUb() && !ub)) {
          llvm::outs() << "interval bounds must be perfify.estimated_cost\n";
          return false;
        }
        hypotheses[pre_post] = {lb ? &lb.getBody() : nullptr,
                                ub ? &ub.getBody() : nullptr};
        return true;
      }
    }
    llvm::outs() << "no bound found for " << stringifyHoareStates(pre_post)
                 << " hypothesis\n";
    return false;
  }

  z3::expr buildHypothesis(CmpPredicate pred, HoareStates pre_post,
                           z3::solver &solver,
                           std::vector<z3::expr> &cost_var) {
    z3::context &ctx = solver.ctx();
    const HypothesisArgs &args = hypotheses.at(pre_post);
    z3::expr cost =
        pre_post == HoareStates::Pre ? cost_var.front() : cost_var.back();
    std::optional<z3::expr> lb, ub;
    if (args.lb)
      lb = evaluateCostRegion(args.lb, solver);
    if (args.ub)
      ub = evaluateCostRegion(args.ub, solver);
    z3::expr in_interval = ctx.bool_val(true);
    if (lb)
      in_interval = in_interval && (*lb <= cost);
    if (ub)
      in_interval = in_interval && (cost <= *ub);
    switch (pred) {
    case CmpPredicate::eq:
      return in_interval;
    case CmpPredicate::ne:
      return !in_interval;
    case CmpPredicate::lt:
      return lb ? z3::expr(cost < *lb) : ctx.bool_val(false);
    case CmpPredicate::le:
      return lb ? z3::expr(cost <= *lb) : ctx.bool_val(false);
    case CmpPredicate::gt:
      return ub ? z3::expr(cost > *ub) : ctx.bool_val(false);
    case CmpPredicate::ge:
      return ub ? z3::expr(cost >= *ub) : ctx.bool_val(false);
    }
    llvm_unreachable("unknown perfify.cmp predicate");
  }

  z3::expr evaluateCostRegion(mlir::Region *costRegion, z3::solver &solver) {
    z3::context &ctx = solver.ctx();
    llvm::DenseMap<mlir::Value, z3::expr> valueExprs;
    std::optional<z3::expr> result;
    for (mlir::Block &block : costRegion->getBlocks()) {
      for (mlir::Operation &regionOp : block.getOperations()) {
        std::optional<z3::expr> expr;
        if (auto constOp = dyn_cast<arith::ConstantOp>(regionOp)) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            llvm::SmallString<40> valueStr;
            intAttr.getValue().toString(valueStr, 10, false);
            expr = ctx.int_val(valueStr.c_str());
          }
        } else if (auto symOp = dyn_cast<SymCostOp>(regionOp)) {
          std::string symName = ("sym_" + symOp.getSymAttr().getValue()).str();
          expr = ctx.int_const(symName.c_str());
          solver.add(*expr >= 0); // symbolic costs are never negative
        } else if (isa<arith::AddIOp, arith::MulIOp>(regionOp)) {
          auto lhs = valueExprs.find(regionOp.getOperand(0));
          auto rhs = valueExprs.find(regionOp.getOperand(1));
          if (lhs != valueExprs.end() && rhs != valueExprs.end()) {
            expr = isa<arith::AddIOp>(regionOp) ? (lhs->second + rhs->second)
                                                : (lhs->second * rhs->second);
          } else {
            llvm::outs() << "unresolvable operand in cost expression for "
                         << regionOp.getName().getStringRef() << "\n";
          }
        } else if (auto yieldOp = dyn_cast<YieldOp>(regionOp)) {
          if (mlir::Value yielded = yieldOp.getValue()) {
            auto it = valueExprs.find(yielded);
            if (it != valueExprs.end()) {
              result = it->second;
            }
          }
        } else {
          llvm::outs() << "unsupported op in cost region: "
                       << regionOp.getName().getStringRef() << "\n";
        }
        if (expr && regionOp.getNumResults() > 0) {
          valueExprs.insert({regionOp.getResult(0), *expr});
          result = expr;
        }
      }
    }
    return result.value_or(ctx.int_val(0));
  }

  void visitRegion(Region &region, AsmState *state, z3::solver &solver,
                   std::vector<z3::expr> &cost_var) {
    for (Block &block : region.getBlocks())
      visitBlock(block, state, solver, cost_var);
  }

  void visitBlock(Block &block, AsmState *state, z3::solver &solver,
                  std::vector<z3::expr> &cost_var) {
    for (Operation &op : block.getOperations())
      visitOperation(&op, state, solver, cost_var);
  }
};
} // namespace
