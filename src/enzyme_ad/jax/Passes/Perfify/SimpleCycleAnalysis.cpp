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
  llvm::DenseMap<mlir::Value, int64_t> args;
  std::unordered_map<mlir::enzyme::perfify::HoareStates, mlir::Region *>
      constant_costs;
  llvm::DenseMap<mlir::Value, std::pair<mlir::Region *, mlir::Region *>>
      interval_costs;
  llvm::DenseMap<mlir::StringAttr, mlir::Region *> funcMap;

  void runOnOperation() override {
    Operation *op = getOperation();
    mlir::AsmState state(op);
    z3::context ctx;
    z3::solver solver(ctx);
    std::vector<z3::expr> cost_var;
    cost_var.push_back(solver.ctx().int_const("cost0"));

    visitOperation(op, &state, solver, cost_var);
  }

  void visitOperation(Operation *op, AsmState *state, z3::solver &solver,
                      std::vector<z3::expr> &cost_var) {
    if (auto costOp = dyn_cast<CostOp>(op)) {
      std::string target_op = costOp.getTargetOp().str();
      if (cost_map.find(target_op) == cost_map.end()) {
        cost_map[target_op] =
            &costOp.getBody(); // store the body of the region for eval
      }
    } else if (auto con_cost = dyn_cast<ConstantCostOp>(op)) {
      mlir::enzyme::perfify::HoareStates pre_post =
          static_cast<mlir::enzyme::perfify::HoareStates>(
              con_cost->getParentRegion()->getRegionNumber());
      constant_costs.insert({pre_post, &con_cost.getBody()});
    } else if (auto interval_cost = dyn_cast<IntervalOp>(op)) {
      std::optional<mlir::Region *>
          lb = nullptr,
          ub = nullptr; // if nullptr, cannot be evaluated.
      if (mlir::Value bound = interval_cost.getLb()) {
        if (auto con_cost = bound.getDefiningOp<ConstantCostOp>()) {
          lb = &con_cost.getBody();
        } else {
          llvm::outs()
              << "interval lower bound must be a perfify.constant_cost\n";
          lb = std::nullopt;
        }
      }
      if (mlir::Value bound = interval_cost.getUb()) {
        if (auto con_cost = bound.getDefiningOp<ConstantCostOp>()) {
          ub = &con_cost.getBody();
        } else {
          llvm::outs()
              << "interval upper bound must be a perfify.constant_cost\n";
          ub = std::nullopt;
        }
      }
      if (lb && ub) {
        interval_costs.insert({interval_cost.getResult(), {*lb, *ub}});
      }
    } else if (auto arg = dyn_cast<ArgOp>(op)) {
      NamedAttribute arg_attr = op->getAttrs()[0];
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(arg_attr.getValue())) {
        args[op->getResult(0)] = intAttr.getInt();
      }
    } else if (auto cond = dyn_cast<ConditionsOp>(op)) {
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

    } else if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      funcMap[func.getNameAttr()] = &func.getFunctionBody();
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      llvm::SmallVector<mlir::Operation *> allOps;
      if (analysis_func != NULL) {
        analysis_func->walk(
            [&allOps](mlir::Operation *op) { allOps.push_back(op); });
      }
      for (Operation *region_op : allOps) {
        auto cost_map_res =
            cost_map.find(region_op->getName().getStringRef().str());
        if (cost_map_res != cost_map.end()) {
          std::string cost_v = cost_str + std::to_string(cost_var.size());
          cost_var.push_back(
              solver.ctx().int_const(cost_v.c_str())); // create a new cost var
          z3::expr post_cost = cost_var[cost_var.size() - 1];
          z3::expr pre_cost = cost_var[cost_var.size() - 2];
          z3::expr op_cost = evaluateCostRegion(cost_map_res->second, solver);
          z3::expr cost_expr =
              (post_cost == pre_cost + op_cost); // increment per op cost

          solver.add(cost_expr); // add to solver
        } else {
          llvm::outs() << "unknown cost for operation "
                       << region_op->getName().getStringRef().str() << "\n";
        }
      }
    } else if (auto cmpOp = dyn_cast<CompareOp>(op)) {
      CmpPredicate pred = cmpOp.getPredicateAttr().getValue();
      auto region_num = cmpOp->getParentRegion()->getRegionNumber();
      z3::context &ctx = solver.ctx();
      z3::expr cost = region_num == 0 ? cost_var[0] : cost_var.back();
      mlir::Value interval;

      for (mlir::Value operand : cmpOp->getOperands()) {
        if (isa<IntervalType>(operand.getType()))
          interval = operand;
      }
      std::optional<z3::expr> holds;
      if (interval) {
        auto bounds = interval_costs.find(interval);
        if (bounds == interval_costs.end()) {
          llvm::outs() << "no interval cost registered for conditions region "
                       << region_num << "\n";
        } else {
          auto lbRegion = bounds->second.first;
          auto ubRegion = bounds->second.second;
          std::optional<z3::expr> lb, ub;
          if (lbRegion)
            lb = evaluateCostRegion(lbRegion, solver);
          if (ubRegion)
            ub = evaluateCostRegion(ubRegion, solver);
          z3::expr in_interval = ctx.bool_val(true);
          if (lb)
            in_interval = in_interval && (*lb <= cost);
          if (ub)
            in_interval = in_interval && (cost <= *ub);
          switch (pred) {
          case CmpPredicate::eq:
            holds = in_interval;
            break;
          case CmpPredicate::ne:
            holds = !in_interval;
            break;
          case CmpPredicate::lt:
            holds = lb ? z3::expr(cost < *lb) : ctx.bool_val(false);
            break;
          case CmpPredicate::le:
            holds = lb ? z3::expr(cost <= *lb) : ctx.bool_val(false);
            break;
          case CmpPredicate::gt:
            holds = ub ? z3::expr(cost > *ub) : ctx.bool_val(false);
            break;
          case CmpPredicate::ge:
            holds = ub ? z3::expr(cost >= *ub) : ctx.bool_val(false);
            break;
          }
        }
      } else {
        auto pre_post_cost = constant_costs.find(
            static_cast<mlir::enzyme::perfify::HoareStates>(region_num));
        if (pre_post_cost == constant_costs.end()) {
          llvm::outs() << "no constant cost registered for conditions region "
                       << region_num << "\n";
        } else {
          z3::expr expected_cost =
              evaluateCostRegion(pre_post_cost->second, solver);
          switch (pred) {
          case CmpPredicate::eq:
            holds = cost == expected_cost;
            break;
          case CmpPredicate::ne:
            holds = cost != expected_cost;
            break;
          case CmpPredicate::lt:
            holds = cost < expected_cost;
            break;
          case CmpPredicate::le:
            holds = cost <= expected_cost;
            break;
          case CmpPredicate::gt:
            holds = cost > expected_cost;
            break;
          case CmpPredicate::ge:
            holds = cost >= expected_cost;
            break;
          }
        }
      }
      if (holds) { // check that we don't have an empty hypothesis
        solver.add(region_num == 0 ? *holds : !*holds);
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
// todo: trigger traceup from the provided register argument, evaluate or
// fetch the cmp set up the hoare triple here
#ifdef Z3_DEBUG_PERFIFY
      std::cout << solver << std::endl;
      llvm::outs() << (((region_num == 0 &&
                         check_res ==
                             1) || // the precondition should be satisfiable
                        (region_num != 0 &&
                         check_res == 0)) // the postcondition should be unsat
                                          // since we negated the predicate
                           ? "Met perf check!"
                           : "Did not meet perf check")
                   << "\n";
#endif
      auto region_num = assumeOp->getParentRegion()->getRegionNumber();
      auto check_res = solver.check();
      if ((region_num == 0 && check_res == 1) ||
          (region_num != 0 && check_res == 0)) {
        assumeOp.setSatresAttr(BoolAttr::get(&getContext(), true));
      } else {
        assumeOp.setSatresAttr(BoolAttr::get(&getContext(), false));
      }
    }
    for (Region &region : op->getRegions())
      visitRegion(region, state, solver, cost_var);
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
