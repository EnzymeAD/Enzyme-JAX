#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "z3++.h"
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
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
enum class HoareStates { Pre = 0, Post = 1 };
const std::string cost_str = "cost";

namespace {

struct SimpleCycleAnalysisPass
    : public enzyme::perfify::impl::SimpleCycleAnalysisPassBase<
          SimpleCycleAnalysisPass> {
  using SimpleCycleAnalysisPassBase::SimpleCycleAnalysisPassBase;
  mlir::Region *analysis_func;
  std::unordered_map<std::string, mlir::Region *> cost_map;
  llvm::DenseMap<mlir::Value, int64_t> args;
  std::unordered_map<HoareStates, mlir::Region *> constant_costs;
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
      HoareStates pre_post = static_cast<HoareStates>(
          con_cost->getParentRegion()->getRegionNumber());
      constant_costs.insert({pre_post, &con_cost.getBody()});
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

    } else if (auto func = dyn_cast<mlir::func::FuncOp>(op)) {
      funcMap[func.getNameAttr()] = &func.getBody();
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      llvm::SmallVector<mlir::Operation *> allOps;

      analysis_func->walk(
          [&allOps](mlir::Operation *op) { allOps.push_back(op); });
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
      auto pre_post_cost =
          constant_costs.find(static_cast<HoareStates>(region_num));
      if (pre_post_cost == constant_costs.end()) {
        llvm::outs() << "no constant cost registered for conditions region "
                     << region_num << "\n";
      } else {
        z3::expr expected_cost =
            evaluateCostRegion(pre_post_cost->second, solver);
        z3::expr p = solver.ctx().bool_val(true);
        z3::expr q = solver.ctx().bool_val(
            false); // should return false if nothing else?
        if (region_num == 0) {
          if (pred == CmpPredicate::eq) {
            p = z3::expr(cost_var[0] == expected_cost); // precondition
          } else if (pred == CmpPredicate::ne) {
            p = z3::expr(cost_var[0] != expected_cost);
          } else if (pred == CmpPredicate::lt) {
            p = z3::expr(cost_var[0] < expected_cost);
          } else if (pred == CmpPredicate::le) {
            p = z3::expr(cost_var[0] <= expected_cost);
          } else if (pred == CmpPredicate::gt) {
            p = z3::expr(cost_var[0] > expected_cost);
          } else if (pred == CmpPredicate::ge) {
            p = z3::expr(cost_var[0] >= expected_cost);
          }
          solver.add(p);
        } else {
          if (pred == CmpPredicate::eq) {
            q = z3::expr(cost_var[cost_var.size() - 1] !=
                         expected_cost); // postcondition
          } else if (pred == CmpPredicate::ne) {
            q = z3::expr(cost_var[cost_var.size() - 1] == expected_cost);
          } else if (pred == CmpPredicate::lt) {
            q = z3::expr(cost_var[cost_var.size() - 1] >= expected_cost);
          } else if (pred == CmpPredicate::le) {
            q = z3::expr(cost_var[cost_var.size() - 1] > expected_cost);
          } else if (pred == CmpPredicate::gt) {
            q = z3::expr(cost_var[cost_var.size() - 1] <= expected_cost);
          } else if (pred == CmpPredicate::ge) {
            q = z3::expr(cost_var[cost_var.size() - 1] < expected_cost);
          }
          solver.add(q); // if satisfiable -> assignment exists s.t. perf
                         // counter doesn't equal expected value?
        }
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
