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
  std::unordered_map<std::string, int64_t> cost_map;
  llvm::DenseMap<mlir::Value, int64_t> args;
  std::unordered_map<HoareStates, int64_t> constant_costs;
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
      if (!op->getAttrs().empty()) {
        NamedAttribute cost_attr = op->getAttrs()[0];
        NamedAttribute op_attr = op->getAttrs()[1];
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(cost_attr.getValue())) {
          if (auto strAttr =
                  mlir::dyn_cast<mlir::StringAttr>(op_attr.getValue())) {
            int64_t value = intAttr.getInt();
            if (cost_map.find(strAttr.getValue().str()) == cost_map.end()) {
              cost_map[strAttr.getValue().str()] = value;
            }
          }
        }
      }
    } else if (auto con_cost = dyn_cast<ConstantCostOp>(op)) {
      if (!op->getAttrs().empty()) {
        NamedAttribute est_cost = op->getAttrs()[0];
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(est_cost.getValue())) {
          HoareStates pre_post = static_cast<HoareStates>(
              con_cost->getParentRegion()->getRegionNumber());
          constant_costs.insert({pre_post, intAttr.getInt()});
        }
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

    } else if (auto func = dyn_cast<mlir::func::FuncOp>(op)) {
      funcMap[func.getNameAttr()] = &func.getBody();
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      // TODO: Make this symbolic
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
          int64_t cost_val = cost_map_res->second;
          z3::expr cost_expr =
              (post_cost == pre_cost + solver.ctx().int_val(
                                           cost_val)); // increment per op cost

          solver.add(cost_expr); // add to solver
        } else {
          llvm::outs() << "unknown cost for operation "
                       << region_op->getName().getStringRef().str() << "\n";
        }
      }
    } else if (auto cmpOp = dyn_cast<CompareOp>(op)) {
      CmpPredicate pred = cmpOp.getPredicateAttr().getValue();
      if (pred == CmpPredicate::eq) {
        auto region_num = cmpOp->getParentRegion()->getRegionNumber();
        auto pre_post_cost =
            constant_costs.find(static_cast<HoareStates>(region_num));
        z3::expr p =
            (cost_var[0] ==
             solver.ctx().int_val(pre_post_cost->second)); // precondition
        z3::expr q =
            (cost_var[cost_var.size() - 1] !=
             solver.ctx().int_val(pre_post_cost->second)); // postcondition
        if (region_num == 0) {
          solver.add(p);
        } else {
          solver.add(q); // if satisfiable -> assignment exists s.t. perf
                         // counter doesn't equal expected value?
        }
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      // todo: trigger traceup from the provided register argument, evaluate or
      // fetch the cmp set up the hoare triple here
      std::cout << solver << std::endl;
      auto region_num = assumeOp->getParentRegion()->getRegionNumber();
      auto check_res = solver.check();
      llvm::outs() << (((region_num == 0 && check_res == 1) ||
                        (region_num != 0 && check_res == 0))
                           ? "Met perf check!"
                           : "Did not meet perf check")
                   << "\n"; // this should always be true
    }
    for (Region &region : op->getRegions())
      visitRegion(region, state, solver, cost_var);
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
