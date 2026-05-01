#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "z3++.h"
#include <cstdint>
#include <string>
#include <iostream>
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
std::unordered_map<std::string, int64_t> cost_map;
llvm::DenseMap<mlir::Value, int64_t> args;
std::unordered_map<HoareStates, int64_t> constant_costs;
llvm::DenseMap<mlir::StringAttr, mlir::Region *> funcMap;
std::vector<z3::expr> cost_var;
const std::string cost_str = "cost";

namespace {

struct SimpleCycleAnalysisPass
    : public enzyme::perfify::impl::SimpleCycleAnalysisPassBase<
          SimpleCycleAnalysisPass> {
  using SimpleCycleAnalysisPassBase::SimpleCycleAnalysisPassBase;
  mlir::Region *analysis_func;
  void runOnOperation() override {
    Operation *op = getOperation();
    mlir::AsmState state(op);
    z3::context c;
    z3::solver s{c};
    cost_var.push_back(c.int_const("cost0"));
    visitOperation(op, &state, s);
  }
  void visitOperation(Operation *op, AsmState *state, z3::solver& s) {
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
          std::string cost_v = std::to_string(cost_var.size());
          std::string fmt = cost_str + cost_v;
          cost_var.push_back(s.ctx().int_const(fmt.c_str())); // create a new cost var
          z3::expr post_cost = cost_var[cost_var.size() - 1];
          z3::expr pre_cost = cost_var[cost_var.size() - 2];
          int64_t cost_val = cost_map_res->second;
          z3::expr cost_expr = (post_cost == pre_cost + s.ctx().int_val(cost_val)); // increment per op cost
          s.add(cost_expr); // add to solver
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
            z3::expr p = (cost_var[0] == s.ctx().int_val(pre_post_cost->second)); // precondition
            z3::expr q = (cost_var[cost_var.size() - 1] != s.ctx().int_val(pre_post_cost->second)); // postcondition
            if (region_num == 0) {
              s.add(p);
            } else {
              s.add(q); // safer to add the false?
            }
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      // todo: trigger traceup from the provided register argument, evaluate or
      // fetch the cmp set up the hoare triple here
      llvm::outs() << s.check() << "\n";
    }
    for (Region &region : op->getRegions())
      visitRegion(region, state, s);
  }

  void visitRegion(Region &region, AsmState *state, z3::solver& s) {
    for (Block &block : region.getBlocks())
      visitBlock(block, state, s);
  }

  void visitBlock(Block &block, AsmState *state, z3::solver& s) {
    for (Operation &op : block.getOperations())
      visitOperation(&op, state, s);
  }
};

} // namespace
