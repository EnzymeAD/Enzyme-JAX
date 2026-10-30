#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  mlir::Region *analysisFunc = nullptr;
  std::unordered_map<std::string, mlir::Region *> costMap;
  llvm::DenseMap<mlir::Value, int64_t> args;
  struct HypothesisArgs {
    mlir::Region *lb = nullptr, *ub = nullptr;
  };
  std::unordered_map<HoareStates, HypothesisArgs> hypotheses;
  llvm::DenseMap<mlir::StringAttr, mlir::Region *> funcRegionMap;
  using ProvenBound = std::optional<std::pair<CmpPredicate, HypothesisArgs>>;
  llvm::DenseMap<ConditionsOp, std::pair<ProvenBound, ProvenBound>>
      funcBoundsMap;
  llvm::DenseMap<mlir::StringAttr, unsigned int> loopsCount;

  void runOnOperation() override {
    Operation *op = getOperation();
    mlir::AsmState state(op);
    z3::context ctx;
    z3::solver solver(ctx);
    std::vector<z3::expr> cost_var;
    cost_var.push_back(solver.ctx().int_const("cost0"));
    solver.add(cost_var.front() >= 0); // the starting cost is never negative

    visitOperation(op, &state, solver, cost_var);
  }

  void visitOperation(Operation *op, AsmState *state, z3::solver &solver,
                      std::vector<z3::expr> &cost_var) {
    if (auto costOp = dyn_cast<CostOp>(op)) {
      std::string target_op = costOp.getTargetOp().str();
      if (costMap.find(target_op) == costMap.end()) {
        costMap[target_op] =
            &costOp.getBody(); // store the body of the region for eval
      } else {
        llvm::outs() << "Duplicate cost definition for " << target_op << "\n";
      }
    } else if (auto arg = dyn_cast<ArgOp>(
                   op)) { // this is an argument to set up the cost hypothesis.
                          // It does not refer to the target function's arity
      NamedAttribute argAttr = op->getAttrs()[0];
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(argAttr.getValue())) {
        args[op->getResult(0)] = intAttr.getInt();
      }
    } else if (auto cond =
                   dyn_cast<ConditionsOp>(op)) { // see a new Hoare triple
      analysisFunc = nullptr;
      Attribute condAttr = op->getAttrs()[0].getValue();
      auto condRef = mlir::cast<mlir::FlatSymbolRefAttr>(condAttr);

      bool verify_huh =
          mlir::cast<mlir::BoolAttr>(op->getAttrs()[1].getValue())
              .getValue(); // can optionally disable verification, so check here
      if (verify_huh) {
        auto res = funcRegionMap.find(condRef.getAttr());
        if (res != funcRegionMap.end()) {
          analysisFunc = res->second;
        } else {
          llvm::outs() << "Could not find " << condRef.getAttr()
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
      funcRegionMap[func.getNameAttr()] = &func.getFunctionBody();
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      if (analysisFunc != nullptr)
        accumulateRegionCost(*analysisFunc, solver.ctx().int_val(1), op, solver,
                             cost_var);
    } else if (auto hyp = dyn_cast<HypothesisOp>(op)) {
      // loop hypotheses have no pre/post; loopTripCount reads them
      if (std::optional<HoareStates> pre_post = hyp.getPrePost()) {
        auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
        if (auto bounds = boundRegions(cmpOp)) {
          hypotheses[*pre_post] = *bounds;
          z3::expr holds =
              buildHypothesis(cmpOp.getPredicate(), *bounds,
                              *pre_post == HoareStates::Pre ? cost_var.front()
                                                            : cost_var.back(),
                              solver);
          solver.add(*pre_post == HoareStates::Pre ? holds : !holds);
        } else {
          llvm::outs() << "no bound found for "
                       << stringifyHoareStates(*pre_post) << " hypothesis\n";
        }
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      // where the z3 checks actually happen
      bool met = false;
      if (auto hyp = assumeOp.getCondition().getDefiningOp<HypothesisOp>()) {
        auto check_res = solver.check();
        met = hyp.getPrePost() == HoareStates::Pre ? check_res == z3::sat
                                                   : check_res == z3::unsat;
        if (met) {
          auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
          auto args = hypotheses.find(*hyp.getPrePost());
          if (args != hypotheses.end()) {
            // read lb/ub with the predicate, as in buildHypothesis
            auto &bounds =
                funcBoundsMap[assumeOp->getParentOfType<ConditionsOp>()];
            (hyp.getPrePost() == HoareStates::Pre ? bounds.first
                                                  : bounds.second) =
                std::make_pair(cmpOp.getPredicate(), args->second);
          }
        }
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
    }
    for (Region &region : op->getRegions())
      visitRegion(region, state, solver, cost_var);
  }

  // Chain the cost of every op in `region` (nested regions first, matching a
  // post-order walk) onto cost_var, each scaled by `scale`, the number of times
  // `region` executes.
  void accumulateRegionCost(mlir::Region &region, z3::expr scale,
                            Operation *fnCostOp, z3::solver &solver,
                            std::vector<z3::expr> &cost_var) {
    for (Operation *region_op : llvm::make_pointer_range(region.getOps())) {
      // perfify annotations (e.g. loop hypotheses) cost nothing
      if (isa_and_nonnull<PerfifyDialect>(region_op->getDialect()))
        continue;
      if (auto whileOp = dyn_cast<scf::WhileOp>(region_op)) {
        // the condition region runs once more than the body
        z3::expr trips =
            loopTripCount(region_op, std::nullopt, whileOp.getAfter(), solver);
        accumulateRegionCost(
            whileOp.getBefore(), scale * (trips + 1), fnCostOp, solver,
            cost_var); // account for the cost of the head check
        accumulateRegionCost(whileOp.getAfter(), scale * trips, fnCostOp,
                             solver, cost_var); // and for the body
      } else {
        // a loop body executes once per iteration
        z3::expr nestedScale = scale;
        if (auto forOp = dyn_cast<scf::ForOp>(region_op)) {
          nestedScale =
              scale * loopTripCount(region_op, forOp.getStaticTripCount(),
                                    forOp.getRegion(), solver);
        }
        for (Region &nested : region_op->getRegions()) {
          accumulateRegionCost(nested, nestedScale, fnCostOp, solver, cost_var);
        }
      }

      if (isa<func::CallOp, LLVM::CallOp>(region_op)) {
        auto targetFunc = dyn_cast<SymbolRefAttr>(
            cast<CallOpInterface>(region_op).getCallableForCallee());
        if (!targetFunc) {
          llvm::outs() << "no cost for indirect call\n";
          continue;
        }
        // use the callee's proven post bound as the cost of the call
        ProvenBound calleeBound;
        for (auto cond : fnCostOp->getParentOfType<AssumptionsOp>()
                             .getBody()
                             .getOps<ConditionsOp>()) {
          auto it = funcBoundsMap.find(cond);
          if (cond.getFuncHandleAttr() == targetFunc &&
              it != funcBoundsMap.end())
            calleeBound = it->second.second;
        }
        if (!calleeBound) {
          llvm::outs() << "no proven cost bound for " << targetFunc << "\n";
          continue;
        }
        std::string cost_v = cost_str + std::to_string(cost_var.size());
        z3::expr call_cost = solver.ctx().int_const((cost_v + "_call").c_str());
        solver.add(call_cost >= 0); // a call never costs less than nothing
        cost_var.push_back(solver.ctx().int_const(cost_v.c_str()));
        solver.add(cost_var.back() ==
                   cost_var[cost_var.size() - 2] + scale * call_cost);
        solver.add(buildHypothesis(calleeBound->first, calleeBound->second,
                                   call_cost, solver));
      } else {
        auto costMapRes =
            costMap.find(region_op->getName().getStringRef().str());
        if (costMapRes != costMap.end()) {
          std::string cost_v = cost_str + std::to_string(cost_var.size());
          cost_var.push_back(
              solver.ctx().int_const(cost_v.c_str())); // create a new cost var
          z3::expr post_cost = cost_var[cost_var.size() - 1];
          z3::expr pre_cost = cost_var[cost_var.size() - 2];
          z3::expr op_cost = evaluateCostRegion(costMapRes->second, solver);
          z3::expr cost_expr =
              (post_cost ==
               pre_cost + scale * op_cost); // increment per op cost

          solver.add(cost_expr); // add to solver
        } else {
          llvm::outs() << "unknown cost for operation "
                       << region_op->getName().getStringRef().str() << "\n";
        }
      }
    }
  }

  z3::expr loopTripCount(Operation *loop, std::optional<APInt> staticCount,
                         Region &body, z3::solver &solver) {
    // We have an iteration bound associated with the loop, fetch
    z3::context &ctx = solver.ctx();
    llvm::SmallString<40> staticLoopStr;
    auto loopHyps = body.getOps<HypothesisOp>();
    // for fixed iteration loops (e.g. for i < 100; i++)
    if (staticCount) {
      if (!loopHyps.empty()) {
        llvm::outs() << "ignoring trip count hypothesis on a loop with a "
                        "static trip count\n";
      }
      staticCount->toString(staticLoopStr, 10, false);
      return ctx.int_val(staticLoopStr.c_str());
    } else {
      // without statically determined bounds, rely on the user provided
      // constraints.
      auto module = loop->getParentOfType<ModuleOp>();
      StringAttr funcName =
          loop->getParentOfType<FunctionOpInterface>().getNameAttr();
      std::string tripName = (funcName.getValue() + "loop_trip" +
                              std::to_string(loopsCount[funcName]++))
                                 .str();
      OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
      DefSymbolOp::create(b, loop->getLoc(), b.getStringAttr(tripName),
                          TypeAttr::get(b.getI64Type()), StringAttr());
      // same z3 name perfify.sym_cost would give the symbol
      z3::expr trips = ctx.int_const(("sym_" + tripName).c_str());
      solver.add(trips >= 0); // trip counts are never negative
      if (!loopHyps.empty()) {
        // the hypothesis is assumed to hold, like a pre-condition
        auto cmpOp = (*loopHyps.begin()).getCmp().getDefiningOp<CompareOp>();
        if (auto bounds = boundRegions(cmpOp))
          solver.add(
              buildHypothesis(cmpOp.getPredicate(), *bounds, trips, solver));
        else
          llvm::outs() << "no bound found for loop hypothesis\n";
      }
      return trips;
    }
  }

  std::optional<HypothesisArgs> boundRegions(CompareOp cmpOp) {
    Operation *bound = cmpOp.getBoundOp();
    if (auto estCost = dyn_cast_or_null<EstimatedCostOp>(bound)) {
      return HypothesisArgs{&estCost.getBody(), &estCost.getBody()};
    } else if (auto interval = dyn_cast_or_null<IntervalOp>(bound)) {
      auto lb = interval.getMinCostOp();
      auto ub = interval.getMaxCostOp();
      return HypothesisArgs{lb ? &lb.getBody() : nullptr,
                            ub ? &ub.getBody() : nullptr};
    }
    return std::nullopt;
  }

  z3::expr buildHypothesis(CmpPredicate pred, const HypothesisArgs &bounds,
                           z3::expr cost, z3::solver &solver) {
    z3::context &ctx = solver.ctx();
    std::optional<z3::expr> lb, ub;
    z3::expr in_interval = ctx.bool_val(true);
    if (bounds.lb) {
      lb = evaluateCostRegion(bounds.lb, solver);
      in_interval = in_interval && (*lb <= cost);
    }
    if (bounds.ub) {
      ub = evaluateCostRegion(bounds.ub, solver);
      in_interval = in_interval && (cost <= *ub);
    }

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
    for (mlir::Operation &regionOp : costRegion->getOps()) {
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
