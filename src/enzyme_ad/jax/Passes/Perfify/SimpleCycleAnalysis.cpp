#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/CostLookup.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "z3++.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/SaveAndRestore.h"
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
  llvm::DenseMap<ConditionsOp, ProvenBound> funcBoundsMap;
  llvm::DenseMap<StringAttr, std::pair<CmpPredicate, HypothesisArgs>>
      funcCostMap;
  bool unprovenAssertions =
      false; // the current function has a cost we cannot bound
  bool preAllowsCallStart = false;
  bool convexOnly = false;
  unsigned int freshNameCount = 0;

  std::string freshName(const std::string &prefix) {
    return prefix + std::to_string(freshNameCount++);
  }

  void visitOperationWithFreshSolver(Operation *op, z3::context &ctx) {
    llvm::SaveAndRestore savedFunc(analysisFunc);
    llvm::SaveAndRestore savedHypotheses(hypotheses);
    llvm::SaveAndRestore savedUnprovenAssertions(unprovenAssertions);
    llvm::SaveAndRestore savedCallStart(preAllowsCallStart);
    llvm::SaveAndRestore savedConvexOnly(convexOnly);

    z3::solver solver(ctx);
    std::vector<z3::expr> cost_var;
    cost_var.push_back(ctx.int_const("cost0"));
    solver.add(cost_var.front() >= 0);

    visitOperation(op, solver, cost_var);
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto costOp = dyn_cast<CostOp>(op)) {
        std::string target_op = costOp.getTargetOp().str();
        if (costMap.find(target_op) == costMap.end()) {
          costMap[target_op] =
              &costOp.getBody(); // store the body of the region for eval
        } else {
          llvm::outs() << "Duplicate cost definition for " << target_op << "\n";
        }
      } else if (auto func = dyn_cast<FunctionOpInterface>(op)) {
        funcRegionMap[func.getNameAttr()] = &func.getFunctionBody();
      } else if (auto funcCostOp = dyn_cast<FuncBoundOp>(op)) {
        if (auto bounds = boundRegions(funcCostOp.getBoundOp()))
          funcCostMap.insert({funcCostOp.getFuncHandleAttr().getAttr(),
                              {funcCostOp.getPredicate(), *bounds}});
      }
    });
    Operation *op = getOperation();
    z3::set_param("pp.single_line", true); // set the printer options, turn off line breaks
    z3::set_param("pp.min_alias_size", std::numeric_limits<int>::max()); // turn off aliasing
    z3::set_param("pp.max_depth", std::numeric_limits<int>::max()); // abbreviate terms
    z3::context ctx;
    visitOperationWithFreshSolver(op, ctx);
  }

  void visitOperation(Operation *op, z3::solver &solver,
                      std::vector<z3::expr> &cost_var) {
    if (auto arg = dyn_cast<ArgOp>(
            op)) { // this is an argument to set up the cost hypothesis.0
                   // It does not refer to the target function's arity
      if (auto intAttr = arg.getVal()) {
        args[op->getResult(0)] = intAttr;
      }
    } else if (auto cond =
                   dyn_cast<ConditionsOp>(op)) { // see a new Hoare triple
      if (!funcBoundsMap.try_emplace(cond)
               .second) { // skip triple that was already checked
        return;
      }
      analysisFunc = nullptr;
      FlatSymbolRefAttr condRef = cond.getFuncHandleAttr();

      bool verify_huh = cond.getVerifyHuh();
      if (verify_huh) {
        auto res = funcRegionMap.find(condRef.getAttr());
        if (res != funcRegionMap.end()) {
          analysisFunc = res->second;
        } else {
          llvm::outs() << "Could not find " << condRef.getAttr()
                       << " in the function map\n";
        }
      } else {
        llvm::outs() << "verify_huh is false for " << condRef << "\n";
        return;
      }
      // Clear the solver state between functions
      hypotheses.clear();
      preAllowsCallStart = false;
      convexOnly = false;
      unprovenAssertions = false;
      cost_var.erase(cost_var.begin() + 1, cost_var.end());
      solver.push();

      for (Region &region : op->getRegions()) {
        visitRegion(region, solver, cost_var);
      }
      solver.pop();
      return;
    } else if (auto funcCost = dyn_cast<FnCostOp>(op)) {
      if (analysisFunc != nullptr && !analysisFunc->empty()) {
        accumulateRegionCost(*analysisFunc, solver.ctx().int_val(1), op, solver,
                             cost_var);
      } else {
        llvm::outs() << "No function body for " << funcCost->getLoc() << "\n";
        unprovenAssertions = true;
      }
    } else if (auto hyp = dyn_cast<HypothesisOp>(op)) {
      // loop hypotheses have no pre/post; loopTripCount reads them
      if (std::optional<HoareStates> pre_post = hyp.getPrePost()) {
        auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
        if (auto bounds = boundRegions(cmpOp.getBoundOp())) {
          hypotheses[*pre_post] = *bounds;
          // a post hypothesis is checked at its perfify.assume
          if (*pre_post == HoareStates::Pre) {
            if (auto holds = buildHypothesis(cmpOp.getPredicate(), *bounds,
                                             cost_var.front(), solver)) {
              solver.add(*holds);
              z3::solver callStart(solver.ctx());
              callStart.add(cost_var.front() == 0 && !*holds);
              preAllowsCallStart =
                  callStart.check() ==
                  z3::unsat; // we want to make sure the callee's precondition
                             // is contained within our current precondition
            } else {
              unprovenAssertions = true;
            }
          }
        } else {
          llvm::outs() << "no bound found for "
                       << stringifyHoareStates(*pre_post) << " hypothesis\n";
        }
      }
    } else if (auto assumeOp = dyn_cast<AssumeOp>(op)) {
      // where the z3 checks actually happen
      auto hyp = assumeOp.getCondition().getDefiningOp<HypothesisOp>();
      std::optional<std::string> invalidRegion;
      auto cond = assumeOp->getParentOfType<ConditionsOp>();
      bool met = hyp.getPrePost() == HoareStates::Pre
                     ? solver.check() == z3::sat
                     : cond.getLeanProven() || checkPostHypothesis(hyp, solver, cost_var, invalidRegion);
      if (met && hyp.getPrePost() == HoareStates::Post && preAllowsCallStart) {
        auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
        auto args = hypotheses.find(*hyp.getPrePost());
        if (args != hypotheses.end()) {
          // read lb/ub with the predicate, as in buildHypothesis
          funcBoundsMap[assumeOp->getParentOfType<ConditionsOp>()] =
              std::make_pair(cmpOp.getPredicate(), args->second);
        }
      }
#ifdef Z3_DEBUG_PERFIFY
      std::cout << solver << std::endl;
      llvm::outs() << (met ? "Met perf check!" : "Did not meet perf check")
                   << "\n";
#endif
      assumeOp.setSatresAttr(BoolAttr::get(&getContext(), met));
      if (!cond.getLeanProven()) {
        if (invalidRegion)
          cond.setInvalidRegion(*invalidRegion);
        else
          cond.removeInvalidRegionAttr();
      }
    }
    for (Region &region : op->getRegions())
      visitRegion(region, solver, cost_var);
  }

  bool checkPostHypothesis(HypothesisOp hyp, z3::solver &solver,
                           const std::vector<z3::expr> &cost_var, std::optional<std::string> &invalidRegionOut) {
    auto cmpOp = hyp.getCmp().getDefiningOp<CompareOp>();
    auto bounds = boundRegions(cmpOp.getBoundOp());
    bool postHypothesisOutput;
    std::optional<z3::expr> holds =
        bounds ? buildHypothesis(cmpOp.getPredicate(), *bounds, cost_var.back(),
                                 solver)
               : std::nullopt;
    if (convexOnly && cmpOp.getPredicate() == CmpPredicate::ne) {
      llvm::outs()
          << "Cannot prove 'ne' for a function that repeats a symbol\n";
      return false;
    }
    if (!holds || solver.check() != z3::sat) {
      llvm::outs() << "constraints are unsatisfiable or undecided\n";
      postHypothesisOutput = false;
    } else {
      solver.push();
      solver.add(!*holds);
      z3::check_result negated = solver.check();
      postHypothesisOutput = negated == z3::unsat && !unprovenAssertions;
      if (negated == z3::sat) {
        invalidRegionOut = failureRegion(solver);
      }
      solver.pop();
    }
    return postHypothesisOutput;
  }

  void accumulateRegionCost(mlir::Region &region, z3::expr scale,
                            Operation *fnCostOp, z3::solver &solver,
                            std::vector<z3::expr> &cost_var) {
    if (!region.empty()) {
      if (region.hasOneBlock()) {
        accumulateBlockCost(region.front(), scale, fnCostOp, solver, cost_var);
      } else {
        llvm::SmallPtrSet<Block *, 8> cyclicBlocks;
        for (auto scc = llvm::scc_begin(&region); !scc.isAtEnd(); ++scc)
          if (scc.hasCycle())
            cyclicBlocks.insert(scc->begin(), scc->end());
        for (Block &block : region) {
          bool cyclic = cyclicBlocks.contains(&block);
          if (block.isEntryBlock() && !cyclic) {
            accumulateBlockCost(block, scale, fnCostOp, solver, cost_var);
            continue;
          }
          z3::expr count = solver.ctx().int_const(freshName("block").c_str());
          solver.add(count >= 0);
          if (!cyclic) {
            solver.add(count <= scale);
          }
          accumulateBlockCost(block, count, fnCostOp, solver, cost_var);
        }
      }
    }
  }

  void accumulateBlockCost(mlir::Block &block, z3::expr scale,
                           Operation *fnCostOp, z3::solver &solver,
                           std::vector<z3::expr> &cost_var) {
    for (Operation *region_op : llvm::make_pointer_range(block)) {
      // perfify annotations (e.g. loop hypotheses) cost nothing
      if (isa_and_nonnull<PerfifyDialect>(region_op->getDialect()))
        continue;
      if (isa<scf::WhileOp, stablehlo::WhileOp>(region_op)) {
        // the condition region runs once more than the body
        mlir::Region &before = region_op->getRegion(0);
        mlir::Region &after = region_op->getRegion(1);
        convexOnly =
            convexOnly || !z3::eq(scale.simplify(), scale.ctx().int_val(1));
        z3::expr trips = loopTripCount(region_op, std::nullopt, after, solver);
        accumulateRegionCost(
            before, scale * (trips + 1), fnCostOp, solver,
            cost_var); // account for the cost of the head check
        accumulateRegionCost(after, scale * trips, fnCostOp, solver,
                             cost_var); // and for the body
      } else if (isa<scf::ForOp, affine::AffineForOp>(region_op)) {
        // a loop body executes once per iteration
        mlir::Region &body = region_op->getRegion(0);
        std::optional<APInt> staticCount =
            cast<LoopLikeOpInterface>(region_op).getStaticTripCount();
        if (!staticCount) {
          convexOnly =
              convexOnly || !z3::eq(scale.simplify(), scale.ctx().int_val(1));
        }
        accumulateRegionCost(
            body, scale * loopTripCount(region_op, staticCount, body, solver),
            fnCostOp, solver, cost_var);
      } else if (isa<scf::IfOp, affine::AffineIfOp, stablehlo::IfOp>(
                     region_op)) {
        z3::expr scaleIterations =
            solver.ctx().int_const(freshName("scale_iterations").c_str());
        solver.add(scaleIterations >= 0 && scaleIterations <= scale);
        accumulateRegionCost(region_op->getRegion(0), scaleIterations, fnCostOp,
                             solver, cost_var);
        accumulateRegionCost(region_op->getRegion(1), scale - scaleIterations,
                             fnCostOp, solver, cost_var);
      } else {
        // we don't know how often the regions of any other op execute, nor how
        // they relate, so generate a scaling factor per region
        for (Region &nested : region_op->getRegions()) {
          z3::expr count = solver.ctx().int_const(freshName("region").c_str());
          solver.add(count >= 0);
          accumulateRegionCost(nested, scale * count, fnCostOp, solver,
                               cost_var);
        }
      }

      if (isa<func::CallOp, LLVM::CallOp>(region_op)) {
        auto targetFunc = dyn_cast<SymbolRefAttr>(
            cast<CallOpInterface>(region_op).getCallableForCallee());
        if (!targetFunc) {
          llvm::outs() << "no cost for indirect call\n";
          unprovenAssertions = true;
          continue;
        }
        SmallVector<std::pair<CmpPredicate, HypothesisArgs>> calleeBounds;
        auto givenCost = funcCostMap.find(targetFunc.getRootReference());
        if (givenCost != funcCostMap.end()) {
          calleeBounds.push_back(givenCost->second);
        } else {
          for (auto cond : fnCostOp->getParentOfType<AssumptionsOp>()
                               .getBody()
                               .getOps<ConditionsOp>()) {
            if (cond.getFuncHandleAttr() != targetFunc) {
              continue;
            }
            visitOperationWithFreshSolver(cond, solver.ctx());
            auto it = funcBoundsMap.find(cond);
            if (it != funcBoundsMap.end() && it->second) {
              calleeBounds.push_back(*it->second);
            }
          }
        }
        if (calleeBounds.empty()) {
          llvm::outs() << "no proven cost bound for " << targetFunc << "\n";
          unprovenAssertions = true;
          continue;
        }
        std::string cost_v = cost_str + std::to_string(cost_var.size());
        z3::expr call_cost = solver.ctx().int_const((cost_v + "_call").c_str());
        convexOnly =
            convexOnly || !z3::eq(scale.simplify(), scale.ctx().int_val(1));
        std::optional<z3::expr> callBound;
        for (auto &[pred, args] : calleeBounds) {
          callBound = buildHypothesis(pred, args, call_cost, solver);
          if (callBound) {
            solver.add(*callBound);
          } else {
            unprovenAssertions = true;
          }
        }
        solver.add(call_cost >= 0); // a call never costs less than nothing
        cost_var.push_back(solver.ctx().int_const(cost_v.c_str()));
        solver.add(cost_var.back() ==
                   cost_var[cost_var.size() - 2] + scale * call_cost);
      } else {
        auto costMapRes = findCost(costMap, region_op);
        if (costMapRes != costMap.end()) {
          std::optional<z3::expr> op_cost =
              evaluateCostRegion(costMapRes->second, solver);
          if (!op_cost) {
            unprovenAssertions = true;
            continue;
          }
          solver.add(*op_cost >= 0);
          std::string cost_v = cost_str + std::to_string(cost_var.size());
          cost_var.push_back(
              solver.ctx().int_const(cost_v.c_str())); // create a new cost var
          z3::expr post_cost = cost_var[cost_var.size() - 1];
          z3::expr pre_cost = cost_var[cost_var.size() - 2];
          z3::expr cost_expr =
              (post_cost ==
               pre_cost + scale * *op_cost); // increment per op cost

          solver.add(cost_expr); // add to solver
        } else {
          llvm::outs() << "unknown cost for operation "
                       << region_op->getName().getStringRef().str() << "\n";
          unprovenAssertions = true;
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
      std::string tripName;
      if (auto tripSym = loop->getAttrOfType<FlatSymbolRefAttr>(StringLiteral("perfify.loop_trip"))) {
        tripName = tripSym.getValue().str();
      } else {
        do {
          tripName = freshName((funcName.getValue() + "loop_trip").str());
        } while (SymbolTable::lookupSymbolIn(module, tripName));
        OpBuilder b = OpBuilder::atBlockBegin(module.getBody());
        DefSymbolOp::create(b, loop->getLoc(), b.getStringAttr(tripName), TypeAttr::get(b.getI64Type()), StringAttr());
        loop->setAttr(StringLiteral("perfify.loop_trip"), FlatSymbolRefAttr::get(b.getContext(), tripName));
      }
      // same z3 name perfify.sym_cost would give the symbol
      z3::expr trips = ctx.int_const(("sym_" + tripName).c_str());
      solver.add(trips >= 0); // trip counts are never negative
      if (!loopHyps.empty()) {
        // the hypothesis is assumed to hold, like a pre-condition
        auto cmpOp = (*loopHyps.begin()).getCmp().getDefiningOp<CompareOp>();
        if (auto bounds = boundRegions(cmpOp.getBoundOp())) {
          if (auto holds = buildHypothesis(cmpOp.getPredicate(), *bounds, trips,
                                           solver)) {
            solver.add(*holds);
          } else {
            unprovenAssertions = true;
          }
        } else {
          llvm::outs() << "no bound found for loop hypothesis\n";
        }
      }
      return trips;
    }
  }

  std::optional<HypothesisArgs> boundRegions(Operation *bound) {
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

  std::optional<z3::expr> buildHypothesis(CmpPredicate pred,
                                          const HypothesisArgs &bounds,
                                          z3::expr cost, z3::solver &solver) {
    z3::context &ctx = solver.ctx();
    std::optional<z3::expr> lb, ub;
    z3::expr in_interval = ctx.bool_val(true);
    if (bounds.lb) {
      lb = evaluateCostRegion(bounds.lb, solver);
      if (!lb)
        return std::nullopt;
      in_interval = in_interval && (*lb <= cost);
    }
    if (bounds.ub) {
      ub = evaluateCostRegion(bounds.ub, solver);
      if (!ub)
        return std::nullopt;
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

  std::optional<z3::expr> evaluateCostRegion(mlir::Region *costRegion,
                                             z3::solver &solver) {
    z3::context &ctx = solver.ctx();
    llvm::DenseMap<mlir::Value, z3::expr> valueExprs;
    std::optional<z3::expr> result;
    for (mlir::Operation &regionOp : costRegion->getOps()) {
      std::optional<z3::expr> expr;
      if (auto constOp = dyn_cast<arith::ConstantOp>(regionOp)) {
        auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
        if (!intAttr) {
          llvm::outs() << "non-integer constant in cost expression\n";
          return std::nullopt;
        }
        llvm::SmallString<40> valueStr;
        intAttr.getValue().toString(valueStr, 10, /*Signed=*/true);
        expr = ctx.int_val(valueStr.c_str());
      } else if (auto symOp = dyn_cast<SymCostOp>(regionOp)) {
        std::string symName = ("sym_" + symOp.getSymAttr().getValue()).str();
        expr = ctx.int_const(symName.c_str());
        solver.add(*expr >= 0); // symbolic costs are never negative
      } else if (isa<arith::AddIOp, arith::MulIOp>(regionOp)) {
        auto lhs = valueExprs.find(regionOp.getOperand(0));
        auto rhs = valueExprs.find(regionOp.getOperand(1));
        if (lhs == valueExprs.end() || rhs == valueExprs.end()) {
          llvm::outs() << "unresolvable operand in cost expression for "
                       << regionOp.getName().getStringRef() << "\n";
          return std::nullopt;
        }
        expr = isa<arith::AddIOp>(regionOp) ? (lhs->second + rhs->second)
                                            : (lhs->second * rhs->second);
      } else if (auto yieldOp = dyn_cast<YieldOp>(regionOp)) {
        if (mlir::Value yielded = yieldOp.getValue()) {
          auto it = valueExprs.find(yielded);
          if (it == valueExprs.end()) {
            llvm::outs() << "unresolvable yield in cost expression\n";
            return std::nullopt;
          }
          result = it->second;
        }
      } else {
        llvm::errs() << "unsupported op in cost region: "
                     << regionOp.getName().getStringRef() << "\n";
        return std::nullopt;
      }
      if (expr && regionOp.getNumResults() > 0) {
        valueExprs.insert({regionOp.getResult(0), *expr});
        result = expr;
      }
    }
    return result;
  }

  std::optional<std::string> failureRegion(z3::solver &solver) {
    z3::expr_vector F = solver.assertions();
    z3::expr res = z3::mk_and(F);
    z3::expr_vector kept(solver.ctx()), eliminated(solver.ctx());
    llvm::DenseSet<unsigned int> seen;
    SmallVector<z3::expr> worklist{res};
    while (!worklist.empty()) {
      z3::expr e = worklist.pop_back_val();
      if (!seen.insert(e.id()).second) {
        continue;
      }
      if (e.is_app() && e.num_args() == 0 && e.decl().decl_kind() == Z3_OP_UNINTERPRETED) {
        if (StringRef(e.decl().name().str()).starts_with("sym_")) {
          kept.push_back(e);
        } else {
          eliminated.push_back(e);
        }
      } else {
        for (unsigned int i = 0, n = e.num_args(); i != n; ++i) {
          worklist.push_back(e.arg(i));
        }
      }
    }
    if (!eliminated.empty()) {
      res = z3::exists(eliminated, res);
    }
    try {
      z3::goal g(solver.ctx());
      g.add(res);
      z3::tactic t = z3::tactic(solver.ctx(), "qe") & 
                      z3::tactic(solver.ctx(), "ctx-solver-simplify") &
                      z3::tactic(solver.ctx(), "propagate-ineqs") &
                      z3::tactic(solver.ctx(), "ctx-solver-simplify") &
                      z3::tactic(solver.ctx(), "propagate-ineqs");
      
      z3::apply_result r = t(g);
      z3::expr_vector fin(solver.ctx());
      for (unsigned int i = 0; i < r.size(); i++) {
        fin.push_back(r[i].as_expr());
      }
      z3::expr region = fin.size() == 1? fin[0] : z3::mk_or(fin);
      return region.to_string();
    } catch (const z3::exception &e) {
      llvm::errs() << "could not compute invalid region: " << e.msg() << "\n";
      return std::nullopt;
    }
  }

  void visitRegion(Region &region, z3::solver &solver,
                   std::vector<z3::expr> &cost_var) {
    for (Block &block : region.getBlocks())
      visitBlock(block, solver, cost_var);
  }

  void visitBlock(Block &block, z3::solver &solver,
                  std::vector<z3::expr> &cost_var) {
    for (Operation &op : block.getOperations())
      visitOperation(&op, solver, cost_var);
  }
};
} // namespace
