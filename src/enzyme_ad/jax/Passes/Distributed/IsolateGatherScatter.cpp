#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/MainFunctionAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ISOLATEGATHERSCATTERPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// A reshard point on `value` whose rule gives every dimension distinct
// operand and result factors, so each dimension is broken. It carries
// `value`'s Shardy sharding, if any, so an sdy.reshard Shardy placed next to
// the op still reads one from it.
Value createBreakingReshardPoint(OpBuilder &builder, Location loc,
                                 Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  MLIRContext *context = builder.getContext();
  int64_t rank = type.getRank();
  SmallVector<int64_t> factorSizes(type.getShape());
  llvm::append_range(factorSizes, type.getShape());
  SmallVector<sdy::DimMappingAttr> operandDims, resultDims;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    operandDims.push_back(sdy::DimMappingAttr::get(context, {dimension}));
    resultDims.push_back(sdy::DimMappingAttr::get(context, {rank + dimension}));
  }
  auto rule = sdy::OpShardingRuleAttr::get(
      context, factorSizes, {sdy::TensorMappingAttr::get(context, operandDims)},
      {sdy::TensorMappingAttr::get(context, resultDims)},
      /*reduction_factors=*/{}, /*need_replication_factors=*/{},
      /*permutation_factors=*/{}, /*blocked_propagation_factors=*/{},
      /*is_custom_rule=*/true);
  auto point = stablehlo::CustomCallOp::create(
      builder, loc, TypeRange{type}, ValueRange{value},
      SmallVector<NamedAttribute>{
          builder.getNamedAttr("call_target_name",
                               builder.getStringAttr(kReshardPointTarget)),
          builder.getNamedAttr(sdy::kShardingRuleAttr, rule)});
  if (sdy::TensorShardingAttr sharding = sdy::getSharding(value))
    sdy::setSharding(point.getResult(0), sharding);
  return point.getResult(0);
}

// Whether a reshard point on `value` can break anything: a scalar has no
// dimension to break.
bool isBreakable(Value value) {
  return cast<RankedTensorType>(value.getType()).getRank() > 0;
}

void isolate(Operation *op) {
  OpBuilder builder(op);
  for (OpOperand &operand : op->getOpOperands()) {
    // An operand that is already a reshard point (one isolated op feeding
    // another) shares that break.
    Operation *producer = operand.get().getDefiningOp();
    if (!isBreakable(operand.get()) || (producer && isReshardPoint(producer)))
      continue;
    operand.set(
        createBreakingReshardPoint(builder, op->getLoc(), operand.get()));
  }
  builder.setInsertionPointAfter(op);
  for (Value result : op->getResults()) {
    if (!isBreakable(result))
      continue;
    Value point = createBreakingReshardPoint(builder, op->getLoc(), result);
    result.replaceAllUsesExcept(point, point.getDefiningOp());
  }
}

struct IsolateGatherScatterPass
    : public impl::IsolateGatherScatterPassBase<IsolateGatherScatterPass> {
  using IsolateGatherScatterPassBase::IsolateGatherScatterPassBase;

  void runOnOperation() override {
    // ConvertMainToDistributedFunction reports a missing or ambiguous main.
    const auto &mainFunction = getAnalysis<FindMainFunctionAnalysis>();
    if (!mainFunction.isValid())
      return;
    func::FuncOp main = mainFunction.getMainFuncOp();
    assert(main && "main is a func.func before conversion to distributed");
    SmallVector<Operation *> toIsolate;
    for (Operation &op : main.getBody().getOps())
      if (runsUnsharded(&op))
        toIsolate.push_back(&op);
    for (Operation *op : toIsolate)
      isolate(op);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
