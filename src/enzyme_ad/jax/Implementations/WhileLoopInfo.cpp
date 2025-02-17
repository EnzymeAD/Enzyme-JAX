#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

LogicalResult WhileLoopInfo::computeInfo() {
  auto &condBlk = op.getCond().front();
  if (condBlk.getOperations().size() != 2)
    return failure();
  auto condTerm = cast<stablehlo::ReturnOp>(&condBlk.back());
  auto condV = condTerm->getOperand(0);
  auto cond = condV.getDefiningOp<stablehlo::CompareOp>();
  if (!cond)
    return failure();

  auto induct = cond.getOperand(0).dyn_cast<BlockArgument>();
  if (!induct)
    return failure();
  if (induct.getOwner() != &condBlk)
    return failure();

  if (cond.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
    return failure();

  auto bodyTerm = cast<stablehlo::ReturnOp>(&op.getBody().front().back());
  auto incV = bodyTerm->getOperand(induct.getArgNumber());
  auto inc = incV.getDefiningOp<stablehlo::AddOp>();
  if (!inc)
    return failure();

  auto loopBodyBlock = &op.getBody().front();

  auto incba = inc.getOperand(0).dyn_cast<BlockArgument>();

  if (!incba)
    return failure();

  if (incba.getOwner() != loopBodyBlock)
    return failure();

  if (incba.getArgNumber() != induct.getArgNumber())
    return failure();

  step = inc.getOperand(1);
  start = op->getOperand(induct.getArgNumber());
  limit = cond.getOperand(1);

  return success();
}

std::optional<int64_t> WhileLoopInfo::getConstantStep() {
  DenseIntElementsAttr stepAttr;
  if (!matchPattern(step, m_Constant(&stepAttr)))
    return std::nullopt;
  return (*stepAttr.begin()).getSExtValue();
}

std::optional<int64_t> WhileLoopInfo::getConstantStart() {
  DenseIntElementsAttr startAttr;
  if (!matchPattern(start, m_Constant(&startAttr)))
    return std::nullopt;
  return (*startAttr.begin()).getSExtValue();
}

std::optional<int64_t> WhileLoopInfo::getConstantLimit() {
  DenseIntElementsAttr limitAttr;
  if (!matchPattern(limit, m_Constant(&limitAttr)))
    return std::nullopt;
  return (*limitAttr.begin()).getSExtValue();
}

int64_t WhileLoopInfo::getConstantNumIters() {
  return (getConstantLimit().value() - getConstantStart().value()) /
         getConstantStep().value();
}

Value WhileLoopInfo::getNumIters(mlir::OpBuilder &builder) {
  auto opReg = op->getParentRegion();
  if (!opReg->isAncestor(limit.getParentRegion()) ||
      !opReg->isAncestor(step.getParentRegion())) {
    // Limit or Step are defined in the Condition/Block regions (respectively).
    return {};
  }

  // numIters = (limit - start) / step;
  Value numIters = builder.create<stablehlo::DivOp>(
      op->getLoc(),
      builder.create<stablehlo::SubtractOp>(op->getLoc(), limit, start), step);

  return numIters;
}
