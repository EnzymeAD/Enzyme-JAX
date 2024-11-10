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

  auto direction = cond.getComparisonDirection();
  if (direction != stablehlo::ComparisonDirection::LT &&
      direction != stablehlo::ComparisonDirection::LE)
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
  inclusive = direction == stablehlo::ComparisonDirection::LE;

  return success();
}

std::optional<DenseIntOrFPElementsAttr> WhileLoopInfo::getConstantStep() {
  DenseIntOrFPElementsAttr stepAttr;
  if (!matchPattern(step, m_Constant(&stepAttr)))
    return std::nullopt;
  return stepAttr;
}

std::optional<DenseIntOrFPElementsAttr> WhileLoopInfo::getConstantStart() {
  DenseIntOrFPElementsAttr startAttr;
  if (!matchPattern(start, m_Constant(&startAttr)))
    return std::nullopt;
  return startAttr;
}

std::optional<DenseIntOrFPElementsAttr> WhileLoopInfo::getConstantLimit() {
  DenseIntOrFPElementsAttr limitAttr;
  if (!matchPattern(limit, m_Constant(&limitAttr)))
    return std::nullopt;
  return limitAttr;
}

std::optional<int64_t> WhileLoopInfo::getConstantNumIters() {
  DenseIntOrFPElementsAttr start = getConstantStart().value(),
                           limit = getConstantLimit().value(),
                           step = getConstantStep().value();

  if (isa<mlir::FloatType>(start.getElementType())) {
    auto start_f = *start.getValues<APFloat>().begin(),
         limit_f = *limit.getValues<APFloat>().begin(),
         step_f = *step.getValues<APFloat>().begin();

    auto numIters_f = (limit_f - start_f) / step_f;

    APSInt numIters_i(64, true);
    bool isExact = false;
    if (numIters_f.convertToInteger(numIters_i, llvm::APFloat::rmTowardZero,
                                    &isExact) != llvm::APFloat::opOK ||
        !isExact)
      return std::nullopt;

    return numIters_i.getSExtValue() + inclusive;
  }

  auto start_i = *start.getValues<APInt>().begin(),
       limit_i = *limit.getValues<APInt>().begin(),
       step_i = *step.getValues<APInt>().begin();
  return ((limit_i - start_i).sdiv(step_i)).getSExtValue() + inclusive;
}

Value WhileLoopInfo::getNumIters(mlir::OpBuilder &builder) {
  auto opReg = op->getParentRegion();
  if (!opReg->isAncestor(limit.getParentRegion()) ||
      !opReg->isAncestor(step.getParentRegion())) {
    // Limit or Step are defined in the Condition/Block regions (respectively).
    // TODO: move operations outside if constant
    return {};
  }

  auto Ty = builder.getI64Type();
  auto unrankedTensorType = RankedTensorType::get({}, Ty);

  // numIters = (int64_t)((limit - start) / step) + inclusive;
  Value numIters = builder.create<stablehlo::ConvertOp>(
      op->getLoc(), unrankedTensorType,
      builder.create<stablehlo::DivOp>(
          op->getLoc(),
          builder.create<stablehlo::SubtractOp>(op->getLoc(), limit, start),
          step));

  if (inclusive) {
    numIters = builder.create<stablehlo::AddOp>(
        op->getLoc(), numIters,
        builder.create<ConstantOp>(
            op->getLoc(), unrankedTensorType,
            SplatElementsAttr::get(
                unrankedTensorType,
                ArrayRef<Attribute>(IntegerAttr::get(Ty, 1)))));
  }

  return numIters;
}
