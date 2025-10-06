#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

LogicalResult WhileLoopInfo::computeInfo() {
  auto &condBlk = op.getCond().front();
  if (condBlk.getOperations().size() != 2)
    return failure();
  auto condTerm = cast<stablehlo::ReturnOp>(condBlk.getTerminator());
  auto condV = condTerm->getOperand(0);
  auto cond = condV.getDefiningOp<stablehlo::CompareOp>();
  if (!cond)
    return failure();

  auto induct = dyn_cast<BlockArgument>(cond.getOperand(0));
  if (!induct)
    return failure();
  if (induct.getOwner() != &condBlk)
    return failure();

  if (cond.getComparisonDirection() != stablehlo::ComparisonDirection::LT)
    return failure();

  auto bodyTerm =
      cast<stablehlo::ReturnOp>(op.getBody().front().getTerminator());
  auto incV = bodyTerm->getOperand(induct.getArgNumber());
  auto inc = incV.getDefiningOp<stablehlo::AddOp>();
  if (!inc)
    return failure();

  auto loopBodyBlock = &op.getBody().front();

  auto incba0 = dyn_cast<BlockArgument>(inc.getOperand(0));
  auto incba1 = dyn_cast<BlockArgument>(inc.getOperand(1));

  bool found = false;

  if (incba0 && (incba0.getOwner() == loopBodyBlock) &&
      (incba0.getArgNumber() == induct.getArgNumber())) {
    step = inc.getOperand(1);
    found = true;
  }

  if (!found && incba1 && (incba1.getOwner() == loopBodyBlock) &&
      (incba1.getArgNumber() == induct.getArgNumber())) {
    step = inc.getOperand(0);
    found = true;
  }

  if (!found)
    return failure();

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
  int64_t start = getConstantStart().value(),
          limit = getConstantLimit().value(), step = getConstantStep().value();

  if (limit < start) // cmp is lt
    return 0;

  if (step < 0 && limit > start)
    return 0;

  return (limit - start) / step;
}

Value WhileLoopInfo::getNumIters(mlir::OpBuilder &builder) {
  auto opReg = op->getParentRegion();
  if (!opReg->isAncestor(limit.getParentRegion()) ||
      !opReg->isAncestor(step.getParentRegion())) {
    // Limit or Step are defined in the Condition/Block regions (respectively).
    return {};
  }

  Value numIters;
  if (isConstant()) {
    numIters = builder.create<stablehlo::ConstantOp>(
        op->getLoc(), start.getType(),
        cast<ElementsAttr>(makeAttr(start.getType(), getConstantNumIters())));
  } else {
    // numIters = (limit - start) / step;
    numIters = builder.create<stablehlo::DivOp>(
        op->getLoc(),
        builder.create<stablehlo::SubtractOp>(op->getLoc(), limit, start),
        step);
  }

  return numIters;
}

void WhileLoopInfo::propagateInductionVarOffsets() {
  auto inductionVar = op.getBody().front().getArgument(0);

  SmallVector<Value> worklist;
  DenseSet<Value> visited;

  worklist.push_back(inductionVar);

  auto inductionType = inductionVar.getType();
  unsigned bitWidth = 64;
  if (auto tensorType = dyn_cast<RankedTensorType>(inductionType)) {
    if (auto intType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      bitWidth = intType.getWidth();
    }
  }

  inductionVarOffsets[inductionVar] = APInt(bitWidth, 0, true);

  while (!worklist.empty()) {
    auto cur = worklist.pop_back_val();
    if (visited.contains(cur))
      continue;
    visited.insert(cur);

    APInt curOffset = inductionVarOffsets[cur];

    for (auto user : cur.getUsers()) {
      APInt newOffset(bitWidth, 0, true);
      Value result;

      if (auto addOp = dyn_cast<stablehlo::AddOp>(user)) {
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();
        APInt constVal(bitWidth, 0, true);

        if (matchPattern(lhs, m_ConstantInt(&constVal))) {
          newOffset = updateOffset(curOffset, constVal);
          result = addOp.getResult();
        } else if (matchPattern(rhs, m_ConstantInt(&constVal))) {
          newOffset = updateOffset(curOffset, constVal);
          result = addOp.getResult();
        }
      } else if (auto subOp = dyn_cast<stablehlo::SubtractOp>(user)) {
        Value lhs = subOp.getLhs();
        Value rhs = subOp.getRhs();
        APInt constVal(bitWidth, 0, true);

        if (matchPattern(rhs, m_ConstantInt(&constVal))) {
          newOffset = updateOffset(curOffset, -constVal);
          result = subOp.getResult();
        }
      } else if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(user)) {
        newOffset = curOffset;
        result = convertOp.getResult();
      }

      if (result && !inductionVarOffsets.contains(result)) {
        inductionVarOffsets[result] = newOffset;
        worklist.push_back(result);
      }
    }
  }
}
