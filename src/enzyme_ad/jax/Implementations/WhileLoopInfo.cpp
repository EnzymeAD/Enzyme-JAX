#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif
#include "stablehlo/dialect/StablehloOps.h"
#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

static bool definedOutside(Value v, Operation *op) {
  return !op->isAncestor(v.getParentBlock()->getParentOp());
}

void WhileLoopInfo::computeConstantValues() {
  constStep = getConstantStepCalculate();
  constLimit = getConstantLimitCalculate();
  constStart = getConstantStartCalculate();
}

Value WhileLoopInfo::getStep(OpBuilder &builder) {
  if (step)
    return step;

  auto Ty =
      RankedTensorType::get({}, builder.getIntegerType(stepInt.getBitWidth()));
  return stablehlo::ConstantOp::create(
      builder, op->getLoc(), Ty,
      cast<ElementsAttr>(makeAttr(Ty, stepInt.getSExtValue())));
}

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

  start = op->getOperand(induct.getArgNumber());
  limit = cond.getOperand(1);

  propagateAffineIndexInfo();
  auto affineIndexInfoMap = getAffineIndexInfo();

  auto bodyTerm =
      cast<stablehlo::ReturnOp>(op.getBody().front().getTerminator());
  auto incV = bodyTerm->getOperand(induct.getArgNumber());

  // if part of the index map then exit early
  if (affineIndexInfoMap.contains(incV)) {
    auto indexInfo = affineIndexInfoMap[incV];
    if (indexInfo.scale.isOne()) {
      stepInt = indexInfo.offset;
      foundStep = true;
      computeConstantValues();
      return success();
    } else {
      return failure();
    }
  }

  // simpler check
  auto inc = incV.getDefiningOp<stablehlo::AddOp>();
  if (!inc)
    return failure();

  auto loopBodyBlock = &op.getBody().front();

  auto incba0 = dyn_cast<BlockArgument>(inc.getOperand(0));
  auto incba1 = dyn_cast<BlockArgument>(inc.getOperand(1));

  foundStep = false;

  if (incba0 && (incba0.getOwner() == loopBodyBlock) &&
      (incba0.getArgNumber() == induct.getArgNumber())) {
    step = inc.getOperand(1);
    foundStep = true;
  }

  if (!foundStep && incba1 && (incba1.getOwner() == loopBodyBlock) &&
      (incba1.getArgNumber() == induct.getArgNumber())) {
    step = inc.getOperand(0);
    foundStep = true;
  }

  if (!foundStep)
    return failure();

  computeConstantValues();
  return success();
}

int64_t getBitWidth(Value v) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  return tensorType.getElementType().getIntOrFloatBitWidth();
}

std::optional<int64_t> WhileLoopInfo::getConstantStepCalculate() {
  if (!step)
    return stepInt.getSExtValue();

  llvm::APInt constVal(getBitWidth(step), 0, true);
  if (isConstantValue(step, constVal))
    return constVal.getSExtValue();
  return std::nullopt;
}

std::optional<int64_t> WhileLoopInfo::getConstantStartCalculate() {
  llvm::APInt constVal(getBitWidth(start), 0, true);
  if (isConstantValue(start, constVal))
    return constVal.getSExtValue();
  return std::nullopt;
}

std::optional<int64_t> WhileLoopInfo::getConstantLimitCalculate() {
  llvm::APInt constVal(getBitWidth(limit), 0, true);
  if (isConstantValue(limit, constVal))
    return constVal.getSExtValue();
  return std::nullopt;
}

int64_t WhileLoopInfo::getConstantNumIters() {
  int64_t start = getConstantStart().value(),
          limit = getConstantLimit().value(), step = getConstantStep().value();

  if (limit < start) // cmp is lt
    return 0;

  if (step < 0 && limit > start)
    return 0;

  return (limit - start + step - 1) / step; // ceil division
}

Value WhileLoopInfo::getNumIters(mlir::OpBuilder &builder) {
  auto opReg = op->getParentRegion();
  if (!opReg->isAncestor(limit.getParentRegion()) ||
      (step && !opReg->isAncestor(step.getParentRegion()))) {
    // Limit or Step are defined in the Condition/Block regions (respectively).
    return {};
  }

  Value numIters;
  if (isConstant()) {
    numIters = stablehlo::ConstantOp::create(
        builder, op->getLoc(), start.getType(),
        cast<ElementsAttr>(makeAttr(start.getType(), getConstantNumIters())));
  } else {
    // numIters = (limit - start) / step;
    Value stepVal;
    if (step) {
      stepVal = step;
    } else {
      stepVal = stablehlo::ConstantOp::create(
          builder, op->getLoc(), start.getType(),
          cast<ElementsAttr>(
              makeAttr(start.getType(), stepInt.getSExtValue())));
    }
    numIters = stablehlo::DivOp::create(
        builder, op->getLoc(),
        stablehlo::SubtractOp::create(builder, op->getLoc(), limit, start),
        stepVal);
  }

  return numIters;
}

WhileLoopInfo::AffineIndexInfo
WhileLoopInfo::updateAffineIndexInfo(AffineIndexInfo curInfo, llvm::APInt scale,
                                     llvm::APInt offset) {
  if (curInfo.offset.getBitWidth() != offset.getBitWidth()) {
    offset = offset.sextOrTrunc(curInfo.offset.getBitWidth());
  }
  if (curInfo.scale.getBitWidth() != scale.getBitWidth()) {
    scale = scale.sextOrTrunc(curInfo.scale.getBitWidth());
  }
  return AffineIndexInfo{scale * curInfo.scale,
                         scale * curInfo.offset + offset};
}

bool WhileLoopInfo::isStepOne() {
  if (step) {
    return matchPattern(step, m_One());
  }

  return stepInt.isOne();
}

bool WhileLoopInfo::isConstantValue(Value v, llvm::APInt &constVal) {
  if (matchPattern(v, m_ConstantInt(&constVal)))
    return true;

  Value outerValue;
  SmallVector<Operation *> canBeHoisted;
  if (isConstantAcrossIterations(v, outerValue, canBeHoisted, false) &&
      matchPattern(outerValue, m_ConstantInt(&constVal)))
    return true;
  return false;
}

void WhileLoopInfo::propagateAffineIndexInfo() {
  auto inductionVar = getInductionVariable();

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

  llvm::APInt baseScaling(bitWidth, 1, true);
  llvm::APInt baseOffset(bitWidth, 0, true);
  affineIndexInfo[inductionVar] = AffineIndexInfo{baseScaling, baseOffset};

  while (!worklist.empty()) {
    auto cur = worklist.pop_back_val();
    if (visited.contains(cur))
      continue;
    visited.insert(cur);

    AffineIndexInfo curInfo = affineIndexInfo[cur];

    for (auto user : cur.getUsers()) {
      AffineIndexInfo newInfo;
      Value result;

      APInt constVal(bitWidth, 0, true);
      if (auto addOp = dyn_cast<stablehlo::AddOp>(user)) {
        if (isConstantValue(addOp.getLhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, baseScaling, constVal);
          result = addOp.getResult();
        } else if (isConstantValue(addOp.getRhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, baseScaling, constVal);
          result = addOp.getResult();
        }
      } else if (auto subOp = dyn_cast<stablehlo::SubtractOp>(user)) {
        if (isConstantValue(subOp.getRhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, baseScaling, -constVal);
          result = subOp.getResult();
        } else if (isConstantValue(subOp.getLhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, -baseScaling, constVal);
          result = subOp.getResult();
        }
      } else if (auto convertOp = dyn_cast<stablehlo::ConvertOp>(user)) {
        newInfo = curInfo;
        result = convertOp.getResult();
      } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(user)) {
        if (isConstantValue(mulOp.getRhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, constVal, baseOffset);
          result = mulOp.getResult();
        } else if (isConstantValue(mulOp.getLhs(), constVal)) {
          newInfo = updateAffineIndexInfo(curInfo, constVal, baseOffset);
          result = mulOp.getResult();
        }
      } else if (auto negOp = dyn_cast<stablehlo::NegOp>(user)) {
        newInfo = updateAffineIndexInfo(curInfo, -baseScaling, baseOffset);
        result = negOp.getResult();
      }

      if (result && !affineIndexInfo.contains(result)) {
        affineIndexInfo[result] = newInfo;
        worklist.push_back(result);
      }
    }
  }
}

bool WhileLoopInfo::isConstantAcrossIterations(Value v, bool checkOperands) {
  Value outerValue;
  SmallVector<Operation *> canBeHoisted;
  return isConstantAcrossIterations(v, outerValue, canBeHoisted, checkOperands);
}

bool WhileLoopInfo::isConstantAcrossIterations(
    Value v, Value &outerValue, SmallVector<Operation *> &canBeHoisted,
    bool checkOperands) {
  if (definedOutside(v, op)) {
    outerValue = v;
    return true;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    int64_t blockArgIndex = blockArg.getArgNumber();
    auto &body = op.getBody().front();
    auto terminator = body.getTerminator();
    if (terminator && blockArgIndex < terminator->getNumOperands() &&
        terminator->getOperand(blockArgIndex) ==
            body.getArgument(blockArgIndex)) {
      outerValue = op->getOperand(blockArgIndex);
      return true;
    }
  }

  if (!checkOperands)
    return false;

  auto defOp = v.getDefiningOp();
  if (!defOp)
    return false;

  // all operands of the defining op are constant across iterations
  // don't populate the outerValue in this case
  if (llvm::all_of(defOp->getOperands(), [&](Value operand) {
        return isConstantAcrossIterations(operand, outerValue, canBeHoisted,
                                          true);
      })) {
    outerValue = nullptr;
    canBeHoisted.push_back(defOp);
    return true;
  }
  return false;
}

template <typename OpTy>
void constructScatterGatherIndices(OpTy op, Value &result, OpBuilder &builder,
                                   SmallVectorImpl<int64_t> &dimensions,
                                   WhileLoopInfo &whileLoopInfo) {
  auto startIndices = op.getStartIndices();
  auto indexTy = startIndices[0].getType();
  auto indexElemTy = cast<ShapedType>(indexTy).getElementType();
  Location loc = op.getLoc();
  auto affineIndexInfo = whileLoopInfo.getAffineIndexInfo();

  SmallVector<int64_t> indexVectorShape = {whileLoopInfo.getConstantNumIters(),
                                           1};
  auto depIndicesTy = RankedTensorType::get(indexVectorShape, indexElemTy);

  auto bcastIndex = [&](Value val) {
    if (cast<RankedTensorType>(val.getType()).getElementType() != indexElemTy) {
      val = stablehlo::ConvertOp::create(
          builder, loc, RankedTensorType::get({}, indexElemTy), val);
    }

    return stablehlo::BroadcastInDimOp::create(
        builder, loc, depIndicesTy, val, builder.getDenseI64ArrayAttr({}));
  };

  Value startVal = bcastIndex(whileLoopInfo.getStart());

  SmallVector<Value> concatInputs;
  for (size_t i = 0; i < startIndices.size(); i++) {
    if (!llvm::is_contained(dimensions, i)) {
      concatInputs.push_back(bcastIndex(startIndices[i]));
      continue;
    }

    auto depIndex = startIndices[i];
    auto indexInfo = affineIndexInfo[depIndex];
    auto scale = indexInfo.scale.getSExtValue();
    auto offset = indexInfo.offset.getSExtValue();

    Value depIndices = stablehlo::IotaOp::create(builder, loc, depIndicesTy, 0);

    // step * iota
    if (!whileLoopInfo.isStepOne()) {
      depIndices = stablehlo::MulOp::create(
          builder, loc, depIndices, bcastIndex(whileLoopInfo.getStep(builder)));
    }

    // step * iota + offset
    depIndices = stablehlo::AddOp::create(builder, loc, depIndices, startVal);

    // apply affine map
    auto scaleConst = stablehlo::ConstantOp::create(
        builder, loc, depIndicesTy,
        cast<ElementsAttr>(makeAttr(depIndicesTy, scale)));
    auto offsetConst = stablehlo::ConstantOp::create(
        builder, loc, depIndicesTy,
        cast<ElementsAttr>(makeAttr(depIndicesTy, offset)));

    depIndices = stablehlo::MulOp::create(builder, loc, depIndices, scaleConst);
    depIndices =
        stablehlo::AddOp::create(builder, loc, depIndices, offsetConst);

    concatInputs.push_back(depIndices);
  }

  result = stablehlo::ConcatenateOp::create(builder, loc, concatInputs, 1);
  return;
}

bool WhileLoopInfo::canHoistOperationFromLoop(
    mlir::stablehlo::DynamicSliceOp sliceOp,
    SmallVectorImpl<int64_t> &dimensions) {
  if (!isConstant())
    return false;

  for (auto dim : dimensions) {
    if (!affineIndexInfo.contains(sliceOp.getStartIndices()[dim]))
      return false;
  }

  return true;
}

bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, mlir::stablehlo::DynamicSliceOp sliceOp,
    int64_t sliceIndex, Value &result) {
  SmallVector<int64_t> dimensions = {sliceIndex};
  if (!canHoistOperationFromLoop(sliceOp, dimensions))
    return false;

  auto depIndex = sliceOp.getStartIndices()[sliceIndex];
  auto indexTy = depIndex.getType();

  // First we emit a DS op that covers the entire sliced region. This
  // might be converted into a Slice Op if the starts are static.
  // Next we do a strided slice of this DS op. If starts are static,
  // these will get fused into a single slice op.
  auto indexInfo = affineIndexInfo[depIndex];
  auto scale = indexInfo.scale.getSExtValue();
  auto offset = indexInfo.offset.getSExtValue();

  auto step = getConstantStep().value();
  auto lb = getConstantStart().value();
  auto ub = getConstantLimit().value();

  auto rawMin = scale * lb + offset;
  auto rawMax = scale * (ub - 1) + offset;

  // flip if negative scale
  auto actualMin = std::min(rawMin, rawMax);
  auto actualMax = std::max(rawMin, rawMax);
  auto actualSize = actualMax - actualMin + 1;

  SmallVector<Value> dSliceStarts(sliceOp.getStartIndices().begin(),
                                  sliceOp.getStartIndices().end());
  SmallVector<int64_t> dSliceSizes(sliceOp.getSliceSizes().begin(),
                                   sliceOp.getSliceSizes().end());

  // i in [lb, ub)
  // idx_min = scale * lb + offset
  // idx_max = scale * (ub - 1) + offset
  // size = idx_max - idx_min + 1 = (N - 1) * scale + 1 where N = ub - lb
  auto idxMinConst = stablehlo::ConstantOp::create(
      builder, sliceOp.getLoc(), indexTy,
      cast<ElementsAttr>(makeAttr(indexTy, actualMin)));
  dSliceStarts[sliceIndex] = idxMinConst;
  dSliceSizes[sliceIndex] = actualSize;

  auto dSlice = stablehlo::DynamicSliceOp::create(
      builder, sliceOp.getLoc(), operand, dSliceStarts,
      builder.getDenseI64ArrayAttr(dSliceSizes));
  auto dType = dSlice.getResult().getType();

  // j(i) = (scale * i + offset) - idx_min = scale * (i - lb)
  SmallVector<int64_t> sliceStarts(dSliceStarts.size(), 0);
  SmallVector<int64_t> sliceStrides(dSliceSizes.size(), 1);
  sliceStrides[sliceIndex] = std::abs(scale) * step;
  SmallVector<int64_t> sliceLimits(dType.getShape().begin(),
                                   dType.getShape().end());

  auto slice =
      stablehlo::SliceOp::create(builder, sliceOp.getLoc(), dSlice,
                                 builder.getDenseI64ArrayAttr(sliceStarts),
                                 builder.getDenseI64ArrayAttr(sliceLimits),
                                 builder.getDenseI64ArrayAttr(sliceStrides));
  result = slice.getResult();

  if (scale < 0) {
    result = stablehlo::ReverseOp::create(
        builder, sliceOp.getLoc(), result,
        builder.getDenseI64ArrayAttr({sliceIndex}));
  }

  return true;
}

bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, mlir::stablehlo::DynamicSliceOp sliceOp,
    SmallVectorImpl<int64_t> &dimensions, Value &result) {
  if (!canHoistOperationFromLoop(sliceOp, dimensions))
    return false;

  if (dimensions.size() == 1) {
    return hoistOperationFromLoop(builder, operand, sliceOp, dimensions[0],
                                  result);
  }

  Value gatherIndices;
  constructScatterGatherIndices(sliceOp, gatherIndices, builder, dimensions,
                                *this);

  auto sliceSizes = sliceOp.getSliceSizes();

  SmallVector<int64_t> gatherSliceSizes;
  for (size_t i = 0; i < sliceSizes.size(); i++) {
    if (!llvm::is_contained(dimensions, i)) {
      gatherSliceSizes.push_back(sliceSizes[i]);
      continue;
    }
    gatherSliceSizes.push_back(1);
  }

  SmallVector<int64_t> offsetDims;
  for (size_t i = 0; i < sliceSizes.size(); i++) {
    if (!llvm::is_contained(dimensions, i))
      offsetDims.push_back(i);
  }

  SmallVector<int64_t> startIndexMap(sliceSizes.size());
  std::iota(startIndexMap.begin(), startIndexMap.end(), 0);

  auto gatherOp = stablehlo::GatherOp::create(
      builder, sliceOp.getLoc(), operand, gatherIndices,
      stablehlo::GatherDimensionNumbersAttr::get(
          sliceOp.getContext(),
          /*offsetDims*/ offsetDims,
          /*collapsedSliceDims*/ dimensions,
          /*operandBatchingDims*/ {},
          /*startIndicesBatchingDims*/ {},
          /*startIndexMap*/ startIndexMap,
          /*indexVectorDim*/ 1),
      gatherSliceSizes);
  result = gatherOp.getResult();

  return true;
}

bool WhileLoopInfo::canHoistOperationFromLoop(
    mlir::stablehlo::DynamicUpdateSliceOp dusOp,
    SmallVectorImpl<int64_t> &dimensions) {
  for (auto dim : dimensions) {
    if (!affineIndexInfo.contains(dusOp.getStartIndices()[dim]))
      return false;
  }

  return true;
}

bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, Value update,
    mlir::stablehlo::DynamicUpdateSliceOp dusOp, int64_t dusIndex,
    Value &result) {
  SmallVector<int64_t> dimensions = {dusIndex};
  if (!canHoistOperationFromLoop(dusOp, dimensions) || !isStepOne())
    return false;

  SmallVector<Value> dusStartIndices(dusOp.getStartIndices().begin(),
                                     dusOp.getStartIndices().end());

  auto depIndex = dusOp.getStartIndices()[dusIndex];
  auto indexTy = depIndex.getType();
  auto indexInfo = affineIndexInfo[depIndex];

  if (!indexInfo.scale.isOne())
    return false;

  // move the update dim = 0 to dusIndex
  SmallVector<int64_t> perm(dusOp.getStartIndices().size() - 1);
  std::iota(perm.begin(), perm.end(), 1);
  perm.insert(perm.begin() + dusIndex, 0);

  update = stablehlo::TransposeOp::create(builder, dusOp.getLoc(), update,
                                          builder.getDenseI64ArrayAttr(perm));

  auto constOffset = stablehlo::ConstantOp::create(
      builder, dusOp.getLoc(), indexTy,
      cast<ElementsAttr>(makeAttr(indexTy, indexInfo.offset.getSExtValue())));

  Value startVal = start;
  if (start.getType() != indexTy) {
    startVal =
        stablehlo::ConvertOp::create(builder, dusOp.getLoc(), indexTy, start);
  }

  dusStartIndices[dusIndex] =
      stablehlo::AddOp::create(builder, dusOp.getLoc(), startVal, constOffset);

  result = stablehlo::DynamicUpdateSliceOp::create(
      builder, dusOp.getLoc(), operand, update, dusStartIndices);
  return true;
}

// dimension 0 of the update is the batch dimension
bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, Value update,
    mlir::stablehlo::DynamicUpdateSliceOp dusOp,
    SmallVectorImpl<int64_t> &dimensions, Value &result) {
  if (!canHoistOperationFromLoop(dusOp, dimensions))
    return false;

  if (dimensions.size() == 1 &&
      hoistOperationFromLoop(builder, operand, update, dusOp, dimensions[0],
                             result)) {
    return true;
  }

  Value scatterIndices;
  constructScatterGatherIndices(dusOp, scatterIndices, builder, dimensions,
                                *this);

  auto updateTy = cast<RankedTensorType>(update.getType());
  auto operandTy = cast<RankedTensorType>(operand.getType());

  SmallVector<int64_t> updateWindowDims(updateTy.getRank() - 1);
  std::iota(updateWindowDims.begin(), updateWindowDims.end(), 1);

  SmallVector<int64_t> scatterDimsToOperandDims(operandTy.getRank());
  std::iota(scatterDimsToOperandDims.begin(), scatterDimsToOperandDims.end(),
            0);

  auto scatterOp = stablehlo::ScatterOp::create(
      builder, dusOp.getLoc(), ValueRange{operand}, scatterIndices,
      ValueRange{update},
      stablehlo::ScatterDimensionNumbersAttr::get(
          dusOp.getContext(),
          /*updateWindowDims*/ updateWindowDims,
          /*insertedWindowDims*/ dimensions,
          /*inputBatchingDims*/ {},
          /*scatterIndicesBatchingDims*/ {},
          /*scatterDimsToOperandDims*/ scatterDimsToOperandDims,
          /*indexVectorDim*/ 1),
      /*indicesAreSorted*/ false, /*uniqueIndices*/ true);
  result = scatterOp.getResult(0);

  Block *updateBody = new Block();
  scatterOp.getUpdateComputation().push_back(updateBody);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(updateBody);
    auto unrankedTy = RankedTensorType::get({}, updateTy.getElementType());
    updateBody->addArgument(unrankedTy, dusOp.getLoc());
    Value updateInBody = updateBody->addArgument(unrankedTy, dusOp.getLoc());
    stablehlo::ReturnOp::create(builder, dusOp.getLoc(), updateInBody);
  }
  return true;
}
