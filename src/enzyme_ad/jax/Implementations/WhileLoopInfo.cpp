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
  if (isConstantAcrossIterations(v, outerValue) &&
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

bool WhileLoopInfo::isConstantAcrossIterations(Value v) {
  Value outerValue;
  return isConstantAcrossIterations(v, outerValue);
}

bool WhileLoopInfo::isConstantAcrossIterations(Value v, Value &outerValue) {
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    int64_t blockArgIndex = blockArg.getArgNumber();
    auto &body = op.getBody().front();
    auto terminator = body.getTerminator();
    if (terminator && terminator->getOperand(blockArgIndex) ==
                          body.getArgument(blockArgIndex)) {
      outerValue = op->getOperand(blockArgIndex);
      return true;
    }
  }

  if (definedOutside(v, op)) {
    outerValue = v;
    return true;
  }
  return false;
}

bool WhileLoopInfo::canHoistOperationFromLoop(
    mlir::stablehlo::DynamicSliceOp sliceOp, int64_t sliceIndex) {
  if (!isConstant())
    return false;

  auto depIndex = sliceOp.getStartIndices()[sliceIndex];
  if (!affineIndexInfo.contains(depIndex))
    return false;

  return true;
}

bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, mlir::stablehlo::DynamicSliceOp sliceOp,
    int64_t sliceIndex, Value &result) {
  // for non constant iteration count we will need a dynamic slice
  if (!isConstant())
    return false;

  auto totalIterCount = getConstantNumIters();

  auto depIndex = sliceOp.getStartIndices()[sliceIndex];
  auto indexTy = depIndex.getType();

  // First we emit a DS op that covers the entire sliced region. This
  // might be converted into a Slice Op if the starts are static.
  // Next we do a strided slice of this DS op. If starts are static,
  // these will get fused into a single slice op.
  if (!affineIndexInfo.contains(depIndex))
    return false;

  auto indexInfo = affineIndexInfo[depIndex];
  auto scale = indexInfo.scale.getSExtValue();
  auto offset = indexInfo.offset.getSExtValue();

  bool negativeScale = scale < 0;

  auto step = getConstantStep().value();
  auto lb = getConstantStart().value();
  auto ub = getConstantLimit().value();
  int64_t N = ub - lb;

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

  if (negativeScale) {
    result = stablehlo::ReverseOp::create(
        builder, sliceOp.getLoc(), result,
        builder.getDenseI64ArrayAttr({sliceIndex}));
  }

  return true;
}

bool WhileLoopInfo::canHoistOperationFromLoop(
    mlir::stablehlo::DynamicUpdateSliceOp dusOp, int64_t dusIndex) {
  auto depIndex = dusOp.getStartIndices()[dusIndex];
  if (!affineIndexInfo.contains(depIndex))
    return false;

  return true;
}

bool WhileLoopInfo::hoistOperationFromLoop(
    OpBuilder &builder, Value operand, Value update,
    mlir::stablehlo::DynamicUpdateSliceOp dusOp, int64_t dusIndex,
    Value &result) {
  auto depIndex = dusOp.getStartIndices()[dusIndex];
  auto indexTy = depIndex.getType();

  if (!affineIndexInfo.contains(depIndex))
    return false;

  auto indexInfo = affineIndexInfo[depIndex];
  auto scale = indexInfo.scale.getSExtValue();
  auto offset = indexInfo.offset.getSExtValue();

  auto dusType = cast<RankedTensorType>(dusOp.getResult().getType());
  SmallVector<Value> dusStartIndices(dusOp.getStartIndices().begin(),
                                     dusOp.getStartIndices().end());

  if (scale == 1) { // avoid scatter if possible
    auto constOffset = stablehlo::ConstantOp::create(
        builder, dusOp.getLoc(), indexTy,
        cast<ElementsAttr>(makeAttr(indexTy, offset)));
    Value startVal = start;
    if (start.getType() != indexTy) {
      startVal =
          stablehlo::ConvertOp::create(builder, dusOp.getLoc(), indexTy, start);
    }
    dusStartIndices[dusIndex] = stablehlo::AddOp::create(
        builder, dusOp.getLoc(), startVal, constOffset);

    result = stablehlo::DynamicUpdateSliceOp::create(
                 builder, dusOp.getLoc(), operand, update, dusStartIndices)
                 .getResult();
  } else { // emit a scatter
    auto updateTy = cast<RankedTensorType>(update.getType());
    auto indexElemTy = cast<ShapedType>(indexTy).getElementType();

    auto bcastIndex = [&](Value val) {
      if (cast<RankedTensorType>(val.getType()).getElementType() !=
          indexElemTy) {
        val = stablehlo::ConvertOp::create(
            builder, dusOp.getLoc(), RankedTensorType::get({}, indexElemTy),
            val);
      }

      return stablehlo::BroadcastInDimOp::create(
          builder, dusOp.getLoc(),
          RankedTensorType::get({updateTy.getDimSize(dusIndex), 1},
                                indexElemTy),
          val, builder.getDenseI64ArrayAttr({}));
    };

    SmallVector<Value> leftScatterIndices, rightScatterIndices;
    for (size_t i = 0; i < dusIndex; i++) {
      leftScatterIndices.push_back(bcastIndex(dusStartIndices[i]));
    }
    for (size_t i = dusIndex + 1; i < dusOp.getStartIndices().size(); i++) {
      rightScatterIndices.push_back(bcastIndex(dusStartIndices[i]));
    }

    auto depIndicesTy =
        RankedTensorType::get({updateTy.getDimSize(dusIndex), 1}, indexElemTy);
    auto depIndices =
        stablehlo::IotaOp::create(builder, dusOp.getLoc(), depIndicesTy, 0)
            .getResult();

    // indices: scale * (idx) + offset -> scale * (lb + i * step) + offset
    // shift by the lower bound
    Value startVal = getStart();
    if (cast<RankedTensorType>(startVal.getType()).getElementType() !=
        indexElemTy) {
      startVal = stablehlo::ConvertOp::create(
          builder, dusOp.getLoc(), RankedTensorType::get({}, indexElemTy),
          startVal);
    }

    if (!isStepOne()) {
      Value stepVal = getStep(builder);
      if (cast<RankedTensorType>(stepVal.getType()).getElementType() !=
          indexElemTy) {
        stepVal = stablehlo::ConvertOp::create(
            builder, dusOp.getLoc(), RankedTensorType::get({}, indexElemTy),
            stepVal);
      }
      depIndices = stablehlo::MulOp::create(builder, dusOp.getLoc(), depIndices,
                                            stepVal);
    }

    depIndices = stablehlo::AddOp::create(
        builder, dusOp.getLoc(), depIndices,
        stablehlo::BroadcastInDimOp::create(builder, dusOp.getLoc(),
                                            depIndicesTy, startVal,
                                            builder.getDenseI64ArrayAttr({})));

    // apply affine map
    depIndices = stablehlo::MulOp::create(
        builder, dusOp.getLoc(), depIndices,
        stablehlo::ConstantOp::create(
            builder, dusOp.getLoc(), depIndicesTy,
            cast<ElementsAttr>(makeAttr(depIndicesTy, scale))));
    depIndices = stablehlo::AddOp::create(
        builder, dusOp.getLoc(), depIndices,
        stablehlo::ConstantOp::create(
            builder, dusOp.getLoc(), depIndicesTy,
            cast<ElementsAttr>(makeAttr(depIndicesTy, offset))));

    SmallVector<Value> concatInputs;
    concatInputs.insert(concatInputs.end(), leftScatterIndices.begin(),
                        leftScatterIndices.end());
    concatInputs.push_back(depIndices);
    concatInputs.insert(concatInputs.end(), rightScatterIndices.begin(),
                        rightScatterIndices.end());

    Value scatterIndices = stablehlo::ConcatenateOp::create(
        builder, dusOp.getLoc(), concatInputs, 1);

    SmallVector<int64_t> updateWindowDims(updateTy.getRank() - 1);
    std::iota(updateWindowDims.begin(), updateWindowDims.begin() + dusIndex, 0);
    std::iota(updateWindowDims.begin() + dusIndex, updateWindowDims.end(),
              dusIndex + 1);

    SmallVector<int64_t> scatterDimsToOperandDims(updateTy.getRank());
    std::iota(scatterDimsToOperandDims.begin(), scatterDimsToOperandDims.end(),
              0);

    auto scatterOp = stablehlo::ScatterOp::create(
        builder, dusOp.getLoc(), ValueRange{operand}, scatterIndices,
        ValueRange{update},
        stablehlo::ScatterDimensionNumbersAttr::get(
            dusOp.getContext(),
            /*updateWindowDims*/ updateWindowDims,
            /*insertedWindowDims*/ {dusIndex},
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
  }
  return true;
}
