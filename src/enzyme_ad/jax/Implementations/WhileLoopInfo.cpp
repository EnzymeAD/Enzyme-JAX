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

#include "mlir/IR/Matchers.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "llvm/ADT/TypeSwitch.h"

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

  auto inductionType = inductionVar.getType();
  unsigned bitWidth = 64;
  if (auto tensorType = dyn_cast<RankedTensorType>(inductionType)) {
    if (auto intType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      bitWidth = intType.getWidth();
    }
  }

  APInt baseScaling(bitWidth, 1, true);
  APInt baseOffset(bitWidth, 0, true);
  SmallVector<Value> newPropagated;

  propagateAffineIndexInfo(
      inductionVar, AffineIndexInfo{baseScaling, baseOffset}, newPropagated);
  return;
}

void WhileLoopInfo::propagateAffineIndexInfo(
    Value v, AffineIndexInfo vInfo, SmallVectorImpl<Value> &newPropagated) {
  SmallVector<Value> worklist;
  worklist.push_back(v);
  affineIndexInfo[v] = vInfo;

  auto bitWidth = vInfo.scale.getBitWidth();
  APInt baseScaling(vInfo.scale.getBitWidth(), 1, true);
  APInt baseOffset(vInfo.offset.getBitWidth(), 0, true);

  while (!worklist.empty()) {
    auto cur = worklist.pop_back_val();
    if (affineIndexPropagationVisited.contains(cur)) {
      continue;
    }
    affineIndexPropagationVisited.insert(cur);

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
      } else if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(user)) {
        if (cast<ShapedType>(reshapeOp.getType()).getNumElements() == 1) {
          newInfo = updateAffineIndexInfo(curInfo, baseScaling, baseOffset);
          result = reshapeOp.getResult();
        }
      }

      if (result && !affineIndexInfo.contains(result)) {
        affineIndexInfo[result] = newInfo;
        newPropagated.push_back(result);
        worklist.push_back(result);
      }
    }
  }

  int64_t totalIterations = 0;
  bool anyNewPropagated;
  do {
    anyNewPropagated = false;
    // if any slice operand is an iota, then we can try to infer the offset
    // and scale
    op.getBody().front().walk([&](stablehlo::DynamicSliceOp sliceOp) {
      // Skip if we've already processed this slice's result
      if (affineIndexInfo.contains(sliceOp.getResult())) {
        return WalkResult::advance();
      }

      auto sliceOutTy = cast<RankedTensorType>(sliceOp.getType());
      if (sliceOutTy.getNumElements() != 1 ||
          !sliceOutTy.getElementType().isInteger() ||
          // skip over boolean types
          sliceOutTy.getElementType().isInteger(1)) {
        return WalkResult::advance();
      }

      int64_t sliceDim = -1;
      for (int64_t i = 0; i < sliceOp.getSliceSizes().size(); i++) {
        if (matchPattern(sliceOp.getStartIndices()[i], m_Zero())) {
          continue;
        }
        if (sliceDim != -1) {
          return WalkResult::advance(); // can't do anything here
        }
        sliceDim = i;
      }

      auto iotaDetection = detectIotaLikeTensor(sliceOp.getOperand());

      if (iotaDetection && sliceDim == iotaDetection.value().dimension) {
        anyNewPropagated = true;
        auto indexInfo = affineIndexInfo[sliceOp.getStartIndices()[sliceDim]];
        auto offset = indexInfo.offset.getSExtValue();
        auto iotaStart = iotaDetection.value().start;
        auto iotaScale = iotaDetection.value().scale;
        // The slice result is: iotaScale * (indexInfo.scale * i +
        //                       indexInfo.offset) + iotaStart
        //                    = (iotaScale * indexInfo.scale) * i + (iotaScale *
        //                    indexInfo.offset + iotaStart)
        auto newScale = indexInfo.scale * iotaScale;
        auto newOffset = iotaScale * offset + iotaStart;

        propagateAffineIndexInfo(
            sliceOp.getResult(),
            WhileLoopInfo::AffineIndexInfo{
                newScale,
                llvm::APInt(indexInfo.offset.getBitWidth(), newOffset, true)},
            newPropagated);
      }

      return WalkResult::advance();
    });
    totalIterations++;
  } while (anyNewPropagated && totalIterations < 4);
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

  // bail out if the operation is not isolated from above. we need to analyze
  // all the operations in the regions to ensure that this is constant across
  // iterations
  if (defOp->getNumRegions() != 0 &&
      !defOp->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
    return false;
  }

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

  auto dSlice = stablehlo::DynamicSliceOpCreate(
      builder, sliceOp.getLoc(), operand, dSliceStarts, dSliceSizes);
  auto dType = dyn_cast<RankedTensorType>(dSlice.getType());
  assert(dType);

  // j(i) = (scale * i + offset) - idx_min = scale * (i - lb)
  SmallVector<int64_t> sliceStarts(dSliceStarts.size(), 0);
  SmallVector<int64_t> sliceStrides(dSliceSizes.size(), 1);
  sliceStrides[sliceIndex] = std::abs(scale) * step;
  SmallVector<int64_t> sliceLimits(dType.getShape().begin(),
                                   dType.getShape().end());

  result = stablehlo::SliceOpCreate(builder, sliceOp.getLoc(), dSlice,
                                    sliceStarts, sliceLimits, sliceStrides);

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

void WhileLoopInfo::propagateBounds() {
  auto inductionVariable = getInductionVariable();

  auto inductionType = inductionVariable.getType();
  unsigned bitWidth = 64;
  if (auto tensorType = dyn_cast<RankedTensorType>(inductionType)) {
    if (auto intType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      bitWidth = intType.getWidth();
    }
  }
  this->boundsBitWidth = bitWidth;

  SmallVector<Value> newPropagated;

  auto start = getConstantStart().value();
  auto limit = getConstantLimit().value();
  auto step = getConstantStep().value();

  // Initialize bounds for the induction variable
  Bounds inductionBounds;
  if (step > 0) {
    inductionBounds =
        Bounds{APInt(bitWidth, start, true), APInt(bitWidth, limit - 1, true)};
  } else {
    inductionBounds =
        Bounds{APInt(bitWidth, limit + 1, true), APInt(bitWidth, start, true)};
  }

  propagateBounds(inductionVariable, inductionBounds, newPropagated);
  return;
}

void WhileLoopInfo::propagateBounds(Value v, Bounds curBounds,
                                    SmallVectorImpl<Value> &newPropagated) {
  SmallVector<Value> worklist;
  DenseSet<Operation *> visited;
  worklist.push_back(v);
  boundsMap[v] = curBounds;

  while (!worklist.empty()) {
    auto cur = worklist.pop_back_val();

    for (auto user : cur.getUsers()) {
      if (visited.contains(user)) {
        continue;
      }
      visited.insert(user);

      auto bounds = computeBounds(user);
      if (bounds.has_value()) {
        for (auto result : user->getResults()) {
          boundsMap[result] = bounds.value();
          newPropagated.push_back(result);
          worklist.push_back(result);
        }
      }
    }
  }
}

std::optional<WhileLoopInfo::Bounds> WhileLoopInfo::getBounds(Value value) {
  if (boundsMap.contains(value)) {
    return boundsMap.lookup(value);
  }

  // Try to read bounds from IR attribute using the utility function
  if (auto irBounds = enzyme::getBoundsFromIR(value, boundsBitWidth)) {
    return Bounds{irBounds->first, irBounds->second};
  }

  SplatElementsAttr splatAttr;
  if (matchPattern(value, m_Constant(&splatAttr))) {
    auto attr = splatAttr.getSplatValue<Attribute>();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      auto intVal = intAttr.getValue().sextOrTrunc(boundsBitWidth);
      return Bounds{intVal, intVal};
    }
  }
  return std::nullopt;
}

// Helper functions for APInt min/max since std::min/max don't work with APInt
static APInt smin(const APInt &a, const APInt &b) { return a.slt(b) ? a : b; }
static APInt smax(const APInt &a, const APInt &b) { return a.sgt(b) ? a : b; }

std::optional<WhileLoopInfo::Bounds>
WhileLoopInfo::computeBounds(Operation *op) {
  if (!op || !stablehlo::hasTraitElementwise(op)) {
    return std::nullopt;
  }

  if (op->getNumOperands() == 1) {
    if (isa<stablehlo::SineOp, stablehlo::CosineOp>(op)) {
      APInt one(boundsBitWidth, 1);
      return Bounds{-one, one};
    }

    auto optionalBounds = getBounds(op->getOperand(0));
    if (!optionalBounds.has_value()) {
      return std::nullopt;
    }
    auto bounds = optionalBounds.value();

    return TypeSwitch<Operation *, std::optional<Bounds>>(op)
        .Case<stablehlo::NegOp>([&](stablehlo::NegOp negOp) {
          return Bounds{-bounds.max, -bounds.min};
        })
        .Case<stablehlo::AbsOp>([&](stablehlo::AbsOp absOp) {
          APInt zero(boundsBitWidth, 0);
          if (bounds.min.sge(zero)) { // all positive
            return Bounds{bounds.min, bounds.max};
          } else if (bounds.max.sle(zero)) { // all negative
            return Bounds{-bounds.max, -bounds.min};
          } else { // mixed signs
            APInt newMax = bounds.max.abs().sgt(bounds.min.abs())
                               ? bounds.max.abs()
                               : (-bounds.min).abs();
            return Bounds{zero, newMax};
          }
        })
        .Case<stablehlo::ConvertOp>([&](stablehlo::ConvertOp convertOp)
                                        -> std::optional<Bounds> {
          auto outType = convertOp.getResult().getType();
          if (auto tensorType = dyn_cast<RankedTensorType>(outType)) {
            if (auto intType =
                    dyn_cast<IntegerType>(tensorType.getElementType())) {
              unsigned outBitWidth = intType.getWidth();
              APInt outMin =
                  APInt::getSignedMinValue(outBitWidth).sext(boundsBitWidth);
              APInt outMax =
                  APInt::getSignedMaxValue(outBitWidth).sext(boundsBitWidth);
              if (bounds.min.sge(outMin) && bounds.max.sle(outMax)) {
                return Bounds{bounds.min, bounds.max};
              }
            }
          }
          return std::nullopt;
        })
        .Default([&](Operation *) { return std::nullopt; });
  }

  if (op->getNumOperands() == 2) {
    auto lhsOptionalBounds = getBounds(op->getOperand(0));
    auto rhsOptionalBounds = getBounds(op->getOperand(1));

    if (!lhsOptionalBounds.has_value() || !rhsOptionalBounds.has_value()) {
      return std::nullopt;
    }

    auto lhsBounds = lhsOptionalBounds.value();
    auto rhsBounds = rhsOptionalBounds.value();

    return TypeSwitch<Operation *, std::optional<Bounds>>(op)
        .Case<stablehlo::AddOp>([&](stablehlo::AddOp addOp) {
          return Bounds{lhsBounds.min + rhsBounds.min,
                        lhsBounds.max + rhsBounds.max};
        })
        .Case<stablehlo::SubtractOp>([&](stablehlo::SubtractOp subOp) {
          return Bounds{lhsBounds.min - rhsBounds.max,
                        lhsBounds.max - rhsBounds.min};
        })
        .Case<stablehlo::MulOp>([&](stablehlo::MulOp mulOp) {
          auto p1 = lhsBounds.min * rhsBounds.min;
          auto p2 = lhsBounds.min * rhsBounds.max;
          auto p3 = lhsBounds.max * rhsBounds.min;
          auto p4 = lhsBounds.max * rhsBounds.max;
          return Bounds{smin(smin(p1, p2), smin(p3, p4)),
                        smax(smax(p1, p2), smax(p3, p4))};
        })
        .Case<stablehlo::DivOp>(
            [&](stablehlo::DivOp divOp) -> std::optional<Bounds> {
              APInt zero(boundsBitWidth, 0);
              if (rhsBounds.min.sle(zero) && rhsBounds.max.sge(zero)) {
                // divisor range includes zero, cannot compute safe bounds
                return std::nullopt;
              }
              auto d1 = lhsBounds.min.sdiv(rhsBounds.min);
              auto d2 = lhsBounds.min.sdiv(rhsBounds.max);
              auto d3 = lhsBounds.max.sdiv(rhsBounds.min);
              auto d4 = lhsBounds.max.sdiv(rhsBounds.max);

              return Bounds{smin(smin(d1, d2), smin(d3, d4)),
                            smax(smax(d1, d2), smax(d3, d4))};
            })
        .Case<stablehlo::MinOp>([&](stablehlo::MinOp minOp) {
          return Bounds{smin(lhsBounds.min, rhsBounds.min),
                        smin(lhsBounds.max, rhsBounds.max)};
        })
        .Case<stablehlo::MaxOp>([&](stablehlo::MaxOp maxOp) {
          return Bounds{smax(lhsBounds.min, rhsBounds.min),
                        smax(lhsBounds.max, rhsBounds.max)};
        })
        .Case<stablehlo::RemOp>(
            [&](stablehlo::RemOp remOp) -> std::optional<Bounds> {
              // Only handle constant positive RHS for now
              if (rhsBounds.min != rhsBounds.max) {
                return std::nullopt;
              }
              APInt zero(boundsBitWidth, 0);
              APInt rhsVal = rhsBounds.min;
              if (rhsVal.sle(zero)) {
                return std::nullopt;
              }
              // For x % M where M > 0:
              // If lhs >= 0: result is in [0, min(lhs_max, M-1)]
              // If lhs < 0: result is in [max(lhs_min, -(M-1)), 0]
              // If mixed: result is in [max(lhs_min, -(M-1)), min(lhs_max,
              // M-1)]
              APInt maxRemainder = rhsVal - 1;
              APInt minRemainder = -(rhsVal - 1);

              if (lhsBounds.min.sge(zero)) {
                // All non-negative
                return Bounds{zero, smin(lhsBounds.max, maxRemainder)};
              } else if (lhsBounds.max.sle(zero)) {
                // All non-positive
                return Bounds{smax(lhsBounds.min, minRemainder), zero};
              } else {
                // Mixed signs
                return Bounds{smax(lhsBounds.min, minRemainder),
                              smin(lhsBounds.max, maxRemainder)};
              }
            })
        .Default([&](Operation *) { return std::nullopt; });
  }

  return std::nullopt;
}
