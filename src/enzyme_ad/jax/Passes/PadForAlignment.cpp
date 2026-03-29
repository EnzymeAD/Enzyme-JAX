#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace enzyme {

struct AlignmentHandler {
  OpBuilder &builder;
  DenseMap<Value, Value> &paddedValues;
  DenseMap<Value, Value> &originalValues;
  SmallVector<Operation *> toErase;

  AlignmentHandler(OpBuilder &b, DenseMap<Value, Value> &pv,
                   DenseMap<Value, Value> &ov)
      : builder(b), paddedValues(pv), originalValues(ov) {}

  static bool needsPadding(Value v) {
    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (!type)
      return false;
    for (auto p : getPaddingAmounts(type)) {
      if (p > 0)
        return true;
    }
    return false;
  }

  static SmallVector<int64_t> getAlignedShape(RankedTensorType type) {
    auto shape = llvm::to_vector(type.getShape());
    auto padding = getPaddingAmounts(type);
    for (unsigned i = 0; i < shape.size(); ++i) {
      shape[i] += padding[i];
    }
    return shape;
  }

  static SmallVector<int64_t> getPaddingAmounts(RankedTensorType type) {
    SmallVector<int64_t> padding(type.getRank(), 0);
    auto rank = type.getRank();
    for (int i = std::max((int64_t)0, rank - 2); i < rank; ++i) {
      if (type.getDimSize(i) >= 64 && type.getDimSize(i) % 128 != 0) {
        padding[i] =
            ((type.getDimSize(i) + 127) / 128 * 128) - type.getDimSize(i);
      }
    }
    return padding;
  }

  Operation *createPlaceholder(Value v) {
    auto type = cast<RankedTensorType>(v.getType());
    auto alignedShape = AlignmentHandler::getAlignedShape(type);
    auto paddedType = type.clone(alignedShape);

    OperationState state(v.getLoc(), "enzyme_temporary_placeholder");
    state.addOperands({v});
    state.addTypes({paddedType});

    auto placeholder = builder.create(state);
    paddedValues[v] = placeholder->getResult(0);
    return placeholder;
  }

  Value getValueOrPadded(Value v) {
    auto it = paddedValues.find(v);
    if (it != paddedValues.end())
      return it->second;
    return v;
  }

  Value getOrCreatePadOp(Value v);
  Value getOrCreateSliceOp(Value paddedV, Value origV);

  bool handleConstantOp(stablehlo::ConstantOp op);
  bool handleReturnOp(stablehlo::ReturnOp op);
  bool handleIfOp(stablehlo::IfOp op);
  bool handleWhileOp(stablehlo::WhileOp op);
  bool handlePadOp(stablehlo::PadOp op);
  bool handleSliceOp(stablehlo::SliceOp op);
  bool handleDynamicUpdateSliceOp(stablehlo::DynamicUpdateSliceOp op);
  bool handleBroadcastInDimOp(stablehlo::BroadcastInDimOp op);
  bool handleElementwiseOp(Operation *op);
  bool handleSelectOp(stablehlo::SelectOp op);
  bool handleConcatenateOp(stablehlo::ConcatenateOp op);
  bool handleDotGeneralOp(stablehlo::DotGeneralOp op);
  bool handleTransposeOp(stablehlo::TransposeOp op);

  // it doesn't update the results of `op`; i.e. just updates the `paddedValues`
  // and may mark it for erasure. the replacement is done on a later stage.
  void eraseWithReplacement(Operation *op, ValueRange replacements,
                            bool should_erase = true) {
    // even if the op results do not need padding (and thus they weren't before 
    // in `paddedValues`), their uses need to be updated
    for (auto [res, rep] : llvm::zip_equal(op->getResults(), replacements))
      paddedValues[res] = rep;
    if (should_erase)
      toErase.push_back(op);
  }
};

#define GEN_PASS_DEF_PADFORALIGNMENTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

class PadForAlignmentPass
    : public impl::PadForAlignmentPassBase<PadForAlignmentPass> {
public:
  using impl::PadForAlignmentPassBase<
      PadForAlignmentPass>::PadForAlignmentPassBase;

  void runOnFunction(func::FuncOp func);

  void runOnOperation() override {
    auto op = getOperation();
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      runOnFunction(func);
    } else if (auto mod = dyn_cast<ModuleOp>(op)) {
      for (auto f : mod.getOps<func::FuncOp>()) {
        runOnFunction(f);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// AlignmentHandler Implementations
//===----------------------------------------------------------------------===//

Value AlignmentHandler::getOrCreateSliceOp(Value paddedV, Value origV) {
  if (originalValues.contains(paddedV))
    return originalValues[paddedV];
  auto origType = cast<RankedTensorType>(origV.getType());
  auto paddedType = cast<RankedTensorType>(paddedV.getType());

  if (origType == paddedType)
    return paddedV;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(paddedV);

  SmallVector<int64_t> start(origType.getRank(), 0);
  SmallVector<int64_t> limit(origType.getShape().begin(),
                             origType.getShape().end());
  SmallVector<int64_t> strides(origType.getRank(), 1);

  auto sliceOp = stablehlo::SliceOp::create(builder, paddedV.getLoc(), origType,
                                            paddedV, start, limit, strides);
  originalValues[paddedV] = sliceOp.getResult();
  return sliceOp.getResult();
}

Value AlignmentHandler::getOrCreatePadOp(Value v) {
  if (paddedValues.contains(v))
    return paddedValues[v];

  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type)
    return v;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(v);

  auto alignedShape = getAlignedShape(type);

  auto elementOp = builder.create<stablehlo::ConstantOp>(
      v.getLoc(),
      DenseElementsAttr::get(RankedTensorType::get({}, type.getElementType()),
                             makeAttr(type.getElementType(), 0)));

  SmallVector<int64_t> edgePaddingLow(type.getRank(), 0);
  SmallVector<int64_t> edgePaddingHigh(type.getRank(), 0);
  SmallVector<int64_t> interiorPadding(type.getRank(), 0);

  auto rank = type.getRank();
  for (int i = std::max((int64_t)0, rank - 2); i < rank; ++i) {
    if (type.getDimSize(i) >= 64 && type.getDimSize(i) % 128 != 0) {
      edgePaddingHigh[i] = alignedShape[i] - type.getDimSize(i);
    }
  }

  auto paddedType = type.clone(alignedShape);
  auto padOp = stablehlo::PadOp::create(builder, v.getLoc(), paddedType, v,
                                        elementOp.getResult(), edgePaddingLow,
                                        edgePaddingHigh, interiorPadding);
  paddedValues[v] = padOp.getResult();
  return padOp.getResult();
}

bool AlignmentHandler::handleConstantOp(stablehlo::ConstantOp op) {
  auto res = op.getResult();
  if (!needsPadding(res))
    return false;

  builder.setInsertionPoint(op);
  auto type = cast<RankedTensorType>(res.getType());
  auto alignedShape = getAlignedShape(type);
  auto paddedType = type.clone(alignedShape);

  auto value = op.getValue();
  if (auto splat = dyn_cast<SplatElementsAttr>(value)) {
    auto bcastValue = splat.getSplatValue<Attribute>();
    auto newAttr = DenseElementsAttr::get(paddedType, bcastValue);
    auto newOp = builder.create<stablehlo::ConstantOp>(op.getLoc(), newAttr);
    eraseWithReplacement(op, newOp.getResult());
  } else {
    auto padZero = builder.create<stablehlo::ConstantOp>(
        op.getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({}, type.getElementType()),
                               builder.getZeroAttr(type.getElementType())));
    auto padding = getPaddingAmounts(type);
    SmallVector<int64_t> low(type.getRank(), 0);
    SmallVector<int64_t> high(type.getRank(), 0);
    SmallVector<int64_t> interior(type.getRank(), 0);
    for (int i = 0; i < type.getRank(); ++i) {
      high[i] = padding[i];
    }
    auto padOp =
        stablehlo::PadOp::create(builder, op.getLoc(), paddedType, res,
                                 padZero.getResult(), low, high, interior);
    eraseWithReplacement(op, padOp.getResult());
  }
  return true;
}

bool AlignmentHandler::handleReturnOp(stablehlo::ReturnOp op) {
  auto parent = op->getParentOp();
  if (!parent || !isa<stablehlo::IfOp, stablehlo::WhileOp>(parent))
    return false;

  SmallVector<Value> newOperands = llvm::to_vector(op.getOperands());
  for (auto &val : newOperands) {
    if (paddedValues.count(val)) {
      val = paddedValues[val];
    }
  }
  op.getResultsMutable().assign(newOperands);
  return true;
}

bool AlignmentHandler::handleIfOp(stablehlo::IfOp op) {
  bool needsUpdate = false;
  SmallVector<Type> newRetTypes;
  for (auto res : op.getResults()) {
    if (paddedValues.contains(res)) {
      needsUpdate = true;
      newRetTypes.push_back(paddedValues[res].getType());
    } else {
      newRetTypes.push_back(res.getType());
    }
  }

  if (needsUpdate) {
    builder.setInsertionPoint(op);
    auto newIf =
        builder.create<stablehlo::IfOp>(op.getLoc(), newRetTypes, op.getPred());

    newIf.getTrueBranch().takeBody(op.getTrueBranch());
    newIf.getFalseBranch().takeBody(op.getFalseBranch());

    // update uses of non-padded results to point to the new if results
    for (auto [i, res] : llvm::enumerate(op->getResults()))
      if (!paddedValues.contains(res))
        res.replaceAllUsesWith(newIf.getResult(i));

    eraseWithReplacement(op, newIf.getResults());
  }
  return true;
}

bool AlignmentHandler::handleWhileOp(stablehlo::WhileOp op) {
  if (!llvm::any_of(op->getOperands(), needsPadding) &&
      !llvm::any_of(op->getResults(), needsPadding))
    return false;

  bool needsUpdate = false;
  SmallVector<Type> newRetTypes;
  SmallVector<Value> newOperands;

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto operand = op->getOperand(i);
    if (paddedValues.contains(operand)) {
      needsUpdate = true;
      newOperands.push_back(paddedValues[operand]);
      newRetTypes.push_back(paddedValues[operand].getType());
    } else {
      newOperands.push_back(operand);
      newRetTypes.push_back(operand.getType());
    }
  }

  if (needsUpdate) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    auto newWhile = builder.create<stablehlo::WhileOp>(op.getLoc(), newRetTypes,
                                                       newOperands);

    newWhile.getCond().takeBody(op.getCond());
    newWhile.getBody().takeBody(op.getBody());

    // update block arguments in both regions
    auto &condBlock = newWhile.getCond().front();
    auto &bodyBlock = newWhile.getBody().front();
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      if (paddedValues.contains(operand)) {
        condBlock.getArgument(i).setType(paddedValues[operand].getType());
        bodyBlock.getArgument(i).setType(paddedValues[operand].getType());

        // skip padding of block arguments since they already have been padded
        // before the while op
        paddedValues[condBlock.getArgument(i)].replaceAllUsesWith(
            condBlock.getArgument(i));
        paddedValues[bodyBlock.getArgument(i)].replaceAllUsesWith(
            bodyBlock.getArgument(i));
        paddedValues.erase(condBlock.getArgument(i));
        paddedValues.erase(bodyBlock.getArgument(i));
      }
    }

    // update uses of non-padded results to point to the new while results
    for (auto [i, res] : llvm::enumerate(op->getResults())) {
      if (!paddedValues.contains(res)) {
        res.replaceAllUsesWith(newWhile.getResult(i));
      }
    }

    eraseWithReplacement(op, newWhile.getResults());
  }

  return true;
}

bool AlignmentHandler::handlePadOp(stablehlo::PadOp op) {
  auto input = op.getOperand();
  auto res = op.getResult();

  if (!needsPadding(res) && !needsPadding(input))
    return false;

  auto type = cast<RankedTensorType>(op.getType());
  auto alignedShape = getAlignedShape(type);
  auto paddedType = type.clone(alignedShape);
  auto resultPadding = getPaddingAmounts(type);
  auto inputPadding = getPaddingAmounts(input.getType());

  SmallVector<int64_t> low = llvm::to_vector(op.getEdgePaddingLow());
  SmallVector<int64_t> high = llvm::to_vector(op.getEdgePaddingHigh());
  SmallVector<int64_t> interior = llvm::to_vector(op.getInteriorPadding());

  // if input is already aligned, just create a bigger padded op
  if (!needsPadding(input)) {
    for (auto [high_i, result_pad_i] : llvm::zip(high, resultPadding))
      high_i += result_pad_i;

    builder.setInsertionPointAfterValue(op);
    auto padOp =
        stablehlo::PadOp::create(builder, op.getLoc(), paddedType, input,
                                 op.getPaddingValue(), low, high, interior);
    eraseWithReplacement(op, padOp.getResult());
    return true;
  }

  bool all_zero_input_padding = true;
  for (int i = 0; i < type.getRank(); ++i) {
    high[i] += resultPadding[i] - (interior[i] + 1) * inputPadding[i];
    if (high[i] != 0)
      all_zero_input_padding = false;

    // check that negative paddings do not remove real data, just the already existing padding
    if (high[i] < 0)
      assert(inputPadding[i] + high[i] >= 0);
  }
  if (all_zero_input_padding) {
    eraseWithReplacement(op, getValueOrPadded(input));
    return true;
  }
  builder.setInsertionPointAfterValue(op);
  auto padOp =
      // stablehlo::PadOp::create(builder, op.getLoc(), paddedType, paddedInput,
      //                          op.getPaddingValue(), low, high, interior);
      stablehlo::PadOp::create(builder, op.getLoc(), paddedType, getValueOrPadded(input),
                               op.getPaddingValue(), low, high, interior);

  eraseWithReplacement(op, padOp.getResult());

  return true;
}

bool AlignmentHandler::handleSliceOp(stablehlo::SliceOp op) {
  auto input = op.getOperand();
  auto res = op.getResult();

  if (!needsPadding(res) && !needsPadding(input))
    return false;

  builder.setInsertionPoint(op);
  Value paddedInput = getValueOrPadded(input);

  auto start = llvm::to_vector(op.getStartIndices());
  auto limit = llvm::to_vector(op.getLimitIndices());
  auto strides = llvm::to_vector(op.getStrides());

  auto outType = cast<RankedTensorType>(res.getType());
  auto alignedOutShape = getAlignedShape(outType);

  SmallVector<int64_t> newLimit = limit;
  auto rank = outType.getRank();
  auto inputType = cast<RankedTensorType>(paddedInput.getType());

  bool needsPadAfter = false;
  auto padding = getPaddingAmounts(outType);
  for (int i = 0; i < rank; ++i) {
    if (padding[i] > 0) {
      long targetLimit = start[i] + alignedOutShape[i];
      if (targetLimit <= inputType.getDimSize(i)) {
        newLimit[i] = targetLimit;
      } else {
        newLimit[i] = inputType.getDimSize(i);
        needsPadAfter = true;
      }
    }
  }

  SmallVector<int64_t> intermediateShape = llvm::to_vector(outType.getShape());
  for (int i = 0; i < rank; ++i) {
    intermediateShape[i] =
        (newLimit[i] - start[i] + strides[i] - 1) / strides[i];
  }
  auto intermediateType = outType.clone(intermediateShape);

  auto sliceOp =
      stablehlo::SliceOp::create(builder, op.getLoc(), intermediateType,
                                 paddedInput, start, newLimit, strides);

  Value sliceRes = sliceOp.getResult();
  auto paddedOutType = outType.clone(alignedOutShape);

  if (needsPadAfter) {
    auto padZero = builder.create<stablehlo::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(
                         RankedTensorType::get({}, outType.getElementType()),
                         builder.getZeroAttr(outType.getElementType())));

    SmallVector<int64_t> edgePaddingLow(rank, 0);
    SmallVector<int64_t> edgePaddingHigh(rank, 0);
    SmallVector<int64_t> interiorPadding(rank, 0);

    for (int i = std::max((int64_t)0, rank - 2); i < rank; ++i) {
      edgePaddingHigh[i] = alignedOutShape[i] - intermediateShape[i];
    }

    auto padOp = stablehlo::PadOp::create(
        builder, op.getLoc(), paddedOutType, sliceRes, padZero.getResult(),
        edgePaddingLow, edgePaddingHigh, interiorPadding);
    paddedValues[res] = padOp.getResult();
    eraseWithReplacement(op, ValueRange{padOp.getResult()});
  } else {
    paddedValues[res] = sliceRes;
    eraseWithReplacement(op, ValueRange{sliceRes});
  }
  return true;
}

bool AlignmentHandler::handleDynamicUpdateSliceOp(
    stablehlo::DynamicUpdateSliceOp op) {
  auto operand = op.getOperand();
  auto update = op.getUpdate();
  auto res = op.getResult();

  if (!paddedValues.contains(operand) && !paddedValues.contains(update))
    return false;

  builder.setInsertionPoint(op);
  auto paddedOperand = getOrCreatePadOp(operand);
  auto paddedUpdate = getOrCreatePadOp(update);

  auto unpaddedUpdateType = cast<RankedTensorType>(update.getType());
  auto paddedUpdateType = cast<RankedTensorType>(paddedUpdate.getType());
  auto unpaddedOperandType = cast<RankedTensorType>(operand.getType());

  bool needsSliceUpdate = false;
  SmallVector<int64_t> updateSliceLimit(paddedUpdateType.getShape().begin(),
                                        paddedUpdateType.getShape().end());

  for (size_t i = 0; i < op.getStartIndices().size(); ++i) {
    auto idxOp = op.getStartIndices()[i].getDefiningOp();
    bool isStatic = false;
    int64_t startVal = 0;

    if (auto constOp = dyn_cast_or_null<stablehlo::ConstantOp>(idxOp)) {
      if (auto attr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
        auto vals = attr.getValues<APInt>();
        if (!vals.empty()) {
          startVal = vals[0].getSExtValue();
          isStatic = true;
        }
      }
    }

    if (!isStatic) {
      updateSliceLimit[i] = unpaddedUpdateType.getDimSize(i);
      needsSliceUpdate = true;
    } else {
      if (startVal + unpaddedUpdateType.getDimSize(i) <
          unpaddedOperandType.getDimSize(i)) {
        updateSliceLimit[i] = unpaddedUpdateType.getDimSize(i);
        needsSliceUpdate = true;
      }
    }
  }

  Value slicedUpdate = paddedUpdate;
  if (needsSliceUpdate) {
    SmallVector<int64_t> start(paddedUpdateType.getRank(), 0);
    SmallVector<int64_t> strides(paddedUpdateType.getRank(), 1);

    auto sliceType = paddedUpdateType.clone(updateSliceLimit);
    auto sliceOp = builder.create<stablehlo::SliceOp>(
        op.getLoc(), sliceType, paddedUpdate, start, updateSliceLimit, strides);
    slicedUpdate = sliceOp.getResult();
  }

  SmallVector<Value> startIndices;
  for (auto idx : op.getStartIndices()) {
    if (paddedValues.contains(idx)) {
      startIndices.push_back(getOrCreateSliceOp(paddedValues[idx], idx));
    } else {
      startIndices.push_back(idx);
    }
  }

  auto outType = cast<RankedTensorType>(res.getType());
  auto alignedOutShape = getAlignedShape(outType);
  auto paddedOutType = outType.clone(alignedOutShape);

  auto newOp = builder.create<stablehlo::DynamicUpdateSliceOp>(
      op.getLoc(), paddedOutType, paddedOperand, slicedUpdate, startIndices);

  paddedValues[res] = newOp.getResult();
  eraseWithReplacement(op, ValueRange{newOp.getResult()});
  return true;
}

bool AlignmentHandler::handleBroadcastInDimOp(stablehlo::BroadcastInDimOp op) {
  auto input = op.getOperand();
  auto res = op.getResult();

  if (!paddedValues.contains(input) && !needsPadding(res))
    return false;

  builder.setInsertionPoint(op);
  auto paddedInput = getOrCreatePadOp(input);
  auto bcastDims = op.getBroadcastDimensions();

  auto outType = cast<RankedTensorType>(res.getType());
  auto alignedOutShape = getAlignedShape(outType);

  SmallVector<int64_t> intermediateShape = alignedOutShape;
  bool needsPadAfter = false;
  bool needsSliceAfter = false;
  SmallVector<int64_t> padHigh(outType.getRank(), 0);
  SmallVector<int64_t> sliceLimit = alignedOutShape;

  auto inputPaddedType = cast<RankedTensorType>(paddedInput.getType());

  for (size_t i = 0; i < bcastDims.size(); ++i) {
    int64_t outDim = bcastDims[i];
    auto inputDimSize = inputPaddedType.getDimSize(i);
    if (inputDimSize != alignedOutShape[outDim]) {
      intermediateShape[outDim] = inputDimSize;
      if (inputDimSize < alignedOutShape[outDim]) {
        padHigh[outDim] = alignedOutShape[outDim] - inputDimSize;
        needsPadAfter = true;
      } else {
        sliceLimit[outDim] = alignedOutShape[outDim];
        needsSliceAfter = true;
      }
    }
  }

  auto intermediateType = outType.clone(intermediateShape);
  auto newOp = builder.create<stablehlo::BroadcastInDimOp>(
      op.getLoc(), intermediateType, paddedInput, bcastDims);

  Value currentOut = newOp.getResult();

  if (needsSliceAfter) {
    SmallVector<int64_t> start(outType.getRank(), 0);
    SmallVector<int64_t> strides(outType.getRank(), 1);

    SmallVector<int64_t> sliceShape = intermediateShape;
    for (int i = 0; i < outType.getRank(); ++i) {
      sliceShape[i] = sliceLimit[i];
    }
    auto sliceType = outType.clone(sliceShape);

    auto sliceOp =
        stablehlo::SliceOp::create(builder, op.getLoc(), sliceType, currentOut,
                                   start, sliceLimit, strides);
    currentOut = sliceOp.getResult();
  }

  if (needsPadAfter) {
    auto padZero = builder.create<stablehlo::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(
                         RankedTensorType::get({}, outType.getElementType()),
                         builder.getZeroAttr(outType.getElementType())));

    SmallVector<int64_t> edgePaddingLow(outType.getRank(), 0);
    SmallVector<int64_t> edgePaddingHigh = padHigh;
    SmallVector<int64_t> interiorPadding(outType.getRank(), 0);

    auto paddedType = outType.clone(alignedOutShape);
    auto padOp = stablehlo::PadOp::create(
        builder, op.getLoc(), paddedType, currentOut, padZero.getResult(),
        edgePaddingLow, edgePaddingHigh, interiorPadding);
    currentOut = padOp.getResult();
  }

  paddedValues[res] = currentOut;
  eraseWithReplacement(op, ValueRange{currentOut});
  return true;
}

bool AlignmentHandler::handleSelectOp(stablehlo::SelectOp op) {
  auto cond = op.getPred();
  auto trueVal = op.getOnTrue();
  auto falseVal = op.getOnFalse();
  auto res = op.getResult();

  bool anyPadded = paddedValues.contains(cond) ||
                   paddedValues.contains(trueVal) ||
                   paddedValues.contains(falseVal);
  if (!anyPadded)
    return false;

  builder.setInsertionPoint(op);

  Value paddedCond = cond;
  if (cast<RankedTensorType>(cond.getType()).getRank() > 0) {
    paddedCond = getOrCreatePadOp(cond);
  }

  auto paddedTrue = getOrCreatePadOp(trueVal);
  auto paddedFalse = getOrCreatePadOp(falseVal);

  auto type = cast<RankedTensorType>(res.getType());
  auto alignedShape = getAlignedShape(type);
  auto paddedType = type.clone(alignedShape);

  auto newOp = builder.create<stablehlo::SelectOp>(
      op.getLoc(), paddedType, paddedCond, paddedTrue, paddedFalse);

  eraseWithReplacement(op, newOp.getResult());
  return true;
}

bool AlignmentHandler::handleConcatenateOp(stablehlo::ConcatenateOp op) {
  auto res = op.getResult();
  int64_t axis = op.getDimension();

  bool anyPadded = false;
  for (auto v : op.getInputs()) {
    if (paddedValues.contains(v))
      anyPadded = true;
  }
  if (!anyPadded)
    return false;

  builder.setInsertionPoint(op);
  SmallVector<Value> operands;

  for (auto v : op.getInputs()) {
    if (!paddedValues.contains(v)) {
      operands.push_back(getOrCreatePadOp(v));
    } else {
      auto paddedV = paddedValues[v];
      auto unpaddedType = cast<RankedTensorType>(v.getType());
      auto paddedType = cast<RankedTensorType>(paddedV.getType());

      if (paddedType.getDimSize(axis) > unpaddedType.getDimSize(axis)) {
        SmallVector<int64_t> start(unpaddedType.getRank(), 0);
        SmallVector<int64_t> limit(paddedType.getShape().begin(),
                                   paddedType.getShape().end());
        limit[axis] = unpaddedType.getDimSize(axis);
        SmallVector<int64_t> strides(unpaddedType.getRank(), 1);

        auto sliceType = paddedType.clone(limit);
        auto sliceOp = builder.create<stablehlo::SliceOp>(
            op.getLoc(), sliceType, paddedV, start, limit, strides);
        operands.push_back(sliceOp.getResult());
      } else {
        operands.push_back(paddedV);
      }
    }
  }

  SmallVector<int64_t> newShape(
      cast<RankedTensorType>(operands[0].getType()).getShape().begin(),
      cast<RankedTensorType>(operands[0].getType()).getShape().end());
  newShape[axis] = 0;
  for (auto opnd : operands) {
    newShape[axis] += cast<RankedTensorType>(opnd.getType()).getDimSize(axis);
  }

  auto outType = cast<RankedTensorType>(res.getType());
  auto paddedOutType = outType.clone(newShape);

  auto newOp = builder.create<stablehlo::ConcatenateOp>(
      op.getLoc(), paddedOutType, operands, op.getDimension());

  paddedValues[res] = newOp.getResult();
  return true;
}

bool AlignmentHandler::handleElementwiseOp(Operation *op) {
  if (!needsPadding(op->getResult(0)))
    return false;

  builder.setInsertionPoint(op);
  SmallVector<Value> paddedArgs;
  for (auto arg : op->getOperands()) {
    paddedArgs.push_back(getValueOrPadded(arg));
  }

  auto res = op->getResult(0);
  auto resType = cast<RankedTensorType>(res.getType());
  auto paddedResType = resType.clone(getAlignedShape(resType));

  OperationState state(op->getLoc(), op->getName().getStringRef());
  state.addOperands(paddedArgs);
  state.addTypes({paddedResType});
  state.addAttributes(op->getAttrs());

  auto newOp = builder.create(state);
  eraseWithReplacement(op, newOp->getResults());
  return true;
}

bool AlignmentHandler::handleDotGeneralOp(stablehlo::DotGeneralOp op) {
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto res = op.getResult();

  if (!paddedValues.contains(lhs) && !paddedValues.contains(rhs))
    return false;

  builder.setInsertionPoint(op);
  auto paddedLhs = getOrCreatePadOp(lhs);
  auto paddedRhs = getOrCreatePadOp(rhs);

  auto resType = cast<RankedTensorType>(res.getType());
  auto alignedResShape = getAlignedShape(resType);
  auto paddedResType = resType.clone(alignedResShape);

  auto newOp = builder.create<stablehlo::DotGeneralOp>(
      op.getLoc(), paddedResType, paddedLhs, paddedRhs,
      op.getDotDimensionNumbers(), op.getPrecisionConfigAttr(),
      op.getAlgorithmAttr());
  eraseWithReplacement(op, newOp.getResult());
  return true;
}

bool AlignmentHandler::handleTransposeOp(stablehlo::TransposeOp op) {
  auto input = op.getOperand();
  auto res = op.getResult();

  if (!paddedValues.contains(input))
    return false;

  builder.setInsertionPoint(op);
  auto paddedInput = getOrCreatePadOp(input);
  auto perm = op.getPermutation();

  auto resType = cast<RankedTensorType>(res.getType());
  auto alignedResShape = getAlignedShape(resType);
  auto paddedResType = resType.clone(alignedResShape);

  auto newOp = builder.create<stablehlo::TransposeOp>(
      op.getLoc(), paddedResType, paddedInput, perm);

  eraseWithReplacement(op, newOp->getResult(0));
  return true;
}

void PadForAlignmentPass::runOnFunction(func::FuncOp func) {
  OpBuilder builder(func.getContext());
  DenseMap<Value, Value> paddedValues;
  DenseMap<Value, Value> originalValues;

  AlignmentHandler handler(builder, paddedValues, originalValues);

  // create placeholders for all original values that need padding
  DenseMap<Value, Operation *> placeholders;
  SmallVector<Operation *> ops;

  func.walk([&](Operation *op) {
    if (op == func.getOperation())
      return;

    ops.push_back(op);

    for (auto res : op->getResults()) {
      if (!handler.needsPadding(res))
        continue;

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(op);
      auto placeholder = handler.createPlaceholder(res);
      placeholders[res] = placeholder;
      paddedValues[res] = placeholder->getResult(0);
    }
  });

  func.walk([&](Block *block) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);

    for (auto arg : block->getArguments()) {
      if (!handler.needsPadding(arg))
        continue;

      auto type = cast<RankedTensorType>(arg.getType());
      auto alignedShape = handler.getAlignedShape(type);

      Operation *replacement = handler.createPlaceholder(arg);
      placeholders[arg] = replacement;
      paddedValues[arg] = replacement->getResult(0);
    }
  });

  // optimal propagation traversal
  for (auto op : ops) {
    bool handled = false;
    if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op)) {
      handled = handler.handleConstantOp(constOp);
    } else if (auto returnOp = dyn_cast<stablehlo::ReturnOp>(op)) {
      handled = handler.handleReturnOp(returnOp);
    } else if (auto ifOp = dyn_cast<stablehlo::IfOp>(op)) {
      handled = handler.handleIfOp(ifOp);
    } else if (auto whileOp = dyn_cast<stablehlo::WhileOp>(op)) {
      handled = handler.handleWhileOp(whileOp);
    } else if (auto pad = dyn_cast<stablehlo::PadOp>(op)) {
      handled = handler.handlePadOp(pad);
    } else if (auto slice = dyn_cast<stablehlo::SliceOp>(op)) {
      handled = handler.handleSliceOp(slice);
    } else if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op)) {
      handled = handler.handleDynamicUpdateSliceOp(dus);
    } else if (auto select = dyn_cast<stablehlo::SelectOp>(op)) {
      handled = handler.handleSelectOp(select);
    } else if (auto concat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
      handled = handler.handleConcatenateOp(concat);
    } else if (stablehlo::hasTraitElementwise(op)) {
      handled = handler.handleElementwiseOp(op);
    } else if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op)) {
      handled = handler.handleDotGeneralOp(dot);
    } else if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
      handled = handler.handleBroadcastInDimOp(bcast);
    } else if (auto transpose = dyn_cast<stablehlo::TransposeOp>(op)) {
      handled = handler.handleTransposeOp(transpose);
    }

    if (!handled) {
      // Boundary fallback: slice any padded inputs
      for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
        if (paddedValues.contains(operand)) {
          builder.setInsertionPoint(op);
          auto slice =
              handler.getOrCreateSliceOp(paddedValues[operand], operand);
          op->setOperand(i, slice);
        }
      }
    }
  }

  // resolve placeholders
  for (auto [orig, ph] : placeholders) {
    Value phRes = ph->getResult(0);

    // placeholder has already been handled, so just replace uses
    if (handler.paddedValues[orig] != phRes) {
      phRes.replaceAllUsesWith(handler.paddedValues[orig]);
      continue;
    }

    // placeholder has not been handled, so create a pad op
    handler.paddedValues.erase(orig); // clear cache lookup
    builder.setInsertionPoint(ph);
    auto realPad = handler.getOrCreatePadOp(orig);
    phRes.replaceAllUsesWith(realPad);
  }

  // safely delete all placeholders
  for (auto [_, ph] : placeholders) {
    ph->erase();
  }

  // safely delete any original ops that are now dead
  for (auto it = handler.toErase.rbegin(); it != handler.toErase.rend(); ++it) {
    auto op = *it;
    assert(op->use_empty());
    op->erase();
  }
}

} // namespace enzyme
} // namespace mlir
