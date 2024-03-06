//===- ArithAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

namespace {
#include "src/enzyme_ad/jax/Implementations/StableHLODerivatives.inc"

// From
// https://github.com/openxla/stablehlo/blob/5d1a9c892500c2e9fecbfedfa66ffe84ff1caf7b/stablehlo/dialect/StablehloOps.cpp#L1498C1-L1532C1
bool hasSameOperandAndResultTypes(Operation &op) {
  Type expected;
  if (op.getNumResults() != 0)
    expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0)
    expected = op.getOperand(0).getType();
  if (!expected)
    return false;

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

static bool isEligibleForCompactPrint(ReduceOp op) {
  // Check E1.
  auto &block = op.getBody().front();
  if (!hasSingleElement(block.without_terminator()))
    return false;

  Operation &innerOp = *block.begin();

  // Check E2.
  if (innerOp.getDialect() != op->getDialect())
    return false;

  if (innerOp.getNumOperands() != 2 ||
      !innerOp.hasTrait<mlir::OpTrait::OneResult>() ||
      !hasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::hlo::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegions>())
    return false;

  // Check E3.
  if (op.getInputs().empty())
    return false;

  auto elemType =
      op.getInputs()[0].getType().cast<ShapedType>().getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType)
    return false;

  // Check E4.
  if (!llvm::equal(block.getArguments(), innerOp.getOperands()))
    return false;

  // Check E5.
  auto retOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!retOp)
    return false;

  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

template <typename OpTy>
class AutoDiffReduceCF : public ControlFlowAutoDiffOpInterface::ExternalModel<
                             AutoDiffReduceCF<OpTy>, OpTy> {
public:
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return builder.create<OpTy>(original->getLoc(), rettys, remappedOperands,
                                original->getAttrs());
  }
};

template <typename OpTy>
class AutoDiffReduceFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffReduceFwd<OpTy>, OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto red = cast<OpTy>(orig);
    if (!isEligibleForCompactPrint(red)) {
      orig->emitError() << "Unsupported operation in reduction autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    Operation &innerOp = red.getBody().front().front();

    if (isa<MaxOp>(innerOp) || isa<MinOp>(innerOp)) {
      llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
      llvm::SmallDenseSet<unsigned> resultPositionsToShadow;
      for (auto operand : orig->getOpResults()) {
        if (!gutils->isConstantValue(operand)) {
          operandPositionsToShadow.insert(
              red.getInitValues().getBeginOperandIndex() +
              operand.getResultNumber());
          operandPositionsToShadow.insert(
              red.getInputs().getBeginOperandIndex() +
              operand.getResultNumber());
          resultPositionsToShadow.insert(operand.getResultNumber());
        }
      }
      return mlir::enzyme::detail::controlFlowForwardHandler(
          orig, builder, gutils, operandPositionsToShadow,
          resultPositionsToShadow);
    }

    if (!isa<AddOp>(innerOp)) {
      orig->emitError() << "Unsupported operation in reduction autodiff(2): "
                        << *orig << "\n";
      return failure();
    }

    Operation *primal = gutils->getNewFromOriginal(orig);

    IRMapping map;
    for (auto &operand : orig->getOpOperands()) {
      if (!gutils->isConstantValue(operand.get())) {
        map.map(operand.get(), gutils->invertPointerM(operand.get(), builder));
        continue;
      }
      if (auto iface =
              dyn_cast<AutoDiffTypeInterface>(operand.get().getType())) {
        if (!iface.isMutable()) {
          Type retTy = iface.getShadowType();
          auto toret = retTy.cast<AutoDiffTypeInterface>().createNullValue(
              builder, operand.get().getLoc());
          map.map(operand.get(), toret);
          continue;
        }
      }
      orig->emitError() << "Unsupported constant arg to reduce forward "
                           "handler(opidx="
                        << operand.getOperandNumber()
                        << ", op=" << operand.get() << ")\n";
      return failure();
    }
    Operation *shadow = builder.clone(*orig, map);

    Value shadowRes = shadow->getResult(0);

    auto invAdd = gutils->invertedPointers.lookup(innerOp.getResult(0));
    gutils->invertedPointers.erase(innerOp.getResult(0));
    gutils->erase(invAdd.getDefiningOp());
    BitVector baToErase(cast<OpTy>(primal).getBody().front().getNumArguments());
    for (auto ba : red.getBody().front().getArguments()) {
      auto invBA = cast<BlockArgument>(gutils->invertedPointers.lookup(ba));
      gutils->invertedPointers.erase(ba);
      baToErase.set(invBA.getArgNumber());
    }
    cast<OpTy>(primal).getBody().front().eraseArguments(baToErase);

    gutils->setDiffe(orig->getResult(0), shadowRes, builder);
    gutils->eraseIfUnused(orig);
    return success();
  }
};

class AutoDiffBroadcastInDimRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffBroadcastInDimRev, BroadcastInDimOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto op = cast<BroadcastInDimOp>(orig);
    auto inTy = op.getOperand().getType();
    auto outTy = op.getType();
    auto inDiffe = gutils->diffe(op, builder);
    gutils->zeroDiffe(op, builder);

    SmallVector<int64_t> bcastDims(op.getBroadcastDimensions().begin(),
                                   op.getBroadcastDimensions().end());

    Value zero = nullptr;
    if (bcastDims.size() == 0 && inTy.getShape().size() == 0) {
      for (size_t i = 0; i < outTy.getShape().size(); i++) {
        bcastDims.push_back(i);
      }
      zero = gutils->getShadowType(inTy)
                    .cast<AutoDiffTypeInterface>()
                    .createNullValue(builder, op.getLoc());
    } else {
      SmallVector<int64_t> dims;
      for (size_t i = 0; i < inTy.getShape().size(); i++) {
          if (llvm::is_contained(bcastDims, i)) continue;
          dims.push_back(i);
      }
      zero = gutils->getShadowType(RankedTensorType::get(dims, inTy.getElementType()))
                    .cast<AutoDiffTypeInterface>()
                    .createNullValue(builder, op.getLoc());
    }

    auto red = builder.create<ReduceOp>(op.getLoc(),
                                        TypeRange(zero.getType()),
                                        inDiffe, zero, bcastDims);
    red.getBody().push_back(new Block());
    Block &body = red.getBody().front();
    OpBuilder bodyBuilder(orig->getContext());
    bodyBuilder.setInsertionPointToEnd(&body);

    body.addArgument(zero.getType(), op.getLoc());
    body.addArgument(zero.getType(), op.getLoc());
    auto add = bodyBuilder.create<AddOp>(op.getLoc(), body.getArgument(0),
                                         body.getArgument(1));
    bodyBuilder.create<ReturnOp>(op.getLoc(), ValueRange(add));

    llvm::errs() << " red: " << *red << "\n";

    Value res = red->getResult(0);
    Type resTy =  gutils->getShadowType(op.getOperand().getType());
    if (res.getType() != resTy)
        res = builder.create<ReshapeOp>(op.getLoc(), resTy, res);

    gutils->addToDiffe(op.getOperand(), res, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class AutoDiffSliceRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffSliceRev,
                                                       SliceOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto op = cast<SliceOp>(orig);
    auto inTy = op.getOperand().getType();
    auto outTy = op.getType();
    auto zero = inTy.cast<AutoDiffTypeInterface>().createNullValue(builder,
                                                                   op.getLoc());
    auto inDiffe = gutils->diffe(op, builder);
    gutils->zeroDiffe(op, builder);

    Value idxs;
    {
      SmallVector<int64_t> concat_data;
      for (size_t i = 0; i < outTy.getShape().size(); i++) {
        concat_data.push_back(outTy.getShape()[i]);
      }
      concat_data.push_back(1);
      auto toConcatType =
          RankedTensorType::get(concat_data, builder.getI32Type());
      std::vector<Value> inds;
      size_t idx = 0;
      for (auto &&[start, limit, stride] : llvm::zip(
               op.getStartIndices(), op.getLimitIndices(), op.getStrides())) {
        std::vector<int32_t> data;
        for (int32_t i = start; i < limit; i += stride) {
          data.push_back(i);
        }

        Value ind = builder.create<ConstantOp>(op.getLoc(), RankedTensorType::get({data.size()}, builder.getI32Type()),
                                               builder.getI32TensorAttr(data));

        auto bcast_ind = builder.getDenseI64ArrayAttr({idx});
        ind = builder.create<BroadcastInDimOp>(op.getLoc(), toConcatType, ind,
                                               bcast_ind);
        inds.push_back(ind);
        idx++;
      }
      idxs = builder.create<ConcatenateOp>(
          op.getLoc(), inds, builder.getI64IntegerAttr(concat_data.size() - 1));
    }

    // empty extra index into the slice
    std::vector<int64_t> update_window_dims;
    std::vector<int64_t> scatter_dims_to_operand_dims;
    std::vector<int64_t> inserted_window_dims;
    for (int i = 0; i < inTy.getShape().size(); i++) {
      scatter_dims_to_operand_dims.push_back(i);
      inserted_window_dims.push_back(i);
    }

    int64_t indexVectorDim = inTy.getShape().size();

    auto dims = ScatterDimensionNumbersAttr::get(
        builder.getContext(), update_window_dims, inserted_window_dims,
        scatter_dims_to_operand_dims, indexVectorDim);

    // auto prev = gutils->diffe(op.getOperand(), builder);

    auto red = builder.create<ScatterOp>(
        op.getLoc(), TypeRange(gutils->getShadowType(inTy)), ValueRange(zero),
        idxs, ValueRange(inDiffe), dims,
        /*indices_are_sorted*/ builder.getBoolAttr(true),
        /*unique_indices*/ builder.getBoolAttr(true));
    
    red.getUpdateComputation().push_back(new Block());
    Block &body = red.getUpdateComputation().front();
    OpBuilder bodyBuilder(orig->getContext());
    bodyBuilder.setInsertionPointToEnd(&body);

    auto TT = RankedTensorType::get({}, inTy.getElementType());
    body.addArgument(TT, op.getLoc());
    body.addArgument(TT, op.getLoc());
    /*
    auto add = bodyBuilder.create<AddOp>(op.getLoc(), body.getArgument(0),
                                         body.getArgument(1));
    bodyBuilder.create<ReturnOp>(op.getLoc(), ValueRange(add));
    */
    bodyBuilder.create<ReturnOp>(op.getLoc(), ValueRange(body.getArgument(1)));

    gutils->addToDiffe(op.getOperand(), red->getResult(0), builder);
    // gutils->setDiffe(op.getOperand(), red->getResult(0), builder);
    
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class AutoDiffReduceRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffReduceRev,
                                                       ReduceOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto op = cast<ReduceOp>(orig);
    if (!isEligibleForCompactPrint(op)) {
      orig->emitError() << "Unsupported operation in reduction rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    Operation &innerOp = op.getBody().front().front();
    
    auto inTy = op->getOperand(0).getType();
    auto zero = inTy.cast<AutoDiffTypeInterface>().createNullValue(builder,
                                                                   op.getLoc());
    auto inDiffe = gutils->diffe(op->getResult(0), builder);
    gutils->zeroDiffe(op->getResult(0), builder);
    
    if (isa<AddOp>(innerOp)) {
        if (!gutils->isConstantValue(op.getInputs()[0])) {
        Value bcast;
        
        if (op->getResult(0).getType().cast<RankedTensorType>().getShape().size() == 0)
            bcast = builder.create<BroadcastInDimOp>(op.getLoc(), gutils->getShadowType(inTy), inDiffe, builder.getDenseI64ArrayAttr({}));
        else
            bcast = builder.create<BroadcastInDimOp>(op.getLoc(), gutils->getShadowType(inTy), inDiffe, op.getDimensions());

        gutils->addToDiffe(op.getInputs()[0], bcast, builder);
        }
        if (!gutils->isConstantValue(op.getInitValues()[0])) {
        gutils->addToDiffe(op.getInitValues()[0], inDiffe, builder);
        }
        return success();
    }

    if (isa<MaxOp>(innerOp) || isa<MinOp>(innerOp)) {
    }
      
    orig->emitError() << "Unsupported operation in reduction rev autodiff(1): "
                        << *orig << "\n";
    return failure();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class AutoDiffConcatenateRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffConcatenateRev,
                                                       ConcatenateOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto op = cast<ConcatenateOp>(orig);

    auto inDiffe = gutils->diffe(op->getResult(0), builder);
    gutils->zeroDiffe(op->getResult(0), builder);

    auto dim = op.getDimension();
    for (auto &ope : op->getOpOperands()) { 
        auto op = ope.get();
        if (gutils->isConstantValue(op)) continue;
        auto inTy = gutils->getShadowType(op.getType());
        SmallVector<int64_t> start;
        SmallVector<int64_t> limit;
        SmallVector<int64_t> strides;
        SmallVector<int64_t> tys;
        auto RT = inTy.cast<RankedTensorType>();
        for (auto i=0; i<RT.getShape().size(); i++) {
            if (i == dim) {
                tys.push_back(1);
                start.push_back(ope.getOperandNumber());
                limit.push_back(ope.getOperandNumber()+1);
                strides.push_back(1);
                continue;
            }
            tys.push_back(RT.getShape()[i]);
            start.push_back(0);
            limit.push_back(RT.getShape()[i]);
            strides.push_back(1);
        }
        auto res = builder.create<SliceOp>(op.getLoc(), RankedTensorType::get(tys, RT.getElementType()), inDiffe, start, limit, strides);
        auto res2 = builder.create<ReshapeOp>(op.getLoc(), inTy, res);
        gutils->addToDiffe(op, res2, builder);
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

} // namespace

void mlir::enzyme::registerStableHLODialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, stablehlo::StablehloDialect *) {
        registerInterfaces(context);
        ReduceOp::attachInterface<AutoDiffReduceFwd<ReduceOp>>(*context);
        ReduceOp::attachInterface<AutoDiffReduceCF<ReduceOp>>(*context);
        BroadcastInDimOp::attachInterface<AutoDiffBroadcastInDimRev>(*context);
        SliceOp::attachInterface<AutoDiffSliceRev>(*context);
        ReduceOp::attachInterface<AutoDiffReduceRev>(*context);
        ConcatenateOp::attachInterface<AutoDiffConcatenateRev>(*context);
      });
}
