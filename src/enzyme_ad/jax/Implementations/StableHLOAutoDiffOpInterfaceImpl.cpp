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

static int64_t to_i64(int64_t x) { return x; }
static int64_t to_i64(llvm::APInt x) { return x.getSExtValue(); }

static mlir::DenseI64ArrayAttr getI64Attr(OpBuilder &builder,
                                          llvm::ArrayRef<int64_t> vals) {
  return builder.getDenseI64ArrayAttr(vals);
}

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

    size_t shadowCnt = 0;
    for (auto origRes : orig->getResults()) {
      if (!gutils->isConstantValue(origRes)) {
        gutils->setDiffe(origRes, shadow->getResult(shadowCnt), builder);
        shadowCnt++;
      }
    }
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

    SmallVector<int64_t> reducedDims;
    SmallVector<int64_t> iterShape;
    for (auto en : llvm::enumerate(outTy.getShape())) {
      ssize_t bcastIdx = -1;
      for (auto en2 : llvm::enumerate(bcastDims)) {
        if (en2.value() == en.index()) {
          bcastIdx = en2.index();
          break;
        }
      }
      if (bcastIdx != -1) {
        if (en.value() != inTy.getShape()[bcastIdx]) {
          reducedDims.push_back(en.index());
          assert(inTy.getShape()[bcastIdx] == 1);
        } else {
          iterShape.push_back(inTy.getShape()[bcastIdx]);
        }
        continue;
      }
      reducedDims.push_back(en.index());
    }

    auto reduceTy = RankedTensorType::get(iterShape, inTy.getElementType());
    auto bodyTy = RankedTensorType::get({}, inTy.getElementType());

    Value zero = gutils->getShadowType(bodyTy)
                     .cast<AutoDiffTypeInterface>()
                     .createNullValue(builder, op.getLoc());

    auto red = builder.create<ReduceOp>(
        op.getLoc(), TypeRange(gutils->getShadowType(reduceTy)), inDiffe, zero,
        reducedDims);
    red.getBody().push_back(new Block());
    Block &body = red.getBody().front();
    OpBuilder bodyBuilder(orig->getContext());
    bodyBuilder.setInsertionPointToEnd(&body);

    body.addArgument(bodyTy, op.getLoc());
    body.addArgument(bodyTy, op.getLoc());
    auto add = bodyBuilder.create<AddOp>(op.getLoc(), body.getArgument(0),
                                         body.getArgument(1));
    bodyBuilder.create<ReturnOp>(op.getLoc(), ValueRange(add));

    Value res = red->getResult(0);
    Type resTy = gutils->getShadowType(op.getOperand().getType());
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
    auto inDiffe = gutils->diffe(op, builder);
    gutils->zeroDiffe(op, builder);

    SmallVector<int64_t> starts;
    SmallVector<int64_t> edge_padding_high;
    SmallVector<int64_t> interior_padding;
    for (auto &&[start, limit, stride, dim] :
         llvm::zip(op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
                   inTy.getShape())) {
      starts.push_back(start);
      edge_padding_high.push_back(dim - limit);
      interior_padding.push_back(stride - 1);
    }

    auto zeroPad = RankedTensorType::get({}, inTy.getElementType())
                       .cast<AutoDiffTypeInterface>()
                       .createNullValue(builder, op.getLoc());
    auto red = builder.create<stablehlo::PadOp>(
        op.getLoc(), inDiffe, zeroPad, builder.getDenseI64ArrayAttr(starts),
        builder.getDenseI64ArrayAttr(edge_padding_high),
        builder.getDenseI64ArrayAttr(interior_padding));

    gutils->addToDiffe(op.getOperand(), red->getResult(0), builder);
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
      orig->emitError()
          << "Unsupported operation in reduction rev autodiff(1): " << *orig
          << "\n";
      return failure();
    }

    Operation &innerOp = op.getBody().front().front();

    auto inTy = op->getOperand(0).getType().cast<RankedTensorType>();
    auto zero = inTy.cast<AutoDiffTypeInterface>().createNullValue(builder,
                                                                   op.getLoc());
    auto inDiffe = gutils->diffe(op->getResult(0), builder);
    gutils->zeroDiffe(op->getResult(0), builder);

    SmallVector<int64_t> toBroadcast;
    for (auto en : llvm::enumerate(inTy.getShape())) {
      if (llvm::is_contained(op.getDimensions(), en.index())) {
        // reduced op
        continue;
      }
      toBroadcast.push_back(en.index());
    }

    if (isa<AddOp>(innerOp)) {
      if (!gutils->isConstantValue(op.getInputs()[0])) {
        Value bcast;

        bcast = builder.create<BroadcastInDimOp>(
            op.getLoc(), gutils->getShadowType(inTy), inDiffe,
            builder.getDenseI64ArrayAttr(toBroadcast));

        gutils->addToDiffe(op.getInputs()[0], bcast, builder);
      }
      if (!gutils->isConstantValue(op.getInitValues()[0])) {
        gutils->addToDiffe(op.getInitValues()[0], inDiffe, builder);
      }
      return success();
    }

    if (isa<MaxOp>(innerOp) || isa<MinOp>(innerOp)) {
      // TODO: technically we should invert the order here to pick the last
      // value (or divide by count) if multiple are the same as the result
      auto ores = gutils->getNewFromOriginal(op->getResult(0));

      if (!gutils->isConstantValue(op.getInputs()[0])) {
        auto oprev = gutils->getNewFromOriginal(op.getInputs()[0]);
        auto attr = builder.getDenseI64ArrayAttr(toBroadcast);
        auto bc = builder.create<BroadcastInDimOp>(op.getLoc(), oprev.getType(),
                                                   ores, attr);

        auto cmp = builder.create<CompareOp>(op.getLoc(), bc, oprev,
                                             ComparisonDirection::EQ);

        auto bc2 = builder.create<BroadcastInDimOp>(
            op.getLoc(), oprev.getType(), inDiffe, attr);

        auto res = builder.create<SelectOp>(op.getLoc(), cmp, bc2, zero);
        gutils->addToDiffe(op.getInputs()[0], res, builder);
      }
      if (!gutils->isConstantValue(op.getInitValues()[0])) {
        auto oprev = gutils->getNewFromOriginal(op.getInitValues()[0]);

        auto zeroI =
            inDiffe.getType().cast<AutoDiffTypeInterface>().createNullValue(
                builder, op.getLoc());

        auto cmp = builder.create<CompareOp>(op.getLoc(), ores, oprev,
                                             ComparisonDirection::EQ);

        auto res = builder.create<SelectOp>(op.getLoc(), cmp, inDiffe, zeroI);
        gutils->addToDiffe(op.getInitValues()[0], res, builder);
      }
      return success();
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
    size_t startDim = 0;
    for (auto &ope : op->getOpOperands()) {
      auto op = ope.get();
      auto inTy = gutils->getShadowType(op.getType());
      SmallVector<int64_t> start;
      SmallVector<int64_t> limit;
      SmallVector<int64_t> strides;
      SmallVector<int64_t> tys;
      auto RT = inTy.cast<RankedTensorType>();
      for (auto i = 0; i < RT.getShape().size(); i++) {
        tys.push_back(RT.getShape()[i]);
        if (i == dim) {
          start.push_back(startDim);
          limit.push_back(startDim + RT.getShape()[i]);
          startDim += RT.getShape()[i];
          strides.push_back(1);
          continue;
        }
        start.push_back(0);
        limit.push_back(RT.getShape()[i]);
        strides.push_back(1);
      }
      if (gutils->isConstantValue(op))
        continue;
      auto res = builder.create<SliceOp>(
          op.getLoc(), RankedTensorType::get(tys, RT.getElementType()), inDiffe,
          start, limit, strides);
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

struct SHLOConstantOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOConstantOpBatchInterface,
                                             ConstantOp> {

  mlir::Operation *createBatch(Operation *src, IRMapping &mapper,
                               Operation::CloneOptions options,
                               std::map<Operation *, Operation *> &opMap,
                               ArrayRef<int64_t> batchSizes) const {

    SmallVector<Type> resultTypes(src->getResultTypes().begin(),
                                  src->getResultTypes().end());
    for (auto &Ty : resultTypes) {
      auto T = cast<TensorType>(Ty);
      SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
      shape.append(T.getShape().begin(), T.getShape().end());
      Ty = T.clone(shape);
    }
    mlir::NamedAttrList attrs;
    for (auto attr : src->getAttrs()) {
      auto eattr = cast<DenseElementsAttr>(attr.getValue());
      attr.setValue(eattr.resizeSplat(cast<ShapedType>(resultTypes[0])));
      attrs.append(attr);
    }
    auto cop = mlir::Operation::create(
        src->getLoc(), src->getName(), resultTypes, {}, std::move(attrs),
        OpaqueProperties(nullptr), mlir::BlockRange(), 0);
    return cop;
  }
};

struct ADDataFlowSortOp
    : public ADDataFlowOpInterface::ExternalModel<ADDataFlowSortOp, SortOp> {

  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   mlir::OpResult v) const {
    auto srt = cast<SortOp>(op);
    return {srt.getInputs()[v.getResultNumber()]};
  }
  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<SortOp>(op);
    size_t num = v.getArgNumber() / 2;
    return {srt.getInputs()[num]};
  }
  // The return of region is a comparison, non differentiable, and non flowing
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    return {};
  }
};

struct RegionBranchCaseOp
    : public RegionBranchOpInterface::ExternalModel<RegionBranchCaseOp,
                                                    CaseOp> {

  void
  getEntrySuccessorRegions(Operation *op, ArrayRef<Attribute> operands,
                           SmallVectorImpl<RegionSuccessor> &successors) const {
    for (auto &reg : op->getRegions())
      successors.push_back(RegionSuccessor(&reg));
  }

  mlir::OperandRange getEntrySuccessorOperands(Operation *op,
                                               RegionBranchPoint bp) const {
    auto end = op->operand_end();
    return ::mlir::OperandRange(end, end);
  }

  void
  getRegionInvocationBounds(Operation *op, ArrayRef<Attribute> operands,
                            SmallVectorImpl<InvocationBounds> &bounds) const {
    bounds.append(op->getNumRegions(), InvocationBounds(/*lb=*/0, /*ub=*/1));
  }

  bool areTypesCompatible(Operation *op, Type lhs, Type rhs) const {
    return lhs == rhs;
  }

  void getSuccessorRegions(Operation *op, RegionBranchPoint point,
                           SmallVectorImpl<RegionSuccessor> &regions) const {
    // The `then` and the `else` region branch back to the parent operation.
    if (!point.isParent()) {
      regions.push_back(RegionSuccessor(op->getResults()));
      return;
    }

    for (auto &reg : op->getRegions())
      regions.push_back(RegionSuccessor(&reg));
  }
};

struct ScatterActivity
    : public ActivityOpInterface::ExternalModel<ScatterActivity, ScatterOp> {
  bool isInactive(Operation *op) const { return false; }
  bool isArgInactive(Operation *op, size_t idx) const {
    auto scat = cast<ScatterOp>(op);
    if (idx >= scat.getInputs().getBeginOperandIndex() &&
        idx < scat.getInputs().getBeginOperandIndex() + scat.getInputs().size())
      return false;
    if (idx >= scat.getUpdates().getBeginOperandIndex() &&
        idx <
            scat.getUpdates().getBeginOperandIndex() + scat.getUpdates().size())
      return false;
    return true;
  }
};

struct ADDataFlowWhileOp
    : public ADDataFlowOpInterface::ExternalModel<ADDataFlowWhileOp, WhileOp> {

  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   mlir::OpResult v) const {
    auto srt = cast<WhileOp>(op);
    auto resvals = cast<ReturnOp>(srt.getBody().back());
    return {
        srt.getOperand()[v.getResultNumber()],
        resvals.getOperands()[v.getResultNumber()],
    };
  }

  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<WhileOp>(op);
    auto resvals = cast<ReturnOp>(srt.getBody().back());
    return {
        srt.getOperand()[v.getArgNumber()],
        resvals.getOperands()[v.getArgNumber()],
    };
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *parent,
                                                 Operation *term,
                                                 Value v) const {
    auto srt = cast<WhileOp>(op);
    if (srt.getCond() == term->getParentRegion())
      return {};
    SmallVector<Value> sv;
    for (auto &&[res, arg] : llvm::zip(srt.getResults(), term->getOperands())) {
      if (arg == v)
        sv.push_back(res);
    }
    return sv;
  }
};

struct ADDataFlowReduceOp
    : public ADDataFlowOpInterface::ExternalModel<ADDataFlowReduceOp,
                                                  ReduceOp> {

  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   mlir::OpResult v) const {
    auto srt = cast<ReduceOp>(op);
    auto resvals = cast<ReturnOp>(op->getRegion(0).front().back());
    return {
        srt.getInputs()[v.getResultNumber()],
        srt.getInitValues()[v.getResultNumber()],
        resvals.getOperands()[v.getResultNumber()],
    };
  }
  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<ReduceOp>(op);
    auto resvals = cast<ReturnOp>(op->getRegion(0).front().back());
    if (v.getArgNumber() < srt.getInitValues().size())
      return {srt.getInitValues()[v.getArgNumber()],
              resvals.getOperands()[v.getArgNumber()]};
    else
      return {
          srt.getInputs()[v.getArgNumber() - srt.getInitValues().size()],
          resvals.getOperands()[v.getArgNumber() - srt.getInitValues().size()]};
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    auto srt = cast<ScatterOp>(op);
    SmallVector<Value> sv;
    for (size_t i = 0; i < srt.getInputs().size(); i++) {
      if (term->getOperands()[i] == v) {
        sv.push_back(srt.getResults()[i]);
        sv.push_back(op->getRegion()
                         .front()
                         .getArguments()[srt.getInitValues().size() + i]);
      }
    }
    return sv;
  }
};

struct ADDataFlowScatterOp
    : public ADDataFlowOpInterface::ExternalModel<ADDataFlowScatterOp,
                                                  ScatterOp> {

  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   mlir::OpResult v) const {
    auto srt = cast<ScatterOp>(op);
    auto resvals = cast<ReturnOp>(op->getRegion(0).front().back());
    return {
        srt.getInputs()[v.getResultNumber()],
        srt.getUpdates()[v.getResultNumber()],
        resvals.getOperands()[v.getResultNumber()],
    };
  }
  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<ScatterOp>(op);
    if (v.getArgNumber() < srt.getInputs().size())
      return {srt.getInputs()[v.getArgNumber()]};
    else
      return {srt.getUpdates()[v.getArgNumber() - srt.getInputs().size()]};
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    auto srt = cast<ScatterOp>(op);
    SmallVector<Value> sv;
    for (size_t i = 0; i < srt.getInputs().size(); i++) {
      if (term->getOperands()[i] == v) {
        sv.push_back(srt.getResults()[i]);
      }
    }
    return sv;
  }
};

class AutoDiffScatter
    : public AutoDiffOpInterface::ExternalModel<AutoDiffScatter, ScatterOp> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto scat = cast<ScatterOp>(op);
    SmallVector<int> inds;

    for (auto idx = scat.getInputs().getBeginOperandIndex();
         idx <
         scat.getInputs().getBeginOperandIndex() + scat.getInputs().size();
         idx++)
      inds.push_back((int)idx);

    for (auto idx = scat.getUpdates().getBeginOperandIndex();
         idx <
         scat.getUpdates().getBeginOperandIndex() + scat.getUpdates().size();
         idx++)
      inds.push_back((int)idx);

    SmallVector<Type> ResultTypes;
    SmallVector<Value> Inputs;
    SmallVector<Value> Updates;
    for (auto &&[res, inp, up] :
         llvm::zip(op->getResults(), scat.getInputs(), scat.getUpdates())) {
      ResultTypes.push_back(res.getType());
      Inputs.push_back(gutils->getNewFromOriginal(inp));
      Updates.push_back(gutils->getNewFromOriginal(up));
      if (!gutils->isConstantValue(res)) {
        ResultTypes.push_back(
            cast<AutoDiffTypeInterface>(res.getType()).getShadowType());
        if (!gutils->isConstantValue(inp)) {
          Inputs.push_back(gutils->invertPointerM(inp, builder));
        } else {
          Inputs.push_back(cast<AutoDiffTypeInterface>(inp.getType())
                               .createNullValue(builder, op->getLoc()));
        }
        if (!gutils->isConstantValue(up)) {
          Updates.push_back(gutils->invertPointerM(up, builder));
        } else {
          Updates.push_back(cast<AutoDiffTypeInterface>(up.getType())
                                .createNullValue(builder, op->getLoc()));
        }
      }
    }

    auto replacement = builder.create<ScatterOp>(
        scat.getLoc(), ResultTypes, Inputs,
        gutils->getNewFromOriginal(scat.getScatterIndices()), Updates,
        scat.getScatterDimensionNumbersAttr(), scat.getIndicesAreSortedAttr(),
        scat.getUniqueIndicesAttr());
    auto newOp = gutils->getNewFromOriginal(op);
    for (auto &&[region, replacementRegion] :
         llvm::zip(newOp->getRegions(), replacement->getRegions())) {
      replacementRegion.takeBody(region);
    }

    auto nb = gutils->getNewFromOriginal(&scat->getRegion(0).front());
    // Rewrite block arguments to match the shadowing
    size_t curidx = 0;
    for (int j = 0; j < 2; j++) {
      for (size_t i = 0; i < scat.getInputs().size(); i++) {
        auto inp = scat.getInputs()[i];
        auto up = scat.getInputs()[i];
        // primal
        curidx++;
        if (gutils->isConstantValue(inp) && gutils->isConstantValue(up)) {
          continue;
        }
        auto ba = scat->getRegion(0)
                      .front()
                      .getArguments()[i + j * scat.getInputs().size()];
        nb->insertArgument(
            curidx, cast<AutoDiffTypeInterface>(ba.getType()).getShadowType(),
            scat.getLoc());
        curidx++;
      }
    }

    // Inject the mapping for the new results into GradientUtil's shadow
    // table.
    SmallVector<Value> reps;
    size_t idx = 0;
    for (Value r : op->getResults()) {
      // TODO only if used
      reps.push_back(replacement->getResult(idx));
      idx++;
      if (!gutils->isConstantValue(r)) {
        auto inverted = gutils->invertedPointers.lookupOrNull(r);
        assert(inverted);
        gutils->invertedPointers.map(r, replacement->getResult(idx));
        inverted.replaceAllUsesWith(replacement->getResult(idx));
        gutils->erase(inverted.getDefiningOp());
        idx++;
      }
    }

    // Differentiate body.
    for (auto &origRegion : op->getRegions()) {
      for (auto &origBlock : origRegion) {
        for (Operation &o : origBlock) {
          if (failed(gutils->visitChild(&o))) {
            return failure();
          }
        }
      }
    }

    // Replace all uses of original results
    gutils->replaceOrigOpWith(op, reps);
    gutils->erase(newOp);
    gutils->originalToNewFnOps[op] = replacement;
    return success();
  }
};

class AutoDiffHLOReturn
    : public AutoDiffOpInterface::ExternalModel<AutoDiffHLOReturn, ReturnOp> {
public:
  LogicalResult createForwardModeTangent(Operation *origTerminator,
                                         OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto parentOp = origTerminator->getParentOp();

    llvm::SmallDenseSet<unsigned> operandsToShadow;
    if (isa<WhileOp>(parentOp) && origTerminator->getParentRegion() ==
                                      &cast<WhileOp>(parentOp).getCond()) {
    } else {
      assert(parentOp->getNumResults() == origTerminator->getNumOperands());
      for (auto res : parentOp->getResults()) {
        if (!gutils->isConstantValue(res))
          operandsToShadow.insert(res.getResultNumber());
      }
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(origTerminator->getNumOperands() +
                        operandsToShadow.size());
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
      if (operandsToShadow.contains(operand.getOperandNumber()))
        newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }

    // Assuming shadows following the originals are fine.
    // TODO: consider extending to have a ShadowableTerminatorOpInterface
    Operation *replTerminator = gutils->getNewFromOriginal(origTerminator);
    replTerminator->setOperands(newOperands);
    return success();
  }
};

class AutoDiffSort
    : public AutoDiffOpInterface::ExternalModel<AutoDiffSort, SortOp> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {

    auto sop = cast<SortOp>(op);

    // TODO: we may need to record, for every successor, which of its inputs
    // need a shadow to recreate the body correctly.
    llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
    llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

    for (auto res : op->getResults())
      if (!gutils->isConstantValue(res)) {
        operandPositionsToShadow.insert(res.getResultNumber());
        resultPositionsToShadow.insert(res.getResultNumber());
      }

    return mlir::enzyme::detail::controlFlowForwardHandler(
        op, builder, gutils, operandPositionsToShadow, resultPositionsToShadow);
  }
};

} // namespace

void mlir::enzyme::registerStableHLODialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, stablehlo::StablehloDialect *) {
        registerInterfaces(context);

        // SortOp::attachInterface<AutoDiffSort>(*context);

        WhileOp::attachInterface<ADDataFlowWhileOp>(*context);
        SortOp::attachInterface<ADDataFlowSortOp>(*context);
        ScatterOp::attachInterface<ADDataFlowScatterOp>(*context);
        ReduceOp::attachInterface<ADDataFlowReduceOp>(*context);

        CaseOp::attachInterface<RegionBranchCaseOp>(*context);

        ScatterOp::attachInterface<ScatterActivity>(*context);
        ScatterOp::attachInterface<AutoDiffScatter>(*context);

        ReturnOp::attachInterface<AutoDiffHLOReturn>(*context);

        ReduceOp::attachInterface<AutoDiffReduceFwd<ReduceOp>>(*context);
        ReduceOp::attachInterface<AutoDiffReduceCF<ReduceOp>>(*context);
        BroadcastInDimOp::attachInterface<AutoDiffBroadcastInDimRev>(*context);
        SliceOp::attachInterface<AutoDiffSliceRev>(*context);
        ReduceOp::attachInterface<AutoDiffReduceRev>(*context);
        ConcatenateOp::attachInterface<AutoDiffConcatenateRev>(*context);
        ConstantOp::attachInterface<SHLOConstantOpBatchInterface>(*context);
      });
}
