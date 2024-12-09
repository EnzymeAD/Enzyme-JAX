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

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
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

static int64_t getI64Value(llvm::ArrayRef<int64_t> attr, size_t pos) {
  return attr[pos];
}

static mlir::DenseBoolArrayAttr getBoolAttr(OpBuilder &builder,
                                            llvm::ArrayRef<bool> vals) {
  return builder.getDenseBoolArrayAttr(vals);
}

static llvm::ArrayRef<bool> getBoolIter(llvm::ArrayRef<bool> vals) {
  return vals;
}

static Value makeI64Constant(Location loc, OpBuilder &builder, int64_t val) {
  auto Ty = builder.getI64Type();
  auto unrankedTensorType = RankedTensorType::get({}, Ty);
  return builder
      .create<ConstantOp>(loc, unrankedTensorType,
                          SplatElementsAttr::get(
                              unrankedTensorType,
                              ArrayRef<Attribute>(IntegerAttr::get(Ty, val))))
      .getResult();
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

class AutoDiffIfFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffIfFwd, IfOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
    llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

    for (auto res : orig->getOpResults()) {
      if (!gutils->isConstantValue(res))
        resultPositionsToShadow.insert(res.getResultNumber());
    }

    return mlir::enzyme::detail::controlFlowForwardHandler(
        orig, builder, gutils, operandPositionsToShadow,
        resultPositionsToShadow);
  }
};

class AutoDiffIfCF
    : public ControlFlowAutoDiffOpInterface::ExternalModel<AutoDiffIfCF, IfOp> {
public:
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    return builder.create<IfOp>(original->getLoc(), rettys, remappedOperands,
                                original->getAttrs());
  }
};

class AutoDiffIfRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffIfRev, IfOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto revOp = builder.create<IfOp>(orig->getLoc(), ArrayRef<mlir::Type>{},
                                      gutils->popCache(caches[0], builder),
                                      orig->getAttrs());

    bool valid = true;
    for (auto &&[origReg, newReg] :
         llvm::zip_equal(orig->getRegions(), revOp->getRegions())) {
      Block *oBB = &origReg.front();

      newReg.push_back(new Block());
      Block *reverseBB = &newReg.front();

      OpBuilder revBuilder(reverseBB, reverseBB->end());
      auto term = oBB->getTerminator();

      for (auto &&[ret, op] :
           llvm::zip_equal(orig->getResults(), term->getOperands())) {
        if (gutils->isConstantValue(ret))
          continue;
        if (gutils->isConstantValue(op))
          continue;

        gutils->addToDiffe(op, gutils->diffe(ret, revBuilder), revBuilder);
      }

      auto first = oBB->rbegin();
      first++; // skip terminator

      auto last = oBB->rend();

      for (auto it = first; it != last; ++it) {
        Operation *op = &*it;
        valid &= gutils->Logic.visitChild(op, revBuilder, gutils).succeeded();
      }

      revBuilder.create<stablehlo::ReturnOp>(orig->getLoc(), ArrayRef<Value>{});
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    SmallVector<Value> caches;

    auto op = cast<IfOp>(orig);

    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);

    Value predCache = gutils->initAndPushCache(
        gutils->getNewFromOriginal(op.getPred()), cacheBuilder);
    caches.push_back(predCache);

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

class AutoDiffWhileFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffWhileFwd, WhileOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<WhileOp>(orig);

    llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
    llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

    for (auto &&[res, arg] :
         llvm::zip(op->getResults(), op.getBody().front().getArguments()))
      if (!gutils->isConstantValue(res) || !gutils->isConstantValue(arg)) {
        operandPositionsToShadow.insert(res.getResultNumber());
        resultPositionsToShadow.insert(res.getResultNumber());
      }

    auto res = mlir::enzyme::detail::controlFlowForwardHandler(
        op, builder, gutils, operandPositionsToShadow, resultPositionsToShadow);

    // Rewrite block arguments to match the shadowing
    for (auto reg : {&op.getCond(), &op.getBody()}) {
      auto nb = gutils->getNewFromOriginal(&reg->front());
      size_t curidx = 0;
      for (auto arg : reg->front().getArguments()) {
        curidx++;
        auto idx = arg.getArgNumber();
        if (resultPositionsToShadow.count(idx)) {
          if (gutils->isConstantValue(arg)) {
            nb->insertArgument(
                curidx,
                cast<AutoDiffTypeInterface>(arg.getType()).getShadowType(),
                op.getLoc());
          }
          curidx++;
        }
      }
    }

    return res;
  }
};

class AutoDiffWhileRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffWhileRev,
                                                       WhileOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    // While op has the same number of results and operands.
    // if the while is not executed (i.e. the condition is false on the first
    // evaluation), then the arguments are returned. This means that we need to
    // pass differentials for all non-constant results or non-constants
    // operands.
    SmallVector<bool> operandsActive(orig->getNumOperands(), false);
    for (int i = 0; i < operandsActive.size(); ++i) {
      operandsActive[i] = !gutils->isConstantValue(orig->getOperand(i)) ||
                          !gutils->isConstantValue(orig->getResult(i));
    }

    // The reverse of the while loop is a for loop where the number
    // of iterations is either known or cached from the augmented primal.
    Value numIters = gutils->popCache(caches[0], builder);

    auto unrankedTensorType = RankedTensorType::get({}, builder.getI64Type());
    auto iterVar =
        builder
            .create<ConstantOp>(
                orig->getLoc(), unrankedTensorType,
                SplatElementsAttr::get(unrankedTensorType,
                                       ArrayRef<Attribute>(IntegerAttr::get(
                                           builder.getI64Type(), 0))))
            .getResult();

    SmallVector<Value> operands;
    operands.reserve(orig->getNumResults() + 1);
    operands.push_back(iterVar);

    for (auto [active, res] : llvm::zip(operandsActive, orig->getResults())) {
      if (active) {
        operands.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    auto revWhile = builder.create<WhileOp>(
        orig->getLoc(), ValueRange(operands).getTypes(), operands);
    auto &condReg = revWhile.getCond();
    auto &bodyReg = revWhile.getBody();

    auto cond = new Block();
    auto body = new Block();

    condReg.push_back(cond);
    bodyReg.push_back(body);

    {
      for (auto operand : operands) {
        cond->addArgument(operand.getType(), orig->getLoc());
      }
      auto condIterVar = cond->getArgument(0);

      OpBuilder condBuilder(cond, cond->end());
      condBuilder.create<ReturnOp>(
          orig->getLoc(),
          ValueRange(condBuilder
                         .create<CompareOp>(orig->getLoc(), condIterVar,
                                            numIters, ComparisonDirection::LT)
                         .getResult()));
    }

    bool valid = true;
    {
      for (auto operand : operands) {
        body->addArgument(operand.getType(), orig->getLoc());
      }
      OpBuilder bodyBuilder(body, body->end());
      auto one = bodyBuilder.create<ConstantOp>(
          orig->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(unrankedTensorType,
                                 ArrayRef<Attribute>(IntegerAttr::get(
                                     bodyBuilder.getI64Type(), 1))));
      Value bodyIterVar =
          bodyBuilder.create<AddOp>(orig->getLoc(), body->getArgument(0), one);

      Block *oBB = &orig->getRegion(1).front();
      auto term = oBB->getTerminator();

      // All values defined in the body should have no use outside this block
      // therefore we can set their diffe to zero upon entering the reverse
      // block to simplify the work of the remove-unnecessary-enzyme-ops pass.
      for (auto operand : oBB->getArguments()) {
        if (!gutils->isConstantValue(operand)) {
          gutils->zeroDiffe(operand, bodyBuilder);
        }
      }

      for (auto &it : oBB->getOperations()) {
        for (auto res : it.getResults()) {
          if (!gutils->isConstantValue(res)) {
            gutils->zeroDiffe(res, bodyBuilder);
          }
        }
      }

      int revIdx = 1;
      for (auto &&[active, operand] :
           llvm::zip(operandsActive, term->getOperands())) {
        if (active) {
          // Set diffe here, not add because it should not accumulate across
          // iterations. Instead the new gradient for this operand is passed in
          // the return of the reverse while body.
          gutils->setDiffe(operand, body->getArgument(revIdx), bodyBuilder);
          revIdx++;
        }
      }

      auto first = oBB->rbegin();
      first++; // skip terminator

      auto last = oBB->rend();

      for (auto it = first; it != last; ++it) {
        Operation *op = &*it;
        valid &= gutils->Logic.visitChild(op, bodyBuilder, gutils).succeeded();
      }

      SmallVector<Value> newResults;
      newResults.reserve(operands.size());
      newResults.push_back(bodyIterVar);

      for (auto &&[active, arg] :
           llvm::zip(operandsActive, oBB->getArguments())) {
        if (active) {
          newResults.push_back(gutils->diffe(arg, bodyBuilder));
          if (!gutils->isConstantValue(arg))
            gutils->zeroDiffe(arg, bodyBuilder);
        }
      }

      bodyBuilder.create<ReturnOp>(orig->getLoc(), newResults);
    }

    int revIdx = 1;
    for (auto &&[active, arg] :
         llvm::zip(operandsActive, orig->getOperands())) {
      if (active) {
        if (!gutils->isConstantValue(arg))
          gutils->addToDiffe(arg, revWhile->getResult(revIdx), builder);
        revIdx++;
      }
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    // The primal is augmented to store the number of iterations
    auto newWhile = cast<WhileOp>(gutils->getNewFromOriginal(orig));
    OpBuilder revBuilder(newWhile);

    Value numIters;

    WhileLoopInfo info(newWhile);
    if (info.computeInfo().succeeded()) {
      numIters = info.getNumIters(revBuilder);
    }

    if (!numIters) {
      auto cond = &newWhile.getCond().front();
      auto body = &newWhile.getBody().front();

      auto unrankedTensorType =
          RankedTensorType::get({}, revBuilder.getI64Type());
      auto numItersInit =
          revBuilder
              .create<ConstantOp>(
                  orig->getLoc(), unrankedTensorType,
                  SplatElementsAttr::get(unrankedTensorType,
                                         ArrayRef<Attribute>(IntegerAttr::get(
                                             revBuilder.getI64Type(), 0))))
              .getResult();

      newWhile->insertOperands(newWhile->getNumOperands(),
                               ValueRange(numItersInit));
      cond->addArgument(numItersInit.getType(), orig->getLoc());
      Value numItersInBlock =
          body->addArgument(numItersInit.getType(), orig->getLoc());

      OpBuilder inBodyBuilder(body, body->begin());
      auto one = inBodyBuilder.create<ConstantOp>(
          orig->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(unrankedTensorType,
                                 ArrayRef<Attribute>(IntegerAttr::get(
                                     revBuilder.getI64Type(), 1))));
      numItersInBlock = inBodyBuilder.create<AddOp>(
          orig->getLoc(), numItersInBlock, one.getResult());
      auto term = body->getTerminator();
      term->insertOperands(term->getNumOperands(), ValueRange(numItersInBlock));

      SmallVector<Type> resultTypes(newWhile->getResultTypes().begin(),
                                    newWhile->getResultTypes().end());
      resultTypes.push_back(numItersInit.getType());

      auto newnewWhile = revBuilder.create<WhileOp>(orig->getLoc(), resultTypes,
                                                    newWhile->getOperands());
      newnewWhile.getCond().takeBody(newWhile.getCond());
      newnewWhile.getBody().takeBody(newWhile.getBody());

      SmallVector<Value> newResults(newnewWhile->getResults().begin(),
                                    --newnewWhile->getResults().end());

      gutils->replaceOrigOpWith(orig, newResults);
      gutils->erase(newWhile);
      gutils->originalToNewFnOps[orig] = newnewWhile;

      Value inductionOut =
          newnewWhile->getResult(newnewWhile->getNumResults() - 1);
      numIters = inductionOut;
      newWhile = newnewWhile;
    }

    revBuilder.setInsertionPointAfter(newWhile);
    Value numItersCache = gutils->initAndPushCache(numIters, revBuilder);

    return {numItersCache};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
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

class AutoDiffDynamicSliceUpdateRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffDynamicSliceUpdateRev, DynamicUpdateSliceOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<DynamicUpdateSliceOp>(orig);
    auto operand = op.getOperand();
    auto update = op.getUpdate();

    SmallVector<Value> startIndices;

    Value diffe;
    if (!gutils->isConstantValue(operand)) {

      for (auto cache : caches) {
        startIndices.push_back(gutils->popCache(cache, builder));
      }

      diffe = gutils->diffe(orig->getResult(0), builder);
      auto operandDiffe = builder.create<DynamicUpdateSliceOp>(
          orig->getLoc(), diffe,
          cast<AutoDiffTypeInterface>(update.getType())
              .createNullValue(builder, orig->getLoc()),
          startIndices);

      gutils->addToDiffe(operand, operandDiffe, builder);
    }

    if (!gutils->isConstantValue(update)) {
      if (startIndices.size() != caches.size()) {
        for (auto cache : caches) {
          startIndices.push_back(gutils->popCache(cache, builder));
        }
      }

      if (!diffe)
        diffe = gutils->diffe(orig->getResult(0), builder);

      auto sliceSizes = builder.getDenseI64ArrayAttr(
          cast<TensorType>(update.getType()).getShape());
      auto updateDiffe = builder.create<DynamicSliceOp>(
          orig->getLoc(), diffe, startIndices, sliceSizes);

      gutils->addToDiffe(update, updateDiffe, builder);
    }

    if (diffe)
      gutils->zeroDiffe(op, builder);

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<DynamicUpdateSliceOp>(orig);

    if (gutils->isConstantValue(op.getOperand()) &&
        gutils->isConstantValue(op.getUpdate()))
      return {};

    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);

    auto startIndices = op.getStartIndices();

    SmallVector<Value> caches;
    caches.reserve(startIndices.size());

    for (auto startIndex : startIndices) {
      Value predCache = gutils->initAndPushCache(
          gutils->getNewFromOriginal(startIndex), cacheBuilder);
      caches.push_back(predCache);
    }

    return caches;
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

// Collects all references in op that are defined either in ref or in a an
// ancestor of ref.
static void getAllReferences(SmallVector<Value> &refs, Operation *op,
                             Region *ref) {
  for (auto operand : op->getOperands()) {
    if (operand.getParentRegion()->isAncestor(ref))
      refs.push_back(operand);
  }

  for (auto &reg : op->getRegions()) {
    for (auto &childOp : reg.getOps()) {
      getAllReferences(refs, &childOp, ref);
    }
  }
}

static mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                        llvm::ArrayRef<int64_t> batchSizes) {
  auto T = cast<TensorType>(Ty);
  SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
  shape.append(T.getShape().begin(), T.getShape().end());
  auto T2 = T.clone(shape);
  return T2;
}

// TODO: make public in Enzyme MLIR?
// this is essentially
// https://github.com/EnzymeAD/Enzyme/blob/342057e3a3e657a33da8295c99acdcd20b0375f4/enzyme/Enzyme/MLIR/Passes/EnzymeBatchPass.cpp#L58-L100
static void batchCloneBlock(Block *srcBlock, Block *destBlock,
                            IRMapping &mapper, ArrayRef<int64_t> batchSizes) {
  for (auto arg : srcBlock->getArguments()) {
    auto batched = destBlock->addArgument(
        applyBatchSizes(arg.getType(), batchSizes), arg.getLoc());
    mapper.map(arg, batched);
  }

  OpBuilder builder(destBlock, destBlock->end());
  for (auto &src : srcBlock->getOperations()) {
    if (auto ifaceOp = dyn_cast<BatchOpInterface>(&src)) {
      auto res = ifaceOp.createBatch(builder, mapper, batchSizes);
      if (res.succeeded())
        continue;
    }

    SmallVector<Value, 8> operands;
    SmallVector<Block *, 2> successors;

    // Remap the operands.
    operands.reserve(src.getNumOperands());
    for (auto opValue : src.getOperands())
      operands.push_back(mapper.lookup(opValue));

    // Remap the successors.
    successors.reserve(src.getNumSuccessors());
    for (Block *successor : src.getSuccessors())
      successors.push_back(mapper.lookup(successor));

    SmallVector<Type> resultTypes(src.getResultTypes().begin(),
                                  src.getResultTypes().end());
    for (auto &Ty : resultTypes) {
      Ty = applyBatchSizes(Ty, batchSizes);
    }

    Operation *newOp = Operation::create(
        src.getLoc(), src.getName(), resultTypes, operands, src.getAttrs(),
        OpaqueProperties(nullptr), successors, src.getNumRegions());

    // // Clone the regions.
    // for (auto &&[oldReg, newReg] :
    //      llvm::zip(src.getRegions(), newOp->getRegions())) {
    //   batchCloneRegion(&oldReg, &newReg, mapper, batchSizes);
    // }

    // Remember the mapping of any results.
    for (unsigned i = 0, e = src.getNumResults(); i != e; ++i)
      mapper.map(src.getResult(i), newOp->getResult(i));

    builder.insert(newOp);
  }
}

// For some ops with nested regions, identify if we can batch the inner regions
// instead
static LogicalResult tryToBatchInner(Operation *src, OpBuilder &builder,
                                     IRMapping &mapper,
                                     ArrayRef<int64_t> batchSizes) {
  if (auto ifOp = dyn_cast<IfOp>(src)) {
    auto predBroadcast =
        mapper.lookup(ifOp.getPred()).getDefiningOp<BroadcastInDimOp>();
    if (predBroadcast && predBroadcast.isSimpleBroadcast() &&
        predBroadcast.getBroadcastDimensions().size() == batchSizes.size()) {
      // %pred = broadcast_in_dim %0
      // if %0 {} {}
      SmallVector<Type> results;
      results.reserve(src->getNumResults());
      for (auto resTy : src->getResultTypes()) {
        results.push_back(applyBatchSizes(resTy, batchSizes));
      }
      auto newIf = builder.create<IfOp>(src->getLoc(), results,
                                        predBroadcast.getOperand());
      newIf.getTrueBranch().push_back(new Block());
      newIf.getFalseBranch().push_back(new Block());

      batchCloneBlock(&ifOp.getTrueBranch().front(),
                      &newIf.getTrueBranch().front(), mapper, batchSizes);
      batchCloneBlock(&ifOp.getFalseBranch().front(),
                      &newIf.getFalseBranch().front(), mapper, batchSizes);

      for (auto &&[oldRes, newRes] :
           llvm::zip(ifOp->getResults(), newIf->getResults())) {
        mapper.map(oldRes, newRes);
      }

      return success();
    }

    auto iszero = matchPattern(ifOp.getPred(), m_Zero());
    auto isone = matchPattern(ifOp.getPred(), m_One());

    if (!iszero && !isone)
      return failure();

    auto &reg = isone ? ifOp.getTrueBranch() : ifOp.getFalseBranch();

    assert(reg.hasOneBlock());  // stablehlo.if only allows 1 or 0 block in the
    auto *block = &reg.front(); // regions

    batchCloneBlock(block, builder.getInsertionBlock(), mapper, batchSizes);
    auto term = builder.getInsertionBlock()->getTerminator();

    for (auto &&[result, operand] :
         llvm::zip(src->getResults(), term->getOperands())) {
      mapper.map(result, operand);
    }

    term->erase();

    return success();
  }

  return failure();
}

template <typename OpTy>
struct SHLOGenericBatchOpInterface
    : public BatchOpInterface::ExternalModel<SHLOGenericBatchOpInterface<OpTy>,
                                             OpTy> {
public:
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    if (tryToBatchInner(src, builder, mapper, batchSizes).succeeded())
      return success();

    SmallVector<Value> operands;
    operands.reserve(src->getNumOperands());

    getAllReferences(operands, src, src->getParentRegion());

    SmallVector<Value> whileOperands;
    whileOperands.reserve(src->getNumResults() + 1);
    whileOperands.push_back(makeI64Constant(src->getLoc(), builder, 0));

    for (auto res : src->getResults()) {
      auto Ty = cast<TensorType>(res.getType());
      SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
      shape.append(Ty.getShape().begin(), Ty.getShape().end());
      auto T2 = cast<AutoDiffTypeInterface>(Ty.clone(shape));
      auto defaultValue = T2.createNullValue(builder, src->getLoc());
      mapper.map(res, defaultValue);
      whileOperands.push_back(defaultValue);
    }

    auto ndims = batchSizes.size();

    SmallVector<int64_t> batchStrides;
    batchStrides.reserve(ndims);
    SmallVector<Value> startIndices;
    startIndices.reserve(ndims);

    int64_t N = 1;
    for (auto batchSize : batchSizes) {
      batchStrides.push_back(N);
      N *= batchSize;
    }

    auto whileOp = builder.create<WhileOp>(src->getLoc(), whileOperands);

    auto whileCond = new Block();
    auto whileBody = new Block();

    whileOp.getCond().push_back(whileCond);
    whileOp.getBody().push_back(whileBody);

    {
      OpBuilder condBuilder(whileCond, whileCond->end());

      for (auto operand : whileOperands) {
        whileCond->addArgument(operand.getType(), src->getLoc());
      }

      condBuilder.create<ReturnOp>(
          src->getLoc(), ValueRange(condBuilder.create<CompareOp>(
                             src->getLoc(), whileCond->getArgument(0),
                             makeI64Constant(src->getLoc(), condBuilder, N),
                             ComparisonDirection::LT)));
    }

    {
      OpBuilder bodyBuilder(whileBody, whileBody->end());

      for (auto operand : whileOperands) {
        whileBody->addArgument(operand.getType(), src->getLoc());
      }

      SmallVector<Value> whileBodyOutputs;
      whileBodyOutputs.reserve(whileBody->getNumArguments());

      whileBodyOutputs.push_back(bodyBuilder.create<AddOp>(
          src->getLoc(), whileBody->getArgument(0),
          makeI64Constant(src->getLoc(), bodyBuilder, 1)));

      for (int d = 0; d < ndims; ++d) {
        // auto idx = (i / batchStrides[d]) % batchSizes[d];
        auto idx = bodyBuilder.create<RemOp>(
            src->getLoc(),
            bodyBuilder.create<DivOp>(
                src->getLoc(), whileBody->getArgument(0),
                makeI64Constant(src->getLoc(), bodyBuilder, batchStrides[d])),
            makeI64Constant(src->getLoc(), bodyBuilder, batchSizes[d]));

        startIndices.push_back(idx);
      }

      IRMapping origToUnbatch;
      for (auto operand : operands) {
        auto batched = mapper.lookup(operand);

        auto Ty = cast<TensorType>(operand.getType());
        SmallVector<int64_t> shape(ndims, 1);
        shape.append(Ty.getShape().begin(), Ty.getShape().end());
        auto sliceTy = Ty.clone(shape);

        auto sliceOp = bodyBuilder.create<DynamicSliceOp>(
            src->getLoc(), sliceTy, batched, startIndices, shape);

        auto reshapeOp = bodyBuilder.create<ReshapeOp>(
            src->getLoc(), operand.getType(), sliceOp->getResult(0));

        origToUnbatch.map(operand, reshapeOp->getResult(0));
      }

      auto newOp = bodyBuilder.clone(*src, origToUnbatch);

      for (auto &&[idx, origRes, newRes] :
           llvm::enumerate(src->getResults(), newOp->getResults())) {
        auto batched = whileBody->getArgument(idx + 1);

        auto Ty = cast<TensorType>(newRes.getType());
        SmallVector<int64_t> shape(ndims, 1);
        shape.append(Ty.getShape().begin(), Ty.getShape().end());
        auto reshapeTy = Ty.clone(shape);

        auto reshapeOp =
            bodyBuilder.create<ReshapeOp>(src->getLoc(), reshapeTy, newRes);

        auto update = bodyBuilder.create<DynamicUpdateSliceOp>(
            src->getLoc(), batched, reshapeOp, startIndices);

        whileBodyOutputs.push_back(update);
      }

      bodyBuilder.create<ReturnOp>(src->getLoc(), whileBodyOutputs);
    }

    for (auto oldRes : src->getOpResults()) {
      mapper.map(oldRes, whileOp->getResult(oldRes.getResultNumber() + 1));
    }

    return success();
  }
};

struct SHLOConstantOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOConstantOpBatchInterface,
                                             ConstantOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
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
    builder.insert(cop);
    mapper.map(src->getResult(0), cop->getResult(0));
    return success();
  }
};

struct SHLOTransposeOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOTransposeOpBatchInterface,
                                             TransposeOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {

    SmallVector<Type> resultTypes(src->getResultTypes().begin(),
                                  src->getResultTypes().end());
    for (auto &Ty : resultTypes) {
      auto T = cast<TensorType>(Ty);
      SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
      shape.append(T.getShape().begin(), T.getShape().end());
      Ty = T.clone(shape);
    }

    // Remap the operands
    SmallVector<Value, 8> operands;
    operands.reserve(src->getNumOperands());
    for (auto opValue : src->getOperands())
      operands.push_back(mapper.lookup(opValue));

    mlir::NamedAttrList attrs;
    for (auto attr : src->getAttrs()) {
      auto eattr = cast<DenseI64ArrayAttr>(attr.getValue());
      SmallVector<int64_t> shape;
      for (size_t i = 0; i < batchSizes.size(); i++)
        shape.push_back(i);
      for (auto val : eattr.asArrayRef())
        shape.push_back(val + batchSizes.size());
      attr.setValue(DenseI64ArrayAttr::get(src->getContext(), shape));
      attrs.append(attr);
    }
    auto cop = mlir::Operation::create(
        src->getLoc(), src->getName(), resultTypes, operands, std::move(attrs),
        OpaqueProperties(nullptr), mlir::BlockRange(), 0);
    builder.insert(cop);
    mapper.map(src->getResult(0), cop->getResult(0));
    return success();
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
    auto resvals = cast<ReturnOp>(srt.getBody().front().back());
    return {
        srt.getOperand()[v.getResultNumber()],
        resvals.getOperands()[v.getResultNumber()],
    };
  }

  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<WhileOp>(op);
    auto resvals = cast<ReturnOp>(srt.getBody().front().back());
    return {
        srt.getOperand()[v.getArgNumber()],
        resvals.getOperands()[v.getArgNumber()],
    };
  }
  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value v) const {
    auto srt = cast<WhileOp>(op);
    if (&srt.getCond() == term->getParentRegion())
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
    auto srt = cast<ReduceOp>(op);
    SmallVector<Value> sv;
    for (size_t i = 0; i < srt.getInputs().size(); i++) {
      if (term->getOperands()[i] == v) {
        sv.push_back(srt.getResults()[i]);
        sv.push_back(op->getRegion(0)
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
        resvals.getOperands()[v.getResultNumber()],
    };
  }
  SmallVector<Value>
  getPotentialIncomingValuesArg(Operation *op, mlir::BlockArgument v) const {
    auto srt = cast<ScatterOp>(op);
    auto resvals = cast<ReturnOp>(op->getRegion(0).front().back());
    if (v.getArgNumber() < srt.getInputs().size())
      return {resvals.getOperands()[v.getArgNumber()],
              srt.getInputs()[v.getArgNumber()]};
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
        auto up = scat.getUpdates()[i];
        // primal
        curidx++;
        if (gutils->isConstantValue(inp) && gutils->isConstantValue(up)) {
          continue;
        }
        auto ba = scat->getRegion(0)
                      .front()
                      .getArguments()[i + j * scat.getInputs().size()];
        if (gutils->isConstantValue(ba)) {
          nb->insertArgument(
              curidx, cast<AutoDiffTypeInterface>(ba.getType()).getShadowType(),
              scat.getLoc());
        }
        // shadow
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

    llvm::SmallDenseSet<unsigned> operandsToPrimal;
    llvm::SmallDenseSet<unsigned> operandsToShadow;
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      operandsToPrimal.insert(operand.getOperandNumber());
    }
    if (auto wop = dyn_cast<WhileOp>(parentOp)) {
      if (origTerminator->getParentRegion() ==
          &cast<WhileOp>(parentOp).getCond()) {
      } else {
        llvm::SmallDenseSet<unsigned> resultPositionsToShadow;

        for (auto &&[res, arg] :
             llvm::zip(wop.getResults(), wop.getBody().front().getArguments()))
          if (!gutils->isConstantValue(res) || !gutils->isConstantValue(arg)) {
            operandsToShadow.insert(res.getResultNumber());
          }
      }
    } else if (isa<FunctionOpInterface>(parentOp)) {
      operandsToPrimal.clear();
      for (OpOperand &operand : origTerminator->getOpOperands()) {
        auto idx = operand.getOperandNumber();
        if (gutils->returnPrimals[idx])
          operandsToPrimal.insert(idx);
        if (gutils->returnShadows[idx])
          operandsToShadow.insert(idx);
      }
    } else {
      assert(parentOp->getNumResults() == origTerminator->getNumOperands());
      for (auto res : parentOp->getResults()) {
        if (!gutils->isConstantValue(res))
          operandsToShadow.insert(res.getResultNumber());
      }
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(operandsToPrimal.size() + operandsToShadow.size());
    for (OpOperand &operand : origTerminator->getOpOperands()) {
      if (operandsToPrimal.contains(operand.getOperandNumber()))
        newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
      if (operandsToShadow.contains(operand.getOperandNumber())) {
        if (!gutils->isConstantValue(operand.get())) {
          newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
        } else {
          Type retTy = operand.get()
                           .getType()
                           .cast<AutoDiffTypeInterface>()
                           .getShadowType();
          newOperands.push_back(
              retTy.cast<AutoDiffTypeInterface>().createNullValue(
                  builder, origTerminator->getLoc()));
        }
      }
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
  registry.addExtension(+[](MLIRContext *context,
                            stablehlo::StablehloDialect *) {
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
    IfOp::attachInterface<AutoDiffIfRev>(*context);
    IfOp::attachInterface<AutoDiffIfFwd>(*context);
    IfOp::attachInterface<AutoDiffIfCF>(*context);

    WhileOp::attachInterface<AutoDiffWhileFwd>(*context);
    WhileOp::attachInterface<AutoDiffWhileRev>(*context);
    ReduceOp::attachInterface<AutoDiffReduceCF<ReduceOp>>(*context);
    WhileOp::attachInterface<AutoDiffReduceCF<WhileOp>>(*context);
    BroadcastInDimOp::attachInterface<AutoDiffBroadcastInDimRev>(*context);
    SliceOp::attachInterface<AutoDiffSliceRev>(*context);
    DynamicUpdateSliceOp::attachInterface<AutoDiffDynamicSliceUpdateRev>(
        *context);
    ReduceOp::attachInterface<AutoDiffReduceRev>(*context);
    ConcatenateOp::attachInterface<AutoDiffConcatenateRev>(*context);

    ConstantOp::attachInterface<SHLOConstantOpBatchInterface>(*context);
    TransposeOp::attachInterface<SHLOTransposeOpBatchInterface>(*context);
    IfOp::attachInterface<SHLOGenericBatchOpInterface<IfOp>>(*context);
    WhileOp::attachInterface<SHLOGenericBatchOpInterface<WhileOp>>(*context);

    ReverseOp::attachInterface<SHLOGenericBatchOpInterface<ReverseOp>>(
        *context); // TODO: simpler version with newly named dims
    ScatterOp::attachInterface<SHLOGenericBatchOpInterface<ScatterOp>>(
        *context); // TODO: simpler version with newly named dims
    ConvolutionOp::attachInterface<SHLOGenericBatchOpInterface<ConvolutionOp>>(
        *context); // TODO: simpler version with newly named dims
  });
}
