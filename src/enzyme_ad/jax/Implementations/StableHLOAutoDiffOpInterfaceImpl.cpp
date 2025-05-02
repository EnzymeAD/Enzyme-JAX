//===- StableHLOAutoDiffOpInterfaceImpl.cpp - Interface external model ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the MLIR stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "Enzyme/MLIR/Passes/RemovalUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include <cstdint>

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
static inline Operation *createAddRegion(Operation *op) {
  mlir::OpBuilder builder(op->getContext());
  mlir::Block *block = new Block();
  op->getRegion(0).push_back(block);
  auto elemType = cast<ShapedType>(op->getResult(0).getType()).getElementType();
  auto tensorType = RankedTensorType::get({}, elemType);
  block->addArguments({tensorType, tensorType}, {op->getLoc(), op->getLoc()});
  builder.setInsertionPointToEnd(block);
  builder.create<mlir::stablehlo::ReturnOp>(
      op->getLoc(),
      builder
          .create<mlir::stablehlo::AddOp>(op->getLoc(), block->getArgument(0),
                                          block->getArgument(1))
          ->getResult(0));
  return op;
}

Operation *cloneWithNewResultTypes(Operation *op, OpBuilder &builder,
                                   IRMapping &mapper,
                                   TypeRange newResultTypes) {
  OperationState state(op->getLoc(), op->getName());
  state.addTypes(newResultTypes);

  SmallVector<Value, 8> operands;
  operands.reserve(op->getNumOperands());
  for (auto opValue : op->getOperands())
    operands.push_back(mapper.lookup(opValue));
  state.addOperands(operands);

  state.addAttributes(op->getAttrs());

  for (Region &region : op->getRegions()) {
    Region *newRegion = state.addRegion();
    region.cloneInto(newRegion, mapper);
  }

  return builder.create(state);
}

namespace {

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

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
      cast<ShapedType>(op.getInputs()[0].getType()).getElementType();
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
          auto toret = cast<AutoDiffTypeInterface>(retTy).createNullValue(
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

    Operation *shadow;
    SmallVector<Type> newResultTypes;
    if (gutils->width > 1) { // batched forward mode
      SmallVector<Type> newResultTypes;
      for (auto result : orig->getResults()) {
        auto oldType = dyn_cast<RankedTensorType>(result.getType());
        if (!oldType || !oldType.hasStaticShape()) {
          orig->emitError("Unsupported result type for batched reduce\n");
          return failure();
        }

        SmallVector<int64_t> newShape;
        newShape.push_back(gutils->width); // prepend batch dim
        newShape.append(oldType.getShape().begin(), oldType.getShape().end());

        newResultTypes.push_back(
            RankedTensorType::get(newShape, oldType.getElementType()));
      }

      shadow = cloneWithNewResultTypes(orig, builder, map, newResultTypes);
    } else {
      shadow = builder.clone(*orig, map);
    }

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

    if (gutils->width > 1) { // batched forward mode
      auto dimsAttr =
          dyn_cast<DenseI64ArrayAttr>(shadow->getAttr("dimensions"));
      if (!dimsAttr) {
        shadow->emitError("Missing 'dimensions' attribute on ReduceOp");
        return failure();
      }

      SmallVector<int64_t> dims;
      for (int64_t d : dimsAttr.asArrayRef())
        dims.push_back(d + 1);

      shadow->setAttr("dimensions",
                      DenseI64ArrayAttr::get(builder.getContext(), dims));
    }

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
    Value numIters;
    if (caches.empty()) {
      WhileLoopInfo info(cast<stablehlo::WhileOp>(orig));
      bool res = info.computeInfo().succeeded();
      assert(res && info.isValid() && info.isConstant());
      (void)res;

      auto iterType = orig->getOperand(0).getType();
      numIters = builder.create<stablehlo::ConstantOp>(
          orig->getLoc(), iterType,
          cast<ElementsAttr>(makeAttr(iterType, info.getConstantNumIters())));
    } else
      numIters = gutils->popCache(caches[0], builder);

    auto unrankedTensorType = RankedTensorType::get({}, builder.getI64Type());
    auto iterVarOp = builder.create<ConstantOp>(
        orig->getLoc(), unrankedTensorType,
        SplatElementsAttr::get(
            unrankedTensorType,
            ArrayRef<Attribute>(IntegerAttr::get(builder.getI64Type(), 0))));
    auto iterVar = iterVarOp.getResult();

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

      auto condIterVarElemType =
          cast<RankedTensorType>(condIterVar.getType()).getElementType();
      auto numItersElemType =
          cast<RankedTensorType>(numIters.getType()).getElementType();
      if (numItersElemType != condIterVarElemType) {
        builder.setInsertionPointAfter(iterVarOp);
        DenseIntElementsAttr numAttr;
        if (matchPattern(numIters, m_Constant(&numAttr))) {
          numIters = builder.create<ConstantOp>(
              orig->getLoc(), condIterVar.getType(),
              cast<ElementsAttr>(makeAttr(condIterVar.getType(),
                                          (*numAttr.begin()).getSExtValue())));
        } else {
          numIters = builder.create<ConvertOp>(orig->getLoc(), numIters,
                                               condIterVarElemType);
        }
        builder.setInsertionPointAfter(revWhile);
      }

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

    Type elementType = loopConditionVariableElementType(newWhile, revBuilder);

    Value numIters;

    WhileLoopInfo info(newWhile);
    if (info.computeInfo().succeeded()) {
      // no need to cache number of iterations if it is a known constant.
      if (info.isValid() && info.isConstant())
        return {};

      numIters = info.getNumIters(revBuilder);
    }

    if (!numIters) {
      auto cond = &newWhile.getCond().front();
      auto body = &newWhile.getBody().front();

      auto unrankedTensorType = RankedTensorType::get({}, elementType);
      auto numItersInit =
          revBuilder
              .create<ConstantOp>(
                  orig->getLoc(), unrankedTensorType,
                  SplatElementsAttr::get(
                      unrankedTensorType,
                      ArrayRef<Attribute>(IntegerAttr::get(elementType, 0))))
              .getResult();

      newWhile->insertOperands(newWhile->getNumOperands(),
                               ValueRange(numItersInit));
      cond->addArgument(numItersInit.getType(), orig->getLoc());
      Value numItersInBlock =
          body->addArgument(numItersInit.getType(), orig->getLoc());

      OpBuilder inBodyBuilder(body, body->begin());
      auto one = inBodyBuilder.create<ConstantOp>(
          orig->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(
              unrankedTensorType,
              ArrayRef<Attribute>(IntegerAttr::get(elementType, 1))));
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

private:
  Type loopConditionVariableElementType(WhileOp whileOp,
                                        OpBuilder &builder) const {
    auto *condBlock = &whileOp.getCond().front();

    auto condReturnOp =
        llvm::dyn_cast<stablehlo::ReturnOp>(condBlock->getTerminator());
    if (!condReturnOp || condReturnOp->getNumOperands() == 0)
      return builder.getI64Type();

    auto condVal = condReturnOp->getOperand(0);
    auto *defOp = condVal.getDefiningOp();

    auto cond = llvm::dyn_cast_or_null<stablehlo::CompareOp>(defOp);
    if (!cond)
      return builder.getI64Type();

    auto lhsType = cast<RankedTensorType>(cond.getOperand(0).getType());
    if (!lhsType)
      return builder.getI64Type();

    return lhsType.getElementType();
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

    SmallVector<int64_t> reshapedShape(outTy.getRank(), -1);
    for (auto [i, sz] : llvm::enumerate(outTy.getShape())) {
      if (llvm::is_contained(reducedDims, i)) {
        reshapedShape[i] = 1;
      } else {
        reshapedShape[i] = sz;
      }
    }

    SmallVector<int64_t> perm(outTy.getRank(), -1);
    SmallVector<int64_t> mapping(outTy.getRank(), -1);
    for (auto [i, dim] : llvm::enumerate(bcastDims)) {
      mapping[dim] = i;
    }

    int next = bcastDims.size();
    for (int i = 0; i < outTy.getRank(); i++) {
      if (mapping[i] == -1) {
        mapping[i] = next++;
      }
    }

    for (int i = 0; i < outTy.getRank(); i++) {
      perm[mapping[i]] = i;
    }

    auto reduceTy = RankedTensorType::get(iterShape, inTy.getElementType());
    auto bodyTy = RankedTensorType::get({}, inTy.getElementType());

    Value zero = cast<AutoDiffTypeInterface>(gutils->getShadowType(bodyTy))
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

    // for simplicity we do grad -> reduce -> reshape (restore 1 dims) ->
    // transpose -> reshape
    // The repeated reshapes are then eliminated via `enzyme-hlo-opt`.
    auto reshapedRed = builder.create<ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(reshapedShape, inTy.getElementType()),
        red->getResult(0));
    auto transposedVal =
        builder.create<TransposeOp>(op.getLoc(), reshapedRed, perm);
    auto res = builder.create<ReshapeOp>(
        op.getLoc(), gutils->getShadowType(op.getOperand().getType()),
        transposedVal);

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

    auto zeroPad = cast<AutoDiffTypeInterface>(
                       RankedTensorType::get({}, inTy.getElementType()))
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

static void makeAddBlock(Region &region, Location loc,
                         Type unrankedTensorType) {
  auto block = new Block();
  region.push_back(block);

  auto a = block->addArgument(unrankedTensorType, loc);
  auto b = block->addArgument(unrankedTensorType, loc);

  OpBuilder builder(block, block->end());
  auto addOp = builder.create<AddOp>(loc, a, b);
  builder.create<stablehlo::ReturnOp>(loc, addOp.getResult());
}

class AutoDiffReduceWindowRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffReduceWindowRev,
                                                       ReduceWindowOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<ReduceWindowOp>(orig);

    if (op.getNumOperands() != 2) {
      orig->emitError() << "Unsupported reduce window rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    auto &region = op.getBody();
    if (!region.hasOneBlock()) {
      orig->emitError() << "Unsupported reduce window rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    if (!gutils->isConstantValue(op.getOperand(1))) {
      orig->emitError() << "Unsupported reduce window rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    auto &block = op.getBody().front();
    if (!hasSingleElement(block.without_terminator())) {
      orig->emitError() << "Unsupported reduce window rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    Operation &innerOp = *block.begin();
    bool ismin = isa<MinOp>(innerOp), ismax = isa<MaxOp>(innerOp),
         isadd = isa<AddOp>(innerOp);

    if (!(ismin || ismax || isadd) ||
        innerOp.getOperand(0) != block.getArgument(0) ||
        innerOp.getOperand(1) != block.getArgument(1)) {
      orig->emitError() << "Unsupported reduce window rev autodiff(1): "
                        << *orig << "\n";
      return failure();
    }

    auto unrankedTensorType = RankedTensorType::get(
        {}, cast<RankedTensorType>(op.getResult(0).getType()).getElementType());

    Value inDiffe = gutils->diffe(op->getResult(0), builder);
    gutils->zeroDiffe(op->getResult(0), builder);

    if (isadd) {
      auto operandType = cast<RankedTensorType>(op->getOperand(0).getType());
      auto outputType = cast<RankedTensorType>(op->getResult(0).getType());

      // Compute padding similarly to conv lhs grad.
      int64_t N = cast<RankedTensorType>(operandType).getShape().size();

      SmallVector<int64_t> paddingValues(2 * N, 0);

      SmallVector<int64_t> paddingHigh(N, 0);
      SmallVector<int64_t> paddingLow(N, 0);
      SmallVector<int64_t> paddingInterior(N, 0);

      auto initialPadding = op.getPadding();
      if (initialPadding.has_value()) {
        paddingValues.assign(initialPadding.value().value_begin<int64_t>(),
                             initialPadding.value().value_end<int64_t>());
      }

      auto dilateShape = [](int64_t shape, int64_t dilation) {
        if (dilation == 1)
          return shape;
        int64_t dilated = 1 + dilation * (shape - 1);
        return dilated < 0 ? 0 : dilated;
      };

      auto lhsDilations = op.getBaseDilations();
      auto rhsDilations = op.getWindowDilations();
      auto windowStrides = op.getWindowStrides();
      auto windowDimensions = op.getWindowDimensions();

      for (int i = 0; i < N; ++i) {
        auto padBefore = paddingValues[2 * i];

        auto lhsDilation =
            lhsDilations.has_value() ? getI64Value(lhsDilations.value(), i) : 1;
        auto rhsDilation =
            rhsDilations.has_value() ? getI64Value(rhsDilations.value(), i) : 1;
        auto windowStride = windowStrides.has_value()
                                ? getI64Value(windowStrides.value(), i)
                                : 1;

        auto lhsShape = dilateShape(operandType.getShape()[i], lhsDilation);
        auto rhsShape = dilateShape(windowDimensions[i], rhsDilation);
        auto outShape = dilateShape(outputType.getShape()[i], windowStride);

        auto newPadBefore = rhsShape - padBefore - 1;

        paddingHigh[i] = newPadBefore;
        paddingLow[i] = lhsShape + rhsShape - 1 - outShape - newPadBefore;
        paddingInterior[i] = windowStride - 1;
      }

      auto paddingType = RankedTensorType::get({N, 2}, builder.getI64Type());

      SmallVector<int64_t> zeroPadding(2 * N, 0);
      auto newPaddingAttr =
          mlir::DenseIntElementsAttr::get(paddingType, zeroPadding);

      SmallVector<int64_t> newBaseDilation(N, 1);

      auto zero = cast<AutoDiffTypeInterface>(unrankedTensorType)
                      .createNullValue(builder, op.getLoc());

      auto paddedIndiffe =
          builder
              .create<stablehlo::PadOp>(op.getLoc(), inDiffe, zero,
                                        getI64Attr(builder, paddingHigh),
                                        getI64Attr(builder, paddingLow),
                                        getI64Attr(builder, paddingInterior))
              .getResult();

      auto revOp = builder.create<stablehlo::ReduceWindowOp>(
          op.getLoc(), operandType, paddedIndiffe,
          /*init_value*/ zero,
          /*window_dimensions*/ op.getWindowDimensionsAttr(),
          /*window_strides*/ op.getBaseDilationsAttr(),
          /*base_dilations*/ getI64Attr(builder, newBaseDilation),
          /*window_dilations*/ op.getWindowDilationsAttr(),
          /*padding*/ newPaddingAttr);

      makeAddBlock(revOp.getBody(), op.getLoc(), unrankedTensorType);
      gutils->addToDiffe(op.getOperand(0), revOp.getResult(0), builder);
    } else if (ismax || ismin) {
      auto select = new Block();
      select->addArgument(unrankedTensorType, op.getLoc());
      select->addArgument(unrankedTensorType, op.getLoc());
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(select);

        auto cmpOp = builder.create<CompareOp>(
            op.getLoc(), select->getArgument(0), select->getArgument(1),
            ismax ? ComparisonDirection::GE : ComparisonDirection::LE);
        builder.create<ReturnOp>(op.getLoc(), cmpOp.getResult());
      }

      auto revOp = builder.create<SelectAndScatterOp>(
          op.getLoc(), op.getOperand(0).getType(),
          gutils->popCache(caches[0], builder), inDiffe,
          cast<AutoDiffTypeInterface>(unrankedTensorType)
              .createNullValue(builder, op.getLoc()),
          op.getWindowDimensionsAttr(), op.getWindowStridesAttr(),
          op.getPaddingAttr());

      revOp.getSelect().push_back(select);
      makeAddBlock(revOp.getScatter(), op.getLoc(), unrankedTensorType);

      gutils->addToDiffe(op.getOperand(0), revOp.getResult(), builder);
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    SmallVector<Value> cachedArguments;

    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);

    Value cache = gutils->initAndPushCache(
        gutils->getNewFromOriginal(orig->getOperand(0)), cacheBuilder);
    cachedArguments.push_back(cache);

    return cachedArguments;
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

    auto inTy = cast<RankedTensorType>(op->getOperand(0).getType());
    auto zero =
        cast<AutoDiffTypeInterface>(inTy).createNullValue(builder, op.getLoc());
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

        auto zeroI = cast<AutoDiffTypeInterface>(inDiffe.getType())
                         .createNullValue(builder, op.getLoc());

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
      auto RT = cast<RankedTensorType>(inTy);
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
          Type retTy = cast<AutoDiffTypeInterface>(operand.get().getType())
                           .getShadowType();
          newOperands.push_back(
              cast<AutoDiffTypeInterface>(retTy).createNullValue(
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

class AutoDiffBatchNormTrainingRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffBatchNormTrainingRev, BatchNormTrainingOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<BatchNormTrainingOp>(orig);

    if (!gutils->isConstantValue(op->getResult(0)) ||
        !gutils->isConstantValue(op->getResult(1)) ||
        !gutils->isConstantValue(op->getResult(2))) {
      gutils->zeroDiffe(op->getResult(0), builder);

      auto opOperand0 = gutils->popCache(caches[0], builder);
      auto opOperand1 = gutils->popCache(caches[1], builder);
      auto opResult1 = gutils->getNewFromOriginal(op->getResult(1));
      auto opResult2 = gutils->getNewFromOriginal(op->getResult(2));

      auto gradOp = builder.create<BatchNormGradOp>(
          op->getLoc(), opOperand0, opOperand1, opResult1, opResult2,
          gutils->diffe(op->getOperand(0), builder), op.getEpsilonAttr(),
          op.getFeatureIndexAttr());

      if (!gutils->isConstantValue(op->getOperand(0))) {
        gutils->addToDiffe(op->getOperand(0), gradOp.getResult(0), builder);
      }
      if (!gutils->isConstantValue(op->getOperand(1))) {
        gutils->addToDiffe(op->getOperand(1), gradOp.getResult(1), builder);
      }
      if (!gutils->isConstantValue(op->getOperand(2))) {
        gutils->addToDiffe(op->getOperand(2), gradOp.getResult(2), builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto op = cast<BatchNormTrainingOp>(orig);

    if (!gutils->isConstantValue(op->getOperand(0)) ||
        !gutils->isConstantValue(op->getOperand(1)) ||
        !gutils->isConstantValue(op->getOperand(2))) {
      SmallVector<Value> caches;

      Operation *newOp = gutils->getNewFromOriginal(op);
      OpBuilder cacheBuilder(newOp);

      auto initCacheOperand0 = gutils->initAndPushCache(
          gutils->getNewFromOriginal(op->getOperand(0)), cacheBuilder);
      caches.push_back(initCacheOperand0);

      auto initCacheOperand1 = gutils->initAndPushCache(
          gutils->getNewFromOriginal(op->getOperand(1)), cacheBuilder);
      caches.push_back(initCacheOperand1);

      return caches;
    }

    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct WhileOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<WhileOpEnzymeOpsRemover,
                                                        stablehlo::WhileOp> {
private:
#define DEBUG_TYPE "enzymexla-stablehlo-while-mincut"

  // A node in the compute graph.
  // Operation nodes have outgoing edges to value nodes that they produce and
  // incoming nodes from values they take as operands.
  struct Node {
    Operation *O;
    Value V;
    enum Type {
      NONE,
      VAL,
      OP,
    } type;

    Node(Operation *O) : O(O), type(OP){};
    Node(Value V) : V(V), type(VAL){};
    Node() : type(NONE){};

    bool operator<(const Node N) const {
      if (type != N.type)
        return type < N.type;
      else if (type == OP)
        return O < N.O;
      else if (type == VAL)
        return V.getAsOpaquePointer() < N.V.getAsOpaquePointer();
      else
        return true;
    }
    void dump() const {
      if (type == VAL)
        llvm::errs() << "[" << V << ", " << "Value" << "]\n";
      else if (type == OP)
        llvm::errs() << "[" << *O << ", " << "Operation" << "]\n";
      else
        llvm::errs() << "[" << "NULL" << ", " << "None" << "]\n";
    }
  };

  typedef std::map<Node, std::set<Node>> Graph;

  static void dump(Graph &G) {
    for (auto &pair : G) {
      pair.first.dump();
      for (const auto &N : pair.second) {
        llvm::errs() << "\t";
        N.dump();
      }
    }
  }

  // parent is populated with a path from each connected leaf node of G to one
  // of the Value in Source.
  static inline void bfs(const Graph &G, const llvm::SetVector<Value> &Sources,
                         std::map<Node, Node> &parent) {
    std::deque<Node> q;
    for (auto V : Sources) {
      Node N(V);
      parent.emplace(N, Node());
      q.push_back(N);
    }

    // Standard BFS Loop
    while (!q.empty()) {
      auto u = q.front();
      q.pop_front();
      auto found = G.find(u);
      if (found == G.end())
        continue;
      for (auto v : found->second) {
        if (parent.find(v) == parent.end()) {
          q.push_back(v);
          parent.emplace(v, u);
        }
      }
    }
  }

  // Whether or not an operation can be moved from the forward region to the
  // reverse region or vice-versa.
  static inline bool isMovable(Operation *op) {
    return mlir::isPure(op) && op->getNumRegions() == 0;
  }

  // Given the full forward/backward compute graph, the push/pop can be seen as
  // a special cut of this graph. This function tries to modifies the boundary
  // of the push/pop to minimize the amount of memory that is live across
  // different loops.
  static void minCutCache(Block *forward, Block *reverse,
                          SmallVector<CacheInfo> &caches,
                          PatternRewriter &rewriter) {
    if (caches.empty())
      return;

    // where to build the new inits
    Operation *entry = caches[0].initOp;

    Graph G;

    LLVM_DEBUG(llvm::dbgs() << "trying min/cut\n");
    LLVM_DEBUG(forward->getParentOp()->getParentOp()->dump());

    SmallVector<Value> worklist;
    for (auto &cache : caches) {
      worklist.push_back(cache.pushedValue());
    }

    // nodes that cannot be recomputed
    SetVector<Value> roots;

    // Walk Backward
    //
    // Roots (sources) are either block arguments or values which are defined
    // outside of forward.
    while (!worklist.empty()) {
      Value todo = worklist.pop_back_val();

      if (todo.getParentBlock() != forward) {
        roots.insert(todo);
        continue;
      }

      Operation *owner = todo.getDefiningOp();
      if (!owner || !isMovable(owner)) {
        roots.insert(todo);
        continue;
      }

      auto &&[_, inserted] = G[Node(owner)].insert(Node(todo));
      if (inserted) {
        for (Value operand : owner->getOperands()) {
          G[Node(operand)].insert(Node(owner));
          worklist.push_back(operand);
        }
      }
    }

    worklist.clear();

    for (auto &info : caches) {
      // insert use of the push through the pop. These define the existing
      // forward/reverse cut that the min cut is trying to improve.
      //
      // Given the following IR:
      //
      // %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>
      // ^forward:
      //   %pushed = "operation.someop"(%somevalue) : (f32) -> f32
      //   "enzyme.push"(%cache, %pushed) : (!enzyme.Cache<f32>, f32) -> ()
      // ^backward:
      //   %poped = "enzyme.pop"(%cache) : (!enzyme.Cache<f32>) -> f32
      //   %use = "operation.use"(%poped) : (f32) -> f32
      //
      // will result in the following graph:
      //
      // [%somevalue, Value]
      //   [%pushed, Operation]
      //     [%pushed, Value]
      //       [%poped, Operation]
      //         [%poped, Value]
      //           [%use, Operation]
      //             [%use, Value]
      //
      Node popNode = Node(static_cast<Operation *>(info.popOp));
      Value poped = info.popOp.getResult();
      G[Node(info.pushedValue())].insert(popNode);
      G[popNode].insert(Node(poped));
      worklist.push_back(poped);
    }

    SetVector<Value> Required;

    // Walk Forward
    while (!worklist.empty()) {
      Value todo = worklist.pop_back_val();

      for (auto user : todo.getUsers()) {
        if (user->getBlock() != reverse && !isMovable(user)) {
          Required.insert(todo);
          continue;
        }

        if (!llvm::all_of(user->getOperands(), [&G, &todo](Value operand) {
              return operand == todo || G.count(Node(operand));
            })) {
          Required.insert(todo);
          continue;
        }

        Node N(user);
        auto &&[_, inserted] = G[Node(todo)].insert(N);
        if (inserted) {
          for (Value res : user->getResults()) {
            G[N].insert(Node(res));
            worklist.push_back(res);
          }
        }
      }
    }

    if (G.empty())
      return;

    LLVM_DEBUG(dump(G));

    Graph Orig = G;

    // Augment the flow while there is a path from source to sink
    while (1) {
      std::map<Node, Node> parent;
      bfs(G, roots, parent);
      Node end;
      for (auto req : Required) {
        if (parent.find(Node(req)) != parent.end()) {
          end = Node(req);
          break;
        }
      }
      if (end.type == Node::NONE)
        break;
      // update residual capacities of the edges and reverse edges
      // along the path
      Node v = end;
      while (1) {
        assert(parent.find(v) != parent.end());
        Node u = parent.find(v)->second;
        assert(u.type != Node::NONE);
        assert(G[u].count(v) == 1);
        assert(G[v].count(u) == 0);
        G[u].erase(v);
        G[v].insert(u);
        if (u.type == Node::VAL && roots.contains(u.V))
          break;
        v = u;
      }
    }
    // Flow is maximum now, find vertices reachable from s

    std::map<Node, Node> parent;
    bfs(G, roots, parent);

    LLVM_DEBUG(llvm::dbgs() << "residual graph: \n";);
    LLVM_DEBUG(dump(G));

    // Those are the new values to cache
    SetVector<Value> newCaches;

    // All edges that are from a reachable vertex to non-reachable vertex in the
    // original graph are edges for the minimum cut. The set of values to cache
    // are the values transported along those edges (either. Value -> Operation
    // or Operation -> Value).
    //
    // Note: we could use more heuristics here to select the actual cached value
    //       based on sizes, existing caches, number of users in the fwd as to
    //       not duplicate work, etc...
    for (auto &pair : Orig) {
      if (parent.find(pair.first) != parent.end()) {
        for (auto N : pair.second) {
          if (parent.find(N) == parent.end()) {
            Value newCache;
            if (pair.first.type == Node::VAL) {
              assert(N.type == Node::OP);
              newCache = pair.first.V;
            } else {
              assert(pair.first.type == Node::OP);
              assert(N.type == Node::VAL);
              newCache = N.V;
            }
            newCaches.insert(newCache);
          }
        }
      }
    }

    // compute path from new caches to required
    parent.clear();
    bfs(Orig, newCaches, parent);

    // The reverse graph is a sub graph of Orig with only pathes from Required
    // to "dominating" caches.
    Graph revGraph;
    for (Value req : Required) {
      auto p = parent.find(Node(req));
      while (p != parent.end()) {
        revGraph[p->second].insert(p->first);
        p = parent.find(p->second);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "revGraph:\n");
    LLVM_DEBUG(dump(revGraph));

    // Refine cached values based on some heuristics
    auto newCacheVec = newCaches.takeVector();

    {
      // sort caches to provide determinism.
      auto sorter = [](const Value &a, const Value &b) -> bool {
        if (a == b)
          return true;

        auto ba = dyn_cast<BlockArgument>(a);
        auto bb = dyn_cast<BlockArgument>(b);
        if (ba && bb) {
          if (ba.getOwner() == bb.getOwner())
            return ba.getArgNumber() < bb.getArgNumber();

          return ba.getOwner()->getParent()->isProperAncestor(
              bb.getOwner()->getParent());
        }

        if (ba)
          return true;

        if (bb)
          return false;

        OpResult ra = cast<OpResult>(a);
        OpResult rb = cast<OpResult>(b);

        if (ra.getParentBlock() == rb.getParentBlock())
          return ra.getDefiningOp()->isBeforeInBlock(rb.getDefiningOp());

        return ra.getParentRegion()->isProperAncestor(rb.getParentRegion());
      };
      std::sort(newCacheVec.begin(), newCacheVec.end(), sorter);
    };

    for (Value newCache : newCacheVec) {
      SetVector<Value> candidates;

      worklist.clear();
      worklist.push_back(newCache);

      while (!worklist.empty()) {
        Value candidate = worklist.pop_back_val();

        auto C = revGraph.find(Node(candidate));
        if (C == revGraph.end())
          continue;

        if (C->second.size() > 1)
          continue;

        if (candidate.getParentBlock() == reverse)
          continue; // TODO: support this

        candidates.insert(candidate);
        for (auto &N : C->second) {
          // not eligible
          if (N.O->getNumResults() > 1)
            continue;

          worklist.append(N.O->getResults().begin(), N.O->getResults().end());
        }
      }

      auto computeSizeOfType = [](Value val) -> int64_t {
        auto T = cast<RankedTensorType>(val.getType());
        if (!T.getElementType().isIntOrFloat())
          return INT64_MAX;
        int64_t sz = T.getElementType().getIntOrFloatBitWidth();
        for (auto sh : T.getShape())
          sz *= sh;
        return sz;
      };

      Value picked = newCache;
      int64_t curSize = computeSizeOfType(picked),
              curRank = cast<RankedTensorType>(picked.getType()).getRank();
      for (Value candidate : candidates) {
        int64_t newSize = computeSizeOfType(candidate),
                newRank = cast<RankedTensorType>(candidate.getType()).getRank();
        if (newSize < curSize || (newSize == curSize && newRank < curRank) ||
            candidate.getDefiningOp<enzyme::PopOp>() != nullptr) {
          curSize = newSize;
          curRank = newRank;
          picked = candidate;
        }
      }

      auto p = parent.find(Node(picked));
      while (p != parent.end()) {
        revGraph.erase(p->second);
        p = parent.find(p->second);
      }

      newCaches.insert(picked);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "new caches: \n";
      for (Value v : newCaches) {
        v.dump();
      }
    });

    SmallVector<CacheInfo> newCacheInfos;
    IRMapping mapping;

    // For all new caches, materialize the path either by moving ops from
    // forward to reverse or reverse to forward.
    for (Value newCache : newCaches) {
      enzyme::InitOp initOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(entry);
        rewriter.create<enzyme::InitOp>(
            newCache.getLoc(),
            enzyme::CacheType::get(newCache.getContext(), newCache.getType()));
      });
      enzyme::PushOp pushOp;
      enzyme::PopOp popOp;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(newCache);

      // TODO: This newCache value might not be available here since it might be
      //       a part of the reverse. The operations needed to create newCache
      //       in the forward should be cloned from forward to reverse.
      assert(newCache.getParentBlock() != reverse && "todo");

      pushOp = rewriter.create<enzyme::PushOp>(newCache.getLoc(),
                                               initOp.getResult(), newCache);

      rewriter.setInsertionPointToStart(reverse);
      popOp = rewriter.create<enzyme::PopOp>(
          newCache.getLoc(), newCache.getType(), initOp.getResult());

      mapping.map(newCache, popOp.getResult());

      CacheInfo info;
      info.initOp = initOp;
      info.pushOp = pushOp;
      info.popOp = popOp;
      newCacheInfos.push_back(info);
    }

    worklist.clear();
    worklist.assign(newCaches.begin(), newCaches.end());

    // Clone ops in the reverse graph to make sure all edges have been mapped.
    while (!worklist.empty()) {
      Value todo = worklist.pop_back_val();

      if (Required.count(todo)) {
        rewriter.replaceAllUsesWith(todo, mapping.lookup(todo));
        continue;
      }

      auto found = revGraph.find(Node(todo));
      assert(found != revGraph.end());

      for (auto N : found->second) {
        assert(N.type == Node::OP);

        // Special case for across forward/reverse boundary.
        if (isa<enzyme::PopOp>(N.O)) {
          rewriter.replaceAllOpUsesWith(N.O, mapping.lookup(todo));
          continue;
        }

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfterValue(mapping.lookup(todo));
        Operation *newO = rewriter.clone(*N.O, mapping);

        for (auto [oldRes, newRes] :
             llvm::zip_equal(N.O->getResults(), newO->getResults()))
          mapping.map(oldRes, newRes);

        auto pair = revGraph.find(N);
        if (pair == revGraph.end())
          continue;

        for (auto NN : pair->second) {
          assert(NN.type == Node::VAL);
          worklist.push_back(NN.V);
        }
      }
    }

    // Remove old caches
    for (auto &info : caches) {
      rewriter.eraseOp(info.popOp);
      rewriter.eraseOp(info.pushOp);
      rewriter.eraseOp(info.initOp);
    }

    // Set new caches
    caches.assign(newCacheInfos.begin(), newCacheInfos.end());
  }

#undef DEBUG_TYPE

public:
  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);
    stablehlo::WhileOp otherWhileOp; // where cache pops are

    Block *body = &whileOp.getBody().front();
    Block *cond = &whileOp.getCond().front();

    // Gradients whose values need to be passed as iteration variables.
    llvm::SmallDenseSet<Value> updatedGradients;

    llvm::MapVector<Value, CacheInfo> cachesMap;

    for (auto &it : *body) {
      Operation *op = &it;

      if (auto setOp = dyn_cast<enzyme::SetOp>(op))
        updatedGradients.insert(setOp.getGradient());

      if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
        CacheInfo info(pushOp.getCache());

        // Forward push to pop
        if (info.pushOp->getBlock() == info.popOp->getBlock()) {
          assert(info.pushOp->isBeforeInBlock(info.popOp));
          info.popOp.getResult().replaceAllUsesWith(info.pushOp.getValue());
          continue;
        }

        Value pushedValue = info.pushedValue();
        if (cachesMap.contains(pushedValue)) {
          info = info.merge(cachesMap.lookup(pushedValue), rewriter);
        }
        cachesMap[pushedValue] = info;

        otherWhileOp = cast<stablehlo::WhileOp>(info.popOp->getParentOp());
      }
    }

    // nothing to do
    if (updatedGradients.empty() && cachesMap.empty())
      return success();

    SmallVector<CacheInfo> caches;
    caches.reserve(cachesMap.size());
    for (auto &&[_, info] : cachesMap) {
      caches.push_back(info);
    }

    WhileLoopInfo info(whileOp);

    // TODO: support non-constant loops by using a dynamic dimension
    // ...   should we fail ? i.e. return failure();
    if (info.computeInfo().failed() || !info.isValid() || !info.isConstant()) {
      return rewriter.notifyMatchFailure(
          op, "WhileOp does not have static iteration count for cache removal");
    }

    // 1. Move enzyme.get outside the body if the variable is not used outside
    // the loop
    for (auto &it : *body) {
      Operation *op = &it;

      auto getOp = dyn_cast<enzyme::GetOp>(op);
      if (!getOp || updatedGradients.contains(getOp.getGradient()))
        continue;

      auto outerGet = rewriter.create<enzyme::GetOp>(
          getOp->getLoc(),
          cast<enzyme::GradientType>(getOp.getResult().getType()).getBasetype(),
          getOp.getGradient());

      rewriter.replaceAllUsesWith(getOp.getResult(), outerGet.getResult());
      rewriter.eraseOp(getOp);
    }

    auto term = body->getTerminator();

    // 2. For gradients whose value are updated during the iterations, the new
    // values need to be passed as arguments to the body and gets should be
    // replaced with the corresponding new argument.
    SmallVector<Value> newOperands(whileOp.getOperands());
    for (auto grad : updatedGradients) {
      auto Ty = cast<enzyme::GradientType>(grad.getType()).getBasetype();
      auto outerGet = rewriter.create<enzyme::GetOp>(grad.getLoc(), Ty, grad);

      newOperands.push_back(outerGet.getResult());
      auto newArg = body->addArgument(Ty, grad.getLoc());
      cond->addArgument(Ty, grad.getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPointToStart(body);
        rewriter.create<enzyme::SetOp>(grad.getLoc(), grad, newArg);

        rewriter.setInsertionPoint(term);

        auto outputVal =
            rewriter.create<enzyme::GetOp>(grad.getLoc(), Ty, grad).getResult();
        term->insertOperands(term->getNumOperands(), ValueRange(outputVal));
      }
    }

    // 3. For enzyme.cache, the iteration number needs to be known. Given a
    // while loop with N iterations. For each of these cache, generate a
    // batched tensor with N prepended. Cache pushes become
    // dynamic_update_slice and cache pops become dynamic_slice.
    auto numIters = info.getConstantNumIters();

    Value inductionVariable; // [0,..., N - 1] counter from within the loop

    if (matchPattern(info.start, m_Zero()) &&
        matchPattern(info.step, m_One())) {
      inductionVariable = body->getArgument(0);
    }

    auto zero = makeI64Constant(whileOp->getLoc(), rewriter, 0);

    // Run min cut partitioning to limit the amount of values to be cached.
    if (!caches.empty() && !whileOp->hasAttr("enzymexla.disable_min_cut")) {
      Block *forward = &whileOp.getBody().front();
      Block *reverse = &otherWhileOp.getBody().front();
      minCutCache(forward, reverse, caches, rewriter);
    }

    for (auto &info : caches) {
      Value cache = info.initOp.getResult();

      // push does not depend on a value inside the loop, we can hoist the
      // push/pop before the for loops.
      if (info.pushedValue().getParentRegion() != whileOp.getBody()) {
        auto newPush = rewriter.create<enzyme::PushOp>(cache.getLoc(), cache,
                                                       info.pushedValue());
        rewriter.eraseOp(info.pushOp);
        info.pushOp = newPush;

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(info.popOp->getParentOp());

          auto popVal = info.popOp.getResult();
          auto newPop = rewriter.create<enzyme::PopOp>(cache.getLoc(),
                                                       popVal.getType(), cache);
          rewriter.replaceAllUsesWith(popVal, newPop.getResult());
          rewriter.eraseOp(info.popOp);
          info.popOp = newPop;
        }

        continue;
      }

      if (!inductionVariable) {
        return rewriter.notifyMatchFailure(
            op, "WhileOp does not have induction variable for cache removal");

        // TODO: support adding an induction variable if not present

        // Value zero = rewriter.create<arith::ConstantOp>(whileOp->getLoc(),
        //                                                rewriter.getIndexAttr(0));
        // newOperands.push_back(zero);

        // inductionVariable = body->addArgument(zero.getType(),
        // whileOp->getLoc());
        //  cond->addArgument(zero.getType(), whileOp->getLoc());
        // {
        //   OpBuilder::InsertionGuard guard(rewriter);
        //   rewriter.setInsertionPoint(term);

        //   auto one = rewriter.create<arith::ConstantOp>(whileOp->getLoc(),
        //                                                rewriter.getIndexAttr(1));
        //   auto newInductionVar = rewriter.create<arith::AddIOp>(
        //       whileOp->getLoc(), inductionVariable, one);
        //   term->insertOperands(term->getNumOperands(),
        //                        ValueRange(newInductionVar));
        // }
      }

      auto newType =
          cast<ShapedType>(cast<AutoDiffTypeInterface>(info.cachedType())
                               .getShadowType(numIters));

      Value initValue = cast<AutoDiffTypeInterface>(newType).createNullValue(
          rewriter, info.initOp->getLoc());

      newOperands.push_back(initValue);

      auto cacheValue = body->addArgument(newType, info.pushOp->getLoc());
      cond->addArgument(newType, info.pushOp->getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.pushOp);

        Value newCacheValue;
        if (auto TT = dyn_cast<TensorType>(info.cachedType())) {
          auto shape = TT.getShape();

          SmallVector<Value> startIndices(shape.size() + 1, zero);
          startIndices[0] = inductionVariable;

          SmallVector<int64_t> updateShape;
          updateShape.push_back(1);
          updateShape.append(shape.begin(), shape.end());
          Value reshapedUpdate = rewriter.create<stablehlo::ReshapeOp>(
              info.pushOp->getLoc(), TT.clone(updateShape),
              info.pushOp.getValue());

          newCacheValue = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
              info.pushOp->getLoc(), cacheValue, reshapedUpdate, startIndices);
        } else {
          assert(false && "todo");
          // newCacheValue = rewriter.create<tensor::InsertOp>(
          //     info.pushOp->getLoc(), info.pushOp.getValue(), cacheValue,
          //     inductionVariable);
        }

        term->insertOperands(term->getNumOperands(), ValueRange(newCacheValue));
      }
    }

    auto numInitArgs = whileOp->getNumOperands();
    auto newWhile =
        rewriter.create<stablehlo::WhileOp>(op->getLoc(), newOperands);

    newWhile.getCond().takeBody(whileOp.getCond());
    newWhile.getBody().takeBody(whileOp.getBody());

    unsigned resultIdx = numInitArgs;
    for (auto grad : updatedGradients) {
      // set the updated gradient after the new for op.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.create<enzyme::SetOp>(grad.getLoc(), grad,
                                     newWhile->getResult(resultIdx));
      ++resultIdx;
    }

    for (auto &&[res, newRes] :
         llvm::zip(whileOp->getResults(), newWhile->getResults())) {
      rewriter.replaceAllUsesWith(res, newRes);
    }

    // 4. On the other while op (the one containing the pops), we add an
    // induction variable and replace pops with slice from the tensor version of
    // the cache.
    if (inductionVariable && caches.size() != 0) {
      if (isa<BlockArgument>(inductionVariable) &&
          cast<BlockArgument>(inductionVariable).getArgNumber() != 0)
        resultIdx++;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(otherWhileOp);
      SmallVector<Value> operands(otherWhileOp->getOperands().begin(),
                                  otherWhileOp->getOperands().end());
      operands.push_back(
          makeI64Constant(otherWhileOp->getLoc(), rewriter, numIters - 1));

      Block *otherBody = &otherWhileOp.getBody().front();
      Block *otherCond = &otherWhileOp.getCond().front();
      Value otherInductionVariable = otherBody->addArgument(
          RankedTensorType::get({}, rewriter.getI64Type()),
          otherWhileOp->getLoc());
      otherCond->addArgument(otherInductionVariable.getType(),
                             otherWhileOp->getLoc());
      auto otherTerm = otherBody->getTerminator();

      rewriter.setInsertionPoint(otherTerm);

      otherInductionVariable =
          rewriter
              .create<stablehlo::SubtractOp>(
                  otherWhileOp->getLoc(), otherInductionVariable,
                  makeI64Constant(otherWhileOp->getLoc(), rewriter, 1))
              .getResult();
      otherTerm->insertOperands(otherTerm->getNumOperands(),
                                ValueRange(otherInductionVariable));

      rewriter.setInsertionPoint(otherWhileOp);
      auto newOtherWhileOp =
          rewriter.create<stablehlo::WhileOp>(otherWhileOp->getLoc(), operands);

      for (auto &&[res, newRes] : llvm::zip(otherWhileOp->getResults(),
                                            newOtherWhileOp->getResults())) {
        rewriter.replaceAllUsesWith(res, newRes);
      }
      newOtherWhileOp.getCond().takeBody(otherWhileOp.getCond());
      newOtherWhileOp.getBody().takeBody(otherWhileOp.getBody());

      rewriter.eraseOp(otherWhileOp);
      otherWhileOp = newOtherWhileOp;
    }

    // 5. Finally, replace pops with slices.
    for (auto &info : caches) {
      if (info.pushedValue().getParentRegion() != newWhile.getBody())
        continue;

      Value cache = info.initOp.getResult();

      auto newType =
          cast<ShapedType>(cast<AutoDiffTypeInterface>(info.cachedType())
                               .getShadowType(numIters));
      enzyme::InitOp newInit = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.initOp);

        rewriter.create<enzyme::InitOp>(
            info.initOp->getLoc(),
            enzyme::CacheType::get(cache.getContext(), newType));
      });
      info.pushOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(newWhile);
        auto newPush =
            rewriter.create<enzyme::PushOp>(cache.getLoc(), newInit.getResult(),
                                            newWhile->getResult(resultIdx));
        rewriter.eraseOp(info.pushOp);
        newPush;
      });

      resultIdx++;

      {
        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPoint(otherWhileOp);

        auto popNewValue = rewriter.create<enzyme::PopOp>(
            info.popOp->getLoc(), newType, newInit.getResult());

        Block *popBody = &otherWhileOp.getBody().front();
        rewriter.setInsertionPoint(info.popOp);

        Value newInductionVariable =
            popBody->getArgument(popBody->getNumArguments() - 1);

        Value popValue;
        if (auto TT = dyn_cast<TensorType>(info.cachedType())) {
          auto shape = TT.getShape();
          SmallVector<Value> startIndices(shape.size() + 1, zero);
          startIndices[0] = newInductionVariable;
          SmallVector<int64_t> sliceSizes;
          sliceSizes.reserve(shape.size() + 1);
          sliceSizes.push_back(1);
          sliceSizes.append(shape.begin(), shape.end());

          popValue = rewriter.create<stablehlo::DynamicSliceOp>(
              info.popOp->getLoc(), TT.clone(sliceSizes), popNewValue,
              startIndices, sliceSizes);
          popValue = rewriter.create<stablehlo::ReshapeOp>(info.popOp->getLoc(),
                                                           TT, popValue);
        } else {
          assert(false && "todo");
          // popValue = tensor.extract(%popNewValue)
        }

        rewriter.replaceAllUsesWith(info.popOp, popValue);
        rewriter.eraseOp(info.popOp);
      }
    }

    rewriter.eraseOp(whileOp);

    return success();
  }
};

static void removalBlockExplore(Block *block, IRMapping &mapping,
                                PatternRewriter &rewriter,
                                llvm::SetVector<Value> &gradients,
                                llvm::MapVector<Value, CacheInfo> &caches) {
  for (auto it = block->begin(), e = block->end(); it != e;) {
    Operation *op = &*it;

    if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
      auto grad = setOp.getGradient();
      auto value = setOp.getValue();
      mapping.map(grad, value);
      gradients.insert(grad);
    }

    if (auto getOp = dyn_cast<enzyme::GetOp>(op)) {
      auto grad = getOp.getGradient();
      Value value = mapping.lookupOrNull(getOp.getGradient());
      if (!value) {
        value = rewriter.create<enzyme::GetOp>(
            getOp->getLoc(), getOp.getResult().getType(), grad);
        mapping.map(grad, value);
      }
      rewriter.replaceAllUsesWith(getOp.getResult(), value);
    }

    if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
      CacheInfo info(pushOp.getCache());

      Value pushedValue = info.pushedValue();

      // Then we can push the value before the if, if it is defined before the
      // if
      if (pushedValue.getParentBlock() != block) {
        rewriter.create<enzyme::PushOp>(pushOp->getLoc(), pushOp.getCache(),
                                        pushedValue);

        ++it; // Increment iterator to allow in place deletion
        rewriter.eraseOp(pushOp);

        // Move the pop before the other if
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.popOp->getParentOp());

        auto newPop = rewriter.create<enzyme::PopOp>(
            info.popOp->getLoc(), pushedValue.getType(), info.popOp.getCache());
        rewriter.replaceAllUsesWith(info.popOp.getResult(), newPop);
        rewriter.eraseOp(info.popOp);

        continue;
      }

      if (caches.contains(pushedValue)) {
        info = info.merge(caches.lookup(pushedValue), rewriter);
      }
      caches[pushedValue] = info;
    }

    ++it;
  }
}

struct IfOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<IfOpEnzymeOpsRemover,
                                                        stablehlo::IfOp> {
  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    // Gradients:
    //
    //  For each set in a branch, we instead set after the if by using the
    //  return value.
    //
    //  if %pred {
    //    enzyme.set %grad, %2
    //  } else {
    //  }
    //
    //  %0 = enzyme.get %grad
    //  %1 = if %pred {
    //    return %2
    //  } else {
    //    return %0
    //  }
    //  enzyme.set %grad, %1
    //
    //  For each get in a branch, we get before and use that instead of the
    //  get.

    // Caches:
    //
    // For each push, push after the if instead add a dummy value in the other
    // branch.
    //
    // For each pop in the reverse if, pop before the if instead of inside a
    // branch.

    auto ifOp = cast<IfOp>(op);

    Block *trueBlock = &ifOp.getTrueBranch().front(),
          *falseBlock = &ifOp.getFalseBranch().front();

    // Gradients whose value is set in either branches.
    llvm::SetVector<Value> gradients;

    // We assume pushes are exclusive.
    llvm::MapVector<Value, CacheInfo> pushedCaches;

    // Grad to value
    IRMapping trueMapping, falseMapping;

    removalBlockExplore(trueBlock, trueMapping, rewriter, gradients,
                        pushedCaches);
    removalBlockExplore(falseBlock, falseMapping, rewriter, gradients,
                        pushedCaches);

    if (gradients.empty() && pushedCaches.empty())
      return success();

    Operation *trueTerm = trueBlock->getTerminator();
    Operation *falseTerm = falseBlock->getTerminator();

    for (auto grad : gradients) {
      auto trueValue = trueMapping.lookupOrNull(grad);
      if (!trueValue) {
        trueValue = rewriter.create<enzyme::GetOp>(
            grad.getLoc(),
            cast<enzyme::GradientType>(grad.getType()).getBasetype(), grad);
      }
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));

      auto falseValue = falseMapping.lookupOrNull(grad);
      if (!falseValue) {
        falseValue = rewriter.create<enzyme::GetOp>(
            grad.getLoc(),
            cast<enzyme::GradientType>(grad.getType()).getBasetype(), grad);
      }
      falseTerm->insertOperands(falseTerm->getNumOperands(),
                                ValueRange(falseValue));
    }

    for (auto &[pushedValue, info] : pushedCaches) {
      Value dummy = cast<AutoDiffTypeInterface>(pushedValue.getType())
                        .createNullValue(rewriter, pushedValue.getLoc());

      Value trueValue =
          pushedValue.getParentBlock() == trueBlock ? pushedValue : dummy;
      Value falseValue =
          pushedValue.getParentBlock() == falseBlock ? pushedValue : dummy;

      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));
      falseTerm->insertOperands(falseTerm->getNumOperands(),
                                ValueRange(falseValue));
    }

    auto newIf = rewriter.create<stablehlo::IfOp>(
        ifOp->getLoc(), trueTerm->getOperandTypes(), ifOp.getPred());
    newIf.getTrueBranch().takeBody(ifOp.getTrueBranch());
    newIf.getFalseBranch().takeBody(ifOp.getFalseBranch());

    size_t idx = ifOp->getNumResults();
    for (auto grad : gradients) {
      rewriter.create<enzyme::SetOp>(grad.getLoc(), grad,
                                     newIf->getResult(idx));
      idx++;
    }

    for (auto &[pushedValue, info] : pushedCaches) {
      rewriter.create<enzyme::PushOp>(info.pushOp->getLoc(),
                                      info.initOp.getResult(),
                                      newIf->getResult(idx));
      rewriter.eraseOp(info.pushOp);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(info.popOp->getParentOp());

      auto newPop = rewriter.create<enzyme::PopOp>(
          info.popOp->getLoc(), info.popOp.getResult().getType(),
          info.popOp.getCache());
      rewriter.replaceAllUsesWith(info.popOp.getResult(), newPop);
      rewriter.eraseOp(info.popOp);

      idx++;
    }

    rewriter.replaceAllUsesWith(
        ifOp->getResults(),
        newIf->getResults().slice(0, ifOp->getNumResults()));
    rewriter.eraseOp(ifOp);

    return success();
  }
};

struct SHLOReduceOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOReduceOpBatchInterface,
                                             ReduceOp> {

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

    stablehlo::ReduceOp reduceOp = cast<stablehlo::ReduceOp>(src);
    if (!reduceOp)
      return failure();

    // Remap the operands
    SmallVector<Value, 8> newReduceInputs;
    newReduceInputs.reserve(reduceOp.getInputs().size());
    for (auto opValue : reduceOp.getInputs())
      newReduceInputs.push_back(mapper.lookup(opValue));

    // The init value would have been batched already, we need to slice it.
    // Constant Folding will fix it up later.
    SmallVector<Value, 8> newReduceInits;
    newReduceInits.reserve(reduceOp.getInitValues().size());
    for (auto opValue : reduceOp.getInitValues()) {
      auto batchedInit = mapper.lookup(opValue);

      SmallVector<int64_t> sliceStrides(batchSizes.size(), 1);
      SmallVector<int64_t> sliceStarts(batchSizes.size());
      SmallVector<int64_t> sliceLimits(batchSizes.size());
      for (int i = 0; i < batchSizes.size(); i++) {
        sliceStarts[i] = batchSizes[i] - 1;
        sliceLimits[i] = batchSizes[i];
      }

      auto elemType =
          cast<RankedTensorType>(batchedInit.getType()).getElementType();
      auto slicedInit = builder.create<SliceOp>(
          batchedInit.getLoc(), RankedTensorType::get(sliceStrides, elemType),
          batchedInit, sliceStarts, sliceLimits, sliceStrides);
      auto reshapedInit = builder.create<ReshapeOp>(
          batchedInit.getLoc(), RankedTensorType::get({}, elemType),
          slicedInit);
      newReduceInits.push_back(reshapedInit->getResult(0));
    }

    SmallVector<int64_t> reduceDims = llvm::to_vector(reduceOp.getDimensions());
    for (int i = 0; i < reduceDims.size(); i++) {
      reduceDims[i] += batchSizes.size();
    }

    auto newReduceOp = builder.create<stablehlo::ReduceOp>(
        src->getLoc(), resultTypes, newReduceInputs, newReduceInits,
        reduceDims);

    {
      OpBuilder::InsertionGuard guard(builder);

      Block &oldBlock = reduceOp.getBody().front();
      Block *newBlock = new Block();

      for (BlockArgument arg : oldBlock.getArguments()) {
        newBlock->addArgument(arg.getType(), arg.getLoc());
      }

      IRMapping regionMapper;
      for (auto [oldArg, newArg] :
           llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
        regionMapper.map(oldArg, newArg);
      }

      for (Operation &op : oldBlock) {
        builder.setInsertionPointToEnd(newBlock);
        builder.clone(op, regionMapper);
      }

      newReduceOp.getBody().push_back(newBlock);
    }

    for (int i = 0; i < reduceOp.getResults().size(); i++) {
      mapper.map(src->getResult(i), newReduceOp.getResult(i));
    }
    return success();
  }
};

struct SHLODotGeneralOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLODotGeneralOpBatchInterface,
                                             DotGeneralOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<DotGeneralOp>(src);
    auto dimensionNumbers = op.getDotDimensionNumbers();

    SmallVector<int64_t> lhsBatchingDimensions, rhsBatchingDimensions,
        lhsContractingDimensions, rhsContractingDimensions;

    for (int i = 0; i < batchSizes.size(); i++) {
      lhsBatchingDimensions.push_back(i);
      rhsBatchingDimensions.push_back(i);
    }
    for (auto &dim : dimensionNumbers.getLhsBatchingDimensions()) {
      lhsBatchingDimensions.push_back(dim + batchSizes.size());
    }
    for (auto &dim : dimensionNumbers.getRhsBatchingDimensions()) {
      rhsBatchingDimensions.push_back(dim + batchSizes.size());
    }

    for (auto &dim : dimensionNumbers.getLhsContractingDimensions()) {
      lhsContractingDimensions.push_back(dim + batchSizes.size());
    }
    for (auto &dim : dimensionNumbers.getRhsContractingDimensions()) {
      rhsContractingDimensions.push_back(dim + batchSizes.size());
    }

    auto dotDimsAttr = stablehlo::DotDimensionNumbersAttr::get(
        op.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
        lhsContractingDimensions, rhsContractingDimensions);

    SmallVector<int64_t> resultShape;
    resultShape.reserve(op.getType().getShape().size() + batchSizes.size());
    for (auto &dim : batchSizes) {
      resultShape.push_back(dim);
    }
    for (auto &dim : op.getType().getShape()) {
      resultShape.push_back(dim);
    }

    auto dotOp = builder.create<stablehlo::DotGeneralOp>(
        op.getLoc(),
        RankedTensorType::get(resultShape, op.getType().getElementType()),
        mapper.lookup(op.getLhs()), mapper.lookup(op.getRhs()), dotDimsAttr,
        op.getPrecisionConfigAttr(), op.getAlgorithmAttr());

    mapper.map(src->getResult(0), dotOp->getResult(0));
    return success();
  }
};

struct SHLOBroadcastInDimOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOBroadcastInDimOpBatchInterface,
                                             BroadcastInDimOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<BroadcastInDimOp>(src);
    auto resultType = op.getType();

    SmallVector<int64_t> bcastDims;
    for (int i = 0; i < batchSizes.size(); i++) {
      bcastDims.push_back(i);
    }
    for (auto &dim : op.getBroadcastDimensions()) {
      bcastDims.push_back(dim + batchSizes.size());
    }

    SmallVector<int64_t> resultShape;
    resultShape.reserve(resultType.getShape().size() + batchSizes.size());
    for (auto &dim : batchSizes) {
      resultShape.push_back(dim);
    }
    for (auto &dim : resultType.getShape()) {
      resultShape.push_back(dim);
    }

    auto bcastOp = builder.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(),
        RankedTensorType::get(resultShape, resultType.getElementType()),
        mapper.lookup(op.getOperand()),
        builder.getDenseI64ArrayAttr(bcastDims));

    mapper.map(src->getResult(0), bcastOp->getResult(0));
    return success();
  }
};

struct StablehloAddSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          StablehloAddSimplifyMathInterface, stablehlo::AddOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<stablehlo::AddOp>(src);

    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct StablehloSubSimplifyMathInterface
    : public MathSimplifyInterface::ExternalModel<
          StablehloSubSimplifyMathInterface, stablehlo::SubtractOp> {
  mlir::LogicalResult simplifyMath(Operation *src,
                                   PatternRewriter &rewriter) const {
    auto op = cast<stablehlo::SubtractOp>(src);

    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero())) {
      rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, op.getRhs());
      return success();
    }

    return failure();
  }
};

} // namespace

void mlir::enzyme::registerStableHLODialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            stablehlo::StablehloDialect *) {
    registerInterfaces(context);

    // SortOp::attachInterface<AutoDiffSort>(*context);

    WhileOp::attachInterface<WhileOpEnzymeOpsRemover>(*context);
    IfOp::attachInterface<IfOpEnzymeOpsRemover>(*context);

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
    ReduceWindowOp::attachInterface<AutoDiffReduceWindowRev>(*context);
    ConcatenateOp::attachInterface<AutoDiffConcatenateRev>(*context);
    BatchNormTrainingOp::attachInterface<AutoDiffBatchNormTrainingRev>(
        *context);

    ConstantOp::attachInterface<SHLOConstantOpBatchInterface>(*context);
    TransposeOp::attachInterface<SHLOTransposeOpBatchInterface>(*context);
    IfOp::attachInterface<SHLOGenericBatchOpInterface<IfOp>>(*context);
    WhileOp::attachInterface<SHLOGenericBatchOpInterface<WhileOp>>(*context);
    ReduceOp::attachInterface<SHLOReduceOpBatchInterface>(*context);
    DotGeneralOp::attachInterface<SHLODotGeneralOpBatchInterface>(*context);
    BroadcastInDimOp::attachInterface<SHLOBroadcastInDimOpBatchInterface>(
        *context);

    ReverseOp::attachInterface<SHLOGenericBatchOpInterface<ReverseOp>>(
        *context); // TODO: simpler version with newly named dims
    ScatterOp::attachInterface<SHLOGenericBatchOpInterface<ScatterOp>>(
        *context); // TODO: simpler version with newly named dims
    ConvolutionOp::attachInterface<SHLOGenericBatchOpInterface<ConvolutionOp>>(
        *context); // TODO: simpler version with newly named dims

    AddOp::attachInterface<StablehloAddSimplifyMathInterface>(*context);
    SubtractOp::attachInterface<StablehloSubSimplifyMathInterface>(*context);
  });
}
