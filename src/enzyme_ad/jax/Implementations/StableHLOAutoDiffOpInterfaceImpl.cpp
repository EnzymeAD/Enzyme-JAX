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
#include "src/enzyme_ad/jax/Implementations/SHLOGenericBatchOpInterface.h"

#include "mlir/Analysis/TopologicalSortUtils.h"

#include "llvm/ADT/PointerUnion.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include "src/enzyme_ad/jax/Utils.h"
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

Value makeIntegerConstant(Location loc, OpBuilder &builder, Type type,
                          int64_t val) {
  auto unrankedTensorType = RankedTensorType::get({}, type);
  return ConstantOp::create(
             builder, loc, unrankedTensorType,
             SplatElementsAttr::get(
                 unrankedTensorType,
                 ArrayRef<Attribute>(IntegerAttr::get(type, val))))
      .getResult();
}

static Value makeI64Constant(Location loc, OpBuilder &builder, int64_t val) {
  return makeIntegerConstant(loc, builder, builder.getI64Type(), val);
}

static Value makeI32Constant(Location loc, OpBuilder &builder, int32_t val) {
  return makeIntegerConstant(loc, builder, builder.getI32Type(), val);
}

static inline Operation *createAddRegion(Operation *op) {
  mlir::OpBuilder builder(op->getContext());
  mlir::Block *block = new Block();
  op->getRegion(0).push_back(block);
  auto elemType = cast<ShapedType>(op->getResult(0).getType()).getElementType();
  auto tensorType = RankedTensorType::get({}, elemType);
  block->addArguments({tensorType, tensorType}, {op->getLoc(), op->getLoc()});
  builder.setInsertionPointToEnd(block);
  mlir::stablehlo::ReturnOp::create(
      builder, op->getLoc(),
      mlir::stablehlo::AddOp::create(
          builder, op->getLoc(), block->getArgument(0), block->getArgument(1))
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

static inline DenseI64ArrayAttr getBroadcastInDimsAttr(OpBuilder &builder,
                                                       ArrayRef<int64_t> dims) {
  return builder.getDenseI64ArrayAttr(dims);
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
    return OpTy::create(builder, original->getLoc(), rettys, remappedOperands,
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
    return IfOp::create(builder, original->getLoc(), rettys, remappedOperands,
                        original->getAttrs());
  }
};

class AutoDiffIfRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffIfRev, IfOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto revOp =
        IfOp::create(builder, orig->getLoc(), ArrayRef<mlir::Type>{},
                     gutils->popCache(caches[0], builder), orig->getAttrs());

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

      stablehlo::ReturnOp::create(revBuilder, orig->getLoc(),
                                  ArrayRef<Value>{});
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

  enum ReverseMode { CONSTANT, CONSTANT_CHECKPOINTING, UNKNOWN };
  struct ReverseModeInfo {
    enum ReverseMode mode = UNKNOWN;
    WhileLoopInfo info;

    ReverseModeInfo(stablehlo::WhileOp op) : info(op) {}
  };

  static struct ReverseModeInfo getReverseMode(Operation *orig) {
    struct ReverseModeInfo revInfo(cast<stablehlo::WhileOp>(orig));

    if (revInfo.info.computeInfo().succeeded() && revInfo.info.isValid() &&
        revInfo.info.isConstant()) {
      const char *checkpointAttrName = "enzymexla.enable_checkpointing";
      auto enableCheckpointing =
          orig->getAttrOfType<BoolAttr>(checkpointAttrName);
      if (enableCheckpointing && enableCheckpointing.getValue())
        revInfo.mode = CONSTANT_CHECKPOINTING;
      else
        revInfo.mode = CONSTANT;
    }

    return revInfo;
  }

  static stablehlo::WhileOp makeForLoop(OpBuilder &builder, Location loc,
                                        int64_t start, int64_t limit,
                                        int64_t step, ValueRange operands) {
    return makeForLoop(builder, loc, makeI64Constant(loc, builder, start),
                       makeI64Constant(loc, builder, limit),
                       makeI64Constant(loc, builder, step), operands);
  }

  static stablehlo::WhileOp makeForLoop(OpBuilder &builder, Location loc,
                                        Value start, Value limit, Value step,
                                        ValueRange operands) {
    OpBuilder::InsertionGuard guard(builder);

    SmallVector<Value> operandsWithInduction{start};
    operandsWithInduction.append(operands.begin(), operands.end());

    auto types = ValueRange(operandsWithInduction).getTypes();
    auto whileOp =
        stablehlo::WhileOp::create(builder, loc, types, operandsWithInduction);

    SmallVector<Location> locs{loc};
    locs.reserve(operands.size() + 1);
    for (Value val : operands)
      locs.push_back(val.getLoc());

    Block *cond = builder.createBlock(&whileOp.getCond(), {}, types, locs);
    Value cmp = stablehlo::CompareOp::create(builder, loc, cond->getArgument(0),
                                             limit, ComparisonDirection::LT);
    stablehlo::ReturnOp::create(builder, loc, cmp);

    Block *body = builder.createBlock(&whileOp.getBody(), {}, types, locs);
    Value newVal =
        stablehlo::AddOp::create(builder, loc, body->getArgument(0), step);
    operandsWithInduction.assign(body->getArguments().begin(),
                                 body->getArguments().end());
    operandsWithInduction[0] = newVal;
    stablehlo::ReturnOp::create(builder, loc, operandsWithInduction);

    return whileOp;
  }

  static LogicalResult reverseWithCheckpointing(stablehlo::WhileOp orig,
                                                struct ReverseModeInfo revInfo,
                                                OpBuilder &builder,
                                                MGradientUtilsReverse *gutils,
                                                SmallVector<Value> caches,
                                                ArrayRef<bool> operandsActive) {
    int64_t numIters = revInfo.info.getConstantNumIters();
    int64_t nInner = std::sqrt(numIters);
    int64_t nOuter = nInner;
    if (nInner * nOuter != revInfo.info.getConstantNumIters()) {
      orig->emitError()
          << "Non square number of iterations for checkpointing, nInner="
          << nInner << " nOuter=" << nOuter
          << " iters=" << revInfo.info.getConstantNumIters() << "\n";
      return failure();
    }

    SetVector<Value> outsideRefs;
    getUsedValuesDefinedAbove(orig->getRegions(), outsideRefs);

    int numOutsideRefs = outsideRefs.size();
    int nargs = caches.size() + 1;
    int nrets = nargs - numOutsideRefs;

    SmallVector<Value> operands;
    for (auto [active, res] : llvm::zip(operandsActive, orig->getResults())) {
      if (active) {
        operands.push_back(gutils->diffe(res, builder));
        if (!gutils->isConstantValue(res))
          gutils->zeroDiffe(res, builder);
      }
    }

    OpBuilder::InsertionGuard guard(builder);

    stablehlo::WhileOp revOuter =
        makeForLoop(builder, orig.getLoc(), 0, nOuter, 1, operands);

    Block *revOuterBody = &revOuter.getBody().front();
    builder.setInsertionPointToStart(revOuterBody);

    Value outerStep = stablehlo::SubtractOp::create(
        builder, orig.getLoc(),
        makeI64Constant(orig.getLoc(), builder, nOuter - 1),
        revOuterBody->getArgument(0));
    Value outerStart = stablehlo::MulOp::create(
        builder, orig.getLoc(), makeI64Constant(orig.getLoc(), builder, nInner),
        outerStep);

    Value lastCache = nullptr;

    SmallVector<Value> cacheVals(nargs - 1, nullptr);
    for (int i = 0; i < nrets - 1; ++i)
      lastCache = cacheVals[i] = gutils->popCache(caches[i], builder);

    builder.setInsertionPoint(revOuter);

    IRMapping mapping;

    for (int i = nrets - 1, refIdx = 0; i < nargs - 1; ++i) {
      Value cached = cacheVals[i] = gutils->popCache(caches[i], builder);
      mapping.map(outsideRefs[refIdx], cached);
      refIdx++;
    }

    if (lastCache)
      builder.setInsertionPointAfterValue(lastCache);
    else
      builder.setInsertionPointToStart(revOuterBody);

    SmallVector<Value> carried;
    for (int i = 0; i < nrets - 1; i++) {
      carried.push_back(cacheVals[i]);
    }

    auto revInner = makeForLoop(builder, orig.getLoc(), 0, nInner, 1, carried);
    Block *revInnerBody = &revInner.getBody().front();

    revInner->setAttrs(orig->getAttrs());
    revInner->removeAttr("enzymexla.enable_checkpointing");

    auto revLoop = makeForLoop(builder, orig.getLoc(), 0, nInner, 1,
                               revOuterBody->getArguments().drop_front());
    Block *revLoopBody = &revLoop.getBody().front();

    builder.setInsertionPointToStart(revInnerBody);

    Value innerIV = stablehlo::SubtractOp::create(
        builder, orig.getLoc(),
        makeI64Constant(orig.getLoc(), builder, nInner - 1),
        revInnerBody->getArgument(0));

    Value currentStep =
        stablehlo::AddOp::create(builder, orig.getLoc(), outerStart, innerIV);
    Value currentIV = stablehlo::AddOp::create(
        builder, orig.getLoc(),
        makeI64Constant(orig.getLoc(), builder,
                        revInfo.info.getConstantStart().value()),
        stablehlo::MulOp::create(
            builder, orig.getLoc(),
            makeI64Constant(orig.getLoc(), builder,
                            revInfo.info.getConstantStep().value()),
            currentStep));

    Block *origBody = &orig.getBody().front();
    for (auto &&[origarg, revinnerarg] : llvm::zip_equal(
             origBody->getArguments(), revInnerBody->getArguments())) {
      mapping.map(origarg, revinnerarg);
      gutils->originalToNewFn.map(origarg, revinnerarg);
    }
    mapping.map(origBody->getArgument(0), currentIV);
    gutils->originalToNewFn.map(origBody->getArgument(0), currentIV);

    for (Operation &op : origBody->without_terminator()) {
      auto newOp = builder.clone(op, mapping);
      gutils->originalToNewFnOps[&op] = newOp;
      for (auto &&[oldv, newv] :
           llvm::zip(op.getResults(), newOp->getResults())) {
        gutils->originalToNewFn.map(oldv, newv);
      }
    }
    {
      auto oldTerm = cast<stablehlo::ReturnOp>(origBody->getTerminator());
      auto newTerm = cast<stablehlo::ReturnOp>(revInnerBody->getTerminator());
      SmallVector<Value> vals;
      for (auto v : oldTerm.getResults().drop_front()) {
        vals.push_back(mapping.lookupOrDefault(v));
      }
      newTerm.getResultsMutable()
          .slice(1, newTerm.getResultsMutable().size() - 1)
          .assign(vals);
    }

    gutils->originalToNewFnOps[orig] = revInner;

    builder.setInsertionPointToStart(revLoopBody);

    int revIdx = 1;
    for (auto &&[active, operand] : llvm::zip_equal(
             operandsActive, origBody->getTerminator()->getOperands())) {
      if (active) {
        gutils->addToDiffe(operand, revLoopBody->getArgument(revIdx), builder);
        revIdx++;
      }
    }

    bool anyFailed = false;

    {
      OpBuilder cacheBuilder(revInner);
      auto loc = orig->getLoc();
      auto cacheCreator = [&](Type t) {
        Value cache = enzyme::InitOp::create(cacheBuilder, loc, t);
        return std::make_pair(cache, cache);
      };
      gutils->registerCacheCreatorHook(cacheCreator);

      auto rstart = origBody->rbegin(), rend = origBody->rend();
      rstart++;
      for (auto it = rstart; it != rend; it++) {
        Operation *op = &*it;
        anyFailed |= gutils->Logic.visitChild(op, builder, gutils).failed();
      }
      gutils->deregisterCacheCreatorHook(cacheCreator);
    }

    SmallVector<Value> newResults;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, origBody->getArguments())) {
      if (active) {
        newResults.push_back(gutils->diffe(arg, builder));
        if (!gutils->isConstantValue(arg))
          gutils->zeroDiffe(arg, builder);
      }
    }

    cast<stablehlo::ReturnOp>(revLoopBody->getTerminator())
        .getResultsMutable()
        .slice(1, revLoop.getNumResults() - 1)
        .assign(newResults);

    cast<stablehlo::ReturnOp>(revOuterBody->getTerminator())
        .getResultsMutable()
        .slice(1, revOuter.getNumResults() - 1)
        .assign(revLoop.getResults().drop_front());

    builder.setInsertionPointAfter(revOuter);

    revIdx = 1;
    for (auto &&[active, arg] :
         llvm::zip_equal(operandsActive, orig->getOperands())) {
      if (active) {
        if (!gutils->isConstantValue(arg)) {
          gutils->addToDiffe(arg, revOuter->getResult(revIdx), builder);
        }
        revIdx++;
      }
    }

    return success(!anyFailed);
  }

public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto whileOp = cast<WhileOp>(orig);
    // While op has the same number of results and operands.
    // if the while is not executed (i.e. the condition is false on the first
    // evaluation), then the arguments are returned. This means that we need to
    // pass differentials for all non-constant results or non-constants
    // operands.
    SmallVector<bool> operandsActive(orig->getNumOperands(), false);
    for (int i = 0; i < operandsActive.size(); ++i) {
      operandsActive[i] =
          !gutils->isConstantValue(orig->getOperand(i)) ||
          !gutils->isConstantValue(whileOp.getCond().front().getArgument(i)) ||
          !gutils->isConstantValue(whileOp.getBody().front().getArgument(i)) ||
          !gutils->isConstantValue(orig->getResult(i));
    }

    auto revInfo = getReverseMode(orig);

    // The reverse of the while loop is a for loop where the number
    // of iterations is either known or cached from the augmented primal.
    Value numIters;
    if (revInfo.mode == CONSTANT_CHECKPOINTING) {
      return reverseWithCheckpointing(cast<stablehlo::WhileOp>(orig), revInfo,
                                      builder, gutils, caches, operandsActive);
    } else if (revInfo.mode == CONSTANT) {
      auto iterType = orig->getOperand(0).getType();
      numIters = stablehlo::ConstantOp::create(
          builder, orig->getLoc(), iterType,
          cast<ElementsAttr>(
              makeAttr(iterType, revInfo.info.getConstantNumIters())));
    } else
      numIters = gutils->popCache(caches[0], builder);

    auto unrankedTensorType = RankedTensorType::get({}, builder.getI64Type());
    auto iterVarOp = ConstantOp::create(
        builder, orig->getLoc(), unrankedTensorType,
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

    auto revWhile = WhileOp::create(builder, orig->getLoc(),
                                    ValueRange(operands).getTypes(), operands);
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
          numIters = ConstantOp::create(
              builder, orig->getLoc(), condIterVar.getType(),
              cast<ElementsAttr>(makeAttr(condIterVar.getType(),
                                          (*numAttr.begin()).getSExtValue())));
        } else {
          numIters = ConvertOp::create(builder, orig->getLoc(), numIters,
                                       condIterVarElemType);
        }
        builder.setInsertionPointAfter(revWhile);
      }

      ReturnOp::create(
          condBuilder, orig->getLoc(),
          ValueRange(CompareOp::create(condBuilder, orig->getLoc(), condIterVar,
                                       numIters, ComparisonDirection::LT)
                         .getResult()));
    }

    bool valid = true;
    {
      for (auto operand : operands) {
        body->addArgument(operand.getType(), orig->getLoc());
      }
      OpBuilder bodyBuilder(body, body->end());
      auto one = ConstantOp::create(
          bodyBuilder, orig->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(unrankedTensorType,
                                 ArrayRef<Attribute>(IntegerAttr::get(
                                     bodyBuilder.getI64Type(), 1))));
      Value bodyIterVar =
          AddOp::create(bodyBuilder, orig->getLoc(), body->getArgument(0), one);

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
          gutils->addToDiffe(operand, body->getArgument(revIdx), bodyBuilder);
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

      ReturnOp::create(bodyBuilder, orig->getLoc(), newResults);
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
      if (info.isValid() && info.isConstant()) {

        // for any value that is a reference from the outside we can hoist the
        // push/pop from outside the outer really.

        if (getReverseMode(orig).mode == CONSTANT_CHECKPOINTING) {
          OpBuilder builder(newWhile);

          SetVector<Value> outsideRefs;
          getUsedValuesDefinedAbove(orig->getRegions(), outsideRefs);
          SmallVector<Value> caches;

          // sqrt scheme
          int64_t nInner = std::sqrt(info.getConstantNumIters());
          int64_t nOuter = nInner;

          if (nInner * nOuter != info.getConstantNumIters()) {
            orig->emitError()
                << "Non square number of iterations for checkpointing, nInner="
                << nInner << " nOuter=" << nOuter
                << " iters=" << info.getConstantNumIters() << "\n";
          } else {
            auto outer = makeForLoop(builder, orig->getLoc(), 0, nOuter, 1,
                                     newWhile->getOperands().slice(
                                         1, newWhile->getNumOperands() - 1));

            Block *outerBody = &outer.getBody().front();
            builder.setInsertionPointToStart(outerBody);

            Value outerIV = stablehlo::MulOp::create(
                builder, newWhile.getLoc(), outerBody->getArgument(0),
                makeI64Constant(newWhile.getLoc(), builder, nOuter));

            for (auto arg : outerBody->getArguments().slice(1)) {
              caches.push_back(gutils->initAndPushCache(arg, builder));
            }

            builder.setInsertionPoint(outer);

            for (auto ref : outsideRefs) {
              caches.push_back(gutils->initAndPushCache(
                  gutils->getNewFromOriginal(ref), builder));
            }

            builder.setInsertionPointAfterValue(outerIV);

            SmallVector<Value> operands(
                outerBody->getArguments().slice(1).begin(),
                outerBody->getArguments().slice(1).end());
            auto inner =
                makeForLoop(builder, orig->getLoc(), 0, nInner, 1, operands);

            outerBody->getTerminator()->setOperands(
                1, inner.getNumResults() - 1,
                inner.getResults().slice(1, inner.getNumResults() - 1));

            Block *innerBody = &inner.getBody().front();
            Block *oldInnerBody = &newWhile.getBody().front();
            builder.setInsertionPointToStart(innerBody);

            IRMapping mapping;

            for (auto [oldArg, newArg] : llvm::zip_equal(
                     oldInnerBody->getArguments(), innerBody->getArguments())) {
              mapping.map(oldArg, newArg);
            }

            Value oldIV = oldInnerBody->getArgument(0);
            Value newIV = stablehlo::AddOp::create(
                builder, oldIV.getLoc(), innerBody->getArgument(0), outerIV);

            mapping.map(oldIV, newIV);

            for (Operation &innerOp : oldInnerBody->without_terminator()) {
              builder.clone(innerOp, mapping);
            }

            SmallVector<Value> newReturns;
            for (auto oldRes :
                 oldInnerBody->getTerminator()->getOperands().slice(
                     1, oldInnerBody->getTerminator()->getNumOperands() - 1)) {
              newReturns.push_back(mapping.lookupOrDefault(oldRes));
            }
            Operation *term = innerBody->getTerminator();
            term->setOperands(1, term->getNumOperands() - 1, newReturns);

            builder.setInsertionPointAfter(outer);
            SmallVector<Value> newResults{makeI64Constant(
                oldIV.getLoc(), builder, *info.getConstantLimit())};
            newResults.append(
                outer->getResults()
                    .slice(1, outer->getNumResults() - 1)
                    .begin(),
                outer->getResults().slice(1, outer->getNumResults() - 1).end());

            gutils->replaceOrigOpWith(orig, newResults);
            gutils->erase(newWhile);
            gutils->originalToNewFnOps[orig] = outer;

            return caches;
          }
        }

        return {};
      }

      numIters = info.getNumIters(revBuilder);
    }

    if (!numIters) {
      auto cond = &newWhile.getCond().front();
      auto body = &newWhile.getBody().front();

      auto unrankedTensorType = RankedTensorType::get({}, elementType);
      auto numItersInit =
          ConstantOp::create(
              revBuilder, orig->getLoc(), unrankedTensorType,
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
      auto one = ConstantOp::create(
          inBodyBuilder, orig->getLoc(), unrankedTensorType,
          SplatElementsAttr::get(
              unrankedTensorType,
              ArrayRef<Attribute>(IntegerAttr::get(elementType, 1))));
      numItersInBlock = AddOp::create(inBodyBuilder, orig->getLoc(),
                                      numItersInBlock, one.getResult());
      auto term = body->getTerminator();
      term->insertOperands(term->getNumOperands(), ValueRange(numItersInBlock));

      SmallVector<Type> resultTypes(newWhile->getResultTypes().begin(),
                                    newWhile->getResultTypes().end());
      resultTypes.push_back(numItersInit.getType());

      auto newnewWhile = WhileOp::create(revBuilder, orig->getLoc(),
                                         resultTypes, newWhile->getOperands());
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

    auto red = ReduceOp::create(builder, op.getLoc(),
                                TypeRange(gutils->getShadowType(reduceTy)),
                                inDiffe, zero, reducedDims);
    red.getBody().push_back(new Block());
    Block &body = red.getBody().front();
    OpBuilder bodyBuilder(orig->getContext());
    bodyBuilder.setInsertionPointToEnd(&body);

    body.addArgument(bodyTy, op.getLoc());
    body.addArgument(bodyTy, op.getLoc());
    auto add = AddOp::create(bodyBuilder, op.getLoc(), body.getArgument(0),
                             body.getArgument(1));
    ReturnOp::create(bodyBuilder, op.getLoc(), ValueRange(add));

    // for simplicity we do grad -> reduce -> reshape (restore 1 dims) ->
    // transpose -> reshape
    // The repeated reshapes are then eliminated via `enzyme-hlo-opt`.
    auto reshapedRed = ReshapeOp::create(
        builder, op.getLoc(),
        RankedTensorType::get(reshapedShape, inTy.getElementType()),
        red->getResult(0));
    auto transposedVal =
        TransposeOp::create(builder, op.getLoc(), reshapedRed, perm);
    auto res = ReshapeOp::create(
        builder, op.getLoc(), gutils->getShadowType(op.getOperand().getType()),
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
    auto red = stablehlo::PadOp::create(
        builder, op.getLoc(), inDiffe, zeroPad,
        builder.getDenseI64ArrayAttr(starts),
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
  auto addOp = AddOp::create(builder, loc, a, b);
  stablehlo::ReturnOp::create(builder, loc, addOp.getResult());
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
          stablehlo::PadOp::create(builder, op.getLoc(), inDiffe, zero,
                                   getI64Attr(builder, paddingHigh),
                                   getI64Attr(builder, paddingLow),
                                   getI64Attr(builder, paddingInterior))
              .getResult();

      auto revOp = stablehlo::ReduceWindowOp::create(
          builder, op.getLoc(), operandType, paddedIndiffe,
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

        auto cmpOp = CompareOp::create(
            builder, op.getLoc(), select->getArgument(0),
            select->getArgument(1),
            ismax ? ComparisonDirection::GE : ComparisonDirection::LE);
        ReturnOp::create(builder, op.getLoc(), cmpOp.getResult());
      }

      auto revOp = SelectAndScatterOp::create(
          builder, op.getLoc(), op.getOperand(0).getType(),
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

        bcast = BroadcastInDimOp::create(
            builder, op.getLoc(), gutils->getShadowType(inTy), inDiffe,
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
        auto bc = BroadcastInDimOp::create(builder, op.getLoc(),
                                           oprev.getType(), ores, attr);

        auto cmp = CompareOp::create(builder, op.getLoc(), bc, oprev,
                                     ComparisonDirection::EQ);

        auto bc2 = BroadcastInDimOp::create(builder, op.getLoc(),
                                            oprev.getType(), inDiffe, attr);

        auto res = SelectOp::create(builder, op.getLoc(), cmp, bc2, zero);
        gutils->addToDiffe(op.getInputs()[0], res, builder);
      }
      if (!gutils->isConstantValue(op.getInitValues()[0])) {
        auto oprev = gutils->getNewFromOriginal(op.getInitValues()[0]);

        auto zeroI = cast<AutoDiffTypeInterface>(inDiffe.getType())
                         .createNullValue(builder, op.getLoc());

        auto cmp = CompareOp::create(builder, op.getLoc(), ores, oprev,
                                     ComparisonDirection::EQ);

        auto res = SelectOp::create(builder, op.getLoc(), cmp, inDiffe, zeroI);
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
      auto res = SliceOp::create(
          builder, op.getLoc(), RankedTensorType::get(tys, RT.getElementType()),
          inDiffe, start, limit, strides);
      auto res2 = ReshapeOp::create(builder, op.getLoc(), inTy, res);
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

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto constOp = cast<ConstantOp>(src);

    auto T = cast<TensorType>(constOp.getType());
    SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
    shape.append(T.getShape().begin(), T.getShape().end());
    auto Ty = T.clone(shape);

    // If splatted attr then we can easily batch it
    auto eattr = cast<DenseElementsAttr>(constOp.getValue());
    if (eattr.isSplat()) {
      auto splatAttr = cast<SplatElementsAttr>(constOp.getValue());
      auto newSplattedConstOp = ConstantOp::create(
          builder, constOp->getLoc(), Ty,
          cast<ElementsAttr>(splatAttr.resizeSplat(cast<ShapedType>(Ty))));
      mapper.map(src->getResult(0), newSplattedConstOp->getResult(0));
      return success();
    }

    // otherwise do a broadcast in dim
    SmallVector<int64_t> mapping(T.getShape().size());
    std::iota(mapping.begin(), mapping.end(), batchSizes.size());

    auto constOpCloned = builder.clone(*constOp);
    auto bcastOp = BroadcastInDimOp::create(
        builder, src->getLoc(), Ty, constOpCloned->getResult(0),
        builder.getDenseI64ArrayAttr(mapping));
    mapper.map(src->getResult(0), bcastOp->getResult(0));
    return success();
  }
};

struct SHLOGetDimensionSizeOpBatchInterface
    : public BatchOpInterface::ExternalModel<
          SHLOGetDimensionSizeOpBatchInterface, GetDimensionSizeOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto getDimSizeOp = cast<GetDimensionSizeOp>(src);

    auto newOp = GetDimensionSizeOp::create(
        builder, src->getLoc(), mapper.lookup(getDimSizeOp.getOperand()),
        cast<IntegerAttr>(getDimSizeOp.getDimensionAttr()).getInt() +
            batchSizes.size());
    auto bcastOp = BroadcastInDimOp::create(
        builder, src->getLoc(),
        RankedTensorType::get(
            batchSizes, cast<RankedTensorType>(newOp->getResult(0).getType())
                            .getElementType()),
        newOp->getResult(0), builder.getDenseI64ArrayAttr({}));
    mapper.map(src->getResult(0), bcastOp->getResult(0));
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
    for (auto &&[res, arg, barg] :
         llvm::zip(srt.getResults(), term->getOperands(),
                   srt.getBody().front().getArguments())) {
      if (arg == v) {
        sv.push_back(res);
        sv.push_back(barg);
      }
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

class AutoDiffScatterFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffScatterFwd, ScatterOp> {
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

    auto replacement = ScatterOp::create(
        builder, scat.getLoc(), ResultTypes, Inputs,
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

class AutoDiffScatterRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffScatterRev,
                                                       ScatterOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto scatterOp = cast<ScatterOp>(op);

    auto checkCommonScatterOp =
        mlir::stablehlo::CheckCommonScatterOp(scatterOp);

    if (!checkCommonScatterOp.isSetindexScatter &&
        !checkCommonScatterOp.isAddScatter) {
      op->emitError("AutoDiffScatterRev only supports Setindex "
                    "and AddScatter operations");
      return failure();
    }

    SmallVector<Value> outputDiffe;
    outputDiffe.reserve(scatterOp.getNumResults());
    for (int i = 0; i < scatterOp.getNumResults(); i++) {
      outputDiffe.push_back(gutils->diffe(scatterOp.getResult(i), builder));
      gutils->zeroDiffe(scatterOp.getResult(i), builder);
    }

    auto scatterIndices = gutils->popCache(caches[0], builder);

    auto gatherDims = stablehlo::getGatherDims(
        scatterOp->getContext(), scatterOp.getScatterDimensionNumbers());
    auto gatherSliceSizes = builder.getDenseI64ArrayAttr(
        stablehlo::computeGatherSliceSizes(scatterOp));

    if (checkCommonScatterOp.isAddScatter) {
      createScatterAddGradientInputs(scatterOp, gutils, scatterIndices,
                                     gatherDims, gatherSliceSizes, outputDiffe,
                                     builder);
    } else {
      createScatterSetindexGradientInputs(scatterOp, gutils, scatterIndices,
                                          gatherDims, gatherSliceSizes,
                                          outputDiffe, builder);
    }

    createGradientUpdates(scatterOp, gutils, scatterIndices, gatherDims,
                          gatherSliceSizes, outputDiffe, builder);
    return success();
  }

  void createScatterAddGradientInputs(
      stablehlo::ScatterOp scatterOp, MGradientUtilsReverse *gutils,
      Value scatterIndices,
      stablehlo::GatherDimensionNumbersAttr gatherDimNumbers,
      DenseI64ArrayAttr gatherSliceSizes, SmallVector<Value> outputDiffe,
      OpBuilder &builder) const {
    for (auto [i, operand] : llvm::enumerate(scatterOp.getInputs())) {
      if (!gutils->isConstantValue(operand)) {
        auto updateDiffe = stablehlo::GatherOp::create(
            builder, scatterOp.getLoc(), outputDiffe[i], scatterIndices,
            gatherDimNumbers, gatherSliceSizes,
            scatterOp.getIndicesAreSortedAttr());
        gutils->addToDiffe(operand, updateDiffe, builder);
      }
    }
    return;
  }

  void createScatterSetindexGradientInputs(
      stablehlo::ScatterOp scatterOp, MGradientUtilsReverse *gutils,
      Value scatterIndices,
      stablehlo::GatherDimensionNumbersAttr gatherDimNumbers,
      DenseI64ArrayAttr gatherSliceSizes, SmallVector<Value> outputDiffe,
      OpBuilder &builder) const {
    auto zeroUpdateType = scatterOp.getUpdates()[0].getType();
    auto zeroUpdate = stablehlo::ConstantOp::create(
        builder, scatterOp.getLoc(), zeroUpdateType,
        cast<ElementsAttr>(makeAttr(zeroUpdateType, 0)));

    auto elemType = cast<RankedTensorType>(zeroUpdateType).getElementType();
    auto zeroScalar = stablehlo::ConstantOp::create(
        builder, scatterOp.getLoc(), RankedTensorType::get({}, elemType),
        cast<ElementsAttr>(makeAttr(RankedTensorType::get({}, elemType), 0)));

    // gradient of the inputs
    SmallVector<Value> selectedOutputDiffe, newScatterUpdates;
    SmallVector<Type> selectedOutputTypes;
    for (auto [i, operand] : llvm::enumerate(scatterOp.getInputs())) {
      if (!gutils->isConstantValue(operand)) {
        selectedOutputDiffe.push_back(outputDiffe[i]);
        newScatterUpdates.push_back(zeroUpdate);
        selectedOutputTypes.push_back(
            cast<RankedTensorType>(outputDiffe[i].getType()));
      }
    }
    int64_t nNonConsts = selectedOutputDiffe.size();

    if (nNonConsts > 0) {
      auto newScatterOp = stablehlo::ScatterOp::create(
          builder, scatterOp.getLoc(), selectedOutputTypes, selectedOutputDiffe,
          scatterIndices, newScatterUpdates,
          scatterOp.getScatterDimensionNumbersAttr(),
          scatterOp.getIndicesAreSortedAttr(),
          scatterOp.getUniqueIndicesAttr());

      auto &updateRegion = newScatterOp.getUpdateComputation();
      auto *block = builder.createBlock(&updateRegion);
      auto argType = RankedTensorType::get({}, elemType);

      for (int i = 0; i < 2 * nNonConsts; i++)
        block->addArgument(argType, scatterOp.getLoc());

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(block);

        SmallVector<Value> returnValues;
        for (int i = nNonConsts; i < 2 * nNonConsts; i++)
          returnValues.push_back(zeroScalar);

        stablehlo::ReturnOp::create(builder, scatterOp.getLoc(), returnValues);
      }

      builder.setInsertionPointAfter(newScatterOp);

      int64_t counter = 0;
      for (auto [i, operand] : llvm::enumerate(scatterOp.getInputs())) {
        if (!gutils->isConstantValue(operand)) {
          gutils->addToDiffe(operand, newScatterOp.getResult(counter++),
                             builder);
        }
      }
    }
    return;
  }

  void
  createGradientUpdates(stablehlo::ScatterOp scatterOp,
                        MGradientUtilsReverse *gutils, Value scatterIndices,
                        stablehlo::GatherDimensionNumbersAttr gatherDimNumbers,
                        DenseI64ArrayAttr gatherSliceSizes,
                        SmallVector<Value> outputDiffe,
                        OpBuilder &builder) const {
    for (auto [i, update] : llvm::enumerate(scatterOp.getUpdates())) {
      if (!gutils->isConstantValue(update)) {
        auto updateDiffe = stablehlo::GatherOp::create(
            builder, scatterOp.getLoc(), outputDiffe[i], scatterIndices,
            gatherDimNumbers, gatherSliceSizes,
            scatterOp.getIndicesAreSortedAttr());
        gutils->addToDiffe(update, updateDiffe, builder);
      }
    }
    return;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto scatterOp = cast<ScatterOp>(orig);

    bool anyNonConst = false;

    for (auto input : scatterOp.getInputs()) {
      if (!gutils->isConstantValue(input)) {
        anyNonConst = true;
        break;
      }
    }

    for (auto update : scatterOp.getUpdates()) {
      if (!gutils->isConstantValue(update)) {
        anyNonConst = true;
        break;
      }
    }

    if (anyNonConst) {
      SmallVector<Value> caches;

      Operation *newOp = gutils->getNewFromOriginal(scatterOp);
      OpBuilder cacheBuilder(newOp);

      Value scatterIndicesCached = gutils->initAndPushCache(
          gutils->getNewFromOriginal(scatterOp.getScatterIndices()),
          cacheBuilder);
      caches.push_back(scatterIndicesCached);

      return caches;
    }

    return {};
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

stablehlo::SortOp
constructSortOpWithExtraOperands(OpBuilder &builder, stablehlo::SortOp original,
                                 SmallVectorImpl<Value> &newOperands) {
  auto newSortOp = stablehlo::SortOp::create(
      builder, original.getLoc(), newOperands, original.getDimensionAttr(),
      original.getIsStableAttr());

  IRMapping regionMapper;
  auto &newComparator = newSortOp.getComparator();
  auto *newBlock = new Block();
  newComparator.push_back(newBlock);

  {
    SmallVector<Type> scalarArgTys;
    for (auto arg : newOperands) {
      auto elTy = RankedTensorType::get(
          {}, cast<TensorType>(arg.getType()).getElementType());
      scalarArgTys.push_back(elTy);
      scalarArgTys.push_back(elTy);
    }
    newBlock->addArguments(
        scalarArgTys,
        SmallVector<Location>(scalarArgTys.size(), original.getLoc()));
  }

  auto &origComparator = original.getComparator();
  auto &origBlock = origComparator.front();

  IRMapping mapper;
  for (int64_t i = 0; i < origBlock.getNumArguments(); i++)
    mapper.map(origBlock.getArgument(i), newBlock->getArgument(i));

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(newBlock);
    for (Operation &origOpInside : origBlock) {
      builder.clone(origOpInside, mapper);
    }
  }

  return newSortOp;
}

class AutoDiffSortFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffSortFwd, SortOp> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    if (gutils->width > 1) {
      op->emitError(
          "TODO: AutoDiffSortFwd does not support batched forward mode");
      return failure();
    }

    auto sortOp = cast<stablehlo::SortOp>(op);

    DenseMap<int32_t, int32_t> gradMapping;

    SmallVector<Value> newOperands;
    for (auto operand : sortOp.getInputs()) {
      newOperands.push_back(gutils->getNewFromOriginal(operand));
    }
    for (auto [i, operand] : llvm::enumerate(sortOp.getInputs())) {
      if (!gutils->isConstantValue(operand)) {
        newOperands.push_back(gutils->invertPointerM(operand, builder));
        gradMapping[i] = newOperands.size() - 1;
      }
    }

    auto newSortOp =
        constructSortOpWithExtraOperands(builder, sortOp, newOperands);

    SmallVector<Value> replacementResults(sortOp.getNumResults());
    for (int32_t i = 0; i < sortOp.getNumResults(); i++) {
      replacementResults[i] = newSortOp.getResults()[i];
      auto origRes = sortOp.getResults()[i];
      if (!gutils->isConstantValue(origRes)) {
        int32_t j = gradMapping[i];
        gutils->setDiffe(origRes, newSortOp.getResults()[j], builder);
      }
    }

    gutils->replaceOrigOpWith(op, replacementResults);
    gutils->originalToNewFnOps[op] = newSortOp;
    gutils->eraseIfUnused(op);
    return success();
  }
};

class AutoDiffSortRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffSortRev,
                                                       stablehlo::SortOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto sortOp = cast<stablehlo::SortOp>(orig);

    if (gutils->width > 1) {
      orig->emitError(
          "TODO: AutoDiffSortRev does not support batched reverse mode");
      return failure();
    }

    auto indices = gutils->popCache(caches[0], builder);
    auto indicesTy = cast<RankedTensorType>(indices.getType());

    SmallVector<int64_t> newIndicesShape(indicesTy.getShape().begin(),
                                         indicesTy.getShape().end());
    newIndicesShape.push_back(1);

    indices = stablehlo::ReshapeOp::create(
        builder, orig->getLoc(),
        RankedTensorType::get(newIndicesShape, indicesTy.getElementType()),
        indices);

    auto inTy = cast<RankedTensorType>(orig->getOperand(0).getType());
    auto inRank = inTy.getRank();

    SmallVector<int64_t> batchingDims;
    for (int32_t d = 0; d < inRank; d++) {
      if (d != sortOp.getDimension()) {
        batchingDims.push_back(d);
      }
    }

    auto scatterDims = stablehlo::ScatterDimensionNumbersAttr::get(
        orig->getContext(), SmallVector<int64_t>(),
        SmallVector<int64_t>{static_cast<int64_t>(sortOp.getDimension())},
        batchingDims, batchingDims,
        SmallVector<int64_t>{static_cast<int64_t>(sortOp.getDimension())},
        indicesTy.getRank());

    for (size_t i = 0; i < orig->getNumResults(); i++) {
      if (gutils->isConstantValue(orig->getResult(i)) ||
          gutils->isConstantValue(orig->getOperand(i)))
        continue;

      // we compute the gradients with scatter_add and then set the original
      auto inDiffe = gutils->diffe(orig->getResult(i), builder);
      auto inDiffeTy = cast<RankedTensorType>(inDiffe.getType());
      gutils->zeroDiffe(orig->getResult(i), builder);

      auto outDiffe = gutils->diffe(orig->getOperand(i), builder);

      Region combiner;
      {
        Block *block = new Block();
        combiner.push_back(block);
        block->addArgument(
            RankedTensorType::get({}, inDiffeTy.getElementType()),
            orig->getLoc());
        block->addArgument(
            RankedTensorType::get({}, inDiffeTy.getElementType()),
            orig->getLoc());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(block);
        stablehlo::ReturnOp::create(
            builder, orig->getLoc(),
            ValueRange{stablehlo::AddOp::create(builder, orig->getLoc(),
                                                block->getArgument(0),
                                                block->getArgument(1))});
      }

      auto scatterOp = stablehlo::ScatterOp::create(
          builder, orig->getLoc(), outDiffe, indices, inDiffe, scatterDims,
          builder.getBoolAttr(false), builder.getBoolAttr(true));
      scatterOp.getUpdateComputation().takeBody(combiner);

      gutils->setDiffe(orig->getOperand(i), scatterOp.getResults()[0], builder);
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    auto sortOp = cast<stablehlo::SortOp>(orig);

    if (gutils->width > 1)
      return {};

    bool allConstant = true;
    for (auto input : sortOp.getInputs()) {
      if (!gutils->isConstantValue(input)) {
        allConstant = false;
        break;
      }
    }

    if (allConstant)
      return {};

    auto newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);

    SmallVector<Value> newOperands(sortOp.getInputs().size() + 1);
    for (auto [i, operand] : llvm::enumerate(sortOp.getInputs())) {
      newOperands[i] = gutils->getNewFromOriginal(operand);
    }
    auto OpTy = cast<TensorType>(newOperands[0].getType());
    auto iotaOp = stablehlo::IotaOp::create(
        cacheBuilder, orig->getLoc(),
        RankedTensorType::get(OpTy.getShape(),
                              cacheBuilder.getIntegerType(32, false)),
        sortOp.getDimensionAttr());
    newOperands[newOperands.size() - 1] = iotaOp.getResult();

    auto newSortOp =
        constructSortOpWithExtraOperands(cacheBuilder, sortOp, newOperands);
    auto newResults = newSortOp.getResults();

    SmallVector<Value> caches;
    caches.push_back(gutils->initAndPushCache(newResults[newResults.size() - 1],
                                              cacheBuilder));

    SmallVector<Value> replacements;
    for (size_t i = 0; i < newResults.size() - 1; i++) {
      replacements.push_back(newResults[i]);
    }

    gutils->replaceOrigOpWith(orig, replacements);
    gutils->eraseIfUnused(orig);
    gutils->originalToNewFnOps[orig] = newSortOp;

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
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
      auto inDiffe = gutils->diffe(op->getResult(0), builder);
      gutils->zeroDiffe(op->getResult(0), builder);

      auto opOperand0 = gutils->popCache(caches[0], builder);
      auto opOperand1 = gutils->popCache(caches[1], builder);
      auto opResult1 = gutils->getNewFromOriginal(op->getResult(1));
      auto opResult2 = gutils->getNewFromOriginal(op->getResult(2));

      auto gradOp = BatchNormGradOp::create(
          builder, op->getLoc(), opOperand0, opOperand1, opResult1, opResult2,
          inDiffe, op.getEpsilonAttr(), op.getFeatureIndexAttr());

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
public:
  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {
    auto whileOp = cast<stablehlo::WhileOp>(op);
    stablehlo::WhileOp otherWhileOp; // where cache pops are

    Block *body = &whileOp.getBody().front();
    Block *cond = &whileOp.getCond().front();

    // Gradients whose values need to be passed as iteration variables.
    llvm::SetVector<Value> updatedGradients;

    llvm::MapVector<Value, CacheInfo> cachesMap;

    if (op->walk([&](enzyme::SetOp sub) {
            if (sub->getParentOp() != op) {
              llvm::errs() << " paren: " << *sub->getParentOp() << "\n";
              llvm::errs() << "op: " << *op << "\n";
              llvm::errs() << "sub: " << sub << "\n";
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          }).wasInterrupted()) {
      return rewriter.notifyMatchFailure(
          op, "had set op which was not a direct descendant");
    }
    if (op->walk([&](enzyme::GetOp sub) {
            if (sub->getParentOp() != op) {
              llvm::errs() << " paren: " << *sub->getParentOp() << "\n";
              llvm::errs() << "op: " << *op << "\n";
              llvm::errs() << "sub: " << sub << "\n";
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          }).wasInterrupted()) {
      return rewriter.notifyMatchFailure(
          op, "had get op which was not a direct descendant");
    }

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

    if (info.computeInfo().failed() || !info.isValid()) {
      return rewriter.notifyMatchFailure(
          op, "WhileOp does not have known iteration count for cache removal");
    }

    DenseMap<Value, SmallVector<Operation *>> updatedGradientUsers;

    // 1. Move enzyme.get outside the body if the variable is not used outside
    // the loop
    for (auto &it : *body) {
      Operation *op = &it;

      auto getOp = dyn_cast<enzyme::GetOp>(op);
      if (getOp && updatedGradients.contains(getOp.getGradient())) {
        updatedGradientUsers[getOp.getGradient()].push_back(getOp);
      } else if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
        updatedGradientUsers[setOp.getGradient()].push_back(setOp);
      }

      if (!getOp || updatedGradients.contains(getOp.getGradient()))
        continue;

      auto outerGet = enzyme::GetOp::create(
          rewriter, getOp->getLoc(),
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
      auto outerGet = enzyme::GetOp::create(rewriter, grad.getLoc(), Ty, grad);

      newOperands.push_back(outerGet.getResult());
      auto newArg = body->addArgument(Ty, grad.getLoc());
      cond->addArgument(Ty, grad.getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);
        // here we do a primitive form of mem2reg within the loop. We have a
        // sorted (by instruction number) list of all users of the instruction.
        Value val = newArg;
        for (auto user : updatedGradientUsers[grad]) {
          if (auto getOp = dyn_cast<enzyme::GetOp>(user)) {
            rewriter.replaceOp(getOp, val);
          } else {
            auto setOp = cast<enzyme::SetOp>(user);
            val = setOp.getValue();
            rewriter.eraseOp(setOp);
          }
        }

        term->insertOperands(term->getNumOperands(), ValueRange(val));
      }
    }

    // 3. For enzyme.cache, the iteration number needs to be known. Given a
    // while loop with N iterations. For each of these cache, generate a
    // batched tensor with N prepended. Cache pushes become
    // dynamic_update_slice and cache pops become dynamic_slice.
    auto numIters =
        info.isConstant() ? info.getConstantNumIters() : ShapedType::kDynamic;

    Value inductionVariable; // [0,..., N - 1] counter from within the loop

    if (matchPattern(info.getStart(), m_Zero()) && info.isStepOne()) {
      inductionVariable = body->getArgument(0);
    }

    auto zero = makeI64Constant(whileOp->getLoc(), rewriter, 0);

    // Run min cut partitioning to limit the amount of values to be cached.
    if (hasMinCut(whileOp) && caches.size()) {
      Block *forward = &whileOp.getBody().front();
      Block *reverse = &otherWhileOp.getBody().front();
      Operation *lastFwd = nullptr;
      IRMapping fwdrevmap;
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(reverse);
      mlir::enzyme::minCutCache(forward, reverse, caches, rewriter, fwdrevmap,
                                lastFwd);
    }

    Value itersV = nullptr;

    for (auto &cinfo : caches) {
      Value cache = cinfo.initOp.getResult();

      // push does not depend on a value inside the loop, we can hoist the
      // push/pop before the for loops.
      if (cinfo.pushedValue().getParentRegion() != &whileOp.getBody()) {
        auto newPush = enzyme::PushOp::create(rewriter, cache.getLoc(), cache,
                                              cinfo.pushedValue());
        rewriter.eraseOp(cinfo.pushOp);
        cinfo.pushOp = newPush;

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(cinfo.popOp->getParentOp());

          auto popVal = cinfo.popOp.getResult();
          auto newPop = enzyme::PopOp::create(rewriter, cache.getLoc(),
                                              popVal.getType(), cache);
          rewriter.replaceAllUsesWith(popVal, newPop.getResult());
          rewriter.eraseOp(cinfo.popOp);
          cinfo.popOp = newPop;
        }

        continue;
      }

      if (!inductionVariable) {
        return rewriter.notifyMatchFailure(
            op, "WhileOp does not have induction variable for cache removal");
      }

      auto newType =
          cast<ShapedType>(cast<AutoDiffTypeInterface>(cinfo.cachedType())
                               .getShadowType(numIters));

      Value initValue;
      if (info.isConstant()) {
        initValue = cast<AutoDiffTypeInterface>(newType).createNullValue(
            rewriter, cinfo.initOp->getLoc());
      } else {
        if (!itersV)
          itersV = info.getNumIters(rewriter);
        SmallVector<int64_t> zeros = llvm::to_vector(newType.getShape());
        for (auto &v : zeros) {
          if (v == ShapedType::kDynamic)
            v = 0;
        }
        auto op = cast<AutoDiffTypeInterface>(
                      RankedTensorType::get(zeros, newType.getElementType()))
                      .createNullValue(rewriter, cinfo.initOp->getLoc());

        auto zeroOp = cast<AutoDiffTypeInterface>(
                          RankedTensorType::get(ArrayRef<int64_t>(),
                                                newType.getElementType()))
                          .createNullValue(rewriter, cinfo.initOp->getLoc());

        auto zeroInt = stablehlo::ConstantOp::create(
            rewriter, cinfo.initOp->getLoc(), itersV.getType(),
            cast<ElementsAttr>(makeAttr(itersV.getType(), 0)));

        auto ST = RankedTensorType::get(
            zeros.size(),
            cast<RankedTensorType>(itersV.getType()).getElementType());
        auto starts =
            stablehlo::ConstantOp::create(rewriter, cinfo.initOp->getLoc(), ST,
                                          cast<ElementsAttr>(makeAttr(ST, 0)));
        auto ints = starts;

        int64_t padStart[] = {0};
        int64_t padEnd[] = {(int64_t)zeros.size() - 1};
        auto iterRS = stablehlo::ReshapeOp::create(
            rewriter, cinfo.initOp->getLoc(),
            RankedTensorType::get(
                {1}, cast<TensorType>(itersV.getType()).getElementType()),
            itersV);
        Value ends = stablehlo::PadOp::create(rewriter, cinfo.initOp->getLoc(),
                                              starts.getType(), iterRS, zeroInt,
                                              padStart, padEnd, padStart);

        initValue = stablehlo::DynamicPadOp::create(
            rewriter, cinfo.initOp->getLoc(), newType, op, zeroOp, starts, ends,
            ints);
      }

      newOperands.push_back(initValue);

      auto cacheValue = body->addArgument(newType, cinfo.pushOp->getLoc());
      cond->addArgument(newType, cinfo.pushOp->getLoc());

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(cinfo.pushOp);

        Value newCacheValue;
        if (auto TT = dyn_cast<TensorType>(cinfo.cachedType())) {
          auto shape = TT.getShape();

          SmallVector<Value> startIndices(shape.size() + 1, zero);
          startIndices[0] = inductionVariable;

          SmallVector<int64_t> updateShape;
          updateShape.push_back(1);
          updateShape.append(shape.begin(), shape.end());
          Value reshapedUpdate = stablehlo::ReshapeOp::create(
              rewriter, cinfo.pushOp->getLoc(), TT.clone(updateShape),
              cinfo.pushOp.getValue());

          newCacheValue = stablehlo::DynamicUpdateSliceOp::create(
              rewriter, cinfo.pushOp->getLoc(), cacheValue, reshapedUpdate,
              startIndices);
        } else {
          assert(false && "todo");
          // newCacheValue = tensor::InsertOp::create(rewriter,
          //     info.pushOp->getLoc(), info.pushOp.getValue(), cacheValue,
          //     inductionVariable);
        }

        term->insertOperands(term->getNumOperands(), ValueRange(newCacheValue));
      }
    }

    auto numInitArgs = whileOp->getNumOperands();
    auto newWhile =
        stablehlo::WhileOp::create(rewriter, op->getLoc(), newOperands);

    newWhile.getCond().takeBody(whileOp.getCond());
    newWhile.getBody().takeBody(whileOp.getBody());

    unsigned resultIdx = numInitArgs;
    for (auto grad : updatedGradients) {
      // set the updated gradient after the new for op.
      OpBuilder::InsertionGuard guard(rewriter);
      enzyme::SetOp::create(rewriter, grad.getLoc(), grad,
                            newWhile->getResult(resultIdx));
      ++resultIdx;
    }

    for (auto &&[res, newRes] :
         llvm::zip(whileOp->getResults(), newWhile->getResults())) {
      rewriter.replaceAllUsesWith(res, newRes);
    }

    // 4. On the other while op (the one containing the pops), we add an
    // induction variable and replace pops with slice from the tensor version
    // of the cache.
    if (inductionVariable && caches.size() != 0) {
      if (isa<BlockArgument>(inductionVariable) &&
          cast<BlockArgument>(inductionVariable).getArgNumber() != 0)
        resultIdx++;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(otherWhileOp);
      SmallVector<Value> operands(otherWhileOp->getOperands().begin(),
                                  otherWhileOp->getOperands().end());
      if (info.isConstant()) {
        operands.push_back(
            makeI64Constant(otherWhileOp->getLoc(), rewriter, numIters - 1));
      } else {
        if (!itersV)
          itersV = info.getNumIters(rewriter);
        auto one = stablehlo::ConstantOp::create(
            rewriter, otherWhileOp->getLoc(), itersV.getType(),
            cast<ElementsAttr>(makeAttr(itersV.getType(), 1)));
        auto sub = stablehlo::SubtractOp::create(
            rewriter, otherWhileOp->getLoc(), itersV, one);
        operands.push_back(sub);
      }

      Block *otherBody = &otherWhileOp.getBody().front();
      Block *otherCond = &otherWhileOp.getCond().front();
      Value otherInductionVariable = otherBody->addArgument(
          operands.back().getType(), otherWhileOp->getLoc());
      otherCond->addArgument(otherInductionVariable.getType(),
                             otherWhileOp->getLoc());
      auto otherTerm = otherBody->getTerminator();

      rewriter.setInsertionPoint(otherTerm);

      otherInductionVariable =
          stablehlo::SubtractOp::create(
              rewriter, otherWhileOp->getLoc(), otherInductionVariable,
              stablehlo::ConstantOp::create(
                  rewriter, otherWhileOp->getLoc(),
                  otherInductionVariable.getType(),
                  cast<ElementsAttr>(
                      makeAttr(otherInductionVariable.getType(), 1))))
              .getResult();
      otherTerm->insertOperands(otherTerm->getNumOperands(),
                                ValueRange(otherInductionVariable));

      rewriter.setInsertionPoint(otherWhileOp);
      auto newOtherWhileOp = stablehlo::WhileOp::create(
          rewriter, otherWhileOp->getLoc(), operands);

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
      if (info.pushedValue().getParentRegion() != &newWhile.getBody())
        continue;

      Value cache = info.initOp.getResult();

      auto newType =
          cast<ShapedType>(cast<AutoDiffTypeInterface>(info.cachedType())
                               .getShadowType(numIters));
      enzyme::InitOp newInit = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.initOp);

        enzyme::InitOp::create(
            rewriter, info.initOp->getLoc(),
            enzyme::CacheType::get(cache.getContext(), newType));
      });
      info.pushOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointAfter(newWhile);
        auto newPush = enzyme::PushOp::create(rewriter, cache.getLoc(),
                                              newInit.getResult(),
                                              newWhile->getResult(resultIdx));
        rewriter.eraseOp(info.pushOp);
        newPush;
      });

      resultIdx++;

      {
        OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPoint(otherWhileOp);

        auto popNewValue = enzyme::PopOp::create(rewriter, info.popOp->getLoc(),
                                                 newType, newInit.getResult());

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

          popValue = stablehlo::DynamicSliceOp::create(
              rewriter, info.popOp->getLoc(), TT.clone(sliceSizes), popNewValue,
              startIndices, sliceSizes);
          popValue = stablehlo::ReshapeOp::create(
              rewriter, info.popOp->getLoc(), TT, popValue);
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
    //  } else {
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
        trueValue = enzyme::GetOp::create(
            rewriter, grad.getLoc(),
            cast<enzyme::GradientType>(grad.getType()).getBasetype(), grad);
      }
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));

      auto falseValue = falseMapping.lookupOrNull(grad);
      if (!falseValue) {
        falseValue = enzyme::GetOp::create(
            rewriter, grad.getLoc(),
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

    auto newIf = stablehlo::IfOp::create(
        rewriter, ifOp->getLoc(), trueTerm->getOperandTypes(), ifOp.getPred());
    newIf.getTrueBranch().takeBody(ifOp.getTrueBranch());
    newIf.getFalseBranch().takeBody(ifOp.getFalseBranch());

    size_t idx = ifOp->getNumResults();
    for (auto grad : gradients) {
      enzyme::SetOp::create(rewriter, grad.getLoc(), grad,
                            newIf->getResult(idx));
      idx++;
    }

    for (auto &[pushedValue, info] : pushedCaches) {
      enzyme::PushOp::create(rewriter, info.pushOp->getLoc(),
                             info.initOp.getResult(), newIf->getResult(idx));
      rewriter.eraseOp(info.pushOp);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(info.popOp->getParentOp());

      auto newPop = enzyme::PopOp::create(rewriter, info.popOp->getLoc(),
                                          info.popOp.getResult().getType(),
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

    SmallVector<Value, 8> newReduceInits;
    newReduceInits.reserve(reduceOp.getInitValues().size());
    for (auto opValue : reduceOp.getInitValues()) {
      auto batchedInit = mapper.lookup(opValue);
      auto scalarInit = getScalarValue(batchedInit.getDefiningOp(), builder);
      if (!scalarInit) {
        // TODO: we need to support broadcasting inits, or do we?
        src->emitError("Unsupported reduce init for batched reduce");
        return failure();
      }
      newReduceInits.push_back(scalarInit);
    }

    SmallVector<int64_t> reduceDims = llvm::to_vector(reduceOp.getDimensions());
    for (int i = 0; i < reduceDims.size(); i++) {
      reduceDims[i] += batchSizes.size();
    }

    auto newReduceOp = stablehlo::ReduceOp::create(builder, src->getLoc(),
                                                   resultTypes, newReduceInputs,
                                                   newReduceInits, reduceDims);

    IRMapping regionMapper;
    Block &oldBlock = reduceOp.getRegion().front();
    for (Operation &op : oldBlock.getOperations()) {
      for (Value operand : op.getOperands()) {
        // If operand is defined outside the region and not yet mapped
        if (operand.getParentRegion() != &reduceOp.getRegion() &&
            !regionMapper.contains(operand)) {
          if (matchPattern(operand, m_Constant())) {
            Operation *definingOp = operand.getDefiningOp();
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(newReduceOp);
            auto clonedOp = builder.clone(*definingOp);
            regionMapper.map(operand, clonedOp->getResult(0));
          } else {
            src->emitError("Currently we don't support non-constants in reduce "
                           "body that are external to the region");
            return failure();
          }
        }
      }
    }
    reduceOp.getRegion().cloneInto(&newReduceOp.getRegion(), regionMapper);

    for (int i = 0; i < reduceOp.getResults().size(); i++) {
      mapper.map(src->getResult(i), newReduceOp.getResult(i));
    }
    return success();
  }
};

struct SHLOReduceWindowOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOReduceWindowOpBatchInterface,
                                             ReduceWindowOp> {

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

    stablehlo::ReduceWindowOp reduceWindowOp =
        cast<stablehlo::ReduceWindowOp>(src);
    if (!reduceWindowOp)
      return failure();

    // Remap the operands
    SmallVector<Value, 8> newReduceWindowInputs;
    newReduceWindowInputs.reserve(reduceWindowOp.getInputs().size());
    for (auto opValue : reduceWindowOp.getInputs())
      newReduceWindowInputs.push_back(mapper.lookup(opValue));

    // The init value would have been batched already, we need to slice it.
    // Constant Folding will fix it up later.
    SmallVector<Value, 8> newReduceWindowInits;
    newReduceWindowInits.reserve(reduceWindowOp.getInitValues().size());
    for (auto opValue : reduceWindowOp.getInitValues()) {
      auto batchedInit = mapper.lookup(opValue);
      auto scalarInit = getScalarValue(batchedInit.getDefiningOp(), builder);
      if (!scalarInit) {
        src->emitError(
            "Unsupported reduce window init for batched reduce window");
        // TODO: we need to support broadcasting inits, or do we?
        return failure();
      }
      newReduceWindowInits.push_back(scalarInit);
    }

    SmallVector<int64_t> windowDims(batchSizes.size(), 1);
    windowDims.append(reduceWindowOp.getWindowDimensions().begin(),
                      reduceWindowOp.getWindowDimensions().end());

    DenseI64ArrayAttr windowStridesAttr;
    if (reduceWindowOp.getWindowStrides()) {
      SmallVector<int64_t> windowStrides(batchSizes.size(), 1);
      windowStrides.append(reduceWindowOp.getWindowStrides()->begin(),
                           reduceWindowOp.getWindowStrides()->end());
      windowStridesAttr = builder.getDenseI64ArrayAttr(windowStrides);
    }

    DenseI64ArrayAttr baseDilationsAttr;
    if (reduceWindowOp.getBaseDilations()) {
      SmallVector<int64_t> baseDilations(batchSizes.size(), 1);
      baseDilations.append(reduceWindowOp.getBaseDilations()->begin(),
                           reduceWindowOp.getBaseDilations()->end());
      baseDilationsAttr = builder.getDenseI64ArrayAttr(baseDilations);
    }

    DenseI64ArrayAttr windowDilationsAttr;
    if (reduceWindowOp.getWindowDilations()) {
      SmallVector<int64_t> windowDilations(batchSizes.size(), 1);
      windowDilations.append(reduceWindowOp.getWindowDilations()->begin(),
                             reduceWindowOp.getWindowDilations()->end());
      windowDilationsAttr = builder.getDenseI64ArrayAttr(windowDilations);
    }

    DenseIntElementsAttr newPaddingAttr;
    if (reduceWindowOp.getPadding()) {
      auto paddingOriginal =
          llvm::to_vector(reduceWindowOp.getPadding()->getValues<int64_t>());
      auto paddingType = RankedTensorType::get(
          {static_cast<int64_t>(batchSizes.size() + paddingOriginal.size() / 2),
           2},
          builder.getI64Type());

      SmallVector<int64_t> newPadding(2 * batchSizes.size(), 0);
      newPadding.append(paddingOriginal.begin(), paddingOriginal.end());
      newPaddingAttr = mlir::DenseIntElementsAttr::get(paddingType, newPadding);
    }

    auto newReduceWindowOp = stablehlo::ReduceWindowOp::create(
        builder, src->getLoc(), resultTypes, newReduceWindowInputs,
        newReduceWindowInits, windowDims, windowStridesAttr, baseDilationsAttr,
        windowDilationsAttr, newPaddingAttr);

    IRMapping regionMapper;
    reduceWindowOp.getRegion().cloneInto(&newReduceWindowOp.getRegion(),
                                         regionMapper);

    for (int i = 0; i < reduceWindowOp.getResults().size(); i++) {
      mapper.map(src->getResult(i), newReduceWindowOp.getResult(i));
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

    auto dotOp = stablehlo::DotGeneralOp::create(
        builder, op.getLoc(),
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

    auto bcastOp = stablehlo::BroadcastInDimOp::create(
        builder, op.getLoc(),
        RankedTensorType::get(resultShape, resultType.getElementType()),
        mapper.lookup(op.getOperand()),
        builder.getDenseI64ArrayAttr(bcastDims));

    mapper.map(src->getResult(0), bcastOp->getResult(0));
    return success();
  }
};

struct SHLOConcatenateOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOConcatenateOpBatchInterface,
                                             stablehlo::ConcatenateOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::ConcatenateOp>(src);

    SmallVector<Value> newInputs;
    for (auto input : op.getInputs()) {
      newInputs.push_back(mapper.lookup(input));
    }

    auto newConcatOp = stablehlo::ConcatenateOp::create(
        builder, op.getLoc(), ValueRange(newInputs),
        op.getDimension() + batchSizes.size());

    mapper.map(src->getResult(0), newConcatOp->getResult(0));
    return success();
  }
};

struct SHLOGatherOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOGatherOpBatchInterface,
                                             stablehlo::GatherOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::GatherOp>(src);

    auto newOperand = mapper.lookup(op.getOperand());
    auto newStartIndices = mapper.lookup(op.getStartIndices());

    SmallVector<int64_t> newOffsetDims, newCollapsedSliceDims,
        newOperandBatchingDims, newStartIndicesBatchingDims, newStartIndexMap,
        newSliceSizes;
    int64_t nBatch = batchSizes.size();

    auto oldGatherDimensionNumbers = op.getDimensionNumbers();

    for (auto offsetDim : oldGatherDimensionNumbers.getOffsetDims())
      newOffsetDims.push_back(offsetDim + nBatch);
    for (auto collapsedSliceDim :
         oldGatherDimensionNumbers.getCollapsedSliceDims())
      newCollapsedSliceDims.push_back(collapsedSliceDim + nBatch);

    for (int64_t i = 0; i < nBatch; i++) {
      newOperandBatchingDims.push_back(i);
      newStartIndicesBatchingDims.push_back(i);
    }
    for (auto operandBatchingDim :
         oldGatherDimensionNumbers.getOperandBatchingDims())
      newOperandBatchingDims.push_back(operandBatchingDim + nBatch);
    for (auto startIndicesBatchingDim :
         oldGatherDimensionNumbers.getStartIndicesBatchingDims())
      newStartIndicesBatchingDims.push_back(startIndicesBatchingDim + nBatch);
    for (auto startIndexMap : oldGatherDimensionNumbers.getStartIndexMap())
      newStartIndexMap.push_back(startIndexMap + nBatch);

    auto newIndexVectorDim =
        oldGatherDimensionNumbers.getIndexVectorDim() + nBatch;

    auto gatherDims = stablehlo::GatherDimensionNumbersAttr::get(
        op.getContext(), newOffsetDims, newCollapsedSliceDims,
        newOperandBatchingDims, newStartIndicesBatchingDims, newStartIndexMap,
        newIndexVectorDim);

    for (int64_t i = 0; i < nBatch; i++)
      newSliceSizes.push_back(1);
    for (auto sliceSize : op.getSliceSizes())
      newSliceSizes.push_back(sliceSize);

    auto newGatherOp =
        stablehlo::GatherOp::create(builder, op.getLoc(), newOperand,
                                    newStartIndices, gatherDims, newSliceSizes);

    mapper.map(src->getResult(0), newGatherOp->getResult(0));
    return success();
  }
};

struct SHLOSliceOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOSliceOpBatchInterface,
                                             stablehlo::SliceOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::SliceOp>(src);

    auto newOperand = mapper.lookup(op.getOperand());

    SmallVector<int64_t> newStartIndices, newLimitIndices, newStrides;
    for (int64_t i = 0; i < batchSizes.size(); i++) {
      newStartIndices.push_back(0);
      newLimitIndices.push_back(batchSizes[i]);
      newStrides.push_back(1);
    }
    for (auto [sIndex, lIndex, stride] : llvm::zip(
             op.getStartIndices(), op.getLimitIndices(), op.getStrides())) {
      newStartIndices.push_back(sIndex);
      newLimitIndices.push_back(lIndex);
      newStrides.push_back(stride);
    }

    auto newSliceOp = stablehlo::SliceOp::create(builder, op.getLoc(),
                                                 newOperand, newStartIndices,
                                                 newLimitIndices, newStrides);

    mapper.map(src->getResult(0), newSliceOp->getResult(0));
    return success();
  }
};

SmallVector<Value> computeBatchedStartIndices(Operation *op, OpBuilder &builder,
                                              SmallVector<Value> startIndices,
                                              IRMapping &mapper,
                                              ArrayRef<int64_t> batchSizes) {
  auto startIndicesType = cast<RankedTensorType>(startIndices[0].getType());
  auto startIndicesElemType = startIndicesType.getElementType();
  auto zeroStart =
      makeIntegerConstant(op->getLoc(), builder, startIndicesElemType, 0);

  SmallVector<Value> newStartIndices;
  for (int64_t i = 0; i < batchSizes.size(); i++)
    newStartIndices.push_back(zeroStart);

  SmallVector<int64_t> innerSliceStarts, innerSliceLimits, innerSliceStrides;
  for (int64_t i = 0; i < batchSizes.size(); i++) {
    innerSliceStarts.push_back(0);
    innerSliceLimits.push_back(1);
    innerSliceStrides.push_back(1);
  }

  for (auto sIndex : startIndices) {
    // We need to slice and extract a single element
    auto newStartIndex = stablehlo::SliceOp::create(
        builder, op->getLoc(), mapper.lookup(sIndex), innerSliceStarts,
        innerSliceLimits, innerSliceStrides);
    auto newStartIndexReshape = stablehlo::ReshapeOp::create(
        builder, op->getLoc(), RankedTensorType::get({}, startIndicesElemType),
        newStartIndex);
    newStartIndices.push_back(newStartIndexReshape.getResult());
  }

  return newStartIndices;
}

struct SHLODynamicSliceOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLODynamicSliceOpBatchInterface,
                                             stablehlo::DynamicSliceOp> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::DynamicSliceOp>(src);

    SmallVector<Value> startIndices = computeBatchedStartIndices(
        op, builder, op.getStartIndices(), mapper, batchSizes);

    SmallVector<int64_t> sliceSizes;
    for (int64_t i = 0; i < batchSizes.size(); i++)
      sliceSizes.push_back(batchSizes[i]);
    for (auto sIndex : op.getSliceSizes())
      sliceSizes.push_back(sIndex);

    auto newSliceOp = stablehlo::DynamicSliceOp::create(
        builder, op.getLoc(), mapper.lookup(op.getOperand()), startIndices,
        sliceSizes);

    mapper.map(src->getResult(0), newSliceOp.getResult());
    return success();
  }
};

struct SHLODynamicUpdateSliceOpBatchInterface
    : public BatchOpInterface::ExternalModel<
          SHLODynamicUpdateSliceOpBatchInterface,
          stablehlo::DynamicUpdateSliceOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::DynamicUpdateSliceOp>(src);

    SmallVector<Value> startIndices = computeBatchedStartIndices(
        op, builder, op.getStartIndices(), mapper, batchSizes);

    auto newDUS = stablehlo::DynamicUpdateSliceOp::create(
        builder, op.getLoc(), mapper.lookup(op.getOperand()),
        mapper.lookup(op.getUpdate()), startIndices);

    mapper.map(src->getResult(0), newDUS.getResult());
    return success();
  }
};

struct SHLOIotaOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOIotaOpBatchInterface,
                                             stablehlo::IotaOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::IotaOp>(src);

    auto origResult = cast<RankedTensorType>(op.getResult().getType());

    SmallVector<int64_t> newShape;
    newShape.append(batchSizes.begin(), batchSizes.end());
    newShape.append(origResult.getShape().begin(), origResult.getShape().end());

    auto newIotaOp = stablehlo::IotaOp::create(
        builder, op.getLoc(),
        RankedTensorType::get(newShape, origResult.getElementType()),
        op.getIotaDimension() + batchSizes.size());

    mapper.map(src->getResult(0), newIotaOp.getResult());
    return success();
  }
};

struct SHLOSortOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOSortOpBatchInterface,
                                             stablehlo::SortOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::SortOp>(src);

    SmallVector<Value> newOperands;
    newOperands.reserve(op.getNumOperands());
    for (auto operand : op.getOperands()) {
      newOperands.push_back(mapper.lookup(operand));
    }

    auto newSortOp = stablehlo::SortOp::create(
        builder, op.getLoc(), ValueRange(newOperands),
        builder.getI64IntegerAttr(op.getDimension() + batchSizes.size()),
        op.getIsStableAttr());

    IRMapping regionMapper;
    op.getComparator().cloneInto(&newSortOp.getComparator(), regionMapper);

    for (int i = 0; i < newSortOp.getNumResults(); i++) {
      mapper.map(src->getResult(i), newSortOp.getResult(i));
    }
    return success();
  }
};

struct SHLOSelectOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOSelectOpBatchInterface,
                                             stablehlo::SelectOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::SelectOp>(src);

    auto opPredOld = op.getPred();
    auto opPredType = cast<RankedTensorType>(opPredOld.getType());
    auto opResultShape =
        cast<RankedTensorType>(op.getResult().getType()).getShape();

    stablehlo::SelectOp newSelectOp;

    if (opPredType.getRank() == 0) {
      // Need a broadcast_in_dim to make the predicate into proper shape
      SmallVector<int64_t> newShape;
      newShape.append(batchSizes.begin(), batchSizes.end());
      newShape.append(opResultShape.begin(), opResultShape.end());

      SmallVector<int64_t> broadcastDims;
      for (int64_t i = 0; i < batchSizes.size(); i++)
        broadcastDims.push_back(i);

      auto newPred = stablehlo::BroadcastInDimOp::create(
          builder, op.getLoc(),
          RankedTensorType::get(newShape, opPredType.getElementType()),
          mapper.lookup(opPredOld),
          builder.getDenseI64ArrayAttr(broadcastDims));

      newSelectOp = stablehlo::SelectOp::create(builder, op.getLoc(), newPred,
                                                mapper.lookup(op.getOnTrue()),
                                                mapper.lookup(op.getOnFalse()));
    } else {
      newSelectOp = stablehlo::SelectOp::create(
          builder, op.getLoc(), mapper.lookup(opPredOld),
          mapper.lookup(op.getOnTrue()), mapper.lookup(op.getOnFalse()));
    }

    mapper.map(src->getResult(0), newSelectOp.getResult());
    return success();
  }
};

struct SHLOReverseOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOReverseOpBatchInterface,
                                             stablehlo::ReverseOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto op = cast<stablehlo::ReverseOp>(src);

    int64_t nBatches = batchSizes.size();
    SmallVector<int64_t> newDims;
    for (auto dim : op.getDimensions())
      newDims.push_back(dim + nBatches);

    auto newReverseOp = stablehlo::ReverseOp::create(
        builder, op.getLoc(), mapper.lookup(op.getOperand()),
        builder.getDenseI64ArrayAttr(newDims));

    mapper.map(src->getResult(0), newReverseOp.getResult());
    return success();
  }
};

// https://github.com/jax-ml/jax/blob/2a8cb54b82f1b0d17181d43f9be78d2b349df333/jax/_src/lax/convolution.py#L613-L629
struct SHLOConvolutionOpBatchInterface
    : public BatchOpInterface::ExternalModel<SHLOConvolutionOpBatchInterface,
                                             stablehlo::ConvolutionOp> {
  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto convolution = cast<stablehlo::ConvolutionOp>(src);
    auto convDimNumbers = convolution.getDimensionNumbers();
    int64_t inputBatchDim = convDimNumbers.getInputBatchDimension();
    int64_t inputFeatureDim = convDimNumbers.getInputFeatureDimension();
    int64_t kernelOutputFeatureDim =
        convDimNumbers.getKernelOutputFeatureDimension();
    int64_t outputFeatureDim = convDimNumbers.getOutputFeatureDimension();

    int64_t batchSize = std::accumulate(batchSizes.begin(), batchSizes.end(), 1,
                                        std::multiplies<int64_t>());

    auto lhs = mapper.lookup(convolution.getLhs());
    auto rhs = mapper.lookup(convolution.getRhs());

    int64_t batchGroupCount = convolution.getBatchGroupCount();
    int64_t featureGroupCount = convolution.getFeatureGroupCount();

    int64_t inputBatchingDimFromGroupCount;
    if (batchGroupCount > 1) {
      inputBatchingDimFromGroupCount = inputBatchDim;
      batchGroupCount *= batchSize;
    } else {
      inputBatchingDimFromGroupCount = inputFeatureDim;
      featureGroupCount *= batchSize;
    }
    auto batchedLhs = reshapeAxisInto(builder, lhs, batchSizes,
                                      inputBatchingDimFromGroupCount);

    auto batchedRhs =
        reshapeAxisInto(builder, rhs, batchSizes, kernelOutputFeatureDim);

    auto outTy = cast<RankedTensorType>(convolution.getResult().getType());
    auto outShape = llvm::to_vector(outTy.getShape());
    outShape[outputFeatureDim] = outShape[outputFeatureDim] * batchSize;
    auto outElemTy = outTy.getElementType();

    auto batchedConvolution = stablehlo::ConvolutionOp::create(
        builder, src->getLoc(), RankedTensorType::get(outShape, outElemTy),
        batchedLhs, batchedRhs, convolution.getWindowStridesAttr(),
        convolution.getPaddingAttr(), convolution.getLhsDilationAttr(),
        convolution.getRhsDilationAttr(), convolution.getWindowReversalAttr(),
        convolution.getDimensionNumbersAttr(),
        builder.getI64IntegerAttr(featureGroupCount),
        builder.getI64IntegerAttr(batchGroupCount),
        convolution.getPrecisionConfigAttr());

    auto transposedOut = reshapeAxisOutOf(builder, batchedConvolution,
                                          batchSizes, outputFeatureDim);
    mapper.map(src->getResult(0), transposedOut);
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

    WhileOp::attachInterface<WhileOpEnzymeOpsRemover>(*context);
    IfOp::attachInterface<IfOpEnzymeOpsRemover>(*context);

    WhileOp::attachInterface<ADDataFlowWhileOp>(*context);
    SortOp::attachInterface<ADDataFlowSortOp>(*context);
    ScatterOp::attachInterface<ADDataFlowScatterOp>(*context);
    ReduceOp::attachInterface<ADDataFlowReduceOp>(*context);

    ScatterOp::attachInterface<ScatterActivity>(*context);
    ScatterOp::attachInterface<AutoDiffScatterFwd>(*context);
    ScatterOp::attachInterface<AutoDiffScatterRev>(*context);

    ReturnOp::attachInterface<AutoDiffHLOReturn>(*context);

    ReduceOp::attachInterface<AutoDiffReduceFwd<ReduceOp>>(*context);
    IfOp::attachInterface<AutoDiffIfRev>(*context);
    IfOp::attachInterface<AutoDiffIfFwd>(*context);
    IfOp::attachInterface<AutoDiffIfCF>(*context);

    SortOp::attachInterface<AutoDiffSortFwd>(*context);
    SortOp::attachInterface<AutoDiffSortRev>(*context);
    WhileOp::attachInterface<AutoDiffWhileFwd>(*context);
    WhileOp::attachInterface<AutoDiffWhileRev>(*context);
    ReduceOp::attachInterface<AutoDiffReduceCF<ReduceOp>>(*context);
    WhileOp::attachInterface<AutoDiffReduceCF<WhileOp>>(*context);
    BroadcastInDimOp::attachInterface<AutoDiffBroadcastInDimRev>(*context);
    SliceOp::attachInterface<AutoDiffSliceRev>(*context);
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
    ReduceWindowOp::attachInterface<SHLOReduceWindowOpBatchInterface>(*context);
    DotGeneralOp::attachInterface<SHLODotGeneralOpBatchInterface>(*context);
    BroadcastInDimOp::attachInterface<SHLOBroadcastInDimOpBatchInterface>(
        *context);
    ConcatenateOp::attachInterface<SHLOConcatenateOpBatchInterface>(*context);
    GatherOp::attachInterface<SHLOGatherOpBatchInterface>(*context);
    SliceOp::attachInterface<SHLOSliceOpBatchInterface>(*context);
    DynamicSliceOp::attachInterface<SHLODynamicSliceOpBatchInterface>(*context);
    DynamicUpdateSliceOp::attachInterface<
        SHLODynamicUpdateSliceOpBatchInterface>(*context);
    CustomCallOp::attachInterface<SHLOGenericBatchOpInterface<CustomCallOp>>(
        *context);
    IotaOp::attachInterface<SHLOIotaOpBatchInterface>(*context);
    SelectOp::attachInterface<SHLOSelectOpBatchInterface>(*context);
    SortOp::attachInterface<SHLOSortOpBatchInterface>(*context);
    GetDimensionSizeOp::attachInterface<SHLOGetDimensionSizeOpBatchInterface>(
        *context);
    ReverseOp::attachInterface<SHLOReverseOpBatchInterface>(*context);
    ConvolutionOp::attachInterface<SHLOConvolutionOpBatchInterface>(*context);

    ScatterOp::attachInterface<SHLOGenericBatchOpInterface<ScatterOp>>(
        *context); // TODO: simpler version with newly named dims

    AddOp::attachInterface<StablehloAddSimplifyMathInterface>(*context);
    SubtractOp::attachInterface<StablehloSubSimplifyMathInterface>(*context);
  });
}
