//===- TritonAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the MLIR tt dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_JAX_ENABLE_TRITON
#define ENZYME_JAX_ENABLE_TRITON 1
#endif

#if ENZYME_JAX_ENABLE_TRITON

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::triton;

namespace {

#include "src/enzyme_ad/jax/Implementations/TritonDerivatives.inc"

class TritonPointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TritonPointerTypeInterface,
                                                  triton::PointerType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm_unreachable("TODO");
  }
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    llvm_unreachable("TODO");
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    // TODO inspect val and memset corresponding size
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

class TritonTensorDescTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TritonTensorDescTypeInterface,
                                                  triton::TensorDescType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm_unreachable("TODO");
  }
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    llvm_unreachable("TODO");
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

class AutoDiffTritonFuncFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffTritonFuncFunctionInterface, triton::FuncOp> {
public:
  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &returnTypes) const {}

  // A tt.func has no comdat or linkage group to leave behind.
  void detachFromPrimalDefinition(Operation *self) const {}

  Operation *createCall(Operation *self, OpBuilder &builder, Location loc,
                        ValueRange args) const {
    return triton::CallOp::create(builder, loc, cast<triton::FuncOp>(self),
                                  args);
  }

  Operation *createReturn(Operation *self, OpBuilder &builder, Location loc,
                          ValueRange retArgs) const {
    return triton::ReturnOp::create(builder, loc, retArgs);
  }
};

// Helpers shared by the forward and reverse TTIR rules. Preserve the original
// attributes when cloning (in particular layout, precision and cache policy).
static Value zero(OpBuilder &b, Location loc, Type type) {
  return cast<AutoDiffTypeInterface>(type).createNullValue(b, loc);
}

static Value tangent(Value value, OpBuilder &b, MGradientUtils *g) {
  return g->isConstantValue(value) ? zero(b, value.getLoc(), value.getType())
                                   : g->invertPointerM(value, b);
}

static Operation *cloneWithOperands(Operation *op, OpBuilder &b,
                                    ValueRange operands) {
  Operation *copy = b.clone(*op);
  copy->setOperands(operands);
  return copy;
}

static bool isHandle(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return isa<triton::PointerType, triton::TensorDescType>(type);
}

static Value sum(OpBuilder &b, Location loc, Value value, unsigned axis) {
  auto reduce = triton::ReduceOp::create(b, loc, ValueRange{value}, axis);
  auto *block = new Block();
  reduce.getRegion().push_back(block);
  auto elem = cast<RankedTensorType>(value.getType()).getElementType();
  block->addArguments({elem, elem}, {loc, loc});
  OpBuilder body = OpBuilder::atBlockBegin(block);
  Value result = arith::AddFOp::create(body, loc, block->getArgument(0),
                                       block->getArgument(1));
  triton::ReduceReturnOp::create(body, loc, ValueRange{result});
  return reduce->getResult(0);
}

static Value expandReduction(OpBuilder &b, Location loc, Value value,
                             RankedTensorType target, unsigned axis) {
  if (!isa<RankedTensorType>(value.getType()))
    return triton::SplatOp::create(b, loc, target, value);
  value = triton::ExpandDimsOp::create(b, loc, value, axis);
  return triton::BroadcastOp::create(b, loc, target, value);
}

// Pointer transformations construct their shadows in the primal sweep. Value
// transformations instead pull the result adjoint back through the shape map.
template <typename OpTy>
struct ShapeRev
    : ReverseAutoDiffOpInterface::ExternalModel<ShapeRev<OpTy>, OpTy> {
  SmallVector<Value> cacheValues(Operation *, MGradientUtilsReverse *) const {
    return {};
  }
  LogicalResult createShadowValues(Operation *op, OpBuilder &b,
                                   MGradientUtilsReverse *g) const {
    if (!isHandle(op->getResult(0).getType()) ||
        g->isConstantValue(op->getResult(0)))
      return success();
    SmallVector<Value> operands = g->getNewFromOriginal(op->getOperands());
    operands[0] = g->invertPointerM(op->getOperand(0), b);
    auto *shadow = cloneWithOperands(op, b, operands);
    g->setInvertedPointer(op->getResult(0), shadow->getResult(0));
    return success();
  }
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &b,
                                         MGradientUtilsReverse *g,
                                         SmallVector<Value>) const {
    Value input = op->getOperand(0), result = op->getResult(0);
    if (isHandle(result.getType()) || g->isConstantValue(result) ||
        g->isConstantValue(input))
      return success();
    Value grad = g->diffe(result, b);
    g->zeroDiffe(result, b);
    auto loc = op->getLoc();
    if constexpr (std::is_same_v<OpTy, triton::SplatOp>) {
      while (auto type = dyn_cast<RankedTensorType>(grad.getType()))
        grad = sum(b, loc, grad, type.getRank() - 1);
    } else if constexpr (std::is_same_v<OpTy, triton::BroadcastOp>) {
      auto type = cast<RankedTensorType>(input.getType());
      auto out = cast<RankedTensorType>(result.getType());
      for (int axis = type.getRank() - 1; axis >= 0; --axis)
        if (type.getDimSize(axis) != out.getDimSize(axis)) {
          grad = sum(b, loc, grad, axis);
          if (isa<RankedTensorType>(grad.getType()))
            grad = triton::ExpandDimsOp::create(b, loc, grad, axis);
          else
            grad = triton::SplatOp::create(b, loc, type, grad);
        }
    } else if constexpr (std::is_same_v<OpTy, triton::ExpandDimsOp>) {
      grad = sum(b, loc, grad, cast<triton::ExpandDimsOp>(op).getAxis());
    } else if constexpr (std::is_same_v<OpTy, triton::ReshapeOp>) {
      grad = triton::ReshapeOp::create(b, loc, input.getType(), grad,
                                       /*allowReorder=*/false);
    } else if constexpr (std::is_same_v<OpTy, triton::TransOp>) {
      auto order = cast<triton::TransOp>(op).getOrder();
      SmallVector<int32_t> inverse(order.size());
      for (auto [i, axis] : llvm::enumerate(order))
        inverse[axis] = i;
      grad = triton::TransOp::create(b, loc, grad, inverse);
    }
    g->addToDiffe(input, grad, b);
    return success();
  }
};

struct LoadActivity
    : ActivityOpInterface::ExternalModel<LoadActivity, triton::LoadOp> {
  bool isInactive(Operation *) const { return false; }
  bool isArgInactive(Operation *, size_t index) const { return index == 1; }
};

struct LoadFwd : AutoDiffOpInterface::ExternalModel<LoadFwd, triton::LoadOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &b,
                                         MGradientUtils *g) const {
    auto op = cast<triton::LoadOp>(orig);
    if (g->isConstantValue(op.getResult()))
      return success();
    Value other = op.getOther() ? tangent(op.getOther(), b, g)
                                : zero(b, op.getLoc(), op.getType());
    Value mask = op.getMask() ? g->getNewFromOriginal(op.getMask()) : Value();
    Value result;
    if (g->isConstantValue(op.getPtr())) {
      result = zero(b, op.getLoc(), op.getType());
      if (mask)
        result = arith::SelectOp::create(b, op.getLoc(), mask, result, other);
    } else {
      auto *copy = b.clone(*g->getNewFromOriginal(orig));
      auto load = cast<triton::LoadOp>(copy);
      load.getPtrMutable().assign(g->invertPointerM(op.getPtr(), b));
      if (op.getOther())
        load.getOtherMutable().assign(other);
      result = load.getResult();
    }
    g->setDiffe(op.getResult(), result, b);
    return success();
  }
};

// Caches include the shadow address and predicates: a later store or loop
// iteration may overwrite the original values before the reverse sweep.
template <typename OpTy, bool IsLoad, bool Descriptor = false>
struct MemoryRev : ReverseAutoDiffOpInterface::ExternalModel<
                       MemoryRev<OpTy, IsLoad, Descriptor>, OpTy> {
  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
    return success();
  }
  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *g) const {
    if constexpr (IsLoad)
      if (g->isConstantValue(op->getResult(0)))
        return {};
    OpBuilder b(g->getNewFromOriginal(op));
    SmallVector<Value> caches;
    if (!g->isConstantValue(op->getOperand(0)))
      caches.push_back(
          g->initAndPushCache(g->invertPointerM(op->getOperand(0), b), b));
    // Loads: descriptor indices or mask; stores: indices or mask after value.
    unsigned start = IsLoad ? 1 : 2;
    unsigned end = Descriptor
                       ? op->getNumOperands()
                       : std::min<unsigned>(op->getNumOperands(), start + 1);
    for (unsigned i = start; i < end; ++i)
      caches.push_back(
          g->initAndPushCache(g->getNewFromOriginal(op->getOperand(i)), b));
    return caches;
  }
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &b,
                                         MGradientUtilsReverse *g,
                                         SmallVector<Value> caches) const {
    auto loc = orig->getLoc();
    if constexpr (IsLoad)
      if (g->isConstantValue(orig->getResult(0)))
        return success();
    unsigned next = 0;
    Value ptr;
    if (!g->isConstantValue(orig->getOperand(0)))
      ptr = g->popCache(caches[next++], b);
    SmallVector<Value> indices;
    while (next < caches.size())
      indices.push_back(g->popCache(caches[next++], b));
    Value mask = !Descriptor && !indices.empty() ? indices[0] : Value();
    if constexpr (IsLoad) {
      Value grad = g->diffe(orig->getResult(0), b);
      g->zeroDiffe(orig->getResult(0), b);
      if (ptr) {
        if constexpr (Descriptor) {
          triton::DescriptorReduceOp::create(
              b, loc, triton::DescriptorReduceKind::ADD, ptr, grad, indices);
        } else {
          // Multiple lanes and CTAs may load the same address.
          triton::AtomicRMWOp::create(
              b, loc, grad.getType(), triton::RMWOp::FADD, ptr, grad, mask,
              triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
        }
      }
      if constexpr (!Descriptor) {
        auto op = cast<triton::LoadOp>(orig);
        if (op.getOther() && !g->isConstantValue(op.getOther())) {
          Value fallback = arith::SelectOp::create(
              b, loc, mask, zero(b, loc, grad.getType()), grad);
          g->addToDiffe(op.getOther(), fallback, b);
        }
      }
    } else if (ptr) {
      Value input = orig->getOperand(1);
      Value z = zero(b, loc, input.getType());
      Value grad;
      if constexpr (Descriptor) {
        grad = triton::DescriptorLoadOp::create(b, loc, input.getType(), ptr,
                                                indices,
                                                /*cachePolicy=*/Attribute());
        triton::DescriptorStoreOp::create(b, loc, ptr, z, indices);
      } else {
        grad = triton::LoadOp::create(b, loc, ptr, mask, mask ? z : Value());
        triton::StoreOp::create(b, loc, ptr, z, mask);
      }
      if (!g->isConstantValue(input))
        g->addToDiffe(input, grad, b);
    }
    return success();
  }
};

static Value transposeMatrix(OpBuilder &b, Location loc, Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int32_t> order;
  for (int i = 0; i < type.getRank(); ++i)
    order.push_back(i);
  std::swap(order[order.size() - 1], order[order.size() - 2]);
  return triton::TransOp::create(b, loc, value, order);
}

static Value convertFloat(OpBuilder &b, Location loc, Value value, Type type) {
  if (value.getType() == type)
    return value;
  auto src = cast<FloatType>(getElementTypeOrSelf(value.getType()));
  auto dst = cast<FloatType>(getElementTypeOrSelf(type));
  if (src.getWidth() < dst.getWidth())
    return arith::ExtFOp::create(b, loc, type, value);
  return arith::TruncFOp::create(b, loc, type, value);
}

struct DotRev
    : ReverseAutoDiffOpInterface::ExternalModel<DotRev, triton::DotOp> {
  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
    return success();
  }
  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *g) const {
    if (g->isConstantValue(op->getResult(0)))
      return {};
    OpBuilder b(g->getNewFromOriginal(op));
    SmallVector<Value> caches;
    for (unsigned i = 0; i < 2; ++i)
      if (!g->isConstantValue(op->getOperand(i)))
        caches.push_back(g->initAndPushCache(
            g->getNewFromOriginal(op->getOperand(1 - i)), b));
    return caches;
  }
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &b,
                                         MGradientUtilsReverse *g,
                                         SmallVector<Value> caches) const {
    auto op = cast<triton::DotOp>(orig);
    if (g->isConstantValue(op.getResult()))
      return success();
    Value grad = g->diffe(op.getResult(), b);
    g->zeroDiffe(op.getResult(), b);
    unsigned next = 0;
    for (unsigned i = 0; i < 2; ++i) {
      Value input = op->getOperand(i);
      if (g->isConstantValue(input))
        continue;
      Value other =
          transposeMatrix(b, op.getLoc(), g->popCache(caches[next++], b));
      // Accumulate in the output precision, then cast to the input precision.
      auto accumElem = getElementTypeOrSelf(grad.getType());
      auto otherType = cast<RankedTensorType>(other.getType()).clone(accumElem);
      other = convertFloat(b, op.getLoc(), other, otherType);
      auto accumType = cast<RankedTensorType>(input.getType()).clone(accumElem);
      Value z = zero(b, op.getLoc(), accumType);
      Value result =
          triton::DotOp::create(b, op.getLoc(), accumType,
                                i == 0 ? grad : other, i == 0 ? other : grad, z,
                                getElementTypeOrSelf(input.getType()).isF32()
                                    ? op.getInputPrecision()
                                    : triton::InputPrecision::IEEE,
                                /*maxNumImpreciseAcc=*/0);
      result = convertFloat(b, op.getLoc(), result, input.getType());
      g->addToDiffe(input, result, b);
    }
    if (!g->isConstantValue(op.getC()))
      g->addToDiffe(op.getC(), grad, b);
    return success();
  }
};

// Recognize exactly the scalar combiner we implement; arbitrary user regions
// must not silently receive a sum derivative.
static bool isSimpleReduction(triton::ReduceOp op, bool &isSum) {
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return false;
  Block &block = op.getRegion().front();
  if (block.getOperations().size() != 2)
    return false;
  Operation &combine = block.front();
  isSum = isa<arith::AddFOp>(combine);
  if (!isSum && !isa<arith::MaxNumFOp, arith::MinNumFOp>(combine))
    return false;
  return combine.getOperand(0) == block.getArgument(0) &&
         combine.getOperand(1) == block.getArgument(1) &&
         block.getTerminator()->getOperand(0) == combine.getResult(0);
}

// At extrema ties use the symmetric subgradient, in both AD modes.
static Value extremaWeights(OpBuilder &b, triton::ReduceOp op, Value input,
                            Value result) {
  auto loc = op.getLoc();
  auto type = cast<RankedTensorType>(input.getType());
  Value expanded = expandReduction(b, loc, result, type, op.getAxis());
  Value matches =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, input, expanded);
  Value weights = arith::UIToFPOp::create(b, loc, type, matches);
  Value count = expandReduction(b, loc, sum(b, loc, weights, op.getAxis()),
                                type, op.getAxis());
  return arith::DivFOp::create(b, loc, weights, count);
}

struct ReduceFwd
    : AutoDiffOpInterface::ExternalModel<ReduceFwd, triton::ReduceOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &b,
                                         MGradientUtils *g) const {
    auto op = cast<triton::ReduceOp>(orig);
    if (llvm::all_of(op->getResults(),
                     [&](Value value) { return g->isConstantValue(value); }))
      return success();
    // Preserve the existing forward rule for NaN-propagating extrema.
    // The shared symmetric rule below covers maxnumf/minnumf.
    auto &combine = op.getRegion().front();
    if (combine.getOperations().size() == 2 &&
        isa<arith::MaximumFOp, arith::MinimumFOp>(combine.front())) {
      llvm::SmallDenseSet<unsigned> operands, results;
      for (auto [i, value] : llvm::enumerate(op->getOperands()))
        if (!g->isConstantValue(value))
          operands.insert(i);
      for (auto [i, value] : llvm::enumerate(op->getResults()))
        if (!g->isConstantValue(value))
          results.insert(i);
      return mlir::enzyme::detail::controlFlowForwardHandler(orig, b, g,
                                                             operands, results);
    }
    bool isSum;
    if (!isSimpleReduction(op, isSum))
      return op.emitError(
          "autodiff supports single-input addf/maxnumf/minnumf reductions");
    // Cloning reserves shadow block arguments and placeholders for all
    // active values. The primal combiner stays unchanged: its derivative is
    // built separately below, so discard those unused reservations.
    auto primal = cast<triton::ReduceOp>(g->getNewFromOriginal(op));
    auto &body = op.getRegion().front();
    for (Operation &nested : body) {
      for (Value result : nested.getResults()) {
        if (Value shadow = g->invertedPointers.lookupOrNull(result)) {
          g->invertedPointers.erase(result);
          g->erase(shadow.getDefiningOp());
        }
      }
    }
    llvm::BitVector erase(primal.getRegion().front().getNumArguments());
    for (BlockArgument arg : body.getArguments()) {
      if (auto shadow = dyn_cast_or_null<BlockArgument>(
              g->invertedPointers.lookupOrNull(arg))) {
        erase.set(shadow.getArgNumber());
        g->invertedPointers.erase(arg);
      }
    }
    primal.getRegion().front().eraseArguments(erase);
    // Extrema weights depend on the primal reduction result.
    b.setInsertionPointAfter(primal);
    Value grad = tangent(op.getOperand(0), b, g);
    if (!isSum) {
      Value weights =
          extremaWeights(b, op, g->getNewFromOriginal(op.getOperand(0)),
                         g->getNewFromOriginal(op->getResult(0)));
      grad = arith::MulFOp::create(b, op.getLoc(), weights, grad);
    }
    g->setDiffe(op->getResult(0), sum(b, op.getLoc(), grad, op.getAxis()), b);
    return success();
  }
};

struct ReduceRev
    : ReverseAutoDiffOpInterface::ExternalModel<ReduceRev, triton::ReduceOp> {
  LogicalResult createShadowValues(Operation *, OpBuilder &,
                                   MGradientUtilsReverse *) const {
    return success();
  }
  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *g) const {
    auto op = cast<triton::ReduceOp>(orig);
    bool isSum;
    if (!isSimpleReduction(op, isSum) || isSum ||
        g->isConstantValue(op->getResult(0)))
      return {};
    OpBuilder b(g->getNewFromOriginal(op));
    b.setInsertionPointAfter(g->getNewFromOriginal(op));
    return {g->initAndPushCache(g->getNewFromOriginal(op.getOperand(0)), b),
            g->initAndPushCache(g->getNewFromOriginal(op->getResult(0)), b)};
  }
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &b,
                                         MGradientUtilsReverse *g,
                                         SmallVector<Value> caches) const {
    auto op = cast<triton::ReduceOp>(orig);
    if (llvm::all_of(op->getResults(),
                     [&](Value value) { return g->isConstantValue(value); }))
      return success();
    bool isSum;
    if (!isSimpleReduction(op, isSum))
      return op.emitError(
          "autodiff supports single-input addf/maxnumf/minnumf reductions");
    Value grad = g->diffe(op->getResult(0), b);
    g->zeroDiffe(op->getResult(0), b);
    auto type = cast<RankedTensorType>(op.getOperand(0).getType());
    grad = expandReduction(b, op.getLoc(), grad, type, op.getAxis());
    if (!isSum) {
      Value input = g->popCache(caches[0], b);
      Value result = g->popCache(caches[1], b);
      Value weights = extremaWeights(b, op, input, result);
      grad = arith::MulFOp::create(b, op.getLoc(), weights, grad);
    }
    g->addToDiffe(op.getOperand(0), grad, b);
    return success();
  }
};

/// Control flow interface for Triton ReduceOp (needed for max/min)
class AutoDiffTritonReduceCF
    : public ControlFlowAutoDiffOpInterface::ExternalModel<
          AutoDiffTritonReduceCF, triton::ReduceOp> {
public:
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    auto reduce = cast<triton::ReduceOp>(original);
    auto newOp = builder.create<triton::ReduceOp>(
        original->getLoc(), remappedOperands, reduce.getAxis());
    IRMapping mapping;
    reduce.getCombineOp().cloneInto(&newOp.getCombineOp(), mapping);
    return newOp;
  }
};

} // end anonymous namespace

void mlir::enzyme::registerTritonDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, triton::TritonDialect *) {
    // Prefer the mixed-precision pullback to the generated same-type dot rule.
    // The generated forward rule is shared with the forward-mode PR.
    triton::DotOp::attachInterface<DotRev>(*context);
    registerInterfaces(context);
    triton::FuncOp::attachInterface<AutoDiffTritonFuncFunctionInterface>(
        *context);
    triton::PointerType::attachInterface<TritonPointerTypeInterface>(*context);
    triton::TensorDescType::attachInterface<TritonTensorDescTypeInterface>(
        *context);
    triton::LoadOp::attachInterface<LoadActivity, LoadFwd,
                                    MemoryRev<triton::LoadOp, true>>(*context);
    triton::StoreOp::attachInterface<MemoryRev<triton::StoreOp, false>>(
        *context);
    triton::DescriptorLoadOp::attachInterface<
        MemoryRev<triton::DescriptorLoadOp, true, true>>(*context);
    triton::DescriptorStoreOp::attachInterface<
        MemoryRev<triton::DescriptorStoreOp, false, true>>(*context);
    triton::ReduceOp::attachInterface<ReduceFwd, ReduceRev,
                                      AutoDiffTritonReduceCF>(*context);
    triton::AddPtrOp::attachInterface<ShapeRev<triton::AddPtrOp>>(*context);
    triton::MakeTensorDescOp::attachInterface<
        ShapeRev<triton::MakeTensorDescOp>>(*context);
    triton::SplatOp::attachInterface<ShapeRev<triton::SplatOp>>(*context);
    triton::BroadcastOp::attachInterface<ShapeRev<triton::BroadcastOp>>(
        *context);
    triton::ExpandDimsOp::attachInterface<ShapeRev<triton::ExpandDimsOp>>(
        *context);
    triton::ReshapeOp::attachInterface<ShapeRev<triton::ReshapeOp>>(*context);
    triton::TransOp::attachInterface<ShapeRev<triton::TransOp>>(*context);
  });
}

#else

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace enzyme {
void registerTritonDialectAutoDiffInterface(DialectRegistry &registry) {}
} // namespace enzyme
} // namespace mlir

#endif
