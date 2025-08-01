#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {

template <typename T> inline Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));

  if (isa<FloatType>(elemType)) {
    return FloatAttr::get(elemType, val);
  } else if (isa<IntegerType>(elemType)) {
    return IntegerAttr::get(elemType, val);
  } else if (auto complexType = dyn_cast<ComplexType>(elemType)) {
    auto elementType = complexType.getElementType();
    auto realAttr = makeAttr(elementType, val);
    auto imagAttr = makeAttr(elementType, 0);

    SmallVector<Attribute> complexVals = {realAttr, imagAttr};
    return ArrayAttr::get(elemType.getContext(), complexVals);
  } else {
    llvm_unreachable("Unsupported type");
  }
}

template <> inline Attribute makeAttr(mlir::Type elemType, llvm::APFloat val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<llvm::APFloat>(TT.getElementType(), val)));

  return FloatAttr::get(elemType, val);
}

// matcher for complex numbers. should probably be upstreamed at some point.
// https://github.com/llvm/llvm-project/blob/be6fc0092e44c7fa3981639cbfe692c78a5eb418/mlir/include/mlir/IR/Matchers.h#L162
struct constant_complex_value_binder {
  FloatAttr::ValueType *bindRealValue;
  FloatAttr::ValueType *bindImagValue;

  constant_complex_value_binder(FloatAttr::ValueType *realValue,
                                FloatAttr::ValueType *imagValue)
      : bindRealValue(realValue), bindImagValue(imagValue) {}

  bool match(Attribute attr) const {
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
      return match(splatAttr.getSplatValue<Attribute>());
    }

    auto arrayAttr = dyn_cast<ArrayAttr>(attr);
    if (!arrayAttr || arrayAttr.size() != 2)
      return false;

    auto realAttr = dyn_cast<FloatAttr>(arrayAttr.getValue()[0]);
    auto imagAttr = dyn_cast<FloatAttr>(arrayAttr.getValue()[1]);

    if (!realAttr || !imagAttr)
      return false;

    *bindRealValue = realAttr.getValue();
    *bindImagValue = imagAttr.getValue();
    return true;
  }

  bool match(Operation *op) const {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;

    Type type = op->getResult(0).getType();
    if (isa<ComplexType, VectorType, RankedTensorType>(type))
      return match(attr);

    return false;
  }
};

struct constant_complex_predicate_matcher {
  bool (*predicate)(const APFloat &, const APFloat &);

  bool match(Attribute attr) const {
    APFloat realValue(APFloat::Bogus());
    APFloat imagValue(APFloat::Bogus());
    return constant_complex_value_binder(&realValue, &imagValue).match(attr) &&
           predicate(realValue, imagValue);
  }

  bool match(Operation *op) const {
    APFloat realValue(APFloat::Bogus());
    APFloat imagValue(APFloat::Bogus());
    return constant_complex_value_binder(&realValue, &imagValue).match(op) &&
           predicate(realValue, imagValue);
  }
};

inline constant_complex_predicate_matcher m_AnyZeroComplex() {
  return {[](const APFloat &realValue, const APFloat &imagValue) {
    return realValue.isZero() && imagValue.isZero();
  }};
}

inline constant_complex_predicate_matcher m_AnyZeroRealComplex() {
  return {[](const APFloat &realValue, const APFloat &imagValue) {
    return realValue.isZero();
  }};
}

inline constant_complex_predicate_matcher m_AnyZeroImagComplex() {
  return {[](const APFloat &realValue, const APFloat &imagValue) {
    return imagValue.isZero();
  }};
}

inline ::mlir::detail::constant_int_predicate_matcher m_NegOne() {
  return {[](const APInt &value) { return value == -1; }};
}

inline ::mlir::detail::constant_float_predicate_matcher m_NegOneFloat() {
  return {[](const APFloat &value) { return value.isExactlyValue(-1.0); }};
}

static inline mlir::scf::IfOp cloneWithResults(mlir::scf::IfOp op,
                                               mlir::OpBuilder &rewriter,
                                               mlir::IRMapping mapping = {}) {
  using namespace mlir;
  return rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                    mapping.lookupOrDefault(op.getCondition()),
                                    true);
}
static inline mlir::affine::AffineIfOp
cloneWithResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
                 mlir::IRMapping mapping = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<affine::AffineIfOp>(op.getLoc(), op.getResultTypes(),
                                             op.getIntegerSet(), lower, true);
}

static inline mlir::scf::IfOp cloneWithoutResults(mlir::scf::IfOp op,
                                                  mlir::OpBuilder &rewriter,
                                                  mlir::IRMapping mapping = {},
                                                  mlir::TypeRange types = {}) {
  using namespace mlir;
  return rewriter.create<scf::IfOp>(
      op.getLoc(), types, mapping.lookupOrDefault(op.getCondition()), true);
}
static inline mlir::affine::AffineIfOp
cloneWithoutResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
                    mlir::IRMapping mapping = {}, mlir::TypeRange types = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<affine::AffineIfOp>(op.getLoc(), types,
                                             op.getIntegerSet(), lower, true);
}

static inline mlir::scf::ForOp
cloneWithoutResults(mlir::scf::ForOp op, mlir::PatternRewriter &rewriter,
                    mlir::IRMapping mapping = {}) {
  using namespace mlir;
  return rewriter.create<scf::ForOp>(
      op.getLoc(), mapping.lookupOrDefault(op.getLowerBound()),
      mapping.lookupOrDefault(op.getUpperBound()),
      mapping.lookupOrDefault(op.getStep()));
}
static inline mlir::affine::AffineForOp
cloneWithoutResults(mlir::affine::AffineForOp op,
                    mlir::PatternRewriter &rewriter,
                    mlir::IRMapping mapping = {}) {
  using namespace mlir;
  SmallVector<Value> lower;
  for (auto o : op.getLowerBoundOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  SmallVector<Value> upper;
  for (auto o : op.getUpperBoundOperands())
    upper.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<affine::AffineForOp>(
      op.getLoc(), lower, op.getLowerBoundMap(), upper, op.getUpperBoundMap(),
      op.getStepAsInt());
}

static inline void clearBlock(mlir::Block *block,
                              mlir::PatternRewriter &rewriter) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    rewriter.eraseOp(&op);
  }
}

static inline mlir::Block *getThenBlock(mlir::scf::IfOp op) {
  return op.thenBlock();
}
static inline mlir::Block *getThenBlock(mlir::affine::AffineIfOp op) {
  return op.getThenBlock();
}
static inline mlir::Block *getElseBlock(mlir::scf::IfOp op) {
  return op.elseBlock();
}
static inline mlir::Block *getElseBlock(mlir::affine::AffineIfOp op) {
  if (op.hasElse())
    return op.getElseBlock();
  else
    return nullptr;
}

static inline mlir::Region &getThenRegion(mlir::scf::IfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getThenRegion(mlir::affine::AffineIfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getElseRegion(mlir::scf::IfOp op) {
  return op.getElseRegion();
}
static inline mlir::Region &getElseRegion(mlir::affine::AffineIfOp op) {
  return op.getElseRegion();
}

static inline mlir::scf::YieldOp getThenYield(mlir::scf::IfOp op) {
  return op.thenYield();
}
static inline mlir::affine::AffineYieldOp
getThenYield(mlir::affine::AffineIfOp op) {
  return llvm::cast<mlir::affine::AffineYieldOp>(
      op.getThenBlock()->getTerminator());
}
static inline mlir::scf::YieldOp getElseYield(mlir::scf::IfOp op) {
  return op.elseYield();
}
static inline mlir::affine::AffineYieldOp
getElseYield(mlir::affine::AffineIfOp op) {
  return llvm::cast<mlir::affine::AffineYieldOp>(
      op.getElseBlock()->getTerminator());
}

static inline bool inBound(mlir::scf::IfOp op, mlir::Value v) {
  return op.getCondition() == v;
}
static inline bool inBound(mlir::affine::AffineIfOp op, mlir::Value v) {
  return llvm::any_of(op.getOperands(), [&](mlir::Value e) { return e == v; });
}
static inline bool inBound(mlir::scf::ForOp op, mlir::Value v) {
  return op.getUpperBound() == v;
}
static inline bool inBound(mlir::affine::AffineForOp op, mlir::Value v) {
  return llvm::any_of(op.getUpperBoundOperands(),
                      [&](mlir::Value e) { return e == v; });
}
static inline bool hasElse(mlir::scf::IfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}
static inline bool hasElse(mlir::affine::AffineIfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}

const std::set<std::string> &getNonCapturingFunctions();

bool collectEffects(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool ignoreBarriers);

bool getEffectsBefore(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool stopAtBarrier);

bool getEffectsAfter(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool stopAtBarrier);

bool isReadOnly(mlir::Operation *);
bool isReadNone(mlir::Operation *);

bool mayReadFrom(mlir::Operation *, mlir::Value);
bool mayWriteTo(mlir::Operation *, mlir::Value, bool ignoreBarrier = false);

bool mayAlias(mlir::MemoryEffects::EffectInstance a,
              mlir::MemoryEffects::EffectInstance b);

bool mayAlias(mlir::MemoryEffects::EffectInstance a, mlir::Value b);

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty);
bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy, Type inTy);

bool anyOperandIsConstant(mlir::Operation *op);
bool allOperandsAreConstant(mlir::Operation *op);

/// Swap side of predicate
static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
    return pred;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

} // namespace enzyme

namespace stablehlo {

stablehlo::GatherDimensionNumbersAttr
getGatherDims(mlir::MLIRContext *ctx,
              stablehlo::ScatterDimensionNumbersAttr scatterDimNumbers);

bool isScatterSetindexOp(stablehlo::ScatterOp &op);

SmallVector<int64_t> computeGatherSliceSizes(stablehlo::ScatterOp &scatterOp);

template <typename T>
stablehlo::ConstantOp createConstantOpFromScalar(PatternRewriter &rewriter,
                                                 Operation *op, T value) {
  return rewriter.create<stablehlo::ConstantOp>(
      op->getLoc(), op->getResult(0).getType(),
      cast<ElementsAttr>(
          mlir::enzyme::makeAttr(op->getResult(0).getType(), value)));
}

stablehlo::ComparisonDirection
reversedComparisonDirection(stablehlo::ComparisonDirection direction);

stablehlo::ComparisonDirection
negatedComparisonDirection(stablehlo::ComparisonDirection direction);

} // namespace stablehlo

} // namespace mlir
