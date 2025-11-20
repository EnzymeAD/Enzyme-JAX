#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <deque>

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
  return scf::IfOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                           mapping.lookupOrDefault(op.getCondition()), true);
}
static inline mlir::affine::AffineIfOp
cloneWithResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
                 mlir::IRMapping mapping = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return affine::AffineIfOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                                    op.getIntegerSet(), lower, true);
}

static inline mlir::scf::IfOp cloneWithoutResults(mlir::scf::IfOp op,
                                                  mlir::OpBuilder &rewriter,
                                                  mlir::IRMapping mapping = {},
                                                  mlir::TypeRange types = {}) {
  using namespace mlir;
  return scf::IfOp::create(rewriter, op.getLoc(), types,
                           mapping.lookupOrDefault(op.getCondition()), true);
}
static inline mlir::affine::AffineIfOp
cloneWithoutResults(mlir::affine::AffineIfOp op, mlir::OpBuilder &rewriter,
                    mlir::IRMapping mapping = {}, mlir::TypeRange types = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return affine::AffineIfOp::create(rewriter, op.getLoc(), types,
                                    op.getIntegerSet(), lower, true);
}

static inline mlir::scf::ForOp
cloneWithoutResults(mlir::scf::ForOp op, mlir::PatternRewriter &rewriter,
                    mlir::IRMapping mapping = {}) {
  using namespace mlir;
  return scf::ForOp::create(rewriter, op.getLoc(),
                            mapping.lookupOrDefault(op.getLowerBound()),
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
  auto newFor = affine::AffineForOp::create(
      rewriter, op.getLoc(), lower, op.getLowerBoundMap(), upper,
      op.getUpperBoundMap(), op.getStepAsInt());
  for (auto attr : op->getDiscardableAttrs())
    newFor->setAttr(attr.getName(), attr.getValue());
  return newFor;
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
bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty,
                          mlir::Operation *op, PatternRewriter &rewriter);
bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy, Type inTy);
bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy, Type inTy,
                          mlir::Operation *op, PatternRewriter &rewriter);

bool canApplySymmetricPattern(mlir::Operation *op, PatternRewriter &rewriter);
bool canApplySymmetricPattern(mlir::Value val, PatternRewriter &rewriter);

template <typename Child> class GuaranteedResultAnalysisBase {
protected:
  llvm::DenseMap<mlir::Value, bool> valueCache;

public:
  enum class State {
    // We know this is _not_ guaranteed.
    NOTGUARANTEED = 0,
    // We know this is guaranteed.
    GUARANTEED = 1,
    // This is guarnateed, pending the results of the new Operations.
    PENDING = 2,
    // Might be returned when parsing the IR for preexisiting guarantees
    UNKNOWN = 3
  };

  bool guaranteed(mlir::Value value, PatternRewriter &rewriter) {
    auto it = valueCache.find(value);
    if (it != valueCache.end())
      return it->second;

    State stateFromIR = lookupGuaranteedFromIR(value, rewriter);
    if (stateFromIR != State::UNKNOWN) {
      return stateFromIR == State::GUARANTEED;
    }

    // Map of values we need to still check. If all of these are guaranteed
    // we therefore know that the operation `op` is guaranteed
    std::deque<mlir::Value> todo = {value};

    // Map of values we have seen before. The target of the map[o] is a list
    // of sub-queries, that if all true prove that `o` is guaranteed.
    llvm::MapVector<mlir::Value, llvm::SmallPtrSet<mlir::Value, 2>> seen;

    // Inverse of seen. A map of values `p` we still need to prove, to a
    // list of values that require `p` to be proven.
    DenseMap<mlir::Value, SmallVector<mlir::Value, 2>> reverseSeen;

    while (!todo.empty()) {
      auto cur = todo.front();
      todo.pop_front();

      if (seen.find(cur) != seen.end())
        continue;

      SmallVector<mlir::Value> localtodo;
      State status;

      {
        auto found = valueCache.find(cur);
        if (found != valueCache.end()) {
          if (found->second) {
            status = State::GUARANTEED;
          } else {
            status = State::NOTGUARANTEED;
          }
        } else {
          status = localGuaranteedWithSetAttr(cur, localtodo, rewriter);
        }
      }

      switch (status) {
      case State::NOTGUARANTEED: {
        SmallVector<Value, 2> rtodo{cur};
        while (!rtodo.empty()) {
          auto rcur = rtodo.pop_back_val();
          if (valueCache.find(rcur) != valueCache.end()) {
            continue;
          }
          valueCache[rcur] = false;
          setGuaranteedInIR(rcur, false, rewriter);

          auto rfound = reverseSeen.find(rcur);
          if (rfound != reverseSeen.end()) {
            for (auto next : rfound->second) {
              rtodo.push_back(next);
            }
            reverseSeen.erase(rfound);
          }
        }

        setGuaranteedInIR(cur, false, rewriter);
        return false;
      }

      case State::GUARANTEED: {
        // Operations which are now guaranteed
        SmallVector<Value, 2> rtodo = {cur};

        while (!rtodo.empty()) {

          auto rcur = rtodo.pop_back_val();
          if (valueCache.find(rcur) != valueCache.end()) {
            continue;
          }

          {
            auto rfound = seen.find(rcur);
            if (rfound != seen.end()) {
              seen.erase(rfound);
            }
          }

          // This is now an value we have not previously marked as
          // guaranteed
          valueCache[rcur] = true;

          // Look if this is one we have previously visited this value as a
          // pending value, and if so, remove the corresponding pending
          // dependencies

          auto rfound = reverseSeen.find(rcur);
          if (rfound == reverseSeen.end()) {
            continue;
          }

          for (auto next : rfound->second) {
            auto bfound = seen.find(next);
            assert(bfound != seen.end());
            bfound->second.erase(rcur);
            if (bfound->second.empty())
              rtodo.push_back(next);
          }

          reverseSeen.erase(rcur);
        }
        break;
      }
      case State::PENDING: {
        assert(localtodo.size());
        assert(seen.find(cur) == seen.end());
        for (auto v : localtodo) {
          reverseSeen[v].push_back(cur);
          todo.push_back(v);
        }
        llvm::SmallPtrSet<mlir::Value, 2> set(localtodo.begin(),
                                              localtodo.end());
        seen[cur] = std::move(set);
        break;
      }
      }
    }

    // We have checked all recursive dependencies, and found no values which
    // would invalidate. Therefore all seen operations [including op] are known
    // to be guaranteed.
    for (auto &sval : seen) {
      valueCache[sval.first] = true;
      setGuaranteedInIR(sval.first, true, rewriter);
    }

    auto found = valueCache.find(value);
    if (found != valueCache.end()) {
      bool guaranteed = found->second;
      setGuaranteedInIR(value, guaranteed, rewriter);
      return guaranteed;
    }
    return false;
  }

  State guaranteedConstant(Value val, PatternRewriter &rewriter) {
    auto it = valueCache.find(val);
    if (it != valueCache.end())
      return it->second ? State::GUARANTEED : State::NOTGUARANTEED;

    DenseElementsAttr denseAttr;
    if (!matchPattern(val, m_Constant(&denseAttr)))
      return State::UNKNOWN;

    State state = State::NOTGUARANTEED;
    if (denseAttr.getType().getShape().size() && denseAttr.isSplat()) {
      denseAttr = denseAttr.resizeSplat(
          RankedTensorType::get({}, denseAttr.getType().getElementType()));
    }

    // For floating point values
    if (isa<FloatType>(denseAttr.getElementType())) {
      if (((Child *)this)->constantFloatCheck(denseAttr)) {
        state = State::GUARANTEED;
      }
    }

    // For integer values
    if (isa<IntegerType>(denseAttr.getElementType())) {
      if (((Child *)this)->constantIntCheck(denseAttr)) {
        state = State::GUARANTEED;
      }
    }

    setGuaranteedInIR(val, state, rewriter);
    return state;
  }

  State localGuaranteedWithSetAttr(Value val, SmallVectorImpl<Value> &localtodo,
                                   PatternRewriter &rewriter) {
    auto stateFromIR = lookupGuaranteedFromIR(val, rewriter);
    if (stateFromIR != State::UNKNOWN)
      return stateFromIR;

    auto stateFromConstant = guaranteedConstant(val, rewriter);
    if (stateFromConstant != State::UNKNOWN)
      return stateFromConstant;

    auto state = ((Child *)this)->localGuaranteed(val, localtodo, rewriter);

    auto attrName = ((Child *)this)->getAttrName();
    setGuaranteedInIR(val, state, rewriter);
    return state;
  }

private:
  State lookupGuaranteedFromIR(Value val, PatternRewriter &rewriter) {
    auto op = val.getDefiningOp();
    if (!op)
      return State::UNKNOWN;

    auto attrName = ((Child *)this)->getAttrName();
    auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!arrayAttr)
      return State::UNKNOWN;

    auto opResult = dyn_cast<mlir::OpResult>(val);
    if (!opResult)
      return State::UNKNOWN;

    // invalidate if arrayAttr size doesn't match. can happen for ops like while
    // where the inputs/results were modified
    if (arrayAttr.size() != op->getNumResults()) {
      rewriter.modifyOpInPlace(op, [&]() { op->removeAttr(attrName); });
      return State::UNKNOWN;
    }

    auto resultNumber = opResult.getResultNumber();
    auto attr = arrayAttr[resultNumber];
    auto enumAttr = dyn_cast<enzymexla::GuaranteedAnalysisResultAttr>(attr);
    assert(enumAttr && "Expected guaranteed analysis result");

    switch (enumAttr.getValue()) {
    case enzymexla::GuaranteedAnalysisResult::GUARANTEED:
      return State::GUARANTEED;
    case enzymexla::GuaranteedAnalysisResult::NOTGUARANTEED:
      return State::NOTGUARANTEED;
    case enzymexla::GuaranteedAnalysisResult::UNKNOWN:
      return State::UNKNOWN;
    default:
      llvm_unreachable("Unexpected guaranteed analysis result");
    }
  }

  void setGuaranteedInIR(Value val, bool guaranteed,
                         PatternRewriter &rewriter) {
    setGuaranteedInIR(
        val, guaranteed ? State::GUARANTEED : State::NOTGUARANTEED, rewriter);
  }

  void setGuaranteedInIR(Value val, State state, PatternRewriter &rewriter) {
    if (state == State::UNKNOWN || state == State::PENDING)
      return;

    auto op = val.getDefiningOp();
    if (!op)
      return;

    auto opResult = dyn_cast<mlir::OpResult>(val);
    if (!opResult)
      return;

    auto resultNumber = opResult.getResultNumber();

    auto attrName = ((Child *)this)->getAttrName();
    auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);

    SmallVector<Attribute> newAttrs;

    // if arrayAttr size doesn't match invalidate the results. can happen
    // for ops like while where the inputs/results were modified
    if (!arrayAttr || arrayAttr.size() != op->getNumResults()) {
      auto unknownAttr = enzymexla::GuaranteedAnalysisResultAttr::get(
          val.getContext(), enzymexla::GuaranteedAnalysisResult::UNKNOWN);

      for (auto i = 0; i < op->getNumResults(); i++) {
        newAttrs.push_back(unknownAttr);
      }
    } else {
      Attribute attr = arrayAttr[resultNumber];
      auto enumAttr = dyn_cast<enzymexla::GuaranteedAnalysisResultAttr>(attr);
      assert(enumAttr && "Expected guaranteed analysis result");

      newAttrs = SmallVector<Attribute>(arrayAttr.begin(), arrayAttr.end());
      assert(newAttrs.size() == op->getNumResults());
    }

    enzymexla::GuaranteedAnalysisResult newValue;
    switch (state) {
    case State::GUARANTEED:
      newValue = enzymexla::GuaranteedAnalysisResult::GUARANTEED;
      break;
    case State::NOTGUARANTEED:
      newValue = enzymexla::GuaranteedAnalysisResult::NOTGUARANTEED;
      break;
    default:
      llvm_unreachable("Unexpected state");
    }

    auto newEnumAttr = enzymexla::GuaranteedAnalysisResultAttr::get(
        val.getContext(), newValue);

    newAttrs[resultNumber] = newEnumAttr;

    rewriter.modifyOpInPlace(op, [&]() {
      op->setAttr(attrName, ArrayAttr::get(val.getContext(), newAttrs));
    });
  }
};

class FiniteResultAnalysis;
class NoNanResultAnalysis;
class SymmetricResultAnalysis;

class SymmetricResultAnalysis
    : public GuaranteedResultAnalysisBase<SymmetricResultAnalysis> {
public:
  State localGuaranteed(Value val, SmallVectorImpl<Value> &localtodo,
                        PatternRewriter &rewriter);

  bool constantFloatCheck(DenseElementsAttr attr);
  bool constantIntCheck(DenseElementsAttr attr);

  StringRef getAttrName() const { return "enzymexla.symmetric_matrix"; }
};

class NoNanResultAnalysis
    : public GuaranteedResultAnalysisBase<NoNanResultAnalysis> {
private:
  std::shared_ptr<FiniteResultAnalysis> finiteResultAnalysis = nullptr;

public:
  State localGuaranteed(Value val, SmallVectorImpl<Value> &localtodo,
                        PatternRewriter &rewriter);

  bool constantFloatCheck(DenseElementsAttr attr);
  bool constantIntCheck(DenseElementsAttr attr);

  StringRef getAttrName() const { return "enzymexla.no_nan"; }

  void setFiniteResultAnalysis(std::shared_ptr<FiniteResultAnalysis> analysis) {
    finiteResultAnalysis = analysis;
  }
};

class FiniteResultAnalysis
    : public GuaranteedResultAnalysisBase<FiniteResultAnalysis> {
private:
  std::shared_ptr<NoNanResultAnalysis> noNanResultAnalysis = nullptr;

public:
  bool constantFloatCheck(DenseElementsAttr attr);
  bool constantIntCheck(DenseElementsAttr attr);

  StringRef getAttrName() const { return "enzymexla.finite"; }

  State localGuaranteed(Value val, SmallVectorImpl<Value> &localtodo,
                        PatternRewriter &rewriter);

  void setNoNanResultAnalysis(std::shared_ptr<NoNanResultAnalysis> analysis) {
    noNanResultAnalysis = analysis;
  }
};

NoNanResultAnalysis initNoNanResultAnalysis();
FiniteResultAnalysis initFiniteResultAnalysis();
SymmetricResultAnalysis initSymmetricResultAnalysis();

template <typename T>
bool runAnalysisOnOperation(T analysis, Operation *op,
                            PatternRewriter &rewriter) {
  if (!op)
    return false;

  for (auto res : op->getResults()) {
    if (!analysis.guaranteed(res, rewriter)) {
      return false;
    }
  }
  return true;
}

inline bool guaranteedNoNanResult(mlir::Value value,
                                  PatternRewriter &rewriter) {
  return initNoNanResultAnalysis().guaranteed(value, rewriter);
}
inline bool guaranteedNoNanResult(Operation *op, PatternRewriter &rewriter) {
  auto analysis = initNoNanResultAnalysis();
  return runAnalysisOnOperation<NoNanResultAnalysis>(analysis, op, rewriter);
}

inline bool guaranteedFiniteResult(mlir::Value value,
                                   PatternRewriter &rewriter) {
  return initFiniteResultAnalysis().guaranteed(value, rewriter);
}
inline bool guaranteedFiniteResult(Operation *op, PatternRewriter &rewriter) {
  auto analysis = initFiniteResultAnalysis();
  return runAnalysisOnOperation<FiniteResultAnalysis>(analysis, op, rewriter);
}

inline bool guaranteedSymmetricResult(mlir::Value value,
                                      PatternRewriter &rewriter) {
  return initSymmetricResultAnalysis().guaranteed(value, rewriter);
}
inline bool guaranteedSymmetricResult(Operation *op,
                                      PatternRewriter &rewriter) {
  auto analysis = initSymmetricResultAnalysis();
  return runAnalysisOnOperation<SymmetricResultAnalysis>(analysis, op,
                                                         rewriter);
}

class NonNegativeResultAnalysis
    : public GuaranteedResultAnalysisBase<NonNegativeResultAnalysis> {
public:
  bool constantFloatCheck(DenseElementsAttr attr);
  bool constantIntCheck(DenseElementsAttr attr);

  StringRef getAttrName() const { return "enzymexla.non_negative"; }

  State localGuaranteed(Value val, SmallVectorImpl<Value> &localtodo,
                        PatternRewriter &rewriter);
};

inline bool guaranteedNonNegativeResult(mlir::Value value,
                                        PatternRewriter &rewriter) {
  return NonNegativeResultAnalysis().guaranteed(value, rewriter);
}
inline bool guaranteedNonNegativeResult(Operation *op,
                                        PatternRewriter &rewriter) {
  auto analysis = NonNegativeResultAnalysis();
  return runAnalysisOnOperation<NonNegativeResultAnalysis>(analysis, op,
                                                           rewriter);
}

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

SmallVector<int64_t> findReshapeInsertionDims(RankedTensorType inputType,
                                              RankedTensorType outputType);
SmallVector<int64_t> findReshapeInsertionDims(ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> outputShape);

bool areValidInsertionDims(RankedTensorType inputType,
                           RankedTensorType outputType,
                           SmallVector<int64_t> insertionDims);

bool isOnlyUsedInOperation(Operation *operation, Operation *parentOp);

} // namespace enzyme

namespace stablehlo {

stablehlo::GatherDimensionNumbersAttr
getGatherDims(mlir::MLIRContext *ctx,
              stablehlo::ScatterDimensionNumbersAttr scatterDimNumbers);

bool isSetindexBlock(mlir::Block *block);

template <typename T> bool isCommutativeOpBlock(mlir::Block *block) {
  if (block->getNumArguments() != 2)
    return false;

  if (!hasSingleElement(block->without_terminator()))
    return false;

  auto op = dyn_cast<T>(block->front());
  if (!op)
    return false;

  if (op.getNumOperands() != 2)
    return false;

  if (!(op->getOperand(0) == block->getArgument(0) &&
        op->getOperand(1) == block->getArgument(1)) &&
      !(op->getOperand(0) == block->getArgument(1) &&
        op->getOperand(1) == block->getArgument(0)))
    return false;

  auto returnOp = block->getTerminator();
  auto stablehloReturnOp = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturnOp)
    return false;

  if (stablehloReturnOp.getNumOperands() != 1)
    return false;

  // The returned value should be the result of the addition
  return stablehloReturnOp.getOperand(0) == op.getResult();
}

struct CheckCommonReduceOp {
public:
  bool isAddReduce;
  bool isMinReduce;
  bool isMaxReduce;
  bool isMulReduce;

  CheckCommonReduceOp(stablehlo::ReduceOp op) {
    auto &region = op.getRegion();
    if (region.getBlocks().size() != 1) {
      isAddReduce = false;
      isMinReduce = false;
      isMaxReduce = false;
      isMulReduce = false;
      return;
    }

    auto &block = region.getBlocks().front();
    isAddReduce = isCommutativeOpBlock<stablehlo::AddOp>(&block);
    isMinReduce = isCommutativeOpBlock<stablehlo::MinOp>(&block);
    isMaxReduce = isCommutativeOpBlock<stablehlo::MaxOp>(&block);
    isMulReduce = isCommutativeOpBlock<stablehlo::MulOp>(&block);
  }

  bool isCommonReduce() {
    return isAddReduce || isMinReduce || isMaxReduce || isMulReduce;
  }
};

struct CheckCommonScatterOp {
public:
  bool isSetindexScatter;
  bool isAddScatter;

  CheckCommonScatterOp(stablehlo::ScatterOp op) {
    auto &updateComputation = op.getUpdateComputation();

    if (!updateComputation.hasOneBlock()) {
      isSetindexScatter = false;
      isAddScatter = false;
      return;
    }

    auto &block = updateComputation.front();
    isSetindexScatter = isSetindexBlock(&block);
    isAddScatter = isCommutativeOpBlock<stablehlo::AddOp>(&block);
  }
};

SmallVector<int64_t> computeGatherSliceSizes(stablehlo::ScatterOp &scatterOp);

template <typename T>
stablehlo::ConstantOp createConstantOpFromScalar(PatternRewriter &rewriter,
                                                 Operation *op, T value) {
  return stablehlo::ConstantOp::create(
      rewriter, op->getLoc(), op->getResult(0).getType(),
      cast<ElementsAttr>(
          mlir::enzyme::makeAttr(op->getResult(0).getType(), value)));
}

stablehlo::ComparisonDirection
reversedComparisonDirection(stablehlo::ComparisonDirection direction);

stablehlo::ComparisonDirection
negatedComparisonDirection(stablehlo::ComparisonDirection direction);

bool reshapeIsTranspose(stablehlo::ReshapeOp reshapeOp);

mlir::Value reshapeAxisInto(OpBuilder &builder, Value input,
                            ArrayRef<int64_t> &batchSizes, int64_t dim);

mlir::Value reshapeAxisOutOf(OpBuilder &builder, Value input,
                             ArrayRef<int64_t> &batchSizes, int64_t dim);

// matches for hasTrait<OpTrait::Elementwise>. Additionally matches for
// hasTrait<OpTrait::HLOBroadcastingElementwise> if all of the operands are
// of the same shape.
bool hasTraitElementwise(Operation *op);

// currently there are no traits for associative ops
bool isAssociativeOp(Operation *op);

} // namespace stablehlo

} // namespace mlir
