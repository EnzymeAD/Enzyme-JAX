#pragma once

#include "Enzyme/MLIR/Interfaces/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Value.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "absl/status/status.h"

#include "shardy/dialect/sdy/ir/utils.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
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

#include <deque>

namespace mlir {
namespace enzyme {

void commonLowerUpdateWithoutCorners(enzymexla::UpdateWithoutCornersOp extend,
                                     PatternRewriter &rewriter);

using namespace ::mlir::enzyme::oputils;

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

template <> inline Attribute makeAttr(mlir::Type elemType, llvm::APInt val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<llvm::APInt>(TT.getElementType(), val)));

  return IntegerAttr::get(elemType, val);
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

inline ::mlir::detail::constant_int_predicate_matcher m_AllOnes() {
  return {[](const APInt &value) { return value.isAllOnes(); }};
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

bool mayReadFrom(mlir::Operation *, mlir::Value);
bool mayWriteTo(mlir::Operation *, mlir::Value, bool ignoreBarrier = false);

template <typename AttrTy, typename T>
SmallVector<Attribute> getUpdatedAttrList(Value val, StringRef attrName,
                                          T unknownValue, T newValue) {
  auto ctx = val.getContext();

  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return {AttrTy::get(ctx, newValue)};
  }

  auto op = val.getDefiningOp();
  assert(op);

  auto resultNumber = cast<OpResult>(val).getResultNumber();

  auto arrayAttr = op->template getAttrOfType<ArrayAttr>(attrName);

  SmallVector<Attribute> newAttrs;

  // if arrayAttr size doesn't match invalidate the results. can happen
  // for ops like while where the inputs/results were modified
  if (!arrayAttr || arrayAttr.size() != op->getNumResults()) {
    auto unknownAttr = AttrTy::get(ctx, unknownValue);

    for (auto i = 0; i < op->getNumResults(); i++) {
      newAttrs.push_back(unknownAttr);
    }
  } else {
    Attribute attr = arrayAttr[resultNumber];
    auto enumAttr = dyn_cast<AttrTy>(attr);
    (void)enumAttr;
    assert(enumAttr && "Expected guaranteed analysis result");

    newAttrs = SmallVector<Attribute>(arrayAttr.begin(), arrayAttr.end());
    assert(newAttrs.size() == op->getNumResults());
  }

  newAttrs[resultNumber] = AttrTy::get(ctx, newValue);
  return newAttrs;
}

template <typename AttrTy, typename T>
T getAttributeFromIR(Value val, StringRef attrName, T unknownValue) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (!parentOp) {
      return unknownValue;
    }

    auto funcOpInterface = dyn_cast<mlir::FunctionOpInterface>(parentOp);
    if (!funcOpInterface) {
      return unknownValue;
    }

    auto argAttrs = funcOpInterface.getArgAttrs(blockArg.getArgNumber());
    for (auto attr : argAttrs) {
      if (attr.getName() == attrName) {
        auto enumAttr = dyn_cast<AttrTy>(attr.getValue());
        assert(enumAttr && "Expected guaranteed analysis result");
        return enumAttr.getValue();
      }
    }

    return unknownValue;
  }

  auto op = val.getDefiningOp();
  assert(op);

  auto arrayAttr = op->template getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() != op->getNumResults()) {
    return unknownValue;
  }

  auto opResult = dyn_cast<OpResult>(val);
  if (!opResult) {
    return unknownValue;
  }

  auto attr = arrayAttr[opResult.getResultNumber()];
  auto enumAttr = dyn_cast<AttrTy>(attr);
  assert(enumAttr && "Expected guaranteed analysis result");
  return enumAttr.getValue();
}

/// Get bounds attribute from IR. Bounds are stored as ArrayAttr with two
/// IntegerAttr elements [min, max] under the attribute name "enzymexla.bounds".
/// Returns nullopt if the attribute is not found or malformed.
inline std::optional<std::pair<APInt, APInt>>
getBoundsFromIR(Value val, unsigned bitWidth) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto parentOp = blockArg.getOwner()->getParentOp();
    if (!parentOp)
      return std::nullopt;

    auto funcOpInterface = dyn_cast<mlir::FunctionOpInterface>(parentOp);
    if (!funcOpInterface)
      return std::nullopt;

    auto argAttrs = funcOpInterface.getArgAttrs(blockArg.getArgNumber());
    for (auto attr : argAttrs) {
      if (attr.getName() == "enzymexla.bounds") {
        auto boundsAttr = dyn_cast<ArrayAttr>(attr.getValue());
        if (!boundsAttr || boundsAttr.size() != 2)
          return std::nullopt;

        auto minAttr = dyn_cast<IntegerAttr>(boundsAttr[0]);
        auto maxAttr = dyn_cast<IntegerAttr>(boundsAttr[1]);
        if (!minAttr || !maxAttr)
          return std::nullopt;

        auto minVal = minAttr.getValue().sextOrTrunc(bitWidth);
        auto maxVal = maxAttr.getValue().sextOrTrunc(bitWidth);
        return std::make_pair(minVal, maxVal);
      }
    }

    return std::nullopt;
  }

  auto op = val.getDefiningOp();
  if (!op)
    return std::nullopt;

  auto boundsAttr = op->getAttrOfType<ArrayAttr>("enzymexla.bounds");
  if (!boundsAttr || boundsAttr.size() != op->getNumResults())
    return std::nullopt;

  auto opResult = dyn_cast<OpResult>(val);
  if (!opResult)
    return std::nullopt;

  auto resultBounds =
      dyn_cast<ArrayAttr>(boundsAttr[opResult.getResultNumber()]);
  if (!resultBounds || resultBounds.size() != 2)
    return std::nullopt;

  auto minAttr = dyn_cast<IntegerAttr>(resultBounds[0]);
  auto maxAttr = dyn_cast<IntegerAttr>(resultBounds[1]);
  if (!minAttr || !maxAttr)
    return std::nullopt;

  auto minVal = minAttr.getValue().sextOrTrunc(bitWidth);
  auto maxVal = maxAttr.getValue().sextOrTrunc(bitWidth);
  return std::make_pair(minVal, maxVal);
}

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
      case State::UNKNOWN:
        llvm_unreachable("Unknown state not handled");
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

    setGuaranteedInIR(val, state, rewriter);
    return state;
  }

protected:
  template <typename ItTy>
  State recursivelyCheckOperands(SmallVectorImpl<Value> &localtodo,
                                 ItTy operands, bool skipIntegerEltypes) {
    assert(!operands.empty() && "expected operands to not be empty");

    bool allOperandsGuaranteed = true;
    for (auto operand : operands) {
      if (skipIntegerEltypes) {
        if (auto TT = dyn_cast<TensorType>(operand.getType())) {
          if (TT.getElementType().isInteger()) {
            continue;
          }
        }
      }

      auto found = valueCache.find(operand);
      if (found != valueCache.end()) {
        if (found->second) {
          continue;
        }
        return State::NOTGUARANTEED;
      }

      localtodo.push_back(operand);
      allOperandsGuaranteed = false;
    }

    return allOperandsGuaranteed ? State::GUARANTEED : State::PENDING;
  }

private:
  State
  GuaranteedAnalysisResultToState(enzymexla::GuaranteedAnalysisResult val) {
    switch (val) {
    case enzymexla::GuaranteedAnalysisResult::GUARANTEED:
      return State::GUARANTEED;
    case enzymexla::GuaranteedAnalysisResult::NOTGUARANTEED:
      return State::NOTGUARANTEED;
    case enzymexla::GuaranteedAnalysisResult::UNKNOWN:
      return State::UNKNOWN;
    default:
      llvm_unreachable("Unhandled state");
    }
  }

  State lookupGuaranteedFromIR(Value val, PatternRewriter &rewriter) {
    return GuaranteedAnalysisResultToState(
        getAttributeFromIR<enzymexla::GuaranteedAnalysisResultAttr>(
            val, ((Child *)this)->getAttrName(),
            enzymexla::GuaranteedAnalysisResult::UNKNOWN));
  }

  void setGuaranteedInIR(Value val, bool guaranteed,
                         PatternRewriter &rewriter) {
    setGuaranteedInIR(
        val, guaranteed ? State::GUARANTEED : State::NOTGUARANTEED, rewriter);
  }

  void setGuaranteedInIR(Value val, State state, PatternRewriter &rewriter) {
    if (state == State::UNKNOWN || state == State::PENDING) {
      return;
    }

    auto op = val.getDefiningOp();
    if (!op) {
      return;
    }

    auto attrName = ((Child *)this)->getAttrName();

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

    auto newAttrs = getUpdatedAttrList<enzymexla::GuaranteedAnalysisResultAttr>(
        val, attrName, enzymexla::GuaranteedAnalysisResult::UNKNOWN, newValue);

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

bool isInsertDimOp(stablehlo::ReshapeOp reshapeOp);
bool isDeleteDimOp(stablehlo::ReshapeOp reshapeOp);

void getSingletonInsertionDims(stablehlo::BroadcastInDimOp bcastOp,
                               SmallVectorImpl<int64_t> &insertionDims);

bool areValidInsertionDims(RankedTensorType inputType,
                           RankedTensorType outputType,
                           SmallVector<int64_t> insertionDims);

bool getCollapsingMapping(
    llvm::ArrayRef<int64_t> oldShape, llvm::ArrayRef<int64_t> newShape,
    llvm::DenseMap<int64_t, llvm::SmallVector<int64_t, 2>> &mapping);

bool isOnlyUsedInOperation(Operation *operation, Operation *parentOp);
bool isValueOnlyUsedInOperation(Value value, Operation *parentOp);

mlir::RankedTensorType removeBatchedDims(mlir::RankedTensorType Ty,
                                         llvm::ArrayRef<int64_t> dims);

enzymexla::LapackTranspose
transposeLapackTranspose(enzymexla::LapackTranspose trans, bool canBeComplex);

enzymexla::LapackUplo transposeLapackUplo(enzymexla::LapackUplo uplo);

enzymexla::LapackUplo standardizeUplo(enzymexla::LapackUplo uplo);

using InputValidatorFn = std::function<bool(mlir::Value)>;

absl::Status detectConstantSetindexScatterOp(
    stablehlo::ScatterOp scatterOp, bool allowedMultipleUses,
    InputValidatorFn inputValidator, SplatElementsAttr &constSetIndexValue);
absl::Status detectConstantSetindexScatterOp(stablehlo::ScatterOp scatterOp,
                                             bool allowedMultipleUses,
                                             InputValidatorFn inputValidator);

absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates,
                                  InputValidatorFn inputValidator);
absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp,
                                  mlir::Value *outUpdates);
absl::Status detectDiagonalTensor(stablehlo::ScatterOp scatterOp);

// Tensor indexing utilities for multi-dimensional arrays

// Compute row-major strides for a given shape
inline llvm::SmallVector<int64_t>
computeStrides(llvm::ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  llvm::SmallVector<int64_t> strides(rank, 1);
  for (int64_t i = rank - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Convert a linear index to multi-dimensional indices
inline void linearToMultiIndex(int64_t linearIdx,
                               llvm::ArrayRef<int64_t> strides,
                               llvm::SmallVectorImpl<int64_t> &indices) {
  indices.resize(strides.size());
  for (size_t d = 0; d < strides.size(); d++) {
    indices[d] = linearIdx / strides[d];
    linearIdx = linearIdx % strides[d];
  }
}

// Convert multi-dimensional indices to a linear index
inline int64_t multiToLinearIndex(llvm::ArrayRef<int64_t> indices,
                                  llvm::ArrayRef<int64_t> strides) {
  int64_t linearIdx = 0;
  for (size_t d = 0; d < strides.size(); d++) {
    linearIdx += indices[d] * strides[d];
  }
  return linearIdx;
}

struct IotaLikeTensor {
  mlir::TypedAttr start;
  int64_t dimension;
  mlir::TypedAttr scale; // multiplicative factor applied to the iota
  mlir::RankedTensorType tensorType;
};

std::optional<IotaLikeTensor> detectIotaLikeTensor(DenseElementsAttr attr);
std::optional<IotaLikeTensor> detectIotaLikeTensor(mlir::Value tensor);

// Represents a constant tensor that can be expressed as
//   pad(innerTensor, paddingValue, lowPadding, highPadding,
//   interiorPadding=[0,...])
struct PaddedTensor {
  mlir::DenseElementsAttr innerTensorAttr; // The smaller constant tensor
  mlir::Attribute paddingValue;            // The padding value (scalar)
  llvm::SmallVector<int64_t> lowPadding;   // Padding at the start of each dim
  llvm::SmallVector<int64_t> highPadding;  // Padding at the end of each dim
  mlir::RankedTensorType resultType;       // The resulting padded tensor type
};

std::optional<PaddedTensor> detectPaddedTensor(mlir::DenseElementsAttr attr);

// Helper to check if a TypedAttr is zero
inline bool isZeroAttr(mlir::TypedAttr attr) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getValue().isZero();
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr))
    return floatAttr.getValue().isZero();
  return false;
}

// Helper to check if a TypedAttr is one
inline bool isOneAttr(mlir::TypedAttr attr) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return intAttr.getValue() == 1;
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    llvm::APFloat one(floatAttr.getValue().getSemantics(), 1);
    return floatAttr.getValue().bitwiseIsEqual(one);
  }
  return false;
}

// Helper to get a double value from a TypedAttr
inline std::optional<double> getDoubleFromAttr(mlir::TypedAttr attr) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
    return static_cast<double>(intAttr.getValue().getSExtValue());
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr))
    return floatAttr.getValueAsDouble();
  return std::nullopt;
}

// Helper to create a TypedAttr from a double value using the given type
inline mlir::TypedAttr createAttrFromDouble(mlir::MLIRContext *ctx,
                                            mlir::Type elemType, double value) {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(elemType))
    return mlir::IntegerAttr::get(intType, static_cast<int64_t>(value));
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(elemType))
    return mlir::FloatAttr::get(floatType, value);
  return nullptr;
}

// TODO: we can do a full analysis and return if the access is on a specific set
// of diagonals. Checks that all accesses for this Op and its users thereoff are
// along the diagonal.
bool allAccessesAreOnMainDiagonal(
    mlir::Operation *op, llvm::SetVector<mlir::Operation *> &opsToReplace);
bool allAccessesAreOnMainDiagonal(
    stablehlo::ReshapeOp op, llvm::SetVector<mlir::Operation *> &opsToReplace);
bool allAccessesAreOnMainDiagonal(
    stablehlo::GatherOp op, llvm::SetVector<mlir::Operation *> &opsToReplace);

} // namespace enzyme

namespace stablehlo {

stablehlo::GatherDimensionNumbersAttr
getGatherDims(mlir::MLIRContext *ctx,
              stablehlo::ScatterDimensionNumbersAttr scatterDimNumbers);

bool isSetindexBlock(mlir::Block *block,
                     std::function<bool(stablehlo::ReturnOp retOp)> fn);

bool isSetindexBlock(mlir::Block *block);
bool isSetindexBlock(mlir::Block *block, mlir::Value &val);
bool isConstantSetindexBlock(mlir::Block *block,
                             mlir::SplatElementsAttr &constant);

// rhs is only considered if commutative is false
template <typename T, bool commutative, bool rhs>
bool isOnlyOpBlock(mlir::Block *block) {
  if (block->getNumArguments() != 2)
    return false;

  if (!hasSingleElement(block->without_terminator()))
    return false;

  auto op = dyn_cast<T>(block->front());
  if (!op)
    return false;

  if (op.getNumOperands() != 2)
    return false;

  if constexpr (commutative) {
    if (!(op->getOperand(0) == block->getArgument(0) &&
          op->getOperand(1) == block->getArgument(1)) &&
        !(op->getOperand(0) == block->getArgument(1) &&
          op->getOperand(1) == block->getArgument(0)))
      return false;
  } else {
    if constexpr (rhs) {
      if (!(op->getOperand(0) == block->getArgument(0) &&
            op->getOperand(1) == block->getArgument(1)))
        return false;
    } else {
      if (!(op->getOperand(0) == block->getArgument(1) &&
            op->getOperand(1) == block->getArgument(0)))
        return false;
    }
  }

  auto returnOp = block->getTerminator();
  auto stablehloReturnOp = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturnOp)
    return false;

  if (stablehloReturnOp.getNumOperands() != 1)
    return false;

  // The returned value should be the result of the addition
  return stablehloReturnOp.getOperand(0) == op.getResult();
}

template <typename T, int nonConstantIndex>
bool isOnlyOpConstantBlock(mlir::Block *block,
                           mlir::SplatElementsAttr &constant) {
  if (block->getNumArguments() != 2)
    return false;

  if (!hasSingleElement(block->without_terminator()))
    return false;

  auto op = dyn_cast<T>(block->front());
  if (!op)
    return false;

  if (op.getNumOperands() != 2)
    return false;

  // the non constant operand needs to be the first argument
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  if (matchPattern(lhs, m_Constant(&constant))) {
    if (rhs != block->getArgument(nonConstantIndex))
      return false;
  } else if (matchPattern(rhs, m_Constant(&constant))) {
    if (lhs != block->getArgument(nonConstantIndex))
      return false;
  } else {
    return false;
  }

  auto returnOp = block->getTerminator();
  auto stablehloReturnOp = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturnOp)
    return false;

  if (stablehloReturnOp.getNumOperands() != 1)
    return false;

  // The returned value should be the result of the addition
  return stablehloReturnOp.getOperand(0) == op.getResult();
}

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

// this doesn't construct the scalar value and instead returns the
// other operand
bool extractMultiplicationFactor(Value v, Value &other, Operation *op,
                                 OpBuilder &builder);
void extractMultiplicationFactor(Value v, Value &scalar, Value &other,
                                 Operation *op, OpBuilder &builder);

Value getScalarValue(Value val, OpBuilder &builder);
Value getScalarValue(Operation *op, OpBuilder &builder);

bool isScalarValue(Value val);
bool isScalarValue(Operation *op);

Value copyTriangularPart(OpBuilder &builder, Value input,
                         enzymexla::LapackUplo uplo);

bool OpIsReshapeLike(stablehlo::BroadcastInDimOp op);
bool OpIsReshapeLike(stablehlo::TransposeOp op);
bool OpIsReshapeLike(stablehlo::TransposeOp op, ArrayRef<int64_t> shape);

bool canMergeSlicesAlongAxis(int dimension, ArrayRef<int64_t> sliceStarts,
                             ArrayRef<int64_t> otherSliceStarts,
                             ArrayRef<int64_t> sliceLimits,
                             ArrayRef<int64_t> otherSliceLimits,
                             ArrayRef<int64_t> sliceStrides,
                             ArrayRef<int64_t> otherSliceStrides);

bool canMergeSlicesAlongAxis(int dimension, stablehlo::SliceOp slice,
                             stablehlo::SliceOp otherSlice);

stablehlo::ConcatenateOp lowerWrap(enzymexla::WrapOp wrap,
                                   PatternRewriter &rewriter, bool replace);

LogicalResult concatSliceSimplify(PatternRewriter &rewriter,
                                  SmallVectorImpl<Value> &operands, int64_t dim,
                                  SmallVectorImpl<Value> &newOperands);
LogicalResult concatReshapeSliceSimplify(PatternRewriter &rewriter,
                                         SmallVectorImpl<Value> &operands,
                                         int64_t dim,
                                         SmallVectorImpl<Value> &newOperands);

Value getIdentityValue(OpBuilder &builder, Location loc, Type elemType,
                       Operation *op);

bool canFuseIntoReduce(Operation *op);

llvm::SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> perm);

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, stablehlo::SliceOp op);
Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicSliceOp op);

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, ArrayRef<int64_t> starts,
                           ArrayRef<int64_t> limits, ArrayRef<int64_t> strides);

Value transposeSliceHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           ArrayRef<Value> sliceStarts,
                           ArrayRef<int64_t> sliceSizes);

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, stablehlo::SliceOp op);
Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicSliceOp op);
Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           stablehlo::DynamicUpdateSliceOp op);

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter, ArrayRef<int64_t> starts,
                           ArrayRef<int64_t> limits, ArrayRef<int64_t> strides);

Value sliceTransposeHelper(stablehlo::TransposeOp transpose,
                           PatternRewriter &rewriter,
                           ArrayRef<Value> sliceStarts,
                           ArrayRef<int64_t> sliceSizes);

// checks if operation 1 can be fused with operation 2. ordering is important
bool isFusible(stablehlo::TransposeOp transpose, Operation *op);
bool isFusible(Operation *op, stablehlo::BroadcastInDimOp bcast);
bool isFusible(Operation *op, stablehlo::ReshapeOp reshape);

template <typename OpTy>
Value getIdentityValueForOp(OpBuilder &builder, Location loc, Type elemType);

Type GetDotGeneralResultType(Value lhs, Value rhs, Type resElemType,
                             stablehlo::DotDimensionNumbersAttr dotDims);

// these add additional checks that prevent no-op creation
Value ConcatenateOpCreate(
    OpBuilder &builder, Location loc, ArrayRef<Value> inputs, int64_t dimension,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt);

Value ReshapeOpCreate(
    OpBuilder &builder, Location loc, Value input, ArrayRef<int64_t> shape,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt);

Value TransposeOpCreate(
    OpBuilder &builder, Location loc, Value input,
    ArrayRef<int64_t> permutation,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt);

Value SliceOpCreate(
    OpBuilder &builder, Location loc, Value input,
    ArrayRef<int64_t> sliceStarts, ArrayRef<int64_t> sliceLimits,
    ArrayRef<int64_t> sliceStrides,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt);

Value DynamicSliceOpCreate(
    OpBuilder &builder, Location loc, Value input, ArrayRef<Value> sliceStarts,
    ArrayRef<int64_t> sliceSizes,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt);

static Value MaybeBroadcastScalarToMatchShape(
    OpBuilder &builder, Location loc, Value src, RankedTensorType targetTy,
    std::optional<sdy::TensorShardingPerValueAttr> sharding) {
  auto srcTy = cast<RankedTensorType>(src.getType());
  if (srcTy == targetTy || targetTy.getRank() == 0) {
    return src;
  }
  assert(srcTy.getRank() == 0);
  auto bcastOp = stablehlo::BroadcastInDimOp::create(
      builder, loc, targetTy, src, builder.getDenseI64ArrayAttr({}));
  if (sharding.has_value()) {
    sdy::setShardings(bcastOp, *sharding);
  }
  return bcastOp.getResult();
}

// allows lhs or rhs to be a scalar in which case it will automatically be
// broadcasted to the correct shape
template <typename OpTy>
Value BinaryOpCreate(
    OpBuilder &builder, Location loc, Value lhs, Value rhs,
    std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt) {
  auto lhsTy = cast<RankedTensorType>(lhs.getType());
  auto rhsTy = cast<RankedTensorType>(rhs.getType());

  lhs = MaybeBroadcastScalarToMatchShape(builder, loc, lhs, rhsTy, sharding);
  rhs = MaybeBroadcastScalarToMatchShape(builder, loc, rhs, lhsTy, sharding);
  auto newOp = OpTy::create(builder, loc, lhs, rhs);
  if (sharding.has_value()) {
    sdy::setShardings(newOp, *sharding);
  }
  return newOp.getResult();
}

#define DEFINE_BINARY_OP_CREATE(OpTy)                                          \
  inline Value OpTy##Create(                                                   \
      OpBuilder &builder, Location loc, Value lhs, Value rhs,                  \
      std::optional<sdy::TensorShardingPerValueAttr> sharding =                \
          std::nullopt) {                                                      \
    return BinaryOpCreate<stablehlo::OpTy>(builder, loc, lhs, rhs, sharding);  \
  }

DEFINE_BINARY_OP_CREATE(AddOp)
DEFINE_BINARY_OP_CREATE(MulOp)
DEFINE_BINARY_OP_CREATE(SubtractOp)
DEFINE_BINARY_OP_CREATE(DivOp)
DEFINE_BINARY_OP_CREATE(MinOp)
DEFINE_BINARY_OP_CREATE(MaxOp)
DEFINE_BINARY_OP_CREATE(AndOp)
DEFINE_BINARY_OP_CREATE(OrOp)
DEFINE_BINARY_OP_CREATE(XorOp)

// walk back starting from `input` and track the operations to determine if
// only part of the matrix is populated.
bool IsTensorFilled(Value input);

// Enum representing the type of reduce operation
enum class ReduceOpKind { Unknown, Add, Min, Max, Mul, And, Or, Xor };

template <typename OpTy> struct CheckCommonReduceLikeOp {
public:
  ReduceOpKind kind;

  CheckCommonReduceLikeOp(OpTy op) {
    auto &region = op.getBody();
    if (region.getBlocks().size() != 1) {
      kind = ReduceOpKind::Unknown;
      return;
    }

    auto &block = region.getBlocks().front();
    if (isOnlyOpBlock<stablehlo::AddOp, true, false>(&block)) {
      kind = ReduceOpKind::Add;
    } else if (isOnlyOpBlock<stablehlo::MinOp, true, false>(&block)) {
      kind = ReduceOpKind::Min;
    } else if (isOnlyOpBlock<stablehlo::MaxOp, true, false>(&block)) {
      kind = ReduceOpKind::Max;
    } else if (isOnlyOpBlock<stablehlo::MulOp, true, false>(&block)) {
      kind = ReduceOpKind::Mul;
    } else if (isOnlyOpBlock<stablehlo::AndOp, true, false>(&block)) {
      kind = ReduceOpKind::And;
    } else if (isOnlyOpBlock<stablehlo::OrOp, true, false>(&block)) {
      kind = ReduceOpKind::Or;
    } else if (isOnlyOpBlock<stablehlo::XorOp, true, false>(&block)) {
      kind = ReduceOpKind::Xor;
    } else {
      kind = ReduceOpKind::Unknown;
    }
  }

  bool isCommutativeOp() const { return kind != ReduceOpKind::Unknown; }

  Value createEquivalentOperation(
      OpBuilder &builder, Location loc, Value lhs, Value rhs,
      std::optional<sdy::TensorShardingPerValueAttr> sharding = std::nullopt) {
    switch (kind) {
    case ReduceOpKind::Add:
      return AddOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::Min:
      return MinOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::Max:
      return MaxOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::Mul:
      return MulOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::And:
      return AndOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::Or:
      return OrOpCreate(builder, loc, lhs, rhs, sharding);
    case ReduceOpKind::Xor:
      return XorOpCreate(builder, loc, lhs, rhs, sharding);
    default:
      llvm_unreachable("Invalid reduce op");
    }
  }
};

// Type alias for backward compatibility
using CheckCommonReduceOp = CheckCommonReduceLikeOp<stablehlo::ReduceOp>;
using CheckCommonReduceWindowOp =
    CheckCommonReduceLikeOp<stablehlo::ReduceWindowOp>;

// Enum representing the type of scatter operation
enum class ScatterOpKind {
  Unknown,
  Setindex,
  ConstantSetindex,
  SetindexOutsideValue,
  Add,
  Min,
  Max,
  Mul,
  And,
  Or,
  Xor,
  Sub,
  MulConstantUpdate,
  MulConstantInput,
  AddConstantUpdate,
  AddConstantInput
};

struct CheckCommonScatterOp {
public:
  ScatterOpKind kind;
  SplatElementsAttr constant;
  Value outsideValue;

  CheckCommonScatterOp(stablehlo::ScatterOp op) {
    auto &updateComputation = op.getUpdateComputation();

    if (!updateComputation.hasOneBlock()) {
      kind = ScatterOpKind::Unknown;
      return;
    }

    auto &block = updateComputation.front();

    // Check for constant operations first (since they are more specific)
    if (isOnlyOpConstantBlock<stablehlo::MulOp, 0>(&block, constant)) {
      kind = ScatterOpKind::MulConstantUpdate;
    } else if (isOnlyOpConstantBlock<stablehlo::MulOp, 1>(&block, constant)) {
      kind = ScatterOpKind::MulConstantInput;
    } else if (isOnlyOpConstantBlock<stablehlo::AddOp, 0>(&block, constant)) {
      kind = ScatterOpKind::AddConstantUpdate;
    } else if (isOnlyOpConstantBlock<stablehlo::AddOp, 1>(&block, constant)) {
      kind = ScatterOpKind::AddConstantInput;
    } else if (isConstantSetindexBlock(&block, constant)) {
      kind = ScatterOpKind::ConstantSetindex;
    } else if (isSetindexBlock(&block)) {
      kind = ScatterOpKind::Setindex;
    } else if (isSetindexBlock(&block, outsideValue)) {
      kind = ScatterOpKind::SetindexOutsideValue;
    } else if (isOnlyOpBlock<stablehlo::AddOp, true, false>(&block)) {
      kind = ScatterOpKind::Add;
    } else if (isOnlyOpBlock<stablehlo::MulOp, true, false>(&block)) {
      kind = ScatterOpKind::Mul;
    } else if (isOnlyOpBlock<stablehlo::MinOp, true, false>(&block)) {
      kind = ScatterOpKind::Min;
    } else if (isOnlyOpBlock<stablehlo::MaxOp, true, false>(&block)) {
      kind = ScatterOpKind::Max;
    } else if (isOnlyOpBlock<stablehlo::AndOp, true, false>(&block)) {
      kind = ScatterOpKind::And;
    } else if (isOnlyOpBlock<stablehlo::OrOp, true, false>(&block)) {
      kind = ScatterOpKind::Or;
    } else if (isOnlyOpBlock<stablehlo::XorOp, true, false>(&block)) {
      kind = ScatterOpKind::Xor;
    } else if (isOnlyOpBlock<stablehlo::SubtractOp, false, true>(&block)) {
      kind = ScatterOpKind::Sub;
    } else {
      kind = ScatterOpKind::Unknown;
    }
  }
};

void ExtractBlockIntoFunction(Block *block, ModuleOp modOp, func::FuncOp &func,
                              llvm::SetVector<Value> &capturedValues,
                              llvm::SmallVectorImpl<Type> &resultTypes,
                              OpBuilder &builder);

} // namespace stablehlo

} // namespace mlir
