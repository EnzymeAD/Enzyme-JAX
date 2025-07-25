//===- EnzymeWrapPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to create wrapper functions which differentiate
// ops.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Types.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include "llvm/ADT/MapVector.h"
#include <iterator>
#include <numeric>
#define DEBUG_TYPE "enzymehloopt"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEHLOOPTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

// Check if any of the pad sizes are negative
bool anyPadSizesNegative(stablehlo::PadOp pad) {
  for (auto &&[low, high, inner] :
       llvm::zip(pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
                 pad.getInteriorPadding())) {
    if (low < 0 || high < 0 || inner < 0)
      return true;
  }
  return false;
}

namespace {

llvm::SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> inversePerm(perm.size(), -1);
  for (int64_t i = 0; i < perm.size(); ++i) {
    inversePerm[perm[i]] = i;
  }
  return inversePerm;
}

class ReshapeDimMapping {
public:
  void addMapping(int64_t left, int64_t right) {
    mapping.push_back(std::make_pair(left, right));
  }

  SmallVector<int64_t> getMappingFromResultDim(int64_t dim) const {
    SmallVector<int64_t> result;
    for (auto &[left, right] : mapping) {
      if (left == dim)
        result.push_back(right);
    }
    return result;
  }

  SmallVector<int64_t> getMappingFromOperandDim(int64_t dim) const {
    SmallVector<int64_t> result;
    for (auto &[left, right] : mapping) {
      if (right == dim)
        result.push_back(left);
    }
    return result;
  }

  bool isOnlySplitting() const {
    llvm::SmallDenseSet<int64_t> keys;
    for (auto &[left, right] : mapping) {
      if (!std::get<1>(keys.insert(left)))
        return false;
    }
    return true;
  }

  void dump() const {
    for (auto &[left, right] : mapping) {
      llvm::outs() << left << " -> " << right << "\n";
    }
  }

private:
  // Left is result dim, right is operand dim.
  SmallVector<std::pair<int64_t, int64_t>> mapping;
};

// Analyze if a reshape is clearly merging or splitting dimensions.
std::optional<ReshapeDimMapping>
tryFindReshapeDimMapping(stablehlo::ReshapeOp op) {
  ReshapeDimMapping mapping;
  int64_t lhsPos = 0;
  int64_t rhsPos = 0;
  auto lhsShape = cast<TensorType>(op.getOperand().getType()).getShape();
  auto rhsShape = cast<TensorType>(op.getResult().getType()).getShape();
  while (lhsPos < lhsShape.size() && rhsPos < rhsShape.size()) {
    if (lhsShape[lhsPos] == rhsShape[rhsPos]) {
      // Nice 1-to-1 mapping.
      mapping.addMapping(lhsPos, rhsPos);
    } else if (lhsShape[lhsPos] < rhsShape[rhsPos]) {
      // Potential many-to-one mapping.
      int64_t product = lhsShape[lhsPos];
      mapping.addMapping(lhsPos, rhsPos);
      while (product < rhsShape[rhsPos]) {
        if (++lhsPos >= lhsShape.size())
          break;
        product *= lhsShape[lhsPos];
        mapping.addMapping(lhsPos, rhsPos);
      }
      if (product != rhsShape[rhsPos])
        return std::nullopt;
    } else {
      // Potential one-to-many mapping.
      assert(lhsShape[lhsPos] > rhsShape[rhsPos]);
      int64_t product = rhsShape[rhsPos];
      mapping.addMapping(lhsPos, rhsPos);
      while (product < lhsShape[lhsPos]) {
        if (++rhsPos >= rhsShape.size())
          break;
        product *= rhsShape[rhsPos];
        mapping.addMapping(lhsPos, rhsPos);
      }
      if (product != lhsShape[lhsPos])
        return std::nullopt;
    }
    ++lhsPos;
    ++rhsPos;
  };
  return mapping;
}

class StaticSlice {
private:
  using VecTy = SmallVector<int64_t>;
  using TensorValue = TypedValue<RankedTensorType>;
  VecTy starts;
  VecTy limits;
  VecTy inputShape;
  VecTy outputShape;
  VecTy strides;
  unsigned rank;
  TensorValue input, output;
  RankedTensorType inputTy, outputTy;

public:
  TensorValue getOutput() { return output; }
  TensorValue getInput() { return input; }
  int64_t getBeginOffset(unsigned dim) { return starts[dim]; }
  int64_t getEndOffset(unsigned dim) { return inputShape[dim] - limits[dim]; }

  int64_t getOutputShape(unsigned dim) const {
    assert(dim < rank);
    return outputShape[dim];
  }

  bool isFullInDim(unsigned dim) const {
    assert(dim < rank);
    return starts[dim] == 0 && limits[dim] == inputShape[dim];
  }

  bool isSliceInDim(unsigned dim) const {
    assert(dim < rank);
    return starts[dim] != 0 || limits[dim] != inputShape[dim];
  }

  bool isFromStartInDim(unsigned dim) const {
    assert(dim < rank);
    return starts[dim] == 0;
  }

  bool isToEndInDim(unsigned dim) const {
    assert(dim < rank);
    return limits[dim] == inputShape[dim];
  }

  std::optional<unsigned> isOneDimSlice() const {
    std::optional<unsigned> found = std::nullopt;
    for (unsigned i = 0; i < rank; i++) {
      if (isSliceInDim(i)) {
        if (!found)
          found = i;
        else
          return std::nullopt;
      }
    }
    return found;
  }

  bool isFullSlice() const {
    for (unsigned i = 0; i < rank; i++)
      if (isSliceInDim(i))
        return false;
    return true;
  }

  bool isStrideOneAtDim(unsigned dim) const { return strides[dim] == 1; }

  bool isStrideOne() const {
    return llvm::all_of(strides, [](int64_t stride) { return stride == 1; });
  }

  static bool isEquivalentInDim(const StaticSlice &a, const StaticSlice &b,
                                unsigned dim) {
    if (a.rank != b.rank)
      return false;
    if (a.input != b.input)
      return false;

    return a.starts[dim] == b.starts[dim] && a.limits[dim] == b.limits[dim];
  }

  static bool isPrefixInDim(const StaticSlice &a, const StaticSlice &b,
                            unsigned dim) {
    if (!isEquivalentExceptDim(a, b, dim))
      return false;
    return a.starts[dim] == b.starts[dim] && a.limits[dim] <= b.limits[dim];
  }

  static bool isSuffixInDim(const StaticSlice &a, const StaticSlice &b,
                            unsigned dim) {
    if (!isEquivalentExceptDim(a, b, dim))
      return false;
    return a.starts[dim] >= b.starts[dim] && a.limits[dim] == b.limits[dim];
  }

  static bool isEquivalentExceptDim(const StaticSlice &a, const StaticSlice &b,
                                    unsigned dim) {
    return llvm::all_of(llvm::seq(a.rank), [&](unsigned i) {
      return dim == i || isEquivalentInDim(a, b, i);
    });
  }

  static std::optional<StaticSlice> get(Value v) {
    if (!v)
      return std::nullopt;

    StaticSlice res;
    RankedTensorType ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty)
      return std::nullopt;

    unsigned rank = ty.getRank();
    res.rank = rank;
    res.output = cast<TypedValue<RankedTensorType>>(v);
    res.outputTy = ty;

    if (stablehlo::SliceOp slice = v.getDefiningOp<stablehlo::SliceOp>()) {
      res.inputTy = slice.getOperand().getType();
      res.starts = VecTy(slice.getStartIndices());
      res.limits = VecTy(slice.getLimitIndices());
      res.strides = VecTy(slice.getStrides());
      res.input = slice.getOperand();
    } else {
      res.inputTy = ty;
      res.starts = VecTy(rank, 0);
      res.limits = VecTy(ty.getShape());
      res.strides = VecTy(rank, 1);
      res.input = res.output;
    }
    res.inputShape = VecTy(res.inputTy.getShape());
    res.outputShape = VecTy(res.outputTy.getShape());

    return res;
  }
};

LogicalResult failIfDynamicShape(Operation *op, PatternRewriter &rewriter) {
  for (auto type : op->getResultTypes()) {
    auto rType = dyn_cast<RankedTensorType>(type);
    if (!rType || !rType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "unsupported dynamic shape for output.");
  }

  for (auto type : op->getOperandTypes()) {
    auto rType = dyn_cast<RankedTensorType>(type);
    if (!rType || !rType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "unsupported dynamic shape for input.");
  }

  return success();
}

LogicalResult failIfFuncOpInterfaceHasAttr(Operation *op, StringRef attrName,
                                           PatternRewriter &rewriter) {
  if (auto func = op->getParentOfType<FunctionOpInterface>()) {
    if (func->hasAttrOfType<UnitAttr>(attrName))
      return rewriter.notifyMatchFailure(op, "disabled by attribute.");
  }

  return success();
}

static constexpr StringRef kDisablePatternAttrName =
    "enzymexla.disable_hlo_opts";

template <typename OpTy, typename Child>
struct CheckedOpRewritePattern : public OpRewritePattern<OpTy> {
  using Base = OpRewritePattern<OpTy>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override final {
    LogicalResult res =
        failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter);
    if (res.failed())
      return res;

    if (!((Child *)this)->supportsDynamicShapes()) {
      LogicalResult res = failIfDynamicShape(op, rewriter);
      if (res.failed())
        return res;
    }

    return ((Child *)this)->matchAndRewriteImpl(op, rewriter);
  }

  bool supportsDynamicShapes() { return false; }
};

template <template <typename> class TraitType, typename Child>
struct CheckedOpTraitRewritePattern : public OpTraitRewritePattern<TraitType> {
  using Base = OpTraitRewritePattern<TraitType>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op,
                  PatternRewriter &rewriter) const override final {
    LogicalResult res =
        failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter);
    if (res.failed())
      return res;

    if (!((Child *)this)->supportsDynamicShapes()) {
      auto res = failIfDynamicShape(op, rewriter);
      if (res.failed())
        return res;
    }

    return ((Child *)this)->matchAndRewriteImpl(op, rewriter);
  }

  bool supportsDynamicShapes() { return false; }
};

struct NoopSlice final
    : CheckedOpRewritePattern<stablehlo::SliceOp, NoopSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();
    for (auto sz : type.getShape())
      if (sz < 0)
        return failure();
    if (op.getOperand().getType() == type) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    return failure();
  }
};

void sliceSliceHelper(stablehlo::SliceOp prev, SmallVector<int64_t> &starts,
                      SmallVector<int64_t> &limits,
                      SmallVector<int64_t> &strides) {
  assert(starts.size() == prev.getType().getShape().size());
  assert(limits.size() == prev.getType().getShape().size());
  assert(strides.size() == prev.getType().getShape().size());

  for (auto &&[pstart, pend, pstep, nstart, nend, nstep, size] : llvm::zip(
           prev.getStartIndices(), prev.getLimitIndices(), prev.getStrides(),
           starts, limits, strides, prev.getOperand().getType().getShape())) {

    auto start2 = pstart + pstep * nstart;
    auto step2 = pstep * nstep;
    auto end2 = pstart + pstep * nstart + pstep * (nend - nstart);
    if (start2 > size)
      start2 = size;
    if (end2 > size)
      end2 = size;
    nstart = start2;
    nstep = step2;
    nend = end2;
  }
}

struct SliceSlice final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto prev = op.getOperand().getDefiningOp<stablehlo::SliceOp>();
    if (!prev)
      return failure();

    SmallVector<int64_t> start(op.getStartIndices().begin(),
                               op.getStartIndices().end());
    SmallVector<int64_t> end(op.getLimitIndices().begin(),
                             op.getLimitIndices().end());
    SmallVector<int64_t> step(op.getStrides().begin(), op.getStrides().end());

    sliceSliceHelper(prev, start, end, step);
    auto resTy = op.getType();
    auto res = rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
        op, prev.getOperand(), start, end, step);
    assert(res.getType() == resTy);
    (void)res;
    (void)resTy;
    return success();
  }
};

struct DynamicSliceToStatic final
    : CheckedOpRewritePattern<stablehlo::DynamicSliceOp, DynamicSliceToStatic> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicSliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    SmallVector<int64_t> starts;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> steps;
    for (auto &&[start, size, shape] :
         llvm::zip(op.getStartIndices(), op.getSliceSizes(),
                   op.getOperand().getType().getShape())) {

      DenseIntElementsAttr startattr;
      if (!matchPattern(start, m_Constant(&startattr))) {
        return failure();
      }
      int64_t startv = (*startattr.begin()).getSExtValue();
      if (startv < 0)
        return failure();
      if (startv + size > shape)
        return failure();
      starts.push_back(startv);
      ends.push_back(startv + size);
      steps.push_back(1);
    }
    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
        op, op.getType(), op.getOperand(), starts, ends, steps);
    return success();
  }
};

struct DynamicUpdateSliceElim final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp,
                              DynamicUpdateSliceElim> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    if (op.getUpdate().getType() != type)
      return failure();

    for (auto start : op.getStartIndices()) {
      DenseIntElementsAttr startattr;
      if (!matchPattern(start, m_Constant(&startattr))) {
        return failure();
      }
      int64_t startv = (*startattr.begin()).getSExtValue();
      if (startv != 0)
        return failure();
    }
    rewriter.replaceOp(op, op.getUpdate());
    return success();
  }
};

// Given a reshape fromType to toType, remove the eliminated indices from start,
// and fill any new dimensions with toFill. Check that any removed indices have
// value checkRemoved, if set
template <typename T>
bool transformReshapeSlice(RankedTensorType fromType, RankedTensorType toType,
                           SmallVectorImpl<T> &start,
                           std::function<T()> toFillFn,
                           T *checkRemoved = nullptr) {
  auto fromShape = fromType.getShape();
  auto toShape = toType.getShape();

  assert(start.size() == fromShape.size());

  int i = 0, j = 0;
  int startidx = 0;
  std::optional<T> toFillVal;
  while (i < fromShape.size() && j < toShape.size()) {
    if (fromShape[i] == toShape[j]) {
      // Leave everything as is and carry on
      i++;
      j++;
      startidx++;
      continue;
    }

    if (fromShape[i] == 1) {
      if (checkRemoved) {
        if (start[startidx] != *checkRemoved) {
          if (i > 0 && j > 0 && fromShape[i - 1] == fromShape[j - 1] &&
              fromShape[i - 1] == 1 && start[startidx - 1] == *checkRemoved) {
            i--;
            j--;
            startidx--;
          } else {
            return false;
          }
        }
      }
      start.erase(start.begin() + startidx);
      i++;
      continue;
    }

    if (toShape[j] == 1) {
      T toFill = toFillVal ? *toFillVal : toFillFn();
      if (!toFillVal) {
        toFillVal = toFill;
      }
      start.insert(start.begin() + startidx, toFill);
      startidx++;
      j++;
      continue;
    }
    return false;
  }

  while (i < fromShape.size()) {
    if (fromShape[i] == 1) {
      if (checkRemoved) {
        if (start[startidx] != *checkRemoved) {
          return false;
        }
      }
      start.erase(start.begin() + startidx);
      i++;
      continue;
    }
    return false;
  }

  while (j < toShape.size()) {
    if (toShape[j] == 1) {
      T toFill = toFillVal ? *toFillVal : toFillFn();
      if (!toFillVal) {
        toFillVal = toFill;
      }
      start.insert(start.begin() + startidx, toFill);
      startidx++;
      j++;
      continue;
    }
  }
  assert(start.size() == toShape.size());
  return true;
}

template <typename T>
bool transformReshapeSlice(RankedTensorType fromType, RankedTensorType toType,
                           SmallVectorImpl<T> &start, T toFill,
                           T *checkRemoved = nullptr) {
  return transformReshapeSlice<T>(
      fromType, toType, start, [=]() { return toFill; }, checkRemoved);
}

template <typename T>
bool transformReshapeSlice(stablehlo::ReshapeOp op, SmallVectorImpl<T> &start,
                           T toFill, T *checkRemoved = nullptr) {
  return transformReshapeSlice<T>(op.getOperand().getType(), op.getType(),
                                  start, toFill, checkRemoved);
}

template <typename T>
bool transformReshapeSlice(stablehlo::ReshapeOp op, SmallVectorImpl<T> &start,
                           std::function<T()> toFill,
                           T *checkRemoved = nullptr) {
  return transformReshapeSlice<T>(op.getOperand().getType(), op.getType(),
                                  start, toFill, checkRemoved);
}

stablehlo::Element conj(const stablehlo::Element &orig) {
  if (stablehlo::isSupportedComplexType(orig.getType())) {
    std::complex<APFloat> val = orig.getComplexValue();
    return stablehlo::Element(orig.getType(),
                              std::complex<APFloat>(val.real(), -val.imag()));
  }

  llvm_unreachable("Unsupported type");
}

stablehlo::Tensor conjOp(const stablehlo::Tensor &orig, ShapedType resultType) {
  stablehlo::Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, conj(orig.get(*it)));
  return result;
}

struct ReshapeDUS final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is a DynamicUpdateSlice
    auto dus = op.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dus)
      return failure();

    if (!llvm::hasSingleElement(dus->getUsers()))
      return failure();

    // %0 = dynamic_update_slice %arg0, %update
    // %1 = reshape %0
    //
    // ->
    //
    // %arg0_reshaped = reshape %arg0
    // %update_resahepd = reshape %update
    // dynamic_update_slice %arg0_reshaped, %update_reshaped

    SmallVector<Value> startIndices(dus.getStartIndices().begin(),
                                    dus.getStartIndices().end());
    auto itype = startIndices.size() > 0
                     ? cast<RankedTensorType>(startIndices[0].getType())
                     : RankedTensorType::get({}, rewriter.getI64Type());

    if (!transformReshapeSlice<mlir::Value>(
            op, startIndices, /*toFill*/ [&]() -> mlir::Value {
              return rewriter.create<stablehlo::ConstantOp>(
                  dus.getLoc(), itype, cast<ElementsAttr>(makeAttr(itype, 0)));
            }))
      return failure();

    SmallVector<int64_t> updateShape(
        dus.getUpdate().getType().getShape().begin(),
        dus.getUpdate().getType().getShape().end());

    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, updateShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    auto newOperand = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(), op.getType(), dus.getOperand());

    auto newUpdate = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(updateShape,
                              dus.getUpdate().getType().getElementType()),
        dus.getUpdate());

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        op, newOperand, newUpdate, startIndices);

    return success();
  }
};

struct ReshapeSlice final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeSlice> {
  bool onlySingleUser;

  ReshapeSlice(bool onlySingleUser, MLIRContext *context,
               PatternBenefit benefit = 1,
               ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is a DynamicUpdateSlice
    auto slice = op.getOperand().getDefiningOp<stablehlo::SliceOp>();
    if (!slice)
      return failure();

    if (onlySingleUser && !llvm::hasSingleElement(slice->getUsers()))
      return failure();

    SmallVector<int64_t> startIndices(slice.getStartIndices().begin(),
                                      slice.getStartIndices().end());

    int64_t zero = 0;
    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, startIndices, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    SmallVector<int64_t> limitIndices(slice.getLimitIndices().begin(),
                                      slice.getLimitIndices().end());

    if (!transformReshapeSlice<int64_t>(op, limitIndices, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    SmallVector<int64_t> stepIndices(slice.getStrides().begin(),
                                     slice.getStrides().end());

    if (!transformReshapeSlice<int64_t>(op, stepIndices, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    SmallVector<int64_t> operandShape(
        slice.getOperand().getType().getShape().begin(),
        slice.getOperand().getType().getShape().end());

    if (!transformReshapeSlice<int64_t>(op, operandShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    auto newOperand = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(operandShape,
                              slice.getOperand().getType().getElementType()),
        slice.getOperand());

    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
        op, newOperand, startIndices, limitIndices, stepIndices);

    return success();
  }
};

// reshape(extend) -> extend(reshape)
struct ReshapeExtend final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeExtend> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is an ExtendOp
    auto extendOp = op.getOperand().getDefiningOp<enzymexla::ExtendOp>();
    if (!extendOp)
      return failure();

    if (!llvm::hasSingleElement(extendOp->getUsers()))
      return failure();

    // Get extend operation parameters
    int64_t extendDim = extendOp.getDimension();
    int64_t lhs = extendOp.getLhs();
    int64_t rhs = extendOp.getRhs();

    // Get the shape of the operand
    SmallVector<int64_t> operandShape(
        extendOp.getOperand().getType().getShape().begin(),
        extendOp.getOperand().getType().getShape().end());

    // Calculate the new extended dimension after the reshape
    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, operandShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    int64_t newExtendDim = -1;
    SmallVector<int64_t> oneHotDim(extendOp.getType().getShape().size(), 0);
    oneHotDim[extendDim] = 1;

    int64_t zero = 0;
    if (!transformReshapeSlice<int64_t>(op, oneHotDim, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    for (size_t i = 0; i < oneHotDim.size(); i++) {
      if (oneHotDim[i]) {
        if (newExtendDim != -1) {
          return failure();
        }
        newExtendDim = i;
      }
    }
    if (newExtendDim == -1)
      return failure();

    // First reshape the extend's operand
    auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(operandShape,
                              extendOp.getOperand().getType().getElementType()),
        extendOp.getOperand());

    // Then create a new extend operation on the reshaped data
    auto newExtendOp = rewriter.create<enzymexla::ExtendOp>(
        op.getLoc(), newReshapeOp.getResult(), lhs, rhs, newExtendDim);

    // Replace the original reshape op with the new extend operation
    rewriter.replaceOp(op, newExtendOp);

    return success();
  }
};

// reshape(wrap) -> wrap(reshape)
struct ReshapeWrap final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is a WrapOp
    auto wrapOp = op.getOperand().getDefiningOp<enzymexla::WrapOp>();
    if (!wrapOp)
      return failure();

    if (!llvm::hasSingleElement(wrapOp->getUsers()))
      return failure();

    // Get wrap operation parameters
    int64_t wrapDim = wrapOp.getDimension();
    int64_t lhs = wrapOp.getLhs();
    int64_t rhs = wrapOp.getRhs();

    // Get the shape of the operand
    SmallVector<int64_t> operandShape(
        wrapOp.getOperand().getType().getShape().begin(),
        wrapOp.getOperand().getType().getShape().end());

    // Calculate the new wrapped dimension after the reshape
    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, operandShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    SmallVector<int64_t> oneHotDim(wrapOp.getType().getShape().size(), 0);
    oneHotDim[wrapDim] = 1;

    int64_t zero = 0;
    if (!transformReshapeSlice<int64_t>(op, oneHotDim, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    int64_t newWrapDim = -1;
    for (size_t i = 0; i < oneHotDim.size(); i++) {
      if (oneHotDim[i]) {
        if (newWrapDim != -1) {
          return failure();
        }
        newWrapDim = i;
      }
    }
    if (newWrapDim == -1)
      return failure();

    // First reshape the wrap's operand
    auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(operandShape,
                              wrapOp.getOperand().getType().getElementType()),
        wrapOp.getOperand());

    // Then create a new wrap operation on the reshaped data
    auto newWrapOp = rewriter.create<enzymexla::WrapOp>(
        op.getLoc(), op.getType(), newReshapeOp.getResult(), lhs, rhs,
        newWrapDim);

    // Replace the original reshape op with the new wrap operation
    rewriter.replaceOp(op, newWrapOp);

    return success();
  }
};

// reshape(rotate) -> rotate(reshape)
struct ReshapeRotate final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeRotate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is a RotateOp
    auto rotateOp = op.getOperand().getDefiningOp<enzymexla::RotateOp>();
    if (!rotateOp)
      return failure();

    if (!llvm::hasSingleElement(rotateOp->getUsers()))
      return failure();

    // Get rotate operation parameters
    int64_t rotateDim = rotateOp.getDimension();
    int64_t rotateAmount = rotateOp.getAmount();

    // Get the shape of the operand
    SmallVector<int64_t> operandShape(
        rotateOp.getOperand().getType().getShape().begin(),
        rotateOp.getOperand().getType().getShape().end());

    // Calculate the new rotated dimension after the reshape
    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, operandShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    SmallVector<int64_t> oneHotDim(rotateOp.getType().getShape().size(), 0);
    oneHotDim[rotateDim] = 1;

    int64_t zero = 0;
    if (!transformReshapeSlice<int64_t>(op, oneHotDim, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    int64_t newRotateDim = -1;
    for (size_t i = 0; i < oneHotDim.size(); i++) {
      if (oneHotDim[i]) {
        if (newRotateDim != -1) {
          return failure();
        }
        newRotateDim = i;
      }
    }
    if (newRotateDim == -1)
      return failure();

    // First reshape the rotate's operand
    auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(operandShape,
                              rotateOp.getOperand().getType().getElementType()),
        rotateOp.getOperand());

    // Then create a new rotate operation on the reshaped data
    auto newRotateOp = rewriter.create<enzymexla::RotateOp>(
        op.getLoc(), newReshapeOp.getResult(), rotateAmount, newRotateDim);

    // Replace the original reshape op with the new rotate operation
    rewriter.replaceOp(op, newRotateOp);

    return success();
  }
};

struct ReshapePad final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Reshape is a DynamicUpdateSlice
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    SmallVector<int64_t> interior(pad.getInteriorPadding().begin(),
                                  pad.getInteriorPadding().end());

    int64_t zero = 0;
    if (!transformReshapeSlice<int64_t>(op, interior, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    SmallVector<int64_t> low(pad.getEdgePaddingLow().begin(),
                             pad.getEdgePaddingLow().end());

    if (!transformReshapeSlice<int64_t>(op, low, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    SmallVector<int64_t> high(pad.getEdgePaddingHigh().begin(),
                              pad.getEdgePaddingHigh().end());

    if (!transformReshapeSlice<int64_t>(op, high, /*toFill*/ 0,
                                        /*checkRemoved*/ &zero))
      return failure();

    SmallVector<int64_t> operandShape(
        pad.getOperand().getType().getShape().begin(),
        pad.getOperand().getType().getShape().end());

    int64_t one = 1;
    if (!transformReshapeSlice<int64_t>(op, operandShape, /*toFill*/ 1,
                                        /*checkRemoved*/ &one))
      return failure();

    auto newOperand = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(operandShape,
                              pad.getOperand().getType().getElementType()),
        pad.getOperand());

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, newOperand, pad.getPaddingValue(), low, high, interior);

    return success();
  }
};

struct TransposeDUS final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    // Check if the input to Transpose is a DynamicUpdateSlice
    auto dus = op.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dus)
      return failure();

    if (!llvm::hasSingleElement(dus->getUsers()))
      return failure();

    SmallVector<int64_t> permutation;
    for (auto perm : op.getPermutation()) {
      permutation.push_back(perm);
    }

    auto loc = op.getLoc();
    auto transposedOperand = rewriter.create<stablehlo::TransposeOp>(
        loc, dus.getOperand(), op.getPermutation());
    auto transposedUpdate = rewriter.create<stablehlo::TransposeOp>(
        loc, dus.getUpdate(), op.getPermutation());

    SmallVector<Value> permutedStartIndices;
    permutedStartIndices.resize(dus.getStartIndices().size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      permutedStartIndices[permutation[i]] = dus.getStartIndices()[i];
    }

    auto newDus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc, op.getType(), transposedOperand, transposedUpdate,
        permutedStartIndices);

    rewriter.replaceOp(op, newDus);
    rewriter.eraseOp(dus);
    return success();
  }
};

stablehlo::ConcatenateOp lowerWrap(enzymexla::WrapOp wrap,
                                   PatternRewriter &rewriter, bool replace) {
  // sl0[end-lhs:end], mid, sl1[0:rhs]
  auto wrapOpT = cast<RankedTensorType>(wrap.getOperand().getType());
  SmallVector<int64_t> strides(wrapOpT.getShape().size(), 1);

  SmallVector<Value> args;

  auto shard = sdy::getShardingPerValue(wrap);

  if (wrap.getLhs() != 0) {
    SmallVector<int64_t> sl0_starts(wrapOpT.getShape().size(), 0);
    SmallVector<int64_t> sl0_ends(wrapOpT.getShape());

    sl0_starts[wrap.getDimension()] =
        wrapOpT.getShape()[wrap.getDimension()] - wrap.getLhs();

    auto sl0 = rewriter.create<stablehlo::SliceOp>(
        wrap.getLoc(), wrap.getOperand(), sl0_starts, sl0_ends, strides);
    if (shard)
      sdy::setShardings(sl0, shard);

    args.push_back(sl0);
  }

  args.push_back(wrap.getOperand());

  if (wrap.getRhs() != 0) {
    SmallVector<int64_t> sl1_starts(wrapOpT.getShape().size(), 0);
    SmallVector<int64_t> sl1_ends(wrapOpT.getShape());

    sl1_ends[wrap.getDimension()] = wrap.getRhs();
    auto sl1 = rewriter.create<stablehlo::SliceOp>(
        wrap.getLoc(), wrap.getOperand(), sl1_starts, sl1_ends, strides);
    if (shard)
      sdy::setShardings(sl1, shard);

    args.push_back(sl1);
  }

  auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
      wrap.getLoc(), args, wrap.getDimension());
  if (replace)
    rewriter.replaceOp(wrap, newConcat);
  if (shard)
    sdy::setShardings(newConcat, shard);
  return newConcat;
}

struct LowerWrap
    : public CheckedOpRewritePattern<enzymexla::WrapOp, LowerWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::WrapOp wrap,
                                    PatternRewriter &rewriter) const {
    auto concat = lowerWrap(wrap, rewriter, /*replace*/ true);
    if (concat.getInputs().size() == 1) {
      rewriter.replaceOp(concat, concat.getInputs()[0]);
    }
    return success();
  }
};

stablehlo::ConcatenateOp lowerExtend(enzymexla::ExtendOp extend,
                                     PatternRewriter &rewriter, bool replace) {
  auto loc = extend.getLoc();
  auto operand = extend.getOperand();

  SmallVector<Value> args;

  SmallVector<int64_t> strides(operand.getType().getRank(), 1);
  auto shard = sdy::getShardingPerValue(extend);

  if (extend.getLhs() != 0) {
    SmallVector<int64_t> lhsStarts(operand.getType().getRank(), 0);
    SmallVector<int64_t> lhsLimits(operand.getType().getShape());
    lhsLimits[extend.getDimension()] = extend.getLhs();
    auto lhs = rewriter.create<stablehlo::SliceOp>(loc, operand, lhsStarts,
                                                   lhsLimits, strides);
    if (shard)
      sdy::setShardings(lhs, shard);

    args.push_back(lhs);
  }

  args.push_back(operand);

  if (extend.getRhs() != 0) {
    SmallVector<int64_t> rhsStarts(operand.getType().getRank(), 0);
    SmallVector<int64_t> rhsLimits(operand.getType().getShape());
    rhsStarts[extend.getDimension()] =
        rhsLimits[extend.getDimension()] - extend.getRhs();
    auto rhs = rewriter.create<stablehlo::SliceOp>(loc, operand, rhsStarts,
                                                   rhsLimits, strides);
    if (shard)
      sdy::setShardings(rhs, shard);

    args.push_back(rhs);
  }

  auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
      extend.getLoc(), args, extend.getDimension());
  if (replace)
    rewriter.replaceOp(extend, newConcat);
  if (shard)
    sdy::setShardings(newConcat, shard);
  return newConcat;
}

struct LowerExtend
    : public CheckedOpRewritePattern<enzymexla::ExtendOp, LowerExtend> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::ExtendOp extend,
                                    PatternRewriter &rewriter) const {
    auto concat = lowerExtend(extend, rewriter, /*replace*/ true);
    if (concat.getInputs().size() == 1) {
      rewriter.replaceOp(concat, concat.getInputs()[0]);
    }
    return success();
  }
};

stablehlo::ConcatenateOp lowerRotate(enzymexla::RotateOp rotate,
                                     PatternRewriter &rewriter, bool replace) {
  // sl0[A:end], sl1[0:A]
  auto shard = sdy::getShardingPerValue(rotate);
  SmallVector<int64_t> strides(rotate.getType().getShape().size(), 1);
  SmallVector<int64_t> sl0_starts(rotate.getType().getShape().size(), 0);
  SmallVector<int64_t> sl0_ends(rotate.getType().getShape());
  SmallVector<int64_t> sl1_starts(rotate.getType().getShape().size(), 0);
  SmallVector<int64_t> sl1_ends(rotate.getType().getShape());
  sl0_starts[rotate.getDimension()] = rotate.getAmount();
  sl1_ends[rotate.getDimension()] = rotate.getAmount();
  auto sl0 = rewriter.create<stablehlo::SliceOp>(
      rotate.getLoc(), rotate.getOperand(), sl0_starts, sl0_ends, strides);
  if (shard) {
    sdy::setShardings(sl0, shard);
  }
  auto sl1 = rewriter.create<stablehlo::SliceOp>(
      rotate.getLoc(), rotate.getOperand(), sl1_starts, sl1_ends, strides);
  if (shard) {
    sdy::setShardings(sl1, shard);
  }
  Value args[] = {sl0, sl1};
  auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
      rotate.getLoc(), args, rotate.getDimension());
  if (replace)
    rewriter.replaceOp(rotate, newConcat);
  if (shard) {
    sdy::setShardings(newConcat, shard);
  }
  return newConcat;
}

struct LowerRotate
    : public CheckedOpRewritePattern<enzymexla::RotateOp, LowerRotate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::RotateOp rotate,
                                    PatternRewriter &rewriter) const {
    lowerRotate(rotate, rewriter, /*replace*/ true);
    return success();
  }
};

// Optimization: DUSConcat
// Pattern:
//   %concat = stablehlo.concatenate %A, %B, %C, dimension = D
//   %dus = stablehlo.dynamic_update_slice %concat, %update, %idx0, %idxD, %idxN
// Constraint: %concat has only %dus as its user.
// Constraint: The slice defined by %idx0..N and %update shape falls entirely
//             within one of the inputs (e.g., %B) along dimension D, and
//             spans fully along other dimensions.
// Rewrite to:
//   %idxD_new = %idxD - offset_of_B
//   %new_dus = stablehlo.dynamic_update_slice %B, %update, %idx0_0, %idxD_new,
//   %idxN_0 %new_concat = stablehlo.concatenate %A, %new_dus, %C, dimension = D
//   replaceAllUsesWith(%dus, %new_concat)
struct DUSConcat final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dus,
                                    PatternRewriter &rewriter) const {
    auto updateType = cast<RankedTensorType>(dus.getUpdate().getType());

    ArrayRef<int64_t> updateShape = updateType.getShape();
    SmallVector<Value> startIndices = dus.getStartIndices();

    int64_t currentOffset = 0;
    int targetInputIdx = -1;

    int64_t concatDim;
    SmallVector<int64_t> inputSizes;
    bool legal = false;

    if (auto concatOp =
            dus.getOperand().getDefiningOp<stablehlo::ConcatenateOp>()) {
      concatDim = concatOp.getDimension();
      for (auto input : concatOp.getInputs()) {
        inputSizes.push_back(
            cast<RankedTensorType>(input.getType()).getShape()[concatDim]);
      }
      legal = true;
    }
    if (auto wrap = dus.getOperand().getDefiningOp<enzymexla::WrapOp>()) {
      concatDim = wrap.getDimension();
      inputSizes.push_back(wrap.getLhs());
      inputSizes.push_back(cast<RankedTensorType>(wrap.getOperand().getType())
                               .getShape()[concatDim]);
      inputSizes.push_back(wrap.getRhs());
      legal = true;
    }
    if (auto extend = dus.getOperand().getDefiningOp<enzymexla::ExtendOp>()) {
      concatDim = extend.getDimension();
      inputSizes.push_back(extend.getLhs());
      inputSizes.push_back(cast<RankedTensorType>(extend.getOperand().getType())
                               .getShape()[concatDim]);
      inputSizes.push_back(extend.getRhs());
      legal = true;
    }
    if (auto rotate = dus.getOperand().getDefiningOp<enzymexla::RotateOp>()) {
      concatDim = rotate.getDimension();
      inputSizes.push_back(cast<RankedTensorType>(rotate.getOperand().getType())
                               .getShape()[concatDim] -
                           rotate.getAmount());
      inputSizes.push_back(rotate.getAmount());
      legal = true;
    }

    if (!legal) {
      return failure();
    }

    DenseIntElementsAttr concatStartAttr;
    if (!matchPattern(startIndices[concatDim], m_Constant(&concatStartAttr))) {
      return rewriter.notifyMatchFailure(
          dus, "Requires constant start index for concat dimension");
    }

    int64_t concatStartVal = (*concatStartAttr.begin()).getSExtValue();
    int64_t concatUpdateSize = updateShape[concatDim];
    int64_t concatEndVal = concatStartVal + concatUpdateSize; // Exclusive end

    for (const auto &indexedInput : llvm::enumerate(inputSizes)) {
      int64_t inputSize = indexedInput.value();

      // Check if the DUS update region falls entirely within this input's
      // region
      if (concatStartVal >= currentOffset &&
          concatEndVal <= (currentOffset + inputSize)) {
        targetInputIdx = indexedInput.index();
        break; // Found the target input
      }
      currentOffset += inputSize;
    }

    // If no suitable input was found
    if (targetInputIdx == -1) {
      return rewriter.notifyMatchFailure(
          dus,
          "DUS update region does not fall entirely within one concat input");
    }

    auto concatOp = dus.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();

    if (auto wrap = dus.getOperand().getDefiningOp<enzymexla::WrapOp>()) {
      concatOp = lowerWrap(wrap, rewriter, /*replace*/ false);
    } else if (auto extend =
                   dus.getOperand().getDefiningOp<enzymexla::ExtendOp>()) {
      concatOp = lowerExtend(extend, rewriter, /*replace*/ false);
    } else if (auto rotate =
                   dus.getOperand().getDefiningOp<enzymexla::RotateOp>()) {
      concatOp = lowerRotate(rotate, rewriter, /*replace*/ false);
    }
    assert(concatOp);
    assert(concatOp.getType() == dus.getOperand().getType());

    SmallVector<Value> newDusStartIndices =
        llvm::to_vector(dus.getStartIndices());
    Location loc = dus.getLoc();

    newDusStartIndices[concatDim] = rewriter.create<stablehlo::ConstantOp>(
        dus.getLoc(), newDusStartIndices[concatDim].getType(),
        cast<ElementsAttr>(makeAttr(newDusStartIndices[concatDim].getType(),
                                    concatStartVal - currentOffset)));

    Value targetInputVal = concatOp.getInputs()[targetInputIdx];

    // Create the new, smaller DUS.
    auto newDus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc,
        targetInputVal
            .getType(), // Result type is the same as the input being updated
        targetInputVal, dus.getUpdate(), newDusStartIndices);

    // Create the list of operands for the new concatenate op.
    SmallVector<Value> newConcatOperands =
        llvm::to_vector(concatOp.getInputs());
    newConcatOperands[targetInputIdx] = newDus.getResult();

    // Create the new concatenate op.
    auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
        concatOp
            .getLoc(), // Use concatOp's location, maybe dus.getLoc() is better?
        newConcatOperands, concatDim);
    assert(
        newConcat.getType() ==
        dus.getType()); // The result type should match the original DUS result

    // Replace the original DUS with the new concatenate op.
    rewriter.replaceOp(dus, newConcat.getResult());

    return success();
  }
};

template <typename T>
struct SimplifyBoundary final
    : CheckedOpRewritePattern<T, SimplifyBoundary<T>> {
  using CheckedOpRewritePattern<T,
                                SimplifyBoundary<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {

    SplatElementsAttr elems;
    if (!matchPattern(op.getOperand(), m_Constant(&elems))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op.getType(), elems.resizeSplat(op.getType()));
    return success();
  }
};

struct SliceInternal final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceInternal> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp slice,
                                    PatternRewriter &rewriter) const {
    int64_t currentOffset = 0;
    int targetInputIdx = -1;

    int64_t concatDim;
    SmallVector<int64_t> inputSizes;
    SmallVector<int64_t> actualStartSizes;
    bool legal = false;

    Value operand;

    if (auto wrap = slice.getOperand().getDefiningOp<enzymexla::WrapOp>()) {
      concatDim = wrap.getDimension();
      inputSizes.push_back(wrap.getLhs());
      inputSizes.push_back(cast<RankedTensorType>(wrap.getOperand().getType())
                               .getShape()[concatDim]);
      inputSizes.push_back(wrap.getRhs());
      actualStartSizes.push_back(
          cast<RankedTensorType>(wrap.getOperand().getType())
              .getShape()[concatDim] -
          wrap.getLhs());
      actualStartSizes.push_back(0);
      actualStartSizes.push_back(0);
      legal = true;
      operand = wrap.getOperand();
    }
    if (auto extend = slice.getOperand().getDefiningOp<enzymexla::ExtendOp>()) {
      concatDim = extend.getDimension();
      inputSizes.push_back(extend.getLhs());
      inputSizes.push_back(cast<RankedTensorType>(extend.getOperand().getType())
                               .getShape()[concatDim]);
      inputSizes.push_back(extend.getRhs());
      actualStartSizes.push_back(0);
      actualStartSizes.push_back(0);
      actualStartSizes.push_back(
          cast<RankedTensorType>(extend.getOperand().getType())
              .getShape()[concatDim]);
      legal = true;
      operand = extend.getOperand();
    }
    if (auto rotate = slice.getOperand().getDefiningOp<enzymexla::RotateOp>()) {
      concatDim = rotate.getDimension();
      inputSizes.push_back(cast<RankedTensorType>(rotate.getOperand().getType())
                               .getShape()[concatDim] -
                           rotate.getAmount());
      inputSizes.push_back(rotate.getAmount());
      actualStartSizes.push_back(rotate.getAmount());
      actualStartSizes.push_back(0);
      operand = rotate.getOperand();
      legal = true;
    }

    if (!legal) {
      return failure();
    }

    for (const auto &indexedInput : llvm::enumerate(inputSizes)) {
      int64_t inputSize = indexedInput.value();

      // Check if the DUS update region falls entirely within this input's
      // region
      if (slice.getStartIndices()[concatDim] >= currentOffset &&
          slice.getLimitIndices()[concatDim] <= (currentOffset + inputSize)) {
        targetInputIdx = indexedInput.index();
        break; // Found the target input
      }
      currentOffset += inputSize;
    }

    // If no suitable input was found
    if (targetInputIdx == -1) {
      return rewriter.notifyMatchFailure(
          slice, "Slice region does not fall entirely within one concat input");
    }

    SmallVector<int64_t> newStart = llvm::to_vector(slice.getStartIndices());
    newStart[concatDim] -= currentOffset;
    newStart[concatDim] += actualStartSizes[targetInputIdx];
    SmallVector<int64_t> newLimit = llvm::to_vector(slice.getLimitIndices());
    newLimit[concatDim] -= currentOffset;
    newLimit[concatDim] += actualStartSizes[targetInputIdx];

    rewriter.modifyOpInPlace(slice, [&]() {
      slice.setStartIndices(newStart);
      slice.setLimitIndices(newLimit);
      slice.getOperandMutable().assign(operand);
    });
    return success();
  }
};

struct ConcatConcatToDUS final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatConcatToDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp outer,
                                    PatternRewriter &rewriter) const {
    if (outer.getOperands().size() < 2)
      return failure();
    SmallVector<stablehlo::ConcatenateOp> inners;

    for (auto v : outer.getOperands()) {
      auto concatOp = v.getDefiningOp<stablehlo::ConcatenateOp>();
      if (!concatOp)
        return failure();
      if (concatOp.getDimension() == outer.getDimension())
        return failure();
      if (concatOp.getOperands().size() < 2)
        return failure();
      if (inners.size()) {
        if (inners[0].getOperands().size() != concatOp.getOperands().size()) {
          return failure();
        }
        if (inners[0].getDimension() != concatOp.getDimension()) {
          return failure();
        }
      }
      inners.push_back(concatOp);
    }

    stablehlo::SliceOp lhs = nullptr;
    if (auto lhsSlice =
            inners[0].getOperands()[0].getDefiningOp<stablehlo::SliceOp>()) {
      bool legal = true;
      size_t nextStart = lhsSlice.getLimitIndices()[outer.getDimension()];
      for (int i = 1; i < inners.size(); i++) {
        auto slice2 =
            inners[i].getOperands()[0].getDefiningOp<stablehlo::SliceOp>();
        if (!slice2) {
          legal = false;
          break;
        }
        if (slice2.getOperand() != lhsSlice.getOperand()) {
          legal = false;
          break;
        }
        for (int j = 0; j < lhsSlice.getType().getShape().size(); j++) {
          if (j == outer.getDimension()) {
            if (slice2.getStartIndices()[j] != nextStart) {
              legal = false;
              break;
            }
          } else {
            if (slice2.getStartIndices()[j] != lhsSlice.getStartIndices()[j]) {
              legal = false;
              break;
            }
            if (slice2.getLimitIndices()[j] != lhsSlice.getLimitIndices()[j]) {
              legal = false;
              break;
            }
            if (slice2.getStrides()[j] != 1 || lhsSlice.getStrides()[j] != 1) {
              legal = false;
              break;
            }
          }
        }
        nextStart = slice2.getLimitIndices()[outer.getDimension()];
      }
      if (legal) {
        lhs = lhsSlice;
      }
    }

    stablehlo::SliceOp rhs = nullptr;
    if (auto rhsSlice = inners[0]
                            .getOperands()
                            .back()
                            .getDefiningOp<stablehlo::SliceOp>()) {
      bool legal = true;
      size_t nextStart = rhsSlice.getLimitIndices()[outer.getDimension()];
      for (int i = 1; i < inners.size(); i++) {
        auto slice2 =
            inners[i].getOperands().back().getDefiningOp<stablehlo::SliceOp>();
        if (!slice2) {
          legal = false;
          break;
        }
        if (slice2.getOperand() != rhsSlice.getOperand()) {
          legal = false;
          break;
        }
        for (int j = 0; j < rhsSlice.getType().getShape().size(); j++) {
          if (j == outer.getDimension()) {
            if (slice2.getStartIndices()[j] != nextStart) {
              legal = false;
              break;
            }
          } else {
            if (slice2.getStartIndices()[j] != rhsSlice.getStartIndices()[j]) {
              legal = false;
              break;
            }
            if (slice2.getLimitIndices()[j] != rhsSlice.getLimitIndices()[j]) {
              legal = false;
              break;
            }
            if (slice2.getStrides()[j] != 1 || rhsSlice.getStrides()[j] != 1) {
              legal = false;
              break;
            }
          }
        }
        nextStart = slice2.getLimitIndices()[outer.getDimension()];
      }
      if (legal) {
        rhs = rhsSlice;
      }
    }

    if (rhs && lhs && rhs.getOperand() == lhs.getOperand() &&
        rhs.getLimitIndices()[inners[0].getDimension()] -
                lhs.getStartIndices()[inners[0].getDimension()] ==
            outer.getType().getShape()[inners[0].getDimension()]) {
      bool legal = true;
      for (int j = 0; j < lhs.getType().getShape().size(); j++) {
        if (j != inners[0].getDimension()) {
          if (lhs.getStartIndices()[j] != rhs.getStartIndices()[j]) {
            legal = false;
            break;
          }
          if (lhs.getLimitIndices()[j] != rhs.getLimitIndices()[j]) {
            legal = false;
            break;
          }
        }
      }
      if (legal) {
        SmallVector<Value> newInner;
        for (auto concat : inners) {
          SmallVector<Value> innerOps;
          for (int i = 1; i < concat.getOperands().size() - 1; i++) {
            innerOps.push_back(concat.getOperands()[i]);
          }
          newInner.push_back(rewriter.create<stablehlo::ConcatenateOp>(
              concat.getLoc(), innerOps, concat.getDimension()));
        }
        auto newOuter = rewriter.create<stablehlo::ConcatenateOp>(
            outer.getLoc(), newInner, outer.getDimension());
        auto newSlice = rewriter.create<stablehlo::SliceOp>(
            outer.getLoc(), lhs.getOperand(), lhs.getStartIndices(),
            inners.back()
                .getOperands()
                .back()
                .getDefiningOp<stablehlo::SliceOp>()
                .getLimitIndices(),
            lhs.getStrides());
        auto iTy = RankedTensorType::get({}, rewriter.getI64Type());
        SmallVector<Value> starts(
            lhs.getType().getShape().size(),
            rewriter.create<stablehlo::ConstantOp>(
                outer.getLoc(), iTy, cast<ElementsAttr>(makeAttr(iTy, 0))));
        starts[outer.getDimension()] = rewriter.create<stablehlo::ConstantOp>(
            outer.getLoc(),
            cast<ElementsAttr>(
                makeAttr(iTy, lhs.getType().getShape()[outer.getDimension()])));
        rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
            outer, newSlice, newOuter, starts);
        return success();
      }
    }

    if (lhs) {
      llvm::errs() << "TODO: implement LHS side concat optimization\n";
    }

    if (rhs) {
      llvm::errs() << "TODO: implement RHS side concat optimization\n";
    }
    return failure();
  }
};

enum class SliceRequirement {
  AnyOperand = 0,
  NeedsSlice = 1,
  NeedsFull = 2,
};

stablehlo::DynamicUpdateSliceOp
concat_to_dus_slice_common(PatternRewriter &rewriter, Location loc,
                           RankedTensorType concatType, int dimension,
                           ArrayRef<Value> operands, SliceRequirement sliceReq,
                           mlir::sdy::TensorShardingPerValueAttr shard) {
  if (operands.size() < 3)
    return nullptr;

  stablehlo::SliceOp lhs = nullptr;
  bool hasSlice = false;
  if (auto lhsSlice = operands[0].getDefiningOp<stablehlo::SliceOp>()) {
    bool legal = true;
    for (int i = 0; i < lhsSlice.getType().getShape().size(); i++) {
      if (lhsSlice.getStartIndices()[i] != 0) {
        hasSlice = true;
      }
      if (lhsSlice.getStrides()[i] != 1) {
        legal = false;
        break;
      }
    }
    if (legal)
      lhs = lhsSlice;
  }

  if (!lhs)
    return nullptr;

  stablehlo::SliceOp rhs = nullptr;
  if (auto rhsSlice = operands.back().getDefiningOp<stablehlo::SliceOp>()) {
    bool legal = true;
    for (int i = 0; i < rhsSlice.getType().getShape().size(); i++) {
      if (rhsSlice.getStrides()[i] != 1) {
        legal = false;
        break;
      }
      if (i == dimension) {
        if (lhs.getStartIndices()[i] + concatType.getShape()[i] !=
            rhsSlice.getLimitIndices()[i]) {
          legal = false;
          break;
        }
      } else {
        if (lhs.getStartIndices()[i] != rhsSlice.getStartIndices()[i]) {
          legal = false;
          break;
        }
        if (lhs.getLimitIndices()[i] != rhsSlice.getLimitIndices()[i]) {
          legal = false;
          break;
        }
      }
      if (rhsSlice.getLimitIndices()[i] != concatType.getShape()[i]) {
        hasSlice = true;
      }
    }
    if (legal)
      rhs = rhsSlice;
  }

  if (!rhs)
    return nullptr;
  if (sliceReq == SliceRequirement::NeedsSlice && !hasSlice)
    return nullptr;
  if (sliceReq == SliceRequirement::NeedsFull && hasSlice)
    return nullptr;

  if (rhs.getOperand() != lhs.getOperand())
    return nullptr;

  SmallVector<Value> newOps;
  int start = lhs ? 1 : 0;
  int end = operands.size() - (rhs ? 1 : 0);
  for (int i = start; i < end; i++) {
    newOps.push_back(operands[i]);
  }
  Value innerConcat = newOps[0];
  if (newOps.size() != 1) {
    auto nConcat =
        rewriter.create<stablehlo::ConcatenateOp>(loc, newOps, dimension);
    innerConcat = nConcat;
    if (shard) {
      sdy::setShardings(nConcat, shard);
    }
  }

  auto iTy = RankedTensorType::get({}, rewriter.getI64Type());

  Value operand = lhs.getOperand();

  if (!shard) {
    if (auto opSharding = mlir::sdy::getSharding(operand)) {
      sdy::TensorShardingAttr opShardings[] = {opSharding};
      shard = sdy::TensorShardingPerValueAttr::get(operand.getContext(),
                                                   opShardings);
    }
  }

  if (hasSlice) {
    auto sloperand = rewriter.create<stablehlo::SliceOp>(
        lhs.getLoc(), lhs.getOperand(), lhs.getStartIndices(),
        rhs.getLimitIndices(), lhs.getStrides());
    if (shard) {
      sdy::setShardings(sloperand, shard);
    }
    operand = sloperand;
  }

  SmallVector<Value> starts(
      concatType.getShape().size(),
      rewriter.create<stablehlo::ConstantOp>(
          loc, iTy, cast<ElementsAttr>(makeAttr(iTy, 0))));

  if (lhs) {
    starts[dimension] = rewriter.create<stablehlo::ConstantOp>(
        loc, iTy,
        cast<ElementsAttr>(makeAttr(iTy, lhs.getType().getShape()[dimension])));
  }

  auto dus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
      loc, operand, innerConcat, starts);
  if (shard) {
    sdy::setShardings(dus, shard);
  }
  return dus;
}

struct ConcatToOneDimDUSSlice final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                              ConcatToOneDimDUSSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp outer,
                                    PatternRewriter &rewriter) const {

    stablehlo::DynamicUpdateSliceOp replacement = concat_to_dus_slice_common(
        rewriter, outer.getLoc(), outer.getType(), outer.getDimension(),
        llvm::to_vector(outer.getOperands()), SliceRequirement::NeedsSlice,
        sdy::getShardingPerValue(outer));
    if (!replacement)
      return failure();
    rewriter.replaceOp(outer, replacement);

    return success();
  }
};

struct ConcatReshapeToOneDimDUS final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                              ConcatReshapeToOneDimDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp outer,
                                    PatternRewriter &rewriter) const {

    SmallVector<Value> pre_reshape;
    for (auto operand : outer.getOperands()) {
      auto re = operand.getDefiningOp<stablehlo::ReshapeOp>();
      if (!re)
        return failure();
      auto pre_shape =
          cast<RankedTensorType>(re.getOperand().getType()).getShape();
      if (re.getType().getShape().size() + 1 != pre_shape.size()) {
        return failure();
      }
      for (auto &&[lhs, rhs] :
           llvm::zip_equal(re.getType().getShape(), pre_shape.slice(1))) {
        if (lhs != rhs)
          return failure();
      }
      if (pre_shape[0] != 1) {
        return failure();
      }
      pre_reshape.push_back(re.getOperand());
    }
    SmallVector<int64_t> subShape = llvm::to_vector(outer.getType().getShape());
    subShape.insert(subShape.begin(), 1);
    RankedTensorType subType =
        RankedTensorType::get(subShape, outer.getType().getElementType());
    stablehlo::DynamicUpdateSliceOp replacement = concat_to_dus_slice_common(
        rewriter, outer.getLoc(), subType, outer.getDimension() + 1,
        pre_reshape, SliceRequirement::AnyOperand, nullptr);
    if (!replacement)
      return failure();

    auto shard = sdy::getShardingPerValue(outer);
    auto reshaped = rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        outer, outer.getType(), replacement);
    if (shard) {
      sdy::setShardings(reshaped, shard);
    }
    return success();
  }
};

struct DUSDUS final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dus,
                                    PatternRewriter &rewriter) const {
    auto dus2 =
        dus.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();

    if (!dus2)
      return failure();

    if (dus.getUpdate().getType() != dus2.getUpdate().getType())
      return failure();

    for (auto &&[start1, start2] :
         llvm::zip_equal(dus.getStartIndices(), dus2.getStartIndices())) {
      if (start1 != start2)
        return failure();
    }
    rewriter.modifyOpInPlace(
        dus, [&]() { dus.getOperandMutable().set(dus2.getOperand()); });
    return success();
  }
};

// Optimization: DUSPad
// Pattern:
//   %padded_val = stablehlo.pad %original_data, %pad_val, low=[L...],
//   high=[H...], interior=[0...] %dus = stablehlo.dynamic_update_slice
//   %padded_val, %update_data, %idx...
// Constraint: The update region defined by %update_data and %idx falls entirely
//             within the %original_data region of %padded_val.
// Constraint: %padded_val has only %dus as its user.
// Constraint: Interior padding is zero.
// Rewrite to:
//   %idx_new = %idx - L // Adjust indices relative to original_data
//   %new_dus = stablehlo.dynamic_update_slice %original_data, %update_data,
//   %idx_new... %new_pad = stablehlo.pad %new_dus, %pad_val, low=[L...],
//   high=[H...], interior=[0...] replaceAllUsesWith(%dus, %new_pad)
struct DUSPad final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSPad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dus,
                                    PatternRewriter &rewriter) const {

    // 1. Check if the operand being updated comes from a pad op.
    auto padOp = dus.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!padOp) {
      return rewriter.notifyMatchFailure(dus, "Operand is not a PadOp");
    }

    // 2. Constraint: Check if the pad op has only the DUS as its user.
    if (!llvm::hasSingleElement(padOp->getUsers())) {
      return rewriter.notifyMatchFailure(dus, "PadOp has multiple users");
    }

    // 3. Constraint: Check for zero interior padding and non-negative edge
    // padding.
    if (llvm::any_of(padOp.getInteriorPadding(),
                     [](int64_t i) { return i != 0; })) {
      return rewriter.notifyMatchFailure(dus, "Requires zero interior padding");
    }
    if (llvm::any_of(padOp.getEdgePaddingLow(),
                     [](int64_t i) { return i < 0; }) ||
        llvm::any_of(padOp.getEdgePaddingHigh(),
                     [](int64_t i) { return i < 0; })) {
      return rewriter.notifyMatchFailure(dus,
                                         "Requires non-negative edge padding");
    }

    // 4. Get shapes and indices, require static shapes.
    auto dusOperandType = dyn_cast<RankedTensorType>(
        dus.getOperand().getType()); // = padOp result type
    auto updateType = dyn_cast<RankedTensorType>(dus.getUpdate().getType());
    auto originalDataType =
        dyn_cast<RankedTensorType>(padOp.getOperand().getType());

    if (!dusOperandType || !updateType || !originalDataType ||
        !dusOperandType.hasStaticShape() || !updateType.hasStaticShape() ||
        !originalDataType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          dus, "Requires static shapes for involved tensors");
    }

    ArrayRef<int64_t> dusOperandShape = dusOperandType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();
    ArrayRef<int64_t> originalDataShape = originalDataType.getShape();
    SmallVector<Value> startIndices = dus.getStartIndices();
    ArrayRef<int64_t> lowPadding = padOp.getEdgePaddingLow();

    SmallVector<Value> newDusStartIndices;
    Location loc = dus.getLoc();
    auto indexElementType =
        cast<ShapedType>(startIndices[0].getType()).getElementType();
    auto indexScalarType = RankedTensorType::get({}, indexElementType);

    // 5. Check if update region is within original data bounds & calculate new
    // indices.
    for (int64_t d = 0; d < dusOperandShape.size(); ++d) {
      DenseIntElementsAttr startAttr;
      if (!matchPattern(startIndices[d], m_Constant(&startAttr)) ||
          startAttr.getNumElements() != 1) {
        return rewriter.notifyMatchFailure(
            dus, "Requires constant scalar start indices");
      }
      int64_t startVal = (*startAttr.begin()).getSExtValue();
      int64_t updateSize = updateShape[d];
      int64_t endVal = startVal + updateSize; // Exclusive end
      int64_t lowPad = lowPadding[d];
      int64_t originalSize = originalDataShape[d];

      // Check bounds: update [startVal, endVal) must be within original data
      // region [lowPad, lowPad + originalSize)
      if (startVal < lowPad || endVal > (lowPad + originalSize)) {
        return rewriter.notifyMatchFailure(
            dus, "Update region extends into padded area for dim " +
                     std::to_string(d));
      }

      // Calculate new start index relative to original data
      int64_t newStartVal = startVal - lowPad;
      auto newStartAttr =
          rewriter.getIntegerAttr(indexElementType, newStartVal);
      auto newStartConst = rewriter.create<stablehlo::ConstantOp>(
          loc, indexScalarType,
          DenseElementsAttr::get(indexScalarType, newStartAttr));
      newDusStartIndices.push_back(newStartConst);
    }

    // 6. Perform the rewrite.
    // Create the new DUS on the original data.
    auto newDus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc,
        originalDataType,    // Result type matches original data
        padOp.getOperand(),  // Update the original data
        dus.getUpdate(),     // Use the same update value
        newDusStartIndices); // Use the adjusted indices

    // Create the new Pad operation using the result of the new DUS.
    auto newPad = rewriter.create<stablehlo::PadOp>(
        padOp.getLoc(),
        dus.getType(),      // Result type matches original DUS result
        newDus.getResult(), // Pad the result of the new DUS
        padOp.getPaddingValue(), padOp.getEdgePaddingLowAttr(),
        padOp.getEdgePaddingHighAttr(),
        padOp.getInteriorPaddingAttr()); // Assumed to be zeros by check above

    // Replace the original DUS with the new Pad operation.
    rewriter.replaceOp(dus, newPad.getResult());

    return success();
  }
};

struct DUSDUSConcat final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSDUSConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dus,
                                    PatternRewriter &rewriter) const {
    auto dus2 =
        dus.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();

    if (!dus2)
      return failure();

    RankedTensorType tys[2];
    stablehlo::DynamicUpdateSliceOp duses[2] = {dus, dus2};
    for (auto en : llvm::enumerate(duses)) {
      auto ty = dyn_cast<RankedTensorType>(en.value().getUpdate().getType());
      if (!ty)
        return failure();
      tys[en.index()] = ty;
    }

    if (dus.getOperand().getType() != dus2.getOperand().getType())
      return failure();

    ssize_t diffidx = -1;
    for (size_t i = 0; i < dus.getStartIndices().size(); i++) {
      if (dus.getStartIndices()[i] == dus2.getStartIndices()[i])
        continue;
      if (diffidx != -1) {
        return failure();
      }
      diffidx = i;
    }

    if (diffidx == -1) {
      for (size_t i = 0; i < dus.getStartIndices().size(); i++) {
        if (tys[0].getShape()[i] == tys[1].getShape()[i])
          continue;
        if (diffidx != -1) {
          return failure();
        }
        diffidx = i;
      }
    }

    if (diffidx == -1) {
      return failure();
    }

    // Sizes must be the same except for the differing index
    for (size_t i = 0; i < dus.getStartIndices().size(); i++) {
      if (i == diffidx)
        continue;
      if (tys[0].getShape()[i] != tys[1].getShape()[i])
        return failure();
    }

    int64_t idxs[2];

    for (auto en : llvm::enumerate(duses)) {
      auto val = en.value().getStartIndices()[diffidx];
      DenseIntElementsAttr startattr;
      if (!matchPattern(val, m_Constant(&startattr))) {
        return failure();
      }
      int64_t ival = (*startattr.begin()).getSExtValue();
      idxs[en.index()] = ival;
    }

    if (idxs[1] == idxs[0] + tys[0].getShape()[diffidx]) {
      Value operands[2] = {dus.getUpdate(), dus2.getUpdate()};
      auto concat = rewriter.create<stablehlo::ConcatenateOp>(
          dus.getLoc(), operands, diffidx);
      rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
          dus, dus2.getOperand(), concat, dus.getStartIndices());
      return success();
    } else if (idxs[0] == idxs[1] + tys[1].getShape()[diffidx]) {
      Value operands[2] = {dus2.getUpdate(), dus.getUpdate()};
      auto concat = rewriter.create<stablehlo::ConcatenateOp>(
          dus.getLoc(), operands, diffidx);
      rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
          dus, dus2.getOperand(), concat, dus2.getStartIndices());
      return success();
    } else if (idxs[1] >= idxs[0] && idxs[1] + tys[1].getShape()[diffidx] <=
                                         idxs[0] + tys[0].getShape()[diffidx]) {
      // the previous update (in dus1) was completely overwritten [e.g. dus0
      // starts before and end later]
      rewriter.modifyOpInPlace(
          dus, [&]() { dus.getOperandMutable().set(dus2.getOperand()); });
      return success();
    } else if (idxs[0] >= idxs[1] && idxs[0] + tys[0].getShape()[diffidx] <=
                                         idxs[1] + tys[1].getShape()[diffidx]) {
      // the new update is entirely within the space of the old update

      auto itype = dus.getStartIndices()[diffidx].getType();
      auto c0 = rewriter.create<stablehlo::ConstantOp>(
          dus.getLoc(), itype, cast<ElementsAttr>(makeAttr(itype, 0)));
      auto cidx = rewriter.create<stablehlo::ConstantOp>(
          dus.getLoc(), itype,
          cast<ElementsAttr>(makeAttr(itype, idxs[0] - idxs[1])));

      SmallVector<Value> idxs(dus.getStartIndices().size());
      for (size_t i = 0; i < dus.getStartIndices().size(); i++)
        idxs[i] = c0;
      idxs[diffidx] = cidx;

      auto within_dus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
          dus2.getLoc(), dus2.getUpdate(), dus.getUpdate(), idxs);
      rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
          dus, dus2.getOperand(), within_dus, dus2.getStartIndices());
      return success();
    }
    return failure();
  }
};

struct DynamicUpdateToConcat final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp,
                              DynamicUpdateToConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    SmallVector<size_t> mismatches;
    size_t idx = 0;
    for (auto &&[start, update_size, res_size] :
         llvm::zip(op.getStartIndices(), op.getUpdate().getType().getShape(),
                   op.getType().getShape())) {
      DenseIntElementsAttr startattr;
      if (!matchPattern(start, m_Constant(&startattr))) {
        return failure();
      }
      int64_t startv = (*startattr.begin()).getSExtValue();
      if (startv < 0)
        return failure();

      if (startv + update_size > res_size)
        return failure();

      if (startv == 0 && update_size == res_size) {
        idx++;
        continue;
      }
      mismatches.push_back(idx);
      idx++;
    }

    if (mismatches.size() != 1)
      return failure();
    auto dim = mismatches[0];

    DenseIntElementsAttr startattr;
    if (!matchPattern(op.getStartIndices()[dim], m_Constant(&startattr))) {
      return failure();
    }
    int64_t startv = (*startattr.begin()).getSExtValue();

    SmallVector<Value> toConcat;

    if (startv != 0) {
      SmallVector<int64_t> starts(op.getType().getShape().size(), 0);
      SmallVector<int64_t> ends(op.getType().getShape().begin(),
                                op.getType().getShape().end());
      SmallVector<int64_t> steps(op.getType().getShape().size(), 1);
      ends[dim] = startv;
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), op.getOperand(), starts, ends, steps));
    }
    toConcat.push_back(op.getUpdate());
    auto update_size = op.getUpdate().getType().getShape()[dim];
    auto res_size = op.getType().getShape()[dim];
    if (startv + update_size != res_size) {
      SmallVector<int64_t> starts(op.getType().getShape().size(), 0);
      SmallVector<int64_t> ends(op.getType().getShape().begin(),
                                op.getType().getShape().end());
      SmallVector<int64_t> steps(op.getType().getShape().size(), 1);
      starts[dim] = startv + update_size;
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), op.getOperand(), starts, ends, steps));
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, op.getType(),
                                                          toConcat, dim);
    return success();
  }
};

struct SliceOfDynamicUpdate final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceOfDynamicUpdate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto dyn = op.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dyn)
      return failure();

    // Try to use the updated value
    {
      SmallVector<int64_t> start;
      SmallVector<int64_t> end;
      SmallVector<int64_t> step;

      bool legal = true;
      for (auto &&[nstart, nend, nstep, update_start, update_size] : llvm::zip(
               op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
               dyn.getStartIndices(), dyn.getUpdate().getType().getShape())) {
        DenseIntElementsAttr startattr;
        if (!matchPattern(update_start, m_Constant(&startattr))) {
          legal = false;
          break;
        }
        int64_t startv = (*startattr.begin()).getSExtValue();
        if (startv < 0) {
          legal = false;
          break;
        }

        // see if the slice is exclusively inside the update.

        // slice starts below insertion
        if (nstart < startv) {
          legal = false;
          break;
        }

        // slice ends after insertion
        if (nend > startv + update_size) {
          legal = false;
          break;
        }

        start.push_back(nstart - startv);
        end.push_back(nend - startv);
        step.push_back(nstep);
      }

      if (legal) {
        rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(op, dyn.getUpdate(),
                                                        start, end, step);
        return success();
      }
    }

    // Try proving that there can be no overlap
    {
      bool no_overlap = false;

      for (auto &&[nstart, nend, nstep, update_start, update_size] : llvm::zip(
               op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
               dyn.getStartIndices(), dyn.getUpdate().getType().getShape())) {
        DenseIntElementsAttr startattr;
        if (!matchPattern(update_start, m_Constant(&startattr))) {
          continue;
        }

        int64_t startv = (*startattr.begin()).getSExtValue();
        // slice ends below insertion
        if (nend <= startv) {
          no_overlap = true;
          break;
        }

        // slice starts after insertion end
        if (nstart >= startv + update_size) {
          no_overlap = true;
          break;
        }
      }

      if (no_overlap) {
        rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
            op, dyn.getOperand(), op.getStartIndices(), op.getLimitIndices(),
            op.getStrides());
        return success();
      }
    }

    return failure();
  }
};

struct SliceDUSToConcat final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceDUSToConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                    PatternRewriter &rewriter) const {

    auto dusOp =
        sliceOp.getOperand().getDefiningOp<stablehlo::DynamicUpdateSliceOp>();
    if (!dusOp) {
      return rewriter.notifyMatchFailure(sliceOp, "Operand is not a DUS");
    }

    // Constraint: DUS has only the slice as user
    if (!dusOp.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(sliceOp, "DUS has multiple users");
    }

    // Get DUS info
    Value targetOperand = dusOp.getOperand();
    Value updateVal = dusOp.getUpdate();
    SmallVector<Value> dusIndexVals = dusOp.getStartIndices();

    auto targetType = dyn_cast<RankedTensorType>(targetOperand.getType());
    auto updateType = dyn_cast<RankedTensorType>(updateVal.getType());
    auto sliceResultType =
        dyn_cast<RankedTensorType>(sliceOp.getType()); // Type of %283

    if (!targetType || !updateType || !sliceResultType ||
        !targetType.hasStaticShape() || !updateType.hasStaticShape() ||
        !sliceResultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(sliceOp, "Requires static shapes");
    }

    ArrayRef<int64_t> updateShape = updateType.getShape();
    int rank = targetType.getRank();

    // Constraint: DUS indices must be constant
    SmallVector<int64_t> dusStartIndices;
    dusStartIndices.reserve(rank);
    for (Value idxVal : dusIndexVals) {
      DenseIntElementsAttr idxAttr;
      if (!matchPattern(idxVal, m_Constant(&idxAttr)) ||
          idxAttr.getNumElements() != 1) {
        return rewriter.notifyMatchFailure(
            sliceOp, "DUS indices must be constant scalars");
      }
      dusStartIndices.push_back((*idxAttr.begin()).getSExtValue());
    }

    if (llvm::any_of(sliceOp.getStrides(), [](int64_t s) { return s != 1; })) {
      return rewriter.notifyMatchFailure(sliceOp,
                                         "Requires slice strides of 1");
    }

    int concatDim = -1; // Dimension along which to concatenate results

    SmallVector<int64_t> dusLimitIndices;
    for (int d = 0; d < rank; ++d) {
      int64_t dusStart = dusStartIndices[d];
      int64_t dusEnd = dusStart + updateShape[d];
      dusLimitIndices.push_back(dusEnd);
      int64_t sliceStart = sliceOp.getStartIndices()[d];
      int64_t sliceLimit = sliceOp.getLimitIndices()[d];

      if (dusStart == sliceStart && dusEnd == sliceLimit) {
        continue;
      }

      if (concatDim != -1) {
        return failure();
      }
      concatDim = d;

      // if entirely within one or the other, use the other pattern
      // 1) if we are outside the update
      if (sliceLimit <= dusStart || sliceStart >= dusEnd) {
        return failure();
      }

      // 2) we are fully within the update
      if (sliceStart >= dusStart & sliceStart <= dusEnd) {
        return failure();
      }
    }

    if (concatDim == -1)
      return failure();

    SmallVector<Value> toConcat;

    if (sliceOp.getStartIndices()[concatDim] < dusStartIndices[concatDim]) {
      assert(sliceOp.getLimitIndices()[concatDim] > dusStartIndices[concatDim]);
      SmallVector<int64_t> newLimit =
          llvm::to_vector(sliceOp.getLimitIndices());
      newLimit[concatDim] = dusStartIndices[concatDim];
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          sliceOp.getLoc(), dusOp.getOperand(), sliceOp.getStartIndices(),
          newLimit, sliceOp.getStrides()));
    }

    if (sliceOp.getLimitIndices()[concatDim] >= dusLimitIndices[concatDim] &&
        sliceOp.getStartIndices()[concatDim] <= dusStartIndices[concatDim]) {
      toConcat.push_back(dusOp.getUpdate());
    } else if (sliceOp.getLimitIndices()[concatDim] <=
               dusLimitIndices[concatDim]) {
      assert(sliceOp.getStartIndices()[concatDim] < dusStartIndices[concatDim]);
      SmallVector<int64_t> newStart(dusStartIndices);
      SmallVector<int64_t> newLimit =
          llvm::to_vector(sliceOp.getLimitIndices());
      for (int i = 0; i < rank; i++) {
        newStart[i] -= dusStartIndices[i];
        newLimit[i] -= dusStartIndices[i];
      }
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          sliceOp.getLoc(), dusOp.getUpdate(), newStart, newLimit,
          sliceOp.getStrides()));
    } else if (sliceOp.getStartIndices()[concatDim] >=
               dusStartIndices[concatDim]) {
      assert(sliceOp.getLimitIndices()[concatDim] > dusLimitIndices[concatDim]);
      SmallVector<int64_t> newStart =
          llvm::to_vector(sliceOp.getStartIndices());
      SmallVector<int64_t> newLimit(dusLimitIndices);
      for (int i = 0; i < rank; i++) {
        newStart[i] -= dusStartIndices[i];
        newLimit[i] -= dusStartIndices[i];
      }
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          sliceOp.getLoc(), dusOp.getUpdate(), newStart, newLimit,
          sliceOp.getStrides()));
    }

    if (sliceOp.getLimitIndices()[concatDim] > dusLimitIndices[concatDim]) {
      assert(sliceOp.getStartIndices()[concatDim] < dusLimitIndices[concatDim]);
      SmallVector<int64_t> newStart =
          llvm::to_vector(sliceOp.getStartIndices());
      newStart[concatDim] = dusLimitIndices[concatDim];
      toConcat.push_back(rewriter.create<stablehlo::SliceOp>(
          sliceOp.getLoc(), dusOp.getOperand(), newStart,
          sliceOp.getLimitIndices(), sliceOp.getStrides()));
    }

    auto preType = sliceOp.getType();

    auto newConcat = rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        sliceOp, toConcat, concatDim);
    assert(preType == newConcat.getType());
    (void)preType;
    (void)newConcat;

    return success();
  }
};

static bool definedOutside(Value v, Operation *op) {
  return !op->isAncestor(v.getParentBlock()->getParentOp());
}

// Replace while op iteration variables which are not updated with their
// upcoming value
template <typename T> struct LICM : public CheckedOpRewritePattern<T, LICM<T>> {
  using CheckedOpRewritePattern<T, LICM<T>>::CheckedOpRewritePattern;

  bool single_user;
  LICM(bool single_user, MLIRContext *context, PatternBenefit benefit = 1,
       ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern<T, LICM<T>>(context, benefit, generatedNames),
        single_user(single_user) {}

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    auto whileOp = op->template getParentOfType<stablehlo::WhileOp>();
    if (!whileOp)
      return failure();
    for (auto operand : op->getOperands()) {
      if (!definedOutside(operand, whileOp))
        return failure();
      if (single_user) {
        for (auto U : operand.getUsers()) {
          if (U == op)
            continue;
          if (whileOp->isAncestor(U))
            return failure();
        }
      }
    }
    rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(whileOp); });
    return success();
  }
};

struct LICMElementwise
    : public CheckedOpTraitRewritePattern<OpTrait::Elementwise,
                                          LICMElementwise> {
  using CheckedOpTraitRewritePattern<
      OpTrait::Elementwise, LICMElementwise>::CheckedOpTraitRewritePattern;

  bool single_user;
  LICMElementwise(bool single_user, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : CheckedOpTraitRewritePattern<OpTrait::Elementwise, LICMElementwise>(
            context, benefit),
        single_user(single_user) {}

  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const {
    if (!isa<stablehlo::WhileOp>(op->getParentOp()))
      return failure();
    for (auto operand : op->getOperands()) {
      if (!definedOutside(operand, op->getParentOp()))
        return failure();
      if (single_user) {
        for (auto U : operand.getUsers()) {
          if (U == op)
            continue;
          if (op->getParentOp()->isAncestor(U))
            return failure();
        }
      }
    }
    rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(op->getParentOp()); });
    return success();
  }
};

// slice(broadcast x) -> broadcast(slice x)
struct SliceBroadcast final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceBroadcast> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto bcast = op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcast)
      return failure();

    if (!llvm::hasSingleElement(bcast->getUsers()))
      return failure();

    SmallVector<int64_t> nbcast_idx;

    auto preShape = cast<RankedTensorType>(bcast.getOperand().getType());
    SmallVector<int64_t> in_start(preShape.getShape().size(), 0);
    SmallVector<int64_t> in_end(preShape.getShape().begin(),
                                preShape.getShape().end());
    SmallVector<int64_t> in_stride(preShape.getShape().size(), 1);

    bool innerSlice = false;

    size_t outidx = 0;
    for (auto &&[start, end, step, indim, outdim] :
         llvm::zip(op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
                   bcast.getType().getShape(), op.getType().getShape())) {
      ssize_t idx = -1;
      for (auto en : llvm::enumerate(bcast.getBroadcastDimensions())) {
        if (en.value() == outidx) {
          idx = en.index();
          break;
        }
      }

      nbcast_idx.push_back(outdim);
      if (idx == -1) {
        // being broadcast just resize the outshape
      } else {
        auto preShapeIdx = preShape.getShape()[idx];

        // slice the inner shape
        if (preShapeIdx == indim) {
          in_start[idx] = start;
          in_end[idx] = end;
          in_stride[idx] = step;
          innerSlice = true;
        } else if (preShapeIdx != 1) {
          return failure();
        }
      }

      outidx++;
    }

    if (innerSlice && !llvm::hasSingleElement(bcast->getUsers()))
      return failure();

    Value tobcast = bcast.getOperand();
    if (innerSlice)
      tobcast = rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), tobcast, in_start, in_end, in_stride);

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), tobcast, bcast.getBroadcastDimensions());
    return success();
  }
};

SmallVector<int64_t> invertPermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> res(perm.size(), 0);
  for (auto en : llvm::enumerate(perm)) {
    res[en.value()] = en.index();
  }
  return res;
}

stablehlo::SliceOp sliceTransposeHelper(stablehlo::TransposeOp transpose,
                                        PatternRewriter &rewriter,
                                        ArrayRef<int64_t> starts,
                                        ArrayRef<int64_t> limits,
                                        ArrayRef<int64_t> strides) {
  SmallVector<int64_t> start;
  SmallVector<int64_t> end;
  SmallVector<int64_t> step;
  for (auto ind : invertPermutation(transpose.getPermutation())) {
    start.push_back(starts[ind]);
    end.push_back(limits[ind]);
    step.push_back(strides[ind]);
  }

  return rewriter.create<stablehlo::SliceOp>(
      transpose.getLoc(), transpose.getOperand(), start, end, step);
}

// slice(transpose x) -> transpose(slice x)
struct SliceTranspose final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto transpose = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transpose || !llvm::hasSingleElement(transpose->getUsers()))
      return failure();

    auto newslice =
        sliceTransposeHelper(transpose, rewriter, op.getStartIndices(),
                             op.getLimitIndices(), op.getStrides());
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, newslice, transpose.getPermutation());
    return success();
  }
};

// slice(reduce_window x, last_idx) -> reduce x
struct SliceReduceWindow
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceReduceWindow> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceReduceWindow>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reduceWindow =
        op.getOperand().getDefiningOp<stablehlo::ReduceWindowOp>();
    if (!reduceWindow)
      return failure();

    if (!llvm::hasSingleElement(reduceWindow->getUsers()))
      return failure();

    // Check window parameters indicate full reduction along one dimension
    auto windowDims = reduceWindow.getWindowDimensions();
    auto windowStrides = reduceWindow.getWindowStrides();
    auto windowDilations = reduceWindow.getWindowDilations();
    auto baseDilations = reduceWindow.getBaseDilations();

    if (!reduceWindow.getPadding())
      return failure();
    auto padding = reduceWindow.getPadding()->getValues<int64_t>();

    // Check if the window strides are all 1 or unspecified
    if (windowStrides && !llvm::all_of(*windowStrides, [](int64_t stride) {
          return stride == 1;
        }))
      return failure();

    // Check if the window dilations are all 1 or unspecified
    if (windowDilations &&
        !llvm::all_of(*windowDilations,
                      [](int64_t dilation) { return dilation == 1; }))
      return failure();

    // Check if the base dilations are all 1 or unspecified
    if (baseDilations && !llvm::all_of(*baseDilations, [](int64_t dilation) {
          return dilation == 1;
        }))
      return failure();

    // Find which dimension has window size > 1 (the reduction dimension)
    int64_t reductionDim = -1;
    auto inputType =
        dyn_cast<ShapedType>(reduceWindow.getInputs()[0].getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    for (unsigned i = 0; i < windowDims.size(); ++i) {
      if (windowDims[i] == inputShape[i]) {
        if (reductionDim != -1)
          return failure();
        reductionDim = i;
      }
    }
    if (reductionDim == -1)
      return failure();

    // Check that padding covers exactly (window_size-1) elements before in
    // reduction dim
    // FIXME: do other padding values have to be 0?
    for (unsigned i = 0; i < windowDims.size(); ++i) {
      if (i == reductionDim) {
        if (padding[2 * i] != windowDims[i] - 1)
          return failure();
      }
    }

    // Check this is a slice taking the last element in reduction dim
    auto sliceStarts = op.getStartIndices();
    auto sliceLimits = op.getLimitIndices();
    auto sliceStrides = op.getStrides();

    for (int64_t i = 0; i < sliceStarts.size(); ++i) {
      if (!sliceStrides.empty() && sliceStrides[i] != 1)
        return failure();

      if (i == reductionDim) {
        if (sliceStarts[i] != windowDims[i] - 1 ||
            sliceLimits[i] != windowDims[i])
          return failure();
      } else {
        if (sliceStarts[i] != 0 || sliceLimits[i] != inputShape[i])
          return failure();
      }
    }

    // Replace with direct reduce
    auto input = reduceWindow.getInputs()[0];
    auto initVal = reduceWindow.getInitValues()[0];
    // Compute the result type for the new ReduceOp by removing the reduction
    // dimension
    SmallVector<int64_t> resultShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (static_cast<int64_t>(i) != reductionDim) {
        resultShape.push_back(inputShape[i]);
      }
    }
    Type resultType =
        RankedTensorType::get(resultShape, inputType.getElementType());

    rewriter.setInsertionPoint(reduceWindow);

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        reduceWindow.getLoc(), TypeRange(resultType), input, initVal,
        rewriter.getDenseI64ArrayAttr({reductionDim}));

    // Clone the reduction body
    rewriter.inlineRegionBefore(reduceWindow.getBody(), reduceOp.getBody(),
                                reduceOp.getBody().end());

    // Create a reshape to match the slice output shape
    Value result = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(), op.getResult().getType(), reduceOp.getResult(0));

    // Replace the slice with the reduce result
    rewriter.replaceOp(op, result);
    rewriter.eraseOp(reduceWindow);

    return success();
  }
};

// transpose(dynamic_slice x) -> dynamic_slice(transpose x)
struct TransposeDynamicSlice final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeDynamicSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto dynamicSlice =
        op.getOperand().getDefiningOp<stablehlo::DynamicSliceOp>();
    if (!dynamicSlice)
      return failure();
    bool singleUser = dynamicSlice->getResult(0).hasOneUse();
    if (!singleUser)
      return failure();

    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), dynamicSlice.getOperand(), op.getPermutation());

    // Extract the original permutation, start indices, limit indices, and
    // strides
    SmallVector<int64_t> permutation = llvm::to_vector(op.getPermutation());

    SmallVector<Value> startIndices(permutation.size());
    SmallVector<int64_t> sliceSizes(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      size_t permIndex = permutation[i];
      startIndices[i] = dynamicSlice.getStartIndices()[permIndex];
      sliceSizes[i] = dynamicSlice.getSliceSizes()[permIndex];
    }

    // Create a new dynamic slice
    rewriter.replaceOpWithNewOp<stablehlo::DynamicSliceOp>(
        op, newTranspose, startIndices, sliceSizes);
    if (singleUser)
      rewriter.eraseOp(dynamicSlice);
    return success();
  }
};

// transpose(slice x) -> slice(transpose x)
struct TransposeSlice final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto slice = op.getOperand().getDefiningOp<stablehlo::SliceOp>();
    // if (!slice || !llvm::hasSingleElement(slice->getUsers()))
    if (!slice)
      return failure();

    bool singleUser = slice->getResult(0).hasOneUse();

    // First create transpose of the slice's operand
    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), slice.getOperand(), op.getPermutation());

    // We need to compute the result type for the new slice
    auto resultType = op.getType();

    // Extract the original permutation, start indices, limit indices, and
    // strides
    SmallVector<int64_t> permutation;
    for (auto val : op.getPermutation()) {
      permutation.push_back(static_cast<int64_t>(val));
    }

    // Get the original indices
    SmallVector<int64_t> startIndices;
    SmallVector<int64_t> limitIndices;
    SmallVector<int64_t> strides;

    // Convert DenseI64ArrayAttr to SmallVector<int64_t>
    for (auto [start, stop, stride] :
         llvm::zip(slice.getStartIndices(), slice.getLimitIndices(),
                   slice.getStrides())) {
      startIndices.push_back(start);
      limitIndices.push_back(stop);
      strides.push_back(stride);
    }

    // Permute the indices
    SmallVector<int64_t> permutedStartIndices(permutation.size());
    SmallVector<int64_t> permutedLimitIndices(permutation.size());
    SmallVector<int64_t> permutedStrides(permutation.size());

    for (size_t i = 0; i < permutation.size(); ++i) {
      size_t permIndex = permutation[i];
      permutedStartIndices[i] = startIndices[permIndex];
      permutedLimitIndices[i] = limitIndices[permIndex];
      permutedStrides[i] = strides[permIndex];
    }

    // Create the new slice operation with permuted indices
    auto newSlice = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), resultType, newTranspose,
        rewriter.getDenseI64ArrayAttr(permutedStartIndices),
        rewriter.getDenseI64ArrayAttr(permutedLimitIndices),
        rewriter.getDenseI64ArrayAttr(permutedStrides));

    rewriter.replaceOp(op, newSlice);
    if (singleUser)
      rewriter.eraseOp(slice);
    return success();
  }
};

struct TransposeAllUsersSlice final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeAllUsersSlice> {
  using CheckedOpRewritePattern<
      stablehlo::TransposeOp, TransposeAllUsersSlice>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    if (llvm::hasSingleElement(op->getUsers()))
      return rewriter.notifyMatchFailure(op,
                                         "should be handled by SliceTranspose");

    SmallVector<stablehlo::SliceOp> sliceOps;
    for (auto user : op->getUsers()) {
      auto sliceOp = dyn_cast<stablehlo::SliceOp>(user);
      if (!sliceOp)
        return failure();

      // only propagate down if we know a different optimization will clean this
      // up
      for (auto downstreamUser : sliceOp->getUsers()) {
        if (!isa<stablehlo::TransposeOp, stablehlo::BroadcastInDimOp,
                 stablehlo::DotGeneralOp>(downstreamUser))
          return failure();
      }

      sliceOps.push_back(sliceOp);
    }

    auto mapping = getInversePermutation(op.getPermutation());

    for (int64_t i = 0; i < sliceOps.size(); ++i) {
      auto origSlice = sliceOps[i];

      SmallVector<int64_t> originalStartIndices =
          llvm::to_vector(origSlice.getStartIndices());
      SmallVector<int64_t> originalLimitIndices =
          llvm::to_vector(origSlice.getLimitIndices());
      SmallVector<int64_t> originalStrides =
          llvm::to_vector(origSlice.getStrides());

      SmallVector<int64_t> newStartIndices;
      SmallVector<int64_t> newLimitIndices;
      SmallVector<int64_t> newStrides;

      for (size_t j = 0; j < mapping.size(); ++j) {
        size_t permIndex = mapping[j];
        newStartIndices.push_back(originalStartIndices[permIndex]);
        newLimitIndices.push_back(originalLimitIndices[permIndex]);
        newStrides.push_back(originalStrides[permIndex]);
      }

      auto newSlice = rewriter.create<stablehlo::SliceOp>(
          origSlice.getLoc(), op.getOperand(),
          rewriter.getDenseI64ArrayAttr(newStartIndices),
          rewriter.getDenseI64ArrayAttr(newLimitIndices),
          rewriter.getDenseI64ArrayAttr(newStrides));
      rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(origSlice, newSlice,
                                                          op.getPermutation());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct SliceElementwise final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceElementwise> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto elem = op.getOperand().getDefiningOp();
    if (!elem)
      return failure();
    if (!elem->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();
    if (llvm::hasSingleElement(elem->getUsers())) {
      SmallVector<Value> ops;
      for (auto v : elem->getOperands()) {
        ops.push_back(rewriter.create<stablehlo::SliceOp>(
            op.getLoc(), v, op.getStartIndices(), op.getLimitIndices(),
            op.getStrides()));
      }
      auto nex = rewriter.create(
          elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
          TypeRange(op->getResult(0).getType()), elem->getAttrs(), {}, {});
      rewriter.replaceOp(op, nex);
      return success();
    }

    SmallVector<int64_t> starts(op.getStartIndices().begin(),
                                op.getStartIndices().end());
    SmallVector<int64_t> stops(op.getLimitIndices().begin(),
                               op.getLimitIndices().end());
    SmallVector<int64_t> ints(op.getStrides().begin(), op.getStrides().end());
    SmallVector<stablehlo::SliceOp> todo;
    SmallVector<int64_t> sizes;
    for (auto u : elem->getUsers()) {
      auto sop = dyn_cast<stablehlo::SliceOp>(u);
      if (!sop)
        return failure();
      for (auto en : llvm::enumerate(sop.getType().getShape())) {
        auto start = sop.getStartIndices()[en.index()];
        auto stop = sop.getLimitIndices()[en.index()];
        auto stride = sop.getStrides()[en.index()];

        if (stride != ints[en.index()])
          ints[en.index()] = 1;
        // If the difference between the starts is not an even multiple of the
        // stride change the stride to 1
        if (start != starts[en.index()]) {
          if ((start - starts[en.index()]) % ints[en.index()] != 0) {
            ints[en.index()] = 1;
          }
        }
        if (start < starts[en.index()])
          starts[en.index()] = start;
        if (stop > stops[en.index()])
          stops[en.index()] = stop;
      }
      todo.push_back(sop);
    }
    bool changed = false;
    for (auto en : llvm::enumerate(op.getOperand().getType().getShape())) {
      if (starts[en.index()] != 0) {
        changed = true;
      }
      if (stops[en.index()] < en.value()) {
        changed = true;
      }
      if (ints[en.index()] != 1) {
        changed = true;
      }
      sizes.push_back((stops[en.index()] - starts[en.index()]) /
                      ints[en.index()]);
    }
    if (!changed)
      return failure();
    rewriter.setInsertionPoint(elem);
    SmallVector<Value> ops;
    for (auto v : elem->getOperands()) {
      ops.push_back(rewriter.create<stablehlo::SliceOp>(op.getLoc(), v, starts,
                                                        stops, ints));
    }
    auto nex = rewriter.create(
        elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
        TypeRange(RankedTensorType::get(
            sizes, op.getOperand().getType().getElementType())),
        elem->getAttrs(), {}, {});

    for (auto sl : todo) {
      SmallVector<int64_t> sstarts;
      SmallVector<int64_t> sstops;
      SmallVector<int64_t> sints;

      for (auto &&[start, stop, stride, ostart, ostop, ostride] :
           llvm::zip(sl.getStartIndices(), sl.getLimitIndices(),
                     sl.getStrides(), starts, stops, ints)) {
        if (stride == ostride) {
          sstarts.push_back(start - ostart);
          sstops.push_back((stop - ostart) / stride);
          sints.push_back(1);
        } else {
          assert(ostride == 1);
          sstarts.push_back(start - ostart);
          sstops.push_back(stop - ostart);
          sints.push_back(stride);
        }
      }
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(sl, nex->getResult(0),
                                                      sstarts, sstops, sints);
    }
    return success();
  }
};

LogicalResult slicePadHelper(
    stablehlo::PadOp pad, ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
    ArrayRef<int64_t> strides, SmallVectorImpl<int64_t> &start,
    SmallVectorImpl<int64_t> &end, SmallVectorImpl<int64_t> &step,
    SmallVectorImpl<int64_t> &lpads, SmallVectorImpl<int64_t> &hpads,
    SmallVectorImpl<int64_t> &interiors, bool &broadcastres, bool &needspad) {
  assert(start.size() == 0);
  assert(end.size() == 0);
  assert(step.size() == 0);
  assert(lpads.size() == 0);
  assert(hpads.size() == 0);
  assert(interiors.size() == 0);
  assert(!broadcastres);
  assert(!needspad);
  assert(starts.size() == pad.getOperand().getType().getShape().size());
  assert(limits.size() == pad.getOperand().getType().getShape().size());
  assert(strides.size() == pad.getOperand().getType().getShape().size());

  if (anyPadSizesNegative(pad))
    return failure();

  for (auto &&[nstart, nend, nstep, lpad, hpad, interior, inshape, outshape] :
       llvm::zip(starts, limits, strides, pad.getEdgePaddingLow(),
                 pad.getEdgePaddingHigh(), pad.getInteriorPadding(),
                 pad.getOperand().getType().getShape(),
                 pad.getType().getShape())) {
    if (nstep != 1)
      return failure();
    if (interior != 0)
      return failure();

    // slice goes from [nstart, nend]
    // pad result is [0..lpad][lpad... outshape - hpad][outshape - hpad...
    // outshape]

    // start of slice starts after end of value being padded
    if (nstart >= outshape - hpad) {
      broadcastres = true;
      return success();
    }
    // slice ends before the start of value being padded
    if (nend <= lpad) {
      broadcastres = true;
      return success();
    }
    if (nstart - lpad < 0) {
      start.push_back(0);
      lpads.push_back(lpad - nstart);
      needspad = true;
    } else {
      start.push_back(nstart - lpad);
      lpads.push_back(0);
    }
    if (nend - lpad > inshape) {
      end.push_back(inshape);
      hpads.push_back(nend - lpad - inshape);
      needspad = true;
    } else {
      end.push_back(nend - lpad);
      hpads.push_back(0);
    }

    step.push_back(1);
    interiors.push_back(0);
  }
  return success();
}

// slice(pad x) -> pad(slice x)
struct SlicePad final : CheckedOpRewritePattern<stablehlo::SliceOp, SlicePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    SmallVector<int64_t> step;

    SmallVector<int64_t> lpads;
    SmallVector<int64_t> hpads;
    SmallVector<int64_t> interiors;

    bool needspad = false;
    bool broadcastres = false;
    if (!slicePadHelper(pad, op.getStartIndices(), op.getLimitIndices(),
                        op.getStrides(), start, end, step, lpads, hpads,
                        interiors, broadcastres, needspad)
             .succeeded())
      return failure();

    if (broadcastres) {
      rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
          op, op.getType(), pad.getPaddingValue(),
          rewriter.getDenseI64ArrayAttr({}));
      return success();
    }

    if (needspad) {
      auto nslice = rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), pad.getOperand(), start, end, step);
      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          op, nslice, pad.getPaddingValue(), lpads, hpads, interiors);
    } else {
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(op, pad.getOperand(),
                                                      start, end, step);
    }
    return success();
  }
};

// From
// https:github.com/openxla/stablehlo/blob/5d1a9c892500c2e9fecbfedfa66ffe84ff1caf7b/stablehlo/dialect/StablehloOps.cpp#L1498C1-L1532C1
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

static bool isEligibleForCompactPrint(stablehlo::ReduceOp op) {
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
  auto retOp = dyn_cast<stablehlo::ReturnOp>(block.getTerminator());
  if (!retOp)
    return false;

  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

struct ReduceToReshape final
    : CheckedOpRewritePattern<stablehlo::ReduceOp, ReduceToReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1)
      return failure();
    if (!isEligibleForCompactPrint(op))
      return failure();
    auto inpTy = cast<RankedTensorType>(op.getInputs()[0].getType());
    for (auto idx : op.getDimensions()) {
      if (inpTy.getShape()[idx] != 1)
        return failure();
    }

    auto reshaped = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(), op.getResult(0).getType(), op.getInputs()[0]);

    Operation &innerOp = op.getBody().front().front();

    auto bcast = rewriter.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(), reshaped.getType(), op.getInitValues()[0],
        rewriter.getDenseI64ArrayAttr({}));
    Value vals[2] = {bcast, reshaped};
    auto res = rewriter.create(
        op.getLoc(), innerOp.getName().getIdentifier(), ValueRange(vals),
        TypeRange(op->getResult(0).getType()), innerOp.getAttrs(), {}, {});

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ReducePad
    : public CheckedOpRewritePattern<stablehlo::ReduceOp, ReducePad> {
  using CheckedOpRewritePattern<stablehlo::ReduceOp,
                                ReducePad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-operand single-init reduce is supported");
    }
    // TODO: min/max can also be an option since they are dropped
    if (!isa<stablehlo::AddOp>(op.getRegion().getBlocks().front().front())) {
      return rewriter.notifyMatchFailure(op, "only add is currently supported");
    }

    Value input = op.getInputs()[0];
    auto pad = input.getDefiningOp<stablehlo::PadOp>();
    if (!pad) {
      return rewriter.notifyMatchFailure(op, "input source is not a pad op");
    }
    if (anyPadSizesNegative(pad))
      return failure();

    if (!matchPattern(pad.getPaddingValue(), m_AnyZeroFloat()))
      return failure();

    SmallVector<int64_t> shape;

    SmallVector<int64_t> low;
    SmallVector<int64_t> high;
    SmallVector<int64_t> inner;
    bool needsPostPad = false;
    for (auto en : llvm::enumerate(
             cast<RankedTensorType>(pad.getOperand().getType()).getShape())) {
      if (llvm::is_contained(op.getDimensions(), en.index()))
        continue;
      shape.push_back(en.value());
      low.push_back(pad.getEdgePaddingLow()[en.index()]);
      high.push_back(pad.getEdgePaddingHigh()[en.index()]);
      inner.push_back(pad.getInteriorPadding()[en.index()]);
      needsPostPad = true;
    }

    auto newReduction = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(),
        TypeRange(RankedTensorType::get(
            shape,
            cast<RankedTensorType>(op->getResultTypes()[0]).getElementType())),
        ValueRange(pad.getOperand()), op.getInitValues(), op.getDimensions());
    newReduction.getRegion().takeBody(op.getRegion());

    Value res = newReduction->getResult(0);
    if (needsPostPad) {
      auto ctype = RankedTensorType::get(
          {}, cast<RankedTensorType>(res.getType()).getElementType());
      res = rewriter.create<stablehlo::PadOp>(
          op.getLoc(), res,
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), ctype, cast<ElementsAttr>(makeAttr(ctype, 0))),
          low, high, inner);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertConcat final
    : CheckedOpRewritePattern<stablehlo::ConvertOp, ConvertConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) const {
    auto concat = op.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    SmallVector<Value> newvals;
    for (auto v : concat.getOperands()) {
      newvals.push_back(rewriter.create<stablehlo::ConvertOp>(
          op.getLoc(),
          RankedTensorType::get(cast<RankedTensorType>(v.getType()).getShape(),
                                op.getType().getElementType()),
          v));
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, newvals, concat.getDimension());
    return success();
  }
};

struct ConvertConvertFloat final
    : CheckedOpRewritePattern<stablehlo::ConvertOp, ConvertConvertFloat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) const {
    auto conv0 = op.getOperand().getDefiningOp<stablehlo::ConvertOp>();
    if (!conv0)
      return failure();

    auto prev = conv0.getOperand();
    if (isa<FloatType>(prev.getType().getElementType()) &&
        isa<FloatType>(op.getType().getElementType()) &&
        isa<FloatType>(conv0.getType().getElementType())) {
      if (prev.getType() == op.getType()) {
        rewriter.replaceOp(op, prev);
        return success();
      }
      rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, op.getType(), prev);
      return success();
    }
    return failure();
  }
};

struct ReduceConcat final
    : CheckedOpRewritePattern<stablehlo::ReduceOp, ReduceConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1)
      return failure();

    auto concat = op.getInputs()[0].getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    auto dim = concat.getDimension();

    if (!llvm::is_contained(op.getDimensions(), dim))
      return failure();

    if (!isEligibleForCompactPrint(op))
      return failure();

    Value prev = op.getInitValues()[0];

    Operation &innerOp = op.getBody().front().front();

    Value identity = nullptr;
    if (isa<stablehlo::AddOp>(&innerOp)) {
      identity = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), prev.getType(),
          cast<ElementsAttr>(makeAttr(prev.getType(), 0)));
    } else if (isa<stablehlo::MaxOp>(&innerOp) ||
               isa<stablehlo::MinOp>(&innerOp))
      identity = prev;
    else {
      return failure();
    }

    if (prev.getType() != op.getResultTypes()[0]) {
      prev = rewriter.create<stablehlo::BroadcastInDimOp>(
          op.getLoc(), op.getResultTypes()[0], prev, ArrayRef<int64_t>());
    }

    for (auto v : concat.getOperands()) {
      IRMapping map;
      map.map(op.getInitValues()[0], identity);
      map.map(op.getInputs()[0], v);
      auto next = rewriter.clone(*op, map)->getResult(0);
      map.map(innerOp.getOperand(0), prev);
      map.map(innerOp.getOperand(1), next);
      Value vals[] = {prev, next};
      prev =
          rewriter
              .create(innerOp.getLoc(), innerOp.getName().getIdentifier(), vals,
                      TypeRange(prev.getType()), innerOp.getAttrs(), {}, {})
              ->getResult(0);
    }

    assert(op.getResultTypes()[0] == prev.getType());
    rewriter.replaceOp(op, prev);
    return success();
  }
};

struct FullReduceReshapeOrTranspose final
    : CheckedOpRewritePattern<stablehlo::ReduceOp,
                              FullReduceReshapeOrTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1)
      return failure();

    auto inpTy = cast<RankedTensorType>(op.getInputs()[0].getType());
    if (op.getDimensions().size() != inpTy.getShape().size())
      return failure();

    RankedTensorType changeType = nullptr;
    SmallVector<Operation *> reshapeOrTransposes;
    llvm::MapVector<Operation *, int> toclone;
    {
      SmallVector<Value> todo = {op.getInputs()[0]};
      while (todo.size()) {
        auto cur = todo.pop_back_val();
        auto curOp = cur.getDefiningOp();
        if (!curOp)
          return failure();
        if (auto rs = dyn_cast<stablehlo::ReshapeOp>(curOp)) {
          if (changeType != nullptr) {
            if (rs.getOperand().getType() != changeType)
              return failure();
          } else {
            changeType = rs.getOperand().getType();
          }
          reshapeOrTransposes.push_back(rs);
          continue;
        }
        if (auto rs = dyn_cast<stablehlo::TransposeOp>(curOp)) {
          if (changeType != nullptr) {
            if (rs.getOperand().getType() != changeType)
              return failure();
          } else {
            changeType = rs.getOperand().getType();
          }
          reshapeOrTransposes.push_back(rs);
          continue;
        }
        if (!curOp->hasTrait<mlir::OpTrait::Elementwise>())
          return failure();
        if (!isMemoryEffectFree(curOp))
          return failure();
        for (auto op : curOp->getOperands())
          todo.push_back(op);
        toclone[curOp] = curOp->getNumOperands();
      }
    }

    IRMapping map;
    SmallVector<Operation *> todo;
    for (auto reshape : reshapeOrTransposes) {
      map.map(reshape->getResult(0), reshape->getOperand(0));
      for (auto u : reshape->getResult(0).getUsers()) {
        if (toclone.contains(u)) {
          toclone[u]--;
          if (toclone[u] == 0) {
            todo.push_back(u);
            toclone.erase(u);
          }
        }
      }
    }
    for (auto pair : toclone) {
      for (auto u : pair.first->getResult(0).getUsers()) {
        if (u == op)
          continue;
        if (llvm::is_contained(reshapeOrTransposes, u))
          continue;
        if (toclone.contains(u))
          continue;
        return failure();
      }
    }

    for (auto cur : todo) {
      for (auto curOp : cur->getOperands()) {
        if (!map.contains(curOp))
          return failure();
      }
    }

    while (todo.size()) {
      auto cur = todo.pop_back_val();

      SmallVector<Value> vals;
      for (auto op : cur->getOperands())
        vals.push_back(map.lookup(op));

      auto changeType2 = RankedTensorType::get(
          changeType.getShape(),
          cast<RankedTensorType>(cur->getResult(0).getType()).getElementType());
      auto res =
          rewriter.create(cur->getLoc(), cur->getName().getIdentifier(), vals,
                          TypeRange(changeType2), cur->getAttrs(), {}, {});

      map.map(cur->getResult(0), res->getResult(0));

      for (auto u : cur->getResult(0).getUsers()) {
        if (toclone.contains(u)) {
          toclone[u]--;
          if (toclone[u] == 0) {
            todo.push_back(u);
            toclone.erase(u);
          }
        }
      }
    }

    SmallVector<int64_t> newReduceDimensions;
    for (size_t i = 0, end = changeType.getShape().size(); i < end; i++)
      newReduceDimensions.push_back(i);

    auto newReduction = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), op->getResultTypes(), map.lookup(op.getInputs()[0]),
        op.getInitValues(), newReduceDimensions);
    newReduction.getRegion().takeBody(op.getRegion());
    rewriter.replaceOp(op, newReduction);
    return success();
  }
};

LogicalResult sliceConcatHelper(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter,
                                ArrayRef<int64_t> starts,
                                ArrayRef<int64_t> limits,
                                ArrayRef<int64_t> strides,
                                SmallVectorImpl<Value> &postConcat) {
  auto dim = concat.getDimension();

  if (strides[dim] != 1)
    return failure();

  assert(postConcat.size() == 0);
  size_t curdim = 0;
  for (auto v : concat.getInputs()) {
    auto ty = cast<RankedTensorType>(v.getType());
    auto nextdim = ty.getShape()[dim];
    if (starts[dim] >= curdim + nextdim) {
      curdim += nextdim;
      continue;
    }
    if (limits[dim] <= curdim) {
      curdim += nextdim;
      continue;
    }
    SmallVector<int64_t> nstart(starts.begin(), starts.end());
    SmallVector<int64_t> nend(limits.begin(), limits.end());
    nstart[dim] -= curdim;
    if (nstart[dim] < 0)
      nstart[dim] = 0;
    nend[dim] -= curdim;
    if (nend[dim] > nextdim)
      nend[dim] = nextdim;
    auto subslice = rewriter.create<stablehlo::SliceOp>(concat.getLoc(), v,
                                                        nstart, nend, strides);
    postConcat.push_back(subslice);
    curdim += nextdim;
  }
  return success();
}

bool canMergeSlicesAlongAxis(int dimension, ArrayRef<int64_t> sliceStarts,
                             ArrayRef<int64_t> otherSliceStarts,
                             ArrayRef<int64_t> sliceLimits,
                             ArrayRef<int64_t> otherSliceLimits,
                             ArrayRef<int64_t> sliceStrides,
                             ArrayRef<int64_t> otherSliceStrides) {
  bool canMerge = true;

  for (int d = 0, ndims = sliceStarts.size(); d < ndims; ++d) {
    if (d == dimension) {
      canMerge &= sliceLimits[d] == otherSliceStarts[d] &&
                  sliceStrides[d] == otherSliceStrides[d];
    } else {
      canMerge &= sliceStarts[d] == otherSliceStarts[d] &&
                  sliceLimits[d] == otherSliceLimits[d] &&
                  sliceStrides[d] == otherSliceStrides[d];
    }
  }
  return canMerge;
}

bool canMergeSlicesAlongAxis(int dimension, stablehlo::SliceOp slice,
                             stablehlo::SliceOp otherSlice) {
  if (otherSlice.getOperand() != slice.getOperand())
    return false;

  // Check that both slices are contiguous only in dim
  ArrayRef<int64_t> sliceStarts = slice.getStartIndices(),
                    otherSliceStarts = otherSlice.getStartIndices(),
                    sliceLimits = slice.getLimitIndices(),
                    otherSliceLimits = otherSlice.getLimitIndices(),
                    sliceStrides = slice.getStrides(),
                    otherSliceStrides = otherSlice.getStrides();

  return canMergeSlicesAlongAxis(dimension, sliceStarts, otherSliceStarts,
                                 sliceLimits, otherSliceLimits, sliceStrides,
                                 otherSliceStrides);
}

struct ConcatSlice final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    auto dim = op.getDimension();

    SmallVector<Value> newOperands;

    bool changed = false;

    for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
      auto operand = op->getOperand(i);
      auto slice = operand.getDefiningOp<stablehlo::SliceOp>();

      if (!slice) {
        newOperands.push_back(operand);
        continue;
      }

      while (i + 1 < e) {
        if (auto otherSlice =
                op->getOperand(i + 1).getDefiningOp<stablehlo::SliceOp>()) {
          if (canMergeSlicesAlongAxis(op.getDimension(), slice, otherSlice)) {
            slice = rewriter.create<stablehlo::SliceOp>(
                slice->getLoc(), slice.getOperand(), slice.getStartIndices(),
                otherSlice.getLimitIndices(), slice.getStrides());
            changed = true;
            i++;
            continue;
          } else
            break;
        }
        if (auto otherWrap =
                op->getOperand(i + 1).getDefiningOp<enzymexla::WrapOp>()) {
          auto wrapSlice =
              otherWrap.getOperand().getDefiningOp<stablehlo::SliceOp>();
          if (wrapSlice && wrapSlice.getOperand() == slice.getOperand() &&
              otherWrap.getLhs() != 0) {
            SmallVector<int64_t> wrapStarts =
                llvm::to_vector(wrapSlice.getStartIndices());
            SmallVector<int64_t> wrapLimits =
                llvm::to_vector(wrapSlice.getLimitIndices());
            if (wrapSlice.getStrides()[dim] == 1) {
              wrapStarts[dim] = wrapLimits[dim] - otherWrap.getLhs();
            }
            if (canMergeSlicesAlongAxis(
                    op.getDimension(), slice.getStartIndices(), wrapStarts,
                    slice.getLimitIndices(), wrapLimits, slice.getStrides(),
                    wrapSlice.getStrides())) {

              changed = true;
              auto c2 = lowerWrap(otherWrap, rewriter, /*replace*/ false);
              auto newSlice = rewriter.create<stablehlo::SliceOp>(
                  slice->getLoc(), slice.getOperand(), slice.getStartIndices(),
                  wrapLimits, slice.getStrides());
              newOperands.push_back(newSlice);
              for (int i = 1; i < c2.getOperands().size(); i++) {
                newOperands.push_back(c2.getOperands()[i]);
              }
              i++;
              slice = nullptr;
              break;
            } else
              break;
          }
        }
        break;
      }

      if (slice)
        newOperands.push_back(slice.getResult());
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, newOperands, dim);
    return success();
  }
};

bool canMergePadsAlongAxis(int dimension, stablehlo::PadOp pad,
                           stablehlo::PadOp otherPad) {
  if (otherPad.getPaddingValue() != pad.getPaddingValue())
    return false;

  for (int d = 0, ndims = pad.getType().getShape().size(); d < ndims; ++d) {
    if (d == dimension) {
      if (pad.getInteriorPadding()[d] != 0) {
        return false;
      }
      if (otherPad.getInteriorPadding()[d] != 0) {
        return false;
      }
      if (pad.getEdgePaddingHigh()[d] != 0) {
        return false;
      }
      if (otherPad.getEdgePaddingLow()[d] != 0) {
        return false;
      }
    } else {
      if (pad.getInteriorPadding()[d] != otherPad.getInteriorPadding()[d]) {
        return false;
      }
      if (pad.getEdgePaddingHigh()[d] != otherPad.getEdgePaddingHigh()[d]) {
        return false;
      }
      if (pad.getEdgePaddingLow()[d] != otherPad.getEdgePaddingLow()[d]) {
        return false;
      }
    }
  }
  return true;
}

struct ConcatMultiPad final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatMultiPad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    auto dim = op.getDimension();

    SmallVector<Value> newOperands;

    for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
      auto operand = op->getOperand(i);
      auto pad = operand.getDefiningOp<stablehlo::PadOp>();

      if (!pad) {
        newOperands.push_back(operand);
        continue;
      }

      stablehlo::PadOp otherPadOp;
      while (i + 1 < e &&
             (otherPadOp =
                  op->getOperand(i + 1).getDefiningOp<stablehlo::PadOp>())) {
        if (canMergePadsAlongAxis(op.getDimension(), pad, otherPadOp)) {
          Value padops[] = {pad.getOperand(), otherPadOp.getOperand()};
          auto subConcat = rewriter.create<stablehlo::ConcatenateOp>(
              op.getLoc(), padops, op.getDimension());
          pad = rewriter.create<stablehlo::PadOp>(
              pad->getLoc(), subConcat, pad.getPaddingValue(),
              pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
              pad.getInteriorPadding());
          i++;
        } else
          break;
      }

      newOperands.push_back(pad.getResult());
    }

    if (newOperands.size() == op->getNumOperands())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, newOperands, dim);
    return success();
  }
};

bool canMergeWrapsAlongAxis(int dimension, enzymexla::WrapOp wrap,
                            enzymexla::WrapOp otherWrap) {
  if (wrap.getDimension() != otherWrap.getDimension())
    return false;

  if (dimension == wrap.getDimension())
    return false;

  if (wrap.getLhs() != otherWrap.getLhs())
    return false;

  if (wrap.getRhs() != otherWrap.getRhs())
    return false;

  return true;
}

struct ConcatWrap final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    auto dim = op.getDimension();

    SmallVector<Value> newOperands;

    for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
      auto operand = op->getOperand(i);
      auto wrap = operand.getDefiningOp<enzymexla::WrapOp>();

      if (!wrap) {
        newOperands.push_back(operand);
        continue;
      }

      enzymexla::WrapOp otherWrap;
      while (i + 1 < e &&
             (otherWrap =
                  op->getOperand(i + 1).getDefiningOp<enzymexla::WrapOp>())) {
        if (canMergeWrapsAlongAxis(op.getDimension(), wrap, otherWrap)) {
          Value padops[] = {wrap.getOperand(), otherWrap.getOperand()};
          auto subConcat = rewriter.create<stablehlo::ConcatenateOp>(
              op.getLoc(), padops, op.getDimension());
          wrap = rewriter.create<enzymexla::WrapOp>(
              wrap->getLoc(), subConcat, wrap.getLhs(), wrap.getRhs(),
              wrap.getDimension());
          i++;
        } else
          break;
      }

      newOperands.push_back(wrap.getResult());
    }

    if (newOperands.size() == op->getNumOperands())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, newOperands, dim);
    return success();
  }
};

struct SliceConcat final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto concat = op.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    auto dim = concat.getDimension();

    SmallVector<Value> postConcat;
    if (!sliceConcatHelper(concat, rewriter, op.getStartIndices(),
                           op.getLimitIndices(), op.getStrides(), postConcat)
             .succeeded())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, postConcat, dim);
    return success();
  }
};

DenseElementsAttr fromTensor(stablehlo::Tensor tensor) {
  return stablehlo::makeDenseElementsAttr(tensor);
}

/*
%22 = stablehlo.dot_general %21, %16, contracting_dims = [1] x [0], precision
= [DEFAULT, DEFAULT] : (tensor<288x288xf32>, tensor<288xf32>) ->
tensor<288xf32> %27 = stablehlo.reshape %22 : (tensor<288xf32>) ->
tensor<144x2xf32> %28 = stablehlo.dot_general %6, %27, batching_dims = [0] x
[0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] :
(tensor<144x2x2xf32>, tensor<144x2xf32>) -> tensor<144x2xf32>

should become

%a21 = stablehlo.reshape %21 : (tensor<288xf32>) -> tensor<144x2xf32>

%22 = stablehlo.dot_general %a21, %16, batching_dims = [1] x [],
contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] :
(tensor<144x2x288xf32>, tensor<288xf32>) -> tensor<2x144xf32>

%28 = stablehlo.dot_general %6, %22, batching_dims = [0] x [1],
contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] :
(tensor<144x2x2xf32>, tensor<144x2xf32>) -> tensor<144x2xf32>

TODO
*/

struct DotReshapeDot final
    : CheckedOpRewritePattern<stablehlo::DotGeneralOp, DotReshapeDot> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DotGeneralOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    return failure();
  }
};

struct PadSimplify final
    : CheckedOpRewritePattern<stablehlo::PadOp, PadSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;
  size_t max_constant_expansion;

  PadSimplify(size_t max_constant_expansion, MLIRContext *context,
              PatternBenefit benefit = 1,
              ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::PadOp op,
                                    PatternRewriter &rewriter) const {

    if (matchPattern(op.getOperand(), m_AnyZeroFloat())) {
      if (matchPattern(op.getPaddingValue(), m_AnyZeroFloat())) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 0)));
        return success();
      }
    }

    {
      DenseElementsAttr inp;
      matchPattern(op.getOperand(), m_Constant(&inp));
      DenseElementsAttr pv;
      matchPattern(op.getPaddingValue(), m_Constant(&pv));
      if (inp && pv) {

        if (inp.isSplat() && pv.isSplat() &&
            inp.getSplatValue<Attribute>() == pv.getSplatValue<Attribute>()) {
          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
              op, op.getType(), inp.resizeSplat(op.getType()));
          return success();
        }

        size_t size = 1;
        for (auto sz : op.getType().getShape())
          size *= sz;
        if (size < max_constant_expansion) {
          auto ten = stablehlo::constantOp(inp);
          auto out = fromTensor(stablehlo::padOp(
              stablehlo::constantOp(inp), stablehlo::constantOp(pv),
              stablehlo::Sizes(op.getEdgePaddingLow()),
              stablehlo::Sizes(op.getInteriorPadding()), op.getType()));

          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                             out);
          return success();
        }
      }
    }

    for (auto &&[low, high, inner] :
         llvm::zip(op.getEdgePaddingLow(), op.getEdgePaddingHigh(),
                   op.getInteriorPadding())) {
      if (low != 0)
        return failure();
      if (high != 0)
        return failure();
      if (inner != 0)
        return failure();
    }
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct RotatePad final
    : CheckedOpRewritePattern<enzymexla::RotateOp, RotatePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::RotateOp rotate,
                                    PatternRewriter &rewriter) const {

    auto pad = rotate.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (pad.getEdgePaddingLow()[rotate.getDimension()] != 0)
      return failure();
    if (pad.getEdgePaddingHigh()[rotate.getDimension()] != 0)
      return failure();
    if (pad.getInteriorPadding()[rotate.getDimension()] != 0)
      return failure();

    auto newRotate = rewriter.create<enzymexla::RotateOp>(
        rotate.getLoc(), pad.getOperand(), rotate.getAmount(),
        rotate.getDimension());

    auto newPad = rewriter.create<stablehlo::PadOp>(
        pad.getLoc(), newRotate, pad.getPaddingValue(), pad.getEdgePaddingLow(),
        pad.getEdgePaddingHigh(), pad.getInteriorPadding());

    rewriter.replaceOp(rotate, newPad);
    return success();
  }
};

struct ShiftRightLogicalSimplify final
    : CheckedOpRewritePattern<stablehlo::ShiftRightLogicalOp,
                              ShiftRightLogicalSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ShiftRightLogicalOp op,
                                    PatternRewriter &rewriter) const {

    DenseElementsAttr lhs;
    matchPattern(op.getLhs(), m_Constant(&lhs));
    DenseElementsAttr rhs;
    matchPattern(op.getRhs(), m_Constant(&rhs));
    if (lhs && rhs) {
      auto out = fromTensor(stablehlo::shiftRightLogicalOp(
          stablehlo::constantOp(lhs), stablehlo::constantOp(rhs),
          op.getType()));

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct WhileDeadResults final
    : CheckedOpRewritePattern<stablehlo::WhileOp, WhileDeadResults> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  void replaceTerminator(PatternRewriter &rewriter, Region &region,
                         ArrayRef<int64_t> deadResults) const {
    Operation *terminator = region.front().getTerminator();
    SmallVector<Value> terminatorOperands;
    for (auto &&[i, operand] : llvm::enumerate(terminator->getOperands())) {
      if (!llvm::is_contained(deadResults, i))
        terminatorOperands.push_back(operand);
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(
        terminator, TypeRange(), terminatorOperands, terminator->getAttrs());
  }

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp op,
                                    PatternRewriter &rewriter) const {
    bool isTotalPure = isPure(op);
    std::map<int64_t, int> forwardIsPure;
    std::map<int64_t, std::set<int64_t>> backwardsUsesOfArguments;
    SmallVector<size_t> emptyResults;
    size_t NumResults = op.getNumResults();
    for (OpResult result : op.getResults()) {
      if (result.use_empty())
        emptyResults.push_back(result.getResultNumber());
    }
    if (emptyResults.size() == 0)
      return failure();

    DenseMap<Value, llvm::SmallPtrSet<size_t, 3> *> users(
        emptyResults.size() + op.getCond().front().getOperations().size());
    llvm::SmallVector<std::unique_ptr<llvm::SmallPtrSet<size_t, 3>>, 3>
        condCache;
    condCache.reserve(emptyResults.size());
    for (auto ores : emptyResults) {
      auto ba = op.getCond().front().getArgument(ores);
      auto up = std::make_unique<llvm::SmallPtrSet<size_t, 3>>();
      up->insert(ores);
      users[ba] = up.get();
      condCache.emplace_back(std::move(up));
    }

    llvm::SmallPtrSet<size_t, 3> nonPure;

    SmallVector<
        std::pair<Operation *, std::unique_ptr<llvm::SmallPtrSet<size_t, 3>>>>
        opUsers;
    for (auto &sop : op.getCond().front().without_terminator()) {
      auto setP = std::make_unique<llvm::SmallPtrSet<size_t, 3>>();
      auto &set = *setP;
      for (auto operand : sop.getOperands()) {
        auto found = users.find(operand);
        if (found != users.end()) {
          set.insert(found->second->begin(), found->second->end());
        }
      }
      if (!(sop.getNumRegions() == 0 ||
            sop.hasTrait<OpTrait::IsIsolatedFromAbove>())) {
        SmallVector<Operation *> todo;
        for (auto &reg : sop.getRegions()) {
          for (auto &blk : reg) {
            for (auto &mop : blk) {
              todo.push_back(&mop);
            }
          }
        }
        while (todo.size()) {
          auto cur = todo.pop_back_val();
          for (auto operand : cur->getOperands()) {
            auto found = users.find(operand);
            if (found != users.end()) {
              set.insert(found->second->begin(), found->second->end());
            }
          }
          if (!(cur->getNumRegions() == 0 ||
                cur->hasTrait<OpTrait::IsIsolatedFromAbove>())) {
            for (auto &reg : cur->getRegions()) {
              for (auto &blk : reg) {
                for (auto &mop : blk) {
                  todo.push_back(&mop);
                }
              }
            }
          }
        }
      }
      if (set.size() == 0)
        continue;

      if (!isTotalPure && !mlir::isPure(&sop)) {
        for (auto arg : set)
          nonPure.insert(arg);
      }
      for (auto res : sop.getResults())
        if (!res.use_empty())
          users.try_emplace(res, &set);
      opUsers.emplace_back(&sop, std::move(setP));
    }
    llvm::SmallPtrSet<size_t, 3> terminatorUsers;
    for (auto v : op.getCond().front().getTerminator()->getOperands()) {
      auto found = users.find(v);
      if (found != users.end()) {
        terminatorUsers.insert(found->second->begin(), found->second->end());
      }
    }

    SmallVector<size_t> emptyNonPure;
    for (auto ores : emptyResults) {
      if (nonPure.count(ores)) {
        continue;
      }
      if (terminatorUsers.count(ores)) {
        continue;
      }
      emptyNonPure.push_back(ores);
    }
    if (emptyNonPure.size() == 0) {
      return failure();
    }

    DenseMap<Value, llvm::SmallPtrSet<size_t, 3> *> usersBody(
        emptyNonPure.size() + op.getBody().front().getOperations().size());
    SmallVector<
        std::pair<Operation *, std::unique_ptr<llvm::SmallPtrSet<size_t, 3>>>>
        opUsersBody;
    terminatorUsers.clear();
    nonPure.clear();

    llvm::SmallVector<std::unique_ptr<llvm::SmallPtrSet<size_t, 3>>, 3>
        bodyCache;
    bodyCache.reserve(emptyNonPure.size());
    for (auto ores : emptyNonPure) {
      auto ba = op.getBody().front().getArgument(ores);
      auto up = std::make_unique<llvm::SmallPtrSet<size_t, 3>>();
      up->insert(ores);
      usersBody[ba] = up.get();
      bodyCache.emplace_back(std::move(up));
    }

    llvm::SmallSet<size_t, 3> nonPure2;

    for (auto &sop : op.getBody().front().without_terminator()) {
      auto setP = std::make_unique<llvm::SmallPtrSet<size_t, 3>>();
      auto &set = *setP;
      for (auto operand : sop.getOperands()) {
        auto found = usersBody.find(operand);
        if (found != usersBody.end()) {
          set.insert(found->second->begin(), found->second->end());
        }
      }
      if (!(sop.getNumRegions() == 0 ||
            sop.hasTrait<OpTrait::IsIsolatedFromAbove>())) {
        SmallVector<Operation *> todo;
        for (auto &reg : sop.getRegions()) {
          for (auto &blk : reg) {
            for (auto &mop : blk) {
              todo.push_back(&mop);
            }
          }
        }
        while (todo.size()) {
          auto cur = todo.pop_back_val();
          for (auto operand : cur->getOperands()) {
            auto found = usersBody.find(operand);
            if (found != usersBody.end()) {
              set.insert(found->second->begin(), found->second->end());
            }
          }
          if (!(cur->getNumRegions() == 0 ||
                cur->hasTrait<OpTrait::IsIsolatedFromAbove>())) {
            for (auto &reg : cur->getRegions()) {
              for (auto &blk : reg) {
                for (auto &mop : blk) {
                  todo.push_back(&mop);
                }
              }
            }
          }
        }
      }
      if (set.size() == 0)
        continue;

      if (!isTotalPure && !mlir::isPure(&sop)) {
        for (auto arg : set)
          nonPure2.insert(arg);
      }
      for (auto res : sop.getResults())
        if (!res.use_empty())
          usersBody.try_emplace(res, &set);
      opUsersBody.emplace_back(&sop, std::move(setP));
    }

    llvm::SmallPtrSet<size_t, 3> emptyNonPure2;
    if (isTotalPure)
      emptyNonPure2.insert(emptyNonPure.begin(), emptyNonPure.end());
    else {
      for (auto ores : emptyNonPure) {
        if (nonPure2.count(ores)) {
          continue;
        }
        emptyNonPure2.insert(ores);
      }
      if (emptyNonPure2.size() == 0) {
        return failure();
      }
    }

    llvm::BitVector removedResults(NumResults, true);
    llvm::BitVector seen(NumResults, false);
    SmallVector<size_t> todo;
    todo.reserve(NumResults);
    for (size_t residx = 0; residx < NumResults; residx++) {
      if (!emptyNonPure2.count(residx)) {
        todo.push_back(residx);
        seen.set(residx);
      }
    }

    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (!removedResults[cur])
        continue;
      removedResults.reset(cur);
      auto v = op.getBody().front().getTerminator()->getOperands()[cur];
      auto found = usersBody.find(v);
      if (found != usersBody.end()) {
        if (isTotalPure) {
          for (auto arg : *found->second) {
            if (!seen.test(arg)) {
              todo.push_back(arg);
              seen.set(arg);
            }
          }
        } else {
          for (auto arg : *found->second) {
            if (emptyNonPure2.contains(arg)) {
              if (!seen.test(arg)) {
                todo.push_back(arg);
                seen.set(arg);
              }
            }
          }
        }
      }
    }
    /*
    llvm::BitVector todo(NumResults, false);
    int start = NumResults;
    for (size_t residx = 0; residx < NumResults; residx++) {
      if (!emptyNonPure2.count(residx)) {
        todo.set(residx);
        if (residx < start) start = residx;
      }
    }

    while (true) {
      auto cur = todo.find_first_in(start, NumResults);
      if (cur == -1) break;
      start = cur+1;
      todo.reset(cur);
      if (!removedResults.test(cur))
        continue;
      removedResults.reset(cur);
      auto v = op.getBody().front().getTerminator()->getOperands()[cur];
      auto found = usersBody.find(v);
      if (found != usersBody.end()) {
        if (isTotalPure) {
          for (auto arg : found->second) {
            if (!removedResults.test(arg))
              continue;
            todo.set(arg);
            if (arg < start) start = arg;
          }
        } else {
          for (auto arg : found->second) {
            if (emptyNonPure2.contains(arg)) {
              if (!removedResults.test(arg))
                continue;
              todo.set(arg);
              if (arg < start) start = arg;
            }
          }
        }
      }
    }
    */
    /*
    SmallVector<size_t> todo;
    for (size_t residx = 0; residx < NumResults; residx++) {
      if (!emptyNonPure2.count(residx))
        todo.push_back(residx);
    }

    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (!removedResults[cur])
        continue;
      removedResults.reset(cur);
      auto v = op.getBody().front().getTerminator()->getOperands()[cur];
      auto found = usersBody.find(v);
      if (found != usersBody.end()) {
        if (isTotalPure)
          todo.append(found->second.begin(), found->second.end());
        else
          for (auto arg : found->second) {
            if (emptyNonPure2.contains(arg))
              todo.push_back(arg);
        }
      }
    }
    */

    if (!removedResults.any())
      return failure();

    SmallVector<int64_t> deadResults;
    for (auto cur = removedResults.find_first(); cur != -1;
         cur = removedResults.find_next(cur)) {
      deadResults.push_back(cur);
    }

    replaceTerminator(rewriter, op.getBody(), deadResults);

    for (auto &cop : llvm::reverse(opUsers)) {
      bool hasDead = false;
      for (int64_t i : deadResults) {
        if (cop.second->count(i)) {
          hasDead = true;
          break;
        }
      }
      if (hasDead)
        rewriter.eraseOp(cop.first);
    }
    for (auto &cop : llvm::reverse(opUsersBody)) {
      bool hasDead = false;
      for (int64_t i : deadResults) {
        if (cop.second->count(i)) {
          hasDead = true;
          break;
        }
      }
      if (hasDead)
        rewriter.eraseOp(cop.first);
    }

    SmallVector<Value> operands;
    SmallVector<Type> resultTypes;
    SmallVector<Location> condBlockArgLocs, bodyBlockArgsLocs;
    for (auto &&[i, operand, resultType] :
         llvm::enumerate(op->getOperands(), op.getResultTypes())) {
      if (llvm::is_contained(deadResults, i))
        continue;

      operands.push_back(operand);
      resultTypes.push_back(resultType);
      condBlockArgLocs.push_back(op.getCond().getArgument(i).getLoc());
      bodyBlockArgsLocs.push_back(op.getBody().getArgument(i).getLoc());
    }

    auto updated = rewriter.create<stablehlo::WhileOp>(
        op->getLoc(), resultTypes, operands, op->getAttrs());
    SmallVector<Value> resultReplacements;
    for (int64_t old = 0, upd = 0, end = op->getNumResults(); old < end;
         ++old) {
      if (llvm::is_contained(deadResults, old)) {
        resultReplacements.push_back(nullptr);
        continue;
      }
      resultReplacements.push_back(updated->getResult(upd));
      ++upd;
    }

    for (int64_t i : llvm::reverse(deadResults))
      op.getCond().eraseArgument(i);
    rewriter.inlineRegionBefore(op.getCond(), updated.getCond(),
                                updated.getCond().begin());

    for (int64_t i : llvm::reverse(deadResults)) {
      op.getBody().eraseArgument(i);
    }
    rewriter.inlineRegionBefore(op.getBody(), updated.getBody(),
                                updated.getBody().begin());

    rewriter.replaceOp(op, resultReplacements);
    return success();
  }
};

struct NegativePadToSlice final
    : CheckedOpRewritePattern<stablehlo::PadOp, NegativePadToSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::PadOp op,
                                    PatternRewriter &rewriter) const {
    SmallVector<int64_t> starts;
    SmallVector<int64_t> limits;
    SmallVector<int64_t> strides;

    bool negative = false;
    for (auto &&[low, high, inner, dim] : llvm::zip(
             op.getEdgePaddingLow(), op.getEdgePaddingHigh(),
             op.getInteriorPadding(), op.getOperand().getType().getShape())) {
      if (low > 0)
        return failure();
      if (high > 0)
        return failure();
      if (inner != 0)
        return failure();
      if (low < 0 || high < 0)
        negative = true;

      starts.push_back(-low);
      limits.push_back(dim + high);
      strides.push_back(1);
    }
    if (!negative)
      return failure();
    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(op, op.getOperand(), starts,
                                                    limits, strides);
    return success();
  }
};

/*

    %1192 = stablehlo.pad %1189, %cst_0, low = [0], high = [1], interior =
    [0] :
   (tensor<1xf32>, tensor<f32>) -> tensor<2xf32> %1193 = arith.addf %1191,
   %1192 : tensor<2xf32>

*/
template <typename T>
struct BinopPadToConcat final
    : CheckedOpRewritePattern<T, BinopPadToConcat<T>> {
  using CheckedOpRewritePattern<T,
                                BinopPadToConcat<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    for (int i = 0; i < 2; i++) {
      if (auto lhs =
              op->getOperand(i).template getDefiningOp<stablehlo::PadOp>()) {
        if (anyPadSizesNegative(lhs))
          continue;
        auto rhs = op->getOperand(1 - i);

        bool match = false;
        if (isa<stablehlo::AddOp>(op)) {
          match = matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat());
        } else if (isa<stablehlo::MulOp>(op)) {
          match = matchPattern(lhs.getPaddingValue(), m_OneFloat()) ||
                  matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat());
        }
        if (!match) {
          SmallVector<Operation *> ops = {op};
          bool legal = true;
          while (!ops.empty()) {
            auto cur = ops.pop_back_val();
            if (isa<stablehlo::SliceOp>(cur))
              continue;
            if (isa<stablehlo::AddOp, stablehlo::MulOp>(cur)) {
              for (auto u : cur->getResult(0).getUsers()) {
                ops.push_back(u);
              }
              continue;
            }
            legal = false;
            break;
          }
          if (!legal)
            return failure();
        }

        bool legal = true;
        for (auto step : lhs.getInteriorPadding()) {
          if (step != 0) {
            legal = true;
            break;
          }
        }
        if (!legal)
          continue;

        ssize_t padidx = -1;

        SmallVector<size_t> idxs;
        for (auto &&[low, high, dim] :
             llvm::zip(lhs.getEdgePaddingLow(), lhs.getEdgePaddingHigh(),
                       type.getShape())) {
          padidx++;
          if (low == 0 && high == 0)
            continue;
          if (low < 0 || high < 0)
            return failure();
          idxs.push_back(padidx);
        }

        if (idxs.size() == 1) {
          auto idx = idxs[0];

          SmallVector<int64_t> strides(type.getShape().size(), 1);
          SmallVector<int64_t> starts(type.getShape().size(), 0);
          SmallVector<int64_t> limits(type.getShape().begin(),
                                      type.getShape().end());

          SmallVector<Value, 1> vals;

          if (lhs.getEdgePaddingLow()[idx] != 0) {
            starts[idx] = 0;
            limits[idx] = lhs.getEdgePaddingLow()[idx];
            Value prevSlice = rewriter.create<stablehlo::SliceOp>(
                op.getLoc(), rhs, starts, limits, strides);

            if (isa<stablehlo::AddOp>(op) &&
                matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat())) {
              // If adding we're adding 0, no need to do extra work
            } else if (isa<stablehlo::MulOp>(op) &&
                       matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat())) {
              // If multiplying by 0, broadcast the zero
              prevSlice = rewriter.create<stablehlo::BroadcastInDimOp>(
                  op.getLoc(), prevSlice.getType(), lhs.getPaddingValue(),
                  ArrayRef<int64_t>());
            } else if (isa<stablehlo::MulOp>(op) &&
                       matchPattern(lhs.getPaddingValue(), m_OneFloat())) {
              // If multiplying by 1, no need to do extra work
            } else
              prevSlice = rewriter.create<T>(
                  op.getLoc(), prevSlice,
                  rewriter.create<stablehlo::BroadcastInDimOp>(
                      op.getLoc(), prevSlice.getType(), lhs.getPaddingValue(),
                      ArrayRef<int64_t>()));
            vals.push_back(prevSlice);
          }

          starts[idx] = lhs.getEdgePaddingLow()[idx];
          limits[idx] = type.getShape()[idx] - lhs.getEdgePaddingHigh()[idx];

          auto midSlice = rewriter.create<stablehlo::SliceOp>(
              op.getLoc(), rhs, starts, limits, strides);
          auto mid =
              rewriter.create<T>(op.getLoc(), midSlice, lhs.getOperand());
          vals.push_back(mid);

          if (lhs.getEdgePaddingHigh()[idx] != 0) {
            starts[idx] = type.getShape()[idx] - lhs.getEdgePaddingHigh()[idx];
            limits[idx] = type.getShape()[idx];
            Value postSlice = rewriter.create<stablehlo::SliceOp>(
                op.getLoc(), rhs, starts, limits, strides);

            if (isa<stablehlo::AddOp>(op) &&
                matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat())) {
              // If adding we're adding 0, no need to do extra work
            } else if (isa<stablehlo::MulOp>(op) &&
                       matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat())) {
              // If multiplying by 0, broadcast the zero
              postSlice = rewriter.create<stablehlo::BroadcastInDimOp>(
                  op.getLoc(), postSlice.getType(), lhs.getPaddingValue(),
                  ArrayRef<int64_t>());
            } else if (isa<stablehlo::MulOp>(op) &&
                       matchPattern(lhs.getPaddingValue(), m_OneFloat())) {
              // If multiplying by 1, no need to do extra work
            } else
              postSlice = rewriter.create<T>(
                  op.getLoc(), postSlice,
                  rewriter.create<stablehlo::BroadcastInDimOp>(
                      op.getLoc(), postSlice.getType(), lhs.getPaddingValue(),
                      ArrayRef<int64_t>()));
            vals.push_back(postSlice);
          }

          rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, vals, idx);
          return success();
        }
      }
    }

    return failure();
  }
};

struct ReshapeIota final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeIota> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto iota = op.getOperand().getDefiningOp<stablehlo::IotaOp>();
    if (!iota)
      return failure();

    size_t curiotaidx = 0;
    size_t iotadim = 0;
    for (auto en : llvm::enumerate(op.getType().getShape())) {
      if (en.value() == 1)
        continue;

      if (curiotaidx == iota.getType().getShape().size())
        return failure();
      auto ival = iota.getType().getShape()[curiotaidx];
      while (ival == 1 && curiotaidx < iota.getType().getShape().size()) {
        if (curiotaidx == iota.getIotaDimension()) {
          return failure();
        }
        curiotaidx++;
        ival = iota.getType().getShape()[curiotaidx];
      }
      if (en.value() == ival) {
        if (curiotaidx == iota.getIotaDimension()) {
          iotadim = en.index();
        }
        curiotaidx++;
        continue;
      }
      return failure();
    }
    rewriter.replaceOpWithNewOp<stablehlo::IotaOp>(op, op.getType(), iotadim);
    return success();
  }
};

struct BroadcastIota final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp, BroadcastIota> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    auto iota = op.getOperand().getDefiningOp<stablehlo::IotaOp>();
    if (!iota)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::IotaOp>(
        op, op.getType(), op.getBroadcastDimensions()[iota.getIotaDimension()]);
    return success();
  }
};

LogicalResult reshapePadHelper(stablehlo::ReshapeOp op,
                               PatternRewriter &rewriter) {
  auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
  if (!pad)
    return failure();
  if (anyPadSizesNegative(pad))
    return failure();
  size_t curiotaidx = 0;
  SmallVector<int64_t> lows;
  SmallVector<int64_t> highs;
  SmallVector<int64_t> interiors;

  SmallVector<int64_t> inner_shape;
  for (auto en : llvm::enumerate(op.getType().getShape())) {
    if (en.value() == 1) {
      lows.push_back(0);
      highs.push_back(0);
      interiors.push_back(0);
      inner_shape.push_back(1);
      continue;
    }

    if (curiotaidx == pad.getType().getShape().size())
      return failure();

    auto ival = pad.getType().getShape()[curiotaidx];
    while (ival == 1 && curiotaidx < pad.getType().getShape().size()) {
      assert(pad.getEdgePaddingLow()[curiotaidx] == 0);
      assert(pad.getEdgePaddingHigh()[curiotaidx] == 0);
      assert(pad.getInteriorPadding()[curiotaidx] == 0);
      curiotaidx++;
      ival = pad.getType().getShape()[curiotaidx];
    }
    if (en.value() == ival) {
      lows.push_back(pad.getEdgePaddingLow()[curiotaidx]);
      highs.push_back(pad.getEdgePaddingHigh()[curiotaidx]);
      interiors.push_back(pad.getInteriorPadding()[curiotaidx]);
      inner_shape.push_back(pad.getOperand().getType().getShape()[curiotaidx]);
      curiotaidx++;
      continue;
    }
    return failure();
  }
  auto inner = rewriter.create<stablehlo::ReshapeOp>(
      op.getLoc(),
      RankedTensorType::get(inner_shape, op.getType().getElementType()),
      pad.getOperand());
  rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
      op, inner, pad.getPaddingValue(), lows, highs, interiors);
  return success();
}

struct DotReshapePad final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, DotReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    for (auto u : op->getUsers())
      if (!isa<stablehlo::DotGeneralOp>(u))
        return failure();

    if (!reshapePadHelper(op, rewriter).succeeded())
      return failure();
    return success();
  }
};

struct ZeroProductReshapePad final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ZeroProductReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    if (!matchPattern(pad.getPaddingValue(), m_AnyZeroFloat()))
      return failure();

    for (auto u : op->getUsers()) {
      if (!isa<stablehlo::MulOp>(u) && !isa<stablehlo::DivOp>(u))
        return failure();
    }
    if (!reshapePadHelper(op, rewriter).succeeded())
      return failure();
    return success();
  }
};

struct PadReshapePad final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, PadReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    for (auto u : op->getUsers()) {
      auto pad2 = dyn_cast<stablehlo::PadOp>(u);
      if (!pad2)
        return failure();
      if (pad2.getPaddingValue() != pad.getPaddingValue())
        return failure();
    }
    if (!reshapePadHelper(op, rewriter).succeeded())
      return failure();
    return success();
  }
};

struct BinopConstReshapePad final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, BinopConstReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    for (auto u : op->getUsers()) {
      if (isa<stablehlo::AddOp>(u) || isa<stablehlo::SubtractOp>(u) ||
          isa<stablehlo::MulOp>(u) || isa<stablehlo::DivOp>(u)) {
        bool hasConst = false;
        for (auto op : u->getOperands())
          hasConst |= op.getDefiningOp<stablehlo::ConstantOp>() != nullptr;
        if (hasConst)
          continue;
      }
      return failure();
    }
    if (!reshapePadHelper(op, rewriter).succeeded())
      return failure();
    return success();
  }
};

struct ConcatAppendingReshape final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                              ConcatAppendingReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() != 2)
      return failure();

    SmallVector<Value> lhs;

    SmallVector<Type> converts;

    size_t frontSize = 0;
    for (auto v : op.getOperands()) {
      if (auto t = v.getDefiningOp<stablehlo::ConvertOp>()) {
        v = t.getOperand();
        converts.push_back(
            cast<RankedTensorType>(v.getType()).getElementType());
      } else
        converts.push_back(nullptr);
      if (auto t = v.getDefiningOp<stablehlo::ReshapeOp>()) {
        lhs.push_back(t->getOperand(0));

        auto prevshape = t.getOperand().getType().getShape();
        auto postshape = t.getType().getShape();
        if (prevshape.size() == 0)
          return failure();
        if (prevshape.size() + 1 != postshape.size())
          return failure();
        if (postshape[0] != 1)
          return failure();

        frontSize += prevshape[0];

        for (auto en : llvm::enumerate(prevshape)) {
          if (en.value() != postshape[1 + en.index()])
            return failure();
        }

      } else
        return failure();
    }

    Type typeconvert = converts[0];
    for (auto c : converts)
      if (c != typeconvert)
        return failure();

    RankedTensorType nextType = op.getType();
    auto nextDim = op.getDimension();
    if (nextDim == 0) {
      SmallVector<int64_t> nextShape(nextType.getShape().begin() + 1,
                                     nextType.getShape().end());

      nextShape[0] = frontSize;
      nextType = RankedTensorType::get(
          nextShape, typeconvert ? typeconvert : nextType.getElementType());
      nextDim = 0;
    } else {
      nextType = RankedTensorType::get(nextType.getShape().drop_front(),
                                       typeconvert ? typeconvert
                                                   : nextType.getElementType());
      nextDim--;
    }
    auto lhs2 = rewriter.create<stablehlo::ConcatenateOp>(op.getLoc(), nextType,
                                                          lhs, nextDim);

    Value res2 = rewriter.create<stablehlo::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(op.getType().getShape(),
                              nextType.getElementType()),
        lhs2);

    if (typeconvert)
      res2 = rewriter.create<stablehlo::ConvertOp>(op.getLoc(), op.getType(),
                                                   res2);

    rewriter.replaceOp(op, res2);
    return success();
  }
};

template <typename T>
struct UnaryPadPush final : CheckedOpRewritePattern<T, UnaryPadPush<T>> {
  using CheckedOpRewritePattern<T, UnaryPadPush<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    auto pad = op->getOperand(0).template getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    auto padval = pad.getPaddingValue();
    auto padval2 = rewriter.create<T>(
        op.getLoc(), RankedTensorType::get({}, op.getType().getElementType()),
        padval);

    auto val = pad.getOperand();
    auto val2 = rewriter.create<T>(
        op.getLoc(),
        RankedTensorType::get(cast<RankedTensorType>(val.getType()).getShape(),
                              op.getType().getElementType()),
        val);

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, val2, padval2, pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
        pad.getInteriorPadding());
    return success();
  }
};

struct TransposePad final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto pad = op->getOperand(0).template getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    auto padval = pad.getPaddingValue();

    auto val = pad.getOperand();
    auto val2 = rewriter.create<stablehlo::TransposeOp>(op.getLoc(), val,
                                                        op.getPermutation());

    SmallVector<int64_t> low;
    SmallVector<int64_t> high;
    SmallVector<int64_t> inner;
    for (auto idx : op.getPermutation()) {
      low.push_back(pad.getEdgePaddingLow()[idx]);
      high.push_back(pad.getEdgePaddingHigh()[idx]);
      inner.push_back(pad.getInteriorPadding()[idx]);
    }

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(op, val2, padval, low, high,
                                                  inner);
    rewriter.eraseOp(pad);
    return success();
  }
};

template <typename T>
struct ConcatPushBinop final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatPushBinop<T>> {
  using CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                ConcatPushBinop<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() != 2)
      return failure();

    SmallVector<Value> lhs;
    SmallVector<Value> rhs;

    SmallVector<Type> converts;

    for (auto v : op.getOperands()) {
      if (auto t = v.getDefiningOp<stablehlo::ConvertOp>()) {
        converts.push_back(
            cast<RankedTensorType>(t.getType()).getElementType());
        v = t.getOperand();
      } else
        converts.push_back(nullptr);
      if (auto t = v.getDefiningOp<T>()) {
        lhs.push_back(t->getOperand(0));
        rhs.push_back(t->getOperand(1));
      } else
        return failure();
    }

    Type typeconvert = converts[0];
    for (auto c : converts)
      if (c != typeconvert)
        return failure();

    auto lhs2 = rewriter.create<stablehlo::ConcatenateOp>(op.getLoc(), lhs,
                                                          op.getDimension());
    auto rhs2 = rewriter.create<stablehlo::ConcatenateOp>(op.getLoc(), rhs,
                                                          op.getDimension());

    Value res2 = rewriter.create<T>(op.getLoc(), lhs2, rhs2);

    if (typeconvert)
      res2 = rewriter.create<stablehlo::ConvertOp>(
          op.getLoc(),
          RankedTensorType::get(
              cast<RankedTensorType>(res2.getType()).getShape(), typeconvert),
          res2);

    rewriter.replaceOp(op, res2);
    return success();
  }
};

struct ConcatFuse final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatFuse> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() == 1 &&
        op->getOperand(0).getType() == op.getType()) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    SmallVector<Value> vals;
    bool changed = false;
    for (auto v : op->getOperands()) {
      if (auto c2 = v.getDefiningOp<stablehlo::ConcatenateOp>()) {
        if (c2.getDimension() == op.getDimension()) {
          for (auto v2 : c2->getOperands())
            vals.push_back(v2);
          changed = true;
          continue;
        }
      }
      if (cast<RankedTensorType>(v.getType()).getShape()[op.getDimension()] ==
          0) {
        changed = true;
        continue;
      }
      vals.push_back(v);
    }
    if (!changed)
      return failure();
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, op.getType(), vals, op.getDimensionAttr());
    return success();
  }
};

struct ConcatToBroadcast final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatToBroadcast> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() <= 1)
      return failure();
    for (auto opv : op->getOperands())
      if (opv != op->getOperand(0))
        return failure();
    SmallVector<int64_t> bcast;
    if (cast<RankedTensorType>(op->getOperand(0).getType())
            .getShape()[op.getDimension()] != 1)
      return failure();
    for (auto en : llvm::enumerate(op.getType().getShape())) {
      bcast.push_back(en.index());
    }
    auto bcast2 = rewriter.getDenseI64ArrayAttr(bcast);
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), op->getOperand(0), bcast2);
    return success();
  }
};

struct GammaConstProp final
    : CheckedOpRewritePattern<chlo::LgammaOp, GammaConstProp> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::LgammaOp op,
                                    PatternRewriter &rewriter) const {
    // return if not constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&inputAttr)))
      return failure();
    Value result =
        stablehlo::materializeLgamma(rewriter, op.getLoc(), op->getOperands());
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct DynamicUpdateSliceConstProp final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp,
                              DynamicUpdateSliceConstProp> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;
  size_t max_constant_expansion;

  DynamicUpdateSliceConstProp(size_t max_constant_expansion,
                              MLIRContext *context, PatternBenefit benefit = 1,
                              ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp op,
                                    PatternRewriter &rewriter) const {

    size_t size = 1;
    for (auto sz : op.getType().getShape())
      size *= sz;
    if (size >= max_constant_expansion)
      return failure();

    auto startIndices = op.getStartIndices();

    bool legal = true;

    DenseElementsAttr operandConstant;
    DenseElementsAttr updateConstant;

    SmallVector<DenseElementsAttr> constants(startIndices.size(),
                                             DenseElementsAttr());
    for (auto &operand : op->getOpOperands()) {
      if (operand.getOperandNumber() == 0)
        legal &= matchPattern(operand.get(), m_Constant(&operandConstant));
      else if (operand.getOperandNumber() == 1)
        legal &= matchPattern(operand.get(), m_Constant(&updateConstant));
      else
        legal &= matchPattern(
            operand.get(),
            m_Constant(&constants[operand.getOperandNumber() - 2]));
    }

    if (operandConstant && updateConstant && operandConstant.isSplat() &&
        updateConstant.isSplat() &&
        operandConstant.getSplatValue<Attribute>() ==
            updateConstant.getSplatValue<Attribute>()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    if (!legal)
      return failure();

    stablehlo::Tensor operandTen = stablehlo::constantOp(operandConstant);
    stablehlo::Tensor updateTen = stablehlo::constantOp(updateConstant);
    SmallVector<stablehlo::Tensor> inps;
    for (auto &c : constants)
      inps.push_back(stablehlo::constantOp(c));

    auto out = stablehlo::dynamicUpdateSliceOp(operandTen, updateTen, inps,
                                               op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                       fromTensor(out));

    return success();
  }
};

template <auto f>
LogicalResult binaryConstProp(Operation *op, PatternRewriter &rewriter) {
  // return if not constant
  DenseElementsAttr lhsAttr;
  DenseElementsAttr rhsAttr;
  if (!matchPattern(op->getOperand(0), m_Constant(&lhsAttr)) ||
      !matchPattern(op->getOperand(1), m_Constant(&rhsAttr)))
    return failure();

  stablehlo::Tensor lhsTen;
  stablehlo::Tensor rhsTen;
  RankedTensorType ty = cast<RankedTensorType>(op->getResultTypes()[0]);

  // only const prop if the constant has a single user to prevent create many
  // constants
  if ((!lhsAttr.isSplat() || !rhsAttr.isSplat()) &&
      !llvm::hasSingleElement(op->getResult(0).getUsers()))
    return failure();

  if (lhsAttr.isSplat() && rhsAttr.isSplat()) {
    ty = RankedTensorType::get(
        {}, cast<ShapedType>(op->getResultTypes()[0]).getElementType());

    lhsTen = stablehlo::makeTensor(lhsAttr.resizeSplat(RankedTensorType::get(
        {}, cast<ShapedType>(op->getOperand(0).getType()).getElementType())));

    rhsTen = stablehlo::makeTensor(rhsAttr.resizeSplat(RankedTensorType::get(
        {}, cast<ShapedType>(op->getOperand(1).getType()).getElementType())));
  } else {
    lhsTen = stablehlo::constantOp(lhsAttr);
    rhsTen = stablehlo::constantOp(rhsAttr);
  }

  // get the resultType
  auto resultType = cast<ShapedType>(ty);

  auto out = fromTensor(f(lhsTen, rhsTen, resultType));

  if (lhsAttr.isSplat() && rhsAttr.isSplat()) {
    out = out.resizeSplat(cast<ShapedType>(op->getResultTypes()[0]));
  }

  // Replace with new constant op containing the computed result
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
      op, op->getResultTypes()[0], out);
  return success();
}

template <typename OpTy, auto constPropFn>
struct BinaryConstProp final
    : CheckedOpRewritePattern<OpTy, BinaryConstProp<OpTy, constPropFn>> {
  using CheckedOpRewritePattern<
      OpTy, BinaryConstProp<OpTy, constPropFn>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter) const {
    return binaryConstProp<constPropFn>(op, rewriter);
  }
};

template <auto f>
LogicalResult unaryConstProp(Operation *op, PatternRewriter &rewriter) {
  // return if not constant
  DenseElementsAttr inputAttr;
  if (!matchPattern(op->getOperand(0), m_Constant(&inputAttr)))
    return failure();

  stablehlo::Tensor inputTen;
  RankedTensorType ty = cast<RankedTensorType>(op->getResultTypes()[0]);

  // only const prop if the constant has a single user to prevent create many
  // constants
  if (!inputAttr.isSplat() &&
      !llvm::hasSingleElement(op->getResult(0).getUsers()))
    return failure();

  if (inputAttr.isSplat()) {
    ty = RankedTensorType::get(
        {}, cast<ShapedType>(op->getResultTypes()[0]).getElementType());
    auto inputTy = RankedTensorType::get(
        {}, cast<ShapedType>(op->getOperand(0).getType()).getElementType());
    inputTen = stablehlo::makeTensor(inputAttr.resizeSplat(inputTy));
  } else {
    inputTen = stablehlo::constantOp(inputAttr);
  }
  // get the resultType
  auto resultType = cast<ShapedType>(ty);

  // Convert constant to tensor, compute log, then convert back to attribute
  auto out = fromTensor(f(inputTen, resultType));

  if (inputAttr.isSplat()) {
    out = out.resizeSplat(cast<ShapedType>(op->getResultTypes()[0]));
  }
  // Replace with new constant op containing the computed result
  rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
      op, op->getResultTypes()[0], out);

  return success();
}

template <typename OpTy, auto constPropFn>
struct UnaryConstProp final
    : CheckedOpRewritePattern<OpTy, UnaryConstProp<OpTy, constPropFn>> {
  using CheckedOpRewritePattern<
      OpTy, UnaryConstProp<OpTy, constPropFn>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter) const {
    return unaryConstProp<constPropFn>(op, rewriter);
  }
};

struct ChloInfConstProp final
    : CheckedOpRewritePattern<chlo::IsInfOp, ChloInfConstProp> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::IsInfOp op,
                                    PatternRewriter &rewriter) const {
    // return if not constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&inputAttr)))
      return failure();

    DenseElementsAttr outAttr;
    auto resultTy = cast<ShapedType>(op->getResultTypes()[0]);

    // handle splat separately
    if (inputAttr.isSplat()) {
      llvm::APInt resVals;
      if (matchPattern(op.getOperand(), m_PosInfFloat()) ||
          matchPattern(op.getOperand(), m_NegInfFloat())) {
        // true
        resVals = llvm::APInt(1, 1);
      } else {
        // false
        resVals = llvm::APInt(1, 0);
      }

      outAttr = DenseElementsAttr::get(resultTy, resVals);
    } else {
      SmallVector<APInt> resVals;
      resVals.reserve(inputAttr.getNumElements());

      // iterate over every element in inputAttr and run check.
      for (APFloat val : inputAttr.getValues<APFloat>()) {
        bool isInf = val.isInfinity();
        resVals.push_back(APInt(1, isInf ? 1 : 0));
      }

      outAttr = DenseElementsAttr::get(resultTy, resVals);
    }

    // replace op with the bool const op
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op->getResultTypes()[0], outAttr);

    return success();
  }
};

struct ConcatConstProp final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatConstProp> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;
  size_t max_constant_expansion;
  ConcatConstProp(size_t max_constant_expansion, MLIRContext *context,
                  PatternBenefit benefit = 1,
                  ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    if (op->getNumOperands() == 1) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }

    {
      SmallVector<Value> subconcat;
      bool changed = false;
      for (auto v : op->getOperands()) {
        if (auto c2 = v.getDefiningOp<stablehlo::ConcatenateOp>())
          if (c2.getDimension() == op.getDimension()) {
            for (auto v2 : c2->getOperands())
              subconcat.push_back(v2);
            changed = true;
            continue;
          }
        subconcat.push_back(v);
      }
      if (changed) {
        rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
            op, subconcat, op.getDimension());
        return success();
      }
    }

    SmallVector<DenseElementsAttr> constants;
    constants.assign(op->getNumOperands(), DenseElementsAttr());
    bool legal = true;
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i) {
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));
      if (!constants[i])
        legal = false;
    }

    if (legal) {

      if (constants[0].isSplat()) {
        bool allSplat = true;
        for (int i = 1; i < op->getNumOperands(); i++) {
          if (!constants[i].isSplat()) {
            allSplat = false;
            break;
          }
          if (constants[0].getSplatValue<Attribute>() !=
              constants[i].getSplatValue<Attribute>()) {
            allSplat = false;
            break;
          }
        }
        if (allSplat) {
          rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
              op, op.getType(), constants[0].resizeSplat(op.getType()));
          return success();
        }
      }
      size_t size = 1;
      for (auto sz : op.getType().getShape())
        size *= sz;
      if (size >= max_constant_expansion)
        return failure();

      SmallVector<stablehlo::Tensor> inps;
      for (auto &c : constants)
        inps.push_back(stablehlo::constantOp(c));
      auto out =
          stablehlo::concatenateOp(inps, op.getDimension(), op.getType());
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                         fromTensor(out));
      return success();
    }
    return failure();
  }
};

struct ReshapeEmptyBroadcast final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeEmptyBroadcast> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto bcast = op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcast)
      return failure();
    if (bcast.getBroadcastDimensions().size() != 0)
      return failure();
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), bcast.getOperand(), bcast.getBroadcastDimensions());
    return success();
  }
};

struct BroadcastReshape final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp, BroadcastReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    SmallVector<int64_t> dims;

    size_t pre_reshape_idx = 0;
    size_t postidx = 0;

    SmallVector<int64_t> oneOutIdxs;
    for (auto en : llvm::enumerate(op.getType().getShape()))
      if (en.value() == 1)
        oneOutIdxs.push_back(en.index());

    for (auto en : llvm::enumerate(reshape.getType().getShape())) {
      if (en.value() == 1) {
        continue;
      }

      if (pre_reshape_idx == reshape.getOperand().getType().getShape().size())
        return failure();
      auto ival = reshape.getOperand().getType().getShape()[pre_reshape_idx];
      while (ival == 1 &&
             pre_reshape_idx + 1 <
                 reshape.getOperand().getType().getShape().size()) {
        if (postidx == oneOutIdxs.size())
          return failure();
        dims.push_back(oneOutIdxs[postidx]);
        postidx++;
        pre_reshape_idx++;
        ival = reshape.getOperand().getType().getShape()[pre_reshape_idx];
      }
      if (en.value() == ival) {
        dims.push_back(op.getBroadcastDimensions()[en.index()]);
        pre_reshape_idx++;
        continue;
      }
      return failure();
    }
    while (pre_reshape_idx !=
           reshape.getOperand().getType().getShape().size()) {
      auto ival = reshape.getOperand().getType().getShape()[pre_reshape_idx];
      assert(ival == 1);
      (void)ival;

      size_t nextdim = 0;
      if (postidx == oneOutIdxs.size()) {
        return failure();
      } else {
        nextdim = oneOutIdxs[postidx];
        postidx++;
      }
      dims.push_back(nextdim);
      pre_reshape_idx++;
    }
    assert(dims.size() == reshape.getOperand().getType().getShape().size());
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), reshape.getOperand(), dims);
    return success();
  }
};

struct BroadcastToReshape final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp, BroadcastToReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();
    assert(op.getBroadcastDimensions().size() ==
           op.getOperand().getType().getShape().size());

    // Ensure these are sorted
    for (auto en : llvm::enumerate(op.getBroadcastDimensions())) {
      if (en.index() == 0)
        continue;
      if (op.getBroadcastDimensions()[en.index() - 1] >= en.value()) {
        return failure();
      }
    }

    // Check that no new data is added
    for (auto en : llvm::enumerate(op.getType().getShape())) {
      ssize_t idx = -1;
      for (auto en2 : llvm::enumerate(op.getBroadcastDimensions())) {
        if (en2.value() == en.index())
          idx = en2.index();
      }
      if (idx != -1) {
        if (en.value() != op.getOperand().getType().getShape()[idx]) {
          return failure();
        }
        continue;
      }
      if (en.value() != 1)
        return failure();
    }

    // replace with reshape
    if (op.getType() == op.getOperand().getType())
      rewriter.replaceOp(op, op.getOperand());
    else
      rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                        op.getOperand());
    return success();
  }
};

struct BroadcastPad final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp, BroadcastPad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {

    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    SplatElementsAttr splat1;
    if (!llvm::hasSingleElement(pad->getUsers()) &&
        !(matchPattern(pad.getOperand(), m_Constant(&splat1))))
      return failure();

    SmallVector<int64_t> paddingLow;
    SmallVector<int64_t> paddingHigh;
    SmallVector<int64_t> paddingInt;

    // broadcast dim map idx from pad output -> which broadcast output

    SmallVector<int64_t> midShape;

    for (auto en : llvm::enumerate(op.getType().getShape())) {
      // which pad in/output dim
      ssize_t origIdx = -1;
      for (auto en2 : llvm::enumerate(op.getBroadcastDimensions())) {
        if (en2.value() == en.index()) {
          origIdx = en2.index();
          break;
        }
      }

      if (origIdx == -1) {
        paddingLow.push_back(0);
        paddingHigh.push_back(0);
        paddingInt.push_back(0);
        midShape.push_back(en.value());
      } else {
        if (pad.getType().getShape()[origIdx] == en.value()) {
          paddingLow.push_back(pad.getEdgePaddingLow()[origIdx]);
          paddingHigh.push_back(pad.getEdgePaddingHigh()[origIdx]);
          paddingInt.push_back(pad.getInteriorPadding()[origIdx]);
          midShape.push_back(pad.getOperand().getType().getShape()[origIdx]);
        } else {
          if (pad.getEdgePaddingLow()[origIdx] != 0)
            return failure();
          if (pad.getEdgePaddingHigh()[origIdx] != 0)
            return failure();
          if (pad.getInteriorPadding()[origIdx] != 0)
            return failure();
          paddingLow.push_back(pad.getEdgePaddingLow()[origIdx]);
          paddingHigh.push_back(pad.getEdgePaddingHigh()[origIdx]);
          paddingInt.push_back(pad.getInteriorPadding()[origIdx]);
          midShape.push_back(en.value());
        }
      }
    }

    auto bcast2 = rewriter.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(),
        RankedTensorType::get(midShape, pad.getType().getElementType()),
        pad.getOperand(), op.getBroadcastDimensions());

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, bcast2, pad.getPaddingValue(), paddingLow, paddingHigh, paddingInt);
    return success();
  }
};

// Given a value and index idx, determine whether all values are the same along
// idx. If so, return said value
std::optional<Value> is_same_in_axis(OpBuilder &rewriter, ShapedType outTy,
                                     Value v, size_t idx) {
  mlir::SplatElementsAttr splat;
  if (matchPattern(v, m_Constant(&splat))) {
    return rewriter.create<stablehlo::ConstantOp>(v.getLoc(), outTy,
                                                  splat.resizeSplat(outTy));
  }

  return {};
}

struct ScatterToDynamicUpdateSlice final
    : CheckedOpRewritePattern<stablehlo::ScatterOp,
                              ScatterToDynamicUpdateSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ScatterOp op,
                                    PatternRewriter &rewriter) const {
    Block &body = op.getUpdateComputation().front();
    if (body.getOperations().size() != 1)
      return failure();

    Operation &innerOp = body.front();
    if (!isa<stablehlo::ReturnOp>(&innerOp)) {
      return failure();
    }
    if (innerOp.getNumOperands() != 1) {
      return failure();
    }

    if (op.getInputs().size() != 1)
      return failure();

    // For us to proceed, either we are returning the last block argument or we
    // are returning a constant
    Value update = nullptr;
    DenseElementsAttr splatAttr;

    auto retop = dyn_cast<BlockArgument>(innerOp.getOperand(0));
    if (retop) {
      if (retop.getOwner() != &body)
        return failure();
      if (retop.getArgNumber() != 1)
        return failure();
      update = op.getUpdates()[0];
    } else {
      DenseElementsAttr attr;
      if (matchPattern(innerOp.getOperand(0), m_Constant(&attr))) {
        splatAttr = DenseElementsAttr::get(
            cast<ShapedType>(op.getUpdates()[0].getType()),
            attr.getSplatValue<Attribute>());
      } else {
        return failure();
      }
    }

    auto dims = op.getScatterDimensionNumbers();

    auto input = op.getInputs()[0];
    auto scatter = op.getScatterIndices();
    auto updateShape =
        cast<ShapedType>(op.getUpdates()[0].getType()).getShape();

    if (dims.getInsertedWindowDims().size() == 0 &&
        dims.getUpdateWindowDims().size() == updateShape.size()) {

      if (update == nullptr) {
        update = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), op.getUpdates()[0].getType(), splatAttr);
      }

      auto ity = RankedTensorType::get(
          {}, cast<ShapedType>(scatter.getType()).getElementType());
      SmallVector<Value> start(updateShape.size(), 0);
      for (auto en : llvm::enumerate(dims.getScatterDimsToOperandDims())) {
        auto startval = is_same_in_axis(rewriter, ity, scatter, en.index());
        if (!startval)
          return failure();
        start[en.value()] = *startval;
      }
      for (auto &v : start) {
        if (v != nullptr)
          continue;
        v = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), ity, cast<ElementsAttr>(makeAttr(ity, 0)));
      }
      rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
          op, op.getResult(0).getType(), input, update, start);
      return success();
    }

    return failure();
  }
};

bool isOnlyUsedInOperation(Operation *operation, Operation *parentOp) {
  if (!operation || !parentOp)
    return false;

  for (Operation *user : operation->getUsers()) {
    if (user != parentOp)
      return false;
  }

  return true;
}

template <typename OpType>
LogicalResult simplifyBinaryOpWithTranspose(OpType op,
                                            PatternRewriter &rewriter) {
  auto lhsOp = op.getLhs().template getDefiningOp<stablehlo::TransposeOp>();
  auto rhsOp = op.getRhs().template getDefiningOp<stablehlo::TransposeOp>();
  if ((lhsOp && rhsOp) && (lhsOp.getPermutation() == rhsOp.getPermutation()) &&
      isOnlyUsedInOperation(lhsOp, op) && isOnlyUsedInOperation(rhsOp, op)) {
    auto newOp = rewriter.create<OpType>(op.getLoc(), lhsOp.getOperand(),
                                         rhsOp.getOperand());
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, newOp,
                                                        lhsOp.getPermutation());
    return success();
  }

  if (lhsOp && isOnlyUsedInOperation(lhsOp, op)) {
    auto rhsConstOp =
        op.getRhs().template getDefiningOp<stablehlo::ConstantOp>();
    if (rhsConstOp && isOnlyUsedInOperation(rhsConstOp, op)) {
      // This will be eliminated by a transpose(constant) -> constant
      // optimization
      auto transposedConstOp = rewriter.create<stablehlo::TransposeOp>(
          rhsConstOp.getLoc(), rhsConstOp,
          getInversePermutation(lhsOp.getPermutation()));
      auto newOp = rewriter.create<OpType>(op.getLoc(), lhsOp.getOperand(),
                                           transposedConstOp);
      rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
          op, newOp, lhsOp.getPermutation());
      return success();
    }
  }

  if (rhsOp && isOnlyUsedInOperation(rhsOp, op)) {
    auto lhsConstOp =
        op.getLhs().template getDefiningOp<stablehlo::ConstantOp>();
    if (lhsConstOp && isOnlyUsedInOperation(lhsConstOp, op)) {
      // This will be eliminated by a transpose(constant) -> constant
      // optimization
      auto transposedConstOp = rewriter.create<stablehlo::TransposeOp>(
          lhsConstOp.getLoc(), lhsConstOp,
          getInversePermutation(rhsOp.getPermutation()));
      auto newOp = rewriter.create<OpType>(op.getLoc(), transposedConstOp,
                                           rhsOp.getOperand());
      rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
          op, newOp, rhsOp.getPermutation());
      return success();
    }
  }

  return failure();
}

template <typename OpType>
struct BinaryOpTransposeSimplify
    : public CheckedOpRewritePattern<OpType,
                                     BinaryOpTransposeSimplify<OpType>> {
  using CheckedOpRewritePattern<
      OpType, BinaryOpTransposeSimplify<OpType>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(OpType op,
                                    PatternRewriter &rewriter) const {
    return simplifyBinaryOpWithTranspose(op, rewriter);
  }
};

template <typename OpType>
struct TransposeUnaryTransposeSimplify
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeUnaryTransposeSimplify<OpType>> {
  using CheckedOpRewritePattern<
      stablehlo::TransposeOp,
      TransposeUnaryTransposeSimplify<OpType>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp outerTransposeOp,
                                    PatternRewriter &rewriter) const {
    auto unaryOp =
        outerTransposeOp.getOperand().template getDefiningOp<OpType>();
    if (!unaryOp && !isOnlyUsedInOperation(unaryOp, outerTransposeOp))
      return failure();

    auto innerTransposeOp =
        unaryOp->getOperand(0).template getDefiningOp<stablehlo::TransposeOp>();
    if (!innerTransposeOp)
      return failure();

    if (outerTransposeOp.getPermutation() != innerTransposeOp.getPermutation())
      return failure();

    rewriter.replaceOpWithNewOp<OpType>(outerTransposeOp,
                                        outerTransposeOp.getType(),
                                        innerTransposeOp.getOperand());

    return success();
  }
};

struct AddSimplify
    : public CheckedOpRewritePattern<stablehlo::AddOp, AddSimplify> {
  using CheckedOpRewritePattern<stablehlo::AddOp,
                                AddSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AddOp op,
                                    PatternRewriter &rewriter) const {

    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero()) ||
        matchPattern(op.getLhs(), m_AnyZeroComplex())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero()) ||
        matchPattern(op.getRhs(), m_AnyZeroComplex())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct ReplaceNegAddWithSubtract
    : public CheckedOpRewritePattern<stablehlo::AddOp,
                                     ReplaceNegAddWithSubtract> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AddOp op,
                                    PatternRewriter &rewriter) const {
    auto negateOp = op.getRhs().getDefiningOp<stablehlo::NegOp>();

    if (!negateOp)
      return failure();

    if (!llvm::hasSingleElement(negateOp->getUsers()))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(op, op.getLhs(),
                                                       negateOp.getOperand());
    return success();
  }
};

struct SubSimplify
    : public CheckedOpRewritePattern<stablehlo::SubtractOp, SubSimplify> {
  using CheckedOpRewritePattern<stablehlo::SubtractOp,
                                SubSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SubtractOp op,
                                    PatternRewriter &rewriter) const {

    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero()) ||
        matchPattern(op.getRhs(), m_AnyZeroComplex())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero()) ||
        matchPattern(op.getLhs(), m_AnyZeroComplex())) {
      rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, op.getRhs());
      return success();
    }

    if (isa<IntegerType>(op.getType().getElementType()) &&
        op.getLhs() == op.getRhs()) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    return failure();
  }
};

struct NoNanSelfSubSimplify
    : public CheckedOpRewritePattern<stablehlo::SubtractOp,
                                     NoNanSelfSubSimplify> {
  using CheckedOpRewritePattern<stablehlo::SubtractOp,
                                NoNanSelfSubSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SubtractOp op,
                                    PatternRewriter &rewriter) const {

    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    return failure();
  }
};

struct AndSimplify
    : public CheckedOpRewritePattern<stablehlo::AndOp, AndSimplify> {
  using CheckedOpRewritePattern<stablehlo::AndOp,
                                AndSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AndOp op,
                                    PatternRewriter &rewriter) const {

    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // false & x -> x
    for (auto v : op.getOperands()) {
      if (matchPattern(v, m_Zero())) {
        rewriter.replaceOp(op, v);
        return success();
      }
    }

    // true & x -> x
    for (int i = 0; i < 2; i++) {
      if (matchPattern(op.getOperand(i), m_One())) {
        rewriter.replaceOp(op, op.getOperand(1 - i));
        return success();
      }
    }

    return failure();
  }
};

struct OrSimplify
    : public CheckedOpRewritePattern<stablehlo::OrOp, OrSimplify> {
  using CheckedOpRewritePattern<stablehlo::OrOp,
                                OrSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::OrOp op,
                                    PatternRewriter &rewriter) const {

    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // true | x -> true
    for (auto v : op.getOperands()) {
      if (matchPattern(v, m_One())) {
        rewriter.replaceOp(op, v);
        return success();
      }
    }

    // false | x -> x
    for (int i = 0; i < 2; i++) {
      if (matchPattern(op.getOperand(i), m_Zero())) {
        rewriter.replaceOp(op, op.getOperand(1 - i));
        return success();
      }
    }

    return failure();
  }
};

struct XorSimplify
    : public CheckedOpRewritePattern<stablehlo::XorOp, XorSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::XorOp op,
                                    PatternRewriter &rewriter) const {

    // false ^ x -> x
    for (int i = 0; i < 2; i++) {
      if (matchPattern(op.getOperand(i), m_Zero())) {
        rewriter.replaceOp(op, op.getOperand(1 - i));
        return success();
      }
    }

    // true ^ x -> not x
    for (auto v : op.getOperands()) {
      if (matchPattern(v, m_One())) {
        rewriter.replaceOpWithNewOp<stablehlo::NotOp>(op, v);
        return success();
      }
    }

    return failure();
  }
};

struct MulSimplify
    : public CheckedOpRewritePattern<stablehlo::MulOp, MulSimplify> {
  using CheckedOpRewritePattern<stablehlo::MulOp,
                                MulSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    // 1 * x -> x
    if (matchPattern(op.getLhs(), m_One()) ||
        matchPattern(op.getLhs(), m_OneFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    // x * 1 -> x
    if (matchPattern(op.getRhs(), m_One()) ||
        matchPattern(op.getRhs(), m_OneFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    return failure();
  }
};

struct DivSimplify
    : public CheckedOpRewritePattern<stablehlo::DivOp, DivSimplify> {
  using CheckedOpRewritePattern<stablehlo::DivOp,
                                DivSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DivOp op,
                                    PatternRewriter &rewriter) const {

    // x / 1 -> x
    if (matchPattern(op.getRhs(), m_OneFloat()) ||
        matchPattern(op.getRhs(), m_One())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // x / const -> x * (1 / const)
    if (isa<FloatType>(op.getType().getElementType())) {
      DenseElementsAttr rhsAttr;
      if (matchPattern(op.getRhs(), m_Constant(&rhsAttr))) {
        {
          DenseElementsAttr lhsAttr;
          if (matchPattern(op.getLhs(), m_Constant(&lhsAttr)))
            return failure(); // const prop will evaluate this
        }

        auto ty = op.getType();
        if (rhsAttr.isSplat()) {
          ty = RankedTensorType::get(
              {}, cast<ShapedType>(op->getResultTypes()[0]).getElementType());
          rhsAttr = rhsAttr.resizeSplat(ty);
        }

        auto rhsTen = stablehlo::constantOp(rhsAttr);
        auto oneTen =
            stablehlo::constantOp(cast<ElementsAttr>(makeAttr(ty, 1)));
        auto out = fromTensor(stablehlo::divideOp(oneTen, rhsTen, ty));

        if (ty != op.getType()) {
          out = out.resizeSplat(op.getType());
        }

        rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
            op, op.getLhs(),
            rewriter.create<stablehlo::ConstantOp>(op.getLoc(), op.getType(),
                                                   out));
        return success();
      }
    }

    return failure();
  }
};

struct NoNanDivSimplify final
    : public CheckedOpRewritePattern<stablehlo::DivOp, NoNanDivSimplify> {
  using CheckedOpRewritePattern<stablehlo::DivOp,
                                NoNanDivSimplify>::CheckedOpRewritePattern;

  NoNanDivSimplify(bool allowOnFloatingPointMath, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : CheckedOpRewritePattern(context, benefit),
        allowOnFloatingPointMath(allowOnFloatingPointMath) {}

  LogicalResult matchAndRewriteImpl(stablehlo::DivOp op,
                                    PatternRewriter &rewriter) const {
    if (!canApplyNoNanPattern(allowOnFloatingPointMath, op.getType()))
      return failure();

    // 0 / x -> 0
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // x / x -> 1
    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 1)));
      return success();
    }

    return failure();
  }

private:
  bool allowOnFloatingPointMath;
};

struct RemSimplify
    : public CheckedOpRewritePattern<stablehlo::RemOp, RemSimplify> {
  using CheckedOpRewritePattern<stablehlo::RemOp,
                                RemSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::RemOp op,
                                    PatternRewriter &rewriter) const {

    if (matchPattern(op.getRhs(), m_One())) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, cast<ElementsAttr>(makeAttr(op.getType(), 0)));
      return success();
    }

    return failure();
  }
};

struct PowSimplify
    : public CheckedOpRewritePattern<stablehlo::PowOp, PowSimplify> {
  using CheckedOpRewritePattern<stablehlo::PowOp,
                                PowSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::PowOp op,
                                    PatternRewriter &rewriter) const {
    // pow(x, 1) -> x
    if (matchPattern(op.getRhs(), m_One()) ||
        matchPattern(op.getRhs(), m_OneFloat())) {
      rewriter.replaceAllUsesWith(op, op.getLhs());
      return success();
    }

    // pow(x, 0) -> 1 || pow(1, x) -> 1
    if ((matchPattern(op.getRhs(), m_Zero()) ||
         matchPattern(op.getRhs(), m_AnyZeroFloat())) ||
        (matchPattern(op.getLhs(), m_One()) ||
         matchPattern(op.getLhs(), m_OneFloat()))) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 1)));
      return success();
    }

    if (isa<FloatType>(op.getType().getElementType())) {
      DenseFPElementsAttr rhs;
      if (matchPattern(op.getRhs(), m_Constant(&rhs))) {
        bool allHalf = true, allNegOne = true, allNegHalf = true, allTwo = true;
        for (auto v : rhs) {
          allHalf &= v.isExactlyValue(0.5);
          allNegOne &= v.isExactlyValue(-1.0);
          allNegHalf &= v.isExactlyValue(-0.5);
          allTwo &= v.isExactlyValue(2.0);
        }

        // pow(X, -1) -> 1 / X
        if (allNegOne) {
          rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
              op,
              rewriter.create<stablehlo::ConstantOp>(
                  op.getLoc(), op.getType(),
                  cast<ElementsAttr>(makeAttr(op.getType(), 1))),
              op.getLhs());
          return success();
        }

        // pow(X, -0.5) -> rsqrt(X)
        if (allNegHalf) {
          rewriter.replaceOpWithNewOp<stablehlo::RsqrtOp>(op, op.getLhs());
          return success();
        }

        // pow(X, 0.5) -> sqrt(X)
        if (allHalf) {
          rewriter.replaceOpWithNewOp<stablehlo::SqrtOp>(op, op.getLhs());
          return success();
        }

        // pow(X, 2) -> X * X
        if (allTwo) {
          rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, op.getLhs(),
                                                        op.getLhs());
          return success();
        }
      }
    }

    return failure();
  }
};

struct NoNanZeroBasePowSimplify final
    : public CheckedOpRewritePattern<stablehlo::PowOp,
                                     NoNanZeroBasePowSimplify> {
  using CheckedOpRewritePattern<
      stablehlo::PowOp, NoNanZeroBasePowSimplify>::CheckedOpRewritePattern;

  NoNanZeroBasePowSimplify(bool allowOnFloatingPointMath, MLIRContext *context,
                           PatternBenefit benefit = 1)
      : CheckedOpRewritePattern(context, benefit),
        allowOnFloatingPointMath(allowOnFloatingPointMath) {}

  LogicalResult matchAndRewriteImpl(stablehlo::PowOp op,
                                    PatternRewriter &rewriter) const {
    if (!canApplyNoNanPattern(allowOnFloatingPointMath, op.getType())) {
      return failure();
    }

    if (matchPattern(op.getLhs(), m_Zero()) ||
        matchPattern(op.getLhs(), m_AnyZeroFloat())) {

      DenseElementsAttr attr;
      if (matchPattern(op.getRhs(), m_Constant(&attr)))
        return failure(); // let constant propagation handle this

      // 0 ^ x => x == 0 ? 1 : (x > 0 ? 0 : Inf)
      auto zero = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), rewriter.getZeroAttr(op.getType()));
      auto nonZeroCase = rewriter.create<stablehlo::SelectOp>(
          op.getLoc(),
          rewriter.create<stablehlo::CompareOp>(
              op.getLoc(), op.getRhs(), zero,
              stablehlo::ComparisonDirection::GT),
          zero,
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), op.getType(),
              cast<ElementsAttr>(makeAttr(
                  op.getType(), std::numeric_limits<float>::infinity()))));
      rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
          op,
          rewriter.create<stablehlo::CompareOp>(
              op.getLoc(), op.getRhs(), zero,
              stablehlo::ComparisonDirection::EQ),
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), op.getType(),
              cast<ElementsAttr>(makeAttr(op.getType(), 1))),
          nonZeroCase);
      return success();
    }

    return failure();
  }

private:
  bool allowOnFloatingPointMath;
};

bool is_broadcastable_compare(Value operand) {
  if (auto cmp = operand.getDefiningOp<stablehlo::CompareOp>()) {

    for (int i = 0; i < 2; i++) {
      auto v = cmp->getOperand(i);
      if (v.getDefiningOp<stablehlo::IotaOp>()) {
        continue;
      }
      DenseElementsAttr attr;
      if (matchPattern(v, m_Constant(&attr))) {
        if (attr.isSplat()) {
          continue;
        }
      }
      return false;
    }
    return true;
  }
  if (auto andv = operand.getDefiningOp<stablehlo::AndOp>()) {
    return is_broadcastable_compare(andv.getLhs()) &&
           is_broadcastable_compare(andv.getRhs());
  }
  return false;
}

struct BroadcastCompare
    : public CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                                     BroadcastCompare> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    if (!is_broadcastable_compare(op.getOperand()))
      return failure();

    auto operand = op.getOperand().getDefiningOp();

    Value newops[2];

    for (int i = 0; i < 2; i++) {
      auto v = operand->getOperand(i);
      auto RT = RankedTensorType::get(
          op.getType().getShape(),
          cast<RankedTensorType>(v.getType()).getElementType());
      newops[i] = rewriter.create<stablehlo::BroadcastInDimOp>(
          op.getLoc(), RT, v, op.getBroadcastDimensions());
    }

    if (auto cmp = op.getOperand().getDefiningOp<stablehlo::CompareOp>()) {
      auto cmp2 = rewriter.create<stablehlo::CompareOp>(
          cmp.getLoc(), newops[0], newops[1], cmp.getComparisonDirection());
      rewriter.replaceOp(op, cmp2);
    } else {
      auto and2 = rewriter.create<stablehlo::AndOp>(op.getOperand().getLoc(),
                                                    newops[0], newops[1]);
      rewriter.replaceOp(op, and2);
    }
    return success();
  }
};

struct IotaSimplify
    : public CheckedOpRewritePattern<stablehlo::IotaOp, IotaSimplify> {
  using CheckedOpRewritePattern<stablehlo::IotaOp,
                                IotaSimplify>::CheckedOpRewritePattern;

  size_t max_constant_expansion;
  IotaSimplify(size_t max_constant_expansion, MLIRContext *context,
               PatternBenefit benefit = 1,
               ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::IotaOp op,
                                    PatternRewriter &rewriter) const {
    size_t size = 1;
    for (auto sz : op.getType().getShape())
      size *= sz;
    if (size >= max_constant_expansion)
      return failure();

    auto out = stablehlo::iotaOp(op.getIotaDimension(), op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                       fromTensor(out));
    return success();
  }
};

struct ConcatToPad
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatToPad> {
  using CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                ConcatToPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getNumOperands() < 2)
      return failure();

    for (unsigned ind : {(unsigned int)0, op.getNumOperands() - 1}) {
      DenseElementsAttr inp;
      if (!matchPattern(op->getOperand(ind), m_Constant(&inp)))
        continue;
      if (!inp.isSplat())
        continue;

      auto subconcat = rewriter.create<stablehlo::ConcatenateOp>(
          op.getLoc(),
          (ind == 0) ? op.getOperands().drop_front()
                     : op.getOperands().drop_back(),
          op.getDimension());

      SmallVector<int64_t> low(op.getType().getShape().size(), 0);
      SmallVector<int64_t> high(op.getType().getShape().size(), 0);
      SmallVector<int64_t> interior(op.getType().getShape().size(), 0);
      if (ind == 0)
        low[op.getDimension()] = inp.getType().getShape()[op.getDimension()];
      else
        high[op.getDimension()] = inp.getType().getShape()[op.getDimension()];
      auto type0 = RankedTensorType::get({}, inp.getType().getElementType());
      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          op, op.getType(), subconcat,
          rewriter.create<stablehlo::ConstantOp>(op.getLoc(), type0,
                                                 inp.resizeSplat(type0)),
          low, high, interior);
      return success();
    }
    return failure();
  }
};

// reduce_window(pad(x, lo, hi, 0)) -> reduce_window(x, pad_lo=lo, pad_hi=hi)
struct PadReduceWindow
    : public CheckedOpRewritePattern<stablehlo::ReduceWindowOp,
                                     PadReduceWindow> {
  using CheckedOpRewritePattern<stablehlo::ReduceWindowOp,
                                PadReduceWindow>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceWindowOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() != 2)
      return failure();

    if (op.getPadding().has_value() &&
        !llvm::all_of(op.getPadding().value(),
                      [](auto pad) { return pad.isZero(); }))
      return failure();

    Value operand = op->getOperand(0), initValue = op->getOperand(1);

    auto padOp = operand.getDefiningOp<stablehlo::PadOp>();
    if (!padOp || !llvm::all_of(padOp.getInteriorPadding(),
                                [](int64_t pad) { return pad == 0; }))
      return failure();

    if (padOp.getPaddingValue() != initValue)
      return failure();

    auto highValues = padOp.getEdgePaddingHigh();
    auto lowValues = padOp.getEdgePaddingLow();

    int64_t N = highValues.size();

    SmallVector<int64_t> newPaddingValues(2 * N, 0);

    for (int i = 0; i < N; ++i) {
      newPaddingValues[2 * i] = lowValues[i];
      newPaddingValues[2 * i + 1] = highValues[i];
    }

    auto paddingType =
        mlir::RankedTensorType::get({N, 2}, rewriter.getI64Type());
    auto newPaddingAttr =
        mlir::DenseIntElementsAttr::get(paddingType, newPaddingValues);

    auto newOp = rewriter.create<stablehlo::ReduceWindowOp>(
        op.getLoc(), op.getResult(0).getType(), padOp.getOperand(), initValue,
        op.getWindowDimensionsAttr(), op.getWindowStridesAttr(),
        op.getBaseDilationsAttr(), op.getWindowDilationsAttr(), newPaddingAttr);
    newOp.getRegion().takeBody(op.getRegion());

    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ConcatPad
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatPad> {
  using CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                ConcatPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getNumOperands() < 2)
      return failure();

    for (unsigned ind : {(unsigned int)0, op.getNumOperands() - 1}) {

      auto pad = op->getOperand(ind).getDefiningOp<stablehlo::PadOp>();
      if (!pad)
        continue;

      if (pad.getInteriorPadding()[op.getDimension()] != 0)
        continue;

      if (ind == 0) {
        if (pad.getEdgePaddingHigh()[op.getDimension()] != 0)
          continue;
      } else {
        if (pad.getEdgePaddingLow()[op.getDimension()] != 0)
          continue;
      }

      bool legal = true;
      for (size_t i = 0; i < pad.getType().getShape().size(); i++) {
        if (i == op.getDimension())
          continue;
        if (pad.getInteriorPadding()[i] != 0) {
          legal = false;
          break;
        }
        if (pad.getEdgePaddingLow()[i] != 0) {
          legal = false;
          break;
        }
        if (pad.getEdgePaddingHigh()[i] != 0) {
          legal = false;
          break;
        }
      }

      if (!legal)
        continue;

      auto prevArgs = (ind == 0) ? op.getOperands().drop_front()
                                 : op.getOperands().drop_back();
      SmallVector<Value> subArgs(prevArgs.begin(), prevArgs.end());
      if (ind == 0)
        subArgs.insert(subArgs.begin(), pad.getOperand());
      else
        subArgs.push_back(pad.getOperand());

      auto subconcat = rewriter.create<stablehlo::ConcatenateOp>(
          op.getLoc(), subArgs, op.getDimension());

      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          op, op.getType(), subconcat, pad.getPaddingValue(),
          pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
          pad.getInteriorPadding());
      return success();
    }
    return failure();
  }
}; // namespace

struct ConvertSimplify
    : public CheckedOpRewritePattern<stablehlo::ConvertOp, ConvertSimplify> {
  using CheckedOpRewritePattern<stablehlo::ConvertOp,
                                ConvertSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      stablehlo::Tensor ten;
      RankedTensorType ty = op.getType();
      if (inp.isSplat()) {
        ten = stablehlo::makeTensor(inp.resizeSplat(
            RankedTensorType::get({}, inp.getType().getElementType())));
        ty = RankedTensorType::get({}, op.getType().getElementType());
      } else {
        ten = stablehlo::constantOp(inp);
      }
      auto out = fromTensor(stablehlo::convertOp(ten, ty));
      if (inp.isSplat())
        out = out.resizeSplat(op.getType());

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct SliceSimplify
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceSimplify> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceSimplify>::CheckedOpRewritePattern;

  static size_t getDenseElementBitWidth(Type eltType) {
    // Align the width for complex to 8 to make storage and interpretation
    // easier.
    if (ComplexType comp = llvm::dyn_cast<ComplexType>(eltType))
      return llvm::alignTo<8>(getDenseElementBitWidth(comp.getElementType())) *
             2;
    if (eltType.isIndex())
      return IndexType::kInternalStorageBitWidth;
    return eltType.getIntOrFloatBitWidth();
  }

  static size_t getDenseElementStorageWidth(size_t origWidth) {
    return origWidth == 1 ? origWidth : llvm::alignTo<8>(origWidth);
  }
  static size_t getDenseElementStorageWidth(Type elementType) {
    return getDenseElementStorageWidth(getDenseElementBitWidth(elementType));
  }

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      DenseElementsAttr out;
      if (inp.isSplat()) {
        out = inp.resizeSplat(op.getType());
      } else {
        bool contiguous = true;
        size_t offset = 0;
        auto inshape = op.getOperand().getType().getShape();
        auto outshape = op.getType().getShape();
        size_t total = 1;
        for (int i = 0; i < inshape.size(); i++) {
          if (op.getStrides()[i] != 1) {
            contiguous = false;
          }
          auto start = op.getStartIndices()[i];
          auto lim = op.getLimitIndices()[i];
          if (start != 0 || lim != inshape[i]) {
            if (offset != 0) {
              contiguous = false;
            }
          }
          offset *= inshape[i];
          offset += start;
          total *= outshape[i];
        }
        auto elementType = op.getOperand().getType().getElementType();
        auto bw = getDenseElementStorageWidth(elementType);
        if (contiguous && bw != 1) {
          const char *elementPtr = inp.getRawData().data() + (bw / 8) * offset;

          auto values = ArrayRef((char *)elementPtr, (bw / 8) * total);
          out =
              DenseIntOrFPElementsAttr::getFromRawBuffer(op.getType(), values);
        } else {
          auto ten = stablehlo::constantOp(inp);
          out = fromTensor(stablehlo::sliceOp(
              ten, stablehlo::Sizes(op.getStartIndices()),
              stablehlo::Sizes(op.getStrides()), op.getType()));
        }
      }
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }

    return failure();
  }
};

struct BroadcastInDimSimplify
    : public CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                                     BroadcastInDimSimplify> {
  using CheckedOpRewritePattern<
      stablehlo::BroadcastInDimOp,
      BroadcastInDimSimplify>::CheckedOpRewritePattern;

  size_t max_constant_expansion;
  BroadcastInDimSimplify(size_t max_constant_expansion, MLIRContext *context,
                         PatternBenefit benefit = 1,
                         ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      DenseElementsAttr out;
      if (inp.isSplat()) {
        out = inp.resizeSplat(op.getType());
      } else {
        size_t size = 1;
        for (auto sz : op.getType().getShape())
          size *= sz;
        if (size >= max_constant_expansion)
          return failure();
        auto ten = stablehlo::constantOp(inp);
        out = fromTensor(stablehlo::broadcastInDimOp(
            ten, stablehlo::Axes(op.getBroadcastDimensions()), op.getType()));
      }

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }

    return failure();
  }
};

stablehlo::ComparisonDirection
reversedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GE;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GT;
  }
}

struct CompareIotaConstSimplify
    : public CheckedOpRewritePattern<stablehlo::CompareOp,
                                     CompareIotaConstSimplify> {
  using CheckedOpRewritePattern<
      stablehlo::CompareOp, CompareIotaConstSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp cmpOp,
                                    PatternRewriter &rewriter) const {
    auto lhs = cmpOp.getLhs();
    auto rhs = cmpOp.getRhs();

    auto lhsIota = lhs.getDefiningOp<stablehlo::IotaOp>();
    auto rhsIota = rhs.getDefiningOp<stablehlo::IotaOp>();

    if ((!lhsIota && !rhsIota) || (lhsIota && rhsIota))
      return rewriter.notifyMatchFailure(cmpOp, "Requires single iota user");

    auto iota = lhsIota ? lhsIota : rhsIota;
    Value cst = lhsIota ? rhs : lhs;

    APInt cstAPInt;
    if (!matchPattern(cst, m_ConstantInt(&cstAPInt)))
      return rewriter.notifyMatchFailure(cmpOp, "Non-constant comparison");

    std::optional<stablehlo::ComparisonType> compType = cmpOp.getCompareType();

    int64_t cstI = compType == stablehlo::ComparisonType::SIGNED
                       ? cstAPInt.getSExtValue()
                       : cstAPInt.getZExtValue();
    auto dir = cmpOp.getComparisonDirection();

    auto T = cast<RankedTensorType>(iota.getType());

    ssize_t max_offset = T.getShape()[iota.getIotaDimension()] - 1;
    ssize_t min_offset = 0;

    if (lhs != iota) {
      dir = reversedComparisonDirection(dir);
    }

    auto boolType = rewriter.getI1Type();
    int64_t lb = 0, ub = T.getShape()[iota.getIotaDimension()];

    auto padInner = [&](bool valueIn, int64_t cstI, int64_t iotaDim) {
      SmallVector<int64_t> shape(T.getShape().begin(), T.getShape().end());
      SmallVector<int64_t> slow(T.getShape().size(), 0);
      SmallVector<int64_t> shigh(T.getShape().size(), 0);
      SmallVector<int64_t> sint(T.getShape().size(), 0);

      shape[iotaDim] = 1;
      slow[iotaDim] = cstI;
      shigh[iotaDim] = T.getShape()[iotaDim] - (cstI + 1);

      Value innerValue = rewriter.create<stablehlo::ConstantOp>(
          cmpOp.getLoc(),
          SplatElementsAttr::get(RankedTensorType::get(shape, boolType),
                                 rewriter.getBoolAttr(valueIn)));
      Value paddingValue = rewriter.create<stablehlo::ConstantOp>(
          cmpOp.getLoc(),
          SplatElementsAttr::get(RankedTensorType::get({}, boolType),
                                 rewriter.getBoolAttr(!valueIn)));

      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          cmpOp, innerValue, paddingValue, slow, shigh, sint);
    };

    switch (dir) {
    case stablehlo::ComparisonDirection::EQ:
      if (cstI < lb || cstI >= ub) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(false)));
        return success();
      }

      padInner(true, cstI, iota.getIotaDimension());
      return success();
    case stablehlo::ComparisonDirection::NE:
      if (cstI < lb || cstI >= ub) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(true)));
        return success();
      }

      padInner(false, cstI, iota.getIotaDimension());
      return success();
    case stablehlo::ComparisonDirection::LE:
    case stablehlo::ComparisonDirection::LT:
    case stablehlo::ComparisonDirection::GE:
    case stablehlo::ComparisonDirection::GT: {
      // iota <= cst [0, 1, 2, 3] .<= 2 -> [1, 1, 1, 0]

      // [0, 1, 2, 3] < 4   => all true if max_offset < cst
      if (dir == stablehlo::ComparisonDirection::LT && max_offset < cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(true)));
        return success();
      }
      // [0, 1, 2, 3] <= 3  => all true if max_offset <= cst
      if (dir == stablehlo::ComparisonDirection::LE && max_offset <= cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(true)));
        return success();
      }
      // [0, 1, 2, 3] > -1  => all true if min_offset > cst
      if (dir == stablehlo::ComparisonDirection::GT && min_offset > cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(true)));
        return success();
      }
      // [0, 1, 2, 3] >= 0  => all true if min_offset > cst
      if (dir == stablehlo::ComparisonDirection::GE && min_offset >= cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(true)));
        return success();
      }

      // [0, 1, 2, 3] < 0   => all false if min_offset >= cst
      if (dir == stablehlo::ComparisonDirection::LT && min_offset >= cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(false)));
        return success();
      }

      // [0, 1, 2, 3] <= -1  => all false if min_offset > cst
      if (dir == stablehlo::ComparisonDirection::LE && min_offset > cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(false)));
        return success();
      }

      // [0, 1, 2, 3] > 3   => all false if max_offset <= cst
      if (dir == stablehlo::ComparisonDirection::GT && max_offset <= cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(false)));
        return success();
      }

      // [0, 1, 2, 3] >= 4   => all false if max_offset < cst
      if (dir == stablehlo::ComparisonDirection::GE && max_offset < cstI) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            cmpOp, SplatElementsAttr::get(cmpOp.getType(),
                                          rewriter.getBoolAttr(false)));
        return success();
      }

      bool isLess = dir == stablehlo::ComparisonDirection::LE ||
                    dir == stablehlo::ComparisonDirection::LT;

      SmallVector<int64_t> leftShape;
      SmallVector<int64_t> rightShape;

      for (auto [i, S] : llvm::enumerate(T.getShape())) {
        if (i == iota.getIotaDimension()) {

          // [0, 1, 2, 3] ?= 2

          // is <= 2, left = [0, 1, 2]
          // is < 2, left = [0, 1]
          // is >= 2, left = [0, 1]
          // is > 2, left = [0, 1, 2]
          int left = (dir == stablehlo::ComparisonDirection::LE ||
                      dir == stablehlo::ComparisonDirection::GT)
                         ? cstI + 1
                         : cstI;
          leftShape.push_back(left);
          rightShape.push_back(S - left);
          continue;
        }

        leftShape.push_back(S);
        rightShape.push_back(S);
      }

      Value ops[] = {
          rewriter.create<stablehlo::ConstantOp>(
              iota.getLoc(),
              SplatElementsAttr::get(RankedTensorType::get(leftShape, boolType),
                                     rewriter.getBoolAttr(isLess))),
          rewriter.create<stablehlo::ConstantOp>(
              iota.getLoc(), SplatElementsAttr::get(
                                 RankedTensorType::get(rightShape, boolType),
                                 rewriter.getBoolAttr(!isLess)))};
      rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
          cmpOp, cmpOp.getType(), ValueRange(ops), iota.getIotaDimension());
      return success();
    }
    default:
      // TODO: other directions
      break;
    }

    return failure();
  }
};

struct BroadcastIotaSimplify
    : public CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                                     BroadcastIotaSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp broadcast,
                                    PatternRewriter &rewriter) const {
    auto operand = broadcast.getOperand();
    DenseIntElementsAttr input;
    matchPattern(operand, m_Constant(&input));

    auto RTO = cast<RankedTensorType>(operand.getType());
    if (input && !input.isSplat() && RTO.getShape().size() == 1 &&
        RTO.getShape()[0] >= 2) {
      auto elemType = input.getElementType();

      if (auto int_attr_arr = input.tryGetValues<::mlir::IntegerAttr>();
          llvm::succeeded(int_attr_arr)) {
        const auto end = int_attr_arr->end();
        auto curr = int_attr_arr->begin();
        auto next = int_attr_arr->begin();
        if (next++ == end)
          return failure();
        if (next == end)
          return failure();

        const auto start = (*curr).getValue().getSExtValue();
        const auto diff = (*next).getValue().getSExtValue() -
                          (*curr).getValue().getSExtValue();

        auto result_type = broadcast->getResultTypes();
        auto loc = broadcast.getLoc();

        // find the dimension to broadcast in
        int broadcast_dim = -1;
        auto result_shape =
            cast<mlir::ShapedType>(result_type.front()).getShape();
        auto max_dims = result_shape.size();

        if (broadcast.getType().getElementType().isInteger(1)) {
          // true, false, .... false.  -> iota == 0
          if (start != 0 && (*next).getValue().isZero()) {
            bool legal = true;
            for (auto idx = next; idx != end; idx++) {
              if (!(*idx).getValue().isZero()) {
                legal = false;
                break;
              }
            }
            // only 1 at the start
            if (legal) {
              auto ITy = RankedTensorType::get(
                  result_shape, rewriter.getIntegerType(32, false));
              auto iota = rewriter.create<stablehlo::IotaOp>(
                  loc, ITy, broadcast.getBroadcastDimensions()[0]);
              auto cmp = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy, cast<ElementsAttr>(makeAttr(ITy, 0))),
                  stablehlo::ComparisonDirection::EQ);
              rewriter.replaceOp(broadcast, cmp);
              return success();
            }
          }

          // false, true, .... true.  -> iota != 0
          if (start == 0 && !(*next).getValue().isZero()) {
            bool legal = true;
            for (auto idx = next; idx != end; idx++) {
              if ((*idx).getValue().isZero()) {
                legal = false;
                break;
              }
            }
            // only 1 at the start
            if (legal) {
              auto ITy = RankedTensorType::get(
                  result_shape, rewriter.getIntegerType(32, false));
              auto iota = rewriter.create<stablehlo::IotaOp>(
                  loc, ITy, broadcast.getBroadcastDimensions()[0]);
              auto cmp = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy, cast<ElementsAttr>(makeAttr(ITy, 0))),
                  stablehlo::ComparisonDirection::NE);
              rewriter.replaceOp(broadcast, cmp);
              return success();
            }
          }

          // false, .... false, true -> iota == end
          auto lastVal = (*(--int_attr_arr->end())).getInt();
          if (lastVal != 0) {
            bool legal = true;
            for (auto idx = int_attr_arr->begin();;) {
              if (!(*idx).getValue().isZero()) {
                legal = false;
                break;
              }
              idx++;
              auto nextv = idx;
              nextv++;
              if (nextv == end) {
                break;
              }
            }
            // only 1 at the end
            if (legal) {
              auto ITy = RankedTensorType::get(
                  result_shape, rewriter.getIntegerType(32, false));
              auto iota = rewriter.create<stablehlo::IotaOp>(
                  loc, ITy, broadcast.getBroadcastDimensions()[0]);
              auto cmp = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy,
                      cast<ElementsAttr>(makeAttr(ITy, RTO.getShape()[0] - 1))),
                  stablehlo::ComparisonDirection::EQ);
              rewriter.replaceOp(broadcast, cmp);
              return success();
            }
          }

          // true, .... true, false  -> iota != end
          if (lastVal == 0) {
            bool legal = true;
            for (auto idx = int_attr_arr->begin();;) {
              if ((*idx).getValue().isZero()) {
                legal = false;
                break;
              }
              idx++;
              auto nextv = idx;
              nextv++;
              if (nextv == end) {
                break;
              }
            }
            // only 1 at the end
            if (legal) {
              auto ITy = RankedTensorType::get(
                  result_shape, rewriter.getIntegerType(32, false));
              auto iota = rewriter.create<stablehlo::IotaOp>(
                  loc, ITy, broadcast.getBroadcastDimensions()[0]);
              auto cmp = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy,
                      cast<ElementsAttr>(makeAttr(ITy, RTO.getShape()[0] - 1))),
                  stablehlo::ComparisonDirection::NE);
              rewriter.replaceOp(broadcast, cmp);
              return success();
            }
          }

          // false, false, ..., true, true, true, ..., false, false, ...
          if (lastVal == 0 && start == 0) {
            bool legal = true;
            int firstTrue = -1;
            int firstFalseAgain = -1;
            int i = 0;
            for (auto idx = int_attr_arr->begin(); idx != end; i++, idx++) {
              bool isTrue = !(*idx).getValue().isZero();
              // Before the transition.
              if (firstTrue == -1) {
                if (isTrue) {
                  firstTrue = i;
                  continue;
                } else {
                  continue;
                }
              }
              // In the true region
              if (firstFalseAgain == -1) {
                if (isTrue) {
                  continue;
                } else {
                  firstFalseAgain = i;
                  continue;
                }
              }

              // Post transition
              if (isTrue) {
                legal = false;
                break;
              }
              continue;
            }
            if (legal && firstTrue != -1 && firstFalseAgain != -1) {
              auto ITy = RankedTensorType::get(
                  result_shape, rewriter.getIntegerType(32, false));
              auto iota = rewriter.create<stablehlo::IotaOp>(
                  loc, ITy, broadcast.getBroadcastDimensions()[0]);
              auto cmp1 = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy, cast<ElementsAttr>(makeAttr(ITy, firstTrue))),
                  stablehlo::ComparisonDirection::GE);
              auto cmp2 = rewriter.create<stablehlo::CompareOp>(
                  loc, iota,
                  rewriter.create<stablehlo::ConstantOp>(
                      loc, ITy,
                      cast<ElementsAttr>(makeAttr(ITy, firstFalseAgain))),
                  stablehlo::ComparisonDirection::LT);
              rewriter.replaceOpWithNewOp<stablehlo::AndOp>(broadcast, cmp1,
                                                            cmp2);
              return success();
            }
          }
        }

        for (broadcast_dim = 0; broadcast_dim < max_dims; ++broadcast_dim) {
          bool found = false;
          for (auto &elem : broadcast.getBroadcastDimensions()) {
            if (elem == broadcast_dim) {
              found = true;
              break;
            }
          }
          if (!found)
            break;
        }
        assert(broadcast_dim != -1);

        if (diff == 0)
          return failure();

        while (next != end) {
          auto curr_diff = (*next).getValue().getSExtValue() -
                           (*curr).getValue().getSExtValue();
          if (curr_diff != diff)
            return failure();
          ++curr;
          ++next;
        }

        // build the replacement operations
        auto iota =
            rewriter.create<stablehlo::IotaOp>(loc, result_type, broadcast_dim);
        auto stride_attr = mlir::DenseElementsAttr::get(
            operand.getType().cloneWith(result_shape, elemType),
            rewriter.getIntegerAttr(elemType, diff));
        auto start_attr = mlir::DenseElementsAttr::get(
            operand.getType().cloneWith(result_shape, elemType),
            rewriter.getIntegerAttr(elemType, start));
        auto stride_const = rewriter.create<stablehlo::ConstantOp>(
            loc, result_type, stride_attr);
        auto start_const = rewriter.create<stablehlo::ConstantOp>(
            loc, result_type, start_attr);
        auto mul = rewriter.create<stablehlo::MulOp>(loc, iota, stride_const);

        rewriter.replaceOpWithNewOp<stablehlo::AddOp>(broadcast, start_const,
                                                      mul);
        return success();
      }
      return failure();
    }

    return failure();
  }
};

struct DotGeneralSimplify
    : public CheckedOpRewritePattern<stablehlo::DotGeneralOp,
                                     DotGeneralSimplify> {
  using CheckedOpRewritePattern<stablehlo::DotGeneralOp,
                                DotGeneralSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DotGeneralOp op,
                                    PatternRewriter &rewriter) const {
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_AnyZeroComplex()) ||
        matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_AnyZeroComplex())) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }
    return failure();
  }
};

struct TransposeSimplify
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeSimplify> {
  using CheckedOpRewritePattern<stablehlo::TransposeOp,
                                TransposeSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {

      DenseElementsAttr out;
      if (inp.isSplat()) {
        out = inp.resizeSplat(op.getType());
      } else {
        out = fromTensor(stablehlo::transposeOp(
            stablehlo::constantOp(inp), stablehlo::Axes(op.getPermutation()),
            op.getType()));
      }
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct MaxSimplify
    : public CheckedOpRewritePattern<stablehlo::MaxOp, MaxSimplify> {
  using CheckedOpRewritePattern<stablehlo::MaxOp,
                                MaxSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MaxOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getOperand(0) == op.getOperand(1)) {
      rewriter.replaceOp(op, op.getOperand(0));
      return success();
    }

    return failure();
  }
};

struct MinSimplify
    : public CheckedOpRewritePattern<stablehlo::MinOp, MinSimplify> {
  using CheckedOpRewritePattern<stablehlo::MinOp,
                                MinSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MinOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getOperand(0) == op.getOperand(1)) {
      rewriter.replaceOp(op, op.getOperand(0));
      return success();
    }

    return failure();
  }
};

template <typename T>
struct BinBroadcastSplat final
    : CheckedOpRewritePattern<T, BinBroadcastSplat<T>> {
  using CheckedOpRewritePattern<T,
                                BinBroadcastSplat<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    for (int i = 0; i < 2; i++) {
      mlir::Value opi = op->getOperand(i);
      if (auto broadcast = opi.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
        SplatElementsAttr other;
        if (matchPattern(op->getOperand(1 - i), m_Constant(&other))) {
          IRMapping map;
          mlir::Value vals[2];
          vals[i] = broadcast.getOperand();
          vals[1 - i] = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), broadcast.getOperand().getType(),
              other.resizeSplat(broadcast.getOperand().getType()));
          auto pushed = rewriter.create<T>(op.getLoc(), vals[0], vals[1]);
          map.map(broadcast.getOperand(), pushed->getResult(0));
          auto bc2 = rewriter.clone(*broadcast, map);
          rewriter.replaceOp(op, bc2);
          return success();
        }
      }
    }
    return failure();
  }
};

struct AllFiniteIsFinite
    : public CheckedOpRewritePattern<stablehlo::IsFiniteOp, AllFiniteIsFinite> {
  using CheckedOpRewritePattern<stablehlo::IsFiniteOp,
                                AllFiniteIsFinite>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IsFiniteOp op,
                                    PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 1)));
    return success();
  }
};

struct AllFiniteIsInf
    : public CheckedOpRewritePattern<chlo::IsInfOp, AllFiniteIsInf> {
  using CheckedOpRewritePattern<chlo::IsInfOp,
                                AllFiniteIsInf>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::IsInfOp op,
                                    PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, rewriter.getZeroAttr(op.getType()));
    return success();
  }
};

struct AllFiniteIsPosInf
    : public CheckedOpRewritePattern<chlo::IsPosInfOp, AllFiniteIsPosInf> {
  using CheckedOpRewritePattern<chlo::IsPosInfOp,
                                AllFiniteIsPosInf>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::IsPosInfOp op,
                                    PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, rewriter.getZeroAttr(op.getType()));
    return success();
  }
};

struct AllFiniteIsNegInf
    : public CheckedOpRewritePattern<chlo::IsNegInfOp, AllFiniteIsNegInf> {
  using CheckedOpRewritePattern<chlo::IsNegInfOp,
                                AllFiniteIsNegInf>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::IsNegInfOp op,
                                    PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, rewriter.getZeroAttr(op.getType()));
    return success();
  }
};

struct NoNan : public CheckedOpRewritePattern<stablehlo::CompareOp, NoNan> {
  using CheckedOpRewritePattern<stablehlo::CompareOp,
                                NoNan>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getLhs() == op.getRhs()) {
      if (op.getComparisonDirection() == stablehlo::ComparisonDirection::EQ) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 1)));
        return success();
      }
      if (op.getComparisonDirection() == stablehlo::ComparisonDirection::NE) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), cast<ElementsAttr>(makeAttr(op.getType(), 0)));
        return success();
      }
    }
    return failure();
  }
};

struct TransposeTranspose
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeTranspose> {
  using CheckedOpRewritePattern<stablehlo::TransposeOp,
                                TransposeTranspose>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto operand = op.getOperand();

    auto convertOp = operand.getDefiningOp<stablehlo::ConvertOp>();
    if (convertOp) {
      operand = convertOp.getOperand();
    }

    auto definingTranspose = operand.getDefiningOp<stablehlo::TransposeOp>();
    if (!definingTranspose)
      return rewriter.notifyMatchFailure(op, "not a transpose(transpose)");

    llvm::ArrayRef<int64_t> thisPermutation = op.getPermutation();
    llvm::ArrayRef<int64_t> prevPermutation =
        definingTranspose.getPermutation();

    SmallVector<int64_t> newPermutation;
    newPermutation.resize(thisPermutation.size());
    for (unsigned i = 0, e = thisPermutation.size(); i != e; ++i) {
      newPermutation[i] = prevPermutation[thisPermutation[i]];
    }

    if (!convertOp) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setPermutation(newPermutation);
        op.setOperand(definingTranspose.getOperand());
      });
    } else {
      auto midPerm = rewriter.create<stablehlo::TransposeOp>(
          op.getLoc(), definingTranspose.getOperand(), newPermutation);
      rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, op.getType(),
                                                        midPerm);
    }
    return success();
  }
};

size_t getBitWidth(mlir::Type ty) {
  if (auto CT = dyn_cast<ComplexType>(ty)) {
    return 2 * getBitWidth(CT.getElementType());
  }
  return ty.getIntOrFloatBitWidth();
}

struct TransposeConvert
    : public CheckedOpRewritePattern<stablehlo::ConvertOp, TransposeConvert> {
  using CheckedOpRewritePattern<stablehlo::ConvertOp,
                                TransposeConvert>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) const {
    auto resultType = cast<TensorType>(op.getResult().getType());
    auto operandType = cast<TensorType>(op.getOperand().getType());
    if (!resultType.hasStaticShape() || !operandType.hasStaticShape())
      return failure();
    if (resultType.getNumElements() *
            getBitWidth(resultType.getElementType()) >=
        operandType.getNumElements() *
            getBitWidth(operandType.getElementType()))
      return failure();

    auto transpose = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transpose || !llvm::hasSingleElement(transpose->getUsers()))
      return failure();

    auto newConvert = rewriter.create<stablehlo::ConvertOp>(
        op.getLoc(), transpose.getOperand(), resultType.getElementType());
    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        transpose.getLoc(), newConvert.getResult(), transpose.getPermutation());
    rewriter.replaceOp(op, newTranspose);
    rewriter.eraseOp(transpose);

    return success();
  }
};

struct TransposeDotReorder
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeDotReorder> {
  using CheckedOpRewritePattern<stablehlo::TransposeOp,
                                TransposeDotReorder>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {

    auto operand = op.getOperand();
    auto convert = operand.getDefiningOp<stablehlo::ConvertOp>();
    if (convert) {
      operand = convert.getOperand();
      if (!llvm::hasSingleElement(convert->getUsers()))
        return failure();
    }

    auto dot = operand.getDefiningOp<stablehlo::DotGeneralOp>();
    if (!dot || !llvm::hasSingleElement(dot->getUsers()))
      return failure();

    auto perm = op.getPermutation();
    auto dimensionNumbers = dot.getDotDimensionNumbers();

    size_t permidx = 0;

    for (size_t bidx = 0,
                end = dimensionNumbers.getLhsBatchingDimensions().size();
         bidx < end; bidx++) {
      if (perm[permidx] != bidx)
        return failure();
      permidx++;
    }

    size_t numLHSResults =
        dot.getLhs().getType().getShape().size() -
        dimensionNumbers.getLhsBatchingDimensions().size() -
        dimensionNumbers.getLhsContractingDimensions().size();

    {
      size_t residx = 0;
      for (size_t ridx = 0, end = dot.getRhs().getType().getShape().size();
           ridx < end; ridx++) {
        if (llvm::is_contained(dimensionNumbers.getRhsBatchingDimensions(),
                               ridx))
          continue;
        if (llvm::is_contained(dimensionNumbers.getRhsContractingDimensions(),
                               ridx))
          continue;
        if (perm[permidx] !=
            dimensionNumbers.getLhsBatchingDimensions().size() + numLHSResults +
                residx)
          return failure();
        permidx++;
        residx++;
      }
    }

    {
      size_t residx = 0;
      for (size_t lidx = 0, end = dot.getLhs().getType().getShape().size();
           lidx < end; lidx++) {
        if (llvm::is_contained(dimensionNumbers.getLhsBatchingDimensions(),
                               lidx))
          continue;
        if (llvm::is_contained(dimensionNumbers.getLhsContractingDimensions(),
                               lidx))
          continue;
        if (perm[permidx] !=
            dimensionNumbers.getLhsBatchingDimensions().size() + residx)
          return failure();
        permidx++;
        residx++;
      }
    }

    auto ndim = stablehlo::DotDimensionNumbersAttr::get(
        dimensionNumbers.getContext(),
        dimensionNumbers.getRhsBatchingDimensions(),
        dimensionNumbers.getLhsBatchingDimensions(),
        dimensionNumbers.getRhsContractingDimensions(),
        dimensionNumbers.getLhsContractingDimensions());
    if (!convert) {
      rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
          op, op.getType(), dot.getRhs(), dot.getLhs(), ndim,
          dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
    } else {
      auto middot = rewriter.create<stablehlo::DotGeneralOp>(
          op.getLoc(),
          RankedTensorType::get(op.getType().getShape(),
                                dot.getType().getElementType()),
          dot.getRhs(), dot.getLhs(), ndim, dot.getPrecisionConfigAttr(),
          dot.getAlgorithmAttr());
      rewriter.replaceOpWithNewOp<stablehlo::ConvertOp>(op, op.getType(),
                                                        middot);
    }
    return success();
  }
};

struct TransposeReduce
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeReduce> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp transpose,
                                    PatternRewriter &rewriter) const {
    auto operand = transpose.getOperand();
    auto reduce = operand.getDefiningOp<stablehlo::ReduceOp>();
    if (!reduce)
      return failure();

    for (auto result : reduce->getResults())
      if (!llvm::hasSingleElement(result.getUsers()))
        return failure();

    unsigned resultNum = std::distance(
        reduce.getResults().begin(), llvm::find(reduce.getResults(), operand));

    auto reduceDims = reduce.getDimensions();
    auto reduceInput = reduce.getInputs()[resultNum];
    auto reduceInputType = dyn_cast<RankedTensorType>(reduceInput.getType());
    if (!reduceInputType)
      return rewriter.notifyMatchFailure(reduce, "Reduce input not tensor");

    auto transposePermutation = transpose.getPermutation();

    SmallVector<int64_t> newTransposePermutation(transposePermutation);

    for (int64_t reduceDim : reduceDims) {
      for (auto &transposeDim : newTransposePermutation) {
        if (reduceDim <= transposeDim)
          transposeDim++;
      }
      newTransposePermutation.insert(
          std::next(newTransposePermutation.begin(), reduceDim), reduceDim);
    }

    rewriter.setInsertionPoint(reduce);
    auto newTransposeOp = rewriter.create<stablehlo::TransposeOp>(
        transpose.getLoc(), reduceInput, newTransposePermutation);

    SmallVector<Type> newReduceResultTypes(reduce.getResultTypes());
    newReduceResultTypes[resultNum] = transpose.getResult().getType();
    SmallVector<Value> newReduceInputs(reduce.getInputs());
    newReduceInputs[resultNum] = newTransposeOp.getResult();

    auto newReduce = rewriter.create<stablehlo::ReduceOp>(
        reduce.getLoc(), newReduceResultTypes, newReduceInputs,
        reduce.getInitValues(), reduceDims);
    rewriter.inlineRegionBefore(reduce.getRegion(), newReduce.getRegion(),
                                newReduce.getRegion().begin());
    for (auto [i, oldRes, newRes] :
         llvm::enumerate(reduce.getResults(), newReduce.getResults())) {
      if (i == resultNum) {
        rewriter.replaceOp(transpose, newRes);
      } else {
        rewriter.replaceAllUsesWith(oldRes, newRes);
      }
    }
    rewriter.eraseOp(reduce);

    return success();
  }
};

struct TransposeConvolution
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeConvolution> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp transpose,
                                    PatternRewriter &rewriter) const {
    auto operand = transpose.getOperand();
    auto conv = operand.getDefiningOp<stablehlo::ConvolutionOp>();
    if (!conv || !llvm::hasSingleElement(operand.getUsers()))
      return failure();

    auto permutation = transpose.getPermutation();
    SmallVector<int64_t> mapping(permutation.size(), -1);
    for (int64_t i = 0; i < permutation.size(); i++)
      mapping[permutation[i]] = i;

    auto dimensionNumbers = conv.getDimensionNumbers();

    int64_t outputBatchDimension =
        mapping[dimensionNumbers.getOutputBatchDimension()];
    int64_t outputFeatureDimension =
        mapping[dimensionNumbers.getOutputFeatureDimension()];

    SmallVector<int64_t> outputSpatialDimensions(
        dimensionNumbers.getOutputSpatialDimensions().begin(),
        dimensionNumbers.getOutputSpatialDimensions().end());
    for (auto &dim : outputSpatialDimensions)
      dim = mapping[dim];

    auto newDimensionNumbers = stablehlo::ConvDimensionNumbersAttr::get(
        dimensionNumbers.getContext(),
        dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(), outputBatchDimension,
        outputFeatureDimension, outputSpatialDimensions);

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        transpose, transpose.getType(), conv.getLhs(), conv.getRhs(),
        conv.getWindowStridesAttr(), conv.getPaddingAttr(),
        conv.getLhsDilationAttr(), conv.getRhsDilationAttr(),
        conv.getWindowReversalAttr(), newDimensionNumbers,
        conv.getFeatureGroupCountAttr(), conv.getBatchGroupCountAttr(),
        conv.getPrecisionConfigAttr());

    return success();
  }
};

struct ConvolutionTranspose
    : public CheckedOpRewritePattern<stablehlo::ConvolutionOp,
                                     ConvolutionTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvolutionOp conv,
                                    PatternRewriter &rewriter) const {
    auto lhs_trans = conv.getLhs().getDefiningOp<stablehlo::TransposeOp>();
    auto rhs_trans = conv.getRhs().getDefiningOp<stablehlo::TransposeOp>();
    if (!lhs_trans && !rhs_trans)
      return failure();

    auto dim = conv.getDimensionNumbers();
    int64_t inputBatchDimension = dim.getInputBatchDimension();
    int64_t inputFeatureDimension = dim.getInputFeatureDimension();
    SmallVector<int64_t> inputSpatialDimensions(
        dim.getInputSpatialDimensions().begin(),
        dim.getInputSpatialDimensions().end());

    int64_t kernelInputFeatureDimension = dim.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDimension =
        dim.getKernelOutputFeatureDimension();
    SmallVector<int64_t> kernelSpatialDimensions(
        dim.getKernelSpatialDimensions().begin(),
        dim.getKernelSpatialDimensions().end());

    if (lhs_trans) {
      inputBatchDimension = lhs_trans.getPermutation()[inputBatchDimension];
      inputFeatureDimension = lhs_trans.getPermutation()[inputFeatureDimension];
      for (auto &dim : inputSpatialDimensions)
        dim = lhs_trans.getPermutation()[dim];
    }

    if (rhs_trans) {
      kernelInputFeatureDimension =
          rhs_trans.getPermutation()[kernelInputFeatureDimension];
      kernelOutputFeatureDimension =
          rhs_trans.getPermutation()[kernelOutputFeatureDimension];
      for (auto &dim : kernelSpatialDimensions)
        dim = rhs_trans.getPermutation()[dim];
    }

    auto ndim = stablehlo::ConvDimensionNumbersAttr::get(
        dim.getContext(), inputBatchDimension, inputFeatureDimension,
        inputSpatialDimensions, kernelInputFeatureDimension,
        kernelOutputFeatureDimension, kernelSpatialDimensions,
        dim.getOutputBatchDimension(), dim.getOutputFeatureDimension(),
        dim.getOutputSpatialDimensions());

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        conv, conv.getType(),
        lhs_trans ? lhs_trans.getOperand() : conv.getLhs(),
        rhs_trans ? rhs_trans.getOperand() : conv.getRhs(),

        conv.getWindowStridesAttr(), conv.getPaddingAttr(),
        conv.getLhsDilationAttr(), conv.getRhsDilationAttr(),
        conv.getWindowReversalAttr(), ndim, conv.getFeatureGroupCountAttr(),
        conv.getBatchGroupCountAttr(), conv.getPrecisionConfigAttr());
    return success();
  }
};

// transpose(einsum) -> einsum
struct TransposeEinsum
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeEinsum> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp transpose,
                                    PatternRewriter &rewriter) const {
    auto operand = transpose.getOperand();
    auto einsum = operand.getDefiningOp<stablehlo::EinsumOp>();
    if (!einsum || !llvm::hasSingleElement(operand.getUsers()))
      return failure();

    auto einsumConfig = einsum.getEinsumConfig();
    auto arrowPos = einsumConfig.find("->");

    if (arrowPos == StringRef::npos)
      return failure();

    auto permutation = transpose.getPermutation();

    if (einsumConfig.size() - (arrowPos + 2) < permutation.size())
      return failure();

    auto newEinsumConfig = std::string(einsumConfig.str());
    for (int i = 0; i < permutation.size(); ++i) {
      newEinsumConfig[arrowPos + 2 + i] =
          einsumConfig[arrowPos + 2 + permutation[i]];
    }

    rewriter.modifyOpInPlace(einsum, [&einsum, &transpose, newEinsumConfig] {
      einsum.setEinsumConfig(
          StringAttr::get(einsum.getContext(), newEinsumConfig));
      einsum.getResult().setType(transpose.getType());
    });
    rewriter.replaceAllUsesWith(transpose.getResult(), einsum.getResult());
    return success();
  }
};

// einsum(transpose(x), transpose(y)) -> einsum(x, y)
struct EinsumTranspose
    : public CheckedOpRewritePattern<stablehlo::EinsumOp, EinsumTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::EinsumOp einsum,
                                    PatternRewriter &rewriter) const {
    llvm::StringRef einsumConfig = einsum.getEinsumConfig();

    auto lhs_trans = einsum.getLhs().getDefiningOp<stablehlo::TransposeOp>();
    auto rhs_trans = einsum.getRhs().getDefiningOp<stablehlo::TransposeOp>();
    if (!lhs_trans && !rhs_trans)
      return failure();

    size_t commaPos = einsumConfig.find(",");
    size_t arrowPos = einsumConfig.find("->");
    if (commaPos != einsum.getLhs().getType().getRank() ||
        einsumConfig.size() - commaPos < einsum.getRhs().getType().getRank() ||
        (arrowPos != StringRef::npos &&
         arrowPos - commaPos < einsum.getRhs().getType().getRank()))
      return failure();

    auto newEinsumConfig = std::string(einsumConfig.str());

    if (lhs_trans) {
      for (int i = 0; i < commaPos; ++i) {
        newEinsumConfig[i] = einsumConfig[lhs_trans.getPermutation()[i]];
      }
    }

    if (rhs_trans) {
      int64_t rhsRank = einsum.getRhs().getType().getRank();
      for (int i = 0; i < rhsRank; ++i) {
        newEinsumConfig[commaPos + 1 + i] =
            einsumConfig[commaPos + 1 + rhs_trans.getPermutation()[i]];
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::EinsumOp>(
        einsum, einsum.getType(),
        lhs_trans ? lhs_trans.getOperand() : einsum.getLhs(),
        rhs_trans ? rhs_trans.getOperand() : einsum.getRhs(),
        StringAttr::get(einsum.getContext(), newEinsumConfig));
    return success();
  }
};

struct DotTranspose
    : public CheckedOpRewritePattern<stablehlo::DotGeneralOp, DotTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DotGeneralOp dot,
                                    PatternRewriter &rewriter) const {

    auto dim = dot.getDotDimensionNumbers();
    size_t numLHSResults = dot.getLhs().getType().getRank() -
                           dim.getLhsBatchingDimensions().size() -
                           dim.getLhsContractingDimensions().size();

    size_t numRHSResults = dot.getRhs().getType().getRank() -
                           dim.getRhsBatchingDimensions().size() -
                           dim.getRhsContractingDimensions().size();

    auto lhsTrans = dot.getLhs().getDefiningOp<stablehlo::TransposeOp>();
    bool fuseLhs = false;
    if (lhsTrans)
      fuseLhs = numLHSResults <= 1; // otherwise we need to transpose the result

    auto rhsTrans = dot.getRhs().getDefiningOp<stablehlo::TransposeOp>();
    bool fuseRhs = false;
    if (rhsTrans)
      fuseRhs = numRHSResults <= 1; // otherwise we need to transpose the result

    if (!fuseLhs && !fuseRhs)
      return failure();

    SmallVector<int64_t> lhsBatch(dim.getLhsBatchingDimensions().begin(),
                                  dim.getLhsBatchingDimensions().end());
    SmallVector<int64_t> rhsBatch(dim.getRhsBatchingDimensions().begin(),
                                  dim.getRhsBatchingDimensions().end());

    SmallVector<int64_t> lhsContract(dim.getLhsContractingDimensions().begin(),
                                     dim.getLhsContractingDimensions().end());
    SmallVector<int64_t> rhsContract(dim.getRhsContractingDimensions().begin(),
                                     dim.getRhsContractingDimensions().end());

    if (fuseLhs) {
      for (auto &dim : lhsBatch)
        dim = lhsTrans.getPermutation()[dim];
      for (auto &dim : lhsContract)
        dim = lhsTrans.getPermutation()[dim];
    }

    if (fuseRhs) {
      for (auto &dim : rhsBatch)
        dim = rhsTrans.getPermutation()[dim];
      for (auto &dim : rhsContract)
        dim = rhsTrans.getPermutation()[dim];
    }

    auto ndim = stablehlo::DotDimensionNumbersAttr::get(
        dim.getContext(), lhsBatch, rhsBatch, lhsContract, rhsContract);

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        dot, dot.getType(), fuseLhs ? lhsTrans.getOperand() : dot.getLhs(),
        fuseRhs ? rhsTrans.getOperand() : dot.getRhs(), ndim,
        dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
    return success();
  }
};

struct BroadcastReduce
    : public CheckedOpRewritePattern<stablehlo::ReduceOp, BroadcastReduce> {
  using CheckedOpRewritePattern<stablehlo::ReduceOp,
                                BroadcastReduce>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-operand single-init reduce is supported");
    }
    // TODO: min/max can also be an option since they are dropped
    if (!isa<stablehlo::AddOp>(op.getRegion().getBlocks().front().front())) {
      return rewriter.notifyMatchFailure(op, "only add is currently supported");
    }

    Value input = op.getInputs()[0];
    auto inputType = cast<TensorType>(input.getType());
    auto broadcast = input.getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!broadcast) {
      return rewriter.notifyMatchFailure(op,
                                         "input source is not a broadcast op");
    }

    // If any of the dimensions that are being reduced was initially
    // broadcasted, we can multiply the result with the dimension instead.
    ArrayRef<int64_t> broadcastDims = broadcast.getBroadcastDimensions();
    SmallVector<int64_t> broadcastFromNothingDims, broadcastFromOneDims;
    auto broadcastSourceType =
        cast<TensorType>(broadcast.getOperand().getType());
    for (int64_t reductionDim : op.getDimensions()) {
      if (inputType.isDynamicDim(reductionDim))
        continue;
      auto it = llvm::find(broadcastDims, reductionDim);
      if (it == broadcastDims.end()) {
        broadcastFromNothingDims.push_back(reductionDim);
        continue;
      }
      size_t originalDim = std::distance(broadcastDims.begin(), it);
      if (broadcastSourceType.getDimSize(originalDim) == 1 &&
          inputType.getDimSize(reductionDim) != 1) {
        broadcastFromOneDims.push_back(reductionDim);
      }
    }
    if (broadcastFromNothingDims.empty() && broadcastFromOneDims.empty())
      return rewriter.notifyMatchFailure(op, "no dimensions to remove");

    int64_t size = 1;
    for (int64_t dim : broadcastFromNothingDims) {
      size *= inputType.getDimSize(dim);
    }
    for (int64_t dim : broadcastFromOneDims) {
      size *= inputType.getDimSize(dim);
    }

    int64_t numRemoved = 0;
    SmallVector<int64_t> newReduceDimensions;
    llvm::sort(broadcastFromNothingDims);
    for (int64_t reductionDim : op.getDimensions()) {
      if (llvm::is_contained(broadcastFromNothingDims, reductionDim)) {
        numRemoved++;
        continue;
      }
      newReduceDimensions.push_back(reductionDim - numRemoved);
    }

    assert(numRemoved == broadcastFromNothingDims.size());

    Value operand = broadcast.getOperand();
    // Some of the broadcasted values aren't being reduced over
    // if (broadcastFromNothingDims.size() + broadcastFromOneDims.size() +
    // cast<RankedTensorType>(op.getType(0)).getShape().size() !=
    // inputType.getShape().size()) {
    SmallVector<int64_t> newBroadcast(
        cast<RankedTensorType>(broadcast.getOperand().getType())
            .getShape()
            .size(),
        -1);
    SmallVector<int64_t> newShape;
    for (int i = 0; i < inputType.getShape().size(); i++) {
      // These dimensions are removed
      if (llvm::is_contained(broadcastFromNothingDims, i)) {
        continue;
      }
      auto it = llvm::find(broadcastDims, i);
      if (llvm::is_contained(broadcastFromOneDims, i))
        newShape.push_back(1);
      else
        newShape.push_back(inputType.getShape()[i]);
      if (it == broadcastDims.end()) {
        // This dimension does not come from an earlier input
      } else {
        size_t originalDim = std::distance(broadcastDims.begin(), it);
        newBroadcast[originalDim] = newShape.size() - 1;
      }
    }
    operand = rewriter.create<stablehlo::BroadcastInDimOp>(
        op.getLoc(),
        RankedTensorType::get(newShape, inputType.getElementType()), operand,
        newBroadcast);
    // }

    auto newReduction = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), op->getResultTypes(), ValueRange{operand},
        op.getInitValues(), newReduceDimensions);
    newReduction.getRegion().takeBody(op.getRegion());

    auto newResultType = cast<TensorType>(newReduction.getResult(0).getType());
    auto constantInt = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(),
        makeAttr(newResultType.clone(rewriter.getI64Type()), size));
    auto converted = rewriter.create<stablehlo::ConvertOp>(
        op.getLoc(), constantInt, newResultType.getElementType());
    assert(op.getType(0) == newReduction.getResult(0).getType());
    assert(op.getType(0) == converted.getType());
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, newReduction.getResult(0),
                                                  converted.getResult());

    return success();
  }
};

template <typename OpTy>
static LogicalResult getDefiningZeroPadding(OpTy op, PatternRewriter &rewriter,
                                            stablehlo::PadOp &pad,
                                            Value &otherArg,
                                            bool &isOtherArgLHS) {
  pad = op.getLhs().template getDefiningOp<stablehlo::PadOp>();
  otherArg = op.getRhs();
  isOtherArgLHS = false;
  if (!pad) {
    pad = op.getRhs().template getDefiningOp<stablehlo::PadOp>();
    otherArg = op.getLhs();
    isOtherArgLHS = true;
  }
  if (!pad)
    return rewriter.notifyMatchFailure(op, "operands not produced by pad");
  // if (!llvm::hasSingleElement(pad->getUsers()))
  //  return rewriter.notifyMatchFailure(op, "pad has multiple users");

  if (!matchPattern(pad.getPaddingValue(), m_AnyZeroFloat()))
    return rewriter.notifyMatchFailure(op, "padding value not zero");
  return success();
}

template <typename T>
struct BinopConstPad : public CheckedOpRewritePattern<T, BinopConstPad<T>> {
  using CheckedOpRewritePattern<T, BinopConstPad<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    for (int i = 0; i < 2; i++) {
      DenseElementsAttr inp;
      if (!matchPattern(op->getOperand(i), m_Constant(&inp)))
        continue;
      if (!inp.isSplat())
        continue;

      auto pad =
          op->getOperand(1 - i).template getDefiningOp<stablehlo::PadOp>();
      if (!pad)
        continue;
      if (anyPadSizesNegative(pad))
        return failure();

      auto pval = pad.getPaddingValue();
      auto pval_cst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), pval.getType(),
          inp.resizeSplat(cast<ShapedType>(pval.getType())));
      auto pval2 = rewriter.create<T>(op.getLoc(), (i == 0) ? pval_cst : pval,
                                      (i == 0) ? pval : pval_cst);

      auto val = pad.getOperand();
      auto val_cst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), val.getType(),
          inp.resizeSplat(cast<ShapedType>(val.getType())));
      auto val2 = rewriter.create<T>(op.getLoc(), (i == 0) ? val_cst : val,
                                     (i == 0) ? val : val_cst);

      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          op, val2, pval2, pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
          pad.getInteriorPadding());
      return success();
    }

    return failure();
  }
};

template <typename T>
struct BinopPadPad : public CheckedOpRewritePattern<T, BinopPadPad<T>> {
  using CheckedOpRewritePattern<T, BinopPadPad<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    auto pad1 = op->getOperand(0).template getDefiningOp<stablehlo::PadOp>();
    if (!pad1 || anyPadSizesNegative(pad1))
      return failure();

    auto pad2 = op->getOperand(1).template getDefiningOp<stablehlo::PadOp>();
    if (!pad2 || anyPadSizesNegative(pad2))
      return failure();

    if (pad1.getEdgePaddingLow() != pad2.getEdgePaddingLow())
      return failure();

    if (pad1.getEdgePaddingHigh() != pad2.getEdgePaddingHigh())
      return failure();

    if (pad1.getInteriorPadding() != pad2.getInteriorPadding())
      return failure();

    auto pv2 = rewriter.create<T>(op.getLoc(), pad1.getPaddingValue(),
                                  pad2.getPaddingValue());
    auto op2 =
        rewriter.create<T>(op.getLoc(), pad1.getOperand(), pad2.getOperand());

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, op2, pv2, pad1.getEdgePaddingLow(), pad1.getEdgePaddingHigh(),
        pad1.getInteriorPadding());
    return success();
  }
};

struct AddPadPadToConcat
    : public CheckedOpRewritePattern<stablehlo::AddOp, AddPadPadToConcat> {
  using CheckedOpRewritePattern<stablehlo::AddOp,
                                AddPadPadToConcat>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AddOp op,
                                    PatternRewriter &rewriter) const {
    auto pad1 = op->getOperand(0).template getDefiningOp<stablehlo::PadOp>();
    if (!pad1 || anyPadSizesNegative(pad1))
      return failure();

    auto pad2 = op->getOperand(1).template getDefiningOp<stablehlo::PadOp>();
    if (!pad2 || anyPadSizesNegative(pad2))
      return failure();

    if (!matchPattern(pad1.getPaddingValue(), m_AnyZeroFloat()))
      return failure();

    if (!matchPattern(pad2.getPaddingValue(), m_AnyZeroFloat()))
      return failure();

    for (auto en : llvm::enumerate(op.getType().getShape())) {
      auto h1 = pad1.getEdgePaddingHigh()[en.index()];
      auto h2 = pad2.getEdgePaddingHigh()[en.index()];

      auto p1 = pad1.getOperand().getType().getShape()[en.index()];
      auto p2 = pad2.getOperand().getType().getShape()[en.index()];

      if (pad1.getInteriorPadding()[en.index()])
        continue;
      if (pad2.getInteriorPadding()[en.index()])
        continue;

      //  pad1: [ 0s ][ 0s   ][ data ][ 0s ]
      //  pad2: [ 0s ][ data ][ 0s   ][ 0s ]
      if (h2 == h1 + p1) {
        bool legal = true;
        for (auto en2 : llvm::enumerate(op.getType().getShape())) {
          if (en2.index() == en.index())
            continue;
          auto sl1 = pad1.getEdgePaddingLow()[en2.index()];
          auto sl2 = pad2.getEdgePaddingLow()[en2.index()];
          auto sh1 = pad1.getEdgePaddingHigh()[en2.index()];
          auto sh2 = pad2.getEdgePaddingHigh()[en2.index()];
          if (sl1 != sl2 || sh1 != sh2) {
            legal = false;
            break;
          }
        }
        if (legal) {
          Value data[] = {pad2.getOperand(), pad1.getOperand()};
          auto concat = rewriter.create<stablehlo::ConcatenateOp>(
              op.getLoc(), data, en.index());

          SmallVector<int64_t> lows(pad2.getEdgePaddingLow().begin(),
                                    pad2.getEdgePaddingLow().end());
          SmallVector<int64_t> highs(pad1.getEdgePaddingHigh().begin(),
                                     pad1.getEdgePaddingHigh().end());
          SmallVector<int64_t> ints(pad1.getInteriorPadding().begin(),
                                    pad1.getInteriorPadding().end());

          rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
              op, concat, pad1.getPaddingValue(), lows, highs, ints);
          return success();
        }
      }

      //  pad2: [ 0s ][ 0s   ][ data ][ 0s ]
      //  pad1: [ 0s ][ data ][ 0s   ][ 0s ]
      if (h1 == h2 + p2) {
        bool legal = true;
        for (auto en2 : llvm::enumerate(op.getType().getShape())) {
          if (en2.index() == en.index())
            continue;
          auto sl1 = pad1.getEdgePaddingLow()[en2.index()];
          auto sl2 = pad2.getEdgePaddingLow()[en2.index()];
          auto sh1 = pad1.getEdgePaddingHigh()[en2.index()];
          auto sh2 = pad2.getEdgePaddingHigh()[en2.index()];
          if (sl1 != sl2 || sh1 != sh2) {
            legal = false;
            break;
          }
        }
        if (legal) {
          Value data[] = {pad1.getOperand(), pad2.getOperand()};

          auto concat = rewriter.create<stablehlo::ConcatenateOp>(
              op.getLoc(), data, en.index());

          SmallVector<int64_t> lows(pad1.getEdgePaddingLow().begin(),
                                    pad1.getEdgePaddingLow().end());
          SmallVector<int64_t> highs(pad2.getEdgePaddingHigh().begin(),
                                     pad2.getEdgePaddingHigh().end());
          SmallVector<int64_t> ints(pad1.getInteriorPadding().begin(),
                                    pad1.getInteriorPadding().end());

          rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
              op, concat, pad1.getPaddingValue(), lows, highs, ints);
          return success();
        }
      }

      //  pad1: [ 0s ][      data            ][ 0s ]
      //  pad2: [ 0s ][ 0s ][  data   ][ 0s  ][ 0s ]
      if (h2 >= h1 && h2 + p2 <= h1 + p1) {
        bool legal = true;
        for (auto en2 : llvm::enumerate(op.getType().getShape())) {
          if (en2.index() == en.index())
            continue;
          auto sl1 = pad1.getEdgePaddingLow()[en2.index()];
          auto sl2 = pad2.getEdgePaddingLow()[en2.index()];
          auto sh1 = pad1.getEdgePaddingHigh()[en2.index()];
          auto sh2 = pad2.getEdgePaddingHigh()[en2.index()];
          if (sl1 != sl2 || sh1 != sh2) {
            legal = false;
            break;
          }
        }
        if (legal) {

          SmallVector<int64_t> slow(pad2.getEdgePaddingLow().size(), 0);
          SmallVector<int64_t> shigh(pad2.getEdgePaddingLow().size(), 0);
          SmallVector<int64_t> sint(pad2.getEdgePaddingLow().size(), 0);

          slow[en.index()] = h1 + p1 - (h2 + p2);
          shigh[en.index()] = h2 - h1;

          auto inPad = rewriter.create<stablehlo::PadOp>(
              op.getLoc(), pad2.getOperand(), pad2.getPaddingValue(), slow,
              shigh, sint);
          assert(inPad.getType() == pad1.getOperand().getType());

          auto add = rewriter.create<stablehlo::AddOp>(op.getLoc(), inPad,
                                                       pad1.getOperand());

          rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
              op, add, pad1.getPaddingValue(), pad1.getEdgePaddingLow(),
              pad1.getEdgePaddingHigh(), pad1.getInteriorPadding());
          return success();
        }
      }

      //  pad2: [ 0s ][      data            ][ 0s ]
      //  pad1: [ 0s ][ 0s ][  data   ][ 0s  ][ 0s ]
      if (h1 >= h2 && h1 + p1 <= h2 + p2) {
        bool legal = true;
        for (auto en2 : llvm::enumerate(op.getType().getShape())) {
          if (en2.index() == en.index())
            continue;
          auto sl1 = pad1.getEdgePaddingLow()[en2.index()];
          auto sl2 = pad2.getEdgePaddingLow()[en2.index()];
          auto sh1 = pad1.getEdgePaddingHigh()[en2.index()];
          auto sh2 = pad2.getEdgePaddingHigh()[en2.index()];
          if (sl1 != sl2 || sh1 != sh2) {
            legal = false;
            break;
          }
        }
        if (legal) {

          SmallVector<int64_t> slow(pad1.getEdgePaddingLow().size(), 0);
          SmallVector<int64_t> shigh(pad1.getEdgePaddingLow().size(), 0);
          SmallVector<int64_t> sint(pad1.getEdgePaddingLow().size(), 0);

          slow[en.index()] = h2 + p2 - (h1 + p1);
          shigh[en.index()] = h1 - h2;

          auto inPad = rewriter.create<stablehlo::PadOp>(
              op.getLoc(), pad1.getOperand(), pad1.getPaddingValue(), slow,
              shigh, sint);
          assert(inPad.getType() == pad2.getOperand().getType());

          auto add = rewriter.create<stablehlo::AddOp>(op.getLoc(), inPad,
                                                       pad2.getOperand());

          rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
              op, add, pad2.getPaddingValue(), pad2.getEdgePaddingLow(),
              pad2.getEdgePaddingHigh(), pad2.getInteriorPadding());
          return success();
        }
      }
    }

    return failure();
  }
};

template <typename T>
struct BinopBinopPadPad
    : public CheckedOpRewritePattern<T, BinopBinopPadPad<T>> {
  using CheckedOpRewritePattern<T,
                                BinopBinopPadPad<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    for (int i = 0; i < 2; i++) {
      auto pad1 = op->getOperand(i).template getDefiningOp<stablehlo::PadOp>();

      DenseElementsAttr inp1;
      if (!pad1) {
        if (!matchPattern(op->getOperand(i), m_Constant(&inp1)))
          continue;
        if (!inp1.isSplat())
          continue;
      } else if (anyPadSizesNegative(pad1))
        return failure();

      auto op2 = op->getOperand(1 - i).template getDefiningOp<T>();
      if (!op2)
        continue;

      for (int j = 0; j < 2; j++) {
        auto pad2 =
            op2->getOperand(j).template getDefiningOp<stablehlo::PadOp>();

        DenseElementsAttr inp2;
        if (!pad2) {
          if (!matchPattern(op2->getOperand(j), m_Constant(&inp2)))
            continue;
          if (!inp2.isSplat())
            continue;
        } else if (anyPadSizesNegative(pad2))
          return failure();

        if (pad1 && pad2) {
          if (pad1.getEdgePaddingLow() != pad2.getEdgePaddingLow())
            continue;

          if (pad1.getEdgePaddingHigh() != pad2.getEdgePaddingHigh())
            continue;

          if (pad1.getInteriorPadding() != pad2.getInteriorPadding())
            continue;
        }
        if (inp1 && inp2)
          continue;

        auto p1val = pad1 ? pad1.getPaddingValue()
                          : rewriter.create<stablehlo::ConstantOp>(
                                op.getLoc(), pad2.getPaddingValue().getType(),
                                inp1.resizeSplat(cast<ShapedType>(
                                    pad2.getPaddingValue().getType())));
        auto p2val = pad2 ? pad2.getPaddingValue()
                          : rewriter.create<stablehlo::ConstantOp>(
                                op.getLoc(), pad1.getPaddingValue().getType(),
                                inp2.resizeSplat(cast<ShapedType>(
                                    pad1.getPaddingValue().getType())));

        auto pval = rewriter.create<T>(op.getLoc(), p1val, p2val);

        auto o1val = pad1 ? pad1.getOperand()
                          : rewriter.create<stablehlo::ConstantOp>(
                                op.getLoc(), pad2.getOperand().getType(),
                                inp1.resizeSplat(cast<ShapedType>(
                                    pad2.getOperand().getType())));
        auto o2val = pad2 ? pad2.getOperand()
                          : rewriter.create<stablehlo::ConstantOp>(
                                op.getLoc(), pad1.getOperand().getType(),
                                inp2.resizeSplat(cast<ShapedType>(
                                    pad1.getOperand().getType())));

        auto val = rewriter.create<T>(op.getLoc(), o1val, o2val);

        auto npad = rewriter.create<stablehlo::PadOp>(
            op.getLoc(), val, pval, (pad1 ? pad1 : pad2).getEdgePaddingLow(),
            (pad1 ? pad1 : pad2).getEdgePaddingHigh(),
            (pad1 ? pad1 : pad2).getInteriorPadding());

        rewriter.replaceOpWithNewOp<T>(op, op2->getOperand(1 - j), npad);
        return success();
      }
    }
    return failure();
  }
};

template <typename T>
struct BinopBinopPadConst
    : public CheckedOpRewritePattern<T, BinopBinopPadConst<T>> {
  using CheckedOpRewritePattern<T,
                                BinopBinopPadConst<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    for (int i = 0; i < 2; i++) {
      auto pad1 = op->getOperand(i).template getDefiningOp<stablehlo::PadOp>();
      if (!pad1)
        continue;
      if (anyPadSizesNegative(pad1))
        return failure();
      auto op2 = op->getOperand(1 - i).template getDefiningOp<T>();
      if (!op2)
        continue;

      for (int j = 0; j < 2; j++) {
        auto pad2 =
            op2->getOperand(j).template getDefiningOp<stablehlo::PadOp>();
        if (!pad2)
          continue;
        if (anyPadSizesNegative(pad2))
          return failure();

        if (pad1.getEdgePaddingLow() != pad2.getEdgePaddingLow())
          continue;

        if (pad1.getEdgePaddingHigh() != pad2.getEdgePaddingHigh())
          continue;

        if (pad1.getInteriorPadding() != pad2.getInteriorPadding())
          continue;

        auto pval = rewriter.create<T>(pad2.getLoc(), pad1.getPaddingValue(),
                                       pad2.getPaddingValue());
        auto val = rewriter.create<T>(pad2.getLoc(), pad1.getOperand(),
                                      pad2.getOperand());

        auto npad = rewriter.create<stablehlo::PadOp>(
            op.getLoc(), val, pval, pad1.getEdgePaddingLow(),
            pad1.getEdgePaddingHigh(), pad1.getInteriorPadding());

        rewriter.replaceOpWithNewOp<T>(op, op2->getOperand(1 - j), npad);
        return success();
      }
    }
    return failure();
  }
};

struct MulZeroPad
    : public CheckedOpRewritePattern<stablehlo::MulOp, MulZeroPad> {
  using CheckedOpRewritePattern<stablehlo::MulOp,
                                MulZeroPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    auto otherArgType = cast<TensorType>(otherArg.getType());
    SmallVector<int64_t> limitDims = llvm::to_vector(otherArgType.getShape());
    for (auto &&[limit, pad] : llvm::zip(limitDims, pad.getEdgePaddingHigh())) {
      limit -= pad;
    }
    SmallVector<int64_t> interior = llvm::to_vector(pad.getInteriorPadding());
    for (int64_t &value : interior) {
      value += 1;
    }

    auto slice = rewriter.create<stablehlo::SliceOp>(
        pad.getLoc(), otherArg, pad.getEdgePaddingLow(), limitDims, interior);
    auto mul = rewriter.create<stablehlo::MulOp>(
        op.getLoc(), otherIsLHS ? slice.getResult() : pad.getOperand(),
        otherIsLHS ? pad.getOperand() : slice.getResult());
    auto newPad = rewriter.create<stablehlo::PadOp>(
        pad.getLoc(), mul.getResult(), pad.getPaddingValue(),
        pad.getEdgePaddingLowAttr(), pad.getEdgePaddingHighAttr(),
        pad.getInteriorPaddingAttr());
    assert(op.getType() == newPad.getType());
    rewriter.replaceOp(op, newPad);

    return success();
  }
};

struct DivZeroPad
    : public CheckedOpRewritePattern<stablehlo::DivOp, DivZeroPad> {
  using CheckedOpRewritePattern<stablehlo::DivOp,
                                DivZeroPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DivOp op,
                                    PatternRewriter &rewriter) const {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    if (otherIsLHS)
      return failure();

    auto otherArgType = cast<TensorType>(otherArg.getType());
    SmallVector<int64_t> limitDims = llvm::to_vector(otherArgType.getShape());
    for (auto &&[limit, pad] : llvm::zip(limitDims, pad.getEdgePaddingHigh())) {
      limit -= pad;
    }
    SmallVector<int64_t> interior = llvm::to_vector(pad.getInteriorPadding());
    for (int64_t &value : interior) {
      value += 1;
    }

    auto slice = rewriter.create<stablehlo::SliceOp>(
        pad.getLoc(), otherArg, pad.getEdgePaddingLow(), limitDims, interior);
    auto mul = rewriter.create<stablehlo::DivOp>(
        op.getLoc(), otherIsLHS ? slice.getResult() : pad.getOperand(),
        otherIsLHS ? pad.getOperand() : slice.getResult());
    auto newPad = rewriter.create<stablehlo::PadOp>(
        pad.getLoc(), mul.getResult(), pad.getPaddingValue(),
        pad.getEdgePaddingLowAttr(), pad.getEdgePaddingHighAttr(),
        pad.getInteriorPaddingAttr());
    rewriter.replaceOp(op, newPad);

    return success();
  }
};

template <typename T> DenseI64ArrayAttr addLists(T lhs, T rhs) {
  MLIRContext *context = lhs.getContext();
  SmallVector<int64_t> sum;
  for (auto &&[lhsv, rhsv] :
       llvm::zip(cast<DenseI64ArrayAttr>(lhs).asArrayRef(),
                 cast<DenseI64ArrayAttr>(rhs).asArrayRef())) {
    sum.push_back(lhsv + rhsv);
  }
  return DenseI64ArrayAttr::get(context, sum);
}

struct PadPad : public CheckedOpRewritePattern<stablehlo::PadOp, PadPad> {
  using CheckedOpRewritePattern<stablehlo::PadOp,
                                PadPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::PadOp op,
                                    PatternRewriter &rewriter) const {
    auto definingPad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!definingPad || definingPad.getPaddingValue() != op.getPaddingValue()) {
      return rewriter.notifyMatchFailure(op, "no compatible defining pad");
    }

    auto allZero = [](ArrayRef<int64_t> values) {
      return llvm::all_of(values, [](int64_t v) { return v == 0; });
    };

    if (!allZero(op.getInteriorPadding()) ||
        !allZero(definingPad.getInteriorPadding())) {
      return rewriter.notifyMatchFailure(op, "cannot combine interior padding");
    }

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, definingPad.getOperand(), definingPad.getPaddingValue(),
        addLists(op.getEdgePaddingLowAttr(),
                 definingPad.getEdgePaddingLowAttr()),
        addLists(op.getEdgePaddingHighAttr(),
                 definingPad.getEdgePaddingHighAttr()),
        addLists(op.getInteriorPaddingAttr(),
                 definingPad.getInteriorPaddingAttr()));
    return success();
  }
};

std::tuple<Value, Value, RankedTensorType>
sliceDotGeneralHelper(stablehlo::DotGeneralOp dot, ArrayRef<int64_t> starts,
                      ArrayRef<int64_t> limits, ArrayRef<int64_t> strides,
                      PatternRewriter &rewriter) {
  SmallVector<int64_t> lhs_lb(dot.getLhs().getType().getShape().size(), 0);
  SmallVector<int64_t> lhs_ub(dot.getLhs().getType().getShape().begin(),
                              dot.getLhs().getType().getShape().end());
  SmallVector<int64_t> lhs_step(dot.getLhs().getType().getShape().size(), 1);

  SmallVector<int64_t> rhs_lb(dot.getRhs().getType().getShape().size(), 0);
  SmallVector<int64_t> rhs_ub(dot.getRhs().getType().getShape().begin(),
                              dot.getRhs().getType().getShape().end());
  SmallVector<int64_t> rhs_step(dot.getRhs().getType().getShape().size(), 1);

  auto dimensionNumbers = dot.getDotDimensionNumbers();

  size_t residx = 0;
  SmallVector<int64_t> resShape;
  for (auto &&[lhs, rhs] :
       llvm::zip(dimensionNumbers.getLhsBatchingDimensions(),
                 dimensionNumbers.getRhsBatchingDimensions())) {
    lhs_lb[lhs] = starts[residx];
    lhs_ub[lhs] = limits[residx];
    lhs_step[lhs] = strides[residx];

    rhs_lb[rhs] = starts[residx];
    rhs_ub[rhs] = limits[residx];
    rhs_step[rhs] = strides[residx];
    resShape.push_back((limits[residx] - starts[residx]) / strides[residx]);
    residx++;
  }

  for (size_t i = 0, end = dot.getLhs().getType().getShape().size(); i < end;
       i++) {
    if (llvm::is_contained(dimensionNumbers.getLhsContractingDimensions(), i))
      continue;
    if (llvm::is_contained(dimensionNumbers.getLhsBatchingDimensions(), i))
      continue;

    lhs_lb[i] = starts[residx];
    lhs_ub[i] = limits[residx];
    lhs_step[i] = strides[residx];
    resShape.push_back((limits[residx] - starts[residx]) / strides[residx]);
    residx++;
  }

  for (size_t i = 0, end = dot.getRhs().getType().getShape().size(); i < end;
       i++) {
    if (llvm::is_contained(dimensionNumbers.getRhsContractingDimensions(), i))
      continue;
    if (llvm::is_contained(dimensionNumbers.getRhsBatchingDimensions(), i))
      continue;

    rhs_lb[i] = starts[residx];
    rhs_ub[i] = limits[residx];
    rhs_step[i] = strides[residx];
    resShape.push_back((limits[residx] - starts[residx]) / strides[residx]);
    residx++;
  }

  assert(residx == dot.getType().getShape().size());

  auto lhs2 = rewriter.create<stablehlo::SliceOp>(dot.getLoc(), dot.getLhs(),
                                                  lhs_lb, lhs_ub, lhs_step);
  auto rhs2 = rewriter.create<stablehlo::SliceOp>(dot.getLoc(), dot.getRhs(),
                                                  rhs_lb, rhs_ub, rhs_step);

  return std::tuple<Value, Value, RankedTensorType>(
      lhs2, rhs2,
      RankedTensorType::get(resShape, dot.getType().getElementType()));
}

struct SliceDotGeneral
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceDotGeneral> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceDotGeneral>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto dot = op.getOperand().getDefiningOp<stablehlo::DotGeneralOp>();
    if (!dot) {
      return rewriter.notifyMatchFailure(op, "defining op is not a reshape");
    }
    if (!llvm::hasSingleElement(dot->getUsers()))
      return failure();

    auto &&[lhs2, rhs2, resTy] =
        sliceDotGeneralHelper(dot, op.getStartIndices(), op.getLimitIndices(),
                              op.getStrides(), rewriter);

    Value operands[2] = {lhs2, rhs2};
    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        op, TypeRange(resTy), operands, dot->getAttrs());

    return success();
  }
};

struct PadDotGeneral
    : public CheckedOpRewritePattern<stablehlo::DotGeneralOp, PadDotGeneral> {
  using CheckedOpRewritePattern<stablehlo::DotGeneralOp,
                                PadDotGeneral>::CheckedOpRewritePattern;

  bool postPad;
  PadDotGeneral(size_t postPad, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        postPad(postPad) {}

  LogicalResult matchAndRewriteImpl(stablehlo::DotGeneralOp op,
                                    PatternRewriter &rewriter) const {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    for (auto u : pad->getUsers())
      if (!isa<stablehlo::DotGeneralOp>(u))
        return failure();

    auto dimensionNumbers = op.getDotDimensionNumbers();
    auto padContractingDimensions =
        dimensionNumbers.getLhsContractingDimensions();
    auto otherContractingDimensions =
        dimensionNumbers.getRhsContractingDimensions();

    auto padBatchDimensions = dimensionNumbers.getLhsBatchingDimensions();
    auto otherBatchDimensions = dimensionNumbers.getRhsBatchingDimensions();

    SmallVector<int64_t> padResultDimensions;
    for (size_t i = 0, end = op.getLhs().getType().getShape().size(); i < end;
         i++) {
      if (llvm::is_contained(dimensionNumbers.getLhsContractingDimensions(), i))
        continue;
      if (llvm::is_contained(dimensionNumbers.getLhsBatchingDimensions(), i))
        continue;
      padResultDimensions.push_back(i);
    }

    SmallVector<int64_t> otherResultDimensions;
    for (size_t i = 0, end = op.getRhs().getType().getShape().size(); i < end;
         i++) {
      if (llvm::is_contained(dimensionNumbers.getRhsContractingDimensions(), i))
        continue;
      if (llvm::is_contained(dimensionNumbers.getRhsBatchingDimensions(), i))
        continue;
      otherResultDimensions.push_back(i);
    }

    if (otherIsLHS) {
      std::swap(padContractingDimensions, otherContractingDimensions);
      std::swap(padBatchDimensions, otherBatchDimensions);
      std::swap(padResultDimensions, otherResultDimensions);
    }

    // Need to figure out which dimension(s) to slice. For this purpose,
    // look the pairs of contracting dimensions.
    SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        otherDimsToSlice;
    for (auto &&[padDim, otherDim] :
         llvm::zip(padContractingDimensions, otherContractingDimensions)) {
      // If padding along the dim, mark the corresponding other dim for
      // slicing.
      int64_t low = pad.getEdgePaddingLow()[padDim];
      int64_t high = pad.getEdgePaddingHigh()[padDim];
      int64_t interior = pad.getInteriorPadding()[padDim];
      if (low == 0 && high == 0 && interior == 0)
        continue;
      if (!postPad)
        otherDimsToSlice.emplace_back(otherDim, low, high, interior);
    }

    SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t>> resultDimsToPad;
    SmallVector<int64_t> resultShape;

    {
      size_t resultidx = 0;
      for (auto &&[padDim, otherDim] :
           llvm::zip(padBatchDimensions, otherBatchDimensions)) {
        // If padding along the dim, mark the corresponding other dim for
        // slicing.
        int64_t low = pad.getEdgePaddingLow()[padDim];
        int64_t high = pad.getEdgePaddingHigh()[padDim];
        int64_t interior = pad.getInteriorPadding()[padDim];
        auto padSize = cast<RankedTensorType>(pad.getOperand().getType())
                           .getShape()[padDim];
        resultShape.push_back(padSize);
        if (low == 0 && high == 0 && interior == 0) {
          resultidx++;
          continue;
        }
        if (!postPad)
          otherDimsToSlice.emplace_back(otherDim, low, high, interior);
        resultDimsToPad.emplace_back(resultidx, low, high, interior);
        resultidx++;
      }

      if (otherIsLHS) {
        for (auto dim : otherResultDimensions) {
          resultidx++;
          resultShape.push_back(
              cast<RankedTensorType>(otherArg.getType()).getShape()[dim]);
        }
      }

      for (auto padDim : padResultDimensions) {
        int64_t low = pad.getEdgePaddingLow()[padDim];
        int64_t high = pad.getEdgePaddingHigh()[padDim];
        int64_t interior = pad.getInteriorPadding()[padDim];
        auto padSize = cast<RankedTensorType>(pad.getOperand().getType())
                           .getShape()[padDim];
        resultShape.push_back(padSize);
        if (low == 0 && high == 0 && interior == 0) {
          resultidx++;
          continue;
        }

        resultDimsToPad.emplace_back(resultidx, low, high, interior);
        resultidx++;
      }

      if (!otherIsLHS) {
        for (auto dim : otherResultDimensions) {
          resultidx++;
          resultShape.push_back(
              cast<RankedTensorType>(otherArg.getType()).getShape()[dim]);
        }
      }
    }

    if (!resultDimsToPad.empty() && !postPad)
      return failure();

    if (otherDimsToSlice.empty() && resultDimsToPad.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "contracting dimensions not padded");
    }

    Value nextOtherArg = otherArg;
    if (!otherDimsToSlice.empty()) {
      SmallVector<int64_t> sliceLow, sliceHigh, sliceStride;
      for (auto &&[pos, size] :
           llvm::enumerate(cast<TensorType>(otherArg.getType()).getShape())) {
        std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>> it;
        for (auto tup : otherDimsToSlice)
          if (std::get<0>(tup) == pos)
            it = tup;
        if (!it) {
          sliceLow.push_back(0);
          sliceHigh.push_back(size);
          sliceStride.push_back(1);
          continue;
        }

        sliceLow.push_back(std::get<1>(*it));
        sliceHigh.push_back(size - std::get<2>(*it));
        sliceStride.push_back(std::get<3>(*it) + 1);
      }

      auto slice = rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), otherArg, sliceLow, sliceHigh, sliceStride);
      nextOtherArg = slice.getResult();
    }

    Value res = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(),
        RankedTensorType::get(resultShape, op.getType().getElementType()),
        otherIsLHS ? nextOtherArg : pad.getOperand(),
        otherIsLHS ? pad.getOperand() : nextOtherArg,
        op.getDotDimensionNumbersAttr(), op.getPrecisionConfigAttr(),
        op.getAlgorithmAttr());

    if (!resultDimsToPad.empty()) {
      SmallVector<int64_t> low(op.getType().getShape().size(), 0);
      SmallVector<int64_t> high(op.getType().getShape().size(), 0);
      SmallVector<int64_t> interior(op.getType().getShape().size(), 0);
      for (auto &&[idx, lval, hval, ival] : resultDimsToPad) {
        low[idx] = lval;
        high[idx] = hval;
        interior[idx] = ival;
      }
      res = rewriter.create<stablehlo::PadOp>(op.getLoc(), op.getType(), res,
                                              pad.getPaddingValue(), low, high,
                                              interior);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

LogicalResult sliceReshapeHelper(stablehlo::SliceOp op,
                                 SmallVectorImpl<int64_t> &starts,
                                 SmallVectorImpl<int64_t> &limits,
                                 SmallVectorImpl<int64_t> &strides) {
  auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
  if (!reshape) {
    return failure();
  }

  assert(starts.size() == 0);
  assert(limits.size() == 0);
  assert(strides.size() == 0);

  auto reshapeOperandType = cast<TensorType>(reshape.getOperand().getType());
  auto reshapeType = cast<TensorType>(reshape.getType());
  size_t indim = 0;
  size_t outdim = 0;
  while (indim < reshapeOperandType.getShape().size() &&
         outdim < reshapeType.getShape().size()) {
    if (reshapeOperandType.getShape()[indim] ==
        reshapeType.getShape()[outdim]) {
      starts.push_back(op.getStartIndices()[outdim]);
      limits.push_back(op.getLimitIndices()[outdim]);
      strides.push_back(op.getStrides()[outdim]);
      indim++;
      outdim++;
      continue;
    }
    if (reshapeOperandType.getShape()[indim] == 1) {
      starts.push_back(0);
      limits.push_back(1);
      strides.push_back(1);
      indim++;
      continue;
    }
    if (reshapeType.getShape()[outdim] == 1) {
      if (op.getStartIndices()[outdim] != 0)
        return failure();
      if (op.getLimitIndices()[outdim] != 1)
        return failure();
      if (op.getStrides()[outdim] != 1)
        return failure();
      outdim++;
      continue;
    }
    return failure();
  }
  while (indim < reshapeOperandType.getShape().size()) {
    if (reshapeOperandType.getShape()[indim] != 1)
      return failure();
    // It's a full slice of the original dimension.
    starts.push_back(0);
    limits.push_back(1);
    strides.push_back(1);
    indim++;
  }
  while (outdim < reshapeType.getShape().size()) {
    if (reshapeType.getShape()[outdim] != 1)
      return failure();
    if (op.getStartIndices()[outdim] != 0)
      return failure();
    if (op.getLimitIndices()[outdim] != 1)
      return failure();
    if (op.getStrides()[outdim] != 1)
      return failure();
    outdim++;
  }
  return success();
}

struct SliceReshape
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshape> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceReshape>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape) {
      return rewriter.notifyMatchFailure(op, "defining op is not a reshape");
    }
    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    auto newSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), reshape.getOperand(), starts, limits, strides);
    auto newReshape = rewriter.create<stablehlo::ReshapeOp>(
        reshape->getLoc(), op.getResult().getType(), newSlice.getResult());
    rewriter.replaceOp(op, newReshape);

    return success();
  }
};

// slice(reshape(pad x)) -> pad(slice x)
struct SliceReshapePad final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshapePad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    auto pad = reshape.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    SmallVector<int64_t> step;

    SmallVector<int64_t> lpads;
    SmallVector<int64_t> hpads;
    SmallVector<int64_t> interiors;

    bool needspad = false;
    bool broadcastres = false;
    if (!slicePadHelper(pad, starts, limits, strides, start, end, step, lpads,
                        hpads, interiors, broadcastres, needspad)
             .succeeded())
      return failure();

    if (broadcastres) {
      rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
          op, op.getType(), pad.getPaddingValue(),
          rewriter.getDenseI64ArrayAttr({}));
      return success();
    }

    if (needspad && !llvm::hasSingleElement(pad->getUsers()) &&
        !llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    mlir::Value nslice = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), pad.getOperand(), start, end, step);
    if (needspad) {
      nslice = rewriter.create<stablehlo::PadOp>(
          op.getLoc(), nslice, pad.getPaddingValue(), lpads, hpads, interiors);
    }
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(), nslice);
    return success();
  }
};

struct SliceIf : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceIf> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceIf>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto ifop = op.getOperand().getDefiningOp<stablehlo::IfOp>();
    if (!ifop)
      return failure();
    if (!llvm::hasSingleElement(op.getOperand().getUsers()))
      return failure();

    auto opres = cast<OpResult>(op.getOperand());
    ssize_t opIdx = opres.getResultNumber();

    SmallVector<Type> ifResultTypes = llvm::to_vector(ifop->getResultTypes());
    ifResultTypes[opIdx] = op.getType();

    auto newIf = rewriter.create<stablehlo::IfOp>(ifop.getLoc(), ifResultTypes,
                                                  ifop.getPred());

    Operation *trueTerm = ifop.getTrueBranch().front().getTerminator();
    Operation *falseTerm = ifop.getFalseBranch().front().getTerminator();

    rewriter.setInsertionPoint(trueTerm);
    auto newTrue = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), trueTerm->getOperands()[opIdx], op.getStartIndices(),
        op.getLimitIndices(), op.getStrides());
    rewriter.modifyOpInPlace(trueTerm,
                             [&] { trueTerm->setOperand(opIdx, newTrue); });

    rewriter.setInsertionPoint(falseTerm);
    auto newFalse = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), falseTerm->getOperands()[opIdx], op.getStartIndices(),
        op.getLimitIndices(), op.getStrides());
    rewriter.modifyOpInPlace(falseTerm,
                             [&] { falseTerm->setOperand(opIdx, newFalse); });

    newIf.getTrueBranch().takeBody(ifop.getTrueBranch());
    newIf.getFalseBranch().takeBody(ifop.getFalseBranch());

    for (int i = 0; i < ifop.getNumResults(); i++) {
      if (i == opIdx) {
        rewriter.replaceOp(op, newIf.getResult(i));
      } else {
        rewriter.replaceAllUsesWith(ifop.getResult(i), newIf.getResult(i));
      }
    }

    rewriter.eraseOp(ifop);
    return success();
  }
};

struct SliceReshapeConcat
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshapeConcat> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceReshapeConcat>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    auto concat =
        reshape.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    auto dim = concat.getDimension();

    SmallVector<Value> postConcat;
    if (!sliceConcatHelper(concat, rewriter, starts, limits, strides,
                           postConcat)
             .succeeded())
      return failure();

    auto c2 = rewriter.create<stablehlo::ConcatenateOp>(concat.getLoc(),
                                                        postConcat, dim);
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(), c2);
    return success();
  }
};

struct SliceReshapeElementwise final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshapeElementwise> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();
    auto elem = reshape.getOperand().getDefiningOp();
    if (!elem)
      return failure();
    if (!elem->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();
    if (!llvm::hasSingleElement(elem->getUsers()))
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    SmallVector<Value> ops;
    for (auto v : elem->getOperands()) {
      ops.push_back(rewriter.create<stablehlo::SliceOp>(op.getLoc(), v, starts,
                                                        limits, strides));
    }
    SmallVector<int64_t> sizes;
    for (auto &&[start, stop, stride] : llvm::zip(starts, limits, strides))
      sizes.push_back((stop - start) / stride);
    auto nex = rewriter.create(
        elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
        TypeRange(RankedTensorType::get(
            sizes, reshape.getOperand().getType().getElementType())),
        elem->getAttrs(), {}, {});
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      nex->getResult(0));
    return success();
  }
};

struct TransposeElementwise final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeElementwise> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  bool onlySingleUser;

  TransposeElementwise(bool onlySingleUser, MLIRContext *context,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto elem = op.getOperand().getDefiningOp();
    if (!elem)
      return failure();

    if (!elem->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();

    bool singleUser = llvm::hasSingleElement(elem->getUsers());
    if (onlySingleUser && !singleUser)
      return failure();

    SmallVector<Value> ops;
    for (auto v : elem->getOperands()) {
      if (auto rop = v.getDefiningOp()) {
        rewriter.setInsertionPointAfter(rop);
      } else if (auto ba = dyn_cast<BlockArgument>(v)) {
        rewriter.setInsertionPointToStart(ba.getOwner());
      }
      ops.push_back(rewriter.create<stablehlo::TransposeOp>(
          op.getLoc(), v, op.getPermutation()));
    }
    if (singleUser) {
      rewriter.modifyOpInPlace(elem, [&]() {
        elem->setOperands(ops);
        elem->getResult(0).setType(op.getType());
      });
      rewriter.replaceOp(op, elem);
    } else {
      rewriter.setInsertionPointAfter(elem);
      auto newOp = rewriter.create(
          elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
          TypeRange(op.getType()), elem->getAttrs(), {}, {});
      rewriter.replaceOp(op, newOp);
    }
    return success();
  }
};

struct TransposeConcat final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto concat = op.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    if (!llvm::hasSingleElement(concat->getUsers()))
      return failure();

    SmallVector<Value> ops;
    for (auto v : concat->getOperands()) {
      ops.push_back(rewriter.create<stablehlo::TransposeOp>(
          op.getLoc(), v, op.getPermutation()));
    }

    auto dim = concat.getDimension();
    auto dim2 = getInversePermutation(op.getPermutation())[dim];

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, ops, dim2);
    rewriter.eraseOp(concat);
    return success();
  }
};

struct TransposeIota final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeIota> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto iota = op.getOperand().getDefiningOp<stablehlo::IotaOp>();
    if (!iota)
      return failure();

    if (!llvm::hasSingleElement(iota->getUsers()))
      return failure();

    auto dim = iota.getIotaDimension();
    auto dim2 = getInversePermutation(op.getPermutation())[dim];

    rewriter.replaceOpWithNewOp<stablehlo::IotaOp>(op, op.getType(), dim2);
    rewriter.eraseOp(iota);
    return success();
  }
};

struct TransposeReduceWindow final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeReduceWindow> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto reduce = op.getOperand().getDefiningOp<stablehlo::ReduceWindowOp>();
    if (!reduce)
      return failure();

    if (reduce->getNumResults() != 1)
      return failure();

    if (!llvm::hasSingleElement(reduce->getResult(0).getUsers()))
      return failure();

    Type restys[] = {op.getType()};

    Value operands[] = {rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), reduce.getOperands()[0], op.getPermutation())};

    int64_t padding_shape[2] = {(int64_t)op.getType().getShape().size(), 2};

    SmallVector<int64_t> win_dim(reduce.getWindowDimensions().begin(),
                                 reduce.getWindowDimensions().end());
    SmallVector<int64_t> win_strides;
    if (reduce.getWindowStrides())
      win_strides.append(reduce.getWindowStrides()->begin(),
                         reduce.getWindowStrides()->end());
    SmallVector<int64_t> base_dialations;
    if (reduce.getBaseDilations())
      base_dialations.append(reduce.getBaseDilations()->begin(),
                             reduce.getBaseDilations()->end());
    SmallVector<int64_t> win_dialations;
    if (reduce.getWindowDilations())
      win_dialations.append(reduce.getWindowDilations()->begin(),
                            reduce.getWindowDilations()->end());
    SmallVector<int64_t> padding_dialations(2 * padding_shape[0]);

    auto perm = op.getPermutation();
    for (int64_t i = 0; i < perm.size(); ++i) {
      win_dim[perm[i]] = reduce.getWindowDimensions()[i];
      if (reduce.getWindowStrides())
        win_strides[perm[i]] = (*reduce.getWindowStrides())[i];
      if (reduce.getBaseDilations())
        base_dialations[perm[i]] = (*reduce.getBaseDilations())[i];
      if (reduce.getWindowDilations())
        win_dialations[perm[i]] = (*reduce.getWindowDilations())[i];
      if (reduce.getPadding()) {
        padding_dialations[2 * perm[i]] =
            (*(reduce.getPadding()->begin() + (2 * i))).getSExtValue();
        padding_dialations[2 * perm[i] + 1] =
            (*(reduce.getPadding()->begin() + (2 * i + 1))).getSExtValue();
      }
    }

    auto red2 = rewriter.replaceOpWithNewOp<stablehlo::ReduceWindowOp>(
        op, restys, operands, reduce.getInitValues(),
        rewriter.getDenseI64ArrayAttr(win_dim),
        (reduce.getWindowStrides() ? rewriter.getDenseI64ArrayAttr(win_strides)
                                   : DenseI64ArrayAttr()),
        (reduce.getBaseDilations()
             ? rewriter.getDenseI64ArrayAttr(base_dialations)
             : DenseI64ArrayAttr()),
        (reduce.getWindowDilations()
             ? rewriter.getDenseI64ArrayAttr(win_dialations)
             : DenseI64ArrayAttr()),
        (reduce.getPadding()
             ? DenseIntElementsAttr::get(
                   RankedTensorType::get(padding_shape,
                                         rewriter.getIntegerType(64)),
                   padding_dialations)
             : DenseIntElementsAttr()));

    red2.getBody().takeBody(reduce.getBody());
    rewriter.eraseOp(reduce);
    return success();
  }
};

// reshape(concat(...)) -> concat(reshape(...))
struct ReshapeOfConcatToConcatOfReshape final
    : public CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                     ReshapeOfConcatToConcatOfReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  int64_t computeNewConcatDim(ArrayRef<int64_t> originalShape,
                              ArrayRef<int64_t> reshapedShape,
                              int64_t originalConcatDim) const {
    int64_t flattenedIndex = 0;
    int64_t newConcatDim = 0;

    for (int64_t i = 0; i < originalConcatDim; ++i) {
      flattenedIndex += originalShape[i];
    }

    int64_t cumulativeSize = 0;
    for (int64_t i = 0; i < reshapedShape.size(); ++i) {
      cumulativeSize += reshapedShape[i];
      if (flattenedIndex < cumulativeSize) {
        newConcatDim = i;
        break;
      }
    }
    return newConcatDim;
  }

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp reshapeOp,
                                    PatternRewriter &rewriter) const {
    // Check if the operand of the reshape is a concatenate operation
    auto concatOp =
        reshapeOp.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concatOp)
      return failure();

    SmallVector<int64_t> oneHot(concatOp.getType().getShape().size(), 0);
    oneHot[concatOp.getDimension()] = 1;

    int64_t zero = 0;
    if (!transformReshapeSlice<int64_t>(reshapeOp, oneHot, /*toFill*/ 0, &zero))
      return failure();

    int64_t newDim = -1;
    for (auto &&[i, val] : llvm::enumerate(oneHot)) {
      if (val == 1) {
        assert(newDim == -1);
        newDim = i;
      }
    }
    assert(newDim != -1);

    // Create reshaped operands for the concat operation
    SmallVector<Value> concatOperands;
    for (auto operand : concatOp.getOperands()) {
      auto operandType = cast<RankedTensorType>(operand.getType());
      if (!operandType)
        return failure();

      SmallVector<int64_t> shape(operandType.getShape().begin(),
                                 operandType.getShape().end());

      int64_t one = 1;
      if (!transformReshapeSlice<int64_t>(reshapeOp, shape, /*toFill*/ 1, &one))
        return failure();

      auto newReshapeType =
          RankedTensorType::get(shape, operandType.getElementType());
      auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
          reshapeOp.getLoc(), newReshapeType, operand);
      concatOperands.push_back(newReshapeOp);
    }

    // Create a new concat operation with the reshaped operands
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        reshapeOp, concatOperands, newDim);
    return success();
  }
};

// reshape(reduce_window(...)) -> reduce_window(reshape(...))
struct ReshapeReduceWindow final
    : public CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                     ReshapeReduceWindow> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp reshapeOp,
                                    PatternRewriter &rewriter) const {
    // Check if the operand of the reshape is a reduce_window operation
    auto reduceWindow =
        reshapeOp.getOperand().getDefiningOp<stablehlo::ReduceWindowOp>();
    if (!reduceWindow)
      return failure();

    // Check if there is any non-reshape user of this reduce_window operation
    if (llvm::any_of(reduceWindow->getUsers(), [&](Operation *user) {
          return !isa<stablehlo::ReshapeOp>(user);
        }))
      return failure();

    auto reduceWindowType = dyn_cast<ShapedType>(reduceWindow.getType(0));
    if (!reduceWindowType || !reduceWindowType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> reduceWindowShape = reduceWindowType.getShape();

    auto reshapeType = dyn_cast<ShapedType>(reshapeOp.getType());
    if (!reshapeType || !reshapeType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> reshapeShape = reshapeType.getShape();

    // Create reshaped operands for the reduce_window operation
    SmallVector<int64_t> windowDims;
    SmallVector<int64_t> windowStrides;
    SmallVector<int64_t> windowDilations;
    SmallVector<int64_t> baseDilations;
    size_t paddingSize =
        reduceWindow.getPadding() ? 2 * reshapeShape.size() : 0;
    SmallVector<int64_t> padding(paddingSize);

    size_t reduceWindowDim = 0;
    size_t reshapeDim = 0;

    while (reduceWindowDim < reduceWindowShape.size() &&
           reshapeDim < reshapeShape.size()) {
      // Original dimension, not being reshaped
      if (reduceWindowShape[reduceWindowDim] == reshapeShape[reshapeDim]) {
        windowDims.push_back(
            reduceWindow.getWindowDimensions()[reduceWindowDim]);
        if (reduceWindow.getWindowStrides())
          windowStrides.push_back(
              (*reduceWindow.getWindowStrides())[reduceWindowDim]);
        if (reduceWindow.getWindowDilations())
          windowDilations.push_back(
              (*reduceWindow.getWindowDilations())[reduceWindowDim]);
        if (reduceWindow.getBaseDilations())
          baseDilations.push_back(
              (*reduceWindow.getBaseDilations())[reduceWindowDim]);
        if (reduceWindow.getPadding()) {
          padding[2 * reshapeDim] =
              (*(reduceWindow.getPadding()->begin() + (2 * reduceWindowDim)))
                  .getSExtValue();
          padding[2 * reshapeDim + 1] = (*(reduceWindow.getPadding()->begin() +
                                           (2 * reduceWindowDim + 1)))
                                            .getSExtValue();
        }
        reduceWindowDim++;
        reshapeDim++;
        continue;
      }
      // Unit dimension dropped by reshape
      if (reduceWindowShape[reduceWindowDim] == 1) {
        reduceWindowDim++;
        continue;
      }
      // Unit dimension added by reshape
      if (reshapeShape[reshapeDim] == 1) {
        windowDims.push_back(1);
        if (reduceWindow.getWindowStrides())
          windowStrides.push_back(1);
        if (reduceWindow.getWindowDilations())
          windowDilations.push_back(1);
        if (reduceWindow.getBaseDilations())
          baseDilations.push_back(1);
        if (reduceWindow.getPadding()) {
          padding[2 * reshapeDim] = 0;
          padding[2 * reshapeDim + 1] = 0;
        }
        reshapeDim++;
        continue;
      }
      // Adding/dropping non-unit dimension is not supported
      return failure();
    }

    while (reduceWindowDim < reduceWindowShape.size()) {
      if (reduceWindowShape[reduceWindowDim] != 1)
        return failure();
      reduceWindowDim++;
    }

    while (reshapeDim < reshapeShape.size()) {
      windowDims.push_back(1);
      if (reduceWindow.getWindowStrides())
        windowStrides.push_back(1);
      if (reduceWindow.getWindowDilations())
        windowDilations.push_back(1);
      if (reduceWindow.getBaseDilations())
        baseDilations.push_back(1);
      if (reduceWindow.getPadding()) {
        padding[2 * reshapeDim] = 0;
        padding[2 * reshapeDim + 1] = 0;
      }
      reshapeDim++;
    }

    auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
        reshapeOp.getLoc(), reshapeType, reduceWindow.getInputs()[0]);
    auto newReduceWindowOp =
        rewriter.replaceOpWithNewOp<stablehlo::ReduceWindowOp>(
            reshapeOp, TypeRange(reshapeType), newReshapeOp->getResults(),
            reduceWindow.getInitValues(),
            rewriter.getDenseI64ArrayAttr(windowDims),
            rewriter.getDenseI64ArrayAttr(windowStrides),
            rewriter.getDenseI64ArrayAttr(baseDilations),
            rewriter.getDenseI64ArrayAttr(windowDilations),
            DenseIntElementsAttr::get(
                RankedTensorType::get({(int64_t)reshapeShape.size(), 2},
                                      rewriter.getIntegerType(64)),
                padding));

    newReduceWindowOp.getBody().takeBody(reduceWindow.getBody());
    rewriter.eraseOp(reduceWindow);
    return success();
  }
};

struct ReshapeElementwise final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeElementwise> {
  bool onlySingleUser;

  ReshapeElementwise(bool onlySingleUser, MLIRContext *context,
                     PatternBenefit benefit = 1,
                     ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto elem = op.getOperand().getDefiningOp();
    if (!elem)
      return failure();

    if (onlySingleUser && !llvm::hasSingleElement(elem->getUsers()))
      return failure();

    if (!elem->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();

    SmallVector<Value> ops;
    for (auto v : elem->getOperands()) {
      ops.push_back(rewriter.create<stablehlo::ReshapeOp>(
          op.getLoc(),
          RankedTensorType::get(
              op.getType().getShape(),
              cast<RankedTensorType>(v.getType()).getElementType()),
          v));
    }
    auto newOp = rewriter.create(
        elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
        TypeRange(op.getType()), elem->getAttrs(), {}, {});
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

// slice(transpose x) -> transpose(slice x)
struct SliceReshapeTranspose final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshapeTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    auto transpose =
        reshape.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transpose || !llvm::hasSingleElement(transpose->getUsers()))
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    auto newslice =
        sliceTransposeHelper(transpose, rewriter, starts, limits, strides);
    auto newtransp = rewriter.create<stablehlo::TransposeOp>(
        transpose.getLoc(), newslice, transpose.getPermutation());
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      newtransp);
    return success();
  }
};

struct SliceReshapeDotGeneral
    : public CheckedOpRewritePattern<stablehlo::SliceOp,
                                     SliceReshapeDotGeneral> {
  using CheckedOpRewritePattern<
      stablehlo::SliceOp, SliceReshapeDotGeneral>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    auto dot = reshape.getOperand().getDefiningOp<stablehlo::DotGeneralOp>();
    if (!dot) {
      return rewriter.notifyMatchFailure(op, "defining op is not a reshape");
    }
    if (!llvm::hasSingleElement(dot->getUsers()))
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    auto &&[lhs2, rhs2, resTy] =
        sliceDotGeneralHelper(dot, starts, limits, strides, rewriter);

    Value operands[2] = {lhs2, rhs2};
    auto newdot = rewriter.create<stablehlo::DotGeneralOp>(
        op.getLoc(), TypeRange(resTy), operands, dot->getAttrs());

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(), newdot);
    return success();
  }
};

struct ReshuffleAndsCompares final
    : CheckedOpRewritePattern<stablehlo::AndOp, ReshuffleAndsCompares> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AndOp andOp,
                                    PatternRewriter &rewriter) const {
    SmallVector<Value> conjuncts;
    SmallVector<Operation *> worklist;
    worklist.push_back(andOp);
    while (!worklist.empty()) {
      auto andOp = cast<stablehlo::AndOp>(worklist.pop_back_val());
      if (auto lhsAndOp = andOp.getLhs().getDefiningOp<stablehlo::AndOp>()) {
        worklist.push_back(lhsAndOp);
      } else {
        conjuncts.push_back(andOp.getLhs());
      }
      if (auto lhsAndOp = andOp.getRhs().getDefiningOp<stablehlo::AndOp>()) {
        worklist.push_back(lhsAndOp);
      } else {
        conjuncts.push_back(andOp.getRhs());
      }
    }
    if (conjuncts.size() <= 2)
      return failure();

    Value compareLhs = nullptr;
    auto compares = llvm::filter_to_vector(conjuncts, [&](Value v) {
      auto cmpOp = v.getDefiningOp<stablehlo::CompareOp>();
      if (!cmpOp)
        return false;
      if (compareLhs == nullptr)
        compareLhs = cmpOp.getLhs();
      else if (compareLhs != cmpOp.getLhs())
        return false;
      return cmpOp.getComparisonDirection() ==
             stablehlo::ComparisonDirection::LE;
    });

    if (compares.size() <= 1)
      return failure();

    Value running = compares[0].getDefiningOp<stablehlo::CompareOp>().getRhs();
    for (unsigned i = 1, e = compares.size(); i < e; ++i) {
      Value minRhs = compares[i].getDefiningOp<stablehlo::CompareOp>().getRhs();
      running =
          rewriter.create<stablehlo::MinOp>(andOp.getLoc(), running, minRhs);
    }
    Value replacement = rewriter.create<stablehlo::CompareOp>(
        compares[0].getLoc(), compareLhs, running,
        stablehlo::ComparisonDirection::LE);

    for (Value conjunct : conjuncts) {
      if (llvm::is_contained(compares, conjunct))
        continue;
      replacement = rewriter.create<stablehlo::AndOp>(andOp.getLoc(),
                                                      replacement, conjunct);
    }
    rewriter.replaceOp(andOp, replacement);
    return success();
  }
};

struct SliceReshapeSlice final
    : CheckedOpRewritePattern<stablehlo::SliceOp, SliceReshapeSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp op,
                                    PatternRewriter &rewriter) const {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape)
      return failure();

    if (!llvm::hasSingleElement(reshape->getUsers()))
      return failure();

    auto prev = reshape.getOperand().getDefiningOp<stablehlo::SliceOp>();
    if (!prev)
      return failure();

    SmallVector<int64_t> starts, limits, strides;
    if (!sliceReshapeHelper(op, starts, limits, strides).succeeded())
      return failure();

    sliceSliceHelper(prev, starts, limits, strides);
    auto newslice = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), prev.getOperand(), starts, limits, strides);
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      newslice);
    return success();
  }
};
} // namespace

// Rewritten from
// https:github.com/openxla/stablehlo/blob/4f180d3c2236a15f82f29aad1b47f6ea2c14fc52/stablehlo/reference/Ops.cpp#L1381
// using https://openxla.org/xla/operation_semantics#gather
// becuase xla/openxla differ in semantics
stablehlo::Index index(stablehlo::Tensor tensor) {
  stablehlo::Index result;
  for (auto it = tensor.index_begin(); it != tensor.index_end(); ++it)
    result.push_back(tensor.get(*it).getIntegerValue().getSExtValue());
  return result;
}

stablehlo::Tensor sliceOp(const stablehlo::Tensor &operand,
                          const stablehlo::Index &index) {
  using namespace stablehlo;
  Sizes start, limit;
  for (auto i = 0; i < operand.getRank(); ++i) {
    if (index[i] == -1) {
      start.push_back(0);
      limit.push_back(operand.getShape()[i]);
    } else {
      start.push_back(index[i]);
      limit.push_back(index[i] + 1);
    }
  }
  Sizes strides(operand.getRank(), 1);

  SmallVector<Type> inferredTypes;
  Builder builder(operand.getType().getContext());
  auto inferStatus = hlo::inferSliceOp({}, operand.getType(), start, limit,
                                       strides, inferredTypes);
  (void)inferStatus;
  assert(!failed(inferStatus));

  return stablehlo::sliceOp(operand, start, strides,
                            cast<mlir::ShapedType>(inferredTypes[0]));
}

bool is_iota(ArrayRef<int64_t> idx) {
  for (auto en : llvm::enumerate(idx))
    if (en.index() != en.value())
      return false;
  return true;
}

struct CSEIota : CheckedOpRewritePattern<stablehlo::IotaOp, CSEIota> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IotaOp op,
                                    PatternRewriter &rewriter) const {
    bool anyCsed = false;
    Operation *next = op->getNextNode();
    while (next) {
      if (auto iota = dyn_cast<stablehlo::IotaOp>(next)) {
        next = next->getNextNode();
        if (iota.getIotaDimension() == op.getIotaDimension() &&
            iota.getType() == op.getType()) {
          rewriter.replaceOp(iota, op);
          anyCsed = true;
        }
      } else
        break;
    }
    return success(anyCsed);
  }
};

bool opaque_cmp(Value a, Value b) {
  return a.getAsOpaquePointer() < b.getAsOpaquePointer();
}

bool isCommutativeEquivalent(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;

  if (lhs.size() == 1) {
    return lhs[0] == rhs[0];
  } else if (lhs.size() == 2) {
    return (lhs[0] == rhs[0] && lhs[1] == rhs[1]) ||
           (lhs[1] == rhs[0] && lhs[0] == rhs[1]);
  } else {
    auto lhsv = llvm::to_vector(lhs);
    auto rhsv = llvm::to_vector(rhs);
    llvm::sort(lhsv, opaque_cmp);
    llvm::sort(rhsv, opaque_cmp);
    return lhsv == rhsv;
  }
}

template <typename T> struct CSE final : CheckedOpRewritePattern<T, CSE<T>> {
  using CheckedOpRewritePattern<T, CSE<T>>::CheckedOpRewritePattern;

  bool supportsDynamicShapes() { return true; }

  LogicalResult matchAndRewriteImpl(T op, PatternRewriter &rewriter) const {
    if (op->getNumOperands() > 0)
      for (auto nop : op->getOperand(0).getUsers()) {
        if (nop == op)
          continue;
        if (!isa<T>(nop))
          continue;
        if (nop->getBlock() != op->getBlock())
          continue;

        if (op->getName() != nop->getName())
          continue;

        if (!OperationEquivalence::isEquivalentTo(
                op, nop, OperationEquivalence::IgnoreLocations)) {
          // stablehlo defines a special trait for commutative operations.
          // check for that here.
          if constexpr (std::is_base_of_v<
                            ::mlir::hlo::OpTrait::IsCommutative<T>, T>) {
            auto opRange = op->getOperands();
            auto nopRange = nop->getOperands();
            if (!isCommutativeEquivalent(opRange, nopRange))
              continue;
          } else {
            continue;
          }
        }

        if (nop->isBeforeInBlock(op)) {
          rewriter.replaceOp(op, nop);
          return success();
        } else {
          rewriter.replaceOp(nop, op);
          return success();
        }
      }
    return failure();
  }
};

struct ConstPropThroughBarrier final
    : CheckedOpRewritePattern<stablehlo::OptimizationBarrierOp,
                              ConstPropThroughBarrier> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::OptimizationBarrierOp op,
                                    PatternRewriter &rewriter) const {

    SmallVector<Value> replacements;
    bool changed = false;
    for (auto &&[res, inp] : llvm::zip(op.getResults(), op.getOperands())) {
      if (inp.getDefiningOp<stablehlo::ConstantOp>()) {
        changed = true;
      } else if (res.use_empty()) {
        changed = true;
      } else {
        replacements.push_back(inp);
      }
    }
    if (!changed) {
      return failure();
    }

    auto nop = rewriter.create<stablehlo::OptimizationBarrierOp>(op.getLoc(),
                                                                 replacements);

    size_t idx = 0;
    SmallVector<Value> results;
    for (auto &&[res, inp] : llvm::zip(op.getResults(), op.getOperands())) {
      if (res.use_empty() || inp.getDefiningOp<stablehlo::ConstantOp>()) {
        results.push_back(inp);
      } else {
        results.push_back(nop->getResult(idx));
        idx++;
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// If there is a dus where part of the updated region is not used by later slice
// operations, pre-slice the operand and update to the original dus
struct DUSSliceSimplify final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp,
                              DUSSliceSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dusOp,
                                    PatternRewriter &rewriter) const {
    auto res = dusOp.getResult();
    auto update = dusOp.getUpdate();
    auto updateShape = update.getType().getShape();
    auto operand = dusOp.getOperand();

    auto resShape = res.getType().getShape();
    auto resRank = res.getType().getRank();

    SmallVector<int64_t> ignoredStart(resShape);
    SmallVector<int64_t> ignoredEnd(resRank, 0);
    SmallVector<stablehlo::SliceOp> slices;
    slices.reserve(resRank);
    for (auto &use : res.getUses()) {
      if (auto slice = dyn_cast<stablehlo::SliceOp>(use.getOwner())) {
        ignoredStart = llvm::map_to_vector(
            llvm::zip(slice.getStartIndices(), ignoredStart), [&](auto p) {
              auto &[start, oldIgnoreStart] = p;
              if (start < oldIgnoreStart)
                return start;
              return oldIgnoreStart;
            });
        ignoredEnd = llvm::map_to_vector(
            llvm::zip(slice.getLimitIndices(), ignoredEnd), [&](auto p) {
              auto &[thisIgnoreEnd, oldIgnoreEnd] = p;
              if (thisIgnoreEnd > oldIgnoreEnd)
                return thisIgnoreEnd;
              return oldIgnoreEnd;
            });
        slices.push_back(slice);
      } else {
        return rewriter.notifyMatchFailure(use.getOwner(),
                                           "Found non slice user");
      }
    }

    if (slices.empty())
      return rewriter.notifyMatchFailure(dusOp, "No slice users found");

    if (llvm::all_of(ignoredStart,
                     [](int64_t ignored) { return ignored == 0; }) &&
        llvm::all_of(llvm::zip(ignoredEnd, resShape), [](auto p) {
          auto &[i, s] = p;
          return i == s;
        }))
      return rewriter.notifyMatchFailure(dusOp, "No ignored regions");

    assert(llvm::all_of(llvm::zip(ignoredStart, ignoredEnd), [](auto p) {
      auto &[s, e] = p;
      return s <= e;
    }));

    SmallVector<Value> dusIndexVals = dusOp.getStartIndices();
    SmallVector<int64_t> dusStartIndices;
    dusStartIndices.reserve(resRank);
    for (Value idxVal : dusIndexVals) {
      DenseIntElementsAttr idxAttr;
      if (!matchPattern(idxVal, m_Constant(&idxAttr)) ||
          idxAttr.getNumElements() != 1)
        return rewriter.notifyMatchFailure(
            dusOp, "DUS indices must be constant scalars");
      dusStartIndices.push_back((*idxAttr.begin()).getSExtValue());
    }
    SmallVector<int64_t> dusEndIndices = llvm::map_to_vector(
        llvm::zip(dusStartIndices, updateShape),
        [](auto p) { return std::get<0>(p) + std::get<1>(p); });

    SmallVector<int64_t> strideOne(resRank, 1);

    auto loc = dusOp->getLoc();
    auto preSliceOperand = rewriter.create<stablehlo::SliceOp>(
        loc, operand, ignoredStart, ignoredEnd, strideOne);

    auto ignoredUpdateStart = llvm::map_to_vector(
        llvm::zip(ignoredStart, dusStartIndices), [](auto p) -> int64_t {
          auto &[ignoredStart, dusStart] = p;
          if (ignoredStart > dusStart)
            return ignoredStart - dusStart;
          return 0;
        });
    auto ignoredUpdateEnd =
        llvm::map_to_vector(llvm::zip(ignoredEnd, dusEndIndices, updateShape),
                            [](auto p) -> int64_t {
                              auto &[ignoredEnd, dusEnd, updateShape] = p;
                              if (ignoredEnd < dusEnd)
                                return updateShape - (dusEnd - ignoredEnd);
                              return updateShape;
                            });
    Value preSliceUpdate = rewriter.create<stablehlo::SliceOp>(
        loc, update, ignoredUpdateStart, ignoredUpdateEnd, strideOne);

    Type itype = dusIndexVals[0].getType();
    SmallVector<Value> newDusIndices = llvm::map_to_vector(
        llvm::zip(ignoredStart, dusStartIndices), [&](auto p) -> Value {
          auto &[ignored, dusStart] = p;
          int64_t start = 0;
          if (ignored < dusStart)
            start = dusStart - ignored;

          return rewriter.create<stablehlo::ConstantOp>(
              loc, itype, cast<ElementsAttr>(makeAttr(itype, start)));
        });

    LLVM_DEBUG(
        for (auto [idx, operandSize, updateSize]
             : llvm::zip_equal(
                 newDusIndices,
                 cast<RankedTensorType>(preSliceOperand.getType()).getShape(),
                 cast<RankedTensorType>(preSliceUpdate.getType()).getShape())) {
          APInt start;
          assert(matchPattern(idx, m_ConstantInt(&start)));
          int64_t vali = start.getSExtValue();
          assert(operandSize >= vali + updateSize);
        });

    auto newDus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
        loc, preSliceOperand, preSliceUpdate, newDusIndices);

    for (auto slice : slices) {
      SmallVector<int64_t> starts(slice.getStartIndices());
      SmallVector<int64_t> sizes(slice.getResult().getType().getShape());
      SmallVector<int64_t> limits;
      starts = llvm::map_to_vector(llvm::zip(starts, ignoredStart),
                                   [](auto p) -> int64_t {
                                     auto &[s, i] = p;
                                     assert(s >= i);
                                     return s - i;
                                   });
      limits =
          llvm::map_to_vector(llvm::zip(starts, sizes), [](auto p) -> int64_t {
            auto &[start, size] = p;
            return start + size;
          });
      rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
          slice, newDus, starts, limits, slice.getStrides());
    }
    rewriter.eraseOp(dusOp);

    return success();
  }
};

struct DUSToI32 final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSToI32> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dusOp,
                                    PatternRewriter &rewriter) const {
    auto i32 = rewriter.getI32Type();

    auto unrankedI32 = RankedTensorType::get({}, i32);

    auto startIndices = dusOp.getStartIndices();

    SmallVector<int64_t> newStartIndicesConst;
    for (auto idx : startIndices) {
      if (cast<RankedTensorType>(idx.getType()).getElementType() == i32)
        return failure();

      llvm::APInt val;
      if (!matchPattern(idx, m_ConstantInt(&val)))
        return failure();

      if (val.getSignificantBits() > 32)
        return failure();

      newStartIndicesConst.push_back(val.getZExtValue());
    }

    SmallVector<Value> newStartIndices;
    for (auto [val, idx] :
         llvm::zip_equal(newStartIndicesConst, startIndices)) {
      newStartIndices.push_back(rewriter.create<stablehlo::ConstantOp>(
          idx.getLoc(), unrankedI32,
          cast<ElementsAttr>(makeAttr(unrankedI32, val))));
    }

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        dusOp, dusOp.getOperand(), dusOp.getUpdate(), newStartIndices);

    return success();
  }
};

// Replaces DUS with a combination of slices and concats.
// Each run of the pattern handles one dimension at a time.
struct DUSToConcat final
    : CheckedOpRewritePattern<stablehlo::DynamicUpdateSliceOp, DUSToConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicUpdateSliceOp dusOp,
                                    PatternRewriter &rewriter) const {

    // --- Get Info & Check Static Requirements ---
    Value targetOperand = dusOp.getOperand();
    Value updateVal = dusOp.getUpdate();
    SmallVector<Value> dusIndexVals = dusOp.getStartIndices();

    auto targetType = dyn_cast<RankedTensorType>(targetOperand.getType());
    auto updateType = dyn_cast<RankedTensorType>(updateVal.getType());

    if (!targetType || !updateType || !targetType.hasStaticShape() ||
        !updateType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(dusOp, "Requires static shapes");
    }

    ArrayRef<int64_t> targetShape = targetType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();
    int rank = targetType.getRank();

    if (rank == 0) {
      // Replace scalar DUS with the update value directly
      rewriter.replaceOp(dusOp, updateVal);
      return success();
    }

    SmallVector<int64_t> dusStartIndices;
    dusStartIndices.reserve(rank);
    for (Value idxVal : dusIndexVals) {
      DenseIntElementsAttr idxAttr;
      if (!matchPattern(idxVal, m_Constant(&idxAttr)) ||
          idxAttr.getNumElements() != 1) {
        return rewriter.notifyMatchFailure(
            dusOp, "Requires constant scalar start indices");
      }
      dusStartIndices.push_back((*idxAttr.begin()).getSExtValue());
    }

    // --- Check Constraints ---
    int differingDim = -1;
    bool singleDifferingDim = false;
    for (int d = 0; d < rank; ++d) {
      // Check if update spans the full dimension
      if (dusStartIndices[d] == 0 && updateShape[d] == targetShape[d]) {
        continue; // Spans fully, this dimension is fine
      }
      if (differingDim == -1)
        singleDifferingDim = true;
      else
        singleDifferingDim = false;
      differingDim = d;
    }

    if (differingDim == -1) {
      // DUS covers the entire tensor, replace with update value
      if (targetType == updateType) {
        rewriter.replaceOp(dusOp, updateVal);
        return success();
      } else {
        // Types mismatch (shouldn't happen with valid DUS?), fail for safety
        return rewriter.notifyMatchFailure(
            dusOp, "Full tensor update but types mismatch");
      }
    }

    // Check if the differing dimension is a prefix or suffix update
    bool dusFromBeginning = (dusStartIndices[differingDim] == 0);
    bool dusToEnd =
        (dusStartIndices[differingDim] + updateShape[differingDim] ==
         targetShape[differingDim]);

    Location loc = dusOp.getLoc();
    SmallVector<Value> newDusIndices = dusIndexVals;

    SmallVector<int64_t> sliceStrides(rank, 1);

    Value slicePre;
    Value slicePost;

    auto getPrePost = [&](int64_t sliceAt) {
      SmallVector<int64_t> slicePreStarts(rank, 0);
      SmallVector<int64_t> slicePreLimits = llvm::to_vector(targetShape);

      SmallVector<int64_t> slicePostStarts(rank, 0);
      SmallVector<int64_t> slicePostLimits = llvm::to_vector(targetShape);

      slicePreStarts[differingDim] = 0;
      slicePreLimits[differingDim] = sliceAt;

      slicePostStarts[differingDim] = sliceAt;
      slicePostLimits[differingDim] = targetShape[differingDim];

      slicePre = rewriter.create<stablehlo::SliceOp>(
          loc, targetOperand, slicePreStarts, slicePreLimits, sliceStrides);

      slicePost = rewriter.create<stablehlo::SliceOp>(
          loc, targetOperand, slicePostStarts, slicePostLimits, sliceStrides);
    };

    auto itype = dusIndexVals[differingDim].getType();
    auto c0 = rewriter.create<stablehlo::ConstantOp>(
        loc, itype, cast<ElementsAttr>(makeAttr(itype, 0)));
    newDusIndices[differingDim] = c0;

    if (!dusFromBeginning) {
      int64_t sliceAt = dusStartIndices[differingDim];
      getPrePost(sliceAt);

      auto c0 = rewriter.create<stablehlo::ConstantOp>(
          loc, itype, cast<ElementsAttr>(makeAttr(itype, 0)));
      newDusIndices[differingDim] = c0;

      if (!(singleDifferingDim && dusToEnd)) {
        slicePost = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
            loc, slicePost, updateVal, newDusIndices);
      }

    } else {
      assert(!dusToEnd);
      int64_t sliceAt =
          dusStartIndices[differingDim] + updateShape[differingDim];
      getPrePost(sliceAt);

      if (!(singleDifferingDim && dusFromBeginning)) {
        slicePre = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
            loc, slicePre, updateVal, newDusIndices);
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        dusOp, dusOp.getType(), ValueRange{slicePre, slicePost}, differingDim);

    return success();
  }
};

// I.   (div a (div b c)) -> (div (mul a c) b)
// II.  (div (div a b) c) -> (div a (mul b c))
// III. (div (div a b) (div c d)) -> (div (mul a d) (mul b c))
struct DivideDivideSimplify
    : public CheckedOpRewritePattern<stablehlo::DivOp, DivideDivideSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DivOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (auto lhsDivOp = lhs.getDefiningOp<stablehlo::DivOp>()) {
      if (!lhsDivOp->hasOneUse())
        return failure();

      if (auto rhsDivOp = rhs.getDefiningOp<stablehlo::DivOp>()) {
        if (!rhsDivOp->hasOneUse())
          return failure();

        // Case III.
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op,
            rewriter.create<stablehlo::MulOp>(
                op.getLoc(), lhsDivOp->getOperand(0), rhsDivOp->getOperand(1)),
            rewriter.create<stablehlo::MulOp>(
                op.getLoc(), lhsDivOp->getOperand(1), rhsDivOp->getOperand(0)));
        return success();
      } else {
        // Case II.
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op, lhsDivOp->getOperand(0),
            rewriter.create<stablehlo::MulOp>(op.getLoc(),
                                              lhsDivOp->getOperand(1), rhs));
        return success();
      }
    } else {
      if (auto rhsDivOp = rhs.getDefiningOp<stablehlo::DivOp>()) {
        if (!rhsDivOp->hasOneUse())
          return failure();

        // Case I.
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op,
            rewriter.create<stablehlo::MulOp>(op.getLoc(), lhs,
                                              rhsDivOp->getOperand(1)),
            rhsDivOp->getOperand(0));
        return success();
      }
    }

    return failure();
  }
};

//////////////// Imported from stablehlo
static bool isIotaRange(ArrayRef<int64_t> dims) {
  return llvm::all_of(llvm::enumerate(dims), [](const auto &it) {
    return static_cast<int64_t>(it.index()) == it.value();
  });
}

/// Matches when either of the submatchers match.
template <typename MatcherA, typename MatcherB> struct m_AnyOf {
  m_AnyOf(MatcherA a, MatcherB b) : matcherA(a), matcherB(b) {}

  bool match(Operation *op) { return matcherA.match(op) || matcherB.match(op); }

  MatcherA matcherA;
  MatcherB matcherB;
};

template <typename MatcherA, typename MatcherB>
m_AnyOf(MatcherA, MatcherB) -> m_AnyOf<MatcherA, MatcherB>;

/// Binary constant folder that used a generic folder function to handle both
/// ints and floats.
template <typename Fn>
static TypedAttr foldBinaryOpIntOrFloat(TypedAttr lhs, TypedAttr rhs,
                                        Fn &&folder) {
  Attribute operands[2] = {lhs, rhs};
  Type elemTy = getElementTypeOrSelf(lhs);

  Attribute res;
  if (isa<IntegerType>(elemTy))
    res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(operands,
                                                                       folder);
  if (isa<FloatType>(elemTy))
    res = constFoldBinaryOp<FloatAttr, FloatAttr::ValueType, void>(operands,
                                                                   folder);
  if (res)
    return cast<TypedAttr>(res);

  return nullptr;
}

static stablehlo::ComparisonDirection
invertDirection(stablehlo::ComparisonDirection direction) {
  using stablehlo::ComparisonDirection;

  switch (direction) {
  case ComparisonDirection::EQ:
  case ComparisonDirection::NE:
    return direction;
  case ComparisonDirection::GE:
    return ComparisonDirection::LE;
  case ComparisonDirection::GT:
    return ComparisonDirection::LT;
  case ComparisonDirection::LE:
    return ComparisonDirection::GE;
  case ComparisonDirection::LT:
    return ComparisonDirection::GT;
  }

  llvm::report_fatal_error("Unhandled case");
}

static APInt calculateComp(stablehlo::ComparisonType kind,
                           stablehlo::ComparisonDirection direction,
                           const APInt &lhs, const APInt &rhs) {
  using stablehlo::ComparisonDirection;
  using stablehlo::ComparisonType;
  assert(llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            kind) &&
         "Not an integer comparison");

  auto asBit = [](bool value) {
    return value ? APInt::getAllOnes(1) : APInt::getZero(1);
  };

  switch (direction) {
  case ComparisonDirection::EQ:
    return asBit(lhs == rhs);
  case ComparisonDirection::NE:
    return asBit(lhs != rhs);
  case ComparisonDirection::GE:
    return asBit(kind == ComparisonType::SIGNED ? lhs.sge(rhs) : lhs.uge(rhs));
  case ComparisonDirection::GT:
    return asBit(kind == ComparisonType::SIGNED ? lhs.sgt(rhs) : lhs.ugt(rhs));
  case ComparisonDirection::LE:
    return asBit(kind == ComparisonType::SIGNED ? lhs.sle(rhs) : lhs.ule(rhs));
  case ComparisonDirection::LT:
    return asBit(kind == ComparisonType::SIGNED ? lhs.slt(rhs) : lhs.ult(rhs));
  }

  llvm_unreachable("Unhandled case");
}

struct CompareOpCanon final
    : CheckedOpRewritePattern<stablehlo::CompareOp, CompareOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType type = op.getType();

    {
      DenseElementsAttr lhs;
      matchPattern(op.getLhs(), m_Constant(&lhs));
      DenseElementsAttr rhs;
      matchPattern(op.getRhs(), m_Constant(&rhs));
      if (lhs && rhs) {
        bool isSplat = lhs.isSplat() && rhs.isSplat();
        auto ty = isSplat
                      ? RankedTensorType::get({}, op.getType().getElementType())
                      : op.getType();
        auto out = fromTensor(stablehlo::compareOp(
            isSplat
                ? stablehlo::makeTensor(lhs.resizeSplat(RankedTensorType::get(
                      {}, lhs.getType().getElementType())))
                : stablehlo::constantOp(lhs),
            isSplat
                ? stablehlo::makeTensor(rhs.resizeSplat(RankedTensorType::get(
                      {}, rhs.getType().getElementType())))
                : stablehlo::constantOp(rhs),
            op.getComparisonDirection(), ty));
        if (isSplat)
          out = out.resizeSplat(op.getType());

        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                           out);
        return success();
      }
    }

    // Bail out on non-integer comparison.
    // TODO: Support more comparison types.
    using stablehlo::ComparisonType;
    std::optional<ComparisonType> compType = op.getCompareType();
    if (!compType ||
        !llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            *compType)) {
      return failure();
    }

    using stablehlo::ComparisonDirection;
    ComparisonDirection direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhs == rhs) {
      switch (direction) {
      case ComparisonDirection::EQ:
      case ComparisonDirection::GE:
      case ComparisonDirection::LE: {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, SplatElementsAttr::get(type, rewriter.getBoolAttr(true)));
        return success();
      }
      case ComparisonDirection::GT:
      case ComparisonDirection::LT:
      case ComparisonDirection::NE: {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, rewriter.getZeroAttr(type));
        return success();
      }
      }
      llvm_unreachable("Unhandled case");
    }

    TypedAttr lhsAttr;
    matchPattern(lhs, m_Constant(&lhsAttr));

    TypedAttr rhsAttr;
    matchPattern(rhs, m_Constant(&rhsAttr));

    // The canonical form has the constant operand as the RHS.
    if (lhsAttr && !rhsAttr) {
      rewriter.modifyOpInPlace(op, [&op, direction, lhs, rhs] {
        op.setComparisonDirection(invertDirection(direction));
        op->setOperands(ValueRange{rhs, lhs});
      });
      return success();
    }

    if (Attribute res;
        lhsAttr && rhsAttr &&
        (res = constFoldBinaryOp<IntegerAttr, IntegerAttr::ValueType, void>(
             ArrayRef<Attribute>({lhsAttr, rhsAttr}), op.getType(),
             [direction, kind = *compType](const APInt &a, const APInt &b) {
               return calculateComp(kind, direction, a, b);
             }))) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct CompareExt final
    : CheckedOpRewritePattern<stablehlo::CompareOp, CompareExt> {
  using CheckedOpRewritePattern<stablehlo::CompareOp,
                                CompareExt>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp op,
                                    PatternRewriter &rewriter) const {
    auto elemType =
        cast<RankedTensorType>(op.getLhs().getType()).getElementType();
    if (!elemType.isInteger())
      return failure();

    auto direction = op.getComparisonDirection();

    auto lhsConvert = op.getLhs().getDefiningOp<stablehlo::ConvertOp>();
    auto rhsConvert = op.getRhs().getDefiningOp<stablehlo::ConvertOp>();
    if (!lhsConvert && !rhsConvert)
      return failure();

    auto isConvertFromBool = [](stablehlo::ConvertOp cvtOp) -> bool {
      return cvtOp && cast<RankedTensorType>(cvtOp.getOperand().getType())
                          .getElementType()
                          .isInteger(1);
    };

    if (isConvertFromBool(lhsConvert) && isConvertFromBool(rhsConvert) &&
        direction == stablehlo::ComparisonDirection::EQ) {
      rewriter.replaceOpWithNewOp<stablehlo::AndOp>(op, lhsConvert.getOperand(),
                                                    rhsConvert.getOperand());
      return success();
    }

    if (isConvertFromBool(lhsConvert) &&
        direction == stablehlo::ComparisonDirection::EQ) {
      if (matchPattern(op.getRhs(), m_One())) {
        rewriter.replaceOp(op, lhsConvert.getOperand());
        return success();
      } else if (matchPattern(op.getRhs(), m_Zero())) {
        rewriter.replaceOpWithNewOp<stablehlo::NotOp>(op,
                                                      lhsConvert.getOperand());
        return success();
      }
    }

    if (isConvertFromBool(rhsConvert) &&
        direction == stablehlo::ComparisonDirection::EQ) {
      if (matchPattern(op.getLhs(), m_One())) {
        rewriter.replaceOp(op, rhsConvert.getOperand());
        return success();
      } else if (matchPattern(op.getLhs(), m_Zero())) {
        rewriter.replaceOpWithNewOp<stablehlo::NotOp>(op,
                                                      rhsConvert.getOperand());
        return success();
      }
    }

    return failure();
  }
};

stablehlo::ComparisonDirection
negatedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GT;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GE;
  }
}

struct SelectCompIotaConstSimplify final
    : CheckedOpRewritePattern<stablehlo::SelectOp,
                              SelectCompIotaConstSimplify> {
  struct slice_data {
    Value tensor;
    int64_t count;
  };
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp selectOp,
                                    PatternRewriter &rewriter) const {
    DenseIntElementsAttr inp;

    Value compare = selectOp.getPred();
    Value trueTensor = selectOp.getOnTrue();
    Value falseTensor = selectOp.getOnFalse();
    if (!matchPattern(selectOp.getOperation(),
                      m_Op<stablehlo::SelectOp>(m_Op<stablehlo::CompareOp>(),
                                                matchers::m_Any(&trueTensor),
                                                matchers::m_Any(&falseTensor))))
      return failure();

    auto tensorType = selectOp.getType();
    if (!tensorType)
      return failure();

    auto shapeLimit = tensorType.getShape();

    Value cmpLHS = nullptr;
    Value cmpRHS = nullptr;
    stablehlo::ComparisonDirection flag;
    if (auto cmp = compare.getDefiningOp<stablehlo::CompareOp>()) {
      cmpLHS = cmp.getLhs();
      cmpRHS = cmp.getRhs();
      flag = cmp.getComparisonDirection();
    } else if (auto notop = compare.getDefiningOp<stablehlo::NotOp>()) {
      if (auto cmp = compare.getDefiningOp<stablehlo::CompareOp>()) {
        cmpLHS = cmp.getLhs();
        cmpRHS = cmp.getRhs();
        flag = negatedComparisonDirection(cmp.getComparisonDirection());
      }
    }

    if (!cmpLHS)
      return failure();

    stablehlo::IotaOp iota;
    if (!(matchPattern(cmpLHS, m_Op<stablehlo::IotaOp>()) &&
          matchPattern(cmpRHS, m_Constant(&inp)))) {
      if (matchPattern(cmpRHS, m_Op<stablehlo::IotaOp>()) &&
          matchPattern(cmpLHS, m_Constant(&inp))) {
        // incoming: const `op` iota
        // treat the match as iota `op` const
        switch (flag) {
        case stablehlo::ComparisonDirection::LT:
          flag = stablehlo::ComparisonDirection::GT;
          break;
        case stablehlo::ComparisonDirection::LE:
          flag = stablehlo::ComparisonDirection::GE;
          break;
        case stablehlo::ComparisonDirection::GT:
          flag = stablehlo::ComparisonDirection::LT;
          break;
        case stablehlo::ComparisonDirection::GE:
          flag = stablehlo::ComparisonDirection::LE;
          break;
        default:
          break;
        }
        iota = cast<stablehlo::IotaOp>(cmpRHS.getDefiningOp());
      } else
        return failure();
    } else
      iota = cast<stablehlo::IotaOp>(cmpLHS.getDefiningOp());

    assert(iota);

    const auto iotaDim = iota.getIotaDimension();

    if (!inp.isSplat())
      return failure();

    auto constValue =
        inp.getSplatValue<IntegerAttr>().getValue().getSExtValue();
    auto endValue = shapeLimit[iotaDim];

    SmallVector<slice_data, 3> slices;

    switch (flag) {
    case stablehlo::ComparisonDirection::LT:
      slices.push_back(slice_data{trueTensor, constValue});
      slices.push_back(slice_data{falseTensor, endValue - constValue});
      break;
    case stablehlo::ComparisonDirection::LE:
      slices.push_back(slice_data{trueTensor, constValue + 1});
      slices.push_back(slice_data{falseTensor, endValue - (constValue + 1)});
      break;

    case stablehlo::ComparisonDirection::GT:
      slices.push_back(slice_data{falseTensor, constValue + 1});
      slices.push_back(slice_data{trueTensor, endValue - (constValue + 1)});
      break;
    case stablehlo::ComparisonDirection::GE:
      slices.push_back(slice_data{falseTensor, constValue});
      slices.push_back(slice_data{trueTensor, endValue - constValue});
      break;

    case stablehlo::ComparisonDirection::EQ:
      slices.push_back(slice_data{falseTensor, constValue});
      slices.push_back(slice_data{trueTensor, 1});
      slices.push_back(slice_data{falseTensor, endValue - constValue - 1});
      break;
    case stablehlo::ComparisonDirection::NE:
      slices.push_back(slice_data{trueTensor, constValue});
      slices.push_back(slice_data{falseTensor, 1});
      slices.push_back(slice_data{trueTensor, endValue - constValue - 1});
      break;
    }

    assert(slices.size() >= 2);

    auto valid_slices =
        std::count_if(slices.begin(), slices.end(),
                      [](slice_data data) { return data.count > 0; });
    if (valid_slices == 1) {
      for (auto &elem : slices) {
        if (elem.count > 0) {
          // if we are the only usable slice, replace the result
          rewriter.replaceAllOpUsesWith(selectOp, elem.tensor);
          return success();
        }
      }
    }

    SmallVector<Value, 3> sliceValues;
    {
      int64_t start = 0;
      const auto loc = selectOp.getLoc();
      SmallVector<int64_t> startIndices(shapeLimit.size(), 0);
      SmallVector<int64_t> limitIndices{shapeLimit};
      SmallVector<int64_t> strides(shapeLimit.size(), 1);

      for (const auto &elem : slices) {
        if (elem.count > 0) {
          startIndices[iotaDim] = start;
          limitIndices[iotaDim] = start + elem.count;
          auto slice = rewriter.create<stablehlo::SliceOp>(
              loc, elem.tensor, llvm::ArrayRef<int64_t>(startIndices),
              llvm::ArrayRef<int64_t>(limitIndices),
              llvm::ArrayRef<int64_t>(strides));
          sliceValues.push_back(slice);
          start += elem.count;
        }
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        selectOp, ValueRange{sliceValues}, iotaDim);

    return success();
  }
};

struct SelectCompIotaConstToDUS final
    : CheckedOpRewritePattern<stablehlo::SelectOp, SelectCompIotaConstToDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp selectOp,
                                    PatternRewriter &rewriter) const {
    auto pred = selectOp.getPred().getDefiningOp<stablehlo::AndOp>();
    if (!pred)
      return failure();
    Value trueTensor = selectOp.getOnTrue();
    Value falseTensor = selectOp.getOnFalse();

    Value lhs[2];
    Value rhs[2];
    stablehlo::ComparisonDirection direction[2];
    for (int i = 0; i < 2; i++) {
      Value cmpLHS = nullptr;
      Value cmpRHS = nullptr;
      stablehlo::ComparisonDirection flag;
      auto compare = pred->getOperand(i);
      if (auto cmp = compare.getDefiningOp<stablehlo::CompareOp>()) {
        cmpLHS = cmp.getLhs();
        cmpRHS = cmp.getRhs();
        flag = cmp.getComparisonDirection();
      } else if (auto notop = compare.getDefiningOp<stablehlo::NotOp>()) {
        if (auto cmp = compare.getDefiningOp<stablehlo::CompareOp>()) {
          cmpLHS = cmp.getLhs();
          cmpRHS = cmp.getRhs();
          flag = negatedComparisonDirection(cmp.getComparisonDirection());
        }
      }

      if (!cmpLHS)
        return failure();
      lhs[i] = cmpLHS;
      rhs[i] = cmpRHS;
      direction[i] = flag;
    }

    if (lhs[0] != lhs[1])
      return failure();

    int dimension = -1;
    int start = 0;

    if (stablehlo::IotaOp iota = lhs[0].getDefiningOp<stablehlo::IotaOp>()) {
      dimension = iota.getIotaDimension();
      start = 0;
    } else if (auto sl = lhs[0].getDefiningOp<stablehlo::SliceOp>()) {
      if (stablehlo::IotaOp iota =
              sl.getOperand().getDefiningOp<stablehlo::IotaOp>()) {
        dimension = iota.getIotaDimension();
        start = sl.getStartIndices()[dimension];
      }
    }

    if (dimension == -1)
      return failure();

    int64_t constants[2];
    for (int i = 0; i < 2; i++) {
      DenseIntElementsAttr constant;
      if (!matchPattern(rhs[i], m_Constant(&constant)))
        return failure();
      if (!constant.isSplat())
        return failure();

      constants[i] =
          constant.getSplatValue<IntegerAttr>().getValue().getSExtValue();
    }

    for (int i = 0; i < 2; i++) {

      auto lb = constants[i] - start;
      auto ub = constants[1 - i] - start;

      auto lb_pred = direction[i];
      bool legalLB = false;
      if (lb_pred == stablehlo::ComparisonDirection::GE) {
        legalLB = true;
      } else if (lb_pred == stablehlo::ComparisonDirection::GT) {
        legalLB = true;
        lb++;
      } else if (lb_pred == stablehlo::ComparisonDirection::NE && lb == 0) {
        legalLB = true;
        lb++;
      }

      if (!legalLB)
        continue;

      auto ub_pred = direction[1 - i];
      if (ub_pred != stablehlo::ComparisonDirection::LT &&
          ub_pred != stablehlo::ComparisonDirection::LE)
        continue;

      if (ub_pred == stablehlo::ComparisonDirection::LE) {
        ub++;
      }

      if (ub > selectOp.getType().getShape()[dimension])
        ub = selectOp.getType().getShape()[dimension];
      if (lb >= ub)
        continue;
      if (lb < 0)
        lb = 0;
      if (ub < 0)
        ub = 0;

      auto ITy = RankedTensorType::get({}, rewriter.getI32Type());

      SmallVector<int64_t> startSlices(selectOp.getType().getShape().size(), 0);
      SmallVector<int64_t> limits =
          llvm::to_vector(selectOp.getType().getShape());
      SmallVector<int64_t> step(selectOp.getType().getShape().size(), 1);
      startSlices[dimension] = lb;
      limits[dimension] = ub;

      auto slicedTrueTensor = rewriter.create<stablehlo::SliceOp>(
          selectOp.getLoc(), trueTensor, startSlices, limits, step);

      SmallVector<Value> starts(
          selectOp.getType().getShape().size(),
          rewriter.create<stablehlo::ConstantOp>(
              selectOp.getLoc(), ITy, cast<ElementsAttr>(makeAttr(ITy, 0))));

      starts[dimension] = rewriter.create<stablehlo::ConstantOp>(
          selectOp.getLoc(), ITy, cast<ElementsAttr>(makeAttr(ITy, lb)));

      rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
          selectOp, falseTensor, slicedTrueTensor, starts);
      return success();
    }

    return failure();
  }
};

struct SelectPadToDUS final
    : CheckedOpRewritePattern<stablehlo::SelectOp, SelectPadToDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp selectOp,
                                    PatternRewriter &rewriter) const {
    auto pad = selectOp.getPred().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();
    Value trueTensor = selectOp.getOnTrue();
    Value falseTensor = selectOp.getOnFalse();

    SplatElementsAttr operand, padded;
    if (!matchPattern(pad.getOperand(), m_Constant(&operand))) {
      return failure();
    }
    if (!matchPattern(pad.getPaddingValue(), m_Constant(&padded))) {
      return failure();
    }

    bool operandV = !operand.getSplatValue<IntegerAttr>().getValue().isZero();
    bool paddedV = !padded.getSplatValue<IntegerAttr>().getValue().isZero();
    if (operandV == paddedV)
      return failure();

    for (auto pv : pad.getInteriorPadding()) {
      if (pv != 0)
        return failure();
    }

    SmallVector<int64_t> startSlices = llvm::to_vector(pad.getEdgePaddingLow());
    SmallVector<int64_t> limits =
        llvm::to_vector(selectOp.getType().getShape());
    for (int i = 0; i < selectOp.getType().getShape().size(); i++) {
      limits[i] -= pad.getEdgePaddingHigh()[i];
    }
    SmallVector<int64_t> step(selectOp.getType().getShape().size(), 1);

    Value dusOperand = rewriter.create<stablehlo::SliceOp>(
        selectOp.getLoc(), operandV ? trueTensor : falseTensor, startSlices,
        limits, step);

    auto ITy = RankedTensorType::get({}, rewriter.getI32Type());
    SmallVector<Value> starts;
    for (int i = 0; i < selectOp.getType().getShape().size(); i++) {
      starts.push_back(rewriter.create<stablehlo::ConstantOp>(
          selectOp.getLoc(), ITy,
          cast<ElementsAttr>(makeAttr(ITy, pad.getEdgePaddingLow()[i]))));
    }

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        selectOp, operandV ? falseTensor : trueTensor, dusOperand, starts);
    return success();
  }
};

struct AndPadPad final : CheckedOpRewritePattern<stablehlo::AndOp, AndPadPad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AndOp andOp,
                                    PatternRewriter &rewriter) const {
    auto padLHS = andOp.getLhs().getDefiningOp<stablehlo::PadOp>();
    if (!padLHS)
      return failure();
    auto padRHS = andOp.getRhs().getDefiningOp<stablehlo::PadOp>();
    if (!padRHS)
      return failure();

    SplatElementsAttr operandLHS, paddedLHS;
    if (!matchPattern(padLHS.getOperand(), m_Constant(&operandLHS))) {
      return failure();
    }
    if (!matchPattern(padLHS.getPaddingValue(), m_Constant(&paddedLHS))) {
      return failure();
    }

    bool operandLHSV =
        !operandLHS.getSplatValue<IntegerAttr>().getValue().isZero();
    bool paddedLHSV =
        !paddedLHS.getSplatValue<IntegerAttr>().getValue().isZero();
    if (operandLHSV == paddedLHSV)
      return failure();

    for (auto pv : padLHS.getInteriorPadding()) {
      if (pv != 0)
        return failure();
    }

    SplatElementsAttr operandRHS, paddedRHS;
    if (!matchPattern(padRHS.getOperand(), m_Constant(&operandRHS))) {
      return failure();
    }
    if (!matchPattern(padRHS.getPaddingValue(), m_Constant(&paddedRHS))) {
      return failure();
    }

    bool operandRHSV =
        !operandRHS.getSplatValue<IntegerAttr>().getValue().isZero();
    bool paddedRHSV =
        !paddedRHS.getSplatValue<IntegerAttr>().getValue().isZero();
    if (operandRHSV == paddedRHSV)
      return failure();

    for (auto pv : padLHS.getInteriorPadding()) {
      if (pv != 0)
        return failure();
    }

    int64_t idx = -1;
    for (int i = 0; i < padLHS.getInteriorPadding().size(); i++) {
      if (padLHS.getEdgePaddingLow()[i] != 0 ||
          padRHS.getEdgePaddingLow()[i] != 0 ||
          padLHS.getEdgePaddingHigh()[i] != 0 ||
          padRHS.getEdgePaddingHigh()[i] != 0) {
        if (idx == -1) {
          idx = i;
        } else {
          return failure();
        }
      }
    }
    if (idx == -1)
      return failure();

    std::set<int64_t> boundaries;
    boundaries.insert(0);
    boundaries.insert(padLHS.getType().getShape()[idx]);
    boundaries.insert(padLHS.getEdgePaddingLow()[idx]);
    boundaries.insert(padRHS.getEdgePaddingLow()[idx]);

    boundaries.insert(padLHS.getType().getShape()[idx] -
                      padLHS.getEdgePaddingHigh()[idx]);
    boundaries.insert(padRHS.getType().getShape()[idx] -
                      padRHS.getEdgePaddingHigh()[idx]);
    SmallVector<int64_t> boundariesV = llvm::to_vector(boundaries);

    SmallVector<bool> toConcat;
    SmallVector<int64_t> start;
    for (int i = 0; i < boundariesV.size() - 1; i++) {
      auto lhsV = ((boundariesV[i] >= padLHS.getEdgePaddingLow()[idx]) &&
                   (boundariesV[i] < (padLHS.getType().getShape()[idx] -
                                      padLHS.getEdgePaddingHigh()[idx])))
                      ? operandLHSV
                      : paddedLHSV;
      auto rhsV = ((boundariesV[i] >= padRHS.getEdgePaddingLow()[idx]) &&
                   (boundariesV[i] < (padRHS.getType().getShape()[idx] -
                                      padRHS.getEdgePaddingHigh()[idx])))
                      ? operandRHSV
                      : paddedRHSV;
      auto newV = lhsV & rhsV;
      if (start.size() == 0 || newV != toConcat.back()) {
        toConcat.push_back(newV);
        start.push_back(boundariesV[i]);
      }
    }

    start.push_back(padLHS.getType().getShape()[idx]);

    if (toConcat.size() == 3) {
      auto shape = llvm::to_vector(andOp.getType().getShape());
      shape[idx] = start[2] - start[1];
      auto RT = RankedTensorType::get(shape, andOp.getType().getElementType());

      auto newInner = rewriter.create<stablehlo::ConstantOp>(
          andOp.getLoc(), RT, cast<ElementsAttr>(makeAttr(RT, toConcat[1])));

      auto RT0D = RankedTensorType::get({}, andOp.getType().getElementType());
      auto newOuter = rewriter.create<stablehlo::ConstantOp>(
          andOp.getLoc(), RT0D,
          cast<ElementsAttr>(makeAttr(RT0D, toConcat[0])));

      SmallVector<int64_t> low(padLHS.getInteriorPadding().size(), 0);
      low[idx] = start[1];

      SmallVector<int64_t> high(padLHS.getInteriorPadding().size(), 0);
      high[idx] = start[3] - start[2];

      rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
          andOp, newInner, newOuter, low, high, padLHS.getInteriorPadding());
      return success();
    }

    SmallVector<Value> toConcatV;
    for (int i = 0; i < start.size() - 1; i++) {
      auto shape = llvm::to_vector(andOp.getType().getShape());
      shape[idx] = start[i + 1] - start[i];
      auto RT = RankedTensorType::get(shape, andOp.getType().getElementType());
      toConcatV.push_back(rewriter.create<stablehlo::ConstantOp>(
          andOp.getLoc(), RT, cast<ElementsAttr>(makeAttr(RT, toConcat[i]))));
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(andOp, toConcatV,
                                                          idx);
    return success();
  }
};

struct SelectOpUsedWithinIf final
    : CheckedOpRewritePattern<stablehlo::SelectOp, SelectOpUsedWithinIf> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp op,
                                    PatternRewriter &rewriter) const {
    Value pred = op.getPred();
    Value result = op.getResult();

    if (cast<TensorType>(pred.getType()).getRank() != 0)
      return failure();

    auto block = op->getBlock();

    bool anyModified = false;

    rewriter.replaceUsesWithIf(result, op.getOnTrue(), [&](auto &use) {
      Operation *user = use.getOwner();
      if (user->getBlock() == block)
        return false;

      Operation *p = user->getParentOp();
      while (p && p != op) {
        if (auto ifOp = dyn_cast<stablehlo::IfOp>(p)) {
          if (ifOp.getPred() == pred &&
              ifOp.getTrueBranch().isAncestor(user->getParentRegion())) {
            anyModified = true;
            return true;
          }
        }
        p = p->getParentOp();
      }
      return false;
    });

    rewriter.replaceUsesWithIf(result, op.getOnFalse(), [&](auto &use) {
      Operation *user = use.getOwner();
      if (user->getBlock() == block)
        return false;

      Operation *p = user->getParentOp();
      while (p && p != op) {
        if (auto ifOp = dyn_cast<stablehlo::IfOp>(p)) {
          if (ifOp.getPred() == pred &&
              ifOp.getFalseBranch().isAncestor(user->getParentRegion())) {
            anyModified = true;
            return true;
          }
        }
        p = p->getParentOp();
      }
      return false;
    });

    return success(anyModified);
  }
};

struct SelectOpCanon final
    : CheckedOpRewritePattern<stablehlo::SelectOp, SelectOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  size_t max_constant_expansion;
  SelectOpCanon(size_t max_constant_expansion, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType type = op.getType();

    Value trueVal = op.getOnTrue();
    Value falseVal = op.getOnFalse();

    // Eliminate select with two identical outcomes.
    if (trueVal == falseVal) {
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    // Simplify when the condition is a constant.
    Value pred = op.getPred();
    ElementsAttr cond;
    if (!matchPattern(pred, m_Constant(&cond)))
      return failure();

    // Handle splat predicate and select either `trueVal` or `falseVal`.
    if (cond.isSplat()) {
      rewriter.replaceOp(op, cond.getSplatValue<bool>() ? trueVal : falseVal);
      return success();
    }

    // Handle elementwise selection when both outcomes are also constants. This
    // will create a new, likely non-splat constant.
    if (cond.getNumElements() > max_constant_expansion)
      return failure();

    ElementsAttr trueAttr;
    if (!matchPattern(trueVal, m_Constant(&trueAttr)))
      return failure();

    ElementsAttr falseAttr;
    if (!matchPattern(falseVal, m_Constant(&falseAttr)))
      return failure();

    SmallVector<Attribute> newValues;
    newValues.reserve(cond.getNumElements());
    for (auto [condElem, trueElem, falseElem] : llvm::zip_equal(
             cond.getValues<bool>(), trueAttr.getValues<Attribute>(),
             falseAttr.getValues<Attribute>())) {
      newValues.push_back(condElem ? trueElem : falseElem);
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(type, newValues));
    return success();
  }
};

struct BroadcastInDimOpCanon final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                              BroadcastInDimOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType type = op.getType();

    TypedValue<RankedTensorType> operand = op.getOperand();
    RankedTensorType operandTy = operand.getType();

    // Fold when broadcast is a noop.
    auto dims = op.getBroadcastDimensions();
    bool isDimsIota = isIotaRange(dims);
    if (type == operandTy && isDimsIota) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Handle splat broadcasts.
    if (SplatElementsAttr cstAttr;
        matchPattern(operand, m_Constant(&cstAttr))) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }

    if (operandTy.hasStaticShape() && type.hasStaticShape() &&
        type.getNumElements() == operandTy.getNumElements()) {
      // BroadcastInDim equivalent to reshape.
      if (isDimsIota) {
        rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, type, operand);
        return success();
      }
      // BroadcastInDim equivalent to transpose.
      if (type.getRank() == operandTy.getRank()) {
        rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
            op, type, operand, getInversePermutation(dims));
        return success();
      }
    }

    // Eliminate redundant nested BroadcastInDim.
    if (auto definingOp =
            operand.getDefiningOp<stablehlo::BroadcastInDimOp>()) {
      auto newIndices = llvm::to_vector(
          llvm::map_range(definingOp.getBroadcastDimensions(),
                          [&dims](int64_t dim) { return dims[dim]; }));
      rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
          op, type, definingOp.getOperand(), newIndices);
      return success();
    }

    return failure();
  }
};

struct TransposeBroadcastInDimToBroadcastInDim final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                              TransposeBroadcastInDimToBroadcastInDim> {
  using CheckedOpRewritePattern<
      stablehlo::BroadcastInDimOp,
      TransposeBroadcastInDimToBroadcastInDim>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    auto transposeOp = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    auto broadcastDims = op.getBroadcastDimensions();
    auto permutation = transposeOp.getPermutation();

    // For each input dimension, find where it maps in final output
    SmallVector<int64_t> newBroadcastDims(
        transposeOp.getOperand().getType().getRank(), -1);
    for (auto [idx, oldDim] : llvm::enumerate(broadcastDims)) {
      newBroadcastDims[permutation[idx]] = oldDim;
    }

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), transposeOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(newBroadcastDims));

    return success();
  }
};

struct BroadcastInDimTransposeToBroadcastInDim final
    : CheckedOpRewritePattern<stablehlo::TransposeOp,
                              BroadcastInDimTransposeToBroadcastInDim> {
  using CheckedOpRewritePattern<
      stablehlo::TransposeOp,
      BroadcastInDimTransposeToBroadcastInDim>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto broadcastOp =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!broadcastOp)
      return failure();

    bool singleUser = broadcastOp->getResult(0).hasOneUse();
    auto broadcastDims = broadcastOp.getBroadcastDimensions();
    auto permutation = op.getPermutation();

    // Compute the inverse permutation
    SmallVector<int64_t> inversePermutation(permutation.size());
    for (auto [idx, perm] : llvm::enumerate(permutation)) {
      inversePermutation[perm] = idx;
    }

    // Adjust the broadcast dimensions using the inverse permutation
    SmallVector<int64_t> newBroadcastDims;
    for (auto oldDim : broadcastDims) {
      newBroadcastDims.push_back(inversePermutation[oldDim]);
    }

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), broadcastOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(newBroadcastDims));
    if (singleUser)
      rewriter.eraseOp(broadcastOp);

    return success();
  }
};

struct ConcatenateOpCanon final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatenateOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  size_t max_constant_expansion;
  ConcatenateOpCanon(size_t max_constant_expansion, MLIRContext *context,
                     PatternBenefit benefit = 1,
                     ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape())
      return failure();

    size_t numElems = type.getNumElements();
    if (numElems > max_constant_expansion)
      return failure();

    // Fold concatenate when all inputs are constants.
    OperandRange inputs = op.getInputs();
    SmallVector<DenseElementsAttr> constants(inputs.size());
    for (auto [input, constant] : llvm::zip_equal(inputs, constants)) {
      if (!matchPattern(input, m_Constant(&constant)))
        return failure();
    }

    uint64_t dim = op.getDimension();
    ArrayRef<int64_t> shape = type.getShape();
    int64_t topSize = std::accumulate(shape.begin(), shape.begin() + dim,
                                      int64_t{1}, std::multiplies<>{});

    SmallVector<Attribute> newElems;
    newElems.reserve(numElems);

    for (int64_t i = 0; i != topSize; ++i) {
      for (ElementsAttr attr : constants) {
        size_t bottomSize = attr.getNumElements() / topSize;
        auto begin = attr.value_begin<Attribute>() + (i * bottomSize);
        newElems.append(begin, begin + bottomSize);
      }
    }

    assert(newElems.size() == numElems);
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }
};

struct ConvertOpCanon final
    : CheckedOpRewritePattern<stablehlo::ConvertOp, ConvertOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) const {
    // Check if this convert is a noop.
    if (op.getOperand().getType() != op.getType())
      return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct DivideSqrtToMultiplyRsqrt final
    : CheckedOpRewritePattern<stablehlo::DivOp, DivideSqrtToMultiplyRsqrt> {
  using CheckedOpRewritePattern<
      stablehlo::DivOp, DivideSqrtToMultiplyRsqrt>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DivOp op,
                                    PatternRewriter &rewriter) const {
    auto rhsOp = op.getRhs().getDefiningOp<stablehlo::SqrtOp>();
    if ((!rhsOp) || !rhsOp->hasOneUse())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
        op, op.getLhs(),
        rewriter.create<stablehlo::RsqrtOp>(op.getLoc(), rhsOp.getOperand()));
    return success();
  }
};

/// Does the same as PatternRewriter::replaceOpWithNewOp, but with a twist.
///
/// Sometimes, we want to replace an op with a new op and simultaneously refine
/// the result type from a dynamically-shaped type to a statically-shaped type.
/// (Search for usages of this function for examples).
//
/// Oftentimes, this works just fine because HLO is designed to accommodate
/// this kind of type refinements. But sometimes, this doesn't work - when
/// the op is used outside of the HLO dialect (e.g. in func.return). In these
/// cases, we insert a tensor.cast to smooth things out.
template <typename OpTy, typename... Args>
static OpTy refineOpWithNewOp(PatternRewriter &rewriter, Operation *op,
                              Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);

  llvm::SmallVector<Value> replacementResults;
  assert(op->getNumResults() == newOp->getNumResults() &&
         "replacement op doesn't match results of original op");
  for (auto [opResult, newOpResult] :
       llvm::zip(op->getResults(), newOp->getResults())) {
    Value replacementResult = newOpResult;
    if (llvm::any_of(opResult.getUsers(), [&](Operation *user) {
          return user->getDialect() != op->getDialect();
        }))
      replacementResult = rewriter.create<mlir::tensor::CastOp>(
          op->getLoc(), opResult.getType(), newOpResult);
    replacementResults.push_back(replacementResult);
  }

  rewriter.replaceOp(op, replacementResults);
  return newOp;
}

/// If a DynamicBroadCastInDimOp is not actually dynamic, use an ordinary
/// BroadcastInDimOp.
struct DynamicBroadcastInDimOpNotActuallyDynamic final
    : CheckedOpRewritePattern<stablehlo::DynamicBroadcastInDimOp,
                              DynamicBroadcastInDimOpNotActuallyDynamic> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicBroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType operandType = op.getOperand().getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires operand static shape");

    RankedTensorType type = op.getType();
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }

    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (llvm::SmallVector<int64_t> shape;
        succeeded(hlo::matchInts(op.getOutputDimensions(), shape))) {
      refineOpWithNewOp<stablehlo::BroadcastInDimOp>(
          rewriter, op, RankedTensorType::get(shape, type.getElementType()),
          op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

struct ChainedDynamicBroadcastInDimCanonicalization final
    : CheckedOpRewritePattern<stablehlo::DynamicBroadcastInDimOp,
                              ChainedDynamicBroadcastInDimCanonicalization> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicBroadcastInDimOp bcast,
                                    PatternRewriter &rewriter) const {
    auto precedingBcast =
        bcast.getOperand().getDefiningOp<stablehlo::DynamicBroadcastInDimOp>();
    if (!precedingBcast)
      return failure();

    // Compose broadcast dimensions.
    SmallVector<int64_t> composition;
    for (int64_t precedingDim : precedingBcast.getBroadcastDimensions()) {
      composition.push_back(bcast.getBroadcastDimensions()[precedingDim]);
    }
    auto composedBcastDims = rewriter.getDenseI64ArrayAttr(composition);

    rewriter.replaceOpWithNewOp<stablehlo::DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};

// If all dimensions are known to be nonexpanding from the attribute, replace
// the dynamic broadcast with a cast.
struct DynamicBroadcastInDimAllDimsNonExpanding final
    : CheckedOpRewritePattern<stablehlo::DynamicBroadcastInDimOp,
                              DynamicBroadcastInDimAllDimsNonExpanding> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicBroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    RankedTensorType type = op.getType();

    if (!op.getKnownNonexpandingDimensions() ||
        static_cast<int64_t>(op.getKnownNonexpandingDimensions()->size()) !=
            type.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "known_nonexpanding_dimensions don't cover all output dims");
    }

    auto cast = rewriter.createOrFold<tensor::CastOp>(op.getLoc(), type,
                                                      op.getOperand());
    rewriter.replaceOp(op, cast);
    return success();
  }
};

struct NoopReduceOpCanon final
    : CheckedOpRewritePattern<stablehlo::ReduceOp, NoopReduceOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    // No dimensions to reduce.
    if (op.getDimensions().empty()) {
      rewriter.replaceOp(op, op.getInputs());
      return success();
    }

    // If all returned values in the ReduceOp region exists outside the
    // region, replace the ReduceOp with those values.
    if (auto retOp = dyn_cast<stablehlo::ReturnOp>(
            op.getBody().front().getTerminator())) {
      Region *retRegion = retOp->getParentRegion();
      if (llvm::any_of(retOp.getResults(), [retRegion](Value result) {
            return result.getParentRegion() == retRegion;
          }))
        return failure();

      SmallVector<Value> vals;
      DenseI64ArrayAttr empty = rewriter.getDenseI64ArrayAttr({});
      for (auto [res, opres] : llvm::zip(retOp.getResults(), op.getResults()))
        vals.push_back(rewriter.create<stablehlo::BroadcastInDimOp>(
            op.getLoc(), opres.getType(), res, empty));
      rewriter.replaceOp(op, vals);
      return success();
    }

    return failure();
  }
};

struct EmptyReduceOpCanon final
    : CheckedOpRewritePattern<stablehlo::ReduceOp, EmptyReduceOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    // We require all reduce shapes to be the same, up to the element types, so
    // we can just use the first operand and the first result as
    // representatives.
    auto elemTy = cast<RankedTensorType>(op.getInputs().getType().front());

    if (op.getDimensions().empty()) {
      rewriter.replaceAllOpUsesWith(op, op.getInputs());
      return success();
    }

    if (!llvm::is_contained(elemTy.getShape(), 0))
      return failure();

    Location loc = op.getLoc();
    DenseI64ArrayAttr empty = rewriter.getDenseI64ArrayAttr({});
    if (elemTy.hasStaticShape()) {
      SmallVector<Value> broadcasts(op.getNumResults());
      for (auto [bcast, init, outTy] : llvm::zip_equal(
               broadcasts, op.getInitValues(), op.getResultTypes())) {
        bcast = rewriter.create<stablehlo::BroadcastInDimOp>(loc, outTy, init,
                                                             empty);
      }
      rewriter.replaceOp(op, broadcasts);
      return success();
    }

    SmallVector<Value> shapes;
    if (failed(op.reifyReturnTypeShapes(rewriter, op.getOperands(), shapes)))
      return failure();

    SmallVector<Value> broadcasts(op.getNumResults());
    for (auto [bcast, init, shape, outTy] : llvm::zip_equal(
             broadcasts, op.getInitValues(), shapes, op.getResultTypes())) {
      bcast = rewriter.create<stablehlo::DynamicBroadcastInDimOp>(
          loc, outTy, init, shape, empty);
    }
    rewriter.replaceOp(op, broadcasts);
    return success();
  }
};

struct DynamicReshapeOpCanon final
    : CheckedOpRewritePattern<stablehlo::DynamicReshapeOp,
                              DynamicReshapeOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // This is a noop when the output type is already a static shape.
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, type,
                                                      op.getOperand());
    return success();
  }
};

struct GetTupleElementOpCanon final
    : CheckedOpRewritePattern<stablehlo::GetTupleElementOp,
                              GetTupleElementOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::GetTupleElementOp op,
                                    PatternRewriter &rewriter) const {
    auto tuple = op.getOperand().getDefiningOp<stablehlo::TupleOp>();
    if (!tuple)
      return failure();

    Value result = tuple.getOperand(op.getIndex());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RealOpCanon final
    : CheckedOpRewritePattern<stablehlo::RealOp, RealOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::RealOp op,
                                    PatternRewriter &rewriter) const {
    auto elTy = op.getOperand().getType().getElementType();
    if (!isa<ComplexType>(elTy)) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    auto complex = op.getOperand().getDefiningOp<stablehlo::ComplexOp>();
    if (!complex)
      return failure();

    rewriter.replaceOp(op, complex.getLhs());
    return success();
  }
};

struct ImagOpCanon final
    : CheckedOpRewritePattern<stablehlo::ImagOp, ImagOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ImagOp op,
                                    PatternRewriter &rewriter) const {
    auto elTy = op.getOperand().getType().getElementType();
    if (!isa<ComplexType>(elTy)) {
      rewriter.replaceOp(op, rewriter.create<stablehlo::ConstantOp>(
                                 op->getLoc(), makeAttr(op.getType(), 0)));
      return success();
    }

    auto complex = op.getOperand().getDefiningOp<stablehlo::ComplexOp>();
    if (!complex)
      return failure();

    rewriter.replaceOp(op, complex.getRhs());
    return success();
  }
};

// (conj (complex a, (neg b))) -> (complex a b)
struct ConjComplexNegate final
    : CheckedOpRewritePattern<chlo::ConjOp, ConjComplexNegate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::ConjOp op,
                                    PatternRewriter &rewriter) const {
    auto complex = op.getOperand().getDefiningOp<stablehlo::ComplexOp>();
    if (!complex)
      return failure();

    auto neg = complex.getRhs().getDefiningOp<stablehlo::NegOp>();
    if (!neg)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ComplexOp>(
        op, op.getType(), complex.getLhs(), neg.getOperand());
    return success();
  }
};

struct GetDimensionSizeOpCanon final
    : CheckedOpRewritePattern<stablehlo::GetDimensionSizeOp,
                              GetDimensionSizeOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::GetDimensionSizeOp op,
                                    PatternRewriter &rewriter) const {
    // Fold get_dimension_size when the queried dim is statically known.
    RankedTensorType operandTy = op.getOperand().getType();

    int64_t dimSize = operandTy.getDimSize(op.getDimension());
    if (dimSize < 0)
      return failure();

    auto elemTy = cast<IntegerType>(op.getType().getElementType());
    IntegerAttr elemVal = rewriter.getIntegerAttr(elemTy, dimSize);
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elemVal));
    return success();
  }
};

struct NoopReverse final
    : CheckedOpRewritePattern<stablehlo::ReverseOp, NoopReverse> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReverseOp op,
                                    PatternRewriter &rewriter) const {
    SmallVector<int64_t> newDimensions;
    auto dimensions = op.getDimensions();
    auto shape = op.getResult().getType().getShape();

    for (auto dim : dimensions) {
      auto size = shape[dim];
      if (size != 1)
        newDimensions.push_back(dim);
    }

    if (newDimensions.empty()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    if (newDimensions.size() == dimensions.size())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ReverseOp>(op, op.getOperand(),
                                                      newDimensions);
    return success();
  }
};

/// Converts gather ops to slice ops in case we have a single set of constant
/// indices.
struct GatherOpCanon final
    : CheckedOpRewritePattern<stablehlo::GatherOp, GatherOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::GatherOp gather,
                                    PatternRewriter &rewriter) const {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    stablehlo::GatherDimensionNumbersAttr dnums = gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO: Remove when the verifier catches this case that is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size())) {
      return failure();
    }

    auto operandType = cast<RankedTensorType>(gather->getOperand(0).getType());
    if (!operandType.hasStaticShape())
      return failure();

    auto sliceEnd = llvm::to_vector(gather.getSliceSizes());
    SmallVector<int64_t> sliceStart(sliceEnd.size(), 0);
    for (auto [mapIndex, value] :
         llvm::zip_equal(dnums.getStartIndexMap(), index.getValues<APInt>())) {
      // Clamp the indices within bounds to faithfully mirror gather semantics.
      int64_t offset =
          std::clamp(value.getSExtValue(), static_cast<int64_t>(0),
                     operandType.getDimSize(mapIndex) - sliceEnd[mapIndex]);
      sliceStart[mapIndex] += offset;
      sliceEnd[mapIndex] += offset;
    }

    SmallVector<int64_t> sliceStride(sliceEnd.size(), 1);
    SmallVector<int64_t> sliceShape(sliceEnd.size());
    for (auto [shapeElem, startElem, endElem] :
         llvm::zip_equal(sliceShape, sliceStart, sliceEnd)) {
      shapeElem = endElem - startElem;
    }

    Type elementType = gather.getType().getElementType();
    auto sliceType = RankedTensorType::get(sliceShape, elementType);
    Value result = rewriter.create<stablehlo::SliceOp>(
        gather.getLoc(), sliceType, gather.getOperand(),
        rewriter.getDenseI64ArrayAttr(sliceStart),
        rewriter.getDenseI64ArrayAttr(sliceEnd),
        rewriter.getDenseI64ArrayAttr(sliceStride));

    ArrayRef<int64_t> collapsedSliceDims = dnums.getCollapsedSliceDims();
    if (!collapsedSliceDims.empty()) {
      llvm::SmallVector<int64_t> reshapeShape;
      for (auto [idx, dim] : llvm::enumerate(sliceShape)) {
        if (!llvm::is_contained(collapsedSliceDims, idx))
          reshapeShape.push_back(dim);
      }
      auto reshapeType = RankedTensorType::get(reshapeShape, elementType);
      result = rewriter.create<stablehlo::ReshapeOp>(gather.getLoc(),
                                                     reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

struct ReshapeOpCanon final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeOpCanon> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Fold noop reshape.
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape of a constant.
    DenseElementsAttr cstAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&cstAttr)))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, cstAttr.reshape(op.getType()));
    return success();
  }
};

struct MergeConsecutiveReshapes final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, MergeConsecutiveReshapes> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    // Fold noop reshape.
    auto operand = op.getOperand();
    if (op.getType() == operand.getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape(reshape(x)).
    auto reshapeOp = operand.getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "requires defining op of operand to be Reshape");

    op.setOperand(reshapeOp->getOperand(0));
    return success();
  }
};

struct TransposeIsReshape final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeIsReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto input = op.getOperand();
    auto permutation = op.getPermutation();

    if (isIotaRange(permutation)) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    RankedTensorType inputTy = input.getType();
    if (!inputTy.hasStaticShape() || !op.getType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "requires input and output to be of a statically-shaped ranked "
              "tensor type");

    SmallVector<int64_t> permValues(permutation);
    SmallVector<int64_t> nonZeroPerms;
    nonZeroPerms.reserve(permValues.size());
    for (auto idx : permValues) {
      auto sz = inputTy.getDimSize(idx);
      if (sz != 1)
        nonZeroPerms.push_back(idx);
    }

    for (int i = 1, s = nonZeroPerms.size(); i < s; ++i)
      if (nonZeroPerms[i - 1] > nonZeroPerms[i])
        return rewriter.notifyMatchFailure(op, "memory layout change");

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(), input);
    return success();
  }
};

struct IfRemoveUnused final
    : CheckedOpRewritePattern<stablehlo::IfOp, IfRemoveUnused> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {

    SmallVector<bool> resultUsed(op->getNumResults(), true);

    bool anyUnused = false;
    for (const auto &it : llvm::enumerate(op->getResults())) {
      bool unused = it.value().use_empty();
      resultUsed[it.index()] = !unused;
      anyUnused |= unused;
    }

    if (!anyUnused)
      return failure();

    SmallVector<Type> newResultTypes;

    Operation *trueTerm = op.getTrueBranch().front().getTerminator();
    Operation *falseTerm = op.getFalseBranch().front().getTerminator();

    unsigned removed = 0;
    for (const auto &it : llvm::enumerate(op->getResults())) {
      bool used = resultUsed[it.index()];
      if (used) {
        newResultTypes.push_back(it.value().getType());
        continue;
      }

      auto i = it.index() - removed;
      rewriter.modifyOpInPlace(trueTerm, [&] { trueTerm->eraseOperand(i); });
      rewriter.modifyOpInPlace(falseTerm, [&] { falseTerm->eraseOperand(i); });
      removed++;
    }

    auto newIf = rewriter.create<stablehlo::IfOp>(op.getLoc(), newResultTypes,
                                                  op.getPred());
    newIf.getTrueBranch().takeBody(op.getTrueBranch());
    newIf.getFalseBranch().takeBody(op.getFalseBranch());

    removed = 0;
    for (const auto &it : llvm::enumerate(resultUsed)) {
      bool used = it.value();
      if (!used) {
        removed++;
        continue;
      }
      auto res = op.getResult(it.index());
      auto newRes = newIf.getResult(it.index() - removed);
      rewriter.replaceAllUsesWith(res, newRes);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct IfPredPropagation final
    : CheckedOpRewritePattern<stablehlo::IfOp, IfPredPropagation> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {
    Value pred = op.getPred();
    bool anyModified = false;

    auto makeBool = [&](bool value) {
      auto i1type = RankedTensorType::get({}, rewriter.getI1Type());
      return rewriter
          .create<stablehlo::ConstantOp>(
              pred.getLoc(), i1type,
              SplatElementsAttr::get(i1type, rewriter.getBoolAttr(value)))
          .getResult();
    };

    Region *trueBranch = &op.getTrueBranch();
    Region *falseBranch = &op.getFalseBranch();

    for (OpOperand &use : pred.getUses()) {
      Operation *owner = use.getOwner();
      Region *useRegion = owner->getParentRegion();

      if (trueBranch->isAncestor(useRegion)) {
        anyModified = true;
        rewriter.modifyOpInPlace(owner, [&]() { use.assign(makeBool(true)); });
      } else if (falseBranch->isAncestor(useRegion)) {
        anyModified = true;
        rewriter.modifyOpInPlace(owner, [&]() { use.assign(makeBool(false)); });
      }
    }

    return success(anyModified);
  }
};

struct IfInline final : CheckedOpRewritePattern<stablehlo::IfOp, IfInline> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {

    auto iszero = matchPattern(op.getPred(), m_Zero());
    auto isone = matchPattern(op.getPred(), m_One());

    if (!iszero && !isone)
      return failure();

    auto &reg = isone ? op.getTrueBranch() : op.getFalseBranch();

    if (reg.empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    assert(reg.hasOneBlock());  // stablehlo.if only allows 1 or 0 block in the
    auto *block = &reg.front(); // regions

    auto term = block->getTerminator();
    rewriter.inlineBlockBefore(block, op);
    rewriter.replaceOp(op, term->getOperands());
    rewriter.eraseOp(term);
    return success();
  }
};

// https:github.com/llvm/llvm-project/blob/74d8f3952c4acf6d57948983d7c5b0d0a7763c28/mlir/lib/Dialect/SCF/IR/SCF.cpp#L2313
struct IfToSelect final
    : public CheckedOpRewritePattern<stablehlo::IfOp, IfToSelect> {
  using CheckedOpRewritePattern<stablehlo::IfOp,
                                IfToSelect>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumResults() == 0 || op.getTrueBranch().empty() ||
        op.getFalseBranch().empty())
      return failure();

    auto pred = op.getPred();

    auto trueTerm = op.getTrueBranch().front().getTerminator();
    auto falseTerm = op.getFalseBranch().front().getTerminator();
    auto trueOperands = trueTerm->getOperands();
    auto falseOperands = falseTerm->getOperands();

    SmallVector<Type> nonHoistable;
    for (auto [trueVal, falseVal] : llvm::zip(trueOperands, falseOperands)) {
      if (&op.getTrueBranch() == trueVal.getParentRegion() ||
          &op.getFalseBranch() == falseVal.getParentRegion())
        nonHoistable.push_back(trueVal.getType());
    }

    // Early exit if there aren't any yielded values we can
    // hoist outside the if.
    if (nonHoistable.size() == op->getNumResults())
      return failure();

    auto replacement =
        rewriter.create<stablehlo::IfOp>(op.getLoc(), nonHoistable, pred);
    replacement.getTrueBranch().takeBody(op.getTrueBranch());
    replacement.getFalseBranch().takeBody(op.getFalseBranch());

    SmallVector<Value> results(op->getNumResults());
    assert(trueOperands.size() == results.size());
    assert(falseOperands.size() == results.size());

    SmallVector<Value> trueReturns;
    SmallVector<Value> falseReturns;
    rewriter.setInsertionPoint(replacement);
    for (const auto &it :
         llvm::enumerate(llvm::zip(trueOperands, falseOperands))) {
      Value trueVal = std::get<0>(it.value());
      Value falseVal = std::get<1>(it.value());
      if (&replacement.getTrueBranch() == trueVal.getParentRegion() ||
          &replacement.getFalseBranch() == falseVal.getParentRegion()) {
        results[it.index()] = replacement.getResult(trueReturns.size());
        trueReturns.push_back(trueVal);
        falseReturns.push_back(falseVal);
      } else if (trueVal == falseVal)
        results[it.index()] = trueVal;
      else
        results[it.index()] = rewriter.create<stablehlo::SelectOp>(
            op.getLoc(), pred, trueVal, falseVal);
    }

    rewriter.setInsertionPointToEnd(&replacement.getTrueBranch().front());
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(trueTerm, trueReturns);

    rewriter.setInsertionPointToEnd(&replacement.getFalseBranch().front());
    rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(falseTerm, falseReturns);

    rewriter.replaceOp(op, results);
    return success();
  }
};

// https://github.com/llvm/llvm-project/blob/74d8f3952c4acf6d57948983d7c5b0d0a7763c28/mlir/lib/Dialect/SCF/IR/SCF.cpp#L2313
struct SpeculateIfPadToSelect final
    : public CheckedOpRewritePattern<stablehlo::IfOp, SpeculateIfPadToSelect> {
  using CheckedOpRewritePattern<
      stablehlo::IfOp, SpeculateIfPadToSelect>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumResults() == 0 || op.getTrueBranch().empty() ||
        op.getFalseBranch().empty())
      return failure();

    auto trueOperands =
        op.getTrueBranch().front().getTerminator()->getOperands();
    auto falseOperands =
        op.getFalseBranch().front().getTerminator()->getOperands();

    bool anyPad = false;
    for (auto [trueVal, falseVal] : llvm::zip(trueOperands, falseOperands)) {
      if (&op.getTrueBranch() != trueVal.getParentRegion() &&
          &op.getFalseBranch() == falseVal.getParentRegion()) {
        if (auto pad = falseVal.getDefiningOp<stablehlo::PadOp>()) {
          if (pad.getOperand().getParentRegion() != &op.getFalseBranch() &&
              pad.getPaddingValue().getParentRegion() != &op.getFalseBranch()) {
            rewriter.modifyOpInPlace(pad, [&]() { pad->moveBefore(op); });
            anyPad = true;
            continue;
          }
        }
      }

      if (&op.getFalseBranch() != falseVal.getParentRegion() &&
          &op.getTrueBranch() == trueVal.getParentRegion()) {
        if (auto pad = trueVal.getDefiningOp<stablehlo::PadOp>()) {
          if (pad.getOperand().getParentRegion() != &op.getTrueBranch() &&
              pad.getPaddingValue().getParentRegion() != &op.getTrueBranch()) {
            rewriter.modifyOpInPlace(pad, [&]() { pad->moveBefore(op); });
            anyPad = true;
            continue;
          }
        }
      }
    }

    if (anyPad)
      return success();
    else
      return failure();
  }
};

bool verifyInversePermutations(stablehlo::TransposeOp innerTrans,
                               stablehlo::TransposeOp outerTrans) {
  auto innerPerm = innerTrans.getPermutation();
  auto outerPerm = outerTrans.getPermutation();

  if (innerPerm.size() != outerPerm.size())
    return false;

  SmallVector<int64_t> composition(innerPerm.size());
  for (size_t i = 0; i < innerPerm.size(); ++i) {
    composition[i] = outerPerm[innerPerm[i]];
  }

  // Check if the composition is the identity permutation
  for (size_t i = 0; i < composition.size(); ++i) {
    if (composition[i] != static_cast<int64_t>(i))
      return false;
  }

  return true;
}

// Currently supports:
// 1. Identifies induction variable
// 2. Addition of constant step value
// 3. Less than comparision
struct WhileOpInductionReplacement
    : public CheckedOpRewritePattern<stablehlo::WhileOp,
                                     WhileOpInductionReplacement> {
  using CheckedOpRewritePattern<
      stablehlo::WhileOp, WhileOpInductionReplacement>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Only handle while loops with identifiable iteration patterns
    bool canonicalized = false;

    // Look for a loop counter variable (induction variable analog)
    Block &bodyBlock = whileOp.getBody().front();
    auto returnOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Find the counter variable and its limit before entering the loop
    // This is used to identify and optimize induction variables
    unsigned counterIdx = 0;
    Value limitValue = nullptr;
    bool hasCounter = findCounterAndLimit(whileOp, counterIdx, limitValue);

    // If we can't find a counter variable, we can't optimize induction
    // variables
    if (!hasCounter)
      return failure();

    if (!definedOutside(limitValue, whileOp))
      return failure();

    // Get the counter argument and its start value for later use
    Value counterArg = whileOp.getBody().getArgument(counterIdx);
    Value startValue = findCounterStartValue(whileOp, counterIdx);

    // Find the counter step value (how much the counter increments each
    // iteration) This is needed to correctly scale the induction variable
    // calculation
    Value counterStepValue = findCounterStepValue(whileOp, counterIdx);
    if (!counterStepValue)
      return failure();

    // Examine each iteration argument and result
    for (unsigned i = 0; i < whileOp.getOperands().size(); ++i) {
      // Skip the counter variable itself - we don't want to optimize it away
      if (i == counterIdx)
        continue;

      // Get the input, the body argument, and the yielded value
      Value initValue = whileOp.getOperands()[i];
      BlockArgument iterArg = bodyBlock.getArgument(i);
      Value yieldedValue = returnOp.getOperand(i);
      Value result = whileOp.getResult(i);

      // Look for a simple addition pattern: either iter_arg + step or step +
      // iter_arg
      auto addOp = yieldedValue.getDefiningOp<stablehlo::AddOp>();
      if (!addOp)
        continue;

      if (!addOp.getType().getElementType().isInteger())
        continue;

      // Check which operand is the iteration argument and which is the step
      Value stepValue;
      if (addOp.getLhs() == iterArg) {
        // Pattern: iter_arg + step
        stepValue = addOp.getRhs();
      } else if (addOp.getRhs() == iterArg) {
        // Pattern: step + iter_arg
        stepValue = addOp.getLhs();
      } else {
        // Neither operand is the iteration argument
        continue;
      }

      // Find if the step is a constant
      auto constOp = stepValue.getDefiningOp<stablehlo::ConstantOp>();
      if (!constOp)
        continue;

      // Now we can replace uses of the iterArg inside the loop
      // with a direct calculation based on the counter:
      // replacement = init_value + ((counter - start_value) * step_value) /
      // counter_step_value
      if (!iterArg.use_empty()) {
        rewriter.setInsertionPointToStart(&bodyBlock);

        // Create the calculation for the current iteration
        Value iterOffset = rewriter.create<stablehlo::SubtractOp>(
            whileOp.getLoc(), counterArg.getType(), counterArg, startValue);

        // First multiply by the step value
        Value scaledOffset = rewriter.create<stablehlo::MulOp>(
            whileOp.getLoc(), iterOffset.getType(), iterOffset, stepValue);

        // Then divide by the counter step value to get the correct scaling
        Value normalizedOffset = rewriter.create<stablehlo::DivOp>(
            whileOp.getLoc(), scaledOffset.getType(), scaledOffset,
            counterStepValue);

        Value replacement = rewriter.create<stablehlo::AddOp>(
            whileOp.getLoc(), iterArg.getType(), initValue, normalizedOffset);

        rewriter.modifyOpInPlace(
            whileOp, [&] { iterArg.replaceAllUsesWith(replacement); });
        canonicalized = true;
      }

      // Similarly replace uses of the result outside the loop
      // with a calculation based on the final counter value
      if (!result.use_empty() && limitValue) {
        rewriter.setInsertionPointAfter(whileOp);

        // Calculate total iterations: limit - start
        Value totalIters = rewriter.create<stablehlo::SubtractOp>(
            whileOp.getLoc(), limitValue.getType(), limitValue, startValue);

        // First multiply by the step value (using the same step value
        // identified earlier)
        Value scaledOffset = rewriter.create<stablehlo::MulOp>(
            whileOp.getLoc(), totalIters.getType(), totalIters, stepValue);

        // Then divide by the counter step value to get the correct scaling
        Value normalizedOffset = rewriter.create<stablehlo::DivOp>(
            whileOp.getLoc(), scaledOffset.getType(), scaledOffset,
            counterStepValue);

        Value finalValue = rewriter.create<stablehlo::AddOp>(
            whileOp.getLoc(), result.getType(), initValue, normalizedOffset);

        rewriter.replaceAllUsesWith(result, finalValue);
        canonicalized = true;
      }
    }

    return success(canonicalized);
  }

private:
  // Helper function to identify the counter variable and its limit
  // Returns the index of the counter argument and the limit value
  bool findCounterAndLimit(stablehlo::WhileOp whileOp, unsigned &counterIdx,
                           Value &limitValue) const {
    // Look in the condition region for a comparison operation
    Block &condBlock = whileOp.getCond().front();
    Operation *terminator = condBlock.getTerminator();

    // Typical pattern: return %comparison
    if (auto returnOp = dyn_cast<stablehlo::ReturnOp>(terminator)) {
      if (returnOp.getNumOperands() != 1)
        return false;

      // Look for a comparison that controls the loop
      auto compareOp =
          returnOp.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
      if (!compareOp)
        return false;

      if (compareOp.getComparisonDirection() !=
          stablehlo::ComparisonDirection::LT)
        return false;

      if (!compareOp.getLhs().getType().getElementType().isInteger())
        return false;

      // Check if one side is a block argument (our counter)
      if (auto blockArg = dyn_cast<BlockArgument>(compareOp.getLhs())) {
        if (blockArg.getOwner() == &condBlock) {
          counterIdx = blockArg.getArgNumber();
          limitValue = compareOp.getRhs();
          return true;
        }
      }

      if (auto blockArg = dyn_cast<BlockArgument>(compareOp.getRhs())) {
        if (blockArg.getOwner() == &condBlock) {
          counterIdx = blockArg.getArgNumber();
          limitValue = compareOp.getLhs();
          return true;
        }
      }
    }

    return false;
  }

  // Helper to find the initial value of the counter
  Value findCounterStartValue(stablehlo::WhileOp whileOp,
                              unsigned counterIdx) const {
    // The initial value is the corresponding operand to the while op
    return whileOp.getOperands()[counterIdx];
  }

  // Helper to find the counter step value (how much it increments each
  // iteration)
  Value findCounterStepValue(stablehlo::WhileOp whileOp,
                             unsigned counterIdx) const {
    // Get the block argument in the body region
    Block &bodyBlock = whileOp.getBody().front();
    BlockArgument counterArg = bodyBlock.getArgument(counterIdx);

    // Find the terminator to get the yielded value
    auto returnOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Get the yielded value for the counter
    Value yieldedCounter = returnOp.getOperand(counterIdx);

    // Look for addition pattern: counter + step or step + counter
    auto addOp =
        dyn_cast_or_null<stablehlo::AddOp>(yieldedCounter.getDefiningOp());
    if (!addOp)
      return nullptr;

    // Check both sides of the addition operation (since addition is
    // commutative)
    if (addOp.getLhs() == counterArg) {
      // Pattern: counter + step
      return addOp.getRhs();
    }

    if (addOp.getRhs() == counterArg) {
      // Pattern: step + counter
      return addOp.getLhs();
    }

    // Counter is not directly used in the addition
    return nullptr;
  }
};

struct TransposeWhile
    : public CheckedOpRewritePattern<stablehlo::WhileOp, TransposeWhile> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct TransposeCandidate {
      unsigned idx;
      stablehlo::TransposeOp outerTranspose;
    };

    llvm::SmallVector<TransposeCandidate, 4> outerTransposes;

    // For each yielded value, check if it comes from a transpose
    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {
      // Check if the corresponding while result is used by a transpose
      Value whileResult = whileOp.getResult(idx);

      // Check that the while result has exactly one use
      if (!whileResult.hasOneUse())
        continue;

      // Get the single user and verify it's a transpose
      Operation *singleUser = *whileResult.getUsers().begin();
      auto outerTransposeOp = dyn_cast<stablehlo::TransposeOp>(singleUser);
      if (!outerTransposeOp)
        continue;

      outerTransposes.emplace_back(TransposeCandidate{idx, outerTransposeOp});
    }

    // If no candidates found, no rewrite needed
    if (outerTransposes.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : outerTransposes) {
      unsigned idx = candidate.idx;
      stablehlo::TransposeOp outerTranspose = candidate.outerTranspose;

      // Create a new transpose before the while loop
      auto inputTranspose = rewriter.create<stablehlo::TransposeOp>(
          whileOp.getLoc(),
          outerTranspose.getType(), // The type after transposition
          whileOperands[idx],       // Original input to while
          outerTranspose.getPermutation());

      // Update the while operand to use the new transposed value
      newOperands[idx] = inputTranspose;
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : outerTransposes) {
        unsigned idx = candidate.idx;
        newReturnValues[idx] = rewriter.create<stablehlo::TransposeOp>(
            whileOp.getLoc(), newReturnValues[idx],
            candidate.outerTranspose.getPermutation());
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock.clear();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      SmallVector<Value> replacements;
      for (auto arg : newBodyBlock.getArguments())
        replacements.push_back(arg);
      for (auto &pair : outerTransposes) {
        replacements[pair.idx] = rewriter.create<stablehlo::TransposeOp>(
            pair.outerTranspose.getLoc(), replacements[pair.idx],
            getInversePermutation(pair.outerTranspose.getPermutation()));
      }
      rewriter.mergeBlocks(&whileOp.getBody().front(), &newBodyBlock,
                           replacements);
    }

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      SmallVector<Value> replacements;
      for (auto arg : newCondBlock.getArguments())
        replacements.push_back(arg);
      for (auto &pair : outerTransposes) {
        replacements[pair.idx] = rewriter.create<stablehlo::TransposeOp>(
            pair.outerTranspose.getLoc(), replacements[pair.idx],
            getInversePermutation(pair.outerTranspose.getPermutation()));
      }
      rewriter.mergeBlocks(&whileOp.getCond().front(), &newCondBlock,
                           replacements);
    }

    // Step 5. Replace outerTranspose with the newWhileOp results
    for (auto &candidate : outerTransposes) {
      unsigned idx = candidate.idx;
      stablehlo::TransposeOp outerTranspose = candidate.outerTranspose;

      // Update the while operand to use the new transposed value
      rewriter.replaceOp(outerTranspose, newWhileOp.getResult(idx));
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, newWhileOp.getResults());
    return success();
  }
};

bool isLegalConcatToOneDimDUS(stablehlo::ConcatenateOp outer,
                              stablehlo::SliceOp *lhsP,
                              stablehlo::SliceOp *rhsP = nullptr,
                              Value *operand = nullptr) {
  if (outer.getOperands().size() < 2)
    return false;
  SmallVector<stablehlo::ConcatenateOp> inners;

  stablehlo::SliceOp lhs = nullptr;
  if (auto lhsSlice =
          outer.getOperands()[0].getDefiningOp<stablehlo::SliceOp>()) {
    for (int i = 0; i < lhsSlice.getType().getShape().size(); i++) {
      if (lhsSlice.getStartIndices()[i] != 0) {
        return false;
      }
      if (i != outer.getDimension()) {
        if (cast<RankedTensorType>(lhsSlice.getOperand().getType())
                .getShape()[i] != outer.getType().getShape()[i]) {
          return false;
        }
      }
      if (lhsSlice.getStrides()[i] != 1) {
        return false;
      }
    }
    lhs = lhsSlice;
  }

  stablehlo::SliceOp rhs = nullptr;
  if (auto rhsSlice =
          outer.getOperands().back().getDefiningOp<stablehlo::SliceOp>()) {
    for (int i = 0; i < rhsSlice.getType().getShape().size(); i++) {
      if (rhsSlice.getLimitIndices()[i] != outer.getType().getShape()[i]) {
        return false;
      }
      if (i != outer.getDimension()) {
        if (cast<RankedTensorType>(rhsSlice.getOperand().getType())
                .getShape()[i] != outer.getType().getShape()[i]) {
          return false;
        }
      }
      if (rhsSlice.getStrides()[i] != 1) {
        return false;
      }
    }
    rhs = rhsSlice;
  }

  if (!lhs && !rhs) {
    return false;
  }
  if (!rhs && cast<RankedTensorType>(lhs.getOperand().getType())
                      .getShape()[outer.getDimension()] !=
                  outer.getType().getShape()[outer.getDimension()]) {
    return false;
  }
  if (!lhs && cast<RankedTensorType>(rhs.getOperand().getType())
                      .getShape()[outer.getDimension()] !=
                  outer.getType().getShape()[outer.getDimension()]) {
    return false;
  }

  if (lhs && rhs && outer.getOperands().size() == 2) {
    return false;
  }
  if (lhs && rhs && lhs.getOperand() != rhs.getOperand()) {
    return false;
  }
  if (lhsP)
    *lhsP = lhs;
  if (rhsP)
    *rhsP = rhs;
  if (operand && lhs)
    *operand = lhs.getOperand();
  if (operand && rhs)
    *operand = rhs.getOperand();
  return true;
}

stablehlo::DynamicUpdateSliceOp
concatToOneDimDUS(PatternRewriter &rewriter, stablehlo::ConcatenateOp outer) {
  stablehlo::SliceOp lhs = nullptr;
  stablehlo::SliceOp rhs = nullptr;
  if (!isLegalConcatToOneDimDUS(outer, &lhs, &rhs))
    return nullptr;

  auto shard = sdy::getShardingPerValue(outer);

  SmallVector<Value> newOps;
  int start = lhs ? 1 : 0;
  int end = outer.getOperands().size() - (rhs ? 1 : 0);
  for (int i = start; i < end; i++) {
    newOps.push_back(outer.getOperands()[i]);
  }
  Value innerConcat = newOps[0];

  auto iTy = RankedTensorType::get({}, rewriter.getI64Type());
  Value operand = lhs ? lhs.getOperand() : rhs.getOperand();
  assert(operand.getType() == outer.getType());
  SmallVector<Value> starts(
      outer.getType().getShape().size(),
      rewriter.create<stablehlo::ConstantOp>(
          outer.getLoc(), iTy, cast<ElementsAttr>(makeAttr(iTy, 0))));

  if (lhs) {
    starts[outer.getDimension()] = rewriter.create<stablehlo::ConstantOp>(
        outer.getLoc(), iTy,
        cast<ElementsAttr>(
            makeAttr(iTy, lhs.getType().getShape()[outer.getDimension()])));
  }

  rewriter.setInsertionPointAfter(outer);
  if (newOps.size() != 1) {
    auto nConcat = rewriter.create<stablehlo::ConcatenateOp>(
        outer.getLoc(), newOps, outer.getDimension());
    innerConcat = nConcat;
    if (shard) {
      sdy::setShardings(nConcat, shard);
    }
  }

  auto OT = outer.getType();
  assert(operand.getType() == OT);
  if (operand.getType() != OT) {
    llvm_unreachable("invalid replacement (0)\n");
  }
  auto dus = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
      outer.getLoc(), operand, innerConcat, starts);
  assert(dus.getType() == OT);
  if (dus.getType() != OT) {
    llvm_unreachable("invalid replacement\n");
  }
  if (shard) {
    sdy::setShardings(dus, shard);
  }
  rewriter.replaceOp(outer, dus);
  return dus;
}

struct ConcatToOneDimDUS final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatToOneDimDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp outer,
                                    PatternRewriter &rewriter) const {
    if (concatToOneDimDUS(rewriter, outer))
      return success();
    else
      return failure();
  }
};

struct WhileDUS : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileDUS> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct DUSCandidate {
      unsigned idx;
      stablehlo::DynamicUpdateSliceOp DUS;
      mlir::Value outerOperand;
      mlir::Value conditionalOperand;
    };

    llvm::SmallVector<DUSCandidate, 4> candidates;
    bool hasConditional = false;

    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      Value DUSOperand = nullptr;

      if (auto DUS = yieldOp.getOperand(idx)
                         .getDefiningOp<stablehlo::DynamicUpdateSliceOp>()) {
        DUSOperand = DUS.getOperand();

        bool legal = true;
        for (auto idx : DUS.getStartIndices()) {
          if (!definedOutside(idx, whileOp)) {
            legal = false;
          }
        }

        if (!legal)
          continue;
      } else if (auto concat = yieldOp.getOperand(idx)
                                   .getDefiningOp<stablehlo::ConcatenateOp>()) {
        Value operand;
        if (isLegalConcatToOneDimDUS(concat, nullptr, nullptr, &operand)) {
          DUSOperand = operand;
        }
      }

      // Check that the while result has exactly one use
      if (!DUSOperand)
        continue;

      mlir::Value conditionalOperand = nullptr;
      if (DUSOperand == whileOp.getBody().front().getArgument(idx)) {
      } else if (definedOutside(DUSOperand, whileOp)) {

        bool hasArgUse = !whileOp.getCond().getArgument(idx).use_empty() ||
                         !whileOp.getBody().getArgument(idx).use_empty();

        if (hasArgUse) {
          continue;
        }

        conditionalOperand = DUSOperand;
        hasConditional = true;
      } else {
        continue;
      }

      auto DUS = yieldOp.getOperand(idx)
                     .getDefiningOp<stablehlo::DynamicUpdateSliceOp>();

      if (!DUS) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto concat =
            yieldOp.getOperand(idx).getDefiningOp<stablehlo::ConcatenateOp>();
        assert(concat);
        DUS = concatToOneDimDUS(rewriter, concat);
        assert(DUS);
        for (auto idx : DUS.getStartIndices()) {
          (void)idx;
          assert(definedOutside(idx, whileOp));
        }
      }

      candidates.emplace_back(DUSCandidate{idx, DUS, whileOp.getOperands()[idx],
                                           conditionalOperand});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : candidates) {
      // Create a new transpose before the while loop

      newOperands[candidate.idx] = rewriter.create<stablehlo::DynamicSliceOp>(
          candidate.DUS.getLoc(), candidate.outerOperand,
          candidate.DUS.getStartIndices(),
          candidate.DUS.getUpdate().getType().getShape());
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        newReturnValues[candidate.idx] = candidate.DUS.getUpdate();
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    {
      mlir::IRMapping mapper;
      Value useInner = nullptr;
      if (hasConditional) {

        for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
          mapper.map(whileOp.getCond().getArgument(i),
                     whileOp.getOperands()[i]);
        }
        for (auto &op : whileOp.getCond().front().getOperations()) {
          // Skip the terminator - we'll add it after all other operations
          if (isa<stablehlo::ReturnOp>(op))
            continue;

          // Clone the operation with the value mapping
          rewriter.clone(op, mapper);
        }
        useInner = whileOp.getCond().front().getTerminator()->getOperand(0);
        useInner = mapper.lookupOrDefault(useInner);
      }
      for (auto &candidate : candidates) {
        Value operand = candidate.outerOperand;
        if (candidate.conditionalOperand) {
          operand = rewriter.create<stablehlo::SelectOp>(
              whileOp.getLoc(), useInner, candidate.conditionalOperand,
              operand);
        }

        results[candidate.idx] =
            rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                candidate.DUS.getLoc(), operand, results[candidate.idx],
                candidate.DUS.getStartIndices());
      }
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    // Create an IR mapper to map values from old op to new op
    mlir::IRMapping mapper;

    // Clear the new body block but keep its arguments
    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock
        .clear(); // This clears operations but preserves block arguments

    // Clone operations from old body to new body
    Block &oldBodyBlock = whileOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      // Set up operand mapping for the body region
      for (unsigned i = 0; i < whileOp.getBody().getNumArguments(); ++i) {
        auto oldArg = whileOp.getBody().getArgument(i);
        Value newArg = newWhileOp.getBody().getArgument(i);
        for (auto &pair : candidates) {
          if (pair.idx == i) {
            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.DUS.getLoc(), pair.outerOperand, newArg,
                pair.DUS.getStartIndices());
            break;
          }
        }
        mapper.map(oldArg, newArg);
      }

      for (auto &op : oldBodyBlock.getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
    }

    // Create a new terminator for the body region using new values
    {
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> newReturnValues;

      // Map old return values to new values using the mapper
      for (auto oldRetVal : yieldOp.getOperands()) {
        Value newRetVal = mapper.lookupOrNull(oldRetVal);
        // If the value isn't in the mapper, maybe it was a block argument or
        // constant
        if (!newRetVal)
          newRetVal = oldRetVal; // Consider more robust handling if needed
        newReturnValues.push_back(newRetVal);
      }

      // Create the return op at the end of the body
      rewriter.setInsertionPointToEnd(&newBodyBlock);
      rewriter.create<stablehlo::ReturnOp>(yieldOp.getLoc(), newReturnValues);
    }

    // Create condition region mapper
    mlir::IRMapping condMapper;

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    Block &oldCondBlock = whileOp.getCond().front();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        auto oldArg = whileOp.getCond().getArgument(i);
        Value newArg = newWhileOp.getCond().getArgument(i);
        for (auto &pair : candidates) {
          if (pair.idx == i) {
            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.DUS.getLoc(), pair.outerOperand, newArg,
                pair.DUS.getStartIndices());
            break;
          }
        }
        condMapper.map(oldArg, newArg);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        rewriter.clone(op, condMapper);
      }
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);

    return success();
  }
};

struct IVInfo {
  int index;    // Index of the induction variable in the while op arguments
  int64_t step; // Step size (how much IV increments each iteration)
  Value limit;
  Value start;
  bool isValid; // Whether we successfully identified the IV and step
  bool zeroStart;
  bool canonical;
};

IVInfo extractSimpleIVInfo(stablehlo::WhileOp whileOp) {
  IVInfo result = {-1, 0, nullptr, nullptr, false, false, false};

  // 1. Get the condition block and its return operation
  if (whileOp.getBody().empty() || whileOp.getCond().empty())
    return result;

  Block &condBlock = whileOp.getCond().front();
  if (condBlock.empty())
    return result;

  auto condReturn = dyn_cast<stablehlo::ReturnOp>(condBlock.getTerminator());
  if (!condReturn || condReturn.getNumOperands() != 1)
    return result;

  // 2. Check if the return operand comes from a compare operation
  auto compareOp =
      condReturn.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
  if (!compareOp)
    return result;

  // 3. Identify which block argument is used in the comparison (the IV)
  Value lhs = compareOp.getLhs();
  Value rhs = compareOp.getRhs();

  // Check which side of the comparison is a block argument
  BlockArgument ivArg = nullptr;
  Value limit;
  for (BlockArgument arg : condBlock.getArguments()) {
    if (lhs == arg && (compareOp.getComparisonDirection() ==
                           stablehlo::ComparisonDirection::LT ||
                       compareOp.getComparisonDirection() ==
                           stablehlo::ComparisonDirection::NE)) {
      ivArg = arg;
      limit = rhs;
      break;
    }
    if (rhs == arg && (compareOp.getComparisonDirection() ==
                           stablehlo::ComparisonDirection::GT ||
                       compareOp.getComparisonDirection() ==
                           stablehlo::ComparisonDirection::NE)) {
      ivArg = arg;
      limit = rhs;
      break;
    }
  }

  if (!ivArg)
    return result;

  if (!definedOutside(limit, whileOp))
    return result;

  result.limit = limit;

  // Record the IV index
  result.index = ivArg.getArgNumber();

  // 4. Now analyze the body region to find the step
  Block &bodyBlock = whileOp.getBody().front();
  auto bodyReturn = dyn_cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
  if (!bodyReturn || bodyReturn.getNumOperands() <= result.index)
    return result;

  // Get the value being returned for the IV position
  Value updatedIV = bodyReturn.getOperand(result.index);

  // Look for an add operation that defines the updated IV
  auto addOp = updatedIV.getDefiningOp<stablehlo::AddOp>();
  if (!addOp)
    return result;

  // Check if one side is a constant (the step)
  for (int i = 0; i < 2; i++) {
    if (addOp->getOperand(i) !=
        whileOp.getBody().getArgument(ivArg.getArgNumber()))
      continue;

    DenseIntElementsAttr idxAttr;
    if (!matchPattern(addOp->getOperand(1 - i), m_Constant(&idxAttr)))
      continue;

    result.step = (*idxAttr.begin()).getSExtValue();
    result.isValid = true;
    break;
  }

  result.start = whileOp.getOperands()[result.index];

  DenseIntElementsAttr inpAttr;
  if (matchPattern(result.start, m_Constant(&inpAttr))) {
    result.zeroStart = (*inpAttr.begin()).getSExtValue() == 0;
  }

  result.canonical = result.zeroStart && result.step == 1;

  return result;
}

/// Detects if a value is a replication padding pattern.
stablehlo::ConcatenateOp detectReplicationPadding(Value yieldOperand) {

  // Check if the value is a concatenate operation
  auto concatOp = yieldOperand.getDefiningOp<stablehlo::ConcatenateOp>();
  if (!concatOp)
    return nullptr;

  // Check if we have exactly 3 operands (left border, main tensor, right
  // border)
  if (concatOp.getInputs().size() != 3)
    return nullptr;

  // Get the concat dimension
  int64_t concatDim = concatOp.getDimension();

  // We need to identify potential slice operations that extract borders
  // Check first and last operands
  auto firstSlice = concatOp.getOperand(0).getDefiningOp<stablehlo::SliceOp>();
  auto lastSlice = concatOp.getOperand(concatOp.getNumOperands() - 1)
                       .getDefiningOp<stablehlo::SliceOp>();

  // Check the middle tensor (should be the main tensor)
  Value middleTensor = concatOp.getOperand(1);

  // Both first and last operands must be slice operations
  if (!firstSlice || !lastSlice)
    return nullptr;

  // The slices must come from the same source tensor
  if (firstSlice.getOperand() != middleTensor ||
      lastSlice.getOperand() != middleTensor)
    return nullptr;

  // Verify the slices are taking border elements
  // Get the shapes
  auto middleTensorType = dyn_cast<RankedTensorType>(middleTensor.getType());
  if (!middleTensorType || !middleTensorType.hasStaticShape())
    return nullptr;

  // Check that first slice is taking the first element along concat dimension
  if (firstSlice.getStartIndices()[concatDim] != 0 ||
      firstSlice.getLimitIndices()[concatDim] != 1)
    return nullptr;

  // Check the rest of the firstSlice indices are the same size as the middle
  // tensor
  if (lastSlice.getStartIndices()[concatDim] !=
          middleTensorType.getShape()[concatDim] - 1 ||
      lastSlice.getLimitIndices()[concatDim] !=
          middleTensorType.getShape()[concatDim])
    return nullptr;

  for (int i = 0; i < firstSlice.getStartIndices().size(); i++) {
    if (i == concatDim) // Already checked
      continue;

    // check limit indices are the same size as the middle tensor
    if (firstSlice.getLimitIndices()[i] != middleTensorType.getShape()[i])
      return nullptr;
    // check start indices are 0
    if (firstSlice.getStartIndices()[i] != 0)
      return nullptr;

    // check limit indices are the same size as the middle tensor
    if (lastSlice.getLimitIndices()[i] != middleTensorType.getShape()[i])
      return nullptr;
    // check start indices are 0
    if (lastSlice.getStartIndices()[i] != 0)
      return nullptr;
  }

  return concatOp;
}

struct WhileRepeatedInductionReduction
    : public CheckedOpRewritePattern<stablehlo::WhileOp,
                                     WhileRepeatedInductionReduction> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  template <typename CRangeT, typename RangeT>
  static stablehlo::IfOp
  createConditional(PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
                    IVInfo &ivInfo, CRangeT &candidates, Value iv,
                    const RangeT iterOperand, bool cloneCond) {
    SmallVector<Type> ifResultTypes;

    SmallVector<Value> oldReturns;
    for (auto &candidate : candidates) {
      ifResultTypes.push_back(whileOp.getOperands()[candidate.idx].getType());
      oldReturns.push_back(whileOp.getOperands()[candidate.idx]);
    }

    Value condition;

    if (cloneCond) {
      IRMapping mapper;

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        mapper.map(whileOp.getCond().getArgument(i), whileOp.getOperands()[i]);
      }
      for (auto &op : whileOp.getCond().front().getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
      condition = whileOp.getCond().front().getTerminator()->getOperand(0);
      condition = mapper.lookupOrDefault(condition);
    } else {
      condition = rewriter.create<stablehlo::CompareOp>(
          whileOp.getLoc(), iv, ivInfo.start,
          stablehlo::ComparisonDirection::EQ);
    }

    auto ifOp = rewriter.create<stablehlo::IfOp>(whileOp.getLoc(),
                                                 ifResultTypes, condition);

    // Create the then and else regions for the if operation

    {
      Region &thenRegion =
          cloneCond ? ifOp.getFalseBranch() : ifOp.getTrueBranch();
      OpBuilder::InsertionGuard guard(rewriter);
      Block *thenBlock = rewriter.createBlock(&thenRegion);
      rewriter.setInsertionPointToStart(thenBlock);
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), oldReturns);
    }

    {
      Region &elseRegion =
          cloneCond ? ifOp.getTrueBranch() : ifOp.getFalseBranch();
      OpBuilder::InsertionGuard guard(rewriter);
      Block *elseBlock = rewriter.createBlock(&elseRegion);
      rewriter.setInsertionPointToStart(elseBlock);
      SmallVector<Value> newReturns;

      for (auto &candidate : candidates) {
        auto T =
            cast<RankedTensorType>(candidate.middleTensor.getType()).getShape();

        SmallVector<int64_t> interior(T.size(), 1);

        SmallVector<int64_t> firstLower(T.size(), 0);

        SmallVector<int64_t> firstUpper = llvm::to_vector(T);
        firstUpper[candidate.concat.getDimension()] = 1;

        SmallVector<int64_t> thirdLower(T.size(), 0);
        thirdLower[candidate.concat.getDimension()] =
            T[candidate.concat.getDimension()] - 1;

        SmallVector<int64_t> thirdUpper = llvm::to_vector(T);

        Value args[3] = {
            rewriter
                .create<stablehlo::SliceOp>(whileOp.getLoc(),
                                            iterOperand[candidate.idx],
                                            firstLower, firstUpper, interior)
                .getResult(),
            iterOperand[candidate.idx],
            rewriter
                .create<stablehlo::SliceOp>(whileOp.getLoc(),
                                            iterOperand[candidate.idx],
                                            thirdLower, thirdUpper, interior)
                .getResult(),
        };

        newReturns.push_back(rewriter.create<stablehlo::ConcatenateOp>(
            candidate.concat.getLoc(), args, candidate.concat.getDimension()));
      }
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), newReturns);
    }

    return ifOp;
  }

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find the index of IV and the step to check for 1 iteration
    auto ivInfo = extractSimpleIVInfo(whileOp);
    if (!ivInfo.isValid)
      return failure();

    if (ivInfo.step == 0)
      return failure();

    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct Candidate {
      unsigned idx;
      stablehlo::ConcatenateOp concat;
      Value middleTensor;
      Value outerOperand;
      SmallVector<int64_t> innerStarts;
      SmallVector<int64_t> innerEnds;
      SmallVector<int64_t> innerStrides;
    };

    llvm::SmallVector<Candidate> candidates;
    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      auto concat = detectReplicationPadding(yieldOp.getOperand(idx));
      if (!concat)
        continue;

      auto T = concat.getType();

      SmallVector<int64_t> innerStarts(T.getShape().size(), 0);
      SmallVector<int64_t> innerEnds = llvm::to_vector(T.getShape());
      SmallVector<int64_t> innerStrides(T.getShape().size(), 1);
      innerStarts[concat.getDimension()] += 1;
      innerEnds[concat.getDimension()] -= 1;
      candidates.emplace_back(Candidate{idx, concat, concat.getOperand(1),
                                        whileOp.getOperands()[idx], innerStarts,
                                        innerEnds, innerStrides});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : candidates) {
      newOperands[candidate.idx] = rewriter.create<stablehlo::SliceOp>(
          candidate.concat.getLoc(), candidate.outerOperand,
          candidate.innerStarts, candidate.innerEnds, candidate.innerStrides);
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        newReturnValues[candidate.idx] = candidate.middleTensor;
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    auto ifOp =
        createConditional(rewriter, whileOp, ivInfo, candidates,
                          whileOp.getOperands()[ivInfo.index], results, true);
    for (auto res : ifOp->getResults()) {
      results[candidates[res.getResultNumber()].idx] = res;
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    // Create an IR mapper to map values from old op to new op
    mlir::IRMapping mapper;

    // Clear the new body block but keep its arguments
    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock
        .clear(); // This clears operations but preserves block arguments

    // Clone operations from old body to new body
    Block &oldBodyBlock = whileOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      // Set up operand mapping for the body region
      for (unsigned i = 0; i < whileOp.getBody().getNumArguments(); ++i) {
        auto oldArg = whileOp.getBody().getArgument(i);
        Value newArg = newWhileOp.getBody().getArgument(i);
        mapper.map(oldArg, newArg);
      }

      for (auto res :
           createConditional(rewriter, whileOp, ivInfo, candidates,
                             newWhileOp.getBody().getArgument(ivInfo.index),
                             newWhileOp.getBody().getArguments(), false)
               ->getResults()) {
        auto idx = candidates[res.getResultNumber()].idx;
        mapper.map(whileOp.getBody().getArgument(idx), res);
      }

      for (auto &op : oldBodyBlock.getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
    }

    // Create a new terminator for the body region using new values
    {
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> newReturnValues;

      // Map old return values to new values using the mapper
      for (auto oldRetVal : yieldOp.getOperands()) {
        Value newRetVal = mapper.lookupOrNull(oldRetVal);
        // If the value isn't in the mapper, maybe it was a block argument or
        // constant
        if (!newRetVal)
          newRetVal = oldRetVal; // Consider more robust handling if needed
        newReturnValues.push_back(newRetVal);
      }

      // Create the return op at the end of the body
      rewriter.setInsertionPointToEnd(&newBodyBlock);
      rewriter.create<stablehlo::ReturnOp>(yieldOp.getLoc(), newReturnValues);
    }

    // Create condition region mapper
    mlir::IRMapping condMapper;

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    Block &oldCondBlock = whileOp.getCond().front();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        auto oldArg = whileOp.getCond().getArgument(i);
        Value newArg = newWhileOp.getCond().getArgument(i);
        condMapper.map(oldArg, newArg);
      }

      for (auto res :
           createConditional(rewriter, whileOp, ivInfo, candidates,
                             newWhileOp.getCond().getArgument(ivInfo.index),
                             newWhileOp.getCond().getArguments(), false)
               ->getResults()) {
        auto idx = candidates[res.getResultNumber()].idx;
        condMapper.map(whileOp.getBody().getArgument(idx), res);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        rewriter.clone(op, condMapper);
      }
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);
    return success();
  }
};

struct WhilePadInductionReduction
    : public CheckedOpRewritePattern<stablehlo::WhileOp,
                                     WhilePadInductionReduction> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  template <typename CRangeT, typename RangeT>
  static stablehlo::IfOp
  createConditional(PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
                    IVInfo &ivInfo, CRangeT &candidates, Value iv,
                    const RangeT iterOperand, bool cloneCond) {
    SmallVector<Type> ifResultTypes;

    SmallVector<Value> oldReturns;
    for (auto &candidate : candidates) {
      ifResultTypes.push_back(whileOp.getOperands()[candidate.idx].getType());
      oldReturns.push_back(whileOp.getOperands()[candidate.idx]);
    }

    Value condition;

    if (cloneCond) {
      IRMapping mapper;

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        mapper.map(whileOp.getCond().getArgument(i), whileOp.getOperands()[i]);
      }
      for (auto &op : whileOp.getCond().front().getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
      condition = whileOp.getCond().front().getTerminator()->getOperand(0);
      condition = mapper.lookupOrDefault(condition);
    } else {
      condition = rewriter.create<stablehlo::CompareOp>(
          whileOp.getLoc(), iv, ivInfo.start,
          stablehlo::ComparisonDirection::EQ);
    }

    auto ifOp = rewriter.create<stablehlo::IfOp>(whileOp.getLoc(),
                                                 ifResultTypes, condition);

    // Create the then and else regions for the if operation

    {
      Region &thenRegion =
          cloneCond ? ifOp.getFalseBranch() : ifOp.getTrueBranch();
      OpBuilder::InsertionGuard guard(rewriter);
      Block *thenBlock = rewriter.createBlock(&thenRegion);
      rewriter.setInsertionPointToStart(thenBlock);
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), oldReturns);
    }

    {
      Region &elseRegion =
          cloneCond ? ifOp.getTrueBranch() : ifOp.getFalseBranch();
      OpBuilder::InsertionGuard guard(rewriter);
      Block *elseBlock = rewriter.createBlock(&elseRegion);
      rewriter.setInsertionPointToStart(elseBlock);
      SmallVector<Value> newReturns;

      for (auto &candidate : candidates) {
        // Create a pad op using these values
        auto padOp = rewriter.create<stablehlo::PadOp>(
            candidate.pad.getLoc(),
            iterOperand[candidate.idx], // operand
            candidate.paddingValue,     // padding_value
            candidate.lowAttr,          // edge_padding_low
            candidate.highAttr,         // edge_padding_high
            candidate.interiorAttr);    // interior_padding
        newReturns.push_back(padOp.getResult());
      }
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), newReturns);
    }

    return ifOp;
  }

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find the index of IV and the step to check for 1 iteration
    auto ivInfo = extractSimpleIVInfo(whileOp);
    if (!ivInfo.isValid)
      return failure();

    if (ivInfo.step == 0)
      return failure();

    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct Candidate {
      unsigned idx;
      stablehlo::PadOp pad;
      Value operand;                  // The operand being padded
      Value paddingValue;             // The padding value
      DenseI64ArrayAttr lowAttr;      // Edge padding low attribute
      DenseI64ArrayAttr highAttr;     // Edge padding high attribute
      DenseI64ArrayAttr interiorAttr; // Interior padding attribute
    };

    llvm::SmallVector<Candidate> candidates;
    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      // check if operand is a pad operation
      auto pad = yieldOp.getOperand(idx).getDefiningOp<stablehlo::PadOp>();
      if (!pad)
        continue;

      if (!definedOutside(pad.getPaddingValue(), whileOp))
        continue;

      // Store original attributes directly
      candidates.emplace_back(
          Candidate{idx, pad, pad.getOperand(), pad.getPaddingValue(),
                    rewriter.getDenseI64ArrayAttr(pad.getEdgePaddingLow()),
                    rewriter.getDenseI64ArrayAttr(pad.getEdgePaddingHigh()),
                    rewriter.getDenseI64ArrayAttr(pad.getInteriorPadding())});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create sliced versions of each padded input
    for (auto &candidate : candidates) {

      // Create an undef op of the same type as the operand
      auto tensorType = cast<TensorType>(candidate.operand.getType());
      auto zeroAttr = DenseElementsAttr::get(
          tensorType, rewriter.getZeroAttr(tensorType.getElementType()));
      auto undef = rewriter.create<stablehlo::ConstantOp>(
          candidate.operand.getLoc(), zeroAttr);
      newOperands[candidate.idx] = undef;
    }

    // Update yield op to use the unpadded values
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        newReturnValues[candidate.idx] = candidate.operand;
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    auto ifOp =
        createConditional(rewriter, whileOp, ivInfo, candidates,
                          whileOp.getOperands()[ivInfo.index], results, true);
    for (auto res : ifOp->getResults()) {
      results[candidates[res.getResultNumber()].idx] = res;
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock.clear();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      SmallVector<Value> replacements;
      for (auto arg : newBodyBlock.getArguments())
        replacements.push_back(arg);
      for (auto res :
           createConditional(rewriter, whileOp, ivInfo, candidates,
                             newWhileOp.getBody().getArgument(ivInfo.index),
                             newWhileOp.getBody().getArguments(), false)
               ->getResults()) {
        auto idx = candidates[res.getResultNumber()].idx;
        replacements[idx] = res;
      }
      rewriter.mergeBlocks(&whileOp.getBody().front(), &newBodyBlock,
                           replacements);
    }

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      SmallVector<Value> replacements;
      for (auto arg : newCondBlock.getArguments())
        replacements.push_back(arg);

      for (auto res :
           createConditional(rewriter, whileOp, ivInfo, candidates,
                             newWhileOp.getCond().getArgument(ivInfo.index),
                             newWhileOp.getCond().getArguments(), false)
               ->getResults()) {
        auto idx = candidates[res.getResultNumber()].idx;
        replacements[idx] = res;
      }

      rewriter.mergeBlocks(&whileOp.getCond().front(), &newCondBlock,
                           replacements);
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);
    return success();
  }
};

struct WhileInductionReduction
    : public CheckedOpRewritePattern<stablehlo::WhileOp,
                                     WhileInductionReduction> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct Candidate {
      unsigned idx;
      SmallVector<int64_t> lowerBounds;
      SmallVector<int64_t> lowerUpdateBounds;
      SmallVector<int64_t> upperBounds;
      SmallVector<int64_t> upperUpdateBounds;
      BlockArgument argOperand;
      BlockArgument condOperand;
      Value outerOperand;
      Value outerSlice;
    };

    llvm::SmallVector<Candidate, 4> candidates;
    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      bool legal = true;
      // Skip DUS candidates which can be removed in a better form by WhileDUS
      if (auto DUS = yieldOp.getOperand(idx)
                         .getDefiningOp<stablehlo::DynamicUpdateSliceOp>()) {

        bool legalDUS = true;
        if (DUS.getOperand() == whileOp.getBody().front().getArgument(idx)) {
        } else if (definedOutside(DUS.getOperand(), whileOp)) {

          bool hasArgUse = !whileOp.getCond().getArgument(idx).use_empty() ||
                           !whileOp.getBody().getArgument(idx).use_empty();

          if (hasArgUse) {
            legalDUS = false;
          }

        } else {
          legalDUS = false;
        }

        if (legalDUS)
          legal = false;
      }

      if (!legal)
        continue;

      auto argOperand = whileOp.getBody().getArgument(idx);
      auto condOperand = whileOp.getCond().getArgument(idx);

      auto T = cast<RankedTensorType>(argOperand.getType());
      auto rank = T.getShape().size();

      if (rank == 0)
        continue;

      SmallVector<int64_t> lowerBounds = llvm::to_vector(T.getShape());
      SmallVector<int64_t> upperBounds(T.getShape().size(), 0);
      SmallVector<int64_t> lowerUpdateBounds = llvm::to_vector(T.getShape());
      SmallVector<int64_t> upperUpdateBounds(T.getShape().size(), 0);

      // We want to find a smaller window to use as the inductive variable
      // This means that no slice can read from outside the window (otherwise we
      // need the data). We also want to ensure that no dynamic update slice
      // overwrites outside the window
      //. This is because we want to just do a single DUS of the original input
      // outside the window

      SmallVector<Value> todo = {argOperand, condOperand};
      bool selfYield = false;
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        for (auto &u : cur.getUses()) {
          Operation *user = u.getOwner();
          if (auto use = dyn_cast<stablehlo::SliceOp>(user)) {
            for (int i = 0; i < rank; i++) {
              if (use.getStrides()[i] != 1) {
                legal = false;
                break;
              }
              lowerBounds[i] =
                  std::min(lowerBounds[i], use.getStartIndices()[i]);
              upperBounds[i] =
                  std::max(upperBounds[i], use.getLimitIndices()[i]);
            }
            continue;
          }

          if (auto use = dyn_cast<stablehlo::DynamicUpdateSliceOp>(user)) {
            if (use.getUpdate() == cur) {
              legal = false;
              break;
            }
            for (int i = 0; i < rank; i++) {

              DenseIntElementsAttr idxAttr;
              if (!matchPattern(use.getStartIndices()[i],
                                m_Constant(&idxAttr)) ||
                  idxAttr.getNumElements() != 1) {
                legal = false;
                break;
              }
              auto start = (*idxAttr.begin()).getSExtValue();
              auto end = start + use.getUpdate().getType().getShape()[i];

              lowerBounds[i] = std::min(lowerBounds[i], start);
              upperBounds[i] = std::max(upperBounds[i], end);
              lowerUpdateBounds[i] = std::min(lowerUpdateBounds[i], start);
              upperUpdateBounds[i] = std::max(upperUpdateBounds[i], end);
            }
            todo.push_back(use);
            continue;
          }

          if (auto use = dyn_cast<stablehlo::ReturnOp>(user)) {
            if (use->getParentOp() != whileOp) {
              legal = false;
              break;
            }
            if (u.getOperandNumber() != idx) {
              legal = false;
              break;
            }
            selfYield = true;
            continue;
          }

          legal = false;
          break;
        }
        if (!legal)
          break;
      }
      if (!legal)
        continue;
      if (!selfYield)
        continue;

      bool seenSlice = false;

      for (int i = 0; i < rank; i++) {
        if (lowerUpdateBounds[i] >= upperUpdateBounds[i]) {
          legal = false;
          continue;
        }
        if (lowerUpdateBounds[i] != 0) {
          seenSlice = true;
          break;
        }
        if (upperUpdateBounds[i] != T.getShape()[i]) {
          seenSlice = true;
          break;
        }
      }

      if (!legal || !seenSlice)
        continue;

      candidates.emplace_back(Candidate{
          idx, lowerBounds, lowerUpdateBounds, upperBounds, upperUpdateBounds,
          argOperand, condOperand, whileOp.getOperands()[idx], nullptr});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : candidates) {
      // Create a new transpose before the while loop
      SmallVector<int64_t> strides(candidate.lowerBounds.size(), 1);
      candidate.outerSlice = rewriter.create<stablehlo::SliceOp>(
          candidate.argOperand.getLoc(), candidate.outerOperand,
          candidate.lowerBounds, candidate.upperBounds, strides);
      SmallVector<int64_t> lower = llvm::to_vector(candidate.lowerUpdateBounds);
      SmallVector<int64_t> upper = llvm::to_vector(candidate.upperUpdateBounds);
      for (int i = 0; i < lower.size(); i++) {
        lower[i] -= candidate.lowerBounds[i];
        upper[i] -= candidate.lowerBounds[i];
      }
      newOperands[candidate.idx] = rewriter.create<stablehlo::SliceOp>(
          candidate.argOperand.getLoc(), candidate.outerSlice, lower, upper,
          strides);
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        SmallVector<int64_t> strides(candidate.lowerBounds.size(), 1);
        newReturnValues[candidate.idx] = rewriter.create<stablehlo::SliceOp>(
            candidate.argOperand.getLoc(), newReturnValues[candidate.idx],
            candidate.lowerUpdateBounds, candidate.upperUpdateBounds, strides);
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    {
      auto ctype = RankedTensorType::get({}, rewriter.getI64Type());
      for (auto &candidate : candidates) {
        SmallVector<Value> starts;
        for (auto idx : candidate.lowerUpdateBounds) {
          starts.push_back(rewriter.create<stablehlo::ConstantOp>(
              candidate.argOperand.getLoc(), ctype,
              cast<ElementsAttr>(makeAttr(ctype, idx))));
        }

        results[candidate.idx] =
            rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                candidate.argOperand.getLoc(), candidate.outerOperand,
                results[candidate.idx], starts);
      }
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    // Create an IR mapper to map values from old op to new op
    mlir::IRMapping mapper;

    // Clear the new body block but keep its arguments
    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock
        .clear(); // This clears operations but preserves block arguments

    // Clone operations from old body to new body
    Block &oldBodyBlock = whileOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      // Set up operand mapping for the body region
      for (unsigned i = 0; i < whileOp.getBody().getNumArguments(); ++i) {
        auto oldArg = whileOp.getBody().getArgument(i);
        Value newArg = newWhileOp.getBody().getArgument(i);
        for (auto &pair : candidates) {
          if (pair.idx == i) {

            auto itype = RankedTensorType::get({}, rewriter.getI64Type());

            SmallVector<Value> update_starts;
            for (int i = 0; i < pair.lowerBounds.size(); i++) {
              update_starts.push_back(rewriter.create<stablehlo::ConstantOp>(
                  pair.argOperand.getLoc(), itype,
                  cast<ElementsAttr>(
                      makeAttr(itype, pair.lowerUpdateBounds[i] -
                                          pair.lowerBounds[i]))));
            }

            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.argOperand.getLoc(), pair.outerSlice, newArg,
                update_starts);

            auto ctype = RankedTensorType::get(
                {}, cast<RankedTensorType>(pair.argOperand.getType())
                        .getElementType());
            auto padVal = rewriter.create<stablehlo::ConstantOp>(
                pair.argOperand.getLoc(), ctype,
                cast<ElementsAttr>(makeAttr(ctype, 0)));

            SmallVector<int64_t> slow = llvm::to_vector(pair.lowerBounds);
            SmallVector<int64_t> shigh = llvm::to_vector(
                cast<RankedTensorType>(pair.argOperand.getType()).getShape());
            for (int i = 0; i < shigh.size(); i++)
              shigh[i] -= pair.upperBounds[i];
            SmallVector<int64_t> sint(shigh.size(), 0);

            newArg = rewriter.create<stablehlo::PadOp>(
                pair.argOperand.getLoc(), newArg, padVal, slow, shigh, sint);
            break;
          }
        }
        mapper.map(oldArg, newArg);
      }

      for (auto &op : oldBodyBlock.getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
    }

    // Create a new terminator for the body region using new values
    {
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> newReturnValues;

      // Map old return values to new values using the mapper
      for (auto oldRetVal : yieldOp.getOperands()) {
        Value newRetVal = mapper.lookupOrNull(oldRetVal);
        // If the value isn't in the mapper, maybe it was a block argument or
        // constant
        if (!newRetVal)
          newRetVal = oldRetVal; // Consider more robust handling if needed
        newReturnValues.push_back(newRetVal);
      }

      // Create the return op at the end of the body
      rewriter.setInsertionPointToEnd(&newBodyBlock);
      rewriter.create<stablehlo::ReturnOp>(yieldOp.getLoc(), newReturnValues);
    }

    // Create condition region mapper
    mlir::IRMapping condMapper;

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    Block &oldCondBlock = whileOp.getCond().front();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        auto oldArg = whileOp.getCond().getArgument(i);
        Value newArg = newWhileOp.getCond().getArgument(i);
        for (auto &pair : candidates) {
          if (pair.idx == i) {
            auto itype = RankedTensorType::get({}, rewriter.getI64Type());

            SmallVector<Value> update_starts;
            for (int i = 0; i < pair.lowerBounds.size(); i++) {
              update_starts.push_back(rewriter.create<stablehlo::ConstantOp>(
                  pair.argOperand.getLoc(), itype,
                  cast<ElementsAttr>(
                      makeAttr(itype, pair.lowerUpdateBounds[i] -
                                          pair.lowerBounds[i]))));
            }

            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.argOperand.getLoc(), pair.outerSlice, newArg,
                update_starts);
            auto ctype = RankedTensorType::get(
                {}, cast<RankedTensorType>(pair.condOperand.getType())
                        .getElementType());
            auto padVal = rewriter.create<stablehlo::ConstantOp>(
                pair.condOperand.getLoc(), ctype,
                cast<ElementsAttr>(makeAttr(ctype, 0)));

            SmallVector<int64_t> slow = llvm::to_vector(pair.lowerBounds);
            SmallVector<int64_t> shigh = llvm::to_vector(
                cast<RankedTensorType>(pair.condOperand.getType()).getShape());
            for (int i = 0; i < shigh.size(); i++)
              shigh[i] -= pair.upperBounds[i];
            SmallVector<int64_t> sint(shigh.size(), 0);

            newArg = rewriter.create<stablehlo::PadOp>(
                pair.condOperand.getLoc(), newArg, padVal, slow, shigh, sint);
            break;
          }
        }
        condMapper.map(oldArg, newArg);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        rewriter.clone(op, condMapper);
      }
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);
    return success();
  }
};

// TODO: this is not valid in general but presumes the inner structure is valid
// from the input
struct WhileConcat
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileConcat> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct Candidate {
      unsigned idx;
      stablehlo::ConcatenateOp concat;
      int lhsSize;
      int rhsSize;
      stablehlo::SliceOp ops[3];
    };

    llvm::SmallVector<Candidate, 4> candidates;

    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      auto concat =
          yieldOp.getOperand(idx).getDefiningOp<stablehlo::ConcatenateOp>();

      // Check that the while result has exactly one use
      if (!concat)
        continue;

      if (!concat->hasOneUse()) {
        continue;
      }

      if (concat.getOperands().size() != 3)
        continue;

      stablehlo::SliceOp ops[3] = {
          concat.getOperands()[0].getDefiningOp<stablehlo::SliceOp>(),
          concat.getOperands()[1].getDefiningOp<stablehlo::SliceOp>(),
          concat.getOperands()[2].getDefiningOp<stablehlo::SliceOp>(),
      };

      if (!ops[0])
        continue;
      if (!ops[1])
        continue;
      if (!ops[2])
        continue;

      if (ops[0].getOperand() != ops[1].getOperand())
        continue;
      if (ops[0].getOperand() != ops[2].getOperand())
        continue;

      bool legal = true;

      for (int opn = 0; opn < 3; opn++)
        for (int i = 0; i < concat.getType().getShape().size(); i++)
          if (i != concat.getDimension()) {
            if (ops[0].getStartIndices()[i] != ops[opn].getStartIndices()[i]) {
              legal = false;
              break;
            }
            if (ops[opn].getStrides()[i] != 1) {
              legal = false;
              break;
            }
            if (ops[0].getLimitIndices()[i] != ops[opn].getLimitIndices()[i]) {
              legal = false;
              break;
            }
          }

      if (!legal)
        continue;

      int lowerLim = ops[1].getStartIndices()[concat.getDimension()];
      int upperLim = ops[1].getLimitIndices()[concat.getDimension()];
      if (ops[2].getStartIndices()[concat.getDimension()] != lowerLim)
        continue;

      if (ops[0].getLimitIndices()[concat.getDimension()] != upperLim)
        continue;

      int lhsSize = ops[0].getLimitIndices()[concat.getDimension()] -
                    ops[0].getStartIndices()[concat.getDimension()];
      if (lhsSize !=
          ops[1].getOperand().getType().getShape()[concat.getDimension()] -
              upperLim)
        continue;
      int rhsSize = ops[2].getLimitIndices()[concat.getDimension()] -
                    ops[2].getStartIndices()[concat.getDimension()];
      if (rhsSize != lowerLim)
        continue;

      // TODO this is unsafe unless the input is verified to have this property
      // For now we will assume it

      candidates.emplace_back(
          Candidate{idx, concat, lhsSize, rhsSize, {ops[0], ops[1], ops[2]}});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : candidates) {
      // Create a new transpose before the while loop

      newOperands[candidate.idx] = rewriter.create<stablehlo::SliceOp>(
          candidate.ops[1].getLoc(), whileOp.getOperands()[candidate.idx],
          candidate.ops[1].getStartIndices(),
          candidate.ops[1].getLimitIndices(), candidate.ops[1].getStrides());
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        newReturnValues[candidate.idx] = candidate.ops[1];
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    {
      for (auto &candidate : candidates) {

        SmallVector<int64_t> lowerStarts(
            candidate.concat.getType().getShape().size(), 0);
        SmallVector<int64_t> upperStarts(
            candidate.concat.getType().getShape().size(), 0);
        SmallVector<int64_t> lowerEnds(
            cast<RankedTensorType>(results[candidate.idx].getType())
                .getShape()
                .begin(),
            cast<RankedTensorType>(results[candidate.idx].getType())
                .getShape()
                .end());
        SmallVector<int64_t> upperEnds(
            cast<RankedTensorType>(results[candidate.idx].getType())
                .getShape()
                .begin(),
            cast<RankedTensorType>(results[candidate.idx].getType())
                .getShape()
                .end());
        SmallVector<int64_t> strides(
            candidate.concat.getType().getShape().size(), 1);

        lowerEnds[candidate.concat.getDimension()] = candidate.lhsSize;

        upperStarts[candidate.concat.getDimension()] =
            upperEnds[candidate.concat.getDimension()] - candidate.rhsSize;

        Value ops[3] = {
            rewriter.create<stablehlo::SliceOp>(
                candidate.concat.getLoc(), results[candidate.idx], lowerStarts,
                lowerEnds, strides),
            results[candidate.idx],
            rewriter.create<stablehlo::SliceOp>(
                candidate.concat.getLoc(), results[candidate.idx], upperStarts,
                upperEnds, strides),

        };
        results[candidate.idx] = rewriter.create<stablehlo::ConcatenateOp>(
            candidate.concat.getLoc(), ops, candidate.concat.getDimension());
      }
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    // Create an IR mapper to map values from old op to new op
    mlir::IRMapping mapper;

    // Clear the new body block but keep its arguments
    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock
        .clear(); // This clears operations but preserves block arguments

    // Clone operations from old body to new body
    Block &oldBodyBlock = whileOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      // Set up operand mapping for the body region
      for (unsigned i = 0; i < whileOp.getBody().getNumArguments(); ++i) {
        auto oldArg = whileOp.getBody().getArgument(i);
        Value newArg = newWhileOp.getBody().getArgument(i);
        for (auto &candidate : candidates) {
          if (candidate.idx == i) {

            SmallVector<int64_t> lowerStarts(
                candidate.concat.getType().getShape().size(), 0);
            SmallVector<int64_t> upperStarts(
                candidate.concat.getType().getShape().size(), 0);
            SmallVector<int64_t> lowerEnds(
                cast<RankedTensorType>(newArg.getType()).getShape().begin(),
                cast<RankedTensorType>(newArg.getType()).getShape().end());
            SmallVector<int64_t> upperEnds(
                cast<RankedTensorType>(newArg.getType()).getShape().begin(),
                cast<RankedTensorType>(newArg.getType()).getShape().end());
            SmallVector<int64_t> strides(
                candidate.concat.getType().getShape().size(), 1);

            lowerEnds[candidate.concat.getDimension()] = candidate.lhsSize;

            upperStarts[candidate.concat.getDimension()] =
                upperEnds[candidate.concat.getDimension()] - candidate.rhsSize;

            Value ops[3] = {
                rewriter.create<stablehlo::SliceOp>(candidate.concat.getLoc(),
                                                    newArg, lowerStarts,
                                                    lowerEnds, strides),
                newArg,
                rewriter.create<stablehlo::SliceOp>(candidate.concat.getLoc(),
                                                    newArg, upperStarts,
                                                    upperEnds, strides),

            };
            newArg = rewriter.create<stablehlo::ConcatenateOp>(
                candidate.concat.getLoc(), ops,
                candidate.concat.getDimension());
            break;
          }
        }
        mapper.map(oldArg, newArg);
      }

      for (auto &op : oldBodyBlock.getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
    }

    // Create a new terminator for the body region using new values
    {
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> newReturnValues;

      // Map old return values to new values using the mapper
      for (auto oldRetVal : yieldOp.getOperands()) {
        Value newRetVal = mapper.lookupOrNull(oldRetVal);
        // If the value isn't in the mapper, maybe it was a block argument or
        // constant
        if (!newRetVal)
          newRetVal = oldRetVal; // Consider more robust handling if needed
        newReturnValues.push_back(newRetVal);
      }

      // Create the return op at the end of the body
      rewriter.setInsertionPointToEnd(&newBodyBlock);
      rewriter.create<stablehlo::ReturnOp>(yieldOp.getLoc(), newReturnValues);
    }

    // Create condition region mapper
    mlir::IRMapping condMapper;

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    Block &oldCondBlock = whileOp.getCond().front();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        auto oldArg = whileOp.getCond().getArgument(i);
        Value newArg = newWhileOp.getCond().getArgument(i);
        for (auto &candidate : candidates) {
          if (candidate.idx == i) {
            SmallVector<int64_t> lowerStarts(
                candidate.concat.getType().getShape().size(), 0);
            SmallVector<int64_t> upperStarts(
                candidate.concat.getType().getShape().size(), 0);
            SmallVector<int64_t> lowerEnds(
                cast<RankedTensorType>(newArg.getType()).getShape().begin(),
                cast<RankedTensorType>(newArg.getType()).getShape().end());
            SmallVector<int64_t> upperEnds(
                cast<RankedTensorType>(newArg.getType()).getShape().begin(),
                cast<RankedTensorType>(newArg.getType()).getShape().end());
            SmallVector<int64_t> strides(
                candidate.concat.getType().getShape().size(), 1);

            lowerEnds[candidate.concat.getDimension()] = candidate.lhsSize;

            upperStarts[candidate.concat.getDimension()] =
                upperEnds[candidate.concat.getDimension()] - candidate.rhsSize;

            Value ops[3] = {
                rewriter.create<stablehlo::SliceOp>(candidate.concat.getLoc(),
                                                    newArg, lowerStarts,
                                                    lowerEnds, strides),
                newArg,
                rewriter.create<stablehlo::SliceOp>(candidate.concat.getLoc(),
                                                    newArg, upperStarts,
                                                    upperEnds, strides),

            };
            newArg = rewriter.create<stablehlo::ConcatenateOp>(
                candidate.concat.getLoc(), ops,
                candidate.concat.getDimension());
            break;
          }
        }
        condMapper.map(oldArg, newArg);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        rewriter.clone(op, condMapper);
      }
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);

    return success();
  }
};

// TODO: this is not valid in general but presumes the inner structure is valid
// from the input
template <typename T>
struct WhileWrap
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileWrap<T>> {
  using CheckedOpRewritePattern<stablehlo::WhileOp,
                                WhileWrap<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp whileOp,
                                    PatternRewriter &rewriter) const {
    // Find yield op in the body
    auto &bodyBlock = whileOp.getBody().front();
    auto yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());

    // Step 1: Track which results need to be transformed
    struct Candidate {
      unsigned idx;
      T concat;
    };

    llvm::SmallVector<Candidate, 4> candidates;

    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {

      auto concat = yieldOp.getOperand(idx).getDefiningOp<T>();

      // Check that the while result has exactly one use
      if (!concat)
        continue;

      candidates.emplace_back(Candidate{idx, concat});
    }

    // If no candidates found, no rewrite needed
    if (candidates.empty())
      return failure();

    // Step 2 : Make transformations in the original while op
    // Get the operands of the while op to use later
    auto whileOperands = llvm::to_vector(whileOp.getOperands());

    // New operands
    SmallVector<Value> newOperands(whileOp.getOperands().begin(),
                                   whileOp.getOperands().end());

    // Create input transposes for each candidate
    for (auto &candidate : candidates) {
      // Create a new transpose before the while loop
      SmallVector<int64_t> starts(candidate.concat.getType().getShape().size(),
                                  0);
      SmallVector<int64_t> steps(candidate.concat.getType().getShape().size(),
                                 1);
      SmallVector<int64_t> limits = llvm::to_vector(
          cast<RankedTensorType>(whileOp.getOperands()[candidate.idx].getType())
              .getShape());
      starts[candidate.concat.getDimension()] += candidate.concat.getLhs();
      limits[candidate.concat.getDimension()] -= candidate.concat.getRhs();

      newOperands[candidate.idx] = rewriter.create<stablehlo::SliceOp>(
          candidate.concat[1].getLoc(), whileOp.getOperands()[candidate.idx],
          starts, limits, steps);
    }

    // Update yield op to use the input of the inner transpose
    {
      // Save the current insertion point
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // New return values
      SmallVector<Value> newReturnValues(yieldOp.getOperands().begin(),
                                         yieldOp.getOperands().end());

      rewriter.setInsertionPoint(yieldOp);
      for (auto &candidate : candidates) {
        newReturnValues[candidate.idx] = candidate.concat.getOperand();
      }
      rewriter.replaceOpWithNewOp<stablehlo::ReturnOp>(yieldOp,
                                                       newReturnValues);
      // Update the yieldOp
      yieldOp = cast<stablehlo::ReturnOp>(bodyBlock.getTerminator());
    }

    // Step 3 : Create a new while op with the new operands and move the body of
    // original whileOp
    SmallVector<Type> newResultTypes;
    newResultTypes.reserve(newOperands.size());

    for (auto operand : newOperands) {
      newResultTypes.push_back(operand.getType());
    }
    auto newWhileOp = rewriter.create<stablehlo::WhileOp>(
        whileOp.getLoc(), newResultTypes, newOperands, whileOp->getAttrs());

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    {
      for (auto &candidate : candidates) {
        results[candidate.idx] = rewriter.create<T>(
            candidate.concat.getLoc(), results[candidate.idx],
            candidate.concat.getLhs(), candidate.concat.getRhs(),
            candidate.concat.getDimension());
      }
    }

    // Create blocks in both regions first
    {
      // Create a block in the condition region
      Block *condBlock = rewriter.createBlock(&newWhileOp.getCond());

      // Add arguments to the condition block matching operand types
      for (auto type : newResultTypes) {
        condBlock->addArgument(type, whileOp.getLoc());
      }

      // Create a block in the body region
      Block *bodyBlock = rewriter.createBlock(&newWhileOp.getBody());

      // Add arguments to the body block matching operand types
      for (auto type : newResultTypes) {
        bodyBlock->addArgument(type, whileOp.getLoc());
      }
    }

    // Create an IR mapper to map values from old op to new op
    mlir::IRMapping mapper;

    // Clear the new body block but keep its arguments
    Block &newBodyBlock = newWhileOp.getBody().front();
    newBodyBlock
        .clear(); // This clears operations but preserves block arguments

    // Clone operations from old body to new body
    Block &oldBodyBlock = whileOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBodyBlock);

      // Set up operand mapping for the body region
      for (unsigned i = 0; i < whileOp.getBody().getNumArguments(); ++i) {
        auto oldArg = whileOp.getBody().getArgument(i);
        Value newArg = newWhileOp.getBody().getArgument(i);
        for (auto &candidate : candidates) {
          if (candidate.idx == i) {
            newArg = rewriter.create<T>(
                candidate.concat.getLoc(), newArg, candidate.concat.getLhs(),
                candidate.concat.getRhs(), candidate.concat.getDimension());
            break;
          }
        }
        mapper.map(oldArg, newArg);
      }

      for (auto &op : oldBodyBlock.getOperations()) {
        // Skip the terminator - we'll add it after all other operations
        if (isa<stablehlo::ReturnOp>(op))
          continue;

        // Clone the operation with the value mapping
        rewriter.clone(op, mapper);
      }
    }

    // Create a new terminator for the body region using new values
    {
      OpBuilder::InsertionGuard guard(rewriter);
      SmallVector<Value> newReturnValues;

      // Map old return values to new values using the mapper
      for (auto oldRetVal : yieldOp.getOperands()) {
        Value newRetVal = mapper.lookupOrNull(oldRetVal);
        // If the value isn't in the mapper, maybe it was a block argument or
        // constant
        if (!newRetVal)
          newRetVal = oldRetVal; // Consider more robust handling if needed
        newReturnValues.push_back(newRetVal);
      }

      // Create the return op at the end of the body
      rewriter.setInsertionPointToEnd(&newBodyBlock);
      rewriter.create<stablehlo::ReturnOp>(yieldOp.getLoc(), newReturnValues);
    }

    // Create condition region mapper
    mlir::IRMapping condMapper;

    // Clear and clone condition region
    Block &newCondBlock = newWhileOp.getCond().front();
    newCondBlock.clear();
    Block &oldCondBlock = whileOp.getCond().front();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newCondBlock);

      for (unsigned i = 0; i < whileOp.getCond().getNumArguments(); ++i) {
        auto oldArg = whileOp.getCond().getArgument(i);
        Value newArg = newWhileOp.getCond().getArgument(i);
        for (auto &candidate : candidates) {
          if (candidate.idx == i) {
            newArg = rewriter.create<T>(
                candidate.concat.getLoc(), newArg, candidate.concat.getLhs(),
                candidate.concat.getRhs(), candidate.concat.getDimension());
            break;
          }
        }
        condMapper.map(oldArg, newArg);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        rewriter.clone(op, condMapper);
      }
    }

    // Finally, replace all uses of the old while op with the new one
    rewriter.replaceOp(whileOp, results);

    return success();
  }
};

// Replace while op iteration variables which are not updated with their
// upcoming value
struct WhileSimplify
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  bool hoist_all;
  WhileSimplify(bool hoist_all, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        hoist_all(hoist_all) {}

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp op,
                                    PatternRewriter &rewriter) const {
    SmallVector<unsigned> operands;

    Block *cond = &op.getCond().front(), *body = &op.getBody().front();
    Operation *bodyTerm = body->getTerminator();

    int deleted = 0;

    for (auto &opOperand : op->getOpOperands()) {
      Value inputValue = opOperand.get();

      auto i = opOperand.getOperandNumber() - deleted;
      Value bodyArg = body->getArgument(i);
      Value condArg = cond->getArgument(i);

      bool canHoist = inputValue.getDefiningOp<stablehlo::ConstantOp>();
      if (auto BA = dyn_cast<BlockArgument>(inputValue)) {
        canHoist |= isa<FunctionOpInterface>(BA.getOwner()->getParentOp());
      } else if (hoist_all) {
        canHoist = true;
      }

      Value bodyRes = bodyTerm->getOperand(i);

      if (canHoist && bodyArg == bodyRes) {
        // This variable is not updated during iterations
        rewriter.replaceAllUsesWith(bodyArg, inputValue);
        rewriter.replaceAllUsesWith(condArg, inputValue);
        rewriter.modifyOpInPlace(bodyTerm,
                                 [&] { bodyTerm->setOperands(i, 1, {}); });
        rewriter.replaceAllUsesWith(op.getResult(opOperand.getOperandNumber()),
                                    inputValue);

        body->eraseArgument(i);
        cond->eraseArgument(i);

        deleted++;
      } else {
        operands.push_back(opOperand.getOperandNumber());
      }
    }

    if (operands.size() == op->getNumOperands())
      return failure();

    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());

    for (auto opOperand : operands) {
      newOperands.push_back(op->getOperand(opOperand));
    }

    auto newWhile = rewriter.create<stablehlo::WhileOp>(
        op.getLoc(), newOperands, op->getAttrs());
    newWhile.getCond().takeBody(op.getCond());
    newWhile.getBody().takeBody(op.getBody());

    // Replace uses for remaining results.
    for (const auto &it : llvm::enumerate(operands)) {
      Value oldRes = op->getResult(it.value());
      Value newRes = newWhile->getResult(it.index());

      rewriter.replaceAllUsesWith(oldRes, newRes);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

// Replace while op iteration variables which are not updated with their
// upcoming value
struct WhileLICM
    : public CheckedOpRewritePattern<stablehlo::WhileOp, WhileLICM> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  bool hoist_all;
  WhileLICM(bool hoist_all, MLIRContext *context, PatternBenefit benefit = 1,
            ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        hoist_all(hoist_all) {}

  LogicalResult matchAndRewriteImpl(stablehlo::WhileOp op,
                                    PatternRewriter &rewriter) const {
    SmallVector<unsigned> operands;

    Block *cond = &op.getCond().front(), *body = &op.getBody().front();
    Operation *bodyTerm = body->getTerminator();

    int deleted = 0;

    // Find the index of IV and the step to check for 1 iteration
    auto ivInfo = extractSimpleIVInfo(op);

    for (auto &opOperand : op->getOpOperands()) {
      Value inputValue = opOperand.get();

      auto i = opOperand.getOperandNumber() - deleted;
      Value bodyArg = body->getArgument(i);
      Value condArg = cond->getArgument(i);

      bool canHoist = inputValue.getDefiningOp<stablehlo::ConstantOp>();
      if (auto BA = dyn_cast<BlockArgument>(inputValue)) {
        canHoist |= isa<FunctionOpInterface>(BA.getOwner()->getParentOp());
      } else if (hoist_all) {
        canHoist = true;
      }

      Value bodyRes = bodyTerm->getOperand(i);

      if (canHoist && definedOutside(bodyRes, op) && ivInfo.isValid &&
          ivInfo.step != 0) {

        Value resultReplacement;
        {
          OpBuilder::InsertionGuard guard(rewriter);
          IRMapping mapper;

          for (unsigned i = 0; i < op.getCond().getNumArguments(); ++i) {
            mapper.map(op.getCond().getArgument(i), op.getOperands()[i]);
          }
          for (auto &op : op.getCond().front().getOperations()) {
            // Skip the terminator - we'll add it after all other operations
            if (isa<stablehlo::ReturnOp>(op))
              continue;

            // Clone the operation with the value mapping
            rewriter.clone(op, mapper);
          }
          Value useInner = op.getCond().front().getTerminator()->getOperand(0);
          useInner = mapper.lookupOrDefault(useInner);

          resultReplacement = rewriter.create<stablehlo::SelectOp>(
              op.getLoc(), useInner, bodyRes, inputValue);
        }

        // This variable is not updated during iterations
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&op.getBody().front());
          auto replacement = rewriter.create<stablehlo::SelectOp>(
              op.getLoc(),
              rewriter.create<stablehlo::CompareOp>(
                  op.getLoc(), op.getBody().getArgument(ivInfo.index),
                  ivInfo.start, stablehlo::ComparisonDirection::EQ),
              inputValue, bodyRes);
          rewriter.replaceAllUsesWith(bodyArg, replacement);
        }

        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&op.getCond().front());
          auto replacement = rewriter.create<stablehlo::SelectOp>(
              op.getLoc(),
              rewriter.create<stablehlo::CompareOp>(
                  op.getLoc(), op.getCond().getArgument(ivInfo.index),
                  ivInfo.start, stablehlo::ComparisonDirection::EQ),
              inputValue, bodyRes);
          rewriter.replaceAllUsesWith(condArg, replacement);
        }

        rewriter.modifyOpInPlace(bodyTerm,
                                 [&] { bodyTerm->setOperands(i, 1, {}); });

        rewriter.replaceAllUsesWith(op.getResult(opOperand.getOperandNumber()),
                                    resultReplacement);

        body->eraseArgument(i);
        cond->eraseArgument(i);

        deleted++;
      } else {
        operands.push_back(opOperand.getOperandNumber());
      }
    }

    if (operands.size() == op->getNumOperands())
      return failure();

    SmallVector<Value> newOperands;
    newOperands.reserve(operands.size());

    for (auto opOperand : operands) {
      newOperands.push_back(op->getOperand(opOperand));
    }

    auto newWhile = rewriter.create<stablehlo::WhileOp>(
        op.getLoc(), newOperands, op->getAttrs());
    newWhile.getCond().takeBody(op.getCond());
    newWhile.getBody().takeBody(op.getBody());

    // Replace uses for remaining results.
    for (const auto &it : llvm::enumerate(operands)) {
      Value oldRes = op->getResult(it.value());
      Value newRes = newWhile->getResult(it.index());

      rewriter.replaceAllUsesWith(oldRes, newRes);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct DynamicGatherOpIsNotDynamic
    : public CheckedOpRewritePattern<stablehlo::DynamicGatherOp,
                                     DynamicGatherOpIsNotDynamic> {
  using CheckedOpRewritePattern<
      stablehlo::DynamicGatherOp,
      DynamicGatherOpIsNotDynamic>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::DynamicGatherOp op,
                                    PatternRewriter &rewriter) const {
    // Check if slice sizes are constant.
    DenseIntElementsAttr sliceSizesAttr;
    if (!matchPattern(op.getSliceSizes(), m_Constant(&sliceSizesAttr))) {
      return failure();
    }

    // dynamic_gather allows non-int64 slice sizes, but we need to convert them
    // to int64 for the gather.
    if (!sliceSizesAttr.getType().getElementType().isInteger(64)) {
      SmallVector<APInt> sliceSizes;
      for (auto size : sliceSizesAttr.getValues<APInt>()) {
        sliceSizes.push_back(size);
      }
      auto newSliceSizesAttr = DenseElementsAttr::get(
          RankedTensorType::get(sliceSizesAttr.getType().getShape(),
                                rewriter.getIntegerType(64)),
          sliceSizes);
      sliceSizesAttr = cast<DenseIntElementsAttr>(newSliceSizesAttr);
    }

    SmallVector<int64_t> sliceSizes;
    for (auto size : sliceSizesAttr.getValues<int64_t>()) {
      sliceSizes.push_back(size);
    }
    auto sliceSizesArrayAttr =
        DenseI64ArrayAttr::get(op.getContext(), sliceSizes);

    rewriter.replaceOpWithNewOp<stablehlo::GatherOp>(
        op, op.getType(), op.getOperand(), op.getStartIndices(),
        stablehlo::GatherDimensionNumbersAttr::get(
            op.getContext(), op.getDimensionNumbers().getOffsetDims(),
            op.getDimensionNumbers().getCollapsedSliceDims(),
            /*operandBatchingDims=*/{},
            /*startIndicesBatchingDims=*/{},
            op.getDimensionNumbers().getStartIndexMap(),
            op.getDimensionNumbers().getIndexVectorDim()),
        sliceSizesArrayAttr);

    return success();
  }
};

/// Check if a `t` is a tensor with zero extents.
static std::optional<RankedTensorType> isZeroExtent(Type t) {
  auto type = dyn_cast<RankedTensorType>(t);
  if (type && type.hasStaticShape() && type.getNumElements() == 0)
    return type;
  return std::nullopt;
}

// Replace instances of zero extent tensors with empty tensors of the same
// type.
struct ZeroExtentTensorCanon final : RewritePattern {
  ZeroExtentTensorCanon(MLIRContext *context,
                        PatternBenefit benefit = PatternBenefit(1))
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const {
    auto disabledByAttr =
        failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter);
    if (disabledByAttr.failed())
      return disabledByAttr;

    auto loc = op->getLoc();

    if (!isa_and_present<stablehlo::StablehloDialect>(op->getDialect()))
      return rewriter.notifyMatchFailure(op, "not stablehlo");

    // If the result is a zero-extent tensor, replace the whole op with an empty
    // tensor.
    bool didUpdate = false;
    for (auto result : op->getResults()) {
      auto resultType = isZeroExtent(result.getType());
      if (!resultType || result.use_empty())
        continue;
      rewriter.replaceAllUsesWith(result, rewriter.create<tensor::EmptyOp>(
                                              loc, resultType->getShape(),
                                              resultType->getElementType()));
      didUpdate = true;
    }

    // If one of the operands is a zero-extent tensor, replace the operand with
    // an empty tensor.
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = isZeroExtent(operand.get().getType());
      if (!operandType || operand.get().getDefiningOp<tensor::EmptyOp>())
        continue;
      Operation *owner = operand.getOwner();
      int operandNum = operand.getOperandNumber();
      auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, operandType->getShape(), operandType->getElementType());
      rewriter.modifyOpInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyTensorOp); });
      didUpdate = true;
    }
    return success(didUpdate);
  }
};

struct ReorderElementwiseAndShapeOp final
    : CheckedOpTraitRewritePattern<OpTrait::Elementwise,
                                   ReorderElementwiseAndShapeOp> {
  using CheckedOpTraitRewritePattern::CheckedOpTraitRewritePattern;

  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const {
    if (op->getOperands().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected to be unary");

    auto definingOp = op->getOperand(0).getDefiningOp();
    if (!definingOp)
      return rewriter.notifyMatchFailure(
          op, "expected to have an op before elementise op");

    if (!isa<stablehlo::ReshapeOp>(definingOp) &&
        !isa<stablehlo::TransposeOp>(definingOp) &&
        !isa<stablehlo::BroadcastOp>(definingOp))
      return rewriter.notifyMatchFailure(
          op, "defining operation of unexpected type");

    // Only reorder if the defining op has no other uses.
    if (!llvm::hasSingleElement(definingOp->getResult(0).getUses()))
      return rewriter.notifyMatchFailure(op, "operation has more than one use");

    Value input = definingOp->getOperand(0);
    Value result = op->getResult(0);
    auto intermediateType = cast<ShapedType>(input.getType())
                                .clone(getElementTypeOrSelf(result.getType()));

    // Reorder the operation and rewire the inputs/outputs.
    op->moveBefore(definingOp);
    definingOp->getResult(0).setType(result.getType());
    rewriter.replaceAllUsesWith(result, definingOp->getResult(0));
    result.setType(intermediateType);
    op->setOperands(input);
    definingOp->setOperands(result);
    return success();
  }
};

struct NoNanMulSimplify final
    : public CheckedOpRewritePattern<stablehlo::MulOp, NoNanMulSimplify> {
  using CheckedOpRewritePattern<stablehlo::MulOp,
                                NoNanMulSimplify>::CheckedOpRewritePattern;

  NoNanMulSimplify(bool allowOnFloatingPointMath, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : CheckedOpRewritePattern(context, benefit),
        allowOnFloatingPointMath(allowOnFloatingPointMath) {}

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    if (!canApplyNoNanPattern(allowOnFloatingPointMath,
                              op.getResult().getType(),
                              op.getOperand(0).getType())) {
      return failure();
    }

    // 0 * x -> 0
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero()) ||
        matchPattern(op.getLhs(), m_AnyZeroComplex())) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, cast<ElementsAttr>(makeAttr(op.getType(), 0)));
      return success();
    }
    // x * 0 -> 0
    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero()) ||
        matchPattern(op.getRhs(), m_AnyZeroComplex())) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, cast<ElementsAttr>(makeAttr(op.getType(), 0)));
      return success();
    }

    return failure();
  }

private:
  bool allowOnFloatingPointMath;
};

// c = a + b; d = c - b => d = a
// c = a + b; d = b - c => d = -a
struct NoNanAddSubSimplify final
    : public CheckedOpRewritePattern<stablehlo::SubtractOp,
                                     NoNanAddSubSimplify> {
  using CheckedOpRewritePattern<stablehlo::SubtractOp,
                                NoNanAddSubSimplify>::CheckedOpRewritePattern;

  NoNanAddSubSimplify(bool allowOnFloatingPointMath, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : CheckedOpRewritePattern(context, benefit),
        allowOnFloatingPointMath(allowOnFloatingPointMath) {}

  LogicalResult matchAndRewriteImpl(stablehlo::SubtractOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto subOutTy = op.getResult().getType();

    // Check if LHS is defined by an AddOp
    if (auto lhsAddOp = lhs.getDefiningOp<stablehlo::AddOp>()) {
      auto addOutTy = lhsAddOp.getResult().getType();
      if (!canApplyNoNanPattern(allowOnFloatingPointMath, addOutTy, subOutTy))
        return failure();

      // Case: c = a + b; d = c - b -> d = a
      if (lhsAddOp.getRhs() == rhs) {
        rewriter.replaceOp(op, lhsAddOp.getLhs());
        return success();
      }

      // Case: c = a + b; d = c - a -> d = b
      if (lhsAddOp.getLhs() == rhs) {
        rewriter.replaceOp(op, lhsAddOp.getRhs());
        return success();
      }
    }

    // Check if RHS is defined by an AddOp
    if (auto rhsAddOp = rhs.getDefiningOp<stablehlo::AddOp>()) {
      auto addOutTy = rhsAddOp.getResult().getType();
      if (!canApplyNoNanPattern(allowOnFloatingPointMath, addOutTy, subOutTy))
        return failure();

      // Case: c = a + b; d = b - c -> d = -a
      if (rhsAddOp.getLhs() == lhs) {
        rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, rhsAddOp.getRhs());
        return success();
      }

      // Case: c = a + b; d = a - c -> d = -b
      if (rhsAddOp.getRhs() == lhs) {
        rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, rhsAddOp.getLhs());
        return success();
      }
    }

    // No simplification pattern matched
    return failure();
  }

private:
  bool allowOnFloatingPointMath = false;
};

// a > b ? a : b or a >= b ? a : b ---> maximum(a, b)
// a < b ? a : b or a <= b ? a : b ---> minimum(a, b)
struct CompareSelectSimplify
    : public CheckedOpRewritePattern<stablehlo::SelectOp,
                                     CompareSelectSimplify> {
  using CheckedOpRewritePattern<stablehlo::SelectOp,
                                CompareSelectSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp op,
                                    PatternRewriter &rewriter) const {
    auto compOp = op.getPred().getDefiningOp<stablehlo::CompareOp>();
    if (!compOp)
      return failure();

    auto selectlhs = op.getOnTrue();
    auto selectrhs = op.getOnFalse();

    auto complhs = compOp.getLhs();
    auto comprhs = compOp.getRhs();

    if ((compOp.getComparisonDirection() ==
         stablehlo::ComparisonDirection::GT) ||
        (compOp.getComparisonDirection() ==
         stablehlo::ComparisonDirection::GE)) {
      // select(a > b || a >= b, a, b)
      if (complhs == selectlhs && comprhs == selectrhs) {
        rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, selectlhs, selectrhs);
        return success();
      }
      // select(a > b || a >= b, b, a)
      if (complhs == selectrhs && comprhs == selectlhs) {
        rewriter.replaceOpWithNewOp<stablehlo::MinOp>(op, selectlhs, selectrhs);
        return success();
      }
    }

    if ((compOp.getComparisonDirection() ==
         stablehlo::ComparisonDirection::LT) ||
        (compOp.getComparisonDirection() ==
         stablehlo::ComparisonDirection::LE)) {
      // select(a < b || a <= b, a, b)
      if (complhs == selectlhs && comprhs == selectrhs) {
        rewriter.replaceOpWithNewOp<stablehlo::MinOp>(op, selectlhs, selectrhs);
        return success();
      }
      // select(a < b || a <= b, b, a)
      if (complhs == selectrhs && comprhs == selectlhs) {
        rewriter.replaceOpWithNewOp<stablehlo::MaxOp>(op, selectlhs, selectrhs);
        return success();
      }
    }

    return failure();
  }
};

// select(!op, lhs, rhs) --> select(op, rhs, lhs)
struct NotSelectSimplify
    : public CheckedOpRewritePattern<stablehlo::SelectOp, NotSelectSimplify> {
  using CheckedOpRewritePattern<stablehlo::SelectOp,
                                NotSelectSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp op,
                                    PatternRewriter &rewriter) const {
    auto notOp = op.getPred().getDefiningOp<stablehlo::NotOp>();
    if (!notOp)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        op, notOp.getOperand(), op.getOnFalse(), op.getOnTrue());
    return success();
  }
};

struct NotCompare
    : public CheckedOpRewritePattern<stablehlo::NotOp, NotCompare> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::NotOp op,
                                    PatternRewriter &rewriter) const {
    auto cmp = op.getOperand().getDefiningOp<stablehlo::CompareOp>();
    if (!cmp)
      return failure();

    if (!llvm::hasSingleElement(cmp->getUsers()))
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::CompareOp>(
        op, cmp.getLhs(), cmp.getRhs(),
        negatedComparisonDirection(cmp.getComparisonDirection()));

    rewriter.eraseOp(cmp);

    return success();
  }
};

struct CommonCompareExpressionRewrite
    : public CheckedOpRewritePattern<stablehlo::CompareOp,
                                     CommonCompareExpressionRewrite> {
  using CheckedOpRewritePattern<
      stablehlo::CompareOp,
      CommonCompareExpressionRewrite>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto negDir = negatedComparisonDirection(op.getComparisonDirection());

    // Check for equivalent users for the value with the fewest other users to
    // check. For ease here we simplify just checking if not constant (since
    // getNumUsers is O(n)).
    auto userCheck = lhs;
    if (matchPattern(userCheck, m_Constant()))
      userCheck = rhs;

    for (auto user : userCheck.getUsers()) {
      auto userCompareOp = dyn_cast<stablehlo::CompareOp>(user);
      if (!userCompareOp || userCompareOp.getComparisonDirection() != negDir)
        continue;

      if (user->getBlock() != op->getBlock())
        continue;

      if (userCompareOp.getLhs() == lhs && userCompareOp.getRhs() == rhs) {
        if (user->isBeforeInBlock(op)) {
          auto negatedCondition = rewriter.create<stablehlo::NotOp>(
              op.getLoc(), userCompareOp.getResult());
          rewriter.replaceOp(op, negatedCondition);
          return success();
        } else {
          auto negatedCondition = rewriter.create<stablehlo::NotOp>(
              userCompareOp.getLoc(), op.getResult());
          rewriter.replaceOp(user, negatedCondition);
          return success();
        }
      }
    }

    return failure();
  }
};

stablehlo::ComparisonDirection
reorderComparisionDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GE;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GT;
  }
}

struct CompareCleanup
    : public CheckedOpRewritePattern<stablehlo::CompareOp, CompareCleanup> {
  using CheckedOpRewritePattern<stablehlo::CompareOp,
                                CompareCleanup>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::CompareOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();

    if (!cast<RankedTensorType>(lhs.getType())
             .getElementType()
             .isSignlessInteger(64)) {
      return failure();
    }

    DenseIntElementsAttr rhs;
    if (!matchPattern(op.getRhs(), m_Constant(&rhs))) {
      return failure();
    }

    if (!rhs.isSplat())
      return failure();

    auto rhsv = rhs.getSplatValue<IntegerAttr>().getValue().getSExtValue();

    if (auto add = lhs.getDefiningOp<stablehlo::AddOp>()) {
      DenseIntElementsAttr c;
      if (matchPattern(add.getRhs(), m_Constant(&c)) && c.isSplat()) {
        auto cv = c.getSplatValue<IntegerAttr>().getValue().getSExtValue();

        auto off = rewriter.create<stablehlo::ConstantOp>(
            add.getLoc(), add.getType(),
            cast<ElementsAttr>(makeAttr(add.getType(), rhsv - cv)));

        // x + cv ?= rhsv -> x ?= rhs - cv
        rewriter.modifyOpInPlace(op, [&]() {
          op.getLhsMutable().assign(add.getLhs());
          op.getRhsMutable().assign(off);
        });

        return success();
      }
    }

    if (auto mul = lhs.getDefiningOp<stablehlo::MulOp>()) {
      DenseIntElementsAttr c;
      if (matchPattern(mul.getRhs(), m_Constant(&c)) && c.isSplat()) {
        auto cv = c.getSplatValue<IntegerAttr>().getValue().getSExtValue();
        if (cv == -1) {
          auto off = rewriter.create<stablehlo::ConstantOp>(
              mul.getLoc(), mul.getType(),
              cast<ElementsAttr>(makeAttr(mul.getType(), -rhsv)));

          // x * -1 ?= rhsv -> x =? -rhs
          rewriter.modifyOpInPlace(op, [&]() {
            op.getLhsMutable().assign(mul.getLhs());
            op.getRhsMutable().assign(off);
            op.setComparisonDirection(
                reorderComparisionDirection(op.getComparisonDirection()));
          });

          return success();
        }
      }
    }

    return failure();
  }
};

struct ScatterUpdateComputationConstProp
    : public CheckedOpRewritePattern<stablehlo::ScatterOp,
                                     ScatterUpdateComputationConstProp> {
  using CheckedOpRewritePattern<
      stablehlo::ScatterOp,
      ScatterUpdateComputationConstProp>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ScatterOp op,
                                    PatternRewriter &rewriter) const {
    if (!op.getUniqueIndices())
      return failure();

    auto &region = op.getUpdateComputation();
    auto &block = region.front();

    // Check all inputs are constant and splat and their values are the same.
    auto [constInput, inputSplatAttr] =
        isConstantSplatValueRange(op.getInputs());

    // Check all updates are constant and splat and their values are the same.
    auto [constUpdate, updateSplatAttr] =
        isConstantSplatValueRange(op.getUpdates());

    if (constInput || constUpdate) {
      bool inputTransformed = false;
      bool updateTransformed = false;
      auto blockArgInput = block.getArgument(0);
      auto blockArgUpdate = block.getArgument(1);

      if (constInput && !blockArgInput.getUses().empty()) {
        inputTransformed = true;
        auto denseAttr = DenseElementsAttr::get(
            cast<ShapedType>(blockArgInput.getType()), inputSplatAttr);
        auto constInputOp =
            rewriter.create<stablehlo::ConstantOp>(op.getLoc(), denseAttr);
        blockArgInput.replaceAllUsesWith(constInputOp);
      }

      if (constUpdate && !blockArgUpdate.getUses().empty()) {
        updateTransformed = true;
        auto denseAttr = DenseElementsAttr::get(
            cast<ShapedType>(blockArgUpdate.getType()), updateSplatAttr);
        auto constUpdateOp =
            rewriter.create<stablehlo::ConstantOp>(op.getLoc(), denseAttr);
        blockArgUpdate.replaceAllUsesWith(constUpdateOp);
      }

      if (!inputTransformed && !updateTransformed)
        return failure();

      auto newOp = rewriter.create<stablehlo::ScatterOp>(
          op.getLoc(), op.getResultTypes(), op.getInputs(),
          op.getScatterIndices(), op.getUpdates(),
          op.getScatterDimensionNumbers(), op.getIndicesAreSorted(),
          op.getUniqueIndices());
      newOp.getUpdateComputation().takeBody(region);
      rewriter.replaceOp(op, newOp);

      return success();
    }

    return failure();
  }

private:
  std::tuple<bool, Attribute>
  isConstantSplatValueRange(ValueRange range) const {
    Attribute splatAttr = nullptr;
    bool isConstant = true;
    for (auto val : range) {
      DenseElementsAttr attr;
      if (matchPattern(val, m_Constant(&attr))) {
        if (attr.isSplat()) {
          if (!splatAttr) {
            splatAttr = attr.getSplatValue<Attribute>();
            continue;
          } else if (splatAttr != attr.getSplatValue<Attribute>()) {
            isConstant = false;
            break;
          }
        } else {
          isConstant = false;
          break;
        }
      } else {
        isConstant = false;
        break;
      }
    }
    return std::make_tuple(isConstant, splatAttr);
  };
};

struct ScatterIndicesAreUnique
    : public CheckedOpRewritePattern<stablehlo::ScatterOp,
                                     ScatterIndicesAreUnique> {
  using CheckedOpRewritePattern<
      stablehlo::ScatterOp, ScatterIndicesAreUnique>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ScatterOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getUniqueIndices())
      return failure(); // already unique, no need to do anything

    auto scatterIndices = op.getScatterIndices();
    Attribute scatterIndicesAttr;
    if (matchPattern(scatterIndices, m_Constant(&scatterIndicesAttr))) {
      auto denseAttr = dyn_cast<DenseIntElementsAttr>(scatterIndicesAttr);

      auto shape = cast<ShapedType>(scatterIndices.getType()).getShape();
      if (shape.empty())
        return failure();

      auto dimNumbers = op.getScatterDimensionNumbers();
      int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

      int64_t tupleSize = shape[indexVectorDim];

      SmallVector<int64_t> strides(shape.size());
      strides[shape.size() - 1] = 1;
      for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }

      SmallVector<int64_t> nonIndexVectorShape;
      for (int64_t i = 0; i < shape.size(); ++i) {
        if (i != indexVectorDim) {
          nonIndexVectorShape.push_back(shape[i]);
        }
      }

      // Iterate over the scatter indices tensor to extract tuples
      SmallVector<SmallVector<int64_t>> indexTuples;
      auto values = denseAttr.getValues<APInt>();

      std::function<void(SmallVector<int64_t>, int64_t)> extractTuples =
          [&](SmallVector<int64_t> currentIndices, int64_t dim) {
            if (dim == nonIndexVectorShape.size()) {
              SmallVector<int64_t> indexTuple;
              for (int64_t component = 0; component < tupleSize; ++component) {
                // Build full multi-dimensional index
                SmallVector<int64_t> fullIndex;
                int64_t nonIndexDim = 0;
                for (int64_t d = 0; d < shape.size(); ++d) {
                  if (d == indexVectorDim) {
                    fullIndex.push_back(component);
                  } else {
                    fullIndex.push_back(currentIndices[nonIndexDim++]);
                  }
                }

                // Convert to linear index
                int64_t linearIdx = 0;
                for (int64_t d = 0; d < shape.size(); ++d) {
                  linearIdx += fullIndex[d] * strides[d];
                }

                auto it = values.begin();
                std::advance(it, linearIdx);
                indexTuple.push_back((*it).getSExtValue());
              }
              indexTuples.push_back(indexTuple);
              return;
            }

            for (int64_t i = 0; i < nonIndexVectorShape[dim]; ++i) {
              SmallVector<int64_t> newIndices = currentIndices;
              newIndices.push_back(i);
              extractTuples(newIndices, dim + 1);
            }
          };

      extractTuples({}, 0);

      bool uniqueIndices = areIndexTuplesUnique(indexTuples);
      if (!uniqueIndices && !op.getUniqueIndices())
        return failure();
      auto newOp = rewriter.create<stablehlo::ScatterOp>(
          op.getLoc(), op.getResultTypes(), op.getInputs(), scatterIndices,
          op.getUpdates(), dimNumbers, op.getIndicesAreSortedAttr(),
          rewriter.getBoolAttr(uniqueIndices));
      newOp.getUpdateComputation().takeBody(op.getUpdateComputation());
      rewriter.replaceOp(op, newOp);
      return success();
    }

    return failure();
  }

private:
  bool areIndexTuplesUnique(
      const SmallVector<SmallVector<int64_t>> &indexTuples) const {
    std::set<SmallVector<int64_t>> uniqueSet;
    for (const auto &tuple : indexTuples) {
      if (!uniqueSet.insert(tuple).second) {
        return false; // Duplicate found
      }
    }
    return true;
  }
};

// (add (mul a x) (mul a y)) -> (mul a (add x y))
template <typename Op>
struct AssociativeCommonMulOpReordering final
    : public CheckedOpRewritePattern<Op, AssociativeCommonMulOpReordering<Op>> {
  using CheckedOpRewritePattern<
      Op, AssociativeCommonMulOpReordering<Op>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(Op op, PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsMul = lhs.template getDefiningOp<stablehlo::MulOp>();
    auto rhsMul = rhs.template getDefiningOp<stablehlo::MulOp>();

    if (!lhsMul || !rhsMul)
      return failure();

    Value common = nullptr, lhsVal, rhsVal;

    if (lhsMul.getLhs() == rhsMul.getRhs()) {
      common = lhsMul.getLhs();
      rhsVal = rhsMul.getLhs();
      lhsVal = lhsMul.getRhs();
    }

    if (lhsMul.getLhs() == rhsMul.getLhs()) {
      common = lhsMul.getLhs();
      rhsVal = rhsMul.getRhs();
      lhsVal = lhsMul.getRhs();
    }

    if (lhsMul.getRhs() == rhsMul.getRhs()) {
      common = lhsMul.getRhs();
      rhsVal = rhsMul.getLhs();
      lhsVal = lhsMul.getLhs();
    }

    if (lhsMul.getRhs() == rhsMul.getLhs()) {
      common = lhsMul.getRhs();
      rhsVal = rhsMul.getRhs();
      lhsVal = lhsMul.getLhs();
    }

    if (!common)
      return failure();

    auto newMul = rewriter.create<Op>(op.getLoc(), lhsVal, rhsVal);
    rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, common,
                                                  newMul.getResult());

    return success();
  }
};

// This lets us reorder the following
// Case 1: (op x (op (op y x) y)) -> (op (op x y) (op x y))
// Case 2: (op x (op (op x y) y)) -> (op (op x y) (op x y))
// Case 3: (op x (op y (op x y))) -> (op (op x y) (op x y))
// Case 4: (op x (op y (op y x))) -> (op (op x y) (op x y))
template <typename Op>
struct AssociativeBinaryOpReordering
    : public CheckedOpRewritePattern<Op, AssociativeBinaryOpReordering<Op>> {
  using CheckedOpRewritePattern<
      Op, AssociativeBinaryOpReordering<Op>>::CheckedOpRewritePattern;

  // TODO: generalize to the case where lhs is an Op
  LogicalResult matchAndRewriteImpl(Op op, PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhsOp = op.getRhs().template getDefiningOp<Op>();
    if (!rhsOp)
      return failure();

    auto rhslhs = rhsOp.getLhs();
    auto rhsrhs = rhsOp.getRhs();

    auto rhslhsOp = rhslhs.template getDefiningOp<Op>();
    if (rhslhsOp) {
      auto rhslhslhs = rhslhsOp.getLhs();
      auto rhslhsrhs = rhslhsOp.getRhs();

      // Case 1
      if (lhs == rhslhsrhs && rhslhslhs == rhsrhs) {
        rewriter.replaceOpWithNewOp<Op>(op, rhslhsOp.getResult(),
                                        rhslhsOp.getResult());
        return success();
      }

      // Case 2
      if (lhs == rhslhslhs && rhslhsrhs == rhsrhs) {
        rewriter.replaceOpWithNewOp<Op>(op, rhslhsOp.getResult(),
                                        rhslhsOp.getResult());
        return success();
      }
    }

    auto rhsrhsOp = rhsrhs.template getDefiningOp<Op>();
    if (rhsrhsOp) {
      auto rhsrhslhs = rhsrhsOp.getLhs();
      auto rhsrhsrhs = rhsrhsOp.getRhs();

      // Case 3
      if (lhs == rhsrhslhs && rhslhs == rhsrhsrhs) {
        rewriter.replaceOpWithNewOp<Op>(op, rhsrhsOp.getResult(),
                                        rhsrhsOp.getResult());
        return success();
      }

      // Case 4
      if (lhs == rhsrhsrhs && rhslhs == rhsrhslhs) {
        rewriter.replaceOpWithNewOp<Op>(op, rhsrhsOp.getResult(),
                                        rhsrhsOp.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct ReduceTransposeSimplify
    : public CheckedOpRewritePattern<stablehlo::ReduceOp,
                                     ReduceTransposeSimplify> {
  using CheckedOpRewritePattern<
      stablehlo::ReduceOp, ReduceTransposeSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1) // TODO: support for multiple inputs
      return failure();

    auto input = op.getInputs()[0];
    auto transposeOp = input.getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    auto transposeInput = transposeOp.getOperand();
    auto transposeInputType = cast<ShapedType>(transposeInput.getType());
    auto transposePermutation = transposeOp.getPermutation();
    auto reduceDimensions = op.getDimensions();

    SmallVector<int64_t> newReduceDimensions;
    for (auto dim : reduceDimensions) {
      int64_t newDim = transposePermutation[dim];
      newReduceDimensions.push_back(newDim);
    }

    // Calculate result shape after reduction
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < transposeInputType.getRank(); ++i) {
      if (!llvm::is_contained(newReduceDimensions, i)) {
        resultShape.push_back(transposeInputType.getDimSize(i));
      }
    }
    auto elementType = cast<ShapedType>(op.getType(0)).getElementType();
    auto newResultType = RankedTensorType::get(resultShape, elementType);

    // Create a new reduce operation with the adjusted dimensions
    auto newReduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), TypeRange(newResultType), ValueRange(transposeInput),
        op.getInitValues(), newReduceDimensions);
    newReduceOp.getRegion().takeBody(op.getRegion());

    // Map non-reduced dimensions
    SmallVector<bool> isReduced(transposeInputType.getRank(), false);
    for (auto dim : newReduceDimensions) {
      isReduced[dim] = true;
    }

    // Count the number of reduced dimensions before the transpose dim
    SmallVector<int64_t> reducedDimsBeforeTranspose(
        transposePermutation.size());
    reducedDimsBeforeTranspose[0] = isReduced[0];
    for (int64_t i = 1; i < transposePermutation.size(); ++i) {
      reducedDimsBeforeTranspose[i] =
          reducedDimsBeforeTranspose[i - 1] + isReduced[i];
    }

    // Create final permutation
    // original permutation - reduced dimensions before transposed dim
    SmallVector<int64_t> finalPermutation(transposePermutation.size() -
                                          newReduceDimensions.size());
    int64_t j = 0;
    for (int64_t i = 0; i < transposePermutation.size(); ++i) {
      if (!isReduced[transposePermutation[i]]) {
        finalPermutation[j] =
            transposePermutation[i] -
            reducedDimsBeforeTranspose[transposePermutation[i]];
        ++j;
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, newReduceOp.getResult(0), finalPermutation);
    return success();
  }
};

// (mul (sign x) (abs x)) -> x
// (mul (abs x) (sign x)) -> x
struct SignAbsSimplify
    : public CheckedOpRewritePattern<stablehlo::MulOp, SignAbsSimplify> {
  using CheckedOpRewritePattern<stablehlo::MulOp,
                                SignAbsSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);

    auto lhsSignOp = lhs.getDefiningOp<stablehlo::SignOp>();
    if (lhsSignOp) {
      auto rhsAbsOp = rhs.getDefiningOp<stablehlo::AbsOp>();
      if (!rhsAbsOp)
        return failure();

      if (lhsSignOp.getOperand() != rhsAbsOp.getOperand())
        return failure();

      rewriter.replaceOp(op, lhsSignOp.getOperand());
      return success();
    }

    auto rhsSignOp = rhs.getDefiningOp<stablehlo::SignOp>();
    if (rhsSignOp) {
      auto lhsAbsOp = lhs.getDefiningOp<stablehlo::AbsOp>();
      if (!lhsAbsOp)
        return failure();

      if (rhsSignOp.getOperand() != lhsAbsOp.getOperand())
        return failure();

      rewriter.replaceOp(op, rhsSignOp.getOperand());
      return success();
    }

    return failure();
  }
};

bool opResultIsAlwaysNonNegative(Operation *op);

template <typename T>
bool opResultNonNegativeIfAllElementsNonNegative(Operation *op) {
  if (!op)
    return false;

  auto specificOp = dyn_cast<T>(op);
  if (!specificOp)
    return false;

  auto lhsOp = specificOp.getLhs().getDefiningOp();
  auto rhsOp = specificOp.getRhs().getDefiningOp();

  if (lhsOp && rhsOp) {
    bool lhsNonNeg = opResultIsAlwaysNonNegative(lhsOp);
    bool rhsNonNeg = opResultIsAlwaysNonNegative(rhsOp);

    if (lhsNonNeg && rhsNonNeg)
      return true;
  }

  return false;
}

bool isConstantNonNegative(stablehlo::ConstantOp constOp) {
  Attribute attr = constOp.getValue();

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (denseAttr.getType().getShape().size() && denseAttr.isSplat()) {
      denseAttr = denseAttr.resizeSplat(
          RankedTensorType::get({}, denseAttr.getType().getElementType()));
    }
    // For floating point values
    if (denseAttr.getElementType().isF32() ||
        denseAttr.getElementType().isF64()) {
      for (auto element : denseAttr.getValues<APFloat>()) {
        if (element.isNegative())
          return false;
      }
      return true;
    }

    // For integer values
    if (denseAttr.getElementType().isIntOrIndex()) {
      for (auto element : denseAttr.getValues<APInt>()) {
        if (element.isNegative())
          return false;
      }
      return true;
    }
  }

  // Default: can't guarantee all elements are non-negative
  return false;
}

bool opResultIsAlwaysNonNegative(Operation *op) {
  if (!op)
    return false;

  if (isa<stablehlo::AbsOp, stablehlo::SqrtOp, stablehlo::ExpOp,
          stablehlo::IotaOp, stablehlo::AndOp, stablehlo::OrOp>(op))
    return true;

  if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op)) {
    // Constant is non-negative if all its elements are non-negative
    return isConstantNonNegative(constOp);
  }

  // Any non-negative operation that produces a non-negative result
  if (auto maxOp = dyn_cast<stablehlo::MaxOp>(op)) {
    for (auto operand : maxOp.getOperands()) {
      if (auto operandOp = operand.getDefiningOp()) {
        if (opResultIsAlwaysNonNegative(operandOp))
          return true;
      }
    }
  }

  // All non-negative operations that produce a non-negative result
  if (isa<stablehlo::MinOp, stablehlo::AddOp, stablehlo::MulOp>(op)) {
    if (opResultNonNegativeIfAllElementsNonNegative<stablehlo::MinOp>(op) ||
        opResultNonNegativeIfAllElementsNonNegative<stablehlo::AddOp>(op) ||
        opResultNonNegativeIfAllElementsNonNegative<stablehlo::MulOp>(op))
      return true;
  }

  // (mul a a) is always non-negative
  if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    auto lhsOp = mulOp.getLhs().getDefiningOp();
    auto rhsOp = mulOp.getRhs().getDefiningOp();

    if (lhsOp == rhsOp)
      return true;
  }

  if (auto clampOp = dyn_cast<stablehlo::ClampOp>(op)) {
    // Clamp is non-negative if the min operand is non-negative

    if (auto minOp = clampOp.getMin().getDefiningOp()) {
      if (opResultIsAlwaysNonNegative(minOp))
        return true;
    }
  }

  // TODO: For NegOp we need a check for if the operand is guaranteed to be
  // non-positive

  // TODO: Mul of 2 negative values is non-negative

  if (auto selectOp = dyn_cast<stablehlo::SelectOp>(op)) {
    // Select produces non-negative results if both branches produce
    // non-negative results
    auto trueOp = selectOp.getOnTrue().getDefiningOp();
    auto falseOp = selectOp.getOnFalse().getDefiningOp();

    if (trueOp && falseOp) {
      return opResultIsAlwaysNonNegative(trueOp) &&
             opResultIsAlwaysNonNegative(falseOp);
    }
  }

  // These operations preserve values, so result is non-negative if operand is
  // non-negative
  if (isa<stablehlo::ReshapeOp, stablehlo::TransposeOp>(op)) {
    if (auto defOp = op->getOperand(0).getDefiningOp())
      return opResultIsAlwaysNonNegative(defOp);
  }

  // Default: can't guarantee non-negative result
  return false;
}

struct AbsPositiveSimplify
    : public CheckedOpRewritePattern<stablehlo::AbsOp, AbsPositiveSimplify> {
  using CheckedOpRewritePattern<stablehlo::AbsOp,
                                AbsPositiveSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::AbsOp op,
                                    PatternRewriter &rewriter) const {

    auto operand = op.getOperand();
    if (isa<ComplexType>(operand.getType().getElementType()))
      return failure();

    if (opResultIsAlwaysNonNegative(operand.getDefiningOp())) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    return failure();
  }
};

static SmallVector<int64_t>
findReshapeInsertionDims(RankedTensorType inputType,
                         RankedTensorType outputType) {
  if (inputType.getRank() >= outputType.getRank())
    return {}; // trivial no insertion case

  SmallVector<int64_t> insertionDims;
  size_t inputDimIndex = 0;

  for (size_t i = 0; i < outputType.getRank(); ++i) {
    auto dim = outputType.getShape()[i];
    if (inputDimIndex < inputType.getRank() &&
        dim == inputType.getShape()[inputDimIndex]) {
      ++inputDimIndex;
    } else if (dim == 1 && (inputDimIndex >= inputType.getShape().size() ||
                            dim != inputType.getShape()[inputDimIndex])) {
      // Singleton dimension inserted by reshape.
      insertionDims.push_back(i);
    } else {
      // Reshape modifies existing dimensions, which we don't handle here.
      return {};
    }
  }

  // If we haven't seen all of the input dimensions, we don't have a valid
  // insertion point.
  if (inputDimIndex != inputType.getRank())
    return {};

  return insertionDims;
}

struct TransposeReshapeToBroadcast final
    : CheckedOpRewritePattern<stablehlo::TransposeOp,
                              TransposeReshapeToBroadcast> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto reshapeOp = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    RankedTensorType reshapeOpInputType = reshapeOp.getOperand().getType();
    RankedTensorType reshapeOpOutputType = reshapeOp.getResult().getType();

    SmallVector<int64_t> insertionDims =
        findReshapeInsertionDims(reshapeOpInputType, reshapeOpOutputType);

    if (insertionDims.size() != 1) // TODO: support more than one insertion dim
      return failure();

    int64_t insertionDim = insertionDims[0];

    auto permutation = op.getPermutation();
    if (permutation.size() != reshapeOpOutputType.getRank())
      return failure();

    SmallVector<int64_t> broadcastDimensions;
    for (int64_t i = 0; i < reshapeOpInputType.getRank(); ++i) {
      int64_t findIdx = i;
      if (i >= insertionDim)
        ++findIdx;

      auto it = llvm::find(permutation, findIdx);
      if (it == permutation.end()) {
        return failure(); // The index was not found in the permutation
      }
      int64_t outputIdx = std::distance(permutation.begin(), it);
      broadcastDimensions.push_back(outputIdx);
    }

    // Create a single broadcast_in_dim operation to replace the reshape +
    // transpose sequence.
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getResult().getType(), reshapeOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(broadcastDimensions));

    return success();
  }
};

struct ReshapeTransposeToBroadcast final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp,
                              ReshapeTransposeToBroadcast> {
  using CheckedOpRewritePattern<
      stablehlo::ReshapeOp,
      ReshapeTransposeToBroadcast>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto transposeOp = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    RankedTensorType reshapeOpInputType = op.getOperand().getType();
    RankedTensorType reshapeOpOutputType = op.getResult().getType();

    SmallVector<int64_t> insertionDims =
        findReshapeInsertionDims(reshapeOpInputType, reshapeOpOutputType);

    if (insertionDims.size() != 1) // TODO: support more than one deletion dim
      return failure();

    int64_t insertionDim = insertionDims[0];
    auto permutation = transposeOp.getPermutation();

    SmallVector<int64_t> broadcastDimensions;
    for (int64_t i = 0; i < reshapeOpInputType.getRank(); ++i) {
      auto it = llvm::find(permutation, i);
      if (it == permutation.end())
        return failure(); // The index was not found in the permutation
      int64_t dim = std::distance(permutation.begin(), it);

      if (dim >= insertionDim)
        broadcastDimensions.push_back(dim + 1);
      else
        broadcastDimensions.push_back(dim);
    }

    // Create a single broadcast_in_dim operation to replace the reshape +
    // transpose sequence.
    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getResult().getType(), transposeOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(broadcastDimensions));

    return success();
  }
};

struct BroadcastInDimIsReshape final
    : CheckedOpRewritePattern<stablehlo::BroadcastInDimOp,
                              BroadcastInDimIsReshape> {
  using CheckedOpRewritePattern<
      stablehlo::BroadcastInDimOp,
      BroadcastInDimIsReshape>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::BroadcastInDimOp op,
                                    PatternRewriter &rewriter) const {
    auto input = op.getOperand();
    auto outputType = op.getType();
    auto inputType = input.getType();
    auto broadcastDims = op.getBroadcastDimensions();

    size_t inputSize = 1;
    for (auto sz : inputType.getShape())
      inputSize *= sz;
    size_t outputSize = 1;
    for (auto sz : outputType.getShape())
      outputSize *= sz;

    if (inputSize != outputSize)
      return failure();

    SmallVector<int64_t> nonSingletonDims;

    for (size_t i = 0; i < broadcastDims.size(); ++i) {
      int64_t dimIdx = broadcastDims[i];
      if (inputType.getRank() > i && inputType.getDimSize(i) != 1) {
        nonSingletonDims.push_back(dimIdx);
      }
    }

    for (int i = 1, s = nonSingletonDims.size(); i < s; ++i) {
      if (nonSingletonDims[i - 1] > nonSingletonDims[i])
        return failure();
    }

    for (size_t i = 0; i < outputType.getRank(); ++i) {
      int64_t dimIdx = outputType.getDimSize(i);
      if (dimIdx == 1)
        continue;
      auto it = llvm::find(broadcastDims, dimIdx);
      if (it == broadcastDims.end()) {
        return failure();
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, outputType, input);
    return success();
  }
};

struct ReshapeToBroadcast final
    : CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeToBroadcast> {
  using CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                ReshapeToBroadcast>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp reshape,
                                    PatternRewriter &rewriter) const {

    auto inShape = reshape.getOperand().getType().getShape();
    auto outShape = reshape.getResult().getType().getShape();
    if (inShape.size() + 1 != outShape.size())
      return failure();

    if (outShape[0] != 1) {
      return failure();
    }
    for (auto &&[lhs, rhs] : llvm::zip_equal(outShape.slice(1), inShape)) {
      if (lhs != rhs) {
        return failure();
      }
    }
    SmallVector<int64_t> vals(inShape.size(), 0);
    for (int i = 0; i < inShape.size(); i++) {
      vals[i] = i + 1;
    }
    auto shard = sdy::getShardingPerValue(reshape);
    auto rep = rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        reshape, reshape.getType(), reshape.getOperand(), vals);
    if (shard) {
      sdy::setShardings(rep, shard);
    }
    return success();
  }
};

struct PadConcatToConcatPad
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                     PadConcatToConcatPad> {
  using CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                PadConcatToConcatPad>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concatOp,
                                    PatternRewriter &rewriter) const {

    if (concatOp.getNumOperands() <= 1) {
      return failure();
    }

    // Check if all operands are pad ops with the same padding value
    SmallVector<stablehlo::PadOp> padOps;
    Value padValue;

    for (Value operand : concatOp.getOperands()) {
      auto padOp = operand.getDefiningOp<stablehlo::PadOp>();
      if (!padOp)
        return failure();

      if (!padOp->hasOneUse())
        return failure();

      if (padOps.empty()) {
        padValue = padOp.getPaddingValue();
      } else if (padValue != padOp.getPaddingValue()) {
        return failure(); // Different padding values not supported
      }

      padOps.push_back(padOp);
    }

    int64_t concatDim = concatOp.getDimension();
    int64_t rank = padOps[0].getEdgePaddingLow().size();

    if (rank <= 0)
      return failure();

    // Compute smallest common padding for all tensors
    SmallVector<int64_t> commonLowPadding(rank,
                                          std::numeric_limits<int64_t>::max());
    SmallVector<int64_t> commonHighPadding(rank,
                                           std::numeric_limits<int64_t>::max());
    SmallVector<int64_t> interiorPadding(rank, 0);

    // Find minimum padding across all inputs (conservative common padding)
    for (auto padOp : padOps) {
      for (int64_t dim = 0; dim < rank; ++dim) {
        commonLowPadding[dim] =
            std::min(commonLowPadding[dim], padOp.getEdgePaddingLow()[dim]);
        commonHighPadding[dim] =
            std::min(commonHighPadding[dim], padOp.getEdgePaddingHigh()[dim]);
      }
    }

    bool commonPad = false;

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (commonLowPadding[dim] != 0 || commonHighPadding[dim] != 0) {
        commonPad = true;
        break;
      }
    }

    if (!commonPad) {
      return failure();
    }

    // Collect original operands with adjusted padding
    SmallVector<Value> adjOperands;

    for (auto padOp : padOps) {

      SmallVector<int64_t> diffLowPadding(rank);
      SmallVector<int64_t> diffHighPadding(rank);

      for (int64_t dim = 0; dim < rank; ++dim) {
        diffLowPadding[dim] =
            padOp.getEdgePaddingLow()[dim] - commonLowPadding[dim];
        diffHighPadding[dim] =
            padOp.getEdgePaddingHigh()[dim] - commonHighPadding[dim];
      }

      bool needsExtraPad = false;
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (diffLowPadding[dim] > 0 || diffHighPadding[dim] > 0) {
          needsExtraPad = true;
          break;
        }
      }

      if (needsExtraPad) {

        auto adjustedOp = rewriter.create<stablehlo::PadOp>(
            padOp.getLoc(),
            padOp.getOperand(), // we pad the input operand
            padOp.getPaddingValue(), diffLowPadding, diffHighPadding,
            padOp.getInteriorPaddingAttr());

        adjOperands.push_back(adjustedOp);
      } else {
        // No extra padding needed, use original tensor
        adjOperands.push_back(padOp.getOperand());
      }
    }

    auto newConcatOp = rewriter.create<stablehlo::ConcatenateOp>(
        concatOp.getLoc(), adjOperands, concatDim);

    // Apply the common padding to get the final result
    auto result = rewriter.create<stablehlo::PadOp>(
        concatOp.getLoc(), newConcatOp, padValue, commonLowPadding,
        commonHighPadding, interiorPadding);

    rewriter.replaceOp(concatOp, result);
    return success();
  }
};

struct SliceSelect
    : public CheckedOpRewritePattern<stablehlo::SliceOp, SliceSelect> {
  using CheckedOpRewritePattern<stablehlo::SliceOp,
                                SliceSelect>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                    PatternRewriter &rewriter) const {

    auto selOp = sliceOp.getOperand().getDefiningOp<stablehlo::SelectOp>();

    if (!selOp)
      return failure();

    if (!selOp->hasOneUse())
      return failure();

    bool scalar_pred = false;
    Value pred = selOp.getPred();
    Value on_true = selOp.getOnTrue();
    Value on_false = selOp.getOnFalse();

    if (dyn_cast<RankedTensorType>(pred.getType()).getRank() == 0) {
      scalar_pred = true;
    }

    Value slicedPred;
    if (!scalar_pred) {
      // slice predicate
      slicedPred = rewriter.create<stablehlo::SliceOp>(
          sliceOp.getLoc(), pred, sliceOp.getStartIndices(),
          sliceOp.getLimitIndices(), sliceOp.getStrides());
    } else {
      slicedPred = pred;
    }
    Value slicedOnTrue = rewriter.create<stablehlo::SliceOp>(
        sliceOp.getLoc(), on_true, sliceOp.getStartIndices(),
        sliceOp.getLimitIndices(), sliceOp.getStrides());

    Value slicedOnFalse = rewriter.create<stablehlo::SliceOp>(
        sliceOp.getLoc(), on_false, sliceOp.getStartIndices(),
        sliceOp.getLimitIndices(), sliceOp.getStrides());

    auto newSelectOp = rewriter.create<stablehlo::SelectOp>(
        sliceOp.getLoc(), slicedPred, slicedOnTrue, slicedOnFalse);

    rewriter.replaceOp(sliceOp, newSelectOp.getResult());

    return success();
    ;
  }
};

struct ConstPadConcatToConcat
    : public CheckedOpRewritePattern<stablehlo::PadOp, ConstPadConcatToConcat> {
  using CheckedOpRewritePattern<
      stablehlo::PadOp, ConstPadConcatToConcat>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::PadOp padOp,
                                    PatternRewriter &rewriter) const {
    auto concatOp =
        padOp.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concatOp)
      return failure();

    if (!concatOp->hasOneUse())
      return failure();

    DenseElementsAttr padConst;
    if (!matchPattern(padOp.getPaddingValue(), m_Constant(&padConst)))
      return failure();

    // Ensure interior padding is all zero.
    for (int64_t ip : padOp.getInteriorPadding())
      if (ip != 0)
        return failure();

    // Count how many dimensions have nonzero pad.
    int nonZeroPads = 0;
    int padDim = -1;
    for (int i = 0, e = padOp.getEdgePaddingLow().size(); i < e; ++i) {
      int64_t low = padOp.getEdgePaddingLow()[i];
      int64_t high = padOp.getEdgePaddingHigh()[i];
      if (low != 0 || high != 0) {
        ++nonZeroPads;
        padDim = i;
      }
    }
    // Only handle the case with a single padded dimension.
    if (nonZeroPads != 1)
      return failure();

    // Ensure the padded dimension (padDim) matches the concatenate op's
    // dimension.
    if (padDim != concatOp.getDimension())
      return failure();

    // For simplicity, require padding on only one side.
    bool padAtLow = (padOp.getEdgePaddingLow()[padDim] > 0);
    bool padAtHigh = (padOp.getEdgePaddingHigh()[padDim] > 0);
    if (padAtLow && padAtHigh)
      return failure();

    // Create a constant tensor to replace the pad.
    // Its shape is the same as the output except in padDim where its size
    // equals the pad amount.
    auto outShape = padOp.getType().getShape();
    int64_t padAmount = padAtLow ? padOp.getEdgePaddingLow()[padDim]
                                 : padOp.getEdgePaddingHigh()[padDim];
    SmallVector<int64_t> constShape(outShape.begin(), outShape.end());
    constShape[padDim] = padAmount;
    auto elemType = padOp.getType().getElementType();
    auto constTensorType = RankedTensorType::get(constShape, elemType);
    // Use the pad constant (splat) to create the constant tensor.
    auto newSplattedConstOp = rewriter.create<stablehlo::ConstantOp>(
        padOp.getLoc(), constTensorType, padConst.resizeSplat(constTensorType));

    // Now, instead of padding the concatenation result, we insert the pad
    // slice via a new concatenate.
    auto origOperands = concatOp.getOperands();
    SmallVector<Value, 4> newOperands(origOperands.begin(), origOperands.end());
    if (padAtLow) {
      // Insert the pad slice at the beginning.
      newOperands.insert(newOperands.begin(), newSplattedConstOp);
    } else {
      // Insert the pad slice at the end.
      newOperands.push_back(newSplattedConstOp);
    }

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        padOp, padOp.getResult().getType(), newOperands,
        concatOp.getDimension());
    return success();
  }
};

template <typename T> struct Term {
  APFloat constantFactor;
  Value valFactor;
  T term;
  Term(APFloat constantFactor, Value valFactor, T term)
      : constantFactor(constantFactor), valFactor(valFactor), term(term) {}
};

template <typename ST, typename Child>
struct SumToReductionBase
    : public CheckedOpRewritePattern<ST, SumToReductionBase<ST, Child>> {
  using CheckedOpRewritePattern<
      ST, SumToReductionBase<ST, Child>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(ST op, PatternRewriter &rewriter) const {
    if (!op.getType().getElementType().isFloat())
      return failure();

    SmallVector<Term<stablehlo::SliceOp>> done0;
    SmallVector<Term<Value>> todo;
    APFloat zero(
        cast<FloatType>(op.getType().getElementType()).getFloatSemantics(),
        "0");
    APFloat one(
        cast<FloatType>(op.getType().getElementType()).getFloatSemantics(),
        "1");
    todo.emplace_back(one, (Value) nullptr, (Value)op.getResult());
    SmallVector<Value> intermediates;
    SmallPtrSet<Operation *, 1> seen;
    while (!todo.empty()) {
      auto cur = todo.pop_back_val();
      if (auto add = cur.term.getDefiningOp<stablehlo::AddOp>()) {
        todo.emplace_back(cur.constantFactor, cur.valFactor, add.getLhs());
        todo.emplace_back(cur.constantFactor, cur.valFactor, add.getRhs());
        intermediates.push_back(add);
        seen.insert(add);
        continue;
      }
      if (auto sub = cur.term.getDefiningOp<stablehlo::SubtractOp>()) {
        todo.emplace_back(cur.constantFactor, cur.valFactor, sub.getLhs());
        todo.emplace_back(-cur.constantFactor, cur.valFactor, sub.getRhs());
        intermediates.push_back(sub);
        seen.insert(sub);
        continue;
      }
      if (auto mul = cur.term.getDefiningOp<stablehlo::MulOp>()) {
        bool legal = false;
        for (int i = 0; i < 2; i++) {
          SplatElementsAttr other;
          if (!matchPattern(mul->getOperand(1 - i), m_Constant(&other)))
            continue;
          todo.emplace_back(cur.constantFactor * other.getSplatValue<APFloat>(),
                            cur.valFactor, mul->getOperand(i));
          legal = true;
          break;
        }
        if (legal) {
          intermediates.push_back(mul);
          seen.insert(mul);
          continue;
        }
      }
      if (auto div = cur.term.getDefiningOp<stablehlo::DivOp>()) {
        SplatElementsAttr other;
        if (matchPattern(div.getRhs(), m_Constant(&other))) {
          todo.emplace_back(cur.constantFactor / other.getSplatValue<APFloat>(),
                            cur.valFactor, div.getLhs());
          intermediates.push_back(div);
          seen.insert(div);
          continue;
        }
      }
      if (auto sl = cur.term.getDefiningOp<stablehlo::SliceOp>()) {
        done0.emplace_back(cur.constantFactor, cur.valFactor, sl);
        seen.insert(sl);
        continue;
      }
      if (auto dfop = cur.term.getDefiningOp())
        return rewriter.notifyMatchFailure(
            dfop, "operand is not a linear slice term");
      else
        return rewriter.notifyMatchFailure(
            op, "had operand that is not a linear slice term");
    }

    if (done0.size() < 2)
      return failure();

    auto T = op.getType();

    for (auto v : intermediates) {
      if (v == op)
        continue;
      for (auto u : v.getUsers()) {
        if (!seen.contains(u)) {
          return rewriter.notifyMatchFailure(
              u, "Had use outside of intermediates");
        }
      }
    }

    llvm::MapVector<Value, SmallVector<Term<stablehlo::SliceOp>>> doneMapping;

    for (auto result : done0) {
      doneMapping[result.term.getOperand()].push_back(result);
    }

    bool hasMerge = false;

    SmallVector<Term<Value>> finalToAdd;

    for (auto &&[_, done] : doneMapping) {
      if (done.size() == 1) {
        for (auto val : done)
          finalToAdd.emplace_back(val.constantFactor, val.valFactor, val.term);
        continue;
      }

      ssize_t offsetDim = -1;
      bool legal = true;
      for (int i = 0; i < done.size(); i++) {
        assert(done[i].term.getOperand() == done[0].term.getOperand());
        if (done[i].valFactor) {
          legal = false;
          break;
        }
        ssize_t mismatch = -1;
        for (int j = 0; j < T.getShape().size(); j++) {
          if (done[i].term.getStrides()[j] != 1) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Non-one stride for " << done[i].term << "\n");
            legal = false;
            break;
          }
          if (done[i].term.getStartIndices()[j] !=
                  done[0].term.getStartIndices()[j] ||
              done[i].term.getLimitIndices()[j] !=
                  done[0].term.getLimitIndices()[j]) {
            if (mismatch != -1) {
              if (mismatch != j) {
                LLVM_DEBUG(llvm::dbgs()
                           << "Multi-dimensional mismatch of slice between "
                           << done[i].term << " and " << done[0].term << "\n");
                legal = false;
                break;
              }
            } else {
              mismatch = j;
            }
          }
        }
        if (mismatch == -1) {
          continue;
        }
        if (offsetDim == -1) {
          offsetDim = mismatch;
          continue;
        }
        if (offsetDim == mismatch) {
          continue;
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "Multi-dimensional(2) mismatch of slice between "
                   << done[i].term << " and " << done[0].term << "\n");
        legal = false;
        break;
      }

      if (!legal) {
        for (auto val : done)
          finalToAdd.emplace_back(val.constantFactor, val.valFactor, val.term);
        continue;
      }

      if (legal && offsetDim == -1) {
        std::optional<APFloat> tally;
        for (auto term : done) {
          if (tally)
            tally = *tally + term.constantFactor;
          else
            tally = term.constantFactor;
        }
        finalToAdd.emplace_back(*tally, nullptr, done[0].term);
        hasMerge = true;
        continue;
      }

      std::map<int, APFloat> terms;
      for (int i = 0; i < done.size(); i++) {
        int offset = (int)done[i].term.getStartIndices()[offsetDim] -
                     (int)done[0].term.getStartIndices()[offsetDim];
        auto found = terms.find(offset);
        if (found != terms.end())
          found->second = found->second + done[i].constantFactor;
        else
          terms.emplace(offset, done[i].constantFactor);
      }
      assert(terms.size() > 1);

      int startidx = terms.begin()->first;
      auto lastItr = terms.end();
      lastItr--;
      int lastidx = lastItr->first;
      assert(lastidx != startidx);

      SmallVector<APFloat> pad(lastidx + 1 - startidx, zero);
      // Check contiguous
      for (auto &term : terms) {
        pad[term.first - startidx] = pad[term.first - startidx] + term.second;
      }

      if (!((Child *)this)->applies(offsetDim, T, terms, startidx, lastidx)) {
        for (auto val : done)
          finalToAdd.emplace_back(val.constantFactor, val.valFactor, val.term);
        continue;
      }

      SmallVector<int64_t> newStart =
          llvm::to_vector(done[0].term.getStartIndices());
      SmallVector<int64_t> newLimit =
          llvm::to_vector(done[0].term.getLimitIndices());
      SmallVector<int64_t> newStride =
          llvm::to_vector(done[0].term.getStrides());
      newStart[offsetDim] += startidx;
      newLimit[offsetDim] += lastidx;

      Value input = rewriter.create<stablehlo::SliceOp>(
          done[0].term.getLoc(), done[0].term.getOperand(), newStart, newLimit,
          newStride);

      auto fty = RankedTensorType::get({lastidx + 1 - startidx},
                                       rewriter.getF64Type());
      Value filter =
          rewriter
              .create<stablehlo::ConstantOp>(op.getLoc(), fty,
                                             DenseFPElementsAttr::get(fty, pad))
              .getResult();
      filter = rewriter
                   .create<stablehlo::ConvertOp>(
                       op.getLoc(),
                       RankedTensorType::get({lastidx + 1 - startidx},
                                             op.getType().getElementType()),
                       filter)
                   .getResult();

      auto conv = ((Child *)this)
                      ->makeReduction(rewriter, input, offsetDim, T, terms,
                                      startidx, lastidx, filter);
      finalToAdd.emplace_back(one, nullptr, conv);
      hasMerge = true;
    }

    if (!hasMerge)
      return failure();

    assert(finalToAdd.size());

    Value result = nullptr;
    for (auto term : finalToAdd) {
      Value intermediate = term.term;
      if (!term.constantFactor.isExactlyValue(1.0)) {
        intermediate = rewriter.create<stablehlo::MulOp>(
            op.getLoc(), intermediate,
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), intermediate.getType(),
                cast<ElementsAttr>(
                    makeAttr(intermediate.getType(), term.constantFactor))));
      }
      if (result == nullptr)
        result = intermediate;
      else
        result = rewriter.create<stablehlo::AddOp>(op.getLoc(), result,
                                                   intermediate);
    }
    assert(result);
    rewriter.replaceOp(op, ValueRange{result});
    return success();
  }

  bool reduceWindowApplies(size_t offsetDim, RankedTensorType T,
                           const std::map<int, APFloat> &pad, int startidx,
                           int lastidx) {
    auto start = pad.find(startidx)->second;
    for (int i = startidx; i <= lastidx; i++) {
      auto found = pad.find(i);
      if (found == pad.end())
        return false;
      if (found->second != start)
        return false;
    }
    return true;
  }
};

template <typename ST>
struct SumToConv : public SumToReductionBase<ST, SumToConv<ST>> {
  bool collapseDims;
  SumToConv(bool collapseDims, MLIRContext *context, PatternBenefit benefit = 1,
            ArrayRef<StringRef> generatedNames = {})
      : SumToReductionBase<ST, SumToConv<ST>>(context, benefit, generatedNames),
        collapseDims(collapseDims) {}

  using SumToReductionBase<ST, SumToConv<ST>>::matchAndRewrite;

  bool applies(size_t offsetDim, RankedTensorType T,
               const std::map<int, APFloat> &pad, int startidx, int lastidx) {
    if (SumToReductionBase<ST, SumToConv<ST>>::reduceWindowApplies(
            offsetDim, T, pad, startidx, lastidx)) {
      return false;
    }
    if (collapseDims)
      return true;
    else {
      if (T.getShape().size() > 2) {
        return false;
      } else {
        return true;
      }
    }
  }

  Value makeReduction(PatternRewriter &rewriter, Value input, size_t offsetDim,
                      RankedTensorType T, const std::map<int, APFloat> &pad,
                      int startidx, int lastidx, Value filter) {

    size_t newOffsetDim = offsetDim;
    RankedTensorType pre_reshape = T;
    size_t reshapeOffsetDim = 0;
    SmallVector<int64_t> permutation;

    if (T.getShape().size() > 2) {
      if (newOffsetDim != 0 && newOffsetDim != T.getShape().size() - 1) {
        for (int i = 0; i < T.getShape().size(); i++)
          permutation.push_back(i);
        permutation[newOffsetDim] = 0;
        permutation[0] = newOffsetDim;
        input = rewriter.create<stablehlo::TransposeOp>(input.getLoc(), input,
                                                        permutation);
        newOffsetDim = 0;
      }
      if (newOffsetDim == 0) {
        auto RT = cast<RankedTensorType>(input.getType());
        pre_reshape = RT;
        reshapeOffsetDim = newOffsetDim;
        int64_t newDims[3] = {RT.getShape()[newOffsetDim], 1, 1};
        for (int i = 1; i < RT.getShape().size(); i++) {
          newDims[1] *= RT.getShape()[i];
        }
        input = rewriter.create<stablehlo::ReshapeOp>(
            input.getLoc(), RankedTensorType::get(newDims, T.getElementType()),
            input);
      } else {
        assert(newOffsetDim == T.getShape().size() - 1);

        auto RT = cast<RankedTensorType>(input.getType());
        pre_reshape = RT;
        reshapeOffsetDim = newOffsetDim;
        int64_t newDims[3] = {1, 1, RT.getShape()[newOffsetDim]};
        for (int i = 0; i < RT.getShape().size() - 1; i++) {
          newDims[0] *= RT.getShape()[i];
        }
        input = rewriter.create<stablehlo::ReshapeOp>(
            input.getLoc(), RankedTensorType::get(newDims, T.getElementType()),
            input);
        newOffsetDim = 2;
      }
    } else if (T.getShape().size() < 3) {
      SmallVector<int64_t> newDims =
          llvm::to_vector(cast<RankedTensorType>(input.getType()).getShape());
      reshapeOffsetDim = newOffsetDim;
      while (newDims.size() < 3) {
        newDims.insert(newDims.begin(), 1);
        newOffsetDim++;
      }
      pre_reshape = cast<RankedTensorType>(input.getType());
      input = rewriter.create<stablehlo::ReshapeOp>(
          input.getLoc(), RankedTensorType::get(newDims, T.getElementType()),
          input);
    }
    SmallVector<int64_t> nonOffsetDims;
    auto CT = cast<RankedTensorType>(input.getType()).getShape();
    for (int i = 0; i < CT.size(); i++) {
      if (i != newOffsetDim)
        nonOffsetDims.push_back(i);
    }
    if (CT[nonOffsetDims[1]] != 1 && CT[nonOffsetDims[0]] == 1) {
      std::swap(nonOffsetDims[1], nonOffsetDims[0]);
    }

    filter = rewriter
                 .create<stablehlo::ReshapeOp>(
                     input.getLoc(),
                     RankedTensorType::get({lastidx + 1 - startidx, 1, 1},
                                           T.getElementType()),
                     filter)
                 .getResult();

    // Create convolution dimension numbers
    auto convDims = stablehlo::ConvDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*input_batch_dimension=*/nonOffsetDims[0],
        /*input_feature_dimension=*/nonOffsetDims[1],
        /*input_spatial_dimensions=*/{(int64_t)newOffsetDim},
        /*kernel_input_feature_dimension=*/1,
        /*kernel_output_feature_dimension=*/2,
        /*kernel_spatial_dimensions=*/{0},
        /*output_batch_dimension=*/nonOffsetDims[0],
        /*output_feature_dimension=*/nonOffsetDims[1],
        /*output_spatial_dimensions=*/{(int64_t)newOffsetDim});

    // Create the convolution operation
    SmallVector<int64_t> outShape =
        llvm::to_vector(cast<RankedTensorType>(input.getType()).getShape());
    outShape[newOffsetDim] -= (lastidx - startidx);
    auto convOutType = RankedTensorType::get(outShape, T.getElementType());
    Value conv = rewriter.create<stablehlo::ConvolutionOp>(
        input.getLoc(), convOutType, input, filter,
        /*window_strides=*/nullptr,
        /*padding=*/nullptr,
        /*lhs_dilation=*/nullptr,
        /*rhs_dilation=*/nullptr,
        /*window_reversal=*/nullptr,
        /*conv_dimension_numbers=*/convDims,
        /*feature_group_count=*/rewriter.getI64IntegerAttr(1),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/nullptr);

    if (conv.getType() != pre_reshape) {
      SmallVector<int64_t> post_shape = llvm::to_vector(pre_reshape.getShape());
      post_shape[reshapeOffsetDim] -= (lastidx - startidx);
      RankedTensorType post_reshape =
          RankedTensorType::get(post_shape, pre_reshape.getElementType());
      conv = rewriter.create<stablehlo::ReshapeOp>(input.getLoc(), post_reshape,
                                                   conv);
    }
    if (permutation.size())
      conv = rewriter.create<stablehlo::TransposeOp>(input.getLoc(), conv,
                                                     permutation);
    return conv;
  }
};

template <typename ST>
struct SumToReduceWindow
    : public SumToReductionBase<ST, SumToReduceWindow<ST>> {
  SumToReduceWindow(MLIRContext *context, PatternBenefit benefit = 1,
                    ArrayRef<StringRef> generatedNames = {})
      : SumToReductionBase<ST, SumToReduceWindow<ST>>(context, benefit,
                                                      generatedNames) {}

  using SumToReductionBase<ST, SumToReduceWindow<ST>>::matchAndRewrite;

  bool applies(size_t offsetDim, RankedTensorType T,
               const std::map<int, APFloat> &pad, int startidx, int lastidx) {
    return SumToReductionBase<ST, SumToReduceWindow<ST>>::reduceWindowApplies(
        offsetDim, T, pad, startidx, lastidx);
  }

  Value makeReduction(PatternRewriter &rewriter, Value input, size_t offsetDim,
                      RankedTensorType T, const std::map<int, APFloat> &pad,
                      int startidx, int lastidx, Value filter) {
    auto factor = pad.begin()->second;

    SmallVector<int64_t> outShape =
        llvm::to_vector(cast<RankedTensorType>(input.getType()).getShape());
    outShape[offsetDim] -= (lastidx - startidx);

    auto unrankedTensorType = RankedTensorType::get({}, T.getElementType());
    Value init_values[1] = {rewriter.create<stablehlo::ConstantOp>(
        input.getLoc(), rewriter.getZeroAttr(unrankedTensorType))};

    SmallVector<int64_t> win_dim(outShape.size(), 1);
    win_dim[offsetDim] = lastidx - startidx + 1;

    SmallVector<int64_t> win_strides(outShape.size(), 1);
    SmallVector<int64_t> win_dialations(outShape.size(), 1);
    SmallVector<int64_t> base_dialations(outShape.size(), 1);
    SmallVector<int64_t> padding_dialations(2 * outShape.size(), 0);

    int64_t padding_shape[2] = {(int64_t)outShape.size(), 2};

    Value operands[1] = {input};
    Type restys[1] = {RankedTensorType::get(outShape, T.getElementType())};
    auto redwin = rewriter.create<stablehlo::ReduceWindowOp>(
        input.getLoc(), restys, operands, init_values,
        rewriter.getDenseI64ArrayAttr(win_dim),
        rewriter.getDenseI64ArrayAttr(win_strides),
        rewriter.getDenseI64ArrayAttr(base_dialations),
        rewriter.getDenseI64ArrayAttr(win_dialations),
        DenseIntElementsAttr::get(
            RankedTensorType::get(padding_shape, rewriter.getIntegerType(64)),
            padding_dialations));

    Type tys[2] = {unrankedTensorType, unrankedTensorType};
    Location locs[2] = {input.getLoc(), input.getLoc()};

    Value result = redwin->getResult(0);
    if (!factor.isExactlyValue(1.0)) {
      result = rewriter.create<stablehlo::MulOp>(
          input.getLoc(), result,
          rewriter.create<stablehlo::ConstantOp>(
              input.getLoc(), result.getType(),
              cast<ElementsAttr>(makeAttr(result.getType(), factor))));
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto block = rewriter.createBlock(&redwin.getBody(), {}, tys, locs);
      rewriter.setInsertionPointToStart(block);
      auto addOp = rewriter.create<stablehlo::AddOp>(
          input.getLoc(), block->getArgument(0), block->getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(input.getLoc(), addOp.getResult());
    }

    return result;
  }
};

struct TransposeSelect
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeSelect> {
  using CheckedOpRewritePattern<stablehlo::TransposeOp,
                                TransposeSelect>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp transposeOp,
                                    PatternRewriter &rewriter) const {
    auto selectOp =
        transposeOp.getOperand().getDefiningOp<stablehlo::SelectOp>();
    if (!selectOp)
      return failure();

    if (!selectOp->hasOneUse())
      return failure();

    Value pred = selectOp.getPred();
    bool scalar_pred =
        dyn_cast<RankedTensorType>(pred.getType()).getRank() == 0;

    Value newPred;
    if (!scalar_pred) {
      newPred = rewriter.create<stablehlo::TransposeOp>(
          transposeOp.getLoc(), pred, transposeOp.getPermutation());
    } else {
      newPred = pred;
    }

    SmallVector<int64_t> permutation;
    for (int i = 0; i < transposeOp.getPermutation().size(); i++) {
      permutation.push_back(transposeOp.getPermutation()[i]);
    }

    auto onTrueTransposed = rewriter.create<stablehlo::TransposeOp>(
        transposeOp.getLoc(), selectOp.getOnTrue(), permutation);
    auto onFalseTransposed = rewriter.create<stablehlo::TransposeOp>(
        transposeOp.getLoc(), selectOp.getOnFalse(), permutation);

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        transposeOp, newPred, onTrueTransposed, onFalseTransposed);
    return success();
  }
};

struct ReshapeSelect
    : public CheckedOpRewritePattern<stablehlo::ReshapeOp, ReshapeSelect> {
  using CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                ReshapeSelect>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp reshapeOp,
                                    PatternRewriter &rewriter) const {
    auto selectOp = reshapeOp.getOperand().getDefiningOp<stablehlo::SelectOp>();
    if (!selectOp)
      return failure();

    if (!selectOp->hasOneUse())
      return failure();

    Value pred = selectOp.getPred();
    bool scalar_pred =
        dyn_cast<RankedTensorType>(pred.getType()).getRank() == 0;

    Value newPred;
    if (!scalar_pred) {
      newPred = rewriter.create<stablehlo::ReshapeOp>(
          reshapeOp.getLoc(),
          RankedTensorType::get(
              reshapeOp.getType().getShape(),
              cast<RankedTensorType>(pred.getType()).getElementType()),
          pred);
    } else {
      newPred = pred;
    }

    auto onTrueReshaped = rewriter.create<stablehlo::ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(), selectOp.getOnTrue());
    auto onFalseReshaped = rewriter.create<stablehlo::ReshapeOp>(
        reshapeOp.getLoc(), reshapeOp.getType(), selectOp.getOnFalse());

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        reshapeOp, newPred, onTrueReshaped, onFalseReshaped);
    return success();
  }
};

template <typename T>
struct GroupComms : public CheckedOpRewritePattern<T, GroupComms<T>> {
  using CheckedOpRewritePattern<T, GroupComms<T>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(T end, PatternRewriter &rewriter) const {
    if (end->template getParentOfType<enzymexla::CommRegionOp>())
      return failure();
    if (end->template getParentOfType<sdy::ManualComputationOp>())
      return failure();
    SetVector<Operation *> done;
    done.insert(end);
    SmallVector<Operation *> todo = {end};
    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (cur != end && done.contains(cur))
        continue;
      if (!isa<stablehlo::SliceOp, stablehlo::ConcatenateOp, stablehlo::PadOp,
               stablehlo::TransposeOp, stablehlo::ReshapeOp,
               stablehlo::DynamicUpdateSliceOp, enzymexla::RotateOp,
               enzymexla::ExtendOp, enzymexla::WrapOp>(cur))
        continue;
      bool allWithin = true;
      if (cur != end)
        for (auto res : cur->getResults()) {
          for (auto u : res.getUsers()) {
            if (!done.contains(u)) {
              allWithin = false;
              break;
            }
          }
        }
      if (!allWithin)
        continue;
      done.insert(cur);
      for (auto op : cur->getOperands()) {
        if (auto v = op.getDefiningOp()) {
          todo.push_back(v);
        }
      }
    }

    done = mlir::topologicalSort(done);

    auto newOp = rewriter.create<enzymexla::CommRegionOp>(
        end.getLoc(), end.getResult().getType());
    rewriter.createBlock(&newOp.getBody(), newOp.getBody().begin());
    IRMapping map;
    for (auto op : done) {
      rewriter.clone(*op, map);
    }
    rewriter.create<stablehlo::ReturnOp>(end.getLoc(),
                                         map.lookup(end.getResult()));
    for (auto op : llvm::reverse(done)) {
      if (op == end) {
        rewriter.replaceOp(op, newOp);
      } else {
        rewriter.eraseOp(op);
      }
    }
    return success();
  }
};

struct LowerCommRegion
    : public CheckedOpRewritePattern<enzymexla::CommRegionOp, LowerCommRegion> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::CommRegionOp end,
                                    PatternRewriter &rewriter) const {
    IRMapping map;
    for (auto &op : end.getBody().front()) {
      if (isa<stablehlo::ReturnOp>(op)) {
        SmallVector<Value> operands;
        for (auto v : op.getOperands())
          operands.push_back(map.lookup(v));
        rewriter.replaceOp(end, operands);
        return success();
      } else {
        rewriter.clone(op, map);
      }
    }
    return failure();
  }
};

bool isRotateLike(int dimension, Value lhs, Value rhs,
                  stablehlo::SliceOp *sl0P = nullptr,
                  stablehlo::SliceOp *sl1P = nullptr) {
  auto sl0 = lhs.getDefiningOp<stablehlo::SliceOp>();
  if (!sl0)
    return false;
  auto sl1 = rhs.getDefiningOp<stablehlo::SliceOp>();
  if (!sl1)
    return false;
  if (sl0.getOperand() != sl1.getOperand())
    return false;

  if (sl0P)
    *sl0P = sl0;
  if (sl1P)
    *sl1P = sl1;

  if (sl0.getOperand() != sl1.getOperand())
    return false;

  // sl0[A:end], sl1[start:A]
  for (int j = 0; j < sl0.getType().getShape().size(); j++) {
    if (j == dimension) {
      if (sl0.getStrides()[j] != 1 || sl1.getStrides()[j] != 1) {
        return false;
      }
      if (sl0.getStartIndices()[j] != sl1.getLimitIndices()[j]) {
        return false;
      }
    } else {
      if (sl0.getLimitIndices()[j] != sl1.getLimitIndices()[j]) {
        return false;
      }
      if (sl0.getStartIndices()[j] != sl1.getStartIndices()[j]) {
        return false;
      }
      if (sl0.getStrides()[j] != sl1.getStrides()[j]) {
        return false;
      }
    }
  }
  return true;
}

struct RecognizeRotate
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                     RecognizeRotate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concat,
                                    PatternRewriter &rewriter) const {
    if (concat.getOperands().size() < 2)
      return failure();
    for (int i = 1; i < concat.getOperands().size(); i++) {
      stablehlo::SliceOp sl0, sl1;
      if (!isRotateLike(concat.getDimension(), concat.getOperands()[i - 1],
                        concat.getOperands()[i], &sl0, &sl1)) {
        continue;
      }
      auto starts = llvm::to_vector(sl1.getStartIndices());
      auto limits = llvm::to_vector(sl0.getLimitIndices());
      Value outerSlice = sl0.getOperand();

      bool needsSlice = false;
      for (int j = 0; j < sl0.getType().getShape().size(); j++) {
        if (starts[j] != 0) {
          needsSlice = true;
          break;
        }
        if (limits[j] !=
            cast<RankedTensorType>(sl0.getOperand().getType()).getShape()[j]) {
          needsSlice = true;
          break;
        }
        if (sl0.getStrides()[j] != 1) {
          needsSlice = true;
          break;
        }
      }
      if (needsSlice) {
        outerSlice = rewriter.create<stablehlo::SliceOp>(
            sl0.getLoc(), sl0.getOperand(), starts, limits, sl0.getStrides());
        if (auto shard = sdy::getShardingPerValue(sl0)) {
          sdy::setShardings(outerSlice.getDefiningOp(), shard);
        }
      }
      auto rotate = rewriter.create<enzymexla::RotateOp>(
          sl1.getLoc(), outerSlice,
          sl1.getType().getShape()[concat.getDimension()],
          concat.getDimension());
      if (auto shard = sdy::getShardingPerValue(concat)) {
        sdy::setShardings(rotate, shard);
      }
      SmallVector<Value> toConcat;
      for (int j = 0; j < i - 1; j++)
        toConcat.push_back(concat.getOperands()[j]);
      toConcat.push_back(rotate);
      for (int j = i + 1; j < concat.getOperands().size(); j++)
        toConcat.push_back(concat.getOperands()[j]);
      if (toConcat.size() == 1) {
        rewriter.replaceOp(concat, toConcat[0]);
      } else {
        auto shard = sdy::getShardingPerValue(concat);
        auto newConcat = rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
            concat, toConcat, concat.getDimension());
        if (shard) {
          sdy::setShardings(newConcat, shard);
        }
      }
      return success();
    }
    return failure();
  }
};

bool isWrapLike(int dim, Value lhs, Value mid, Value rhs,
                stablehlo::SliceOp *sl0P = nullptr,
                stablehlo::SliceOp *sl1P = nullptr) {
  auto sl0 = lhs.getDefiningOp<stablehlo::SliceOp>();
  if (!sl0)
    return false;
  auto midT = cast<RankedTensorType>(mid.getType());
  auto sl1 = rhs.getDefiningOp<stablehlo::SliceOp>();
  if (!sl1)
    return false;
  if (sl0.getOperand() != sl1.getOperand())
    return false;

  if (sl0P)
    *sl0P = sl0;
  if (sl1P)
    *sl1P = sl1;

  SmallVector<int64_t> midStarts(midT.getShape().size(), 0);
  SmallVector<int64_t> midLimits = llvm::to_vector(midT.getShape());
  SmallVector<int64_t> midStrides(midT.getShape().size(), 1);
  if (sl0.getOperand() != mid) {
    auto slM = mid.getDefiningOp<stablehlo::SliceOp>();
    if (!slM)
      return false;
    if (slM.getOperand() != sl0.getOperand())
      return false;
    midStarts = llvm::to_vector(slM.getStartIndices());
    midLimits = llvm::to_vector(slM.getLimitIndices());
    midStrides = llvm::to_vector(slM.getStrides());
  }

  // sl0[B-lhs:B], mid[A:B] sl1[A:A+rhs]
  for (int j = 0; j < sl0.getType().getShape().size(); j++) {
    if (j == dim) {
      if (sl0.getStrides()[j] != 1 || sl1.getStrides()[j] != 1 ||
          midStrides[j] != 1) {
        return false;
      }
      if (sl0.getLimitIndices()[j] != midLimits[j]) {
        return false;
      }
      if (midStarts[j] != sl1.getStartIndices()[j]) {
        return false;
      }
    } else {
      if (sl0.getLimitIndices()[j] != sl1.getLimitIndices()[j]) {
        return false;
      }
      if (sl0.getLimitIndices()[j] != midLimits[j]) {
        return false;
      }
      if (sl0.getStartIndices()[j] != sl1.getStartIndices()[j]) {
        return false;
      }
      if (sl0.getStartIndices()[j] != midStarts[j]) {
        return false;
      }
      if (sl0.getStrides()[j] != sl1.getStrides()[j]) {
        return false;
      }
      if (sl0.getStrides()[j] != midStrides[j]) {
        return false;
      }
    }
  }
  return true;
}

bool isOuterReducingReshape(stablehlo::ReshapeOp op) {
  auto prevT = cast<RankedTensorType>(op.getOperand().getType());
  if (prevT.getShape().size() != op.getType().getShape().size() + 1)
    return false;
  if (prevT.getShape()[0] != 1)
    return false;
  for (int i = 1; i < prevT.getShape().size(); i++) {
    if (prevT.getShape()[i] != op.getType().getShape()[i - 1])
      return false;
  }
  return true;
}

struct RecognizeWrap
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp, RecognizeWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concat,
                                    PatternRewriter &rewriter) const {

    int concatDim = concat.getDimension();
    SmallVector<Value> operands = llvm::to_vector(concat.getOperands());
    auto shard = sdy::getShardingPerValue(concat);

    if (operands.size() < 2)
      return failure();
    for (int i = 2; i < operands.size(); i++) {
      stablehlo::SliceOp sl0;
      auto mid = operands[i - 1];
      stablehlo::SliceOp sl1;
      if (isWrapLike(concatDim, operands[i - 2], mid, operands[i], &sl0,
                     &sl1)) {
        auto wrap = rewriter.create<enzymexla::WrapOp>(
            sl0.getLoc(), mid, sl0.getType().getShape()[concatDim],
            sl1.getType().getShape()[concatDim], concatDim);
        if (auto shard = sdy::getShardingPerValue(sl0)) {
          sdy::setShardings(wrap, shard);
        }
        SmallVector<Value> toConcat;
        for (int j = 0; j < i - 2; j++)
          toConcat.push_back(operands[j]);
        toConcat.push_back(wrap);
        for (int j = i + 1; j < operands.size(); j++)
          toConcat.push_back(operands[j]);
        if (toConcat.size() == 1) {
          rewriter.replaceOp(concat, toConcat[0]);
        } else {
          auto newConcat =
              rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
                  concat, toConcat, concatDim);
          if (shard) {
            sdy::setShardings(newConcat, shard);
          }
        }
        return success();
      }
      auto rs0 = operands[i - 2].getDefiningOp<stablehlo::ReshapeOp>();
      auto rsmid = operands[i - 1].getDefiningOp<stablehlo::ReshapeOp>();
      auto rs1 = operands[i].getDefiningOp<stablehlo::ReshapeOp>();
      if (rs0 && rsmid && rs1 && isOuterReducingReshape(rs0) &&
          isOuterReducingReshape(rsmid) && isOuterReducingReshape(rs1)) {
        if (isWrapLike(concatDim + 1, rs0.getOperand(), rsmid.getOperand(),
                       rs1.getOperand(), &sl0, &sl1)) {
          auto wrap = rewriter.create<enzymexla::WrapOp>(
              sl0.getLoc(), rsmid.getOperand(),
              sl0.getType().getShape()[concatDim + 1],
              sl1.getType().getShape()[concatDim + 1], concatDim + 1);
          if (auto shard = sdy::getShardingPerValue(sl0)) {
            sdy::setShardings(wrap, shard);
          }
          SmallVector<Value> toConcat;
          for (int j = 0; j < i - 2; j++)
            toConcat.push_back(operands[j]);
          SmallVector<int64_t> newShape =
              llvm::to_vector(wrap.getType().getShape());
          assert(newShape[0] == 1);
          newShape.erase(newShape.begin());
          auto reshape = rewriter.create<stablehlo::ReshapeOp>(
              concat.getLoc(),
              RankedTensorType::get(newShape, wrap.getType().getElementType()),
              wrap);
          if (auto shard = sdy::getShardingPerValue(rs0)) {
            sdy::setShardings(reshape, shard);
          }
          toConcat.push_back(reshape);
          for (int j = i + 1; j < operands.size(); j++)
            toConcat.push_back(operands[j]);
          if (toConcat.size() == 1) {
            rewriter.replaceOp(concat, toConcat[0]);
          } else {
            auto shard = sdy::getShardingPerValue(rs0);
            rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
                concat, toConcat, concatDim);
            if (shard) {
              sdy::setShardings(reshape, shard);
            }
          }
          return success();
        }
      }
    }
    return failure();
  }
};

struct ExtendSplat
    : public CheckedOpRewritePattern<enzymexla::ExtendOp, ExtendSplat> {
  using CheckedOpRewritePattern<enzymexla::ExtendOp,
                                ExtendSplat>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(enzymexla::ExtendOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr cstAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&cstAttr)))
      return failure();

    if (!cstAttr.isSplat())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, SplatElementsAttr::get(op.getType(),
                                   cstAttr.getSplatValue<Attribute>()));

    return success();
  }
};

template <typename EnzymeOp>
LogicalResult commUnaryOpElementwise(bool onlySingleUser, EnzymeOp op,
                                     PatternRewriter &rewriter) {
  if (onlySingleUser && !llvm::hasSingleElement(op->getUsers()))
    return failure();

  bool anyModified = false;
  for (auto elem : llvm::make_early_inc_range(op->getUsers())) {
    if (!elem->template hasTrait<mlir::OpTrait::Elementwise>() ||
        elem->getNumResults() != 1 || elem->getNumOperands() != 1)
      continue;

    auto newOp = rewriter.create(
        elem->getLoc(), elem->getName().getIdentifier(),
        ValueRange(op.getOperand()),
        TypeRange(cast<RankedTensorType>(elem->getResult(0).getType())
                      .clone(cast<RankedTensorType>(op.getOperand().getType())
                                 .getShape())),
        elem->getAttrs(), {}, {});
    rewriter.replaceOpWithNewOp<EnzymeOp>(
        elem, newOp->getResult(0), op.getLhs(), op.getRhs(), op.getDimension());
    anyModified = true;
  }

  return success(anyModified);
}

struct ExtendUnaryElementwise
    : public CheckedOpRewritePattern<enzymexla::ExtendOp,
                                     ExtendUnaryElementwise> {
  bool onlySingleUser;

  ExtendUnaryElementwise(bool onlySingleUser, MLIRContext *context,
                         PatternBenefit benefit = 1,
                         ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewriteImpl(enzymexla::ExtendOp op,
                                    PatternRewriter &rewriter) const {
    return commUnaryOpElementwise(onlySingleUser, op, rewriter);
  }
};

struct WrapUnaryElementwise
    : public CheckedOpRewritePattern<enzymexla::WrapOp, WrapUnaryElementwise> {
  bool onlySingleUser;

  WrapUnaryElementwise(bool onlySingleUser, MLIRContext *context,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : CheckedOpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewriteImpl(enzymexla::WrapOp op,
                                    PatternRewriter &rewriter) const {
    return commUnaryOpElementwise(onlySingleUser, op, rewriter);
  }
};

template <typename Op, typename EnzymeOp>
LogicalResult commBinOpElementWise(Op op, PatternRewriter &rewriter) {
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();

  auto lhsExtend = lhs.template getDefiningOp<EnzymeOp>();
  auto rhsExtend = rhs.template getDefiningOp<EnzymeOp>();

  if (!lhsExtend || !rhsExtend)
    return failure();

  if (lhsExtend.getLhs() != rhsExtend.getLhs() ||
      lhsExtend.getRhs() != rhsExtend.getRhs() ||
      lhsExtend.getDimension() != rhsExtend.getDimension())
    return failure();

  if (!llvm::hasSingleElement(lhs.getUsers()) ||
      !llvm::hasSingleElement(rhs.getUsers()))
    return failure();

  auto elementWise = rewriter.create<Op>(op.getLoc(), lhsExtend.getOperand(),
                                         rhsExtend.getOperand());
  rewriter.replaceOpWithNewOp<EnzymeOp>(op, elementWise, lhsExtend.getLhs(),
                                        lhsExtend.getRhs(),
                                        lhsExtend.getDimension());

  return success();
};

template <typename Op>
struct ExtendElementwise
    : public CheckedOpRewritePattern<Op, ExtendElementwise<Op>> {
  using CheckedOpRewritePattern<Op,
                                ExtendElementwise<Op>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(Op op, PatternRewriter &rewriter) const {
    return commBinOpElementWise<Op, enzymexla::ExtendOp>(op, rewriter);
  }
};

template <typename Op>
struct WrapElementwise
    : public CheckedOpRewritePattern<Op, WrapElementwise<Op>> {
  using CheckedOpRewritePattern<Op,
                                WrapElementwise<Op>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(Op op, PatternRewriter &rewriter) const {
    return commBinOpElementWise<Op, enzymexla::WrapOp>(op, rewriter);
  }
};

LogicalResult isExtendLike(int dim, Value _lhs, Value _mid, Value _rhs,
                           Location loc, RewriterBase &rewriter,
                           StaticSlice *lhsSS = nullptr,
                           StaticSlice *midSS = nullptr,
                           StaticSlice *rhsSS = nullptr) {
  std::optional<StaticSlice> lhs, mid, rhs;
  if (_lhs)
    lhs = StaticSlice::get(_lhs);
  if (_mid)
    mid = StaticSlice::get(_mid);
  if (_rhs)
    rhs = StaticSlice::get(_rhs);

  if (!mid)
    return rewriter.notifyMatchFailure(loc, "lhs or mid or rhs not slice");

  if (!lhs && !rhs) {
    return rewriter.notifyMatchFailure(loc, "lhs or rhs must be slice");
  }

  if (_lhs && !StaticSlice::isPrefixInDim(*lhs, *mid, dim))
    return rewriter.notifyMatchFailure(loc, "lhs not a prefix of mid");
  if (_lhs && lhs->getOutputShape(dim) != 1) {
    return failure();
  }
  if (_rhs && rhs->getOutputShape(dim) != 1) {
    return failure();
  }
  if (_rhs && !StaticSlice::isSuffixInDim(*rhs, *mid, dim))
    return rewriter.notifyMatchFailure(loc, "rhs is not suffix of mid");
  if (_rhs && !rhs->isStrideOneAtDim(dim))
    return rewriter.notifyMatchFailure(loc, "RHS not stride one");
  if (_lhs && !lhs->isStrideOneAtDim(dim))
    return rewriter.notifyMatchFailure(loc, "LHS not stride one");
  if (!mid->isStrideOneAtDim(dim))
    return rewriter.notifyMatchFailure(loc, "Mid not stride one");
  if (_rhs && rhs->getInput() != mid->getInput())
    return rewriter.notifyMatchFailure(loc,
                                       "mid and rhs not on the same input");
  if (_lhs && lhs->getInput() != mid->getInput())
    return rewriter.notifyMatchFailure(loc,
                                       "mid and lhs not on the same input");

  if (_lhs && lhsSS)
    *lhsSS = *lhs;
  if (_rhs && rhsSS)
    *rhsSS = *rhs;
  if (midSS)
    *midSS = *mid;

  return success();
}

struct RecognizeExtend
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                     RecognizeExtend> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concat,
                                    PatternRewriter &rewriter) const {
    unsigned dim = concat.getDimension();
    if (concat.getNumOperands() == 2) {
      StaticSlice lhs;
      StaticSlice mid;

      if (succeeded(isExtendLike(dim, concat.getOperand(0),
                                 concat.getOperand(1), nullptr, concat.getLoc(),
                                 rewriter, &lhs, &mid, nullptr))) {
        auto extend = rewriter.create<enzymexla::ExtendOp>(
            concat.getLoc(), mid.getOutput(), lhs.getOutputShape(dim), 0, dim);
        if (auto shard = sdy::getShardingPerValue(concat)) {
          sdy::setShardings(extend, shard);
        }
        rewriter.replaceOp(concat, extend);
        return success();
      }
    }
    if (concat.getNumOperands() == 2) {
      StaticSlice rhs;
      StaticSlice mid;

      if (succeeded(isExtendLike(dim, nullptr, concat.getOperand(0),
                                 concat.getOperand(1), concat.getLoc(),
                                 rewriter, nullptr, &mid, &rhs))) {
        auto extend = rewriter.create<enzymexla::ExtendOp>(
            concat.getLoc(), mid.getOutput(), 0, rhs.getOutputShape(dim), dim);
        if (auto shard = sdy::getShardingPerValue(concat)) {
          sdy::setShardings(extend, shard);
        }
        rewriter.replaceOp(concat, extend);
        return success();
      }
    }
    for (unsigned i = 2; i < concat.getNumOperands(); i++) {
      auto finish = [&](Value extend) {
        SmallVector<Value> toConcat;
        for (unsigned j = 0; j < i - 2; j++)
          toConcat.push_back(concat.getOperand(j));
        toConcat.push_back(extend);
        for (unsigned j = i + 1; j < concat.getNumOperands(); j++)
          toConcat.push_back(concat.getOperand(j));

        if (toConcat.size() == 1)
          rewriter.replaceOp(concat, extend);
        else {
          auto shard = sdy::getShardingPerValue(concat);
          auto newConcat =
              rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
                  concat, toConcat, dim);
          if (shard) {
            sdy::setShardings(newConcat, shard);
          }
        }
      };
      Value lhsv = concat.getOperand(i - 2);
      Value midv = concat.getOperand(i - 1);
      Value rhsv = concat.getOperand(i - 0);

      StaticSlice lhs;
      StaticSlice mid;
      StaticSlice rhs;

      if (succeeded(isExtendLike(dim, lhsv, midv, rhsv, concat.getLoc(),
                                 rewriter, &lhs, &mid, &rhs))) {
        auto extend = rewriter.create<enzymexla::ExtendOp>(
            concat.getLoc(), mid.getOutput(), lhs.getOutputShape(dim),
            rhs.getOutputShape(dim), dim);
        if (auto shard = sdy::getShardingPerValue(concat)) {
          sdy::setShardings(extend, shard);
        }
        finish(extend);
        return success();
      }

      std::optional<unsigned> removedDim = std::nullopt;
      auto peelReshape = [&](Value v) -> Value {
        if (auto reshape = v.getDefiningOp<stablehlo::ReshapeOp>()) {
          auto inShape = reshape.getOperand().getType().getShape();
          auto outShape = reshape.getResult().getType().getShape();
          if (inShape.size() != outShape.size() + 1)
            return nullptr;

          if (inShape[0] == 1) {
            bool legal = true;
            for (auto &&[lhs, rhs] :
                 llvm::zip_equal(outShape, inShape.slice(1))) {
              if (lhs != rhs) {
                legal = false;
                break;
              }
            }
            if (legal) {
              if (removedDim) {
                if (*removedDim != 0) {
                  return nullptr;
                }
              } else {
                removedDim = 0;
              }
              return reshape.getOperand();
            }
          }

          for (unsigned inI = 0, outI = 0; inI < inShape.size();) {
            if (outI == outShape.size())
              return nullptr;
            if (inShape[inI] == outShape[outI]) {
              inI++;
              outI++;
            } else if (inShape[inI] == 1) {
              if (removedDim == inI) {
                inI++;
              } else if (!removedDim) {
                removedDim = inI;
                inI++;
              } else {
                return nullptr;
              }
            } else {
              return nullptr;
            }
          }
          return reshape.getOperand();
        }
        return nullptr;
      };
      lhsv = peelReshape(lhsv);
      midv = peelReshape(midv);
      rhsv = peelReshape(rhsv);
      if (!lhsv || !midv || !rhsv || !removedDim) {
        continue;
      }
      unsigned reshapedDim = dim;
      if (reshapedDim >= *removedDim)
        reshapedDim++;

      if (succeeded(isExtendLike(reshapedDim, lhsv, midv, rhsv, concat.getLoc(),
                                 rewriter, &lhs, &mid, &rhs))) {
        auto midO = mid.getOutput();
        auto extend = rewriter.create<enzymexla::ExtendOp>(
            concat.getLoc(), midO, lhs.getOutputShape(reshapedDim),
            rhs.getOutputShape(reshapedDim), reshapedDim);
        if (auto midOp = midO.getDefiningOp())
          if (auto shard = sdy::getShardingPerValue(midOp)) {
            sdy::setShardings(extend, shard);
          }
        auto shape = llvm::to_vector(extend.getResult().getType().getShape());
        assert(shape[*removedDim] == 1);
        shape.erase(std::next(shape.begin(), *removedDim),
                    std::next(shape.begin(), *removedDim + 1));
        auto reshape = rewriter.create<stablehlo::ReshapeOp>(
            concat.getLoc(),
            RankedTensorType::get(
                shape, concat.getResult().getType().getElementType()),
            extend);
        if (auto shard = sdy::getShardingPerValue(concat)) {
          sdy::setShardings(reshape, shard);
        }
        finish(reshape);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(concat, "Could not find extend pattern");
  }
};

bool isAxisFusible(int dimension, ArrayRef<Value> vals) {
  assert(vals.size());

  for (int i = 0; i < vals.size(); i++) {
    if (auto concat = vals[i].getDefiningOp<stablehlo::ConcatenateOp>()) {
      if (concat.getDimension() == dimension)
        return true;
    }
  }

  for (int i = 1; i < vals.size(); i++) {
    auto sl0 = vals[i - 1].getDefiningOp<stablehlo::SliceOp>();
    auto sl1 = vals[i].getDefiningOp<stablehlo::SliceOp>();
    if (sl0 && sl1 && canMergeSlicesAlongAxis(dimension, sl0, sl1)) {
      return true;
    }
  }

  for (int i = 1; i < vals.size(); i++) {
    auto pad0 = vals[i - 1].getDefiningOp<stablehlo::PadOp>();
    auto pad1 = vals[i].getDefiningOp<stablehlo::PadOp>();
    if (pad0 && pad1 && canMergePadsAlongAxis(dimension, pad0, pad1)) {
      return true;
    }
  }

  for (int i = 1; i < vals.size(); i++) {
    auto sl0 = vals[i - 1].getDefiningOp<enzymexla::WrapOp>();
    auto sl1 = vals[i].getDefiningOp<enzymexla::WrapOp>();
    if (sl0 && sl1 && canMergeWrapsAlongAxis(dimension, sl0, sl1)) {
      return true;
    }
  }

  for (int i = 1; i < vals.size(); i++) {
    if (isRotateLike(dimension, vals[i - 1], vals[i])) {
      return true;
    }
  }

  for (int i = 2; i < vals.size(); i++) {
    if (isWrapLike(dimension, vals[i - 2], vals[i - 1], vals[i])) {
      return true;
    }
  }

  auto rewriter = IRRewriter(vals[0].getContext());
  for (int i = 2; i < vals.size(); i++) {
    if (succeeded(isExtendLike(dimension, vals[i - 2], vals[i - 1], vals[i],
                               vals[i].getLoc(), rewriter))) {
      return true;
    }
  }

  return false;
}

struct SliceExtend final
    : CheckedOpRewritePattern<enzymexla::ExtendOp, SliceExtend> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  struct CandidateInfo {
    enzymexla::ExtendOp extendOp;
    stablehlo::SliceOp sliceOp; // Null if it's a direct extend user
  };

  LogicalResult matchAndRewriteImpl(enzymexla::ExtendOp triggerExtendOp,
                                    PatternRewriter &rewriter) const {

    Value triggerOperand = triggerExtendOp.getOperand();
    auto triggerSliceOp = triggerOperand.getDefiningOp<stablehlo::SliceOp>();

    if (!triggerSliceOp) {
      return failure();
    }

    Value baseOperand = triggerSliceOp.getOperand();

    // --- Get Target Extension Parameters ---
    int64_t targetExtendDim = triggerExtendOp.getDimension();
    int64_t targetLhs = triggerExtendOp.getLhs();
    int64_t targetRhs = triggerExtendOp.getRhs();
    Location loc = triggerExtendOp.getLoc();

    // --- Check Validity of Trigger Slice ---
    auto baseOperandType = dyn_cast<RankedTensorType>(baseOperand.getType());
    if (!baseOperandType || !baseOperandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(triggerExtendOp,
                                         "Base operand requires static shape");
    }
    if (triggerSliceOp.getStartIndices()[targetExtendDim] != 0 ||
        triggerSliceOp.getLimitIndices()[targetExtendDim] !=
            baseOperandType.getShape()[targetExtendDim] ||
        triggerSliceOp.getStrides()[targetExtendDim] != 1) {
      return rewriter.notifyMatchFailure(
          triggerExtendOp,
          "Trigger SliceOp modifies the dimension being extended");
    }

    llvm::SmallVector<CandidateInfo> candidates;
    candidates.push_back({triggerExtendOp, triggerSliceOp});

    for (auto const &userOp : baseOperand.getUsers()) {
      // Skip the slice that defines the trigger operand
      if (userOp == triggerSliceOp.getOperation())
        continue;

      // Case 1: Direct Extend
      if (auto directExtend = dyn_cast<enzymexla::ExtendOp>(userOp)) {
        if (directExtend.getDimension() == targetExtendDim &&
            directExtend.getLhs() == targetLhs &&
            directExtend.getRhs() == targetRhs) {
          candidates.push_back({directExtend, nullptr});
        }
      }
      // Case 2: Extend of Slice
      else if (auto sliceUser = dyn_cast<stablehlo::SliceOp>(userOp)) {
        if (!sliceUser->hasOneUse())
          continue;
        auto extendOfSlice =
            dyn_cast<enzymexla::ExtendOp>(*sliceUser->user_begin());
        if (!extendOfSlice)
          continue;

        if (extendOfSlice.getDimension() == targetExtendDim &&
            extendOfSlice.getLhs() == targetLhs &&
            extendOfSlice.getRhs() == targetRhs) {
          // Check validity: sliceUser must not modify targetExtendDim
          if (sliceUser.getStartIndices()[targetExtendDim] == 0 &&
              sliceUser.getLimitIndices()[targetExtendDim] ==
                  baseOperandType.getShape()[targetExtendDim] &&
              sliceUser.getStrides()[targetExtendDim] == 1) {
            candidates.push_back({extendOfSlice, sliceUser});
          }
        }
      }
    }

    if (candidates.size() <= 1) {
      return rewriter.notifyMatchFailure(
          triggerExtendOp,
          "Rewrite condition not met (only found the trigger candidate)");
    }

    SmallVector<int64_t> newBaseExtendShape =
        llvm::to_vector(baseOperandType.getShape());
    newBaseExtendShape[targetExtendDim] += (targetLhs + targetRhs);
    auto newBaseExtendType = RankedTensorType::get(
        newBaseExtendShape, baseOperandType.getElementType());

    if (auto subOp = baseOperand.getDefiningOp())
      rewriter.setInsertionPointAfter(subOp);
    else
      rewriter.setInsertionPointToStart(
          cast<BlockArgument>(baseOperand).getOwner());

    auto newBaseExtendOp = rewriter.create<enzymexla::ExtendOp>(
        loc, newBaseExtendType, baseOperand, targetLhs, targetRhs,
        targetExtendDim);
    RankedTensorType newBaseExtendResultType = newBaseExtendOp.getType();

    for (const auto &candidate : candidates) {
      enzymexla::ExtendOp oldExtendOp = candidate.extendOp;
      stablehlo::SliceOp oldSliceOp = candidate.sliceOp; // Might be null

      if (!oldSliceOp) {
        // Direct Extend - Replace directly
        if (oldExtendOp.getResult().getType() == newBaseExtendOp.getType()) {
          rewriter.replaceOp(oldExtendOp, newBaseExtendOp);
        } else {
          auto castOp = rewriter.create<tensor::CastOp>(
              loc, oldExtendOp.getResult().getType(), newBaseExtendOp);
          rewriter.replaceOp(oldExtendOp, castOp);
        }
      } else {
        SmallVector<int64_t> newSliceStarts =
            llvm::to_vector(oldSliceOp.getStartIndices());
        SmallVector<int64_t> newSliceLimits =
            llvm::to_vector(oldSliceOp.getLimitIndices());
        SmallVector<int64_t> newSliceStrides =
            llvm::to_vector(oldSliceOp.getStrides());

        newSliceStarts[targetExtendDim] = 0;
        newSliceLimits[targetExtendDim] =
            newBaseExtendResultType.getDimSize(targetExtendDim);
        newSliceStrides[targetExtendDim] = 1;
        assert(newBaseExtendOp);
        rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(
            oldExtendOp, newBaseExtendOp, newSliceStarts, newSliceLimits,
            newSliceStrides);
      }
    }

    return success();
  }
};

struct SliceWrap final : CheckedOpRewritePattern<enzymexla::WrapOp, SliceWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  struct CandidateInfo {
    enzymexla::WrapOp wrapOp;
    stablehlo::SliceOp sliceOp; // Null if it's a direct wrap user
  };

  LogicalResult matchAndRewriteImpl(enzymexla::WrapOp triggerWrapOp,
                                    PatternRewriter &rewriter) const {

    Value triggerOperand = triggerWrapOp.getOperand();
    auto triggerSliceOp = triggerOperand.getDefiningOp<stablehlo::SliceOp>();

    if (!triggerSliceOp) {
      return failure();
    }

    Value baseOperand = triggerSliceOp.getOperand();

    // --- Get Target Wrap Parameters ---
    int64_t targetWrapDim = triggerWrapOp.getDimension();
    int64_t targetLhs = triggerWrapOp.getLhs();
    int64_t targetRhs = triggerWrapOp.getRhs();
    Location loc = triggerWrapOp.getLoc();

    // --- Check Validity of Trigger Slice ---
    auto baseOperandType = dyn_cast<RankedTensorType>(baseOperand.getType());
    if (!baseOperandType || !baseOperandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(triggerWrapOp,
                                         "Base operand requires static shape");
    }
    if (triggerSliceOp.getStartIndices()[targetWrapDim] != 0 ||
        triggerSliceOp.getLimitIndices()[targetWrapDim] !=
            baseOperandType.getShape()[targetWrapDim] ||
        triggerSliceOp.getStrides()[targetWrapDim] != 1) {
      return rewriter.notifyMatchFailure(
          triggerWrapOp,
          "Trigger SliceOp modifies the dimension being wrapped");
    }

    llvm::SmallVector<CandidateInfo> candidates;
    candidates.push_back({triggerWrapOp, triggerSliceOp});

    for (auto const &userOp : baseOperand.getUsers()) {
      // Skip the slice that defines the trigger operand
      if (userOp == triggerSliceOp.getOperation())
        continue;

      // Case 1: Direct Wrap
      if (auto directWrap = dyn_cast<enzymexla::WrapOp>(userOp)) {
        if (directWrap.getDimension() == targetWrapDim &&
            directWrap.getLhs() == targetLhs &&
            directWrap.getRhs() == targetRhs) {
          candidates.push_back({directWrap, nullptr});
        }
      }
      // Case 2: Wrap of Slice
      else if (auto sliceUser = dyn_cast<stablehlo::SliceOp>(userOp)) {
        if (!sliceUser->hasOneUse())
          continue;
        auto wrapOfSlice =
            dyn_cast<enzymexla::WrapOp>(*sliceUser->user_begin());
        if (!wrapOfSlice)
          continue;

        if (wrapOfSlice.getDimension() == targetWrapDim &&
            wrapOfSlice.getLhs() == targetLhs &&
            wrapOfSlice.getRhs() == targetRhs) {
          // Check validity: sliceUser must not modify targetWrapDim
          if (sliceUser.getStartIndices()[targetWrapDim] == 0 &&
              sliceUser.getLimitIndices()[targetWrapDim] ==
                  baseOperandType.getShape()[targetWrapDim] &&
              sliceUser.getStrides()[targetWrapDim] == 1) {
            candidates.push_back({wrapOfSlice, sliceUser});
          }
        }
      }
    }

    if (candidates.size() <= 1) {
      return rewriter.notifyMatchFailure(
          triggerWrapOp,
          "Rewrite condition not met (only found the trigger candidate)");
    }

    SmallVector<int64_t> newBaseWrapShape =
        llvm::to_vector(baseOperandType.getShape());
    newBaseWrapShape[targetWrapDim] += (targetLhs + targetRhs);
    auto newBaseWrapType = RankedTensorType::get(
        newBaseWrapShape, baseOperandType.getElementType());

    if (auto subOp = baseOperand.getDefiningOp())
      rewriter.setInsertionPointAfter(subOp);
    else
      rewriter.setInsertionPointToStart(
          cast<BlockArgument>(baseOperand).getOwner());

    auto newBaseWrapOp = rewriter.create<enzymexla::WrapOp>(
        loc, newBaseWrapType, baseOperand, targetLhs, targetRhs, targetWrapDim);
    Value newBaseWrapResult = newBaseWrapOp.getResult();
    RankedTensorType newBaseWrapResultType =
        cast<RankedTensorType>(newBaseWrapResult.getType());

    for (const auto &candidate : candidates) {
      enzymexla::WrapOp oldWrapOp = candidate.wrapOp;
      stablehlo::SliceOp oldSliceOp = candidate.sliceOp; // Might be null

      if (!oldSliceOp) {
        // Direct Wrap - Replace directly
        if (oldWrapOp.getResult().getType() == newBaseWrapResult.getType()) {
          rewriter.replaceOp(oldWrapOp, newBaseWrapResult);
        } else {
          auto castOp = rewriter.create<tensor::CastOp>(
              loc, oldWrapOp.getResult().getType(), newBaseWrapResult);
          rewriter.replaceOp(oldWrapOp, castOp);
        }
      } else {
        SmallVector<int64_t> newSliceStarts =
            llvm::to_vector(oldSliceOp.getStartIndices());
        SmallVector<int64_t> newSliceLimits =
            llvm::to_vector(oldSliceOp.getLimitIndices());
        SmallVector<int64_t> newSliceStrides =
            llvm::to_vector(oldSliceOp.getStrides());

        newSliceStarts[targetWrapDim] = 0;
        newSliceLimits[targetWrapDim] =
            newBaseWrapResultType.getDimSize(targetWrapDim);
        newSliceStrides[targetWrapDim] = 1;

        auto newSlice = rewriter.create<stablehlo::SliceOp>(
            oldWrapOp.getLoc(),
            oldWrapOp.getResult()
                .getType(), // Use original wrap op's result type
            newBaseWrapResult, newSliceStarts, newSliceLimits, newSliceStrides);
        rewriter.replaceAllOpUsesWith(oldWrapOp, newSlice.getResult());
      }
    }

    return success();
  }
};

// transpose(wrap) -> wrap(transpose)
struct TransposeWrap final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeWrap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    // Check if the operand is a wrap operation
    auto wrapOp = op.getOperand().getDefiningOp<enzymexla::WrapOp>();
    if (!wrapOp)
      return failure();

    // Get wrap operation parameters
    int64_t wrapDim = wrapOp.getDimension();
    int64_t lhs = wrapOp.getLhs();
    int64_t rhs = wrapOp.getRhs();

    // Get permutation array
    SmallVector<int64_t> permutation;
    for (auto val : op.getPermutation()) {
      permutation.push_back(static_cast<int64_t>(val));
    }

    // The new wrap dimension will be the permuted dimension
    int64_t newWrapDim = -1;
    for (size_t i = 0; i < permutation.size(); ++i) {
      if (permutation[i] == wrapDim) {
        newWrapDim = i;
        break;
      }
    }

    if (newWrapDim == -1)
      return failure();

    // First transpose the wrap's operand
    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), wrapOp.getOperand(), op.getPermutation());

    // Then create a new wrap operation on the transposed data
    auto newWrapType = op.getType();
    auto newWrapOp = rewriter.create<enzymexla::WrapOp>(
        op.getLoc(), newWrapType, newTranspose.getResult(), lhs, rhs,
        newWrapDim);

    // Replace the original op with the new wrap operation
    rewriter.replaceOp(op, newWrapOp);
    return success();
  }
};

// transpose(extend) -> extend(transpose)
struct TransposeExtend final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeExtend> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    // Check if the operand is an extend operation
    auto extendOp = op.getOperand().getDefiningOp<enzymexla::ExtendOp>();
    if (!extendOp)
      return failure();

    // Get extend operation parameters
    int64_t extendDim = extendOp.getDimension();
    int64_t lhs = extendOp.getLhs();
    int64_t rhs = extendOp.getRhs();

    // Get permutation array
    SmallVector<int64_t> permutation;
    for (auto val : op.getPermutation()) {
      permutation.push_back(static_cast<int64_t>(val));
    }

    // The new extend dimension will be the permuted dimension
    int64_t newExtendDim = -1;
    for (size_t i = 0; i < permutation.size(); ++i) {
      if (permutation[i] == extendDim) {
        newExtendDim = i;
        break;
      }
    }

    if (newExtendDim == -1)
      return failure();

    // First transpose the extend's operand
    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), extendOp.getOperand(), op.getPermutation());

    // Then create a new extend operation on the transposed data
    auto newExtendOp = rewriter.create<enzymexla::ExtendOp>(
        op.getLoc(), newTranspose.getResult(), lhs, rhs, newExtendDim);

    // Replace the original op with the new extend operation
    rewriter.replaceOp(op, newExtendOp);
    return success();
  }
};

// transpose(rotate) -> rotate(transpose)
struct TransposeRotate final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeRotate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    // Check if the operand is a rotate operation
    auto rotateOp = op.getOperand().getDefiningOp<enzymexla::RotateOp>();
    if (!rotateOp)
      return failure();

    // Get rotate operation parameters
    int64_t rotateDim = rotateOp.getDimension();
    int64_t amount = rotateOp.getAmount();

    // Get permutation array
    SmallVector<int64_t> permutation;
    for (auto val : op.getPermutation()) {
      permutation.push_back(static_cast<int64_t>(val));
    }

    // The new rotate dimension will be the permuted dimension
    int64_t newRotateDim = -1;
    for (size_t i = 0; i < permutation.size(); ++i) {
      if (permutation[i] == rotateDim) {
        newRotateDim = i;
        break;
      }
    }

    if (newRotateDim == -1)
      return failure();

    // First transpose the rotate's operand
    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), rotateOp.getOperand(), op.getPermutation());

    // Then create a new rotate operation on the transposed data
    auto newRotateOp = rewriter.create<enzymexla::RotateOp>(
        op.getLoc(), newTranspose.getResult(), amount, newRotateDim);

    // Replace the original op with the new rotate operation
    rewriter.replaceOp(op, newRotateOp);
    return success();
  }
};

struct ConcatConcatAxisSwap final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatConcatAxisSwap> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp outer,
                                    PatternRewriter &rewriter) const {
    if (outer.getOperands().size() < 2)
      return failure();

    SmallVector<stablehlo::ConcatenateOp> inners;

    // Get a full square of concat of compatible dimensions for axis swap
    for (auto v : outer.getOperands()) {
      auto concatOp = v.getDefiningOp<stablehlo::ConcatenateOp>();
      if (!concatOp)
        return failure();
      if (concatOp.getDimension() == outer.getDimension())
        return failure();
      if (concatOp.getOperands().size() < 2)
        return failure();
      if (inners.size()) {
        if (inners[0].getOperands().size() != concatOp.getOperands().size()) {
          return failure();
        }
        if (inners[0].getDimension() != concatOp.getDimension()) {
          return failure();
        }
        for (int i = 0; i < inners[0].getOperands().size(); i++) {
          if (cast<RankedTensorType>(concatOp.getOperands()[i].getType())
                  .getShape()[concatOp.getDimension()] !=
              cast<RankedTensorType>(inners[0].getOperands()[i].getType())
                  .getShape()[concatOp.getDimension()])
            return failure();
        }
      }
      inners.push_back(concatOp);
    }

    // Check that we don't have a current axis fuse opportunity, and wait for
    // those fusions
    for (auto inner : inners) {
      if (isAxisFusible(inner.getDimension(),
                        llvm::to_vector(inner.getOperands()))) {
        return failure();
      }
    }

    bool anyFusible = false;
    for (int i = 0; i < inners[0].getOperands().size(); i++) {
      SmallVector<Value> newOperands;
      for (int j = 0; j < outer.getOperands().size(); j++) {
        newOperands.push_back(inners[j].getOperands()[i]);
      }
      if (isAxisFusible(outer.getDimension(), newOperands)) {
        anyFusible = true;
        break;
      }
    }

    if (!anyFusible)
      return failure();

    SmallVector<Value> newOuters;

    for (int i = 0; i < inners[0].getOperands().size(); i++) {
      SmallVector<Value> newOperands;
      for (int j = 0; j < outer.getOperands().size(); j++) {
        newOperands.push_back(inners[j].getOperands()[i]);
      }
      newOuters.push_back(rewriter.create<stablehlo::ConcatenateOp>(
          outer.getLoc(), newOperands, outer.getDimension()));
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        outer, newOuters, inners[0].getDimension());
    return success();
  }
};

struct SliceRotate final
    : CheckedOpRewritePattern<enzymexla::RotateOp, SliceRotate> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  struct CandidateInfo {
    enzymexla::RotateOp rotateOp;
    stablehlo::SliceOp sliceOp; // Null if it's a direct extend user
  };

  LogicalResult matchAndRewriteImpl(enzymexla::RotateOp triggerRotateOp,
                                    PatternRewriter &rewriter) const {

    Value triggerOperand = triggerRotateOp.getOperand();
    auto triggerSliceOp = triggerOperand.getDefiningOp<stablehlo::SliceOp>();

    if (!triggerSliceOp) {
      return failure();
    }

    Value baseOperand = triggerSliceOp.getOperand();

    // --- Get Target Rotate Parameters ---
    int targetAmount = triggerRotateOp.getAmount();
    int targetRotateDim = triggerRotateOp.getDimension();
    Location loc = triggerRotateOp.getLoc();

    // --- Check Validity of Trigger Slice ---
    auto baseOperandType = dyn_cast<RankedTensorType>(baseOperand.getType());
    if (!baseOperandType || !baseOperandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(triggerRotateOp,
                                         "Base operand requires static shape");
    }
    if (triggerSliceOp.getStartIndices()[targetRotateDim] != 0 ||
        triggerSliceOp.getLimitIndices()[targetRotateDim] !=
            baseOperandType.getShape()[targetRotateDim] ||
        triggerSliceOp.getStrides()[targetRotateDim] != 1) {
      return rewriter.notifyMatchFailure(
          triggerRotateOp,
          "Trigger SliceOp modifies the dimension being extended");
    }

    llvm::SmallVector<CandidateInfo> candidates;
    candidates.push_back({triggerRotateOp, triggerSliceOp});

    for (auto const &userOp : baseOperand.getUsers()) {
      // Skip the slice that defines the trigger operand
      if (userOp == triggerSliceOp.getOperation())
        continue;

      // Case 1: Direct Rotate
      if (auto directRotate = dyn_cast<enzymexla::RotateOp>(userOp)) {
        if (directRotate.getDimension() == targetRotateDim &&
            directRotate.getAmount() == targetAmount) {
          candidates.push_back({directRotate, nullptr});
        }
      }
      // Case 2: Rotate of Slice
      else if (auto sliceUser = dyn_cast<stablehlo::SliceOp>(userOp)) {
        if (!sliceUser->hasOneUse())
          continue;
        auto rotateOfSlice =
            dyn_cast<enzymexla::RotateOp>(*sliceUser->user_begin());
        if (!rotateOfSlice)
          continue;

        if (rotateOfSlice.getDimension() == targetRotateDim &&
            rotateOfSlice.getAmount() == targetAmount) {
          // Check validity: sliceUser must not modify targetRotateDim
          if (sliceUser.getStartIndices()[targetRotateDim] == 0 &&
              sliceUser.getLimitIndices()[targetRotateDim] ==
                  baseOperandType.getShape()[targetRotateDim] &&
              sliceUser.getStrides()[targetRotateDim] == 1) {
            candidates.push_back({rotateOfSlice, sliceUser});
          }
        }
      }
    }

    if (candidates.size() <= 1) {
      return rewriter.notifyMatchFailure(
          triggerRotateOp,
          "Rewrite condition not met (only found the trigger candidate)");
    }

    SmallVector<int64_t> newBaseExtendShape =
        llvm::to_vector(baseOperandType.getShape());

    auto newBaseRotateType = baseOperandType;

    if (auto subOp = baseOperand.getDefiningOp())
      rewriter.setInsertionPointAfter(subOp);
    else
      rewriter.setInsertionPointToStart(
          cast<BlockArgument>(baseOperand).getOwner());

    auto newBaseRotateOp = rewriter.create<enzymexla::RotateOp>(
        loc, newBaseRotateType, baseOperand, targetAmount, targetRotateDim);
    Value newBaseRotateResult = newBaseRotateOp.getResult();
    RankedTensorType newBaseRotateResultType = newBaseRotateType;

    for (const auto &candidate : candidates) {
      enzymexla::RotateOp oldRotateOp = candidate.rotateOp;
      stablehlo::SliceOp oldSliceOp = candidate.sliceOp; // Might be null

      if (!oldSliceOp) {
        // Direct Rotate - Replace directly
        rewriter.replaceOp(oldRotateOp, newBaseRotateResult);
      } else {
        SmallVector<int64_t> newSliceStarts =
            llvm::to_vector(oldSliceOp.getStartIndices());
        SmallVector<int64_t> newSliceLimits =
            llvm::to_vector(oldSliceOp.getLimitIndices());
        SmallVector<int64_t> newSliceStrides =
            llvm::to_vector(oldSliceOp.getStrides());

        newSliceStarts[targetRotateDim] = 0;
        newSliceLimits[targetRotateDim] =
            newBaseRotateResultType.getDimSize(targetRotateDim);
        newSliceStrides[targetRotateDim] = 1;

        auto newSlice = rewriter.create<stablehlo::SliceOp>(
            oldRotateOp.getLoc(),
            oldRotateOp.getResult()
                .getType(), // Use original extend op's result type
            newBaseRotateResult, newSliceStarts, newSliceLimits,
            newSliceStrides);
        rewriter.replaceAllOpUsesWith(oldRotateOp, newSlice.getResult());
      }
    }

    return success();
  }
};

struct SquareAbsSimplify
    : public CheckedOpRewritePattern<stablehlo::MulOp, SquareAbsSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (lhs != rhs)
      return failure();

    auto absOp = lhs.getDefiningOp<stablehlo::AbsOp>();
    if (!absOp)
      return failure();

    auto operand = absOp.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType)
      return failure();

    if (isa<ComplexType>(operandType.getElementType())) {
      // abs(z)^2 = real(z * conj(z)) -- only applied if abs(z) is used in this
      // operation
      if (!isOnlyUsedInOperation(absOp, op))
        return failure();
      rewriter.replaceOpWithNewOp<stablehlo::RealOp>(
          op, rewriter.create<stablehlo::MulOp>(
                  op.getLoc(), operand,
                  rewriter.create<chlo::ConjOp>(op.getLoc(), operand)));
      return success();
    } else {
      // abs(x)^2 = x * x
      rewriter.replaceOpWithNewOp<stablehlo::MulOp>(op, operand, operand);
      return success();
    }
  }
};

struct ConcatReshapeSlice
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                     ConcatReshapeSlice> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concatOp,
                                    PatternRewriter &rewriter) const {
    auto concatDim = concatOp.getDimension();
    auto ndims = cast<RankedTensorType>(concatOp.getType()).getRank();

    // Ensure all operands are reshapes of slices
    SmallVector<stablehlo::SliceOp> sliceOps;
    SmallVector<stablehlo::ReshapeOp> reshapeOps;
    Value sourceTensor;

    for (auto operand : concatOp.getOperands()) {
      auto reshape = operand.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshape || (!reshape->hasOneUse()))
        return failure();

      if (cast<RankedTensorType>(reshape.getResult().getType())
              .getShape()[concatDim] != 1)
        return failure();

      auto slice = reshape.getOperand().getDefiningOp<stablehlo::SliceOp>();
      if ((!slice) || (!slice->hasOneUse()))
        return failure();

      // Make sure all slices come from the same source
      if (!sourceTensor) {
        sourceTensor = slice.getOperand();
      } else if (sourceTensor != slice.getOperand()) {
        return failure();
      }

      auto sliceStrides = slice.getStrides();
      for (int64_t i = 0; i < sliceStrides.size(); i++) {
        if (sliceStrides[i] != 1)
          return failure();
      }

      reshapeOps.push_back(reshape);
      sliceOps.push_back(slice);
    }

    SmallVector<int64_t> sliceStrides(ndims, 1);
    SmallVector<int64_t> sliceStarts, sliceLimits;
    int64_t srcSliceDim = -1;

    for (int i = 0; i < sliceOps.size(); i++) {
      auto sliceOp = sliceOps[i];
      auto reshapeOp = reshapeOps[i];

      auto sliceShape =
          cast<RankedTensorType>(sliceOp.getResult().getType()).getShape();
      auto reshapeShape =
          cast<RankedTensorType>(reshapeOp.getResult().getType()).getShape();

      if (sliceShape.size() != reshapeShape.size())
        return failure();

      int64_t singletonSliceDim = -1;
      int64_t nSingletonSlices = 0;
      for (int64_t i = 0; i < sliceShape.size(); i++) {
        if (sliceShape[i] == 1) {
          singletonSliceDim = i;
          nSingletonSlices++;
        }
      }

      if (nSingletonSlices != 1)
        return failure();

      if (srcSliceDim == -1) {
        srcSliceDim = singletonSliceDim;
        sliceStarts = llvm::to_vector(sliceOp.getStartIndices());
        sliceLimits = llvm::to_vector(sliceOp.getLimitIndices());
      } else {
        if (!canMergeSlicesAlongAxis(srcSliceDim, sliceOps[i - 1], sliceOp))
          return failure();
      }

      // Ensure that the reshape is a permutation of the slice
      SmallVector<int64_t> srcNoSingleton, dstNoSingleton;
      for (int64_t i = 0; i < sliceShape.size(); i++) {
        if (i == singletonSliceDim)
          continue;
        srcNoSingleton.push_back(sliceShape[i]);
      }
      for (int64_t i = 0; i < reshapeShape.size(); i++) {
        if (i == concatDim)
          continue;
        dstNoSingleton.push_back(reshapeShape[i]);
      }

      if (srcNoSingleton != dstNoSingleton)
        return failure();
    }

    int64_t startIndex = sliceOps[0].getStartIndices()[srcSliceDim];
    int64_t limitIndex =
        sliceOps[sliceOps.size() - 1].getLimitIndices()[srcSliceDim];
    sliceStarts[srcSliceDim] = startIndex;
    sliceLimits[srcSliceDim] = limitIndex;

    auto newSlice = rewriter.create<stablehlo::SliceOp>(
        concatOp.getLoc(), sourceTensor, sliceStarts, sliceLimits,
        sliceStrides);

    SmallVector<int64_t> mapping(ndims, 0);
    for (int64_t i = 0; i < ndims; i++) {
      mapping[i] = i;
    }
    mapping[srcSliceDim] = concatDim;
    if (srcSliceDim > concatDim) {
      for (int64_t i = concatDim; i < srcSliceDim; i++) { // shift right
        mapping[i]++;
      }
    } else {
      for (int64_t i = srcSliceDim + 1; i <= concatDim; i++) { // shift left
        mapping[i]--;
      }
    }

    SmallVector<int64_t> permutation(ndims, 0);
    for (int64_t i = 0; i < ndims; i++) {
      permutation[mapping[i]] = i;
    }

    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(concatOp, newSlice,
                                                        permutation);
    return success();
  }
};

bool reshapeOfEquivalentReduces(stablehlo::ReshapeOp reshapeOp,
                                SmallVector<stablehlo::ReduceOp> &allOperands,
                                SmallVector<Value> &reduceOpOperands) {
  auto rank =
      cast<RankedTensorType>(reshapeOp.getOperand().getType()).getRank();
  if (rank != 0)
    return false;

  auto reduceOp = reshapeOp.getOperand().getDefiningOp<stablehlo::ReduceOp>();
  if (!reduceOp)
    return false;

  if (!isOnlyUsedInOperation(reduceOp, reshapeOp))
    return false;

  if (reduceOp.getInputs().size() != 1)
    return false;

  if (allOperands.size() >= 1 &&
      !OperationEquivalence::isEquivalentTo(
          reduceOp, allOperands[0],
          OperationEquivalence::ignoreValueEquivalence, nullptr,
          OperationEquivalence::IgnoreLocations, nullptr))
    return false;

  reduceOpOperands.push_back(reduceOp.getInputs()[0]);
  allOperands.push_back(reduceOp);
  return true;
}

struct ConcatElementwise final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatElementwise> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concatOp,
                                    PatternRewriter &rewriter) const {
    if (concatOp.getNumOperands() <= 1)
      return failure();

    SmallVector<Operation *> concatOpOperands;

    for (auto [i, v] : llvm::enumerate(concatOp.getOperands())) {
      auto vdefOp = v.getDefiningOp();
      if (!vdefOp)
        return failure();

      if (isa<stablehlo::ConvertOp>(vdefOp)) // Conflicts with ConvertConcat
        return failure();

      if (vdefOp->hasTrait<mlir::OpTrait::Elementwise>()) {
        if (concatOpOperands.size() != 0) {
          if (!OperationEquivalence::isEquivalentTo(
                  concatOpOperands[0], vdefOp,
                  OperationEquivalence::ignoreValueEquivalence, nullptr,
                  OperationEquivalence::IgnoreLocations, nullptr))
            return failure();
        }

        if (!isOnlyUsedInOperation(vdefOp, concatOp))
          return failure();

        concatOpOperands.push_back(vdefOp);
      } else {
        return failure();
      }
    }

    SmallVector<Value> elementwiseOperands;

    for (int i = 0; i < concatOpOperands[0]->getNumOperands(); i++) {
      SmallVector<Value> newConcatOperands;
      for (auto v : concatOpOperands) {
        newConcatOperands.push_back(v->getOperand(i));
      }
      auto newConcatOp = rewriter.create<stablehlo::ConcatenateOp>(
          concatOp.getLoc(), newConcatOperands, concatOp.getDimension());
      elementwiseOperands.push_back(newConcatOp.getResult());
    }

    auto newElementwiseOp = rewriter.create(
        concatOp.getLoc(), concatOpOperands[0]->getName().getIdentifier(),
        ValueRange(elementwiseOperands), TypeRange(concatOp.getType()),
        concatOpOperands[0]->getAttrs(), {}, {});

    rewriter.replaceOp(concatOp, newElementwiseOp);
    return success();
  }
};

struct ConcatReshapeReduce final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatReshapeReduce> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concatOp,
                                    PatternRewriter &rewriter) const {
    if (concatOp.getNumOperands() <= 1)
      return failure();

    SmallVector<stablehlo::ReduceOp> allOperands;
    SmallVector<Value> reduceOpOperands;
    for (auto v : concatOp.getOperands()) {
      if (auto reshapeOp = v.getDefiningOp<stablehlo::ReshapeOp>()) {
        if (!isOnlyUsedInOperation(reshapeOp, concatOp))
          return failure();
        if (!reshapeOfEquivalentReduces(reshapeOp, allOperands,
                                        reduceOpOperands))
          return failure();
      } else {
        return rewriter.notifyMatchFailure(concatOp,
                                           "Operand is not a reshape.");
      }
    }

    auto reduceDims = llvm::to_vector(allOperands[0].getDimensions());

    auto concatDim = concatOp.getDimension();
    for (int64_t i = 0; i < reduceDims.size(); i++) {
      if (reduceDims[i] >= concatDim)
        reduceDims[i]++;
    }

    SmallVector<Value> newConcatOperands;
    auto elemTy =
        cast<RankedTensorType>(reduceOpOperands[0].getType()).getElementType();

    SmallVector<int64_t> oldShape = llvm::to_vector(
        cast<RankedTensorType>(reduceOpOperands[0].getType()).getShape());

    SmallVector<int64_t> preConcatShape(oldShape.size() + 1, 0);
    for (int64_t i = 0; i < preConcatShape.size(); i++) {
      if (i == concatDim) {
        preConcatShape[i] = 1;
      } else if (i < concatDim) {
        preConcatShape[i] = oldShape[i];
      } else {
        preConcatShape[i] = oldShape[i - 1];
      }
    }

    for (int64_t i = 0; i < reduceOpOperands.size(); i++) {
      newConcatOperands.push_back(rewriter.create<stablehlo::ReshapeOp>(
          concatOp.getLoc(), RankedTensorType::get(preConcatShape, elemTy),
          reduceOpOperands[i]));
    }

    auto newConcatOp = rewriter.create<stablehlo::ConcatenateOp>(
        concatOp.getLoc(), newConcatOperands, concatDim);

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        concatOp.getLoc(),
        TypeRange(RankedTensorType::get(
            {static_cast<int64_t>(reduceOpOperands.size())}, elemTy)),
        ValueRange(newConcatOp), ValueRange(allOperands[0].getInitValues()[0]),
        rewriter.getDenseI64ArrayAttr(reduceDims));

    // Clone the reduction body
    rewriter.inlineRegionBefore(allOperands[0].getBody(), reduceOp.getBody(),
                                reduceOp.getBody().end());

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        concatOp, concatOp.getType(), reduceOp.getResult(0));
    return success();
  }
};

// reverse(transpose x) -> transpose(reverse x)
struct ReverseTranspose final
    : CheckedOpRewritePattern<stablehlo::ReverseOp, ReverseTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReverseOp op,
                                    PatternRewriter &rewriter) const {
    auto transposeOp = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    if (!transposeOp->getResult(0).hasOneUse())
      return failure();

    SmallVector<int64_t> permutation =
        llvm::to_vector(transposeOp.getPermutation());

    SmallVector<int64_t> newReverseDims(op.getDimensions().size(), 0);
    for (int64_t i = 0; i < op.getDimensions().size(); i++) {
      newReverseDims[i] = permutation[op.getDimensions()[i]];
    }

    auto newReverseOp = rewriter.create<stablehlo::ReverseOp>(
        op.getLoc(), transposeOp.getOperand(), newReverseDims);
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, newReverseOp,
                                                        permutation);
    return success();
  }
};

// transpose(reverse x) -> reverse(transpose x)
struct TransposeReverse final
    : CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeReverse> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto reverseOp = op.getOperand().getDefiningOp<stablehlo::ReverseOp>();
    if (!reverseOp)
      return failure();

    if (!reverseOp->getResult(0).hasOneUse())
      return failure();

    SmallVector<int64_t> permutation = llvm::to_vector(op.getPermutation());

    SmallVector<int64_t> newReverseDims(reverseOp.getDimensions().size(), 0);
    for (int64_t i = 0; i < reverseOp.getDimensions().size(); i++) {
      newReverseDims[i] = permutation[reverseOp.getDimensions()[i]];
    }

    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), reverseOp.getOperand(), permutation);
    rewriter.replaceOpWithNewOp<stablehlo::ReverseOp>(op, newTranspose,
                                                      newReverseDims);
    return success();
  }
};

struct ElementwiseReshapeLike
    : public CheckedOpTraitRewritePattern<OpTrait::Elementwise,
                                          ElementwiseReshapeLike> {
  using CheckedOpTraitRewritePattern<
      OpTrait::Elementwise,
      ElementwiseReshapeLike>::CheckedOpTraitRewritePattern;

  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const {
    SmallVector<Value> parentOperands;
    Operation *operandOp = nullptr;
    for (auto operand : op->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!defOp)
        return failure();

      if (!isa<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(defOp))
        return failure();

      if (!isOnlyUsedInOperation(defOp, op))
        return rewriter.notifyMatchFailure(
            op, "operand is used in more than one op");

      if (!operandOp) {
        operandOp = defOp;
      } else {
        if (!OperationEquivalence::isEquivalentTo(
                operandOp, defOp, OperationEquivalence::ignoreValueEquivalence,
                nullptr, OperationEquivalence::IgnoreLocations, nullptr)) {
          return rewriter.notifyMatchFailure(
              op, "operand operations are not equivalent");
        }
      }

      parentOperands.push_back(defOp->getOperand(0));
    }

    if (parentOperands.size() == 0)
      return failure();

    auto elemOp = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), ValueRange(parentOperands),
        TypeRange{RankedTensorType::get(
            cast<RankedTensorType>(parentOperands[0].getType()).getShape(),
            cast<RankedTensorType>(op->getResult(0).getType())
                .getElementType())},
        op->getAttrs(), {}, {});
    auto reshapeLikeOp = rewriter.create(
        op->getLoc(), operandOp->getName().getIdentifier(),
        ValueRange{elemOp->getResult(0)}, TypeRange{op->getResult(0).getType()},
        operandOp->getAttrs(), {}, {});
    rewriter.replaceOp(op, reshapeLikeOp);
    return success();
  }
};

struct ConcatTranspose final
    : CheckedOpRewritePattern<stablehlo::ConcatenateOp, ConcatTranspose> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getNumOperands() < 2)
      return failure();

    SmallVector<Value> parentOperands;
    stablehlo::TransposeOp transposeOp;
    for (auto operand : op.getOperands()) {
      auto defOp = operand.getDefiningOp<stablehlo::TransposeOp>();
      if (!defOp)
        return failure();

      if (!isOnlyUsedInOperation(defOp, op))
        return rewriter.notifyMatchFailure(
            op, "operand is used in more than one op");

      if (!transposeOp) {
        transposeOp = defOp;
      } else {
        if (!OperationEquivalence::isEquivalentTo(
                transposeOp, defOp,
                OperationEquivalence::ignoreValueEquivalence, nullptr,
                OperationEquivalence::IgnoreLocations, nullptr)) {
          return rewriter.notifyMatchFailure(
              op, "operand operations are not equivalent");
        }
      }

      parentOperands.push_back(defOp->getOperand(0));
    }

    auto permutation = transposeOp.getPermutation();
    auto newConcatDim = permutation[op.getDimension()];

    auto newConcat = rewriter.create<stablehlo::ConcatenateOp>(
        op.getLoc(), parentOperands, newConcatDim);
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(op, newConcat,
                                                        permutation);
    return success();
  }
};

struct ReduceReduce final
    : CheckedOpRewritePattern<stablehlo::ReduceOp, ReduceReduce> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReduceOp op,
                                    PatternRewriter &rewriter) const {
    if (op.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce op has more than one input. not yet supported");

    auto redOp = op.getInputs()[0].getDefiningOp<stablehlo::ReduceOp>();
    if (!redOp)
      return failure();

    if (redOp.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce op has more than one input. not yet supported");

    if (!llvm::hasSingleElement(op.getInputs()[0].getUses()))
      return failure();

    if (!OperationEquivalence::isEquivalentTo(
            redOp.getInitValues()[0].getDefiningOp(),
            op.getInitValues()[0].getDefiningOp(),
            OperationEquivalence::IgnoreLocations))
      return rewriter.notifyMatchFailure(
          op, "reduce op init values are not equivalent");

    if (!OperationEquivalence::isRegionEquivalentTo(
            &redOp.getBody(), &op.getBody(),
            OperationEquivalence::IgnoreLocations))
      return failure();

    SmallVector<int64_t> innerDimensions =
        llvm::to_vector(redOp.getDimensions());
    SmallVector<int64_t> outerDimensions = llvm::to_vector(op.getDimensions());
    SmallVector<int64_t> mergedDimensions = innerDimensions;

    int64_t inputRank =
        cast<RankedTensorType>(redOp.getInputs()[0].getType()).getRank();

    llvm::SmallBitVector reduced(inputRank, false);
    SmallVector<int64_t> dimensionMap;

    for (auto dim : innerDimensions)
      reduced.set(dim);
    for (int64_t i = 0; i < inputRank; ++i) {
      if (!reduced[i])
        dimensionMap.push_back(i);
    }
    for (auto dim : outerDimensions) {
      if (dim >= dimensionMap.size())
        return rewriter.notifyMatchFailure(op, "out of bounds dimension map");

      mergedDimensions.push_back(dimensionMap[dim]);
    }

    auto newReduceOp = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), TypeRange(op.getType(0)), ValueRange(redOp.getInputs()),
        ValueRange(redOp.getInitValues()), mergedDimensions);
    rewriter.inlineRegionBefore(redOp.getBody(), newReduceOp.getBody(),
                                newReduceOp.getBody().end());
    rewriter.replaceOp(op, newReduceOp.getResult(0));
    rewriter.eraseOp(redOp);
    return success();
  }
};

struct ConjReal final : public CheckedOpRewritePattern<chlo::ConjOp, ConjReal> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  bool supportsDynamicShapes() { return true; }

  LogicalResult matchAndRewriteImpl(chlo::ConjOp op,
                                    PatternRewriter &rewriter) const {
    auto input = op.getOperand();
    auto elemType = cast<RankedTensorType>(input.getType()).getElementType();
    if (isa<ComplexType>(elemType))
      return rewriter.notifyMatchFailure(op, "can't apply to complex numbers");

    rewriter.replaceAllUsesWith(op.getResult(), input);
    return success();
  }
};

struct ConcatReshapeElementwise final
    : public CheckedOpRewritePattern<stablehlo::ConcatenateOp,
                                     ConcatReshapeElementwise> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConcatenateOp concatOp,
                                    PatternRewriter &rewriter) const {
    if (concatOp.getNumOperands() <= 1)
      return failure();

    auto concatDim = concatOp.getDimension();

    SmallVector<Operation *> concatOpOperands;

    for (auto [i, v] : llvm::enumerate(concatOp.getOperands())) {
      auto reshapeOp = v.getDefiningOp<stablehlo::ReshapeOp>();
      if (!reshapeOp)
        return rewriter.notifyMatchFailure(concatOp, "not a reshape op");

      RankedTensorType reshapeOpInputType = reshapeOp.getOperand().getType();
      RankedTensorType reshapeOpOutputType = reshapeOp.getResult().getType();

      SmallVector<int64_t> insertionDims =
          findReshapeInsertionDims(reshapeOpInputType, reshapeOpOutputType);

      if (insertionDims.size() != 1)
        return rewriter.notifyMatchFailure(
            concatOp, "reshape op has more than one insertion dim");

      if (insertionDims[0] != concatDim)
        return rewriter.notifyMatchFailure(
            concatOp, "concat dim is not same as insertion dim");

      auto vdefOp = reshapeOp.getOperand().getDefiningOp();
      if (!vdefOp)
        return failure();

      if (vdefOp->hasTrait<mlir::OpTrait::Elementwise>()) {
        if (concatOpOperands.size() != 0) {
          if (!OperationEquivalence::isEquivalentTo(
                  concatOpOperands[0], vdefOp,
                  OperationEquivalence::ignoreValueEquivalence, nullptr,
                  OperationEquivalence::IgnoreLocations, nullptr))
            return rewriter.notifyMatchFailure(
                concatOp, "elementwise op is not equivalent to first");
        }

        if (!isOnlyUsedInOperation(vdefOp, reshapeOp))
          return rewriter.notifyMatchFailure(
              concatOp, "elementwise op is not only used in reshape op");

        concatOpOperands.push_back(vdefOp);
      } else {
        return rewriter.notifyMatchFailure(concatOp, "not a valid elementwise");
      }
    }

    SmallVector<Value> elementwiseOperands;

    for (int i = 0; i < concatOpOperands[0]->getNumOperands(); i++) {
      SmallVector<Value> newConcatOperands;
      for (auto v : concatOpOperands) {
        auto inputOp = v->getOperand(i);

        auto inputType = cast<RankedTensorType>(inputOp.getType());
        auto inputShape = inputType.getShape();
        SmallVector<int64_t> outputShape;
        for (int j = 0; j < concatDim; j++)
          outputShape.push_back(inputShape[j]);
        outputShape.push_back(1);
        for (int j = concatDim + 1; j < (inputShape.size() + 1); j++)
          outputShape.push_back(inputShape[j - 1]);

        auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
            concatOp.getLoc(),
            RankedTensorType::get(outputShape, inputType.getElementType()),
            inputOp);
        newConcatOperands.push_back(newReshapeOp.getResult());
      }
      auto newConcatOp = rewriter.create<stablehlo::ConcatenateOp>(
          concatOp.getLoc(), newConcatOperands, concatDim);
      elementwiseOperands.push_back(newConcatOp.getResult());
    }

    auto newElementwiseOp = rewriter.create(
        concatOp.getLoc(), concatOpOperands[0]->getName().getIdentifier(),
        ValueRange(elementwiseOperands), TypeRange(concatOp.getType()),
        concatOpOperands[0]->getAttrs(), {}, {});

    rewriter.replaceOp(concatOp, newElementwiseOp);
    return success();
  }
};

struct TransposeBatchNormTraining final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeBatchNormTraining> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto batchnormTrainingOp =
        op.getOperand().getDefiningOp<stablehlo::BatchNormTrainingOp>();
    if (!batchnormTrainingOp)
      return failure();

    if (op.getOperand() != batchnormTrainingOp.getOutput())
      return failure();

    auto permutation = op.getPermutation();
    auto newFeatureIndex = permutation[batchnormTrainingOp.getFeatureIndex()];

    rewriter.setInsertionPoint(batchnormTrainingOp);
    auto newTransposeOp = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), batchnormTrainingOp.getOperand(), permutation);
    auto newBatchNormOp = rewriter.create<stablehlo::BatchNormTrainingOp>(
        op.getLoc(), newTransposeOp.getResult(), batchnormTrainingOp.getScale(),
        batchnormTrainingOp.getOffset(), batchnormTrainingOp.getEpsilon(),
        newFeatureIndex);

    rewriter.replaceAllUsesWith(op->getResult(0), newBatchNormOp->getResult(0));
    rewriter.replaceAllUsesWith(batchnormTrainingOp->getResult(1),
                                newBatchNormOp->getResult(1));
    rewriter.replaceAllUsesWith(batchnormTrainingOp->getResult(2),
                                newBatchNormOp->getResult(2));

    return success();
  }
};

struct TransposeBatchNormInference final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeBatchNormInference> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto batchnormInferenceOp =
        op.getOperand().getDefiningOp<stablehlo::BatchNormInferenceOp>();
    if (!batchnormInferenceOp)
      return failure();

    if (op.getOperand() != batchnormInferenceOp.getResult())
      return failure();

    auto permutation = op.getPermutation();
    auto newFeatureIndex = permutation[batchnormInferenceOp.getFeatureIndex()];

    auto newTransposeOp = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), batchnormInferenceOp.getOperand(), permutation);
    rewriter.replaceOpWithNewOp<stablehlo::BatchNormInferenceOp>(
        op, newTransposeOp.getResult(), batchnormInferenceOp.getScale(),
        batchnormInferenceOp.getOffset(), batchnormInferenceOp.getMean(),
        batchnormInferenceOp.getVariance(), batchnormInferenceOp.getEpsilon(),
        newFeatureIndex);
    return success();
  }
};

struct TransposeBatchNormGrad final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp,
                                     TransposeBatchNormGrad> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto batchnormGradOp =
        op.getOperand().getDefiningOp<stablehlo::BatchNormGradOp>();
    if (!batchnormGradOp)
      return failure();

    if (op.getOperand() != batchnormGradOp.getGradOperand())
      return failure();

    auto permutation = op.getPermutation();
    auto newFeatureIndex = permutation[batchnormGradOp.getFeatureIndex()];

    rewriter.setInsertionPoint(batchnormGradOp);
    auto transposeOperand = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), batchnormGradOp.getOperand(), permutation);
    auto transposeGradOutput = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), batchnormGradOp.getGradOutput(), permutation);

    auto newBatchNormOp = rewriter.create<stablehlo::BatchNormGradOp>(
        op.getLoc(), transposeOperand, batchnormGradOp.getScale(),
        batchnormGradOp.getMean(), batchnormGradOp.getVariance(),
        transposeGradOutput, batchnormGradOp.getEpsilon(), newFeatureIndex);

    rewriter.replaceAllUsesWith(op->getResult(0), newBatchNormOp->getResult(0));
    rewriter.replaceAllUsesWith(batchnormGradOp->getResult(1),
                                newBatchNormOp->getResult(1));
    rewriter.replaceAllUsesWith(batchnormGradOp->getResult(2),
                                newBatchNormOp->getResult(2));
    return success();
  }
};

struct SelectBroadcastInDim final
    : public CheckedOpRewritePattern<stablehlo::SelectOp,
                                     SelectBroadcastInDim> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::SelectOp op,
                                    PatternRewriter &rewriter) const {
    auto bcastOp = op.getPred().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcastOp)
      return failure();

    auto bcastOpOperand = bcastOp.getOperand();
    auto bcastOpOperandType = cast<RankedTensorType>(bcastOpOperand.getType());
    auto bcastOpOperandShape = bcastOpOperandType.getShape();

    if (bcastOpOperandShape.size() == 0) {
      // 0-dim tensor
      rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
          op, bcastOpOperand, op.getOnTrue(), op.getOnFalse());
      return success();
    } else if (bcastOpOperandShape.size() == 1) {
      if (bcastOpOperandShape[0] != 1)
        return failure();

      // 1-dim tensor of size (1,)
      auto reshapedPred = rewriter.create<stablehlo::ReshapeOp>(
          op.getLoc(),
          RankedTensorType::get({}, bcastOpOperandType.getElementType()),
          bcastOpOperand);
      rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
          op, reshapedPred, op.getOnTrue(), op.getOnFalse());
      return success();
    }

    return failure();
  }
};

struct TransposeIf final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeIf> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto ifOp = op.getOperand().getDefiningOp<stablehlo::IfOp>();
    if (!ifOp)
      return failure();
    if (!llvm::hasSingleElement(op.getOperand().getUses()))
      return failure();

    auto opRes = cast<OpResult>(op.getOperand());
    ssize_t opIdx = opRes.getResultNumber();

    SmallVector<Type> ifResultTypes = llvm::to_vector(ifOp.getResultTypes());
    ifResultTypes[opIdx] = op.getType();

    rewriter.setInsertionPoint(ifOp);
    auto newIfOp = rewriter.create<stablehlo::IfOp>(op.getLoc(), ifResultTypes,
                                                    ifOp.getPred());

    Operation *trueTerm = ifOp.getTrueBranch().front().getTerminator();
    Operation *falseTerm = ifOp.getFalseBranch().front().getTerminator();

    rewriter.setInsertionPoint(trueTerm);
    auto newTrue = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), trueTerm->getOperands()[opIdx], op.getPermutation());
    rewriter.modifyOpInPlace(trueTerm,
                             [&] { trueTerm->setOperand(opIdx, newTrue); });

    rewriter.setInsertionPoint(falseTerm);
    auto newFalse = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), falseTerm->getOperands()[opIdx], op.getPermutation());
    rewriter.modifyOpInPlace(falseTerm,
                             [&] { falseTerm->setOperand(opIdx, newFalse); });

    newIfOp.getTrueBranch().takeBody(ifOp.getTrueBranch());
    newIfOp.getFalseBranch().takeBody(ifOp.getFalseBranch());

    for (int i = 0; i < ifOp.getNumResults(); i++) {
      if (i == opIdx) {
        rewriter.replaceAllUsesWith(op.getResult(), newIfOp.getResult(i));
      } else {
        rewriter.replaceAllUsesWith(ifOp.getResult(i), newIfOp.getResult(i));
      }
    }

    rewriter.eraseOp(op);
    rewriter.eraseOp(ifOp);
    return success();
  }
};

struct IfOpLiftCommonOps final
    : public CheckedOpRewritePattern<stablehlo::IfOp, IfOpLiftCommonOps> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::IfOp op,
                                    PatternRewriter &rewriter) const {
    mlir::Region &trueRegion = op.getTrueBranch();
    mlir::Region &falseRegion = op.getFalseBranch();

    SmallVector<std::pair<Operation *, Operation *>> opsToLift;
    llvm::SmallPtrSet<Operation *, 8> falseOpsMatched;

    auto &trueBlock = trueRegion.front();
    auto &falseBlock = falseRegion.front();

    for (auto trueIt = trueBlock.begin();
         !trueIt->hasTrait<mlir::OpTrait::IsTerminator>(); ++trueIt) {
      Operation *trueOp = &*trueIt;

      for (auto falseIt = falseBlock.begin();
           !falseIt->hasTrait<mlir::OpTrait::IsTerminator>(); ++falseIt) {
        Operation *falseOp = &*falseIt;

        if (falseOpsMatched.contains(falseOp))
          continue;

        if (OperationEquivalence::isEquivalentTo(
                trueOp, falseOp, OperationEquivalence::IgnoreLocations)) {
          opsToLift.emplace_back(trueOp, falseOp);
          falseOpsMatched.insert(falseOp);
          break;
        }
      }
    }

    if (opsToLift.empty())
      return rewriter.notifyMatchFailure(op, "no common ops found");

    for (auto [trueOp, falseOp] : opsToLift) {
      rewriter.modifyOpInPlace(trueOp, [&]() { trueOp->moveBefore(op); });
      rewriter.replaceOp(falseOp, trueOp);
    }

    return success();
  }
};

// used for ops that dont define the Involution trait
template <typename OpTy>
struct InvolutionSimplify
    : public CheckedOpRewritePattern<OpTy, InvolutionSimplify<OpTy>> {
  using CheckedOpRewritePattern<
      OpTy, InvolutionSimplify<OpTy>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter) const {
    auto operandOp = op.getOperand().template getDefiningOp<OpTy>();
    if (!operandOp)
      return failure();

    rewriter.replaceOp(op, operandOp.getOperand());
    return success();
  }
};

struct RealConjSimplify final
    : public CheckedOpRewritePattern<stablehlo::RealOp, RealConjSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::RealOp op,
                                    PatternRewriter &rewriter) const {
    auto operandOp = op.getOperand().getDefiningOp<chlo::ConjOp>();
    if (!operandOp)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::RealOp>(op, operandOp.getOperand());
    return success();
  }
};

struct ConjComplexSimplify final
    : public CheckedOpRewritePattern<chlo::ConjOp, ConjComplexSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(chlo::ConjOp op,
                                    PatternRewriter &rewriter) const {
    auto operandOp = op.getOperand().getDefiningOp<stablehlo::ComplexOp>();
    if (!operandOp)
      return failure();

    auto rhs = operandOp.getRhs();
    auto rhsConstantOp = rhs.getDefiningOp<stablehlo::ConstantOp>();
    if (!rhsConstantOp)
      return failure();

    auto negateRhs = rewriter.create<stablehlo::NegOp>(op.getLoc(), rhs);
    rewriter.replaceOpWithNewOp<stablehlo::ComplexOp>(op, operandOp.getLhs(),
                                                      negateRhs);
    return success();
  }
};

// workaround for https://github.com/openxla/xla/issues/27446
struct SplitConvolutionIntoReverseConvolution final
    : public CheckedOpRewritePattern<stablehlo::ConvolutionOp,
                                     SplitConvolutionIntoReverseConvolution> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ConvolutionOp op,
                                    PatternRewriter &rewriter) const {
    auto windowReversal = op.getWindowReversal();
    if (!windowReversal)
      return rewriter.notifyMatchFailure(op, "no window reversal");

    if (!llvm::any_of(windowReversal.value(), [](bool b) { return b; }))
      return rewriter.notifyMatchFailure(op, "all window reversals are false");

    auto convDims = op.getDimensionNumbers();

    SmallVector<int64_t> reversalDims;
    for (auto [dim, reversal] : llvm::zip(convDims.getKernelSpatialDimensions(),
                                          windowReversal.value())) {
      if (reversal) {
        reversalDims.push_back(dim);
      }
    }

    auto reverseOp = rewriter.create<stablehlo::ReverseOp>(
        op.getLoc(), op.getRhs(), reversalDims);

    rewriter.replaceOpWithNewOp<stablehlo::ConvolutionOp>(
        op, op.getType(), op.getLhs(), reverseOp.getResult(),
        op.getWindowStridesAttr(), op.getPaddingAttr(), op.getLhsDilationAttr(),
        op.getRhsDilationAttr(), nullptr, op.getDimensionNumbersAttr(),
        op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr(),
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct ScatterMultiplySimplify final
    : public CheckedOpRewritePattern<stablehlo::MulOp,
                                     ScatterMultiplySimplify> {
  using CheckedOpRewritePattern<
      stablehlo::MulOp, ScatterMultiplySimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    stablehlo::ScatterOp scatterOp;
    mlir::Value otherValue;

    auto lhsScatterOp = lhs.getDefiningOp<stablehlo::ScatterOp>();
    auto rhsScatterOp = rhs.getDefiningOp<stablehlo::ScatterOp>();
    if (!lhsScatterOp && !rhsScatterOp) {
      return failure();
    } else {
      if (lhsScatterOp) {
        scatterOp = lhsScatterOp;
        otherValue = rhs;
      } else {
        scatterOp = rhsScatterOp;
        otherValue = lhs;
      }
    }

    if (scatterOp.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "ScatterOp with more than one input not supported");

    auto input = scatterOp.getInputs()[0];
    if (!matchPattern(input, m_AnyZeroFloat()) &&
        !matchPattern(input, m_Zero()))
      return rewriter.notifyMatchFailure(op, "ScatterOp with non-zero input");

    if (!scatterOp.getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(op, "ScatterOp with multiple uses");

    if (!isScatterSetindexOp(scatterOp))
      return rewriter.notifyMatchFailure(op, "ScatterOp with non-setindex");

    auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();

    SmallVector<int64_t> sliceSizes = computeGatherSliceSizes(scatterOp);

    auto gatheredValues = rewriter.create<stablehlo::GatherOp>(
        op.getLoc(), otherValue, scatterOp.getScatterIndices(),
        getGatherDims(rewriter.getContext(), scatterDimNumbers),
        rewriter.getDenseI64ArrayAttr(sliceSizes),
        scatterOp.getIndicesAreSortedAttr());

    auto newUpdates = rewriter.create<stablehlo::MulOp>(
        op.getLoc(), gatheredValues, scatterOp.getUpdates()[0]);

    auto newScatterOp = rewriter.create<stablehlo::ScatterOp>(
        op.getLoc(), scatterOp.getResultTypes(), scatterOp.getInputs(),
        scatterOp.getScatterIndices(), ValueRange(newUpdates),
        scatterOp.getScatterDimensionNumbersAttr(),
        scatterOp.getIndicesAreSortedAttr(), scatterOp.getUniqueIndicesAttr());
    newScatterOp.getUpdateComputation().takeBody(
        scatterOp.getUpdateComputation());
    rewriter.replaceOp(op, newScatterOp);

    return success();
  }
};

struct GatherConstProp final
    : public CheckedOpRewritePattern<stablehlo::GatherOp, GatherConstProp> {
  using CheckedOpRewritePattern<stablehlo::GatherOp,
                                GatherConstProp>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::GatherOp op,
                                    PatternRewriter &rewriter) const {
    DenseElementsAttr operandAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&operandAttr)))
      return rewriter.notifyMatchFailure(op,
                                         "GatherOp with non-constant input");

    if (operandAttr.isSplat()) {
      // In this case the indices don't matter and we can construct a new
      // splatted result
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), operandAttr.resizeSplat(op.getType()));
      return success();
    }

    DenseElementsAttr startIndicesAttr;
    if (!matchPattern(op.getStartIndices(), m_Constant(&startIndicesAttr))) {
      return rewriter.notifyMatchFailure(
          op, "GatherOp with non-constant start indices and unsplatted input");
    }

    stablehlo::Tensor operandTensor = stablehlo::constantOp(operandAttr);
    stablehlo::Tensor startIndicesTensor =
        stablehlo::constantOp(startIndicesAttr);
    auto gatherDims = op.getDimensionNumbers();

    auto sliceSizes = op.getSliceSizes();
    auto elementType = rewriter.getIntegerType(64);
    auto attrType =
        RankedTensorType::get({(int64_t)sliceSizes.size()}, elementType);
    auto sliceSizesAttr = DenseElementsAttr::get(attrType, sliceSizes);

    auto result = stablehlo::gatherOp(
        operandTensor, startIndicesTensor,
        stablehlo::Axes(gatherDims.getOffsetDims()),
        stablehlo::Axes(gatherDims.getCollapsedSliceDims()),
        stablehlo::Axes(gatherDims.getOperandBatchingDims()),
        stablehlo::Axes(gatherDims.getStartIndicesBatchingDims()),
        stablehlo::Axes(gatherDims.getStartIndexMap()),
        stablehlo::Axis(gatherDims.getIndexVectorDim()),
        stablehlo::makeSizes(stablehlo::constantOp(sliceSizesAttr)),
        op.getIndicesAreSorted(), op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                       fromTensor(result));
    return success();
  }
};

struct UnaryElementwiseScatterSimplify final
    : public CheckedOpTraitRewritePattern<OpTrait::Elementwise,
                                          UnaryElementwiseScatterSimplify> {
  using CheckedOpTraitRewritePattern<
      OpTrait::Elementwise,
      UnaryElementwiseScatterSimplify>::CheckedOpTraitRewritePattern;

  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const {
    if (op->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "not a unary elementwise op");

    auto input = op->getOperand(0);
    auto scatterOp = input.getDefiningOp<stablehlo::ScatterOp>();
    if (!scatterOp)
      return rewriter.notifyMatchFailure(op, "not a scatter op");

    if (scatterOp.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(
          op, "ScatterOp with more than one input not supported");

    if (!scatterOp.getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(op, "ScatterOp with multiple uses");

    if (!isScatterSetindexOp(scatterOp))
      return rewriter.notifyMatchFailure(op, "ScatterOp with non-setindex");

    auto scatterInput = scatterOp.getInputs()[0];
    DenseElementsAttr scatterInputAttr;
    // In this case, we are will definitely increase the compute cost
    if (!matchPattern(scatterInput, m_Constant(&scatterInputAttr)))
      return rewriter.notifyMatchFailure(op,
                                         "ScatterOp with non-constant input");

    auto elemType =
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType();

    // should get constant propagated
    auto scatterInputElem = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(), ValueRange(scatterInput),
        TypeRange{RankedTensorType::get(
            cast<RankedTensorType>(scatterInput.getType()).getShape(),
            elemType)},
        op->getAttrs(), {}, {});

    auto scatterUpdatesElem = rewriter.create(
        op->getLoc(), op->getName().getIdentifier(),
        ValueRange(scatterOp.getUpdates()),
        TypeRange{RankedTensorType::get(
            cast<RankedTensorType>(scatterOp.getUpdates()[0].getType())
                .getShape(),
            elemType)},
        op->getAttrs(), {}, {});

    auto resultType = RankedTensorType::get(
        cast<RankedTensorType>(scatterOp.getResultTypes()[0]).getShape(),
        elemType);

    auto newScatterOp = rewriter.create<stablehlo::ScatterOp>(
        op->getLoc(), TypeRange(resultType),
        ValueRange(scatterInputElem->getResult(0)),
        scatterOp.getScatterIndices(),
        ValueRange(scatterUpdatesElem->getResult(0)),
        scatterOp.getScatterDimensionNumbersAttr(),
        scatterOp.getIndicesAreSortedAttr(), scatterOp.getUniqueIndicesAttr());

    auto &updateRegion = newScatterOp.getUpdateComputation();
    auto *block = rewriter.createBlock(&updateRegion);
    auto argType = RankedTensorType::get({}, elemType);
    block->addArgument(argType, op->getLoc());
    block->addArgument(argType, op->getLoc());
    rewriter.setInsertionPointToStart(block);
    rewriter.create<stablehlo::ReturnOp>(op->getLoc(), block->getArgument(1));

    rewriter.replaceOp(op, newScatterOp->getResult(0));
    return success();
  }
};

struct GatherElementwise
    : public CheckedOpRewritePattern<stablehlo::GatherOp, GatherElementwise> {
  using CheckedOpRewritePattern<stablehlo::GatherOp,
                                GatherElementwise>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::GatherOp op,
                                    PatternRewriter &rewriter) const {
    auto gatherInput = op.getOperand();
    auto defOp = gatherInput.getDefiningOp();
    if (!defOp || !defOp->hasTrait<mlir::OpTrait::Elementwise>())
      return rewriter.notifyMatchFailure(op,
                                         "GatherOp with non-elementwise input");

    if (!isOnlyUsedInOperation(defOp, op))
      return failure();

    int64_t outElemCount = 1, inElemCount = 1;
    for (auto dim : cast<RankedTensorType>(op.getType()).getShape())
      outElemCount *= dim;
    for (auto dim : cast<RankedTensorType>(gatherInput.getType()).getShape())
      inElemCount *= dim;

    if (outElemCount >= inElemCount)
      return rewriter.notifyMatchFailure(
          op, "GatherOp has more output elements than input elements");

    SmallVector<Value> newElementwiseInputs;
    for (auto input : defOp->getOperands()) {
      auto neeGatherOp = rewriter.create<stablehlo::GatherOp>(
          op.getLoc(), input, op.getStartIndices(),
          op.getDimensionNumbersAttr(), op.getSliceSizesAttr(),
          op.getIndicesAreSortedAttr());
      newElementwiseInputs.push_back(neeGatherOp->getResult(0));
    }

    auto newElemOp = rewriter.create(
        op.getLoc(), defOp->getName().getIdentifier(),
        ValueRange(newElementwiseInputs), TypeRange{op.getResult().getType()},
        defOp->getAttrs(), {}, {});
    rewriter.replaceOp(op, newElemOp->getResult(0));
    return success();
  }
};

struct ChainedMultiplyToPower final
    : public CheckedOpRewritePattern<stablehlo::MulOp, ChainedMultiplyToPower> {
  using CheckedOpRewritePattern<
      stablehlo::MulOp, ChainedMultiplyToPower>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsDefOp = lhs.getDefiningOp<stablehlo::MulOp>();
    auto rhsDefOp = rhs.getDefiningOp<stablehlo::MulOp>();

    if (lhsDefOp && isOnlyUsedInOperation(lhsDefOp, op)) {
      auto lhsLhs = lhsDefOp.getLhs();
      auto lhsRhs = lhsDefOp.getRhs();
      if (lhsLhs == lhsRhs) {
        if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) {
          auto rhsLhs = rhsDefOp.getLhs();
          auto rhsRhs = rhsDefOp.getRhs();
          if (rhsLhs == rhsRhs &&
              rhsLhs == lhsRhs) { // (mul (mul x x) (mul x x))
            auto powType = RankedTensorType::get(
                cast<RankedTensorType>(lhs.getType()).getShape(),
                lhs.getType().getElementType());
            auto powVal = rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), powType, cast<ElementsAttr>(makeAttr(powType, 4)));
            rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, rhsLhs, powVal);
            return success();
          }
        }

        if (lhsLhs == rhs) { // (mul (mul x x) x)
          auto powType = RankedTensorType::get(
              cast<RankedTensorType>(rhs.getType()).getShape(),
              rhs.getType().getElementType());
          auto powVal = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), powType, cast<ElementsAttr>(makeAttr(powType, 3)));
          rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, rhs, powVal);
          return success();
        }
      }
    }

    if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) { // mul x (mul x x)
      auto rhsLhs = rhsDefOp.getLhs();
      auto rhsRhs = rhsDefOp.getRhs();
      if (rhsLhs == rhsRhs) {
        if (rhsLhs == lhs) {
          auto powType = RankedTensorType::get(
              cast<RankedTensorType>(lhs.getType()).getShape(),
              lhs.getType().getElementType());
          auto powVal = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), powType, cast<ElementsAttr>(makeAttr(powType, 3)));
          rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, lhs, powVal);
          return success();
        }
      }
    }

    return failure();
  }
};

struct PowerMultiplyToPower final
    : public CheckedOpRewritePattern<stablehlo::MulOp, PowerMultiplyToPower> {
  using CheckedOpRewritePattern<stablehlo::MulOp,
                                PowerMultiplyToPower>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::MulOp op,
                                    PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsDefOp = lhs.getDefiningOp<stablehlo::PowOp>();
    auto rhsDefOp = rhs.getDefiningOp<stablehlo::PowOp>();

    auto rType = cast<RankedTensorType>(op.getType());

    if (lhsDefOp && isOnlyUsedInOperation(lhsDefOp, op)) {
      auto lhsLhs = lhsDefOp.getLhs();

      if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) {
        auto rhsLhs = rhsDefOp.getLhs();

        if (lhsLhs ==
            rhsLhs) { // (mul (pow a b) (pow a c)) => (pow a (add b c))
          auto newPowVal = rewriter.create<stablehlo::AddOp>(
              op.getLoc(), lhsDefOp.getRhs(), rhsDefOp.getRhs());
          rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, lhsLhs, newPowVal);
          return success();
        }
      }

      auto rhsMulDefOp = rhs.getDefiningOp<stablehlo::MulOp>();
      if (rhsMulDefOp && isOnlyUsedInOperation(rhsMulDefOp, op)) {
        auto rhsMulLhs = rhsMulDefOp.getLhs();
        auto rhsMulRhs = rhsMulDefOp.getRhs();

        if (rhsMulLhs == rhsMulRhs &&
            rhsMulLhs ==
                lhsLhs) { // (mul (pow a b) (mul a a)) => (pow a (add b 2))
          auto newPowVal = rewriter.create<stablehlo::AddOp>(
              op.getLoc(), lhsDefOp.getRhs(),
              rewriter.create<stablehlo::ConstantOp>(
                  op.getLoc(), rType, cast<ElementsAttr>(makeAttr(rType, 2))));
          rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, lhsLhs, newPowVal);
          return success();
        }
      }

      if (lhsLhs == rhs) { // (mul (pow x y) x) => (pow x (add y 1))
        auto newPowVal = rewriter.create<stablehlo::AddOp>(
            op.getLoc(), lhsDefOp.getRhs(),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), rType, cast<ElementsAttr>(makeAttr(rType, 1))));
        rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, lhsLhs, newPowVal);
        return success();
      }
    }

    if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) {
      auto rhsLhs = rhsDefOp.getLhs();

      if (rhsLhs == lhs) { // (mul x (pow x y)) => (pow x (add y 1))
        auto newPowVal = rewriter.create<stablehlo::AddOp>(
            op.getLoc(), rhsDefOp.getRhs(),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), rType, cast<ElementsAttr>(makeAttr(rType, 1))));
        rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, rhsLhs, newPowVal);
        return success();
      }

      auto lhsMulDefOp = lhs.getDefiningOp<stablehlo::MulOp>();
      if (lhsMulDefOp && isOnlyUsedInOperation(lhsMulDefOp, op)) {
        auto lhsMulLhs = lhsMulDefOp.getLhs();
        auto lhsMulRhs = lhsMulDefOp.getRhs();

        if (lhsMulLhs == lhsMulRhs &&
            lhsMulLhs ==
                rhsLhs) { // (mul (mul a a) (pow a b)) => (pow a (add b 2))
          auto newPowVal = rewriter.create<stablehlo::AddOp>(
              op.getLoc(), rhsDefOp.getRhs(),
              rewriter.create<stablehlo::ConstantOp>(
                  op.getLoc(), rType, cast<ElementsAttr>(makeAttr(rType, 2))));
          rewriter.replaceOpWithNewOp<stablehlo::PowOp>(op, rhsLhs, newPowVal);
          return success();
        }
      }
    }

    return failure();
  }
};

template <typename Op>
struct CommonAssociativeCommutativeOpReorder final
    : public CheckedOpRewritePattern<
          Op, CommonAssociativeCommutativeOpReorder<Op>> {
  using CheckedOpRewritePattern<
      Op, CommonAssociativeCommutativeOpReorder<Op>>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(Op op, PatternRewriter &rewriter) const {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    if (lhs == rhs)
      return failure(); // already in a good form

    DenseFPElementsAttr constAttr;
    if (matchPattern(lhs, m_Constant(&constAttr)) &&
        matchPattern(rhs, m_Constant(&constAttr)))
      return rewriter.notifyMatchFailure(op, "const prop has higher priority");

    auto lhsDefOp = lhs.template getDefiningOp<Op>();
    auto rhsDefOp = rhs.template getDefiningOp<Op>();

    if (lhsDefOp && isOnlyUsedInOperation(lhsDefOp, op)) {
      auto lhsLhs = lhsDefOp.getLhs();
      auto lhsRhs = lhsDefOp.getRhs();

      if (lhsLhs == lhsRhs)
        return failure(); // already in a good form

      if (matchPattern(lhsRhs, m_Constant(&constAttr)) &&
          matchPattern(lhsLhs, m_Constant(&constAttr)))
        return rewriter.notifyMatchFailure(op,
                                           "const prop has higher priority");

      if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) {
        auto rhsLhs = rhsDefOp.getLhs();
        auto rhsRhs = rhsDefOp.getRhs();

        if (rhsLhs == rhsRhs)
          return failure(); // already in a good form

        if (matchPattern(rhsRhs, m_Constant(&constAttr)) &&
            matchPattern(rhsLhs, m_Constant(&constAttr)))
          return rewriter.notifyMatchFailure(op,
                                             "const prop has higher priority");

        Value commonOperand, otherOperand1, otherOperand2;
        bool foundCommonOperand = false;

        if (lhsLhs ==
            rhsLhs) { // (op (op x y) (op x z)) => (op (op x x) (op y z))
          if (lhsRhs == rhsRhs)
            return failure(); // we can CSE this case
          commonOperand = lhsLhs;
          otherOperand1 = lhsRhs;
          otherOperand2 = rhsRhs;
          foundCommonOperand = true;
        }

        if (!foundCommonOperand &&
            lhsLhs == rhsRhs) { // (op (op x y) (op z x)) => (op (op x x)
                                // (op y z))
          if (lhsRhs == rhsLhs)
            return failure(); // we can CSE this case
          commonOperand = lhsLhs;
          otherOperand1 = rhsLhs;
          otherOperand2 = lhsRhs;
          foundCommonOperand = true;
        }

        if (!foundCommonOperand &&
            lhsRhs == rhsLhs) { // (op (op y x) (op x z)) => (op (op x x)
                                // (op y z))
          if (lhsLhs == rhsRhs)
            return failure(); // we can CSE this case
          commonOperand = lhsRhs;
          otherOperand1 = lhsLhs;
          otherOperand2 = rhsRhs;
          foundCommonOperand = true;
        }

        if (!foundCommonOperand &&
            lhsRhs == rhsRhs) { // (op (op y x) (op z x)) => (op (op x x)
                                // (op y z))
          if (lhsLhs == rhsLhs)
            return failure(); // we can CSE this case
          commonOperand = lhsRhs;
          otherOperand1 = rhsLhs;
          otherOperand2 = lhsLhs;
          foundCommonOperand = true;
        }

        if (foundCommonOperand) {
          rewriter.replaceOpWithNewOp<Op>(
              op,
              rewriter.create<Op>(op.getLoc(), commonOperand, commonOperand),
              rewriter.create<Op>(op.getLoc(), otherOperand1, otherOperand2));
          return success();
        }
      }

      if (lhsLhs == rhs) { // (op (op x y) x) => (op (op x x) y)
        rewriter.replaceOpWithNewOp<Op>(
            op, rewriter.create<Op>(op.getLoc(), lhsLhs, lhsLhs), lhsRhs);
        return success();
      }

      if (lhsRhs == rhs) { // (op (op y x) x) => (op y (op x x))
        rewriter.replaceOpWithNewOp<Op>(
            op, lhsLhs, rewriter.create<Op>(op.getLoc(), lhsRhs, lhsRhs));
        return success();
      }
    }

    if (rhsDefOp && isOnlyUsedInOperation(rhsDefOp, op)) {
      auto rhsLhs = rhsDefOp.getLhs();
      auto rhsRhs = rhsDefOp.getRhs();

      if (rhsLhs == rhsRhs)
        return failure(); // already in a good form

      if (matchPattern(rhsRhs, m_Constant(&constAttr)) &&
          matchPattern(rhsLhs, m_Constant(&constAttr)))
        return rewriter.notifyMatchFailure(op,
                                           "const prop has higher priority");

      if (rhsLhs == lhs) { // (op x (op x y)) => (op (op x x) y)
        rewriter.replaceOpWithNewOp<Op>(
            op, rewriter.create<Op>(op.getLoc(), rhsLhs, rhsLhs), rhsRhs);
        return success();
      }

      if (rhsRhs == lhs) { // (op x (op y x)) => (op y (op x x))
        rewriter.replaceOpWithNewOp<Op>(
            op, rhsLhs, rewriter.create<Op>(op.getLoc(), rhsRhs, rhsRhs));
        return success();
      }
    }

    return failure();
  }
};

struct LogSimplify final
    : public CheckedOpRewritePattern<stablehlo::LogOp, LogSimplify> {
  using CheckedOpRewritePattern<stablehlo::LogOp,
                                LogSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::LogOp op,
                                    PatternRewriter &rewriter) const {
    { // log(exp(x)) -> x
      auto defOp = op.getOperand().getDefiningOp<stablehlo::ExpOp>();
      if (defOp) {
        rewriter.replaceAllUsesWith(op.getResult(), defOp.getOperand());
        return success();
      }
    }

    { // log(pow(x, y)) -> y * log(x)
      auto defOp = op.getOperand().getDefiningOp<stablehlo::PowOp>();
      if (defOp) {
        rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
            op, defOp.getRhs(),
            rewriter.create<stablehlo::LogOp>(op.getLoc(), defOp.getLhs()));
        return success();
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::MulOp>();
      if (defOp) {
        auto lhs = defOp.getLhs();
        auto rhs = defOp.getRhs();
        if (lhs == rhs) { // log(mul(a, a)) -> 2 * log(a)
          rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
              op,
              rewriter.create<stablehlo::ConstantOp>(
                  op.getLoc(), lhs.getType(),
                  cast<ElementsAttr>(makeAttr(lhs.getType(), 2))),
              rewriter.create<stablehlo::LogOp>(op.getLoc(), lhs));
          return success();
        }

        if (anyOperandIsConstant(defOp) &&
            !allOperandsAreConstant(defOp)) { // log(mul(a, b)) -> log(a) +
                                              // log(b) if a or b is constant
          rewriter.replaceOpWithNewOp<stablehlo::AddOp>(
              op, rewriter.create<stablehlo::LogOp>(op.getLoc(), lhs),
              rewriter.create<stablehlo::LogOp>(op.getLoc(), rhs));
          return success();
        }
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::AddOp>();
      if (defOp) {
        auto lhs = defOp.getLhs();
        auto rhs = defOp.getRhs();
        if (lhs == rhs) { // log(add(a, a)) -> log(2) + log(a)
          rewriter.replaceOpWithNewOp<stablehlo::AddOp>(
              op,
              rewriter.create<stablehlo::LogOp>(
                  op.getLoc(),
                  rewriter.create<stablehlo::ConstantOp>(
                      op.getLoc(), lhs.getType(),
                      cast<ElementsAttr>(makeAttr(lhs.getType(), 2)))),
              rewriter.create<stablehlo::LogOp>(op.getLoc(), lhs));
          return success();
        }
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::DivOp>();
      if (defOp) {
        auto lhs = defOp.getLhs();
        auto rhs = defOp.getRhs();

        if (anyOperandIsConstant(defOp) &&
            !allOperandsAreConstant(defOp)) { // log(div(a, b)) -> log(a) -
                                              // log(b) if a or b is constant
          rewriter.replaceOpWithNewOp<stablehlo::SubtractOp>(
              op, rewriter.create<stablehlo::LogOp>(op.getLoc(), lhs),
              rewriter.create<stablehlo::LogOp>(op.getLoc(), rhs));
          return success();
        }
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::SqrtOp>();
      if (defOp &&
          isOnlyUsedInOperation(defOp, op)) { // log(sqrt(x)) -> log(x) / 2
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op,
            rewriter.create<stablehlo::LogOp>(op.getLoc(), defOp.getOperand()),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), defOp.getType(),
                cast<ElementsAttr>(makeAttr(defOp.getType(), 2))));
        return success();
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::CbrtOp>();
      if (defOp &&
          isOnlyUsedInOperation(defOp, op)) { // log(cbrt(x)) -> log(x) / 3
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op,
            rewriter.create<stablehlo::LogOp>(op.getLoc(), defOp.getOperand()),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), defOp.getType(),
                cast<ElementsAttr>(makeAttr(defOp.getType(), 3))));
        return success();
      }
    }

    {
      auto defOp = op.getOperand().getDefiningOp<stablehlo::RsqrtOp>();
      if (defOp &&
          isOnlyUsedInOperation(defOp, op)) { // log(rsqrt(x)) -> -log(x) / 2
        rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
            op,
            rewriter.create<stablehlo::LogOp>(op.getLoc(), defOp.getOperand()),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), defOp.getType(),
                cast<ElementsAttr>(makeAttr(defOp.getType(), -2))));
        return success();
      }
    }

    return failure();
  }
};

struct NegMulConstSimplify final
    : public CheckedOpRewritePattern<stablehlo::NegOp, NegMulConstSimplify> {
  using CheckedOpRewritePattern<stablehlo::NegOp,
                                NegMulConstSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::NegOp op,
                                    PatternRewriter &rewriter) const {
    auto mulOp = op.getOperand().getDefiningOp<stablehlo::MulOp>();
    if (!mulOp)
      return failure();

    auto lhs = mulOp.getLhs();
    auto rhs = mulOp.getRhs();

    DenseElementsAttr lhsAttr;
    bool lhsIsConst = matchPattern(lhs, m_Constant(&lhsAttr));

    DenseElementsAttr rhsAttr;
    bool rhsIsConst = matchPattern(rhs, m_Constant(&rhsAttr));

    if (lhsIsConst && rhsIsConst)
      return failure(); // const prop will evaluate this

    if (lhsIsConst) {
      rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
          op, rewriter.create<stablehlo::NegOp>(op.getLoc(), lhs), rhs);
      return success();
    }

    if (rhsIsConst) {
      rewriter.replaceOpWithNewOp<stablehlo::MulOp>(
          op, lhs, rewriter.create<stablehlo::NegOp>(op.getLoc(), rhs));
      return success();
    }

    return failure();
  }
};

struct NegDivConstSimplify final
    : public CheckedOpRewritePattern<stablehlo::NegOp, NegDivConstSimplify> {
  using CheckedOpRewritePattern<stablehlo::NegOp,
                                NegDivConstSimplify>::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::NegOp op,
                                    PatternRewriter &rewriter) const {
    auto divOp = op.getOperand().getDefiningOp<stablehlo::DivOp>();
    if (!divOp)
      return failure();

    auto lhs = divOp.getLhs();
    auto rhs = divOp.getRhs();

    DenseElementsAttr lhsAttr;
    bool lhsIsConst = matchPattern(lhs, m_Constant(&lhsAttr));

    DenseElementsAttr rhsAttr;
    bool rhsIsConst = matchPattern(rhs, m_Constant(&rhsAttr));

    if (lhsIsConst && rhsIsConst)
      return failure(); // const prop will evaluate this

    if (lhsIsConst) {
      rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
          op, rewriter.create<stablehlo::NegOp>(op.getLoc(), lhs), rhs);
      return success();
    }

    if (rhsIsConst) {
      rewriter.replaceOpWithNewOp<stablehlo::DivOp>(
          op, lhs, rewriter.create<stablehlo::NegOp>(op.getLoc(), rhs));
      return success();
    }

    return failure();
  }
};

struct ReshapeDeletionsBroadcastInDimSimplify final
    : public CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                     ReshapeDeletionsBroadcastInDimSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto bcastInDimOp =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcastInDimOp)
      return failure();

    if (!isOnlyUsedInOperation(bcastInDimOp, op))
      return failure();

    auto deletionDims =
        findReshapeInsertionDims(op.getType(), op.getOperand().getType());
    if (deletionDims.empty())
      return failure();

    SmallVector<int64_t> newBcastDims;
    for (auto dim : bcastInDimOp.getBroadcastDimensions()) {
      int64_t nDeleted = 0;
      for (auto delDim : deletionDims) {
        if (delDim == dim)
          return failure();
        if (delDim < dim)
          nDeleted++;
      }
      int64_t newDim = dim - nDeleted;
      if (newDim < 0)
        return failure(); // this should not happen
      newBcastDims.push_back(newDim);
    }

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), bcastInDimOp.getOperand(), newBcastDims);
    return success();
  }
};

struct ReshapeInsertionsBroadcastInDimSimplify final
    : public CheckedOpRewritePattern<stablehlo::ReshapeOp,
                                     ReshapeInsertionsBroadcastInDimSimplify> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::ReshapeOp op,
                                    PatternRewriter &rewriter) const {
    auto bcastInDimOp =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcastInDimOp)
      return failure();

    if (!isOnlyUsedInOperation(bcastInDimOp, op))
      return failure();

    auto insertionDims =
        findReshapeInsertionDims(op.getOperand().getType(), op.getType());
    if (insertionDims.empty())
      return failure();

    SmallVector<int64_t> newPositions;
    for (int i = 0; i < cast<ShapedType>(op.getType()).getRank(); i++) {
      if (!llvm::is_contained(insertionDims, i))
        newPositions.push_back(i);
    }

    SmallVector<int64_t> newBcastDims;
    for (auto dim : bcastInDimOp.getBroadcastDimensions()) {
      if (dim >= newPositions.size())
        return failure(); // this should not happen
      newBcastDims.push_back(newPositions[dim]);
    }

    rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
        op, op.getType(), bcastInDimOp.getOperand(), newBcastDims);
    return success();
  }
};

struct TransposeFFT final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeFFT> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto fftOp = op.getOperand().getDefiningOp<stablehlo::FftOp>();
    if (!fftOp)
      return failure();

    auto fftLength = llvm::to_vector(fftOp.getFftLength());

    auto [preTransposePerm, postTransposePerm, newFftLength, needsPostTranspose,
          invalidMove] =
        splitPermutation(op.getPermutation(),
                         cast<ShapedType>(op.getOperand().getType()).getRank(),
                         fftLength.size(), fftOp.getFftType(), fftLength);

    if (invalidMove)
      return rewriter.notifyMatchFailure(fftOp,
                                         "Can't move transpose above FFT");

    auto preTransposeOp = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), fftOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(preTransposePerm));
    auto newFftOp = rewriter.create<stablehlo::FftOp>(
        fftOp.getLoc(), preTransposeOp.getResult(), fftOp.getFftType(),
        newFftLength);

    if (needsPostTranspose) {
      rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
          op, newFftOp.getResult(),
          rewriter.getDenseI64ArrayAttr(postTransposePerm));
    } else {
      rewriter.replaceOp(op, newFftOp.getResult());
    }

    return success();
  }

private:
  bool isIotaVector(SmallVector<int64_t> v) const {
    for (int64_t i = 0; i < v.size(); ++i) {
      if (v[i] != i)
        return false;
    }
    return true;
  }

  std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>,
             bool, bool>
  splitPermutation(ArrayRef<int64_t> permutation, int64_t rank, int64_t nDimFft,
                   stablehlo::FftType fftType,
                   SmallVector<int64_t> fftLength) const {
    SmallVector<int64_t> preTransposePerm(rank), postTransposePerm(rank);
    bool needsPostTranspose = false, invalidMove = false;
    SmallVector<int64_t> newFftLength(fftLength.size());

    std::iota(preTransposePerm.begin(), preTransposePerm.end(), 0);
    std::iota(postTransposePerm.begin(), postTransposePerm.end(), 0);

    int64_t nonFftDims = rank - nDimFft;

    for (int64_t i = 0; i < rank; ++i) {
      int64_t target = permutation[i];
      bool sourceIsNonFft = (i < nonFftDims);
      bool targetIsNonFft = (target < nonFftDims);

      if (sourceIsNonFft != targetIsNonFft) {
        // Cross-boundary permutation - this is not a valid move
        invalidMove = true;
        break;
      }
    }

    if (invalidMove)
      return {preTransposePerm, postTransposePerm, newFftLength,
              needsPostTranspose, invalidMove};

    SmallVector<int64_t> nonFftTargets, nonFftSources;
    for (int64_t i = 0; i < nonFftDims; i++) {
      int64_t target = permutation[i];
      if (target < nonFftDims) {
        nonFftTargets.push_back(target);
        nonFftSources.push_back(i);
      }
    }

    for (size_t i = 0; i < nonFftSources.size(); ++i) {
      preTransposePerm[nonFftTargets[i]] = nonFftSources[i];
    }

    if (fftType == stablehlo::FftType::FFT ||
        fftType == stablehlo::FftType::IFFT) { // can permute fft dims without
                                               // restriction
      for (int64_t i = 0; i < nDimFft; ++i) {
        int64_t fftDimIdx = nonFftDims + i;
        if (permutation[fftDimIdx] != fftDimIdx) {
          preTransposePerm[fftDimIdx] = permutation[fftDimIdx];
          newFftLength[i] = fftLength[permutation[fftDimIdx] - nonFftDims];
        } else {
          newFftLength[i] = fftLength[i];
        }
      }
    } else { // for IRFFT & RFFT we can't move the last dim
      for (int64_t i = 0; i < nDimFft - 1; ++i) {
        int64_t fftDimIdx = nonFftDims + i;
        if (permutation[fftDimIdx] != fftDimIdx) {
          preTransposePerm[fftDimIdx] = permutation[fftDimIdx];
          newFftLength[i] = fftLength[permutation[fftDimIdx] - nonFftDims];
        } else {
          newFftLength[i] = fftLength[i];
        }
      }

      int64_t lastDimIdx = nonFftDims + nDimFft - 1;
      newFftLength[nDimFft - 1] = fftLength[nDimFft - 1];
      if (permutation[lastDimIdx] != lastDimIdx) {
        needsPostTranspose = true;
        postTransposePerm[lastDimIdx] = permutation[lastDimIdx];
      }
    }

    // If we determined iota then we can't move the transpose above the FFT
    invalidMove = invalidMove || isIotaVector(preTransposePerm);
    // Adding a couple of sanity checks below to ensure that the permutation
    // is valid
    if (!invalidMove) {
      SmallVector<int64_t> preTransposePerm2(preTransposePerm.size());
      for (int i = 0; i < preTransposePerm.size(); ++i) {
        preTransposePerm2[i] = preTransposePerm[i];
      }
      std::sort(preTransposePerm2.begin(), preTransposePerm2.end());
      invalidMove = !isIotaVector(preTransposePerm2);
    }
    if (!invalidMove) {
      SmallVector<int64_t> postTransposePerm2(postTransposePerm.size());
      for (int i = 0; i < postTransposePerm.size(); ++i) {
        postTransposePerm2[i] = postTransposePerm[i];
      }
      std::sort(postTransposePerm2.begin(), postTransposePerm2.end());
      invalidMove = !isIotaVector(postTransposePerm2);
    }

    return {preTransposePerm, postTransposePerm, newFftLength,
            needsPostTranspose, invalidMove};
  }
};

// This applies if reshape expands certain dimensions and transpose doesn't move
// those dimensions around. See lit_tests/transpose_reshape.mlir for examples of
// where this is applicable
struct TransposeReshape final
    : public CheckedOpRewritePattern<stablehlo::TransposeOp, TransposeReshape> {
  using CheckedOpRewritePattern::CheckedOpRewritePattern;

  LogicalResult matchAndRewriteImpl(stablehlo::TransposeOp op,
                                    PatternRewriter &rewriter) const {
    auto reshapeOp = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return failure();

    auto inputShape = llvm::to_vector(
        cast<RankedTensorType>(reshapeOp.getOperand().getType()).getShape());
    auto outputShape =
        llvm::to_vector(cast<RankedTensorType>(reshapeOp.getType()).getShape());

    auto [validExpansion, dimOrdering, outputIdxStarts, extrasList] =
        dimOrderingToPreserve(inputShape, outputShape);

    if (!validExpansion)
      return rewriter.notifyMatchFailure(op, "Invalid expansion for reshape");

    auto permutation = op.getPermutation();
    SmallVector<int64_t> newPermutation(inputShape.size(), -1);

    if (dimOrdering.size() != inputShape.size())
      return rewriter.notifyMatchFailure(op, "Invalid dimOrdering");

    for (auto [i, dimList] : llvm::enumerate(dimOrdering)) {
      if (dimList.size() == 0)
        return rewriter.notifyMatchFailure(op,
                                           "Empty dimList. Should not happen!");

      auto firstDim = dimList[0];
      auto idx = std::find(permutation.begin(), permutation.end(), firstDim);
      if (idx == permutation.end())
        return rewriter.notifyMatchFailure(op, "Invalid permutation");

      int64_t index = std::distance(permutation.begin(), idx);

      for (int i = 1; i < dimList.size(); ++i) {
        if (i + index >= permutation.size() ||
            permutation[i + index] != dimList[i])
          return rewriter.notifyMatchFailure(op, "unsupported permutation");
      }
    }

    int64_t outputIdx = 0;
    for (int i = 0; i < inputShape.size(); ++i) {
      auto transposeIdx = permutation[outputIdx];

      auto idx = std::find(outputIdxStarts.begin(), outputIdxStarts.end(),
                           transposeIdx);
      auto index = std::distance(outputIdxStarts.begin(), idx);

      newPermutation[i] = transposeIdx - extrasList[index];

      for (int j = 0; j < dimOrdering.size(); ++j) {
        if (dimOrdering[j][0] == transposeIdx) {
          outputIdx += dimOrdering[j].size();
          break;
        }
      }
    }

    auto newTranspose = rewriter.create<stablehlo::TransposeOp>(
        op.getLoc(), reshapeOp.getOperand(),
        rewriter.getDenseI64ArrayAttr(newPermutation));
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      newTranspose.getResult());
    return success();
  }

private:
  std::tuple<bool, SmallVector<SmallVector<int64_t>>, SmallVector<int64_t>,
             SmallVector<int64_t>>
  dimOrderingToPreserve(SmallVector<int64_t> inputShape,
                        SmallVector<int64_t> outputShape) const {
    SmallVector<SmallVector<int64_t>> dimOrdering;
    SmallVector<int64_t> outputIdxStarts, extrasList;
    bool validExpansion = true;

    int64_t outputIdx = 0, i = 0, extras = 0;
    for (; i < inputShape.size() && validExpansion &&
           outputIdx < outputShape.size();
         ++i) {
      int64_t outputSize = 1;
      SmallVector<int64_t> dims;

      outputIdxStarts.push_back(outputIdx);
      extrasList.push_back(extras);

      do {
        outputSize *= outputShape[outputIdx];
        dims.push_back(outputIdx);
        outputIdx++;
      } while (outputIdx < outputShape.size() && outputSize < inputShape[i]);

      if (outputSize != inputShape[i]) {
        validExpansion = false;
        break;
      }

      dimOrdering.push_back(dims);
      extras += dims.size() - 1;
    }

    if (i != inputShape.size() || outputIdx != outputShape.size())
      validExpansion = false;

    return {validExpansion, dimOrdering, outputIdxStarts, extrasList};
  }
};

///////////////  End Imported from stablehlo

// clang-format off
namespace mlir {
namespace enzyme {
#include "src/enzyme_ad/jax/Passes/StablehloOptPatterns.cpp.inc"
}; // namespace enzyme
}; // namespace mlir

#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.cpp.inc"
// clang-format on

void mlir::transform::addPadDotGeneral(RewritePatternSet &patterns,
                                       bool postPad, MLIRContext &context,
                                       PatternBenefit benefit) {
  patterns.insert<PadDotGeneral>(postPad, &context, benefit);
}

void mlir::transform::addIotaSimplify(RewritePatternSet &patterns,
                                      int64_t maxConstantExpansion,
                                      MLIRContext &context,
                                      PatternBenefit benefit) {
  patterns.insert<IotaSimplify>(maxConstantExpansion, &context, benefit);
}

void mlir::transform::addConcatConstProp(RewritePatternSet &patterns,
                                         int64_t maxConstantExpansion,
                                         MLIRContext &context,
                                         PatternBenefit benefit) {
  patterns.insert<ConcatConstProp>(maxConstantExpansion, &context, benefit);
}

void mlir::transform::addPadSimplify(RewritePatternSet &patterns,
                                     int64_t maxConstantExpansion,
                                     MLIRContext &context,
                                     PatternBenefit benefit) {
  patterns.insert<PadSimplify>(maxConstantExpansion, &context, benefit);
}

void mlir::transform::addDynamicUpdateSliceConstProp(
    RewritePatternSet &patterns, int64_t maxConstantExpansion,
    MLIRContext &context, PatternBenefit benefit) {
  patterns.insert<DynamicUpdateSliceConstProp>(maxConstantExpansion, &context,
                                               benefit);
}

void mlir::transform::addWhileSimplify(RewritePatternSet &patterns,
                                       bool hoistAll, MLIRContext &context,
                                       PatternBenefit benefit) {
  patterns.insert<WhileSimplify>(hoistAll, &context, benefit);
}

void mlir::transform::addWhileLICM(RewritePatternSet &patterns, bool hoistAll,
                                   MLIRContext &context,
                                   PatternBenefit benefit) {
  patterns.insert<WhileLICM>(hoistAll, &context, benefit);
}

void mlir::transform::addSliceLICM(RewritePatternSet &patterns,
                                   bool single_user, MLIRContext &context,
                                   PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::SliceOp>>(single_user, &context, benefit);
}

void mlir::transform::addDUSLICM(RewritePatternSet &patterns, bool single_user,
                                 MLIRContext &context, PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::DynamicUpdateSliceOp>>(single_user, &context,
                                                         benefit);
}

void mlir::transform::addSumToConv(RewritePatternSet &patterns,
                                   bool collapseDims, MLIRContext &context,
                                   PatternBenefit benefit) {
  patterns
      .insert<SumToConv<stablehlo::AddOp>, SumToConv<stablehlo::SubtractOp>>(
          collapseDims, &context, benefit);
}

void mlir::transform::addPadLICM(RewritePatternSet &patterns, bool single_user,
                                 MLIRContext &context, PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::PadOp>>(single_user, &context, benefit);
}

void mlir::transform::addElementwiseLICM(RewritePatternSet &patterns,
                                         bool single_user, MLIRContext &context,
                                         PatternBenefit benefit) {
  patterns.insert<LICMElementwise>(single_user, &context, benefit);
}

void mlir::transform::addConcatenateLICM(RewritePatternSet &patterns,
                                         bool single_user, MLIRContext &context,
                                         PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::ConcatenateOp>>(single_user, &context,
                                                  benefit);
}

void mlir::transform::addBroadcastInDimLICM(RewritePatternSet &patterns,
                                            bool single_user,
                                            MLIRContext &context,
                                            PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::BroadcastInDimOp>>(single_user, &context,
                                                     benefit);
}

void mlir::transform::addReshapeLICM(RewritePatternSet &patterns,
                                     bool single_user, MLIRContext &context,
                                     PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::ReshapeOp>>(single_user, &context, benefit);
}

void mlir::transform::addTransposeLICM(RewritePatternSet &patterns,
                                       bool single_user, MLIRContext &context,
                                       PatternBenefit benefit) {
  patterns.insert<LICM<stablehlo::TransposeOp>>(single_user, &context, benefit);
}

void mlir::transform::addNoNanAddSubSimplify(RewritePatternSet &patterns,
                                             bool allowOnFloatingPointMath,
                                             MLIRContext &context,
                                             PatternBenefit benefit) {
  patterns.insert<NoNanAddSubSimplify>(allowOnFloatingPointMath, &context,
                                       benefit);
}

void mlir::transform::addNoNanMulSimplify(RewritePatternSet &patterns,
                                          bool allowOnFloatingPointMath,
                                          MLIRContext &context,
                                          PatternBenefit benefit) {
  patterns.insert<NoNanMulSimplify>(allowOnFloatingPointMath, &context,
                                    benefit);
}

void mlir::transform::addNoNanDivSimplify(RewritePatternSet &patterns,
                                          bool allowOnFloatingPointMath,
                                          MLIRContext &context,
                                          PatternBenefit benefit) {
  patterns.insert<NoNanDivSimplify>(allowOnFloatingPointMath, &context,
                                    benefit);
}

void mlir::transform::addNoNanZeroBasePowSimplify(RewritePatternSet &patterns,
                                                  bool allowOnFloatingPointMath,
                                                  MLIRContext &context,
                                                  PatternBenefit benefit) {
  patterns.insert<NoNanZeroBasePowSimplify>(allowOnFloatingPointMath, &context,
                                            benefit);
}

void mlir::transform::addBroadcastInDimSimplify(RewritePatternSet &patterns,
                                                int64_t maxConstantExpansion,
                                                MLIRContext &context,
                                                PatternBenefit benefit) {
  patterns.insert<BroadcastInDimSimplify>(maxConstantExpansion, &context,
                                          benefit);
}

void mlir::transform::addSelectOpCanon(RewritePatternSet &patterns,
                                       int64_t maxConstantExpansion,
                                       MLIRContext &context,
                                       PatternBenefit benefit) {
  patterns.insert<SelectOpCanon>(maxConstantExpansion, &context, benefit);
}

void mlir::transform::addConcatenateOpCanon(RewritePatternSet &patterns,
                                            int64_t maxConstantExpansion,
                                            MLIRContext &context,
                                            PatternBenefit benefit) {
  patterns.insert<ConcatenateOpCanon>(maxConstantExpansion, &context, benefit);
}

void mlir::transform::addTransposeElementwise(RewritePatternSet &patterns,
                                              bool onlySingleUser,
                                              MLIRContext &context,
                                              PatternBenefit benefit) {
  patterns.insert<TransposeElementwise>(onlySingleUser, &context, benefit);
}

void mlir::transform::addReshapeElementwise(RewritePatternSet &patterns,
                                            bool onlySingleUser,
                                            MLIRContext &context,
                                            PatternBenefit benefit) {
  patterns.insert<ReshapeElementwise>(onlySingleUser, &context, benefit);
}

void mlir::transform::addReshapeSlice(RewritePatternSet &patterns,
                                      bool onlySingleUser, MLIRContext &context,
                                      PatternBenefit benefit) {
  patterns.insert<ReshapeSlice>(onlySingleUser, &context, benefit);
}

void mlir::transform::addExtendUnaryElementwise(RewritePatternSet &patterns,
                                                bool onlySingleUser,
                                                MLIRContext &context,
                                                PatternBenefit benefit) {
  patterns.insert<ExtendUnaryElementwise>(onlySingleUser, &context, benefit);
}

void mlir::transform::addWrapUnaryElementwise(RewritePatternSet &patterns,
                                              bool onlySingleUser,
                                              MLIRContext &context,
                                              PatternBenefit benefit) {
  patterns.insert<WrapUnaryElementwise>(onlySingleUser, &context, benefit);
}

namespace {

struct EnzymeHLOOptPass
    : public enzyme::impl::EnzymeHLOOptPassBase<EnzymeHLOOptPass> {
  using EnzymeHLOOptPassBase::EnzymeHLOOptPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    RewritePatternSet patterns(context);
    mlir::enzyme::populateWithGenerated(patterns);

    patterns.add<SliceExtend>(context);
    patterns.add<SliceRotate>(context);
    patterns.add<SliceWrap>(context);
    patterns.add<ReshapeWrap>(context);
    patterns.add<ReshapeExtend>(context);
    patterns.add<ReshapeRotate>(context);
    patterns.add<TransposeWrap>(context);
    patterns.add<TransposeExtend>(context);
    patterns.add<TransposeRotate>(context);

    patterns.add<
        AddSimplify, SubSimplify, AndSimplify, MaxSimplify, MinSimplify,
        OrSimplify, XorSimplify, MulSimplify, DivSimplify, RemSimplify,
        PowSimplify, NoopSlice, NoopReverse, SliceSlice, LogSimplify,
        ShiftRightLogicalSimplify, NegativePadToSlice, SliceSimplify,
        ConvertSimplify, TransposeSimplify, DotGeneralSimplify,
        DynamicSliceToStatic, DynamicUpdateSliceElim, ReduceToReshape,
        BroadcastToReshape, ReshapeEmptyBroadcast, BroadcastReshape,
        ConstPropThroughBarrier, ReplaceNegAddWithSubtract, SignAbsSimplify,
        AbsPositiveSimplify, SimplifyBoundary<enzymexla::ExtendOp>,
        SimplifyBoundary<enzymexla::WrapOp>,
        SimplifyBoundary<enzymexla::RotateOp>, TransposeReshapeToBroadcast,
        ReshapeTransposeToBroadcast, SelectBroadcastInDim, PowerMultiplyToPower,
        NegMulConstSimplify, NegDivConstSimplify,
        ReshapeDeletionsBroadcastInDimSimplify,
        ReshapeInsertionsBroadcastInDimSimplify>(context,
                                                 PatternBenefit(65000));

    patterns.add<IotaSimplify, BroadcastInDimSimplify, ConcatConstProp,
                 DynamicUpdateSliceConstProp, PadSimplify>(
        max_constant_expansion, context, PatternBenefit(65000));

    patterns.add<ConvertConcat, DynamicUpdateToConcat, SliceOfDynamicUpdate,
                 SliceElementwise, SliceReshapeElementwise, SlicePad,
                 SliceReshapePad, DotReshapeDot, ChloInfConstProp,
                 GammaConstProp, ConcatFuse, ConcatToBroadcast, PadPad,
                 PadReshapePad, ConcatPushBinop<stablehlo::AddOp>,
                 ConcatPushBinop<stablehlo::MulOp>, ScatterToDynamicUpdateSlice,
                 ReduceConcat, ConcatSlice, ConcatMultiPad, ConcatWrap,
                 ConcatConcatAxisSwap, SliceConcat, SliceIf, SliceReshapeConcat,
                 BinBroadcastSplat<stablehlo::AddOp>,
                 BinBroadcastSplat<stablehlo::SubtractOp>,
                 BinBroadcastSplat<stablehlo::DivOp>,
                 BinBroadcastSplat<stablehlo::MulOp>, RotatePad, ConjReal>(
        context);

    // Unary constant propagation patterns
    patterns.add<UnaryConstProp<stablehlo::NotOp, stablehlo::notOp>,
                 UnaryConstProp<stablehlo::IsFiniteOp, stablehlo::isFiniteOp>,
                 UnaryConstProp<stablehlo::LogOp, stablehlo::logOp>,
                 UnaryConstProp<stablehlo::Log1pOp, stablehlo::log1pOp>,
                 UnaryConstProp<stablehlo::AbsOp, stablehlo::absOp>,
                 UnaryConstProp<stablehlo::NegOp, stablehlo::negOp>,
                 UnaryConstProp<stablehlo::SqrtOp, stablehlo::sqrtOp>,
                 UnaryConstProp<stablehlo::RsqrtOp, stablehlo::rsqrtOp>,
                 UnaryConstProp<stablehlo::CosineOp, stablehlo::cosineOp>,
                 UnaryConstProp<stablehlo::SineOp, stablehlo::sineOp>,
                 UnaryConstProp<stablehlo::ExpOp, stablehlo::exponentialOp>,
                 UnaryConstProp<stablehlo::Expm1Op, stablehlo::expm1Op>,
                 UnaryConstProp<stablehlo::TanhOp, stablehlo::tanhOp>,
                 UnaryConstProp<stablehlo::LogisticOp, stablehlo::logisticOp>,
                 UnaryConstProp<chlo::ConjOp, conjOp>,
                 UnaryConstProp<stablehlo::CeilOp, stablehlo::ceilOp>,
                 UnaryConstProp<stablehlo::CbrtOp, stablehlo::cbrtOp>,
                 UnaryConstProp<stablehlo::RealOp, stablehlo::realOp>,
                 UnaryConstProp<stablehlo::ImagOp, stablehlo::imagOp>,
                 UnaryConstProp<stablehlo::RoundOp, stablehlo::roundOp>,
                 UnaryConstProp<stablehlo::RoundNearestEvenOp,
                                stablehlo::roundNearestEvenOp>,
                 UnaryConstProp<stablehlo::SignOp, stablehlo::signOp>,
                 UnaryConstProp<stablehlo::FloorOp, stablehlo::floorOp>,
                 UnaryConstProp<stablehlo::TanOp, stablehlo::tanOp>>(context);

    // binary constant propagation patterns
    patterns.add<BinaryConstProp<stablehlo::AddOp, stablehlo::addOp>,
                 BinaryConstProp<stablehlo::AndOp, stablehlo::andOp>,
                 BinaryConstProp<stablehlo::Atan2Op, stablehlo::atan2Op>,
                 BinaryConstProp<stablehlo::ComplexOp, stablehlo::complexOp>,
                 BinaryConstProp<stablehlo::DivOp, stablehlo::divideOp>,
                 BinaryConstProp<stablehlo::MaxOp, stablehlo::maxOp>,
                 BinaryConstProp<stablehlo::MinOp, stablehlo::minOp>,
                 BinaryConstProp<stablehlo::MulOp, stablehlo::multiplyOp>,
                 BinaryConstProp<stablehlo::OrOp, stablehlo::orOp>,
                 BinaryConstProp<stablehlo::PowOp, stablehlo::powerOp>,
                 BinaryConstProp<stablehlo::RemOp, stablehlo::remOp>,
                 BinaryConstProp<stablehlo::SubtractOp, stablehlo::subtractOp>,
                 BinaryConstProp<stablehlo::XorOp, stablehlo::xorOp>>(context);

    patterns.add<GatherConstProp>(context);

    patterns.add<BinaryOpTransposeSimplify<stablehlo::AddOp>,
                 BinaryOpTransposeSimplify<stablehlo::SubtractOp>,
                 BinaryOpTransposeSimplify<stablehlo::MulOp>,
                 BinaryOpTransposeSimplify<stablehlo::DivOp>,
                 BinaryOpTransposeSimplify<stablehlo::MinOp>,
                 BinaryOpTransposeSimplify<stablehlo::MaxOp>,
                 BinaryOpTransposeSimplify<stablehlo::AndOp>,
                 BinaryOpTransposeSimplify<stablehlo::OrOp>,
                 BinaryOpTransposeSimplify<stablehlo::XorOp>,
                 BinaryOpTransposeSimplify<stablehlo::PowOp>,
                 BinaryOpTransposeSimplify<stablehlo::RemOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::AbsOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::CeilOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::ConvertOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::CosineOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::ExpOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::Expm1Op>,
                 TransposeUnaryTransposeSimplify<stablehlo::LogOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::Log1pOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::NegOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::RsqrtOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::SignOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::SineOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::SqrtOp>,
                 TransposeUnaryTransposeSimplify<stablehlo::TanhOp>,
                 AssociativeBinaryOpReordering<stablehlo::AddOp>,
                 AssociativeBinaryOpReordering<stablehlo::MulOp>,
                 AssociativeBinaryOpReordering<stablehlo::MinOp>,
                 AssociativeBinaryOpReordering<stablehlo::MaxOp>,
                 AssociativeBinaryOpReordering<stablehlo::AndOp>,
                 AssociativeBinaryOpReordering<stablehlo::OrOp>,
                 AssociativeBinaryOpReordering<stablehlo::XorOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::AddOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::MulOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::MinOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::MaxOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::AndOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::OrOp>,
                 CommonAssociativeCommutativeOpReorder<stablehlo::XorOp>>(
        context);

    patterns.add<BinopPadToConcat<stablehlo::AddOp>,
                 BinopPadToConcat<stablehlo::MulOp>, ConcatPad,
                 PadConcatToConcatPad, SliceSelect, PadReduceWindow>(context);

    if (passses & 512) {
      patterns.add<TransposeDotReorder, DotTranspose, ConvolutionTranspose,
                   TransposeConvolution, EinsumTranspose, TransposeEinsum,
                   ConvertConvertFloat, ConcatToPad, ConcatAppendingReshape,
                   ReshapeIota, DUSDUS, DUSDUSConcat, DUSConcat, DUSPad,
                   SliceDUSToConcat, ConcatConcatToDUS>(context);
      patterns.add<LICM<stablehlo::DynamicUpdateSliceOp>>(false, context);
    }

    if (passses & 1024)
      patterns.add<FullReduceReshapeOrTranspose>(context);

    if (passses & 1)
      patterns.add<SliceTranspose, SliceReshapeTranspose, SliceBroadcast,
                   SliceReduceWindow>(context);

    if (passses & 2)
      patterns.add<ReducePad, BroadcastPad>(context);
    if (passses & 4)
      patterns.add<MulZeroPad, DivZeroPad, ZeroProductReshapePad>(context);
    if (passses & 8)
      patterns.add<BinopConstReshapePad, BinopConstPad<stablehlo::AddOp>,
                   BinopConstPad<stablehlo::SubtractOp>,
                   BinopConstPad<stablehlo::MulOp>,
                   BinopConstPad<stablehlo::DivOp>>(context);

    if (passses & 16)
      patterns.add<
          BinopBinopPadPad<stablehlo::AddOp>, AddPadPadToConcat,
          BinopBinopPadPad<stablehlo::MulOp>, BinopPadPad<stablehlo::AddOp>,
          BinopPadPad<stablehlo::SubtractOp>, BinopPadPad<stablehlo::MulOp>,
          BinopPadPad<stablehlo::DivOp>, BinopPadPad<stablehlo::MinOp>,
          BinopPadPad<stablehlo::MaxOp>>(context);

    if (passses & 32)
      patterns
          .add<UnaryPadPush<stablehlo::ConvertOp>,
               UnaryPadPush<stablehlo::TanhOp>, UnaryPadPush<stablehlo::ExpOp>>(
              context);

    if (passses & 64)
      patterns.add<TransposePad>(context);

    if (passses & 128)
      patterns.add<ReshapePad>(context);

    if (cse) {
      patterns.add<CSE<stablehlo::BroadcastInDimOp>, CSE<stablehlo::SliceOp>,
                   CSE<stablehlo::TransposeOp>, CSE<stablehlo::ConvertOp>,
                   CSE<stablehlo::PadOp>, CSE<stablehlo::DotGeneralOp>,
                   CSE<stablehlo::ReshapeOp>, CSE<stablehlo::MulOp>,
                   CSE<stablehlo::DivOp>, CSE<stablehlo::AddOp>,
                   CSE<stablehlo::SubtractOp>, CSE<stablehlo::MinOp>,
                   CSE<stablehlo::ConcatenateOp>, CSE<stablehlo::MaxOp>,
                   CSE<stablehlo::NegOp>, CSE<stablehlo::AbsOp>,
                   CSE<enzymexla::RotateOp>, CSE<enzymexla::WrapOp>,
                   CSE<enzymexla::ExtendOp>, CSEIota>(context,
                                                      PatternBenefit(65000));
    }

    if (passses & 256)
      patterns.add<TransposeConvert>(context);

    if (passses & 2048)
      patterns.add<TransposeTranspose>(context);

    if (passses & (2048 * 2))
      patterns.add<BroadcastReduce, SliceDotGeneral, SliceReshapeDotGeneral>(
          context);

    if (passses & (2048 * 4)) {
      patterns.add<PadDotGeneral>(false, context);
      patterns.add<DotReshapePad>(context);
    }
    if (passses & (2048 * 8))
      patterns.add<SliceReshape>(context);

    if (passses & (2048 * 16)) {
      patterns.add<PadDotGeneral>(true, context);
      patterns.add<DotReshapePad>(context);
    }

    if (passses & (2048 * 32)) {
      patterns.add<TransposeWhile, TransposeSlice, TransposeConcat,
                   TransposeDUS, TransposeIota, TransposeReduceWindow,
                   TransposeReduce, TransposeSelect, TransposeDynamicSlice,
                   TransposeReverse, TransposeBatchNormTraining,
                   TransposeBatchNormInference, TransposeBatchNormGrad,
                   TransposeIf, TransposeFFT, TransposeReshape>(context);
      patterns.add<TransposeElementwise>(true, context);
    }

    if (passses & (2048 * 64)) {
      // add reshape push up cases here
      patterns.add<ReshapeElementwise, ReshapeSlice>(true, context);
      patterns.add<ReshapeOfConcatToConcatOfReshape, ReshapeDUS, ReshapePad,
                   ReshapeReduceWindow, ReshapeSelect>(context);
    }

    if (passses & (2048 * 128)) {
      // Conflicts with ConcatPad
      patterns.add<ConstPadConcatToConcat>(context);
    }

    if (passses & (2048 * 256)) {
      patterns.add<RecognizeRotate, RecognizeWrap, RecognizeExtend>(context);
    }

    if (passses & (2048 * 512)) {
      patterns.add<LowerRotate, LowerWrap, LowerExtend>(context);
    }

    if (passses & (2048 * 1024)) {
      patterns.add<ConcatToOneDimDUS>(context);
    }

    if (passses & (2048 * 2048)) {
      // push reshapes down
      patterns.add<ElementwiseReshapeLike>(context);
    }

    if (all_finite)
      patterns.add<AllFiniteIsFinite, AllFiniteIsInf, AllFiniteIsPosInf,
                   AllFiniteIsNegInf>(context);
    if (no_nan || all_finite)
      patterns.add<NoNan, NoNanSelfSubSimplify>(context);
    patterns.add<NoNanAddSubSimplify, NoNanMulSimplify, NoNanDivSimplify>(
        (no_nan || all_finite), context);

    // clang-format off
    patterns.add<
        WhileRepeatedInductionReduction,
        WhilePadInductionReduction,
        WhileOpInductionReplacement,
        BroadcastInDimOpCanon,
        ChainedDynamicBroadcastInDimCanonicalization,
        CompareOpCanon,
        CompareExt,
        ConjComplexNegate,
        ConvertOpCanon,
        DivideSqrtToMultiplyRsqrt,
        DynamicBroadcastInDimAllDimsNonExpanding,
        DynamicBroadcastInDimOpNotActuallyDynamic,
        DynamicGatherOpIsNotDynamic,
        DynamicReshapeOpCanon,
        EmptyReduceOpCanon,
        GatherOpCanon,
        GetDimensionSizeOpCanon,
        GetTupleElementOpCanon,
        IfRemoveUnused,
        IfInline,
        IfToSelect,
        IfPredPropagation,
        ImagOpCanon,
        MergeConsecutiveReshapes,
        NoopReduceOpCanon,
        RealOpCanon,
        ReorderElementwiseAndShapeOp,
        ReshapeOpCanon,
        SelectCompIotaConstSimplify,
        SelectCompIotaConstToDUS,
        SelectPadToDUS,
        AndPadPad,
        SelectOpUsedWithinIf,
        TransposeBroadcastInDimToBroadcastInDim,
        BroadcastInDimTransposeToBroadcastInDim,
        TransposeIsReshape,
        BroadcastInDimIsReshape,
        ReshuffleAndsCompares,
        WhileDeadResults,
        ZeroExtentTensorCanon,
        CompareSelectSimplify,
        NotSelectSimplify,
        CommonCompareExpressionRewrite,
        ScatterUpdateComputationConstProp,
        ScatterIndicesAreUnique,
        ReduceTransposeSimplify,
        BroadcastIotaSimplify,
        BroadcastIota,
        BroadcastCompare,
        NotCompare,
        SliceInternal,
        SquareAbsSimplify,
        DivideDivideSimplify,
        ConcatReshapeSlice,
        ConcatReshapeReduce,
        ConcatElementwise,
        ConcatReshapeElementwise,
        TransposeAllUsersSlice,
        ReduceReduce,
        IfOpLiftCommonOps,
        InvolutionSimplify<stablehlo::NegOp>,
        InvolutionSimplify<stablehlo::NotOp>,
        InvolutionSimplify<chlo::ConjOp>,
        RealConjSimplify,
        ConjComplexSimplify,
        SplitConvolutionIntoReverseConvolution,
        ScatterMultiplySimplify,
        UnaryElementwiseScatterSimplify,
        GatherElementwise
      >(context);

    patterns.add<SumToReduceWindow<stablehlo::AddOp>,
    SumToReduceWindow<stablehlo::SubtractOp>>(context);

    patterns.add<WhileSimplify>(false, context);

    patterns.add<WhileLICM>(false, context);

    // clang-format on
    patterns.add<SelectOpCanon>(max_constant_expansion, context,
                                PatternBenefit(65000));
    patterns.add<ConcatenateOpCanon>(max_constant_expansion, context,
                                     PatternBenefit(65000));

    GreedyRewriteConfig config;
    config.setMaxIterations(max_iterations);
    config.setUseTopDownTraversal(top_down);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
