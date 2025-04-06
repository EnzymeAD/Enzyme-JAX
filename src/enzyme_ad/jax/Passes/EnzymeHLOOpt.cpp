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
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "llvm/ADT/MapVector.h"
#include <iterator>
#define DEBUG_TYPE "enzymehloopt"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEHLOOPTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

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
  auto rhsShape = op.getOperand().getType().cast<TensorType>().getShape();
  auto lhsShape = op.getResult().getType().cast<TensorType>().getShape();
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

struct NoopSlice final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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

struct SliceSlice final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::DynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicSliceOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
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
          return false;
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
bool transformReshapeSlice(mlir::stablehlo::ReshapeOp op,
                           SmallVectorImpl<T> &start, T toFill,
                           T *checkRemoved = nullptr) {
  return transformReshapeSlice<T>(op.getOperand().getType(), op.getType(),
                                  start, toFill, checkRemoved);
}

template <typename T>
bool transformReshapeSlice(mlir::stablehlo::ReshapeOp op,
                           SmallVectorImpl<T> &start, std::function<T()> toFill,
                           T *checkRemoved = nullptr) {
  return transformReshapeSlice<T>(op.getOperand().getType(), op.getType(),
                                  start, toFill, checkRemoved);
}

struct ReshapeDUS final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
                     ? startIndices[0].getType().cast<RankedTensorType>()
                     : RankedTensorType::get({}, rewriter.getI64Type());

    if (!transformReshapeSlice<mlir::Value>(
            op, startIndices, /*toFill*/ [&]() -> mlir::Value {
              return rewriter.create<stablehlo::ConstantOp>(
                  dus.getLoc(), itype, makeAttr(itype, 0).cast<ElementsAttr>());
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicUpdateSliceOp>(
        op, newOperand, newUpdate, startIndices);

    return success();
  }
};

struct ReshapeSlice final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the input to Reshape is a DynamicUpdateSlice
    auto slice = op.getOperand().getDefiningOp<stablehlo::SliceOp>();
    if (!slice)
      return failure();

    if (!llvm::hasSingleElement(slice->getUsers()))
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::SliceOp>(
        op, newOperand, startIndices, limitIndices, stepIndices);

    return success();
  }
};

struct ReshapePad final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::PadOp>(
        op, newOperand, pad.getPaddingValue(), low, high, interior);

    return success();
  }
};

struct TransposeDUS final : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {
    auto concatOp = dus.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concatOp) {
      return failure();
    }

    auto dusOperandType =
        dyn_cast<RankedTensorType>(dus.getOperand().getType());
    auto updateType = dyn_cast<RankedTensorType>(dus.getUpdate().getType());
    if (!dusOperandType || !updateType || !dusOperandType.hasStaticShape() ||
        !updateType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          dus, "Requires static shapes for DUS operand and update");
    }

    int64_t concatDim = concatOp.getDimension();
    ArrayRef<int64_t> dusOperandShape = dusOperandType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();
    SmallVector<Value> startIndices = dus.getStartIndices();

    DenseIntElementsAttr concatStartAttr;
    if (!matchPattern(startIndices[concatDim], m_Constant(&concatStartAttr))) {
      return rewriter.notifyMatchFailure(
          dus, "Requires constant start index for concat dimension");
    }
    if (concatStartAttr.getNumElements() != 1) {
      return rewriter.notifyMatchFailure(dus,
                                         "Concat start index must be a scalar");
    }

    int64_t concatStartVal = (*concatStartAttr.begin()).getSExtValue();
    int64_t concatUpdateSize = updateShape[concatDim];
    int64_t concatEndVal = concatStartVal + concatUpdateSize; // Exclusive end

    int64_t currentOffset = 0;
    int targetInputIdx = -1;
    Value targetInputVal;

    for (const auto &indexedInput : llvm::enumerate(concatOp.getInputs())) {
      Value input = indexedInput.value();
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      if (!inputType || !inputType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            dus, "Requires static shapes for ConcatOp inputs");
      }
      int64_t inputSize = inputType.getShape()[concatDim];

      // Check if the DUS update region falls entirely within this input's
      // region
      if (concatStartVal >= currentOffset &&
          concatEndVal <= (currentOffset + inputSize)) {
        targetInputIdx = indexedInput.index();
        targetInputVal = input;
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

    SmallVector<Value> newDusStartIndices =
        llvm::to_vector(dus.getStartIndices());
    Location loc = dus.getLoc();
    auto indexElementType = startIndices[concatDim]
                                .getType()
                                .cast<ShapedType>()
                                .getElementType(); // Assuming all start indices
                                                   // have same scalar type
    auto indexScalarType = RankedTensorType::get({}, indexElementType);

    int64_t newConcatStartVal = concatStartVal - currentOffset;
    auto newStartAttr =
        rewriter.getIntegerAttr(indexElementType, newConcatStartVal);
    newDusStartIndices[concatDim] = rewriter.create<stablehlo::ConstantOp>(
        dus.getLoc(), newDusStartIndices[concatDim].getType(),
        makeAttr(newDusStartIndices[concatDim].getType(),
                 concatStartVal - currentOffset)
            .cast<ElementsAttr>());

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
        dus.getType(), // The result type should match the original DUS result
        newConcatOperands, concatDim);

    // Replace the original DUS with the new concatenate op.
    rewriter.replaceOp(dus, newConcat.getResult());

    return success();
  }
};

struct DUSDUS final : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {
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
struct DUSPad final : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {

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
    // High padding isn't directly used for the check below but needed for
    // reconstruction
    ArrayRef<int64_t> highPadding = padOp.getEdgePaddingHigh();

    SmallVector<Value> newDusStartIndices;
    Location loc = dus.getLoc();
    auto indexElementType =
        startIndices[0].getType().cast<ShapedType>().getElementType();
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
    : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp dus,
                                PatternRewriter &rewriter) const override {
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
          dus.getLoc(), itype, makeAttr(itype, 0).cast<ElementsAttr>());
      auto cidx = rewriter.create<stablehlo::ConstantOp>(
          dus.getLoc(), itype,
          makeAttr(itype, idxs[0] - idxs[1]).cast<ElementsAttr>());

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
    : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
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

struct SliceOfDynamicUpdate final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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

struct SliceDUSToConcat final : OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

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
template <typename T> struct LICM : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  bool single_user;
  LICM(bool single_user, MLIRContext *context, PatternBenefit benefit = 1,
       ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<T>(context, benefit, generatedNames),
        single_user(single_user) {}

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
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

struct LICMElementwise : public OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern<OpTrait::Elementwise>::OpTraitRewritePattern;
  bool single_user;
  LICMElementwise(bool single_user, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpTraitRewritePattern<OpTrait::Elementwise>(context, benefit),
        single_user(single_user) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
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
struct SliceBroadcast final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto bcast = op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!bcast)
      return failure();

    if (!llvm::hasSingleElement(bcast->getUsers()))
      return failure();

    SmallVector<int64_t> nbcast_idx;

    auto preShape = bcast.getOperand().getType().cast<RankedTensorType>();
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
struct SliceTranspose final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
struct SliceReduceWindow : public OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern<mlir::stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
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
        reduceWindow.getInputs()[0].getType().dyn_cast<ShapedType>();
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

// transpose(slice x) -> slice(transpose x)
struct TransposeSlice final : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    auto slice = op.getOperand().getDefiningOp<stablehlo::SliceOp>();
    // if (!slice || !llvm::hasSingleElement(slice->getUsers()))
    if (!slice)
      return failure();

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
    return success();
  }
};

struct SliceElementwise final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
        if (start < starts[en.index()])
          starts[en.index()] = start;
        if (stop > stops[en.index()])
          stops[en.index()] = stop;
        if (stride != ints[en.index()])
          ints[en.index()] = 1;
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
    // pad result is [0..lpad][lpad...outshape-hpad][outshape-hpad...outshape]

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
struct SlicePad final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
      op.getInputs()[0].getType().cast<ShapedType>().getElementType();
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

struct ReduceToReshape final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return failure();
    if (!isEligibleForCompactPrint(op))
      return failure();
    auto inpTy = op.getInputs()[0].getType().cast<RankedTensorType>();
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

struct ReducePad : public OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-operand single-init reduce is supported");
    }
    // TODO: min/max can also be an option since they are dropped
    if (!isa<stablehlo::AddOp>(op.getRegion().getBlocks().front().front())) {
      return rewriter.notifyMatchFailure(op, "only add is currently supported");
    }

    Value input = op.getInputs()[0];
    auto pad = input.getDefiningOp<mlir::stablehlo::PadOp>();
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
             pad.getOperand().getType().cast<RankedTensorType>().getShape())) {
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
            op->getResultTypes()[0].cast<RankedTensorType>().getElementType())),
        ValueRange(pad.getOperand()), op.getInitValues(), op.getDimensions());
    newReduction.getRegion().takeBody(op.getRegion());

    Value res = newReduction->getResult(0);
    if (needsPostPad) {
      auto ctype = RankedTensorType::get(
          {}, res.getType().cast<RankedTensorType>().getElementType());
      res = rewriter.create<stablehlo::PadOp>(
          op.getLoc(), res,
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), ctype, makeAttr(ctype, 0).cast<ElementsAttr>()),
          low, high, inner);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertConcat final : OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto concat = op.getOperand().getDefiningOp<stablehlo::ConcatenateOp>();
    if (!concat)
      return failure();

    SmallVector<Value> newvals;
    for (auto v : concat.getOperands()) {
      newvals.push_back(rewriter.create<stablehlo::ConvertOp>(
          op.getLoc(),
          RankedTensorType::get(v.getType().cast<RankedTensorType>().getShape(),
                                op.getType().getElementType()),
          v));
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(
        op, newvals, concat.getDimension());
    return success();
  }
};

struct ConvertConvertFloat final
    : OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto conv0 = op.getOperand().getDefiningOp<stablehlo::ConvertOp>();
    if (!conv0)
      return failure();

    auto prev = conv0.getOperand();
    if (prev.getType().getElementType().isa<FloatType>() &&
        op.getType().getElementType().isa<FloatType>() &&
        conv0.getType().getElementType().isa<FloatType>()) {
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

struct ReduceConcat final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return failure();

    auto inpTy = op.getInputs()[0].getType().cast<RankedTensorType>();
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
    while (todo.size()) {
      auto cur = todo.pop_back_val();

      SmallVector<Value> vals;
      for (auto op : cur->getOperands())
        vals.push_back(map.lookup(op));

      auto changeType2 = RankedTensorType::get(changeType.getShape(),
                                               cur->getResult(0)
                                                   .getType()
                                                   .cast<RankedTensorType>()
                                                   .getElementType());
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
    auto ty = v.getType().cast<RankedTensorType>();
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

struct ConcatSlice final : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto dim = op.getDimension();

    SmallVector<Value> newOperands;

    for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
      auto operand = op->getOperand(i);
      auto slice = operand.getDefiningOp<stablehlo::SliceOp>();

      if (!slice) {
        newOperands.push_back(operand);
        continue;
      }

      stablehlo::SliceOp otherSlice;
      while (i + 1 < e &&
             (otherSlice =
                  op->getOperand(i + 1).getDefiningOp<stablehlo::SliceOp>())) {
        if (otherSlice.getOperand() != slice.getOperand())
          break;

        bool canMerge = true;

        // Check that both slices are contiguous only in dim
        ArrayRef<int64_t> sliceStarts = slice.getStartIndices(),
                          otherSliceStarts = otherSlice.getStartIndices(),
                          sliceLimits = slice.getLimitIndices(),
                          otherSliceLimits = otherSlice.getLimitIndices(),
                          sliceStrides = slice.getStrides(),
                          otherSliceStrides = otherSlice.getStrides();

        for (int d = 0, ndims = sliceStarts.size(); d < ndims; ++d) {
          if (d == dim) {
            canMerge &= sliceLimits[d] == otherSliceStarts[d] &&
                        sliceStrides[d] == otherSliceStrides[d];
          } else {
            canMerge &= sliceStarts[d] == otherSliceStarts[d] &&
                        sliceLimits[d] == otherSliceLimits[d] &&
                        sliceStrides[d] == otherSliceStrides[d];
          }
        }

        if (canMerge) {
          slice = rewriter.create<stablehlo::SliceOp>(
              slice->getLoc(), slice.getOperand(), sliceStarts,
              otherSliceLimits, sliceStrides);
          i++;
        } else
          break;
      }

      newOperands.push_back(slice.getResult());
    }

    if (newOperands.size() == op->getNumOperands())
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, newOperands, dim);
    return success();
  }
};

struct SliceConcat final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
  return mlir::stablehlo::makeDenseElementsAttr(tensor);
}

/*
%22 = stablehlo.dot_general %21, %16, contracting_dims = [1] x [0], precision =
[DEFAULT, DEFAULT] : (tensor<288x288xf32>, tensor<288xf32>) -> tensor<288xf32>
%27 = stablehlo.reshape %22 : (tensor<288xf32>) -> tensor<144x2xf32>
%28 = stablehlo.dot_general %6, %27, batching_dims = [0] x [0], contracting_dims
= [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<144x2x2xf32>,
tensor<144x2xf32>) -> tensor<144x2xf32>

should become

%a21 = stablehlo.reshape %21 : (tensor<288xf32>) -> tensor<144x2xf32>

%22 = stablehlo.dot_general %a21, %16, batching_dims = [1] x [],
contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] :
(tensor<144x2x288xf32>, tensor<288xf32>) -> tensor<2x144xf32>

%28 = stablehlo.dot_general %6, %22, batching_dims = [0] x [1], contracting_dims
= [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<144x2x2xf32>,
tensor<144x2xf32>) -> tensor<144x2xf32>

TODO
*/

struct DotReshapeDot final : OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    return failure();
  }
};

struct PadSimplify final : OpRewritePattern<mlir::stablehlo::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::PadOp op,
                                PatternRewriter &rewriter) const override {

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
        auto ten = mlir::stablehlo::constantOp(inp);
        auto out = fromTensor(mlir::stablehlo::padOp(
            mlir::stablehlo::constantOp(inp), mlir::stablehlo::constantOp(pv),
            stablehlo::Sizes(op.getEdgePaddingLow()),
            stablehlo::Sizes(op.getInteriorPadding()), op.getType()));

        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                           out);
        return success();
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

struct ShiftRightLogicalSimplify final
    : OpRewritePattern<mlir::stablehlo::ShiftRightLogicalOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ShiftRightLogicalOp op,
                                PatternRewriter &rewriter) const override {

    DenseElementsAttr lhs;
    matchPattern(op.getLhs(), m_Constant(&lhs));
    DenseElementsAttr rhs;
    matchPattern(op.getRhs(), m_Constant(&rhs));
    if (lhs && rhs) {
      auto out = fromTensor(mlir::stablehlo::shiftRightLogicalOp(
          mlir::stablehlo::constantOp(lhs), mlir::stablehlo::constantOp(rhs),
          op.getType()));

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct WhileDeadResults final : OpRewritePattern<mlir::stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  bool isLoopResultDead(OpResult result, ArrayRef<int64_t> deadResults,
                        bool &retryIfNewDead) const {
    // Not dead if the result is in use.
    if (!result.use_empty())
      return false;

    // Or if the corresponding argument is being used in computing the
    // condition.
    auto whileOp = cast<mlir::stablehlo::WhileOp>(result.getOwner());
    Value condArgument =
        whileOp.getCond().getArgument(result.getResultNumber());
    SetVector<Operation *> forwardSlice;
    getForwardSlice(condArgument, &forwardSlice);
    if (!llvm::all_of(forwardSlice, mlir::isPure))
      return false;
    if (forwardSlice.contains(whileOp.getCond().front().getTerminator()))
      return false;

    // Or in computing another result. We first do a fast-path check of having
    // the argument not influencing the terminator operation, before going into
    // finer-grain analysis.
    //
    // TODO: it is possible that this argument does influence another terminator
    // operand, but that operand in turn corresponds to a dead value, but
    // handling that would require more complex logic of detecting dead cycles
    // of value chains.
    forwardSlice.clear();
    assert(llvm::hasSingleElement(whileOp.getBody()));
    Value bodyArgument =
        whileOp.getBody().getArgument(result.getResultNumber());
    getForwardSlice(bodyArgument, &forwardSlice);
    if (!llvm::all_of(forwardSlice, mlir::isPure))
      return false;

    Operation *bodyTerminator = whileOp.getBody().front().getTerminator();
    if (!forwardSlice.contains(bodyTerminator))
      return true;

    for (OpOperand &terminatorOperand : bodyTerminator->getOpOperands()) {
      if (terminatorOperand.getOperandNumber() == result.getResultNumber())
        continue;

      if (llvm::is_contained(deadResults, terminatorOperand.getOperandNumber()))
        continue;
      // We directly yield an argument from a different index (since we skip
      // the return of the given result).
      if (auto ba = dyn_cast<BlockArgument>(terminatorOperand.get())) {
        if (ba.getOwner()->getParentOp() == whileOp) {
          if (terminatorOperand.get() == bodyArgument) {
            retryIfNewDead = true;
            return false;
          } else {
            continue;
          }
        }
      }

      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      getBackwardSlice(terminatorOperand.get(), &backwardSlice, options);
      for (Operation *op : backwardSlice) {
        if (llvm::is_contained(op->getOperands(), bodyArgument)) {
          retryIfNewDead = true;
          return false;
        }
      }
    }
    return true;
  }

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
    auto term2 = rewriter.replaceOpWithNewOp<mlir::stablehlo::ReturnOp>(
        terminator, TypeRange(), terminatorOperands, terminator->getAttrs());
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> deadResults;
    do {
      bool newDead = false;
      bool retry = false;
      for (OpResult result : op.getResults()) {
        if (llvm::is_contained(deadResults, result.getResultNumber()))
          continue;
        if (!isLoopResultDead(result, deadResults, retry))
          continue;
        deadResults.push_back(result.getResultNumber());
        newDead = true;
      }
      if (newDead && retry)
        continue;
    } while (false);
    if (deadResults.empty())
      return failure();

    llvm::sort(deadResults);

    SetVector<Operation *> condSlice, bodySlice;
    for (int64_t i : deadResults) {
      getForwardSlice(op.getCond().getArgument(i), &condSlice);
      getForwardSlice(op.getBody().getArgument(i), &bodySlice);
    }
    condSlice.remove(op.getCond().front().getTerminator());
    bodySlice.remove(op.getBody().front().getTerminator());
    replaceTerminator(rewriter, op.getBody(), deadResults);

    condSlice = mlir::topologicalSort(condSlice);
    bodySlice = mlir::topologicalSort(bodySlice);
    for (Operation *erasable : llvm::reverse(condSlice))
      rewriter.eraseOp(erasable);
    for (Operation *erasable : llvm::reverse(bodySlice))
      rewriter.eraseOp(erasable);

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

    auto updated = rewriter.create<mlir::stablehlo::WhileOp>(
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

struct NegativePadToSlice final : OpRewritePattern<mlir::stablehlo::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::PadOp op,
                                PatternRewriter &rewriter) const override {
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

    %1192 = stablehlo.pad %1189, %cst_0, low = [0], high = [1], interior = [0] :
   (tensor<1xf32>, tensor<f32>) -> tensor<2xf32> %1193 = arith.addf %1191, %1192
   : tensor<2xf32>

*/
template <typename T> struct BinopPadToConcat final : OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
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

struct ReshapeIota final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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

struct DotReshapePad final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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

struct PadReshapePad final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2)
      return failure();

    SmallVector<Value> lhs;

    SmallVector<Type> converts;

    size_t frontSize = 0;
    for (auto v : op.getOperands()) {
      if (auto t = v.getDefiningOp<stablehlo::ConvertOp>()) {
        v = t.getOperand();
        converts.push_back(
            v.getType().cast<RankedTensorType>().getElementType());
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

template <typename T> struct UnaryPadPush final : OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
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
        RankedTensorType::get(
            val.getType().template cast<RankedTensorType>().getShape(),
            op.getType().getElementType()),
        val);

    rewriter.replaceOpWithNewOp<stablehlo::PadOp>(
        op, val2, padval2, pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
        pad.getInteriorPadding());
    return success();
  }
};

struct TransposePad final : OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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
    return success();
  }
};

template <typename T>
struct ConcatPushBinop final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2)
      return failure();

    SmallVector<Value> lhs;
    SmallVector<Value> rhs;

    SmallVector<Type> converts;

    for (auto v : op.getOperands()) {
      if (auto t = v.getDefiningOp<stablehlo::ConvertOp>()) {
        converts.push_back(
            t.getType().cast<RankedTensorType>().getElementType());
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
              res2.getType().cast<RankedTensorType>().getShape(), typeconvert),
          res2);

    rewriter.replaceOp(op, res2);
    return success();
  }
};

struct ConcatFuse final : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
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
      if (v.getType().cast<RankedTensorType>().getShape()[op.getDimension()] ==
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
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() <= 1)
      return failure();
    for (auto opv : op->getOperands())
      if (opv != op->getOperand(0))
        return failure();
    SmallVector<int64_t> bcast;
    if (op->getOperand(0)
            .getType()
            .cast<RankedTensorType>()
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

struct GammaConstProp final : OpRewritePattern<mlir::chlo::LgammaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::chlo::LgammaOp op,
                                PatternRewriter &rewriter) const override {
    // return if not constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&inputAttr)))
      return failure();
    Value result = mlir::stablehlo::materializeLgamma(rewriter, op.getLoc(),
                                                      op->getOperands());
    rewriter.replaceOp(op, result);

    return success();
  }
};

struct DynamicUpdateSliceConstProp final
    : OpRewritePattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp op,
                                PatternRewriter &rewriter) const override {
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

    if (!legal)
      return failure();

    if (operandConstant.isSplat() && updateConstant.isSplat() &&
        ((isa<FloatType>(op.getType().getElementType()) &&
          operandConstant.getSplatValue<llvm::APFloat>() ==
              updateConstant.getSplatValue<llvm::APFloat>()) ||
         (isa<IntegerType>(op.getType().getElementType()) &&
          operandConstant.getSplatValue<llvm::APInt>() ==
              updateConstant.getSplatValue<llvm::APInt>()))) {
      rewriter.replaceAllUsesWith(op.getResult(), op.getOperand());
      return success();
    }

    stablehlo::Tensor operandTen = mlir::stablehlo::constantOp(operandConstant);
    stablehlo::Tensor updateTen = mlir::stablehlo::constantOp(updateConstant);
    SmallVector<stablehlo::Tensor> inps;
    for (auto &c : constants)
      inps.push_back(mlir::stablehlo::constantOp(c));

    auto out = mlir::stablehlo::dynamicUpdateSliceOp(operandTen, updateTen,
                                                     inps, op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                       fromTensor(out));

    return success();
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
    inputTen = mlir::stablehlo::constantOp(inputAttr);
  }
  // get the resultType
  auto resultType = ty.cast<ShapedType>();

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

struct NotConstProp final : OpRewritePattern<mlir::stablehlo::NotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::NotOp op,
                                PatternRewriter &rewriter) const override {
    return unaryConstProp<mlir::stablehlo::notOp>(op, rewriter);
  }
};

struct IsFiniteConstProp final : OpRewritePattern<mlir::stablehlo::IsFiniteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IsFiniteOp op,
                                PatternRewriter &rewriter) const override {
    return unaryConstProp<mlir::stablehlo::isFiniteOp>(op, rewriter);
  }
};

struct LogConstProp final : OpRewritePattern<mlir::stablehlo::LogOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::LogOp op,
                                PatternRewriter &rewriter) const override {
    return unaryConstProp<mlir::stablehlo::logOp>(op, rewriter);
  }
};

struct LogPlusConstProp final : OpRewritePattern<mlir::stablehlo::Log1pOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::Log1pOp op,
                                PatternRewriter &rewriter) const override {
    return unaryConstProp<stablehlo::log1pOp>(op, rewriter);
  }
};

struct ChloInfConstProp final : OpRewritePattern<mlir::chlo::IsInfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::chlo::IsInfOp op,
                                PatternRewriter &rewriter) const override {

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
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, op->getResultTypes()[0], outAttr);

    return success();
  }
};

struct ConcatConstProp final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
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

      SmallVector<stablehlo::Tensor> inps;
      for (auto &c : constants)
        inps.push_back(mlir::stablehlo::constantOp(c));
      auto out =
          mlir::stablehlo::concatenateOp(inps, op.getDimension(), op.getType());
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                         fromTensor(out));
      return success();
    }
    return failure();
  }
};

struct ReshapeEmptyBroadcast final
    : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, op.getType(),
                                                      op.getOperand());
    return success();
  }
};

struct BroadcastPad final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {

    auto pad = op.getOperand().getDefiningOp<mlir::stablehlo::PadOp>();
    if (!pad)
      return failure();

    if (!llvm::hasSingleElement(pad->getUsers()))
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
    : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
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

    auto retop = innerOp.getOperand(0).dyn_cast<BlockArgument>();
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
            op.getUpdates()[0].getType().cast<ShapedType>(),
            attr.getSplatValue<Attribute>());
      } else {
        return failure();
      }
    }

    auto dims = op.getScatterDimensionNumbers();

    auto input = op.getInputs()[0];
    auto scatter = op.getScatterIndices();
    auto updateShape =
        op.getUpdates()[0].getType().cast<ShapedType>().getShape();

    if (dims.getInsertedWindowDims().size() == 0 &&
        dims.getUpdateWindowDims().size() == updateShape.size()) {

      if (update == nullptr) {
        update = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), op.getUpdates()[0].getType(), splatAttr);
      }

      auto ity = RankedTensorType::get(
          {}, scatter.getType().cast<ShapedType>().getElementType());
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
            op.getLoc(), ity, makeAttr(ity, 0).template cast<ElementsAttr>());
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

llvm::SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> perm) {
  llvm::SmallVector<int64_t> inversePerm(perm.size(), -1);
  for (int64_t i = 0; i < perm.size(); ++i) {
    inversePerm[perm[i]] = i;
  }
  return inversePerm;
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
struct BinaryOpTransposeSimplify : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    return simplifyBinaryOpWithTranspose(op, rewriter);
  }
};

template <typename OpType>
struct TransposeUnaryTransposeSimplify
    : public OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern<stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp outerTransposeOp,
                                PatternRewriter &rewriter) const override {
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

struct AddSimplify : public OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern<mlir::stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {

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

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                APFloat res2(a);
                res2.add(b, llvm::RoundingMode::NearestTiesToEven);
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                APInt res2(a);
                res2 += b;
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct ReplaceNegAddWithSubtract : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {
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

struct SubSimplify : public OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern<mlir::stablehlo::SubtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const final {

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

    if (isa<IntegerType>(op.getType().getElementType()) &&
        op.getLhs() == op.getRhs()) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                APFloat res2(a);
                res2.subtract(b, llvm::RoundingMode::NearestTiesToEven);
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                APInt res2(a);
                res2 -= b;
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct NoNanSelfSubSimplify
    : public OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern<mlir::stablehlo::SubtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const final {

    if (op.getLhs() == op.getRhs()) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }

    return failure();
  }
};

struct NegateSimplify : public OpRewritePattern<mlir::stablehlo::NegOp> {
  using OpRewritePattern<mlir::stablehlo::NegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::NegOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res =
              mlir::constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType,
                                                void>(
                  constants, [](const APFloat &a) -> std::optional<APFloat> {
                    return -a;
                  })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else {
      if (auto res =
              mlir::constFoldUnaryOpConditional<IntegerAttr,
                                                IntegerAttr::ValueType, void>(
                  constants,
                  [](const APInt &a) -> std::optional<APInt> { return -a; })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct AndSimplify : public OpRewritePattern<mlir::stablehlo::AndOp> {
  using OpRewritePattern<mlir::stablehlo::AndOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AndOp op,
                                PatternRewriter &rewriter) const final {

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

    SmallVector<Attribute> constants(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));
    if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                IntegerAttr::ValueType, void>(
            constants,
            [](const APInt &a, const APInt &b) -> std::optional<APInt> {
              APInt res2(a);
              res2 &= b;
              return res2;
            })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct OrSimplify : public OpRewritePattern<mlir::stablehlo::OrOp> {
  using OpRewritePattern<mlir::stablehlo::OrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::OrOp op,
                                PatternRewriter &rewriter) const final {

    // true | x -> x
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

    SmallVector<Attribute> constants(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));
    if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                IntegerAttr::ValueType, void>(
            constants,
            [](const APInt &a, const APInt &b) -> std::optional<APInt> {
              APInt res2(a);
              res2 |= b;
              return res2;
            })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct MulSimplify : public OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const final {

    // 0 * x -> x
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // x * 0 -> x
    if (matchPattern(op.getRhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

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

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                APFloat res2(a);
                res2.multiply(b, llvm::RoundingMode::NearestTiesToEven);
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                APInt res2(a);
                res2 *= b;
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct DivSimplify : public OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op,
                                PatternRewriter &rewriter) const final {

    // 0 / x -> 0 [assume non nan here]
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getLhs(), m_Zero())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // x / 1 -> x
    if (matchPattern(op.getRhs(), m_OneFloat()) ||
        matchPattern(op.getRhs(), m_One())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                APFloat res2(a);
                res2.divide(b, llvm::RoundingMode::NearestTiesToEven);
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                APInt res2(a);
                return res2.sdiv(b);
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct RemSimplify : public OpRewritePattern<mlir::stablehlo::RemOp> {
  using OpRewritePattern<mlir::stablehlo::RemOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RemOp op,
                                PatternRewriter &rewriter) const final {

    if (matchPattern(op.getRhs(), m_One())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                APFloat res2(a);
                res2.remainder(b);
                return res2;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                APInt res2(a);
                return res2.srem(b);
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }
    return failure();
  }
};

struct PowSimplify : public OpRewritePattern<mlir::stablehlo::PowOp> {
  using OpRewritePattern<mlir::stablehlo::PowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::PowOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {
      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                if (a.getSizeInBits(a.getSemantics()) == 64 &&
                    b.getSizeInBits(b.getSemantics()) == 64)
                  return APFloat(pow(a.convertToDouble(), b.convertToDouble()));

                if (a.getSizeInBits(a.getSemantics()) == 32 &&
                    b.getSizeInBits(b.getSemantics()) == 32)
                  return APFloat(powf(a.convertToFloat(), b.convertToFloat()));

                return {};
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }

      // pow(X, 0.5) -> sqrt(X)
      {
        DenseFPElementsAttr rhs;
        if (matchPattern(op.getRhs(), m_Constant(&rhs))) {
          bool allHalf = true;
          for (auto v : rhs) {
            if (!v.isExactlyValue(0.5)) {
              allHalf = false;
              break;
            }
          }
          if (allHalf) {
            rewriter.replaceOpWithNewOp<stablehlo::SqrtOp>(op, op.getLhs());
            return success();
          }
        }
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                if (b.isNegative())
                  return {}; // Ignore the negative case

                APInt result = APInt(a.getBitWidth(), 1);
                APInt base = a;
                uint64_t exponent = b.getLimitedValue();

                while (exponent > 0) {
                  if (exponent % 2 == 1) {
                    result *= base;
                  }
                  base *= base;
                  exponent /= 2;
                }

                return result;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }

    return failure();
  }
};

struct IotaSimplify : public OpRewritePattern<mlir::stablehlo::IotaOp> {
  using OpRewritePattern<mlir::stablehlo::IotaOp>::OpRewritePattern;
  size_t max_constant_expansion;
  IotaSimplify(size_t max_constant_expansion, MLIRContext *context,
               PatternBenefit benefit = 1,
               ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}
  LogicalResult matchAndRewrite(mlir::stablehlo::IotaOp op,
                                PatternRewriter &rewriter) const final {
    size_t size = 1;
    for (auto sz : op.getType().getShape())
      size *= sz;
    if (size >= max_constant_expansion)
      return failure();

    auto out = mlir::stablehlo::iotaOp(op.getIotaDimension(), op.getType());
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                       fromTensor(out));
    return success();
  }
};

struct ConcatToPad : public OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const final {
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
    : public OpRewritePattern<mlir::stablehlo::ReduceWindowOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceWindowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const final {
    if (op->getNumOperands() != 2)
      return failure();

    if (op.getPadding().has_value() &&
        !llvm::all_of(op.getPadding().value(),
                      [](auto pad) { return pad.isZero(); }))
      return failure();

    Value operand = op->getOperand(0), initValue = op->getOperand(1);

    auto padOp = operand.getDefiningOp<mlir::stablehlo::PadOp>();
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

    auto newOp = rewriter.create<mlir::stablehlo::ReduceWindowOp>(
        op.getLoc(), op.getResult(0).getType(), padOp.getOperand(), initValue,
        op.getWindowDimensionsAttr(), op.getWindowStridesAttr(),
        op.getBaseDilationsAttr(), op.getWindowDilationsAttr(), newPaddingAttr);
    newOp.getRegion().takeBody(op.getRegion());

    rewriter.replaceAllUsesWith(op.getResult(0), newOp.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConcatPad : public OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const final {
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
};

struct ConvertSimplify : public OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern<mlir::stablehlo::ConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const final {
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
        ten = mlir::stablehlo::constantOp(inp);
      }
      auto out = fromTensor(mlir::stablehlo::convertOp(ten, ty));
      if (inp.isSplat())
        out = out.resizeSplat(op.getType());

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct SliceSimplify : public OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern<mlir::stablehlo::SliceOp>::OpRewritePattern;

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

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
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
          auto ten = mlir::stablehlo::constantOp(inp);
          out = fromTensor(mlir::stablehlo::sliceOp(
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
    : public OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern<mlir::stablehlo::BroadcastInDimOp>::OpRewritePattern;

  size_t max_constant_expansion;
  BroadcastInDimSimplify(size_t max_constant_expansion, MLIRContext *context,
                         PatternBenefit benefit = 1,
                         ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const final {
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
        auto ten = mlir::stablehlo::constantOp(inp);
        out = fromTensor(mlir::stablehlo::broadcastInDimOp(
            ten, mlir::stablehlo::Axes(op.getBroadcastDimensions()),
            op.getType()));
      }

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }

    return failure();
  }
};

struct BroadcastIotaSimplify
    : public OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp broadcast,
                                PatternRewriter &rewriter) const final {
    auto operand = broadcast.getOperand();
    DenseIntElementsAttr input;
    matchPattern(operand, m_Constant(&input));

    if (input) {
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

        const auto start = (*curr).getInt();
        const auto diff = (*next).getInt() - (*curr).getInt();

        if (diff == 0)
          return failure();

        while (next != end) {
          auto curr_diff = (*next).getInt() - (*curr).getInt();
          if (curr_diff != diff)
            return failure();
          ++curr;
          ++next;
        }
        auto result_type = broadcast->getResultTypes();
        auto loc = broadcast.getLoc();
        rewriter.setInsertionPointAfter(operand.getDefiningOp());

        // find the dimension to broadcast in
        auto broadcast_dim = 0Z;
        auto result_shape =
            result_type.front().template cast<mlir::ShapedType>().getShape();
        auto max_dims = result_shape.size();

        for (broadcast_dim = 0Z; broadcast_dim < max_dims; ++broadcast_dim) {
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

        // build the replacement operations
        auto iota = rewriter.create<mlir::stablehlo::IotaOp>(loc, result_type,
                                                             broadcast_dim);
        auto stride_attr = mlir::DenseElementsAttr::get(
            operand.getType().cloneWith(result_shape, elemType),
            rewriter.getIntegerAttr(elemType, diff));
        auto start_attr = mlir::DenseElementsAttr::get(
            operand.getType().cloneWith(result_shape, elemType),
            rewriter.getIntegerAttr(elemType, start));
        auto stride_const = rewriter.create<mlir::stablehlo::ConstantOp>(
            loc, result_type, stride_attr);
        auto start_const = rewriter.create<mlir::stablehlo::ConstantOp>(
            loc, result_type, start_attr);
        auto mul =
            rewriter.create<mlir::stablehlo::MulOp>(loc, iota, stride_const);

        rewriter.replaceOpWithNewOp<mlir::stablehlo::AddOp>(broadcast,
                                                            start_const, mul);
        return success();
      }
      return failure();
    }

    return failure();
  }
};

struct DotGeneralSimplify
    : public OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern<mlir::stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const final {
    if (matchPattern(op.getLhs(), m_AnyZeroFloat()) ||
        matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, rewriter.getZeroAttr(op.getType()));
      return success();
    }
    return failure();
  }
};

struct TransposeSimplify
    : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {

      DenseElementsAttr out;
      if (inp.isSplat()) {
        out = inp.resizeSplat(op.getType());
      } else {
        out = fromTensor(mlir::stablehlo::transposeOp(
            stablehlo::constantOp(inp),
            mlir::stablehlo::Axes(op.getPermutation()), op.getType()));
      }
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }
    return failure();
  }
};

struct MaxSimplify : public OpRewritePattern<mlir::stablehlo::MaxOp> {
  using OpRewritePattern<mlir::stablehlo::MaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MaxOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getOperand(0) == op.getOperand(1)) {
      rewriter.replaceOp(op, op.getOperand(0));
      return success();
    }
    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {

      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                return (a > b) ? a : b;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                return a.sgt(b) ? a : b;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }
    return failure();
  }
};

struct MinSimplify : public OpRewritePattern<mlir::stablehlo::MinOp> {
  using OpRewritePattern<mlir::stablehlo::MinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MinOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getOperand(0) == op.getOperand(1)) {
      rewriter.replaceOp(op, op.getOperand(0));
      return success();
    }
    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (op.getType().getElementType().isa<FloatType>()) {

      if (auto res = constFoldBinaryOpConditional<FloatAttr,
                                                  FloatAttr::ValueType, void>(
              constants,
              [](const APFloat &a, const APFloat &b) -> std::optional<APFloat> {
                return (a < b) ? a : b;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    } else if (op.getType().getElementType().isa<IntegerType>()) {
      if (auto res = constFoldBinaryOpConditional<IntegerAttr,
                                                  IntegerAttr::ValueType, void>(
              constants,
              [](const APInt &a, const APInt &b) -> std::optional<APInt> {
                return a.slt(b) ? a : b;
              })) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), res.cast<ElementsAttr>());
        return success();
      }
    }
    return failure();
  }
};

struct CosSimplify : public OpRewritePattern<mlir::stablehlo::CosineOp> {
  using OpRewritePattern<mlir::stablehlo::CosineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CosineOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res =
            constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants, [](const APFloat &a) -> std::optional<APFloat> {
                  if (a.getSizeInBits(a.getSemantics()) == 64)
                    return APFloat(cos(a.convertToDouble()));

                  if (a.getSizeInBits(a.getSemantics()) == 32)
                    return APFloat(cosf(a.convertToFloat()));
                  return {};
                })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct SinSimplify : public OpRewritePattern<mlir::stablehlo::SineOp> {
  using OpRewritePattern<mlir::stablehlo::SineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SineOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res =
            constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants, [](const APFloat &a) -> std::optional<APFloat> {
                  if (a.getSizeInBits(a.getSemantics()) == 64)
                    return APFloat(sin(a.convertToDouble()));

                  if (a.getSizeInBits(a.getSemantics()) == 32)
                    return APFloat(sinf(a.convertToFloat()));
                  return {};
                })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct SqrtSimplify : public OpRewritePattern<mlir::stablehlo::SqrtOp> {
  using OpRewritePattern<mlir::stablehlo::SqrtOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SqrtOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res =
            constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants, [](const APFloat &a) -> std::optional<APFloat> {
                  if (a.getSizeInBits(a.getSemantics()) == 64)
                    return APFloat(sqrt(a.convertToDouble()));

                  if (a.getSizeInBits(a.getSemantics()) == 32)
                    return APFloat(sqrtf(a.convertToFloat()));
                  return {};
                })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct TanhSimplify : public OpRewritePattern<mlir::stablehlo::TanhOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TanhOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res =
            constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants, [](const APFloat &a) -> std::optional<APFloat> {
                  if (a.getSizeInBits(a.getSemantics()) == 64)
                    return APFloat(tanh(a.convertToDouble()));

                  if (a.getSizeInBits(a.getSemantics()) == 32)
                    return APFloat(tanhf(a.convertToFloat()));

                  bool losesInfo = false;
                  APFloat fres(tanh(a.convertToDouble()));
                  fres.convert(a.getSemantics(),
                               llvm::RoundingMode::NearestTiesToEven,
                               &losesInfo);
                  return fres;
                })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

struct ExpSimplify : public OpRewritePattern<mlir::stablehlo::ExpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ExpOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res =
            constFoldUnaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants, [](const APFloat &a) -> std::optional<APFloat> {
                  if (a.getSizeInBits(a.getSemantics()) == 64)
                    return APFloat(exp(a.convertToDouble()));

                  if (a.getSizeInBits(a.getSemantics()) == 32)
                    return APFloat(expf(a.convertToFloat()));
                  return {};
                })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
    }

    return failure();
  }
};

template <typename T> struct BinBroadcastSplat final : OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
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

struct AllFinite : public OpRewritePattern<mlir::stablehlo::IsFiniteOp> {
  using OpRewritePattern<mlir::stablehlo::IsFiniteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IsFiniteOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, op.getType(), makeAttr(op.getType(), 1).cast<ElementsAttr>());
    return success();
  }
};

struct NoNan : public OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern<mlir::stablehlo::CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getLhs() == op.getRhs()) {
      if (op.getComparisonDirection() ==
          mlir::stablehlo::ComparisonDirection::EQ) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), makeAttr(op.getType(), 1).cast<ElementsAttr>());
        return success();
      }
      if (op.getComparisonDirection() ==
          mlir::stablehlo::ComparisonDirection::NE) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(), makeAttr(op.getType(), 0).cast<ElementsAttr>());
        return success();
      }
    }
    return failure();
  }
};

struct TransposeTranspose
    : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const final {
    auto operand = op.getOperand();

    auto convertOp = operand.getDefiningOp<mlir::stablehlo::ConvertOp>();
    if (convertOp) {
      operand = convertOp.getOperand();
    }

    auto definingTranspose =
        operand.getDefiningOp<mlir::stablehlo::TransposeOp>();
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

struct TransposeConvert : public OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern<mlir::stablehlo::ConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = op.getResult().getType().cast<TensorType>();
    auto operandType = op.getOperand().getType().cast<TensorType>();
    if (!resultType.hasStaticShape() || !operandType.hasStaticShape())
      return failure();
    if (resultType.getNumElements() *
            getBitWidth(resultType.getElementType()) >=
        operandType.getNumElements() *
            getBitWidth(operandType.getElementType()))
      return failure();

    auto transpose =
        op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>();
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
    : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const final {

    auto operand = op.getOperand();
    auto convert = operand.getDefiningOp<mlir::stablehlo::ConvertOp>();
    if (convert) {
      operand = convert.getOperand();
      if (!llvm::hasSingleElement(convert->getUsers()))
        return failure();
    }

    auto dot = operand.getDefiningOp<mlir::stablehlo::DotGeneralOp>();
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

struct TransposeReduce : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto operand = transpose.getOperand();
    auto reduce = operand.getDefiningOp<mlir::stablehlo::ReduceOp>();
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
    : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto operand = transpose.getOperand();
    auto conv = operand.getDefiningOp<mlir::stablehlo::ConvolutionOp>();
    if (!conv || !llvm::hasSingleElement(operand.getUsers()))
      return failure();

    auto permutation = transpose.getPermutation();

    auto dimensionNumbers = conv.getDimensionNumbers();
    int64_t outputBatchDimension =
        permutation[dimensionNumbers.getOutputBatchDimension()];
    int64_t outputFeatureDimension =
        permutation[dimensionNumbers.getOutputFeatureDimension()];
    SmallVector<int64_t> outputSpatialDimensions(
        dimensionNumbers.getOutputSpatialDimensions().begin(),
        dimensionNumbers.getOutputSpatialDimensions().end());

    for (auto &dim : outputSpatialDimensions) {
      dim = permutation[dim];
    }

    auto newDimensionNumbers = stablehlo::ConvDimensionNumbersAttr::get(
        dimensionNumbers.getContext(),
        dimensionNumbers.getInputBatchDimension(),
        dimensionNumbers.getInputFeatureDimension(),
        dimensionNumbers.getInputSpatialDimensions(),
        dimensionNumbers.getKernelInputFeatureDimension(),
        dimensionNumbers.getKernelOutputFeatureDimension(),
        dimensionNumbers.getKernelSpatialDimensions(), outputBatchDimension,
        outputFeatureDimension, outputSpatialDimensions);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConvolutionOp>(
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
    : public OpRewritePattern<mlir::stablehlo::ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvolutionOp conv,
                                PatternRewriter &rewriter) const final {
    auto lhs_trans =
        conv.getLhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    auto rhs_trans =
        conv.getRhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
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
struct TransposeEinsum : public OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp transpose,
                                PatternRewriter &rewriter) const final {
    auto operand = transpose.getOperand();
    auto einsum = operand.getDefiningOp<mlir::stablehlo::EinsumOp>();
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
struct EinsumTranspose : public OpRewritePattern<mlir::stablehlo::EinsumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::EinsumOp einsum,
                                PatternRewriter &rewriter) const final {
    llvm::StringRef einsumConfig = einsum.getEinsumConfig();

    auto lhs_trans =
        einsum.getLhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    auto rhs_trans =
        einsum.getRhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::EinsumOp>(
        einsum, einsum.getType(),
        lhs_trans ? lhs_trans.getOperand() : einsum.getLhs(),
        rhs_trans ? rhs_trans.getOperand() : einsum.getRhs(),
        StringAttr::get(einsum.getContext(), newEinsumConfig));
    return success();
  }
};

struct DotTranspose : public OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp dot,
                                PatternRewriter &rewriter) const final {

    auto dim = dot.getDotDimensionNumbers();
    size_t numLHSResults = dot.getLhs().getType().getShape().size() -
                           dim.getLhsBatchingDimensions().size() -
                           dim.getLhsContractingDimensions().size();

    size_t numRHSResults = dot.getRhs().getType().getShape().size() -
                           dim.getRhsBatchingDimensions().size() -
                           dim.getRhsContractingDimensions().size();

    auto lhs_trans = dot.getLhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (lhs_trans && numLHSResults != 1)
      return failure();
    auto rhs_trans = dot.getRhs().getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (rhs_trans && numRHSResults != 1)
      return failure();
    if (!lhs_trans && !rhs_trans)
      return failure();

    SmallVector<int64_t> lhsBatch(dim.getLhsBatchingDimensions().begin(),
                                  dim.getLhsBatchingDimensions().end());
    SmallVector<int64_t> rhsBatch(dim.getRhsBatchingDimensions().begin(),
                                  dim.getRhsBatchingDimensions().end());

    SmallVector<int64_t> lhsContract(dim.getLhsContractingDimensions().begin(),
                                     dim.getLhsContractingDimensions().end());
    SmallVector<int64_t> rhsContract(dim.getRhsContractingDimensions().begin(),
                                     dim.getRhsContractingDimensions().end());

    if (lhs_trans) {
      for (auto &dim : lhsBatch)
        dim = lhs_trans.getPermutation()[dim];
      for (auto &dim : lhsContract)
        dim = lhs_trans.getPermutation()[dim];
    }

    if (rhs_trans) {
      for (auto &dim : rhsBatch)
        dim = rhs_trans.getPermutation()[dim];
      for (auto &dim : rhsContract)
        dim = rhs_trans.getPermutation()[dim];
    }

    auto ndim = stablehlo::DotDimensionNumbersAttr::get(
        dim.getContext(), lhsBatch, rhsBatch, lhsContract, rhsContract);

    rewriter.replaceOpWithNewOp<stablehlo::DotGeneralOp>(
        dot, dot.getType(), lhs_trans ? lhs_trans.getOperand() : dot.getLhs(),
        rhs_trans ? rhs_trans.getOperand() : dot.getRhs(), ndim,
        dot.getPrecisionConfigAttr(), dot.getAlgorithmAttr());
    return success();
  }
};

struct BroadcastReduce : public OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern<mlir::stablehlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-operand single-init reduce is supported");
    }
    // TODO: min/max can also be an option since they are dropped
    if (!isa<stablehlo::AddOp>(op.getRegion().getBlocks().front().front())) {
      return rewriter.notifyMatchFailure(op, "only add is currently supported");
    }

    Value input = op.getInputs()[0];
    auto inputType = input.getType().cast<TensorType>();
    auto broadcast = input.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!broadcast) {
      return rewriter.notifyMatchFailure(op,
                                         "input source is not a broadcast op");
    }

    // If any of the dimensions that are being reduced was initially
    // broadcasted, we can multiply the result with the dimension instead.
    ArrayRef<int64_t> broadcastDims = broadcast.getBroadcastDimensions();
    SmallVector<int64_t> broadcastFromNothingDims, broadcastFromOneDims;
    auto broadcastSourceType =
        broadcast.getOperand().getType().cast<TensorType>();
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

    auto newReduction = rewriter.create<stablehlo::ReduceOp>(
        op.getLoc(), op->getResultTypes(), ValueRange{broadcast.getOperand()},
        op.getInitValues(), newReduceDimensions);
    newReduction.getRegion().takeBody(op.getRegion());

    auto newResultType = newReduction.getResult(0).getType().cast<TensorType>();
    auto constantInt = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(),
        makeAttr(newResultType.clone(rewriter.getI64Type()), size));
    auto converted = rewriter.create<stablehlo::ConvertOp>(
        op.getLoc(), constantInt, newResultType.getElementType());
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

template <typename T> struct BinopConstPad : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
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
          inp.resizeSplat(pval.getType().template cast<ShapedType>()));
      auto pval2 = rewriter.create<T>(op.getLoc(), (i == 0) ? pval_cst : pval,
                                      (i == 0) ? pval : pval_cst);

      auto val = pad.getOperand();
      auto val_cst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), val.getType(),
          inp.resizeSplat(val.getType().template cast<ShapedType>()));
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

template <typename T> struct BinopPadPad : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
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

struct AddPadPadToConcat : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern<stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {
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

template <typename T> struct BinopBinopPadPad : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
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

        auto p1val =
            pad1 ? pad1.getPaddingValue()
                 : rewriter.create<stablehlo::ConstantOp>(
                       op.getLoc(), pad2.getPaddingValue().getType(),
                       inp1.resizeSplat(pad2.getPaddingValue()
                                            .getType()
                                            .template cast<ShapedType>()));
        auto p2val =
            pad2 ? pad2.getPaddingValue()
                 : rewriter.create<stablehlo::ConstantOp>(
                       op.getLoc(), pad1.getPaddingValue().getType(),
                       inp2.resizeSplat(pad1.getPaddingValue()
                                            .getType()
                                            .template cast<ShapedType>()));

        auto pval = rewriter.create<T>(op.getLoc(), p1val, p2val);

        auto o1val =
            pad1 ? pad1.getOperand()
                 : rewriter.create<stablehlo::ConstantOp>(
                       op.getLoc(), pad2.getOperand().getType(),
                       inp1.resizeSplat(pad2.getOperand()
                                            .getType()
                                            .template cast<ShapedType>()));
        auto o2val =
            pad2 ? pad2.getOperand()
                 : rewriter.create<stablehlo::ConstantOp>(
                       op.getLoc(), pad1.getOperand().getType(),
                       inp2.resizeSplat(pad1.getOperand()
                                            .getType()
                                            .template cast<ShapedType>()));

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

template <typename T> struct BinopBinopPadConst : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
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

struct MulZeroPad : public OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const final {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    auto otherArgType = otherArg.getType().cast<TensorType>();
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

struct DivZeroPad : public OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op,
                                PatternRewriter &rewriter) const final {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
      return failure();
    if (anyPadSizesNegative(pad))
      return failure();

    if (otherIsLHS)
      return failure();

    auto otherArgType = otherArg.getType().cast<TensorType>();
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

struct PadPad : public OpRewritePattern<mlir::stablehlo::PadOp> {
  using OpRewritePattern<mlir::stablehlo::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::PadOp op,
                                PatternRewriter &rewriter) const final {
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

struct SliceDotGeneral : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
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

struct PadDotGeneral : public OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern<mlir::stablehlo::DotGeneralOp>::OpRewritePattern;

  bool postPad;
  PadDotGeneral(size_t postPad, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames), postPad(postPad) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const final {
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
        auto padSize = pad.getOperand()
                           .getType()
                           .cast<RankedTensorType>()
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
              otherArg.getType().cast<RankedTensorType>().getShape()[dim]);
        }
      }

      for (auto padDim : padResultDimensions) {
        int64_t low = pad.getEdgePaddingLow()[padDim];
        int64_t high = pad.getEdgePaddingHigh()[padDim];
        int64_t interior = pad.getInteriorPadding()[padDim];
        auto padSize = pad.getOperand()
                           .getType()
                           .cast<RankedTensorType>()
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
              otherArg.getType().cast<RankedTensorType>().getShape()[dim]);
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
           llvm::enumerate(otherArg.getType().cast<TensorType>().getShape())) {
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

  auto reshapeOperandType = reshape.getOperand().getType().cast<TensorType>();
  auto reshapeType = reshape.getType().cast<TensorType>();
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

struct SliceReshape : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
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
struct SliceReshapePad final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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

struct SliceIf : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto ifop = op.getOperand().getDefiningOp<stablehlo::IfOp>();
    if (!ifop)
      return failure();
    if (!llvm::hasSingleElement(op.getOperand().getUsers()))
      return failure();

    ssize_t opIdx = -1;
    for (OpOperand &use : op.getOperand().getUses()) {
      if (opIdx != -1) {
        llvm_unreachable("multi use is not possible");
      } else {
        opIdx = use.getOperandNumber();
      }
    }
    if (opIdx == -1)
      llvm_unreachable("zero use is not possible");

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

struct SliceReshapeConcat : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  bool onlySingleUser;

  TransposeElementwise(bool onlySingleUser, MLIRContext *context,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        onlySingleUser(onlySingleUser) {}

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto elem = op.getOperand().getDefiningOp();
    if (!elem)
      return failure();

    if (onlySingleUser && !llvm::hasSingleElement(elem->getUsers()))
      return failure();

    if (!elem->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();

    SmallVector<Value> ops;
    for (auto v : elem->getOperands()) {
      ops.push_back(rewriter.create<stablehlo::TransposeOp>(
          op.getLoc(), v, op.getPermutation()));
    }
    auto newOp = rewriter.create(
        elem->getLoc(), elem->getName().getIdentifier(), ValueRange(ops),
        TypeRange(op.getType()), elem->getAttrs(), {}, {});
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TransposeConcat final : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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
    return success();
  }
};

struct TransposeIota final : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto iota = op.getOperand().getDefiningOp<stablehlo::IotaOp>();
    if (!iota)
      return failure();

    if (!llvm::hasSingleElement(iota->getUsers()))
      return failure();

    auto dim = iota.getIotaDimension();
    auto dim2 = getInversePermutation(op.getPermutation())[dim];

    rewriter.replaceOpWithNewOp<stablehlo::IotaOp>(op, op.getType(), dim2);
    return success();
  }
};

struct TransposeReduceWindow final
    : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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
    : public OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

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

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the operand of the reshape is a concatenate operation
    auto concatOp =
        reshapeOp.getOperand().getDefiningOp<mlir::stablehlo::ConcatenateOp>();
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
      auto operandType = operand.getType().cast<RankedTensorType>();
      if (!operandType)
        return failure();

      SmallVector<int64_t> shape(operandType.getShape().begin(),
                                 operandType.getShape().end());

      int64_t one = 1;
      if (!transformReshapeSlice<int64_t>(reshapeOp, shape, /*toFill*/ 1, &one))
        return failure();

      auto newReshapeType =
          RankedTensorType::get(shape, operandType.getElementType());
      auto newReshapeOp = rewriter.create<mlir::stablehlo::ReshapeOp>(
          reshapeOp.getLoc(), newReshapeType, operand);
      concatOperands.push_back(newReshapeOp);
    }

    // Create a new concat operation with the reshaped operands
    auto origReshapeOperand = reshapeOp.getOperand().getType().getShape();
    auto origReshapeResult = reshapeOp.getResult().getType().getShape();
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConcatenateOp>(
        reshapeOp, concatOperands, newDim);

    return success();
  }
};

// reshape(reduce_window(...)) -> reduce_window(reshape(...))
struct ReshapeReduceWindow final
    : public OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the operand of the reshape is a reduce_window operation
    auto reduceWindow =
        reshapeOp.getOperand().getDefiningOp<mlir::stablehlo::ReduceWindowOp>();
    if (!reduceWindow)
      return failure();

    // Check if there is any non-reshape user of this reduce_window operation
    if (llvm::any_of(reduceWindow->getUsers(), [&](Operation *user) {
          return !isa<mlir::stablehlo::ReshapeOp>(user);
        }))
      return failure();

    auto reduceWindowType = reduceWindow.getType(0).dyn_cast<ShapedType>();
    if (!reduceWindowType || !reduceWindowType.hasStaticShape())
      return failure();
    ArrayRef<int64_t> reduceWindowShape = reduceWindowType.getShape();

    auto reshapeType = reshapeOp.getType().dyn_cast<ShapedType>();
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

    auto newReshapeOp = rewriter.create<mlir::stablehlo::ReshapeOp>(
        reshapeOp.getLoc(), reshapeType, reduceWindow.getInputs()[0]);
    auto newReduceWindowOp =
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ReduceWindowOp>(
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

struct ReshapeElementwise final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto elem = op.getOperand().getDefiningOp();
    if (!elem)
      return failure();

    if (!llvm::hasSingleElement(elem->getUsers()))
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
    : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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

struct SliceReshapeDotGeneral : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
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

struct SliceReshapeSlice final : OpRewritePattern<mlir::stablehlo::SliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
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
// https://github.com/openxla/stablehlo/blob/4f180d3c2236a15f82f29aad1b47f6ea2c14fc52/stablehlo/reference/Ops.cpp#L1381
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
                            inferredTypes[0].cast<mlir::ShapedType>());
}

bool is_iota(ArrayRef<int64_t> idx) {
  for (auto en : llvm::enumerate(idx))
    if (en.index() != en.value())
      return false;
  return true;
}

/// Converts gather ops to slice ops in case we have a single set of constant
/// indices.
struct GatherSimplify final : OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr startIndicesCst;
    if (!matchPattern(op.getStartIndices(), m_Constant(&startIndicesCst)))
      return failure();

    {
      DenseIntElementsAttr operandVals;
      if (matchPattern(op.getOperand(), m_Constant(&operandVals))) {
        auto out = stablehlo::gatherOp(
            stablehlo::constantOp(operandVals),
            stablehlo::constantOp(startIndicesCst),
            stablehlo::Axes(op.getDimensionNumbers().getOffsetDims()),
            stablehlo::Axes(op.getDimensionNumbers().getCollapsedSliceDims()),
            stablehlo::Axes(op.getDimensionNumbers().getOperandBatchingDims()),
            stablehlo::Axes(
                op.getDimensionNumbers().getStartIndicesBatchingDims()),
            stablehlo::Axes(op.getDimensionNumbers().getStartIndexMap()),
            stablehlo::Axis(op.getDimensionNumbers().getIndexVectorDim()),
            stablehlo::Sizes(op.getSliceSizes()), op.getIndicesAreSorted(),
            op.getType());

        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                           fromTensor(out));
        return success();
      }
    }
    return failure();
  }
};

template <typename T> struct CSE final : OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() > 0)
      for (auto nop : op->getOperand(0).getUsers()) {
        if (nop == op)
          continue;
        if (!isa<T>(nop))
          continue;
        if (!OperationEquivalence::isEquivalentTo(
                op, nop, OperationEquivalence::IgnoreLocations))
          continue;
        if (nop->getBlock() != op->getBlock())
          continue;
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
    : OpRewritePattern<mlir::stablehlo::OptimizationBarrierOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::OptimizationBarrierOp op,
                                PatternRewriter &rewriter) const override {

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

struct DUSToI32 final : OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicUpdateSliceOp dusOp,
                                PatternRewriter &rewriter) const override {
    auto i32 = rewriter.getI32Type();

    auto unrankedI32 = RankedTensorType::get({}, i32);

    auto startIndices = dusOp.getStartIndices();

    SmallVector<int64_t> newStartIndicesConst;
    for (auto idx : startIndices) {
      if (idx.getType().cast<RankedTensorType>().getElementType() == i32)
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
          makeAttr(unrankedI32, val).cast<ElementsAttr>()));
    }

    rewriter.replaceOpWithNewOp<stablehlo::DynamicUpdateSliceOp>(
        dusOp, dusOp.getOperand(), dusOp.getUpdate(), newStartIndices);

    return success();
  }
};

// Replaces DUS with a combination of slices and concats.
// Each run of the pattern handles one dimension at a time.
struct DUSToConcat final : OpRewritePattern<stablehlo::DynamicUpdateSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicUpdateSliceOp dusOp,
                                PatternRewriter &rewriter) const override {

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
        loc, itype, makeAttr(itype, 0).cast<ElementsAttr>());
    newDusIndices[differingDim] = c0;

    if (!dusFromBeginning) {
      int64_t sliceAt = dusStartIndices[differingDim];
      getPrePost(sliceAt);

      auto c0 = rewriter.create<stablehlo::ConstantOp>(
          loc, itype, makeAttr(itype, 0).cast<ElementsAttr>());
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
    return res.cast<TypedAttr>();

  return nullptr;
}

static mlir::stablehlo::ComparisonDirection
invertDirection(mlir::stablehlo::ComparisonDirection direction) {
  using mlir::stablehlo::ComparisonDirection;

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

static APInt calculateComp(mlir::stablehlo::ComparisonType kind,
                           mlir::stablehlo::ComparisonDirection direction,
                           const APInt &lhs, const APInt &rhs) {
  using mlir::stablehlo::ComparisonDirection;
  using mlir::stablehlo::ComparisonType;
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

struct CompareOpCanon final : OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
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
        auto out = fromTensor(mlir::stablehlo::compareOp(
            isSplat
                ? stablehlo::makeTensor(lhs.resizeSplat(RankedTensorType::get(
                      {}, lhs.getType().getElementType())))
                : mlir::stablehlo::constantOp(lhs),
            isSplat
                ? stablehlo::makeTensor(rhs.resizeSplat(RankedTensorType::get(
                      {}, rhs.getType().getElementType())))
                : mlir::stablehlo::constantOp(rhs),
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
    using mlir::stablehlo::ComparisonType;
    std::optional<ComparisonType> compType = op.getCompareType();
    if (!compType ||
        !llvm::is_contained({ComparisonType::SIGNED, ComparisonType::UNSIGNED},
                            *compType)) {
      return failure();
    }

    using mlir::stablehlo::ComparisonDirection;
    ComparisonDirection direction = op.getComparisonDirection();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (lhs == rhs) {
      switch (direction) {
      case ComparisonDirection::EQ:
      case ComparisonDirection::GE:
      case ComparisonDirection::LE: {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
            op, SplatElementsAttr::get(type, rewriter.getBoolAttr(true)));
        return success();
      }
      case ComparisonDirection::GT:
      case ComparisonDirection::LT:
      case ComparisonDirection::NE: {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
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
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, res);
      return success();
    }

    return failure();
  }
};

struct CompareExt final : OpRewritePattern<mlir::stablehlo::CompareOp> {
  using OpRewritePattern<mlir::stablehlo::CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getLhs()
             .getType()
             .cast<RankedTensorType>()
             .getElementType()
             .isInteger())
      return failure();

    auto direction = op.getComparisonDirection();

    auto lhsConvert = op.getLhs().getDefiningOp<mlir::stablehlo::ConvertOp>();
    auto rhsConvert = op.getRhs().getDefiningOp<mlir::stablehlo::ConvertOp>();
    if (!lhsConvert && !rhsConvert)
      return failure();

    auto isConvertFromBool = [](mlir::stablehlo::ConvertOp cvtOp) -> bool {
      return cvtOp && cvtOp.getOperand()
                          .getType()
                          .cast<RankedTensorType>()
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
        rewriter.replaceAllUsesWith(op.getResult(), lhsConvert.getOperand());
        return success();
      } else if (matchPattern(op.getRhs(), m_Zero())) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::NotOp>(
            op, lhsConvert.getOperand());
        return success();
      }
    }

    if (isConvertFromBool(rhsConvert) &&
        direction == stablehlo::ComparisonDirection::EQ) {
      if (matchPattern(op.getLhs(), m_One())) {
        rewriter.replaceAllUsesWith(op.getResult(), rhsConvert.getOperand());
        return success();
      } else if (matchPattern(op.getLhs(), m_Zero())) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::NotOp>(
            op, rhsConvert.getOperand());
        return success();
      }
    }

    return failure();
  }
};

struct SelectCompIotaConstSimplify final
    : OpRewritePattern<mlir::stablehlo::SelectOp> {
  struct slice_data {
    Value tensor;
    int64_t count;
  };
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::stablehlo::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
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
    auto compareOp = dyn_cast<stablehlo::CompareOp>(compare.getDefiningOp());
    auto flag = compareOp.getComparisonDirection();

    stablehlo::IotaOp iota;
    if (!matchPattern(
            compare, m_Op<stablehlo::CompareOp>(m_Op<mlir::stablehlo::IotaOp>(),
                                                m_Constant(&inp)))) {

      if (matchPattern(compare, m_Op<stablehlo::CompareOp>(
                                    m_Constant(&inp),
                                    m_Op<mlir::stablehlo::IotaOp>()))) {
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
        iota = cast<stablehlo::IotaOp>(compareOp.getRhs().getDefiningOp());
      } else
        return failure();
    } else
      iota = cast<stablehlo::IotaOp>(compareOp.getLhs().getDefiningOp());

    assert(iota);

    const auto iotaDim = iota.getIotaDimension();

    if (!inp.isSplat())
      return failure();

    auto constValue = inp.getSplatValue<IntegerAttr>().getInt();
    auto endValue = shapeLimit[iotaDim];

    rewriter.setInsertionPointAfterValue(compare);

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
      const auto elemType = tensorType.getElementType();
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

struct SelectOpUsedWithinIf final
    : OpRewritePattern<mlir::stablehlo::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
    Value pred = op.getPred();
    Value result = op.getResult();

    if (pred.getType().cast<TensorType>().getShape().size() != 0)
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

struct SelectOpCanon final : OpRewritePattern<mlir::stablehlo::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  size_t max_constant_expansion;
  SelectOpCanon(size_t max_constant_expansion, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}
  LogicalResult matchAndRewrite(mlir::stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(type, newValues));
    return success();
  }
};

struct BroadcastInDimOpCanon final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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
      rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
          op, SplatElementsAttr::get(op.getType(),
                                     cstAttr.getSplatValue<Attribute>()));
      return success();
    }

    if (operandTy.hasStaticShape() && type.hasStaticShape() &&
        type.getNumElements() == operandTy.getNumElements()) {
      // BroadcastInDim equivalent to reshape.
      if (isDimsIota) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                                operand);
        return success();
      }
      // BroadcastInDim equivalent to transpose.
      if (type.getRank() == operandTy.getRank()) {
        rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
            op, type, operand, getInversePermutation(dims));
        return success();
      }
    }

    // Eliminate redundant nested BroadcastInDim.
    if (auto definingOp =
            operand.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
      auto newIndices = llvm::to_vector(
          llvm::map_range(definingOp.getBroadcastDimensions(),
                          [&dims](int64_t dim) { return dims[dim]; }));
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          op, type, definingOp.getOperand(), newIndices);
      return success();
    }

    return failure();
  }
};

struct TransposeBroadcastInDimToBroadcastInDim final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern<mlir::stablehlo::BroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern<mlir::stablehlo::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcastOp =
        op.getOperand().getDefiningOp<stablehlo::BroadcastInDimOp>();
    if (!broadcastOp)
      return failure();

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

    return success();
  }
};

struct ConcatenateOpCanon final
    : OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;
  size_t max_constant_expansion;
  ConcatenateOpCanon(size_t max_constant_expansion, MLIRContext *context,
                     PatternBenefit benefit = 1,
                     ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        max_constant_expansion(max_constant_expansion) {}
  LogicalResult matchAndRewrite(mlir::stablehlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
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
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), newElems));
    return success();
  }
};

struct ConvertOpCanon final : OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    // Check if this convert is a noop.
    if (op.getOperand().getType() != op.getType())
      return failure();

    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

struct DivideSqrtToMultiplyRsqrt final
    : OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType operandType = op.getOperand().getType();
    if (!operandType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "requires operand static shape");

    RankedTensorType type = op.getType();
    // output has static shape, replace with broadcast_in_dim
    if (type.hasStaticShape()) {
      rewriter.replaceOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          op, type, op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }

    // output_dimensions are constant, set output shape with output_dimensions,
    // then replace with broadcast_in_dim
    if (llvm::SmallVector<int64_t> shape;
        succeeded(hlo::matchInts(op.getOutputDimensions(), shape))) {
      refineOpWithNewOp<mlir::stablehlo::BroadcastInDimOp>(
          rewriter, op, RankedTensorType::get(shape, type.getElementType()),
          op.getOperand(), op.getBroadcastDimensionsAttr());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "requires output static shape or constant broadcast dimensions");
  }
};

struct ChainedDynamicBroadcastInDimCanonicalization final
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp bcast,
                                PatternRewriter &rewriter) const override {
    auto precedingBcast =
        bcast.getOperand()
            .getDefiningOp<mlir::stablehlo::DynamicBroadcastInDimOp>();
    if (!precedingBcast)
      return failure();

    // Compose broadcast dimensions.
    SmallVector<int64_t> composition;
    for (int64_t precedingDim : precedingBcast.getBroadcastDimensions()) {
      composition.push_back(bcast.getBroadcastDimensions()[precedingDim]);
    }
    auto composedBcastDims = rewriter.getDenseI64ArrayAttr(composition);

    rewriter.replaceOpWithNewOp<mlir::stablehlo::DynamicBroadcastInDimOp>(
        bcast, bcast.getType(), precedingBcast.getOperand(),
        bcast.getOutputDimensions(), composedBcastDims);
    return success();
  }
};

// If all dimensions are known to be nonexpanding from the attribute, replace
// the dynamic broadcast with a cast.
struct DynamicBroadcastInDimAllDimsNonExpanding final
    : OpRewritePattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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

struct NoopReduceOpCanon final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // No dimensions to reduce.
    if (op.getDimensions().empty()) {
      rewriter.replaceOp(op, op.getInputs());
      return success();
    }

    // If all returned values in the ReduceOp region exists outside the
    // region, replace the ReduceOp with those values.
    if (auto retOp = dyn_cast<mlir::stablehlo::ReturnOp>(
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

struct EmptyReduceOpCanon final : OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    // We require all reduce shapes to be the same, up to the element types, so
    // we can just use the first operand and the first result as
    // representatives.
    auto elemTy = op.getInputs().getType().front().cast<RankedTensorType>();

    if (!llvm::is_contained(elemTy.getShape(), 0))
      return failure();

    Location loc = op.getLoc();
    DenseI64ArrayAttr empty = rewriter.getDenseI64ArrayAttr({});
    if (elemTy.hasStaticShape()) {
      SmallVector<Value> broadcasts(op.getNumResults());
      for (auto [bcast, init, outTy] : llvm::zip_equal(
               broadcasts, op.getInitValues(), op.getResultTypes())) {
        bcast = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(loc, outTy,
                                                                   init, empty);
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
      bcast = rewriter.create<mlir::stablehlo::DynamicBroadcastInDimOp>(
          loc, outTy, init, shape, empty);
    }
    rewriter.replaceOp(op, broadcasts);
    return success();
  }
};

struct DynamicReshapeOpCanon final
    : OpRewritePattern<mlir::stablehlo::DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // This is a noop when the output type is already a static shape.
    RankedTensorType type = op.getType();
    if (!type.hasStaticShape())
      return failure();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, type,
                                                            op.getOperand());
    return success();
  }
};

struct GetTupleElementOpCanon final
    : OpRewritePattern<mlir::stablehlo::GetTupleElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetTupleElementOp op,
                                PatternRewriter &rewriter) const override {
    auto tuple = op.getOperand().getDefiningOp<mlir::stablehlo::TupleOp>();
    if (!tuple)
      return failure();

    Value result = tuple.getOperand(op.getIndex());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RealOpCanon final : OpRewritePattern<mlir::stablehlo::RealOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RealOp op,
                                PatternRewriter &rewriter) const override {
    auto elTy = op.getOperand().getType().getElementType();
    if (!isa<ComplexType>(elTy)) {
      rewriter.replaceAllUsesWith(op.getResult(), op.getOperand());
      return success();
    }

    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex)
      return failure();

    rewriter.replaceOp(op, complex.getLhs());
    return success();
  }
};

struct ImagOpCanon final : OpRewritePattern<mlir::stablehlo::ImagOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ImagOp op,
                                PatternRewriter &rewriter) const override {
    auto elTy = op.getOperand().getType().getElementType();
    if (!isa<ComplexType>(elTy)) {
      rewriter.replaceOp(op, rewriter.create<stablehlo::ConstantOp>(
                                 op->getLoc(), makeAttr(op.getType(), 0)));
      return success();
    }

    auto complex = op.getOperand().getDefiningOp<mlir::stablehlo::ComplexOp>();
    if (!complex)
      return failure();

    rewriter.replaceOp(op, complex.getRhs());
    return success();
  }
};

// (conj (complex a, (neg b))) -> (complex a b)
struct ConjComplexNegate final : OpRewritePattern<mlir::chlo::ConjOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(chlo::ConjOp op,
                                PatternRewriter &rewriter) const override {
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
    : OpRewritePattern<mlir::stablehlo::GetDimensionSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold get_dimension_size when the queried dim is statically known.
    RankedTensorType operandTy = op.getOperand().getType();

    int64_t dimSize = operandTy.getDimSize(op.getDimension());
    if (dimSize < 0)
      return failure();

    auto elemTy = op.getType().getElementType().cast<IntegerType>();
    IntegerAttr elemVal = rewriter.getIntegerAttr(elemTy, dimSize);
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, DenseElementsAttr::get(op.getType(), elemVal));
    return success();
  }
};

struct NoopReverse final : OpRewritePattern<mlir::stablehlo::ReverseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReverseOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> newDimensions;
    auto dimensions = op.getDimensions();
    auto shape = op.getResult().getType().getShape();

    for (auto dim : dimensions) {
      auto size = shape[dim];
      if (size != 1)
        newDimensions.push_back(dim);
    }

    if (newDimensions.empty()) {
      rewriter.replaceAllUsesWith(op.getResult(), op.getOperand());
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
struct GatherOpCanon final : OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp gather,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr index;
    if (!matchPattern(gather.getStartIndices(), m_Constant(&index)))
      return failure();

    mlir::stablehlo::GatherDimensionNumbersAttr dnums =
        gather.getDimensionNumbers();
    if (dnums.getIndexVectorDim() != 0 || index.getType().getRank() > 1)
      return failure();

    // TODO: Remove when the verifier catches this case that is
    // invalid if all previous condition holds.
    if (index.getNumElements() !=
        static_cast<int64_t>(dnums.getStartIndexMap().size())) {
      return failure();
    }

    auto operandType = gather->getOperand(0).getType().cast<RankedTensorType>();
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
    Value result = rewriter.create<mlir::stablehlo::SliceOp>(
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
      result = rewriter.create<mlir::stablehlo::ReshapeOp>(gather.getLoc(),
                                                           reshapeType, result);
    }

    result.setType(gather.getType());
    rewriter.replaceOp(gather, result);
    return success();
  }
};

struct ReshapeOpCanon final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold noop reshape.
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape of a constant.
    DenseElementsAttr cstAttr;
    if (!matchPattern(op.getOperand(), m_Constant(&cstAttr)))
      return failure();

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(
        op, cstAttr.reshape(op.getType()));
    return success();
  }
};

struct MergeConsecutiveReshapes final
    : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    // Fold noop reshape.
    auto operand = op.getOperand();
    if (op.getType() == operand.getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }

    // Fold reshape(reshape(x)).
    auto reshapeOp = operand.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!reshapeOp)
      return rewriter.notifyMatchFailure(
          op, "requires defining op of operand to be Reshape");

    op.setOperand(reshapeOp->getOperand(0));
    return success();
  }
};

struct TransposeIsReshape final
    : OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReshapeOp>(op, op.getType(),
                                                            input);
    return success();
  }
};

struct IfRemoveUnused final : OpRewritePattern<mlir::stablehlo::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IfOp op,
                                PatternRewriter &rewriter) const override {

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

struct IfPredPropagation final : OpRewritePattern<mlir::stablehlo::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IfOp op,
                                PatternRewriter &rewriter) const override {
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

struct IfInline final : OpRewritePattern<mlir::stablehlo::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IfOp op,
                                PatternRewriter &rewriter) const override {

    auto iszero = matchPattern(op.getPred(), m_Zero());
    auto isone = matchPattern(op.getPred(), m_One());

    if (!iszero && !isone)
      return failure();

    auto current = op->getBlock();

    auto &reg = isone ? op.getTrueBranch() : op.getFalseBranch();

    if (reg.empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    assert(reg.hasOneBlock());  // stablehlo.if only allows 1 or 0 block in the
    auto *block = &reg.front(); // regions

    auto term = block->getTerminator();
    rewriter.replaceAllOpUsesWith(op, term->getOperands());
    rewriter.eraseOp(term);

    auto newBlock = rewriter.splitBlock(current, Block::iterator(op));

    rewriter.inlineRegionBefore(reg, newBlock);

    rewriter.mergeBlocks(block, current);
    rewriter.mergeBlocks(newBlock, current);

    rewriter.eraseOp(op);

    return success();
  }
};

// https://github.com/llvm/llvm-project/blob/74d8f3952c4acf6d57948983d7c5b0d0a7763c28/mlir/lib/Dialect/SCF/IR/SCF.cpp#L2313
struct IfToSelect final : public OpRewritePattern<mlir::stablehlo::IfOp> {
  using OpRewritePattern<mlir::stablehlo::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() == 0 || op.getTrueBranch().empty() ||
        op.getFalseBranch().empty())
      return failure();

    auto pred = op.getPred();

    auto trueOperands =
        op.getTrueBranch().front().getTerminator()->getOperands();
    auto falseOperands =
        op.getFalseBranch().front().getTerminator()->getOperands();

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
        rewriter.create<mlir::stablehlo::IfOp>(op.getLoc(), nonHoistable, pred);
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
        results[it.index()] = rewriter.create<mlir::stablehlo::SelectOp>(
            op.getLoc(), pred, trueVal, falseVal);
    }

    rewriter.setInsertionPointToEnd(&replacement.getTrueBranch().front());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReturnOp>(
        replacement.getTrueBranch().front().getTerminator(), trueReturns);

    rewriter.setInsertionPointToEnd(&replacement.getFalseBranch().front());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReturnOp>(
        replacement.getFalseBranch().front().getTerminator(), falseReturns);

    rewriter.replaceOp(op, results);
    return success();
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
    : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern<stablehlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const final {
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
      if (auto blockArg = compareOp.getLhs().dyn_cast<BlockArgument>()) {
        if (blockArg.getOwner() == &condBlock) {
          counterIdx = blockArg.getArgNumber();
          limitValue = compareOp.getRhs();
          return true;
        }
      }

      if (auto blockArg = compareOp.getRhs().dyn_cast<BlockArgument>()) {
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

struct TransposeWhile : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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
        whileOp.getLoc(), newResultTypes, newOperands);

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
        for (auto &pair : outerTransposes) {
          if (pair.idx == i) {
            newArg = rewriter.create<stablehlo::TransposeOp>(
                pair.outerTranspose.getLoc(), newArg,
                getInversePermutation(pair.outerTranspose.getPermutation()));
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
        for (auto &pair : outerTransposes) {
          if (pair.idx == i) {
            newArg = rewriter.create<stablehlo::TransposeOp>(
                pair.outerTranspose.getLoc(), newArg,
                getInversePermutation(pair.outerTranspose.getPermutation()));
            break;
          }
        }
        condMapper.map(oldArg, newArg);
      }

      for (auto &op : oldCondBlock.getOperations()) {
        // if (isa<stablehlo::ReturnOp>(op))
        //   continue;
        rewriter.clone(op, condMapper);
      }
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

struct WhileDUS : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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

      auto DUS = yieldOp.getOperand(idx)
                     .getDefiningOp<stablehlo::DynamicUpdateSliceOp>();

      // Check that the while result has exactly one use
      if (!DUS)
        continue;

      if (!DUS->hasOneUse()) {
      }

      mlir::Value conditionalOperand = nullptr;
      if (DUS.getOperand() == whileOp.getBody().front().getArgument(idx)) {
      } else if (definedOutside(DUS.getOperand(), whileOp)) {

        bool hasArgUse = !whileOp.getCond().getArgument(idx).use_empty() ||
                         !whileOp.getBody().getArgument(idx).use_empty();

        if (hasArgUse) {
          continue;
        }

        conditionalOperand = DUS.getOperand();
        hasConditional = true;
      } else {
        continue;
      }

      bool legal = true;
      for (auto idx : DUS.getStartIndices()) {
        if (!definedOutside(idx, whileOp)) {
          legal = false;
        }
      }

      if (!legal)
        continue;

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
        whileOp.getLoc(), newResultTypes, newOperands);

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
        unsigned idx = candidate.idx;
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
    if (rhs == arg && compareOp.getComparisonDirection() ==
                          stablehlo::ComparisonDirection::GT ||
        compareOp.getComparisonDirection() ==
            stablehlo::ComparisonDirection::NE) {
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

  // One operand should be the IV, the other the step
  Value addLhs = addOp.getLhs();
  Value addRhs = addOp.getRhs();

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
  auto middleTensorType = middleTensor.getType().dyn_cast<RankedTensorType>();
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
    : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

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
    Region &thenRegion = ifOp.getTrueBranch();
    Region &elseRegion = ifOp.getFalseBranch();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *thenBlock = rewriter.createBlock(&thenRegion);
      rewriter.setInsertionPointToStart(thenBlock);
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), oldReturns);
    }

    {
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

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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
        whileOp.getLoc(), newResultTypes, newOperands);

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
    : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

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
    Region &thenRegion = ifOp.getTrueBranch();
    Region &elseRegion = ifOp.getFalseBranch();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *thenBlock = rewriter.createBlock(&thenRegion);
      rewriter.setInsertionPointToStart(thenBlock);
      rewriter.create<stablehlo::ReturnOp>(whileOp.getLoc(), oldReturns);
    }

    {
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

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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
      auto tensorType = candidate.operand.getType().cast<TensorType>();
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
        whileOp.getLoc(), newResultTypes, newOperands);

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

struct WhileInductionReduction : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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
          candidate.argOperand.getLoc(), candidate.outerOperand, lower, upper,
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
        whileOp.getLoc(), newResultTypes, newOperands);

    SmallVector<Value> results;
    for (auto res : newWhileOp.getResults())
      results.push_back(res);

    {
      auto ctype = RankedTensorType::get({}, rewriter.getI64Type());
      for (auto &candidate : candidates) {
        unsigned idx = candidate.idx;
        Value operand = candidate.outerOperand;
        SmallVector<Value> starts;
        for (auto idx : candidate.lowerUpdateBounds) {
          starts.push_back(rewriter.create<stablehlo::ConstantOp>(
              candidate.argOperand.getLoc(), ctype,
              makeAttr(ctype, idx).cast<ElementsAttr>()));
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
                  makeAttr(itype,
                           pair.lowerUpdateBounds[i] - pair.lowerBounds[i])
                      .cast<ElementsAttr>()));
            }

            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.argOperand.getLoc(), pair.outerSlice, newArg,
                update_starts);

            auto ctype = RankedTensorType::get(
                {}, cast<RankedTensorType>(pair.argOperand.getType())
                        .getElementType());
            auto padVal = rewriter.create<stablehlo::ConstantOp>(
                pair.argOperand.getLoc(), ctype,
                makeAttr(ctype, 0).cast<ElementsAttr>());

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
                  makeAttr(itype,
                           pair.lowerUpdateBounds[i] - pair.lowerBounds[i])
                      .cast<ElementsAttr>()));
            }

            newArg = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                pair.argOperand.getLoc(), pair.outerSlice, newArg,
                update_starts);
            auto ctype = RankedTensorType::get(
                {}, cast<RankedTensorType>(pair.condOperand.getType())
                        .getElementType());
            auto padVal = rewriter.create<stablehlo::ConstantOp>(
                pair.condOperand.getLoc(), ctype,
                makeAttr(ctype, 0).cast<ElementsAttr>());

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
struct WhileConcat : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
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
    bool hasConditional = false;

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
          Candidate{idx, concat, lhsSize, rhsSize, ops[0], ops[1], ops[2]});
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
        whileOp.getLoc(), newResultTypes, newOperands);

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

// Replace while op iteration variables which are not updated with their
// upcoming value
struct WhileSimplify : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  bool hoist_all;
  WhileSimplify(bool hoist_all, MLIRContext *context,
                PatternBenefit benefit = 1,
                ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        hoist_all(hoist_all) {}

  LogicalResult matchAndRewrite(stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
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

    auto newWhile =
        rewriter.create<stablehlo::WhileOp>(op.getLoc(), newOperands);
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
struct WhileLICM : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  bool hoist_all;
  WhileLICM(bool hoist_all, MLIRContext *context, PatternBenefit benefit = 1,
            ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern(context, benefit, generatedNames),
        hoist_all(hoist_all) {}

  LogicalResult matchAndRewrite(stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
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
              op.getLoc(), useInner, inputValue, bodyRes);
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

    auto newWhile =
        rewriter.create<stablehlo::WhileOp>(op.getLoc(), newOperands);
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
    : public OpRewritePattern<stablehlo::DynamicGatherOp> {
  using OpRewritePattern<stablehlo::DynamicGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicGatherOp op,
                                PatternRewriter &rewriter) const override {
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
      sliceSizesAttr = newSliceSizesAttr.cast<DenseIntElementsAttr>();
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
  auto type = t.dyn_cast<RankedTensorType>();
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
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (!isa_and_present<mlir::stablehlo::StablehloDialect>(op->getDialect()))
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
    : OpTraitRewritePattern<OpTrait::Elementwise> {
  using OpTraitRewritePattern::OpTraitRewritePattern;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getOperands().size() != 1)
      return rewriter.notifyMatchFailure(op, "expected to be unary");

    auto definingOp = op->getOperand(0).getDefiningOp();
    if (!definingOp)
      return rewriter.notifyMatchFailure(
          op, "expected to have an op before elementise op");

    if (!isa<mlir::stablehlo::ReshapeOp>(definingOp) &&
        !isa<mlir::stablehlo::TransposeOp>(definingOp) &&
        !isa<mlir::stablehlo::BroadcastOp>(definingOp))
      return rewriter.notifyMatchFailure(
          op, "defining operation of unexpected type");

    // Only reorder if the defining op has no other uses.
    if (!llvm::hasSingleElement(definingOp->getResult(0).getUses()))
      return rewriter.notifyMatchFailure(op, "operation has more than one use");

    Value input = definingOp->getOperand(0);
    Value result = op->getResult(0);
    auto intermediateType = input.getType().cast<ShapedType>().clone(
        getElementTypeOrSelf(result.getType()));

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

// c = a + b; d = c - b => d = a
// c = a + b; d = b - c => d = -a
struct NoNanAddSubSimplify final
    : public OpRewritePattern<stablehlo::SubtractOp> {
  using OpRewritePattern<stablehlo::SubtractOp>::OpRewritePattern;

  NoNanAddSubSimplify(bool allowOnFloatingPointMath, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        allowOnFloatingPointMath(allowOnFloatingPointMath) {}

  // Apply the pattern only if the output types are integers or if the pattern
  // is allowed on floating point math.
  bool canApplyPattern(bool allowOnFloatingPointMath, Type addOutTy,
                       Type subOutTy) const {
    addOutTy = getElementTypeOrSelf(addOutTy);
    subOutTy = getElementTypeOrSelf(subOutTy);
    if (addOutTy.isInteger() && subOutTy.isInteger())
      return true;
    return allowOnFloatingPointMath;
  }

  LogicalResult matchAndRewrite(stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const final {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto subOutTy = op.getResult().getType();

    // Check if LHS is defined by an AddOp
    if (auto lhsAddOp = lhs.getDefiningOp<stablehlo::AddOp>()) {
      auto addOutTy = lhsAddOp.getResult().getType();
      if (!canApplyPattern(allowOnFloatingPointMath, addOutTy, subOutTy))
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
      if (!canApplyPattern(allowOnFloatingPointMath, addOutTy, subOutTy))
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
struct CompareSelectSimplify : public OpRewritePattern<stablehlo::SelectOp> {
  using OpRewritePattern<stablehlo::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const final {
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
struct NotSelectSimplify : public OpRewritePattern<stablehlo::SelectOp> {
  using OpRewritePattern<stablehlo::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SelectOp op,
                                PatternRewriter &rewriter) const final {
    auto notOp = op.getPred().getDefiningOp<stablehlo::NotOp>();
    if (!notOp)
      return failure();

    rewriter.replaceOpWithNewOp<stablehlo::SelectOp>(
        op, notOp.getOperand(), op.getOnFalse(), op.getOnTrue());
    return success();
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

struct CommonCompareExpressionRewrite
    : public OpRewritePattern<stablehlo::CompareOp> {
  using OpRewritePattern<stablehlo::CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompareOp op,
                                PatternRewriter &rewriter) const final {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto negDir = negatedComparisonDirection(op.getComparisonDirection());

    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto opOperand = op.getOperand(i);
      for (auto user : opOperand.getUsers()) {
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
    }

    return failure();
  }
};

struct ScatterUpdateComputationConstProp
    : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern<stablehlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
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
            blockArgInput.getType().cast<ShapedType>(), inputSplatAttr);
        auto constInputOp =
            rewriter.create<stablehlo::ConstantOp>(op.getLoc(), denseAttr);
        blockArgInput.replaceAllUsesWith(constInputOp);
      }

      if (constUpdate && !blockArgUpdate.getUses().empty()) {
        updateTransformed = true;
        auto denseAttr = DenseElementsAttr::get(
            blockArgUpdate.getType().cast<ShapedType>(), updateSplatAttr);
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

struct ScatterIndicesAreUnique : public OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern<stablehlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getUniqueIndices())
      return failure(); // already unique, no need to do anything

    auto scatterIndices = op.getScatterIndices();
    Attribute scatterIndicesAttr;
    if (matchPattern(scatterIndices, m_Constant(&scatterIndicesAttr))) {
      auto denseAttr = scatterIndicesAttr.dyn_cast<DenseIntElementsAttr>();

      auto shape = scatterIndices.getType().cast<ShapedType>().getShape();
      if (shape.empty())
        return failure();

      int64_t numTuples = 1;
      for (int64_t i = 0; i < shape.size() - 1; ++i) {
        numTuples *= shape[i];
      }
      int64_t tupleSize = shape.back();

      // Iterate over the scatter indices tensor to extract tuples
      SmallVector<SmallVector<int64_t>> indexTuples;
      auto values = denseAttr.getValues<APInt>();
      auto it = values.begin();
      for (int64_t i = 0; i < numTuples; ++i) {
        SmallVector<int64_t> indexTuple;
        for (int64_t j = 0; j < tupleSize; ++j) {
          if (it == values.end()) {
            return failure(); // Unexpected end of values
          }
          indexTuple.push_back((*it).getSExtValue());
          ++it;
        }
        indexTuples.push_back(indexTuple);
      }

      if (areIndexTuplesUnique(indexTuples)) {
        auto newOp = rewriter.create<stablehlo::ScatterOp>(
            op.getLoc(), op.getResultTypes(), op.getInputs(),
            op.getScatterIndices(), op.getUpdates(),
            op.getScatterDimensionNumbers(), op.getIndicesAreSortedAttr(),
            rewriter.getBoolAttr(true));
        newOp.getUpdateComputation().takeBody(op.getUpdateComputation());
        rewriter.replaceOp(op, newOp);
        return success();
      }
    }

    return failure();
  }

private:
  bool areIndexTuplesUnique(
      const SmallVector<SmallVector<int64_t>> &indexTuples) const {
    bool hasUnique = true;
    for (int64_t i = 0; i < indexTuples.size() && hasUnique; ++i) {
      for (int64_t j = i + 1; j < indexTuples.size() && hasUnique; ++j) {
        if (std::equal(indexTuples[i].begin(), indexTuples[i].end(),
                       indexTuples[j].begin(), indexTuples[j].end())) {
          hasUnique = false;
          break;
        }
      }
    }
    return hasUnique;
  }
};

// (add (mul a x) (mul a y)) -> (mul a (add x y))
template <typename Op>
struct AssociativeCommonMulOpReordering final : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
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
struct AssociativeBinaryOpReordering : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
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

struct ReduceTransposeSimplify : public OpRewritePattern<stablehlo::ReduceOp> {
  using OpRewritePattern<stablehlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1) // TODO: support for multiple inputs
      return failure();

    auto input = op.getInputs()[0];
    auto transposeOp = input.getDefiningOp<stablehlo::TransposeOp>();
    if (!transposeOp)
      return failure();

    auto transposeInput = transposeOp.getOperand();
    auto transposeInputType = transposeInput.getType().cast<ShapedType>();
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
    auto elementType = op.getType(0).cast<ShapedType>().getElementType();
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
struct SignAbsSimplify : public OpRewritePattern<stablehlo::MulOp> {
  using OpRewritePattern<stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::MulOp op,
                                PatternRewriter &rewriter) const override {
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

struct AbsPositiveSimplify : public OpRewritePattern<stablehlo::AbsOp> {
  using OpRewritePattern<stablehlo::AbsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AbsOp op,
                                PatternRewriter &rewriter) const override {

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

  return insertionDims;
}

struct TransposeReshapeToBroadcast final
    : OpRewritePattern<stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
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

struct BroadcastInDimIsReshape final
    : OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern<mlir::stablehlo::BroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
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

struct PadConcatToConcatPad
    : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern<stablehlo::ConcatenateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concatOp,
                                PatternRewriter &rewriter) const override {

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

struct SliceSelect : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

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

struct ConstPadConcatToConcat : public OpRewritePattern<stablehlo::PadOp> {
  using OpRewritePattern<stablehlo::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::PadOp padOp,
                                PatternRewriter &rewriter) const override {
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
  double constantFactor;
  Value valFactor;
  T term;
  Term(double constantFactor, Value valFactor, T term)
      : constantFactor(constantFactor), valFactor(valFactor), term(term) {}
};

template <typename ST, typename Child>
struct SumToReductionBase : public OpRewritePattern<ST> {
  using OpRewritePattern<ST>::OpRewritePattern;

  LogicalResult matchAndRewrite(ST op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().getElementType().isFloat())
      return failure();

    SmallVector<Term<stablehlo::SliceOp>> done0;
    SmallVector<Term<Value>> todo;
    todo.emplace_back(1.0, (Value) nullptr, (Value)op.getResult());
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
          todo.emplace_back(cur.constantFactor * other.getSplatValue<double>(),
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
          todo.emplace_back(cur.constantFactor / other.getSplatValue<double>(),
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
        double tally = 0;
        for (auto term : done)
          tally += term.constantFactor;

        finalToAdd.emplace_back(tally, nullptr, done[0].term);
        hasMerge = true;
        continue;
      }

      std::map<int, double> terms;
      for (int i = 0; i < done.size(); i++) {
        int offset = (int)done[i].term.getStartIndices()[offsetDim] -
                     (int)done[0].term.getStartIndices()[offsetDim];
        terms[offset] += done[i].constantFactor;
      }
      assert(terms.size() > 1);

      int startidx = terms.begin()->first;
      auto lastItr = terms.end();
      lastItr--;
      int lastidx = lastItr->first;
      assert(lastidx != startidx);

      SmallVector<double> pad(lastidx + 1 - startidx);
      // Check contiguous
      for (auto &term : terms) {
        pad[term.first - startidx] += term.second;
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
      finalToAdd.emplace_back(1, nullptr, conv);
      hasMerge = true;
    }

    if (!hasMerge)
      return failure();

    assert(finalToAdd.size());

    Value result = nullptr;
    for (auto term : finalToAdd) {
      Value intermediate = term.term;
      if (term.constantFactor != 1) {

        intermediate = rewriter.create<stablehlo::MulOp>(
            op.getLoc(), intermediate,
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), intermediate.getType(),
                makeAttr(intermediate.getType(), term.constantFactor)
                    .cast<ElementsAttr>()));
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
                           const std::map<int, double> &pad, int startidx,
                           int lastidx) {
    double start = pad.find(startidx)->second;
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
               const std::map<int, double> &pad, int startidx, int lastidx) {
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
                      RankedTensorType T, const std::map<int, double> &pad,
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
               const std::map<int, double> &pad, int startidx, int lastidx) {
    return SumToReductionBase<ST, SumToReduceWindow<ST>>::reduceWindowApplies(
        offsetDim, T, pad, startidx, lastidx);
  }

  Value makeReduction(PatternRewriter &rewriter, Value input, size_t offsetDim,
                      RankedTensorType T, const std::map<int, double> &pad,
                      int startidx, int lastidx, Value filter) {
    double factor = pad.begin()->second;

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
    if (factor != 1) {
      result = rewriter.create<stablehlo::MulOp>(
          input.getLoc(), result,
          rewriter.create<stablehlo::ConstantOp>(
              input.getLoc(), result.getType(),
              makeAttr(result.getType(), factor).cast<ElementsAttr>()));
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto block = rewriter.createBlock(&redwin.getBody(), {}, tys, locs);
      rewriter.setInsertionPointToStart(block);
      auto addOp = rewriter.create<stablehlo::AddOp>(
          input.getLoc(), block->getArgument(0), block->getArgument(1));
      auto ret = rewriter.create<stablehlo::ReturnOp>(input.getLoc(),
                                                      addOp.getResult());
    }

    return result;
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

namespace {

struct EnzymeHLOOptPass
    : public enzyme::impl::EnzymeHLOOptPassBase<EnzymeHLOOptPass> {
  using EnzymeHLOOptPassBase::EnzymeHLOOptPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    RewritePatternSet patterns(context);
    mlir::enzyme::populateWithGenerated(patterns);

    patterns
        .add<AddSimplify, SubSimplify, AndSimplify, MaxSimplify, MinSimplify,
             OrSimplify, NegateSimplify, MulSimplify, DivSimplify, RemSimplify,
             PowSimplify, SqrtSimplify, CosSimplify, SinSimplify, NoopSlice,
             NoopReverse, SliceSlice, PadSimplify, ShiftRightLogicalSimplify,
             NegativePadToSlice, TanhSimplify, ExpSimplify, SliceSimplify,
             ConvertSimplify, TransposeSimplify, DotGeneralSimplify,
             DynamicSliceToStatic, DynamicUpdateSliceElim, ReduceToReshape,
             BroadcastToReshape, GatherSimplify, ReshapeEmptyBroadcast,
             BroadcastReshape, ConstPropThroughBarrier,
             ReplaceNegAddWithSubtract, SignAbsSimplify, AbsPositiveSimplify,
             TransposeReshapeToBroadcast>(context, PatternBenefit(65000));
    patterns.add<IotaSimplify, BroadcastInDimSimplify>(
        max_constant_expansion, context, PatternBenefit(65000));

    patterns.add<ConvertConcat, DynamicUpdateToConcat, SliceOfDynamicUpdate,
                 SliceElementwise, SliceReshapeElementwise, SlicePad,
                 SliceReshapePad, DotReshapeDot, ConcatConstProp,
                 DynamicUpdateSliceConstProp, NotConstProp, IsFiniteConstProp,
                 LogConstProp, LogPlusConstProp, ChloInfConstProp,
                 GammaConstProp, ConcatFuse, ConcatToBroadcast, PadPad,
                 PadReshapePad, ConcatPushBinop<stablehlo::AddOp>,
                 ConcatPushBinop<stablehlo::MulOp>, ScatterToDynamicUpdateSlice,
                 ReduceConcat, ConcatSlice, SliceConcat, SliceIf,
                 SliceReshapeConcat, BinBroadcastSplat<stablehlo::AddOp>,
                 BinBroadcastSplat<stablehlo::SubtractOp>,
                 BinBroadcastSplat<stablehlo::DivOp>,
                 BinBroadcastSplat<stablehlo::MulOp>>(context);

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
                 AssociativeBinaryOpReordering<stablehlo::OrOp>>(context);

    patterns.add<BinopPadToConcat<stablehlo::AddOp>,
                 BinopPadToConcat<stablehlo::MulOp>, ConcatPad,
                 PadConcatToConcatPad, SliceSelect, PadReduceWindow>(context);

    if (passses & 512) {
      patterns.add<TransposeDotReorder, DotTranspose, ConvolutionTranspose,
                   TransposeConvolution, EinsumTranspose, TransposeEinsum,
                   ConvertConvertFloat, ConcatToPad, ConcatAppendingReshape,
                   ReshapeIota, DUSDUS, DUSDUSConcat, DUSConcat, DUSPad,
                   SliceDUSToConcat>(context);
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
                   CSE<stablehlo::NegOp>>(context, PatternBenefit(65000));
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
      patterns
          .add<TransposeWhile, TransposeSlice, TransposeConcat, TransposeDUS,
               TransposeIota, TransposeReduceWindow, TransposeReduce>(context);
      patterns.add<TransposeElementwise>(true, context);
    }

    if (passses & (2048 * 64)) {
      // add reshape push up cases here
      patterns.add<ReshapeElementwise, ReshapeOfConcatToConcatOfReshape,
                   ReshapeDUS, ReshapeSlice, ReshapePad, ReshapeReduceWindow>(
          context);
    }

    if (passses & (2048 * 128)) {
      // Conflicts with ConcatPad
      patterns.add<ConstPadConcatToConcat>(context);
    }

    if (all_finite)
      patterns.add<AllFinite>(context);
    if (no_nan || all_finite) {
      patterns.add<NoNan, NoNanSelfSubSimplify>(context);
    }
    patterns.add<NoNanAddSubSimplify>((no_nan || all_finite), context);

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
        SelectOpUsedWithinIf,
        TransposeBroadcastInDimToBroadcastInDim,
        BroadcastInDimTransposeToBroadcastInDim,
        TransposeIsReshape,
        BroadcastInDimIsReshape,
        WhileDeadResults,
        ZeroExtentTensorCanon,
        CompareSelectSimplify,
        NotSelectSimplify,
        CommonCompareExpressionRewrite,
        ScatterUpdateComputationConstProp,
        ScatterIndicesAreUnique,
        ReduceTransposeSimplify,
        BroadcastIotaSimplify
      >(context);

    patterns.add<SumToReduceWindow<stablehlo::AddOp>, SumToReduceWindow<stablehlo::SubtractOp>>(context);

    patterns.add<WhileSimplify>(false, context);

    patterns.add<WhileLICM>(false, context);

    // clang-format on
    patterns.add<SelectOpCanon>(max_constant_expansion, context,
                                PatternBenefit(65000));
    patterns.add<ConcatenateOpCanon>(max_constant_expansion, context,
                                     PatternBenefit(65000));

    GreedyRewriteConfig config;
    config.maxIterations = max_iterations;
    config.useTopDownTraversal = top_down;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
