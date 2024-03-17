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

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/transforms/Passes.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
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
      if (product != lhsShape[rhsPos])
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

    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    SmallVector<int64_t> step;

    for (auto &&[pstart, pend, pstep, nstart, nend, nstep] : llvm::zip(
             prev.getStartIndices(), prev.getLimitIndices(), prev.getStrides(),
             op.getStartIndices(), op.getLimitIndices(), op.getStrides())) {
      start.push_back(pstart + pstep * nstart);
      step.push_back(pstep * nstep);
      end.push_back(pstart + pstep * nstart + pstep * nstep * (nend - nstart));
    }
    rewriter.replaceOpWithNewOp<stablehlo::SliceOp>(op, prev.getOperand(),
                                                    start, end, step);
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
    if (!matchPattern(op.getStartIndices()[0], m_Constant(&startattr))) {
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

    SmallVector<int64_t> start;
    SmallVector<int64_t> end;
    SmallVector<int64_t> step;
    for (auto ind : invertPermutation(transpose.getPermutation())) {
      start.push_back(op.getStartIndices()[ind]);
      end.push_back(op.getLimitIndices()[ind]);
      step.push_back(op.getStrides()[ind]);
    }

    auto newslice = rewriter.create<stablehlo::SliceOp>(
        op.getLoc(), transpose.getOperand(), start, end, step);
    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, newslice, transpose.getPermutation());
    return success();
  }
};

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
    for (auto &&[nstart, nend, nstep, lpad, hpad, interior, inshape] :
         llvm::zip(op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
                   pad.getEdgePaddingLow(), pad.getEdgePaddingHigh(),
                   pad.getInteriorPadding(),
                   pad.getOperand().getType().getShape())) {
      if (nstep != 1)
        return failure();
      if (interior != 0)
        return failure();

      // start of slice starts after end of value being padded
      if (nstart - lpad >= inshape) {
        rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
            op, op.getType(), pad.getPaddingValue(),
            rewriter.getDenseI64ArrayAttr({}));
        return success();
      }
      // slice ends before the start of value being padded
      if (nend - lpad < inshape) {
        rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
            op, op.getType(), pad.getPaddingValue(),
            rewriter.getDenseI64ArrayAttr({}));
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

    for (auto v : concat.getOperands()) {
      IRMapping map;
      map.map(op.getInitValues()[0], identity);
      map.map(op.getInputs()[0], v);
      auto next = rewriter.clone(*op, map)->getResult(0);
      map.map(innerOp.getOperand(0), prev);
      map.map(innerOp.getOperand(1), next);
      prev = rewriter.clone(innerOp, map)->getResult(0);
    }

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
    DenseMap<Operation *, int> toclone;
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
    while (todo.size()) {
      auto cur = todo.pop_back_val();

      SmallVector<Value> vals;
      for (auto op : cur->getOperands())
        vals.push_back(map.lookup(op));

      auto res =
          rewriter.create(cur->getLoc(), cur->getName().getIdentifier(), vals,
                          TypeRange(changeType), cur->getAttrs(), {}, {});

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

    if (op.getStrides()[dim] != 1)
      return failure();

    SmallVector<Value> postConcat;
    size_t curdim = 0;
    for (auto v : concat.getInputs()) {
      auto ty = v.getType().cast<RankedTensorType>();
      auto nextdim = ty.getShape()[dim];
      if (op.getStartIndices()[dim] >= curdim + nextdim) {
        curdim += nextdim;
        continue;
      }
      if (op.getLimitIndices()[dim] <= curdim) {
        curdim += nextdim;
        continue;
      }
      SmallVector<int64_t> nstart(op.getStartIndices().begin(),
                                  op.getStartIndices().end());
      SmallVector<int64_t> nend(op.getLimitIndices().begin(),
                                op.getLimitIndices().end());
      nstart[dim] -= curdim;
      if (nstart[dim] < 0)
        nstart[dim] = 0;
      nend[dim] -= curdim;
      if (nend[dim] > nextdim)
        nend[dim] = nextdim;
      auto subslice = rewriter.create<stablehlo::SliceOp>(
          op.getLoc(), v, nstart, nend, op.getStrides());
      postConcat.push_back(subslice);
      curdim += nextdim;
    }
    rewriter.replaceOpWithNewOp<stablehlo::ConcatenateOp>(op, postConcat, dim);
    return success();
  }
};

DenseElementsAttr fromTensor(stablehlo::Tensor inp) {
  auto type = inp.getType();
  auto elemType = type.getElementType();

  if (elemType.isBF16()) {
    auto floatValues =
        ArrayRef((char *)inp.getData(), 2 * inp.getNumElements());
    return DenseFPElementsAttr::getFromRawBuffer(type, floatValues);
  }

  if (elemType.isF32()) {
    auto floatValues = ArrayRef((float *)inp.getData(), inp.getNumElements());
    return DenseFPElementsAttr::get(type, floatValues);
  }

  if (elemType.isF64()) {
    auto floatValues = ArrayRef((double *)inp.getData(), inp.getNumElements());
    return DenseFPElementsAttr::get(type, floatValues);
  }

  if (elemType.isSignlessInteger(8)) {
    auto floatValues = ArrayRef((int8_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isSignlessInteger(16)) {
    auto floatValues = ArrayRef((int16_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isSignlessInteger(32)) {
    auto floatValues = ArrayRef((int32_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isSignlessInteger(64)) {
    auto floatValues = ArrayRef((int64_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isUnsignedInteger(8)) {
    auto floatValues = ArrayRef((uint8_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isUnsignedInteger(16)) {
    auto floatValues =
        ArrayRef((uint16_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isUnsignedInteger(32)) {
    auto floatValues =
        ArrayRef((uint32_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }
  if (elemType.isUnsignedInteger(64)) {
    auto floatValues =
        ArrayRef((uint64_t *)inp.getData(), inp.getNumElements());
    return DenseIntElementsAttr::get(type, floatValues);
  }

  assert(0);
  return {};
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

/*

    %1192 = stablehlo.pad %1189, %cst_0, low = [0], high = [1], interior = [0] :
   (tensor<1xf32>, tensor<f32>) -> tensor<2xf32> %1193 = arith.addf %1191, %1192
   : tensor<2xf32>

*/
struct AddPad final : OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    for (int i = 0; i < 2; i++) {
      if (auto lhs = op->getOperand(i).getDefiningOp<stablehlo::PadOp>()) {
        auto rhs = op->getOperand(1 - i);

        if (!matchPattern(lhs.getPaddingValue(), m_AnyZeroFloat())) {
          continue;
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
            auto prevSlice = rewriter.create<stablehlo::SliceOp>(
                op.getLoc(), rhs, starts, limits, strides);
            vals.push_back(prevSlice);
          }

          starts[idx] = lhs.getEdgePaddingLow()[idx];
          limits[idx] = type.getShape()[idx] - lhs.getEdgePaddingHigh()[idx];

          auto midSlice = rewriter.create<stablehlo::SliceOp>(
              op.getLoc(), rhs, starts, limits, strides);
          auto mid = rewriter.create<stablehlo::AddOp>(op.getLoc(), midSlice,
                                                       lhs.getOperand());
          vals.push_back(mid);

          if (lhs.getEdgePaddingHigh()[idx] != 0) {
            starts[idx] = type.getShape()[idx] - lhs.getEdgePaddingHigh()[idx];
            limits[idx] = type.getShape()[idx];
            auto postSlice = rewriter.create<stablehlo::SliceOp>(
                op.getLoc(), rhs, starts, limits, strides);
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

struct ReshapePad final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
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
        inner_shape.push_back(
            pad.getOperand().getType().getShape()[curiotaidx]);
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
        inps.push_back(mlir::stablehlo::evalConstantOp(c));
      auto out = mlir::stablehlo::evalConcatenateOp(inps, op.getDimension(),
                                                    op.getType());
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                         fromTensor(out));
      return success();
    }
    return failure();
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

#if 0
struct ScatterToPad final : OpRewritePattern<mlir::stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto type = dyn_cast<RankedTensorType>(op.getType());
    if (!type)
      return failure();

    Block &body = op.getUpdateComputation().front();
    if (body.size() != 1)

    Operation &innerOp = body.front();
    if (!isa<ReturnOp>(&innerOp)) {
      return failure();
    }
    if (innerOp->getNumOperands() != 1) {
      return failure();
    }
    auto retop = innerOp->getOperand(0).dyn_cast<BlockArgument>();
    if (!retop) return failure();
    if (retop.getOwner() != &body) return failure();
    if (retop.getArgNumber() != 1) return failure();

    if (op.getInputs().size() != 1) return failure();

    mlir::SplatElementsAttr prev;
    if (!matchPattern(op.getInputs()[0], m_Constant(&prev))) {
      return failure();
    }

    mlir::DenseElementsAttr idx;
    if (!matchPattern(op.getScatterIndices()[0], m_Constant(&idx))) {
      return failure();
    }
    auto idx2 = mlir::stablehlo::evalConstantOp(idx);

    if (!op.getIndicesAreSorted()) return failure();
    if (!op.getUniqueIndices()) return failure();

    auto dims = op.getScatterDimensionNumbers();
    if (dims.getInsertedWindowDims() != op.getScatterDimsToOperandDims())
      return failure();
    for (auto en : llvm::enumerate(dims.getInsertedWindowDims())) {
      if (en.value() != en.index()) return failure();
    }
    
    auto update = op.getUpdates()[0];
    auto updateTy = update.getType().cast<RankedTensorType>();
    if (op.getIndexVectorDim() != updateTy.getShape().size()) return failure();

    SmallVector<int64_t> starts;
    SmallVector<int64_t> edge_padding_high;
    SmallVector<int64_t> interior_padding;
    for (size_t lidx = 0; lidx < idx2.getShape()[idx2.getShape().size()-1]; lidx++) {

      uint64_t start = 0;
      uint64_t step = 0
      for (size_t incidx = 0; incidx < idx2.getShape()[lidx]; incidx++) {
        std::optional<stablehlo::Element> value;
        bool legal = true;
        std::function<void(SmallVector<int64_t>)> checkAllEqual = [&](SmallVector<int64_t> prefix) {
          if (prefix.size() == lidx)
            prefix.push_back(incidx);

          if (prefix.size() == idx2.getShape().size()-1) {
            prefix.push_back(lidx);
            auto cur = idx2.get(prefix);
            if (value) {
              legal &= value == cur;
            } else {
              value = cur;
            }
            return;
          }
          for (size_t j = 0; j < idx2.getShape()[prefix.size()]; j++) {
            SmallVector<int64_t> prefix2(prefix);
            prefix2.push_back(j);
            checkAllEqual(prefix2);
          }
        };
        checkAllEqual({});
        assert(value);

        uint64_t cur = (*value).getIntegerValue().getZExtValue();
        if (incidx == 0) {
          start = cur;
        } else if (incidx == 1) {
          step = cur - start;
        } else {
          // Only support step size of one
          if (start + incidx * step != cur) {
            return failure();
          }
        }

      }
      start.push_back(start);
      edge_padding_high.push_back(idx2.getShape()[lidx] - start - );
      interior_padding.push_back(step - 1);
    }

    auto padval = builder.create<stablehlo::ConstantOp>(op.getLoc(), RankedTensorType::get({}, prev.getType().getElementType()), prev.getSplatValue<Attribute>());
    auto pad = builder.replaceOpWithNewOp<stablehlo::PadOp>(op, update, padval)
    return failure();
};
#endif

struct AddSimplify : public OpRewritePattern<mlir::stablehlo::AddOp> {
  using OpRewritePattern<mlir::stablehlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
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
    } else {
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

struct SubSimplify : public OpRewritePattern<mlir::stablehlo::SubtractOp> {
  using OpRewritePattern<mlir::stablehlo::SubtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op,
                                PatternRewriter &rewriter) const final {

    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOpWithNewOp<stablehlo::NegOp>(op, op.getRhs());
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
    } else {
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

struct NegateSimplify : public OpRewritePattern<mlir::stablehlo::NegOp> {
  using OpRewritePattern<mlir::stablehlo::NegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::NegOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

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

    return failure();
  }
};

struct MulSimplify : public OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const final {

    // 0 * x -> x
    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // x * 0 -> x
    if (matchPattern(op.getRhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    // 1 * x -> x
    if (matchPattern(op.getLhs(), m_OneFloat())) {
      rewriter.replaceOp(op, op.getRhs());
      return success();
    }

    // x * 1 -> x
    if (matchPattern(op.getRhs(), m_OneFloat())) {
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
    } else {
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
    if (matchPattern(op.getLhs(), m_AnyZeroFloat())) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }

    // x / 1 -> x
    if (matchPattern(op.getRhs(), m_OneFloat())) {
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
    } else {
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

struct PowSimplify : public OpRewritePattern<mlir::stablehlo::PowOp> {
  using OpRewritePattern<mlir::stablehlo::PowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::PowOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<Attribute> constants;
    constants.assign(op->getNumOperands(), Attribute());
    for (unsigned i = 0, e = op->getNumOperands(); i != e; ++i)
      matchPattern(op->getOperand(i), m_Constant(&constants[i]));

    if (auto res = constFoldBinaryOpConditional<FloatAttr, FloatAttr::ValueType,
                                                void>(
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

    auto out = mlir::stablehlo::evalIotaOp(op.getIotaDimension(), op.getType());
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
        ten = mlir::stablehlo::evalConstantOp(inp);
      }
      auto out = fromTensor(mlir::stablehlo::evalConvertOp(ten, ty));
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

  LogicalResult matchAndRewrite(mlir::stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      DenseElementsAttr out;
      if (inp.isSplat()) {
        out = inp.resizeSplat(op.getType());
      } else {
        auto ten = mlir::stablehlo::evalConstantOp(inp);
        out = fromTensor(mlir::stablehlo::evalSliceOp(
            ten, stablehlo::Sizes(op.getStartIndices()),
            stablehlo::Sizes(op.getStrides()), op.getType()));
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
        auto ten = mlir::stablehlo::evalConstantOp(inp);
        out = fromTensor(mlir::stablehlo::evalBroadcastInDimOp(
            ten, mlir::stablehlo::Axes(op.getBroadcastDimensions()),
            op.getType()));
      }

      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(), out);
      return success();
    }

    return failure();
  }
};

struct ReshapeSimplify : public OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern<mlir::stablehlo::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(),
          inp.isSplat() ? inp.resizeSplat(op.getType())
                        : inp.reshape(op.getType()));
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
    if (auto res =
            constFoldBinaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants,
                [](const APFloat &a, const APFloat &b)
                    -> std::optional<APFloat> { return (a > b) ? a : b; })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
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
    if (auto res =
            constFoldBinaryOpConditional<FloatAttr, FloatAttr::ValueType, void>(
                constants,
                [](const APFloat &a, const APFloat &b)
                    -> std::optional<APFloat> { return (a < b) ? a : b; })) {
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
          op, op.getType(), res.cast<ElementsAttr>());
      return success();
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
                  return {};
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
    if (auto definingTranspose =
            op.getOperand().getDefiningOp<mlir::stablehlo::TransposeOp>()) {
      llvm::ArrayRef<int64_t> thisPermutation = op.getPermutation();
      llvm::ArrayRef<int64_t> prevPermutation =
          definingTranspose.getPermutation();

      SmallVector<int64_t> newPermutation;
      newPermutation.resize(thisPermutation.size());
      for (unsigned i = 0, e = thisPermutation.size(); i != e; ++i) {
        newPermutation[i] = prevPermutation[thisPermutation[i]];
      }

      rewriter.modifyOpInPlace(op, [&]() {
        op.setPermutation(newPermutation);
        op.setOperand(definingTranspose.getOperand());
      });

      return success();
    }
    return rewriter.notifyMatchFailure(op, "not a transpose(transpose)");
  }
};

struct TransposeConvert : public OpRewritePattern<mlir::stablehlo::ConvertOp> {
  using OpRewritePattern<mlir::stablehlo::ConvertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op,
                                PatternRewriter &rewriter) const final {
    auto resultType = op.getResult().getType().cast<TensorType>();
    auto operandType = op.getOperand().getType().cast<TensorType>();
    if (!resultType.hasStaticShape() || !operandType.hasStaticShape())
      return failure();
    if (resultType.getNumElements() * resultType.getElementTypeBitWidth() >=
        operandType.getNumElements() * operandType.getElementTypeBitWidth())
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
      }

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
        }

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
      auto op2 = op->getOperand(1 - i).template getDefiningOp<T>();
      if (!op2)
        continue;

      for (int j = 0; j < 2; j++) {
        auto pad2 =
            op2->getOperand(j).template getDefiningOp<stablehlo::PadOp>();
        if (!pad2)
          continue;

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

struct MulPad : public OpRewritePattern<mlir::stablehlo::MulOp> {
  using OpRewritePattern<mlir::stablehlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::MulOp op,
                                PatternRewriter &rewriter) const final {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
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
    rewriter.replaceOp(op, newPad);

    return success();
  }
};

struct DivPad : public OpRewritePattern<mlir::stablehlo::DivOp> {
  using OpRewritePattern<mlir::stablehlo::DivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op,
                                PatternRewriter &rewriter) const final {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
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

struct PadDotGeneral : public OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern<mlir::stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const final {
    stablehlo::PadOp pad;
    Value otherArg;
    bool otherIsLHS;
    if (failed(getDefiningZeroPadding(op, rewriter, pad, otherArg, otherIsLHS)))
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
      // If padding along the dim, mark the corresponding other dim for slicing.
      int64_t low = pad.getEdgePaddingLow()[padDim];
      int64_t high = pad.getEdgePaddingHigh()[padDim];
      int64_t interior = pad.getInteriorPadding()[padDim];
      if (low == 0 && high == 0 && interior == 0)
        continue;
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

    if (otherDimsToSlice.empty() && resultDimsToPad.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "contracting dimensions not padded");
    }

    Value nextOtherArg = otherArg;
    if (!otherDimsToSlice.empty()) {
      SmallVector<int64_t> sliceLow, sliceHigh, sliceStride;
      for (auto &&[pos, size] :
           llvm::enumerate(otherArg.getType().cast<TensorType>().getShape())) {
        auto it = llvm::find_if(otherDimsToSlice, [&](auto &tup) {
          return std::get<0>(tup) == pos;
        });
        if (it == otherDimsToSlice.end()) {
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
        op.getDotDimensionNumbersAttr(), op.getPrecisionConfigAttr());

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

struct ReshapeToSlice : public OpRewritePattern<stablehlo::SliceOp> {
  using OpRewritePattern<stablehlo::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::SliceOp op,
                                PatternRewriter &rewriter) const final {
    auto reshape = op.getOperand().getDefiningOp<stablehlo::ReshapeOp>();
    if (!reshape) {
      return rewriter.notifyMatchFailure(op, "defining op is not a reshape");
    }
    std::optional<ReshapeDimMapping> mapping =
        tryFindReshapeDimMapping(reshape);
    if (!mapping) {
      return rewriter.notifyMatchFailure(
          reshape, "reshape is not clearly merging or splitting dimensions");
    }
    if (!mapping->isOnlySplitting()) {
      // TODO: it may still be possible to handle this depending on the slice
      // configuration.
      return rewriter.notifyMatchFailure(reshape,
                                         "reshape is merging dimensions");
    }

    auto sliceOperandType = op.getOperand().getType().cast<TensorType>();
    SmallVector<bool> notSlicedDims;
    notSlicedDims.reserve(sliceOperandType.getRank());
    for (auto [start, limit, stride, dim] :
         llvm::zip(op.getStartIndices(), op.getLimitIndices(), op.getStrides(),
                   sliceOperandType.getShape())) {
      notSlicedDims.push_back(start == 0 && limit == dim && stride == 1);
    }

    auto reshapeOperandType = reshape.getOperand().getType().cast<TensorType>();
    SmallVector<int64_t> starts, limits, strides;
    for (auto [i, dim] : llvm::enumerate(reshapeOperandType.getShape())) {
      SmallVector<int64_t> resultDims = mapping->getMappingFromOperandDim(i);
      if (llvm::hasSingleElement(resultDims)) {
        // Keep existing.
        starts.push_back(op.getStartIndices()[resultDims[0]]);
        limits.push_back(op.getLimitIndices()[resultDims[0]]);
        strides.push_back(op.getStrides()[resultDims[0]]);
        continue;
      }

      if (!llvm::all_of(resultDims,
                        [&](int64_t dim) { return notSlicedDims[dim]; })) {
        return rewriter.notifyMatchFailure(reshape,
                                           "split dimension is also sliced");
      }

      // It's a full slice of the original dimension.
      starts.push_back(0);
      limits.push_back(reshapeOperandType.getDimSize(i));
      strides.push_back(1);
    }

    auto newSlice = rewriter.create<stablehlo::SliceOp>(
        op->getLoc(), reshape.getOperand(), starts, limits, strides);
    auto newReshape = rewriter.create<stablehlo::ReshapeOp>(
        reshape->getLoc(), op.getResult().getType(), newSlice.getResult());
    rewriter.replaceOp(op, newReshape);

    return success();
  }
};

struct EnzymeHLOOptPass : public EnzymeHLOOptPassBase<EnzymeHLOOptPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<
        FullReduceReshapeOrTranspose, ConcatToPad, ConcatAppendingReshape,
        ReshapeIota, ReshapePad, ConvertConcat, DynamicSliceToStatic,
        DynamicUpdateSliceElim, DynamicUpdateToConcat, SliceOfDynamicUpdate,
        SliceTranspose, SlicePad, SliceBroadcast, ReducePad, SliceSlice, AddPad,
        MulPad, DivPad, BinopConstPad<stablehlo::AddOp>,
        BinopConstPad<stablehlo::SubtractOp>, BinopConstPad<stablehlo::MulOp>,
        BinopConstPad<stablehlo::DivOp>, BinopBinopPadPad<stablehlo::AddOp>,
        BinopBinopPadPad<stablehlo::MulOp>, PadSimplify, DotReshapeDot,
        ConcatConstProp, ConcatFuse, ConcatPushBinop<stablehlo::AddOp>,
        ConcatPushBinop<stablehlo::MulOp>, UnaryPadPush<stablehlo::ConvertOp>,
        UnaryPadPush<stablehlo::TanhOp>, UnaryPadPush<stablehlo::ExpOp>,
        TransposePad,
        /*ScatterToPad, */ BroadcastToReshape, ReduceToReshape, ConvertSimplify,
        ReshapeSimplify, SliceSimplify, ReduceConcat, SliceConcat, NoopSlice,
        CosSimplify, SinSimplify, SqrtSimplify, TanhSimplify, ExpSimplify,
        AddSimplify, SubSimplify, AndSimplify, MaxSimplify, MinSimplify,
        OrSimplify, NegateSimplify, MulSimplify, DivSimplify, PowSimplify,
        BinBroadcastSplat<stablehlo::AddOp>,
        BinBroadcastSplat<stablehlo::SubtractOp>,
        BinBroadcastSplat<stablehlo::DivOp>,
        BinBroadcastSplat<stablehlo::MulOp>, TransposeTranspose,
        TransposeConvert, BroadcastReduce, PadDotGeneral, ReshapeToSlice>(
        context);
    patterns.add<IotaSimplify, BroadcastInDimSimplify>(max_constant_expansion,
                                                       context);
    if (all_finite)
      patterns.add<AllFinite>(context);
    if (no_nan || all_finite)
      patterns.add<NoNan>(context);
    mlir::stablehlo::populateStablehloCanonicalizationPatterns(context,
                                                               &patterns);

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

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEnzymeHLOOptPass() {
  return std::make_unique<EnzymeHLOOptPass>();
}
} // namespace enzyme
} // namespace mlir
