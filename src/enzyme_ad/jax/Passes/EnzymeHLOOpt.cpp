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

  bool isLoopResultDead(OpResult result) const {
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

      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      getBackwardSlice(terminatorOperand.get(), &backwardSlice, options);
      for (Operation *op : backwardSlice) {
        if (llvm::is_contained(op->getOperands(), bodyArgument))
          return false;
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
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ReturnOp>(
        terminator, TypeRange(), terminatorOperands, terminator->getAttrs());
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> deadResults;
    for (OpResult result : op.getResults()) {
      if (!isLoopResultDead(result))
        continue;

      deadResults.push_back(result.getResultNumber());
    }
    if (deadResults.empty())
      return failure();

    SetVector<Operation *> condSlice, bodySlice;
    for (int64_t i : deadResults) {
      getForwardSlice(op.getCond().getArgument(i), &condSlice);
      getForwardSlice(op.getBody().getArgument(i), &bodySlice);
    }
    condSlice.remove(op.getCond().front().getTerminator());
    bodySlice.remove(op.getBody().front().getTerminator());
    replaceTerminator(rewriter, op.getCond(), deadResults);
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

    for (int64_t i : llvm::reverse(deadResults))
      op.getBody().eraseArgument(i);
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
struct ReshapePad final : OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto pad = op.getOperand().getDefiningOp<stablehlo::PadOp>();
    if (!pad)
      return failure();
    if (!llvm::hasSingleElement(pad->getUsers()))
      return failure();

    if (!reshapePadHelper(op, rewriter).succeeded())
      return failure();
    return success();
  }
};

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

  if (inputAttr.isSplat()) {

    ty = RankedTensorType::get(
        {}, cast<ShapedType>(op->getResultTypes()[0]).getElementType());
    inputTen = stablehlo::makeTensor(inputAttr.resizeSplat(ty));
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

// ((add x cst0) cst1) -> (add x1 (add cst0 cst1))
template <typename T> struct BinOpConstSimplify : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    // Only apply to integers
    if (!isa<IntegerType>(op.getType().getElementType()))
      return failure();

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsConst = matchPattern(lhs, m_Constant());
    auto rhsConst = matchPattern(rhs, m_Constant());

    if (!lhsConst && !rhsConst)
      return failure();

    auto constVal = lhsConst ? lhs : rhs;
    auto otherOp = lhsConst ? rhs.template getDefiningOp<T>()
                            : lhs.template getDefiningOp<T>();

    if (!otherOp)
      return failure();

    auto otherLhs = otherOp.getRhs();
    auto otherRhs = otherOp.getLhs();

    auto otherLhsConst = matchPattern(otherLhs, m_Constant());
    auto otherRhsConst = matchPattern(otherRhs, m_Constant());

    if (!otherLhsConst && !otherRhsConst)
      return failure();

    // Both op and other have a constant operand
    // group constants to a new op.
    auto otherConstVal = otherLhsConst ? otherLhs : otherRhs;
    auto otherOperand = otherLhsConst ? otherRhs : otherLhs;

    auto constantAdd = rewriter.create<T>(
        otherOp.getLoc(), op.getResult().getType(), constVal, otherConstVal);
    rewriter.replaceOpWithNewOp<T>(op, otherOperand, constantAdd);

    return success();
  }
};

struct ReplaceNegAddWithSubtract : public OpRewritePattern<stablehlo::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::AddOp op,
                                PatternRewriter &rewriter) const final {
    auto negateOp = op.getRhs().getDefiningOp<stablehlo::NegOp>();

    if (!negateOp)
      return failure();

    if (!negateOp->hasOneUse())
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
            op, type, operand, dims);
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

// Replace while op iteration variables which are not updated with their
// upcoming value
struct WhileSimplify : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> operands;

    Block *cond = &op.getCond().front(), *body = &op.getBody().front();
    Operation *bodyTerm = body->getTerminator();

    int deleted = 0;
    for (auto &opOperand : op->getOpOperands()) {
      Value inputValue = opOperand.get();

      auto i = opOperand.getOperandNumber() - deleted;
      Value bodyArg = body->getArgument(i);
      Value condArg = cond->getArgument(i);

      if (inputValue.getDefiningOp<stablehlo::ConstantOp>() &&
          bodyArg == bodyTerm->getOperand(i)) {
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

struct TransposeReduceSimplify : public OpRewritePattern<stablehlo::ReduceOp> {
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
    for (auto dim : reduceDimensions) {
      isReduced[dim] = true;
    }

    SmallVector<int64_t> oldDims, newDims;
    for (int64_t i = 0; i < transposePermutation.size(); ++i) {
      if (!isReduced[transposePermutation[i]]) {
        oldDims.push_back(transposePermutation[i]);
      }
      if (!isReduced[i]) {
        newDims.push_back(i);
      }
    }

    // Create final permutation
    SmallVector<int64_t> finalPermutation(newDims.size());
    for (int64_t i = 0; i < newDims.size(); ++i) {
      for (int64_t j = 0; j < oldDims.size(); ++j) {
        if (newDims[i] == oldDims[j]) {
          finalPermutation[j] = i;
          break;
        }
      }
    }

    rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
        op, newReduceOp.getResult(0), finalPermutation);
    return success();
  }
};

///////////////  End Imported from stablehlo

#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.cpp.inc"

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

namespace {
struct EnzymeHLOOptPass
    : public enzyme::impl::EnzymeHLOOptPassBase<EnzymeHLOOptPass> {
  using EnzymeHLOOptPassBase::EnzymeHLOOptPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    RewritePatternSet patterns(context);
    patterns.add<AddSimplify, SubSimplify, AndSimplify, MaxSimplify,
                 MinSimplify, OrSimplify, NegateSimplify, MulSimplify,
                 DivSimplify, RemSimplify, PowSimplify, SqrtSimplify,
                 CosSimplify, SinSimplify, NoopSlice, NoopReverse, SliceSlice,
                 PadSimplify, ShiftRightLogicalSimplify, NegativePadToSlice,
                 TanhSimplify, ExpSimplify, SliceSimplify, ConvertSimplify,
                 TransposeSimplify, DotGeneralSimplify, DynamicSliceToStatic,
                 DynamicUpdateSliceElim, ReduceToReshape, BroadcastToReshape,
                 GatherSimplify, ReshapeEmptyBroadcast, BroadcastReshape,
                 ConstPropThroughBarrier, ReplaceNegAddWithSubtract>(
        context, PatternBenefit(65000));
    patterns.add<IotaSimplify, BroadcastInDimSimplify>(
        max_constant_expansion, context, PatternBenefit(65000));

    patterns.add<
        ConvertConcat, DynamicUpdateToConcat, SliceOfDynamicUpdate,
        SliceElementwise, SliceReshapeElementwise, SlicePad, SliceReshapePad,
        DotReshapeDot, ConcatConstProp, DynamicUpdateSliceConstProp,
        LogConstProp, LogPlusConstProp, ChloInfConstProp, GammaConstProp,
        ConcatFuse, ConcatToBroadcast, PadPad, PadReshapePad,
        ConcatPushBinop<stablehlo::AddOp>, ConcatPushBinop<stablehlo::MulOp>,
        ScatterToDynamicUpdateSlice, ReduceConcat, ConcatSlice, SliceConcat,
        SliceReshapeConcat, BinBroadcastSplat<stablehlo::AddOp>,
        BinBroadcastSplat<stablehlo::SubtractOp>,
        BinBroadcastSplat<stablehlo::DivOp>,
        BinBroadcastSplat<stablehlo::MulOp>,
        BinOpConstSimplify<stablehlo::AddOp>,
        BinOpConstSimplify<stablehlo::MulOp>>(context);

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
                 BinopPadToConcat<stablehlo::MulOp>, ConcatPad>(context);

    if (passses & 512)
      patterns.add<TransposeDotReorder, DotTranspose, ConvolutionTranspose,
                   TransposeConvolution, EinsumTranspose, TransposeEinsum,
                   ConvertConvertFloat, ConcatToPad, ConcatAppendingReshape,
                   ReshapeIota>(context);

    if (passses & 1024)
      patterns.add<FullReduceReshapeOrTranspose>(context);

    if (passses & 1)
      patterns.add<SliceTranspose, SliceReshapeTranspose, SliceBroadcast>(
          context);
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

    if (all_finite)
      patterns.add<AllFinite>(context);
    if (no_nan || all_finite) {
      patterns.add<NoNan, NoNanSelfSubSimplify>(context);
    }
    patterns.add<NoNanAddSubSimplify>((no_nan || all_finite), context);

    // clang-format off
    patterns.add<
        BroadcastInDimOpCanon,
        ChainedDynamicBroadcastInDimCanonicalization,
        CompareOpCanon,
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
        SelectOpUsedWithinIf,
        TransposeBroadcastInDimToBroadcastInDim,
        BroadcastInDimTransposeToBroadcastInDim,
        TransposeIsReshape,
        WhileDeadResults,
        WhileSimplify,
        ZeroExtentTensorCanon,
        CompareSelectSimplify,
        NotSelectSimplify,
        CommonCompareExpressionRewrite,
        ScatterUpdateComputationConstProp,
        ScatterIndicesAreUnique,
        TransposeReduceSimplify
      >(context);
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
