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

struct SliceSimplification final : OpRewritePattern<mlir::stablehlo::SliceOp> {
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
    DenseElementsAttr inp;
    matchPattern(op->getOperand(0), m_Constant(&inp));
    if (inp) {
      if (inp.isSplat()) {
        rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
            op, op.getType(),
            mlir::SplatElementsAttr::get(op.getType(),
                                         inp.getSplatValue<mlir::Attribute>()));
        return success();
      }
      auto inp0 = mlir::stablehlo::evalConstantOp(inp);
      auto out = mlir::stablehlo::evalBroadcastInDimOp(
          inp0, mlir::stablehlo::Axes(op.getBroadcastDimensions()),
          op.getType());
      rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(op, op.getType(),
                                                         fromTensor(out));
      return success();
    }

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

struct EnzymeHLOOptPass : public EnzymeHLOOptPassBase<EnzymeHLOOptPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);
    patterns.add<DynamicSliceToStatic, SlicePad, SliceSlice, AddPad,
                 PadSimplify, DotReshapeDot, ConcatConstProp, ConcatFuse,
                 /*ScatterToPad, */ BroadcastToReshape, ReduceToReshape,
                 ReduceConcat, SliceConcat, SliceSimplification, CosSimplify,
                 SinSimplify, SqrtSimplify, AddSimplify, SubSimplify,
                 AndSimplify, OrSimplify, NegateSimplify, MulSimplify,
                 DivSimplify, PowSimplify, BinBroadcastSplat<stablehlo::AddOp>,
                 BinBroadcastSplat<stablehlo::SubtractOp>,
                 BinBroadcastSplat<stablehlo::DivOp>,
                 BinBroadcastSplat<stablehlo::MulOp>>(context);
    mlir::stablehlo::populateStablehloCanonicalizationPatterns(context,
                                                               &patterns);

    GreedyRewriteConfig config;
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
