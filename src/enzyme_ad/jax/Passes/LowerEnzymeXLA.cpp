//===- LowerEnzymeXLA.cpp - Lower generic enzymexla ops -------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers `enzymexla.stencil` to StableHLO / Triton.
//
// Two lowering paths are provided, applied in priority order:
//
//  1. **Primary – Triton (static rank-1 shapes):** Lower to an
//     `enzymexla_tt_ext.call` wrapping a JIT-compiled Triton kernel.
//     The Triton IR is built directly with C++ op builders (no textual
//     MLIR string generation/parsing).
//
//     1-D stencils (rank == 1):
//       The kernel tiles the output into blocks of BLOCK_SIZE=1024 lanes.
//       Each block issues K `tt.load` calls covering adjacent, overlapping
//       input ranges [base+k, base+BLOCK+k). With num_stages=3, Triton's
//       software pipeliner stages these loads through SMEM.
//
//  2. **Fallback – convolution (static shapes):** Used when
//     `--force-slice-fallback` is set or when Triton matching fails.
//     Lowers to a single `stablehlo.convolution` (valid, stride-1,
//     no padding). Requires statically-known shapes.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

using namespace mlir::enzymexla::triton_ext;

//===----------------------------------------------------------------------===//
// Triton MLIR code generation helpers
//===----------------------------------------------------------------------===//

static std::string ci(int64_t v) { return std::to_string(v); }

// Create a vector i32 constant: dense<v> : tensor<Bxi32>.
static Value makeVecI32Const(PatternRewriter &rewriter, Location loc,
                             int64_t B, int64_t v) {
  auto ty = RankedTensorType::get({B}, rewriter.getI32Type());
  auto attr = DenseElementsAttr::get(ty, rewriter.getI32IntegerAttr(v));
  return arith::ConstantOp::create(rewriter, loc, ty, attr);
}

// Create a zero-filled vector of element type `elemTy`: tensor<BxelemTy>.
static Value makeVecZeroConst(PatternRewriter &rewriter, Location loc,
                              int64_t B, Type elemTy) {
  auto ty = RankedTensorType::get({B}, elemTy);
  TypedAttr zero = isa<Float64Type>(elemTy)
                       ? TypedAttr(rewriter.getF64FloatAttr(0.0))
                       : TypedAttr(rewriter.getF32FloatAttr(0.0f));
  return arith::ConstantOp::create(rewriter, loc, ty,
                                   DenseElementsAttr::get(ty, zero));
}

// Build a rank-1 Triton stencil kernel directly using C++ op builders.
//
// Signature is:
//   tt.func @name(%input: !tt.ptr<T>, %weights: !tt.ptr<T>, %output: !tt.ptr<T>)
//
// with T in {f32, f64}. The computation is:
//   output[i] = sum_{k=0..K-1} input[i + k] * weights[k]
// for i in [0, Nout), executed in blocks of B lanes.
static LogicalResult buildTriton1DKernel(triton::FuncOp func,
                                         PatternRewriter &rewriter,
                                         Location loc, int64_t B, int64_t K,
                                         int64_t Nout) {
  Block *entry = func.addEntryBlock();
  rewriter.setInsertionPointToStart(entry);

  Value inputPtr = entry->getArgument(0);
  Value weightPtr = entry->getArgument(1);
  Value outputPtr = entry->getArgument(2);

  auto ptrTy = inputPtr.getType();
  Type elemTy = triton::getPointeeType(ptrTy);
  auto vecI32Ty = RankedTensorType::get({B}, rewriter.getI32Type());
  auto vecElemTy = RankedTensorType::get({B}, elemTy);
  auto vecPtrTy = RankedTensorType::get({B}, ptrTy);

  Value cB = arith::ConstantIntOp::create(rewriter, loc, B, 32);
  Value pid = triton::GetProgramIdOp::create(rewriter, loc,
                                             triton::ProgramIDDim::X);
  Value base = arith::MulIOp::create(rewriter, loc, pid, cB);
  Value rng = triton::MakeRangeOp::create(rewriter, loc, vecI32Ty, 0u,
                                          static_cast<uint32_t>(B));
  Value bspl = triton::SplatOp::create(rewriter, loc, vecI32Ty, base);
  Value oidx = arith::AddIOp::create(rewriter, loc, bspl, rng);

  Value nout = makeVecI32Const(rewriter, loc, B, Nout);
  Value omask = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                      oidx, nout);

  Value ispl = triton::SplatOp::create(rewriter, loc, vecPtrTy, inputPtr);
  Value acc = makeVecZeroConst(rewriter, loc, B, elemTy);

  for (int64_t k = 0; k < K; ++k) {
    Value iidx = oidx;
    if (k != 0) {
      Value kvec = makeVecI32Const(rewriter, loc, B, k);
      iidx = arith::AddIOp::create(rewriter, loc, oidx, kvec);
    }

    Value iptrs = triton::AddPtrOp::create(rewriter, loc, vecPtrTy, ispl, iidx);
    Value ival = triton::LoadOp::create(
        rewriter, loc, iptrs, omask, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, /*isVolatile=*/false);

    Value wval;
    if (k == 0) {
      wval = triton::LoadOp::create(rewriter, loc, weightPtr,
                                    triton::CacheModifier::NONE,
                                    triton::EvictionPolicy::NORMAL,
                                    /*isVolatile=*/false);
    } else {
      Value ck = arith::ConstantIntOp::create(rewriter, loc, k, 32);
      Value wptr =
          triton::AddPtrOp::create(rewriter, loc, ptrTy, weightPtr, ck);
      wval = triton::LoadOp::create(rewriter, loc, wptr,
                                    triton::CacheModifier::NONE,
                                    triton::EvictionPolicy::NORMAL,
                                    /*isVolatile=*/false);
    }

    Value wspl = triton::SplatOp::create(rewriter, loc, vecElemTy, wval);
    Value prod = arith::MulFOp::create(rewriter, loc, ival, wspl);
    acc = arith::AddFOp::create(rewriter, loc, acc, prod);
  }

  Value ospl = triton::SplatOp::create(rewriter, loc, vecPtrTy, outputPtr);
  Value optrs = triton::AddPtrOp::create(rewriter, loc, vecPtrTy, ospl, oidx);
  triton::StoreOp::create(rewriter, loc, optrs, acc, omask,
                          triton::CacheModifier::NONE,
                          triton::EvictionPolicy::NORMAL);
  triton::ReturnOp::create(rewriter, loc, ValueRange{});
  return success();
}

//===----------------------------------------------------------------------===//
// Primary Triton lowering: StencilOp → TritonCallOp
//
// Requires static rank-1 input/weight/output shapes. The Triton kernel IR is
// built through C++ op builders (no parseSourceString path).
//===----------------------------------------------------------------------===//

struct StencilOpTritonLowering
    : public OpRewritePattern<enzymexla::StencilOp> {
  StencilOpTritonLowering(MLIRContext *ctx, int32_t numWarps = 4,
                          int32_t numStages = 3, PatternBenefit benefit = 2)
      : OpRewritePattern(ctx, benefit), numWarps(numWarps),
        numStages(numStages) {}

  int32_t numWarps;
  int32_t numStages;

  LogicalResult matchAndRewrite(enzymexla::StencilOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto weightsType = dyn_cast<RankedTensorType>(op.getWeights().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !weightsType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    if (!inputType.hasStaticShape() || !weightsType.hasStaticShape() ||
        !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
                                         "requires static tensor shapes");

    if (inputType.getRank() != 1 || weightsType.getRank() != 1 ||
        outputType.getRank() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "Triton builder path currently supports rank-1 only");

    Type elemType = outputType.getElementType();
    if (!isa<Float32Type, Float64Type>(elemType))
      return rewriter.notifyMatchFailure(op, "only f32/f64 are supported");

    const int64_t Nout = outputType.getShape()[0];
    const int64_t K = weightsType.getShape()[0];
    const int64_t BLOCK = 1024;
    const int64_t gridX = (Nout + BLOCK - 1) / BLOCK;

    auto loc = op.getLoc();
    auto *ctx = op.getContext();

    std::string funcName = "stencil_r1_" + ci(Nout) + "x" +
                           (isa<Float64Type>(elemType) ? "f64" : "f32");

    SymbolTable symTable(SymbolTable::getNearestSymbolTable(op));
    OpBuilder::InsertionGuard guard(rewriter);

    rewriter.setInsertionPointToEnd(&symTable.getOp()->getRegion(0).front());
    auto outerMod = TritonModuleOp::create(rewriter, loc, "stencil_tt_mod");
    Block *body = rewriter.createBlock(&outerMod.getBodyRegion());
    symTable.insert(outerMod);

    rewriter.setInsertionPointToStart(body);
    auto innerMod = ModuleOp::create(loc);
    innerMod.setSymName("stencil_tt_inner");
    innerMod->setAttr("enzymexla.num_warps",
                      rewriter.getI32IntegerAttr(numWarps));
    innerMod->setAttr("enzymexla.num_stages",
                      rewriter.getI32IntegerAttr(numStages));
    rewriter.insert(innerMod);

    rewriter.setInsertionPointToStart(innerMod.getBody());
    auto ptrTy = triton::getPointerType(elemType);
    auto fnTy = rewriter.getFunctionType({ptrTy, ptrTy, ptrTy}, {});
    auto func = triton::FuncOp::create(rewriter, loc, funcName, fnTy);

    if (failed(buildTriton1DKernel(func, rewriter, loc, BLOCK, K, Nout)))
      return rewriter.notifyMatchFailure(op, "failed to build Triton kernel");

    SmallVector<FlatSymbolRefAttr, 2> nestedRefs = {
        FlatSymbolRefAttr::get(innerMod.getSymNameAttr()),
        FlatSymbolRefAttr::get(StringAttr::get(ctx, funcName))};
    SymbolRefAttr fn =
        SymbolRefAttr::get(ctx, outerMod.getSymNameAttr(), nestedRefs);

    auto TI64 = [&](int64_t v) -> Value {
      auto ty = RankedTensorType::get({}, rewriter.getI64Type());
      return stablehlo::ConstantOp::create(
          rewriter, loc, ty,
          cast<ElementsAttr>(mlir::enzyme::makeAttr(ty, v)));
    };

    rewriter.setInsertionPoint(op);
    auto callOp = TritonCallOp::create(
        rewriter, loc, TypeRange{outputType}, fn,
        TI64(gridX), TI64(1), TI64(1),
        TI64(1), TI64(1), TI64(1),
        ValueRange{op.getInput(), op.getWeights()},
        StringAttr::get(ctx, ""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/ArrayAttr::get(ctx, {}),
        /*res_attrs=*/ArrayAttr::get(ctx, {}),
        /*output_operand_aliases=*/nullptr,
        rewriter.getUnitAttr());

    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fallback lowering: StencilOp → stablehlo.convolution
//
// Maps the N-D stencil directly to a batched valid convolution:
//   input   [d0,...,dR-1]  →  lhs  [1, 1, d0,...,dR-1]  (batch=0, feat=1)
//   weights [k0,...,kR-1]  →  rhs  [1, 1, k0,...,kR-1]  (in_feat=0, out_feat=1)
//   output  [o0,...,oR-1]  ←  conv [1, 1, o0,...,oR-1]  → reshape
//
// No padding, stride-1, no dilation (valid cross-correlation).
// Requires statically-known shapes.
//===----------------------------------------------------------------------===//

struct StencilOpFallbackLowering
    : public OpRewritePattern<enzymexla::StencilOp> {
  using OpRewritePattern<enzymexla::StencilOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::StencilOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto weightsType = cast<RankedTensorType>(op.getWeights().getType());
    auto outputType = cast<RankedTensorType>(op.getResult().getType());

    if (!inputType.hasStaticShape() || !weightsType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");

    auto loc = op.getLoc();
    const int64_t ndim = inputType.getRank();
    Type elemTy = outputType.getElementType();

    // Reshape input: [d0,...,dR-1] → [1, 1, d0,...,dR-1]
    SmallVector<int64_t> inputConvShape = {1, 1};
    for (int64_t d : inputType.getShape())
      inputConvShape.push_back(d);
    Value inputConv = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get(inputConvShape, elemTy),
        op.getInput());

    // Reshape weights: [k0,...,kR-1] → [1, 1, k0,...,kR-1]
    SmallVector<int64_t> weightsConvShape = {1, 1};
    for (int64_t k : weightsType.getShape())
      weightsConvShape.push_back(k);
    Value weightsConv = stablehlo::ReshapeOp::create(
        rewriter, loc, RankedTensorType::get(weightsConvShape, elemTy),
        op.getWeights());

    // Spatial dims: [2, 3, ..., ndim+1]
    SmallVector<int64_t> spatialDims(ndim);
    for (int64_t i = 0; i < ndim; ++i)
      spatialDims[i] = i + 2;

    auto dnums = stablehlo::ConvDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/1,
        /*input_spatial_dimensions=*/spatialDims,
        /*kernel_input_feature_dimension=*/0,
        /*kernel_output_feature_dimension=*/1,
        /*kernel_spatial_dimensions=*/spatialDims,
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/1,
        /*output_spatial_dimensions=*/spatialDims);

    // Conv output shape: [1, 1, o0,...,oR-1]
    SmallVector<int64_t> convOutShape = {1, 1};
    for (int64_t d : outputType.getShape())
      convOutShape.push_back(d);

    Value convResult = stablehlo::ConvolutionOp::create(
        rewriter, loc, RankedTensorType::get(convOutShape, elemTy),
        inputConv, weightsConv,
        /*window_strides=*/nullptr,
        /*padding=*/nullptr,
        /*lhs_dilation=*/nullptr,
        /*rhs_dilation=*/nullptr,
        /*window_reversal=*/nullptr,
        dnums,
        /*feature_group_count=*/rewriter.getI64IntegerAttr(1),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/nullptr);

    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(op, outputType,
                                                      convResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct LowerEnzymeXLAPass
    : public enzyme::impl::LowerEnzymeXLAPassBase<LowerEnzymeXLAPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    if (forceSliceFallback) {
      patterns.add<StencilOpFallbackLowering>(context);
    } else {
      // Primary: Triton kernel (static rank-1 shapes, GPU).
      patterns.add<StencilOpTritonLowering>(context);
      // Fallback: static-shape slice/mul/add unrolling.
      patterns.add<StencilOpFallbackLowering>(context);
    }

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }

    // Verify all StencilOps were lowered.
    auto walkResult = getOperation()->walk([&](Operation *innerOp) {
      if (isa<enzymexla::StencilOp>(innerOp)) {
        innerOp->emitError("Failed to lower enzymexla.stencil");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
