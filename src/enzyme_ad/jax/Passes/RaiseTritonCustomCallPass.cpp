//===- RaiseTritonCustomCallPass.cpp - Raise triton custom calls ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise stablehlo triton custom calls to the
// triton ext dialect.
//
//===----------------------------------------------------------------------===//

#include <zlib.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "jaxlib/gpu/triton.pb.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RAISETRITONCUSTOMCALLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzymexla;
using namespace stablehlo;

namespace {

static Value TI64(OpBuilder &builder, Location loc, uint64_t v) {
  auto TT = RankedTensorType::get({}, builder.getI64Type());
  return stablehlo::ConstantOp::create(
      builder, loc, TT, cast<ElementsAttr>(mlir::enzyme::makeAttr(TT, v)));
}

static LogicalResult
replaceWithTritonCall(stablehlo::CustomCallOp callOp, PatternRewriter &rewriter,
                      ModuleOp innerMod, StringRef kernelName, uint64_t gridx,
                      uint64_t gridy, uint64_t gridz,
                      std::optional<int32_t> numWarps = std::nullopt,
                      std::optional<int32_t> numStages = std::nullopt) {
  SymbolTable symTable(
      SymbolTable::getNearestSymbolTable(callOp.getOperation()));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&symTable.getOp()->getRegion(0).front());

  auto outerMod = enzymexla::triton_ext::TritonModuleOp::create(
      rewriter, callOp->getLoc(), "triton_module");

  Block *b = rewriter.createBlock(&outerMod.getBodyRegion());

  symTable.insert(outerMod);

  rewriter.setInsertionPointToStart(b);
  rewriter.insert(innerMod);

  innerMod.setSymName("triton_module_inner");
  if (numWarps) {
    innerMod.getOperation()->setAttr("enzymexla.num_warps",
                                     rewriter.getI32IntegerAttr(*numWarps));
  }
  if (numStages) {
    innerMod.getOperation()->setAttr("enzymexla.num_stages",
                                     rewriter.getI32IntegerAttr(*numStages));
  }

  SmallVector<FlatSymbolRefAttr, 2> nestedRefs = {
      FlatSymbolRefAttr::get(innerMod.getSymNameAttr()),
      FlatSymbolRefAttr::get(StringAttr::get(callOp.getContext(), kernelName))};

  SymbolRefAttr fn = SymbolRefAttr::get(callOp.getContext(),
                                        outerMod.getSymNameAttr(), nestedRefs);

  rewriter.setInsertionPoint(callOp);
  rewriter.replaceOpWithNewOp<enzymexla::triton_ext::TritonCallOp>(
      callOp, callOp->getResultTypes(), fn,

      TI64(rewriter, callOp->getLoc(), gridx),
      TI64(rewriter, callOp->getLoc(), gridy),
      TI64(rewriter, callOp->getLoc(), gridz),

      TI64(rewriter, callOp->getLoc(), 1), TI64(rewriter, callOp->getLoc(), 1),
      TI64(rewriter, callOp->getLoc(), 1),

      callOp.getInputs(),
      /* backendConfig */ StringAttr::get(callOp.getContext(), ""),
      callOp.getOperandLayoutsAttr(), callOp.getResultLayoutsAttr(),
      /* argAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
      /* resAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
      callOp.getOutputOperandAliasesAttr(),
      callOp.getHasSideEffect() ? nullptr : rewriter.getUnitAttr());

  return success();
}

struct RaiseTritonCustomCallPattern final
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp callOp,
                                PatternRewriter &rewriter) const {
    auto callTargetName = callOp.getCallTargetName();
    if (callTargetName != "triton_kernel_call")
      return failure();

    auto configAttr = callOp.getBackendConfig();
    if (!configAttr)
      return failure();

    auto configStrAttr = dyn_cast<StringAttr>(*configAttr);
    if (!configStrAttr)
      return failure();

    std::string config(configStrAttr.getValue());
    uLongf destlen = config.size() * 5;
    std::string data;

    while (true) {
      data.resize(destlen);
      int ret =
          uncompress(reinterpret_cast<Bytef *>(data.data()), &destlen,
                     reinterpret_cast<Bytef *>(config.data()), config.size());
      if (ret == Z_OK) {
        data.resize(destlen);
        break;
      } else if (ret == Z_BUF_ERROR) {
        destlen *= 2;
      } else {
        // cannot decompress backend config
        return failure();
      }
    }

    jax_triton::TritonAnyKernelCall proto;
    if (!proto.ParseFromString(data))
      return failure();

    // TODO: support autotuned kernel calls
    if (proto.has_autotuned_kernel_call())
      return failure();

    auto kCall = proto.kernel_call();
    OwningOpRef<ModuleOp> mir = mlir::parseSourceString<ModuleOp>(
        kCall.kernel().ttir(), callOp.getContext());
    if (!mir)
      return failure();

    auto innerMod = cast<mlir::ModuleOp>(mir.release().getOperation());
    return replaceWithTritonCall(callOp, rewriter, innerMod,
                                 kCall.kernel().kernel_name(), kCall.grid_0(),
                                 kCall.grid_1(), kCall.grid_2());
  }
};

struct RaiseXLAGpuTritonCustomCallPattern final
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp callOp,
                                PatternRewriter &rewriter) const {
    auto callTargetName = callOp.getCallTargetName();
    if (callTargetName != "__gpu$xla.gpu.triton")
      return failure();

    auto configAttr = callOp.getBackendConfig();
    if (!configAttr)
      return failure();

    auto configDict = dyn_cast<DictionaryAttr>(*configAttr);
    if (!configDict)
      return failure();

    auto irAttr = configDict.getAs<StringAttr>("ir");
    auto nameAttr = configDict.getAs<StringAttr>("name");
    if (!irAttr || !nameAttr)
      return failure();

    OwningOpRef<ModuleOp> mir = mlir::parseSourceString<ModuleOp>(
        irAttr.getValue(), callOp.getContext());
    if (!mir)
      return failure();

    auto getInt = [&](StringRef name, int64_t defaultVal) -> int64_t {
      if (auto attr = configDict.getAs<IntegerAttr>(name))
        return attr.getInt();
      return defaultVal;
    };

    int64_t gridx = getInt("grid_x", 1);
    int64_t gridy = getInt("grid_y", 1);
    int64_t gridz = getInt("grid_z", 1);
    int64_t numWarps = getInt("num_warps", 4);
    int64_t numStages = getInt("num_stages", 3);

    auto innerMod = cast<mlir::ModuleOp>(mir.release().getOperation());
    return replaceWithTritonCall(
        callOp, rewriter, innerMod, nameAttr.getValue(), gridx, gridy, gridz,
        static_cast<int32_t>(numWarps), static_cast<int32_t>(numStages));
  }
};

struct RaiseTritonCustomCallPass
    : public mlir::enzyme::impl::RaiseTritonCustomCallPassBase<
          RaiseTritonCustomCallPass> {
  using RaiseTritonCustomCallPassBase::RaiseTritonCustomCallPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<RaiseTritonCustomCallPattern, RaiseXLAGpuTritonCustomCallPattern>(
            patterns.getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace
