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

    SymbolTable symTable(SymbolTable::getNearestSymbolTable(callOp));

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&symTable.getOp()->getRegion(0).front());

    auto innerMod = cast<mlir::ModuleOp>(mir.release().getOperation());
    auto outerMod = enzymexla::triton_ext::TritonModuleOp::create(
        rewriter, callOp.getLoc(), "triton_module");

    Block *b = rewriter.createBlock(&outerMod.getBodyRegion());

    symTable.insert(outerMod);

    rewriter.setInsertionPointToStart(b);
    rewriter.insert(innerMod);

    innerMod.setSymName("triton_module_inner");

    uint64_t gridx = kCall.grid_0(), gridy = kCall.grid_1(),
             gridz = kCall.grid_2();
    uint64_t clusterx = kCall.kernel().cluster_dim_0(),
             clustery = kCall.kernel().cluster_dim_1(),
             clusterz = kCall.kernel().cluster_dim_2();

    SmallVector<FlatSymbolRefAttr, 2> nestedRefs = {
        FlatSymbolRefAttr::get(innerMod.getSymNameAttr()),
        FlatSymbolRefAttr::get(StringAttr::get(callOp.getContext(),
                                               kCall.kernel().kernel_name()))};

    SymbolRefAttr fn = SymbolRefAttr::get(
        callOp.getContext(), outerMod.getSymNameAttr(), nestedRefs);

    auto TI64 = [&](uint64_t v) -> Value {
      auto TT = RankedTensorType::get({}, rewriter.getI64Type());
      return stablehlo::ConstantOp::create(
          rewriter, callOp.getLoc(), TT,
          cast<ElementsAttr>(mlir::enzyme::makeAttr(TT, v)));
    };

    rewriter.setInsertionPoint(callOp);
    rewriter.replaceOpWithNewOp<enzymexla::triton_ext::TritonCallOp>(
        callOp, callOp->getResultTypes(), fn,

        TI64(gridx), TI64(gridy), TI64(gridz),

        TI64(clusterx), TI64(clustery), TI64(clusterz),

        callOp.getInputs(),
        /* backendConfig */ StringAttr::get(callOp.getContext(), ""),
        callOp.getOperandLayoutsAttr(), callOp.getResultLayoutsAttr(),
        /* argAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
        /* resAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
        callOp.getOutputOperandAliasesAttr(), nullptr);

    return success();
  }
};

struct RaiseTritonCustomCallPass
    : public mlir::enzyme::impl::RaiseTritonCustomCallPassBase<
          RaiseTritonCustomCallPass> {
  using RaiseTritonCustomCallPassBase::RaiseTritonCustomCallPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseTritonCustomCallPattern>(patterns.getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace
