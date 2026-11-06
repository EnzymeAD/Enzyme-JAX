//===- RaiseMosaicCustomCallPass.cpp - Raise Mosaic TPU custom calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise StableHLO `tpu_custom_call` operations
// carrying base64-encoded Mosaic MLIR payloads into `enzymexla_mosaic_ext`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"

#include "src/enzyme_ad/jax/Dialect/MosaicExt/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_RAISEMOSAICCUSTOMCALLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzymexla;

namespace {

static Value createConstI64Tensor(OpBuilder &builder, Location loc,
                                  uint64_t val) {
  auto tensorTy = RankedTensorType::get({}, builder.getI64Type());
  auto attr = DenseIntElementsAttr::get(tensorTy, static_cast<int64_t>(val));
  return stablehlo::ConstantOp::create(builder, loc, tensorTy, attr);
}

static Value createZeroTensor(OpBuilder &builder, Location loc,
                              RankedTensorType resTy) {
  if (TypedAttr zeroAttr = builder.getZeroAttr(resTy)) {
    return stablehlo::ConstantOp::create(builder, loc, resTy,
                                         cast<ElementsAttr>(zeroAttr));
  }
  auto fallbackAttr = cast<ElementsAttr>(mlir::enzyme::makeAttr(resTy, 0));
  return stablehlo::ConstantOp::create(builder, loc, resTy, fallbackAttr);
}

static bool hasSparseCoreFunc(ModuleOp mod) {
  bool found = false;
  mod.walk([&](func::FuncOp fn) {
    if (auto coreAttr = fn->getAttrOfType<tpu::CoreTypeAttr>("tpu.core_type")) {
      tpu::CoreType ct = coreAttr.getValue();
      if (ct == tpu::CoreType::kScScalarSubcore ||
          ct == tpu::CoreType::kScVectorSubcore) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

static bool isDefaultCompactSparseCoreConfig(const llvm::json::Object &obj) {
  if (obj.size() != 1)
    return false;
  std::optional<StringRef> tiling = obj.getString("tiling");
  return tiling && *tiling == "COMPACT";
}

// Strip `body` and default Mosaic `tpu_custom_call` fields from the parsed
// `backend_config` JSON object. If only default fields were present, return an
// empty string so raised `enzymexla_mosaic_ext.call` IR stays clean; otherwise
// serialize the remaining JSON configuration so lowering restores it verbatim.
static std::string extractResidualBackendConfig(llvm::json::Object rootObj,
                                                ModuleOp innerMod) {
  bool isSparseCore = hasSparseCoreFunc(innerMod);
  auto [hasComm, _] = tpu::mightCommunicateBetweenChips(innerMod);

  if (llvm::json::Object *ccObj = rootObj.getObject("custom_call_config")) {
    ccObj->erase("body");

    if (std::optional<int64_t> fmt =
            ccObj->getInteger("serialization_format")) {
      if (*fmt == 1)
        ccObj->erase("serialization_format");
    }
    if (!isSparseCore) {
      if (std::optional<bool> needsLayout =
              ccObj->getBoolean("needs_layout_passes")) {
        if (*needsLayout)
          ccObj->erase("needs_layout_passes");
      }
    }
    if (hasComm) {
      if (std::optional<bool> hc = ccObj->getBoolean("has_communication")) {
        if (*hc)
          ccObj->erase("has_communication");
      }
    }
    if (ccObj->empty())
      rootObj.erase("custom_call_config");
  }

  if (isSparseCore) {
    if (std::optional<StringRef> devType = rootObj.getString("device_type")) {
      if (*devType == "DEVICE_TYPE_SPARSECORE")
        rootObj.erase("device_type");
    }
    if (const llvm::json::Object *scObj =
            rootObj.getObject("sparse_core_config")) {
      if (isDefaultCompactSparseCoreConfig(*scObj))
        rootObj.erase("sparse_core_config");
    }
  }

  if (rootObj.empty())
    return "";

  std::string residual;
  llvm::raw_string_ostream os(residual);
  os << llvm::json::Value(std::move(rootObj));
  os.flush();
  return residual;
}

static func::FuncOp findEntryFunction(ModuleOp mod, StringRef kernelNameHint) {
  if (auto mainFn = mod.lookupSymbol<func::FuncOp>("main"))
    return mainFn;
  if (!kernelNameHint.empty()) {
    if (auto namedFn = mod.lookupSymbol<func::FuncOp>(kernelNameHint))
      return namedFn;
  }
  func::FuncOp soleFn = nullptr;
  unsigned count = 0;
  for (auto fn : mod.getOps<func::FuncOp>()) {
    soleFn = fn;
    ++count;
  }
  return count == 1 ? soleFn : nullptr;
}

static void extractGridDims(func::FuncOp entryFn, uint64_t &gx, uint64_t &gy,
                            uint64_t &gz) {
  gx = 1;
  gy = 1;
  gz = 1;
  auto boundsAttr =
      entryFn->getAttrOfType<DenseI64ArrayAttr>("iteration_bounds");
  if (!boundsAttr)
    return;
  ArrayRef<int64_t> bounds = boundsAttr.asArrayRef();
  auto safeDim = [](int64_t d) -> uint64_t {
    return (d > 0 && !ShapedType::isDynamic(d)) ? static_cast<uint64_t>(d) : 1;
  };
  if (bounds.size() >= 1)
    gx = safeDim(bounds[0]);
  if (bounds.size() >= 2)
    gy = safeDim(bounds[1]);
  if (bounds.size() >= 3) {
    uint64_t prod = 1;
    for (size_t i = 2; i < bounds.size(); ++i)
      prod *= safeDim(bounds[i]);
    gz = prod;
  }
}

static bool isCustomCallInherentOrHandledAttr(StringRef name) {
  return name == "call_target_name" || name == "has_side_effect" ||
         name == "backend_config" || name == "mhlo.backend_config" ||
         name == "api_version" || name == "called_computations" ||
         name == "operand_layouts" || name == "result_layouts" ||
         name == "output_operand_aliases" || name == "kernel_name";
}

struct RaiseMosaicCustomCallPattern final
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (callOp.getCallTargetName() != "tpu_custom_call")
      return failure();

    for (Type resTy : callOp->getResultTypes()) {
      auto ranked = dyn_cast<RankedTensorType>(resTy);
      if (!ranked || !ranked.hasStaticShape())
        return failure();
    }

    StringAttr configStrAttr;
    if (auto configOpt = callOp.getBackendConfig())
      configStrAttr = dyn_cast<StringAttr>(*configOpt);
    if (!configStrAttr)
      configStrAttr = callOp->getAttrOfType<StringAttr>("mhlo.backend_config");
    if (!configStrAttr)
      return failure();

    llvm::Expected<llvm::json::Value> parsedJson =
        llvm::json::parse(configStrAttr.getValue());
    if (!parsedJson) {
      llvm::consumeError(parsedJson.takeError());
      return failure();
    }

    llvm::json::Object *rootObj = parsedJson->getAsObject();
    if (!rootObj)
      return failure();
    llvm::json::Object *ccConfig = rootObj->getObject("custom_call_config");
    if (!ccConfig)
      return failure();
    std::optional<StringRef> bodyB64 = ccConfig->getString("body");
    if (!bodyB64)
      return failure();

    std::vector<char> rawPayload;
    if (llvm::Error err = llvm::decodeBase64(*bodyB64, rawPayload)) {
      llvm::consumeError(std::move(err));
      return failure();
    }

    MLIRContext *ctx = callOp.getContext();
    bool prevAllowUnregistered = ctx->allowsUnregisteredDialects();
    ctx->allowUnregisteredDialects(true);
    llvm::scope_exit restoreUnregistered(
        [&]() { ctx->allowUnregisteredDialects(prevAllowUnregistered); });

    ParserConfig parserConfig(ctx, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> parsedMod = mlir::parseSourceString<ModuleOp>(
        StringRef(rawPayload.data(), rawPayload.size()), parserConfig);
    if (!parsedMod)
      return failure();

    if ((*parsedMod)->hasAttr("stable_mosaic.version")) {
      PassManager pm(ctx);
      pm.enableVerifier(true);
      pm.addPass(mlir::tpu::createMosaicSerdePass(
          mlir::tpu::MosaicSerdePassOptions{/*serialize=*/false,
                                            /*target_version=*/-1}));
      if (failed(pm.run(*parsedMod)))
        return failure();
      (*parsedMod)->removeAttr("stable_mosaic.version");
    }

    StringAttr kernelNameAttr =
        callOp->getAttrOfType<StringAttr>("kernel_name");
    StringRef kernelNameHint =
        kernelNameAttr ? kernelNameAttr.getValue() : StringRef();
    if (std::optional<StringRef> origModSym = parsedMod->getSymName()) {
      (*parsedMod)
          ->setAttr("enzymexla.module_sym_name",
                    rewriter.getStringAttr(*origModSym));
    }

    func::FuncOp entryFn = findEntryFunction(*parsedMod, kernelNameHint);
    if (!entryFn)
      return failure();

    // Rename `@main` to `kernel_name` when safe so the raised `MosaicExt`
    // module exposes the human-readable kernel symbol on the call op.
    StringRef calleeName = entryFn.getSymName();
    if (kernelNameAttr) {
      (*parsedMod)->setAttr("enzymexla.kernel_name", kernelNameAttr);
      if (!kernelNameHint.empty() && kernelNameHint != calleeName &&
          !parsedMod->lookupSymbol(kernelNameHint) &&
          SymbolTable::symbolKnownUseEmpty(entryFn, *parsedMod)) {
        entryFn.setSymName(kernelNameHint);
        calleeName = entryFn.getSymName();
      }
    }

    uint64_t gx = 1, gy = 1, gz = 1;
    extractGridDims(entryFn, gx, gy, gz);

    std::string residualBackendConfig =
        extractResidualBackendConfig(std::move(*rootObj), *parsedMod);

    Operation *symTableOp =
        SymbolTable::getNearestSymbolTable(callOp.getOperation());
    if (!symTableOp)
      return failure();
    SymbolTable symTable(symTableOp);

    ModuleOp innerMod = *parsedMod;
    (void)parsedMod.release();

    Location loc = callOp.getLoc();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&symTableOp->getRegion(0).front());
      auto outerMod = enzymexla::mosaic_ext::MosaicModuleOp::create(
          rewriter, loc, "mosaic_module", /*sym_visibility=*/nullptr);
      Block *bodyBlock = rewriter.createBlock(&outerMod.getBodyRegion());
      symTable.insert(outerMod);

      rewriter.setInsertionPointToStart(bodyBlock);
      rewriter.insert(innerMod);
      innerMod.setSymName("mosaic_module_inner");

      SmallVector<FlatSymbolRefAttr, 2> nestedRefs = {
          FlatSymbolRefAttr::get(innerMod.getSymNameAttr()),
          FlatSymbolRefAttr::get(rewriter.getStringAttr(calleeName))};
      SymbolRefAttr fnRef =
          SymbolRefAttr::get(ctx, outerMod.getSymNameAttr(), nestedRefs);

      rewriter.setInsertionPoint(callOp);
      Value gridxVal = createConstI64Tensor(rewriter, loc, gx);
      Value gridyVal = createConstI64Tensor(rewriter, loc, gy);
      Value gridzVal = createConstI64Tensor(rewriter, loc, gz);
      Value clusterxVal = createConstI64Tensor(rewriter, loc, 1);
      Value clusteryVal = createConstI64Tensor(rewriter, loc, 1);
      Value clusterzVal = createConstI64Tensor(rewriter, loc, 1);

      SmallVector<Value> inputs(callOp.getInputs());
      SmallVector<Attribute> aliases;
      SmallVector<Attribute> operandLayouts;
      if (auto layouts = callOp.getOperandLayoutsAttr())
        operandLayouts.append(layouts.begin(), layouts.end());
      bool hasOperandLayouts = callOp.getOperandLayoutsAttr() != nullptr;
      ArrayAttr origResultLayouts = callOp.getResultLayoutsAttr();

      unsigned numResults = callOp->getNumResults();
      SmallVector<int64_t> aliasedOperandForResult(numResults, -1);
      if (auto origAliases = callOp.getOutputOperandAliasesAttr()) {
        for (Attribute attr : origAliases) {
          aliases.push_back(attr);
          auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
          ArrayRef<int64_t> outIdxs = alias.getOutputTupleIndices();
          unsigned resIdx =
              (numResults == 1 || outIdxs.empty()) ? 0 : outIdxs[0];
          if (resIdx < numResults)
            aliasedOperandForResult[resIdx] = alias.getOperandIndex();
        }
      }

      SmallVector<int64_t> syntheticOutputResults;
      bool needExplicitResultLayouts = false;
      for (unsigned i = 0; i < numResults; ++i) {
        if (aliasedOperandForResult[i] >= 0) {
          if (hasOperandLayouts && origResultLayouts &&
              i < origResultLayouts.size()) {
            int64_t opIdx = aliasedOperandForResult[i];
            if (opIdx < static_cast<int64_t>(operandLayouts.size()) &&
                operandLayouts[opIdx] != origResultLayouts[i]) {
              needExplicitResultLayouts = true;
            }
          }
          continue;
        }
        auto resTy = cast<RankedTensorType>(callOp->getResult(i).getType());
        Value zeroVal = createZeroTensor(rewriter, loc, resTy);
        inputs.push_back(zeroVal);
        syntheticOutputResults.push_back(static_cast<int64_t>(i));

        aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            ctx,
            numResults == 1 ? ArrayRef<int64_t>{}
                            : ArrayRef<int64_t>{static_cast<int64_t>(i)},
            static_cast<int64_t>(inputs.size() - 1), {}));

        if (hasOperandLayouts) {
          if (origResultLayouts && i < origResultLayouts.size()) {
            operandLayouts.push_back(origResultLayouts[i]);
          } else {
            SmallVector<int64_t> minorToMajor;
            for (int64_t d = resTy.getRank() - 1; d >= 0; --d)
              minorToMajor.push_back(d);
            operandLayouts.push_back(rewriter.getIndexTensorAttr(minorToMajor));
          }
        }
      }

      if (!syntheticOutputResults.empty()) {
        innerMod->setAttr(
            "enzymexla.synthetic_output_results",
            rewriter.getDenseI64ArrayAttr(syntheticOutputResults));
      }
      if (needExplicitResultLayouts && origResultLayouts) {
        innerMod->setAttr("enzymexla.result_layouts", origResultLayouts);
      }

      SmallVector<NamedAttribute> extraAttrs;
      for (NamedAttribute namedAttr : callOp->getAttrs()) {
        if (!isCustomCallInherentOrHandledAttr(namedAttr.getName().strref()))
          extraAttrs.push_back(namedAttr);
      }

      auto mosaicCall = enzymexla::mosaic_ext::MosaicCallOp::create(
          rewriter, loc, callOp->getResultTypes(), fnRef, gridxVal, gridyVal,
          gridzVal, clusterxVal, clusteryVal, clusterzVal, inputs,
          rewriter.getStringAttr(residualBackendConfig),
          hasOperandLayouts ? rewriter.getArrayAttr(operandLayouts) : nullptr,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
          rewriter.getArrayAttr(aliases),
          callOp.getHasSideEffect() ? nullptr : rewriter.getUnitAttr());

      for (NamedAttribute extra : extraAttrs)
        mosaicCall->setAttr(extra.getName(), extra.getValue());

      rewriter.replaceOp(callOp, mosaicCall.getResults());
    }

    return success();
  }
};

struct RaiseMosaicCustomCallPass
    : public mlir::enzyme::impl::RaiseMosaicCustomCallPassBase<
          RaiseMosaicCustomCallPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseMosaicCustomCallPattern>(patterns.getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // end anonymous namespace
