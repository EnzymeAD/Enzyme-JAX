//===- LowerMosaic.cpp - Lower MosaicExt kernels to XLA tpu_custom_call ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower `enzymexla_mosaic_ext` modules and calls
// into StableHLO `@tpu_custom_call` operations carrying base64-encoded
// `mosaic-serde` MLIR bytecode inside the JSON `backend_config`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeWriter.h"
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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/JSON.h"

#include "src/enzyme_ad/jax/Dialect/MosaicExt/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

#define DEBUG_TYPE "lower-mosaic"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERMOSAICPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;

namespace {

struct MosaicKernelInfo {
  mosaic_ext::MosaicCallOp call;
  func::FuncOp funcOp;
  ModuleOp innerMod;
  mosaic_ext::MosaicModuleOp mosaicMod;
};

static LogicalResult
collectMosaicKernels(SmallVectorImpl<MosaicKernelInfo> &kernels,
                     SymbolTableCollection &symbolTable,
                     mosaic_ext::MosaicCallOp op) {
  Operation *calleeOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
  if (!calleeOp) {
    return op.emitError() << "Failed to find function '" << op.getFn()
                          << "' in module";
  }

  auto funcOp = dyn_cast<func::FuncOp>(calleeOp);
  if (!funcOp) {
    return op.emitError() << "Expected `func.func` target for Mosaic lowering, "
                             "got '"
                          << calleeOp->getName() << "'";
  }

  auto mosaicMod = funcOp->getParentOfType<mosaic_ext::MosaicModuleOp>();
  if (!mosaicMod) {
    return op.emitError() << "No `enzymexla_mosaic_ext.module` parent found";
  }

  ModuleOp innerMod = nullptr;
  if (auto parentMod = funcOp->getParentOfType<ModuleOp>()) {
    if (mosaicMod->isProperAncestor(parentMod))
      innerMod = parentMod;
  }

  kernels.push_back({op, funcOp, innerMod, mosaicMod});
  return success();
}

static std::optional<uint64_t> getIntFromConstant(Value v) {
  if (!v)
    return 1;
  DenseIntElementsAttr attr;
  if (matchPattern(v, m_Constant(&attr)))
    return attr.getSplatValue<APInt>().getZExtValue();
  return std::nullopt;
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

// Return true if `memrefVal` may be read or atomically accumulated (`add =
// true`) inside the kernel, meaning an explicit output buffer must be
// initialized from `memrefVal` before replacing remaining uses.
static bool hasReadOrAtomicAdd(Value memrefVal) {
  SmallVector<Value, 8> worklist{memrefVal};
  llvm::SmallPtrSet<Value, 8> visited{memrefVal};

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
        if (viewOp.getViewSource() == cur) {
          Value viewRes = viewOp->getResult(0);
          if (visited.insert(viewRes).second)
            worklist.push_back(viewRes);
          continue;
        }
      }
      if (auto tpuReinterp = dyn_cast<tpu::ReinterpretCastOp>(user)) {
        if (tpuReinterp.getInput() == cur) {
          Value viewRes = tpuReinterp.getResult();
          if (visited.insert(viewRes).second)
            worklist.push_back(viewRes);
          continue;
        }
      }
      if (auto vecStore = dyn_cast<tpu::VectorStoreOp>(user)) {
        if (vecStore.getBase() == cur && vecStore.getValueToStore() != cur &&
            !vecStore.getAdd())
          continue;
        return true;
      }
      if (auto tpuStore = dyn_cast<tpu::StoreOp>(user)) {
        if (tpuStore.getBase() == cur && tpuStore.getValueToStore() != cur &&
            !tpuStore.getAdd())
          continue;
        return true;
      }
      if (auto stdVecStore = dyn_cast<vector::StoreOp>(user)) {
        if (stdVecStore.getBase() == cur &&
            stdVecStore.getValueToStore() != cur)
          continue;
        return true;
      }
      if (auto memStore = dyn_cast<memref::StoreOp>(user)) {
        if (memStore.getMemRef() == cur && memStore.getValueToStore() != cur)
          continue;
        return true;
      }
      return true;
    }
  }
  return false;
}

static LogicalResult emitInitialBufferCopy(OpBuilder &builder, Location loc,
                                           func::FuncOp entryFn,
                                           unsigned numGrid, Value srcMemRef,
                                           Value dstMemRef) {
  auto memTy = dyn_cast<MemRefType>(srcMemRef.getType());
  if (!memTy || !memTy.hasStaticShape() || dstMemRef.getType() != memTy)
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entryFn.getBody().front());

  auto emitCopyOps = [&](OpBuilder &b) {
    if (memTy.getRank() == 0) {
      Value loaded = memref::LoadOp::create(b, loc, srcMemRef, ValueRange{});
      memref::StoreOp::create(b, loc, loaded, dstMemRef, ValueRange{});
      return;
    }
    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    SmallVector<Value> zeroIndices(memTy.getRank(), c0);
    auto vecTy = VectorType::get(memTy.getShape(), memTy.getElementType());
    Value loaded =
        vector::LoadOp::create(b, loc, vecTy, srcMemRef, zeroIndices);
    vector::StoreOp::create(b, loc, loaded, dstMemRef, zeroIndices);
  };

  if (numGrid > 0 && !entryFn->hasAttr("window_params")) {
    Value cond;
    for (unsigned g = 0; g < numGrid; ++g) {
      Value gid = entryFn.getArgument(g);
      Type gidTy = gid.getType();
      TypedAttr zeroAttr = builder.getZeroAttr(gidTy);
      if (!zeroAttr)
        return failure();
      Value zeroVal = arith::ConstantOp::create(builder, loc, zeroAttr);
      Value isZero = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, gid, zeroVal);
      if (!cond) {
        cond = isZero;
      } else {
        cond = arith::AndIOp::create(builder, loc, cond, isZero);
      }
    }
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, cond,
                                  /*withElseRegion=*/false);
    OpBuilder thenBuilder = OpBuilder::atBlockBegin(ifOp.thenBlock());
    emitCopyOps(thenBuilder);
  } else {
    emitCopyOps(builder);
  }
  return success();
}

static void pruneUnusedHelperFuncs(ModuleOp mod, func::FuncOp entryFn) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (func::FuncOp fn :
         llvm::make_early_inc_range(mod.getOps<func::FuncOp>())) {
      if (fn == entryFn)
        continue;
      if (SymbolTable::symbolKnownUseEmpty(fn, mod)) {
        fn.erase();
        changed = true;
      }
    }
  }
}

static bool isMosaicCallInherentAttr(StringRef name) {
  return name == "fn" || name == "backend_config" ||
         name == "operand_layouts" || name == "arg_attrs" ||
         name == "res_attrs" || name == "output_operand_aliases" ||
         name == "xla_side_effect_free";
}

struct LowerMosaicPass
    : public mlir::enzyme::impl::LowerMosaicPassBase<LowerMosaicPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp modOp = getOperation();
    MLIRContext *ctx = modOp.getContext();

    SmallVector<MosaicKernelInfo> kernels;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);

    bool collectFailed = false;
    modOp->walk([&](mosaic_ext::MosaicCallOp op) {
      if (failed(collectMosaicKernels(kernels, symbolTable, op)))
        collectFailed = true;
    });
    if (collectFailed) {
      signalPassFailure();
      return;
    }

    llvm::SetVector<mosaic_ext::MosaicModuleOp> candidateModules;
    OpBuilder builder(ctx);

    bool anyFailed = false;
    for (const MosaicKernelInfo &info : kernels) {
      mosaic_ext::MosaicCallOp mosaicCall = info.call;
      candidateModules.insert(info.mosaicMod);

      OwningOpRef<ModuleOp> kernelMod;
      if (info.innerMod) {
        kernelMod = cast<ModuleOp>(info.innerMod->clone());
      } else {
        kernelMod = ModuleOp::create(mosaicCall.getLoc());
        OpBuilder modBuilder = OpBuilder::atBlockBegin(kernelMod->getBody());
        modBuilder.insert(info.funcOp->clone());
      }

      std::string origFuncName = mosaicCall.getFn().getLeafReference().str();
      auto entryFn = kernelMod->lookupSymbol<func::FuncOp>(origFuncName);
      if (!entryFn) {
        mosaicCall.emitError() << "Failed to locate callee '" << origFuncName
                               << "' in cloned module";
        anyFailed = true;
        continue;
      }

      // Determine kernel_name and module sym_name metadata before stripping
      // internal helper attributes.
      bool emitKernelNameAttr = false;
      std::string kernelName = origFuncName;
      if (auto savedKernelName =
              (*kernelMod)
                  ->getAttrOfType<StringAttr>("enzymexla.kernel_name")) {
        kernelName = savedKernelName.getValue().str();
        emitKernelNameAttr = true;
      } else if (origFuncName != "main") {
        kernelName = origFuncName;
        emitKernelNameAttr = true;
      }

      if (auto savedModSym =
              (*kernelMod)
                  ->getAttrOfType<StringAttr>("enzymexla.module_sym_name")) {
        kernelMod->setSymName(savedModSym.getValue());
      } else if (emitKernelNameAttr && !kernelName.empty()) {
        kernelMod->setSymName(kernelName);
      } else if (kernelMod->getSymName() &&
                 *kernelMod->getSymName() == "mosaic_module_inner") {
        kernelMod->getOperation()->removeAttr(SymbolTable::getSymbolAttrName());
      }

      llvm::DenseSet<int64_t> syntheticResults;
      if (auto synAttr = (*kernelMod)
                             ->getAttrOfType<DenseI64ArrayAttr>(
                                 "enzymexla.synthetic_output_results")) {
        for (int64_t idx : synAttr.asArrayRef())
          syntheticResults.insert(idx);
      }

      ArrayAttr savedResultLayouts =
          (*kernelMod)->getAttrOfType<ArrayAttr>("enzymexla.result_layouts");

      (*kernelMod)->removeAttr("enzymexla.kernel_name");
      (*kernelMod)->removeAttr("enzymexla.module_sym_name");
      (*kernelMod)->removeAttr("enzymexla.synthetic_output_results");
      (*kernelMod)->removeAttr("enzymexla.result_layouts");

      // Rename the entry function to `@main` as required by XLA TPU Mosaic.
      if (entryFn.getSymName() != "main") {
        if (Operation *existingMain = kernelMod->lookupSymbol("main")) {
          if (existingMain != entryFn.getOperation()) {
            mosaicCall.emitError(
                "Mosaic module already defines a distinct '@main' symbol");
            anyFailed = true;
            continue;
          }
        }
        if (failed(SymbolTable::replaceAllSymbolUses(
                entryFn, builder.getStringAttr("main"), *kernelMod))) {
          mosaicCall.emitError("Failed to update symbol uses of kernel entry");
          anyFailed = true;
          continue;
        }
        entryFn.setSymName("main");
      }
      entryFn.setSymVisibilityAttr(StringAttr());
      pruneUnusedHelperFuncs(*kernelMod, entryFn);

      // Partition `MosaicCallOp` inputs into real inputs vs synthetic output
      // zero-initializers introduced when raising unaliased `tpu_custom_call`s.
      unsigned totalInputs = mosaicCall.getInputs().size();
      unsigned numResults = mosaicCall->getNumResults();
      SmallVector<int64_t> resultAliasSlot(numResults, -1);
      if (ArrayAttr aliasesAttr = mosaicCall.getOutputOperandAliasesAttr()) {
        for (Attribute attr : aliasesAttr) {
          auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
          ArrayRef<int64_t> outIdxs = alias.getOutputTupleIndices();
          unsigned resIdx =
              (numResults == 1 || outIdxs.empty()) ? 0 : outIdxs[0];
          if (resIdx < numResults)
            resultAliasSlot[resIdx] = alias.getOperandIndex();
        }
      }

      SmallVector<bool> usedByNonSyntheticResult(totalInputs, false);
      for (unsigned r = 0; r < numResults; ++r) {
        int64_t slot = resultAliasSlot[r];
        if (!syntheticResults.contains(r) && slot >= 0 &&
            slot < static_cast<int64_t>(totalInputs)) {
          usedByNonSyntheticResult[slot] = true;
        }
      }

      SmallVector<bool> isSyntheticSlot(totalInputs, false);
      for (unsigned r = 0; r < numResults; ++r) {
        int64_t slot = resultAliasSlot[r];
        if (syntheticResults.contains(r) && slot >= 0 &&
            slot < static_cast<int64_t>(totalInputs) &&
            !usedByNonSyntheticResult[slot]) {
          isSyntheticSlot[slot] = true;
        }
      }

      SmallVector<Value> realInputs;
      SmallVector<int64_t> oldToRealSlot(totalInputs, -1);
      for (unsigned i = 0; i < totalInputs; ++i) {
        if (!isSyntheticSlot[i]) {
          oldToRealSlot[i] = static_cast<int64_t>(realInputs.size());
          realInputs.push_back(mosaicCall.getInputs()[i]);
        }
      }

      SmallVector<Attribute> realAliases;
      SmallVector<int64_t> realResultToOperand(numResults, -1);
      if (ArrayAttr aliasesAttr = mosaicCall.getOutputOperandAliasesAttr()) {
        for (Attribute attr : aliasesAttr) {
          auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
          ArrayRef<int64_t> outIdxs = alias.getOutputTupleIndices();
          unsigned resIdx =
              (numResults == 1 || outIdxs.empty()) ? 0 : outIdxs[0];
          int64_t oldSlot = alias.getOperandIndex();
          if (resIdx < numResults && oldSlot >= 0 &&
              oldSlot < static_cast<int64_t>(totalInputs) &&
              !isSyntheticSlot[oldSlot]) {
            int64_t newSlot = oldToRealSlot[oldSlot];
            realResultToOperand[resIdx] = newSlot;
            realAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
                ctx, alias.getOutputTupleIndices(), newSlot,
                alias.getOperandTupleIndices()));
          }
        }
      }

      auto gx = getIntFromConstant(mosaicCall.getGridx());
      auto gy = getIntFromConstant(mosaicCall.getGridy());
      auto gz = getIntFromConstant(mosaicCall.getGridz());
      if (!gx || !gy || !gz) {
        mosaicCall.emitError(
            "Dynamic grid dimensions are not supported for Mosaic lowering");
        anyFailed = true;
        continue;
      }

      int64_t numScratch = 0;
      if (auto scratchAttr =
              entryFn->getAttrOfType<IntegerAttr>("scratch_operands")) {
        numScratch = scratchAttr.getInt();
      }

      unsigned numGrid = 0;
      if (auto boundsAttr =
              entryFn->getAttrOfType<DenseI64ArrayAttr>("iteration_bounds")) {
        numGrid = boundsAttr.size();
      } else {
        // Infer leading i32 grid index arguments for hand-authored kernels
        // without an explicit `iteration_bounds` attribute.
        unsigned leadingI32 = 0;
        while (
            leadingI32 < entryFn.getNumArguments() && leadingI32 < 3 &&
            entryFn.getArgument(leadingI32).getType().isSignlessInteger(32)) {
          ++leadingI32;
        }
        int64_t rem = static_cast<int64_t>(entryFn.getNumArguments()) -
                      static_cast<int64_t>(leadingI32) - numScratch;
        if (leadingI32 > 0 &&
            (rem == static_cast<int64_t>(realInputs.size()) ||
             rem == static_cast<int64_t>(realInputs.size() + numResults))) {
          numGrid = leadingI32;
          uint64_t callGrid[3] = {*gx, *gy, *gz};
          SmallVector<int64_t> inferredBounds;
          SmallVector<Attribute> inferredSem;
          for (unsigned g = 0; g < numGrid; ++g) {
            uint64_t bound = callGrid[g];
            if (g + 1 == numGrid) {
              for (unsigned rest = g + 1; rest < 3; ++rest)
                bound *= callGrid[rest];
            }
            inferredBounds.push_back(static_cast<int64_t>(bound));
            inferredSem.push_back(tpu::DimensionSemanticsAttr::get(
                ctx, tpu::DimensionSemantics::parallel));
          }
          entryFn->setAttr("iteration_bounds",
                           builder.getDenseI64ArrayAttr(inferredBounds));
          if (!entryFn->hasAttr("dimension_semantics"))
            entryFn->setAttr("dimension_semantics",
                             builder.getArrayAttr(inferredSem));
        }
      }

      int64_t numBufferArgs = static_cast<int64_t>(entryFn.getNumArguments()) -
                              static_cast<int64_t>(numGrid) - numScratch;
      int64_t expectedExplicit =
          static_cast<int64_t>(realInputs.size() + numResults);
      int64_t expectedInPlace = static_cast<int64_t>(realInputs.size());

      if (numBufferArgs == expectedExplicit) {
        // Entry function already declares explicit output memref arguments.
      } else if (numBufferArgs == expectedInPlace && numResults > 0) {
        // Native `MosaicExt` in-place kernel: append one explicit output memref
        // argument per result so `@main` matches XLA's `(ins..., outs...)` ABI.
        int64_t scalarPrefetch = 0;
        if (auto spAttr =
                entryFn->getAttrOfType<IntegerAttr>("scalar_prefetch")) {
          scalarPrefetch = spAttr.getInt();
        }
        SmallVector<Attribute> updatedWindowParams;
        ArrayAttr origWindowParams =
            entryFn->getAttrOfType<ArrayAttr>("window_params");
        bool updateWindowParams =
            origWindowParams && static_cast<int64_t>(origWindowParams.size()) ==
                                    expectedInPlace - scalarPrefetch;
        if (updateWindowParams) {
          updatedWindowParams.append(origWindowParams.begin(),
                                     origWindowParams.end());
        }

        bool argRewriteFailed = false;
        for (unsigned r = 0; r < numResults; ++r) {
          int64_t aliasIdx = realResultToOperand[r];
          if (aliasIdx < 0 ||
              aliasIdx >= static_cast<int64_t>(realInputs.size())) {
            mosaicCall.emitError() << "In-place Mosaic kernel result #" << r
                                   << " does not alias a valid input operand";
            argRewriteFailed = true;
            break;
          }
          unsigned inArgPos = numGrid + static_cast<unsigned>(aliasIdx);
          Value inArg = entryFn.getArgument(inArgPos);
          DictionaryAttr inArgAttrs = entryFn.getArgAttrDict(inArgPos);
          bool needsCopy = hasReadOrAtomicAdd(inArg);

          unsigned outArgPos =
              numGrid + static_cast<unsigned>(realInputs.size()) + r;
          if (failed(entryFn.insertArgument(outArgPos, inArg.getType(),
                                            inArgAttrs, mosaicCall.getLoc()))) {
            mosaicCall.emitError("Failed to append Mosaic output argument");
            argRewriteFailed = true;
            break;
          }
          Value outArg = entryFn.getArgument(outArgPos);
          inArg.replaceAllUsesWith(outArg);

          if (needsCopy) {
            if (failed(emitInitialBufferCopy(builder, mosaicCall.getLoc(),
                                             entryFn, numGrid, inArg,
                                             outArg))) {
              mosaicCall.emitError("Failed to emit initial VMEM copy for "
                                   "in-place Mosaic output");
              argRewriteFailed = true;
              break;
            }
          }

          if (updateWindowParams) {
            int64_t wpIdx = aliasIdx - scalarPrefetch;
            if (wpIdx >= 0 &&
                wpIdx < static_cast<int64_t>(origWindowParams.size())) {
              updatedWindowParams.push_back(origWindowParams[wpIdx]);
            } else {
              updatedWindowParams.push_back(builder.getDictionaryAttr({}));
            }
          }
        }
        if (argRewriteFailed) {
          anyFailed = true;
          continue;
        }
        if (updateWindowParams) {
          entryFn->setAttr("window_params",
                           builder.getArrayAttr(updatedWindowParams));
        }
      } else {
        mosaicCall.emitError()
            << "Unexpected number of Mosaic kernel buffer arguments: got "
            << numBufferArgs << ", expected " << expectedExplicit << " (or "
            << expectedInPlace << " for in-place MosaicExt kernels)";
        anyFailed = true;
        continue;
      }

      // Inspect core type and chip communication before `mosaic-serde` mangles
      // `func.func` and `tpu.*` operations into `stable_mosaic.*`.
      bool isSparseCore = hasSparseCoreFunc(*kernelMod);
      auto [hasComm, _] = tpu::mightCommunicateBetweenChips(*kernelMod);

      int serVersion = target_version >= 0
                           ? target_version
                           : mlir::tpu::MosaicSerdePass::kVersion;
      {
        bool prevAllowUnregistered = ctx->allowsUnregisteredDialects();
        ctx->allowUnregisteredDialects(true);
        llvm::scope_exit restoreUnregistered(
            [&]() { ctx->allowUnregisteredDialects(prevAllowUnregistered); });

        PassManager pm(ctx);
        pm.enableVerifier(false);
        pm.addPass(mlir::tpu::createMosaicSerdePass(
            mlir::tpu::MosaicSerdePassOptions{/*serialize=*/true,
                                              /*target_version=*/serVersion}));
        if (failed(pm.run(*kernelMod))) {
          mosaicCall.emitError("Failed to run mosaic-serde serialization pass");
          anyFailed = true;
          continue;
        }
      }

      std::string bytecode;
      llvm::raw_string_ostream bcStream(bytecode);
      BytecodeWriterConfig writerConfig;
      writerConfig.setDesiredBytecodeVersion(0);
      if (failed(writeBytecodeToFile(*kernelMod, bcStream, writerConfig))) {
        mosaicCall.emitError("Failed to serialize Mosaic module bytecode");
        anyFailed = true;
        continue;
      }
      bcStream.flush();
      std::string b64Body = llvm::encodeBase64(bytecode);

      llvm::json::Object rootObj;
      StringRef existingConfig = mosaicCall.getBackendConfig();
      if (!existingConfig.empty()) {
        llvm::Expected<llvm::json::Value> parsedConfig =
            llvm::json::parse(existingConfig);
        if (!parsedConfig || !parsedConfig->getAsObject()) {
          if (!parsedConfig)
            llvm::consumeError(parsedConfig.takeError());
          mosaicCall.emitError(
              "Expected JSON object in MosaicCallOp backend_config");
          anyFailed = true;
          continue;
        }
        rootObj = std::move(*parsedConfig->getAsObject());
      }

      llvm::json::Object *ccObj = rootObj.getObject("custom_call_config");
      if (!ccObj) {
        rootObj["custom_call_config"] = llvm::json::Object{};
        ccObj = rootObj.getObject("custom_call_config");
      }
      (*ccObj)["body"] = b64Body;
      if (!ccObj->get("serialization_format"))
        (*ccObj)["serialization_format"] = 1;
      if (!isSparseCore && !ccObj->get("needs_layout_passes"))
        (*ccObj)["needs_layout_passes"] = true;
      if (hasComm && !ccObj->get("has_communication"))
        (*ccObj)["has_communication"] = true;

      if (isSparseCore) {
        if (!rootObj.get("device_type"))
          rootObj["device_type"] = "DEVICE_TYPE_SPARSECORE";
        if (!rootObj.get("sparse_core_config")) {
          rootObj["sparse_core_config"] =
              llvm::json::Object{{"tiling", "COMPACT"}};
        }
      }

      std::string jsonBackendConfig;
      llvm::raw_string_ostream jsonStream(jsonBackendConfig);
      jsonStream << llvm::json::Value(std::move(rootObj));
      jsonStream.flush();

      builder.setInsertionPoint(mosaicCall);
      auto customCall = stablehlo::CustomCallOp::create(
          builder, mosaicCall.getLoc(), mosaicCall.getResultTypes(),
          realInputs);
      customCall.setCallTargetName("tpu_custom_call");
      customCall.setBackendConfigAttr(builder.getStringAttr(jsonBackendConfig));
      customCall.setApiVersion(
          ::mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL);

      if (Attribute layoutsAttr = mosaicCall.getOperandLayoutsAttr()) {
        auto origLayouts = cast<ArrayAttr>(layoutsAttr);
        if (origLayouts.size() != totalInputs) {
          mosaicCall.emitError("operand_layouts size does not match inputs");
          anyFailed = true;
          continue;
        }
        SmallVector<Attribute> realOperandLayouts;
        for (unsigned i = 0; i < totalInputs; ++i) {
          if (!isSyntheticSlot[i])
            realOperandLayouts.push_back(origLayouts[i]);
        }
        customCall.setOperandLayoutsAttr(
            builder.getArrayAttr(realOperandLayouts));

        if (savedResultLayouts) {
          customCall.setResultLayoutsAttr(savedResultLayouts);
        } else {
          SmallVector<Attribute> resultLayouts;
          for (unsigned r = 0; r < numResults; ++r) {
            int64_t slot = resultAliasSlot[r];
            if (slot >= 0 && slot < static_cast<int64_t>(origLayouts.size())) {
              resultLayouts.push_back(origLayouts[slot]);
            } else {
              auto resTy =
                  cast<RankedTensorType>(mosaicCall.getResult(r).getType());
              SmallVector<int64_t> minorToMajor;
              for (int64_t d = resTy.getRank() - 1; d >= 0; --d)
                minorToMajor.push_back(d);
              resultLayouts.push_back(builder.getIndexTensorAttr(minorToMajor));
            }
          }
          customCall.setResultLayoutsAttr(builder.getArrayAttr(resultLayouts));
        }
      } else if (savedResultLayouts) {
        customCall.setResultLayoutsAttr(savedResultLayouts);
      }

      if (!realAliases.empty() || mosaicCall.getOutputOperandAliasesAttr()) {
        customCall.setOutputOperandAliasesAttr(
            builder.getArrayAttr(realAliases));
      }

      if (!mosaicCall.getXlaSideEffectFreeAttr())
        customCall.setHasSideEffect(true);

      if (emitKernelNameAttr)
        customCall->setAttr("kernel_name", builder.getStringAttr(kernelName));

      for (NamedAttribute namedAttr : mosaicCall->getAttrs()) {
        if (!isMosaicCallInherentAttr(namedAttr.getName().strref()))
          customCall->setAttr(namedAttr.getName(), namedAttr.getValue());
      }

      mosaicCall.replaceAllUsesWith(customCall.getResults());
      mosaicCall.erase();
    }

    if (anyFailed) {
      signalPassFailure();
      return;
    }

    for (mosaic_ext::MosaicModuleOp mosaicMod : candidateModules) {
      if (SymbolTable::symbolKnownUseEmpty(mosaicMod, modOp))
        mosaicMod.erase();
    }
  }
};

} // end anonymous namespace
