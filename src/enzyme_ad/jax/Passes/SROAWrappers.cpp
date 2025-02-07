//===- SROAWrappers.cpp - Run SROA on ABI conversion wrappers --------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/SROA.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"

#include <optional>

#define DEBUG_TYPE "sroa-wrappers"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SROAWRAPPERSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir::enzyme;

namespace {

struct SROAWrappersPass
    : public mlir::enzyme::impl::SROAWrappersPassBase<SROAWrappersPass> {
  using SROAWrappersPassBase::SROAWrappersPassBase;

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    mlir::OpBuilder b(m);

    auto mToTranslate = b.cloneWithoutRegions(m);

    llvm::SmallVector<mlir::Operation *> toOpt;
    for (auto [oldRegion, newRegion] :
         llvm::zip(m->getRegions(), mToTranslate->getRegions())) {
      for (auto &oldBlock : oldRegion.getBlocks()) {
        assert(oldBlock.getNumArguments() == 0);
        b.createBlock(&newRegion, newRegion.end());
        for (auto &op : oldBlock) {
          // FIXME in reality, this check should be whether the entirety
          // (all nested ops with all (transitively) used symbol as well) of
          // the op is translatable to llvm ir.
          // FIXME we also need to mark them `used` so the llvm optimizer
          // does not get rid of them.
          if (llvm::isa<mlir::LLVM::LLVMDialect>(op.getDialect())) {
            // There should be no need for mapping because all top level
            // operations in the module should be isolated from above
            b.clone(op);
            toOpt.push_back(&op);
          }
        }
      }
    }

    mlir::PassManager pm(mToTranslate.getContext());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertNVVMToLLVMPass());
    auto subres = pm.run(mToTranslate);
    if (!subres.succeeded()) {
      return;
    }

    llvm::LLVMContext llvmCtx;
    auto llvmModule = mlir::translateModuleToLLVMIR(mToTranslate, llvmCtx);

    if (dump_prellvm)
      llvm::errs() << "sroa pre llvm\n" << *llvmModule << "\n";
    {
      using namespace llvm;
      PipelineTuningOptions PTO;
      PTO.LoopUnrolling = false;
      PTO.LoopInterleaving = false;
      PTO.LoopVectorization = false;
      PTO.SLPVectorization = false;
      PTO.MergeFunctions = false;
      PTO.CallGraphProfile = false;
      PTO.UnifiedLTO = false;

      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;

      PassInstrumentationCallbacks PIC;
      PassBuilder PB(nullptr, PTO, std::nullopt, nullptr);

      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      ModulePassManager MPM;
      FunctionPassManager FPM;
      if (sroa)
        MPM.addPass(createModuleToFunctionPassAdaptor(
            SROAPass(SROAOptions::ModifyCFG)));
      if (instcombine)
        MPM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
      if (instsimplify)
        MPM.addPass(createModuleToFunctionPassAdaptor(InstSimplifyPass()));
      if (attributor)
        MPM.addPass(llvm::AttributorPass());
      MPM.run(*llvmModule, MAM);
    }
    if (dump_postllvm)
      llvm::errs() << "sroa post_llvm\n" << *llvmModule << "\n";
    auto translatedFromLLVMIR = mlir::translateLLVMIRToModule(
        std::move(llvmModule), m->getContext(), /*emitExpensiveWarnings*/ true,
        /*dropDICompositeTypeElements*/ false, /*loadAllDialects*/ false);

    b.setInsertionPoint(m);
    mlir::ModuleOp newM = *translatedFromLLVMIR;

    for (auto op : toOpt) {
      op->erase();
    }
    for (auto [oldRegion, newRegion] :
         llvm::zip(m->getRegions(), newM->getRegions())) {
      for (auto [oldBlock, newBlock] :
           llvm::zip(oldRegion.getBlocks(), newRegion.getBlocks())) {
        b.setInsertionPointToEnd(&oldBlock);
        for (auto &op : newBlock) {
          assert(op.hasTrait<mlir::OpTrait::IsIsolatedFromAbove>());
          assert(llvm::isa<mlir::LLVM::LLVMDialect>(op.getDialect()));
          // There should be no need for mapping because all top level
          // operations in the module should be isolated from above
          b.clone(op);
        }
      }
    }

    mToTranslate->erase();
  }
};

} // end anonymous namespace
