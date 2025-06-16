//===- enzymemlir-opt.cpp - The enzymemlir-opt driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'enzymemlir-opt' tool, which is the enzyme analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

extern "C" std::string runLLVMToMLIRRoundTrip(std::string input) {
  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  auto llvmModule =
      llvm::parseIR(llvm::MemoryBufferRef(input, "conversion"), Err, Context);
  if (!llvmModule) {
    std::string err_str;
    llvm::raw_string_ostream err_stream(err_str);
    Err.print(/*ProgName=*/"LLVMToMLIR", err_stream);
    err_stream.flush();
    exit(1);
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::DialectRegistry registry;
  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);
  mlir::enzyme::registerInterfaces(registry);
  mlir::enzyme::initializePasses();

  mlir::MLIRContext context(registry);
  auto mod = mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                           /*emitExpensiveWarnings*/ false,
                                           /*dropDICompositeElements*/ false);
  if (!mod) {
    exit(1);
  }

  llvm::errs() << " mod: " << *mod << "\n";

  using namespace llvm;
  using namespace mlir;
  std::string pass_pipeline =
      "inline{default-pipeline=canonicalize "
      "max-iterations=4},sroa-wrappers{set_private=false},gpu-launch-"
      "recognition,canonicalize,libdevice-funcs-raise,canonicalize,parallel-lower{wrapParallelOps=true},llvm-to-"
      "memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,"
      "canonicalize,polygeist-mem2reg,canonicalize,enzyme-lift-cf-to-scf,"
      "canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,"
      "canonicalize,affine-cfg,canonicalize,"
      "func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,"
      "canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,"
      "affine-cfg,canonicalize,llvm-to-affine-access,canonicalize,func.func("
      "affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-"
      "affine-to-stablehlo{prefer_while_raising=false "
      "dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true},"
      "symbol-dce,convert-parallel-to-gpu1,gpu-kernel-outlining,canonicalize,"
      "convert-parallel-to-gpu2,lower-affine,convert-polygeist-to-llvm,strip-"
      "gpu-info,gpu-"
      "module-to-binary";
  if (auto pipe2 = getenv("OVERRIDE_PASS_PIPELINE")) {
    pass_pipeline = pipe2;
  }
  mlir::PassManager pm(mod->getContext());
  std::string error_message;
  llvm::raw_string_ostream error_stream(error_message);
  mlir::LogicalResult result =
      mlir::parsePassPipeline(pass_pipeline, pm, error_stream);
  if (mlir::failed(result)) {
    llvm::errs() << " failed to parse pass pipeline: " << error_message << "\n";
    exit(2);
  }

  DiagnosticEngine &engine = mod->getContext()->getDiagEngine();
  error_stream << "Pipeline failed:\n";
  DiagnosticEngine::HandlerID id =
      engine.registerHandler([&](Diagnostic &diag) -> LogicalResult {
        error_stream << diag << "\n";
        return failure();
      });
  if (!mlir::succeeded(pm.run(cast<mlir::ModuleOp>(*mod)))) {
    llvm::errs() << error_stream.str() << "\n";
    return "";
  }

  llvm::LLVMContext llvmContext;
  auto outModule = translateModuleToLLVMIR(*mod, llvmContext);

  if (auto F = outModule->getFunction("mgpuModuleLoad")) {
    for (auto U : llvm::make_early_inc_range(F->users())) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        if (GlobalVariable *glob =
                dyn_cast<GlobalVariable>(CI->getArgOperand(0))) {
          GlobalVariable *newMod = nullptr;
          for (auto U2 : llvm::make_early_inc_range(CI->users())) {
            auto ST = cast<StoreInst>(U2);
            newMod = cast<GlobalVariable>(ST->getPointerOperand());
            ST->eraseFromParent();
          }
          CI->eraseFromParent();
          assert(newMod);
          for (auto U : llvm::make_early_inc_range(newMod->users())) {
            for (auto U2 : llvm::make_early_inc_range(U->users())) {
              cast<Instruction>(U2)->eraseFromParent();
            }
            cast<Instruction>(U)->eraseFromParent();
          }
          newMod->eraseFromParent();
          auto oldName = (glob->getName().substr(0, glob->getName().size() -
                                                        strlen("_binary")) +
                          "_gpubin_cst")
                             .str();
          llvm::errs() << "oldName: " << oldName << "\n";
          outModule->dump();
          auto oldG = outModule->getGlobalVariable(oldName, true);
          assert(oldG);
          oldG->replaceAllUsesWith(glob);
          oldG->eraseFromParent();
          break;
        }
      }
    }
  }
  std::string res;
  llvm::raw_string_ostream ss(res);
  ss << *outModule;

  return res;
}
