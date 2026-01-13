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
#include <system_error>

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

extern "C" std::string runLLVMToMLIRRoundTrip(std::string input,
                                              std::string outfile,
                                              std::string backend,
                                              std::string library) {
  llvm::LLVMContext Context;
  Context.setDiscardValueNames(false);
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

  mlir::OpPrintingFlags flags;
  if (getenv("DEBUG_REACTANT_INFO"))
    flags.enableDebugInfo(true, /*pretty*/ false);
  else
    flags.enableDebugInfo(false, /*pretty*/ false);

  if (auto path = getenv("DEBUG_REACTANT_IMPORTED_MLIR_MOD_PATH")) {
    std::error_code EC;
    llvm::raw_fd_ostream os(path, EC);
    mod->print(os, flags);
  }
  if (getenv("DEBUG_REACTANT")) {
    llvm::errs() << " imported mlir mod: ";
    mod->print(llvm::errs(), flags);
    llvm::errs() << "\n";
  }

  using namespace llvm;
  using namespace mlir;
  // clang-format off
  std::string pass_pipeline =
      "inline{default-pipeline=canonicalize "
      "max-iterations=4},sroa-wrappers{set_private=false attributor=false},gpu-launch-"
      "recognition{backend=";
      pass_pipeline += backend;
      pass_pipeline += "}";
      pass_pipeline += ","
      "canonicalize,libdevice-funcs-raise,canonicalize,symbol-dce,";
  
  if (backend == "cpu")
    pass_pipeline += "parallel-lower{wrapParallelOps=false},";
  else
    pass_pipeline += "parallel-lower{wrapParallelOps=true},";
  pass_pipeline += "llvm-to-"
      "memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,"
      "canonicalize,polygeist-mem2reg,canonicalize,enzyme-lift-cf-to-scf,"
      "canonicalize,"
      "func.func(canonicalize-loops),"
      "llvm.func(canonicalize-loops),"
      "canonicalize-scf-for,"
      "canonicalize,affine-cfg,canonicalize,"
      "func.func(canonicalize-loops),"
      "llvm.func(canonicalize-loops),"
      "canonicalize,llvm-to-affine-access,"
      "canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,"
      "affine-cfg,canonicalize,llvm-to-affine-access,canonicalize,"
      "func.func(affine-loop-invariant-code-motion),"
      "canonicalize,sort-memory,";
  if (StringRef(backend).starts_with("xla")) {
      pass_pipeline += "raise-affine-to-stablehlo{prefer_while_raising=false "
      "dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true},"
      "symbol-dce";
      if (outfile.size() && getenv("EXPORT_REACTANT")) {
        pass_pipeline += ",print{filename="+outfile+".mlir}";
      }
      pass_pipeline += ",lower-affine";
      if (getenv("REACTANT_OMP")) {
        pass_pipeline += ",convert-scf-to-openmp,";
      } else {
        pass_pipeline += ",parallel-serialization,";
      }
      pass_pipeline += "canonicalize,convert-polygeist-to-llvm{backend=";
      pass_pipeline += backend;
      pass_pipeline += "}";
  } else {
      if (outfile.size() && getenv("EXPORT_REACTANT")) {
        pass_pipeline += "print{filename="+outfile+".mlir},";
      }
      pass_pipeline += "symbol-dce,enzyme,remove-unnecessary-enzyme-ops,lower-affine";
      if (backend != "cpu")
	pass_pipeline += ",convert-parallel-to-gpu1,gpu-kernel-outlining,canonicalize,convert-parallel-to-gpu2{backend=";
      pass_pipeline += backend;
      pass_pipeline += "}";
      pass_pipeline += ","
      "lower-affine";
      if (getenv("REACTANT_OMP")) {
        pass_pipeline += ",convert-scf-to-openmp,";
      } else {
	      pass_pipeline += ",parallel-serialization,";
      }
      pass_pipeline += "canonicalize,convert-polygeist-to-llvm{backend=";
      pass_pipeline += backend;
      pass_pipeline += "},strip-"
      "gpu-info,gpu-"
      "module-to-binary";
      if (!library.empty()) {
        pass_pipeline += "{l=";
        pass_pipeline += library;
        pass_pipeline += "}";
      }
  }

  // clang-format on
  if (auto pipe2 = getenv("OVERRIDE_PASS_PIPELINE")) {
    pass_pipeline = pipe2;
  }
  if (getenv("DEBUG_REACTANT")) {
    llvm::errs() << " passes to run: " << pass_pipeline << "\n";
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

  if (getenv("DEBUG_REACTANT")) {
    llvm::errs() << " final mlir mod: ";
    mod->print(llvm::errs(), flags);
    llvm::errs() << "\n";
  }

  llvm::LLVMContext llvmContext;
  llvmContext.setDiscardValueNames(false);
  auto outModule = translateModuleToLLVMIR(*mod, llvmContext);
  if (!outModule) {
    llvm::errs() << "failed to translate MLIR to LLVM IR\n";
    return "";
  }

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
          if (getenv("DEBUG_REACTANT")) {
            llvm::errs() << "oldName: " << oldName << "\n";
            llvm::errs() << " gpumod: " << *outModule << "\n";
          }
          auto oldG = outModule->getGlobalVariable(oldName, true);
          assert(oldG);
          oldG->replaceAllUsesWith(glob);
          oldG->eraseFromParent();
        }
      }
    }
  }
  std::string res;
  llvm::raw_string_ostream ss(res);
  ss << *outModule;

  if (getenv("DEBUG_REACTANT")) {
    llvm::errs() << " final llvm:" << res << "\n";
  }

  return res;
}
