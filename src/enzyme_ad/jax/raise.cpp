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

#include "src/enzyme_ad/jax/raise.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"

#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "llvm/Support/TargetSelect.h"
#include <system_error>

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

extern "C" std::string runLLVMToMLIRRoundTrip(std::string input,
                                              std::string outfile,
                                              std::string backend,
                                              std::string library,
                                              MLIRRoundTripOptions *options) {
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
  if (options->lowerInvoke) {
    llvm::PassBuilder PB;
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::LowerInvokePass());
    // The landing pads the lowering leaves behind are unreachable, and what
    // reads the module next has no more use for them than it had for the edge.
    FPM.addPass(llvm::SimplifyCFGPass());

    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.run(*llvmModule, MAM);
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
      "max-iterations=4},sroa-wrappers{set_private=false attributor=false},"
      "lift-tessera-annotations,parse-optimization-rules,"
      "gpu-launch-recognition{backend=";
  pass_pipeline += backend;
  pass_pipeline += "}";
  pass_pipeline += ","
      "canonicalize-parallel,libdevice-funcs-raise,restore-preserve-nvvm,canonicalize-parallel,"
      "inline-enzyme-regions,symbol-dce,";
  
  if (backend == "cpu")
    pass_pipeline += "parallel-lower{wrapParallelOps=false},";
  else
    pass_pipeline += "parallel-lower{wrapParallelOps=true},";
  pass_pipeline += "llvm-to-"
      "memref-access,polygeist-mem2reg,canonicalize-parallel,convert-llvm-to-cf,"
      "canonicalize-parallel,polygeist-mem2reg,canonicalize-parallel,enzyme-lift-cf-to-scf,"
      "canonicalize-parallel,"
      "func.func(canonicalize-loops),"
      "llvm.func(canonicalize-loops),"
      "canonicalize-scf-for,"
      "canonicalize-parallel,affine-cfg,canonicalize-parallel,"
      "func.func(canonicalize-loops),"
      "llvm.func(canonicalize-loops),"
      "canonicalize-parallel,llvm-to-affine-access,"
      "canonicalize-parallel,delinearize-indexing,canonicalize-parallel,simplify-affine-exprs,"
      "affine-cfg,canonicalize-parallel,llvm-to-affine-access,canonicalize-parallel,"
      "func.func(affine-loop-invariant-code-motion),"
      "canonicalize-parallel,sort-memory,llvm-to-tessera,tessera-apply-pdl,tessera-to-llvm,";
  if (StringRef(backend).starts_with("xla")) {
      pass_pipeline += "func.func(kernelcast),raise-affine-to-stablehlo{prefer_while_raising=false "
      "dump_failed_lockstep=true},canonicalize-parallel,arith-raise{stablehlo=true},"
      "symbol-dce";
      if (outfile.size() && getenv("EXPORT_REACTANT")) {
        pass_pipeline += ",print{filename="+outfile+".mlir}";
      }
      pass_pipeline += ",lower-aligned-affine-accesses,lower-affine";
      if (getenv("REACTANT_OMP")) {
        pass_pipeline += ",convert-scf-to-openmp,";
      } else {
        pass_pipeline += ",parallel-serialization,";
      }
      pass_pipeline += "canonicalize-parallel,hoist-allocas,convert-polygeist-to-llvm{backend=";
      pass_pipeline += backend;
      pass_pipeline += "}";
  } else {
      if (outfile.size() && getenv("EXPORT_REACTANT")) {
        pass_pipeline += "print{filename="+outfile+".mlir},";
      }
      pass_pipeline += "symbol-dce,raise-llvm-ext,outline-enzyme-regions,";
      if (options->preADLowerAffine)
        pass_pipeline += "lower-aligned-affine-accesses,lower-affine,";

      // A checkpointed loop must not capture both a value and a view of it:
      // it would snapshot the same buffer twice. Has to precede `enzyme`,
      // which is what reads the captures.
      pass_pipeline += "sink-checkpoint-views,";

      pass_pipeline += "enzyme{";
      if (options->dataflow)
        pass_pipeline += "dataflow ";
      if (options->markReadonly)
        pass_pipeline += "markReadonly ";
      // Each generated derivative function is cleaned of enzyme cache ops
      // the moment it is created: nested differentiation hands the outer AD
      // the inner function as input, and enzyme.push/pop have no derivative
      // of their own.
      pass_pipeline += "postpasses=\"canonicalize,";
      if (options->splitMultiResults)
        pass_pipeline += "split-multi-results,";
      pass_pipeline += "remove-unnecessary-enzyme-ops,"
        // binomial checkpointing leaves enzyme.binomial_progress behind; it has
        // no lowering of its own further down, so expand it here.
        "flatten-enzyme-caches,lower-enzyme-binomial-progress,";
      if (options->hoistLoopAllocations)
        pass_pipeline += "hoist-loop-allocations,";
      pass_pipeline += "enzyme-simplify-math\"";
      pass_pipeline += "},"
        // The one module-level survivor: llvm_ext ops also live outside the
        // generated functions the postpasses clean -- a ptr_size_hint sits in
        // the primal that carries the user's marker -- and any left behind
        // fail translation to LLVM IR.
        "lower-llvm-ext,"
        "inline{default-pipeline=canonicalize max-iterations=4},"
        "polygeist-mem2reg,canonicalize-parallel,symbol-dce,"
        // canonicalize-parallel here folds away memref.subview ops before gpu-kernel-outlining
        "canonicalize-parallel,cse";
      if (options->removeAtomics)
        pass_pipeline += ",affine-cfg,remove-atomics";
      if (options->sortBlockMemory)
        pass_pipeline += ",sort-block-memory";
      pass_pipeline += ",lower-aligned-affine-accesses,lower-affine,"
                       "lower-affine-atomic-rmw";
      if (backend == "rocm")
        pass_pipeline += ",convert-cudart-to-hiprt";
      if (backend != "cpu") {
        pass_pipeline += ",convert-parallel-to-gpu1,symbol-dce,gpu-kernel-outlining,canonicalize-parallel,symbol-dce,";
        pass_pipeline += "convert-parallel-to-gpu2{backend=";
        pass_pipeline += backend;
        pass_pipeline += "}";
        pass_pipeline += ",lower-aligned-affine-accesses,lower-affine";
      }
      if (getenv("REACTANT_OMP")) {
        pass_pipeline += ",convert-scf-to-openmp,";
      } else {
	      pass_pipeline += ",parallel-serialization,";
      }
      pass_pipeline += "canonicalize-parallel,hoist-allocas,convert-polygeist-to-llvm{backend=";
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
  // The pass manager verifies the whole module -- a symbol-table walk and
  // dominance check over every operation -- after every pass. Over this
  // pipeline's ~65 passes on a large TU that is a third of the pipeline's
  // wall time, re-proving unchanged exception-handling functions well-formed.
  // Verify once at the end instead (below); options->verifyEach restores the
  // per-pass verification for debugging a miscompile to a pass.
  if (!options->verifyEach)
    pm.enableVerifier(false);
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
        for (auto &note : diag.getNotes())
          error_stream << "  note: " << note << "\n";
        return failure();
      });
  if (!mlir::succeeded(pm.run(cast<mlir::ModuleOp>(*mod)))) {
    llvm::errs() << error_stream.str() << "\n";
    return "";
  }

  // The one verification that still stands guard: malformed IR fails here,
  // with diagnostics, rather than inside the LLVM translator.
  if (!options->verifyEach && mlir::failed(mlir::verify(*mod))) {
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
          if (oldG->hasSection())
            glob->setSection(oldG->getSection());
          glob->setAlignment(oldG->getAlign());
          oldG->replaceAllUsesWith(glob);
          oldG->eraseFromParent();
        }
      }
    }
  }
  // Hand the module back as bitcode. The module crosses this boundary as bytes
  // either way, and bitcode is the cheaper and more faithful spelling of it: on
  // MFEM's dFEM tests it is a fifth the size of the textual form and parses in
  // half the time, and it does not depend on a printer and a parser agreeing
  // about syntax. The reader is llvm::parseIR, which sniffs the bitcode magic
  // and dispatches, so it takes either and no version of it has to be taught
  // this.
  std::string res;
  llvm::raw_string_ostream ss(res);
  llvm::WriteBitcodeToFile(*outModule, ss);

  if (getenv("DEBUG_REACTANT")) {
    llvm::errs() << " final llvm:" << *outModule << "\n";
  }

  return res;
}
