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

#include "Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Passes/Passes.h"
#include "Implementations/XLADerivatives.h"
#include "Passes/Passes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IRReader/IRReader.h"

#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/tests/CheckOps.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
void registerGenerateApplyPatternsPass();
void registerRemoveTransformPass();
} // namespace enzyme
} // namespace mlir

void prepareRegistry(mlir::DialectRegistry &registry);

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
      "recognition,canonicalize,parallel-lower{wrapParallelOps=true},llvm-to-"
      "memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,"
      "canonicalize,polygeist-mem2reg,canonicalize,enzyme-lift-cf-to-scf,"
      "canonicalize,func.func(canonicalize-loops),canonicalize-scf-for,"
      "canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,"
      "func.func(canonicalize-loops),canonicalize,llvm-to-affine-access,"
      "canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,"
      "affine-cfg,canonicalize,llvm-to-affine-access,canonicalize,func.func("
      "affine-loop-invariant-code-motion),canonicalize,sort-memory,raise-"
      "affine-to-stablehlo{prefer_while_raising=false "
      "dump_failed_lockstep=true},canonicalize,arith-raise{stablehlo=true},"
      "symbol-dce";
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
  }

  std::string res;
  llvm::raw_string_ostream ss(res);
  ss << *mod;

  return res;
}
