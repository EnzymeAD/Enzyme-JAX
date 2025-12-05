//===- InlineGPULaunchFuncs.cpp - SPMD Pass -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the pass InlineGPULaunchFuncs which inlines GPU kernels
// into the calling launchFunc operation. 
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/IRMapping.h"


#define DEBUG_TYPE "inline-gpu-launch-funcs"
#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::enzyme  {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_INLINEGPULAUNCHFUNCS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"


namespace {

//===----------------------------------------------------------------------===//
struct InlineGPULaunchFuncs : impl::InlineGPULaunchFuncsBase<InlineGPULaunchFuncs> {
  using InlineGPULaunchFuncsBase::InlineGPULaunchFuncsBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Create a SymbolTable for the top-level ModuleOp
    SymbolTable moduleSymbolTable(moduleOp);

    // Lookup the gpu.module in the module symbol table
    gpu::GPUModuleOp gpuModuleOp = moduleSymbolTable.lookup<gpu::GPUModuleOp>("__mlir_gpu_module");
    if (!gpuModuleOp) {
      LLVM_DEBUG(DBGS() << "No gpu.module found with symbol name '__mlir_gpu_module'.\n");
      return;
    }

    LLVM_DEBUG(DBGS() << "Found gpu.module '__mlir_gpu_module'.\n");

    // Create a SymbolTable for the found gpu.module
    SymbolTable gpuSymbolTable(gpuModuleOp);

    // Collect all operations before modifying the IR
    struct InliningInfo {
      gpu::LaunchFuncOp launchFuncOp;
      LLVM::LLVMFuncOp calleeFuncOp;
      bool isExternal;
    };
    SmallVector<InliningInfo, 8> toProcess;

    moduleOp.walk([&](gpu::LaunchFuncOp launchFuncOp) {
      // Get the target function name
      StringRef funcName = launchFuncOp.getKernelName();

      LLVM_DEBUG(DBGS() << "Processing gpu.launch_func: " << funcName << "\n");

      // Lookup function within the gpu.module's symbol table
      Operation *calleeFuncOpr = gpuSymbolTable.lookup(funcName);
      if (!calleeFuncOpr) {
        LLVM_DEBUG(DBGS() << "Skipping inlining: Function " << funcName
                                << " not found in module.\n");
        return;
      }
      auto calleeFuncOp = dyn_cast<LLVM::LLVMFuncOp>(calleeFuncOpr);
      if (!calleeFuncOp) {
        LLVM_DEBUG(DBGS() << "Skipping inlining: Found " << funcName
                                << " symbol not a function.\n");
        return;
      }

      bool isExternal = calleeFuncOp.isExternal();
      // Store information for later inlining
      toProcess.push_back({launchFuncOp, calleeFuncOp, isExternal});
    });

   
    // Process inlining and external calls
    OpBuilder builder(moduleOp.getContext());
    InlinerInterface inliner(moduleOp.getContext());
    
    for (auto &entry : toProcess) {
      auto launchFuncOp = entry.launchFuncOp;
      auto calleeFuncOp = entry.calleeFuncOp;

      if (entry.isExternal) {
        // Case: External function -> Replace with LLVM::CallOp
        LLVM_DEBUG(DBGS() << "Replacing gpu.launch_func with LLVM::CallOp for external function: "
                          << calleeFuncOp.getName() << "\n");

        // Create the call operation
        builder.create<LLVM::CallOp>(
            launchFuncOp.getLoc(), calleeFuncOp.getFunctionType(),
            calleeFuncOp.getSymName(), launchFuncOp.getKernelOperands());
      } 
      else {
        // Case: Internal function -> Inline
        LLVM_DEBUG(DBGS() << "Inlining function: " << calleeFuncOp.getName() << "\n");

        Operation *inlinePoint = launchFuncOp.getOperation();
        SmallVector<Value, 0> resultsToReplace; // Always empty since CUDA kernels return void

        // Prepare the region to inline
        Region *calleeRegion = &calleeFuncOp.getBody();
        IRMapping mapper;

        // Clone callback: clones each operation using the builder
        auto cloneCallback = [](OpBuilder &builder, Region *region,
                                Block *destBlock, Block *beforeBlock,
                                IRMapping &mapper, bool) {
            for (Block &block : *region) {
                for (Operation &op : block) {
                    builder.clone(op, mapper);
                }
            }
        };
        LogicalResult inlineResult = inlineRegion(
            inliner,
            cloneCallback,       // clone callback required
            calleeRegion,        // region to inline
            inlinePoint, // insertion point operation
            launchFuncOp.getKernelOperands(), // operands for region arguments
            resultsToReplace,                  // results to replace (empty if void)
            launchFuncOp.getLoc(),   // optional location
            true                 // whether to clone the inlined region
        );
        if (failed(inlineResult)) {
          LLVM_DEBUG(DBGS() << "Inlining failed for function: " << calleeFuncOp.getName() << "\n");
          continue;
        }
        LLVM_DEBUG(DBGS() << "Successfully inlined\n");
      }
      launchFuncOp.erase();
    }

    // Iterate over the gpu.module region and collect external functions
    SmallVector<LLVM::LLVMFuncOp, 8> externalFuncsToAdd;
    SmallVector<LLVM::LLVMFuncOp, 8> nonExternalFuncsToAdd;
    gpuModuleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
      // Check if the function has external linkage
      if (funcOp.isExternal()) {
        externalFuncsToAdd.push_back(funcOp); // Collect the external function
        LLVM_DEBUG(DBGS() << "Found external function in GPU module: " << funcOp.getName() << "\n");
      }
      else {
        nonExternalFuncsToAdd.push_back(funcOp); // Collect the non-external function
        LLVM_DEBUG(DBGS() << "Found non-external function in GPU module: " << funcOp.getName() << "\n");
      }
    });

    // Now, add external function symbols to the top of the moduleOp's region
    builder.setInsertionPointToStart(moduleOp.getBody());  // Insert at the beginning of the region

    for (auto &funcOp : externalFuncsToAdd) {
      // Ensure the function exists in moduleOp (move it from gpu.module if needed)
      if (!moduleSymbolTable.lookup<LLVM::LLVMFuncOp>(funcOp.getSymName())) {
        // Insert the function definition at the beginning of the module
        //TODO: some attributes are wrong when cloning e.g. targets etc.
        builder.clone(*funcOp.getOperation());  // clone function at the beginning of moduleOp's body
        // Ensure the function is registered in the module's symbol table
        // moduleSymbolTable.insert(funcOp);
        LLVM_DEBUG(DBGS() << "Added external function " << funcOp.getName() << " to moduleOp.\n");
      }
    }
    LLVM_DEBUG(DBGS() << "Finished processing external functions.\n");

    // Now, add non-external function symbols to the top of the moduleOp's region
    for (auto &funcOp : nonExternalFuncsToAdd) {
      // Ensure the function exists in moduleOp (move it from gpu.module if needed)
      if (!moduleSymbolTable.lookup<LLVM::LLVMFuncOp>(funcOp.getSymName())) {
        // Insert the function definition at the beginning of the module
        builder.clone(*funcOp.getOperation());  // clone function at the beginning of moduleOp's body
        // Ensure the function is registered in the module's symbol table
        // moduleSymbolTable.insert(funcOp);
        LLVM_DEBUG(DBGS() << "Added non-external function " << funcOp.getName() << " to moduleOp.\n");
      }
    }
    LLVM_DEBUG(DBGS() << "Finished processing non-external functions.\n");
    gpuModuleOp.erase();  // Remove the gpu.module after processing
    LLVM_DEBUG(DBGS() << "Removed gpu.module from moduleOp.\n");
    LLVM_DEBUG(DBGS()  << "Finished processing module.\n");
  }
};
//===----------------------------------------------------------------------===//

} // namespace
} // namespace mlir::enyzme
