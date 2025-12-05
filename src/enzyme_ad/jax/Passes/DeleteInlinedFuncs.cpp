//===- SimplifyControlFlow.cpp - SPMD Pass -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the pass SimplifyControlFlow which converts all cf.SwitchOps 
// first to scf.IndexSwitchOps and then all scf.IndexSwitchOps to scf.IfOps.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::enzyme  {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_DELETEINLINEDFUNCS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

//pattern to remove unused llvm func ops that arent called in the main function after being inlined
struct LLVMFuncErase : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp llvmfuncOp,
                                PatternRewriter &rewriter) const final {
    //skip main func
    if (llvmfuncOp.getSymName() == "main") { 
      return failure();
    }

    OpBuilder builder(llvmfuncOp);
    ModuleOp moduleOp = llvmfuncOp->getParentOfType<ModuleOp>();

   // look into all functions in module if any of them calls the funcOp possibly to be erased
    for (Operation &opr : moduleOp.getBodyRegion().getOps()) {
      if (auto llvmfuncOp2 = dyn_cast<LLVM::LLVMFuncOp>(&opr)) {
        //look, if symbol uses are given, that all of them are recursive calls
        if (llvmfuncOp2.getSymName() != llvmfuncOp.getSymName()) { 
          if (SymbolTable::symbolKnownUseEmpty(llvmfuncOp.getSymNameAttr(), llvmfuncOp2) == false) {
            return failure();  
          } 
        }
      }
    }

    rewriter.eraseOp(llvmfuncOp);
    return success();   
  }
};

//pattern to remove unused func ops that arent called in the main function after being inlined by polygeist
// struct FuncErase : public OpRewritePattern<func::FuncOp> {
//   using OpRewritePattern<func::FuncOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(func::FuncOp funcOp,
//                                 PatternRewriter &rewriter) const final {
//     OpBuilder builder(funcOp);

//     if (funcOp.getSymName() == "main") {
//       return failure();
//     }

//     ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();

//    // look into all functions in module if any of them calls the funcOp possibly to be erased
//     for (Operation &opr : moduleOp.getBodyRegion().getOps()) {
//       if (auto funcOp2 = dyn_cast<func::FuncOp>(&opr)) {
        
//     //look, if symbol uses are given, that all of them are recursive calls
//         if (funcOp2.getSymName() != funcOp.getSymName()) {
//           if (SymbolTable::symbolKnownUseEmpty(funcOp.getSymNameAttr(), funcOp2) == false) {
//             return failure();  
//           } 
//         }
//       }
//     }

//     rewriter.eraseOp(funcOp);
//     return success();   
//   }
// };


struct DeleteInlinedFuncs : impl::DeleteInlinedFuncsBase<DeleteInlinedFuncs> {
  using DeleteInlinedFuncsBase::DeleteInlinedFuncsBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    RewritePatternSet patterns(&getContext());
    // GreedyRewriteConfig config;
    patterns.insert<LLVMFuncErase>(&getContext());
    (void)applyPatternsGreedily(moduleOp, std::move(patterns)/*, config*/);
  }
};

} // namespace
} // namespace mlir::enyzme
