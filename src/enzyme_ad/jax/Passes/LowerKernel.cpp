//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "lower-kernel"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;
using namespace mlir::enzymexla;
using namespace enzymexla;

using namespace stablehlo;

typedef size_t KernelContext[7];
typedef void XlaCustomCallStatus;

// See API details at https://github.com/openxla/xla/blob/37fb0612d36ac3d08ff984b1d61e4bc4dedf4809/xla/service/hlo.proto#L73
extern "C" void EnzymeGPUCustomCall(void* __restrict__ stream, void** __restrict__ buffers, void** __restrict__ opaqueptr,
                               size_t opaque_len, XlaCustomCallStatus* __restrict__ status) {
  auto ptr = (void(*)(void*, void**)) (opaqueptr[0]);
  //auto ptr = (void(*)(void*, void**, size_t, size_t, size_t, size_t, size_t, size_t)) (opaqueptr[0][0]);

  //size_t gridx = opaqueptr[0][1];
  //size_t gridy = opaqueptr[0][2];
  //size_t gridz = opaqueptr[0][3];

  //size_t blockx = opaqueptr[0][4];
  //size_t blocky = opaqueptr[0][5];
  //size_t blockz = opaqueptr[0][6];

  ptr(stream, buffers);//, gridx, gridy, gridz, blockx, blocky, blockz);
}

void* CompileKernel(SymbolTableCollection &symbolTable, mlir::Location loc, FunctionOpInterface op, bool jit, size_t gridx, size_t gridy, size_t gridz, size_t blockx, size_t blocky, size_t blockz) {

  OpBuilder builder(op);

  auto ptrty = LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Type intys[] = {ptrty, ptrty};
  FunctionType calleeType = builder.getFunctionType(intys, {});

  FunctionType gpuTy = dyn_cast<FunctionType>(op.getFunctionType());
  if (!gpuTy) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(op.getFunctionType())) {
      gpuTy = builder.getFunctionType(lty.getParams(), {});
    } else {
      op.emitError("Require target operand to have functiontype or llvmfunctiontype");
      return nullptr;
    }
  }
 
  auto submod = builder.create<ModuleOp>(loc, "offload");
  submod->setAttr("gpu.container_module", builder.getUnitAttr());
  builder.setInsertionPointToStart(&submod.getBodyRegion().front());
  
  auto gpumod = builder.create<gpu::GPUModuleOp>(loc, "gpumodname");
  builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());
  
  auto gpufunc = builder.create<gpu::GPUFuncOp>(loc, "kernel", gpuTy);
  {
    IRMapping map;
    map.map(op.getArguments(), gpufunc.getArguments());
    op.getFunctionBody().cloneInto(&gpufunc.getBody(), map);
    gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());
   
    auto entry = &gpufunc.getBody().front();
    auto second = entry->getNextNode();
    entry->getOperations().splice(entry->getOperations().end(),
                                  second->getOperations());

    second->erase();

    gpufunc->walk([](LLVM::ReturnOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
    });
    
    gpufunc->walk([](LLVM::UnreachableOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
    });
    
    gpufunc->walk([](func::ReturnOp op) {
        OpBuilder rewriter(op);
        rewriter.create<gpu::ReturnOp>(op.getLoc());
        op.erase();
    });
  
    llvm::errs() << gpufunc << "\n";
  }
  SmallVector<Operation*> tocopy;
  op->walk([&](CallOpInterface cop) {
    if (auto op2 = cop.resolveCallable())
      tocopy.push_back(op2);
  });
  SmallPtrSet<Operation*, 1> done;
  
  builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());
  while (tocopy.size()) {
      auto cur = tocopy.pop_back_val();
      if (done.count(cur)) continue;
      done.insert(cur);
      builder.clone(*cur);
      cur->walk([&](CallOpInterface cop) {
        if (auto op2 = cop.resolveCallable())
          tocopy.push_back(op2);
      });
  }

  builder.setInsertionPointToEnd(&submod.getBodyRegion().front());

  auto func = builder.create<func::FuncOp>(loc, "caller", calleeType);
  
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  mlir::Value stream = entryBlock.getArgument(0);
  auto buffers = entryBlock.getArgument(1);
 
  auto i32 = builder.getIntegerType(32);
  gpu::KernelDim3 gridSize{
      builder.create<arith::ConstantIntOp>(loc, gridx, i32),
      builder.create<arith::ConstantIntOp>(loc, gridy, i32),
      builder.create<arith::ConstantIntOp>(loc, gridz, i32),
  };
  
  gpu::KernelDim3 blockSize{
      builder.create<arith::ConstantIntOp>(loc, blockx, i32),
      builder.create<arith::ConstantIntOp>(loc, blocky, i32),
      builder.create<arith::ConstantIntOp>(loc, blockz, i32),
  };

  SmallVector<mlir::Value> arguments;
  for (auto arg : op.getArguments()) {
      LLVM::GEPArg args[1] = {arg.getArgNumber()};
      auto gep = builder.create<LLVM::GEPOp>(loc, ptrty, ptrty, buffers, args, true);
      auto ld = builder.create<LLVM::LoadOp>(loc, arg.getType(), gep);
      arguments.push_back(ld);
  }
  auto dynshmem = builder.create<arith::ConstantIntOp>(loc, 0, i32);
  stream = builder.create<UnrealizedConversionCastOp>(loc, gpu::AsyncTokenType::get(stream.getContext()), stream)->getResult(0);
  builder.create<gpu::LaunchFuncOp>(
      loc, gpufunc, gridSize, blockSize, dynshmem, arguments, stream.getType(), ValueRange(stream));

  builder.create<mlir::func::ReturnOp>(loc);

  llvm::errs() << func << "\n";
  
  llvm::errs() << submod << "\n";

  if (!jit)
      return nullptr;

  op->emitError() << "JIT compilation of kernels not yet implemented";
  return nullptr;
};

namespace {

struct LowerKernelPass : public LowerKernelPassBase<LowerKernelPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());


    getOperation()->walk([&](KernelCallOp op) {
        mlir::ArrayAttr operand_layouts = op.getOperandLayouts() ? cast<mlir::ArrayAttr>(*op.getOperandLayouts()) : nullptr;
        mlir::ArrayAttr result_layouts = op.getResultLayouts() ? cast<mlir::ArrayAttr>(*op.getResultLayouts()) : nullptr;
        mlir::ArrayAttr output_operand_aliases = op.getOutputOperandAliases();

        KernelContext data;


        auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
        auto fn = cast<FunctionOpInterface>(symbolOp);

        Value vals[6] = {op.getGridx(), op.getGridy(), op.getGridz(), op.getBlockx(), op.getBlocky(), op.getBlockz()};
        for (auto en : llvm::enumerate(vals)) {
          DenseIntElementsAttr stepAttr;
          if (!matchPattern(en.value(), m_Constant(&stepAttr))) {
            op->emitError() << "Cannot lower kernel with a grid/block size which is not a constant integer tensor";
            return;
          }
          if (stepAttr.size() != 1) {
            op->emitError() << "Cannot lower kernel with a grid/block size which is not a constant integer tensor of size 1";
            return;
          }
          auto val = (*stepAttr.begin()).getZExtValue();
          data[1+en.index()] = val;
        }

        // Compiled kernel goes here once ready
        data[0] = (size_t)CompileKernel(symbolTable, op.getLoc(), fn, jit, data[1], data[2], data[3], data[4], data[5], data[6]);

        std::string backendinfo((char*)&data, sizeof(void*));
       
        OpBuilder rewriter(op);
        auto replacement = rewriter.create<stablehlo::CustomCallOp>(op.getLoc(),
            op.getResultTypes(),
            op.getInputs(),
            rewriter.getStringAttr("enzymexla_gpu"),
            /* has_side_effect*/rewriter.getBoolAttr(false),
            /*backend_config*/rewriter.getStringAttr(backendinfo),
            /* api_version*/CustomCallApiVersionAttr::get(
                                 rewriter.getContext(),mlir::stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
            /*calledcomputations*/nullptr,
            operand_layouts,
            result_layouts,
            output_operand_aliases
            );

        op.replaceAllUsesWith(replacement);
        op.erase();
    });

  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createLowerKernelPass() {
  return std::make_unique<LowerKernelPass>();
}
} // namespace enzyme
} // namespace mlir
