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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "lower-kernel"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERKERNELPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::gpu;
using namespace enzyme;
using namespace mlir::enzymexla;
using namespace enzymexla;

using namespace stablehlo;

bool CompileGPUKernel(SymbolTableCollection &symbolTable, mlir::Location loc,
                      FunctionOpInterface op, size_t gridx, size_t gridy,
                      size_t gridz, size_t blockx, size_t blocky, size_t blockz,
                      size_t shmem, enzymexla::KernelCallOp kcall) {

  OpBuilder builder(op);

  FunctionType gpuTy0 = dyn_cast<FunctionType>(op.getFunctionType());
  if (!gpuTy0) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(op.getFunctionType())) {
      gpuTy0 = builder.getFunctionType(lty.getParams(), {});
    } else {
      op.emitError(
          "Require target operand to have functiontype or llvmfunctiontype");
      return false;
    }
  }
  SmallVector<Type, 1> newParams;
  for (Type p : gpuTy0.getInputs()) {
    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(p)) {
      p = AT.getElementType();
    }
    newParams.push_back(p);
  }
  FunctionType gpuTy = builder.getFunctionType(newParams, {});

  std::string legalName = op.getName().str();
  std::replace(legalName.begin(), legalName.end(), '#', '_');
  std::string gpumodname = "gpumod_" + legalName;

  gpu::GPUModuleOp gpumod = nullptr;
  gpu::GPUFuncOp gpufunc = nullptr;

  for (auto &gop : *op.getOperation()->getBlock()) {
    auto gmod = dyn_cast<gpu::GPUModuleOp>(gop);
    if (!gmod)
      continue;
    if (gmod.getName() == gpumodname) {
      gpumod = gmod;
      break;
    }
  }

  if (gpumod) {
    for (auto &gop : *(gpumod.getBody())) {
      auto gfunc = dyn_cast<gpu::GPUFuncOp>(gop);
      if (!gfunc)
        continue;
      if (gfunc.getName() == legalName) {
        gpufunc = gfunc;
        break;
      }
    }
    assert(gpufunc);
  } else {
    gpumod = builder.create<gpu::GPUModuleOp>(loc, gpumodname);
    builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());

    gpufunc = builder.create<gpu::GPUFuncOp>(loc, legalName, gpuTy);
    {
      auto entry = &gpufunc.getBody().front();
      builder.setInsertionPointToEnd(entry);
      IRMapping map;
      for (auto &&[oldarg, newarg] :
           zip(op.getArguments(), gpufunc.getArguments())) {
        Value newval = newarg;

        if (auto AT = dyn_cast<LLVM::LLVMArrayType>(oldarg.getType())) {
          auto ud =
              builder.create<LLVM::UndefOp>(newarg.getLoc(), oldarg.getType());
          int64_t c0[1] = {0};
          newval = builder.create<LLVM::InsertValueOp>(
              newarg.getLoc(), oldarg.getType(), ud, newval, c0);
        }

        map.map(oldarg, newval);
      }

      op.getFunctionBody().cloneInto(&gpufunc.getBody(), map);
      gpufunc->setAttr("gpu.kernel", builder.getUnitAttr());

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
    }
    SmallVector<Operation *> tocopy;
    op->walk([&](CallOpInterface cop) {
      if (auto op2 = cop.resolveCallable())
        tocopy.push_back(op2);
    });
    op->walk([&](LLVM::AddressOfOp cop) {
      if (auto op2 = cop.getGlobal(symbolTable))
        tocopy.push_back(op2);
      else if (auto op2 = cop.getFunction(symbolTable))
        tocopy.push_back(op2);
    });
    SmallPtrSet<Operation *, 1> done;

    builder.setInsertionPointToStart(&gpumod.getBodyRegion().front());
    while (tocopy.size()) {
      auto cur = tocopy.pop_back_val();
      if (done.count(cur))
        continue;
      done.insert(cur);
      builder.clone(*cur);
      cur->walk([&](CallOpInterface cop) {
        if (auto op2 = cop.resolveCallable())
          tocopy.push_back(op2);
      });
      cur->walk([&](LLVM::AddressOfOp cop) {
        if (auto op2 = cop.getGlobal(symbolTable))
          tocopy.push_back(op2);
        else if (auto op2 = cop.getFunction(symbolTable))
          tocopy.push_back(op2);
      });
    }
  }

  builder.setInsertionPoint(op);

  static size_t callid = 0;
  callid++;
  auto callName = (op.getName() + "$call$" + std::to_string(callid)).str();
  auto func = builder.create<func::FuncOp>(loc, callName, gpuTy);
  func.setVisibility(SymbolTable::Visibility::Private);

  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  auto idx = builder.getIntegerType(64);
  auto i32 = builder.getIntegerType(32);
  gpu::KernelDim3 gridSize{
      builder.create<arith::ConstantIntOp>(loc, gridx, idx),
      builder.create<arith::ConstantIntOp>(loc, gridy, idx),
      builder.create<arith::ConstantIntOp>(loc, gridz, idx),
  };

  gpu::KernelDim3 blockSize{
      builder.create<arith::ConstantIntOp>(loc, blockx, idx),
      builder.create<arith::ConstantIntOp>(loc, blocky, idx),
      builder.create<arith::ConstantIntOp>(loc, blockz, idx),
  };

  auto dynshmem = builder.create<arith::ConstantIntOp>(loc, shmem, i32);

  Value stream = builder.create<enzymexla::GetStreamOp>(
      loc, gpu::AsyncTokenType::get(kcall.getContext()));

  builder.create<gpu::LaunchFuncOp>(loc, gpufunc, gridSize, blockSize, dynshmem,
                                    entryBlock.getArguments(), stream.getType(),
                                    ValueRange(stream));

  builder.create<mlir::func::ReturnOp>(loc);

  if (!op->getParentOp()->hasAttr(
          gpu::GPUDialect::getContainerModuleAttrName()))
    op->getParentOp()->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                               UnitAttr::get(kcall.getContext()));

  OpBuilder rewriter(kcall);
  auto replacement = rewriter.create<enzymexla::JITCallOp>(
      kcall.getLoc(), kcall.getResultTypes(),
      mlir::FlatSymbolRefAttr::get(kcall.getContext(), callName),
      kcall.getInputs(), kcall.getBackendConfigAttr(),
      kcall.getOperandLayoutsAttr(), kcall.getResultLayoutsAttr(),
      kcall.getOutputOperandAliasesAttr());
  kcall.replaceAllUsesWith(replacement);
  kcall.erase();
  return true;
};

bool CompileCPUKernel(SymbolTableCollection &symbolTable, mlir::Location loc,
                      FunctionOpInterface op, size_t gridx, size_t gridy,
                      size_t gridz, size_t blockx, size_t blocky, size_t blockz,
                      size_t shmem, enzymexla::KernelCallOp kcall) {
  OpBuilder builder(op);

  FunctionType gpuTy0 = dyn_cast<FunctionType>(op.getFunctionType());
  if (!gpuTy0) {
    if (auto lty = dyn_cast<LLVM::LLVMFunctionType>(op.getFunctionType())) {
      gpuTy0 = builder.getFunctionType(lty.getParams(), {});
    } else {
      op.emitError(
          "Require target operand to have functiontype or llvmfunctiontype");
      return false;
    }
  }
  SmallVector<Type, 1> newParams;
  for (Type p : gpuTy0.getInputs()) {
    if (auto AT = dyn_cast<LLVM::LLVMArrayType>(p)) {
      p = AT.getElementType();
    }
    newParams.push_back(p);
  }

  static int id = 0;
  auto callName = (op.getName() + "$" + "par" + std::to_string(id)).str();
  id++;
  auto func = builder.create<func::FuncOp>(loc, callName, gpuTy0);
  func.setVisibility(SymbolTable::Visibility::Private);
  auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  SmallVector<mlir::Value> inits;
  SmallVector<mlir::Value> finals;
  SmallVector<mlir::Value> incs;
  for (auto val : {gridx, gridy, gridz, blockx, blocky, blockz}) {
    inits.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
    incs.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
    finals.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }

  IRMapping map;
  map.map(op.getArguments(), entryBlock.getArguments());

  auto context = loc.getContext();
  SmallVector<AffineMap> idMaps, zeroMaps;
  auto zeroMap = AffineMap::getConstantMap(0, context);
  zeroMaps.insert(zeroMaps.begin(), 6, zeroMap);
  for (unsigned i = 0; i < 6; i++) {
    auto idMap = AffineMap::get(0, 6, getAffineSymbolExpr(i, context));
    idMaps.push_back(idMap);
  }

  SmallVector<int64_t> steps(6, 1);
  auto par = builder.create<affine::AffineParallelOp>(
      loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), zeroMaps,
      ValueRange(), idMaps, finals, steps);

  builder.create<mlir::func::ReturnOp>(loc);

  builder.setInsertionPointToStart(&par.getRegion().front());
  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());

  op.getFunctionBody().cloneInto(&executeRegion.getRegion(), map);

  executeRegion->walk([](LLVM::ReturnOp op) {
    OpBuilder rewriter(op);
    rewriter.create<scf::YieldOp>(op.getLoc());
    op.erase();
  });

  executeRegion->walk([](func::ReturnOp op) {
    OpBuilder rewriter(op);
    rewriter.create<scf::YieldOp>(op.getLoc());
    op.erase();
  });

  executeRegion->walk([](LLVM::UnreachableOp op) {
    OpBuilder rewriter(op);
    rewriter.create<scf::YieldOp>(op.getLoc());
    op.erase();
  });

  // block idx
  executeRegion->walk([&](NVVM::BlockIdXOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[0]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });
  executeRegion->walk([&](NVVM::BlockIdYOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[1]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });
  executeRegion->walk([&](NVVM::BlockIdZOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[2]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });

  // thread idx
  executeRegion->walk([&](NVVM::ThreadIdXOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[3]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });
  executeRegion->walk([&](NVVM::ThreadIdYOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[4]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });
  executeRegion->walk([&](NVVM::ThreadIdZOp idxOp) {
    OpBuilder rewriter(idxOp);
    auto rep = rewriter.create<arith::IndexCastUIOp>(
        op.getLoc(), idxOp.getType(), par.getIVs()[5]);
    idxOp.replaceAllUsesWith(rep.getResult());
    idxOp.erase();
  });

  OpBuilder rewriter(kcall);
  auto replacement = rewriter.create<enzymexla::JITCallOp>(
      kcall.getLoc(), kcall.getResultTypes(),
      mlir::FlatSymbolRefAttr::get(kcall.getContext(), callName),
      kcall.getInputs(), kcall.getBackendConfigAttr(),
      kcall.getOperandLayoutsAttr(), kcall.getResultLayoutsAttr(),
      kcall.getOutputOperandAliasesAttr());
  kcall.replaceAllUsesWith(replacement);
  kcall.erase();
  return true;
};

namespace {

struct LowerKernelPass
    : public mlir::enzyme::impl::LowerKernelPassBase<LowerKernelPass> {
  using LowerKernelPassBase::LowerKernelPassBase;

  void runOnOperation() override {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());

    getOperation()->walk([&](KernelCallOp op) {
      size_t data[8];

      auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
      auto fn = cast<FunctionOpInterface>(symbolOp);
      if (fn.getArguments().size() != op.getInputs().size()) {
        op->emitError() << "Kernel_call had " << op.getInputs().size()
                        << " whereas called kernel requires "
                        << fn.getArguments().size() << "\n";
        return;
      }

      Value vals[] = {op.getGridx(),  op.getGridy(),  op.getGridz(),
                      op.getBlockx(), op.getBlocky(), op.getBlockz(),
                      op.getShmem()};
      for (auto en : llvm::enumerate(vals)) {
        DenseIntElementsAttr stepAttr;
        if (!matchPattern(en.value(), m_Constant(&stepAttr))) {
          op->emitError() << "Cannot lower kernel with a grid/block size which "
                             "is not a constant integer tensor";
          return;
        }
        if (stepAttr.size() != 1) {
          op->emitError() << "Cannot lower kernel with a grid/block size which "
                             "is not a constant integer tensor of size 1";
          return;
        }
        auto val = (*stepAttr.begin()).getZExtValue();
        data[1 + en.index()] = val;
      }

      // Compiled kernel goes here once ready
      if (backend == "cuda") {
        CompileGPUKernel(symbolTable, op.getLoc(), fn, data[1], data[2],
                         data[3], data[4], data[5], data[6], data[7], op);
      } else if (backend == "cpu") {
        CompileCPUKernel(symbolTable, op.getLoc(), fn, data[1], data[2],
                         data[3], data[4], data[5], data[6], data[7], op);
      } else {
        op->emitError() << "Cannot lower kernel to unknown backend \""
                        << backend << "\"";
      }
    });
  }
};

} // end anonymous namespace
