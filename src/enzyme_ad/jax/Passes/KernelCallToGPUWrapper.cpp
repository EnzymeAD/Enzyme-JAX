//===- KernelCallToGPUWrapper.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "kernel-call-to-gpu-wrapper"

namespace mlir::enzyme {
#define GEN_PASS_DEF_KERNELCALLTOGPUWRAPPERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

struct KernelCallToGPUWrapperPass
    : public mlir::enzyme::impl::KernelCallToGPUWrapperPassBase<
          KernelCallToGPUWrapperPass> {
  using mlir::enzyme::impl::KernelCallToGPUWrapperPassBase<
      KernelCallToGPUWrapperPass>::KernelCallToGPUWrapperPassBase;

  FailureOr<int64_t> getConstant(enzymexla::KernelCallOp call, Value value,
                                 int64_t defaultValue, StringRef description) {
    if (!value)
      return defaultValue;
    APInt constantValue;
    if (!matchPattern(value, m_ConstantInt(&constantValue))) {
      return call.emitError()
             << description << " must be a constant scalar integer tensor";
    }
    return constantValue.getSExtValue();
  }

  template <typename OpTy> static void replaceNVVMId(OpTy op, Value iv) {
    OpBuilder builder(op);
    Value replacement =
        arith::IndexCastUIOp::create(builder, op.getLoc(), op.getType(), iv);
    op.replaceAllUsesWith(replacement);
    op.erase();
  }

  static Value dimensionValue(gpu::Dimension dimension, ValueRange values) {
    switch (dimension) {
    case gpu::Dimension::x:
      return values[0];
    case gpu::Dimension::y:
      return values[1];
    case gpu::Dimension::z:
      return values[2];
    }
    llvm_unreachable("unknown GPU dimension");
  }

  LogicalResult rewriteCall(enzymexla::KernelCallOp call,
                            SymbolTableCollection &symbolTables) {
    Operation *symbol =
        symbolTables.lookupNearestSymbolFrom(call, call.getFnAttr());
    auto kernel = dyn_cast_or_null<FunctionOpInterface>(symbol);
    if (!kernel)
      return call.emitError("kernel symbol does not name a function");
    if (kernel.isExternal())
      return call.emitError("cannot expose an external kernel body");
    if (kernel.getNumArguments() != call.getInputs().size())
      return call.emitError("kernel argument count does not match call inputs");

    SmallVector<Value> launchValues{call.getGridx(),  call.getGridy(),
                                    call.getGridz(),  call.getBlockx(),
                                    call.getBlocky(), call.getBlockz()};
    SmallVector<int64_t> launchDims;
    static constexpr StringLiteral dimensionNames[] = {
        "grid x", "grid y", "grid z", "block x", "block y", "block z"};
    for (auto [value, name] : llvm::zip(launchValues, dimensionNames)) {
      FailureOr<int64_t> dimension = getConstant(call, value, 1, name);
      if (failed(dimension))
        return failure();
      if (*dimension < 1)
        return call.emitError() << name << " must be positive";
      launchDims.push_back(*dimension);
    }

    FailureOr<int64_t> sharedMemory =
        getConstant(call, call.getShmem(), 0, "shared memory size");
    if (failed(sharedMemory))
      return failure();
    if (*sharedMemory != 0)
      return call.emitError(
          "nonzero dynamic shared memory is not supported by this pass");

    for (auto [value, name] : llvm::zip(
             ValueRange{call.getClusterx(), call.getClustery(),
                        call.getClusterz()},
             ArrayRef<StringRef>{"cluster x", "cluster y", "cluster z"})) {
      FailureOr<int64_t> dimension = getConstant(call, value, 1, name);
      if (failed(dimension))
        return failure();
      if (*dimension != 1)
        return call.emitError(
            "cluster launches are not supported by this pass");
    }

    OpBuilder builder(call);

    SmallVector<Value> bounds;
    for (int64_t dimension : launchDims)
      bounds.push_back(
          arith::ConstantIndexOp::create(builder, call.getLoc(), dimension));

    IRMapping mapping;

    SmallVector<Value> kargMemrefs;
    for (auto [karg, operand] :
         llvm::zip_equal(kernel.getArguments(), call.getArgOperands())) {
      auto T = cast<TensorType>(operand.getType());
      auto MT = MemRefType::get(T.getShape(), T.getElementType(),
                                /* layout= */ MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(1));
      Value memref = enzymexla::Tensor2MemrefOp::create(
          builder, operand.getLoc(), MT, operand);
      kargMemrefs.push_back(memref);
      Value ptr = enzymexla::Memref2PointerOp::create(builder, operand.getLoc(),
                                                      karg.getType(), memref);
      mapping.map(karg, ptr);
    }

    auto wrapper =
        enzymexla::GPUWrapperOp::create(builder, call.getLoc(), bounds);
    builder.setInsertionPointToStart(wrapper.getBody());

    MLIRContext *context = call.getContext();
    SmallVector<AffineMap> lowerMaps(6, AffineMap::getConstantMap(0, context));
    SmallVector<AffineMap> upperMaps;
    for (unsigned i = 0; i < 6; ++i)
      upperMaps.push_back(
          AffineMap::get(0, 6, getAffineSymbolExpr(i, context)));
    SmallVector<int64_t> steps(6, 1);

    auto parallel = affine::AffineParallelOp::create(
        builder, call.getLoc(), TypeRange(), ArrayRef<arith::AtomicRMWKind>(),
        lowerMaps, ValueRange(), upperMaps, bounds, steps);

    builder.setInsertionPointToStart(parallel.getBody());
    auto execute =
        scf::ExecuteRegionOp::create(builder, call.getLoc(), TypeRange());

    kernel.getFunctionBody().cloneInto(&execute.getRegion(), mapping);

    execute.walk([](LLVM::ReturnOp op) {
      OpBuilder builder(op);
      scf::YieldOp::create(builder, op.getLoc());
      op.erase();
    });
    execute.walk([](func::ReturnOp op) {
      OpBuilder builder(op);
      scf::YieldOp::create(builder, op.getLoc());
      op.erase();
    });
    execute.walk([](LLVM::UnreachableOp op) {
      OpBuilder builder(op);
      scf::YieldOp::create(builder, op.getLoc());
      op.erase();
    });

    ValueRange ivs = parallel.getIVs();
    execute.walk([&](NVVM::BlockIdXOp op) { replaceNVVMId(op, ivs[0]); });
    execute.walk([&](NVVM::BlockIdYOp op) { replaceNVVMId(op, ivs[1]); });
    execute.walk([&](NVVM::BlockIdZOp op) { replaceNVVMId(op, ivs[2]); });
    execute.walk([&](NVVM::ThreadIdXOp op) { replaceNVVMId(op, ivs[3]); });
    execute.walk([&](NVVM::ThreadIdYOp op) { replaceNVVMId(op, ivs[4]); });
    execute.walk([&](NVVM::ThreadIdZOp op) { replaceNVVMId(op, ivs[5]); });

    execute.walk([&](NVVM::BlockDimXOp op) { replaceNVVMId(op, bounds[3]); });
    execute.walk([&](NVVM::BlockDimYOp op) { replaceNVVMId(op, bounds[4]); });
    execute.walk([&](NVVM::BlockDimZOp op) { replaceNVVMId(op, bounds[5]); });
    execute.walk([&](NVVM::GridDimXOp op) { replaceNVVMId(op, bounds[0]); });
    execute.walk([&](NVVM::GridDimYOp op) { replaceNVVMId(op, bounds[1]); });
    execute.walk([&](NVVM::GridDimZOp op) { replaceNVVMId(op, bounds[2]); });

    execute.walk([&](gpu::BlockIdOp op) {
      op.replaceAllUsesWith(
          dimensionValue(op.getDimension(), ivs.take_front(3)));
      op.erase();
    });
    execute.walk([&](gpu::ThreadIdOp op) {
      op.replaceAllUsesWith(
          dimensionValue(op.getDimension(), ivs.drop_front(3)));
      op.erase();
    });
    execute.walk([&](gpu::GridDimOp op) {
      op.replaceAllUsesWith(
          dimensionValue(op.getDimension(), ValueRange(bounds).take_front(3)));
      op.erase();
    });
    execute.walk([&](gpu::BlockDimOp op) {
      op.replaceAllUsesWith(
          dimensionValue(op.getDimension(), ValueRange(bounds).drop_front(3)));
      op.erase();
    });
    execute.walk([&](NVVM::BarrierOp op) {
      OpBuilder builder(op);
      enzymexla::BarrierOp::create(builder, op.getLoc(), ivs.drop_front(3));
      op.erase();
    });
    execute.walk([&](gpu::BarrierOp op) {
      OpBuilder builder(op);
      enzymexla::BarrierOp::create(builder, op.getLoc(), ivs.drop_front(3));
      op.erase();
    });

    // OpBuilder callBuilder(call);
    // auto replacement = enzymexla::JITCallOp::create(
    //     callBuilder, call.getLoc(), call.getResultTypes(),
    //     SymbolRefAttr::get(context, helperName), call.getInputs(),
    //     call.getBackendConfigAttr(), call.getOperandLayoutsAttr(),
    //     call.getResultLayoutsAttr(), call.getArgAttrsAttr(),
    //     call.getResAttrsAttr(), call.getOutputOperandAliasesAttr(),
    //     call.getXlaSideEffectFreeAttr());
    // call.replaceAllUsesWith(replacement.getResults());
    builder.setInsertionPoint(call);

    for (auto [result, alias_attr] :
         llvm::zip_equal(call.getResults(), call.getOutputOperandAliases())) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
      auto aliasOperandIndex = alias.getOperandIndex();

      Value memref = kargMemrefs[aliasOperandIndex];
      Value tensor = enzymexla::Memref2TensorOp::create(
          builder, result.getLoc(), result.getType(), memref);
      result.replaceAllUsesWith(tensor);
    }

    call.erase();
    return success();
  }

  void runOnOperation() override {
    SymbolTableCollection symbolTables;
    SmallVector<enzymexla::KernelCallOp> calls;
    getOperation().walk(
        [&](enzymexla::KernelCallOp call) { calls.push_back(call); });
    for (enzymexla::KernelCallOp call : calls) {
      if (failed(rewriteCall(call, symbolTables))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
