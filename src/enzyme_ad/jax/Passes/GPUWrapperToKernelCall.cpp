//===- GPUWrapperToKernelCall.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "Dialect/Ops.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Implementations/SHLOGenericBatchOpInterface.h"
#include "src/enzyme_ad/jax/Passes/ConvertPolygeistToLLVM.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "gpu-wrapper-to-kernel-call"

namespace mlir::enzyme {
#define GEN_PASS_DEF_GPUWRAPPERTOKERNELCALLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

struct GPUWrapperToKernelCallPass
    : public mlir::enzyme::impl::GPUWrapperToKernelCallPassBase<
          GPUWrapperToKernelCallPass> {
  using mlir::enzyme::impl::GPUWrapperToKernelCallPassBase<
      GPUWrapperToKernelCallPass>::GPUWrapperToKernelCallPassBase;

  LLVM::LLVMFuncOp convertGPUFuncToLLVMFunc(gpu::GPUFuncOp funcOp,
                                            SymbolTableCollection &collection,
                                            ValueRange callOperands) {

    SmallVector<Type> arguments;

    for (int i = 0, e = funcOp.getFunctionType().getNumInputs(); i < e; ++i) {
      arguments.push_back(
          LLVM::LLVMPointerType::get(funcOp.getContext(), /*addressSpace=*/1));
    }

    assert(funcOp.getFunctionType().getResults().empty());

    auto fnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(funcOp->getContext()), arguments);

    auto moduleOp = funcOp->getParentOfType<gpu::GPUModuleOp>();

    std::string fnName =
        (moduleOp.getName() + "::" + funcOp.getName() + "_to_llvm").str();

    OpBuilder builder(funcOp->getParentOp());
    auto fn =
        LLVM::LLVMFuncOp::create(builder, funcOp->getLoc(), fnName, fnType);
    fn.setPrivate();

    Block *entry = fn.addEntryBlock(builder);
    Block *gpuBody = &funcOp.getBody().front();

    builder.setInsertionPointToEnd(entry);

    IRMapping mapping;
    for (auto [argument, gpu_argument, callOperand] : llvm::zip_equal(
             entry->getArguments(), gpuBody->getArguments(), callOperands)) {
      if (auto MT = dyn_cast<MemRefType>(gpu_argument.getType())) {
        Value memref = enzymexla::Pointer2MemrefOp::create(
            builder, argument.getLoc(), MT, argument);
        mapping.map(gpu_argument, memref);
        continue;
      }

      // This argument was already a raw pointer at the gpu.launch_func
      // boundary (e.g. an Enzyme-AD kernel operand), produced by casting a
      // memref with enzymexla.memref2pointer before outlining. Recover that
      // memref's type from the caller-side cast and re-materialize the
      // pointer from inside the function body instead, so the call boundary
      // itself only ever needs to carry pointers matching this argument's
      // declared type while the shape survives via the nested cast.
      auto m2p = callOperand.getDefiningOp<enzymexla::Memref2PointerOp>();
      assert(m2p && "expected a raw-pointer kernel operand to be produced by "
                    "enzymexla.memref2pointer");
      Value memref = enzymexla::Pointer2MemrefOp::create(
          builder, argument.getLoc(), m2p.getSource().getType(), argument);
      Value ptr = enzymexla::Memref2PointerOp::create(
          builder, argument.getLoc(), gpu_argument.getType(), memref);
      mapping.map(gpu_argument, ptr);
    }

    for (Operation &op : gpuBody->without_terminator()) {
      builder.clone(op, mapping);
    }

    LLVM::ReturnOp::create(builder, gpuBody->getTerminator()->getLoc(),
                           ValueRange{});

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(getOperation()));

    LLVMTypeConverter converter(fn.getContext(), options, &dataLayoutAnalysis);

    std::string backend = "cuda";

    // Memrefs have statically known shapes here (they come from a
    // gpu.launch_func's operands), so lower them to bare pointers plus GEP
    // arithmetic instead of the LLVM memref descriptor struct.
    converter.addConversion([&](MemRefType type) -> std::optional<Type> {
      auto elTy = convertMemrefElementTypeForLLVMPointer(type, converter);
      if (!elTy)
        return Type();
      return LLVM::LLVMPointerType::get(type.getContext(),
                                        type.getMemorySpaceAsInt());
    });
    converter.addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                           ValueRange inputs, Location loc,
                                           Type originalType) -> Value {
      if (inputs.size() != 1)
        return Value();

      auto resPtrType = dyn_cast<LLVM::LLVMPointerType>(resultType);
      auto inPtrType =
          dyn_cast<LLVM::LLVMPointerType>(inputs.front().getType());
      if (resPtrType && inPtrType &&
          resPtrType.getAddressSpace() != inPtrType.getAddressSpace()) {
        emitWarning(inputs.front().getLoc())
            << "mismatched address space, emitting addrspacecast";
        return LLVM::AddrSpaceCastOp::create(builder, loc, resultType,
                                             inputs.front());
      }
      return Value();
    });

    RewritePatternSet patterns(&getContext());

    populatePolygeistToLLVMConversionPatterns(converter, patterns);
    populateCStyleGPUFuncLoweringPatterns(patterns, converter, backend, false);
    populateCStyleMemRefLoweringPatterns(patterns, converter, backend);

    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

    ConversionConfig conversionConfig;

    ConversionTarget target(getContext());

    target.addIllegalDialect<gpu::GPUDialect, memref::MemRefDialect>();
    target.addIllegalOp<enzyme::AtomicRMWOp, enzyme::AffineAtomicRMWOp>();
    target.addLegalDialect<NVVM::NVVMDialect, LLVM::LLVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(fn, target, std::move(patterns),
                                      conversionConfig))) {
      fn->emitError("failed to apply partial conversion in "
                    "GPUWrapperToKernelCallPass");
    }

    moduleOp->erase();

    return fn;
  }

  // The buffer of `region` a kernel operand points to, looking through the
  // pointer casts between them.
  static FailureOr<BlockArgument> getBuffer(Value operand,
                                            enzymexla::JITRegionOp region) {
    while (true) {
      if (auto arg = dyn_cast<BlockArgument>(operand);
          arg && arg.getOwner() == region.getBodyBlock())
        return arg;
      if (auto m2p = operand.getDefiningOp<enzymexla::Memref2PointerOp>()) {
        operand = m2p.getSource();
        continue;
      }
      if (auto p2m = operand.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
        operand = p2m.getSource();
        continue;
      }
      return failure();
    }
  }

  // Creates a kernel_call launching the kernel of `launch` on `operands`, the
  // contents of its buffers. Each result is the contents of the buffer of the
  // corresponding operand after the launch.
  FailureOr<enzymexla::KernelCallOp>
  createKernelCall(OpBuilder &builder, gpu::LaunchFuncOp launch,
                   ValueRange operands, SymbolTableCollection &symbolTables) {
    auto toTensorI64 = [&](Value v) -> Value {
      llvm::APInt val;
      if (!matchPattern(v, m_ConstantInt(&val)))
        return nullptr;
      return details::makeI64Constant(v.getLoc(), builder, val.getSExtValue());
    };

    Value gridx = toTensorI64(launch.getGridSizeX());
    Value gridy = toTensorI64(launch.getGridSizeY());
    Value gridz = toTensorI64(launch.getGridSizeZ());
    Value blockx = toTensorI64(launch.getBlockSizeX());
    Value blocky = toTensorI64(launch.getBlockSizeY());
    Value blockz = toTensorI64(launch.getBlockSizeZ());
    Value shmem =
        launch.getDynamicSharedMemorySize()
            ? toTensorI64(launch.getDynamicSharedMemorySize())
            : details::makeI64Constant(launch.getLoc(), builder, 0);
    if (!gridx || !gridy || !gridz || !blockx || !blocky || !blockz || !shmem)
      return launch.emitError("launch dimensions must be constants");

    Value clusterx, clustery, clusterz;
    if (launch.hasClusterSize()) {
      clusterx = toTensorI64(launch.getClusterSizeX());
      clustery = toTensorI64(launch.getClusterSizeY());
      clusterz = toTensorI64(launch.getClusterSizeZ());
      if (!clusterx || !clustery || !clusterz)
        return launch.emitError("cluster dimensions must be constants");
    }

    MLIRContext *ctx = launch.getContext();
    SmallVector<Attribute> aliases;
    for (unsigned i = 0, e = operands.size(); i < e; ++i)
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, /*outputTupleIndices=*/{}, /*operandIndex=*/i,
          /*operandTupleIndices=*/{}));

    auto gpuFunc = symbolTables.lookupNearestSymbolFrom<gpu::GPUFuncOp>(
        launch, launch.getKernel());
    auto fn = convertGPUFuncToLLVMFunc(gpuFunc, symbolTables,
                                       launch.getKernelOperands());

    return enzymexla::KernelCallOp::create(
        builder, launch.getLoc(), operands.getTypes(),
        SymbolRefAttr::get(fn.getSymNameAttr()), gridx, gridy, gridz, blockx,
        blocky, blockz, shmem, clusterx, clustery, clusterz, operands,
        /*backend_config=*/builder.getStringAttr(""),
        /*operand_layouts=*/nullptr, /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        ArrayAttr::get(ctx, aliases), /*xla_side_effect_free=*/nullptr);
  }

  // Replaces a jit_region by the kernel_calls of the kernels it launches,
  // tracking the contents of each of its buffers as a tensor: they start as
  // the region's inputs, each launch updates those of the buffers it is
  // passed, and the region's results are their final values.
  LogicalResult lowerJITRegion(enzymexla::JITRegionOp region,
                               SymbolTableCollection &symbolTables) {
    Block *body = region.getBodyBlock();
    DenseMap<Value, Value> contents;
    for (auto [arg, input] :
         llvm::zip_equal(body->getArguments(), region.getInputs()))
      contents[arg] = input;

    OpBuilder builder(region);
    for (Operation &op :
         llvm::make_early_inc_range(body->without_terminator())) {
      if (auto errOp = dyn_cast<enzymexla::GPUErrorOp>(&op)) {
        Block &launchBlock = errOp.getRegion().front();
        auto launch = dyn_cast<gpu::LaunchFuncOp>(&launchBlock.front());
        if (!launch || launchBlock.getOperations().size() != 2)
          return errOp.emitError(
              "expected a gpu_error holding a single gpu.launch_func");
        if (!errOp->use_empty())
          return errOp.emitError(
              "the error code of a kernel launch in a jit_region is not "
              "supported");

        SmallVector<BlockArgument> buffers;
        for (auto [i, operand] : llvm::enumerate(launch.getKernelOperands())) {
          FailureOr<BlockArgument> buffer = getBuffer(operand, region);
          if (failed(buffer))
            return launch.emitError()
                   << "kernel operand #" << i
                   << " is not a buffer of the enclosing jit_region";
          // Each operand of a kernel_call has its own buffer.
          if (llvm::is_contained(buffers, *buffer))
            return launch.emitError()
                   << "kernel operand #" << i
                   << " aliases another operand of the kernel";
          buffers.push_back(*buffer);
        }

        SmallVector<Value> operands;
        for (BlockArgument buffer : buffers)
          operands.push_back(contents[buffer]);

        builder.setInsertionPoint(region);
        FailureOr<enzymexla::KernelCallOp> call =
            createKernelCall(builder, launch, operands, symbolTables);
        if (failed(call))
          return failure();
        for (auto [buffer, result] :
             llvm::zip_equal(buffers, call->getResults()))
          contents[buffer] = result;

        errOp.erase();
        continue;
      }

      if (!isMemoryEffectFree(&op) || op.getNumRegions() != 0)
        return op.emitError("cannot lower an operation with side effects in "
                            "a jit_region to kernel calls");

      // Values the launches depend on are kept; views of the buffers are
      // left behind, and die with the region.
      if (llvm::none_of(op.getOperands(), [&](Value operand) {
            return region.getBody().isAncestor(operand.getParentRegion());
          }))
        op.moveBefore(region);
    }

    for (auto [result, arg] :
         llvm::zip_equal(region.getResults(), body->getArguments()))
      result.replaceAllUsesWith(contents[arg]);
    region.erase();
    return success();
  }

  void runOnOperation() override {
    SymbolTableCollection symbolTables;

    SmallVector<enzymexla::JITRegionOp> regions;
    getOperation().walk(
        [&](enzymexla::JITRegionOp region) { regions.push_back(region); });

    bool failed = false;
    for (enzymexla::JITRegionOp region : regions)
      failed |= mlir::failed(lowerJITRegion(region, symbolTables));

    getOperation().walk([&](enzymexla::GPUErrorOp errOp) {
      errOp.emitError("expected a kernel launch in a jit_region");
      failed = true;
    });

    if (failed)
      signalPassFailure();
  }
};

} // namespace
