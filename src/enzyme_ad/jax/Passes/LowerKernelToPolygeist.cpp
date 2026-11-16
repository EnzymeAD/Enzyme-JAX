//===- LowerKernelToPolygeist.cpp -----------------------------------------===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "lower-kernel-to-polygeist"

namespace mlir::enzyme {
#define GEN_PASS_DEF_LOWERKERNELTOPOLYGEISTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

struct LowerKernelToPolygeistPass
    : public mlir::enzyme::impl::LowerKernelToPolygeistPassBase<
          LowerKernelToPolygeistPass> {
  using mlir::enzyme::impl::LowerKernelToPolygeistPassBase<
      LowerKernelToPolygeistPass>::LowerKernelToPolygeistPassBase;

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

  // Whether `op` is one of the operations the pass rewrites in the kernel
  // body: it reads the position of the thread in the launch or the launch
  // dimensions, synchronizes the block, or takes the address of shared memory.
  static bool usesLaunchContext(Operation *op) {
    if (isa<NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp,
            NVVM::ThreadIdXOp, NVVM::ThreadIdYOp, NVVM::ThreadIdZOp,
            NVVM::BlockDimXOp, NVVM::BlockDimYOp, NVVM::BlockDimZOp,
            NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp,
            NVVM::BarrierOp, gpu::BlockIdOp, gpu::ThreadIdOp, gpu::GridDimOp,
            gpu::BlockDimOp, gpu::BarrierOp>(op))
      return true;
    if (auto addressOf = dyn_cast<LLVM::AddressOfOp>(op))
      return cast<LLVM::LLVMPointerType>(addressOf.getType())
                 .getAddressSpace() == 3;
    return false;
  }

  // The function `call` calls, if it has a body to inline.
  static CallableOpInterface getDefinedCallee(CallOpInterface call,
                                              SymbolTableCollection &tables) {
    auto callee = dyn_cast_or_null<CallableOpInterface>(
        call.resolveCallableInTable(&tables));
    if (!callee)
      return nullptr;
    Region *body = callee.getCallableRegion();
    if (!body || body->empty())
      return nullptr;
    return callee;
  }

  // Whether `callable` uses the launch context, itself or through the
  // functions it calls, memoized in `usesContext`. Fails on recursion, which
  // inlining cannot remove; `stack` holds the functions being visited.
  FailureOr<bool> callsUseLaunchContext(
      enzymexla::KernelCallOp call, CallableOpInterface callable,
      SymbolTableCollection &symbolTables, SetVector<Operation *> &stack,
      DenseMap<Operation *, bool> &usesContext) {
    if (auto it = usesContext.find(callable); it != usesContext.end())
      return it->second;
    if (!stack.insert(callable)) {
      return call.emitError()
             << "the kernel calls "
             << cast<SymbolOpInterface>(callable.getOperation()).getName()
             << " recursively, which cannot be inlined";
    }
    bool uses = false;
    WalkResult result = callable->walk([&](Operation *op) {
      if (usesLaunchContext(op)) {
        uses = true;
        return WalkResult::advance();
      }
      auto callOp = dyn_cast<CallOpInterface>(op);
      if (!callOp)
        return WalkResult::advance();
      CallableOpInterface callee = getDefinedCallee(callOp, symbolTables);
      if (!callee)
        return WalkResult::advance();
      FailureOr<bool> calleeUses =
          callsUseLaunchContext(call, callee, symbolTables, stack, usesContext);
      if (failed(calleeUses))
        return WalkResult::interrupt();
      uses |= *calleeUses;
      return WalkResult::advance();
    });
    stack.pop_back();
    if (result.wasInterrupted())
      return failure();
    usesContext[callable] = uses;
    return uses;
  }

  // Inlines the functions the kernel body in `execute` calls, and the ones
  // they call, so that the launch context they use is rewritten with the
  // body's. A call that cannot be inlined stays, unless its callee uses the
  // launch context. `usesContext` holds every function the body reaches.
  LogicalResult inlineCallees(enzymexla::KernelCallOp call,
                              scf::ExecuteRegionOp execute,
                              SymbolTableCollection &symbolTables,
                              const DenseMap<Operation *, bool> &usesContext) {
    InlinerInterface interface(call.getContext());
    auto cloneCallback = [](OpBuilder &builder, Region *src, Block *inlineBlock,
                            Block *postInsertBlock, IRMapping &mapper,
                            bool shouldCloneInlinedRegion) {
      assert(shouldCloneInlinedRegion && "callees are cloned, not moved");
      src->cloneInto(inlineBlock->getParent(), postInsertBlock->getIterator(),
                     mapper);
    };

    // The callees have no recursion, so inlining the calls round after round
    // stops once no call remains that can be inlined.
    DenseSet<Operation *> kept;
    bool inlined = true;
    while (inlined) {
      inlined = false;
      SmallVector<CallOpInterface> calls;
      execute.walk([&](CallOpInterface callOp) {
        if (!kept.contains(callOp))
          calls.push_back(callOp);
      });
      for (CallOpInterface callOp : calls) {
        CallableOpInterface callee = getDefinedCallee(callOp, symbolTables);
        if (!callee) {
          kept.insert(callOp);
          continue;
        }
        // A callee with several blocks needs a region that can hold them.
        if (!isa<scf::ExecuteRegionOp>(callOp->getParentOp())) {
          OpBuilder builder(callOp);
          auto wrapper = scf::ExecuteRegionOp::create(builder, callOp.getLoc(),
                                                      callOp->getResultTypes());
          Block *block = new Block();
          wrapper.getRegion().push_back(block);
          callOp->replaceAllUsesWith(wrapper.getResults());
          callOp->moveBefore(block, block->end());
          builder.setInsertionPointToEnd(block);
          scf::YieldOp::create(builder, callOp.getLoc(), callOp->getResults());
        }
        if (failed(inlineCall(interface, cloneCallback, callOp, callee,
                              callee.getCallableRegion()))) {
          kept.insert(callOp);
          continue;
        }
        callOp->erase();
        inlined = true;
      }
    }

    for (Operation *op : kept) {
      CallableOpInterface callee =
          getDefinedCallee(cast<CallOpInterface>(op), symbolTables);
      if (callee && usesContext.lookup(callee)) {
        return call.emitError()
               << "cannot inline "
               << cast<SymbolOpInterface>(callee.getOperation()).getName()
               << ", which uses the position of the thread, the launch "
                  "dimensions, barriers or shared memory";
      }
    }
    return success();
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

    Value clusterValues[] = {call.getClusterx(), call.getClustery(),
                             call.getClusterz()};
    static constexpr StringLiteral clusterNames[] = {"cluster x", "cluster y",
                                                     "cluster z"};
    for (auto [value, name] : llvm::zip(clusterValues, clusterNames)) {
      if (!value)
        continue;
      FailureOr<int64_t> dimension = getConstant(call, value, 1, name);
      if (failed(dimension))
        return failure();
      if (*dimension != 1)
        return call.emitError(
            "cluster launches are not supported by this pass");
    }

    // Each result is the final contents of the operand buffer it aliases.
    ArrayAttr aliases = call.getOutputOperandAliases();
    if (aliases.size() != call.getNumResults())
      return call.emitError("every result must alias a kernel operand");
    for (Attribute attr : aliases) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
      if (alias.getOperandIndex() >= (int64_t)call.getInputs().size())
        return call.emitError("operand alias index out of range");
    }

    for (Value operand : call.getInputs())
      if (!isa<RankedTensorType>(operand.getType()))
        return call.emitError("kernel operands must be ranked tensors");

    SetVector<Operation *> stack;
    DenseMap<Operation *, bool> usesContext;
    if (failed(callsUseLaunchContext(
            call, cast<CallableOpInterface>(kernel.getOperation()),
            symbolTables, stack, usesContext)))
      return failure();

    // The function takes the pointers the kernel does, one per buffer.
    SmallVector<Type> argTypes;
    for (auto [i, karg] : llvm::enumerate(kernel.getArguments())) {
      if (!isa<LLVM::LLVMPointerType>(karg.getType()))
        return call.emitError()
               << "kernel argument #" << i << " is not a pointer";
      argTypes.push_back(karg.getType());
    }

    ModuleOp module = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
    // The device kernel outlined from this function inherits its name, and
    // PTX identifiers may only contain [A-Za-z0-9_$], so legalize kernel
    // symbols such as Julia's `##kernel#123`.
    std::string wrapperName = (call.getFn().getLeafReference().getValue() +
                               (gpuWrapper ? "_gpu_wrapper" : "_host"))
                                  .str();
    for (char &c : wrapperName)
      if (!llvm::isAlnum(c) && c != '_' && c != '$')
        c = '_';
    auto wrapperFn = LLVM::LLVMFuncOp::create(
        builder, call.getLoc(), wrapperName,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(call.getContext()),
                                    argTypes));
    wrapperFn.setLinkage(LLVM::Linkage::Private);
    symbolTables.getSymbolTable(module).insert(wrapperFn);
    Block *entry = wrapperFn.addEntryBlock(builder);
    ValueRange buffers = entry->getArguments();
    builder.setInsertionPointToEnd(entry);
    LLVM::ReturnOp::create(builder, call.getLoc(), ValueRange());

    builder.setInsertionPointToStart(entry);

    SmallVector<Value> bounds;
    for (int64_t dimension : launchDims)
      bounds.push_back(
          arith::ConstantIndexOp::create(builder, call.getLoc(), dimension));

    IRMapping mapping;

    // On the device, the loop runs in a gpu_wrapper; on the host, as is.
    if (gpuWrapper) {
      auto wrapper =
          enzymexla::GPUWrapperOp::create(builder, call.getLoc(), bounds);
      builder.setInsertionPoint(wrapper.getBody()->getTerminator());
    }
    for (auto [karg, buffer] : llvm::zip_equal(kernel.getArguments(), buffers))
      mapping.map(karg, buffer);

    // A loop over the blocks of the grid holds a loop over the threads of a
    // block, so that each block has a body of its own for its shared memory,
    // and barriers synchronize the threads of the inner loop.
    MLIRContext *context = call.getContext();
    SmallVector<AffineMap> lowerMaps(3, AffineMap::getConstantMap(0, context));
    SmallVector<AffineMap> upperMaps;
    for (unsigned i = 0; i < 3; ++i)
      upperMaps.push_back(
          AffineMap::get(0, 3, getAffineSymbolExpr(i, context)));
    SmallVector<int64_t> steps(3, 1);
    auto createParallel = [&](ValueRange upperBounds) {
      auto parallel = affine::AffineParallelOp::create(
          builder, call.getLoc(), TypeRange(), ArrayRef<arith::AtomicRMWKind>(),
          lowerMaps, ValueRange(), upperMaps, upperBounds, steps);
      builder.setInsertionPointToStart(parallel.getBody());
      return parallel;
    };
    auto blockParallel = createParallel(ValueRange(bounds).take_front(3));
    auto threadParallel = createParallel(ValueRange(bounds).drop_front(3));

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

    if (failed(inlineCallees(call, execute, symbolTables, usesContext)))
      return failure();

    ValueRange blockIds = blockParallel.getIVs();
    ValueRange threadIds = threadParallel.getIVs();
    execute.walk([&](NVVM::BlockIdXOp op) { replaceNVVMId(op, blockIds[0]); });
    execute.walk([&](NVVM::BlockIdYOp op) { replaceNVVMId(op, blockIds[1]); });
    execute.walk([&](NVVM::BlockIdZOp op) { replaceNVVMId(op, blockIds[2]); });
    execute.walk(
        [&](NVVM::ThreadIdXOp op) { replaceNVVMId(op, threadIds[0]); });
    execute.walk(
        [&](NVVM::ThreadIdYOp op) { replaceNVVMId(op, threadIds[1]); });
    execute.walk(
        [&](NVVM::ThreadIdZOp op) { replaceNVVMId(op, threadIds[2]); });

    execute.walk([&](NVVM::BlockDimXOp op) { replaceNVVMId(op, bounds[3]); });
    execute.walk([&](NVVM::BlockDimYOp op) { replaceNVVMId(op, bounds[4]); });
    execute.walk([&](NVVM::BlockDimZOp op) { replaceNVVMId(op, bounds[5]); });
    execute.walk([&](NVVM::GridDimXOp op) { replaceNVVMId(op, bounds[0]); });
    execute.walk([&](NVVM::GridDimYOp op) { replaceNVVMId(op, bounds[1]); });
    execute.walk([&](NVVM::GridDimZOp op) { replaceNVVMId(op, bounds[2]); });

    execute.walk([&](gpu::BlockIdOp op) {
      op.replaceAllUsesWith(dimensionValue(op.getDimension(), blockIds));
      op.erase();
    });
    execute.walk([&](gpu::ThreadIdOp op) {
      op.replaceAllUsesWith(dimensionValue(op.getDimension(), threadIds));
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
      enzymexla::BarrierOp::create(builder, op.getLoc(), threadIds);
      op.erase();
    });
    execute.walk([&](gpu::BarrierOp op) {
      OpBuilder builder(op);
      enzymexla::BarrierOp::create(builder, op.getLoc(), threadIds);
      op.erase();
    });

    IRRewriter rewriter(context);
    enzyme::privatizeSharedMemory(rewriter, execute, blockParallel.getBody(),
                                  symbolTables);

    // The jit_call has the results, aliases and attributes of the kernel call.
    builder.setInsertionPoint(call);
    auto jitCall = enzymexla::JITCallOp::create(
        builder, call.getLoc(), call.getResultTypes(),
        SymbolRefAttr::get(wrapperFn.getSymNameAttr()), call.getInputs(),
        /*backend_config=*/builder.getStringAttr(""),
        call.getOperandLayoutsAttr(), call.getArgAttrsAttr(),
        call.getResAttrsAttr(), aliases, call.getXlaSideEffectFreeAttr());

    call.replaceAllUsesWith(jitCall.getResults());
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
