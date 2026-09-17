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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include <algorithm>

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

struct Tensor2MemrefEliminate
    : public OpRewritePattern<enzymexla::Tensor2MemrefOp> {
  using OpRewritePattern<enzymexla::Tensor2MemrefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::Tensor2MemrefOp op,
                                PatternRewriter &rewriter) const override {

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // tensor2memref(memref2tensor(x)) -> x
    if (auto def =
            op.getTensor().getDefiningOp<enzymexla::Memref2TensorOp>()) {
      if (def.getMemref().getType() == op.getMemref().getType()) {
        rewriter.replaceOp(op, def.getMemref());
        return success();
      }
    }

    if (!llvm::all_of(op.getMemref().getUsers(), [&](Operation *user) {
          auto cast = dyn_cast<enzymexla::Memref2TensorOp>(user);
          return cast && cast.getTensor().getType() == op.getTensor().getType();
        }))
      return failure();

    // Every user just converts the memref straight back to a tensor: forward
    // the original tensor to them directly instead, collecting the users
    // first since erasing them below mutates op.getMemref()'s use-list.
    SmallVector<Operation *> users(op.getMemref().getUsers());
    for (Operation *user : users) {
      rewriter.replaceAllUsesWith(user->getResult(0), op.getTensor());
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

struct Memref2TensorEliminate
    : public OpRewritePattern<enzymexla::Memref2TensorOp> {
  using OpRewritePattern<enzymexla::Memref2TensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::Memref2TensorOp op,
                                PatternRewriter &rewriter) const override {

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // memref2tensor(tensor2memref(x)) -> x
    if (auto def =
            op.getMemref().getDefiningOp<enzymexla::Tensor2MemrefOp>()) {
      if (def.getTensor().getType() == op.getTensor().getType()) {
        rewriter.replaceOp(op, def.getTensor());
        return success();
      }
    }

    if (!llvm::all_of(op.getTensor().getUsers(), [&](Operation *user) {
          auto cast = dyn_cast<enzymexla::Tensor2MemrefOp>(user);
          return cast && cast.getMemref().getType() == op.getMemref().getType();
        }))
      return failure();

    // Every user just converts the tensor straight back to a memref: forward
    // the original memref to them directly instead, collecting the users
    // first since erasing them below mutates op.getTensor()'s use-list.
    SmallVector<Operation *> users(op.getTensor().getUsers());
    for (Operation *user : users) {
      rewriter.replaceAllUsesWith(user->getResult(0), op.getMemref());
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);

    return success();
  }
};

// // mem2reg-lite forwarding for the tensor/memref bridge ops: memref2tensor
// // is a load, tensor2memref is an alloc+store. This walks each block in
// // program order, tracking -- for every memref this pass has a fact about --
// // the tensor value currently known to match its contents, and forwards
// // later memref2tensor reads directly to that value instead of leaving a
// // fresh (but redundant) cast in the IR.
// //
// // enzymexla.kernel_call gets no bespoke treatment here, even though its
// // output_operand_aliases documents that it may physically donate one of its
// // (tensor-typed) operand's buffers to a result: this pass never decodes
// // that attribute or looks at what the op does with the buffer, only its
// // generic MemoryEffectsOpInterface-reported effects. If the effects say the
// // op isn't memory-effect-free, any memref reachable from its operands
// // (including, structurally, through a tensor operand that is itself
// // nothing but a memref2tensor read) has its cached value forgotten --
// // never replaced with a guess at the new one. That keeps every op besides
// // the bridge casts themselves fully opaque: a later read of the same
// // memref just becomes a real load again instead of being folded to a
// // (possibly wrong) prior value.
// //
// // This never removes or reorders the underlying memref-level ops themselves
// // (allocs, copies, kernel calls, dealloc): those still run for real; only
// // redundant SSA-level memref2tensor casts are eliminated.
// static void forwardTensorMemRefValues(Region &region) {
//   for (Block &block : region) {
//     for (Operation &nested : block)
//       for (Region &nestedRegion : nested.getRegions())
//         forwardTensorMemRefValues(nestedRegion);

//     // memref -> the tensor value currently known to match its contents.
//     DenseMap<Value, Value> currentValue;

//     SmallVector<Operation *> toErase;
//     for (Operation &opRef : block) {
//       Operation *op = &opRef;

//       if (auto m2t = dyn_cast<enzymexla::Memref2TensorOp>(op)) {
//         Value memref = m2t.getMemref();
//         auto it = currentValue.find(memref);
//         if (it != currentValue.end() &&
//             it->second.getType() == m2t.getTensor().getType()) {
//           m2t.getTensor().replaceAllUsesWith(it->second);
//           toErase.push_back(op);
//         } else {
//           // This is now the freshest known content of the memref -- until
//           // something writes to it, later reads can forward to this one.
//           currentValue[memref] = m2t.getTensor();
//         }
//         continue;
//       }

//       if (auto t2m = dyn_cast<enzymexla::Tensor2MemrefOp>(op)) {
//         currentValue[t2m.getMemref()] = t2m.getTensor();
//         continue;
//       }

//       if (auto copy = dyn_cast<memref::CopyOp>(op)) {
//         auto it = currentValue.find(copy.getSource());
//         if (it != currentValue.end())
//           currentValue[copy.getTarget()] = it->second;
//         else
//           currentValue.erase(copy.getTarget());
//         continue;
//       }

//       // A fresh allocation's contents are unknown (there is nothing to
//       // forward yet, and it cannot alias any memref already in the map), so
//       // there is nothing to do beyond leaving it out of the map.
//       if (isa<memref::AllocOp, memref::AllocaOp, gpu::AllocOp>(op))
//         continue;

//       // Fills the whole buffer with zero: record that as its known content
//       // so a later read forwards to a zero constant instead of a fresh
//       // cast.
//       if (auto fillZero = dyn_cast<enzyme::FillZeroOp>(op)) {
//         auto memrefTy = cast<MemRefType>(fillZero.getMemref().getType());
//         if (memrefTy.hasStaticShape()) {
//           auto tensorTy = RankedTensorType::get(memrefTy.getShape(),
//                                                 memrefTy.getElementType());
//           OpBuilder builder(op);
//           builder.setInsertionPointAfter(op);
//           Value zero = arith::ConstantOp::create(
//               builder, op->getLoc(),
//               SplatElementsAttr::get(tensorTy,
//                                      builder.getZeroAttr(
//                                          memrefTy.getElementType())));
//           currentValue[fillZero.getMemref()] = zero;
//         }
//         continue;
//       }

//       // Conservatively forget anything we thought we knew about a memref
//       // that flows into an op we don't otherwise understand and that may
//       // write through it -- whether as a direct memref operand, or (as with
//       // enzymexla.kernel_call, whose operands/results are all tensors) as a
//       // tensor operand that is itself nothing but a read of that memref. No
//       // op-specific knowledge of what such an op does with the buffer is
//       // used here, only its own reported memory effects: we only know to
//       // stop trusting the cached value, never what it becomes.
//       if (!isMemoryEffectFree(op)) {
//         for (Value operand : op->getOperands()) {
//           if (isa<MemRefType>(operand.getType())) {
//             currentValue.erase(operand);
//           } else if (auto read =
//                          operand.getDefiningOp<enzymexla::Memref2TensorOp>())
//                          {
//             currentValue.erase(read.getMemref());
//           }
//         }
//       }
//     }

//     for (Operation *op : toErase)
//       op->erase();
//   }
// }

// // Once forwardTensorMemRefValues has forwarded every memref2tensor read
// // it safely could, some allocations may be left with no read at all --
// // only writes (enzyme.fill_zero, memref.copy targets) and a final
// // deallocation. Nothing ever observes such a buffer's contents, so the
// // allocation and everything that writes or frees it can be deleted outright.
// // This is a fixed point: deleting one allocation's writes can turn some
// // *other* memref's only remaining use (as a memref.copy source feeding this
// // now-dead target) into something worth reconsidering elsewhere, though in
// // practice that only matters for chains of copies.
// static void eliminateDeadMemRefAllocations(Region &region) {
//   for (Block &block : region) {
//     for (Operation &nested : block)
//       for (Region &nestedRegion : nested.getRegions())
//         eliminateDeadMemRefAllocations(nestedRegion);

//     auto getAllocatedMemref = [](Operation *op) -> Value {
//       if (auto alloc = dyn_cast<memref::AllocOp>(op))
//         return alloc.getMemref();
//       if (auto alloca = dyn_cast<memref::AllocaOp>(op))
//         return alloca.getMemref();
//       if (auto galloc = dyn_cast<gpu::AllocOp>(op))
//         return galloc.getMemref();
//       return Value();
//     };

//     bool changed = true;
//     while (changed) {
//       changed = false;

//       // Collect this round's dead allocations from a read-only scan first:
//       // erasing a user while `block` is still being iterated could free the
//       // op an in-flight iterator is already holding onto as "next".
//       SmallVector<Operation *> dead;
//       for (Operation &opRef : block) {
//         Value memref = getAllocatedMemref(&opRef);
//         if (!memref)
//           continue;

//         bool hasRealRead = llvm::any_of(memref.getUsers(), [&](Operation
//         *user) {
//           if (auto copy = dyn_cast<memref::CopyOp>(user))
//             return copy.getSource() == memref;
//           if (isa<enzyme::FillZeroOp, memref::DeallocOp,
//           gpu::DeallocOp>(user))
//             return false;
//           return true;
//         });
//         if (!hasRealRead)
//           dead.push_back(&opRef);
//       }

//       for (Operation *op : dead) {
//         Value memref = getAllocatedMemref(op);
//         SmallVector<Operation *> users(memref.getUsers());
//         for (Operation *user : users)
//           user->erase();
//         op->erase();
//         changed = true;
//       }
//     }
//   }
// }

struct GPUWrapperToKernelCallPass
    : public mlir::enzyme::impl::GPUWrapperToKernelCallPassBase<
          GPUWrapperToKernelCallPass> {
  using mlir::enzyme::impl::GPUWrapperToKernelCallPassBase<
      GPUWrapperToKernelCallPass>::GPUWrapperToKernelCallPassBase;

  LogicalResult cleanupMemrefs(func::FuncOp func) {
    SmallVector<Operation *> toDelete;
    DenseMap<Value, Value> mappings; // tensor <-> memref

    if (!func.getBody().hasOneBlock())
      return failure();

    for (auto &it : func.getBody().front()) {
      Operation *op = &it;

      // This is a nested op
      if (isa<enzyme::EnzymeOpsRemoverOpInterface>(op)) {
        return op->emitError() << "cannot yet handle nested op";
      }

      if (isa<enzyme::PlaceholderOp, gpu::AllocOp, gpu::DeallocOp>(op)) {
        toDelete.push_back(op);
        continue;
      }

      if (auto m2t = dyn_cast<enzymexla::Memref2TensorOp>(op)) {
        auto it = mappings.find(m2t.getMemref());
        toDelete.push_back(op);

        Value tensor;
        if (it == mappings.end()) {
          auto TT = cast<AutoDiffTypeInterface>(m2t.getTensor().getType());
          OpBuilder builder(op);
          tensor = TT.createNullValue(builder, m2t->getLoc());
        } else {
          tensor = it->second;
        }

        m2t.getTensor().replaceAllUsesWith(tensor);
        continue;
      }

      if (auto t2m = dyn_cast<enzymexla::Tensor2MemrefOp>(op)) {
        mappings[t2m.getMemref()] = t2m.getTensor();
        toDelete.push_back(t2m);
        continue;
      }

      if (auto memcpy = dyn_cast<memref::CopyOp>(op)) {
        auto it = mappings.find(memcpy.getSource());
        if (it == mappings.end()) {
          continue;
        }

        mappings[memcpy.getTarget()] = it->second;
        toDelete.push_back(op);
        continue;
      }

      if (auto fz = dyn_cast<enzyme::FillZeroOp>(op)) {
        auto memref = fz.getMemref();
        auto TT = cast<AutoDiffTypeInterface>(RankedTensorType::get(
            memref.getType().getShape(), memref.getType().getElementType()));

        OpBuilder builder(op);
        mappings[memref] = TT.createNullValue(builder, memref.getLoc());
        toDelete.push_back(fz);
        continue;
      }
    }

    while (!toDelete.empty()) {
      toDelete.pop_back_val()->erase();
    }

    return success();
  }

  LLVM::LLVMFuncOp convertGPUFuncToLLVMFunc(gpu::GPUFuncOp funcOp,
                                            SymbolTableCollection &collection) {

    SmallVector<Type> arguments;

    for (auto _ : funcOp.getFunctionType().getInputs()) {
      arguments.push_back(
          LLVM::LLVMPointerType::get(funcOp.getContext(), /*addressSpace=*/1));
    }

    assert(funcOp.getFunctionType().getResults().empty());

    auto fnType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(funcOp->getContext()), arguments);

    auto moduleOp = funcOp->getParentOfType<gpu::GPUModuleOp>();

    SmallVector<char> buf;
    auto fnName = moduleOp.getName() + "::" + funcOp.getName() + "_to_llvm";

    OpBuilder builder(funcOp->getParentOp());
    auto fn = LLVM::LLVMFuncOp::create(builder, funcOp->getLoc(),
                                       fnName.toStringRef(buf), fnType);
    fn.setPrivate();

    Block *entry = fn.addEntryBlock(builder);
    Block *gpuBody = &funcOp.getBody().front();

    builder.setInsertionPointToEnd(entry);

    IRMapping mapping;
    for (auto [argument, gpu_argument] :
         llvm::zip_equal(entry->getArguments(), gpuBody->getArguments())) {
      Value memref = enzymexla::Pointer2MemrefOp::create(
          builder, argument.getLoc(), gpu_argument.getType(), argument);

      mapping.map(gpu_argument, memref);
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
    target.addIllegalOp<enzyme::AtomicRMWOp>();
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

  LogicalResult emitAlternatives(enzymexla::AlternativesOp alt,
                                 SymbolTableCollection &symbolTables) {
    Region &content = alt.getRegions().front();

    if (!content.hasOneBlock()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "skipping alternative: region 0 does not have exactly "
                    "one block\n");
      return failure();
    }

    Block *entry = &content.front();

    enzymexla::GPUErrorOp errOp =
        dyn_cast<enzymexla::GPUErrorOp>(&entry->front());
    if (!errOp) {
      LLVM_DEBUG(llvm::dbgs() << "skipping alternative: first op in region 0 "
                                 "is not a gpu_error, got "
                              << entry->front() << "\n");
      return failure();
    }

    auto launchOp =
        dyn_cast<gpu::LaunchFuncOp>(&errOp.getRegion().front().front());
    if (!launchOp) {
      LLVM_DEBUG(llvm::dbgs() << "skipping alternative: first op in "
                                 "gpu_error is not a gpu.launch_func, got "
                              << errOp.getRegion().front().front() << "\n");
      return failure();
    }

    OpBuilder builder(alt);

    SmallVector<Value> kernelCallOperands;
    SmallVector<Type> resultTypes;
    for (auto [i, koperand] : llvm::enumerate(launchOp.getKernelOperands())) {
      auto MT = dyn_cast<MemRefType>(koperand.getType());
      if (!MT)
        return launchOp->emitError()
               << "kernel operand #" << i + 1 << " is not a memref, got "
               << koperand.getType();

      auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
      kernelCallOperands.push_back(enzymexla::Memref2TensorOp::create(
          builder, koperand.getLoc(), TT, koperand));
      resultTypes.push_back(TT);
    }

    auto toTensorI64 = [&](Value v) -> Value {
      llvm::APInt val;
      if (!matchPattern(v, m_ConstantInt(&val)))
        return nullptr;

      return details::makeI64Constant(v.getLoc(), builder, val.getSExtValue());
    };
    auto zeroTensorI64 = [&]() -> Value {
      return details::makeI64Constant(launchOp.getLoc(), builder, 0);
    };

    Value gridx = toTensorI64(launchOp.getGridSizeX());
    Value gridy = toTensorI64(launchOp.getGridSizeY());
    Value gridz = toTensorI64(launchOp.getGridSizeZ());
    Value blockx = toTensorI64(launchOp.getBlockSizeX());
    Value blocky = toTensorI64(launchOp.getBlockSizeY());
    Value blockz = toTensorI64(launchOp.getBlockSizeZ());
    Value shmem = launchOp.getDynamicSharedMemorySize()
                      ? toTensorI64(launchOp.getDynamicSharedMemorySize())
                      : zeroTensorI64();

    if (!gridx || !gridy || !gridz || !blockx || !blocky || !blockz || !shmem)
      return failure();

    Value clusterx, clustery, clusterz;
    if (launchOp.hasClusterSize()) {
      clusterx = toTensorI64(launchOp.getClusterSizeX());
      clustery = toTensorI64(launchOp.getClusterSizeY());
      clusterz = toTensorI64(launchOp.getClusterSizeZ());
    }

    // Each kernel operand is mutated in place, so the i-th result aliases the
    // i-th operand.
    MLIRContext *ctx = alt.getContext();
    SmallVector<Attribute> aliases;
    for (auto [i, resultType] : llvm::enumerate(resultTypes))
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, /*outputTupleIndices=*/{}, /*operandIndex=*/(int64_t)i,
          /*operandTupleIndices=*/{}));

    auto gpuFuncOp = symbolTables.lookupNearestSymbolFrom<gpu::GPUFuncOp>(
        launchOp, launchOp.getKernel());
    auto fn = convertGPUFuncToLLVMFunc(gpuFuncOp, symbolTables);

    auto kernelCall = enzymexla::KernelCallOp::create(
        builder, launchOp.getLoc(), resultTypes,
        SymbolRefAttr::get(fn.getSymNameAttr()), gridx, gridy, gridz, blockx,
        blocky, blockz, shmem, clusterx, clustery, clusterz, kernelCallOperands,
        /*backend_config=*/builder.getStringAttr(""),
        /*operand_layouts=*/nullptr, /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        ArrayAttr::get(ctx, aliases), /*xla_side_effect_free=*/nullptr);

    for (auto [koperand, result] : llvm::zip_equal(launchOp.getKernelOperands(),
                                                   kernelCall.getResults())) {
      Value memref = enzymexla::Tensor2MemrefOp::create(
          builder, koperand.getLoc(), koperand.getType(), result);
      memref::CopyOp::create(builder, koperand.getLoc(), memref, koperand);
    }

    alt->erase();

    return success();
  }

  void runOnOperation() override {
    SymbolTableCollection symbolTables;

    SmallVector<enzymexla::AlternativesOp> alts;

    bool anyFailed = false;

    SmallVector<func::FuncOp> funcs;

    getOperation().walk(
        [&](enzymexla::AlternativesOp alt) { alts.push_back(alt); });
    for (enzymexla::AlternativesOp alt : alts) {
      auto func = alt->getParentOfType<func::FuncOp>();
      funcs.push_back(func);

      if (failed(emitAlternatives(alt, symbolTables))) {
        anyFailed = true;
      }
    }

    for (auto func : funcs)
      if (cleanupMemrefs(func).failed())
        signalPassFailure();

    // for (Region &region : getOperation()->getRegions()) {
    //   forwardTensorMemRefValues(region);
    //   eliminateDeadMemRefAllocations(region);
    // }

    // RewritePatternSet patterns(&getContext());
    // patterns.add<Tensor2MemrefEliminate, Memref2TensorEliminate>(
    //     &getContext());

    // (void)applyPatternsGreedily(getOperation(), std::move(patterns),
    //                             GreedyRewriteConfig().enableFolding());

    if (anyFailed)
      signalPassFailure();
  }
};

} // namespace
