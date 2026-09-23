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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Mem2Reg.h"

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
    if (auto def = op.getTensor().getDefiningOp<enzymexla::Memref2TensorOp>()) {
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
    if (auto def = op.getMemref().getDefiningOp<enzymexla::Tensor2MemrefOp>()) {
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

// pointer2memref(memref2pointer(x)) -> x, for every pointer2memref user
// that recovers x's exact memref type.
struct Memref2PointerForward
    : public OpRewritePattern<enzymexla::Memref2PointerOp> {
  using OpRewritePattern<enzymexla::Memref2PointerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::Memref2PointerOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    SmallVector<enzymexla::Pointer2MemrefOp> users;
    bool allChanged = true;

    for (Operation *user : op.getResult().getUsers()) {
      if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(user)) {
        auto RT = cast<MemRefType>(p2m.getResult().getType());
        auto ST = cast<MemRefType>(source.getType());
        if (RT == ST || (!RT.hasStaticShape() && ST.hasStaticShape() &&
                         RT.getRank() == 1)) {
          users.push_back(p2m);
          continue;
        }
      }
      allChanged = false;
    }

    if (users.empty())
      return failure();

    for (auto p2m : users)
      rewriter.replaceOp(p2m, source);

    if (allChanged)
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

  // Folds every enzymexla.tensor2memref this function's gpu_error scaffolding
  // created into the tensor value it was given, forwarding it to matching
  // enzymexla.memref2tensor reads (and memref.copy stores) via MLIR's
  // Mem2Reg machinery instead of a bespoke DenseMap-based scan: promotion
  // threads a slot through nested regions (e.g. a stablehlo.while, see
  // WhileOpMemorySlotPromotion) with proper dominance handling and
  // fixed-point retries for free, and simply leaves any slot it cannot fully
  // promote (a real, non-scratch memref, or one touched by a construct that
  // doesn't implement the relevant Promotable*Interface) untouched rather
  // than erroring -- only Tensor2MemrefOp results are ever registered as
  // allocators here, so no other memref in the function is a candidate.
  // True for scratch scaffolding this pass is willing to delete outright
  // once nothing uses it any more: placeholders/allocations that never carry
  // meaning on their own, and this pass' own tensor/memref/pointer bridging
  // ops (some of which -- see Memref2PointerOp and memref.copy's
  // PromotableMemOpInterface impls -- get freshly recreated as a promoted
  // slot's blocking use is redirected, and can end up unused if their own
  // sole consumer is independently eliminated later).
  static bool isDeletableScratchOp(Operation *op) {
    return isa<enzyme::PlaceholderOp, gpu::AllocOp, gpu::DeallocOp,
               enzymexla::Tensor2MemrefOp, enzymexla::Memref2TensorOp,
               enzymexla::Memref2PointerOp>(op);
  }

  LogicalResult cleanupMemrefs(func::FuncOp func) {
    SmallVector<PromotableAllocationOpInterface> allocators;
    func.walk([&](enzymexla::Tensor2MemrefOp op) {
      allocators.push_back(
          cast<PromotableAllocationOpInterface>(op.getOperation()));
    });

    if (!allocators.empty()) {
      DominanceInfo dominance(func);
      DataLayout dataLayout(func->getParentOfType<ModuleOp>());
      OpBuilder builder(&func.getBody().front(),
                        func.getBody().front().begin());
      (void)tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance);
    }

    // Sweep away anything left with no remaining uses. This is a plain
    // worklist DCE fixed point (deleting a dead op can make the op that fed
    // one of its own operands dead in turn), not another mem2reg pass, so
    // unlike re-running promotion itself it is cheap and always terminates:
    // a redirected blocking use that has no real destination to redirect to
    // (e.g. a memref.copy into a real, non-promotable memref) recreates an
    // equivalent tensor2memref every time it is reconsidered, which would
    // make looping promotion to a fixed point spin forever instead of
    // converging.
    SmallVector<Operation *> worklist;
    func.walk([&](Operation *op) {
      if (isDeletableScratchOp(op))
        worklist.push_back(op);
    });
    // The same op can land on the worklist twice (once from the walk above,
    // again as a cascade target of two different now-dead consumers) before
    // either entry is popped; erasing it twice is a use-after-free, so track
    // what has already been resolved (erased or found still live) instead of
    // processing every entry blindly.
    DenseSet<Operation *> resolved;
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!resolved.insert(op).second)
        continue;
      if (!op->use_empty())
        continue;
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp())
          if (isDeletableScratchOp(def) && !resolved.contains(def))
            worklist.push_back(def);
      op->erase();
    }

    return success();
  }

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

  LogicalResult emitErrorOp(enzymexla::GPUErrorOp errOp,
                            SymbolTableCollection &symbolTables) {
    auto launchOp =
        dyn_cast<gpu::LaunchFuncOp>(&errOp.getRegion().front().front());
    if (!launchOp) {
      LLVM_DEBUG(llvm::dbgs() << "skipping alternative: first op in "
                                 "gpu_error is not a gpu.launch_func, got "
                              << errOp.getRegion().front().front() << "\n");
      return failure();
    }

    OpBuilder builder(errOp);

    SmallVector<Value> kernelCallOperands;
    SmallVector<Type> resultTypes;

    // The memref actually backing each kernel operand: either the operand
    // itself, or -- for an operand that reached the launch_func as a raw
    // pointer (see the memref2pointer walk below) -- the memref that pointer
    // was cast from.
    SmallVector<Value> operandMemrefs;
    for (auto [i, koperand] : llvm::enumerate(launchOp.getKernelOperands())) {
      Value kernelOperand = nullptr;
      Value memrefOperand = koperand;

      auto MT = dyn_cast<MemRefType>(koperand.getType());

      if (!MT) {
        // Enzyme-AD kernels can capture a memref2pointer cast of a memref
        // rather than the memref itself (its body wants a bare pointer).
        // Walk back through that cast to recover the underlying memref so
        // this pass can still round-trip its shape through the kernel_call.
        auto m2p = koperand.getDefiningOp<enzymexla::Memref2PointerOp>();
        if (!m2p)
          return launchOp->emitError()
                 << "kernel operand #" << i + 1 << " is not a memref, got "
                 << koperand.getType();
        memrefOperand = m2p.getSource();
        MT = cast<MemRefType>(memrefOperand.getType());
      }

      while (true) {
        if (auto p2m =
                memrefOperand.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
          if (auto m2p = p2m.getSource()
                             .getDefiningOp<enzymexla::Memref2PointerOp>()) {
            memrefOperand = m2p.getSource();
            continue;
          }
        }

        if (auto t2m =
                memrefOperand.getDefiningOp<enzymexla::Tensor2MemrefOp>();
            t2m && memrefOperand.hasOneUse()) {
          kernelOperand = t2m.getTensor();
          memrefOperand = nullptr;
          break;
        }

        break;
      }

      if (!kernelOperand) {
        auto TT = RankedTensorType::get(MT.getShape(), MT.getElementType());
        kernelOperand = enzymexla::Memref2TensorOp::create(
            builder, koperand.getLoc(), TT, memrefOperand);
      }

      auto TT = cast<RankedTensorType>(kernelOperand.getType());

      kernelCallOperands.push_back(kernelOperand);
      resultTypes.push_back(TT);
      operandMemrefs.push_back(memrefOperand);
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
    MLIRContext *ctx = errOp.getContext();
    SmallVector<Attribute> aliases;
    for (auto [i, resultType] : llvm::enumerate(resultTypes))
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, /*outputTupleIndices=*/{}, /*operandIndex=*/(int64_t)i,
          /*operandTupleIndices=*/{}));

    auto gpuFuncOp = symbolTables.lookupNearestSymbolFrom<gpu::GPUFuncOp>(
        launchOp, launchOp.getKernel());
    auto fn = convertGPUFuncToLLVMFunc(gpuFuncOp, symbolTables,
                                       launchOp.getKernelOperands());

    auto kernelCall = enzymexla::KernelCallOp::create(
        builder, launchOp.getLoc(), resultTypes,
        SymbolRefAttr::get(fn.getSymNameAttr()), gridx, gridy, gridz, blockx,
        blocky, blockz, shmem, clusterx, clustery, clusterz, kernelCallOperands,
        /*backend_config=*/builder.getStringAttr(""),
        /*operand_layouts=*/nullptr, /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        ArrayAttr::get(ctx, aliases), /*xla_side_effect_free=*/nullptr);

    for (auto [memrefOperand, result] :
         llvm::zip_equal(operandMemrefs, kernelCall.getResults())) {
      if (!memrefOperand)
        continue;
      Value memref = enzymexla::Tensor2MemrefOp::create(
          builder, memrefOperand.getLoc(), memrefOperand.getType(), result);
      memref::CopyOp::create(builder, memrefOperand.getLoc(), memref,
                             memrefOperand);
    }

    errOp->erase();

    return success();
  }

  void runOnOperation() override {
    SymbolTableCollection symbolTables;

    SmallVector<enzymexla::GPUErrorOp> errs;

    bool anyFailed = false;

    // A func can contain many GPUErrorOps (one per kernel); cleanupMemrefs
    // walks and mem2reg's the whole function, so it must run once per
    // function, not once per error op it happened to contain.
    llvm::SetVector<func::FuncOp> funcs;

    getOperation().walk(
        [&](enzymexla::GPUErrorOp alt) { errs.push_back(alt); });
    for (enzymexla::GPUErrorOp err : errs) {
      auto func = err->getParentOfType<func::FuncOp>();
      funcs.insert(func);

      if (failed(emitErrorOp(err, symbolTables))) {
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
