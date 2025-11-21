#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-triton-extension-ops"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERTRITONEXTENSIONOPSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::enzymexla::triton_ext;

struct JITCallScratchMemoryLowering
    : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::JITCallOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();

    BitVector rewriteScratchMemoryIdxs(inputs.size(), false);
    SmallVector<Value> newInputs;
    bool hasScratchMemory = false;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (auto scratchMemoryOp =
              inputs[i].getDefiningOp<triton_ext::ScratchMemoryOp>()) {
        hasScratchMemory = true;
        rewriteScratchMemoryIdxs.set(i);
        continue;
      }
      newInputs.push_back(inputs[i]);
    }

    if (!hasScratchMemory)
      return failure(); // nothing to do

    // hoist the scratch memory allocation and use gpu.alloc to allocate this
    // memory in the jit call function
    auto modOp = op->getParentOfType<ModuleOp>();
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(modOp);
    auto funcOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
    if (!funcOp) {
      op->emitError("Failed to find function '") << op.getFn() << "' in module";
      return failure();
    }

    auto funcOpInterface = dyn_cast<FunctionOpInterface>(funcOp);

    auto &fnBody = funcOp->getRegion(0).front();

    for (unsigned idx : rewriteScratchMemoryIdxs.set_bits()) {
      rewriter.setInsertionPoint(&fnBody, fnBody.begin());
      auto scratchMemoryOp =
          inputs[idx].getDefiningOp<triton_ext::ScratchMemoryOp>();
      auto outTy =
          cast<RankedTensorType>(scratchMemoryOp.getResult().getType());
      assert(outTy.getRank() == 1);

      auto outMemrefType = MemRefType::get(
          outTy.getShape(), outTy.getElementType(), MemRefLayoutAttrInterface{},
          rewriter.getI64IntegerAttr(
              cast<LLVM::LLVMPointerType>(fnBody.getArgument(idx).getType())
                  .getAddressSpace()));

      auto allocOp =
          memref::AllocOp::create(rewriter, op.getLoc(), outMemrefType,
                                  scratchMemoryOp.getAlignmentAttr());
      auto ptrOp = enzymexla::Memref2PointerOp::create(
          rewriter, op.getLoc(),
          LLVM::LLVMPointerType::get(rewriter.getContext(),
                                     outMemrefType.getMemorySpaceAsInt()),
          allocOp.getResult());
      rewriter.replaceAllUsesWith(fnBody.getArgument(idx), ptrOp.getResult());

      // clang-format off
      // FIXME: This is producing
      //  error: 'llvm.call' op operand type mismatch for operand 0: '!llvm.ptr<1>' != '!llvm.ptr'
      // see current operation: "llvm.call"(%61, %60) <{CConv = #llvm.cconv<ccc>, TailCallKind = #llvm.tailcallkind<none>, callee = @mgpuMemFree, fastmathFlags = #llvm.fastmath<none>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 2, 0>}> : (!llvm.ptr<1>, !llvm.ptr) -> ()
      // SmallVector<Value> deps;
      // Operation *lastUser = ptrOp;
      // for (auto u : ptrOp->getUsers()) {
      //   if (auto gpuLaunchOp = dyn_cast<gpu::LaunchFuncOp>(u)) {
      //     deps.push_back(gpuLaunchOp.getAsyncToken());
      //   }

      //   if (lastUser->isBeforeInBlock(u)) {
      //     lastUser = u;
      //   }
      // }

      // rewriter.setInsertionPointAfter(lastUser);
      // gpu::DeallocOp::create(rewriter, op.getLoc(),
      //                        gpu::AsyncTokenType::get(rewriter.getContext()),
      //                        ValueRange(deps), allocOp.getResult());
      // clang-format on
    }

    funcOpInterface.eraseArguments(rewriteScratchMemoryIdxs);

    // TODO: to be safe we should rework the other attributes if they are being
    // removed...
    rewriter.setInsertionPoint(op);
    auto newJitCallOp = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), op.getFn(), newInputs,
        op.getBackendConfigAttr(), op.getOperandLayoutsAttr(),
        op.getResultLayoutsAttr(), op.getArgAttrsAttr(), op.getResAttrsAttr(),
        op.getOutputOperandAliasesAttr(), op.getXlaSideEffectFreeAttr());
    rewriter.replaceOp(op, newJitCallOp);
    return success();
  }
};

struct LowerTritonExtensionOpsPass
    : public mlir::enzyme::impl::LowerTritonExtensionOpsPassBase<
          LowerTritonExtensionOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    RewritePatternSet patterns(context);
    patterns.add<JITCallScratchMemoryLowering>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
