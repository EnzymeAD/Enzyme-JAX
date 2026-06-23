#include "Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fuse-jit-calls"

namespace mlir {
namespace enzyme {

#define GEN_PASS_DEF_FUSEJITCALLSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

// Trace the request operand of Wait/Waitall back through the StableHLO
// request-array materialization to the async MPI calls that produced it.
static SmallVector<enzymexla::JITCallOp> traceRequestHandles(Value requestVal) {
  SmallVector<enzymexla::JITCallOp> result;
  SmallVector<Value> worklist = {requestVal};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (Operation *defOp = current.getDefiningOp()) {
      if (auto concatOp = dyn_cast<stablehlo::ConcatenateOp>(defOp)) {
        auto operands = concatOp.getOperands();
        for (int i = operands.size() - 1; i >= 0; --i) {
          worklist.push_back(operands[i]);
        }
      } else if (auto bcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(defOp)) {
        worklist.push_back(bcastOp.getOperand());
      } else if (auto jitCall = dyn_cast<enzymexla::JITCallOp>(defOp)) {
        StringRef fnName = jitCall.getFn().getRootReference().getValue();
        if (fnName.contains("MPI_Irecv") || fnName.contains("MPI_Isend")) {
          result.push_back(jitCall);
        }
      }
    }
  }
  return result;
}

// Rewrite pattern that matches MPI_Wait and MPI_Waitall
struct FuseJITCallsPattern : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::JITCallOp waitCall,
                                PatternRewriter &rewriter) const override {
    StringRef waitFnName = waitCall.getFn().getRootReference().getValue();
    if (!waitFnName.contains("MPI_Wait") && !waitFnName.contains("MPI_Waitall"))
      return failure();

    if (waitCall.getNumOperands() == 0)
      return failure();

    // MPI Wait/Waitall wrappers take the request value as their last operand.
    Value requestVal = waitCall.getOperand(waitCall.getNumOperands() - 1);

    SmallVector<enzymexla::JITCallOp> asyncCalls =
        traceRequestHandles(requestVal);
    if (asyncCalls.empty())
      return failure();

    auto module = waitCall->getParentOfType<ModuleOp>();
    auto waitFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(waitFnName);
    if (!waitFunc || waitFunc.empty())
      return failure();

    enzymexla::JITCallOp firstAsyncCall = asyncCalls.front();
    StringRef asyncFnName =
        firstAsyncCall.getFn().getRootReference().getValue();
    auto asyncFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(asyncFnName);
    if (!asyncFunc || asyncFunc.empty())
      return failure();

    std::string fusedName = asyncFnName.str() + "_" + waitFnName.str();

    // Collect the non-request inputs needed by the async calls and the wait.
    SmallVector<Value> fusedInputs;

    // The request is materialized inside the fused wrapper, so it is not an
    // external result of the fused jit_call.
    llvm::SmallPtrSet<Value, 4> requestHandles;
    for (auto call : asyncCalls) {
      if (call.getNumResults() > 0) {
        requestHandles.insert(call.getResult(call.getNumResults() - 1));
      }
    }

    for (auto call : asyncCalls) {
      for (Value operand : call.getOperands()) {
        if (llvm::find(fusedInputs, operand) == fusedInputs.end()) {
          fusedInputs.push_back(operand);
        }
      }
    }

    for (auto [idx, operand] : llvm::enumerate(waitCall.getOperands())) {
      if (idx == waitCall.getNumOperands() - 1)
        continue;
      if (llvm::find(fusedInputs, operand) == fusedInputs.end()) {
        fusedInputs.push_back(operand);
      }
    }

    // JIT wrapper arguments are lowered as LLVM pointers.
    SmallVector<Type> llvmArgTypes(
        fusedInputs.size(), LLVM::LLVMPointerType::get(rewriter.getContext()));

    auto fusedFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(fusedName);
    if (!fusedFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(waitFunc);

      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(rewriter.getContext()), llvmArgTypes);
      fusedFunc = rewriter.create<LLVM::LLVMFuncOp>(waitCall.getLoc(),
                                                    fusedName, funcType);

      Block *fusedBlock = fusedFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToEnd(fusedBlock);

      IRMapping mapping;

      // Keep request handles local to the fused wrapper; the outer StableHLO
      // graph only observes the data buffers after the wait has completed.
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);
      auto numRequests = rewriter.create<LLVM::ConstantOp>(
          waitCall.getLoc(), i32Type,
          rewriter.getI32IntegerAttr(asyncCalls.size()));
      Value localReqArray = rewriter.create<LLVM::AllocaOp>(
          waitCall.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
          i32Type, numRequests, /*alignment=*/0);

      for (size_t i = 0; i < asyncCalls.size(); ++i) {
        auto call = asyncCalls[i];

        for (auto [argIdx, arg] : llvm::enumerate(asyncFunc.getArguments())) {
          if (argIdx == asyncFunc.getNumArguments() - 1) {
            auto offset = rewriter.create<LLVM::ConstantOp>(
                waitCall.getLoc(), i32Type, rewriter.getI32IntegerAttr(i));
            Value reqPtr = rewriter.create<LLVM::GEPOp>(
                waitCall.getLoc(),
                LLVM::LLVMPointerType::get(rewriter.getContext()), i32Type,
                localReqArray, ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(offset)});
            mapping.map(arg, reqPtr);
          } else if (argIdx < call.getNumOperands()) {
            Value originalOperand = call.getOperand(argIdx);
            auto it = llvm::find(fusedInputs, originalOperand);
            size_t fusedIdx = std::distance(fusedInputs.begin(), it);
            mapping.map(arg, fusedFunc.getArgument(fusedIdx));
          }
        }

        for (auto &op : asyncFunc.front().without_terminator()) {
          rewriter.clone(op, mapping);
        }
      }

      // The wait wrapper consumes the local request array after all async calls
      // have populated it.
      for (auto [argIdx, arg] : llvm::enumerate(waitFunc.getArguments())) {
        if (argIdx == waitFunc.getNumArguments() - 1) {
          mapping.map(arg, localReqArray);
        } else if (argIdx < waitCall.getNumOperands() - 1) {
          Value originalOperand = waitCall.getOperand(argIdx);
          auto it = llvm::find(fusedInputs, originalOperand);
          size_t fusedIdx = std::distance(fusedInputs.begin(), it);
          mapping.map(arg, fusedFunc.getArgument(fusedIdx));
        }
      }

      for (auto &op : waitFunc.front().without_terminator()) {
        rewriter.clone(op, mapping);
      }

      rewriter.create<LLVM::ReturnOp>(waitCall.getLoc(), ValueRange{});
    }

    SmallVector<Type> fusedResultTypes;
    DenseMap<Value, int> originalToFusedResultIdx;

    for (auto call : asyncCalls) {
      for (auto res : call.getResults()) {
        if (!requestHandles.count(res)) {
          originalToFusedResultIdx[res] = fusedResultTypes.size();
          fusedResultTypes.push_back(res.getType());
        }
      }
    }

    auto newCall = rewriter.create<enzymexla::JITCallOp>(
        waitCall.getLoc(), fusedResultTypes,
        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), fusedName),
        fusedInputs, StringAttr::get(rewriter.getContext(), ""),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*xla_side_effect_free=*/nullptr);

    // Replace only data results; request results disappear with the wait.
    for (auto call : asyncCalls) {
      for (auto res : call.getResults()) {
        if (!requestHandles.count(res)) {
          int mappedIdx = originalToFusedResultIdx[res];
          rewriter.replaceAllUsesWith(res, newCall.getResult(mappedIdx));
        }
      }
      // Erase the original call after replacing its results.
      rewriter.eraseOp(call);
    }
    // Erase the wait call after replacing its results.
    rewriter.eraseOp(waitCall);

    return success();
  }
};

struct FuseJITCallsPass : public impl::FuseJITCallsPassBase<FuseJITCallsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FuseJITCallsPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace enzyme
} // namespace mlir
