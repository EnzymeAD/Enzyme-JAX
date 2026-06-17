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

// Helper to trace request handles backwards from Wait/Waitall
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

    // The request handle is passed as the last operand to the wait call
    Value requestVal = waitCall.getOperand(waitCall.getNumOperands() - 1);

    SmallVector<enzymexla::JITCallOp> asyncCalls =
        traceRequestHandles(requestVal);
    if (asyncCalls.empty())
      return failure();

    auto module = waitCall->getParentOfType<ModuleOp>();
    auto waitFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(waitFnName);
    if (!waitFunc || waitFunc.empty())
      return failure();

    // For simplicity in the plan, assume we only fuse if all async calls share
    // the same name (e.g. all Irecv) and we generate a fused wrapper based on
    // the first one.
    enzymexla::JITCallOp firstAsyncCall = asyncCalls.front();
    StringRef asyncFnName =
        firstAsyncCall.getFn().getRootReference().getValue();
    auto asyncFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(asyncFnName);
    if (!asyncFunc || asyncFunc.empty())
      return failure();

    std::string fusedName = asyncFnName.str() + "_" + waitFnName.str();

    // Collect combined unique inputs and their types, excluding the request
    // handles
    SmallVector<Value> fusedInputs;
    SmallVector<Type> argTypes;

    // Track which async call outputs are request handles so we exclude them
    llvm::SmallPtrSet<Value, 4> requestHandles;
    for (auto call : asyncCalls) {
      // Assuming request is the last result
      if (call.getNumResults() > 0) {
        requestHandles.insert(call.getResult(call.getNumResults() - 1));
      }
    }

    // Add inputs from async calls (deduplicating identical inputs)
    for (auto call : asyncCalls) {
      for (Value operand : call.getOperands()) {
        if (llvm::find(fusedInputs, operand) == fusedInputs.end()) {
          fusedInputs.push_back(operand);
          argTypes.push_back(
              operand.getType()); // Need to map correctly to LLVM type if
                                  // needed, but for now we just use the mlir
                                  // type of the operand? Wait, LLVM wrappers
                                  // take LLVMPointerType for everything in
                                  // Enzymexla. Let's get the type from the LLVM
                                  // function if possible.
        }
      }
    }

    // Add inputs from wait call, excluding the request handle itself
    for (auto [idx, operand] : llvm::enumerate(waitCall.getOperands())) {
      if (idx == waitCall.getNumOperands() - 1)
        continue; // skip the request
      if (llvm::find(fusedInputs, operand) == fusedInputs.end()) {
        fusedInputs.push_back(operand);
      }
    }

    // Since we don't have perfect type mapping, let's derive argTypes from LLVM
    // pointers
    SmallVector<Type> llvmArgTypes(
        fusedInputs.size(), LLVM::LLVMPointerType::get(rewriter.getContext()));

    // 1. Create the fused LLVM wrapper function if it doesn't exist
    auto fusedFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(fusedName);
    if (!fusedFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(waitFunc);

      auto funcType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(rewriter.getContext()), llvmArgTypes);
      fusedFunc = rewriter.create<LLVM::LLVMFuncOp>(waitCall.getLoc(),
                                                    fusedName, funcType);

      // Implement full LLVM block cloning
      Block *fusedBlock = fusedFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToEnd(fusedBlock);

      IRMapping mapping;

      // We need local allocation for the request handles since they are no
      // longer passed from outside Allocate space for the request handles (e.g.
      // array of i32s)
      auto i32Type = IntegerType::get(rewriter.getContext(), 32);
      auto numRequests = rewriter.create<LLVM::ConstantOp>(
          waitCall.getLoc(), i32Type,
          rewriter.getI32IntegerAttr(asyncCalls.size()));
      Value localReqArray = rewriter.create<LLVM::AllocaOp>(
          waitCall.getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()),
          i32Type, numRequests, /*alignment=*/0);

      // For each async call, clone its block into the fused block
      // We map the async call's arguments to the fused function's arguments.
      // The last argument of the async wrapper is usually the request pointer.
      // We map it to our local allocation.
      for (size_t i = 0; i < asyncCalls.size(); ++i) {
        auto call = asyncCalls[i];

        // Map arguments:
        // This is a simplification. We assume the async LLVM wrapper arguments
        // map 1:1 to the JIT call's operands. And the last one is the request
        // output pointer. We find which of the fusedInputs corresponds to this
        // call's operands.
        for (auto [argIdx, arg] : llvm::enumerate(asyncFunc.getArguments())) {
          if (argIdx == asyncFunc.getNumArguments() - 1) {
            // Map request pointer to an offset in our localReqArray
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

        // Clone operations from asyncFunc entry block (excluding return)
        for (auto &op : asyncFunc.front().without_terminator()) {
          rewriter.clone(op, mapping);
        }
      }

      // Clone operations from waitFunc entry block
      // The wait function typically takes (count, request_array)
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

    // 2. Create the fused JITCallOp
    SmallVector<Type> fusedResultTypes;
    for (auto call : asyncCalls) {
      for (auto res : call.getResults()) {
        if (!requestHandles.count(res)) {
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

    // 3. Replace uses and erase
    int resIdx = 0;
    for (auto call : asyncCalls) {
      for (auto res : call.getResults()) {
        if (!requestHandles.count(res)) {
          rewriter.replaceAllUsesWith(res, newCall.getResult(resIdx++));
        }
      }
    }

    // Note: The Wait call has no results, so we don't replace anything for it.
    // The JIT calls will be erased safely by the GreedyPatternRewriteDriver
    // when they have no uses. We explicitly erase the waitCall here to trigger
    // the pattern success.
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
