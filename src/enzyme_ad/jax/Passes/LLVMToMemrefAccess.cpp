#include "Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#define DEBUG_TYPE "llvm-to-memref-access"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LLVMTOMEMREFACCESSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

using PtrVal = TypedValue<LLVM::LLVMPointerType>;

namespace mlir {
MemRefType recoverMemRefTypeFromKernelCall(enzymexla::KernelCallOp kernelCallOp,
                                           unsigned argIndex) {
  auto tensorType =
      kernelCallOp.getInputs()[argIndex].getType().dyn_cast<RankedTensorType>();
  if (!tensorType) {
    return nullptr;
  }
  // Convert tensor type to memref type
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
}

} // namespace mlir

struct LLVMToMemrefAccessPass
    : public enzyme::impl::LLVMToMemrefAccessPassBase<LLVMToMemrefAccessPass> {
  using LLVMToMemrefAccessPassBase::LLVMToMemrefAccessPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    SymbolTableCollection symbolTableCollection;
    SymbolTable symbolTable = symbolTableCollection.getSymbolTable(op);

    MLIRContext *context = &getContext();

    // Find all enzymexla.kernel_call ops and recover memref types for callees
    op->walk([&](enzymexla::KernelCallOp kernelCallOp) {
      auto *symbolOp = symbolTableCollection.lookupNearestSymbolFrom(
          kernelCallOp, kernelCallOp.getFnAttr());
      auto fn = cast<FunctionOpInterface>(symbolOp);

      // 1. Recover memref types for all arguments of the callee
      SmallVector<Type, 4> newArgTypes;
      // Tracks which arguments are updated
      SmallVector<bool, 4> isMemrefArg;
      for (unsigned i = 0; i < kernelCallOp.getInputs().size(); ++i) {
        MemRefType memrefType =
            recoverMemRefTypeFromKernelCall(kernelCallOp, i);
        if (!memrefType)
          newArgTypes.push_back(fn.getArguments()[i].getType());
        else
          newArgTypes.push_back(memrefType);
        isMemrefArg.push_back(memrefType != nullptr);
      }

      // 2. Rewrite function signatures and body operations for all affected
      // functions
      auto oldFunc = dyn_cast<LLVM::LLVMFuncOp>(symbolOp);
      if (!oldFunc)
        return;

      // 3. Create a new function with updated argument types
      OpBuilder builder(oldFunc);
      auto newFuncType = FunctionType::get(
          context, newArgTypes, oldFunc.getFunctionType().getReturnType());
      auto newFunc = builder.create<func::FuncOp>(oldFunc.getLoc(),
                                                  fn.getName(), newFuncType);

      // 4. Copy attributes from the old function to the new one
      for (auto attr : oldFunc->getAttrs()) {
        newFunc->setAttr(attr.getName(), attr.getValue());
      }
      newFunc->setAttr("function_type", TypeAttr::get(newFuncType));

      // 5. Create all blocks in the new function
      IRMapping mapper;

      // 5.1. Create a new entry block and remap arguments
      auto *newEntryBlock = builder.createBlock(
          &newFunc.getBody(),
          /*insertBefore=*/newFunc.getBody().end(), newArgTypes,
          llvm::SmallVector<mlir::Location>(newArgTypes.size(),
                                            oldFunc.getLoc()));
      builder.setInsertionPointToStart(newEntryBlock);
      for (auto [oldArg, newArg, isMemref] :
           llvm::zip(oldFunc.getArguments(), newEntryBlock->getArguments(),
                     isMemrefArg)) {
        if (isMemref) {
          // Insert memref2ptr operation for memref arguments
          auto memref2ptr = builder.create<enzymexla::Memref2PointerOp>(
              oldFunc.getLoc(), oldArg.getType(), newArg);
          // Map old argument to memref2ptr result
          mapper.map(oldArg, memref2ptr);
        } else {
          // Directly map unchanged arguments
          mapper.map(oldArg, newArg);
        }
      }
      auto &oldEntryBlock = oldFunc.getBody().front();
      for (Operation &op : oldEntryBlock) {
        Operation *clonedOp = builder.clone(op, mapper);
        for (auto [oldResult, newResult] :
             llvm::zip(op.getResults(), clonedOp->getResults())) {
          // Map results of the original op to the cloned op
          mapper.map(oldResult, newResult);
        }
      }

      // 5.2. Move all non-entry blocks from old function to the new function
      auto &oldBlocks = oldFunc.getBody();
      auto &newBlocks = newFunc.getBody();

      for (auto &block : llvm::make_early_inc_range(oldBlocks)) {
        // Skip the entry block
        if (&block == &oldEntryBlock) {
          continue;
        }

        block.moveBefore(&newBlocks, /*insertBefore=*/newBlocks.end());

        // Remap operands in moved blocks
        for (Operation &op : block) {
          for (OpOperand &operand : op.getOpOperands()) {
            if (mapper.contains(operand.get())) {
              operand.set(mapper.lookup(operand.get()));
            }
          }
        }

        // Remap block arguments if necessary
        for (auto arg : block.getArguments()) {
          if (mapper.contains(arg)) {
            arg.replaceAllUsesWith(mapper.lookup(arg));
          }
        }
      }

      // 6. Replace all uses of the old function with the new one
      oldFunc->replaceAllUsesWith(newFunc);

      // 7. Erase the old function
      oldFunc.erase();
    });
  }
};
