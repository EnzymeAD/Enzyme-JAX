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
    auto moduleOp = getOperation();
    auto *ctx = moduleOp->getContext();
    OpBuilder builder(ctx);
    SymbolTable symTable(moduleOp);

    // Find all kernels and their enzymexla.kernel_call and enzymexla.jit_call
    // callers
    DenseMap<FunctionOpInterface, SetVector<CallOpInterface>> funcToKernelMap;
    moduleOp->walk([&](enzymexla::JITCallOp callOp) {
      auto symbolName =
          dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
      auto callee =
          symTable.lookup<FunctionOpInterface>(symbolName.getLeafReference());
      if (!callee)
        return;
      funcToKernelMap[callee].insert(callOp);
    });
    moduleOp->walk([&](enzymexla::KernelCallOp callOp) {
      auto symbolName =
          dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
      auto callee =
          symTable.lookup<FunctionOpInterface>(symbolName.getLeafReference());
      if (!callee)
        return;
      funcToKernelMap[callee].insert(callOp);
    });
    if (funcToKernelMap.empty())
      return;

    // Recover memref types for callees
    for (auto [callee, callers] : funcToKernelMap) {
      SmallVector<Type> newTypes;
      SmallVector<unsigned> indices;
      for (auto [index, calleeArgTy, calleeArg] :
           llvm::enumerate(callee.getArgumentTypes(), callee.getArguments())) {
        assert(!callers.empty());
        if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(calleeArgTy)) {
          bool sameElementTypeAcrossCallers = true;
          bool sameShapeAcrossCallers = true;
          ArrayRef<int64_t> shape;
          Type elTy;
          // A kernel can be called by multiple KernelCallOps
          for (auto caller : callers) {
            Value callerArg = caller.getArgOperands()[index];
            Type callerArgTy = callerArg.getType();

            if (auto rtt = dyn_cast<RankedTensorType>(callerArgTy)) {
              auto thisShape = rtt.getShape();
              auto thisElTy = rtt.getElementType();
              if (!elTy) {
                elTy = thisElTy;
                shape = thisShape;
              } else if (shape != thisShape) {
                sameShapeAcrossCallers = false;
                break;
              } else if (elTy != thisElTy) {
                sameElementTypeAcrossCallers = false;
                break;
              }
            } else {
              sameElementTypeAcrossCallers = false;
              break;
            }
          }
          // If the element types differ across callers, we need to be
          // conservative and keep the llvm.ptr type.
          if (sameElementTypeAcrossCallers) {
            Attribute addrSpace;
            if (ptrTy.getAddressSpace() == 0)
              addrSpace = nullptr;
            else
              addrSpace =
                  IntegerAttr::get(IntegerType::get(calleeArg.getContext(), 64),
                                   ptrTy.getAddressSpace());

            // If the shapes differ across callers, we need to be conservative
            // and convert to dynamic shape
            MemRefType memrefTy = MemRefType::get(
                sameShapeAcrossCallers
                    ? shape
                    : SmallVector<int64_t>{ShapedType::kDynamic},
                elTy,
                // TODO do we need a layout?
                MemRefLayoutAttrInterface{}, Attribute(addrSpace));

            newTypes.push_back(memrefTy);
            indices.push_back(index);
          } else {
            newTypes.push_back(calleeArgTy);
          }
        }
      }

      if (indices.empty())
        continue;

      // Rewrite function signatures and body operations for all affected
      // kernels
      FunctionType newFuncTy = nullptr;
      if (auto fty =
              dyn_cast<LLVM::LLVMFunctionType>(callee.getFunctionType())) {
        if (fty.getReturnType() == LLVM::LLVMVoidType::get(ctx) &&
            !fty.isVarArg()) {
          newFuncTy = FunctionType::get(ctx, newTypes, fty.getReturnType());
        }
      } else if (auto fty = dyn_cast<FunctionType>(callee.getFunctionType())) {
        if (fty.getResults().empty()) {
          newFuncTy = FunctionType::get(ctx, newTypes, {});
        }
      }
      if (newFuncTy) {
        // Update entry block's argument types
        Block *entry = &callee.getFunctionBody().front();
        builder.setInsertionPointToStart(entry);
        for (auto index : indices) {
          auto newType = newTypes[index];
          auto oldArg = callee.getArgument(index);
          auto newArg = entry->insertArgument(index, newType, oldArg.getLoc());
          auto newPtr = builder.create<enzymexla::Memref2PointerOp>(
              newArg.getLoc(), oldArg.getType(), newArg);
          oldArg.replaceAllUsesWith(newPtr);
          entry->eraseArgument(index + 1);
        }

        builder.setInsertionPoint(callee);
        auto newFunc = builder.create<func::FuncOp>(
            callee.getLoc(), callee.getName(), newFuncTy);
        newFunc.getBlocks().splice(newFunc.getBlocks().begin(),
                                   callee.getFunctionBody().getBlocks());

        // Copy attributes from the old function to the new one
        for (auto attr : callee->getAttrs()) {
          newFunc->setAttr(attr.getName(), attr.getValue());
        }
        // Fix function_type attr because the above copying overwrote it with
        // the old one
        newFunc->setAttr("function_type", TypeAttr::get(newFuncTy));

        callee->erase();
      }
    }
  }
};
