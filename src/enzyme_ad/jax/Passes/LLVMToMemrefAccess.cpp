#include "Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
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
      dyn_cast<RankedTensorType>(kernelCallOp.getInputs()[argIndex].getType());
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
    SetVector<FunctionOpInterface> candidateFunctions;
    for (auto &entry : funcToKernelMap)
      candidateFunctions.insert(entry.first);
    // The existing path discovers callees through enzymexla custom-call ops.
    // Imported LLVM kernels are standalone functions with no such callers, so
    // an explicit argument contract must also make a function a candidate.
    moduleOp->walk([&](FunctionOpInterface function) {
      for (unsigned index = 0; index < function.getNumArguments(); ++index) {
        if (function.getArgAttr(index, "enzymexla.memref_type")) {
          candidateFunctions.insert(function);
          break;
        }
      }
    });
    if (candidateFunctions.empty())
      return;

    // Recover memref types for callees
    for (auto callee : candidateFunctions) {
      auto callers = funcToKernelMap.lookup(callee);
      SmallVector<Type> newTypes;
      SmallVector<unsigned> indices;
      for (auto [index, calleeArgTy, calleeArg] :
           llvm::enumerate(callee.getArgumentTypes(), callee.getArguments())) {
        if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(calleeArgTy)) {
          if (auto typeAttr = dyn_cast_or_null<TypeAttr>(
                  callee.getArgAttr(index, "enzymexla.memref_type"))) {
            auto memrefType = dyn_cast<MemRefType>(typeAttr.getValue());
            if (!memrefType) {
              callee.emitError() << "enzymexla.memref_type"
                                 << " must contain a memref type";
              signalPassFailure();
              return;
            }
            newTypes.push_back(memrefType);
            indices.push_back(index);
            continue;
          }
          if (callers.empty()) {
            newTypes.push_back(calleeArgTy);
            continue;
          }
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

              if (isa<FloatType>(thisElTy)) {
                auto targetFloatType =
                    callee->getAttrOfType<TypeAttr>("enzymexla.float_type");
                if (targetFloatType) {
                  if (thisElTy != targetFloatType.getValue()) {
                    sameElementTypeAcrossCallers = false;
                  } else if (auto srcTyAttr = callee->getAttrOfType<TypeAttr>(
                                 "enzymexla.src_float_type")) {
                    thisElTy = srcTyAttr.getValue();
                  } else {
                    thisElTy = mlir::Float32Type::get(thisElTy.getContext());
                  }
                }
              }

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
        } else {
          newTypes.push_back(calleeArgTy);
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
          newFuncTy = FunctionType::get(ctx, newTypes, {});
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
          auto newPtr = enzymexla::Memref2PointerOp::create(
              builder, newArg.getLoc(), oldArg.getType(), newArg);
          oldArg.replaceAllUsesWith(newPtr);
          entry->eraseArgument(index + 1);
        }

        builder.setInsertionPoint(callee);
        auto newFunc = func::FuncOp::create(builder, callee.getLoc(),
                                            callee.getName(), newFuncTy);
        newFunc.getBlocks().splice(newFunc.getBlocks().begin(),
                                   callee.getFunctionBody().getBlocks());

        // Copy attributes from the old function to the new one
        for (auto attr : callee->getAttrs()) {
          newFunc->setAttr(attr.getName(), attr.getValue());
        }
        // Fix function_type attr because the above copying overwrote it with
        // the old one
        newFunc->setAttr("function_type", TypeAttr::get(newFuncTy));

        // Copy argument attributes, except for the type contract consumed by
        // this rewrite.
        for (unsigned i = 0; i < callee.getNumArguments(); ++i) {
          SmallVector<NamedAttribute> attributes;
          for (auto attribute : callee.getArgAttrs(i)) {
            if (attribute.getName() != "enzymexla.memref_type")
              attributes.push_back(attribute);
          }
          newFunc.setArgAttrs(i, DictionaryAttr::get(ctx, attributes));
        }
        callee->erase();
      }
    }
  }
};
