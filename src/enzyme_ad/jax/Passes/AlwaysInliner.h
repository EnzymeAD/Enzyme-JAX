#ifndef _POLYGEIST_PASSES_ALWAYSINLINER_H_
#define _POLYGEIST_PASSES_ALWAYSINLINER_H_

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

struct AlwaysInlinerInterface : public mlir::InlinerInterface {
  using InlinerInterface::InlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    if (!isa<mlir::func::ReturnOp, mlir::LLVM::ReturnOp>(op))
      return;

    // Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       op->getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
    // Replace the values directly with the return operands.
    assert(op->getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(op->getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

static mlir::InlinerInterface::CloneCallbackTy cloneCallback =
    [](mlir::OpBuilder &builder, mlir::Region *src, mlir::Block *inlineBlock,
       mlir::Block *postInsertBlock, mlir::IRMapping &mapper,
       bool shouldCloneInlinedRegion) {
      // Check to see if the region is being cloned, or moved inline. In
      // either case, move the new blocks after the 'insertBlock' to improve
      // IR readability.
      mlir::Region *insertRegion = inlineBlock->getParent();
      if (shouldCloneInlinedRegion)
        src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
      else
        insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                         src->getBlocks(), src->begin(),
                                         src->end());
    };

[[maybe_unused]] static void alwaysInlineCall(mlir::func::CallOp caller) {
  // Build the inliner interface.
  AlwaysInlinerInterface interface(caller.getContext());

  auto callable = caller.getCallableForCallee();
  mlir::CallableOpInterface callableOp;
  if (mlir::SymbolRefAttr symRef =
          mlir::dyn_cast<mlir::SymbolRefAttr>(callable)) {
    auto *symbolOp =
        caller->getParentOfType<mlir::ModuleOp>().lookupSymbol(symRef);
    callableOp = mlir::dyn_cast_or_null<mlir::CallableOpInterface>(symbolOp);
  } else {
    return;
  }
  mlir::Region *targetRegion = callableOp.getCallableRegion();
  if (!targetRegion)
    return;
  if (targetRegion->empty())
    return;
  if (inlineCall(interface, cloneCallback, caller, callableOp, targetRegion,
                 /*shouldCloneInlinedRegion=*/true)
          .succeeded()) {
    caller.erase();
  }
};

#endif // _POLYGEIST_PASSES_ALWAYSINLINER_H_
