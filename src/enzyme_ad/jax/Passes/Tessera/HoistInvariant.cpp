//===----------------------------------------------------------------------===//
//
// Moves calls to pure tessera ops out of loops when nothing in the loop may
// change what they read, along with a memcpy that fills what they read and the
// free of what they return. Upstream loop-invariant code motion moves only ops
// that touch no memory, and a library call reads the objects its handles
// name; whether the loop may change those objects is what property
// propagation already works out.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAHOISTINVARIANTPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

// Whether `op` frees `value`: a call to the C library's free, made directly
// or through the tessera op a header declared it as.
bool freesValue(Operation *op, Value value) {
  StringRef name;
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    if (auto callee = call.getCalleeAttr())
      name = callee.getValue();
  } else if (auto call = dyn_cast<CallOp>(op)) {
    if (auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
            call, call.getCalleeAttr().getAttr()))
      if (auto original =
              define->getAttrOfType<StringAttr>("tessera.original_name"))
        name = original.getValue();
  }
  return name == "free" && op->getNumOperands() == 1 &&
         op->getOperand(0) == value;
}

// The ops at the top of `loop`'s body that free `value`. Once what returns
// `value` moves out of the loop, they move after it.
SmallVector<Operation *> freesInLoop(Value value, LoopLikeOpInterface loop) {
  SmallVector<Operation *> frees;
  Block *body = &loop.getLoopRegions().front()->front();
  for (Operation *user : value.getUsers())
    if (user->getBlock() == body && freesValue(user, value))
      frees.push_back(user);
  return frees;
}

// Whether `op` is a call to a pure tessera op whose pointer arguments name
// objects nothing else in `loop` may change, and whose result nothing in the
// loop changes either, as every later step will use the one it returned: a
// free of it, which moves after the loop, is the only exception. A pure op
// changes no memory, and is taken to be safe to run whenever its operands are
// defined, so it may run before the loop even if the loop then doesn't.
bool readsOnlyInvariantObjects(Operation *op, LoopLikeOpInterface loop) {
  auto call = dyn_cast<CallOp>(op);
  if (!call)
    return false;
  auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
      call, call.getCalleeAttr().getAttr());
  if (!define || !define.getPure())
    return false;
  if (!llvm::all_of(call.getArgOperands(), [&](Value arg) {
        return !isa<LLVM::LLVMPointerType>(arg.getType()) ||
               objectUnchangedIn(arg, {call.getOperation()}, loop);
      }))
    return false;
  return llvm::all_of(call->getResults(), [&](Value result) {
    if (isa<IntegerType, FloatType, IndexType>(result.getType()))
      return true;
    if (!isa<LLVM::LLVMPointerType>(result.getType()))
      return false;
    SmallVector<Operation *> ignored{call};
    llvm::append_range(ignored, freesInLoop(result, loop));
    return objectUnchangedIn(result, ignored, loop);
  });
}

// Whether `loop` is known to run at least once, or a test that it does can be
// built before it.
enum class Runs { Always, IfTested, Unknown };
Runs whetherLoopRuns(Operation *loop) {
  if (auto forOp = dyn_cast<affine::AffineForOp>(loop)) {
    if (forOp.hasConstantBounds())
      return forOp.getConstantLowerBound() < forOp.getConstantUpperBound()
                 ? Runs::Always
                 : Runs::Unknown;
    return forOp.getLowerBoundMap().getNumResults() == 1 &&
                   forOp.getUpperBoundMap().getNumResults() == 1
               ? Runs::IfTested
               : Runs::Unknown;
  }
  return isa<scf::ForOp>(loop) ? Runs::IfTested : Runs::Unknown;
}

// A test, built before `loop`, that it runs at least once. Only for a loop
// whetherLoopRuns says can be tested.
Value buildRunsTest(OpBuilder &builder, Operation *loop) {
  Location loc = loop->getLoc();
  if (auto forOp = dyn_cast<affine::AffineForOp>(loop)) {
    Value lb = affine::AffineApplyOp::create(
        builder, loc, forOp.getLowerBoundMap(), forOp.getLowerBoundOperands());
    Value ub = affine::AffineApplyOp::create(
        builder, loc, forOp.getUpperBoundMap(), forOp.getUpperBoundOperands());
    return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, lb,
                                 ub);
  }
  auto forOp = cast<scf::ForOp>(loop);
  return arith::CmpIOp::create(builder, loc,
                               forOp.getUnsignedCmp()
                                   ? arith::CmpIPredicate::ult
                                   : arith::CmpIPredicate::slt,
                               forOp.getLowerBound(), forOp.getUpperBound());
}

// Whether `op` is a memcpy that writes the same thing on every iteration of
// `loop`: nothing else in the loop may change its source or its destination.
// It may move out of the loop, under a test that the loop runs, since a loop
// that doesn't run wouldn't have written.
bool isInvariantCopy(Operation *op, LoopLikeOpInterface loop) {
  auto copy = dyn_cast<LLVM::MemcpyOp>(op);
  if (!copy || copy.getIsVolatile())
    return false;
  if (whetherLoopRuns(loop) == Runs::Unknown)
    return false;
  return objectUnchangedIn(copy.getSrc(), {copy.getOperation()}, loop) &&
         objectUnchangedIn(copy.getDst(), {copy.getOperation()}, loop);
}

// The pure ops in `loop` that compute, directly or not, an operand of a tessera
// call or memcpy in it. These may move out so that the call or copy can. Other
// ops are left to the pipeline's own loop-invariant code motion.
llvm::SmallPtrSet<Operation *, 8>
operandsOfCandidates(LoopLikeOpInterface loop) {
  llvm::SmallPtrSet<Operation *, 8> needed;
  SmallVector<Value> worklist;
  for (Region *region : loop.getLoopRegions())
    region->walk([&](Operation *op) {
      if (isa<CallOp, LLVM::MemcpyOp>(op))
        llvm::append_range(worklist, op->getOperands());
    });
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || !loop->isProperAncestor(def) || !isPure(def) ||
        !needed.insert(def).second)
      continue;
    llvm::append_range(worklist, def->getOperands());
  }
  return needed;
}

struct TesseraHoistInvariantPass
    : public enzyme::tessera::impl::TesseraHoistInvariantPassBase<
          TesseraHoistInvariantPass> {
  using TesseraHoistInvariantPassBase::TesseraHoistInvariantPassBase;

  void runOnOperation() override {
    // Inner loops first, so what leaves an inner loop can leave the outer.
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      SmallVector<Operation *> movedCalls;
      llvm::SmallPtrSet<Operation *, 8> needed = operandsOfCandidates(loop);
      moveLoopInvariantCode(
          loop.getLoopRegions(),
          [&](Value value, Region *) {
            return loop.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *) {
            return needed.contains(op) || readsOnlyInvariantObjects(op, loop) ||
                   isInvariantCopy(op, loop);
          },
          [&](Operation *op, Region *) {
            if (!isa<LLVM::MemcpyOp>(op) ||
                whetherLoopRuns(loop) == Runs::Always) {
              loop.moveOutOfLoop(op);
              if (isa<CallOp>(op))
                movedCalls.push_back(op);
              return;
            }
            OpBuilder builder(loop);
            auto ifOp = scf::IfOp::create(builder, loop->getLoc(),
                                          buildRunsTest(builder, loop),
                                          /*withElseRegion=*/false);
            op->moveBefore(ifOp.thenBlock()->getTerminator());
          });
      // What a moved call returns is now made once, so it's freed once,
      // after the loop.
      for (Operation *call : movedCalls)
        for (Value result : call->getResults())
          for (Operation *free : freesInLoop(result, loop))
            free->moveAfter(loop);
    });
  }
};

} // namespace
