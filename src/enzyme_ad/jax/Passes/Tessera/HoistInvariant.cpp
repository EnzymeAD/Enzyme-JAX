//===----------------------------------------------------------------------===//
//
// Moves calls to pure tessera ops out of loops when nothing in the loop may
// change what they read, nor what they return. Upstream loop-invariant code
// motion moves only ops that touch no memory, and a library call reads the
// objects its handles name; whether the loop may change those objects is what
// property propagation already works out.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

// Whether `op` is a call to a pure tessera op whose pointer arguments name
// objects nothing else in `loop` may change, and whose result nothing in the
// loop changes either, as every later step will use the one it returned. A
// pure op
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
    return objectUnchangedIn(result, {call.getOperation()}, loop);
  });
}

// The pure ops in `loop` that compute, directly or not, an operand of a tessera
// call in it. These may move out so that the call can. Other ops are left to
// the pipeline's own loop-invariant code motion.
llvm::SmallPtrSet<Operation *, 8>
operandsOfCandidates(LoopLikeOpInterface loop) {
  llvm::SmallPtrSet<Operation *, 8> needed;
  SmallVector<Value> worklist;
  for (Region *region : loop.getLoopRegions())
    region->walk([&](Operation *op) {
      if (isa<CallOp>(op))
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
      llvm::SmallPtrSet<Operation *, 8> needed = operandsOfCandidates(loop);
      moveLoopInvariantCode(
          loop.getLoopRegions(),
          [&](Value value, Region *) {
            return loop.isDefinedOutsideOfLoop(value);
          },
          [&](Operation *op, Region *) {
            return needed.contains(op) || readsOnlyInvariantObjects(op, loop);
          },
          [&](Operation *op, Region *) { loop.moveOutOfLoop(op); });
    });
  }
};

} // namespace
