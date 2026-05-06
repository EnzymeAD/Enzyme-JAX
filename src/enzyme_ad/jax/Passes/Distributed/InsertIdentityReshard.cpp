#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace enzyme {
namespace distributed {

#define GEN_PASS_DEF_INSERTIDENTITYRESHARDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

bool isTensorOperand(Value value) { return isa<ShapedType>(value.getType()); }

LogicalResult insertIdentityReshardForFunction(func::FuncOp funcOp) {
  llvm::SmallVector<Operation *> operations;
  funcOp.walk([&](Operation *op) {
    if (!isa<sdy::ReshardOp>(op)) {
      operations.push_back(op);
    }
  });

  IRRewriter rewriter(funcOp.getContext());
  for (Operation *op : operations) {
    if (!op || !op->getBlock()) {
      continue;
    }

    rewriter.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      if (!isTensorOperand(value)) {
        continue;
      }

      FailureOr<sdy::TensorShardingAttr> sharding = sdy::getSharding(value);
      if (failed(sharding)) {
        op->emitError()
            << "expected every tensor operand to carry explicit sharding";
        return failure();
      }

      if (value.getDefiningOp<sdy::ReshardOp>()) {
        continue;
      }

      auto reshard =
          rewriter.create<sdy::ReshardOp>(op->getLoc(), value, *sharding);
      operand.set(reshard.getResult());
    }
  }

  return success();
}

struct InsertIdentityReshardPass
    : public enzyme::distributed::impl::InsertIdentityReshardPassBase<
          InsertIdentityReshardPass> {
  using InsertIdentityReshardPassBase::InsertIdentityReshardPassBase;

  void runOnOperation() override {
    if (failed(insertIdentityReshardForFunction(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace distributed
} // namespace enzyme
} // namespace mlir