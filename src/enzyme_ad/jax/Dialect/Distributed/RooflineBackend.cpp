#include "RooflineBackend.h"

#include "ComputeCost.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::enzyme::distributed {

LogicalResult RooflineBackend::annotate(ModuleOp kernelModule,
                                        const CostTarget &target) {
  auto kernel = SymbolTable(kernelModule).lookup<func::FuncOp>("kernel");
  assert(kernel && "a kernel module holds func.func @kernel");
  for (Operation &op : kernel.getBody().front().without_terminator())
    setLatency(&op, costOp(&op).value_or(OpCost{}).time(target.device));
  return success();
}

} // namespace mlir::enzyme::distributed
