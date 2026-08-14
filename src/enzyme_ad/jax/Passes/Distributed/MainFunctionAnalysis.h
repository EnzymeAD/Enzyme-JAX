#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_MAINFUNCTIONANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_MAINFUNCTIONANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir::enzyme::distributed {

// Module-scoped helper analysis that resolves the unique symbol named "main"
// across both func.func and distributed.DistributedFunction.
class FindMainFunctionAnalysis {
public:
  explicit FindMainFunctionAnalysis(ModuleOp module);

  bool isValid() const { return valid; }
  bool hasMainFunction() const { return mainFunctionOp != nullptr; }

  Operation *getMainFunctionOp() const { return mainFunctionOp; }
  func::FuncOp getMainFuncOp() const;
  DistributedFunctionOp getMainDistributedFunctionOp() const;

  // Returns the entry block only for single-block function-like mains.
  Block *getMainBlock() const;

private:
  bool valid = true;
  Operation *mainFunctionOp = nullptr;
};

} // namespace mlir::enzyme::distributed

#endif