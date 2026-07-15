#ifndef ENZYME_AD_JAX_PASSES_DISTRIBUTED_FINDSHARDYFUNCTIONSANALYSIS_H
#define ENZYME_AD_JAX_PASSES_DISTRIBUTED_FINDSHARDYFUNCTIONSANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::distributed {

class FindShardyFunctionsAnalysis {
public:
  struct FunctionInfo {
    func::FuncOp funcOp;
    StringAttr symName;
    llvm::SmallVector<sdy::MeshAttr> meshes;
  };

  explicit FindShardyFunctionsAnalysis(ModuleOp module);

  llvm::ArrayRef<FunctionInfo> getShardyFunctions() const { return functions; }

  bool isShardyFunction(func::FuncOp funcOp) const {
    return functionIndices.contains(funcOp.getOperation());
  }

  llvm::ArrayRef<sdy::MeshAttr> getMeshes(func::FuncOp funcOp) const;

  bool isValid() const { return valid; }

private:
  bool valid = true;
  llvm::SmallVector<FunctionInfo> functions;
  llvm::DenseMap<Operation *, unsigned> functionIndices;
};

} // namespace mlir::enzyme::distributed

#endif
