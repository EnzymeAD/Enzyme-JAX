#ifndef DISTRIBUTED_PASSES_H
#define DISTRIBUTED_PASSES_H

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

struct TensorBindingChoice {
  ::mlir::Value localizedAxis;
  int64_t tensorAxis = 0;
  int64_t chosenDeviceIndex = 0;
};

using TensorBindingMap = ::llvm::DenseMap<::mlir::Value, TensorBindingChoice>;

llvm::LogicalResult rewriteShardyCollectivesInFunction(func::FuncOp funcOp);
llvm::LogicalResult localizeMeshComputation(MeshComputationOp meshComputationOp);
llvm::LogicalResult parameterizedLocalizeMeshComputation(
	MeshComputationOp meshComputationOp, const TensorBindingMap &bindings);

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
