#ifndef DISTRIBUTED_PASSES_H
#define DISTRIBUTED_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Distributed/LocalizationTypes.h"

namespace mlir {
namespace enzyme {
namespace distributed {

llvm::LogicalResult rewriteShardyCollectivesInFunction(func::FuncOp funcOp);
llvm::LogicalResult
localizeMeshComputation(MeshComputationOp meshComputationOp);
llvm::LogicalResult
parameterizedLocalizeMeshComputation(MeshComputationOp meshComputationOp,
                                     const TensorBindingMap &bindings);

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

} // namespace distributed
} // namespace enzyme
} // namespace mlir

#endif
