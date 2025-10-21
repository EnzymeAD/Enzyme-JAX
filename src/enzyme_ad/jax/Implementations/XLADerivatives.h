#pragma once

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/APFloat.h"

#include "Enzyme/MLIR/Passes/RemovalUtils.h"

namespace mlir {
namespace enzyme {
void registerEnzymeXLADialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerMHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerStableHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerCHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerEnzymeXLADialectAutoDiffInterface(mlir::DialectRegistry &registry);

static inline void
registerXLAAutoDiffInterfaces(mlir::DialectRegistry &registry) {
  registerEnzymeXLADialectAutoDiffInterface(registry);
  registerMHLODialectAutoDiffInterface(registry);
  registerStableHLODialectAutoDiffInterface(registry);
  registerCHLODialectAutoDiffInterface(registry);
  registerEnzymeXLADialectAutoDiffInterface(registry);
}

void removalBlockExplore(Block *block, IRMapping &mapping,
                         PatternRewriter &rewriter,
                         llvm::SetVector<Value> &gradients,
                         llvm::MapVector<Value, CacheInfo> &caches);
} // namespace enzyme
} // namespace mlir
  //
