#pragma once

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace enzyme {
void registerMHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerStableHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerCHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerEnzymeXLADialectAutoDiffInterface(mlir::DialectRegistry &registry);

static inline void
registerXLAAutoDiffInterfaces(mlir::DialectRegistry &registry) {
  registerMHLODialectAutoDiffInterface(registry);
  registerStableHLODialectAutoDiffInterface(registry);
  registerCHLODialectAutoDiffInterface(registry);
  registerEnzymeXLADialectAutoDiffInterface(registry);
}
} // namespace enzyme
} // namespace mlir
