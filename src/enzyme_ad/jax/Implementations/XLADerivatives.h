#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace enzyme {
void registerMHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);
void registerStableHLODialectAutoDiffInterface(mlir::DialectRegistry &registry);

static inline void
registerXLAAutoDiffInterfaces(mlir::DialectRegistry &registry) {
  registerMHLODialectAutoDiffInterface(registry);
  registerStableHLODialectAutoDiffInterface(registry);
}
} // namespace enzyme
} // namespace mlir
