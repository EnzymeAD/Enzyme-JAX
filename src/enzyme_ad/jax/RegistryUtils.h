#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace enzyme {
void prepareRegistry(mlir::DialectRegistry &registry);

void registerDialects(mlir::DialectRegistry &ctx);

void registerInterfaces(mlir::DialectRegistry &registry);

void initializePasses(mlir::DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir
