#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace enzyme {
void prepareRegistry(mlir::DialectRegistry &registry);

void registerDialects(mlir::DialectRegistry &ctx);

void registerInterfaces(mlir::DialectRegistry &registry);

void loadAllRegisteredDialects(mlir::MLIRContext &context);

void initializePasses();
} // namespace enzyme
} // namespace mlir
