#pragma once

namespace mlir {
class DialectRegistry;
}

void prepareRegistry(mlir::DialectRegistry &registry);
