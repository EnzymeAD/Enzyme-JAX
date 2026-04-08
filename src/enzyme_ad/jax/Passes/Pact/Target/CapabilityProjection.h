#ifndef PACT_TARGET_CAPABILITY_PROJECTION_H
#define PACT_TARGET_CAPABILITY_PROJECTION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

#include "CompilationConfig.h"
#include "RawDescriptor.h"

namespace mlir::enzyme::pact {

class CapabilityProjection {
public:
  static DictionaryAttr project(const RawDescriptor &raw, MLIRContext *ctx);
};

} // namespace mlir::enzyme::pact

#endif // PACT_TARGET_CAPABILITY_PROJECTION_H
