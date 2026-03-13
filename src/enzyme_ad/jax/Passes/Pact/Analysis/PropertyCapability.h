#ifndef PACT_ANALYSIS_PROPERTY_CAPABILITY_H
#define PACT_ANALYSIS_PROPERTY_CAPABILITY_H

#include "CompareResult.h"
#include "PipelineLevel.h"
#include "Strategy.h"
#include "mlir/IR/Attributes.h"
#include "src/enzyme_ad/jax/Passes/Pact/PropertyScheme.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::pact {

class CapabilityProjection;
class EvalContext;
struct RawDescriptor;
struct CompilationConfig;

class PropertyCapability {};

} // namespace mlir::enzyme::pact

#endif // PACT_ANALYSIS_PROPERTY_CAPABILITY_H