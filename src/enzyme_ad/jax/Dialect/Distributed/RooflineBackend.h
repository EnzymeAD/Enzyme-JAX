#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ROOFLINE_BACKEND_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ROOFLINE_BACKEND_H

#include "CostBackend.h"

namespace mlir::enzyme::distributed {

// Prices each top-level op of @kernel by ComputeCost.h's per-op roofline
// (costOp), at 0 when costOp cannot cost it. Never groups ops.
class RooflineBackend : public CostBackend {
public:
  llvm::StringRef name() const override { return "roofline"; }
  LogicalResult annotate(ModuleOp kernelModule,
                         const CostTarget &target) override;
};

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ROOFLINE_BACKEND_H
