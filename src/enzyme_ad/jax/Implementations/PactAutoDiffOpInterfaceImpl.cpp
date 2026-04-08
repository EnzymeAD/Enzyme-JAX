#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "src/enzyme_ad/jax/Dialect/Pact/Dialect.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;

// namespace {
// #include "src/enzyme_ad/jax/Implementations/PactDerivatives.inc"
// } // namespace

void mlir::enzyme::registerPactDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, mlir::enzyme::pact::PactDialect *) {
        // registerInterfaces(context);
      });
}
