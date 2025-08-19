#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

#include "src/enzyme_ad/jax/Implementations/EnzymeXLADerivatives.inc"

class Pointer2MemrefRev : public ReverseAutoDiffOpInterface::ExternalModel<
                              Pointer2MemrefRev, enzymexla::Pointer2MemrefOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto p2m = cast<enzymexla::Pointer2MemrefOp>(op);
    if (!gutils->isConstantValue(p2m)) {
      Value dres = gutils->invertPointerM(p2m.getSource(), builder);
      Value shadow = builder.create<enzymexla::Pointer2MemrefOp>(
          p2m.getLoc(), p2m.getType(), dres);
      gutils->setDiffe(p2m, shadow, builder);
    }
  }
};
} // namespace

void mlir::enzyme::registerEnzymeXLADialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            enzymexla::EnzymeXLADialect *) {
    enzymexla::Pointer2MemrefOp::attachInterface<Pointer2MemrefRev>(*context);
  });
}
