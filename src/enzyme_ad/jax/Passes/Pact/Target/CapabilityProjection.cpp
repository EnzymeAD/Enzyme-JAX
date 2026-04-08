#include "CapabilityProjection.h"
#include "mlir/IR/Builders.h"

using namespace mlir::enzyme::pact;

mlir::DictionaryAttr CapabilityProjection::project(const RawDescriptor &raw,
                                                   mlir::MLIRContext *ctx) {
  mlir::Builder b(ctx);
  mlir::NamedAttrList attrs;

  attrs.set("capability.exec.subgroup_width",
            b.getI32IntegerAttr(raw.wave.compute.width));

  return mlir::DictionaryAttr::get(ctx, attrs);
}