#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/APFloat.h"

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

static inline mlir::DenseFPElementsAttr getTensorAttr(mlir::Type type,
                                                       llvm::StringRef value) {
  using namespace mlir;
  auto T = cast<TensorType>(type);
  size_t num = 1;
  for (auto sz : T.getShape())
    num *= sz;
  APFloat apvalue(T.getElementType().cast<FloatType>().getFloatSemantics(),
                  value);
  SmallVector<APFloat> supportedValues(num, apvalue);
  return DenseFPElementsAttr::get(type.cast<ShapedType>(), supportedValues);
}
