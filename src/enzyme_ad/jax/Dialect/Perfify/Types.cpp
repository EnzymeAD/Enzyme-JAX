#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPE_INTERFACE_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypeInterfaces.cpp.inc"

using namespace mlir;

bool enzyme::perfify::ConstantCostType::isPolynomial() const { return false; }

bool enzyme::perfify::PolyCostType::isPolynomial() const { return true; }