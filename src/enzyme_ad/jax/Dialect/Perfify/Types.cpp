#include "Dialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypes.cpp.inc"

#define GET_TYPE_INTERFACE_CLASES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyTypeInterfaces.cpp.inc"
using namespace mlir;

bool enzyme::perfify::PerfifyTypes::ConstantCostType::isPolynomial() {
    return false;
}

bool enzyme::perfify::PerfifyTypes::PolyCostType::isPolynomial() {
    return true;
}