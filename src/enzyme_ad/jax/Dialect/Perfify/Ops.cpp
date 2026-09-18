#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::enzyme::perfify {} // namespace mlir::enzyme::perfify

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"

using namespace mlir;
using namespace mlir::enzyme::perfify;
LogicalResult SymCostOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto def = symbolTable.lookupNearestSymbolFrom<DefSymbolOp>(*this, getSymAttr());
  if (!def)
    return emitOpError("'") << getSym() << "' does not reference a valid symbol";
  if (def.getType() != getResult().getType())
    return emitOpError("type mismatch: symbol is ") << def.getType()
           << " but result is " << getResult().getType();
  return success();
}


