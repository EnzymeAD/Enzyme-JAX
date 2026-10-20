#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::enzyme::perfify {
// Handle optional type suffix (either "perfify.cmp <comparator>, %a, %b" or
// "perfify.cmp <comparator>, %a, %b : !perfify.cost, !perfify.interval" is
// valid)
static ParseResult parseCmpOperandTypes(OpAsmParser &parser, Type &lhsType,
                                        Type &rhsType) {
  if (succeeded(parser.parseOptionalColon()))
    return failure(parser.parseType(lhsType) || parser.parseComma() ||
                   parser.parseType(rhsType));
  lhsType = rhsType = CostType::get(parser.getContext());
  return success();
}

static void printCmpOperandTypes(OpAsmPrinter &printer, Operation *,
                                 Type lhsType, Type rhsType) {
  if (isa<CostType>(lhsType) && isa<CostType>(rhsType))
    return;
  printer << " : " << lhsType << ", " << rhsType;
}
} // namespace mlir::enzyme::perfify

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Perfify/PerfifyOps.cpp.inc"

using namespace mlir;
using namespace mlir::enzyme::perfify;
LogicalResult SymCostOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto def =
      symbolTable.lookupNearestSymbolFrom<DefSymbolOp>(*this, getSymAttr());
  if (!def)
    return emitOpError("'")
           << getSym() << "' does not reference a valid symbol";
  if (def.getType() != getResult().getType())
    return emitOpError("type mismatch: symbol is ")
           << def.getType() << " but result is " << getResult().getType();
  return success();
}

LogicalResult IntervalOp::verify() {
  if (!getLb() && !getUb())
    return emitOpError("requires at least one of 'lb' or 'ub'");
  return success();
}

LogicalResult CompareOp::verify() {
  if (isa<IntervalType>(getExpCost().getType()) &&
      isa<IntervalType>(getFuncCost().getType()))
    return emitOpError("cannot compare two intervals");
  return success();
}
