#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::enzyme::perfify {
// Parses "[min_cost: @a, max_cost: @b]", with the keys in either order and
// either one optional.
static ParseResult parseIntervalBounds(OpAsmParser &parser,
                                       FlatSymbolRefAttr &minCost,
                                       FlatSymbolRefAttr &maxCost) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Square, [&]() -> ParseResult {
        SMLoc loc = parser.getCurrentLocation();
        StringRef key;
        if (parser.parseKeyword(&key) || parser.parseColon())
          return failure();
        FlatSymbolRefAttr *bound = key == "min_cost"   ? &minCost
                                   : key == "max_cost" ? &maxCost
                                                       : nullptr;
        if (!bound)
          return parser.emitError(loc, "expected 'min_cost' or 'max_cost'");
        if (*bound)
          return parser.emitError(loc, "duplicate '") << key << "'";
        return parser.parseAttribute(*bound);
      });
}

static void printIntervalBounds(OpAsmPrinter &printer, Operation *,
                                FlatSymbolRefAttr minCost,
                                FlatSymbolRefAttr maxCost) {
  printer << "[";
  if (minCost)
    printer << "min_cost: " << minCost;
  if (minCost && maxCost)
    printer << ", ";
  if (maxCost)
    printer << "max_cost: " << maxCost;
  printer << "]";
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

// Finds the estimated_cost or interval that `ref` names in `from`'s region.
static Operation *lookupCostBound(Operation *from, FlatSymbolRefAttr ref) {
  if (ref)
    for (Operation &op : from->getParentRegion()->getOps())
      if (isa<EstimatedCostOp, IntervalOp>(op) &&
          op.getAttr("name") == ref.getAttr())
        return &op;
  return nullptr;
}

Operation *CompareOp::getBoundOp() {
  return lookupCostBound(*this, getBoundAttr());
}

EstimatedCostOp IntervalOp::getMinCostOp() {
  return dyn_cast_or_null<EstimatedCostOp>(
      lookupCostBound(*this, getMinCostAttr()));
}

EstimatedCostOp IntervalOp::getMaxCostOp() {
  return dyn_cast_or_null<EstimatedCostOp>(
      lookupCostBound(*this, getMaxCostAttr()));
}

LogicalResult EstimatedCostOp::verify() {
  if (lookupCostBound(*this, FlatSymbolRefAttr::get(getNameAttr())) !=
      getOperation())
    return emitOpError("name is already defined in this region");
  return success();
}

LogicalResult IntervalOp::verify() {
  if (!getMinCostAttr() && !getMaxCostAttr())
    return emitOpError("requires at least one of 'min_cost' or 'max_cost'");
  if ((getMinCostAttr() && !getMinCostOp()) ||
      (getMaxCostAttr() && !getMaxCostOp()))
    return emitOpError("bounds must name a perfify.estimated_cost in this "
                       "region");
  if (lookupCostBound(*this, FlatSymbolRefAttr::get(getNameAttr())) !=
      getOperation())
    return emitOpError("name is already defined in this region");
  return success();
}

LogicalResult CompareOp::verify() {
  if (!getBoundOp())
    return emitOpError("'")
           << getBoundAttr()
           << "' does not name a perfify.estimated_cost or perfify.interval "
              "in this region";
  return success();
}

LogicalResult HypothesisOp::verify() {
  if (!getCmp().getDefiningOp<CompareOp>())
    return emitOpError("operand must be defined by perfify.cmp");
  if (isa_and_nonnull<ConditionsOp>((*this)->getParentOp()) &&
      (*this)->getParentRegion()->getRegionNumber() !=
          static_cast<unsigned>(getPrePost()))
    return emitOpError("'") << stringifyHoareStates(getPrePost())
                            << "' hypothesis is in the wrong conditions region";
  return success();
}
