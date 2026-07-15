#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H

#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

static ParseResult parseReductionGroups(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionGroups,
    ArrayAttr &reductionFunctions) {
  SmallVector<Attribute> parsedReductionFunctions;

  if (parser.parseLParen()) {
    return failure();
  }

  if (succeeded(parser.parseOptionalRParen())) {
    reductionFunctions = ArrayAttr::get(parser.getBuilder().getContext(),
                                        parsedReductionFunctions);
    return success();
  }

  while (true) {
    OpAsmParser::UnresolvedOperand reductionGroup;
    if (parser.parseOperand(reductionGroup)) {
      return failure();
    }
    reductionGroups.push_back(reductionGroup);

    Attribute reductionFunctionAttr;
    if (parser.parseAttribute(reductionFunctionAttr)) {
      return failure();
    }
    auto reductionFunction =
        dyn_cast_or_null<FlatSymbolRefAttr>(reductionFunctionAttr);
    if (!reductionFunction) {
      return parser.emitError(parser.getCurrentLocation())
             << "requires reduction function to be a flat symbol reference";
    }
    parsedReductionFunctions.push_back(reductionFunction);

    if (succeeded(parser.parseOptionalComma())) {
      continue;
    }
    if (parser.parseRParen()) {
      return failure();
    }
    break;
  }

  reductionFunctions = ArrayAttr::get(parser.getBuilder().getContext(),
                                      parsedReductionFunctions);
  return success();
}

static void printReductionGroups(OpAsmPrinter &printer, Operation *op,
                                 OperandRange reductionGroups,
                                 ArrayAttr reductionFunctions) {
  printer << '(';
  for (auto [idx, reductionGroup] : llvm::enumerate(reductionGroups)) {
    if (idx != 0) {
      printer << ", ";
    }
    printer << reductionGroup << ' ' << reductionFunctions[idx];
  }
  printer << ')';
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H