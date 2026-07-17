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

static ParseResult
parseAxisMapping(OpAsmParser &parser,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mappingLhs,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mappingRhs) {
  if (parser.parseLParen()) {
    return failure();
  }

  if (succeeded(parser.parseOptionalRParen())) {
    return success();
  }

  while (true) {
    OpAsmParser::UnresolvedOperand lhs;
    OpAsmParser::UnresolvedOperand rhs;
    if (parser.parseOperand(lhs) || parser.parseArrow() ||
        parser.parseOperand(rhs)) {
      return failure();
    }
    mappingLhs.push_back(lhs);
    mappingRhs.push_back(rhs);

    if (succeeded(parser.parseOptionalComma())) {
      continue;
    }
    if (parser.parseRParen()) {
      return failure();
    }
    break;
  }

  return success();
}

static void printAxisMapping(OpAsmPrinter &printer, Operation *op,
                             OperandRange mappingLhs, OperandRange mappingRhs) {
  printer << '(';
  for (auto [idx, lhs] : llvm::enumerate(mappingLhs)) {
    if (idx != 0) {
      printer << ", ";
    }
    printer << lhs << " -> " << mappingRhs[idx];
  }
  printer << ')';
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H