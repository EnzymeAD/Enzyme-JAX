#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H

#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

static ParseResult parseReductionGroups(
    OpAsmParser &parser,
  SmallVectorImpl<OpAsmParser::UnresolvedOperand> &reductionGroups) {

  if (failed(parser.parseLParen())) {
    return failure();
  }

  if (succeeded(parser.parseOptionalRParen())) {
    return success();
  }

  while (true) {
    OpAsmParser::UnresolvedOperand reductionGroup;
    if (parser.parseOperand(reductionGroup)) {
      return failure();
    }
    reductionGroups.push_back(reductionGroup);

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

static void printReductionGroups(OpAsmPrinter &printer, Operation *op,
                                 OperandRange reductionGroups) {
  printer << '(';
  for (auto [idx, reductionGroup] : llvm::enumerate(reductionGroups)) {
    if (idx != 0) {
      printer << ", ";
    }
    printer << reductionGroup;
  }
  printer << ')';
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H