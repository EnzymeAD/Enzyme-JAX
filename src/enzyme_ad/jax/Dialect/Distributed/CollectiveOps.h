#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H

#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

/**
 *  Parses a type-annotated variadic as:
 *  (%arg1 : type1, %arg2 : type2, ..., %argN : typeN)
 *  OR
 *  ()
 */
static ParseResult parseVariadicWithTypes(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ssa_operands,
    SmallVectorImpl<Type> &types) {

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
    ssa_operands.push_back(reductionGroup);
    Type type;
    if (parser.parseColonType(type)) {
      return failure();
    }
    types.push_back(type);

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

static void printVariadicWithTypes(OpAsmPrinter &printer, Operation *op,
                                   OperandRange ssa_operands, TypeRange types) {
  printer << '(';
  for (auto [idx, pair] : llvm::enumerate(llvm::zip(ssa_operands, types))) {
    auto [ssa_operand, type] = pair;
    if (idx != 0) {
      printer << ", ";
    }
    printer << ssa_operand << " : " << type;
  }
  printer << ')';
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H