#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H

#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

// Shared with the axis dialect: parses/prints a type-annotated variadic as
// "(%arg1 : type1, ..., %argN : typeN)" or "()".
using ::mlir::enzyme::axis::parseVariadicWithTypes;
using ::mlir::enzyme::axis::printVariadicWithTypes;

// Result-list counterpart: "(type1, ..., typeN)" or "()". Literal parens in the
// assembly format are not enough, since the generated list parser is given no
// delimiter and so rejects the empty list its printer emits.
inline ParseResult parseParenTypes(OpAsmParser &parser,
                                   SmallVectorImpl<Type> &types) {
  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
    return parser.parseType(types.emplace_back());
  });
}

inline void printParenTypes(OpAsmPrinter &printer, Operation *op,
                            TypeRange types) {
  printer << '(';
  llvm::interleaveComma(types, printer);
  printer << ')';
}

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_COLLECTIVE_OPS_H