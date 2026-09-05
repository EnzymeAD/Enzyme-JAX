#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"

namespace mlir::enzyme::distributed {

// Backing implementation for EmptyableArrayRefParameter (Attributes.td).
//
// ODS's default list parser is FieldParser<SmallVector<T>>, which calls
// parseCommaSeparatedList with Delimiter::None. MLIR only permits an empty list
// when a delimiter is supplied, so the default parser rejects the `[]` that its
// own printer emits. Owning the delimiter here is what makes the empty case
// round-trip.
template <typename ContainerT>
static FailureOr<ContainerT> parseEmptyableArrayRef(AsmParser &parser) {
  ContainerT elements;
  auto parseElement = [&]() -> ParseResult {
    auto element = FieldParser<typename ContainerT::value_type>::parse(parser);
    if (failed(element))
      return failure();
    elements.push_back(std::move(*element));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                     parseElement))
    return failure();
  return elements;
}

template <typename RangeT>
static void printEmptyableArrayRef(AsmPrinter &printer, RangeT &&elements) {
  printer << '[';
  llvm::interleaveComma(elements, printer, [&](auto element) {
    printer.printStrippedAttrOrType(element);
  });
  printer << ']';
}

} // namespace mlir::enzyme::distributed

// Include the .cpp.inc files
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedAttrDefs.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.cpp.inc"

// Initialize the dialect
void mlir::enzyme::distributed::DistributedDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
      >();
}