#include "llvm/ADT/TypeSwitch.h"

#include "CostBackend.h"
#include "Dialect.h"
#include "GrammarRegistry.h"

#include "shardy/dialect/sdy/ir/parsers.h"
#include "shardy/dialect/sdy/ir/printers.h"

namespace mlir::enzyme::distributed {

// OperatorAxesAttr prints its axis extents as #sdy.op_sharding_rule prints its
// factor sizes (`{i=4, j=8}`), through custom<FactorSizes>.
using ::mlir::sdy::parseFactorSizes;
using ::mlir::sdy::printFactorSizes;

// OperatorAxesAttr's mapping lists, e.g. `([i, k], [k, j])`, in Shardy's
// syntax except that a dimension with no operator axes is written `1`, which
// Shardy's syntax cannot express. The parser skips every `1` token within a
// dimension (`[1]`, `[1 i]`); the printer emits a single `1` exactly when a
// dimension has no axes.
static ParseResult
parseTensorMappings(AsmParser &parser,
                    SmallVector<sdy::TensorMappingAttr> &mappings) {
  auto parseDim = [&](SmallVector<sdy::DimMappingAttr> &dims) -> ParseResult {
    SMLoc loc = parser.getCurrentLocation();
    SmallVector<int64_t> axes;
    bool sawToken = false;
    while (true) {
      int64_t one;
      OptionalParseResult integer = parser.parseOptionalInteger(one);
      if (integer.has_value()) {
        if (failed(*integer))
          return failure();
        if (one != 1)
          return parser.emitError(loc)
                 << "expected operator axes or 1 for a dimension, got " << one;
        sawToken = true;
        continue;
      }
      // Shardy's DimMappingAttr parser reads one run of axis symbols (`ij`),
      // and parses nothing when the next token isn't a keyword.
      auto dim = cast_or_null<sdy::DimMappingAttr>(
          sdy::DimMappingAttr::parse(parser, Type()));
      if (!dim)
        return failure();
      if (dim.getFactorIndices().empty())
        break;
      llvm::append_range(axes, dim.getFactorIndices());
      sawToken = true;
    }
    if (!sawToken)
      return parser.emitError(loc,
                              "expected operator axes or 1 for a dimension");
    dims.push_back(sdy::DimMappingAttr::get(parser.getContext(), axes));
    return success();
  };
  auto parseTensor = [&]() -> ParseResult {
    SmallVector<sdy::DimMappingAttr> dims;
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       [&] { return parseDim(dims); }))
      return failure();
    mappings.push_back(sdy::TensorMappingAttr::get(parser.getContext(), dims));
    return success();
  };
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                        parseTensor);
}

static void printTensorMappings(AsmPrinter &printer,
                                ArrayRef<sdy::TensorMappingAttr> mappings) {
  printer << '(';
  llvm::interleaveComma(mappings, printer, [&](sdy::TensorMappingAttr mapping) {
    printer << '[';
    llvm::interleaveComma(mapping.getDimMappings(), printer,
                          [&](sdy::DimMappingAttr dim) {
                            if (dim.getFactorIndices().empty())
                              printer << '1';
                            else
                              dim.print(printer);
                          });
    printer << ']';
  });
  printer << ')';
}

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
  grammarRegistry = std::make_unique<GrammarRegistry>();
  registerBuiltinGrammarEntries(*grammarRegistry);
}

mlir::enzyme::distributed::GrammarRegistry &
mlir::enzyme::distributed::DistributedDialect::getGrammarRegistry() {
  return *grammarRegistry;
}

// Discardable attributes in this dialect's namespace are unchecked except
// distributed.latency, whose consumers rely on it being a duration.
mlir::LogicalResult
mlir::enzyme::distributed::DistributedDialect::verifyOperationAttribute(
    Operation *op, NamedAttribute attribute) {
  if (attribute.getName() != kLatencyAttrName)
    return success();
  auto latency = dyn_cast<FloatAttr>(attribute.getValue());
  // Written as !(x >= 0) so that NaN is rejected too.
  if (!latency || !latency.getType().isF64() ||
      !(latency.getValueAsDouble() >= 0))
    return op->emitOpError()
           << "expects '" << kLatencyAttrName
           << "' to be a non-negative f64 number of nanoseconds, got "
           << attribute.getValue();
  return success();
}