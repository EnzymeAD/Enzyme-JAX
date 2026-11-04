#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "src/enzyme_ad/jax/Dialect/Julia/JuliaDialect.cpp.inc"
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaAttrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaAttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

using namespace mlir;
using namespace mlir::enzymexla::julia;

namespace {

static Type parseShapedBody(
    AsmParser &parser,
    llvm::function_ref<Type(ArrayRef<int64_t>, Type)> builderFn) {
  if (parser.parseLess())
    return Type();
  SmallVector<int64_t> dims;
  if (parser.parseDimensionList(dims, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return Type();
  Type elemType;
  if (parser.parseType(elemType) || parser.parseGreater())
    return Type();
  return builderFn(dims, elemType);
}

static void printShapedBody(AsmPrinter &printer, ArrayRef<int64_t> shape,
                            Type elemType) {
  printer << "<";
  if (!shape.empty()) {
    printer.printDimensionList(shape);
    printer << "x";
  }
  printer << elemType << ">";
}

} // namespace

Type ArrayType::parse(AsmParser &parser) {
  return parseShapedBody(parser, [&](ArrayRef<int64_t> dims, Type elem) {
    return ArrayType::get(parser.getContext(), dims, elem);
  });
}

void ArrayType::print(AsmPrinter &printer) const {
  printShapedBody(printer, getShape(), getElementType());
}

Type ViewType::parse(AsmParser &parser) {
  return parseShapedBody(parser, [&](ArrayRef<int64_t> dims, Type elem) {
    return ViewType::get(parser.getContext(), dims, elem);
  });
}

void ViewType::print(AsmPrinter &printer) const {
  printShapedBody(printer, getShape(), getElementType());
}

void JuliaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaAttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaOps.cpp.inc"
      >();
}

Operation *JuliaDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (llvm::isa<ShapedType>(type))
    return nullptr;
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(value))
    return ConstantOp::create(builder, loc, type, typedAttr);
  return nullptr;
}
