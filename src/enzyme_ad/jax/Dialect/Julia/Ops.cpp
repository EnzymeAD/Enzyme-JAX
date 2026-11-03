#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::enzymexla::julia;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  if (!argAttrs.empty()) {
    assert(type.getNumInputs() == argAttrs.size());
    call_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/{},
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
  }
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto fn = cast<FuncOp>((*this)->getParentOp());
  ArrayRef<Type> results = fn.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << fn.getName() << ") returns " << results.size();
  for (auto [idx, expected] : llvm::enumerate(results)) {
    if (getOperand(idx).getType() != expected)
      return emitOpError("operand ")
             << idx << " type " << getOperand(idx).getType()
             << " does not match function result type " << expected;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand cond;
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion))
    return failure();
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion))
      return failure();
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition();
  p.printOptionalArrowTypeList(getResultTypes());
  p << " ";
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> iterArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  if (parser.parseAssignmentList(iterArgs, initOperands))
    return failure();

  FunctionType fnType;
  if (parser.parseColonType(fnType))
    return failure();
  if (iterArgs.size() != fnType.getNumInputs())
    return parser.emitError(parser.getNameLoc(),
                            "mismatch between iter args and input types");

  for (auto [arg, ty] : llvm::zip(iterArgs, fnType.getInputs()))
    arg.type = ty;

  if (parser.resolveOperands(initOperands, fnType.getInputs(),
                             parser.getNameLoc(), result.operands))
    return failure();
  result.addTypes(fnType.getResults());

  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();
  if (parser.parseKeyword("cond") || parser.parseRegion(*condRegion, iterArgs))
    return failure();

  SmallVector<OpAsmParser::Argument> bodyArgs;
  if (parser.parseKeyword("body") || parser.parseRegion(*bodyRegion))
    return failure();

  return parser.parseOptionalAttrDict(result.attributes);
}

void WhileOp::print(OpAsmPrinter &p) {
  p << " (";
  Block &condEntry = getCond().front();
  llvm::interleaveComma(
      llvm::zip(condEntry.getArguments(), getInits()), p, [&](auto it) {
        p << std::get<0>(it) << " = " << std::get<1>(it);
      });
  p << ") : ";
  p.printFunctionalType(getInits().getTypes(), getResultTypes());
  p << " cond ";
  p.printRegion(getCond(), /*printEntryBlockArgs=*/false);
  p << " body ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true);
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// ConstantOp & LoadOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (llvm::isa_and_nonnull<KernelOp, KernelInPlaceOp>(
          (*this)->getParentOp())) {
    return;
  }
  effects.emplace_back(MemoryEffects::Read::get());
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Julia/JuliaOps.cpp.inc"
