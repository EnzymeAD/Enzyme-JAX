#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme::tessera;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.cpp.inc"

namespace mlir::enzyme::tessera {} // namespace mlir::enzyme::tessera

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

void DefineOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     LLVM::LLVMFunctionType type,
                     ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumParams() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag, std::string &) {
        Type returnType = results.empty()
                              ? LLVM::LLVMVoidType::get(builder.getContext())
                              : results[0];
        return LLVM::LLVMFunctionType::get(returnType, argTypes);
      };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DefineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void DefineOp::cloneInto(DefineOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
DefineOp DefineOp::clone(IRMapping &mapper) {
  // Create the new function.
  DefineOp newFunc = cast<DefineOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    auto oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumParams();
    SmallVector<Type, 4> newParams;
    newParams.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newParams.push_back(oldType.getParams()[i]);

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newParams.size() != oldNumArgs) {
      newFunc.setType(LLVM::LLVMFunctionType::get(
          oldType.getReturnType(), newParams, oldType.isVarArg()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newParams.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}
DefineOp DefineOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<DefineOp>((*this)->getParentOp());
  auto fnType = function.getFunctionType();
  bool isVoid = mlir::isa<LLVM::LLVMVoidType>(fnType.getReturnType());

  if (isVoid && getNumOperands() != 0)
    return emitOpError("has operands but enclosing function (@")
           << function.getName() << ") returns void";

  if (!isVoid && getNumOperands() != 1)
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns one value";

  if (!isVoid && getOperand(0).getType() != fnType.getReturnType())
    return emitError() << "type of return operand 0 ("
                       << getOperand(0).getType()
                       << ") doesn't match function result type ("
                       << fnType.getReturnType() << ") in function @"
                       << function.getName();
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  DefineOp fn = symbolTable.lookupNearestSymbolFrom<DefineOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumParams() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumParams(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getParams()[i])
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getParams()[i] << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (mlir::isa<LLVM::LLVMVoidType>(fnType.getReturnType()) !=
      (getNumResults() == 0))
    return emitOpError("incorrect number of results for callee");

  if (getNumResults() > 0) {
    Type resultType = getResult(0).getType();
    if (resultType != fnType.getReturnType()) {
      auto diag = emitOpError("result type mismatch");
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getReturnType();
      return diag;
    }
  }

  return success();
}

LLVM::LLVMFunctionType CallOp::getCalleeType() {
  auto resultTypes = getResultTypes();
  auto returnType = resultTypes.empty() ? LLVM::LLVMVoidType::get(getContext())
                                        : resultTypes[0];
  SmallVector<Type> argTypes(getOperandTypes());
  return LLVM::LLVMFunctionType::get(returnType, argTypes);
}
