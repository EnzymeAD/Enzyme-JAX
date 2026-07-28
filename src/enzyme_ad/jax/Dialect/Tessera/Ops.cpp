#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
                     FunctionType type, ArrayAttr byRefTypes, bool pure,
                     ArrayAttr resultArgTypes,
                     StringAttr sym_visibility, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute("pure", builder.getBoolAttr(pure));
  state.addAttribute("byRefTypes", byRefTypes);

  if (resultArgTypes)
    state.addAttribute("resultArgTypes", resultArgTypes);

  if (sym_visibility)
    state.addAttribute(getSymVisibilityAttrName(state.name), sym_visibility);

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (!argAttrs.empty()) {
    assert(type.getNumInputs() == argAttrs.size());
    call_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/{},
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
  }
}

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

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
  DefineOp newFunc = cast<DefineOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
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

Attribute DefineOp::getSretAttr() {
  if (getFunctionType().getNumInputs() == 0)
    return nullptr;
  if (auto argAttrs = getAllArgAttrs())
    return cast<DictionaryAttr>(argAttrs[0])
        .get(LLVM::LLVMDialect::getStructRetAttrName());
  return nullptr;
}

// Override getArgAttr to map call-side indices to define-side indices.
// tessera::DefineOp has one extra argument at index 0 for sret, which
// is not present in tessera::CallOp operands. This allows generic
// FunctionOpInterface callers to use call-side indices directly.
Attribute DefineOp::getArgAttr(unsigned index, StringAttr name) {
  unsigned offset = getSretAttr() != nullptr ? 1 : 0;
  if (auto dict = mlir::function_interface_impl::getArgAttrDict(
          cast<FunctionOpInterface>(getOperation()), index + offset))
    return dict.get(name);
  return nullptr;
}

Attribute DefineOp::getArgAttr(unsigned index, StringRef name) {
  unsigned offset = getSretAttr() != nullptr ? 1 : 0;
  if (auto dict = mlir::function_interface_impl::getArgAttrDict(
          cast<FunctionOpInterface>(getOperation()), index + offset))
    return dict.get(name);
  return nullptr;
}

ArrayRef<Attribute> DefineOp::getByRefTypesArray() {
  return getByRefTypes().getValue();
}

Type DefineOp::getByRefType(unsigned argIdx) {
  auto types = getByRefTypesArray();
  assert(argIdx < types.size() && "argIdx out of bounds in getByRefType");
  auto attr = types[argIdx];
  if (!isa<TypeAttr>(attr))
    return nullptr;
  return cast<TypeAttr>(attr).getValue();
}

ArrayRef<Attribute> DefineOp::getResultArgTypesArray() {
  if (auto resultArgTypes = getResultArgTypes())
    return resultArgTypes->getValue();
  else
    return ArrayRef<Attribute>();
}

Type DefineOp::getResultArgType(unsigned argIdx) {
  auto types = getResultArgTypesArray();
  if (types.empty())
    return nullptr;
  assert(argIdx < types.size() && "argIdx out of bounds in getResultArgType");
  auto attr = types[argIdx];
  if (!isa<TypeAttr>(attr))
    return nullptr;
  return cast<TypeAttr>(attr).getValue();
}

unsigned DefineOp::getNumResultArgs() {
  if (auto resultArgTypes = getResultArgTypes()) {
    unsigned count = 0;
    for (Attribute a : resultArgTypes->getValue()) {
      if (isa<TypeAttr>(a))
        count++;
    }
    return count;
  } else {
    return 0;
  }
}

bool DefineOp::isResultOnlyArg(unsigned argIdx) {
  return getResultArgType(argIdx) != nullptr &&
         getByRefType(argIdx) == nullptr;
}

unsigned DefineOp::getNumCallOperands() {
  unsigned count = 0;
  for (unsigned i = 0, e = getByRefTypesArray().size(); i != e; ++i)
    if (!isResultOnlyArg(i))
      count++;
  return count;
}

LogicalResult DefineOp::verify() {
  for (Attribute a : getByRefTypes())
    if (!isa<TypeAttr>(a) && !isa<UnitAttr>(a))
      return emitOpError(
                 "byRefTypes entry must be TypeAttr or UnitAttr, but got ")
             << a;

  unsigned offset = getSretAttr() != nullptr ? 1 : 0;
  if (getByRefTypes().size() != getFunctionType().getNumInputs() - offset)
    return emitOpError("byRefTypes size (")
           << getByRefTypes().size() << ") must match number of args ("
           << getFunctionType().getNumInputs() - offset << ")";

  if (auto resultArgTypes = getResultArgTypes()) {
    for (Attribute a : resultArgTypes->getValue())
      if (!isa<TypeAttr>(a) && !isa<UnitAttr>(a))
        return emitOpError(
                   "resultArgTypes entry must be TypeAttr or UnitAttr, but got ")
               << a;
    if (resultArgTypes->getValue().size() != getByRefTypes().size())
      return emitOpError("resultArgTypes size (")
             << resultArgTypes->getValue().size() << ") must match number of byRef types ("
             << getByRefTypes().size() << ")";
  }

  if (getSretAttr() && getNumResultArgs() > 0)
    return emitOpError("cannot have both sret and resultArgTypes");

  if (getSretAttr() && !getFunctionType().getResults().empty())
    return emitOpError("sret function must have a void (empty) natural result list");

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

  auto fnType = fn.getFunctionType();

  // tessera.call operand count = every non-sret argument, minus any
  // pure-output (:result but not :byref) args, which are dropped from the
  // operand list and added as trailing results.
  unsigned expectedNumOperands = fn.getNumCallOperands();
  if (getNumOperands() != expectedNumOperands)
    return emitOpError("incorrect number of operands for callee: expected ")
           << expectedNumOperands << ", got " << getNumOperands();

  // Walk the callee's sret-excluded arguments in order, matching each
  // argument to the next tessera.call operand, skipping any pure-output
  // args are skipped. Allow type mismatch only for byref pointer args 
  // that have been converted to values.
  bool has_sret = fn.getSretAttr() != nullptr;
  unsigned argOffset = has_sret ? 1 : 0;
  unsigned operandIdx = 0;
  for (unsigned i = 0, e = fn.getByRefTypesArray().size(); i != e; ++i) {
    if (fn.isResultOnlyArg(i))
      continue; // call operand list does not include pure-output args
    Type expectedType = fnType.getInput(i + argOffset);
    if (getOperand(operandIdx).getType() == expectedType) {
      operandIdx++;
      continue; // operand type matches expected type
    }
    if (isa<LLVM::LLVMPointerType>(expectedType) &&
        (fn.getArgAttr(i, LLVM::LLVMDialect::getByValAttrName()) ||
         fn.getByRefType(i))) {
      operandIdx++;
      continue; // operand was loaded from a byref pointer, so type mismatch is allowed
    }
    return emitOpError("operand type mismatch: expected operand type ")
           << expectedType << ", but provided "
           << getOperand(operandIdx).getType() << " for operand number "
           << operandIdx;
  }

  // tessera.call result count = one leading result per :result-marked
  // argument (in argument order), followed by the callee's natural
  // function_type results; if a sret is present, the result count is 1
  // (the sret-derived value).
  unsigned calleeNumResults = fnType.getNumResults();
  unsigned numResultArgs = fn.getNumResultArgs();
  unsigned expectedNumResults =
      has_sret ? 1 : (calleeNumResults + numResultArgs);
  if (getNumResults() != expectedNumResults)
    return emitOpError("incorrect number of results for callee: expected ")
           << expectedNumResults << ", got " << getNumResults();

  if (has_sret) {
    auto sret = fn.getSretAttr();
    auto sretType = cast<TypeAttr>(sret).getValue();
    if (getResult(0).getType() != sretType)
      return emitOpError("result type mismatch: expected ")
             << sretType << " but got " << getResult(0).getType();
  } else {
    // Verify leading, :result-arg-derived result types, in argument order.
    unsigned resultArgIdx = 0;
    for (unsigned i = 0, e = fn.getByRefTypesArray().size(); i != e; ++i) {
      Type resultType = fn.getResultArgType(i);
      if (!resultType)
        continue;
      Value result = getResult(resultArgIdx);
      if (result.getType() != resultType)
        return emitOpError("result type mismatch for output-param argument ")
               << i << ": expected " << resultType << " but got "
               << result.getType();
      resultArgIdx++;
    }

    // Verify the callee's natural (function_type) results, trailing after
    // the result-arg-derived ones.
    for (unsigned i = 0, e = calleeNumResults; i != e; ++i)
      if (getResult(numResultArgs + i).getType() != fnType.getResult(i)) {
        auto diag = emitOpError("result type mismatch at index ")
                    << numResultArgs + i;
        diag.attachNote() << "      op result types: " << getResultTypes();
        diag.attachNote() << "function result types: " << fnType.getResults();
        return diag;
      }
  }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

void CallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return;
  DefineOp fn = SymbolTable::lookupNearestSymbolFrom<DefineOp>(*this, fnAttr);
  if (!fn)
    return;
  if (fn.getPure())
    return; // return nothing = no effects = side effect free

  // if not side effect free, add all possible memory effects
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto fn = cast<DefineOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = fn.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << fn.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType() << ") in function @"
                         << fn.getName()
                         << " doesn't match function result type ("
                         << results[i] << ")";
  return success();
}
