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

#include <optional>

using namespace mlir;
using namespace mlir::enzyme::tessera;

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Tessera/TesseraOps.cpp.inc"

namespace mlir::enzyme::tessera {} // namespace mlir::enzyme::tessera

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

void DefineOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayAttr argModes, bool pure,
                     StringAttr sym_visibility, ArrayRef<NamedAttribute> attrs,
                     ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute("pure", builder.getBoolAttr(pure));
  state.addAttribute("argModes", argModes);

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

// Translate a tessera.call operand index into the corresponding raw
// tessera.define argument index. tessera.call's operand list excludes the
// sret argument (if any) and any pure result/output arguments (see
// CallOpRewrite in LLVMToTessera.cpp), so both must be skipped when mapping
// back onto the define op's own argument list.
static std::optional<unsigned> translateCallOperandIndex(DefineOp defineOp,
                                                         unsigned index) {
  unsigned offset = defineOp.getSretAttr() != nullptr ? 1 : 0;
  unsigned seen = 0;
  for (unsigned i = 0, e = defineOp.getArgModeEntries().size(); i != e; ++i) {
    if (defineOp.argIsWritten(i) && !defineOp.argIsRead(i))
      continue;
    if (seen == index)
      return i + offset;
    ++seen;
  }
  return std::nullopt;
}

// Override getArgAttr to map call-side indices to define-side indices, so
// that generic FunctionOpInterface callers (e.g. mem2reg) can index by a
// tessera.call's operand position directly.
Attribute DefineOp::getArgAttr(unsigned index, StringAttr name) {
  auto rawIndex = translateCallOperandIndex(*this, index);
  if (!rawIndex)
    return nullptr;
  if (auto dict = mlir::function_interface_impl::getArgAttrDict(
          cast<FunctionOpInterface>(getOperation()), *rawIndex))
    return dict.get(name);
  return nullptr;
}

Attribute DefineOp::getArgAttr(unsigned index, StringRef name) {
  auto rawIndex = translateCallOperandIndex(*this, index);
  if (!rawIndex)
    return nullptr;
  if (auto dict = mlir::function_interface_impl::getArgAttrDict(
          cast<FunctionOpInterface>(getOperation()), *rawIndex))
    return dict.get(name);
  return nullptr;
}

// Field names inside a lifted argument's mode dictionary.
static constexpr llvm::StringLiteral argModeTypeField = "type";
static constexpr llvm::StringLiteral argModeDirField = "dir";

ArrayRef<Attribute> DefineOp::getArgModeEntries() {
  return getArgModes().getValue();
}

// The mode dictionary for an argument, or null if the argument is not lifted.
static DictionaryAttr getArgModeDict(DefineOp op, unsigned argIdx) {
  auto modes = op.getArgModeEntries();
  assert(argIdx < modes.size() && "argIdx out of bounds in getArgModeDict");
  return dyn_cast<DictionaryAttr>(modes[argIdx]);
}

// The direction string of a lifted argument, or empty if it is not lifted.
static StringRef getArgDir(DefineOp op, unsigned argIdx) {
  auto dict = getArgModeDict(op, argIdx);
  if (!dict)
    return StringRef();
  if (auto dirAttr = dyn_cast_or_null<StringAttr>(dict.get(argModeDirField)))
    return dirAttr.getValue();
  return StringRef();
}

Type DefineOp::getArgLiftedType(unsigned argIdx) {
  auto dict = getArgModeDict(*this, argIdx);
  if (!dict)
    return nullptr;
  auto typeAttr = dyn_cast_or_null<TypeAttr>(dict.get(argModeTypeField));
  if (!typeAttr)
    return nullptr;
  return typeAttr.getValue();
}

bool DefineOp::isLiftedArg(unsigned argIdx) {
  return getArgModeDict(*this, argIdx) != nullptr;
}

bool DefineOp::argIsRead(unsigned argIdx) {
  StringRef dir = getArgDir(*this, argIdx);
  return dir == "in" || dir == "inout";
}

bool DefineOp::argIsWritten(unsigned argIdx) {
  StringRef dir = getArgDir(*this, argIdx);
  return dir == "out" || dir == "inout";
}

unsigned DefineOp::getNumWrittenArgs() {
  unsigned count = 0;
  for (unsigned i = 0, e = getArgModeEntries().size(); i != e; ++i)
    if (argIsWritten(i))
      count++;
  return count;
}

Type DefineOp::getCallOperandPointeeType(unsigned callOperandIdx) {
  // getArgAttr already maps call-operand indices onto define-argument ones.
  if (auto byValAttr =
          getArgAttr(callOperandIdx, LLVM::LLVMDialect::getByValAttrName()))
    return cast<TypeAttr>(byValAttr).getValue();
  auto rawIndex = translateCallOperandIndex(*this, callOperandIdx);
  if (!rawIndex)
    return nullptr;
  // argModes is indexed sret-exclusive, so undo the sret offset. Only a
  // read argument reaches the callee through a loaded value; a write-only
  // ("out") argument has no operand slot to carry one.
  unsigned offset = getSretAttr() != nullptr ? 1 : 0;
  if (!argIsRead(*rawIndex - offset))
    return nullptr;
  return getArgLiftedType(*rawIndex - offset);
}

unsigned DefineOp::getNumCallOperands() {
  unsigned count = 0;
  for (unsigned i = 0, e = getArgModeEntries().size(); i != e; ++i)
    if (!argIsWritten(i) || argIsRead(i))
      count++;
  return count;
}

LogicalResult DefineOp::verify() {
  // Each entry is either UnitAttr (argument not lifted) or a dictionary
  // carrying exactly one pointee type and one direction. Storing the type once
  // is deliberate: the previous representation kept it in two parallel arrays,
  // which could disagree without any verifier noticing.
  for (auto [i, a] : llvm::enumerate(getArgModes())) {
    if (isa<UnitAttr>(a))
      continue;
    auto dict = dyn_cast<DictionaryAttr>(a);
    if (!dict)
      return emitOpError("argModes entry ")
             << i << " must be a DictionaryAttr or UnitAttr, but got " << a;

    auto typeAttr = dyn_cast_or_null<TypeAttr>(dict.get(argModeTypeField));
    if (!typeAttr)
      return emitOpError("argModes entry ")
             << i << " must have a '" << argModeTypeField << "' TypeAttr";

    auto dirAttr = dyn_cast_or_null<StringAttr>(dict.get(argModeDirField));
    if (!dirAttr)
      return emitOpError("argModes entry ")
             << i << " must have a '" << argModeDirField << "' StringAttr";

    StringRef dir = dirAttr.getValue();
    if (dir != "in" && dir != "out" && dir != "inout")
      return emitOpError("argModes entry ")
             << i << " has invalid direction '" << dir
             << "', expected 'in', 'out', or 'inout'";
  }

  // One entry per sret-excluded argument. Write-only ("out") arguments stay in
  // the define op's signature -- it is tessera.call's operand list, not this
  // one, that drops them (see getNumCallOperands).
  unsigned offset = getSretAttr() != nullptr ? 1 : 0;
  if (getArgModes().size() != getFunctionType().getNumInputs() - offset)
    return emitOpError("argModes size (")
           << getArgModes().size() << ") must match number of args ("
           << getFunctionType().getNumInputs() - offset << ")";

  if (getSretAttr() && getNumWrittenArgs() > 0)
    return emitOpError("cannot have both sret and write ('out'/'inout') args");

  if (getSretAttr() && !getFunctionType().getResults().empty())
    return emitOpError(
        "sret function must have a void (empty) natural result list");

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
  // write-only ("out") args, which are dropped from the operand list and
  // added as trailing results.
  unsigned expectedNumOperands = fn.getNumCallOperands();
  if (getNumOperands() != expectedNumOperands)
    return emitOpError("incorrect number of operands for callee: expected ")
           << expectedNumOperands << ", got " << getNumOperands();

  // Walk the callee's sret-excluded arguments in order, matching each
  // argument to the next tessera.call operand and skipping the write-only
  // ones. Allow a type mismatch only where the operand carries a pointee
  // value loaded at the call site rather than the pointer itself.
  bool has_sret = fn.getSretAttr() != nullptr;
  unsigned argOffset = has_sret ? 1 : 0;
  unsigned operandIdx = 0;
  for (unsigned i = 0, e = fn.getArgModeEntries().size(); i != e; ++i) {
    if (fn.argIsWritten(i) && !fn.argIsRead(i))
      continue; // call operand list does not include write-only args
    Type expectedType = fnType.getInput(i + argOffset);
    if (getOperand(operandIdx).getType() == expectedType) {
      operandIdx++;
      continue; // operand type matches expected type
    }
    if (isa<LLVM::LLVMPointerType>(expectedType) &&
        (fn.getArgAttr(operandIdx, LLVM::LLVMDialect::getByValAttrName()) ||
         fn.argIsRead(i))) {
      operandIdx++;
      continue; // operand was loaded from the pointer, so a type mismatch
                // against the pointer type is expected here
    }
    return emitOpError("operand type mismatch: expected operand type ")
           << expectedType << ", but provided "
           << getOperand(operandIdx).getType() << " for operand number "
           << operandIdx;
  }

  // tessera.call result count = one leading result per written ("out" or
  // "inout") argument, in argument order, followed by the callee's natural
  // function_type results; if a sret is present, the result count is 1
  // (the sret-derived value).
  unsigned calleeNumResults = fnType.getNumResults();
  unsigned numResultArgs = fn.getNumWrittenArgs();
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
    // Verify the leading results derived from written arguments, in argument
    // order. The direction check is load-bearing: getArgLiftedType is also
    // non-null for read-only ("in") args, which contribute no result.
    unsigned resultArgIdx = 0;
    for (unsigned i = 0, e = fn.getArgModeEntries().size(); i != e; ++i) {
      if (!fn.argIsWritten(i))
        continue;
      Type resultType = fn.getArgLiftedType(i);
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
