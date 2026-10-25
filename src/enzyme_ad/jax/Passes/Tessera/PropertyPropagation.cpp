//===----------------------------------------------------------------------===//
//
// Deriving the properties of values from declarations, and the pass that
// records them on call arguments. See PropertyPropagation.h for the model.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_TESSERAPROPAGATEPROPERTIESPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

// Properties that imply others. `positive_definite` (what its runtime check
// tests, which includes symmetry) is the same property as `SPD`.
struct Implication {
  llvm::StringLiteral from, to;
};
constexpr Implication kImplications[] = {
    {"SPD", "positive_definite"},       {"positive_definite", "SPD"},
    {"positive_definite", "symmetric"}, {"positive_definite", "invertible"},
    {"orthogonal", "invertible"},       {"identity", "diagonal"},
    {"identity", "positive_definite"},  {"identity", "orthogonal"},
    {"diagonal", "symmetric"},          {"diagonal", "triangular_upper"},
    {"diagonal", "triangular_lower"},
};

// How far back through calls a fact is followed before giving up.
constexpr unsigned kMaxDerivationDepth = 64;

// Step through operations that pass a value along unchanged.
Value stripCasts(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<LLVM::FreezeOp, LLVM::BitcastOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

DictionaryAttr attrDictAt(ArrayAttr list, unsigned index) {
  if (!list || index >= list.size())
    return nullptr;
  return dyn_cast<DictionaryAttr>(list[index]);
}

// Append each property named in `list`, an array of strings such as
// ["SPD", "symmetric"], that `names` does not have yet.
void appendPropertyNames(SmallVectorImpl<StringAttr> &names, Attribute list) {
  auto array = dyn_cast_or_null<ArrayAttr>(list);
  if (!array)
    return;
  for (Attribute entry : array)
    if (auto name = dyn_cast<StringAttr>(entry))
      if (!llvm::is_contained(names, name))
        names.push_back(name);
}

// Append the facts recorded in a call argument's attributes.
void appendRecordedFacts(SmallVectorImpl<StringAttr> &names,
                         DictionaryAttr argAttrs) {
  if (argAttrs)
    appendPropertyNames(names, argAttrs.get(kPropertyAttr));
}

// The declaration `name` (`tessera.assumes`, `tessera.guarantees`) in entry
// `index` of a function's argument or result attributes, or null.
Attribute declaredPropertiesAt(ArrayAttr list, unsigned index, StringRef name) {
  DictionaryAttr dict = attrDictAt(list, index);
  return dict ? dict.get(name) : nullptr;
}

// The function an llvm.call calls directly, or null.
LLVM::LLVMFuncOp calleeOf(LLVM::CallOp call) {
  auto calleeAttr = call.getCalleeAttr();
  return calleeAttr ? SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
                          call, calleeAttr)
                    : nullptr;
}

// What the callee of the call producing `result` guarantees of it, or null.
Attribute guaranteesOf(OpResult result) {
  Operation *op = result.getOwner();
  unsigned resultNumber = result.getResultNumber();

  if (auto call = dyn_cast<CallOp>(op)) {
    auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, call.getCalleeAttr().getAttr());
    if (!define)
      return nullptr;
    // A call's results start with its written arguments, and argModes don't
    // count the sret, so map the result number back to what it holds.
    unsigned offset = define.getSretAttr() ? 1 : 0;
    if (auto arg = define.getWrittenArgForCallResult(resultNumber))
      return declaredPropertiesAt(define.getArgAttrsAttr(), *arg + offset,
                                  kGuaranteesAttr);
    if (offset)
      return declaredPropertiesAt(define.getArgAttrsAttr(), 0, kGuaranteesAttr);
    return declaredPropertiesAt(define.getResAttrsAttr(),
                                resultNumber - define.getNumWrittenArgs(),
                                kGuaranteesAttr);
  }

  // A function that is not a tessera op can still declare what it returns.
  if (auto call = dyn_cast<LLVM::CallOp>(op))
    if (LLVM::LLVMFuncOp func = calleeOf(call))
      return declaredPropertiesAt(func.getResAttrsAttr(), resultNumber,
                                  kGuaranteesAttr);
  return nullptr;
}

// The properties the callee preserves into `result`, each with the values
// the call passed for that rule's inputs. Rules naming an input the call
// didn't pass are skipped.
SmallVector<std::pair<StringAttr, SmallVector<Value>>>
preservedFrom(OpResult result) {
  Operation *op = result.getOwner();
  ArrayAttr rules;
  // The value the call passed as each of the callee's own parameters, sret
  // included; null where it passed none.
  SmallVector<Value> passed;

  if (auto call = dyn_cast<CallOp>(op)) {
    auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, call.getCalleeAttr().getAttr());
    if (!define)
      return {};
    // A preserves rule applies to the return value, or, if the function
    // returns nothing, to the one argument it writes.
    unsigned offset = define.getSretAttr() ? 1 : 0;
    bool returnsValue = offset || define.getFunctionType().getNumResults() != 0;
    if (define.getWrittenArgForCallResult(result.getResultNumber()) &&
        (returnsValue || define.getNumWrittenArgs() != 1))
      return {};
    rules = define->getAttrOfType<ArrayAttr>(kPreservesAttr);
    passed.resize(define.getNumArguments());
    for (unsigned param = offset; param < passed.size(); ++param)
      if (std::optional<unsigned> operand =
              define.getCallOperandForArg(param - offset))
        if (*operand < call.getArgOperands().size())
          passed[param] = call.getArgOperands()[*operand];
  } else if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    LLVM::LLVMFuncOp func = calleeOf(call);
    if (!func)
      return {};
    rules = func->getAttrOfType<ArrayAttr>(kPreservesAttr);
    llvm::append_range(passed, call.getArgOperands());
  }
  if (!rules)
    return {};

  SmallVector<std::pair<StringAttr, SmallVector<Value>>> preserved;
  for (Attribute entry : rules) {
    auto rule = dyn_cast<DictionaryAttr>(entry);
    auto property = rule ? rule.getAs<StringAttr>("property") : nullptr;
    auto inputs = rule ? rule.getAs<ArrayAttr>("inputs") : nullptr;
    if (!property || !inputs || inputs.empty())
      continue;
    SmallVector<Value> values;
    for (Attribute index : inputs) {
      auto param = dyn_cast<IntegerAttr>(index);
      if (!param || param.getInt() < 0 ||
          param.getInt() >= (int64_t)passed.size() || !passed[param.getInt()])
        break;
      values.push_back(passed[param.getInt()]);
    }
    if (values.size() == inputs.size())
      preserved.push_back({property, std::move(values)});
  }
  return preserved;
}

// What a definition assumes of one of its own parameters on entry, or null
// if `arg` is not such a parameter.
Attribute assumptionsOnParameter(BlockArgument arg) {
  Block *block = arg.getOwner();
  auto func = dyn_cast<FunctionOpInterface>(block->getParentOp());
  if (!func || !block->isEntryBlock())
    return nullptr;
  return declaredPropertiesAt(func.getArgAttrsAttr(), arg.getArgNumber(),
                              kAssumesAttr);
}

//===----------------------------------------------------------------------===//
// Facts about handles
//===----------------------------------------------------------------------===//

// Operations that hand a pointer on as it is, or view the memory it points
// to as a memref.
bool isPointerCast(Operation *op) {
  return isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::FreezeOp,
             enzymexla::Pointer2MemrefOp>(op);
}

// The variable through which calls reach a handle's object. Either:
//  - a pointer variable holding the handle, `Mat A`. C code reloads A for
//    each use, so each call gets a different SSA value, all naming the same
//    object until something stores a new handle into A; or
//  - an object variable, `BilinearForm a`, whose address is the handle.
struct HandleVariable {
  // The variable's address and casts of it.
  DenseSet<Value> addresses;
  // The values calls are given as the handle: loads from a pointer variable,
  // or an object variable's address and casts of it.
  DenseSet<Value> handleValues;
  // Ops that may change what the handle names: stores, lifetime markers, and
  // calls given the variable's address (for an object, an address inside it).
  DenseSet<Operation *> modifyingOps;
  // Calls to fact markers given the variable's address. They state facts
  // about the handle and change nothing.
  DenseSet<Operation *> markerCalls;

  bool isHandleValue(Value value) const {
    return handleValues.contains(stripCasts(value));
  }
};

// Whether `pointer`, and any pointer offset from it, is only loaded from.
bool isOnlyLoadedThrough(Value pointer) {
  return llvm::all_of(pointer.getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    if (isPointerCast(user) ||
        (isa<LLVM::GEPOp>(user) && use.getOperandNumber() == 0))
      return isOnlyLoadedThrough(user->getResult(0));
    return isa<LLVM::LoadOp>(user) ||
           (isa<affine::AffineLoadOp, memref::LoadOp>(user) &&
            use.getOperandNumber() == 0);
  });
}

// Whether a loaded handle is only passed to calls or guards, compared, or
// loaded through (as `a->f()` loads the vtable). A copy stored anywhere else
// could reach a call the walk can't see.
bool isOnlyPassedToCalls(Value handle) {
  return llvm::all_of(handle.getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::FreezeOp>(user))
      return isOnlyPassedToCalls(user->getResult(0));
    // Given to a call, rather than called: an indirect call's callee is its
    // first operand.
    if (auto call = dyn_cast<LLVM::CallOp>(user))
      return call.getCalleeAttr() || use.getOperandNumber() != 0;
    // Read through. A load's one operand is its address.
    if (isa<LLVM::LoadOp>(user))
      return true;
    if (isa<enzymexla::Pointer2MemrefOp, LLVM::GEPOp>(user) &&
        use.getOperandNumber() == 0)
      return isOnlyLoadedThrough(user->getResult(0));
    return isa<CallOp, GuardOp, LLVM::ICmpOp>(user);
  });
}

// Whether `op` stores to `address`, as opposed to storing it.
bool isStoreTo(Operation *op, Value address) {
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return store.getAddr() == address;
  if (auto store = dyn_cast<affine::AffineStoreOp>(op))
    return store.getMemref() == address;
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getMemref() == address;
  return false;
}

// Whether `op` runs at most once per call of its function: it's in no loop,
// and in the entry block of every enclosing region.
bool runsAtMostOnce(Operation *op) {
  while (true) {
    Region *region = op->getParentRegion();
    if (!op->getBlock()->isEntryBlock())
      return false;
    Operation *parent = region->getParentOp();
    if (isa<FunctionOpInterface>(parent))
      return true;
    if (isa<LoopLikeOpInterface>(parent))
      return false;
    if (auto branch = dyn_cast<RegionBranchOpInterface>(parent))
      if (branch.isRepetitiveRegion(region->getRegionNumber()))
        return false;
    op = parent;
  }
}

// The variable through which `handle`, passed to `use`, reaches its object.
// Nothing if there isn't one, or if it's used in a way this can't track.
std::optional<HandleVariable> findHandleVariable(Value handle,
                                                 Operation *use) {
  // A handle in a pointer variable, `Mat A`, is loaded from it; an object
  // variable, `BilinearForm a(...)`, is its own handle.
  Value address = stripCasts(handle);
  Operation *load = address.getDefiningOp();
  bool isPointerVariable =
      load && isa<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(load);
  if (isPointerVariable) {
    // Loaded for `use` alone, so that a fact recorded on it is of this call
    // only (see objectFactsAt).
    if (address != handle ||
        llvm::any_of(handle.getUsers(),
                     [&](Operation *user) { return user != use; }))
      return std::nullopt;
    // A load's address is its first operand.
    address = load->getOperand(0);
    while (Operation *def = address.getDefiningOp()) {
      if (!isPointerCast(def))
        break;
      address = def->getOperand(0);
    }
  }

  auto alloca = address.getDefiningOp<LLVM::AllocaOp>();
  if (!alloca ||
      isa<LLVM::LLVMPointerType>(alloca.getElemType()) != isPointerVariable)
    return std::nullopt;
  // One pointer, so that every load reads the same one.
  if (isPointerVariable && !matchPattern(alloca.getArraySize(), m_One()))
    return std::nullopt;

  HandleVariable variable;
  // Each address, and whether it is of the whole variable rather than of a
  // field inside an object.
  SmallVector<std::pair<Value, bool>> worklist{{alloca.getResult(), true}};
  while (!worklist.empty()) {
    auto [current, whole] = worklist.pop_back_val();
    if (whole) {
      if (!variable.addresses.insert(current).second)
        continue;
      if (!isPointerVariable)
        variable.handleValues.insert(current);
    }
    for (OpOperand &operand : current.getUses()) {
      Operation *user = operand.getOwner();
      if (isPointerCast(user)) {
        worklist.push_back({user->getResult(0), whole});
        continue;
      }
      // An address inside an object. A pointer variable has no fields.
      if (!isPointerVariable && isa<LLVM::GEPOp>(user) &&
          operand.getOperandNumber() == 0) {
        worklist.push_back({user->getResult(0), false});
        continue;
      }
      // Loading from a pointer variable gives the handle; loading from an
      // object only reads it.
      if (isa<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(user)) {
        if (isPointerVariable)
          variable.handleValues.insert(user->getResult(0));
        continue;
      }
      // A fact marker's body is empty, so it doesn't change the variable. Any
      // other call given its address might, as MatDestroy(&A) does.
      auto llvmCall = dyn_cast<LLVM::CallOp>(user);
      LLVM::LLVMFuncOp callee =
          whole && llvmCall ? calleeOf(llvmCall) : nullptr;
      if (callee && callee->hasAttr(kFactMarkerAttr)) {
        variable.markerCalls.insert(user);
        continue;
      }
      // A call given an object's address is given the handle; what it does
      // to the object is for the walk to decide.
      if (whole && !isPointerVariable && isa<CallOp, LLVM::CallOp>(user))
        continue;
      if (isStoreTo(user, current) ||
          isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp, CallOp, LLVM::CallOp>(
              user)) {
        variable.modifyingOps.insert(user);
        continue;
      }
      // Its address escapes, or is offset into: anything could write it.
      return std::nullopt;
    }
  }
  if (!isPointerVariable)
    return variable;

  // If `use` runs only once, a handle loaded after it may be copied anywhere
  // (as `bfs.Append(form)` does): the copy can't reach a call before `use`.
  bool useRunsOnce = runsAtMostOnce(use);
  auto isLoadedAfterUse = [&](Value load) {
    if (!useRunsOnce)
      return false;
    for (Operation *at = use; at; at = at->getParentOp()) {
      Operation *op =
          at->getParentRegion()->findAncestorOpInRegion(*load.getDefiningOp());
      if (op)
        return op->getBlock() != at->getBlock() || at->isBeforeInBlock(op);
    }
    return false;
  };
  if (!llvm::all_of(variable.handleValues, [&](Value load) {
        return isLoadedAfterUse(load) || isOnlyPassedToCalls(load);
      }))
    return std::nullopt;
  return variable;
}

// The attributes of the parameter a call passes argument `index` to, or null.
DictionaryAttr calleeParamAttrs(Operation *op, unsigned index) {
  if (auto call = dyn_cast<CallOp>(op)) {
    auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, call.getCalleeAttr().getAttr());
    if (!define)
      return nullptr;
    std::optional<unsigned> param = define.getArgIndexForCallOperand(index);
    return param ? attrDictAt(define.getArgAttrsAttr(), *param) : nullptr;
  }
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    LLVM::LLVMFuncOp func = calleeOf(call);
    // Past the declared parameters, a variadic argument has none.
    if (!func || index >= func.getNumArguments())
      return nullptr;
    return attrDictAt(func.getArgAttrsAttr(), index);
  }
  return nullptr;
}

// What an operation does to the object a handle names.
enum class HandleEffect { None, Establishes, Changes };

// What `op` itself (not ops nested in it) does to the handle's object.
// Properties it establishes are appended to `established`.
HandleEffect effectOnHandle(Operation *op, const HandleVariable &variable,
                            SmallVectorImpl<StringAttr> &established) {
  if (variable.modifyingOps.contains(op))
    return HandleEffect::Changes;

  // A fact marker does nothing, so what it assumes on entry still holds on
  // exit: both its establishes and its assumes count.
  if (variable.markerCalls.contains(op)) {
    for (auto [index, arg] :
         llvm::enumerate(cast<LLVM::CallOp>(op).getArgOperands()))
      if (variable.addresses.contains(arg))
        if (DictionaryAttr attrs = calleeParamAttrs(op, index)) {
          appendPropertyNames(established, attrs.get(kEstablishesAttr));
          appendPropertyNames(established, attrs.get(kAssumesAttr));
        }
    return established.empty() ? HandleEffect::None : HandleEffect::Establishes;
  }

  ValueRange args;
  if (auto call = dyn_cast<CallOp>(op))
    args = call.getArgOperands();
  else if (auto call = dyn_cast<LLVM::CallOp>(op))
    args = call.getArgOperands();
  else
    // Only calls can be given a handle (see isOnlyPassedToCalls); a guard
    // only reads it, and a comparison only compares it.
    return HandleEffect::None;

  // A call that establishes a property does so on exit, whatever else it
  // does to the object on the way.
  bool establishes = false, changes = false;
  for (auto [index, arg] : llvm::enumerate(args)) {
    if (!variable.isHandleValue(arg))
      continue;
    DictionaryAttr attrs = calleeParamAttrs(op, index);
    if (Attribute properties = attrs ? attrs.get(kEstablishesAttr) : nullptr) {
      appendPropertyNames(established, properties);
      establishes = true;
    } else if (!attrs || !attrs.get(kReadonlyAttr)) {
      changes = true;
    }
  }
  return establishes ? HandleEffect::Establishes
         : changes   ? HandleEffect::Changes
                     : HandleEffect::None;
}

// The properties a call keeps on the handle's object even if it changes it:
// those a void callee preserves from that object alone, as
// `tessera_preserves("P", "this")` on a setter declares.
SmallVector<StringAttr> propertiesKeptBy(Operation *op,
                                         const HandleVariable &variable) {
  SmallVector<StringAttr> kept;
  auto call = dyn_cast<LLVM::CallOp>(op);
  LLVM::LLVMFuncOp func = call ? calleeOf(call) : nullptr;
  auto preserves =
      func ? func->getAttrOfType<ArrayAttr>(kPreservesAttr) : nullptr;
  if (!preserves || call.getNumResults() != 0)
    return kept;
  for (auto rule : preserves.getAsRange<DictionaryAttr>()) {
    auto inputs = rule.getAs<ArrayAttr>("inputs");
    auto position = inputs && inputs.size() == 1
                        ? dyn_cast<IntegerAttr>(inputs[0])
                        : nullptr;
    if (position && position.getInt() >= 0 &&
        position.getInt() < (int64_t)call.getArgOperands().size() &&
        variable.isHandleValue(call.getArgOperands()[position.getInt()]))
      kept.push_back(rule.getAs<StringAttr>("property"));
  }
  return kept;
}

// Whether anything in `op`, nested ops included, might change the handle's
// object. An establishing call counts too: it might not run, or might run
// before a change.
bool mayChangeHandle(Operation *op, const HandleVariable &variable) {
  return op
      ->walk([&](Operation *nested) {
        SmallVector<StringAttr> ignored;
        return effectOnHandle(nested, variable, ignored) == HandleEffect::None
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      })
      .wasInterrupted();
}

// The properties established for the object that argument `index` of `use`
// names, and not changed since.
//
// Walks back from `use` through earlier ops, then out through enclosing ops.
// Gives up at an op with regions (such as `scf.if`) or a loop that might
// change the object, at a region with several blocks, and at the function's
// start.
SmallVector<StringAttr> objectFactsAt(CallOp use, unsigned index) {
  Value handle = use.getArgOperands()[index];
  if (!isa<LLVM::LLVMPointerType>(handle.getType()))
    return {};
  // A fact recorded on an argument is read as true at every use of that
  // value, but an object fact holds only here. So it's recorded only on a
  // handle loaded for this call alone, or on a copy recordFactsOnCall makes.
  std::optional<HandleVariable> variable = findHandleVariable(handle, use);
  if (!variable)
    return {};

  Operation *point = use;
  // If a call on the way changed the object, the properties it kept.
  std::optional<SmallVector<StringAttr>> kept;
  auto keep = [&](SmallVector<StringAttr> &facts) {
    if (kept)
      llvm::erase_if(
          facts, [&](StringAttr p) { return !llvm::is_contained(*kept, p); });
  };
  while (true) {
    for (Operation *op = point->getPrevNode(); op; op = op->getPrevNode()) {
      if (op->getNumRegions() != 0) {
        if (mayChangeHandle(op, *variable))
          return {};
        continue;
      }
      SmallVector<StringAttr> established;
      switch (effectOnHandle(op, *variable, established)) {
      case HandleEffect::None:
        continue;
      case HandleEffect::Establishes:
        keep(established);
        return established;
      case HandleEffect::Changes: {
        SmallVector<StringAttr> keptHere = propertiesKeptBy(op, *variable);
        keep(keptHere);
        if (keptHere.empty())
          return {};
        kept = std::move(keptHere);
        continue;
      }
      }
    }

    Region *region = point->getParentRegion();
    Operation *parent = region->getParentOp();
    if (!region->hasOneBlock() || !parent || isa<FunctionOpInterface>(parent))
      return {};
    bool repeats;
    if (isa<GuardOp>(parent))
      repeats = false;
    else if (auto branch = dyn_cast<RegionBranchOpInterface>(parent))
      repeats = branch.isRepetitiveRegion(region->getRegionNumber());
    else if (isa<LoopLikeOpInterface>(parent))
      repeats = true;
    else
      return {};
    if (repeats && mayChangeHandle(parent, *variable))
      return {};
    point = parent;
  }
}

// Whether having `have` implies `want`, directly or through a chain (SPD
// implies symmetric).
bool impliesProperty(llvm::StringRef have, llvm::StringRef want) {
  SmallVector<llvm::StringRef> worklist{have};
  llvm::SmallDenseSet<llvm::StringRef> seen;
  seen.insert(have);
  while (!worklist.empty()) {
    llvm::StringRef current = worklist.pop_back_val();
    if (current == want)
      return true;
    for (const Implication &implication : kImplications)
      if (implication.from == current && seen.insert(implication.to).second)
        worklist.push_back(implication.to);
  }
  return false;
}

} // namespace

SmallVector<StringAttr> PropertyAnalysis::cachedValueFacts(Value value,
                                                           unsigned depth) {
  value = stripCasts(value);
  if (auto it = valueFactsCache.find(value); it != valueFactsCache.end())
    return it->second;
  // Cached as empty while computing, so a cycle (only possible in a graph
  // region) stops instead of recursing forever.
  valueFactsCache[value] = {};
  SmallVector<StringAttr> facts = computeValueFacts(value, depth);
  valueFactsCache[value] = facts;
  return facts;
}

SmallVector<StringAttr> PropertyAnalysis::computeValueFacts(Value value,
                                                            unsigned depth) {
  SmallVector<StringAttr> facts;

  // Facts already recorded at any call the value is passed to. A value's
  // facts hold at all its uses.
  for (OpOperand &use : value.getUses())
    if (auto call = dyn_cast<CallOp>(use.getOwner()))
      appendRecordedFacts(
          facts, attrDictAt(call.getArgAttrsAttr(), use.getOperandNumber()));

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    appendPropertyNames(facts, assumptionsOnParameter(arg));
    return facts;
  }

  Operation *op = value.getDefiningOp();
  if (op->getNumResults() == 1)
    appendPropertyNames(facts, op->getAttr(kPropertyAttr));

  if (depth >= kMaxDerivationDepth)
    return facts;

  // A value loaded from a parameter gets the parameter's assumptions, if the
  // parameter is only ever loaded from (so nothing changed what it points to).
  if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
    Value address = stripCasts(load.getAddr());
    if (auto arg = dyn_cast<BlockArgument>(address))
      if (llvm::all_of(arg.getUsers(),
                       [](Operation *user) { return isa<LLVM::LoadOp>(user); }))
        appendPropertyNames(facts, assumptionsOnParameter(arg));
    return facts;
  }

  // Either branch of a guard may have produced the result, so only what is
  // known of both holds.
  if (auto guard = dyn_cast<GuardOp>(op)) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    auto yielded = [&](Region &region) -> SmallVector<StringAttr> {
      if (region.empty())
        return {};
      Operation *terminator = region.front().getTerminator();
      if (!terminator || index >= terminator->getNumOperands())
        return {};
      return cachedValueFacts(terminator->getOperand(index), depth + 1);
    };
    SmallVector<StringAttr> thenFacts = yielded(guard.getThenRegion());
    SmallVector<StringAttr> elseFacts = yielded(guard.getElseRegion());
    for (StringAttr name : thenFacts)
      if (llvm::is_contained(elseFacts, name) &&
          !llvm::is_contained(facts, name))
        facts.push_back(name);
    return facts;
  }

  auto result = cast<OpResult>(value);

  // Guaranteed by the callee, whatever its inputs were.
  appendPropertyNames(facts, guaranteesOf(result));

  // Preserved by the callee: holds of the output if it held of every input
  // the declaration lists.
  for (const auto &[property, inputs] : preservedFrom(result)) {
    if (llvm::is_contained(facts, property))
      continue;
    bool everyInputHasIt = true;
    for (Value input : inputs) {
      bool hasIt = false;
      for (StringAttr known : cachedValueFacts(input, depth + 1))
        hasIt = hasIt || impliesProperty(known.getValue(), property.getValue());
      everyInputHasIt = everyInputHasIt && hasIt;
    }
    if (everyInputHasIt)
      facts.push_back(property);
  }
  return facts;
}

void PropertyAnalysis::recordFactsOnCall(CallOp call) {
  MLIRContext *ctx = call.getContext();
  unsigned numArgs = call.getArgOperands().size();
  ArrayAttr existing = call.getArgAttrsAttr();

  SmallVector<Attribute> dicts;
  bool any = false;
  for (unsigned i = 0; i != numArgs; ++i) {
    DictionaryAttr dict = attrDictAt(existing, i);
    SmallVector<StringAttr> facts;
    appendRecordedFacts(facts, dict);
    for (StringAttr name :
         cachedValueFacts(call.getArgOperands()[i], /*depth=*/0))
      if (!llvm::is_contained(facts, name))
        facts.push_back(name);
    SmallVector<StringAttr> objectFacts = objectFactsAt(call, i);
    for (StringAttr name : objectFacts)
      if (!llvm::is_contained(facts, name))
        facts.push_back(name);
    // Give the call its own copy, so the object fact isn't read as true at
    // the value's other uses.
    Value arg = call.getArgOperands()[i];
    if (!objectFacts.empty() &&
        llvm::any_of(arg.getUsers(),
                     [&](Operation *user) { return user != call; })) {
      OpBuilder builder(call);
      call.getArgOperandsMutable()[i].set(
          LLVM::FreezeOp::create(builder, call.getLoc(), arg));
    }

    NamedAttrList attrs(dict ? dict : DictionaryAttr::get(ctx));
    if (!facts.empty()) {
      llvm::sort(facts, [](StringAttr a, StringAttr b) {
        return a.getValue() < b.getValue();
      });
      attrs.set(kPropertyAttr,
                ArrayAttr::get(
                    ctx, SmallVector<Attribute>(facts.begin(), facts.end())));
    }
    any |= !attrs.empty();
    dicts.push_back(attrs.getDictionary(ctx));
  }
  if (any)
    call.setArgAttrsAttr(ArrayAttr::get(ctx, dicts));
}

bool mlir::enzyme::tessera::hasRecordedProperty(Value value,
                                                llvm::StringRef property) {
  for (OpOperand &use : value.getUses()) {
    auto call = dyn_cast<CallOp>(use.getOwner());
    if (!call)
      continue;
    DictionaryAttr dict =
        attrDictAt(call.getArgAttrsAttr(), use.getOperandNumber());
    auto list = dict ? dict.getAs<ArrayAttr>(kPropertyAttr) : nullptr;
    if (!list)
      continue;
    for (Attribute entry : list)
      if (auto name = dyn_cast<StringAttr>(entry))
        if (impliesProperty(name.getValue(), property))
          return true;
  }
  return false;
}

namespace {

struct TesseraPropagatePropertiesPass
    : public enzyme::tessera::impl::TesseraPropagatePropertiesPassBase<
          TesseraPropagatePropertiesPass> {
  using TesseraPropagatePropertiesPassBase::TesseraPropagatePropertiesPassBase;

  void runOnOperation() override {
    PropertyAnalysis facts;
    getOperation()->walk([&](CallOp call) { facts.recordFactsOnCall(call); });
  }
};

} // namespace
