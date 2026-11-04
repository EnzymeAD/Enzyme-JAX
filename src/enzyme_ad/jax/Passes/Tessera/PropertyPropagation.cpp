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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
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

// The memory a pointer points into, for telling whether two pointers can
// reach the same object: a stack variable, a global, a parameter, or a value
// nothing here derives from another, such as a call result. A pointer loaded
// from memory counts as part of that memory, since whatever reaches the
// memory reaches it too.
Value pointerRoot(Value pointer) {
  while (Operation *def = pointer.getDefiningOp()) {
    if (!isPointerCast(def) &&
        !isa<LLVM::GEPOp, LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(
            def))
      break;
    pointer = def->getOperand(0);
  }
  return pointer;
}

// Whether a value of `type` can contain a pointer.
bool typeHasPointer(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return typeHasPointer(array.getElementType());
  if (auto shaped = dyn_cast<ShapedType>(type))
    return typeHasPointer(shaped.getElementType());
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type))
    return structType.isOpaque() ||
           llvm::any_of(structType.getBody(),
                        [](Type field) { return typeHasPointer(field); });
  return false;
}

// Whether `root` may be memory the function can reach through a pointer
// argument: not a global (see PropertyPropagation.h) or a constant.
bool isMemory(Value root) {
  return !isa_and_nonnull<LLVM::AddressOfOp, LLVM::ZeroOp, LLVM::UndefOp,
                          LLVM::PoisonOp, LLVM::ConstantOp>(
      root.getDefiningOp());
}

// Whether the memory at `root` can hold a pointer to an object, for a later
// call to reach the object through. A null pointer or other constant holds
// nothing, and nor does a stack variable whose type has no pointer in it,
// such as `PetscInt idx[4]`. A global, PETSC_COMM_WORLD say, is left out by
// the encapsulation assumption (see PropertyPropagation.h).
bool mayBeHolder(Value root) {
  Operation *def = root.getDefiningOp();
  if (!def)
    return true;
  if (!isMemory(root))
    return false;
  if (auto alloca = dyn_cast<LLVM::AllocaOp>(def))
    return typeHasPointer(alloca.getElemType());
  if (auto alloca = dyn_cast<memref::AllocaOp>(def))
    return typeHasPointer(alloca.getType().getElementType());
  return true;
}

// A call given pointers, as the search for holders sees it.
struct PointerCall {
  Operation *call;
  // Per argument, the root (see pointerRoot) of the pointer passed, or null
  // for an argument that isn't one.
  SmallVector<Value> roots;
  // Per argument, whether the call may store it.
  SmallVector<bool> mayStore;
};

// The variable through which calls reach a handle's object. One of:
//  - a pointer variable holding the handle, `Mat A`. C code reloads A for
//    each use, so each call gets a different SSA value, all naming the same
//    object until something stores a new handle into A;
//  - an object variable, `BilinearForm a`, whose address is the handle; or
//  - a pointer parameter the function assumes something of, `IntVec &ids`,
//    which is the handle. Its object is as assumed on entry. Unlike a
//    variable's, it was there before the call, so other pointers -- another
//    parameter, a global -- may reach it too; or
//  - a pointer a call returned, `double *G = form_gram_matrix(...)`, which is
//    the handle. Its object is as that call left it.
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
  // The parameter, if the handle is one.
  BlockArgument parameter;
  // The call result, if the handle is one.
  OpResult producer;
  // Where the variable is, as pointerRoot gives it: its alloca, the
  // parameter, or the call result.
  Value root;
  // The calls in the function given pointers, found the first time
  // reachingRootsAt needs them, and its answers by point.
  mutable std::optional<SmallVector<PointerCall>> pointerCalls;
  mutable DenseMap<Operation *, DenseSet<Value>> reachingRoots;
  // Ops a rewrite is about to remove: they neither change the object nor
  // store it anywhere.
  ArrayRef<Operation *> replaced;

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

// Fill in `variable` from the uses of `root`, the variable's address. False if
// it's used in a way this can't track.
bool collectVariableUses(Value root, bool isPointerVariable,
                         HandleVariable &variable) {
  // Each address, and whether it is of the whole variable rather than of a
  // field inside an object.
  SmallVector<std::pair<Value, bool>> worklist{{root, true}};
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
      return false;
    }
  }
  return true;
}

// The handle variable for a pointer parameter the function assumes something
// of, or nothing if `arg` isn't one or is used in a way this can't track.
std::optional<HandleVariable> findParameterVariable(BlockArgument arg) {
  if (!isa<LLVM::LLVMPointerType>(arg.getType()) ||
      !assumptionsOnParameter(arg))
    return std::nullopt;
  HandleVariable variable;
  variable.parameter = arg;
  variable.root = arg;
  if (!collectVariableUses(arg, /*isPointerVariable=*/false, variable))
    return std::nullopt;
  return variable;
}

// The variable through which `handle`, passed to `use`, reaches its object.
// Nothing if there isn't one, or if it's used in a way this can't track.
std::optional<HandleVariable> findHandleVariable(Value handle, Operation *use) {
  // A handle in a pointer variable, `Mat A`, is loaded from it; an object
  // variable, `BilinearForm a(...)`, is its own handle, and so is a pointer
  // parameter, `IntVec &ids`.
  Value address = stripCasts(handle);
  if (auto arg = dyn_cast<BlockArgument>(address))
    return findParameterVariable(arg);
  Operation *load = address.getDefiningOp();
  bool isPointerVariable =
      load && isa<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(load);
  if (isPointerVariable) {
    // The load itself, not a copy recordFactsOnCall made of it. The load may
    // be shared with other calls, as when the variable's address doesn't
    // escape and LLVM reuses one load; recordFactsOnCall then gives `use` its
    // own copy.
    if (address != handle)
      return std::nullopt;
    // A load's address is its first operand.
    address = load->getOperand(0);
    while (Operation *def = address.getDefiningOp()) {
      if (!isPointerCast(def))
        break;
      address = def->getOperand(0);
    }
  } else if (auto result = dyn_cast<OpResult>(address)) {
    // A pointer a call returned. A fact the callee gave it is about its
    // object, so it holds only until something changes the object. Only an
    // annotated callee gives a fact, and an annotated callee returns an
    // object that none of its arguments holds (see PropertyPropagation.h),
    // so at first only the result reaches it.
    if (isa<CallOp, LLVM::CallOp>(result.getOwner()) &&
        isa<LLVM::LLVMPointerType>(result.getType())) {
      HandleVariable variable;
      variable.producer = result;
      variable.root = result;
      if (!collectVariableUses(result, /*isPointerVariable=*/false, variable))
        return std::nullopt;
      return variable;
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
  variable.root = alloca.getResult();
  if (!collectVariableUses(alloca.getResult(), isPointerVariable, variable))
    return std::nullopt;
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

//===----------------------------------------------------------------------===//
// Holders
//===----------------------------------------------------------------------===//

// Whether `op`, a call, may store the pointer it is given as argument `index`
// where a later call can reach it. Any call may, annotated or not, unless the
// parameter says otherwise (see PropertyPropagation.h).
bool mayStoreArgument(Operation *op, unsigned index) {
  auto marks = [](DictionaryAttr attrs) {
    return attrs && (attrs.get(kNoEscapeAttr) ||
                     attrs.get(LLVM::LLVMDialect::getNoCaptureAttrName()));
  };
  if (marks(calleeParamAttrs(op, index)))
    return false;
  if (auto call = dyn_cast<CallOp>(op)) {
    // tessera.call keeps the call site's arg_attrs, one per operand.
    auto siteAttrs = op->getAttrOfType<ArrayAttr>("arg_attrs");
    return !marks(attrDictAt(siteAttrs, index));
  }
  auto call = dyn_cast<LLVM::CallOp>(op);
  LLVM::LLVMFuncOp func = call ? calleeOf(call) : nullptr;
  // An indirect call, or a variadic argument, could be anything.
  if (!func || index >= func.getNumArguments())
    return true;
  if (func->hasAttr(kFactMarkerAttr))
    return false;
  return !marks(attrDictAt(call.getArgAttrsAttr(), index));
}

// The calls in `func` given at least one pointer.
SmallVector<PointerCall> pointerCallsIn(Operation *func) {
  SmallVector<PointerCall> calls;
  func->walk([&](Operation *op) {
    ValueRange args;
    if (auto call = dyn_cast<CallOp>(op))
      args = call.getArgOperands();
    else if (auto call = dyn_cast<LLVM::CallOp>(op))
      args = call.getArgOperands();
    else
      return;
    PointerCall entry{op, {}, {}};
    bool anyPointer = false;
    for (auto [index, arg] : llvm::enumerate(args)) {
      bool isPointer = isa<LLVM::LLVMPointerType, MemRefType>(arg.getType());
      entry.roots.push_back(isPointer ? pointerRoot(arg) : Value());
      entry.mayStore.push_back(isPointer && mayStoreArgument(op, index));
      anyPointer |= isPointer;
    }
    if (anyPointer)
      calls.push_back(std::move(entry));
  });
  return calls;
}

// `op` and the ops enclosing it, outermost first, up to the function.
SmallVector<Operation *> enclosingOps(Operation *op) {
  SmallVector<Operation *> chain;
  for (; op && !isa<FunctionOpInterface>(op); op = op->getParentOp())
    chain.push_back(op);
  std::reverse(chain.begin(), chain.end());
  return chain;
}

// Whether control can get from the end of block `from` to the start of block
// `to`, in the same region.
bool blockReaches(Block *from, Block *to) {
  SmallVector<Block *> worklist(from->getSuccessors());
  DenseSet<Block *> seen;
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (block == to)
      return true;
    if (seen.insert(block).second)
      llvm::append_range(worklist, block->getSuccessors());
  }
  return false;
}

// Whether ops in `block` may run more than once per call of the function: in
// a loop, a region that repeats, or a block that branches back to itself.
bool mayRepeat(Block *block) {
  for (; block; block = block->getParentOp() ? block->getParentOp()->getBlock()
                                             : nullptr) {
    if (blockReaches(block, block))
      return true;
    Operation *parent = block->getParentOp();
    if (!parent || isa<FunctionOpInterface>(parent))
      return false;
    if (isa<LoopLikeOpInterface>(parent))
      return true;
    if (auto branch = dyn_cast<RegionBranchOpInterface>(parent))
      if (branch.isRepetitiveRegion(block->getParent()->getRegionNumber()))
        return true;
  }
  return false;
}

// Whether `earlier` may run before `later` in one call of the function.
bool mayRunBefore(Operation *earlier, Operation *later) {
  SmallVector<Operation *> a = enclosingOps(earlier), b = enclosingOps(later);
  size_t level = 0;
  while (level < a.size() && level < b.size() && a[level] == b[level])
    ++level;
  // The same op, or one inside the other.
  if (level == a.size() || level == b.size())
    return earlier != later || mayRepeat(earlier->getBlock());
  Operation *x = a[level], *y = b[level];
  // In different regions of one op: both may run unless the regions exclude
  // each other, as an if's do, and the op runs once.
  if (x->getParentRegion() != y->getParentRegion())
    return !insideMutuallyExclusiveRegions(earlier, later) ||
           mayRepeat(x->getParentOp()->getBlock());
  Block *xBlock = x->getBlock(), *yBlock = y->getBlock();
  if (xBlock == yBlock && x->isBeforeInBlock(y))
    return true;
  if (xBlock != yBlock && blockReaches(xBlock, yBlock))
    return true;
  return mayRepeat(yBlock);
}

// For a returned pointer, the roots of the pointer arguments the call that
// returned it may have taken it from.
SmallVector<Value> producerArgumentRoots(const HandleVariable &variable) {
  SmallVector<Value> roots;
  if (!variable.producer)
    return roots;
  Operation *call = variable.producer.getOwner();
  ValueRange args;
  if (auto tesseraCall = dyn_cast<CallOp>(call))
    args = tesseraCall.getArgOperands();
  else
    args = cast<LLVM::CallOp>(call).getArgOperands();
  for (auto [index, arg] : llvm::enumerate(args))
    if (isa<LLVM::LLVMPointerType, MemRefType>(arg.getType()) &&
        mayStoreArgument(call, index) && isMemory(pointerRoot(arg)))
      roots.push_back(pointerRoot(arg));
  return roots;
}

// Whether `call`, given the pointer a call returned, may run between that call
// and `point`, and so be given the object `point` sees. In a loop, a call
// later in the body than `point` runs before it only in an earlier iteration,
// which the returning call ran in again after, as `free(G)` and the solve do
// in `for (...) { G = form(V); solve(G, b); free(G); }`: each was given the
// G of the iteration before.
bool mayRunSinceProduced(const HandleVariable &variable, Operation *call,
                         Operation *point) {
  Block *block = variable.producer.getOwner()->getBlock();
  Operation *callAt = block->findAncestorOpInBlock(*call);
  Operation *pointAt = block->findAncestorOpInBlock(*point);
  if (!callAt || !pointAt)
    return mayRunBefore(call, point);
  // Both inside one op after the returning call, such as a loop: an earlier
  // iteration of it, the point's own included, was given the same object.
  if (callAt == pointAt)
    return call != point || pointAt != point;
  return callAt->isBeforeInBlock(pointAt);
}

// The roots of the pointers through which a call at `point` can reach the
// variable's object: the variable's own, and its holders' -- pointers an
// earlier call may have stored it in, and for a returned pointer, those the
// call that returned it was given and may hand it out of.
const DenseSet<Value> &reachingRootsAt(const HandleVariable &variable,
                                       Operation *point) {
  if (auto it = variable.reachingRoots.find(point);
      it != variable.reachingRoots.end())
    return it->second;
  if (!variable.pointerCalls)
    variable.pointerCalls =
        pointerCallsIn(point->getParentOfType<FunctionOpInterface>());

  DenseSet<Value> roots{variable.root};
  auto add = [&](Value root) {
    return root && mayBeHolder(root) && roots.insert(root).second;
  };
  // A getter returns what an argument holds, or a pointer into it, as
  // `v.data()` does; a noescape argument has nothing it lets out. Even memory
  // with no pointer in it, `double m[16]`, may be what a row points into.
  for (Value root : producerArgumentRoots(variable))
    roots.insert(root);
  SmallVector<const PointerCall *> before;
  for (const PointerCall &call : *variable.pointerCalls)
    if (!llvm::is_contained(variable.replaced, call.call) &&
        mayRunBefore(call.call, point))
      before.push_back(&call);
  // Holders of holders count too, so repeat until nothing is added.
  bool added = true;
  while (added) {
    added = false;
    for (const PointerCall *call : before) {
      bool storesOne = false;
      for (auto [root, mayStore] : llvm::zip(call->roots, call->mayStore))
        storesOne |= mayStore && root && roots.contains(root) &&
                     (root != variable.root || !variable.producer ||
                      mayRunSinceProduced(variable, call->call, point));
      if (!storesOne)
        continue;
      // It may have stored the pointer in any other pointer it was given, or
      // in what it returned.
      for (Value root : call->roots)
        added |= add(root);
      for (Value result : call->call->getResults())
        if (isa<LLVM::LLVMPointerType>(result.getType()))
          added |= add(result);
    }
  }
  return variable.reachingRoots[point] = std::move(roots);
}

// Whether `pointer`, given to or written through by `op`, can reach the
// variable's object.
bool reachesObject(const HandleVariable &variable, Operation *op,
                   Value pointer) {
  if (!isa<LLVM::LLVMPointerType, MemRefType>(pointer.getType()))
    return false;
  return reachingRootsAt(variable, op).contains(pointerRoot(pointer));
}

// Whether `op`, not a call, writes through a pointer it loaded from a holder,
// as `ksp->A->data[0] = x` would. A store into the holder itself, replacing
// the pointer it holds, changes nothing the object has.
bool writesThroughHolder(Operation *op, const HandleVariable &variable) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](MemoryEffects::EffectInstance &instance) {
    Value address = instance.getValue();
    if (!address ||
        !isa<MemoryEffects::Write, MemoryEffects::Free>(instance.getEffect()))
      return false;
    // Only through a pointer loaded from memory, and so not the holder's own.
    Value through = address;
    while (Operation *def = through.getDefiningOp()) {
      if (!isPointerCast(def) && !isa<LLVM::GEPOp>(def))
        break;
      through = def->getOperand(0);
    }
    return isa_and_nonnull<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(
               through.getDefiningOp()) &&
           reachesObject(variable, op, address);
  });
}

// Whether `op`, not a call, writes straight into memory the call that
// returned the variable's pointer was given, which that pointer may point
// into, as `m[0] = x` does after `p = row(m, 0)`.
bool writesIntoProducerArgument(Operation *op, const HandleVariable &variable) {
  if (!variable.producer)
    return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<Value> roots = producerArgumentRoots(variable);
  if (roots.empty())
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](MemoryEffects::EffectInstance &instance) {
    Value address = instance.getValue();
    if (!address ||
        !isa<MemoryEffects::Write, MemoryEffects::Free>(instance.getEffect()))
      return false;
    while (Operation *def = address.getDefiningOp()) {
      if (!isPointerCast(def) && !isa<LLVM::GEPOp>(def))
        break;
      address = def->getOperand(0);
    }
    return llvm::is_contained(roots, address);
  });
}

// Whether `address` points into one of the function's own stack variables,
// which nothing from before the call can reach.
bool isLocalMemory(Value address) {
  while (Operation *def = address.getDefiningOp()) {
    if (isa<LLVM::AllocaOp, memref::AllocaOp>(def))
      return true;
    if (!isPointerCast(def) && !isa<LLVM::GEPOp>(def))
      return false;
    address = def->getOperand(0);
  }
  return false;
}

// Whether `op` itself (not ops nested in it) might write memory other than
// the function's own stack variables.
bool mayWriteNonLocal(Operation *op) {
  if (op->getNumRegions() != 0 ||
      isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(op))
    return false;
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    LLVM::LLVMFuncOp callee = calleeOf(call);
    if (callee && callee->hasAttr(kFactMarkerAttr))
      return false;
    LLVM::MemoryEffectsAttr memory = call.getMemoryEffectsAttr();
    if (!memory && callee)
      memory = callee.getMemoryEffectsAttr();
    if (!memory)
      return true;
    auto writes = [](LLVM::ModRefInfo info) {
      return info == LLVM::ModRefInfo::Mod || info == LLVM::ModRefInfo::ModRef;
    };
    // Through its arguments, it can write only what they point to.
    return writes(memory.getOther()) ||
           (writes(memory.getArgMem()) &&
            llvm::any_of(call.getArgOperands(), [](Value arg) {
              return isa<LLVM::LLVMPointerType>(arg.getType()) &&
                     !isLocalMemory(arg);
            }));
  }
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return true;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [](MemoryEffects::EffectInstance &instance) {
    return isa<MemoryEffects::Write, MemoryEffects::Free>(
               instance.getEffect()) &&
           (!instance.getValue() || !isLocalMemory(instance.getValue()));
  });
}

// What an operation does to the object a handle names.
enum class HandleEffect { None, Establishes, Changes };

// What `op` itself (not ops nested in it) does to the handle's object.
// Properties it establishes are appended to `established`.
HandleEffect effectOnHandle(Operation *op, const HandleVariable &variable,
                            SmallVectorImpl<StringAttr> &established) {
  if (llvm::is_contained(variable.replaced, op))
    return HandleEffect::None;
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
    // only reads it, and a comparison only compares it. But a parameter's
    // object may also be reached through another pointer, and written, any
    // object through a pointer loaded from a holder, and a returned one
    // through what its call was given.
    return (variable.parameter && mayWriteNonLocal(op)) ||
                   writesThroughHolder(op, variable) ||
                   writesIntoProducerArgument(op, variable)
               ? HandleEffect::Changes
               : HandleEffect::None;

  // A fact marker given another variable changes nothing.
  if (auto call = dyn_cast<LLVM::CallOp>(op))
    if (LLVM::LLVMFuncOp callee = calleeOf(call))
      if (callee->hasAttr(kFactMarkerAttr))
        return HandleEffect::None;

  // A call that establishes a property does so on exit, whatever else it
  // does to the object on the way.
  bool establishes = false, changes = false, given = false;
  for (auto [index, arg] : llvm::enumerate(args)) {
    DictionaryAttr attrs = calleeParamAttrs(op, index);
    bool unchanged = attrs && attrs.get(kUnchangedAttr);
    if (!variable.isHandleValue(arg)) {
      // Given a holder, or a pointer loaded from inside the object, the call
      // can reach the object through it. Leaving that argument unchanged
      // leaves what it holds unchanged too.
      if (!unchanged && reachesObject(variable, op, arg))
        changes = true;
      continue;
    }
    given = true;
    if (Attribute properties = attrs ? attrs.get(kEstablishesAttr) : nullptr) {
      appendPropertyNames(established, properties);
      establishes = true;
    } else if (!unchanged) {
      changes = true;
    }
  }
  // A call not given a parameter's object may still reach it another way.
  // One given it unchanged doesn't change it any other way.
  if (!given && variable.parameter)
    changes = changes || mayWriteNonLocal(op);
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

// The properties established for the variable's object before `point`, and
// not changed since.
//
// Walks back from `point` through earlier ops, then out through enclosing ops.
// Gives up at an op with regions (such as `scf.if`) or a loop that might
// change the object, and at a region with several blocks. At the function's
// start, a parameter's object has what the function assumes of it; any other
// has nothing. At the call that returned the handle, the object has what
// `produced` says that call gives its result.
SmallVector<StringAttr> objectFactsBefore(
    Operation *point, const HandleVariable &variable,
    function_ref<SmallVector<StringAttr>(OpResult)> produced = nullptr) {
  // If a call on the way changed the object, the properties it kept.
  std::optional<SmallVector<StringAttr>> kept;
  auto keep = [&](SmallVector<StringAttr> &facts) {
    if (kept)
      llvm::erase_if(
          facts, [&](StringAttr p) { return !llvm::is_contained(*kept, p); });
  };
  while (true) {
    for (Operation *op = point->getPrevNode(); op; op = op->getPrevNode()) {
      if (variable.producer && op == variable.producer.getOwner()) {
        SmallVector<StringAttr> facts;
        if (produced)
          facts = produced(variable.producer);
        keep(facts);
        return facts;
      }
      if (op->getNumRegions() != 0) {
        if (mayChangeHandle(op, variable))
          return {};
        continue;
      }
      SmallVector<StringAttr> established;
      switch (effectOnHandle(op, variable, established)) {
      case HandleEffect::None:
        continue;
      case HandleEffect::Establishes:
        keep(established);
        return established;
      case HandleEffect::Changes: {
        SmallVector<StringAttr> keptHere = propertiesKeptBy(op, variable);
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
    if (!region->hasOneBlock() || !parent)
      return {};
    if (isa<FunctionOpInterface>(parent)) {
      if (!variable.parameter ||
          variable.parameter.getOwner() != &region->front())
        return {};
      SmallVector<StringAttr> assumed;
      appendPropertyNames(assumed, assumptionsOnParameter(variable.parameter));
      keep(assumed);
      return assumed;
    }
    bool repeats;
    if (isa<GuardOp>(parent))
      repeats = false;
    else if (auto branch = dyn_cast<RegionBranchOpInterface>(parent))
      repeats = branch.isRepetitiveRegion(region->getRegionNumber());
    else if (isa<LoopLikeOpInterface>(parent))
      repeats = true;
    else
      return {};
    if (repeats && mayChangeHandle(parent, variable))
      return {};
    point = parent;
  }
}

// The properties established for the object that argument `index` of `use`
// names, and not changed since, leaving out the `replaced` ops.
SmallVector<StringAttr>
objectFactsAt(CallOp use, unsigned index,
              function_ref<SmallVector<StringAttr>(OpResult)> produced,
              ArrayRef<Operation *> replaced) {
  Value handle = use.getArgOperands()[index];
  if (!isa<LLVM::LLVMPointerType>(handle.getType()))
    return {};
  // A fact recorded on an argument is read as true at every use of that
  // value, but an object fact holds only here. So it's recorded only on a
  // value only this call uses, or on a copy recordFactsOnCall makes.
  std::optional<HandleVariable> variable = findHandleVariable(handle, use);
  if (!variable)
    return {};
  variable->replaced = replaced;
  return objectFactsBefore(use, *variable, produced);
}

// Whether `value` is a pointer a call returned, whose facts are about its
// object (see findHandleVariable).
bool isReturnedPointer(Value value) {
  auto result = dyn_cast<OpResult>(stripCasts(value));
  return result && isa<CallOp, LLVM::CallOp>(result.getOwner()) &&
         isa<LLVM::LLVMPointerType>(result.getType());
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

  // What a pointer parameter is assumed to point to holds on entry, until the
  // function changes it; objectFactsAt follows it from there.
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (!isa<LLVM::LLVMPointerType>(arg.getType()))
      appendPropertyNames(facts, assumptionsOnParameter(arg));
    return facts;
  }

  Operation *op = value.getDefiningOp();
  if (op->getNumResults() == 1)
    appendPropertyNames(facts, op->getAttr(kPropertyAttr));

  if (depth >= kMaxDerivationDepth)
    return facts;

  // A value loaded from a parameter gets the parameter's assumptions, if
  // nothing could have changed what it points to since the function started.
  // A loaded pointer gets nothing: its object could change after the load.
  if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
    Value address = stripCasts(load.getAddr());
    if (auto arg = dyn_cast<BlockArgument>(address))
      if (!isa<LLVM::LLVMPointerType>(value.getType()))
        if (std::optional<HandleVariable> variable = findParameterVariable(arg))
          for (StringAttr name : objectFactsBefore(load, *variable))
            if (!llvm::is_contained(facts, name))
              facts.push_back(name);
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

  // What a callee says of a pointer it returns is about the object, which
  // may change after the call; objectFactsAt follows it from there.
  if (isReturnedPointer(result))
    return facts;
  for (StringAttr name : callResultFacts(result, depth))
    if (!llvm::is_contained(facts, name))
      facts.push_back(name);
  return facts;
}

SmallVector<StringAttr> PropertyAnalysis::callResultFacts(OpResult result,
                                                          unsigned depth) {
  // Guaranteed by the callee, whatever its inputs were.
  SmallVector<StringAttr> facts;
  appendPropertyNames(facts, guaranteesOf(result));
  if (depth >= kMaxDerivationDepth)
    return facts;

  // Preserved by the callee: holds of the output if it held of every input
  // the declaration lists. An input the call was given as a returned pointer
  // has its object's facts at that call.
  Operation *call = result.getOwner();
  for (const auto &[property, inputs] : preservedFrom(result)) {
    if (llvm::is_contained(facts, property))
      continue;
    bool everyInputHasIt = true;
    for (Value input : inputs) {
      SmallVector<StringAttr> known = cachedValueFacts(input, depth + 1);
      if (isReturnedPointer(input))
        if (std::optional<HandleVariable> variable =
                findHandleVariable(input, call))
          llvm::append_range(
              known, objectFactsBefore(call, *variable, [&](OpResult r) {
                return callResultFacts(r, depth + 1);
              }));
      StringRef wanted = property.getValue();
      bool hasIt = llvm::any_of(known, [&](StringAttr name) {
        return impliesProperty(name.getValue(), wanted);
      });
      everyInputHasIt = everyInputHasIt && hasIt;
    }
    if (everyInputHasIt)
      facts.push_back(property);
  }
  return facts;
}

void PropertyAnalysis::recordFactsOnCall(CallOp call,
                                         ArrayRef<Operation *> replaced) {
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
    auto produced = [&](OpResult result) {
      return callResultFacts(result, /*depth=*/1);
    };
    SmallVector<StringAttr> objectFacts =
        objectFactsAt(call, i, produced, replaced);
    for (StringAttr name : objectFacts)
      if (!llvm::is_contained(facts, name))
        facts.push_back(name);
    // Give the call its own copy, so the object fact isn't read as true at
    // the value's other uses. A returned pointer's facts that hold at every
    // call given it need none, which keeps it one value for a rule that names
    // it twice, `exa.assemble(exa.btcb(B, C, K, n), C, G, n)`. The calls a
    // rewrite replaces aren't other uses: they are about to go.
    Value arg = call.getArgOperands()[i];
    auto isOther = [&](Operation *user) {
      return user != call && !llvm::is_contained(replaced, user);
    };
    auto holdsAt = [&](OpOperand &use) {
      auto other = dyn_cast<CallOp>(use.getOwner());
      if (!other || !isOther(other))
        return true;
      SmallVector<StringAttr> there =
          objectFactsAt(other, use.getOperandNumber(), produced, replaced);
      return llvm::all_of(objectFacts, [&](StringAttr name) {
        return llvm::is_contained(there, name);
      });
    };
    bool holdsEverywhere =
        isReturnedPointer(arg) && llvm::all_of(arg.getUses(), holdsAt);
    if (!objectFacts.empty() && !holdsEverywhere &&
        llvm::any_of(arg.getUsers(), isOther)) {
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
