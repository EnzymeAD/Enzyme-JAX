//===----------------------------------------------------------------------===//
//
// Deriving the properties of values from declarations, and the pass that
// records them on call arguments. See Properties.h for the model.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Tessera/Properties.h"
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
#include <functional>
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

/// Properties that bring others with them. `positive_definite` means what its
/// runtime check tests, which includes symmetry, so it and `SPD` are the same
/// property under two names.
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

/// How far back through calls a fact is followed before giving up.
constexpr unsigned kMaxDepth = 64;

/// Step through operations that pass a value along unchanged.
Value lookThroughForwarding(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<LLVM::FreezeOp, LLVM::BitcastOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

DictionaryAttr attrsAt(ArrayAttr list, unsigned index) {
  if (!list || index >= list.size())
    return nullptr;
  return dyn_cast<DictionaryAttr>(list[index]);
}

void addProperties(SmallVectorImpl<StringAttr> &set, Attribute list) {
  auto array = dyn_cast_or_null<ArrayAttr>(list);
  if (!array)
    return;
  for (Attribute entry : array)
    if (auto name = dyn_cast<StringAttr>(entry))
      if (!llvm::is_contained(set, name))
        set.push_back(name);
}

void addProperties(SmallVectorImpl<StringAttr> &set, DictionaryAttr dict) {
  if (dict)
    addProperties(set, dict.get(kPropertyAttr));
}

bool anyImplies(ArrayRef<StringAttr> have, llvm::StringRef want) {
  return llvm::any_of(have, [&](StringAttr name) {
    return impliesProperty(name.getValue(), want);
  });
}

/// What the callee of a call result declares about the output that result
/// carries, and how to reach the values the call passed.
struct Producer {
  /// The attributes of the output: the result, or the parameter it comes
  /// back through.
  DictionaryAttr output;
  /// Whether the output is the one a preserves declaration applies to.
  bool isPreservedOutput = false;
  ArrayAttr preserves;
  /// The value the call passed as parameter k, or null.
  std::function<Value(unsigned)> input;
};

std::optional<Producer> viewProducer(OpResult result) {
  Operation *op = result.getOwner();
  unsigned resultNumber = result.getResultNumber();

  if (auto call = dyn_cast<CallOp>(op)) {
    auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, call.getCalleeAttr().getAttr());
    if (!define)
      return std::nullopt;

    // argModes, and so the written-argument positions, leave out the sret.
    unsigned offset = define.getSretAttr() ? 1 : 0;
    bool returnsValue = offset || define.getFunctionType().getNumResults() != 0;

    Producer producer;
    producer.preserves = define->getAttrOfType<ArrayAttr>(kPreservesAttr);

    // Written arguments come back as the leading results, so a result's
    // number is not the index of the function result it holds.
    if (auto arg = define.getWrittenArgForCallResult(resultNumber)) {
      // A parameter the callee also reads describes it on entry and on exit
      // at once, so it cannot be read as either.
      if (!define.argIsRead(*arg))
        producer.output = attrsAt(define.getArgAttrsAttr(), *arg + offset);
      producer.isPreservedOutput =
          !returnsValue && define.getNumWrittenArgs() == 1;
    } else {
      producer.output =
          offset ? attrsAt(define.getArgAttrsAttr(), 0)
                 : attrsAt(define.getResAttrsAttr(),
                           resultNumber - define.getNumWrittenArgs());
      producer.isPreservedOutput = true;
    }

    producer.input = [call, define, offset](unsigned param) mutable -> Value {
      if (param < offset)
        return Value();
      std::optional<unsigned> operand =
          define.getCallOperandForArg(param - offset);
      if (!operand || *operand >= call.getArgOperands().size())
        return Value();
      return call.getArgOperands()[*operand];
    };
    return producer;
  }

  // A function that is not a tessera op can still declare what it returns.
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return std::nullopt;
    auto func =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op, calleeAttr);
    if (!func)
      return std::nullopt;

    Producer producer;
    producer.output = attrsAt(func.getResAttrsAttr(), resultNumber);
    producer.isPreservedOutput = true;
    producer.preserves = func->getAttrOfType<ArrayAttr>(kPreservesAttr);
    producer.input = [call](unsigned param) mutable -> Value {
      if (param >= call.getArgOperands().size())
        return Value();
      return call.getArgOperands()[param];
    };
    return producer;
  }
  return std::nullopt;
}

/// What a definition assumes of one of its own parameters on entry, or null
/// if `arg` is not such a parameter.
DictionaryAttr assumedOf(BlockArgument arg) {
  Block *block = arg.getOwner();
  if (!block->isEntryBlock())
    return nullptr;
  Operation *parent = block->getParentOp();
  unsigned param = arg.getArgNumber();

  if (auto define = dyn_cast<DefineOp>(parent)) {
    unsigned offset = define.getSretAttr() ? 1 : 0;
    if (param < offset || param - offset >= define.getArgModeEntries().size())
      return nullptr;
    // Only an input: a written parameter's properties hold on exit.
    if (define.argIsWritten(param - offset))
      return nullptr;
    return attrsAt(define.getArgAttrsAttr(), param);
  }
  if (auto func = dyn_cast<LLVM::LLVMFuncOp>(parent)) {
    if (func.getArgAttr(param, LLVM::LLVMDialect::getStructRetAttrName()))
      return nullptr;
    return attrsAt(func.getArgAttrsAttr(), param);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Facts about handles
//===----------------------------------------------------------------------===//

/// Operations that hand a pointer on as it is, or view the memory it points
/// to as a memref.
bool isPointerView(Operation *op) {
  return isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::FreezeOp,
             enzymexla::Pointer2MemrefOp>(op);
}

/// A handle kept in a stack slot. C code keeps `Mat A` in one as soon as
/// `MatCreate(comm, &A)` takes its address, and loads it afresh for each use,
/// so two calls given `A` are given different SSA values. They are the same
/// handle as long as nothing puts another one in the slot in between, and the
/// operations that could are all listed here.
struct HandleSlot {
  /// The slot's address and every view of it.
  DenseSet<Value> views;
  /// The loads of the handle from the slot.
  DenseSet<Value> loads;
  /// Operations that may leave another handle in the slot, or end its life:
  /// stores into it, lifetime markers, and calls given its address.
  DenseSet<Operation *> writers;
  /// Calls to a fact marker given the slot's address (see Properties.h).
  /// They establish what the marker's parameter carries of whichever handle
  /// the slot holds at that point, and change nothing.
  DenseSet<Operation *> factSites;

  bool isHandle(Value value) const {
    return loads.contains(lookThroughForwarding(value));
  }
};

/// Whether a loaded handle goes nowhere but into calls and guards (that read
/// it) and pointer comparisons. A handle copied anywhere else -- stored into
/// another variable, say -- could reach a call this does not see as taking
/// it.
bool onlyPassedToCalls(Value handle) {
  return llvm::all_of(handle.getUses(), [](OpOperand &use) {
    Operation *user = use.getOwner();
    if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::FreezeOp>(user))
      return onlyPassedToCalls(user->getResult(0));
    // Given to a call, rather than called: an indirect call's callee is its
    // first operand.
    if (auto call = dyn_cast<LLVM::CallOp>(user))
      return call.getCalleeAttr() || use.getOperandNumber() != 0;
    return isa<CallOp, GuardOp, LLVM::ICmpOp>(user);
  });
}

/// Whether `op` calls a marker the plugin generated for a fact stated on a
/// statement. The marker's body is empty, so given a slot's address it leaves
/// the slot as it was. Only a marker: any other call given the address may
/// put another handle there, as MatDestroy(&A) does.
bool isFactMarkerCall(Operation *op) {
  auto call = dyn_cast<LLVM::CallOp>(op);
  if (!call || !call.getCalleeAttr())
    return false;
  auto func = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      op, call.getCalleeAttr());
  return func && func->hasAttr(kFactMarkerAttr);
}

/// The slot `handle` was loaded from, or nothing if it was not loaded from a
/// stack slot, or the slot or a handle loaded from it is used in some way this
/// cannot follow.
std::optional<HandleSlot> slotOf(Value handle) {
  Operation *load = lookThroughForwarding(handle).getDefiningOp();
  Value address;
  if (auto llvmLoad = dyn_cast_or_null<LLVM::LoadOp>(load))
    address = llvmLoad.getAddr();
  else if (auto affineLoad = dyn_cast_or_null<affine::AffineLoadOp>(load))
    address = affineLoad.getMemref();
  else if (auto memrefLoad = dyn_cast_or_null<memref::LoadOp>(load))
    address = memrefLoad.getMemref();
  else
    return std::nullopt;
  while (Operation *def = address.getDefiningOp()) {
    if (!isPointerView(def))
      break;
    address = def->getOperand(0);
  }

  // One pointer, so that every load reads the same one.
  auto alloca = address.getDefiningOp<LLVM::AllocaOp>();
  if (!alloca || !isa<LLVM::LLVMPointerType>(alloca.getElemType()) ||
      !matchPattern(alloca.getArraySize(), m_One()))
    return std::nullopt;

  HandleSlot slot;
  SmallVector<Value> worklist{alloca.getResult()};
  while (!worklist.empty()) {
    Value view = worklist.pop_back_val();
    if (!slot.views.insert(view).second)
      continue;
    for (OpOperand &use : view.getUses()) {
      Operation *user = use.getOwner();
      if (isPointerView(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      if (isa<LLVM::LoadOp, affine::AffineLoadOp, memref::LoadOp>(user)) {
        slot.loads.insert(user->getResult(0));
        continue;
      }
      if (isFactMarkerCall(user)) {
        slot.factSites.insert(user);
        continue;
      }
      // The slot as the address stored to, not as the value stored.
      bool isStoreInto =
          (isa<LLVM::StoreOp>(user) &&
           cast<LLVM::StoreOp>(user).getAddr() == view) ||
          (isa<affine::AffineStoreOp>(user) &&
           cast<affine::AffineStoreOp>(user).getMemref() == view) ||
          (isa<memref::StoreOp>(user) &&
           cast<memref::StoreOp>(user).getMemref() == view);
      if (isStoreInto ||
          isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp, CallOp, LLVM::CallOp>(
              user)) {
        slot.writers.insert(user);
        continue;
      }
      // Its address escapes, or is offset into: anything could write it.
      return std::nullopt;
    }
  }

  if (!llvm::all_of(slot.loads, onlyPassedToCalls))
    return std::nullopt;
  return slot;
}

/// The attributes of the parameter a call passes argument `index` to, or null.
DictionaryAttr paramAttrs(Operation *op, unsigned index) {
  if (auto call = dyn_cast<CallOp>(op)) {
    auto define = SymbolTable::lookupNearestSymbolFrom<DefineOp>(
        op, call.getCalleeAttr().getAttr());
    if (!define)
      return nullptr;
    std::optional<unsigned> param = define.getArgIndexForCallOperand(index);
    return param ? attrsAt(define.getArgAttrsAttr(), *param) : nullptr;
  }
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return nullptr;
    auto func =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op, calleeAttr);
    // Past the declared parameters, a variadic argument has none.
    if (!func || index >= func.getNumArguments())
      return nullptr;
    return attrsAt(func.getArgAttrsAttr(), index);
  }
  return nullptr;
}

/// What an operation does to the object a handle names.
enum class HandleEffect { None, Establishes, Changes };

/// What `op` itself, not counting anything in its regions, does to the
/// handle in `slot`. If it establishes properties, they are added to
/// `established`.
HandleEffect effectOn(Operation *op, const HandleSlot &slot,
                      SmallVectorImpl<StringAttr> &established) {
  if (slot.writers.contains(op))
    return HandleEffect::Changes;

  // A fact stated of the variable, of whatever handle it holds here. The
  // marker does nothing, so what it assumes of the object on entry
  // (tessera_assumes on the statement) holds on exit as much as what it
  // guarantees.
  if (slot.factSites.contains(op)) {
    for (auto [index, arg] :
         llvm::enumerate(cast<LLVM::CallOp>(op).getArgOperands()))
      if (slot.views.contains(arg))
        if (DictionaryAttr attrs = paramAttrs(op, index)) {
          addProperties(established, attrs.get(kEstablishesAttr));
          addProperties(established, attrs.get(kPropertyAttr));
        }
    return established.empty() ? HandleEffect::None : HandleEffect::Establishes;
  }

  ValueRange args;
  if (auto call = dyn_cast<CallOp>(op))
    args = call.getArgOperands();
  else if (auto call = dyn_cast<LLVM::CallOp>(op))
    args = call.getArgOperands();
  else
    // Only calls can be given a handle (see onlyPassedToCalls); a guard
    // only reads it, and a comparison only compares it.
    return HandleEffect::None;

  // A call that establishes a property does so on exit, whatever else it
  // does to the object on the way.
  bool establishes = false, changes = false;
  for (auto [index, arg] : llvm::enumerate(args)) {
    if (!slot.isHandle(arg))
      continue;
    DictionaryAttr attrs = paramAttrs(op, index);
    if (Attribute properties = attrs ? attrs.get(kEstablishesAttr) : nullptr) {
      addProperties(established, properties);
      establishes = true;
    } else if (!attrs || !attrs.get(kReadonlyAttr)) {
      changes = true;
    }
  }
  return establishes ? HandleEffect::Establishes
         : changes   ? HandleEffect::Changes
                     : HandleEffect::None;
}

/// Whether anything in `op`, its regions included, might change the object
/// in `slot`. Here an establishing call counts as a change: it may or may not
/// run, or run before something else that changes the object.
bool mayChange(Operation *op, const HandleSlot &slot) {
  return op
      ->walk([&](Operation *nested) {
        SmallVector<StringAttr> ignored;
        return effectOn(nested, slot, ignored) == HandleEffect::None
                   ? WalkResult::advance()
                   : WalkResult::interrupt();
      })
      .wasInterrupted();
}

/// The properties established of the object the handle passed as argument
/// `index` of `use` names, and not changed since.
///
/// The walk goes back from `use` through the operations that run before it:
/// earlier operations in its block, then out to the operation holding the
/// region and on before that. An earlier operation with regions of its own,
/// such as an `scf.if`, might have run any part of them, so anything in them
/// that might change the object ends the walk. So does leaving a loop's body
/// if anything in the loop might change the object, since that could run on
/// an earlier iteration. The walk gives up at a region of more than one
/// block, and at the function's entry.
SmallVector<StringAttr> handleFactsAt(CallOp use, unsigned index) {
  Value handle = use.getArgOperands()[index];
  if (!isa<LLVM::LLVMPointerType>(handle.getType()))
    return {};
  // What is recorded on an argument is read as true of the value passed
  // there, wherever else it goes (PropertyFacts::derive, hasRecordedProperty).
  // A fact of the object holds only here, so it is recorded only on a handle
  // loaded for this call alone.
  if (lookThroughForwarding(handle) != handle ||
      !llvm::all_of(handle.getUsers(),
                    [&](Operation *user) { return user == use; }))
    return {};
  std::optional<HandleSlot> slot = slotOf(handle);
  if (!slot)
    return {};

  Operation *point = use;
  while (true) {
    for (Operation *op = point->getPrevNode(); op; op = op->getPrevNode()) {
      if (op->getNumRegions() != 0) {
        if (mayChange(op, *slot))
          return {};
        continue;
      }
      SmallVector<StringAttr> established;
      switch (effectOn(op, *slot, established)) {
      case HandleEffect::None:
        continue;
      case HandleEffect::Establishes:
        return established;
      case HandleEffect::Changes:
        return {};
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
    if (repeats && mayChange(parent, *slot))
      return {};
    point = parent;
  }
}

} // namespace

bool mlir::enzyme::tessera::impliesProperty(llvm::StringRef have,
                                            llvm::StringRef want) {
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

SmallVector<StringAttr> PropertyFacts::of(Value value) {
  return get(value, /*depth=*/0);
}

SmallVector<StringAttr> PropertyFacts::get(Value value, unsigned depth) {
  value = lookThroughForwarding(value);
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  // Settled as nothing while the answer is worked out, so a cycle -- possible
  // only in a graph region -- ends rather than recursing.
  cache[value] = {};
  SmallVector<StringAttr> facts = derive(value, depth);
  cache[value] = facts;
  return facts;
}

SmallVector<StringAttr> PropertyFacts::derive(Value value, unsigned depth) {
  SmallVector<StringAttr> facts;

  // Recorded already, wherever the value is passed: a fact of an SSA value
  // holds at every use of it.
  for (OpOperand &use : value.getUses())
    if (auto call = dyn_cast<CallOp>(use.getOwner()))
      addProperties(facts,
                    attrsAt(call.getArgAttrsAttr(), use.getOperandNumber()));

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    addProperties(facts, assumedOf(arg));
    return facts;
  }

  Operation *op = value.getDefiningOp();
  if (op->getNumResults() == 1)
    addProperties(facts, op->getAttr(kPropertyAttr));

  if (depth >= kMaxDepth)
    return facts;

  // A matrix a parameter refers to, loaded from it. What the definition
  // assumes of the parameter is assumed of the matrix it points to, so it
  // holds of the load as long as nothing in the function could have changed
  // that matrix first -- which here means the parameter is used for nothing
  // but loading.
  if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
    Value address = lookThroughForwarding(load.getAddr());
    if (auto arg = dyn_cast<BlockArgument>(address))
      if (llvm::all_of(arg.getUsers(),
                       [](Operation *user) { return isa<LLVM::LoadOp>(user); }))
        addProperties(facts, assumedOf(arg));
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
      return get(terminator->getOperand(index), depth + 1);
    };
    SmallVector<StringAttr> thenFacts = yielded(guard.getThenRegion());
    SmallVector<StringAttr> elseFacts = yielded(guard.getElseRegion());
    for (StringAttr name : thenFacts)
      if (llvm::is_contained(elseFacts, name) &&
          !llvm::is_contained(facts, name))
        facts.push_back(name);
    return facts;
  }

  std::optional<Producer> producer = viewProducer(cast<OpResult>(value));
  if (!producer)
    return facts;

  // Guaranteed by the callee, whatever its inputs were.
  addProperties(facts, producer->output);

  // Preserved by the callee: holds of the output if it held of every input
  // the declaration lists.
  if (!producer->isPreservedOutput || !producer->preserves)
    return facts;
  for (Attribute entry : producer->preserves) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      continue;
    auto kept = dict.getAs<StringAttr>("property");
    auto inputs = dict.getAs<ArrayAttr>("inputs");
    if (!kept || !inputs || inputs.empty() || llvm::is_contained(facts, kept))
      continue;
    bool allInputs = llvm::all_of(inputs, [&](Attribute index) {
      auto position = dyn_cast<IntegerAttr>(index);
      if (!position || position.getInt() < 0)
        return false;
      Value in = producer->input(position.getInt());
      return in && anyImplies(get(in, depth + 1), kept.getValue());
    });
    if (allInputs)
      facts.push_back(kept);
  }
  return facts;
}

void PropertyFacts::annotate(CallOp call) {
  MLIRContext *ctx = call.getContext();
  unsigned numArgs = call.getArgOperands().size();
  ArrayAttr existing = call.getArgAttrsAttr();

  SmallVector<Attribute> dicts;
  bool any = false;
  for (unsigned i = 0; i != numArgs; ++i) {
    DictionaryAttr dict = attrsAt(existing, i);
    SmallVector<StringAttr> facts;
    addProperties(facts, dict);
    for (StringAttr name : of(call.getArgOperands()[i]))
      if (!llvm::is_contained(facts, name))
        facts.push_back(name);
    for (StringAttr name : handleFactsAt(call, i))
      if (!llvm::is_contained(facts, name))
        facts.push_back(name);

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
        attrsAt(call.getArgAttrsAttr(), use.getOperandNumber());
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
    PropertyFacts facts;
    getOperation()->walk([&](CallOp call) { facts.annotate(call); });
  }
};

} // namespace
