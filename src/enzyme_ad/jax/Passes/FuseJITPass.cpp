#include "Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

#define DEBUG_TYPE "fuse-jit"

namespace mlir {
namespace enzyme {

#define GEN_PASS_DEF_FUSEJITPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

// replacing several JIT calls with one: which wrappers to combine,
// how their input and output buffers map to the fused call, and which constants
// must move before the calls.
struct JITFusionInfo {
  SmallVector<enzymexla::JITCallOp> fusionCalls;
  SmallVector<LLVM::LLVMFuncOp> fusionFuncs;
  SmallVector<Value> fusedArgs;
  SmallVector<Attribute> operandLayouts;
  SmallVector<SmallVector<unsigned>> callOperandSlots;
  SmallVector<Value> fusedReturns;
  DenseMap<Value, unsigned> resultSlotMap;
  SetVector<Operation *> opsToHoist;
};

constexpr llvm::StringLiteral FusedNamePrefix = "fused__";

LogicalResult validateSingleBlockWrapper(LLVM::LLVMFuncOp func) {
  // Body cloning below copies one block and supplies its own terminator.
  return success(llvm::hasSingleElement(func.getBody()));
}

std::string buildFusedName(ArrayRef<enzymexla::JITCallOp> fusionCalls) {
  // Ordered callee names distinguish different wrapper sequences.
  std::string fusedName = FusedNamePrefix.str();
  llvm::raw_string_ostream os(fusedName);

  for (auto [idx, call] : llvm::enumerate(fusionCalls)) {
    if (idx != 0)
      os << "_";
    enzymexla::JITCallOp jitCall = call;
    os << jitCall.getFn().getRootReference().getValue();
  }

  return fusedName;
}

FailureOr<LLVM::LLVMFuncOp> lookupFusionFunction(ModuleOp module,
                                                 enzymexla::JITCallOp call) {
  // Backend config is opaque, and argument/result attrs need slot remapping.
  if (!call.getBackendConfig().empty() || call.getArgAttrsAttr() ||
      call.getResAttrsAttr())
    return failure();

  // Fusion currently supports single-block, pointer-only, void wrappers.
  StringRef fnName = call.getFn().getRootReference().getValue();
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(fnName);
  if (!func || func.empty())
    return failure();
  if (failed(validateSingleBlockWrapper(func)))
    return failure();
  if (func.getNumArguments() < call.getNumOperands())
    return failure();
  auto funcType = func.getFunctionType();
  if (funcType.isVarArg() ||
      !isa<LLVM::LLVMVoidType>(funcType.getReturnType()) ||
      llvm::any_of(funcType.getParams(),
                   [](Type type) { return !isa<LLVM::LLVMPointerType>(type); }))
    return failure();
  return func;
}

std::string getAvailableFusedName(ModuleOp module, StringRef baseName) {
  // TODO: Reuse duplicates only with a key for wrapper order and fused slot
  // mappings; the callee names alone do not determine the fused ABI.
  std::string fusedName = baseName.str();
  unsigned suffix = 0;
  while (module.lookupSymbol<LLVM::LLVMFuncOp>(fusedName)) {
    fusedName = (baseName + "_" + Twine(++suffix)).str();
  }
  return fusedName;
}

bool aliasMatchesResult(stablehlo::OutputOperandAliasAttr alias,
                        unsigned resultIdx, unsigned numResults) {
  auto outputTupleIndices = alias.getOutputTupleIndices();
  if (numResults == 1)
    return outputTupleIndices.empty();

  return outputTupleIndices.size() == 1 &&
         outputTupleIndices.front() == static_cast<int64_t>(resultIdx);
}

FailureOr<int64_t> findAliasedOperandIndex(enzymexla::JITCallOp call,
                                           unsigned resultIdx) {
  for (Attribute attr : call.getOutputOperandAliases()) {
    auto alias = dyn_cast<stablehlo::OutputOperandAliasAttr>(attr);
    if (!alias)
      return failure();
    if (!aliasMatchesResult(alias, resultIdx, call.getNumResults()))
      continue;
    if (!alias.getOperandTupleIndices().empty())
      return failure();

    int64_t operandIndex = alias.getOperandIndex();
    if (operandIndex < 0 ||
        operandIndex >= static_cast<int64_t>(call.getNumOperands()))
      return failure();
    return operandIndex;
  }

  // Missing aliases are supported by the parser but not by fusion validation.
  return -1;
}

// XLA orders JIT calls without side effects by their operands alone, whatever
// effects their callee declares.
bool isReorderable(Operation *op) {
  if (auto call = dyn_cast<enzymexla::JITCallOp>(op))
    if (call.getXlaSideEffectFreeAttr())
      return true;
  return isMemoryEffectFree(op);
}

// Only this call's additions are staged; a rejected extension changes no state.
struct JITFusionExtension {
  SmallVector<Value> newArgs;
  SmallVector<Attribute> operandLayouts;
  DenseMap<Value, unsigned> readOnlyArgs;
  SmallVector<unsigned> operandSlots;
  SmallVector<unsigned> resultSlots;
  SetVector<Operation *> opsToHoist;
};

// Stages in `toHoist` the ops computing `value` that must move above `firstOp`
// for the fused call to read it. Only effect-free ops of the same block without
// regions move, and never a JIT call: a value computed from a result of the
// group cannot be an input of the group.
LogicalResult stageHoistedProducers(Value value, Operation *firstOp,
                                    DominanceInfo &dominance,
                                    const SetVector<Operation *> &hoisted,
                                    SetVector<Operation *> &toHoist) {
  SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (dominance.properlyDominates(current, firstOp))
      continue;
    Operation *defOp = current.getDefiningOp();
    if (!defOp || defOp->getBlock() != firstOp->getBlock() ||
        isa<enzymexla::JITCallOp>(defOp) || defOp->getNumRegions() != 0 ||
        !isMemoryEffectFree(defOp))
      return failure();
    if (hoisted.contains(defOp) || !toHoist.insert(defOp))
      continue;
    llvm::append_range(worklist, defOp->getOperands());
  }
  return success();
}

LogicalResult validateFusionExtension(
    enzymexla::JITCallOp call, LLVM::LLVMFuncOp wrapperFunc, Operation *firstOp,
    DominanceInfo &dominance, const JITFusionInfo &info,
    const DenseMap<Value, unsigned> &readOnlyArgs,
    const SetVector<Operation *> &hoisted, JITFusionExtension &extension) {
  auto layouts = dyn_cast_or_null<ArrayAttr>(call.getOperandLayoutsAttr());
  if (call.getOperandLayoutsAttr() &&
      (!layouts || layouts.size() != call.getNumOperands()))
    return failure();
  // Keep unspecified layouts distinct from explicitly laid-out calls.
  if (!info.fusionCalls.empty()) {
    enzymexla::JITCallOp firstCall = info.fusionCalls.front();
    if (static_cast<bool>(layouts) !=
        static_cast<bool>(firstCall.getOperandLayoutsAttr()))
      return failure();
  }

  SmallVector<unsigned> aliasedOperands;
  for (unsigned resultIdx = 0; resultIdx < call.getNumResults(); ++resultIdx) {
    FailureOr<int64_t> aliasedOperand =
        findAliasedOperandIndex(call, resultIdx);
    // This ABI maps each result to an aliased input slot. Separate result
    // storage would need an explicit output slot in the fused ABI.
    if (failed(aliasedOperand) || aliasedOperand.value() < 0)
      return failure();
    aliasedOperands.push_back(aliasedOperand.value());
  }

  for (auto [operandIdx, operand] : llvm::enumerate(call.getOperands())) {
    // Internal results reuse their producer's storage slot.
    auto source = info.resultSlotMap.find(operand);
    if (source != info.resultSlotMap.end()) {
      // Writing the slot clobbers the result it holds, so only the calls of
      // the group before this one may read that result.
      if (llvm::is_contained(aliasedOperands, operandIdx) &&
          llvm::any_of(operand.getUsers(), [&](Operation *user) {
            return user != call.getOperation() &&
                   !(llvm::any_of(info.fusionCalls,
                                  [&](enzymexla::JITCallOp member) {
                                    return member.getOperation() == user;
                                  }) &&
                     user->isBeforeInBlock(call));
          }))
        return failure();
      extension.operandSlots.push_back(source->second);
      continue;
    }

    // Only share unaliased, load-only inputs. Writable or unknown uses keep
    // their (call, operand index) slot, even with the same initializer.
    bool readOnly =
        !llvm::is_contained(aliasedOperands, operandIdx) &&
        llvm::all_of(wrapperFunc.getArgument(operandIdx).getUsers(),
                     [](Operation *op) { return isa<LLVM::LoadOp>(op); });
    if (readOnly) {
      auto existing = readOnlyArgs.find(operand);
      if (existing != readOnlyArgs.end()) {
        extension.operandSlots.push_back(existing->second);
        continue;
      }
      auto staged = extension.readOnlyArgs.find(operand);
      if (staged != extension.readOnlyArgs.end()) {
        extension.operandSlots.push_back(staged->second);
        continue;
      }
    }

    // Inputs computed between the calls (e.g. MPI request initializers or
    // layout transposes) move above the group with what they are computed
    // from; an input that cannot move keeps this call out of the group.
    if (failed(stageHoistedProducers(operand, firstOp, dominance, hoisted,
                                     extension.opsToHoist)))
      return failure();
    unsigned slot = info.fusedArgs.size() + extension.newArgs.size();
    extension.operandSlots.push_back(slot);
    extension.newArgs.push_back(operand);
    if (layouts)
      extension.operandLayouts.push_back(layouts[operandIdx]);
    if (readOnly)
      extension.readOnlyArgs[operand] = slot;
  }

  // Each fused pointer slot has one layout. This also checks reused read-only
  // operands and producer/result slots reached by an internal dependency.
  if (layouts) {
    for (auto [operandIdx, slot] : llvm::enumerate(extension.operandSlots)) {
      Attribute slotLayout =
          slot < info.fusedArgs.size()
              ? info.operandLayouts[slot]
              : extension.operandLayouts[slot - info.fusedArgs.size()];
      if (layouts[operandIdx] != slotLayout)
        return failure();
    }
  }

  for (unsigned operandIdx : aliasedOperands)
    extension.resultSlots.push_back(extension.operandSlots[operandIdx]);

  // Inputs and any appended result pointers cover a prefix of wrapper args.
  unsigned mappedArgs = call.getNumOperands() + call.getNumResults();
  for (BlockArgument arg : wrapperFunc.getArguments()) {
    if (arg.getArgNumber() >= mappedArgs && !arg.use_empty())
      return failure();
  }
  return success();
}

// Scan forward in block order, allowing effect-free non-JIT ops between
// calls. Keep the longest supported prefix with effects or escaping results.
// JITFusionInfo collectGeneralizedFusionInfo(enzymexla::JITCallOp firstCall) {
//   JITFusionInfo info;
//   DominanceInfo dominance;
//   if (!dominance.hasSSADominance(firstCall->getBlock()))
//     return info;
//   auto module = firstCall->getParentOfType<ModuleOp>();
//   DenseMap<Value, unsigned> readOnlyArgs;
//   SmallPtrSet<Operation *, 8> movedConstants;
//   DenseMap<Value, size_t> outsideUses;
//   size_t escapingResults = 0;
//   bool hasEffects = false;
//   size_t legalCalls = 0, legalArgs = 0, legalConstants = 0;

//   // Boundary bookkeeping does bounded work per operand/result/use per
//   attempt.
//   // Alias searches, greedy revisits and repeated wrapper cloning can cost
//   more. for (Operation *op = firstCall; op; op = op->getNextNode()) {
//     if (op->hasTrait<OpTrait::IsTerminator>())
//       break;
//     auto call = dyn_cast<enzymexla::JITCallOp>(op);
//     if (!call) {
//       // Effects may only be crossed when their JIT bodies join the group.
//       if (!isMemoryEffectFree(op))
//         break;
//       continue;
//     }

//     FailureOr<LLVM::LLVMFuncOp> func = lookupFusionFunction(module, call);
//     if (failed(func))
//       break;
//     JITFusionExtension extension;
//     if (failed(validateFusionExtension(call, func.value(), firstCall,
//     dominance,
//                                        info, readOnlyArgs, movedConstants,
//                                        extension)))
//       break;

//     // Commit the extension only after its complete preflight succeeds.
//     info.fusionCalls.push_back(call);
//     info.fusionFuncs.push_back(func.value());
//     info.callOperandSlots.push_back(std::move(extension.operandSlots));
//     llvm::append_range(info.fusedArgs, extension.newArgs);
//     readOnlyArgs.insert(extension.readOnlyArgs.begin(),
//                         extension.readOnlyArgs.end());
//     for (Operation *constant : extension.constantsToMove)
//       movedConstants.insert(constant);
//     llvm::append_range(info.constantsToMove, extension.constantsToMove);
//     for (Value operand : call.getOperands()) {
//       auto useIt = outsideUses.find(operand);
//       if (useIt != outsideUses.end()) {
//         assert(useIt->second != 0 && "missing external operand use");
//         // Repeated operands consume separate uses of the same result.
//         if (--useIt->second == 0)
//           --escapingResults;
//       }
//     }
//     for (auto [idx, result] : llvm::enumerate(call.getResults())) {
//       size_t uses = std::distance(result.use_begin(), result.use_end());
//       outsideUses[result] = uses;
//       escapingResults += uses != 0;
//       info.resultSlotMap[result] = extension.resultSlots[idx];
//     }
//     hasEffects |= !isMemoryEffectFree(call);
//     if (info.fusionCalls.size() >= 2 && (hasEffects || escapingResults != 0))
//     {
//       legalCalls = info.fusionCalls.size();
//       legalArgs = info.fusedArgs.size();
//       legalConstants = info.constantsToMove.size();
//     }
//   }

//   if (legalCalls == 0)
//     return JITFusionInfo();

//   // A pure suffix may hide all escaping results. Rewind it once to the last
//   // observable group, without revalidating prefixes or copying growing maps.
//   for (size_t i = info.fusionCalls.size(); i > legalCalls; --i) {
//     auto call = info.fusionCalls[i - 1];
//     for (Value operand : call.getOperands()) {
//       auto useIt = outsideUses.find(operand);
//       if (useIt != outsideUses.end())
//         ++useIt->second;
//     }
//     for (Value result : call.getResults()) {
//       outsideUses.erase(result);
//       info.resultSlotMap.erase(result);
//     }
//   }
//   info.fusionCalls.resize(legalCalls);
//   info.fusionFuncs.resize(legalCalls);
//   info.callOperandSlots.resize(legalCalls);
//   info.fusedArgs.resize(legalArgs);
//   info.constantsToMove.resize(legalConstants);

//   // Keep external results first, in lexical result order.
//   DenseSet<unsigned> returnedSlots;
//   for (auto call : info.fusionCalls) {
//     for (Value result : call.getResults()) {
//       if (outsideUses.lookup(result) != 0) {
//         info.fusedReturns.push_back(result);
//         returnedSlots.insert(info.resultSlotMap.lookup(result));
//       }
//     }
//   }
//   // Preserve an alias for every written input, even if its result is
//   internal
//   // or unused. Otherwise constant request buffers can be treated as
//   read-only. for (auto call : info.fusionCalls) {
//     for (Value result : call.getResults()) {
//       if (returnedSlots.insert(info.resultSlotMap.lookup(result)).second)
//         info.fusedReturns.push_back(result);
//     }
//   }
//   return info;
// }

JITFusionInfo collectDependencyFusionInfo(enzymexla::JITCallOp seedCall) {
  // Collect the same-block component connected by direct JIT result/operand
  // edges. Unrelated calls are excluded even when they are adjacent.
  JITFusionInfo info;
  Block *block = seedCall->getBlock();
  SmallVector<enzymexla::JITCallOp> worklist;
  worklist.push_back(seedCall);
  SmallPtrSet<Operation *, 8> seenCalls;
  seenCalls.insert(seedCall.getOperation());

  // Follow only direct JIT result-to-operand edges within the seed block.
  while (!worklist.empty()) {
    enzymexla::JITCallOp call = worklist.pop_back_val();
    auto addCall = [&](enzymexla::JITCallOp related) {
      if (related && related->getBlock() == block &&
          seenCalls.insert(related.getOperation()).second)
        worklist.push_back(related);
    };

    for (Value operand : call.getOperands())
      addCall(operand.getDefiningOp<enzymexla::JITCallOp>());
    for (Value result : call.getResults())
      for (Operation *user : result.getUsers())
        addCall(dyn_cast<enzymexla::JITCallOp>(user));
  }

  if (seenCalls.size() < 2)
    return info;
  SmallVector<enzymexla::JITCallOp> candidates;
  for (Operation *op : seenCalls)
    candidates.push_back(cast<enzymexla::JITCallOp>(op));
  llvm::sort(candidates,
             [](enzymexla::JITCallOp lhs, enzymexla::JITCallOp rhs) {
               return lhs->isBeforeInBlock(rhs);
             });

  DominanceInfo dominance;
  if (!dominance.hasSSADominance(block))
    return JITFusionInfo();

  // Walk the block from the first candidate, adding each candidate whose
  // inputs can be made available above the group. Members move up to the
  // first one, so none may cross an op whose effects must stay ordered with
  // it; a candidate left out of the group is such an op unless XLA may reorder
  // it.
  ModuleOp module = seedCall->getParentOfType<ModuleOp>();
  Operation *firstOp = nullptr;
  Operation *lastOp = candidates.back().getOperation();
  DenseMap<Value, unsigned> readOnlyArgs;
  for (Operation *op = candidates.front(); op != lastOp->getNextNode();
       op = op->getNextNode()) {
    auto call = dyn_cast<enzymexla::JITCallOp>(op);
    if (call && seenCalls.contains(op)) {
      FailureOr<LLVM::LLVMFuncOp> func = lookupFusionFunction(module, call);
      JITFusionExtension extension;
      if (succeeded(func) &&
          succeeded(validateFusionExtension(
              call, func.value(), firstOp ? firstOp : op, dominance, info,
              readOnlyArgs, info.opsToHoist, extension))) {
        if (!firstOp)
          firstOp = op;
        info.fusionCalls.push_back(call);
        info.fusionFuncs.push_back(func.value());
        info.callOperandSlots.push_back(std::move(extension.operandSlots));
        llvm::append_range(info.fusedArgs, extension.newArgs);
        llvm::append_range(info.operandLayouts, extension.operandLayouts);
        readOnlyArgs.insert(extension.readOnlyArgs.begin(),
                            extension.readOnlyArgs.end());
        info.opsToHoist.insert(extension.opsToHoist.begin(),
                               extension.opsToHoist.end());
        for (auto [resultIdx, result] : llvm::enumerate(call.getResults()))
          info.resultSlotMap[result] = extension.resultSlots[resultIdx];
        continue;
      }
    }
    // Ops before the group stay where they are.
    if (firstOp && !isReorderable(op))
      break;
  }

  if (info.fusionCalls.size() < 2)
    return JITFusionInfo();

  SmallPtrSet<Operation *, 8> memberOps;
  for (enzymexla::JITCallOp call : info.fusionCalls)
    memberOps.insert(call.getOperation());

  bool hasEffects = false;
  bool hasEscapingUse = false;
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    hasEffects |= !isMemoryEffectFree(call);
    for (Value result : call.getResults())
      hasEscapingUse |= llvm::any_of(result.getUsers(), [&](Operation *user) {
        return !memberOps.contains(user);
      });
  }

  if (!hasEffects && !hasEscapingUse)
    return JITFusionInfo();

  // Return escaping values first, then retain each written storage slot once.
  DenseSet<unsigned> returnedSlots;
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    for (Value result : call.getResults()) {
      if (llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !memberOps.contains(user);
          })) {
        info.fusedReturns.push_back(result);
        returnedSlots.insert(info.resultSlotMap.lookup(result));
      }
    }
  }
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    for (Value result : call.getResults()) {
      if (returnedSlots.insert(info.resultSlotMap.lookup(result)).second)
        info.fusedReturns.push_back(result);
    }
  }
  return info;
}

// Binds a wrapper argument to a fused pointer, casting the pointer when the
// wrapper expects another address space.
void mapPointer(OpBuilder &builder, BlockArgument wrapperArg, Value fusedArg,
                IRMapping &mapping) {
  if (fusedArg.getType() != wrapperArg.getType())
    fusedArg = builder.create<LLVM::AddrSpaceCastOp>(
        wrapperArg.getLoc(), wrapperArg.getType(), fusedArg);
  mapping.map(wrapperArg, fusedArg);
}

void mapCallInputs(OpBuilder &builder, LLVM::LLVMFuncOp wrapperFunc,
                   LLVM::LLVMFuncOp fusedFunc, ArrayRef<unsigned> operandSlots,
                   IRMapping &mapping) {
  // Bind source wrapper arguments to their physical fused pointers.
  for (auto [argIdx, slot] : llvm::enumerate(operandSlots))
    mapPointer(builder, wrapperFunc.getArgument(argIdx),
               fusedFunc.getArgument(slot), mapping);
}

void mapCallResults(OpBuilder &builder, enzymexla::JITCallOp call,
                    LLVM::LLVMFuncOp wrapperFunc, LLVM::LLVMFuncOp fusedFunc,
                    const DenseMap<Value, unsigned> &resultSlotMap,
                    IRMapping &mapping) {
  // JIT results name their aliased operand pointers, not LLVM return values.
  for (auto [resultIdx, result] : llvm::enumerate(call.getResults())) {
    auto slot = resultSlotMap.find(result);
    assert(slot != resultSlotMap.end() && "fusion mapping was not validated");
    Value mappedResult = fusedFunc.getArgument(slot->second);

    // Map an appended result pointer when the wrapper ABI contains one.
    unsigned wrapperArgIdx = call.getNumOperands() + resultIdx;
    if (wrapperArgIdx < wrapperFunc.getNumArguments())
      mapPointer(builder, wrapperFunc.getArgument(wrapperArgIdx), mappedResult,
                 mapping);
  }
}

// Purity is preserved only when every original call declares it.
UnitAttr
getFusedSideEffectFreeAttr(ArrayRef<enzymexla::JITCallOp> fusionCalls) {
  if (llvm::all_of(fusionCalls, [](enzymexla::JITCallOp call) {
        return static_cast<bool>(call.getXlaSideEffectFreeAttr());
      })) {
    enzymexla::JITCallOp firstCall = fusionCalls.front();
    return UnitAttr::get(firstCall.getContext());
  }
  return UnitAttr();
}

LogicalResult rewriteFusion(enzymexla::JITCallOp firstCall,
                            PatternRewriter &rewriter,
                            JITFusionInfo &fusionInfo) {
  SmallVector<enzymexla::JITCallOp> &fusionCalls = fusionInfo.fusionCalls;
  auto module = firstCall->getParentOfType<ModuleOp>();
  SmallVector<LLVM::LLVMFuncOp> &fusionFuncs = fusionInfo.fusionFuncs;
  SmallVector<Value> &fusedArgs = fusionInfo.fusedArgs;
  SmallVector<Value> &fusedReturns = fusionInfo.fusedReturns;
  const DenseMap<Value, unsigned> &resultSlotMap = fusionInfo.resultSlotMap;

  // Collection has completed all checks; no IR moves on a failed match.
  // Keeping the block order of the hoisted ops keeps their definitions first.
  SmallVector<Operation *> opsToHoist(fusionInfo.opsToHoist.begin(),
                                      fusionInfo.opsToHoist.end());
  llvm::sort(opsToHoist, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  for (Operation *op : opsToHoist)
    rewriter.moveOpBefore(op, firstCall);

  // The JIT wrapper ABI passes every tensor or scalar buffer as a pointer,
  // in the address space of the first wrapper argument bound to it.
  SmallVector<Type> llvmArgTypes(fusedArgs.size());
  for (auto [wrapperFunc, operandSlots] :
       llvm::zip_equal(fusionFuncs, fusionInfo.callOperandSlots))
    for (auto [argIdx, slot] : llvm::enumerate(operandSlots))
      if (!llvmArgTypes[slot])
        llvmArgTypes[slot] = wrapperFunc.getArgument(argIdx).getType();

  std::string fusedName =
      getAvailableFusedName(module, buildFusedName(fusionCalls));

  // Clone wrapper bodies in block order into one LLVM function.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(fusionFuncs.front());

    auto funcType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmArgTypes);
    // TODO: Merge or recompute function and argument attrs for all cloned
    // wrappers. Map argument attrs through callOperandSlots; missing effect
    // summaries must remain conservative.
    auto fusedFunc = rewriter.create<LLVM::LLVMFuncOp>(firstCall.getLoc(),
                                                       fusedName, funcType);
    // Nothing else can call the fusion of private wrappers. Its symbol is
    // private too, so that symbol-dce erases it once fused again.
    if (llvm::all_of(fusionFuncs, [](LLVM::LLVMFuncOp func) {
          return func.getLinkage() == LLVM::Linkage::Private;
        })) {
      fusedFunc.setLinkage(LLVM::Linkage::Private);
      fusedFunc.setPrivate();
    }

    Block *fusedBlock = fusedFunc.addEntryBlock(rewriter);
    rewriter.setInsertionPointToEnd(fusedBlock);

    for (size_t i = 0; i < fusionCalls.size(); ++i) {
      auto call = fusionCalls[i];
      auto wrapperFunc = fusionFuncs[i];
      // Wrapper arguments are reused SSA keys, so each clone needs a fresh map.
      IRMapping mapping;

      mapCallInputs(rewriter, wrapperFunc, fusedFunc,
                    fusionInfo.callOperandSlots[i], mapping);
      mapCallResults(rewriter, call, wrapperFunc, fusedFunc, resultSlotMap,
                     mapping);

      for (auto &op : wrapperFunc.front().without_terminator())
        rewriter.clone(op, mapping);
    }

    rewriter.create<LLVM::ReturnOp>(firstCall.getLoc(), ValueRange{});
  }

  // Retained results include unused outputs needed to preserve input aliases.
  SmallVector<Type> fusedResultTypes;
  fusedResultTypes.reserve(fusedReturns.size());
  for (Value result : fusedReturns)
    fusedResultTypes.push_back(result.getType());

  // Express retained results as aliases of the fused argument list.
  SmallVector<Attribute> fusedOutputAliases;
  fusedOutputAliases.reserve(fusedReturns.size());
  for (auto [resultIdx, result] : llvm::enumerate(fusedReturns)) {
    auto sourceIt = resultSlotMap.find(result);
    assert(sourceIt != resultSlotMap.end() &&
           "fusion mapping was not validated");

    SmallVector<int64_t> outputTupleIndices;
    if (fusedReturns.size() != 1)
      outputTupleIndices.push_back(static_cast<int64_t>(resultIdx));
    fusedOutputAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        rewriter.getContext(), outputTupleIndices,
        static_cast<int64_t>(sourceIt->second), {}));
  }

  // Replace the group with one JIT call preserving its results and aliases.
  auto newCall = rewriter.create<enzymexla::JITCallOp>(
      firstCall.getLoc(), fusedResultTypes,
      mlir::FlatSymbolRefAttr::get(rewriter.getContext(), fusedName), fusedArgs,
      StringAttr::get(rewriter.getContext(), ""),
      /*operand_layouts=*/firstCall.getOperandLayoutsAttr()
          ? rewriter.getArrayAttr(fusionInfo.operandLayouts)
          : nullptr,
      /*arg_attrs=*/nullptr,
      /*res_attrs=*/nullptr,
      /*output_operand_aliases=*/rewriter.getArrayAttr(fusedOutputAliases),
      getFusedSideEffectFreeAttr(fusionCalls));

  // Replace external users before removing the original SSA producers.
  for (auto [idx, result] : llvm::enumerate(fusedReturns))
    rewriter.replaceAllUsesWith(result, newCall.getResult(idx));

  // Erase consumers before producers so internal SSA uses disappear first.
  for (enzymexla::JITCallOp call : llvm::reverse(fusionCalls)) {
    assert(call->use_empty() && "fusion call still has uses after rewrite");
    rewriter.eraseOp(call);
  }

  return success();
}

// fuses SSA-connected calls
struct FuseDAG : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(enzymexla::JITCallOp jitCallOp,
                                PatternRewriter &rewriter) const override {
    JITFusionInfo fusionInfo = collectDependencyFusionInfo(jitCallOp);
    if (fusionInfo.fusionCalls.size() < 2)
      return failure();
    // The fused call replaces the group at its first call, which need not be
    // the seed.
    enzymexla::JITCallOp firstCall = fusionInfo.fusionCalls.front();
    rewriter.setInsertionPoint(firstCall);
    return rewriteFusion(firstCall, rewriter, fusionInfo);
  }
};

// fuses adjacent jit_calls
// struct FuseAdjacent : public OpRewritePattern<enzymexla::JITCallOp> {
//   using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;
//   void initialize() { setHasBoundedRewriteRecursion(); }

//   LogicalResult matchAndRewrite(enzymexla::JITCallOp jitCallOp,
//                                 PatternRewriter &rewriter) const override {
//     JITFusionInfo fusionInfo = collectGeneralizedFusionInfo(jitCallOp);
//     if (fusionInfo.fusionCalls.size() < 2)
//       return failure();
//     return rewriteFusion(jitCallOp, rewriter, fusionInfo);
//   }
// };

struct FuseJITPass : public impl::FuseJITPassBase<FuseJITPass> {
  using FuseJITPassBase::FuseJITPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (dag)
      patterns.add<FuseDAG>(context);
    // if (adjacent) patterns.add<FuseAdjacent>(context);

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(top_down);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace enzyme
} // namespace mlir
