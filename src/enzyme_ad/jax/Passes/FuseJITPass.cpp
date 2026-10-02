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
  SmallVector<SmallVector<unsigned>> callOperandSlots;
  SmallVector<Value> fusedReturns;
  DenseMap<Value, unsigned> resultSlotMap;
  SmallVector<Operation *> constantsToMove;
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
  // TODO: Remap layouts and argument/result attrs to fused slots. Backend
  // config is opaque, so it needs an explicit composition rule before fusion.
  if (!call.getBackendConfig().empty() || call.getOperandLayoutsAttr() ||
      call.getResultLayoutsAttr() || call.getArgAttrsAttr() ||
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

// Only this call's additions are staged; a rejected extension changes no state.
struct JITFusionExtension {
  SmallVector<Value> newArgs;
  DenseMap<Value, unsigned> readOnlyArgs;
  SmallVector<unsigned> operandSlots;
  SmallVector<unsigned> resultSlots;
  SmallVector<Operation *> constantsToMove;
};

LogicalResult validateFusionExtension(
    enzymexla::JITCallOp call, LLVM::LLVMFuncOp wrapperFunc, Operation *firstOp,
    DominanceInfo &dominance, const JITFusionInfo &info,
    const DenseMap<Value, unsigned> &readOnlyArgs,
    const llvm::SmallPtrSetImpl<Operation *> &movedConstants,
    JITFusionExtension &extension) {
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

  SmallPtrSet<Operation *, 8> newConstants;
  for (auto [operandIdx, operand] : llvm::enumerate(call.getOperands())) {
    // Internal results reuse their producer's storage slot.
    auto source = info.resultSlotMap.find(operand);
    if (source != info.resultSlotMap.end()) {
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

    if (!dominance.properlyDominates(operand, firstOp)) {
      Operation *defOp = operand.getDefiningOp();
      // MPI request initializers may be placed next to later calls. Only
      // operand-free constants can move; a late computed input ends the group.
      if (!defOp || defOp->getBlock() != firstOp->getBlock() ||
          !isa<stablehlo::ConstantOp>(defOp) || defOp->getNumOperands() != 0)
        return failure();
      if (!movedConstants.contains(defOp) && newConstants.insert(defOp).second)
        extension.constantsToMove.push_back(defOp);
    }
    unsigned slot = info.fusedArgs.size() + extension.newArgs.size();
    extension.operandSlots.push_back(slot);
    extension.newArgs.push_back(operand);
    if (readOnly)
      extension.readOnlyArgs[operand] = slot;
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

JITFusionInfo collectGeneralizedFusionInfo(enzymexla::JITCallOp firstCall) {
  // Scan forward in block order, allowing effect-free non-JIT ops between
  // calls. Keep the longest supported prefix with effects or escaping results.
  JITFusionInfo info;
  DominanceInfo dominance;
  if (!dominance.hasSSADominance(firstCall->getBlock()))
    return info;
  auto module = firstCall->getParentOfType<ModuleOp>();
  DenseMap<Value, unsigned> readOnlyArgs;
  SmallPtrSet<Operation *, 8> movedConstants;
  DenseMap<Value, size_t> outsideUses;
  size_t escapingResults = 0;
  bool hasEffects = false;
  size_t legalCalls = 0, legalArgs = 0, legalConstants = 0;

  // Boundary bookkeeping does bounded work per operand/result/use per attempt.
  // Alias searches, greedy revisits and repeated wrapper cloning can cost more.
  for (Operation *op = firstCall; op; op = op->getNextNode()) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;
    auto call = dyn_cast<enzymexla::JITCallOp>(op);
    if (!call) {
      // Effects may only be crossed when their JIT bodies join the group.
      if (!isMemoryEffectFree(op))
        break;
      continue;
    }

    FailureOr<LLVM::LLVMFuncOp> func = lookupFusionFunction(module, call);
    if (failed(func))
      break;
    JITFusionExtension extension;
    if (failed(validateFusionExtension(call, func.value(), firstCall, dominance,
                                       info, readOnlyArgs, movedConstants,
                                       extension)))
      break;

    // Commit the extension only after its complete preflight succeeds.
    info.fusionCalls.push_back(call);
    info.fusionFuncs.push_back(func.value());
    info.callOperandSlots.push_back(std::move(extension.operandSlots));
    llvm::append_range(info.fusedArgs, extension.newArgs);
    readOnlyArgs.insert(extension.readOnlyArgs.begin(),
                        extension.readOnlyArgs.end());
    for (Operation *constant : extension.constantsToMove)
      movedConstants.insert(constant);
    llvm::append_range(info.constantsToMove, extension.constantsToMove);
    for (Value operand : call.getOperands()) {
      auto useIt = outsideUses.find(operand);
      if (useIt != outsideUses.end()) {
        assert(useIt->second != 0 && "missing external operand use");
        // Repeated operands consume separate uses of the same result.
        if (--useIt->second == 0)
          --escapingResults;
      }
    }
    for (auto [idx, result] : llvm::enumerate(call.getResults())) {
      size_t uses = std::distance(result.use_begin(), result.use_end());
      outsideUses[result] = uses;
      escapingResults += uses != 0;
      info.resultSlotMap[result] = extension.resultSlots[idx];
    }
    hasEffects |= !isMemoryEffectFree(call);
    if (info.fusionCalls.size() >= 2 && (hasEffects || escapingResults != 0)) {
      legalCalls = info.fusionCalls.size();
      legalArgs = info.fusedArgs.size();
      legalConstants = info.constantsToMove.size();
    }
  }

  if (legalCalls == 0)
    return JITFusionInfo();

  // A pure suffix may hide all escaping results. Rewind it once to the last
  // observable group, without revalidating prefixes or copying growing maps.
  for (size_t i = info.fusionCalls.size(); i > legalCalls; --i) {
    auto call = info.fusionCalls[i - 1];
    for (Value operand : call.getOperands()) {
      auto useIt = outsideUses.find(operand);
      if (useIt != outsideUses.end())
        ++useIt->second;
    }
    for (Value result : call.getResults()) {
      outsideUses.erase(result);
      info.resultSlotMap.erase(result);
    }
  }
  info.fusionCalls.resize(legalCalls);
  info.fusionFuncs.resize(legalCalls);
  info.callOperandSlots.resize(legalCalls);
  info.fusedArgs.resize(legalArgs);
  info.constantsToMove.resize(legalConstants);

  // Keep external results first, in lexical result order.
  DenseSet<unsigned> returnedSlots;
  for (auto call : info.fusionCalls) {
    for (Value result : call.getResults()) {
      if (outsideUses.lookup(result) != 0) {
        info.fusedReturns.push_back(result);
        returnedSlots.insert(info.resultSlotMap.lookup(result));
      }
    }
  }
  // Preserve an alias for every written input, even if its result is internal
  // or unused. Otherwise constant request buffers can be treated as read-only.
  for (auto call : info.fusionCalls) {
    for (Value result : call.getResults()) {
      if (returnedSlots.insert(info.resultSlotMap.lookup(result)).second)
        info.fusedReturns.push_back(result);
    }
  }
  return info;
}

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
  for (Operation *op : seenCalls)
    info.fusionCalls.push_back(cast<enzymexla::JITCallOp>(op));
  llvm::sort(info.fusionCalls,
             [](enzymexla::JITCallOp lhs, enzymexla::JITCallOp rhs) {
               return lhs->isBeforeInBlock(rhs);
             });

  SmallPtrSet<Operation *, 8> memberOps;
  for (enzymexla::JITCallOp call : info.fusionCalls)
    memberOps.insert(call.getOperation());

  // Moving the group to its first call must preserve every outside effect.
  Operation *firstOp = info.fusionCalls.front().getOperation();
  Operation *lastOp = info.fusionCalls.back().getOperation();
  for (Operation *op = firstOp->getNextNode(); op && op != lastOp;
       op = op->getNextNode()) {
    if (!memberOps.contains(op) && !isMemoryEffectFree(op))
      return JITFusionInfo();
  }

  DominanceInfo dominance;
  if (!dominance.hasSSADominance(block))
    return JITFusionInfo();

  ModuleOp module = seedCall->getParentOfType<ModuleOp>();
  DenseMap<Value, unsigned> readOnlyArgs;
  SmallPtrSet<Operation *, 8> movedConstants;
  bool hasEffects = false;
  bool hasEscapingUse = false;
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    FailureOr<LLVM::LLVMFuncOp> func = lookupFusionFunction(module, call);
    if (failed(func))
      return JITFusionInfo();

    JITFusionExtension extension;
    if (failed(validateFusionExtension(call, func.value(), firstOp, dominance,
                                       info, readOnlyArgs, movedConstants,
                                       extension)))
      return JITFusionInfo();

    info.fusionFuncs.push_back(func.value());
    info.callOperandSlots.push_back(std::move(extension.operandSlots));
    llvm::append_range(info.fusedArgs, extension.newArgs);
    readOnlyArgs.insert(extension.readOnlyArgs.begin(),
                        extension.readOnlyArgs.end());
    for (Operation *constant : extension.constantsToMove)
      movedConstants.insert(constant);
    llvm::append_range(info.constantsToMove, extension.constantsToMove);
    for (auto [resultIdx, result] : llvm::enumerate(call.getResults()))
      info.resultSlotMap[result] = extension.resultSlots[resultIdx];
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

void mapCallInputs(LLVM::LLVMFuncOp wrapperFunc, LLVM::LLVMFuncOp fusedFunc,
                   ArrayRef<unsigned> operandSlots, IRMapping &mapping) {
  // Bind source wrapper arguments to their physical fused pointers.
  for (auto [argIdx, slot] : llvm::enumerate(operandSlots))
    mapping.map(wrapperFunc.getArgument(argIdx), fusedFunc.getArgument(slot));
}

void mapCallResults(enzymexla::JITCallOp call, LLVM::LLVMFuncOp wrapperFunc,
                    LLVM::LLVMFuncOp fusedFunc,
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
      mapping.map(wrapperFunc.getArgument(wrapperArgIdx), mappedResult);
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
  for (Operation *constant : fusionInfo.constantsToMove)
    rewriter.moveOpBefore(constant, firstCall);

  // The JIT wrapper ABI passes every tensor or scalar buffer as !llvm.ptr.
  SmallVector<Type> llvmArgTypes(
      fusedArgs.size(), LLVM::LLVMPointerType::get(rewriter.getContext()));

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

    Block *fusedBlock = fusedFunc.addEntryBlock(rewriter);
    rewriter.setInsertionPointToEnd(fusedBlock);

    for (size_t i = 0; i < fusionCalls.size(); ++i) {
      auto call = fusionCalls[i];
      auto wrapperFunc = fusionFuncs[i];
      // Wrapper arguments are reused SSA keys, so each clone needs a fresh map.
      IRMapping mapping;

      mapCallInputs(wrapperFunc, fusedFunc, fusionInfo.callOperandSlots[i],
                    mapping);
      mapCallResults(call, wrapperFunc, fusedFunc, resultSlotMap, mapping);

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
      /*operand_layouts=*/nullptr,
      /*result_layouts=*/nullptr,
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

struct FuseJITPattern : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(enzymexla::JITCallOp jitCallOp,
                                PatternRewriter &rewriter) const override {
    JITFusionInfo fusionInfo = collectGeneralizedFusionInfo(jitCallOp);
    if (fusionInfo.fusionCalls.size() < 2)
      return failure();
    return rewriteFusion(jitCallOp, rewriter, fusionInfo);
  }
};

struct FuseJITDependencyPattern
    : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(enzymexla::JITCallOp jitCallOp,
                                PatternRewriter &rewriter) const override {
    JITFusionInfo fusionInfo = collectDependencyFusionInfo(jitCallOp);
    if (fusionInfo.fusionCalls.size() < 2 ||
        fusionInfo.fusionCalls.front() != jitCallOp)
      return failure();
    return rewriteFusion(jitCallOp, rewriter, fusionInfo);
  }
};

struct FuseJITPass : public impl::FuseJITPassBase<FuseJITPass> {
  using FuseJITPassBase::FuseJITPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // `dependencies` fuses SSA-connected calls; `generalized` scans a lexical
    // prefix and can include independent calls.
    if (strategy == "dependencies") {
      patterns.add<FuseJITDependencyPattern>(context);
    } else if (strategy == "generalized") {
      patterns.add<FuseJITPattern>(context);
    } else {
      module.emitError() << "unknown fuse-jit strategy '" << strategy
                         << "' (expected 'dependencies' or 'generalized')";
      signalPassFailure();
      return;
    }

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace enzyme
} // namespace mlir
