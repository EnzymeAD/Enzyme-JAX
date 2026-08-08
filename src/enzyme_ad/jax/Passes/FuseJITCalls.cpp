#include "Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fuse-jit-calls"

namespace mlir {
namespace enzyme {

#define GEN_PASS_DEF_FUSEJITCALLSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

// Describes one direct SSA-connected JIT-call component and its IR boundary.
struct JITFusionInfo {
  SmallVector<enzymexla::JITCallOp> fusionCalls;
  SmallVector<Value> fusedArgs;
  SmallVector<Value> fusedReturns;
};

static constexpr llvm::StringLiteral FusedNamePrefix = "__enzyme_fused_";

static bool isGeneratedFusedCall(enzymexla::JITCallOp call) {
  // Prevent the greedy driver from folding a generated call again.
  return call.getFn().getRootReference().getValue().starts_with(
      FusedNamePrefix);
}

static bool containsOperation(ArrayRef<enzymexla::JITCallOp> calls,
                              Operation *op) {
  return llvm::any_of(calls, [&](enzymexla::JITCallOp call) {
    return call.getOperation() == op;
  });
}

static JITFusionInfo collectJITFusionInfo(enzymexla::JITCallOp jitCallOp) {
  JITFusionInfo info;
  SmallVector<enzymexla::JITCallOp> worklist;
  SmallPtrSet<Operation *, 8> seenCalls;

  // Deduplication also makes the bidirectional walk terminate on diamonds.
  auto pushCall = [&](enzymexla::JITCallOp call) {
    if (!call || isGeneratedFusedCall(call))
      return;
    if (seenCalls.insert(call.getOperation()).second)
      worklist.push_back(call);
  };

  pushCall(jitCallOp);
  while (!worklist.empty()) {
    enzymexla::JITCallOp call = worklist.pop_back_val();
    info.fusionCalls.push_back(call);

    // Walk direct JIT producer and consumer edges in both directions.
    // TODO: Extend discovery through selected pure forwarding ops if needed.
    for (Value operand : call.getOperands()) {
      if (auto producer = operand.getDefiningOp<enzymexla::JITCallOp>())
        pushCall(producer);
    }

    for (Value result : call.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (auto consumer = dyn_cast<enzymexla::JITCallOp>(user))
          pushCall(consumer);
      }
    }
  }

  // Lexical order defines cloning order and the fused function ABI.
  if (!info.fusionCalls.empty()) {
    Block *block = info.fusionCalls.front()->getBlock();
    if (llvm::all_of(info.fusionCalls, [&](enzymexla::JITCallOp call) {
          return call->getBlock() == block;
        })) {
      llvm::sort(info.fusionCalls,
                 [](enzymexla::JITCallOp lhs, enzymexla::JITCallOp rhs) {
                   return lhs->isBeforeInBlock(rhs);
                 });
    }
  }

  // Sets only deduplicate; vector insertion order remains deterministic.
  SmallPtrSet<Value, 8> seenArgs;
  SmallPtrSet<Value, 8> seenReturns;

  // Values crossing into the component become fused arguments.
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    for (Value operand : call.getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && containsOperation(info.fusionCalls, defOp))
        continue;
      if (seenArgs.insert(operand).second)
        info.fusedArgs.push_back(operand);
    }
  }

  // Keep only results observed outside the component.
  for (enzymexla::JITCallOp call : info.fusionCalls) {
    for (Value result : call.getResults()) {
      bool hasExternalUser =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !containsOperation(info.fusionCalls, user);
          });
      if (hasExternalUser && seenReturns.insert(result).second)
        info.fusedReturns.push_back(result);
    }
  }

  return info;
}

static LogicalResult validateSameBlock(ArrayRef<enzymexla::JITCallOp> calls) {
  // Cross-block fusion needs control-flow-aware placement and dominance fixes.
  if (calls.empty())
    return failure();

  Block *block = calls.front()->getBlock();
  for (enzymexla::JITCallOp call : calls) {
    if (call->getBlock() != block)
      return failure();
  }
  return success();
}

static LogicalResult validateSingleBlockWrapper(LLVM::LLVMFuncOp func) {
  // Body cloning below copies one block and supplies its own terminator.
  return success(llvm::hasSingleElement(func.getBody()));
}

static std::string buildFusedName(ArrayRef<enzymexla::JITCallOp> fusionCalls) {
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

static FailureOr<SmallVector<LLVM::LLVMFuncOp>>
lookupFusionFunctions(ModuleOp module,
                      ArrayRef<enzymexla::JITCallOp> fusionCalls) {
  // Fusion currently supports single-block, pointer-only, void wrappers.
  SmallVector<LLVM::LLVMFuncOp> fusionFuncs;
  fusionFuncs.reserve(fusionCalls.size());

  for (enzymexla::JITCallOp call : fusionCalls) {
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
        llvm::any_of(funcType.getParams(), [](Type type) {
          return !isa<LLVM::LLVMPointerType>(type);
        }))
      return failure();
    fusionFuncs.push_back(func);
  }

  return fusionFuncs;
}

static std::string getAvailableFusedName(ModuleOp module, StringRef baseName) {
  // TODO: Reuse equivalent fused wrappers once a stable key is available.
  std::string fusedName = baseName.str();
  unsigned suffix = 0;
  while (module.lookupSymbol<LLVM::LLVMFuncOp>(fusedName)) {
    fusedName = (baseName + "_" + Twine(++suffix)).str();
  }
  return fusedName;
}

static FailureOr<unsigned> findValueIndex(Value value, ArrayRef<Value> values) {
  for (auto [idx, candidate] : llvm::enumerate(values)) {
    if (candidate == value)
      return static_cast<unsigned>(idx);
  }
  return failure();
}

static bool aliasMatchesResult(stablehlo::OutputOperandAliasAttr alias,
                               unsigned resultIdx, unsigned numResults) {
  // StableHLO uses [] for one result and [i] for multiple results.
  auto outputTupleIndices = alias.getOutputTupleIndices();
  if (numResults == 1)
    return outputTupleIndices.empty();

  return outputTupleIndices.size() == 1 &&
         outputTupleIndices.front() == static_cast<int64_t>(resultIdx);
}

static FailureOr<int64_t> findAliasedOperandIndex(enzymexla::JITCallOp call,
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

// Resolve a JIT SSA value to the external operand that owns its storage.
static FailureOr<Value>
lookupSourceForValue(Value value, ArrayRef<Value> fusedArgs,
                     const DenseMap<Value, Value> &resultSourceMap) {
  auto resultIt = resultSourceMap.find(value);
  if (resultIt != resultSourceMap.end())
    return resultIt->second;
  if (llvm::is_contained(fusedArgs, value))
    return value;
  return failure();
}

// Preflight every mapping before mutation and record each result's source.
static LogicalResult
validateFusionMappings(ArrayRef<enzymexla::JITCallOp> fusionCalls,
                       ArrayRef<LLVM::LLVMFuncOp> fusionFuncs,
                       ArrayRef<Value> fusedArgs,
                       DenseMap<Value, Value> &resultSourceMap) {
  for (size_t i = 0; i < fusionCalls.size(); ++i) {
    enzymexla::JITCallOp call = fusionCalls[i];
    LLVM::LLVMFuncOp wrapperFunc = fusionFuncs[i];
    // Used wrapper arguments must all receive a value before cloning.
    SmallPtrSet<Value, 8> mappedWrapperArgs;

    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      if (failed(lookupSourceForValue(operand, fusedArgs, resultSourceMap)))
        return failure();
      mappedWrapperArgs.insert(wrapperFunc.getArgument(argIdx));
    }

    for (auto [resultIdx, result] : llvm::enumerate(call.getResults())) {
      FailureOr<int64_t> aliasedOperand =
          findAliasedOperandIndex(call, resultIdx);
      // TODO: Support unaliased results with separate output storage.
      if (failed(aliasedOperand) || aliasedOperand.value() < 0)
        return failure();

      FailureOr<Value> mappedResult = lookupSourceForValue(
          call.getOperand(static_cast<unsigned>(aliasedOperand.value())),
          fusedArgs, resultSourceMap);
      if (failed(mappedResult))
        return failure();

      // Some wrappers append one pointer argument for each JIT result.
      unsigned wrapperArgIdx = call.getNumOperands() + resultIdx;
      if (wrapperArgIdx < wrapperFunc.getNumArguments())
        mappedWrapperArgs.insert(wrapperFunc.getArgument(wrapperArgIdx));
      resultSourceMap[result] = mappedResult.value();
    }

    for (BlockArgument arg : wrapperFunc.getArguments()) {
      if (!mappedWrapperArgs.contains(arg) && !arg.use_empty())
        return failure();
    }
  }
  return success();
}

// The fused call replaces the first call and must not cross other effects.
static LogicalResult
validateRewriteCanMoveToFirstCall(ArrayRef<enzymexla::JITCallOp> fusionCalls,
                                  ArrayRef<Value> fusedArgs) {
  enzymexla::JITCallOp firstCall = fusionCalls.front();
  enzymexla::JITCallOp lastCall = fusionCalls.back();
  Operation *firstOp = firstCall.getOperation();
  Operation *lastOp = lastCall.getOperation();
  Block *block = firstOp->getBlock();

  // Every external operand must dominate the new position at the first call.
  for (Value arg : fusedArgs) {
    Operation *defOp = arg.getDefiningOp();
    if (!defOp || defOp->getBlock() != block)
      continue;
    if (firstOp->isBeforeInBlock(defOp))
      return failure();
  }

  // Later calls may move across pure operations, but never across effects.
  for (Operation *op = firstOp->getNextNode(); op && op != lastOp;
       op = op->getNextNode()) {
    if (containsOperation(fusionCalls, op))
      continue;
    if (!isMemoryEffectFree(op))
      return failure();
  }

  return success();
}

static bool hasSideEffectingCall(ArrayRef<enzymexla::JITCallOp> fusionCalls) {
  // A resultless component is observable only if at least one call has effects.
  return llvm::any_of(fusionCalls, [](enzymexla::JITCallOp call) {
    return !isMemoryEffectFree(call);
  });
}

// LLVM-level counterpart of lookupSourceForValue used while cloning wrappers.
static Value
lookupPointerForValue(Value value, LLVM::LLVMFuncOp fusedFunc,
                      ArrayRef<Value> fusedArgs,
                      const DenseMap<Value, Value> &resultValueMap) {
  auto resultIt = resultValueMap.find(value);
  if (resultIt != resultValueMap.end())
    return resultIt->second;

  FailureOr<unsigned> fusedArgIdx = findValueIndex(value, fusedArgs);
  assert(succeeded(fusedArgIdx) && "fusion mapping was not validated");
  return fusedFunc.getArgument(fusedArgIdx.value());
}

static void mapCallInputs(enzymexla::JITCallOp call,
                          LLVM::LLVMFuncOp wrapperFunc,
                          LLVM::LLVMFuncOp fusedFunc, ArrayRef<Value> fusedArgs,
                          const DenseMap<Value, Value> &resultValueMap,
                          IRMapping &mapping) {
  // Bind source wrapper arguments to their physical fused pointers.
  for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
    Value mappedOperand =
        lookupPointerForValue(operand, fusedFunc, fusedArgs, resultValueMap);
    mapping.map(wrapperFunc.getArgument(argIdx), mappedOperand);
  }
}

static void
mapCallResults(enzymexla::JITCallOp call, LLVM::LLVMFuncOp wrapperFunc,
               LLVM::LLVMFuncOp fusedFunc, ArrayRef<Value> fusedArgs,
               DenseMap<Value, Value> &resultValueMap, IRMapping &mapping) {
  // JIT results name their aliased operand pointers, not LLVM return values.
  for (auto [resultIdx, result] : llvm::enumerate(call.getResults())) {
    FailureOr<int64_t> aliasedOperand =
        findAliasedOperandIndex(call, resultIdx);
    assert(succeeded(aliasedOperand) && aliasedOperand.value() >= 0 &&
           "fusion mapping was not validated");

    Value mappedResult = lookupPointerForValue(
        call.getOperand(static_cast<unsigned>(aliasedOperand.value())),
        fusedFunc, fusedArgs, resultValueMap);

    // Map an appended result pointer when the wrapper ABI contains one.
    unsigned wrapperArgIdx = call.getNumOperands() + resultIdx;
    if (wrapperArgIdx < wrapperFunc.getNumArguments())
      mapping.map(wrapperFunc.getArgument(wrapperArgIdx), mappedResult);
    resultValueMap[result] = mappedResult;
  }
}

// Purity is preserved only when every original call declares it.
static UnitAttr
getFusedSideEffectFreeAttr(ArrayRef<enzymexla::JITCallOp> fusionCalls) {
  if (llvm::all_of(fusionCalls, [](enzymexla::JITCallOp call) {
        return static_cast<bool>(call.getXlaSideEffectFreeAttr());
      })) {
    enzymexla::JITCallOp firstCall = fusionCalls.front();
    return UnitAttr::get(firstCall.getContext());
  }
  return UnitAttr();
}

struct FuseJITCallsPattern : public OpRewritePattern<enzymexla::JITCallOp> {
  using OpRewritePattern<enzymexla::JITCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::JITCallOp jitCallOp,
                                PatternRewriter &rewriter) const override {
    if (isGeneratedFusedCall(jitCallOp))
      return failure();

    JITFusionInfo fusionInfo = collectJITFusionInfo(jitCallOp);
    SmallVector<enzymexla::JITCallOp> &fusionCalls = fusionInfo.fusionCalls;
    if (fusionCalls.size() < 2)
      return failure();
    // Only the earliest call owns the rewrite for this component.
    if (fusionCalls.front() != jitCallOp)
      return failure();
    if (failed(validateSameBlock(fusionCalls)))
      return failure();
    // Avoid replacing a pure component whose results are all dead or internal.
    if (fusionInfo.fusedReturns.empty() && !hasSideEffectingCall(fusionCalls))
      return failure();
    // TODO: Merge compatible call metadata instead of rejecting it.
    if (llvm::any_of(fusionCalls, [](enzymexla::JITCallOp call) {
          return !call.getBackendConfig().empty() ||
                 call.getOperandLayoutsAttr() || call.getResultLayoutsAttr() ||
                 call.getArgAttrsAttr() || call.getResAttrsAttr();
        }))
      return failure();
    if (failed(validateRewriteCanMoveToFirstCall(fusionCalls,
                                                 fusionInfo.fusedArgs)))
      return failure();

    auto module = jitCallOp->getParentOfType<ModuleOp>();
    FailureOr<SmallVector<LLVM::LLVMFuncOp>> fusionFuncsOr =
        lookupFusionFunctions(module, fusionCalls);
    if (failed(fusionFuncsOr))
      return failure();

    SmallVector<LLVM::LLVMFuncOp> &fusionFuncs = fusionFuncsOr.value();
    SmallVector<Value> &fusedArgs = fusionInfo.fusedArgs;
    SmallVector<Value> &fusedReturns = fusionInfo.fusedReturns;
    DenseMap<Value, Value> resultSourceMap;
    // Keep pattern failure atomic by completing validation before IR creation.
    if (failed(validateFusionMappings(fusionCalls, fusionFuncs, fusedArgs,
                                      resultSourceMap)))
      return failure();

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
      // TODO: Merge wrapper attributes such as enzymexla.memory_effects.
      auto fusedFunc = rewriter.create<LLVM::LLVMFuncOp>(jitCallOp.getLoc(),
                                                         fusedName, funcType);

      Block *fusedBlock = fusedFunc.addEntryBlock(rewriter);
      rewriter.setInsertionPointToEnd(fusedBlock);

      // Carry producer result pointers into later consumer wrappers.
      DenseMap<Value, Value> resultValueMap;

      for (size_t i = 0; i < fusionCalls.size(); ++i) {
        auto call = fusionCalls[i];
        auto wrapperFunc = fusionFuncs[i];
        // Wrapper arguments are reused SSA keys, so each clone needs a fresh
        // map.
        IRMapping mapping;

        mapCallInputs(call, wrapperFunc, fusedFunc, fusedArgs, resultValueMap,
                      mapping);
        mapCallResults(call, wrapperFunc, fusedFunc, fusedArgs, resultValueMap,
                       mapping);

        for (auto &op : wrapperFunc.front().without_terminator())
          rewriter.clone(op, mapping);
      }

      rewriter.create<LLVM::ReturnOp>(jitCallOp.getLoc(), ValueRange{});
    }

    // Internal results disappear; external results retain their original types.
    SmallVector<Type> fusedResultTypes;
    fusedResultTypes.reserve(fusedReturns.size());
    for (Value result : fusedReturns)
      fusedResultTypes.push_back(result.getType());

    // Express external results as aliases of the fused argument list.
    SmallVector<Attribute> fusedOutputAliases;
    fusedOutputAliases.reserve(fusedReturns.size());
    for (auto [resultIdx, result] : llvm::enumerate(fusedReturns)) {
      auto sourceIt = resultSourceMap.find(result);
      assert(sourceIt != resultSourceMap.end() &&
             "fusion mapping was not validated");
      FailureOr<unsigned> fusedArgIdx =
          findValueIndex(sourceIt->second, fusedArgs);
      assert(succeeded(fusedArgIdx) && "fusion mapping was not validated");

      SmallVector<int64_t> outputTupleIndices;
      if (fusedReturns.size() != 1)
        outputTupleIndices.push_back(static_cast<int64_t>(resultIdx));
      fusedOutputAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          rewriter.getContext(), outputTupleIndices,
          static_cast<int64_t>(fusedArgIdx.value()), {}));
    }

    // Replace the component with one JIT call exposing its external results.
    auto newCall = rewriter.create<enzymexla::JITCallOp>(
        jitCallOp.getLoc(), fusedResultTypes,
        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), fusedName),
        fusedArgs, StringAttr::get(rewriter.getContext(), ""),
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
};

struct FuseJITCallsPass : public impl::FuseJITCallsPassBase<FuseJITCallsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<FuseJITCallsPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace enzyme
} // namespace mlir
