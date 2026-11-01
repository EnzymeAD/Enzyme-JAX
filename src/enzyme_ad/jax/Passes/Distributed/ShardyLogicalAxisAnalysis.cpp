#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::distributed {
using namespace mlir::sdy;

// Resolves a call to its callee function-like op (a func.func before
// conversion, a distributed.DistributedFunction after).
static Operation *resolveCallee(CallOpInterface call) {
  Operation *callee = call.resolveCallable();
  assert(callee && isa<FunctionOpInterface>(callee) &&
         "axis analysis requires every call to target a function");
  assert(callee->getRegion(0).hasOneBlock() &&
         "axis analysis currently only supports single-block callees");
  return callee;
}

// These operations are introduced after Shardy propagation to materialize
// communication and local tensor views. They intentionally have no Shardy
// rule and should be ignored by a later analysis.
// In the context of our shardy --> distributed pipeline, these are all
// centered around collectives, which shouldn't have propagation anyways.
static bool isNoncomputationalOp(Operation *op) {
  return isa<UnrealizedConversionCastOp, DistributedCollectiveOp,
             DistributedAwait>(op) ||
         op->getName().getDialectNamespace() == "axis";
}

bool structurallyEqual(sdy::DimensionShardingAttr a,
                       sdy::DimensionShardingAttr b) {
  ArrayRef<AxisRefAttr> a_axes = a.getAxes();
  ArrayRef<AxisRefAttr> b_axes = b.getAxes();
  if (a_axes.size() != b_axes.size()) {
    return false;
  }
  for (size_t i = 0; i < a_axes.size(); ++i) {
    // TODO: equality for axis ref attr?
    // Does shardy already have?
    if (a_axes[i] != b_axes[i]) {
      return false;
    }
  }
  return true;
}

// Per dimension of a resharding op's tensor, whether the op may change that
// dimension's sharding, i.e. whether its two sides get distinct symbols.
//   - A reshard point says so itself (see getReshardPointBrokenDims), in
//     search and baseline mode alike.
//   - An sdy.reshard touches exactly the dimensions whose Shardy sharding
//     differs across it. These shardings only recover the reshard's own
//     structure, not any value's physical axis, so without them its
//     semantics are unknown, which is fatal.
// Emits an error and returns nullopt on failure.
static std::optional<llvm::SmallVector<bool>>
getReshardedDims(ReshardingOp reshard) {
  if (isReshardPoint(reshard.op)) {
    FailureOr<llvm::SmallVector<bool>> broken =
        getReshardPointBrokenDims(reshard.op);
    if (failed(broken)) {
      return std::nullopt;
    }
    return *broken;
  }
  TensorShardingAttr inSharding = getSharding(reshard.input);
  TensorShardingAttr outSharding = getSharding(reshard.result);
  if (!inSharding || !outSharding) {
    reshard.op->emitError()
        << "could not recover a sharding for this reshard's "
        << (!inSharding ? "input" : "result")
        << "; axis analysis cannot determine the reshard's semantics "
           "without it";
    return std::nullopt;
  }
  llvm::SmallVector<bool> resharded;
  for (auto [inDim, outDim] : llvm::zip_equal(inSharding.getDimShardings(),
                                              outSharding.getDimShardings())) {
    resharded.push_back(!structurallyEqual(inDim, outDim));
  }
  return resharded;
}

RankedTensorType getLocalTensorType(
    RankedTensorType globalType,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> partitioningAxes) {
  auto globalShape = globalType.getShape();
  assert(globalShape.size() == partitioningAxes.size() &&
         "global shape and partitioning axes must have the same rank");
  llvm::SmallVector<int64_t> localShape;
  localShape.reserve(globalShape.size());
  for (size_t i = 0; i < globalShape.size(); ++i) {
    int64_t globalDim = globalShape[i];
    uint64_t extent = 1;
    for (const auto &symbol : partitioningAxes[i]) {
      extent *= symbol.getExtent();
    }
    assert(globalDim % extent == 0 &&
           "global dimension must be divisible by partitioning extent");
    localShape.push_back(globalDim / extent);
  }
  return RankedTensorType::get(localShape, globalType.getElementType());
}

ShardyLogicalAxisAnalysis::ShardyLogicalAxisAnalysis(Operation *sdy_func)
    : sdy_func(sdy_func) {
  assert(sdy_func && sdy_func->getNumRegions() == 1 &&
         sdy_func->getRegion(0).hasOneBlock() &&
         "axis analysis currently only supports single-block main ops");
  collectAnalyzedFunctions();
  initShardyBaseline();
  if (!valid) {
    return;
  }
  // Every function's symbols must exist before any union runs, since a call
  // boundary reads the callee's argument and return symbols.
  for (Operation *func : analyzedFuncs) {
    buildInitialSymbolsFor(func);
  }
  seedShardyShardings();
  if (!valid) {
    return;
  }
  // Call boundary edges merge first, across every function, so a shared
  // callee's argument and return classes are settled before any interior
  // edge can claim symbols that would make a call site's merge overlap.
  for (bool boundaryEdges : {true, false}) {
    for (Operation *func : analyzedFuncs) {
      buildUnionFor(func, boundaryEdges);
    }
  }
  validateLogicalAxisAssignments();
  validateShardyBaseline();
}

void ShardyLogicalAxisAnalysis::collectAnalyzedFunctions() {
  llvm::SmallPtrSet<Operation *, 8> seen;
  llvm::SmallVector<Operation *> worklist{sdy_func};
  while (!worklist.empty()) {
    Operation *func = worklist.pop_back_val();
    if (!seen.insert(func).second) {
      continue;
    }
    analyzedFuncs.push_back(func);
    Block &body = func->getRegion(0).front();
    for (CallOpInterface call : body.getOps<CallOpInterface>()) {
      worklist.push_back(resolveCallee(call));
    }
  }
}

MainFunctionShardyLogicalAxisAnalysis::MainFunctionShardyLogicalAxisAnalysis(
    ModuleOp module, AnalysisManager &analysisManager) {
  const auto &mainFunctionAnalysis =
      analysisManager.getAnalysis<FindMainFunctionAnalysis, ModuleOp>();
  if (!mainFunctionAnalysis.isValid()) {
    valid = false;
    return;
  }

  Operation *mainOp = mainFunctionAnalysis.getMainFunctionOp();
  if (!mainOp) {
    valid = false;
    return;
  }
  if (mainOp->getNumRegions() != 1 || !mainOp->getRegion(0).hasOneBlock()) {
    valid = false;
    return;
  }

  analysis =
      &analysisManager.getChildAnalysis<ShardyLogicalAxisAnalysis>(mainOp);
  valid = analysis->isValid();
}

ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis
ShardyLogicalAxisAnalysis::getPartitioningAxes(Operation *op) {
  // Ops may be moved after analysis construction (e.g. into
  // distributed.function). We key by operation identity, not parent op type.
  auto pre_factoring = opToPartitioningAxes.lookup(op);
  SymbolsPerPartitioningAxis result;
  result.reserve(pre_factoring.size());
  for (const auto &dim_factors : pre_factoring) {
    result.push_back(axisConstraints.resolve(dim_factors));
  }
  return result;
}

void ShardyLogicalAxisAnalysis::markRewrite(Operation *from, Operation *to) {
  if (!from || !to || from == to) {
    return;
  }

  if (auto opIt = opToPartitioningAxes.find(from);
      opIt != opToPartitioningAxes.end()) {
    opToPartitioningAxes[to] = std::move(opIt->second);
    opToPartitioningAxes.erase(opIt);
  }

  if (auto lhsIt = reshardLHSSymbols.find(from);
      lhsIt != reshardLHSSymbols.end()) {
    reshardLHSSymbols[to] = std::move(lhsIt->second);
    reshardLHSSymbols.erase(lhsIt);
  }

  if (auto rhsIt = reshardRHSSymbols.find(from);
      rhsIt != reshardRHSSymbols.end()) {
    reshardRHSSymbols[to] = std::move(rhsIt->second);
    reshardRHSSymbols.erase(rhsIt);
  }

  auto moveLocalSymbols = [&](TensorLocalSymbols &symbols, unsigned count) {
    for (unsigned idx = 0; idx < count; ++idx) {
      if (auto it = symbols.find({from, idx}); it != symbols.end()) {
        symbols[{to, idx}] = std::move(it->second);
        symbols.erase(it);
      }
    }
  };
  moveLocalSymbols(operandLocalSymbols, from->getNumOperands());
  moveLocalSymbols(resultLocalSymbols, from->getNumResults());
}

llvm::SmallVector<AxisSymbol>
ShardyLogicalAxisAnalysis::getReductionAxes(OpResult result) {
  Operation *op = result.getOwner();
  // Resharding ops have no rule, so they reduce nothing.
  OpShardingRuleAttr shardingRule = getOrSynthesizeOpShardingRule(op).rule;
  if (!shardingRule) {
    return {};
  }

  assert(!shardingRule.getIsCustomRule() &&
         "TODO: custom sharding rules need dedicated handling");
  // Permutation and need-replication factors are never reduction factors
  // (Shardy documents all three factor kinds as disjoint, non-overlapping
  // index sets), so the loop below already skips them correctly -- this is
  // purely informational. Both kinds are tagged unshardable in
  // buildInitialSymbols, so neither is ever handed out as a real logical
  // axis for materialization either.
#ifndef NDEBUG
  if (!shardingRule.getPermutationFactors().empty() &&
      !emittedPermutationFactorRemark) {
    emittedPermutationFactorRemark = true;
    op->emitRemark() << "op has permutation factors in its sharding rule; "
                        "the corresponding logical axes are treated as "
                        "unshardable rather than reduced";
  }
  if (!shardingRule.getNeedReplicationFactors().empty() &&
      !emittedNeedReplicationFactorRemark) {
    emittedNeedReplicationFactorRemark = true;
    op->emitRemark() << "op has need-replication factors in its sharding "
                        "rule; the corresponding logical axes are treated as "
                        "unshardable rather than replicated";
  }
#endif

  llvm::SmallVector<AxisSymbol> reductionSymbols;
  for (int64_t factorIdx : shardingRule.getReductionFactors()) {
    assert(factorIdx >= 0 &&
           factorIdx < static_cast<int64_t>(opToPartitioningAxes[op].size()) &&
           "reduction factor index must refer to a partitioning symbol");
    llvm::SmallVector<AxisSymbol> resolvedFactors =
        axisConstraints.resolve(opToPartitioningAxes[op][factorIdx]);
    reductionSymbols.append(resolvedFactors.begin(), resolvedFactors.end());
  }

  return reductionSymbols;
}

bool ShardyLogicalAxisAnalysis::isUnshardable(AxisSymbol symbol) {
  if (axisConstraints.isLocal(symbol)) {
    return true;
  }
  if (!shardyBaseline) {
    return false;
  }
  return llvm::all_of(axisConstraints.resolve(symbol), [&](AxisSymbol root) {
    return !getPhysicalBinding(root);
  });
}

ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
ShardyLogicalAxisAnalysis::excludeUnshardable(
    const TensorAxesToPartitionAxes &axes) {
  TensorAxesToPartitionAxes filtered;
  filtered.reserve(axes.size());
  for (const auto &dimSymbols : axes) {
    auto &kept = filtered.emplace_back();
    for (AxisSymbol symbol : dimSymbols) {
      if (!isUnshardable(symbol)) {
        kept.push_back(symbol);
      }
    }
  }
  return filtered;
}

llvm::SmallVector<AxisSymbol> ShardyLogicalAxisAnalysis::excludeUnshardable(
    llvm::ArrayRef<AxisSymbol> symbols) {
  llvm::SmallVector<AxisSymbol> kept;
  for (AxisSymbol symbol : symbols) {
    if (!isUnshardable(symbol)) {
      kept.push_back(symbol);
    }
  }
  return kept;
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDims(OpResult result) {
  return getTensorPartitionDims(result.getOwner(), /*isLHS=*/true,
                                result.getResultNumber());
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDims(OpOperand &use) {
  return getTensorPartitionDims(use.getOwner(), /*isLHS=*/false,
                                use.getOperandNumber());
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDims(BlockArgument arg) {
  auto it = argToPartitioningAxes.find(arg);
  if (it == argToPartitioningAxes.end()) {
    return std::nullopt;
  }
  for (int i = 0; i < it->second.size(); ++i) {
    it->second[i] = axisConstraints.resolve(it->second[i]);
  }
  return it->second;
}

// Whether two SSA factors denote the same axis factor. Axis ops without side
// effects (factor, segment, product, and physical mesh axis lookups) are
// values: two such ops with equivalent operands are interchangeable, and CSE
// may merge any subset of them. Only declarative ops (a logical or
// device-local axis declaration) carry identity of their own.
static bool areEquivalentFactors(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  auto lhsResult = dyn_cast<OpResult>(lhs);
  auto rhsResult = dyn_cast<OpResult>(rhs);
  if (!lhsResult || !rhsResult ||
      lhsResult.getResultNumber() != rhsResult.getResultNumber() ||
      !isMemoryEffectFree(lhsResult.getOwner()) ||
      !isMemoryEffectFree(rhsResult.getOwner())) {
    return false;
  }
  return OperationEquivalence::isEquivalentTo(
      lhsResult.getOwner(), rhsResult.getOwner(),
      [](Value a, Value b) { return success(areEquivalentFactors(a, b)); },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

LogicalResult ShardyLogicalAxisAnalysis::assignLogicalAxis(AxisSymbol symbol,
                                                           Value factor) {
  llvm::SmallVector<AxisSymbol> resolved = axisConstraints.resolve(symbol);
  if (resolved.size() != 1) {
    sdy_func->emitError()
        << "cannot anchor a factored logical axis to one SSA factor";
    valid = false;
    return failure();
  }

  AxisSymbol resolvedSymbol = resolved.front();

  // Two symbols our union-find never connected (e.g. a function argument's
  // own symbol and the symbol reached by walking through the yielded
  // result's producer chain -- the yield carries no sharding rule, so
  // buildUnion() never propagates a merge through it) can both be anchored
  // here to the same frozen SSA factor. That shared anchor is ground truth
  // from an earlier pass that they are the same logical axis, so try
  // unioning them now instead of immediately treating it as a conflict.
  if (auto existingIt = factorToLogicalAxis.find(factor);
      existingIt != factorToLogicalAxis.end() &&
      !(existingIt->second == resolvedSymbol)) {
    AxisSymbol existingSymbol = existingIt->second;
    axisConstraints.partialMeetNoLocalize({existingSymbol}, {resolvedSymbol});
    llvm::SmallVector<AxisSymbol> mergedExisting =
        axisConstraints.resolve(existingSymbol);
    llvm::SmallVector<AxisSymbol> mergedNew =
        axisConstraints.resolve(resolvedSymbol);
    rootBindingsDirty = true;
    if (mergedExisting.size() == 1 && mergedNew.size() == 1 &&
        mergedExisting.front() == mergedNew.front()) {
      resolvedSymbol = mergedNew.front();
      existingIt->second = resolvedSymbol;
    }
  }

  auto [symbolIt, insertedSymbol] =
      logicalAxisToFactor.try_emplace(resolvedSymbol, factor);
  auto [factorIt, insertedFactor] =
      factorToLogicalAxis.try_emplace(factor, resolvedSymbol);
  if ((!insertedSymbol && !areEquivalentFactors(symbolIt->second, factor)) ||
      (!insertedFactor && !(factorIt->second == resolvedSymbol))) {
    sdy_func->emitError() << "conflicting logical axis and SSA factor "
                          << "assignment";
    valid = false;
    return failure();
  }
  return success();
}

Value ShardyLogicalAxisAnalysis::getLogicalAxis(AxisSymbol symbol) const {
  auto it = logicalAxisToFactor.find(symbol);
  return it == logicalAxisToFactor.end() ? Value() : it->second;
}

std::optional<AxisSymbol> ShardyLogicalAxisAnalysis::getOrCreateSymbolForFactor(
    TypedValue<axis::AxisFactorType> factor) {
  auto factorIt = factorToLogicalAxis.find(factor);
  if (factorIt != factorToLogicalAxis.end()) {
    return factorIt->second;
  }
  AxisSymbol symbol = AxisSymbol::create(factor.getType().getExtent());
  if (failed(assignLogicalAxis(symbol, factor))) {
    return std::nullopt;
  }
  // A device-local axis is never sharded across the mesh; local symbols are
  // materialized as device-local axes.
  if (isa<DeviceLocalAxisType>(factor.getType().getAxisType())) {
    axisConstraints.localize({symbol});
  }
  if (shardyBaseline) {
    // A factor over a physical axis is already a decided binding.
    FailureOr<TypedValue<axis::AxisTypeInterface>> provenance =
        axis::getFactorProvenanceAxis(factor);
    auto getAxes = succeeded(provenance)
                       ? dyn_cast_or_null<GetPhysicalMeshAxesOp>(
                             provenance->getDefiningOp())
                       : GetPhysicalMeshAxesOp();
    if (getAxes) {
      boundAtoms.push_back(
          {symbol,
           {static_cast<int64_t>(cast<OpResult>(*provenance).getResultNumber()),
            axis::getFactorExtent(factor), axis::getFactorStride(factor)},
           getAxes,
           "frozen physical factor"});
      rootBindingsDirty = true;
    }
  }
  return factorToLogicalAxis.find(factor)->second;
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDimsFromPartitioningAxes(
    Operation *op) {
  ValueRange partitioningAxes;
  if (auto globalToLocal = dyn_cast<DistributedCastGlobalToLocalOp>(op)) {
    partitioningAxes = globalToLocal.getPartitioningAxes();
  } else if (auto localToGlobal =
                 dyn_cast<DistributedCastLocalToGlobalOp>(op)) {
    partitioningAxes = localToGlobal.getPartitioningAxes();
  } else {
    partitioningAxes = cast<AnchorPartitioningOp>(op).getPartitioningAxes();
  }

  TensorAxesToPartitionAxes mapping;
  mapping.reserve(partitioningAxes.size());
  for (Value partitioningAxis : partitioningAxes) {
    auto factorGroup =
        dyn_cast<TypedValue<axis::FactorGroupType>>(partitioningAxis);
    if (!factorGroup) {
      return std::nullopt;
    }
    FailureOr<SmallVector<TypedValue<axis::AxisFactorType>>> factors =
        axis::getProductProvenanceFactors(factorGroup);
    if (failed(factors)) {
      return std::nullopt;
    }

    // A factor group may represent several logical axes. Preserve that basis
    // rather than assigning a single symbol to the group itself.
    SmallVector<AxisSymbol> dimensionSymbols;
    dimensionSymbols.reserve(factors->size());
    for (TypedValue<axis::AxisFactorType> factor : *factors) {
      std::optional<AxisSymbol> symbol = getOrCreateSymbolForFactor(factor);
      if (!symbol) {
        return std::nullopt;
      }
      dimensionSymbols.push_back(*symbol);
    }
    mapping.push_back(axisConstraints.resolve(dimensionSymbols));
  }
  return mapping;
}

void ShardyLogicalAxisAnalysis::buildTensorLocalSymbols(
    Operation *op, OpShardingRuleAttr shardingRule) {
  llvm::SmallDenseSet<int64_t> localFactors;
  localFactors.insert_range(shardingRule.getPermutationFactors());
  localFactors.insert_range(shardingRule.getNeedReplicationFactors());

  auto build = [&](llvm::ArrayRef<TensorMappingAttr> mappings, TypeRange types,
                   TensorLocalSymbols &out) {
    for (auto [valueIdx, mapping] : llvm::enumerate(mappings)) {
      auto tensorType = dyn_cast<RankedTensorType>(types[valueIdx]);
      if (!tensorType) {
        continue;
      }
      DimToSymbol dimSymbols(tensorType.getRank());
      llvm::SmallVector<AxisSymbol> newSymbols;
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        llvm::ArrayRef<int64_t> factors = dimMapping.getFactorIndices();
        if (factors.size() == 1 && localFactors.contains(factors[0]) &&
            !tensorType.isDynamicDim(dim)) {
          dimSymbols[dim] = AxisSymbol::create(
              static_cast<uint64_t>(tensorType.getDimSize(dim)));
          newSymbols.push_back(dimSymbols[dim]);
        }
      }
      if (newSymbols.empty()) {
        continue;
      }
      axisConstraints.localize(newSymbols);
      // Distinct dimensions of one tensor are distinct axes.
      llvm::SmallVector<AxisSymbol> tensorSymbols;
      for (auto [dim, dimMapping] : llvm::enumerate(mapping.getDimMappings())) {
        if (!dimSymbols[dim].isNull()) {
          tensorSymbols.push_back(dimSymbols[dim]);
          continue;
        }
        for (int64_t factor : dimMapping.getFactorIndices()) {
          tensorSymbols.push_back(opToPartitioningAxes[op][factor]);
        }
      }
      axisConstraints.overlap(tensorSymbols);
      out[{op, valueIdx}] = std::move(dimSymbols);
    }
  };
  build(shardingRule.getOperandMappings(), op->getOperandTypes(),
        operandLocalSymbols);
  build(shardingRule.getResultMappings(), op->getResultTypes(),
        resultLocalSymbols);
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getFrozenArgumentAxes(Operation *func,
                                                 BlockArgument arg) {
  auto function = dyn_cast<DistributedFunctionOp>(func);
  if (!function) {
    return std::nullopt;
  }
  IndexedTensorShardingAttr sharding =
      function.getArgumentShardings().getShardings()[arg.getArgNumber()];
  TensorAxesToPartitionAxes dims;
  for (DenseI64ArrayAttr dimAxes : sharding.getDimPartitioningAxes()) {
    SmallVector<AxisSymbol> dimSymbols;
    for (int64_t idx : dimAxes.asArrayRef()) {
      auto factorGroup = cast<TypedValue<axis::FactorGroupType>>(
          function.getPartitioningAxes()[idx]);
      auto factors = axis::getProductProvenanceFactors(factorGroup);
      if (failed(factors)) {
        return std::nullopt;
      }
      for (TypedValue<axis::AxisFactorType> factor : *factors) {
        std::optional<AxisSymbol> symbol = getOrCreateSymbolForFactor(factor);
        if (!symbol) {
          return std::nullopt;
        }
        dimSymbols.push_back(*symbol);
      }
    }
    dims.push_back(axisConstraints.resolve(dimSymbols));
  }
  return dims;
}

void ShardyLogicalAxisAnalysis::buildInitialSymbolsFor(Operation *func) {
  // For each tensor argument, add a partitioning dimension per
  // tensor axis.
  Block &bodyBlock = func->getRegion(0).front();
  for (BlockArgument arg : bodyBlock.getArguments()) {
    auto tensorType = dyn_cast_or_null<RankedTensorType>(arg.getType());
    if (!tensorType) {
      continue;
    }

    // A converted function's axes are already frozen in its metadata, so its
    // arguments start from exactly those factors, as a cast's would.
    if (auto anchored = getFrozenArgumentAxes(func, arg)) {
      axisConstraints.overlap(flattenNested(*anchored));
      argToPartitioningAxes[arg] = std::move(*anchored);
      continue;
    }

    TensorAxesToPartitionAxes dims;
    dims.reserve(tensorType.getRank());
    for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      if (tensorType.isDynamicDim(dimIdx)) {
        sdy_func->emitError()
            << "function argument tensor at position " << arg.getArgNumber()
            << " has a dynamic dimension at index " << dimIdx
            << "; axis analysis requires static-ranked arguments";
        return;
      }
      llvm::SmallVector<AxisSymbol> dimSymbols;
      dimSymbols.push_back(AxisSymbol::create(
          static_cast<uint64_t>(tensorType.getDimSize(dimIdx))));
      dims.push_back(std::move(dimSymbols));
    }
    axisConstraints.overlap(flattenNested(dims));
    argToPartitioningAxes[arg] = std::move(dims);
  }

  // For each direct op in the entry block, if it
  // has a sharding rule, give it a new symbol per partitioning factor.
  for (Operation &opRef : bodyBlock.getOperations()) {
    Operation *op = &opRef;

    if (std::optional<ReshardingOp> reshard = getReshardingOp(op)) {
      std::optional<llvm::SmallVector<bool>> reshardedDims =
          getReshardedDims(*reshard);
      if (!reshardedDims) {
        valid = false;
        return;
      }
      auto type = cast<RankedTensorType>(reshard->result.getType());
      assert(type.hasStaticShape() && reshard->input.getType() == type &&
             "a resharding op keeps its static tensor type");
      auto &lhs_list = reshardLHSSymbols[op];
      auto &rhs_list = reshardRHSSymbols[op];
      for (auto [dimIdx, resharded] : llvm::enumerate(*reshardedDims)) {
        AxisSymbol lhs_sym =
            AxisSymbol::create(static_cast<uint64_t>(type.getDimSize(dimIdx)));
        lhs_list.push_back(lhs_sym);
        rhs_list.push_back(resharded ? AxisSymbol::create(lhs_sym.getExtent())
                                     : lhs_sym);
      }
      axisConstraints.overlap(lhs_list);
      axisConstraints.overlap(rhs_list);
    } else if (isa<DistributedCastGlobalToLocalOp,
                   DistributedCastLocalToGlobalOp, AnchorPartitioningOp>(op)) {
      // Their partitioning_axes are ground truth for the tensors they bind. An
      // anchor is a same-scope identity marker, exactly like the two casts.
      auto mapping = getTensorPartitionDimsFromPartitioningAxes(op);
      if (!mapping) {
        op->emitError() << "failed to recover partitioning axes from tensor "
                           "view cast or anchor";
        continue;
      }
      axisConstraints.overlap(flattenNested(*mapping));
    } else if (isa<CallOpInterface>(op)) {
      // A call has no symbols of its own: it shares its callee's argument and
      // return symbols (see getTensorPartitionDimsForCall).
    } else if (sdy::OpShardingRuleAttr sharding_rule =
                   getOrSynthesizeOpShardingRule(op).rule) {
      int64_t numDims = sharding_rule.getNumFactors();
      DimToSymbol &symbols = opToPartitioningAxes[op];
      symbols.reserve(numDims);
      for (int64_t i = 0; i < numDims; ++i) {
        symbols.push_back(AxisSymbol::create(
            static_cast<uint64_t>(sharding_rule.getFactorSize(i))));
      }
      axisConstraints.overlap(symbols);
      // Permutation factors (e.g. the sliced-away dimension of a static
      // per-layer stablehlo.slice) would need a collective-permute, and
      // need-replication factors (e.g. the concatenated dimension of a
      // stablehlo.concatenate) would need an all-gather, if ever actually
      // sharded; we don't implement either collective, so guarantee neither
      // factor kind ever is by excluding both from real logical axis
      // materialization later.
      llvm::SmallVector<AxisSymbol> mustStayLocalSymbols;
      for (int64_t idx : sharding_rule.getPermutationFactors()) {
        mustStayLocalSymbols.push_back(symbols[idx]);
      }
      for (int64_t idx : sharding_rule.getNeedReplicationFactors()) {
        mustStayLocalSymbols.push_back(symbols[idx]);
      }
      if (!mustStayLocalSymbols.empty()) {
        axisConstraints.localize(mustStayLocalSymbols);
        buildTensorLocalSymbols(op, sharding_rule);
      }
    } else if (auto collective_op = dyn_cast<sdy::CollectiveOpInterface>(op)) {
      // TBD whether this is a real problem
      collective_op.emitWarning(
          "Analysis called on non-reshard collective operation");
    } else if (isNoncomputationalOp(op)) {
      // These values form materialization boundaries, so analysis deliberately
      // records no logical axes for them.
    } else if (isa<func::ReturnOp, DistributedYieldOp>(op)) {
      // do nothing.
    } else {
#ifndef NDEBUG
      op->emitRemark("Operation has no sharding rule: possible "
                     "to-be-implemented");
#endif
    }
  }
}

// Returns a list partition axis indices to their footprint on the
// tensor axes.
ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
ShardyLogicalAxisAnalysis::getTensorPartitionDims(
    Operation *op, OpShardingRuleAttr sharding_rule, bool isLHS, int valueIdx) {
  TensorAxesToPartitionAxes mapping;
  llvm::ArrayRef<TensorMappingAttr> tensor_mappings =
      isLHS ? sharding_rule.getResultMappings()
            : sharding_rule.getOperandMappings();

  assert(tensor_mappings.size() ==
             (isLHS ? op->getNumResults() : op->getNumOperands()) &&
         "Mismatch between number of tensor mappings and operands/results: "
         "double-check rule attr semantics");
  TensorMappingAttr tensor_mapping = tensor_mappings[valueIdx];
  const TensorLocalSymbols &localSymbols =
      isLHS ? resultLocalSymbols : operandLocalSymbols;
  for (auto [dim, dim_mapping] :
       llvm::enumerate(tensor_mapping.getDimMappings())) {
    mapping.emplace_back();
    auto &dim_vec = mapping.back();
    if (auto localIt = localSymbols.find({op, valueIdx});
        localIt != localSymbols.end() && !localIt->second[dim].isNull()) {
      dim_vec.push_back(localIt->second[dim]);
      dim_vec = axisConstraints.resolve(dim_vec);
      continue;
    }
    ArrayRef<int64_t> factorIndices = dim_mapping.getFactorIndices();
    for (int64_t factorIdx : factorIndices) {
      dim_vec.push_back(opToPartitioningAxes[op][factorIdx]);
    }
    dim_vec = axisConstraints.resolve(dim_vec);
  }

  return mapping;
}

ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
ShardyLogicalAxisAnalysis::getTensorPartitionDims(ReshardingOp reshard,
                                                  bool isLHS) {
  TensorAxesToPartitionAxes mapping;
  auto symbolList =
      isLHS ? reshardRHSSymbols[reshard.op] : reshardLHSSymbols[reshard.op];
  mapping.resize(symbolList.size());
  for (auto [dim, sym] : llvm::enumerate(symbolList)) {
    mapping[dim] = axisConstraints.resolve(sym);
  }
  return mapping;
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDims(Operation *op, bool isLHS,
                                                  int valueIdx) {
  if (isa<DistributedCastGlobalToLocalOp, DistributedCastLocalToGlobalOp,
          AnchorPartitioningOp>(op)) {
    return getTensorPartitionDimsFromPartitioningAxes(op);
  }
  if (std::optional<ReshardingOp> reshard = getReshardingOp(op)) {
    return getTensorPartitionDims(*reshard, isLHS);
  }
  // A localized DistributedCall is a call like any other: its operands and
  // results carry the callee's argument and result axes, and are bound to
  // the surrounding global values by the casts around it.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    return getTensorPartitionDimsForCall(call, isLHS, valueIdx);
  }

  if (OpShardingRuleAttr sharding_rule =
          getOrSynthesizeOpShardingRule(op).rule) {
    return getTensorPartitionDims(op, sharding_rule, isLHS, valueIdx);
  }

  return std::nullopt;
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDimsForCall(CallOpInterface call,
                                                         bool isLHS,
                                                         int valueIdx) {
  Block &calleeBody = resolveCallee(call)->getRegion(0).front();
  if (!isLHS) {
    return getTensorPartitionDims(calleeBody.getArgument(valueIdx));
  }
  Value returned = calleeBody.getTerminator()->getOperand(valueIdx);
  if (auto arg = dyn_cast<BlockArgument>(returned)) {
    return getTensorPartitionDims(arg);
  }
  return getTensorPartitionDims(cast<OpResult>(returned));
}

static void mergeProducedAndConsumedAxes(
    const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes &producerMapping,
    const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes &consumerMapping,
    axis::IncrementalAxisConstraints &axisConstraints) {
  if (producerMapping.size() != consumerMapping.size()) {
    return;
  }

  for (auto [prod_dim_factors, cons_dim_factors] :
       llvm::zip_equal(producerMapping, consumerMapping)) {
    axisConstraints.partialMeetNoLocalize(prod_dim_factors, cons_dim_factors);
  }
}

void ShardyLogicalAxisAnalysis::validateLogicalAxisAssignments() {
  llvm::DenseMap<AxisSymbol, Value> resolvedSymbolToFactor;
  for (const auto &[symbol, factor] : logicalAxisToFactor) {
    llvm::SmallVector<AxisSymbol> resolved = axisConstraints.resolve(symbol);
    if (resolved.size() != 1) {
      sdy_func->emitError()
          << "factored logical axis cannot have one SSA factor anchor";
      valid = false;
      return;
    }

    AxisSymbol resolvedSymbol = resolved.front();
    auto [it, inserted] =
        resolvedSymbolToFactor.try_emplace(resolvedSymbol, factor);
    if (!inserted && !areEquivalentFactors(it->second, factor)) {
      sdy_func->emitError()
          << "unified logical axis has distinct SSA factor anchors "
          << it->second << " and " << factor;
      valid = false;
      return;
    }
  }

  // Future lookups use resolved symbols, so retain anchors under the same
  // canonical keys used by partitioning and kernel construction.
  logicalAxisToFactor = std::move(resolvedSymbolToFactor);
  for (auto &[factor, symbol] : factorToLogicalAxis) {
    symbol = axisConstraints.resolve(symbol).front();
  }
}

/**
 * This somewhat replicates Shardy propogation, but our goal here
 * is to decide when two axes have been equivalent, not make decisions:
 * Shardy has already (mostly) inserted communication ops where necessary,
 * but is on a physical mesh basis, resulting in accidental collisions onto the
 * same physical axis. We want to completely distangangle different symbols for
 * an axes without a propogation dependency. Some of these axes are hidden
 * within serial / non-sharded partition axes. So, our logic here is:
 *  - If neither the producer nor consumer are a reshard / collective,
 *    then we merge all symbols with a propagation edge.
 *  - If the producer is a collective, we merge only the RHS of the producers
 *    symbols
 *  - If the consumer is a collective, we merge only its LHS symbols.
 * In this propagation / unioning, we respect exising reshard operations:
 * any decisions made by Shardy are kept as distinct degrees of freedom /
 * separate axes in our analysis. However, that doesn't gaurantee sharding to be
 * entirely compatible: we can have discovered sharding on new previously-serial
 * axes that may introduce conflicts. We need to be ensure that we never place
 * the same sharding axis symbol in two distinct places / axes on an operation
 * or tensor. This is done by maintaining an "overlapping" set of symbols in our
 * union-factor-find: we can never merge any symbols that are overlapping. This
 * rejection is located in the datastructure itself.
 */
void ShardyLogicalAxisAnalysis::buildUnionFor(Operation *func,
                                              bool boundaryEdges) {
  auto mergeUses = [&](const TensorAxesToPartitionAxes &producerMapping,
                       Operation *producer, Value producerValue,
                       auto &&getUses) {
    for (OpOperand &use : getUses()) {
      bool isBoundary = isa<CallOpInterface>(use.getOwner()) ||
                        (producer && isa<CallOpInterface>(producer));
      if (isBoundary != boundaryEdges) {
        continue;
      }
      auto maybeConsumerMapping = getTensorPartitionDims(use);
      if (!maybeConsumerMapping.has_value()) {
        continue;
      }
      const TensorAxesToPartitionAxes &consumerMapping =
          maybeConsumerMapping.value();
      if (shardyBaseline) {
        mergeEdgeRespectingShardy(producerMapping, producerValue,
                                  consumerMapping, use, isBoundary);
        continue;
      }
      mergeProducedAndConsumedAxes(producerMapping, consumerMapping,
                                   axisConstraints);
    }
  };

  Block &bodyBlock = func->getRegion(0).front();
  for (BlockArgument arg : bodyBlock.getArguments()) {
    auto it = argToPartitioningAxes.find(arg);
    if (it == argToPartitioningAxes.end()) {
      continue;
    }

    const TensorAxesToPartitionAxes &producerMapping = it->second;
    mergeUses(producerMapping, nullptr, arg, [&]() { return arg.getUses(); });
  }

  for (Operation &opRef : bodyBlock.getOperations()) {
    Operation *op = &opRef;
    for (OpResult result : op->getResults()) {
      auto maybeProducerMapping = getTensorPartitionDims(result);
      if (!maybeProducerMapping.has_value()) {
        continue;
      }
      TensorAxesToPartitionAxes producerMapping = maybeProducerMapping.value();
      mergeUses(producerMapping, op, result,
                [&]() { return result.getUses(); });
    }
  }
} // end of buildUnionFor

// Prints one symbol as its resolved SSA factor name, falling back to a
// synthetic "aN" name when no factor has been assigned yet (e.g. mid-rewrite).
static void printAxisSymbol(llvm::raw_ostream &os,
                            ShardyLogicalAxisAnalysis &axisAnalysis,
                            AxisSymbol symbol) {
  if (Value factor = axisAnalysis.getLogicalAxis(symbol)) {
    factor.printAsOperand(os, OpPrintingFlags());
  } else {
    os << "a" << symbol.getId();
  }
  if (axisAnalysis.isUnshardable(symbol)) {
    os << "(unshardable)";
  }
}

// Prints a "list of lists" of axis symbols, one bracketed group per tensor
// dimension, e.g. " [%ax1, %ax2] [%ax3]".
static void printAxesPerDimension(
    llvm::raw_ostream &os, ShardyLogicalAxisAnalysis &axisAnalysis,
    llvm::ArrayRef<llvm::SmallVector<AxisSymbol>> axesPerDim) {
  for (const auto &dimSymbols : axesPerDim) {
    os << " [";
    llvm::interleaveComma(dimSymbols, os, [&](const AxisSymbol &symbol) {
      printAxisSymbol(os, axisAnalysis, symbol);
    });
    os << "]";
  }
}

void dumpValueAxes(llvm::raw_ostream &os, Block *block,
                   ShardyLogicalAxisAnalysis &axisAnalysis) {
  os << "===== Logical Axes per Value =====\n";
  for (BlockArgument arg : block->getArguments()) {
    auto axes = axisAnalysis.getTensorPartitionDims(arg);
    if (!axes) {
      continue;
    }
    arg.printAsOperand(os, OpPrintingFlags());
    os << " :";
    printAxesPerDimension(os, axisAnalysis, *axes);
    os << "\n";
  }
  for (Operation &op : block->getOperations()) {
    for (OpResult result : op.getResults()) {
      auto axes = axisAnalysis.getTensorPartitionDims(result);
      if (!axes) {
        continue;
      }
      result.printAsOperand(os, OpPrintingFlags());
      os << " :";
      printAxesPerDimension(os, axisAnalysis, *axes);
      os << "\n";
    }
  }
  os << "===== End Logical Axes per Value =====\n";
}

void dumpOperationAxes(llvm::raw_ostream &os, Block *block,
                       ShardyLogicalAxisAnalysis &axisAnalysis) {
  os << "===== Logical Axes per Operation =====\n";
  for (Operation &op : block->getOperations()) {
    auto axes = axisAnalysis.getPartitioningAxes(&op);
    op.print(os, OpPrintingFlags().skipRegions());
    os << " :";
    printAxesPerDimension(os, axisAnalysis, axes);
    os << "\n";
  }
  os << "===== End Logical Axes per Operation =====\n";
}

TypedValue<axis::AxisFactorType>
createLogicalAxisFactor(AxisSymbol symbol,
                        ShardyLogicalAxisAnalysis &axisAnalysis,
                        OpBuilder &builder, Location loc) {
  if (axisAnalysis.isShardyBaseline()) {
    if (std::optional<PhysicalFactorRef> binding =
            axisAnalysis.getPhysicalBinding(symbol)) {
      return axisAnalysis.createPhysicalFactor(*binding, builder, loc);
    }
    Value localAxis =
        builder.create<DeviceLocalAxisOp>(loc, symbol.getExtent()).getAxis();
    return axis::viewAxisAsFactor(localAxis, builder, loc);
  }
  Value axisValue;
  if (axisAnalysis.isUnshardable(symbol)) {
    axisValue =
        builder.create<DeviceLocalAxisOp>(loc, symbol.getExtent()).getAxis();
  } else {
    axisValue =
        builder.create<LogicalMeshAxesOp>(loc, symbol.getExtent()).getAxis();
  }
  return axis::viewAxisAsFactor(axisValue, builder, loc);
}
} // namespace mlir::enzyme::distributed