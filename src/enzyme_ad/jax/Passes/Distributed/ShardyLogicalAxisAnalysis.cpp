#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/IR/Attributes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::distributed {
using namespace mlir::sdy;

llvm::SmallVector<AxisSymbol> SymbolFactorMerge::resolve(AxisSymbol sym) {
  llvm::SmallVector<AxisSymbol> result;
  _resolve(sym, result);
  return result;
}

llvm::SmallVector<AxisSymbol>
SymbolFactorMerge::resolve(llvm::ArrayRef<AxisSymbol> syms) {
  llvm::SmallVector<AxisSymbol> result;
  for (AxisSymbol sym : syms) {
    _resolve(sym, result);
  }
  return result;
}

void SymbolFactorMerge::_resolve(AxisSymbol sym,
                                 llvm::SmallVector<AxisSymbol> &result) {
  // sym may be new to our equivalence class
  AxisSymbol root = symbolUnion.getOrInsertLeaderValue(sym);
  auto it = factorizations.find(root);
  if (it != factorizations.end()) {
    for (AxisSymbol factor : it->second) {
      _resolve(factor, result);
    }
  } else {
    result.push_back(root);
  }
}

static uint64_t extentOfList(llvm::ArrayRef<AxisSymbol> syms) {
  uint64_t extent = 1;
  for (AxisSymbol sym : syms) {
    extent *= sym.getExtent();
  }
  return extent;
}

static llvm::SmallVector<uint64_t>
symbolsToExtents(llvm::ArrayRef<AxisSymbol> syms) {
  llvm::SmallVector<uint64_t> extents;
  extents.reserve(syms.size());
  for (AxisSymbol sym : syms) {
    extents.push_back(sym.getExtent());
  }
  return extents;
}

void SymbolFactorMerge::markOverlapping(
    llvm::ArrayRef<AxisSymbol> overlapping) {
  auto overlapping_resolved = resolve(overlapping);
  for (AxisSymbol sym : overlapping_resolved) {
    auto &overlappingSet = overlappingSymbols[sym];
    for (AxisSymbol other : overlapping_resolved) {
      if (!(other == sym)) {
        overlappingSet.insert(other);
      }
    }
  }
}

SymbolFactorMerge::OverlapSet
SymbolFactorMerge::_getOverlappingForRoot(AxisSymbol sym) {
  assert(sym == resolve(sym).front() && "symbol must be a root symbol");
  auto &overlapping = overlappingSymbols[sym];
  // Whenever we get this set, we should take the chance to deduplicate.
  llvm::SmallVector<AxisSymbol> remove;
  llvm::SmallVector<AxisSymbol> insert;

  for (AxisSymbol other : overlapping) {
    AxisSymbol other_root = symbolUnion.getOrInsertLeaderValue(other);
    if (!(other_root == other)) {
      remove.push_back(other);
      insert.push_back(other_root);
    }
  }
  for (AxisSymbol r : remove) {
    overlapping.erase(r);
  }
  for (AxisSymbol i : insert) {
    if (!(i == sym)) {
      overlapping.insert(i);
    }
  }
  return overlapping;
}

void SymbolFactorMerge::_appendOverlaps(const OverlapSet &from, AxisSymbol to) {
  auto &overlaps = overlappingSymbols[to];
  for (AxisSymbol sym : from) {
    if (!(sym == to)) {
      overlaps.insert(sym);
    }
  }
}

SymbolFactorMerge::OverlapSet
SymbolFactorMerge::getOverlapping(AxisSymbol sym) {
  auto factors = resolve(sym);
  OverlapSet result;
  for (AxisSymbol factor : factors) {
    auto factor_overlaps = _getOverlappingForRoot(factor);
    result.insert(factor_overlaps.begin(), factor_overlaps.end());
  }
  return result;
}

// Conservatively reject merges that would collapse one logical axis onto
// another by reusing the same factor on both sides at different positions
// within the axis, such as propagating through `x + x.T` for a square tensor.
static bool hasSharedFactorOrderConflict(llvm::ArrayRef<AxisSymbol> lhs,
                                         llvm::ArrayRef<AxisSymbol> rhs) {
  auto buildPositions = [](llvm::ArrayRef<AxisSymbol> factors) {
    llvm::DenseMap<AxisSymbol, uint64_t> positions;
    uint64_t minorProduct = 1;
    for (AxisSymbol factor : llvm::reverse(factors)) {
      positions.try_emplace(factor, minorProduct);
      minorProduct *= factor.getExtent();
    }
    return positions;
  };

  llvm::DenseMap<AxisSymbol, uint64_t> lhsPositions = buildPositions(lhs);
  lhsPositions.reserve(lhs.size());

  uint64_t minorProduct = 1;
  for (AxisSymbol factor : llvm::reverse(rhs)) {
    auto it = lhsPositions.find(factor);
    if (it != lhsPositions.end() && it->second != minorProduct) {
      return true;
    }
    minorProduct *= factor.getExtent();
  }
  return false;
}

// Applies one split plan to a symbolic factor list, reusing whole symbols when
// possible and recording any newly introduced subfactors in `factorizations`.
static llvm::SmallVector<llvm::SmallVector<AxisSymbol>> materializeSplitGroups(
    llvm::ArrayRef<AxisSymbol> factors, llvm::ArrayRef<uint64_t> splits,
    llvm::DenseMap<AxisSymbol, llvm::SmallVector<AxisSymbol>> &factorizations) {
  llvm::SmallVector<llvm::SmallVector<AxisSymbol>> groups;
  auto splitSlices =
      axis::computeSplitExtentSlices(symbolsToExtents(factors), splits);
  groups.reserve(splitSlices.size());

  llvm::SmallVector<llvm::SmallVector<AxisSymbol>> factorPiecesByIndex;
  factorPiecesByIndex.resize(factors.size());
  for (const llvm::SmallVector<axis::SplitExtentSlice> &groupSlices :
       splitSlices) {
    llvm::SmallVector<AxisSymbol> currentGroup;
    currentGroup.reserve(groupSlices.size());
    for (const axis::SplitExtentSlice &slice : groupSlices) {
      AxisSymbol factor = factors[slice.extentIdx];
      bool reuseWholeFactor =
          slice.subExtent == factor.getExtent() && slice.stride == 1;
      AxisSymbol piece =
          reuseWholeFactor ? factor : AxisSymbol::create(slice.subExtent);
      currentGroup.push_back(piece);
      factorPiecesByIndex[slice.extentIdx].push_back(piece);
    }
    groups.push_back(std::move(currentGroup));
  }

  for (auto [idx, factor] : llvm::enumerate(factors)) {
    llvm::SmallVector<AxisSymbol> &currentFactorPieces =
        factorPiecesByIndex[idx];
    if (!(currentFactorPieces.size() == 1 &&
          currentFactorPieces.front() == factor)) {
      factorizations[factor] = currentFactorPieces;
    }
  }
  return groups;
}

void SymbolFactorMerge::_factorSymbol(AxisSymbol sym,
                                      llvm::ArrayRef<AxisSymbol> factors) {
#ifndef NDEBUG
  assert(sym == resolve(sym).front() && "symbol must be a root symbol");
  llvm::SmallVector<AxisSymbol> resolvedFactors = resolve(factors);
  assert(llvm::ArrayRef<AxisSymbol>(resolvedFactors) == factors &&
         "factors must be root symbols");
#endif

  const auto &overlapping_sym = getOverlapping(sym);
  for (AxisSymbol factor : factors) {
    if (overlapping_sym.count(factor)) {
      return;
    }
  }

  factorizations[sym] =
      llvm::SmallVector<AxisSymbol>(factors.begin(), factors.end());
  auto overlaps = getOverlapping(sym);
  for (AxisSymbol factor : factors) {
    _appendOverlaps(overlaps, factor);
  }
  _clearOverlapping(sym);
}

void SymbolFactorMerge::_mergeSymbols(AxisSymbol a, AxisSymbol b) {
  assert(a == resolve(a).front() && "symbol must be a root symbol");
  assert(b == resolve(b).front() && "symbol must be a root symbol");

  const auto &overlap_a = getOverlapping(a);
  const auto &overlap_b = getOverlapping(b);
  // technically this should be symmetric.
  if (overlap_a.count(b) || overlap_b.count(a)) {
    // If a and b are already overlapping, we don't need to merge them.
    return;
  }

  auto new_leader = *symbolUnion.unionSets(a, b);
  if (!(new_leader == a)) {
    _appendOverlaps(overlap_a, new_leader);
    _clearOverlapping(a);
  }
  if (!(new_leader == b)) {
    _appendOverlaps(overlap_b, new_leader);
    _clearOverlapping(b);
  }
}

void SymbolFactorMerge::attemptMergeSymbols(llvm::ArrayRef<AxisSymbol> a,
                                            llvm::ArrayRef<AxisSymbol> b) {
  auto lhs_factors = resolve(a);
  auto rhs_factors = resolve(b);
  assert(extentOfList(lhs_factors) == extentOfList(rhs_factors) &&
         "Cannot merge symbols with different extents");
  if (hasSharedFactorOrderConflict(lhs_factors, rhs_factors)) {
    return;
  }
  // Simple recursive base cases:
  if (lhs_factors.size() == 1 && rhs_factors.size() == 1) {
    _mergeSymbols(lhs_factors[0], rhs_factors[0]);
    return;
  } else if (lhs_factors.size() == 1) {
    // Merging a single factor with a list of factors:
    // this means we are asserting a new factorization
    _factorSymbol(lhs_factors[0], rhs_factors);
    return;
  } else if (rhs_factors.size() == 1) {
    _factorSymbol(rhs_factors[0], lhs_factors);
    return;
  }
  // We have two lists of factors to compare.
  // Sometimes we will get factors with the same extent
  // in the same position (by stride not index), and we can
  // merge those two. Othertimes we will get a cut where one sublist
  // corresponds to a single factor in the other list, and we can
  // merge those two using our recursive case (they become factors).
  // Sometimes we have overlapping factors that can be divided into
  // compatible sublists, like [4, 2] and [2, 4] --> [2, 2, 2].
  // Example: [a=2, b=3, x=5, y=7] and [c=6, 5=7, w=5] factors c --> [2, 3].
  // Other times nothing can be done: [5, 7] and [7, 5] have no
  // compatible factorization, and cannot merge. In this case we give up on
  // the mismatching part, and merge any other clean cuts we can find. When
  // introducing new factors, we may need to create new symbols.

  llvm::SmallVector<uint64_t> splits = axis::computeSplits(
      symbolsToExtents(lhs_factors), symbolsToExtents(rhs_factors));

  auto lhsGroups = materializeSplitGroups(lhs_factors, splits, factorizations);
  auto rhsGroups = materializeSplitGroups(rhs_factors, splits, factorizations);
  assert(lhsGroups.size() == rhsGroups.size() &&
         "split plans must produce aligned symbolic groups");

  for (auto [lhsGroup, rhsGroup] : llvm::zip_equal(lhsGroups, rhsGroups)) {
    if (lhsGroup.size() == 1 || rhsGroup.size() == 1) {
      attemptMergeSymbols(lhsGroup, rhsGroup);
      continue;
    }

    llvm::SmallVector<uint64_t> groupSplits = axis::computeSplits(
        symbolsToExtents(lhsGroup), symbolsToExtents(rhsGroup));

    if (groupSplits.size() == 1 &&
        groupSplits.front() == extentOfList(lhsGroup)) {
      continue;
    }
    attemptMergeSymbols(lhsGroup, rhsGroup);
  }
}

/**
 * TODO: we will need to add unification process for shardable
 * dataflow ops like loops, since we want to simplify llama into
 * loops over layers to reduce the degrees of freedom.
 */

// TBD: reshard op, or CollectiveOpInterface?
mlir::sdy::ReshardOp toCollective(Operation *op) {
  return dyn_cast<mlir::sdy::ReshardOp>(op);
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

ShardyLogicalAxisAnalysis::ShardyLogicalAxisAnalysis(func::FuncOp sdy_func)
    : sdy_func(sdy_func) {
  assert(sdy_func.getBody().hasOneBlock() &&
         "axis analysis currently only supports single-block functions");
  buildInitialSymbols();
  buildUnion();
}

ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis
ShardyLogicalAxisAnalysis::getPartitioningAxes(Operation *op) {
  // Ops may be moved after analysis construction (e.g. into
  // distributed.function). We key by operation identity, not parent op type.
  auto pre_factoring = opToPartitioningAxes.lookup(op);
  SymbolsPerPartitioningAxis result;
  result.reserve(pre_factoring.size());
  for (const auto &dim_factors : pre_factoring) {
    result.push_back(symbolFactorMerge.resolve(dim_factors));
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
}

llvm::SmallVector<AxisSymbol>
ShardyLogicalAxisAnalysis::getReductionAxes(OpResult result) {
  Operation *op = result.getOwner();
  if (auto reshardOp = toCollective(op)) {
    return {};
  }

  OpShardingRuleAttr shardingRule = getOrSynthesizeOpShardingRule(op).rule;
  if (!shardingRule) {
    return {};
  }

  assert(!shardingRule.getIsCustomRule() &&
         "TODO: custom sharding rules need dedicated handling");
  assert(shardingRule.getNeedReplicationFactors().empty() &&
         "TODO: need-replication factors need dedicated handling");
  assert(shardingRule.getPermutationFactors().empty() &&
         "TODO: permutation factors need dedicated handling");

  llvm::SmallVector<AxisSymbol> reductionSymbols;
  for (int64_t factorIdx : shardingRule.getReductionFactors()) {
    assert(factorIdx >= 0 &&
           factorIdx < static_cast<int64_t>(opToPartitioningAxes[op].size()) &&
           "reduction factor index must refer to a partitioning symbol");
    llvm::SmallVector<AxisSymbol> resolvedFactors =
        symbolFactorMerge.resolve(opToPartitioningAxes[op][factorIdx]);
    reductionSymbols.append(resolvedFactors.begin(), resolvedFactors.end());
  }

  return reductionSymbols;
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
  return it->second;
}

void ShardyLogicalAxisAnalysis::buildInitialSymbols() {
  // For each tensor argument, add a partitioning dimension per
  // tensor axis.
  for (BlockArgument arg : sdy_func.getArguments()) {
    auto tensorType = dyn_cast_or_null<RankedTensorType>(arg.getType());
    if (!tensorType) {
      continue;
    }

    TensorAxesToPartitionAxes dims;
    dims.reserve(tensorType.getRank());
    for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
      if (tensorType.isDynamicDim(dimIdx)) {
        sdy_func.emitError()
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
    symbolFactorMerge.markOverlapping(flattenNested(dims));
    argToPartitioningAxes[arg] = std::move(dims);
  }

  // For each direct op in the entry block, if it
  // has a sharding rule, give it a new symbol per partitioning factor.
  for (Operation &opRef : sdy_func.getBody().front().getOperations()) {
    Operation *op = &opRef;

    if (auto reshard_op = toCollective(op)) {
      auto &lhs_list = reshardLHSSymbols[op];
      auto &rhs_list = reshardRHSSymbols[op];
      auto inputType =
          dyn_cast<RankedTensorType>(reshard_op.getInput().getType());
      auto outputType =
          dyn_cast<RankedTensorType>(reshard_op.getResult().getType());
      assert(inputType && outputType &&
             "reshard operands must be ranked tensors for axis analysis");
      // for each dimension in the tensor being resharded,
      // if the input and output shardings are the same,
      // LHS and RHS get the same symbol, otherwise different.
      TensorShardingAttr in_sharding = getSharding(reshard_op.getInput());
      TensorShardingAttr out_sharding = getSharding(reshard_op.getResult());
      for (auto [dimIdx, dimShardings] :
           llvm::enumerate(llvm::zip_equal(in_sharding.getDimShardings(),
                                           out_sharding.getDimShardings()))) {
        auto [in_dim, out_dim] = dimShardings;
        assert(!inputType.isDynamicDim(dimIdx) &&
               !outputType.isDynamicDim(dimIdx) &&
               "reshard tensors must have static dimensions for axis analysis");
        uint64_t lhsExtent =
            static_cast<uint64_t>(inputType.getDimSize(dimIdx));
        uint64_t rhsExtent =
            static_cast<uint64_t>(outputType.getDimSize(dimIdx));

        AxisSymbol lhs_sym = AxisSymbol::create(lhsExtent);
        AxisSymbol rhs_sym = lhs_sym;
        if (structurallyEqual(in_dim, out_dim)) {
          assert(lhsExtent == rhsExtent &&
                 "identity reshard dimensions must preserve extent");
          rhs_sym = lhs_sym;
        } else {
          rhs_sym = AxisSymbol::create(rhsExtent);
        }
        lhs_list.push_back(lhs_sym);
        rhs_list.push_back(rhs_sym);
      }
      symbolFactorMerge.markOverlapping(lhs_list);
      symbolFactorMerge.markOverlapping(rhs_list);
    } else if (sdy::OpShardingRuleAttr sharding_rule =
                   getOrSynthesizeOpShardingRule(op).rule) {
      int64_t numDims = sharding_rule.getNumFactors();
      DimToSymbol &symbols = opToPartitioningAxes[op];
      symbols.reserve(numDims);
      for (int64_t i = 0; i < numDims; ++i) {
        symbols.push_back(AxisSymbol::create(
            static_cast<uint64_t>(sharding_rule.getFactorSize(i))));
      }
      symbolFactorMerge.markOverlapping(symbols);
    } else if (auto collective_op = dyn_cast<sdy::CollectiveOpInterface>(op)) {
      // TBD whether this is a real problem
      collective_op.emitWarning(
          "Analysis called on non-reshard collective operation");
    } else if (isa<func::ReturnOp>(op)) {
      // do nothing.
    } else {
      // Remark the op type encountered that isn't shardable
      // for debug. We will have to handle this case
      // eventually: TBD
      op->emitRemark("Operation has no sharding rule: possible "
                     "to-be-implemented");
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
  for (auto [dim, dim_mapping] :
       llvm::enumerate(tensor_mapping.getDimMappings())) {
    mapping.emplace_back();
    auto &dim_vec = mapping.back();
    ArrayRef<int64_t> factorIndices = dim_mapping.getFactorIndices();
    for (int64_t factorIdx : factorIndices) {
      dim_vec.push_back(opToPartitioningAxes[op][factorIdx]);
    }
    dim_vec = symbolFactorMerge.resolve(dim_vec);
  }

  return mapping;
}

ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes
ShardyLogicalAxisAnalysis::getTensorPartitionDims(ReshardOp op, bool isLHS,
                                                  int valueIdx) {
  TensorAxesToPartitionAxes mapping;
  (void)valueIdx;
  auto symbolList = isLHS ? reshardRHSSymbols[op] : reshardLHSSymbols[op];
  mapping.resize(symbolList.size());
  for (auto [dim, sym] : llvm::enumerate(symbolList)) {
    mapping[dim].push_back(sym);
  }
  return mapping;
}

std::optional<ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes>
ShardyLogicalAxisAnalysis::getTensorPartitionDims(Operation *op, bool isLHS,
                                                  int valueIdx) {
  if (auto reshard_op = toCollective(op)) {
    return getTensorPartitionDims(reshard_op, isLHS, valueIdx);
  }

  if (OpShardingRuleAttr sharding_rule =
          getOrSynthesizeOpShardingRule(op).rule) {
    return getTensorPartitionDims(op, sharding_rule, isLHS, valueIdx);
  }

  return std::nullopt;
}

static void mergeProducedAndConsumedAxes(
    const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes &producerMapping,
    const ShardyLogicalAxisAnalysis::TensorAxesToPartitionAxes &consumerMapping,
    SymbolFactorMerge &symbolFactorMerge) {
  if (producerMapping.size() != consumerMapping.size()) {
    return;
  }

  for (auto [prod_dim_factors, cons_dim_factors] :
       llvm::zip_equal(producerMapping, consumerMapping)) {
    symbolFactorMerge.mergeSymbols(prod_dim_factors, cons_dim_factors);
  }
}

/**
 * This somewhat replicates Shardy propogation, but our goal here
 * is to decide when two axes have been equivalent, not make decisions:
 * Shardy has already inserted communication ops where necessary, but
 * but is on a logical mesh basis, wheras we want to completely
 * distangangle different symbols for an axes without a propogation
 * dependency. Some of these axes are hidden within serial / non-sharded
 * partition axes. So, our logic here is:
 *  - If neither the producer nor consumer are a reshard / collective,
 *    then we merge all symbols with a propagation edge.
 *  - If the producer is a collective, we merge only the RHS of the producers
 *    symbols
 *  - If the consumer is a collective, we merge only its LHS symbols.
 * As a precondition for the pass, we require all reshards to be
 * explicit. This means that each producer / consumer op has consistent
 * sharding, simplifying things.
 */
void ShardyLogicalAxisAnalysis::buildUnion() {
  auto mergeUses = [&](const TensorAxesToPartitionAxes &producerMapping,
                       auto &&getUses) {
    for (OpOperand &use : getUses()) {
      auto maybeConsumerMapping = getTensorPartitionDims(use);
      if (!maybeConsumerMapping.has_value()) {
        continue;
      }
      const TensorAxesToPartitionAxes &consumerMapping =
          maybeConsumerMapping.value();
      mergeProducedAndConsumedAxes(producerMapping, consumerMapping,
                                   symbolFactorMerge);
    }
  };

  for (BlockArgument arg : sdy_func.getArguments()) {
    auto it = argToPartitioningAxes.find(arg);
    if (it == argToPartitioningAxes.end()) {
      continue;
    }

    const TensorAxesToPartitionAxes &producerMapping = it->second;
    mergeUses(producerMapping, [&]() { return arg.getUses(); });
  }

  sdy_func.walk([&](Operation *op) {
    // if we are a sharding op or a reshard op,
    // need to consider merges between producer / consumer
    for (OpResult result : op->getResults()) {
      auto maybeProducerMapping = getTensorPartitionDims(result);
      if (!maybeProducerMapping.has_value()) {
        continue;
      }
      TensorAxesToPartitionAxes producerMapping = maybeProducerMapping.value();
      mergeUses(producerMapping, [&]() { return result.getUses(); });
    }
  }); // end of sdy_func.walk
} // end of buildUnion
} // namespace mlir::enzyme::distributed