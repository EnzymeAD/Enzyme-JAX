#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "mlir/IR/Attributes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::distributed {
using namespace mlir::sdy;

static constexpr uint64_t kReplaceThis = 0;

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
    symbolUnion.unionSets(lhs_factors[0], rhs_factors[0]);
    return;
  } else if (lhs_factors.size() == 1) {
    // Merging a single factor with a list of factors:
    // this means we are asserting a new factorization
    factorizations[lhs_factors[0]] = rhs_factors;
    return;
  } else if (rhs_factors.size() == 1) {
    factorizations[rhs_factors[0]] = lhs_factors;
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
  // compatible factorization, and cannot merge. In this case we give up on the
  // mismatching part, and merge any other clean cuts we can find.
  // When introducing new factors, we may need to create new symbols.

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
  buildInitialSymbols();
  buildUnion();
}

ShardyLogicalAxisAnalysis::SymbolsPerPartitioningAxis
ShardyLogicalAxisAnalysis::getPartitioningAxes(Operation *op) {
  // op should be a child of the function
  assert(op->getParentOfType<func::FuncOp>() == sdy_func &&
         "Operation is not a child of the function");
  auto pre_factoring = opToPartitioningAxes.lookup(op);
  SymbolsPerPartitioningAxis result;
  result.reserve(pre_factoring.size());
  for (const auto &dim_factors : pre_factoring) {
    result.push_back(symbolFactorMerge.resolve(dim_factors));
  }
  return result;
}

llvm::SmallVector<AxisSymbol>
ShardyLogicalAxisAnalysis::getReductionAxes(OpResult result) {
  Operation *op = result.getOwner();
  if (auto reshardOp = toCollective(op)) {
    return {};
  }

  OpShardingRuleAttr shardingRule =
      sdy::getOrCreateShardingRule(op,
                                   /*conservativePropagation=*/false,
                                   /*setShardingRuleOnOp=*/false);
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

void ShardyLogicalAxisAnalysis::buildInitialSymbols() {
  // for each operation in the function, if it has a
  // sharding rule, give it a new symbol per
  // dimensions
  sdy_func.walk([&](Operation *op) {
    if (auto reshard_op = toCollective(op)) {
      auto lhs_list = reshardLHSSymbols[op];
      auto rhs_list = reshardRHSSymbols[op];
      // for each dimension in the tensor being resharded,
      // if the input and output shardings are the same,
      // LHS and RHS get the same symbol, otherwise different.
      TensorShardingAttr in_sharding = getSharding(reshard_op.getInput());
      TensorShardingAttr out_sharding = getSharding(reshard_op.getResult());
      for (auto [in_dim, out_dim] : llvm::zip_equal(
               in_sharding.getDimShardings(), out_sharding.getDimShardings())) {
        AxisSymbol lhs_sym = AxisSymbol::create(kReplaceThis);
        AxisSymbol rhs_sym = lhs_sym;
        if (structurallyEqual(in_dim, out_dim)) {
          rhs_sym = lhs_sym;
        } else {
          rhs_sym = AxisSymbol::create(kReplaceThis);
        }
        lhs_list.push_back(lhs_sym);
        rhs_list.push_back(rhs_sym);
      }
    } else if (sdy::OpShardingRuleAttr sharding_rule =
                   sdy::getOrCreateShardingRule(
                       op,
                       /*conservativePropagation=*/
                       false,
                       /*setShardingRuleOnOp=*/false)) {
      int64_t numDims = sharding_rule.getNumFactors();
      DimToSymbol &symbols = opToPartitioningAxes[op];
      symbols.reserve(numDims);
      for (int64_t i = 0; i < numDims; ++i) {
        symbols.push_back(AxisSymbol::create(
            static_cast<uint64_t>(sharding_rule.getFactorSize(i))));
      }

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
  });
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
          sdy::getOrCreateShardingRule(op,
                                       /*conservativePropagation=*/false,
                                       /*setShardingRuleOnOp=*/false)) {
    return getTensorPartitionDims(op, sharding_rule, isLHS, valueIdx);
  }

  return std::nullopt;
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
  sdy_func.walk([&](Operation *op) {
    // if we are a sharding op or a reshard op,
    // need to consider merges between producer / consumer
    for (OpResult result : op->getResults()) {
      auto maybeProducerMapping = getTensorPartitionDims(result);
      if (!maybeProducerMapping.has_value()) {
        continue;
      }
      TensorAxesToPartitionAxes producerMapping = maybeProducerMapping.value();

      for (OpOperand &use : result.getUses()) {
        auto maybeConsumerMapping = getTensorPartitionDims(use);
        if (!maybeConsumerMapping.has_value()) {
          continue;
        }
        TensorAxesToPartitionAxes consumerMapping =
            maybeConsumerMapping.value();
        if (producerMapping.size() != consumerMapping.size()) {
          continue;
        }

        for (auto [prod_dim_factors, cons_dim_factors] :
             llvm::zip_equal(producerMapping, consumerMapping)) {
          symbolFactorMerge.mergeSymbols(prod_dim_factors, cons_dim_factors);
        }
      }
    }
  }); // end of sdy_func.walk
} // end of buildUnion
} // namespace mlir::enzyme::distributed