#include "src/enzyme_ad/jax/Dialect/Axis/IncrementalAxisConstraints.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include <functional>
#include <numeric>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::enzyme::axis {

llvm::SmallVector<AxisSymbol>
IncrementalAxisConstraints::resolve(AxisSymbol symbol) {
  llvm::SmallVector<AxisSymbol> result;
  resolveInto(symbol, result);
  return result;
}

llvm::SmallVector<AxisSymbol>
IncrementalAxisConstraints::resolve(llvm::ArrayRef<AxisSymbol> symbols) {
  llvm::SmallVector<AxisSymbol> result;
  for (AxisSymbol symbol : symbols) {
    resolveInto(symbol, result);
  }
  return result;
}

void IncrementalAxisConstraints::resolveInto(
    AxisSymbol symbol, llvm::SmallVector<AxisSymbol> &result) {
  // symbol may be new to our equivalence class
  AxisSymbol root = symbolUnion.getOrInsertLeaderValue(symbol);
  auto it = factorizations.find(root);
  if (it != factorizations.end()) {
    for (AxisSymbol piece : it->second) {
      resolveInto(piece, result);
    }
  } else {
    result.push_back(root);
  }
}

static uint64_t extentOfList(llvm::ArrayRef<AxisSymbol> symbols) {
  uint64_t extent = 1;
  for (AxisSymbol symbol : symbols) {
    extent *= symbol.getExtent();
  }
  return extent;
}

void IncrementalAxisConstraints::overlap(llvm::ArrayRef<AxisSymbol> symbols) {
  auto roots = resolve(symbols);
  for (AxisSymbol root : roots) {
    auto &overlappingSet = overlappingSymbols[root];
    for (AxisSymbol other : roots) {
      if (!(other == root)) {
        overlappingSet.insert(other);
      }
    }
  }
}

void IncrementalAxisConstraints::localize(llvm::ArrayRef<AxisSymbol> symbols) {
  for (AxisSymbol root : resolve(symbols)) {
    localSymbols.insert(root);
  }
}

bool IncrementalAxisConstraints::isLocal(AxisSymbol symbol) {
  for (AxisSymbol root : resolve(symbol)) {
    if (localSymbols.contains(root)) {
      return true;
    }
  }
  return false;
}

IncrementalAxisConstraints::OverlapSet
IncrementalAxisConstraints::overlappingOfRoot(AxisSymbol root) {
  assert(root == resolve(root).front() && "symbol must be a root symbol");
  auto &overlapping = overlappingSymbols[root];
  // The set may name symbols that have since been factored or merged away;
  // replace each with its current roots.
  llvm::SmallVector<AxisSymbol> remove;
  llvm::SmallVector<AxisSymbol> insert;

  for (AxisSymbol other : overlapping) {
    auto resolution = resolve(other);
    if (!(resolution.size() == 1 && resolution.front() == other)) {
      remove.push_back(other);
      insert.append(resolution.begin(), resolution.end());
    }
  }
  for (AxisSymbol r : remove) {
    overlapping.erase(r);
  }
  for (AxisSymbol i : insert) {
    if (!(i == root)) {
      overlapping.insert(i);
    }
  }
  return overlapping;
}

void IncrementalAxisConstraints::appendOverlaps(const OverlapSet &from,
                                                AxisSymbol to) {
  auto &overlaps = overlappingSymbols[to];
  for (AxisSymbol symbol : from) {
    if (!(symbol == to)) {
      overlaps.insert(symbol);
    }
  }
}

IncrementalAxisConstraints::OverlapSet
IncrementalAxisConstraints::overlapping(AxisSymbol symbol) {
  AxisSymbol arr[] = {symbol};
  return overlapping(arr);
}

IncrementalAxisConstraints::OverlapSet
IncrementalAxisConstraints::overlapping(llvm::ArrayRef<AxisSymbol> symbols) {
  OverlapSet result;
  for (AxisSymbol root : resolve(symbols)) {
    auto rootOverlaps = overlappingOfRoot(root);
    result.insert(rootOverlaps.begin(), rootOverlaps.end());
  }
  return result;
}

static llvm::SmallVector<uint64_t>
symbolExtents(llvm::ArrayRef<AxisSymbol> symbols) {
  return llvm::map_to_vector(
      symbols, [](AxisSymbol symbol) { return symbol.getExtent(); });
}

// Conservatively reject merges that would collapse one logical axis onto
// another by reusing the same factor on both sides at different positions
// within the axis, such as propagating through `x + x.T` for a square tensor.
static bool hasSharedFactorOrderConflict(llvm::ArrayRef<AxisSymbol> lhs,
                                         llvm::ArrayRef<AxisSymbol> rhs) {
  llvm::DenseMap<AxisSymbol, uint64_t> lhsStrides;
  for (auto [factor, stride] :
       llvm::zip_equal(lhs, digitStrides(symbolExtents(lhs))))
    lhsStrides[factor] = stride;

  for (auto [factor, stride] :
       llvm::zip_equal(rhs, digitStrides(symbolExtents(rhs)))) {
    auto it = lhsStrides.find(factor);
    if (it != lhsStrides.end() && it->second != stride)
      return true;
  }
  return false;
}

static bool hasOverlappingSymbolsConflict(
    const llvm::ArrayRef<AxisSymbol> lhs, const llvm::ArrayRef<AxisSymbol> rhs,
    const IncrementalAxisConstraints::OverlapSet &lhsOverlaps,
    const IncrementalAxisConstraints::OverlapSet &rhsOverlaps) {
  // if any symbol on either side has a non-identity overlap in the other side,
  // we cannot merge
  for (AxisSymbol symbol : lhs) {
    if (llvm::is_contained(rhsOverlaps, symbol)) {
      return true;
    }
  }
  for (AxisSymbol symbol : rhs) {
    if (llvm::is_contained(lhsOverlaps, symbol)) {
      return true;
    }
  }
  return false;
}

llvm::SmallVector<AxisSymbol>
IncrementalAxisConstraints::factor(AxisSymbol symbol,
                                   llvm::ArrayRef<uint64_t> extents) {
  assert(std::accumulate(extents.begin(), extents.end(), uint64_t{1},
                         std::multiplies<uint64_t>()) == symbol.getExtent() &&
         "pieces must cover the symbol");
  llvm::SmallVector<AxisSymbol> pieces =
      llvm::map_to_vector(extents, AxisSymbol::create);
  factorInto(symbol, pieces);
  return pieces;
}

void IncrementalAxisConstraints::factorInto(AxisSymbol symbol,
                                            llvm::ArrayRef<AxisSymbol> pieces) {
#ifndef NDEBUG
  assert(symbol == resolve(symbol).front() && "symbol must be a root symbol");
  llvm::SmallVector<AxisSymbol> resolvedPieces = resolve(pieces);
  assert(llvm::ArrayRef<AxisSymbol>(resolvedPieces) == pieces &&
         "pieces must be root symbols");
#endif

  // Factoring a local symbol into finer pieces doesn't make those pieces any
  // more shardable: every piece stays local too.
  if (localSymbols.contains(symbol)) {
    localSymbols.erase(symbol);
    for (AxisSymbol piece : pieces) {
      localSymbols.insert(piece);
    }
  }

  // Each piece inherits what `symbol` itself overlapped. The set must be read
  // before the factorization is recorded: afterwards `symbol` resolves to its
  // pieces, and the set would be the union of the pieces' own overlaps,
  // handing every piece its siblings' unrelated conflicts.
  OverlapSet overlaps = overlapping(symbol);

  factorizations[symbol] =
      llvm::SmallVector<AxisSymbol>(pieces.begin(), pieces.end());
  for (AxisSymbol piece : pieces) {
    appendOverlaps(overlaps, piece);
  }
  overlappingSymbols.erase(symbol);
  overlap(pieces);
}

void IncrementalAxisConstraints::meetSymbols(AxisSymbol a, AxisSymbol b) {
  assert(a == resolve(a).front() && "symbol must be a root symbol");
  assert(b == resolve(b).front() && "symbol must be a root symbol");

  auto overlapA = overlapping(a);
  auto overlapB = overlapping(b);
  bool local = localSymbols.contains(a) || localSymbols.contains(b);

  auto newLeader = *symbolUnion.unionSets(a, b);
  if (!(newLeader == a)) {
    appendOverlaps(overlapA, newLeader);
    overlappingSymbols.erase(a);
    localSymbols.erase(a);
  }
  if (!(newLeader == b)) {
    appendOverlaps(overlapB, newLeader);
    overlappingSymbols.erase(b);
    localSymbols.erase(b);
  }
  if (local) {
    localSymbols.insert(newLeader);
  }
}

namespace {

using MergeConflict = IncrementalAxisConstraints::MergeConflict;

// `extent` digits of a root symbol, whose minor-most has stride `stride`
// within it: a piece align() splits the symbol into.
struct SymbolSlice {
  AxisSymbol symbol;
  uint64_t extent;
  uint64_t stride;
};

struct PlannedPair {
  llvm::SmallVector<SymbolSlice> lhs, rhs;
  MergeConflict conflict;
};

llvm::SmallVector<SymbolSlice>
wholeSymbols(llvm::ArrayRef<AxisSymbol> symbols) {
  return llvm::map_to_vector(symbols, [](AxisSymbol symbol) {
    return SymbolSlice{symbol, symbol.getExtent(), 1};
  });
}

// The slices of `symbols` that one side of an alignExtents piece covers.
llvm::SmallVector<SymbolSlice>
symbolSlices(llvm::ArrayRef<AxisSymbol> symbols,
             llvm::ArrayRef<SplitExtentSlice> slices) {
  return llvm::map_to_vector(slices, [&](const SplitExtentSlice &slice) {
    return SymbolSlice{symbols[slice.extentIdx], slice.subExtent, slice.stride};
  });
}

// Plans a stretch with no shared symbols and no overlap conflict. A side of
// one symbol pairs with the whole other side, as a merge or a factorization
// onto the other side's symbols. Otherwise the sides split into alignExtents'
// pieces: aligned pieces merge, and indivisible ones, such as [2, 3] against
// [3, 2], are unaligned.
void planGroups(llvm::ArrayRef<AxisSymbol> lhs, llvm::ArrayRef<AxisSymbol> rhs,
                llvm::SmallVector<PlannedPair> &plan) {
  if (lhs.size() == 1 || rhs.size() == 1) {
    plan.push_back({wholeSymbols(lhs), wholeSymbols(rhs), MergeConflict::None});
    return;
  }
  for (const ExtentPiece &piece :
       alignExtents(symbolExtents(lhs), symbolExtents(rhs)))
    plan.push_back(
        {symbolSlices(lhs, piece.lhs), symbolSlices(rhs, piece.rhs),
         piece.aligned() ? MergeConflict::None : MergeConflict::Unaligned});
}

// Plans a stretch with no shared symbols. An overlap between any symbol of
// one side and any of the other conflicts on the whole stretch.
void planSegment(IncrementalAxisConstraints &constraints,
                 llvm::ArrayRef<AxisSymbol> lhs, llvm::ArrayRef<AxisSymbol> rhs,
                 llvm::SmallVector<PlannedPair> &plan) {
  if (hasOverlappingSymbolsConflict(lhs, rhs, constraints.overlapping(lhs),
                                    constraints.overlapping(rhs))) {
    plan.push_back(
        {wholeSymbols(lhs), wholeSymbols(rhs), MergeConflict::Overlap});
    return;
  }
  planGroups(lhs, rhs, plan);
}

// How two resolved strings of equal extent align, major-first, described as
// slices of their symbols so that nothing is split yet. An order conflict
// covers both whole strings.
llvm::SmallVector<PlannedPair>
planAlignment(IncrementalAxisConstraints &constraints,
              llvm::ArrayRef<AxisSymbol> lhs, llvm::ArrayRef<AxisSymbol> rhs) {
  llvm::SmallVector<PlannedPair> orderConflict = {
      {wholeSymbols(lhs), wholeSymbols(rhs), MergeConflict::Order}};
  if (hasSharedFactorOrderConflict(lhs, rhs))
    return orderConflict;

  // A symbol on both sides sits at the same position in each (checked above),
  // so it maps to itself and cuts both strings into segments of equal extent,
  // each planned on its own. The overlap check must not see the shared
  // symbols: a shared symbol overlaps its neighbors on both sides, so in
  // [a, b] ~ [a, c] it would look like a conflict between a and c, although
  // only b and c are being unified.
  auto isShared = [&](AxisSymbol symbol) {
    return llvm::is_contained(lhs, symbol) && llvm::is_contained(rhs, symbol);
  };
  llvm::SmallVector<PlannedPair> plan;
  if (!llvm::any_of(lhs, isShared)) {
    planSegment(constraints, lhs, rhs, plan);
    return plan;
  }
  llvm::SmallVector<AxisSymbol> lhsSegment, rhsSegment;
  auto planPendingSegment = [&]() {
    // An empty segment is extent 1, so the other side holds only extent-1
    // symbols and there is nothing to unify.
    if (!lhsSegment.empty() && !rhsSegment.empty())
      planSegment(constraints, lhsSegment, rhsSegment, plan);
    lhsSegment.clear();
    rhsSegment.clear();
  };
  auto rhsIt = rhs.begin();
  for (AxisSymbol symbol : lhs) {
    if (!isShared(symbol)) {
      lhsSegment.push_back(symbol);
      continue;
    }
    for (; !(*rhsIt == symbol); ++rhsIt) {
      // Positions only order shared symbols of extent > 1; two extent-1
      // shared symbols may still appear in opposite orders.
      if (isShared(*rhsIt))
        return orderConflict;
      rhsSegment.push_back(*rhsIt);
    }
    ++rhsIt;
    planPendingSegment();
    plan.push_back(
        {wholeSymbols(symbol), wholeSymbols(symbol), MergeConflict::None});
  }
  rhsSegment.append(rhsIt, rhs.end());
  planPendingSegment();
  return plan;
}

} // namespace

llvm::SmallVector<IncrementalAxisConstraints::GroupPair>
IncrementalAxisConstraints::align(llvm::ArrayRef<AxisSymbol> a,
                                  llvm::ArrayRef<AxisSymbol> b) {
  llvm::SmallVector<AxisSymbol> lhs = resolve(a);
  llvm::SmallVector<AxisSymbol> rhs = resolve(b);
  assert(extentOfList(lhs) == extentOfList(rhs) &&
         "aligned strings must have equal extent");
  llvm::SmallVector<PlannedPair> plan = planAlignment(*this, lhs, rhs);

  // Split every sliced symbol into its pieces, lhs first and each side in plan
  // order, which is major-first within each symbol. A symbol on both sides is
  // never sliced, so it maps to itself from either side.
  llvm::DenseMap<std::pair<AxisSymbol, uint64_t>, AxisSymbol> pieceAt;
  for (bool isLhs : {true, false}) {
    llvm::MapVector<AxisSymbol, llvm::SmallVector<SymbolSlice>> slicesOf;
    for (const PlannedPair &pair : plan)
      for (const SymbolSlice &slice : isLhs ? pair.lhs : pair.rhs)
        slicesOf[slice.symbol].push_back(slice);
    for (auto &[symbol, slices] : slicesOf) {
      if (slices.size() == 1) {
        assert(slices.front().extent == symbol.getExtent() &&
               "slices of a symbol cover it");
        pieceAt[{symbol, 1}] = symbol;
        continue;
      }
      llvm::SmallVector<AxisSymbol> pieces;
      for (const SymbolSlice &slice : slices) {
        AxisSymbol piece = AxisSymbol::create(slice.extent);
        pieces.push_back(piece);
        pieceAt[{symbol, slice.stride}] = piece;
      }
      factorInto(symbol, pieces);
    }
  }

  auto piecesOf = [&](llvm::ArrayRef<SymbolSlice> slices) {
    return llvm::map_to_vector(slices, [&](const SymbolSlice &slice) {
      return pieceAt.at({slice.symbol, slice.stride});
    });
  };
  llvm::SmallVector<GroupPair> pairs;
  for (const PlannedPair &pair : plan)
    pairs.push_back({piecesOf(pair.lhs), piecesOf(pair.rhs), pair.conflict});
  return pairs;
}

llvm::SmallVector<IncrementalAxisConstraints::ConflictRange>
IncrementalAxisConstraints::conflicts(llvm::ArrayRef<AxisSymbol> a,
                                      llvm::ArrayRef<AxisSymbol> b) {
  llvm::SmallVector<AxisSymbol> lhs = resolve(a);
  llvm::SmallVector<AxisSymbol> rhs = resolve(b);
  assert(extentOfList(lhs) == extentOfList(rhs) &&
         "compared strings must have equal extent");

  llvm::SmallVector<ConflictRange> ranges;
  // The stride of everything minor to the pairs visited so far.
  uint64_t stride = extentOfList(lhs);
  for (const PlannedPair &pair : planAlignment(*this, lhs, rhs)) {
    uint64_t extent = 1;
    for (const SymbolSlice &slice : pair.lhs)
      extent *= slice.extent;
    stride /= extent;
    if (pair.conflict != MergeConflict::None)
      ranges.push_back({stride, extent, pair.conflict});
  }
  return ranges;
}

void IncrementalAxisConstraints::meet(llvm::ArrayRef<AxisSymbol> a,
                                      llvm::ArrayRef<AxisSymbol> b) {
  for (const GroupPair &pair : align(a, b)) {
    if (pair.conflict == MergeConflict::None) {
      mergeAlignedPair(pair);
      continue;
    }
    localize(pair.lhs);
    localize(pair.rhs);
  }
}

LogicalResult
IncrementalAxisConstraints::checkMeetNoLocalize(llvm::ArrayRef<AxisSymbol> a,
                                                llvm::ArrayRef<AxisSymbol> b) {
  return success(conflicts(a, b).empty());
}

void IncrementalAxisConstraints::partialMeetNoLocalize(
    llvm::ArrayRef<AxisSymbol> a, llvm::ArrayRef<AxisSymbol> b) {
  for (const GroupPair &pair : align(a, b))
    if (pair.conflict == MergeConflict::None)
      mergeAlignedPair(pair);
}

LogicalResult
IncrementalAxisConstraints::strictMeetNoLocalize(llvm::ArrayRef<AxisSymbol> a,
                                                 llvm::ArrayRef<AxisSymbol> b) {
  if (failed(checkMeetNoLocalize(a, b)))
    return failure();
  meet(a, b);
  return success();
}

void IncrementalAxisConstraints::mergeAlignedPair(const GroupPair &pair) {
  assert(pair.conflict == MergeConflict::None &&
         "only conflict-free pairs merge");
  if (pair.lhs.size() == 1 && pair.rhs.size() == 1) {
    meetSymbols(pair.lhs.front(), pair.rhs.front());
  } else if (pair.lhs.size() == 1) {
    factorInto(pair.lhs.front(), pair.rhs);
  } else {
    assert(pair.rhs.size() == 1 && "an aligned pair has a one-symbol side");
    factorInto(pair.rhs.front(), pair.lhs);
  }
}

} // namespace mlir::enzyme::axis
