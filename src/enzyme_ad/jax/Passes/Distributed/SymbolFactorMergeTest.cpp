// Unit tests for SymbolFactorMerge's alignment of two symbol lists: the meet,
// the read-only conflict check, the analysis's conflict-free merge, and the
// overlaps of pieces made by splitting a symbol.

#include "src/enzyme_ad/jax/Passes/Distributed/ShardyLogicalAxisAnalysis.h"

#include "gtest/gtest.h"

namespace mlir::enzyme::distributed {
namespace {

using MergeConflict = SymbolFactorMerge::MergeConflict;
using Symbols = llvm::SmallVector<AxisSymbol>;

bool sameRoots(SymbolFactorMerge &merge, llvm::ArrayRef<AxisSymbol> a,
               llvm::ArrayRef<AxisSymbol> b) {
  return merge.resolve(a) == merge.resolve(b);
}

bool overlaps(SymbolFactorMerge &merge, AxisSymbol a, AxisSymbol b) {
  return merge.getOverlapping(a).contains(merge.resolve(b).front());
}

// Symbols that overlap don't merge under the meet: both become L.
TEST(SymbolFactorMergeTest, MeetOverlapBecomesUnshardable) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(4), c = AxisSymbol::create(4);
  merge.markOverlapping({a, c});
  merge.meet({a}, {c});
  EXPECT_FALSE(sameRoots(merge, {a}, {c}));
  EXPECT_TRUE(merge.isUnshardable(a));
  EXPECT_TRUE(merge.isUnshardable(c));
}

// 2*3 against 3*2 has no common refinement: all four symbols become L.
TEST(SymbolFactorMergeTest, MeetUnalignedBecomesUnshardable) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(3), d = AxisSymbol::create(2);
  merge.meet({a, b}, {c, d});
  for (AxisSymbol sym : {a, b, c, d}) {
    EXPECT_TRUE(merge.isUnshardable(sym));
    EXPECT_EQ(merge.resolve(sym), Symbols{sym});
  }
}

// 4*2 against 2*4 refines to 2*2*2 on both sides, and every piece merges.
TEST(SymbolFactorMergeTest, MeetMergesCommonRefinement) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  merge.meet({a, b}, {c, d});
  Symbols roots = merge.resolve({a, b});
  EXPECT_EQ(roots.size(), 3u);
  EXPECT_EQ(roots, merge.resolve({c, d}));
  for (AxisSymbol root : roots)
    EXPECT_FALSE(merge.isUnshardable(root));
}

// conflicts() reports what meet() lowers to L, without splitting anything:
// [4, 2, 3] against [2, 2, 3, 2] aligns 4 with 2*2 and leaves the minor 2*3
// against 3*2 unaligned.
TEST(SymbolFactorMergeTest, ConflictsAgreesWithMeetAndSplitsNothing) {
  SymbolFactorMerge merge;
  AxisSymbol g = AxisSymbol::create(4), a = AxisSymbol::create(2),
             b = AxisSymbol::create(3);
  AxisSymbol h = AxisSymbol::create(2), k = AxisSymbol::create(2),
             c = AxisSymbol::create(3), d = AxisSymbol::create(2);

  auto ranges = merge.conflicts({g, a, b}, {h, k, c, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 6u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Unaligned);
  for (AxisSymbol sym : {g, a, b, h, k, c, d}) {
    EXPECT_EQ(merge.resolve(sym), Symbols{sym});
    EXPECT_FALSE(merge.isUnshardable(sym));
  }

  merge.meet({g, a, b}, {h, k, c, d});
  EXPECT_TRUE(sameRoots(merge, {g}, {h, k}));
  EXPECT_FALSE(merge.isUnshardable(g));
  for (AxisSymbol sym : {a, b, c, d})
    EXPECT_TRUE(merge.isUnshardable(sym));
}

// A shared symbol at different positions on the two sides is an order
// conflict over both whole lists.
TEST(SymbolFactorMergeTest, ConflictsReportsOrder) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(2);
  auto ranges = merge.conflicts({a, b}, {b, a});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 4u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Order);
}

// Extent-1 shared symbols in opposite orders pass the position check but
// still can't pair up: an order conflict over both whole lists, so even the
// segment ahead of them doesn't merge.
TEST(SymbolFactorMergeTest, ExtentOneSharedSymbolsOutOfOrder) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(2), s = AxisSymbol::create(5),
             u = AxisSymbol::create(1), v = AxisSymbol::create(1),
             b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(3);
  auto ranges = merge.conflicts({a, s, u, v, b}, {c, s, v, u, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 30u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Order);

  EXPECT_TRUE(failed(merge.mergeSymbols({a, s, u, v, b}, {c, s, v, u, d})));
  EXPECT_FALSE(sameRoots(merge, {a}, {c}));
}

// The analysis's merge leaves conflicting stretches apart without lowering
// them, and reports that it did.
TEST(SymbolFactorMergeTest, MergeSymbolsSkipsConflictsAndFails) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(3), d = AxisSymbol::create(2);
  EXPECT_TRUE(failed(merge.mergeSymbols({a, b}, {c, d})));
  for (AxisSymbol sym : {a, b, c, d}) {
    EXPECT_FALSE(merge.isUnshardable(sym));
    EXPECT_EQ(merge.resolve(sym), Symbols{sym});
  }

  AxisSymbol e = AxisSymbol::create(4), f = AxisSymbol::create(4);
  EXPECT_TRUE(succeeded(merge.mergeSymbols(e, f)));
  EXPECT_TRUE(sameRoots(merge, {e}, {f}));
}

// An extent-1 symbol merged with a digitless dimension factors into nothing.
TEST(SymbolFactorMergeTest, MergeWithEmptyListDropsExtentOneSymbol) {
  SymbolFactorMerge merge;
  AxisSymbol j = AxisSymbol::create(1);
  EXPECT_TRUE(succeeded(merge.mergeSymbols(Symbols{j}, Symbols{})));
  EXPECT_TRUE(merge.resolve(j).empty());
}

// Shared symbols cut the lists into segments merged on their own: an overlap
// in one segment leaves only that segment apart.
TEST(SymbolFactorMergeTest, MergeSymbolsOverlapRejectsOnlyItsSegment) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(2), s = AxisSymbol::create(3),
             b = AxisSymbol::create(5);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(5);
  merge.markOverlapping({a, c});
  EXPECT_TRUE(failed(merge.mergeSymbols({a, s, b}, {c, s, d})));
  EXPECT_FALSE(sameRoots(merge, {a}, {c}));
  EXPECT_TRUE(sameRoots(merge, {b}, {d}));
}

// Pieces of a split symbol overlap each other, so a later merge can't
// identify two digits of one axis.
TEST(SymbolFactorMergeTest, SplitPiecesOverlapSiblings) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  merge.meet({a, b}, {c, d});
  Symbols pieces = merge.resolve(a);
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_TRUE(overlaps(merge, pieces[0], pieces[1]));
  EXPECT_TRUE(overlaps(merge, pieces[1], pieces[0]));

  // Factoring a symbol onto existing symbols makes those siblings too.
  AxisSymbol x = AxisSymbol::create(6), p = AxisSymbol::create(2),
             q = AxisSymbol::create(3);
  merge.meet({x}, {p, q});
  EXPECT_TRUE(overlaps(merge, p, q));
  EXPECT_TRUE(overlaps(merge, q, p));
}

// A split symbol's pieces inherit its overlaps, readable from either side.
TEST(SymbolFactorMergeTest, SplitOverlapsReadBackSymmetrically) {
  SymbolFactorMerge merge;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  AxisSymbol x = AxisSymbol::create(8);
  merge.markOverlapping({a, x});
  merge.meet({a, b}, {c, d});
  for (AxisSymbol piece : merge.resolve(a)) {
    EXPECT_TRUE(overlaps(merge, piece, x));
    EXPECT_TRUE(overlaps(merge, x, piece));
  }
}

} // namespace
} // namespace mlir::enzyme::distributed
