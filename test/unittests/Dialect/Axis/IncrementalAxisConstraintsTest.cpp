// Unit tests for IncrementalAxisConstraints: the string meets and their
// no-localize variants, the read-only conflict check, and the digit-level
// factor, merge and overlap operations.

#include "src/enzyme_ad/jax/Dialect/Axis/IncrementalAxisConstraints.h"

#include "gtest/gtest.h"

namespace mlir::enzyme::axis {
namespace {

using MergeConflict = IncrementalAxisConstraints::MergeConflict;
using Symbols = llvm::SmallVector<AxisSymbol>;

bool sameRoots(IncrementalAxisConstraints &constraints,
               llvm::ArrayRef<AxisSymbol> a, llvm::ArrayRef<AxisSymbol> b) {
  return constraints.resolve(a) == constraints.resolve(b);
}

bool overlaps(IncrementalAxisConstraints &constraints, AxisSymbol a,
              AxisSymbol b) {
  return constraints.overlapping(a).contains(constraints.resolve(b).front());
}

// Symbols that overlap don't merge under the meet: both become L.
TEST(IncrementalAxisConstraintsTest, MeetOverlapLocalizes) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), c = AxisSymbol::create(4);
  constraints.overlap({a, c});
  constraints.meet({a}, {c});
  EXPECT_FALSE(sameRoots(constraints, {a}, {c}));
  EXPECT_TRUE(constraints.isLocal(a));
  EXPECT_TRUE(constraints.isLocal(c));
}

// 2*3 against 3*2 has no common refinement: all four symbols become L.
TEST(IncrementalAxisConstraintsTest, MeetUnalignedLocalizes) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(3), d = AxisSymbol::create(2);
  constraints.meet({a, b}, {c, d});
  for (AxisSymbol sym : {a, b, c, d}) {
    EXPECT_TRUE(constraints.isLocal(sym));
    EXPECT_EQ(constraints.resolve(sym), Symbols{sym});
  }
}

// 4*2 against 2*4 refines to 2*2*2 on both sides, and every piece merges.
TEST(IncrementalAxisConstraintsTest, MeetMergesCommonRefinement) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  constraints.meet({a, b}, {c, d});
  Symbols roots = constraints.resolve({a, b});
  EXPECT_EQ(roots.size(), 3u);
  EXPECT_EQ(roots, constraints.resolve({c, d}));
  for (AxisSymbol root : roots)
    EXPECT_FALSE(constraints.isLocal(root));
}

// conflicts() reports what meet() lowers to L, without splitting anything:
// [4, 2, 3] against [2, 2, 3, 2] aligns 4 with 2*2 and leaves the minor 2*3
// against 3*2 unaligned.
TEST(IncrementalAxisConstraintsTest, ConflictsAgreesWithMeetAndSplitsNothing) {
  IncrementalAxisConstraints constraints;
  AxisSymbol g = AxisSymbol::create(4), a = AxisSymbol::create(2),
             b = AxisSymbol::create(3);
  AxisSymbol h = AxisSymbol::create(2), k = AxisSymbol::create(2),
             c = AxisSymbol::create(3), d = AxisSymbol::create(2);

  auto ranges = constraints.conflicts({g, a, b}, {h, k, c, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 6u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Unaligned);
  for (AxisSymbol sym : {g, a, b, h, k, c, d}) {
    EXPECT_EQ(constraints.resolve(sym), Symbols{sym});
    EXPECT_FALSE(constraints.isLocal(sym));
  }

  constraints.meet({g, a, b}, {h, k, c, d});
  EXPECT_TRUE(sameRoots(constraints, {g}, {h, k}));
  EXPECT_FALSE(constraints.isLocal(g));
  for (AxisSymbol sym : {a, b, c, d})
    EXPECT_TRUE(constraints.isLocal(sym));
}

// A shared symbol at different positions on the two sides is an order
// conflict over both whole strings.
TEST(IncrementalAxisConstraintsTest, ConflictsReportsOrder) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(2);
  auto ranges = constraints.conflicts({a, b}, {b, a});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 4u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Order);
}

// Extent-1 shared symbols in opposite orders pass the position check but
// still can't pair up: an order conflict over both whole strings, so even the
// segment ahead of them doesn't merge.
TEST(IncrementalAxisConstraintsTest, ExtentOneSharedSymbolsOutOfOrder) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), s = AxisSymbol::create(5),
             u = AxisSymbol::create(1), v = AxisSymbol::create(1),
             b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(3);
  auto ranges = constraints.conflicts({a, s, u, v, b}, {c, s, v, u, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 30u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Order);

  constraints.partialMeetNoLocalize({a, s, u, v, b}, {c, s, v, u, d});
  EXPECT_FALSE(sameRoots(constraints, {a}, {c}));
}

// The partial meet leaves conflicting stretches apart without localizing
// them, and merges the conflict-free ones.
TEST(IncrementalAxisConstraintsTest, PartialMeetSkipsConflicts) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(3), d = AxisSymbol::create(2);
  constraints.partialMeetNoLocalize({a, b}, {c, d});
  for (AxisSymbol sym : {a, b, c, d}) {
    EXPECT_FALSE(constraints.isLocal(sym));
    EXPECT_EQ(constraints.resolve(sym), Symbols{sym});
  }

  AxisSymbol e = AxisSymbol::create(4), f = AxisSymbol::create(4);
  constraints.partialMeetNoLocalize({e}, {f});
  EXPECT_TRUE(sameRoots(constraints, {e}, {f}));
}

// An extent-1 symbol merged with a digitless dimension factors into nothing.
TEST(IncrementalAxisConstraintsTest,
     PartialMeetWithEmptyStringDropsExtentOneSymbol) {
  IncrementalAxisConstraints constraints;
  AxisSymbol j = AxisSymbol::create(1);
  constraints.partialMeetNoLocalize({j}, {});
  EXPECT_TRUE(constraints.resolve(j).empty());
}

// Shared symbols cut the strings into segments merged on their own: an overlap
// in one segment leaves only that segment apart.
TEST(IncrementalAxisConstraintsTest, PartialMeetOverlapRejectsOnlyItsSegment) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), s = AxisSymbol::create(3),
             b = AxisSymbol::create(5);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(5);
  constraints.overlap({a, c});
  constraints.partialMeetNoLocalize({a, s, b}, {c, s, d});
  EXPECT_FALSE(sameRoots(constraints, {a}, {c}));
  EXPECT_TRUE(sameRoots(constraints, {b}, {d}));
}

// An overlap between symbols in different aligned pieces is no conflict: with
// a overlapping d, the pieces a~c and b~d are each clean and both merge.
TEST(IncrementalAxisConstraintsTest, OverlapOutsideAlignedPiecesMerges) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(3);
  constraints.overlap({a, d});
  EXPECT_TRUE(constraints.conflicts({a, b}, {c, d}).empty());
  constraints.partialMeetNoLocalize({a, b}, {c, d});
  EXPECT_TRUE(sameRoots(constraints, {a}, {c}));
  EXPECT_TRUE(sameRoots(constraints, {b}, {d}));
}

// A single symbol against an overlapping counterpart is split so that the
// clean piece still merges: with a overlapping d, a's major piece merges with
// c and only the minor stretch, against d, conflicts.
TEST(IncrementalAxisConstraintsTest, OverlapSplitsOffCleanPieces) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(6);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(3);
  constraints.overlap({a, d});
  auto ranges = constraints.conflicts({a}, {c, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 3u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Overlap);

  constraints.partialMeetNoLocalize({a}, {c, d});
  Symbols pieces = constraints.resolve(a);
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0], constraints.resolve(c).front());
  EXPECT_FALSE(pieces[1] == constraints.resolve(d).front());
  for (AxisSymbol sym : {a, c, d})
    EXPECT_FALSE(constraints.isLocal(sym));
}

// Adjacent conflicting pieces form one stretch: a symbol whose every piece
// conflicts stays whole.
TEST(IncrementalAxisConstraintsTest, ConflictingPiecesCoalesce) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(16);
  AxisSymbol c = AxisSymbol::create(4), d = AxisSymbol::create(4);
  constraints.overlap({a, c, d});
  auto ranges = constraints.conflicts({a}, {c, d});
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].stride, 1u);
  EXPECT_EQ(ranges[0].extent, 16u);
  EXPECT_EQ(ranges[0].conflict, MergeConflict::Overlap);

  constraints.partialMeetNoLocalize({a}, {c, d});
  EXPECT_EQ(constraints.resolve(a), Symbols{a});
}

// Pieces of a split symbol overlap each other, so a later merge can't
// identify two digits of one axis.
TEST(IncrementalAxisConstraintsTest, SplitPiecesOverlapSiblings) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  constraints.meet({a, b}, {c, d});
  Symbols pieces = constraints.resolve(a);
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_TRUE(overlaps(constraints, pieces[0], pieces[1]));
  EXPECT_TRUE(overlaps(constraints, pieces[1], pieces[0]));

  // Factoring a symbol onto existing symbols makes those siblings too.
  AxisSymbol x = AxisSymbol::create(6), p = AxisSymbol::create(2),
             q = AxisSymbol::create(3);
  constraints.meet({x}, {p, q});
  EXPECT_TRUE(overlaps(constraints, p, q));
  EXPECT_TRUE(overlaps(constraints, q, p));
}

// A split symbol's pieces inherit its overlaps, readable from either side.
TEST(IncrementalAxisConstraintsTest, SplitOverlapsReadBackSymmetrically) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  AxisSymbol x = AxisSymbol::create(8);
  constraints.overlap({a, x});
  constraints.meet({a, b}, {c, d});
  for (AxisSymbol piece : constraints.resolve(a)) {
    EXPECT_TRUE(overlaps(constraints, piece, x));
    EXPECT_TRUE(overlaps(constraints, x, piece));
  }
}

// checkMeetNoLocalize succeeds exactly when conflicts() is empty, and splits
// nothing either way.
TEST(IncrementalAxisConstraintsTest, CheckMeetNoLocalizeAgreesWithConflicts) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  EXPECT_TRUE(succeeded(constraints.checkMeetNoLocalize({a, b}, {c, d})));

  AxisSymbol e = AxisSymbol::create(2), f = AxisSymbol::create(3);
  AxisSymbol g = AxisSymbol::create(3), h = AxisSymbol::create(2);
  EXPECT_TRUE(failed(constraints.checkMeetNoLocalize({e, f}, {g, h})));

  for (AxisSymbol sym : {a, b, c, d, e, f, g, h}) {
    EXPECT_EQ(constraints.resolve(sym), Symbols{sym});
    EXPECT_FALSE(constraints.isLocal(sym));
  }
}

// On a conflict the strict meet changes nothing, not even the conflict-free
// stretch that the partial meet would merge.
TEST(IncrementalAxisConstraintsTest, StrictMeetChangesNothingOnConflict) {
  IncrementalAxisConstraints constraints;
  AxisSymbol g = AxisSymbol::create(4), a = AxisSymbol::create(2),
             b = AxisSymbol::create(3);
  AxisSymbol h = AxisSymbol::create(2), k = AxisSymbol::create(2),
             c = AxisSymbol::create(3), d = AxisSymbol::create(2);
  EXPECT_TRUE(
      failed(constraints.strictMeetNoLocalize({g, a, b}, {h, k, c, d})));
  for (AxisSymbol sym : {g, a, b, h, k, c, d}) {
    EXPECT_EQ(constraints.resolve(sym), Symbols{sym});
    EXPECT_FALSE(constraints.isLocal(sym));
  }
}

// Without conflicts the strict meet is the meet: 4*2 against 2*4 refines to
// 2*2*2 on both sides and every piece merges.
TEST(IncrementalAxisConstraintsTest, StrictMeetMergesWithoutConflicts) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(2);
  AxisSymbol c = AxisSymbol::create(2), d = AxisSymbol::create(4);
  EXPECT_TRUE(succeeded(constraints.strictMeetNoLocalize({a, b}, {c, d})));
  Symbols roots = constraints.resolve({a, b});
  EXPECT_EQ(roots.size(), 3u);
  EXPECT_EQ(roots, constraints.resolve({c, d}));
  for (AxisSymbol root : roots)
    EXPECT_FALSE(constraints.isLocal(root));
}

// factor() splits a symbol into pieces of the given extents, which inherit
// its overlaps and overlap each other.
TEST(IncrementalAxisConstraintsTest,
     FactorPiecesInheritOverlapsAndOverlapEachOther) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(6), x = AxisSymbol::create(6);
  constraints.overlap({a, x});
  Symbols pieces = constraints.factor(a, {2, 3});
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].getExtent(), 2u);
  EXPECT_EQ(pieces[1].getExtent(), 3u);
  EXPECT_EQ(constraints.resolve(a), pieces);
  EXPECT_TRUE(overlaps(constraints, pieces[0], pieces[1]));
  EXPECT_TRUE(overlaps(constraints, pieces[1], pieces[0]));
  for (AxisSymbol piece : pieces) {
    EXPECT_TRUE(overlaps(constraints, piece, x));
    EXPECT_TRUE(overlaps(constraints, x, piece));
  }
}

// Pieces of a local symbol are local.
TEST(IncrementalAxisConstraintsTest, FactorOfLocalSymbolStaysLocal) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4);
  constraints.localize({a});
  for (AxisSymbol piece : constraints.factor(a, {2, 2}))
    EXPECT_TRUE(constraints.isLocal(piece));
}

// meetSymbols() makes one class that keeps both sides' overlaps and is local
// if either side was.
TEST(IncrementalAxisConstraintsTest, MeetSymbolsUnionsOverlapsAndLocality) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(4), b = AxisSymbol::create(4),
             x = AxisSymbol::create(4);
  constraints.overlap({a, x});
  constraints.localize({b});
  constraints.meetSymbols(a, b);
  EXPECT_TRUE(sameRoots(constraints, {a}, {b}));
  EXPECT_TRUE(overlaps(constraints, b, x));
  EXPECT_TRUE(overlaps(constraints, x, b));
  EXPECT_TRUE(constraints.isLocal(a));
}

// overlap() relates every pair of its symbols, and no symbol to itself.
TEST(IncrementalAxisConstraintsTest, OverlapIsPairwiseAndIrreflexive) {
  IncrementalAxisConstraints constraints;
  AxisSymbol a = AxisSymbol::create(2), b = AxisSymbol::create(3),
             c = AxisSymbol::create(5);
  constraints.overlap({a, b, c});
  IncrementalAxisConstraints::OverlapSet expected = {b, c};
  EXPECT_EQ(constraints.overlapping(a), expected);
}

} // namespace
} // namespace mlir::enzyme::axis
