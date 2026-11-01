// Unit tests for the axis utilities that work on extents and strides alone:
// aligning two factorings of one index space, cutting an axis into atoms, and
// finding a common atom basis for several factors.

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "gtest/gtest.h"

#include <utility>
#include <vector>

namespace mlir::enzyme::axis {
namespace {

using Extents = llvm::SmallVector<uint64_t>;
// (extent, stride) of each atom, major-first.
using AtomList = std::vector<std::pair<uint64_t, uint64_t>>;
// (extentIdx, subExtent, stride) of each slice in one cut.
using SliceList = std::vector<std::tuple<size_t, uint64_t, uint64_t>>;

AtomList atomList(llvm::ArrayRef<AxisAtom> atoms) {
  AtomList list;
  for (const AxisAtom &atom : atoms)
    list.push_back({atom.extent, atom.stride});
  return list;
}

std::vector<SliceList>
sliceLists(llvm::ArrayRef<llvm::SmallVector<SplitExtentSlice>> cuts) {
  std::vector<SliceList> lists;
  for (const auto &cut : cuts) {
    SliceList &list = lists.emplace_back();
    for (const SplitExtentSlice &slice : cut)
      list.push_back({slice.extentIdx, slice.subExtent, slice.stride});
  }
  return lists;
}

//===----------------------------------------------------------------------===//
// computeMappingSplits
//===----------------------------------------------------------------------===//

// Identical factorings split into their own factors.
TEST(ComputeMappingSplitsTest, IdenticalFactorings) {
  EXPECT_EQ(computeMappingSplits(Extents{4, 8}, Extents{4, 8}),
            (Extents{4, 8}));
}

// Compatible borders refine both sides: 4*2 against 2*4 is 2*2*2.
TEST(ComputeMappingSplitsTest, CommonRefinement) {
  EXPECT_EQ(computeMappingSplits(Extents{4, 2}, Extents{2, 4}),
            (Extents{2, 2, 2}));
  EXPECT_EQ(computeMappingSplits(Extents{4, 3}, Extents{2, 6}),
            (Extents{2, 2, 3}));
}

// One factor against several splits at the other side's borders.
TEST(ComputeMappingSplitsTest, OneAgainstMany) {
  EXPECT_EQ(computeMappingSplits(Extents{6}, Extents{2, 3}), (Extents{2, 3}));
}

// Sides with no common refinement stay one indivisible piece, and the rest of
// the lists still splits.
TEST(ComputeMappingSplitsTest, IndivisibleStretch) {
  EXPECT_EQ(computeMappingSplits(Extents{2, 3}, Extents{3, 2}), (Extents{6}));
  EXPECT_EQ(computeMappingSplits(Extents{2, 3, 5}, Extents{3, 2, 5}),
            (Extents{6, 5}));
}

// Empty factorings have nothing to split.
TEST(ComputeMappingSplitsTest, Empty) {
  EXPECT_TRUE(computeMappingSplits(Extents{}, Extents{}).empty());
}

//===----------------------------------------------------------------------===//
// computeSplitExtentSlices
//===----------------------------------------------------------------------===//

// A factor cut in two gives its major piece first, at the minor piece's
// extent as stride.
TEST(ComputeSplitExtentSlicesTest, CutsInsideAFactor) {
  EXPECT_EQ(
      sliceLists(computeSplitExtentSlices(Extents{4, 2}, Extents{2, 2, 2})),
      (std::vector<SliceList>{{{0, 2, 2}}, {{0, 2, 1}}, {{1, 2, 1}}}));
}

// A cut spanning several factors lists each of them whole.
TEST(ComputeSplitExtentSlicesTest, CutSpanningFactors) {
  EXPECT_EQ(sliceLists(computeSplitExtentSlices(Extents{2, 3}, Extents{6})),
            (std::vector<SliceList>{{{0, 2, 1}, {1, 3, 1}}}));
}

//===----------------------------------------------------------------------===//
// applyCutsSorted and getFactorAtomRange
//===----------------------------------------------------------------------===//

// Cuts may be unsorted, repeated or out of range; atoms come out major-first.
TEST(ApplyCutsSortedTest, AtomsMajorFirst) {
  auto atoms = applyCutsSorted(12, {6, 2, 6, 1, 12, 24});
  ASSERT_TRUE(succeeded(atoms));
  EXPECT_EQ(atomList(*atoms), (AtomList{{2, 6}, {3, 2}, {2, 1}}));
}

// An uncut axis is a single atom spanning it.
TEST(ApplyCutsSortedTest, NoCutsIsOneAtom) {
  auto atoms = applyCutsSorted(8, {});
  ASSERT_TRUE(succeeded(atoms));
  EXPECT_EQ(atomList(*atoms), (AtomList{{8, 1}}));
}

// An extent-1 axis still yields one trivial atom.
TEST(ApplyCutsSortedTest, ExtentOneAxis) {
  auto atoms = applyCutsSorted(1, {});
  ASSERT_TRUE(succeeded(atoms));
  EXPECT_EQ(atomList(*atoms), (AtomList{{1, 1}}));
}

// Cuts at 4 and 6 in an axis of 12 are not a divisibility chain.
TEST(ApplyCutsSortedTest, NonChainFails) {
  EXPECT_TRUE(failed(applyCutsSorted(12, {4, 6})));
}

// A factor covers the run of atoms between its two cuts.
TEST(GetFactorAtomRangeTest, RunBetweenCuts) {
  auto atoms = applyCutsSorted(12, {2, 6});
  ASSERT_TRUE(succeeded(atoms));
  // extent 6 at stride 2: the atoms of 2 at stride 6 and 3 at stride 2.
  EXPECT_EQ(getFactorAtomRange(*atoms, 6, 2),
            std::make_pair(size_t(0), size_t(2)));
  EXPECT_EQ(getFactorAtomRange(*atoms, 2, 1),
            std::make_pair(size_t(2), size_t(1)));
  EXPECT_EQ(getFactorAtomRange(*atoms, 1, 1),
            std::make_pair(size_t(0), size_t(0)));
}

//===----------------------------------------------------------------------===//
// computeCommonAtoms
//===----------------------------------------------------------------------===//

// Each axis is cut at the borders of every factor over it; extent-1 factors
// add no cuts.
TEST(ComputeCommonAtomsTest, NestingFactors) {
  auto common =
      computeCommonAtoms({12, 4}, {{0, 6, 2}, {0, 2, 1}, {1, 4, 1}, {1, 1, 4}});
  ASSERT_TRUE(succeeded(common));
  EXPECT_EQ(atomList(common->atomsOf(0)), (AtomList{{6, 2}, {2, 1}}));
  EXPECT_EQ(atomList(common->atomsOf(1)), (AtomList{{4, 1}}));
  EXPECT_EQ(common->rangeOf({0, 6, 2}), std::make_pair(size_t(0), size_t(1)));
}

// Factors whose borders don't nest (3 at stride 2 and 4 at stride 1, in 12)
// have no common basis.
TEST(ComputeCommonAtomsTest, NonNestingFails) {
  EXPECT_TRUE(failed(computeCommonAtoms({12}, {{0, 3, 2}, {0, 4, 1}})));
}

//===----------------------------------------------------------------------===//
// computeCommonAtomsAndMappingSplits
//===----------------------------------------------------------------------===//

// A pair mapping a whole axis onto a 2*2 axis cuts the whole one to match.
TEST(ComputeCommonAtomsAndMappingSplitsTest, PairForcesCut) {
  AtomFactorPair pair{{{0, 4, 1}}, {{1, 2, 2}, {1, 2, 1}}};
  auto common = computeCommonAtomsAndMappingSplits({4, 4}, {pair});
  ASSERT_TRUE(succeeded(common));
  EXPECT_EQ(atomList(common->atomsOf(0)), (AtomList{{2, 2}, {2, 1}}));
  EXPECT_EQ(atomList(common->atomsOf(1)), (AtomList{{2, 2}, {2, 1}}));
}

// A cut forced by one pair reaches an axis only another pair touches.
TEST(ComputeCommonAtomsAndMappingSplitsTest, CutPropagatesAcrossPairs) {
  AtomFactorPair first{{{0, 4, 1}}, {{1, 2, 2}, {1, 2, 1}}};
  AtomFactorPair second{{{2, 4, 1}}, {{0, 4, 1}}};
  auto common = computeCommonAtomsAndMappingSplits({4, 4, 4}, {first, second});
  ASSERT_TRUE(succeeded(common));
  EXPECT_EQ(atomList(common->atomsOf(2)), (AtomList{{2, 2}, {2, 1}}));
}

// 2*3 against 3*2 has no common basis.
TEST(ComputeCommonAtomsAndMappingSplitsTest, IndivisiblePairFails) {
  AtomFactorPair pair{{{0, 2, 3}, {0, 3, 1}}, {{1, 3, 2}, {1, 2, 1}}};
  EXPECT_TRUE(failed(computeCommonAtomsAndMappingSplits({6, 6}, {pair})));
}

} // namespace
} // namespace mlir::enzyme::axis
