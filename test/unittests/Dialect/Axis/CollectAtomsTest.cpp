// Unit tests for collectAtoms and the atom-set operations: comparing factor
// values by the digits they cover, across the roots in scope.

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

namespace mlir::enzyme::axis {
namespace {

using Factor = TypedValue<AxisFactorType>;

class CollectAtomsTest : public ::testing::Test {
protected:
  CollectAtomsTest() : builder(&context) {
    context.loadDialect<AxisDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
  }

  // A root: axis 0 of a 1-D shape of the given extent, from a new
  // axis.getaxis op. Shape axes are equivalent by type, so two calls with the
  // same extent give distinct ops that are equivalent roots; calls with
  // different extents give unrelated roots.
  Value root(int64_t extent) {
    return createAxesForRankedShape(
        RankedTensorType::get({extent}, builder.getF32Type()), builder,
        builder.getUnknownLoc())[0];
  }

  Factor factor(Value axis, int64_t extent, int64_t stride) {
    return builder.create<AxisFactorOp>(builder.getUnknownLoc(), axis, extent,
                                        stride);
  }

  // The atom set of each factor.
  SmallVector<AtomSet> atomsOf(ArrayRef<Value> factors) {
    SmallVector<ValueRange> lists;
    for (const Value &factor : factors)
      lists.push_back(factor);
    FailureOr<CollectedAtoms> atoms = collectAtoms(lists);
    EXPECT_TRUE(succeeded(atoms));
    return succeeded(atoms) ? atoms->sets
                            : SmallVector<AtomSet>(factors.size());
  }

  MLIRContext context;
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
};

// Two axis.factor ops of one root with the same extent and stride are one
// digit, also when the roots are distinct but equivalent ops.
TEST_F(CollectAtomsTest, IdenticalFactorsAreOneAtomSet) {
  Value axis = root(12);
  Value equivalentAxis = root(12);
  SmallVector<AtomSet> atoms = atomsOf(
      {factor(axis, 3, 4), factor(axis, 3, 4), factor(equivalentAxis, 3, 4)});
  EXPECT_FALSE(atoms[0].empty());
  EXPECT_EQ(atoms[0], atoms[1]);
  EXPECT_EQ(atoms[0], atoms[2]);
}

// A whole axis contains each of its factors, and minus one factor leaves the
// others.
TEST_F(CollectAtomsTest, WholeAxisContainsItsFactors) {
  Value axis = root(12);
  SmallVector<AtomSet> atoms =
      atomsOf({factor(axis, 12, 1), factor(axis, 3, 4), factor(axis, 4, 1)});
  const AtomSet &whole = atoms[0], &major = atoms[1], &minor = atoms[2];
  EXPECT_TRUE(atomSetContains(whole, major));
  EXPECT_TRUE(atomSetContains(whole, minor));
  EXPECT_FALSE(atomSetContains(major, whole));
  EXPECT_TRUE(areAtomSetsDisjoint(major, minor));
  EXPECT_EQ(atomSetMinus(whole, major), minor);
}

// Factors that share a digit are not disjoint: on an axis of 12, (6, stride 2)
// and (4, stride 1) share the digit of extent 2 at stride 2.
TEST_F(CollectAtomsTest, OverlappingFactorsAreNotDisjoint) {
  Value axis = root(12);
  SmallVector<AtomSet> atoms =
      atomsOf({factor(axis, 6, 2), factor(axis, 4, 1)});
  EXPECT_FALSE(areAtomSetsDisjoint(atoms[0], atoms[1]));
  EXPECT_FALSE(atomSetContains(atoms[0], atoms[1]));
}

// Factors of different roots never share a digit.
TEST_F(CollectAtomsTest, DifferentRootsAreDisjoint) {
  SmallVector<AtomSet> atoms =
      atomsOf({factor(root(4), 4, 1), factor(root(8), 8, 1)});
  EXPECT_TRUE(areAtomSetsDisjoint(atoms[0], atoms[1]));
}

// A list's set is the union of its factors' digits.
TEST_F(CollectAtomsTest, ListIsUnionOfItsFactors) {
  Value axis = root(12);
  Factor major = factor(axis, 3, 4), minor = factor(axis, 4, 1);
  SmallVector<Value> both = {major, minor};
  FailureOr<CollectedAtoms> atoms =
      collectAtoms({ValueRange(both), ValueRange(factor(axis, 12, 1))});
  ASSERT_TRUE(succeeded(atoms));
  EXPECT_EQ(atoms->sets[0], atoms->sets[1]);
}

// Each atom has its extent, and a factor's atoms are listed major-first: on
// an axis of 12 cut at (3, stride 4) and (2, stride 2), the atoms are 3, 2, 2
// from the major end, and a second root's atoms follow the first's.
TEST_F(CollectAtomsTest, AtomExtentsMajorFirst) {
  Value axis = root(12);
  Factor major = factor(axis, 3, 4), middle = factor(axis, 2, 2);
  Factor whole = factor(axis, 12, 1), other = factor(root(5), 5, 1);
  FailureOr<CollectedAtoms> atoms =
      collectAtoms({ValueRange(major), ValueRange(middle), ValueRange(whole),
                    ValueRange(other)});
  ASSERT_TRUE(succeeded(atoms));
  EXPECT_EQ(atoms->extents, (SmallVector<uint64_t>{3, 2, 2, 5}));
  EXPECT_EQ(atoms->sets[0], (AtomSet{0}));
  EXPECT_EQ(atoms->sets[1], (AtomSet{1}));
  EXPECT_EQ(atoms->sets[2], (AtomSet{0, 1, 2}));
  EXPECT_EQ(atoms->sets[3], (AtomSet{3}));
}

// Cuts that don't nest (2 and 3 on an axis of 6) have no common atoms.
TEST_F(CollectAtomsTest, NonNestingCutsFail) {
  Value axis = root(6);
  SmallVector<Value> factors = {factor(axis, 2, 1), factor(axis, 3, 1)};
  EXPECT_TRUE(failed(collectAtoms({ValueRange(factors)})));
}

} // namespace
} // namespace mlir::enzyme::axis
