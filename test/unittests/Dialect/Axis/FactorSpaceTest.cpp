// Unit tests for coalesceFactorSpace and unionFactorSpaces: the fewest factors
// covering a set of digits, independent of the order the factors come in.

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

#include <utility>
#include <vector>

namespace mlir::enzyme::axis {
namespace {

using Factor = TypedValue<AxisFactorType>;
// (extent, stride) of each factor, in result order.
using FactorList = std::vector<std::pair<int64_t, int64_t>>;

class FactorSpaceTest : public ::testing::Test {
protected:
  FactorSpaceTest() : builder(&context) {
    context.loadDialect<AxisDialect>();
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());
  }

  // A root: axis 0 of a 1-D shape of the given extent. Shape axes are
  // equivalent by type, so two calls with the same extent give equivalent
  // roots; calls with different extents give unrelated roots.
  Value root(int64_t extent) {
    return createAxesForRankedShape(
        RankedTensorType::get({extent}, builder.getF32Type()), builder,
        builder.getUnknownLoc())[0];
  }

  Factor factor(Value axis, int64_t extent, int64_t stride) {
    return builder.create<AxisFactorOp>(builder.getUnknownLoc(), axis, extent,
                                        stride);
  }

  SmallVector<Factor> coalesce(ArrayRef<Factor> space) {
    return coalesceFactorSpace(space, builder, builder.getUnknownLoc());
  }

  SmallVector<Factor>
  unite(ArrayRef<TypedValueArrayRef<AxisFactorType>> spaces) {
    return unionFactorSpaces(spaces, builder, builder.getUnknownLoc());
  }

  static FactorList factorList(ArrayRef<Factor> factors) {
    FactorList list;
    for (Factor factor : factors)
      list.push_back({getFactorExtent(factor), getFactorStride(factor)});
    return list;
  }

  MLIRContext context;
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
};

// Touching factors of one root become the one factor spanning them.
TEST_F(FactorSpaceTest, TouchingFactorsMerge) {
  Value axis = root(24);
  EXPECT_EQ(factorList(coalesce(
                {factor(axis, 2, 1), factor(axis, 3, 2), factor(axis, 4, 6)})),
            (FactorList{{24, 1}}));
}

// Factors with a gap between them stay apart, major-first.
TEST_F(FactorSpaceTest, GapIsKept) {
  Value axis = root(24);
  EXPECT_EQ(factorList(coalesce({factor(axis, 2, 1), factor(axis, 4, 6)})),
            (FactorList{{4, 6}, {2, 1}}));
}

// The result doesn't depend on the order the factors come in.
TEST_F(FactorSpaceTest, OrderDoesNotMatter) {
  Value axis = root(48);
  Factor low = factor(axis, 2, 1), middle = factor(axis, 3, 2),
         high = factor(axis, 2, 24);
  FactorList expected{{2, 24}, {6, 1}};
  EXPECT_EQ(factorList(coalesce({low, middle, high})), expected);
  EXPECT_EQ(factorList(coalesce({high, low, middle})), expected);
  EXPECT_EQ(factorList(coalesce({middle, high, low})), expected);
}

// A run that is a single given factor is returned as that same factor value.
TEST_F(FactorSpaceTest, LoneFactorIsReused) {
  Value axis = root(24);
  Factor lone = factor(axis, 4, 6);
  SmallVector<Factor> result = coalesce({factor(axis, 2, 1), lone});
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], lone);
}

// Factors of different roots never merge; roots keep their first appearance
// order. Factors of equivalent roots are factors of one root.
TEST_F(FactorSpaceTest, RootsByEquivalence) {
  Value first = root(4), second = root(6), firstAgain = root(4);
  SmallVector<Factor> result = coalesce(
      {factor(first, 2, 1), factor(second, 6, 1), factor(firstAgain, 2, 2)});
  EXPECT_EQ(factorList(result), (FactorList{{4, 1}, {6, 1}}));
  ASSERT_EQ(result.size(), 2u);
  EXPECT_TRUE(areAxesEquivalent(*getFactorProvenanceAxis(result[0]),
                                cast<TypedValue<AxisTypeInterface>>(first)));
}

// Extent-1 factors hold no digit and are left out.
TEST_F(FactorSpaceTest, ExtentOneIsDropped) {
  Value axis = root(8);
  EXPECT_EQ(factorList(coalesce({factor(axis, 1, 4), factor(axis, 2, 1)})),
            (FactorList{{2, 1}}));
}

// Spaces sharing digits unite into the fewest factors over all of them:
// [1, 4), [2, 8) and [8, 24) on a root of 24 are the whole root.
TEST_F(FactorSpaceTest, UnionMergesOverlapAndTouch) {
  Value axis = root(24);
  SmallVector<Factor> lhs{factor(axis, 4, 1)};
  SmallVector<Factor> rhs{factor(axis, 4, 2), factor(axis, 3, 8)};
  EXPECT_EQ(factorList(unite({lhs, rhs})), (FactorList{{24, 1}}));
}

// A factor already covering the others' digits is the union.
TEST_F(FactorSpaceTest, UnionReusesCoveringFactor) {
  Value axis = root(12);
  Factor whole = factor(axis, 12, 1);
  SmallVector<Factor> lhs{factor(axis, 3, 4)}, rhs{whole},
      repeated{factor(axis, 3, 4)};
  SmallVector<Factor> result = unite({lhs, rhs, repeated});
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], whole);
}

// Of several factors spanning a run, a placed one wins over a detached one,
// whichever comes first.
TEST_F(FactorSpaceTest, UnionPrefersPlacedFactor) {
  Value axis = root(12);
  Factor detached = factor(axis, 3, 4), placed = factor(axis, 3, 4);
  detached.getDefiningOp()->remove();
  SmallVector<Factor> lhs{detached}, rhs{placed};
  SmallVector<Factor> result = unite({lhs, rhs});
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], placed);
  detached.getDefiningOp()->erase();
}

// The same digit from several spaces appears once.
TEST_F(FactorSpaceTest, UnionOfSameDigit) {
  Value axis = root(12);
  SmallVector<Factor> lhs{factor(axis, 3, 4)}, rhs{factor(axis, 3, 4)};
  EXPECT_EQ(factorList(unite({lhs, rhs})), (FactorList{{3, 4}}));
}

} // namespace
} // namespace mlir::enzyme::axis
