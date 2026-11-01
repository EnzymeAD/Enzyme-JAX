// Unit tests for refineAxes: splitting a grammar description's operator axes
// at digit boundaries given per tensor dimension.

#include "src/enzyme_ad/jax/Dialect/Distributed/GrammarRegistry.h"

#include "gtest/gtest.h"

namespace mlir::enzyme::distributed {
namespace {

using TensorDigits = SmallVector<SmallVector<int64_t>>;

class RefineAxesTest : public ::testing::Test {
protected:
  RefineAxesTest() { context.loadDialect<DistributedDialect>(); }

  OperatorAxesAttr axes(ArrayRef<int64_t> extents,
                        ArrayRef<TensorDigits> operands,
                        ArrayRef<TensorDigits> results) {
    return OperatorAxesAttr::get(&context, extents, operands, results);
  }

  GrammarDescription description(OperatorAxesAttr axes,
                                 SmallVector<AxisSignature> signature) {
    return GrammarDescription{StringAttr::get(&context, "test"), axes,
                              /*params=*/nullptr,
                              BaseSignature{std::move(signature)}};
  }

  MLIRContext context;
};

// Cuts at existing axis boundaries leave the description unchanged.
TEST_F(RefineAxesTest, CutsAtAxisBoundariesChangeNothing) {
  // ([i, j])->([i, j]) {i=4, j=8}
  OperatorAxesAttr original = axes({4, 8}, {{{0, 1}}}, {{{0, 1}}});
  FailureOr<RefinedAxes> refined =
      refineAxes(description(original, {AxisSignature::parallel(),
                                        AxisSignature::local()}),
                 {{{4, 8}}}, {{{32}}});
  ASSERT_TRUE(succeeded(refined));
  EXPECT_EQ(refined->description.axes, original);
  EXPECT_EQ(refined->split, (SmallVector<SmallVector<int64_t>>{{0}, {1}}));
}

// An axis cut on one dimension is split on every dimension it occurs on, and
// each piece keeps the axis's signature.
TEST_F(RefineAxesTest, SplitAxisSplitsEverywhere) {
  // ([i], [i])->([i]) {i=8}, cut 2·4 on the first operand only.
  FailureOr<RefinedAxes> refined =
      refineAxes(description(axes({8}, {{{0}}, {{0}}}, {{{0}}}),
                             {AxisSignature::custom()}),
                 {{{2, 4}}, {{8}}}, {{{8}}});
  ASSERT_TRUE(succeeded(refined));
  // ([i, j], [i, j])->([i, j]) {i=2, j=4}
  EXPECT_EQ(refined->description.axes,
            axes({2, 4}, {{{0, 1}}, {{0, 1}}}, {{{0, 1}}}));
  EXPECT_EQ(refined->split, (SmallVector<SmallVector<int64_t>>{{0, 1}}));
  EXPECT_EQ(refined->description.signature.axes,
            (SmallVector<AxisSignature>{AxisSignature::custom(),
                                        AxisSignature::custom()}));
}

// A cut inside the major axis of a dimension splits only that axis; pieces are
// numbered in place, so later axes shift.
TEST_F(RefineAxesTest, CutInsideMajorAxis) {
  // ([i, j])->([j, i]) {i=4, j=8}, the operand cut 2·16.
  FailureOr<RefinedAxes> refined = refineAxes(
      description(axes({4, 8}, {{{0, 1}}}, {{{1}, {0}}}),
                  {AxisSignature::local(), AxisSignature::parallel()}),
      {{{2, 16}}}, {{{8}, {4}}});
  ASSERT_TRUE(succeeded(refined));
  // ([i, j, k])->([k], [i, j]) {i=2, j=2, k=8}
  EXPECT_EQ(refined->description.axes,
            axes({2, 2, 8}, {{{0, 1, 2}}}, {{{2}, {0, 1}}}));
  EXPECT_EQ(refined->split, (SmallVector<SmallVector<int64_t>>{{0, 1}, {2}}));
  EXPECT_EQ(refined->description.signature.axes,
            (SmallVector<AxisSignature>{AxisSignature::local(),
                                        AxisSignature::local(),
                                        AxisSignature::parallel()}));
}

// Cuts from two dimensions at 4 and at 6 within an axis of 12 have no common
// refinement.
TEST_F(RefineAxesTest, NonNestedCutsFail) {
  // ([i])->([i]) {i=12}
  FailureOr<RefinedAxes> refined = refineAxes(
      description(axes({12}, {{{0}}}, {{{0}}}), {AxisSignature::parallel()}),
      {{{3, 4}}}, {{{2, 6}}});
  EXPECT_TRUE(failed(refined));
}

// A cut that doesn't divide the axis it falls in fails.
TEST_F(RefineAxesTest, CutNotDividingAxisFails) {
  // ([i, j])->([i, j]) {i=4, j=6}, cut 3·8: 8 falls inside i, whose strides
  // run from 6 to 24.
  FailureOr<RefinedAxes> refined = refineAxes(
      description(axes({4, 6}, {{{0, 1}}}, {{{0, 1}}}),
                  {AxisSignature::parallel(), AxisSignature::parallel()}),
      {{{3, 8}}}, {{{24}}});
  EXPECT_TRUE(failed(refined));
}

} // namespace
} // namespace mlir::enzyme::distributed
