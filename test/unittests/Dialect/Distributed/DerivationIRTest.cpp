// Unit tests for building derivation IR (DerivationBuilder, BatchScope) and for
// the dft kind's batch peeling.

#include "src/enzyme_ad/jax/Dialect/Distributed/DerivationIR.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/GrammarRegistry.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

namespace mlir::enzyme::distributed {
namespace {

using TensorDigits = SmallVector<SmallVector<int64_t>>;

class DerivationIRTest : public ::testing::Test {
protected:
  DerivationIRTest()
      : opBuilder(&context), builder(opBuilder, opBuilder.getUnknownLoc()) {
    context.loadDialect<DistributedDialect, axis::AxisDialect>();
    module = ModuleOp::create(opBuilder.getUnknownLoc());
    opBuilder.setInsertionPointToEnd(module->getBody());
  }

  OperatorAxesAttr roles(ArrayRef<int64_t> extents,
                         ArrayRef<TensorDigits> inputs,
                         ArrayRef<TensorDigits> outputs) {
    return OperatorAxesAttr::get(&context, extents, inputs, outputs);
  }

  // The factors a tensor node holds, in order.
  static SmallVector<Value> axesOf(DerivationTensor tensor) {
    return SmallVector<Value>(
        cast<DerivationTensorOp>(tensor.getDefiningOp()).getAxes());
  }

  // A one-input, one-output term reading `in` and writing `out` through
  // <([i])->([j])>.
  DerivationTensor map(DerivationTensor input, AxisFactor in, AxisFactor out) {
    OperatorAxesAttr mapRoles =
        roles({axis::getFactorExtent(in), axis::getFactorExtent(out)}, {{{0}}},
              {{{1}}});
    OperatorAxis inAxis = builder.product({in});
    OperatorAxis outAxis = builder.product({out});
    return builder.term("dft", {input}, {inAxis, outAxis}, mapRoles,
                        /*params=*/nullptr, {complex()})[0];
  }

  Type complex() { return ComplexType::get(Float32Type::get(&context)); }

  MLIRContext context;
  OpBuilder opBuilder;
  DerivationBuilder builder;
  OwningOpRef<ModuleOp> module;
};

// An output holds its role factors and the active batch factors; the input
// keeps exactly the factors it was given, so it is broadcast over the batch.
TEST_F(DerivationIRTest, OutputGetsRoleAndBatchFactors) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  BatchScope batch(builder, {b});
  DerivationTensor x = builder.input(complex(), {n});
  DerivationTensor y = map(x, n, k);

  EXPECT_EQ(axesOf(x), (SmallVector<Value>{n}));
  EXPECT_EQ(axesOf(y), (SmallVector<Value>{k, b}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Scopes add up while nested and stop applying once closed. Each term reads
// an input batched like its output, so every term verifies.
TEST_F(DerivationIRTest, BatchScopesNest) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  AxisFactor c = builder.newAxis(2);

  DerivationTensor inner, outer, none;
  {
    BatchScope outerScope(builder, {b});
    {
      BatchScope innerScope(builder, {c});
      inner = map(builder.input(complex(), {n, b, c}), n, k);
    }
    outer = map(builder.input(complex(), {n, b}), n, k);
  }
  none = map(builder.input(complex(), {n}), n, k);

  EXPECT_EQ(axesOf(inner), (SmallVector<Value>{k, b, c}));
  EXPECT_EQ(axesOf(outer), (SmallVector<Value>{k, b}));
  EXPECT_EQ(axesOf(none), (SmallVector<Value>{k}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Splitting a factor composes its digits onto the root, major-first, at the
// strides of the factor's index space.
TEST_F(DerivationIRTest, SplitComposesOntoRoot) {
  AxisFactor n = builder.newAxis(16);
  SmallVector<AxisFactor> halves = builder.split(n, {4, 4});
  SmallVector<AxisFactor> quarters = builder.split(halves[1], {2, 2});

  auto rootOf = [](AxisFactor factor) {
    return *axis::getFactorProvenanceAxis(factor);
  };
  auto shape = [](AxisFactor factor) {
    return std::pair(axis::getFactorExtent(factor),
                     axis::getFactorStride(factor));
  };
  EXPECT_EQ(shape(halves[0]), (std::pair<int64_t, int64_t>(4, 4)));
  EXPECT_EQ(shape(halves[1]), (std::pair<int64_t, int64_t>(4, 1)));
  EXPECT_EQ(shape(quarters[0]), (std::pair<int64_t, int64_t>(2, 2)));
  EXPECT_EQ(shape(quarters[1]), (std::pair<int64_t, int64_t>(2, 1)));
  EXPECT_EQ(rootOf(quarters[0]), rootOf(n));
}

// dft peels its both-sided axes and drops their dimensions:
// <([i, j])->([i, k])> is batch i over <([j])->([k])>.
TEST_F(DerivationIRTest, DftPeelsBothSidedDimensions) {
  const GrammarKindEntry *dft = getGrammarRegistry(&context).lookup("dft");
  ASSERT_TRUE(dft && dft->peelBatch);
  DictionaryAttr params = DictionaryAttr::get(
      &context, {NamedAttribute(StringAttr::get(&context, "inverse"),
                                BoolAttr::get(&context, true))});
  PeeledBatch peeled =
      dft->peelBatch(roles({2, 16, 16}, {{{0}, {1}}}, {{{0}, {2}}}), params);

  EXPECT_EQ(peeled.batchAxes, (SmallVector<int64_t>{0}));
  EXPECT_EQ(peeled.innerRoles, roles({16, 16}, {{{0}}}, {{{1}}}));
  EXPECT_EQ(peeled.innerAxes, (SmallVector<int64_t>{1, 2}));
  EXPECT_EQ(peeled.innerParams, params);
}

// Dimensions with no axes are dropped too, and a dft without batch keeps all
// its axes: <([1, i])->([j, 1])> is <([i])->([j])>.
TEST_F(DerivationIRTest, DftPeelDropsAxislessDimensions) {
  const GrammarKindEntry *dft = getGrammarRegistry(&context).lookup("dft");
  PeeledBatch peeled = dft->peelBatch(roles({8, 8}, {{{}, {0}}}, {{{1}, {}}}),
                                      /*params=*/nullptr);

  EXPECT_TRUE(peeled.batchAxes.empty());
  EXPECT_EQ(peeled.innerRoles, roles({8, 8}, {{{0}}}, {{{1}}}));
  EXPECT_EQ(peeled.innerAxes, (SmallVector<int64_t>{0, 1}));
}

// On roles dft's verify rejects, peeling stays total: the mixed dimension [ij]
// keeps its both-sided i, so i's output dimension stays too, while the pure
// batch dimension of b still peels. <([ij], [b])->([i], [k], [b])> is batch b
// over <([ij])->([i], [k])>.
TEST_F(DerivationIRTest, DftPeelKeepsMixedDimensionsWhole) {
  const GrammarKindEntry *dft = getGrammarRegistry(&context).lookup("dft");
  PeeledBatch peeled =
      dft->peelBatch(roles({2, 8, 8, 4}, {{{0, 1}, {3}}}, {{{0}, {2}, {3}}}),
                     /*params=*/nullptr);

  EXPECT_EQ(peeled.batchAxes, (SmallVector<int64_t>{3}));
  EXPECT_EQ(peeled.innerRoles, roles({2, 8, 8}, {{{0, 1}}}, {{{0}, {2}}}));
  EXPECT_EQ(peeled.innerAxes, (SmallVector<int64_t>{0, 1, 2}));
}

// Keeping an axis can mix a dimension that was pure batch: [ij] keeps i, then
// [il] keeps l, then [lm] keeps m. Axes i, j, l, m, k; nothing peels in
// <([ij], [lm])->([il], [k], [m])>.
TEST_F(DerivationIRTest, DftPeelKeepsAxesAcrossDimensions) {
  const GrammarKindEntry *dft = getGrammarRegistry(&context).lookup("dft");
  OperatorAxesAttr mixed =
      roles({2, 4, 2, 2, 4}, {{{0, 1}, {2, 3}}}, {{{0, 2}, {4}, {3}}});
  PeeledBatch peeled = dft->peelBatch(mixed, /*params=*/nullptr);

  EXPECT_TRUE(peeled.batchAxes.empty());
  EXPECT_EQ(peeled.innerRoles, mixed);
  EXPECT_EQ(peeled.innerAxes, (SmallVector<int64_t>{0, 1, 2, 3, 4}));
}

} // namespace
} // namespace mlir::enzyme::distributed
