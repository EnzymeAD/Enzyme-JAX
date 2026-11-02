// Unit tests for building derivation IR (DerivationBuilder) and for the dft
// kind's batch peeling.

#include "src/enzyme_ad/jax/Dialect/Distributed/DerivationIR.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/GrammarRegistry.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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

// An output holds its role factors and its input's batch: whatever the
// input's tensor holds beyond its role digits.
TEST_F(DerivationIRTest, OutputGetsItsInputsBatch) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  AxisFactor c = builder.newAxis(2);
  DerivationTensor batched = map(builder.input(complex(), {n, b, c}), n, k);
  DerivationTensor unbatched = map(builder.input(complex(), {n}), n, k);

  EXPECT_EQ(axesOf(batched), (SmallVector<Value>{k, b, c}));
  EXPECT_EQ(axesOf(unbatched), (SmallVector<Value>{k}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

// A batch digit can sit inside an axis the input holds whole: the term reads
// n's major digit, so the minor digit is batch and reaches the output as a
// factor of n's root.
TEST_F(DerivationIRTest, BatchWithinAnAxisOfTheInput) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(2);
  SmallVector<AxisFactor> digits = builder.split(n, {2, 4});
  DerivationTensor y = map(builder.input(complex(), {n}), digits[0], k);

  SmallVector<Value> factors = axesOf(y);
  ASSERT_EQ(factors.size(), 2u);
  EXPECT_EQ(factors[0], k);
  auto minor = cast<AxisFactor>(factors[1]);
  EXPECT_EQ(*axis::getFactorProvenanceAxis(minor),
            *axis::getFactorProvenanceAxis(n));
  EXPECT_EQ(axis::getFactorExtent(minor), 4);
  EXPECT_EQ(axis::getFactorStride(minor), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Every output of a kind without declared dependence depends on every input,
// so its batch is the union of theirs: a broadcast input adds nothing, and a
// digit two inputs bring appears once, as the larger of its factors.
TEST_F(DerivationIRTest, BatchIsTheUnionOverInputs) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  AxisFactor bMajor = builder.split(b, {2, 2})[0];
  auto combine = [&](ArrayRef<AxisFactor> first, ArrayRef<AxisFactor> second) {
    // <([i], [i])->([j])>
    OperatorAxesAttr combineRoles = roles({8, 8}, {{{0}}, {{0}}}, {{{1}}});
    return builder.term(
        "combine",
        {builder.input(complex(), first), builder.input(complex(), second)},
        {builder.product({n}), builder.product({k})}, combineRoles,
        /*params=*/nullptr, {complex()})[0];
  };
  DerivationTensor broadcast = combine({n, b}, {n});
  DerivationTensor shared = combine({n, b}, {n, b});
  DerivationTensor nested = combine({n, bMajor}, {n, b});

  EXPECT_EQ(axesOf(broadcast), (SmallVector<Value>{k, b}));
  EXPECT_EQ(axesOf(shared), (SmallVector<Value>{k, b}));
  EXPECT_EQ(axesOf(nested), (SmallVector<Value>{k, b}));
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Batch factors of one root that overlap without nesting merge into the
// factor spanning both: on a root of 8, (4, stride 1) and (4, stride 2)
// together cover all of it.
TEST_F(DerivationIRTest, OverlappingBatchFactorsMergeIntoTheirSpan) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor c = builder.newAxis(8);
  Value root = *axis::getFactorProvenanceAxis(c);
  AxisFactor minor = opBuilder.create<axis::AxisFactorOp>(
      opBuilder.getUnknownLoc(), root, 4, 1);
  AxisFactor major = opBuilder.create<axis::AxisFactorOp>(
      opBuilder.getUnknownLoc(), root, 4, 2);
  // <([i], [i])->([j])>
  DerivationTensor output =
      builder.term("combine",
                   {builder.input(complex(), {n, minor}),
                    builder.input(complex(), {n, major})},
                   {builder.product({n}), builder.product({k})},
                   roles({8, 8}, {{{0}}, {{0}}}, {{{1}}}), /*params=*/nullptr,
                   {complex()})[0];

  SmallVector<Value> factors = axesOf(output);
  ASSERT_EQ(factors.size(), 2u);
  auto span = cast<AxisFactor>(factors[1]);
  EXPECT_EQ(*axis::getFactorProvenanceAxis(span), root);
  EXPECT_EQ(axis::getFactorExtent(span), 8);
  EXPECT_EQ(axis::getFactorStride(span), 1);
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Batch factors the builder makes but no output uses are erased: a term
// reading the major digit of x twice gets x's minor digit once.
TEST_F(DerivationIRTest, UnusedBatchFactorsAreErased) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(2);
  SmallVector<AxisFactor> digits = builder.split(n, {2, 4});
  DerivationTensor x = builder.input(complex(), {n});
  // <([i], [i])->([j])>
  builder.term(
      "combine", {x, x}, {builder.product({digits[0]}), builder.product({k})},
      roles({2, 2}, {{{0}}, {{0}}}, {{{1}}}), /*params=*/nullptr, {complex()});
  ASSERT_TRUE(succeeded(verify(*module)));

  // Only the split's own minor digit, which nothing reads, is unused.
  int64_t unused = 0;
  module->walk([&](axis::AxisFactorOp factor) {
    if (factor->use_empty())
      ++unused;
  });
  EXPECT_EQ(unused, 1);
}

// A declared dependence limits each output's batch to the inputs it depends
// on, and the verifier holds outputs to it.
TEST_F(DerivationIRTest, DeclaredDependenceLimitsBatch) {
  getGrammarRegistry(&context).registerEntry(
      "test.pair", {[](OperatorAxesAttr pairRoles, DictionaryAttr) {
                      return BaseSignature{SmallVector<AxisSignature>(
                          pairRoles.getNumAxes(), AxisSignature::parallel())};
                    },
                    /*peelBatch=*/nullptr, /*verify=*/nullptr,
                    [](OperatorAxesAttr, DictionaryAttr) {
                      return SmallVector<SmallVector<int64_t>>{{0, 1}, {1}};
                    }});
  AxisFactor i = builder.newAxis(4);
  AxisFactor j = builder.newAxis(8);
  AxisFactor b = builder.newAxis(2);
  // <([i], [j])->([i], [j])>: the second output depends on the second input
  // only.
  SmallVector<DerivationTensor> outputs = builder.term(
      "test.pair",
      {builder.input(complex(), {i, b}), builder.input(complex(), {j})},
      {builder.product({i}), builder.product({j})},
      roles({4, 8}, {{{0}}, {{1}}}, {{{0}}, {{1}}}), /*params=*/nullptr,
      {complex(), complex()});

  EXPECT_EQ(axesOf(outputs[0]), (SmallVector<Value>{i, b}));
  EXPECT_EQ(axesOf(outputs[1]), (SmallVector<Value>{j}));
  EXPECT_TRUE(succeeded(verify(*module)));

  // The second output carrying x's batch anyway is rejected.
  auto second = cast<DerivationTensorOp>(outputs[1].getDefiningOp());
  Value definition = second.getDefinition();
  opBuilder.setInsertionPoint(second);
  opBuilder.create<DerivationTensorOp>(second.getLoc(), second.getType(),
                                       definition, SmallVector<Value>{j, b});
  second.erase();
  ScopedDiagnosticHandler silence(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(*module)));
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
