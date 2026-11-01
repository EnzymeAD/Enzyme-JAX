// Unit tests for the derivation engine's expansion layer: propagation keys and
// derivation rules registered in the grammar registry.

#include "src/enzyme_ad/jax/Dialect/Distributed/DerivationExpander.h"

#include "gtest/gtest.h"

namespace mlir::enzyme::distributed {
namespace {

using TensorDigits = SmallVector<SmallVector<int64_t>>;

class DerivationExpanderTest : public ::testing::Test {
protected:
  DerivationExpanderTest() { context.loadDialect<DistributedDialect>(); }

  OperatorAxesAttr axes(ArrayRef<int64_t> extents,
                        ArrayRef<TensorDigits> operands,
                        ArrayRef<TensorDigits> results) {
    return OperatorAxesAttr::get(&context, extents, operands, results);
  }

  TypedDescription typed(StringRef kind, OperatorAxesAttr axes,
                         SmallVector<AxisSignature> signature,
                         SmallVector<Type> operandTypes,
                         SmallVector<Type> resultTypes) {
    return TypedDescription{
        GrammarDescription{StringAttr::get(&context, kind), axes,
                           /*params=*/nullptr,
                           BaseSignature{std::move(signature)}},
        std::move(operandTypes), std::move(resultTypes)};
  }

  Type tensor(ArrayRef<int64_t> shape, Type element) {
    return RankedTensorType::get(shape, element);
  }
  Type f32() { return Float32Type::get(&context); }
  Type f64() { return Float64Type::get(&context); }

  MLIRContext context;
};

// Descriptions that differ only in how their axes are numbered give the same
// key, and the axis map says where each description axis went.
TEST_F(DerivationExpanderTest, AxisNumberingDoesNotChangeTheKey) {
  Type matrix = tensor({4, 8}, f32());
  Type transposed = tensor({8, 4}, f32());
  // ([i], [j])->([j], [i]) {i=4, j=8}
  TypedDescription first =
      typed("test.transpose", axes({4, 8}, {{{0}, {1}}}, {{{1}, {0}}}),
            {AxisSignature::parallel(), AxisSignature::local()}, {matrix},
            {transposed});
  // The same operator with i and j numbered the other way round.
  TypedDescription second =
      typed("test.transpose", axes({8, 4}, {{{1}, {0}}}, {{{0}, {1}}}),
            {AxisSignature::local(), AxisSignature::parallel()}, {matrix},
            {transposed});

  auto [firstKey, firstMap] = PropagationKey::from(first);
  auto [secondKey, secondMap] = PropagationKey::from(second);
  EXPECT_TRUE(firstKey == secondKey);
  EXPECT_EQ(hash_value(firstKey), hash_value(secondKey));
  EXPECT_EQ(firstMap, (SmallVector<int64_t>{0, 1}));
  EXPECT_EQ(secondMap, (SmallVector<int64_t>{1, 0}));
  EXPECT_TRUE(secondKey.isBaseShardable(0));
  EXPECT_FALSE(secondKey.isBaseShardable(1));
}

// Operand and result element types are both part of the key, since scratch IR
// is built from it.
TEST_F(DerivationExpanderTest, ElementTypesSeparateKeys) {
  OperatorAxesAttr elementwise = axes({8}, {{{0}}}, {{{0}}});
  auto convert = [&](Type from, Type to) {
    return PropagationKey::from(typed("test.convert", elementwise,
                                      {AxisSignature::parallel()},
                                      {tensor({8}, from)}, {tensor({8}, to)}))
        .first;
  };

  EXPECT_FALSE(convert(f32(), f32()) == convert(f64(), f32()));
  EXPECT_FALSE(convert(f32(), f32()) == convert(f32(), f64()));
}

// The base case shards parallel, reduce and custom-terminal axes by itself;
// local and custom axes need a rule.
TEST_F(DerivationExpanderTest, BaseShardableAxes) {
  // ([a, b, c, d, e])->([a, b, c, d, e]), one axis per signature variant.
  OperatorAxesAttr fiveAxes = axes({2, 2, 2, 2, 2}, {{{0}, {1}, {2}, {3}, {4}}},
                                   {{{0}, {1}, {2}, {3}, {4}}});
  Type type = tensor({2, 2, 2, 2, 2}, f32());
  PropagationKey key =
      PropagationKey::from(
          typed("test.mixed", fiveAxes,
                {AxisSignature::parallel(),
                 AxisSignature::reduce(stablehlo::ReduceOpKind::Add),
                 AxisSignature::customTerminal(), AxisSignature::local(),
                 AxisSignature::custom()},
                {type}, {type}))
          .first;

  EXPECT_TRUE(key.isBaseShardable(0));
  EXPECT_TRUE(key.isBaseShardable(1));
  EXPECT_TRUE(key.isBaseShardable(2));
  EXPECT_FALSE(key.isBaseShardable(3));
  EXPECT_FALSE(key.isBaseShardable(4));
}

// Reductions that differ only in how partial results combine share a key.
TEST_F(DerivationExpanderTest, ReduceKindsShareAKey) {
  // ([i, j])->([i]) {i=4, j=8}
  OperatorAxesAttr rowReduction = axes({4, 8}, {{{0}, {1}}}, {{{0}}});
  auto reduction = [&](stablehlo::ReduceOpKind reduceKind) {
    return typed("test.reduce", rowReduction,
                 {AxisSignature::parallel(), AxisSignature::reduce(reduceKind)},
                 {tensor({4, 8}, f32())}, {tensor({4}, f32())});
  };

  PropagationKey sum =
      PropagationKey::from(reduction(stablehlo::ReduceOpKind::Add)).first;
  PropagationKey max =
      PropagationKey::from(reduction(stablehlo::ReduceOpKind::Max)).first;
  EXPECT_TRUE(sum == max);
  EXPECT_EQ(hash_value(sum), hash_value(max));
}

// A key's typed description has the canonical axes, the key's types, and the
// signature of the kind's registry entry.
TEST_F(DerivationExpanderTest, TypedDescriptionIsCanonical) {
  Type type = tensor({2, 8}, ComplexType::get(f32()));
  // A batched DFT numbered out of canonical order:
  // ([b, i])->([b, j]) {i=8, b=2, j=8}
  TypedDescription dft =
      typed("dft", axes({8, 2, 8}, {{{1}, {0}}}, {{{1}, {2}}}),
            {AxisSignature::local(), AxisSignature::parallel(),
             AxisSignature::local()},
            {type}, {type});

  TypedDescription canonical =
      PropagationKey::from(dft).first.typedDescription();
  EXPECT_EQ(canonical.description.kind.getValue(), "dft");
  EXPECT_EQ(canonical.description.axes,
            axes({2, 8, 8}, {{{0}, {1}}}, {{{0}, {2}}}));
  EXPECT_EQ(canonical.description.signature.axes,
            (SmallVector<AxisSignature>{AxisSignature::parallel(),
                                        AxisSignature::local(),
                                        AxisSignature::local()}));
  EXPECT_EQ(canonical.operandTypes, (SmallVector<Type>{type}));
  EXPECT_EQ(canonical.resultTypes, (SmallVector<Type>{type}));
}

// Each kind lists its own rules, in registration order.
TEST_F(DerivationExpanderTest, RulesListPerKindInRegistrationOrder) {
  GrammarRegistry &registry = getGrammarRegistry(&context);
  RuleId first = registry.registerRule("dft", {"first", nullptr, nullptr});
  RuleId other = registry.registerRule("twiddle", {"other", nullptr, nullptr});
  RuleId second = registry.registerRule("dft", {"second", nullptr, nullptr});

  EXPECT_EQ(SmallVector<RuleId>(registry.getRules("dft")),
            (SmallVector<RuleId>{first, second}));
  EXPECT_EQ(SmallVector<RuleId>(registry.getRules("twiddle")),
            (SmallVector<RuleId>{other}));
  EXPECT_TRUE(registry.getRules("test.unruled").empty());
  EXPECT_EQ(registry.getRule(second).name, "second");
}

} // namespace
} // namespace mlir::enzyme::distributed
