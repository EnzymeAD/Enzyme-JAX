// Unit tests for the derivation engine's expansion layer: propagation keys and
// derivation rules registered in the grammar registry.

#include "src/enzyme_ad/jax/Dialect/Distributed/DerivationExpander.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

namespace mlir::enzyme::distributed {
namespace {

using TensorDigits = SmallVector<SmallVector<int64_t>>;

KeyAxisRef key(int64_t axis) { return KeyAxisRef::keyAxis(axis); }
KeyAxisRef peeled(int64_t number) { return KeyAxisRef::peeledAxis(number); }

class DerivationExpanderTest : public ::testing::Test {
protected:
  DerivationExpanderTest()
      : opBuilder(&context), builder(opBuilder, opBuilder.getUnknownLoc()) {
    context.loadDialect<DistributedDialect, axis::AxisDialect>();
    module = ModuleOp::create(opBuilder.getUnknownLoc());
    opBuilder.setInsertionPointToEnd(module->getBody());
  }

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

  // A dft description over `roles`, with the signature the registry gives,
  // as recognize() fills it in.
  TypedDescription dftTyped(OperatorAxesAttr roles, Type operandType,
                            Type resultType) {
    BaseSignature signature =
        getGrammarRegistry(&context).lookup("dft")->getBaseSignature(
            roles, /*params=*/nullptr);
    return typed("dft", roles, std::move(signature.axes), {operandType},
                 {resultType});
  }

  // A dft's description over `type`, transforming its minor dimension, the
  // way the fft recognizer describes it.
  TypedDescription dftDescription(RankedTensorType type) {
    int64_t rank = type.getRank();
    SmallVector<int64_t> extents(type.getShape());
    extents.push_back(type.getShape().back());
    TensorDigits operandDims, resultDims;
    for (int64_t dim = 0; dim < rank; ++dim) {
      operandDims.push_back({dim});
      resultDims.push_back({dim});
    }
    resultDims.back() = {rank};
    return dftTyped(axes(extents, {operandDims}, {resultDims}), type, type);
  }

  // A one-input, one-output dft term reading `in` and writing `out` through
  // <([i])->([j])>.
  GrammarTermOp dftTerm(DerivationTensor input, AxisFactor in, AxisFactor out) {
    OperatorAxesAttr roles =
        axes({axis::getFactorExtent(in), axis::getFactorExtent(out)}, {{{0}}},
             {{{1}}});
    DerivationTensor output = builder.term(
        "dft", {input}, {builder.product({in}), builder.product({out})}, roles,
        /*params=*/nullptr, {complex()})[0];
    return termOf(output);
  }

  static GrammarTermOp termOf(DerivationTensor output) {
    return cast<DerivationTensorOp>(output.getDefiningOp())
        .getDefinition()
        .getDefiningOp<GrammarTermOp>();
  }

  // The key of a term expected to have one; an empty key on failure.
  PropagationKey keyOf(GrammarTermOp term) {
    FailureOr<KeyMap<PropagationKey>> keyMap = PropagationKey::from(term);
    if (failed(keyMap)) {
      ADD_FAILURE() << "the term has no key";
      return PropagationKey();
    }
    return keyMap->key;
  }

  RankedTensorType tensor(ArrayRef<int64_t> shape, Type element) {
    return RankedTensorType::get(shape, element);
  }
  Type f32() { return Float32Type::get(&context); }
  Type f64() { return Float64Type::get(&context); }
  Type complex() { return ComplexType::get(f32()); }

  MLIRContext context;
  OpBuilder opBuilder;
  DerivationBuilder builder;
  OwningOpRef<ModuleOp> module;
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

  KeyMap<PropagationKey> firstKey = PropagationKey::from(first);
  KeyMap<PropagationKey> secondKey = PropagationKey::from(second);
  EXPECT_TRUE(firstKey.key == secondKey.key);
  EXPECT_EQ(hash_value(firstKey.key), hash_value(secondKey.key));
  EXPECT_EQ(firstKey.axisMap, (SmallVector<KeyAxisRef>{key(0), key(1)}));
  EXPECT_EQ(secondKey.axisMap, (SmallVector<KeyAxisRef>{key(1), key(0)}));
  EXPECT_TRUE(secondKey.key.isBaseShardable(0));
  EXPECT_FALSE(secondKey.key.isBaseShardable(1));
}

// Input and output element types are both part of the key, since the key's
// instance is built from it.
TEST_F(DerivationExpanderTest, ElementTypesSeparateKeys) {
  OperatorAxesAttr elementwise = axes({8}, {{{0}}}, {{{0}}});
  auto convert = [&](Type from, Type to) {
    return PropagationKey::from(typed("test.convert", elementwise,
                                      {AxisSignature::parallel()},
                                      {tensor({8}, from)}, {tensor({8}, to)}))
        .key;
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
          .key;

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
      PropagationKey::from(reduction(stablehlo::ReduceOpKind::Add)).key;
  PropagationKey max =
      PropagationKey::from(reduction(stablehlo::ReduceOpKind::Max)).key;
  EXPECT_TRUE(sum == max);
  EXPECT_EQ(hash_value(sum), hash_value(max));
}

// At the boundary, a real op's batch dimensions peel: an fft over 2x3x8 has
// the key of an unbatched fft over 8, and each batch dimension's operator axis
// is one peeled number, shared by the input and output tensors it is on.
TEST_F(DerivationExpanderTest, BoundaryBatchIsPeeled) {
  KeyMap<PropagationKey> batched =
      PropagationKey::from(dftDescription(tensor({2, 3, 8}, complex())));
  KeyMap<PropagationKey> unbatched =
      PropagationKey::from(dftDescription(tensor({8}, complex())));

  EXPECT_TRUE(batched.key == unbatched.key);
  EXPECT_EQ(hash_value(batched.key), hash_value(unbatched.key));
  // Axes a, b, i, j: ([a], [b], [i])->([a], [b], [j]).
  EXPECT_EQ(batched.axisMap,
            (SmallVector<KeyAxisRef>{peeled(0), peeled(1), key(0), key(1)}));
  EXPECT_EQ(batched.key.getRoles(), axes({8, 8}, {{{0}}}, {{{1}}}));
}

// A key's instance is one term of the key's kind and canonical roles, over
// one fresh root per key axis, with boundary input tensors holding exactly
// their role axes; the instance's term has the key it was built from.
TEST_F(DerivationExpanderTest, InstanceIsCanonical) {
  // A batched DFT numbered out of canonical order:
  // ([b, i])->([b, j]) {i=8, b=2, j=8}
  RankedTensorType type = tensor({2, 8}, complex());
  TypedDescription dft =
      dftTyped(axes({8, 2, 8}, {{{1}, {0}}}, {{{1}, {2}}}), type, type);
  PropagationKey dftKey = PropagationKey::from(dft).key;

  KeyInstance instance = dftKey.buildInstance(builder);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_EQ(instance.axes.size(), 2u);
  ASSERT_EQ(instance.outputs.size(), 1u);
  GrammarTermOp term = termOf(instance.outputs[0]);
  EXPECT_EQ(term.getKind(), "dft");
  EXPECT_EQ(term.getRoles(), axes({8, 8}, {{{0}}}, {{{1}}}));
  EXPECT_EQ(
      SmallVector<Value>(term.getSlotTensor(/*isOutput=*/false, 0).getAxes()),
      (SmallVector<Value>{instance.axes[0]}));
  EXPECT_EQ(
      SmallVector<Value>(term.getSlotTensor(/*isOutput=*/true, 0).getAxes()),
      (SmallVector<Value>{instance.axes[1]}));

  FailureOr<KeyMap<PropagationKey>> rebuilt = PropagationKey::from(term);
  ASSERT_TRUE(succeeded(rebuilt));
  EXPECT_TRUE(rebuilt->key == dftKey);
  EXPECT_EQ(rebuilt->axisMap, (SmallVector<KeyAxisRef>{key(0), key(1)}));
}

// A term's batch never reaches its key: a dft term unbatched, batched over a
// digit, and with its input broadcast over that digit share a key.
TEST_F(DerivationExpanderTest, TermBatchDoesNotChangeTheKey) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  GrammarTermOp unbatched = dftTerm(builder.input(complex(), {n}), n, k);
  GrammarTermOp batched, broadcast;
  {
    BatchScope batch(builder, {b});
    batched = dftTerm(builder.input(complex(), {n, b}), n, k);
    broadcast = dftTerm(builder.input(complex(), {n}), n, k);
  }
  ASSERT_TRUE(succeeded(verify(*module)));

  PropagationKey unbatchedKey = keyOf(unbatched);
  EXPECT_TRUE(keyOf(batched) == unbatchedKey);
  EXPECT_TRUE(keyOf(broadcast) == unbatchedKey);
  EXPECT_EQ(hash_value(keyOf(batched)), hash_value(unbatchedKey));
}

// The kind's peelBatch axes peel from a term too: a dft term with a
// both-sided operator axis, <([b], [i])->([b], [j])>, has the key of the
// unbatched dft, with b peeled.
TEST_F(DerivationExpanderTest, TermPeelsKindBatchAxes) {
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  OperatorAxesAttr roles = axes({4, 8, 8}, {{{0}, {1}}}, {{{0}, {2}}});
  DerivationTensor output = builder.term(
      "dft", {builder.input(complex(), {b, n})},
      {builder.product({b}), builder.product({n}), builder.product({k})}, roles,
      /*params=*/nullptr, {complex()})[0];
  GrammarTermOp withBatchAxis = termOf(output);
  GrammarTermOp plain = dftTerm(builder.input(complex(), {n}), n, k);
  ASSERT_TRUE(succeeded(verify(*module)));

  FailureOr<KeyMap<PropagationKey>> keyMap =
      PropagationKey::from(withBatchAxis);
  ASSERT_TRUE(succeeded(keyMap));
  EXPECT_TRUE(keyMap->key == keyOf(plain));
  EXPECT_EQ(keyMap->axisMap,
            (SmallVector<KeyAxisRef>{peeled(0), key(0), key(1)}));
}

// An operator axis made of digits of two roots keys like a single axis of
// the same extent: keys are coarse, one key axis per product.
TEST_F(DerivationExpanderTest, ProductKeysLikeOneAxis) {
  AxisFactor a = builder.newAxis(4);
  AxisFactor c = builder.newAxis(2);
  AxisFactor k = builder.newAxis(8);
  SmallVector<AxisFactor> aDigits = builder.split(a, {2, 2});
  // <([i])->([j])> with i = (c, a's minor digit) and j = k's minor digit,
  // both of extent 4.
  OperatorAxesAttr roles = axes({4, 4}, {{{0}}}, {{{1}}});
  AxisFactor smallK = builder.split(k, {2, 4})[1];
  DerivationTensor output = builder.term(
      "dft", {builder.input(complex(), {c, aDigits[1]})},
      {builder.product({c, aDigits[1]}), builder.product({smallK})}, roles,
      /*params=*/nullptr, {complex()})[0];
  GrammarTermOp product = termOf(output);
  AxisFactor m = builder.newAxis(4);
  AxisFactor o = builder.newAxis(4);
  GrammarTermOp single = dftTerm(builder.input(complex(), {m}), m, o);
  ASSERT_TRUE(succeeded(verify(*module)));

  EXPECT_TRUE(keyOf(product) == keyOf(single));
}

// A term's key keeps each slot's element type, read from the slot's own
// tensor: a conversion from f32 to f64.
TEST_F(DerivationExpanderTest, TermKeepsSlotElementTypes) {
  getGrammarRegistry(&context).registerEntry(
      "test.convert", {[](OperatorAxesAttr roles, DictionaryAttr) {
        return BaseSignature{SmallVector<AxisSignature>(
            roles.getNumAxes(), AxisSignature::parallel())};
      }});
  AxisFactor n = builder.newAxis(8);
  OperatorAxesAttr roles = axes({8}, {{{0}}}, {{{0}}});
  DerivationTensor output = builder.term(
      "test.convert", {builder.input(f32(), {n})}, {builder.product({n})},
      roles, /*params=*/nullptr, {f64()})[0];
  ASSERT_TRUE(succeeded(verify(*module)));

  PropagationKey convertKey = keyOf(termOf(output));
  EXPECT_EQ(SmallVector<Type>(convertKey.getInputElementTypes()),
            (SmallVector<Type>{f32()}));
  EXPECT_EQ(SmallVector<Type>(convertKey.getOutputElementTypes()),
            (SmallVector<Type>{f64()}));
}

// A term whose kind has no registry entry gets a warning and no key.
TEST_F(DerivationExpanderTest, UnregisteredTermKindWarns) {
  AxisFactor n = builder.newAxis(8);
  OperatorAxesAttr roles = axes({8}, {{{0}}}, {{{0}}});
  DerivationTensor output = builder.term(
      "test.unregistered", {builder.input(f32(), {n})}, {builder.product({n})},
      roles, /*params=*/nullptr, {f32()})[0];
  ASSERT_TRUE(succeeded(verify(*module)));

  std::string warning;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    if (diagnostic.getSeverity() == DiagnosticSeverity::Warning)
      warning = diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(PropagationKey::from(termOf(output))));
  EXPECT_NE(warning.find("'test.unregistered' is not registered"),
            std::string::npos);
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
