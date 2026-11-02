// Unit tests for the derivation engine's expansion layer: propagation keys,
// rules registered in the grammar registry, and the expander's steps and step
// graphs.

#include "src/enzyme_ad/jax/Dialect/Distributed/DerivationExpander.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
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
    context.loadDialect<DistributedDialect, axis::AxisDialect,
                        func::FuncDialect>();
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

  // A derivation IR container in the module; terms are built into it until
  // endFunc returns `outputs`.
  func::FuncOp beginFunc() {
    auto func =
        opBuilder.create<func::FuncOp>(opBuilder.getUnknownLoc(), "container",
                                       opBuilder.getFunctionType({}, {}));
    opBuilder.setInsertionPointToStart(func.addEntryBlock());
    return func;
  }
  void endFunc(func::FuncOp func, ValueRange outputs) {
    opBuilder.create<func::ReturnOp>(opBuilder.getUnknownLoc(), outputs);
    func.setFunctionType(opBuilder.getFunctionType({}, outputs.getTypes()));
  }

  // A one-to-one kind whose sides share nothing, standing in for relayout.
  void registerRelayout() {
    getGrammarRegistry(&context).registerEntry(
        "test.relayout", {[](OperatorAxesAttr roles, DictionaryAttr) {
          return BaseSignature{SmallVector<AxisSignature>(
              roles.getNumAxes(), AxisSignature::local())};
        }});
  }

  // A two-input kind, <([i], [i])->([j])>, whose output depends on both
  // inputs; with one input lacking a batch digit, it is broadcast over it.
  void registerCombine() {
    getGrammarRegistry(&context).registerEntry(
        "test.combine", {[](OperatorAxesAttr roles, DictionaryAttr) {
          return BaseSignature{SmallVector<AxisSignature>(
              roles.getNumAxes(), AxisSignature::parallel())};
        }});
  }
  DerivationTensor combineTerm(DerivationTensor first, DerivationTensor second,
                               AxisFactor in, AxisFactor out) {
    OperatorAxesAttr roles =
        axes({axis::getFactorExtent(in), axis::getFactorExtent(out)},
             {{{0}}, {{0}}}, {{{1}}});
    return builder.term("test.combine", {first, second},
                        {builder.product({in}), builder.product({out})}, roles,
                        /*params=*/nullptr, {complex()})[0];
  }

  // A <([i])->([j])> term of `kind` reading `in` from `input` and writing
  // `out`.
  DerivationTensor mapTerm(StringRef kind, DerivationTensor input,
                           AxisFactor in, AxisFactor out) {
    OperatorAxesAttr roles =
        axes({axis::getFactorExtent(in), axis::getFactorExtent(out)}, {{{0}}},
             {{{1}}});
    return builder.term(kind, {input},
                        {builder.product({in}), builder.product({out})}, roles,
                        /*params=*/nullptr, {complex()})[0];
  }

  Attribute choice(int64_t value) { return opBuilder.getI64IntegerAttr(value); }

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
// instance is built from it. Shapes are not: what the roles don't cover is
// batch, which peels (BoundaryBatchIsPeeled).
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

// The same holds for a real program op as describe() gives it: a batched
// stablehlo.fft keys as the unbatched dft.
TEST_F(DerivationExpanderTest, RealFftBoundaryKey) {
  context.loadDialect<stablehlo::StablehloDialect>();
  OwningOpRef<ModuleOp> program = parseSourceString<ModuleOp>(
      R"mlir(
        func.func @f(%x: tensor<2x8xcomplex<f32>>) -> tensor<2x8xcomplex<f32>> {
          %0 = stablehlo.fft %x, type = FFT, length = [8]
              : (tensor<2x8xcomplex<f32>>) -> tensor<2x8xcomplex<f32>>
          return %0 : tensor<2x8xcomplex<f32>>
        })mlir",
      &context);
  ASSERT_TRUE(program);
  Operation *fft =
      &(*program->getOps<func::FuncOp>().begin()).getBody().front().front();
  FailureOr<TypedDescription> td = describe(fft);
  ASSERT_TRUE(succeeded(td));

  KeyMap<PropagationKey> keyMap = PropagationKey::from(*td);
  EXPECT_TRUE(keyMap.key ==
              PropagationKey::from(dftDescription(tensor({8}, complex()))).key);
  EXPECT_EQ(keyMap.axisMap,
            (SmallVector<KeyAxisRef>{peeled(0), key(0), key(1)}));
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
  EXPECT_EQ(SmallVector<Value>(term.getTensor(/*isOutput=*/false, 0).getAxes()),
            (SmallVector<Value>{instance.axes[0]}));
  EXPECT_EQ(SmallVector<Value>(term.getTensor(/*isOutput=*/true, 0).getAxes()),
            (SmallVector<Value>{instance.axes[1]}));

  FailureOr<KeyMap<PropagationKey>> rebuilt = PropagationKey::from(term);
  ASSERT_TRUE(succeeded(rebuilt));
  EXPECT_TRUE(rebuilt->key == dftKey);
  EXPECT_EQ(rebuilt->axisMap, (SmallVector<KeyAxisRef>{key(0), key(1)}));
}

// A term's batch never reaches its key: a dft term unbatched and batched over
// one or two digits share a key, and so do a two-input term unbatched and
// with one input broadcast over a digit.
TEST_F(DerivationExpanderTest, TermBatchDoesNotChangeTheKey) {
  registerCombine();
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  AxisFactor c = builder.newAxis(2);
  GrammarTermOp unbatched = dftTerm(builder.input(complex(), {n}), n, k);
  GrammarTermOp batched = dftTerm(builder.input(complex(), {n, b}), n, k);
  GrammarTermOp twice = dftTerm(builder.input(complex(), {n, b, c}), n, k);
  GrammarTermOp combined = termOf(combineTerm(
      builder.input(complex(), {n}), builder.input(complex(), {n}), n, k));
  GrammarTermOp broadcast = termOf(combineTerm(
      builder.input(complex(), {n, b}), builder.input(complex(), {n}), n, k));
  ASSERT_TRUE(succeeded(verify(*module)));

  PropagationKey unbatchedKey = keyOf(unbatched);
  EXPECT_TRUE(keyOf(batched) == unbatchedKey);
  EXPECT_TRUE(keyOf(twice) == unbatchedKey);
  EXPECT_EQ(hash_value(keyOf(batched)), hash_value(unbatchedKey));
  EXPECT_TRUE(keyOf(broadcast) == keyOf(combined));
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

// A term's key keeps each tensor position's element type, read from the
// term's tensor there: a conversion from f32 to f64.
TEST_F(DerivationExpanderTest, TermKeepsTensorElementTypes) {
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

// A key's steps are listed once: the base step, then each rule's choices in
// registration order and the rule's own order. splits runs on the key's
// instance. A kind without rules has only its base step.
TEST_F(DerivationExpanderTest, StepsListedOncePerKey) {
  GrammarRegistry &registry = getGrammarRegistry(&context);
  int splitsCalls = 0;
  SmallVector<OperatorAxesAttr> seenRoles;
  RuleId pairs = registry.registerRule(
      "dft", {"pairs",
              [&](GrammarTermOp term) {
                ++splitsCalls;
                seenRoles.push_back(term.getRoles());
                return SmallVector<Attribute>{choice(1), choice(2)};
              },
              nullptr});
  RuleId single =
      registry.registerRule("dft", {"single",
                                    [&](GrammarTermOp) {
                                      ++splitsCalls;
                                      return SmallVector<Attribute>{choice(3)};
                                    },
                                    nullptr});
  Expander<PropagationKey> expander(&context);
  KeyId dft = expander.intern(dftDescription(tensor({2, 8}, complex()))).first;

  ArrayRef<Step> steps = expander.steps(dft);
  ASSERT_EQ(steps.size(), 4u);
  EXPECT_TRUE(steps[0].isBase());
  EXPECT_EQ(steps[1].rule, pairs);
  EXPECT_EQ(steps[1].choice, choice(1));
  EXPECT_EQ(steps[2].rule, pairs);
  EXPECT_EQ(steps[2].choice, choice(2));
  EXPECT_EQ(steps[3].rule, single);
  EXPECT_EQ(steps[3].choice, choice(3));
  expander.steps(dft);
  EXPECT_EQ(splitsCalls, 2);
  EXPECT_EQ(seenRoles,
            (SmallVector<OperatorAxesAttr>{expander.key(dft).getRoles()}));

  KeyId twiddle =
      expander
          .intern(typed("twiddle", axes({8}, {{{0}}}, {{{0}}}),
                        {AxisSignature::parallel()}, {tensor({8}, complex())},
                        {tensor({8}, complex())}))
          .first;
  ASSERT_EQ(expander.steps(twiddle).size(), 1u);
  EXPECT_TRUE(expander.steps(twiddle)[0].isBase());
}

// A rule step's graph is built on first request only, and indexes what the
// rule built: a dft as two dfts through a fresh middle axis. The occurrences
// come in IR order and connect through the middle tensor, which both mention.
TEST_F(DerivationExpanderTest, GraphIndexesTheRuleOnFirstRequest) {
  int applyCalls = 0;
  getGrammarRegistry(&context).registerRule(
      "dft",
      {"two-stages",
       [&](GrammarTermOp) { return SmallVector<Attribute>{choice(0)}; },
       [&](DerivationBuilder &ruleBuilder, GrammarTermOp term, Attribute) {
         ++applyCalls;
         SmallVector<SmallVector<Value>> factors =
             term.getOperatorAxisFactors();
         auto in = cast<AxisFactor>(factors[0][0]);
         auto out = cast<AxisFactor>(factors[1][0]);
         AxisFactor middle = ruleBuilder.newAxis(axis::getFactorExtent(in));
         DerivationTensor middleTensor = ruleBuilder.term(
             "dft", {term.getInputs()[0]},
             {ruleBuilder.product({in}), ruleBuilder.product({middle})},
             term.getRoles(),
             /*params=*/nullptr, {complex()})[0];
         return ruleBuilder.term(
             "dft", {middleTensor},
             {ruleBuilder.product({middle}), ruleBuilder.product({out})},
             term.getRoles(), /*params=*/nullptr, {complex()});
       }});
  Expander<PropagationKey> expander(&context);
  KeyId dft = expander.intern(dftDescription(tensor({8}, complex()))).first;
  ASSERT_EQ(expander.steps(dft).size(), 2u);
  EXPECT_EQ(applyCalls, 0);

  const StepGraph &graph = expander.graph({dft, 1});
  expander.graph({dft, 1});
  EXPECT_EQ(applyCalls, 1);

  // Digits 0 and 1 are the key's axes, 2 the middle axis.
  EXPECT_EQ(graph.digitExtents, (SmallVector<uint64_t>{8, 8, 8}));
  EXPECT_EQ(graph.external, (SmallVector<SmallVector<DigitId>>{{0}, {1}}));
  // The input, middle and output tensors.
  EXPECT_EQ(graph.tensors, (SmallVector<axis::AtomSet>{{0}, {2}, {1}}));
  ASSERT_EQ(graph.occurrences.size(), 2u);
  const StepGraph::Occurrence &first = graph.occurrences[0];
  const StepGraph::Occurrence &second = graph.occurrences[1];
  EXPECT_EQ(first.key, dft);
  EXPECT_EQ(second.key, dft);
  EXPECT_EQ(first.axes, (SmallVector<SmallVector<DigitId>>{{0}, {2}}));
  EXPECT_EQ(second.axes, (SmallVector<SmallVector<DigitId>>{{2}, {1}}));
  EXPECT_EQ(first.tensorIndices, (SmallVector<unsigned>{0, 1}));
  EXPECT_EQ(second.tensorIndices, (SmallVector<unsigned>{1, 2}));
}

// An operator axis's digits are its factors' digits in the product's order,
// each major-first: a product of a fresh root and a factor of a factor maps
// to both, unsorted. Digit extents come from the common atoms, and a factor
// of the external axis is its digits.
TEST_F(DerivationExpanderTest, IndexMapsProductsToDigitsInOrder) {
  func::FuncOp func = beginFunc();
  AxisFactor n = builder.newAxis(16);
  AxisFactor c = builder.newAxis(2);
  AxisFactor k = builder.newAxis(8);
  SmallVector<AxisFactor> halves = builder.split(n, {4, 4});
  SmallVector<AxisFactor> quarters = builder.split(halves[1], {2, 2});
  // <([i])->([j])> with i = (c, quarters[0]) of extent 4 and j = k.
  DerivationTensor output = builder.term(
      "dft", {builder.input(complex(), {c, quarters[0]})},
      {builder.product({c, quarters[0]}), builder.product({k})},
      axes({4, 8}, {{{0}}}, {{{1}}}), /*params=*/nullptr, {complex()})[0];
  // A boundary tensor holding a one-level factor of n.
  builder.input(complex(), {halves[0]});
  endFunc(func, output);
  ASSERT_TRUE(succeeded(verify(func)));

  Expander<PropagationKey> expander(&context);
  StepGraph graph = expander.index(func, {n, c, k});
  // n's atoms 4, 2, 2 (digits 0 to 2), then c (3), then k (4).
  EXPECT_EQ(graph.digitExtents, (SmallVector<uint64_t>{4, 2, 2, 2, 8}));
  EXPECT_EQ(graph.external,
            (SmallVector<SmallVector<DigitId>>{{0, 1, 2}, {3}, {4}}));
  ASSERT_EQ(graph.occurrences.size(), 1u);
  EXPECT_EQ(graph.occurrences[0].axes,
            (SmallVector<SmallVector<DigitId>>{{3, 1}, {4}}));
  EXPECT_EQ(graph.tensors, (SmallVector<axis::AtomSet>{{1, 3}, {4}, {0}}));
}

// Two terms reading one tensor share its digits, with nothing linking them
// but the digits; a relayout's two sides share none.
TEST_F(DerivationExpanderTest, SharedTensorsIdentifyAndRelayoutsSeparate) {
  registerRelayout();
  func::FuncOp func = beginFunc();
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor p = builder.newAxis(8);
  AxisFactor r = builder.newAxis(8);
  DerivationTensor y = mapTerm("dft", builder.input(complex(), {n}), n, k);
  DerivationTensor transformed = mapTerm("dft", y, k, p);
  DerivationTensor relaid = mapTerm("test.relayout", y, k, r);
  endFunc(func, {transformed, relaid});
  ASSERT_TRUE(succeeded(verify(func)));

  Expander<PropagationKey> expander(&context);
  StepGraph graph = expander.index(func, {n, p, r});
  ASSERT_EQ(graph.occurrences.size(), 3u);
  unsigned shared = graph.occurrences[0].tensorIndices[1];
  EXPECT_EQ(graph.occurrences[1].tensorIndices[0], shared);
  EXPECT_EQ(graph.occurrences[2].tensorIndices[0], shared);
  EXPECT_EQ(graph.occurrences[0].axes[1], graph.occurrences[1].axes[0]);
  EXPECT_NE(graph.occurrences[2].key, graph.occurrences[1].key);
  const StepGraph::Occurrence &relayout = graph.occurrences[2];
  EXPECT_TRUE(
      axis::areAtomSetsDisjoint(graph.tensors[relayout.tensorIndices[0]],
                                graph.tensors[relayout.tensorIndices[1]]));
}

// A dft unbatched, batched over a digit, and with a both-sided operator axis
// the kind peels share a KeyId; each tensor's replicated digits are its batch,
// peeled axis included. A two-input term with one input broadcast over a
// digit has it on the other input and the output only.
TEST_F(DerivationExpanderTest, ReplicatedDigitsPerTensorPosition) {
  registerCombine();
  func::FuncOp func = beginFunc();
  AxisFactor n = builder.newAxis(8);
  AxisFactor k = builder.newAxis(8);
  AxisFactor b = builder.newAxis(4);
  AxisFactor peeledAxis = builder.newAxis(2);
  SmallVector<Value> outputs;
  outputs.push_back(mapTerm("dft", builder.input(complex(), {n}), n, k));
  outputs.push_back(mapTerm("dft", builder.input(complex(), {n, b}), n, k));
  // <([b], [i])->([b], [j])>
  outputs.push_back(builder.term("dft",
                                 {builder.input(complex(), {peeledAxis, n})},
                                 {builder.product({peeledAxis}),
                                  builder.product({n}), builder.product({k})},
                                 axes({2, 8, 8}, {{{0}, {1}}}, {{{0}, {2}}}),
                                 /*params=*/nullptr, {complex()})[0]);
  outputs.push_back(combineTerm(builder.input(complex(), {n, b}),
                                builder.input(complex(), {n}), n, k));
  endFunc(func, outputs);
  ASSERT_TRUE(succeeded(verify(func)));

  Expander<PropagationKey> expander(&context);
  StepGraph graph = expander.index(func, {n, k, b, peeledAxis});
  ASSERT_EQ(graph.occurrences.size(), 4u);
  for (const StepGraph::Occurrence &occurrence :
       ArrayRef(graph.occurrences).take_front(3))
    EXPECT_EQ(occurrence.key, graph.occurrences[0].key);
  // The peeled occurrence's key axes skip its peeled operator axis, whose
  // digits are external[3].
  EXPECT_EQ(graph.occurrences[2].axes,
            (SmallVector<SmallVector<DigitId>>{graph.external[0],
                                               graph.external[1]}));
  // Per occurrence, input then output.
  const axis::AtomSet &bDigits = graph.external[2];
  const axis::AtomSet &peeledDigits = graph.external[3];
  SmallVector<std::pair<axis::AtomSet, axis::AtomSet>> expected = {
      {{}, {}}, {bDigits, bDigits}, {peeledDigits, peeledDigits}};
  for (auto [occurrence, digits] :
       llvm::zip_equal(ArrayRef(graph.occurrences).take_front(3), expected)) {
    EXPECT_EQ(replicatedDigits(graph, occurrence, 0), digits.first);
    EXPECT_EQ(replicatedDigits(graph, occurrence, 1), digits.second);
  }
  const StepGraph::Occurrence &broadcast = graph.occurrences[3];
  EXPECT_EQ(replicatedDigits(graph, broadcast, 0), bDigits);
  EXPECT_EQ(replicatedDigits(graph, broadcast, 1), axis::AtomSet());
  EXPECT_EQ(replicatedDigits(graph, broadcast, 2), bDigits);
}

// A rewrite preserves dependence when each output depends, through the
// step's occurrences, on exactly the inputs its kind declares. For a
// two-input kind whose output depends on both inputs, a step that combines
// them preserves it, and one that reads only the first input drops one.
TEST_F(DerivationExpanderTest, PreservesDependence) {
  registerCombine();
  Expander<PropagationKey> expander(&context);
  auto step = [&](bool readBoth) {
    func::FuncOp func = beginFunc();
    AxisFactor n = builder.newAxis(8);
    AxisFactor k = builder.newAxis(8);
    DerivationTensor first = builder.input(complex(), {n});
    DerivationTensor second = builder.input(complex(), {n});
    DerivationTensor output = readBoth ? combineTerm(first, second, n, k)
                                       : mapTerm("dft", first, n, k);
    endFunc(func, output);
    EXPECT_TRUE(succeeded(verify(func)));
    StepGraph graph = expander.index(func, {n, k});
    opBuilder.setInsertionPointToEnd(module->getBody());
    return graph;
  };
  StepGraph both = step(/*readBoth=*/true);
  StepGraph firstOnly = step(/*readBoth=*/false);
  KeyId combine = both.occurrences[0].key;

  EXPECT_EQ(both.inputs.size(), 2u);
  EXPECT_EQ(both.outputs.size(), 1u);
  EXPECT_TRUE(expander.preservesDependence(combine, both));
  EXPECT_FALSE(expander.preservesDependence(combine, firstOnly));
}

// The boundary key of a batched fft is the key of the instance it builds,
// and a term of an unregistered kind fails to intern.
TEST_F(DerivationExpanderTest, InternAtBoundaryAndFromTerms) {
  Expander<PropagationKey> expander(&context);
  auto [boundary, axisMap] =
      expander.intern(dftDescription(tensor({2, 8}, complex())));
  EXPECT_EQ(axisMap, (SmallVector<KeyAxisRef>{peeled(0), key(0), key(1)}));
  KeyInstance instance = expander.key(boundary).buildInstance(builder);
  FailureOr<KeyId> fromInstance = expander.intern(termOf(instance.outputs[0]));
  ASSERT_TRUE(succeeded(fromInstance));
  EXPECT_EQ(*fromInstance, boundary);

  AxisFactor n = builder.newAxis(8);
  DerivationTensor unregistered = builder.term(
      "test.unregistered", {builder.input(f32(), {n})}, {builder.product({n})},
      axes({8}, {{{0}}}, {{{0}}}), /*params=*/nullptr, {f32()})[0];
  ScopedDiagnosticHandler silence(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(expander.intern(termOf(unregistered))));
}

} // namespace
} // namespace mlir::enzyme::distributed
