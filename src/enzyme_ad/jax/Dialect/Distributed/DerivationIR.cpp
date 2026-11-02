#include "DerivationIR.h"

#include "GrammarRegistry.h"
#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

namespace {

// Each input's batch: its tensor's factors without its role digits. The
// factors are either the input's own (already in place) or new factors from
// subtractSpace, which come back detached: getOutputFactors inserts those an
// output uses, and eraseUnusedBatch drops the rest. A batch digit is never a
// digit of the term's operator axes.
SmallVector<SmallVector<AxisFactor>>
getInputBatch(ValueRange inputs, OperatorAxesAttr roles,
              ArrayRef<SmallVector<Value>> axisFactors, OpBuilder &builder) {
  auto asFactors = [](auto values) {
    return llvm::map_to_vector(
        values, [](Value value) { return cast<AxisFactor>(value); });
  };
  SmallVector<SmallVector<AxisFactor>> inputBatch;
  for (auto [position, input] : llvm::enumerate(inputs)) {
    FailureOr<SmallVector<AxisFactor>> batch = axis::subtractSpace(
        asFactors(cast<DerivationTensorOp>(input.getDefiningOp()).getAxes()),
        asFactors(
            getRoleFactors(roles, axisFactors, /*isOutput=*/false, position)),
        builder);
    assert(succeeded(batch) &&
           "an input's tensor and roles cut each root in a divisibility chain");
    inputBatch.push_back(std::move(*batch));
  }
#ifndef NDEBUG
  SmallVector<Value> roleFactors, batchFactors;
  for (ArrayRef<Value> factors : axisFactors)
    llvm::append_range(roleFactors, factors);
  for (ArrayRef<AxisFactor> batch : inputBatch)
    llvm::append_range(batchFactors, batch);
  FailureOr<axis::CollectedAtoms> atoms =
      axis::collectAtoms({roleFactors, batchFactors});
  assert(succeeded(atoms) &&
         axis::areAtomSetsDisjoint(atoms->sets[0], atoms->sets[1]) &&
         "an input's tensor holds no role digit of another tensor position");
#endif
  return inputBatch;
}

// Each output's factors: its role digits, then the union of the batch of the
// inputs it depends on. Several inputs can bring the same batch digit, as
// different factors of it. A detached batch factor an output uses is inserted
// at the builder's insertion point.
SmallVector<SmallVector<Value>>
getOutputFactors(ArrayRef<SmallVector<AxisFactor>> inputBatch,
                 ArrayRef<SmallVector<int64_t>> dependence,
                 OperatorAxesAttr roles,
                 ArrayRef<SmallVector<Value>> axisFactors, OpBuilder &builder,
                 Location loc) {
  SmallVector<SmallVector<Value>> outputFactors;
  for (auto [position, dependsOn] : llvm::enumerate(dependence)) {
    SmallVector<ArrayRef<AxisFactor>> batches;
    for (int64_t input : dependsOn)
      batches.push_back(inputBatch[input]);
    SmallVector<Value> &factors = outputFactors.emplace_back(
        getRoleFactors(roles, axisFactors, /*isOutput=*/true, position));
    for (AxisFactor factor : axis::unionFactorSpaces(batches, builder, loc)) {
      if (Operation *op = factor.getDefiningOp(); !op->getBlock())
        builder.insert(op);
      factors.push_back(factor);
    }
  }
  return outputFactors;
}

// Erases the batch factors of getInputBatch that no output used.
void eraseUnusedBatch(ArrayRef<SmallVector<AxisFactor>> inputBatch) {
  for (ArrayRef<AxisFactor> batch : inputBatch)
    for (AxisFactor factor : batch)
      if (Operation *op = factor.getDefiningOp(); !op->getBlock())
        op->erase();
}

} // namespace

AxisFactor DerivationBuilder::newAxis(int64_t extent) {
  auto root = builder.create<LogicalMeshAxesOp>(loc, extent);
  return axis::viewAxisAsFactor(root.getAxis(), builder, loc);
}

SmallVector<AxisFactor> DerivationBuilder::split(AxisFactor factor,
                                                 ArrayRef<int64_t> extents) {
  assert(llvm::product_of(extents) == axis::getFactorExtent(factor) &&
         "digit extents must multiply to the factor's extent");
  SmallVector<uint64_t> unsignedExtents(extents.begin(), extents.end());
  SmallVector<AxisFactor> digits;
  for (auto [extent, stride] :
       llvm::zip_equal(extents, axis::digitStrides(unsignedExtents)))
    digits.push_back(
        axis::createSubfactor(factor, extent, stride, builder, loc));
  return digits;
}

OperatorAxis DerivationBuilder::product(ArrayRef<AxisFactor> factors) {
  return axis::viewFactorsAsProduct(factors, builder, loc);
}

DerivationTensor DerivationBuilder::input(Type elementType,
                                          ArrayRef<AxisFactor> factors) {
  auto tensor = builder.create<DerivationTensorOp>(
      loc, DerivationTensorType::get(builder.getContext(), elementType),
      /*definition=*/nullptr, SmallVector<Value>(factors));
  return tensor.getTensor();
}

SmallVector<DerivationTensor>
DerivationBuilder::term(StringRef kind, ValueRange inputs,
                        ArrayRef<OperatorAxis> axes, OperatorAxesAttr roles,
                        DictionaryAttr params, TypeRange outputElementTypes) {
  assert(inputs.size() == roles.getMappings(/*isResult=*/false).size() &&
         "the roles have one input tensor position per input");
  assert(outputElementTypes.size() ==
             roles.getMappings(/*isResult=*/true).size() &&
         "the roles have one output tensor position per output element type");
  assert(static_cast<int64_t>(axes.size()) == roles.getNumAxes() &&
         "the roles have one axis per operator axis");
  SmallVector<Type> outputTypes(outputElementTypes.size(),
                                TermOutputType::get(builder.getContext()));
  auto termOp = builder.create<GrammarTermOp>(
      loc, outputTypes, inputs, kind, SmallVector<Value>(axes), roles, params);
  SmallVector<SmallVector<Value>> axisFactors = termOp.getOperatorAxisFactors();

  // An output holds its role digits and the batch of the inputs it depends
  // on. Batch factors are created before the term; outputs follow it.
  SmallVector<SmallVector<Value>> outputFactors;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(termOp);
    SmallVector<SmallVector<AxisFactor>> inputBatch =
        getInputBatch(inputs, roles, axisFactors, builder);
    outputFactors = getOutputFactors(
        inputBatch, getDependence(builder.getContext(), kind, roles, params),
        roles, axisFactors, builder, loc);
    eraseUnusedBatch(inputBatch);
  }
  SmallVector<DerivationTensor> outputs;
  for (auto [position, output] : llvm::enumerate(termOp.getOutputs())) {
    auto tensor = builder.create<DerivationTensorOp>(
        loc,
        DerivationTensorType::get(builder.getContext(),
                                  outputElementTypes[position]),
        output, outputFactors[position]);
    outputs.push_back(tensor.getTensor());
  }
  return outputs;
}

} // namespace mlir::enzyme::distributed
