#include "DerivationIR.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"

namespace mlir::enzyme::distributed {

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
         "the roles have one input slot per input");
  assert(outputElementTypes.size() ==
             roles.getMappings(/*isResult=*/true).size() &&
         "the roles have one output slot per output element type");
  assert(static_cast<int64_t>(axes.size()) == roles.getNumAxes() &&
         "the roles have one axis per operator axis");
  SmallVector<Value> batchFactors;
  for (ArrayRef<AxisFactor> scope : batchScopes)
    llvm::append_range(batchFactors, scope);

  SmallVector<Type> outputTypes(outputElementTypes.size(),
                                TermOutputType::get(builder.getContext()));
  auto termOp = builder.create<GrammarTermOp>(
      loc, outputTypes, inputs, kind, SmallVector<Value>(axes), roles, params);

  SmallVector<SmallVector<Value>> axisFactors = termOp.getOperatorAxisFactors();
#ifndef NDEBUG
  SmallVector<Value> roleFactors;
  for (ArrayRef<Value> factors : axisFactors)
    llvm::append_range(roleFactors, factors);
  FailureOr<axis::CollectedAtoms> atoms =
      axis::collectAtoms({roleFactors, batchFactors});
  assert(succeeded(atoms) &&
         axis::areAtomSetsDisjoint(atoms->sets[0], atoms->sets[1]) &&
         "a batch factor must not be a digit of the term's operator axes");
#endif

  SmallVector<DerivationTensor> outputs;
  for (auto [slot, output] : llvm::enumerate(termOp.getOutputs())) {
    SmallVector<Value> factors =
        getSlotRoleFactors(roles, axisFactors, /*isOutput=*/true, slot);
    llvm::append_range(factors, batchFactors);
    auto tensor = builder.create<DerivationTensorOp>(
        loc,
        DerivationTensorType::get(builder.getContext(),
                                  outputElementTypes[slot]),
        output, factors);
    outputs.push_back(tensor.getTensor());
  }
  return outputs;
}

BatchScope::BatchScope(DerivationBuilder &builder, ArrayRef<AxisFactor> factors)
    : builder(builder) {
  builder.batchScopes.emplace_back(factors);
}

BatchScope::~BatchScope() { builder.batchScopes.pop_back(); }

} // namespace mlir::enzyme::distributed
