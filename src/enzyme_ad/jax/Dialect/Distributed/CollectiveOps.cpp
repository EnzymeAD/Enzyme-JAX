#include "CollectiveOps.h"

namespace mlir::enzyme::distributed {
using namespace ::mlir::enzyme::axis;

// small helper
llvm::SmallVector<::mlir::Value> concatRanges(::mlir::ValueRange lhs,
                                              ::mlir::ValueRange rhs) {
  llvm::SmallVector<::mlir::Value> result;
  result.reserve(lhs.size() + rhs.size());
  result.append(lhs.begin(), lhs.end());
  result.append(rhs.begin(), rhs.end());
  return result;
}

template <typename VT>
llvm::SmallVector<TypedValue<VT>>
concatTypedRanges(TypedValueArrayRef<VT> lhs, TypedValueArrayRef<VT> rhs) {
  llvm::SmallVector<TypedValue<VT>> result;
  result.reserve(lhs.size() + rhs.size());
  result.append(lhs.begin(), lhs.end());
  result.append(rhs.begin(), rhs.end());
  return result;
}

LogicalResult DistributedCollectiveOp::verify() {
  auto inputMeshFactors = axis::getProductProvenanceFactors(getInputMesh());
  if (failed(inputMeshFactors)) {
    return emitOpError()
           << "requires input_mesh to be produced by axis.product";
  }

  auto outputMeshFactors = axis::getProductProvenanceFactors(getOutputMesh());
  if (failed(outputMeshFactors)) {
    return emitOpError()
           << "requires output_mesh to be produced by axis.product";
  }

  if (!axis::areFactorIndexSpacesEqual(*inputMeshFactors, *outputMeshFactors)) {
    return emitOpError()
           << "requires input_mesh and output_mesh to have equal index space";
  }

  ArrayAttr reductionFunctions = getReductionFunctionsAttr();
  if (!reductionFunctions) {
    return emitOpError() << "requires reduction_functions attribute";
  }
  if (reductionFunctions.size() != getReductionGroups().size()) {
    return emitOpError() << "requires reduction_functions size to match "
                            "reduction_groups size ("
                         << reductionFunctions.size()
                         << " != " << getReductionGroups().size() << ")";
  }

  for (auto [idx, reductionFunctionAttr] :
       llvm::enumerate(reductionFunctions)) {
    auto reductionFunction =
        dyn_cast_or_null<FlatSymbolRefAttr>(reductionFunctionAttr);
    if (!reductionFunction) {
      return emitOpError() << "requires reduction_functions[" << idx
                           << "] to be a FlatSymbolRefAttr";
    }

    if (!lookupSymbolInEnclosingScopes(*this, reductionFunction)) {
      return emitOpError() << "references unknown reduction function symbol "
                           << reductionFunction;
    }
  }

  auto typedReductionGroups = axis::castTypedValueList<axis::FactorGroupType>(
      getReductionGroups(), "FactorGroupType");
  auto typedMappingLHS = axis::castTypedValueList<axis::FactorGroupType>(
      getMappingLhs(), "FactorGroupType");
  auto typedMappingRHS = axis::castTypedValueList<axis::FactorGroupType>(
      getMappingRhs(), "FactorGroupType");
  auto reduction_group_factors =
      axis::flattenGroupsToFactors(typedReductionGroups);
  SmallVector<TypedValue<AxisFactorType>> mapping_lhs_factors =
      axis::flattenGroupsToFactors(typedMappingLHS);
  auto mapping_rhs_factors = axis::flattenGroupsToFactors(typedMappingRHS);
  auto lhs_filtered = filterOutReplicationFactors(mapping_lhs_factors);
  auto rhs_filtered = filterOutReplicationFactors(mapping_rhs_factors);

  // Create the set of axis we expect to see from the input, output types.
  OpBuilder builder(getContext());
  builder.clearInsertionPoint();
  Location loc = getLoc();
  auto expected_input_tensor_axes =
      axis::createAxesForRankedShape(getInputObject().getType(), builder, loc);
  auto expected_output_tensor_axes =
      axis::createAxesForRankedShape(getOutputTensorType(), builder, loc);

  auto expected_input_factors =
      axis::viewAxesAsFactors(expected_input_tensor_axes, builder, loc);
  auto expected_output_factors =
      axis::viewAxesAsFactors(expected_output_tensor_axes, builder, loc);

  /*
   * Want:
   * - reduction disjoint with lhs, rhs (rhs and lhs may overlap)
   * - reduction + lhs_filtered = input_mesh + tensor axes
   * - rhs_filtered = output_mesh + tensor axes
   */
  // llvm::SmallVector<TypedValue<AxisFactorType>> lhs_space;
  auto lhs_space =
      concatTypedRanges<AxisFactorType>(reduction_group_factors, lhs_filtered);
  auto expected_input_space = concatTypedRanges<AxisFactorType>(
      *inputMeshFactors, expected_input_factors);
  auto expected_output_space = concatTypedRanges<AxisFactorType>(
      *outputMeshFactors, expected_output_factors);
  if (!axis::areFactorsDisjoint(lhs_space)) {
    return emitOpError()
           << "requires reduction_groups + mapping_lhs to be disjoint";
  }
  if (!axis::areFactorsDisjoint(rhs_filtered)) {
    return emitOpError() << "requires mapping_rhs to be disjoint";
  }
  if (!axis::areFactorIndexSpacesEqual(lhs_space, expected_input_space)) {
    return emitOpError()
           << "requires reduction_groups + mapping_lhs to match input_mesh "
              "+ input_tensor axes";
  }
  if (!axis::areFactorIndexSpacesEqual(rhs_filtered, expected_output_space)) {
    return emitOpError()
           << "requires mapping_rhs to match output_mesh + output_tensor axes";
  }

  if (!axis::areFactorGroupsDisjoint(typedReductionGroups)) {
    return emitOpError() << "requires reduction_groups to be pairwise disjoint";
  }

  if (getMappingLhs().size() != getMappingRhs().size()) {
    return emitOpError() << "requires mapping_lhs and mapping_rhs to have the"
                         << " same length (" << getMappingLhs().size()
                         << " != " << getMappingRhs().size() << ")";
  }

  for (auto [idx, lhsMapping] : llvm::enumerate(getMappingLhs())) {
    Value rhsMapping = getMappingRhs()[idx];

    FailureOr<uint64_t> lhsExtent = axis::getFactorGroupExtent(
        cast<TypedValue<axis::FactorGroupType>>(lhsMapping));
    if (failed(lhsExtent)) {
      return emitOpError() << "requires mapping_lhs[" << idx
                           << "] to be produced by axis.product";
    }

    FailureOr<uint64_t> rhsExtent = axis::getFactorGroupExtent(
        cast<TypedValue<axis::FactorGroupType>>(rhsMapping));
    if (failed(rhsExtent)) {
      return emitOpError() << "requires mapping_rhs[" << idx
                           << "] to be produced by axis.product";
    }

    if (*lhsExtent != *rhsExtent) {
      return emitOpError() << "requires mapping pair #" << idx
                           << " to have matching extent (" << *lhsExtent
                           << " != " << *rhsExtent << ")";
    }
  }

  return success();
}

LogicalResult DistributedCollectiveOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DistributedCollectiveOpAdaptor adaptor(operands, attributes, properties,
                                         regions);
  inferredReturnTypes.push_back(
      AsynchHandleType::get(context, adaptor.getOutputTensorType()));
  return success();
}

LogicalResult DistributedAwait::verify() {
  auto handleType = dyn_cast<AsynchHandleType>(getAsyncHandle().getType());
  if (!handleType) {
    return emitOpError() << "requires async_handle to be an AsynchHandleType";
  }

  Type expectedValueType = handleType.getValueType();
  Type actualValueType = getValue().getType();
  if (actualValueType != expectedValueType) {
    return emitOpError() << "requires result type to match awaited handle "
                         << "value type " << expectedValueType << ", but got "
                         << actualValueType;
  }

  return success();
}

LogicalResult DistributedAwait::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  DistributedAwaitAdaptor adaptor(operands, attributes, properties, regions);
  auto handleType =
      dyn_cast<AsynchHandleType>(adaptor.getAsyncHandle().getType());
  if (!handleType) {
    return failure();
  }

  inferredReturnTypes.push_back(handleType.getValueType());
  return success();
}

} // namespace mlir::enzyme::distributed