#include "Dialect.h"
#include "Utilities.h"

namespace mlir::enzyme::distributed {

LogicalResult DistributedFunctionOp::verify() {
  if (!(*this)->getParentOfType<MeshComputationOp>()) {
    return emitOpError() << "must be nested in distributed.MeshComputation";
  }

  ArrayAttr argumentTypesAttr =
      (*this)->getAttrOfType<ArrayAttr>("argument_types");
  if (!argumentTypesAttr) {
    return emitOpError() << "requires argument_types attribute";
  }
  for (auto [idx, argumentTypeAttr] : llvm::enumerate(argumentTypesAttr)) {
    if (!isa<TypeAttr>(argumentTypeAttr)) {
      return emitOpError() << "requires argument_types[" << idx
                           << "] to be a TypeAttr";
    }
  }

  Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != argumentTypesAttr.size()) {
    return emitOpError() << "requires body block argument count to equal "
                         << "argument_types size ("
                         << bodyBlock.getNumArguments()
                         << " != " << argumentTypesAttr.size() << ")";
  }
  for (auto [idx, argumentTypeAttr] : llvm::enumerate(argumentTypesAttr)) {
    Type expectedType = cast<TypeAttr>(argumentTypeAttr).getValue();
    Type actualType = bodyBlock.getArgument(idx).getType();
    if (expectedType != actualType) {
      return emitOpError() << "requires body block argument #" << idx
                           << " to have type " << expectedType << ", but got "
                           << actualType;
    }
  }

  Operation *terminator = bodyBlock.getTerminator();
  auto yieldOp = dyn_cast_or_null<DistributedYieldOp>(terminator);
  if (!yieldOp) {
    return emitOpError()
           << "requires body to terminate with distributed.DistributedYield";
  }

  ArrayAttr returnTypesAttr = (*this)->getAttrOfType<ArrayAttr>("return_types");
  if (!returnTypesAttr) {
    return emitOpError() << "requires return_types attribute";
  }

  ValueRange yieldedValues = yieldOp.getReturns();
  if (yieldedValues.size() != returnTypesAttr.size()) {
    return emitOpError() << "requires distributed.DistributedYield to yield "
                         << returnTypesAttr.size() << " value(s), but got "
                         << yieldedValues.size();
  }

  for (auto [idx, returnTypeAttr] : llvm::enumerate(returnTypesAttr)) {
    auto typeAttr = dyn_cast<TypeAttr>(returnTypeAttr);
    if (!typeAttr) {
      return emitOpError() << "requires return_types[" << idx
                           << "] to be a TypeAttr";
    }

    Type expectedType = typeAttr.getValue();
    Type actualType = yieldedValues[idx].getType();
    if (expectedType != actualType) {
      return emitOpError() << "requires distributed.DistributedYield operand #"
                           << idx << " to have type " << expectedType
                           << ", but got " << actualType;
    }
  }

  return success();
}

LogicalResult DistributedCallOp::verify() {
  FailureOr<TypedValue<axis::FactorGroupType>> callerContext =
      getEnclosingExecutionContext(*this);
  if (failed(callerContext)) {
    return emitOpError() << "must be nested in distributed.function with a "
                         << "FactorGroupType execution_context";
  }

  FailureOr<DistributedFunctionOp> callee =
      resolveSymbolOpFromAttr<DistributedFunctionOp>(*this, getCalleeAttr());
  if (failed(callee)) {
    return emitOpError() << "references unknown distributed function symbol "
                         << getCalleeAttr();
  }

  ArrayAttr calleeReturnTypesAttr =
      (*callee)->getAttrOfType<ArrayAttr>("return_types");
  if (!calleeReturnTypesAttr) {
    return emitOpError() << "callee " << callee->getSymName()
                         << " is missing return_types attribute";
  }
  if (getReturns().size() != calleeReturnTypesAttr.size()) {
    return emitOpError() << "requires call result count to match callee "
                         << "return_types size (" << getReturns().size()
                         << " != " << calleeReturnTypesAttr.size() << ")";
  }
  for (auto [idx, returnTypeAttr] : llvm::enumerate(calleeReturnTypesAttr)) {
    auto typeAttr = dyn_cast<TypeAttr>(returnTypeAttr);
    if (!typeAttr) {
      return emitOpError() << "requires callee return_types[" << idx
                           << "] to be a TypeAttr";
    }
    Type expectedType = typeAttr.getValue();
    Type actualType = getReturns()[idx].getType();
    if (expectedType != actualType) {
      return emitOpError() << "requires call result #" << idx << " to have "
                           << "type " << expectedType << ", but got "
                           << actualType;
    }
  }

  auto calleeContext = dyn_cast<TypedValue<axis::FactorGroupType>>(
      callee->getExecutionContext());
  auto replicateOver =
      dyn_cast<TypedValue<axis::FactorGroupType>>(getReplicateOver());
  if (!calleeContext || !replicateOver) {
    return emitOpError() << "requires both callee execution_context and "
                         << "replicate_over to be FactorGroupType";
  }

  auto callerFactors = axis::getProductProvenanceFactors(*callerContext);
  auto calleeFactors = axis::getProductProvenanceFactors(calleeContext);
  auto replicateFactors = axis::getProductProvenanceFactors(replicateOver);
  if (failed(callerFactors) || failed(calleeFactors) ||
      failed(replicateFactors)) {
    return emitOpError() << "requires execution contexts to be produced by "
                         << "axis.product so factor provenance can be checked";
  }

  auto expectedCallerFactors = *calleeFactors;
  expectedCallerFactors.append(replicateFactors->begin(),
                               replicateFactors->end());
  if (!axis::areFactorIndexSpacesEqual(*callerFactors, expectedCallerFactors)) {
    return emitOpError()
           << "requires caller execution context to equal callee "
           << "execution_context x replicate_over (permutation-insensitive)";
  }

  return success();
}

} // namespace mlir::enzyme::distributed
