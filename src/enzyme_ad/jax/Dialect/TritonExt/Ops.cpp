#include "mlir/IR/Builders.h"

#include "Dialect.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/CommonCanonicalizers.h"
#include "src/enzyme_ad/jax/Dialect/Utils.h"

namespace mlir {
namespace enzyme {

template <>
triton_ext::TritonCallOp ReadOnlyArg<triton_ext::TritonCallOp>::create(
    PatternRewriter &rewriter, triton_ext::TritonCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return rewriter.create<triton_ext::TritonCallOp>(
      launchOp.getLoc(), resTys, launchOp.getFn(), launchOp.getGridx(),
      launchOp.getGridy(), launchOp.getGridz(), launchOp.getShmem(),
      launchOp.getInputs(), launchOp.getBackendConfigAttr(),
      launchOp.getOperandLayoutsAttr(), /*resultLayouts*/ nullptr,
      launchOp.getArgAttrsAttr(), launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

namespace triton_ext {

// ------------
// TritonCallOp
// ------------

LogicalResult
TritonCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  // Verify that the referenced symbol is specifically a tt.func operation
  if (global->getName().getStringRef() != "tt.func")
    return emitOpError("'")
           << getFn() << "' does not reference a valid tt.func operation, got: "
           << global->getName();

  return success();
}

void TritonCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable TritonCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

Operation::operand_range TritonCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange TritonCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

void TritonCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  assert(module && "TritonCallOp must be inside a ModuleOp");

  auto callee = module.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "TritonCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
};

void TritonCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ReadOnlyArg<TritonCallOp>, ReadNoneArg<TritonCallOp>>(context);
}

} // namespace triton_ext
} // namespace enzyme
} // namespace mlir

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/TritonExt/TritonExtOps.cpp.inc"
