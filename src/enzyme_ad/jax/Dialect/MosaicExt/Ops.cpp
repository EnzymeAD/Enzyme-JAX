#include "mlir/IR/Builders.h"

#include "Dialect.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/Canonicalizers.h"
#include "src/enzyme_ad/jax/Dialect/Utils.h"

namespace mlir {
namespace enzymexla {

LogicalResult mosaic_ext::MosaicCallOp::verify() {
  return verifyResultsAliasOperands(getOperation(), getOutputOperandAliases(),
                                    getInputs().size());
}

template <>
mosaic_ext::MosaicCallOp ReadOnlyArg<mosaic_ext::MosaicCallOp>::create(
    PatternRewriter &rewriter, mosaic_ext::MosaicCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return mosaic_ext::MosaicCallOp::create(
      rewriter, launchOp.getLoc(), resTys, launchOp.getFn(),
      launchOp.getGridx(), launchOp.getGridy(), launchOp.getGridz(),
      launchOp.getClusterx(), launchOp.getClustery(), launchOp.getClusterz(),
      launchOp.getInputs(), launchOp.getBackendConfigAttr(),
      launchOp.getOperandLayoutsAttr(), launchOp.getArgAttrsAttr(),
      launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

template <>
void ReadNoneArg<mosaic_ext::MosaicCallOp>::updateOperandSegmentSizes(
    mosaic_ext::MosaicCallOp call, int32_t numLiveOperands,
    PatternRewriter &rewriter) const {}

namespace mosaic_ext {

// ------------
// MosaicCallOp
// ------------

LogicalResult
MosaicCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global) {
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";
  }
  auto fnKind = global->getName().getStringRef();
  if (fnKind != "func.func" && fnKind != "llvm.func")
    return emitOpError("'")
           << getFn()
           << "' does not reference a valid func.func/llvm.func "
              "operation, got: "
           << global->getName();

  return success();
}

void MosaicCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable MosaicCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

Operation::operand_range MosaicCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange MosaicCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

void MosaicCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  assert(module && "MosaicCallOp must be inside a ModuleOp");

  auto callee = module.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "MosaicCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
}

void MosaicCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {}

} // namespace mosaic_ext
} // namespace enzymexla
} // namespace mlir

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/MosaicExt/MosaicExtOps.cpp.inc"
