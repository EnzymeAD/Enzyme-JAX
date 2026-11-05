#include "mlir/IR/Builders.h"

#include "Dialect.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "src/enzyme_ad/jax/Dialect/Canonicalizers.h"
#include "src/enzyme_ad/jax/Dialect/Utils.h"

namespace mlir {
namespace enzymexla {

LogicalResult cuda_tile_ext::CudaTileCallOp::verify() {
  return verifyResultsAliasOperands(getOperation(), getOutputOperandAliases(),
                                    getInputs().size());
}

template <>
cuda_tile_ext::CudaTileCallOp
ReadOnlyArg<cuda_tile_ext::CudaTileCallOp>::create(
    PatternRewriter &rewriter, cuda_tile_ext::CudaTileCallOp launchOp,
    ArrayRef<Type> resTys, ArrayAttr outputAliases) const {
  return cuda_tile_ext::CudaTileCallOp::create(
      rewriter, launchOp.getLoc(), resTys, launchOp.getFn(),
      launchOp.getGridx(), launchOp.getGridy(), launchOp.getGridz(),
      launchOp.getClusterx(), launchOp.getClustery(), launchOp.getClusterz(),
      launchOp.getInputs(), launchOp.getBackendConfigAttr(),
      launchOp.getOperandLayoutsAttr(), launchOp.getArgAttrsAttr(),
      launchOp.getResAttrsAttr(), outputAliases,
      launchOp.getXlaSideEffectFreeAttr());
}

template <>
void ReadNoneArg<cuda_tile_ext::CudaTileCallOp>::updateOperandSegmentSizes(
    cuda_tile_ext::CudaTileCallOp call, int32_t numLiveOperands,
    PatternRewriter &rewriter) const {}

namespace cuda_tile_ext {

// --------------
// CudaTileCallOp
// --------------

LogicalResult
CudaTileCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global) {
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";
  }
  auto fnKind = global->getName().getStringRef();
  if (fnKind != "cuda_tile.entry" && fnKind != "llvm.func")
    return emitOpError("'")
           << getFn()
           << "' does not reference a valid cuda_tile.entry/llvm.func "
              "operation, got: "
           << global->getName();

  return success();
}

void CudaTileCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setFnAttr(cast<SymbolRefAttr>(callee));
}

CallInterfaceCallable CudaTileCallOp::getCallableForCallee() {
  auto attr = getFnAttr();
  return SymbolRefAttr::get(getContext(), attr.getRootReference(),
                            attr.getNestedReferences());
}

Operation::operand_range CudaTileCallOp::getArgOperands() {
  return getInputs();
}

MutableOperandRange CudaTileCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

void CudaTileCallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  ModuleOp module = (*this)->getParentOfType<ModuleOp>();
  assert(module && "CudaTileCallOp must be inside a ModuleOp");

  auto callee = module.lookupSymbol<FunctionOpInterface>(getFnAttr());
  assert(callee && "CudaTileCallOp must have a valid function");

  auto effectsAttr =
      callee->getAttrOfType<ArrayAttr>("enzymexla.memory_effects");
  if (!effectsAttr) {
    addAllMemoryEffects(effects);
    return;
  }

  addMemoryEffectsFromAttr(effects, effectsAttr);
}

void CudaTileCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {}

} // namespace cuda_tile_ext
} // namespace enzymexla
} // namespace mlir

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/CudaTileExt/CudaTileExtOps.cpp.inc"
