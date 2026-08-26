#pragma once

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzymexla {

void addMemoryEffectsFromAttr(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
    ArrayAttr effectsAttr);

void addAllMemoryEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects);

} // namespace enzymexla
} // namespace mlir
