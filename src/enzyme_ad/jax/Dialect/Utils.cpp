#include "src/enzyme_ad/jax/Dialect/Utils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace enzymexla {

void addMemoryEffectsFromAttr(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
    ArrayAttr effectsAttr) {
  for (auto attr : effectsAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    assert(strAttr &&
           "enzymexla.memory_effects must be a ArrayAttr<StringAttr>");

    StringRef kind = strAttr.getValue();
    if (kind == "allocate")
      effects.emplace_back(MemoryEffects::Allocate::get());
    else if (kind == "free")
      effects.emplace_back(MemoryEffects::Free::get());
    else if (kind == "write")
      effects.emplace_back(MemoryEffects::Write::get());
    else if (kind == "read")
      effects.emplace_back(MemoryEffects::Read::get());
    else
      assert(false && "enzymexla.memory_effects has an invalid value");
  }
}

void addAllMemoryEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

} // namespace enzymexla
} // namespace mlir
