#ifndef PACT_TARGET_TARGET_REGISTRY_H
#define PACT_TARGET_TARGET_REGISTRY_H

#include "RawDescriptor.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::pact {

class TargetRegistry {
public:
  static const RawDescriptor *lookup(llvm::StringRef target_id);
  static void registerTarget(std::string id, RawDescriptor desc);

private:
  static llvm::StringMap<RawDescriptor> &getRegistry();
};

} // namespace mlir::enzyme::pact

#endif // PACT_TARGET_TARGET_REGISTRY_H