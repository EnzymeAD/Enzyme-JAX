#include "TargetRegistry.h"

using namespace mlir::enzyme::pact;

llvm::StringMap<RawDescriptor> &TargetRegistry::getRegistry() {
  static llvm::StringMap<RawDescriptor> registry;
  return registry;
}

const RawDescriptor *TargetRegistry::lookup(llvm::StringRef target_id) {
  auto &reg = getRegistry();
  auto it = reg.find(target_id);
  return it != reg.end() ? &it->second : nullptr;
}

void TargetRegistry::registerTarget(std::string id, RawDescriptor desc) {
  getRegistry()[id] = std::move(desc);
}