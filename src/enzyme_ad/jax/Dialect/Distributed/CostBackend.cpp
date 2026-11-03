#include "CostBackend.h"

#include "RooflineBackend.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <mutex>

namespace mlir::enzyme::distributed {

std::optional<double> getLatency(Operation *op) {
  if (auto latency = op->getAttrOfType<FloatAttr>(kLatencyAttrName))
    return latency.getValueAsDouble();
  return std::nullopt;
}

void setLatency(Operation *op, double nanoseconds) {
  op->setAttr(kLatencyAttrName,
              Builder(op->getContext()).getF64FloatAttr(nanoseconds));
}

namespace {

// Process-wide map from device_target to backend. A backend is created from
// its factory on first lookup, so registering one costs nothing until a mesh
// names it.
class CostBackendRegistry {
public:
  CostBackendRegistry() {
    auto roofline = [] { return std::make_unique<RooflineBackend>(); };
    entries["cpu"].factory = roofline;
    entries["roofline"].factory = roofline;
  }

  void add(llvm::StringRef name, CostBackendFactory factory) {
    std::lock_guard<std::mutex> lock(mutex);
    // A backend handed out by lookup stays alive for the process, so a name
    // is never re-registered over it.
    [[maybe_unused]] bool inserted =
        entries.try_emplace(name, Entry{std::move(factory), nullptr}).second;
    assert(inserted && "a cost backend name is registered once");
  }

  CostBackend *lookup(llvm::StringRef name) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(name);
    if (it == entries.end())
      return nullptr;
    Entry &entry = it->second;
    if (!entry.backend)
      entry.backend = entry.factory();
    return entry.backend.get();
  }

  std::vector<std::string> names() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::string> result;
    for (const auto &entry : entries)
      result.push_back(entry.getKey().str());
    llvm::sort(result);
    return result;
  }

private:
  struct Entry {
    CostBackendFactory factory;
    std::unique_ptr<CostBackend> backend;
  };

  std::mutex mutex;
  llvm::StringMap<Entry> entries;
};

CostBackendRegistry &getRegistry() {
  static CostBackendRegistry registry;
  return registry;
}

} // namespace

void registerCostBackend(llvm::StringRef name, CostBackendFactory factory) {
  getRegistry().add(name, std::move(factory));
}

CostBackend *lookupCostBackend(llvm::StringRef deviceTarget) {
  return getRegistry().lookup(deviceTarget);
}

std::vector<std::string> registeredCostBackendNames() {
  return getRegistry().names();
}

} // namespace mlir::enzyme::distributed
