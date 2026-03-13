#ifndef PACT_TARGET_COMPILATION_CONFIG_H
#define PACT_TARGET_COMPILATION_CONFIG_H

#include <optional>
#include <string>

#include "RawDescriptor.h"

namespace mlir::enzyme::pact {

struct CompilationConfig {
  std::optional<int> effective_wave_size;
  std::string target_features;
  bool wave_size_locked = false;

  static CompilationConfig fromDefaults(const RawDescriptor &raw);
};

} // namespace mlir::enzyme::pact

#endif // PACT_TARGET_COMPILATION_CONFIG_H
