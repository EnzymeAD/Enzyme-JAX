#ifndef PACT_ANALYSIS_STRATEGY_H
#define PACT_ANALYSIS_STRATEGY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::enzyme::pact {

// Forward-declare the enum (fixed underlying type allows this).
// Full definition lives in FactState.h.
enum class FactState : uint8_t;

enum class StrategyScope { Op, Region, Kernel };

struct StrategyRequirement {
  bool requires_dpp = false;
  bool requires_ds_permute = false;
  int min_wave_size = 0;
  int min_lds_available = 0;
};

struct StrategyCost {
  float relative_overhead = 0.0f;

  int delta_instructions = 0;
  int delta_vgprs = 0;
  int delta_lds_bytes = 0;
  int delta_barriers = 0;

  bool preserves_full_semantics = true;

  bool dominates(const StrategyCost &other) const;
};

struct SideEffect {
  llvm::StringRef targetProperty;
  FactState forcedState;
  std::string reason;
};

struct Strategy {
  std::string name;

  StrategyRequirement requirement;
  StrategyCost cost;
  StrategyScope scope = StrategyScope::Op;

  llvm::SmallVector<SideEffect> sideEffects;

  llvm::SmallVector<std::string> implies;
  llvm::SmallVector<std::string> conflicts;
  llvm::SmallVector<std::string> activates;

  llvm::StringMap<std::string> params;
};

} // namespace mlir::enzyme::pact

#endif // PACT_ANALYSIS_STRATEGY_H