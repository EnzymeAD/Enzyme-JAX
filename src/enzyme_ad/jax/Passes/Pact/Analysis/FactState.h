#ifndef PACT_ANALYSIS_FACT_STATE_H
#define PACT_ANALYSIS_FACT_STATE_H

#include "CompareResult.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace mlir::enzyme::pact {

enum class FactState : uint8_t {
  Unknown,
  Tentative,
  Match,
  Mismatch,
  ConditionalMatch,
  Repaired,
  Discharged,
  NotApplicable,
  Degraded,
  ConservativeDefault
};

inline llvm::StringRef factStateName(FactState s) {
  switch (s) {
  case FactState::Unknown:
    return "unknown";
  case FactState::Tentative:
    return "tentative";
  case FactState::Match:
    return "match";
  case FactState::Mismatch:
    return "mismatch";
  case FactState::ConditionalMatch:
    return "conditionalMatch";
  case FactState::Repaired:
    return "repaired";
  case FactState::Discharged:
    return "discharged";
  case FactState::NotApplicable:
    return "not_applicable";
  case FactState::Degraded:
    return "degraded";
  case FactState::ConservativeDefault:
    return "conservative_default";
  }
  return "?";
}

} // namespace mlir::enzyme::pact

#include "Strategy.h"

namespace mlir::enzyme::pact {

struct FactEntry {
  FactState state = FactState::Unknown;
  CompareResult matchResult;
  std::optional<Strategy> selectedStrategy;
  std::string reason;

  bool isResolved() const;
  bool needsRepair() const { return state == FactState::Mismatch; }
};

} // namespace mlir::enzyme::pact

#endif // PACT_ANALYSIS_FACT_STATE_H