#include "FactState.h"

using namespace mlir::enzyme::pact;

bool FactEntry::isResolved() const {
  switch (state) {
  case FactState::Match:
  case FactState::Repaired:
  case FactState::Discharged:
  case FactState::NotApplicable:
  case FactState::ConservativeDefault:
    return true;
  default:
    return false;
  }
}