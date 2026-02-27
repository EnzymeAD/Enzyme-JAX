#ifndef PACT_COMPARE_RESULT_H
#define PACT_COMPARE_RESULT_H

#include "mlir/IR/Attributes.h"
#include "src/enzyme_ad/jax/Passes/Pact/PropertyScheme.h"
#include <string>

namespace mlir::enzyme::pact {

struct CompareResult {
  enum Kind { Match, Mismatch, ConditionalMatch };
  Kind kind = Match;
  Attribute sourceVal, targetVal;
  scheme::Severity severity = scheme::Severity::MustAdapt;
  std::string reason;

  static CompareResult makeMatch() { return {Match, {}, {}, {}, {}}; }

  static CompareResult makeMismatch(mlir::Attribute src, mlir::Attribute tgt,
                                    scheme::Severity sev,
                                    std::string reason = "") {
    return {Mismatch, src, tgt, sev, std::move(reason)};
  }

  bool isMatch() const { return kind == Match; }
  bool isMismatch() const { return kind == Mismatch; }
  bool isBlocking() const {
    return kind == Mismatch && severity == scheme::Severity::Blocking;
  }
};

} // namespace mlir::enzyme::pact

#endif // PACT_COMPARE_RESULT_H