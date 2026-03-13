#ifndef PACT_PROPERTY_SCHEME_H
#define PACT_PROPERTY_SCHEME_H

#include "llvm/ADT/StringRef.h"

namespace mlir::enzyme::pact::scheme {
constexpr llvm::StringLiteral kExecSubgroupWidth = "exec.subgroup_width";
constexpr llvm::StringLiteral kExecProgressModel = "exec.progress_model";
constexpr llvm::StringLiteral kComputeMathAssociative =
    "compute.math_associative";
constexpr llvm::StringLiteral kComputeMathCommutative =
    "compute.math_commutative";
constexpr llvm::StringLiteral kComputeReassociationPermitted =
    "compute.reassociation_permitted";
constexpr llvm::StringLiteral kExecParticipation = "exec.participation";
constexpr llvm::StringLiteral kCrosslaneMechanism = "crosslane.mechanism";

enum class PropertyKind { ContractVsCapability, ContractOnly, ContractDerived };
enum class MatchType { DirectCompare, Lookup, EvaluatorRequired };
enum class Repairability { Repairable, NotRepairable, Delegated };
enum class Severity { MustAdapt, PerformanceRisk, Blocking };
enum class ProgressModel { LockstepAssumed, ExplicitSync, Unknown };
enum class ProgressCapability { LockstepWithinWave, Independent, Configuable };
enum class Participation { Full, Partial };
enum class MechanismKind {
  ShuffleDown,
  ShuffleXor,
  ShuffleBroadcast,
  DPP,
  Bpermute,
  Readlane,
  LDS
};

} // namespace mlir::enzyme::pact::scheme

#endif // PACT_PROPERTY_SCHEME_H