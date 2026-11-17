//===- DependenceInfo.cpp - Calculate dependency information for a Scop ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/DependenceInfo.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/PollyDebug.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "isl/aff.h"
#include "isl/aff_type.h"
#include "isl/ctx.h"
#include "isl/flow.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_map_type.h"
#include "isl/union_set.h"

#include <cstdlib>

using namespace polymer;
using namespace llvm;

#define DEBUG_TYPE "polymer-dependence"

static cl::OptionCategory
    PolymerCategory("Polymer Options", "Configure the polymer loop optimizer");

static cl::opt<int> OptComputeOut(
    "polymer-dependences-computeout",
    cl::desc("Bound the dependence analysis by a maximal amount of "
             "computational steps (0 means no bound)"),
    cl::Hidden, cl::init(0), cl::cat(PolymerCategory));

static cl::opt<bool>
    LegalityCheckDisabled("disable-polymer-legality",
                          cl::desc("Disable polly legality check"), cl::Hidden,
                          cl::cat(PolymerCategory));

namespace {
enum AnalysisType { VALUE_BASED_ANALYSIS, MEMORY_BASED_ANALYSIS };
} // namespace

static cl::opt<enum AnalysisType> OptAnalysisType(
    "polymer-dependences-analysis-type",
    cl::desc("The kind of dependence analysis to use"),
    cl::values(clEnumValN(VALUE_BASED_ANALYSIS, "value-based",
                          "Exact dependences without transitive dependences"),
               clEnumValN(MEMORY_BASED_ANALYSIS, "memory-based",
                          "Overapproximation of dependences")),
    cl::Hidden, cl::init(VALUE_BASED_ANALYSIS), cl::cat(PolymerCategory));

static void printIslStr(raw_ostream &OS, char *str) {
  if (str) {
    OS << str;
    std::free(str);
  } else {
    OS << "null";
  }
}

/// Tag the @p Relation domain with @p TagId
static __isl_give isl_map *tag(__isl_take isl_map *Relation,
                               __isl_take isl_id *TagId) {
  isl_space *Space = isl_map_get_space(Relation);
  Space = isl_space_drop_dims(Space, isl_dim_out, 0,
                              isl_map_dim(Relation, isl_dim_out));
  Space = isl_space_set_tuple_id(Space, isl_dim_out, TagId);
  isl_multi_aff *Tag = isl_multi_aff_domain_map(Space);
  Relation = isl_map_preimage_domain_multi_aff(Relation, Tag);
  return Relation;
}

/// Tag the @p Relation domain with either MA->getArrayId() or
/// MA->getId() based on @p TagLevel
static __isl_give isl_map *tag(__isl_take isl_map *Relation, MemoryAccess *MA,
                               Dependences::AnalysisLevel TagLevel) {
  if (TagLevel == Dependences::AL_Reference)
    return tag(Relation, MA->getArrayId().release());

  if (TagLevel == Dependences::AL_Access)
    return tag(Relation, MA->getId().release());

  return Relation;
}

static void collectAsyncCopyInfo(Scop &S, isl_union_map *&CopyMustWrite,
                                 isl_union_map *&CopyRead) {
  isl_space *Space = S.getParamSpace().release();
  CopyRead = isl_union_map_empty(isl_space_copy(Space));
  CopyMustWrite = isl_union_map_empty(Space);

  for (ScopStmt &Stmt : S) {
    if (!Stmt.isValidAsyncCopy())
      continue;

    for (MemoryAccess *MA : Stmt) {
      isl_set *domcp = Stmt.getDomain().release();
      isl_map *accdom = MA->getAccessRelation().release();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if (MA->isRead())
        CopyRead = isl_union_map_add_map(CopyRead, accdom);
      else if (MA->isMayWrite())
        llvm_unreachable("may writes not allowed for copies");
      else if (MA->isMustWrite())
        CopyMustWrite = isl_union_map_add_map(CopyMustWrite, accdom);
      else if (MA->isKill())
        llvm_unreachable("kills not allowed for copies");
      else
        llvm_unreachable("unknown access type");
    }
  }

  CopyRead = isl_union_map_coalesce(CopyRead);
  CopyMustWrite = isl_union_map_coalesce(CopyMustWrite);
}

static void collectInfo(Scop &S, isl_union_map *&Read,
                        isl_union_map *&MustWrite, isl_union_map *&MayWrite,
                        isl_union_map *&Kill, isl_union_map *&ReductionTagMap,
                        isl_union_set *&TaggedStmtDomain,
                        Dependences::AnalysisLevel Level) {
  isl_space *Space = S.getParamSpace().release();
  Read = isl_union_map_empty(isl_space_copy(Space));
  MustWrite = isl_union_map_empty(isl_space_copy(Space));
  MayWrite = isl_union_map_empty(isl_space_copy(Space));
  Kill = isl_union_map_empty(isl_space_copy(Space));
  ReductionTagMap = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *StmtSchedule = isl_union_map_empty(Space);

  SmallPtrSet<const ScopArrayInfo *, 8> ReductionArrays;

  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      isl_set *domcp = Stmt.getDomain().release();
      isl_map *accdom = MA->getAccessRelation().release();

      accdom = isl_map_intersect_domain(accdom, domcp);

      if (ReductionArrays.count(MA->getScopArrayInfo())) {
        ReductionTagMap =
            isl_union_map_add_map(ReductionTagMap, isl_map_copy(accdom));
        accdom = isl_map_range_map(accdom);
      } else {
        accdom = tag(accdom, MA, Level);
        if (Level > Dependences::AL_Statement) {
          isl_map *StmtScheduleMap = Stmt.getSchedule().release();
          assert(StmtScheduleMap &&
                 "Schedules that contain extension nodes require special "
                 "handling.");
          isl_map *Schedule = tag(StmtScheduleMap, MA, Level);
          StmtSchedule = isl_union_map_add_map(StmtSchedule, Schedule);
        }
      }

      if (MA->isRead())
        Read = isl_union_map_add_map(Read, accdom);
      else if (MA->isMayWrite())
        MayWrite = isl_union_map_add_map(MayWrite, accdom);
      else if (MA->isMustWrite())
        MustWrite = isl_union_map_add_map(MustWrite, accdom);
      else if (MA->isKill())
        Kill = isl_union_map_add_map(Kill, accdom);
      else
        llvm_unreachable("unknown access type");
    }

    if (!ReductionArrays.empty() && Level == Dependences::AL_Statement)
      StmtSchedule =
          isl_union_map_add_map(StmtSchedule, Stmt.getSchedule().release());
  }

  StmtSchedule = isl_union_map_intersect_params(
      StmtSchedule, S.getAssumedContext().release());
  TaggedStmtDomain = isl_union_map_domain(StmtSchedule);

  ReductionTagMap = isl_union_map_coalesce(ReductionTagMap);
  Read = isl_union_map_coalesce(Read);
  MustWrite = isl_union_map_coalesce(MustWrite);
  MayWrite = isl_union_map_coalesce(MayWrite);
}

static void fixSetToZero(isl::set Zero, isl::union_set *User) {
  for (auto i : polly::rangeIslSize(0, Zero.tuple_dim()))
    Zero = Zero.fix_si(isl::dim::set, i, 0);
  *User = User->unite(Zero);
}

void Dependences::addPrivatizationDependences() {
  isl_union_map *PrivRAW, *PrivWAW, *PrivWAR;

  TC_RED = isl_union_map_transitive_closure(isl_union_map_copy(RED), nullptr);

  isl_union_set *UDeltas = isl_union_map_deltas(isl_union_map_copy(TC_RED));
  isl_union_set *Universe = isl_union_set_universe(isl_union_set_copy(UDeltas));
  isl::union_set Zero =
      isl::manage(isl_union_set_empty(isl_union_set_get_space(Universe)));

  for (isl::set Set : isl::manage_copy(Universe).get_set_list())
    fixSetToZero(Set, &Zero);

  isl_union_map *NonPositive =
      isl_union_set_lex_le_union_set(UDeltas, Zero.release());

  TC_RED = isl_union_map_subtract(TC_RED, NonPositive);

  TC_RED = isl_union_map_union(
      TC_RED, isl_union_map_reverse(isl_union_map_copy(TC_RED)));
  TC_RED = isl_union_map_coalesce(TC_RED);

  isl_union_map **Maps[] = {&RAW, &WAW, &WAR};
  isl_union_map **PrivMaps[] = {&PrivRAW, &PrivWAW, &PrivWAR};
  for (unsigned u = 0; u < 3; u++) {
    isl_union_map **Map = Maps[u], **PrivMap = PrivMaps[u];

    *PrivMap = isl_union_map_apply_range(isl_union_map_copy(*Map),
                                         isl_union_map_copy(TC_RED));
    *PrivMap = isl_union_map_union(
        *PrivMap, isl_union_map_apply_range(isl_union_map_copy(TC_RED),
                                            isl_union_map_copy(*Map)));

    *Map = isl_union_map_union(*Map, *PrivMap);
  }

  isl_union_set_free(Universe);
}

static __isl_give isl_union_flow *buildFlow(__isl_keep isl_union_map *Snk,
                                            __isl_keep isl_union_map *Src,
                                            __isl_keep isl_union_map *MaySrc,
                                            __isl_keep isl_union_map *Kill,
                                            __isl_keep isl_schedule *Schedule) {
  isl_union_access_info *AI;

  AI = isl_union_access_info_from_sink(isl_union_map_copy(Snk));
  if (MaySrc)
    AI = isl_union_access_info_set_may_source(AI, isl_union_map_copy(MaySrc));
  if (Src)
    AI = isl_union_access_info_set_must_source(AI, isl_union_map_copy(Src));
  if (Kill)
    AI = isl_union_access_info_set_kill(AI, isl_union_map_copy(Kill));
  AI = isl_union_access_info_set_schedule(AI, isl_schedule_copy(Schedule));
  auto *Flow = isl_union_access_info_compute_flow(AI);
  POLLY_DEBUG(if (!Flow) dbgs()
                  << "last error: "
                  << isl_ctx_last_error(isl_schedule_get_ctx(Schedule)) << " "
                  << isl_ctx_last_error_msg(isl_schedule_get_ctx(Schedule))
                  << '\n';);
  return Flow;
}

void Dependences::calculateDependences(Scop &S) {
  isl_union_map *Read, *MustWrite, *MayWrite, *ReductionTagMap, *Kill;
  isl_schedule *Schedule;
  isl_union_set *TaggedStmtDomain;

  collectInfo(S, Read, MustWrite, MayWrite, Kill, ReductionTagMap,
              TaggedStmtDomain, Level);

  bool HasReductions = !isl_union_map_is_empty(ReductionTagMap);

  Schedule = S.getScheduleTree().release();

  if (!HasReductions) {
    isl_union_map_free(ReductionTagMap);
    if (Level > AL_Statement) {
      auto *TaggedMap =
          isl_union_set_unwrap(isl_union_set_copy(TaggedStmtDomain));
      auto *Tags = isl_union_map_domain_map_union_pw_multi_aff(TaggedMap);
      Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
    }
  } else {
    isl_union_map *IdentityMap;
    isl_union_pw_multi_aff *ReductionTags, *IdentityTags, *Tags;

    ReductionTags =
        isl_union_map_domain_map_union_pw_multi_aff(ReductionTagMap);
    IdentityMap = isl_union_set_identity(isl_union_set_copy(TaggedStmtDomain));
    IdentityTags = isl_union_pw_multi_aff_from_union_map(IdentityMap);
    Tags = isl_union_pw_multi_aff_union_add(ReductionTags, IdentityTags);
    Schedule = isl_schedule_pullback_union_pw_multi_aff(Schedule, Tags);
  }

  isl_union_map *StrictWAW = nullptr;
  {
    polly::IslMaxOperationsGuard MaxOpGuard(IslCtx.get(), OptComputeOut);

    RAW = WAW = WAR = RED = nullptr;
    isl_union_map *Write = isl_union_map_union(isl_union_map_copy(MustWrite),
                                               isl_union_map_copy(MayWrite));

    isl_union_flow *Flow = buildFlow(Write, Write, Read, nullptr, Schedule);
    StrictWAW = isl_union_flow_get_must_dependence(Flow);
    isl_union_flow_free(Flow);

    if (OptAnalysisType == VALUE_BASED_ANALYSIS) {
      Flow = buildFlow(Read, MustWrite, MayWrite, nullptr, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, MustWrite, MayWrite, nullptr, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Read, MustWrite, Schedule);
      WAR = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);
    } else {
      Flow = buildFlow(Read, nullptr, Write, nullptr, Schedule);
      RAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Read, nullptr, Schedule);
      WAR = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);

      Flow = buildFlow(Write, nullptr, Write, nullptr, Schedule);
      WAW = isl_union_flow_get_may_dependence(Flow);
      isl_union_flow_free(Flow);
    }

    isl_union_map_free(Write);
    isl_union_map_free(MustWrite);
    isl_union_map_free(MayWrite);
    isl_union_map_free(Read);
    isl_union_map_free(Kill);
    isl_schedule_free(Schedule);

    RAW = isl_union_map_coalesce(RAW);
    WAW = isl_union_map_coalesce(WAW);
    WAR = isl_union_map_coalesce(WAR);
  }

  if (isl_ctx_last_error(IslCtx.get()) == isl_error_quota) {
    isl_union_map_free(RAW);
    isl_union_map_free(WAW);
    isl_union_map_free(WAR);
    isl_union_map_free(StrictWAW);
    RAW = WAW = WAR = StrictWAW = nullptr;
    isl_ctx_reset_error(IslCtx.get());
  }

  if (!HasReductions && Level == AL_Statement) {
    RED = isl_union_map_empty(isl_union_map_get_space(RAW));
    TC_RED = isl_union_map_empty(isl_union_set_get_space(TaggedStmtDomain));
    isl_union_set_free(TaggedStmtDomain);
    isl_union_map_free(StrictWAW);
    return;
  }

  isl_union_map *STMT_RAW, *STMT_WAW, *STMT_WAR;
  STMT_RAW = isl_union_map_intersect_domain(
      isl_union_map_copy(RAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAW = isl_union_map_intersect_domain(
      isl_union_map_copy(WAW), isl_union_set_copy(TaggedStmtDomain));
  STMT_WAR =
      isl_union_map_intersect_domain(isl_union_map_copy(WAR), TaggedStmtDomain);

  RED = isl_union_map_empty(isl_union_map_get_space(RAW));
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;
      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_map *Identity =
          isl_map_from_domain_and_range(isl_set_copy(AccDomW), AccDomW);
      RED = isl_union_map_add_map(RED, Identity);
    }
  }

  RED = isl_union_map_intersect(RED, isl_union_map_copy(RAW));
  RED = isl_union_map_intersect(RED, StrictWAW);

  if (!isl_union_map_is_empty(RED)) {
    RAW = isl_union_map_subtract(RAW, isl_union_map_copy(RED));
    WAW = isl_union_map_subtract(WAW, isl_union_map_copy(RED));
    WAR = isl_union_map_subtract(WAR, isl_union_map_copy(RED));
    addPrivatizationDependences();
  } else {
    TC_RED = isl_union_map_empty(isl_union_map_get_space(RED));
  }

  isl_union_map *RED_SIN = isl_union_map_empty(isl_union_map_get_space(RAW));

  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isReductionLike())
        continue;

      isl_set *AccDomW = isl_map_wrap(MA->getAccessRelation().release());
      isl_union_map *AccRedDepU = isl_union_map_intersect_domain(
          isl_union_map_copy(TC_RED), isl_union_set_from_set(AccDomW));
      if (isl_union_map_is_empty(AccRedDepU)) {
        isl_union_map_free(AccRedDepU);
        continue;
      }

      isl_map *AccRedDep = isl_map_from_union_map(AccRedDepU);
      RED_SIN = isl_union_map_add_map(RED_SIN, isl_map_copy(AccRedDep));
      AccRedDep = isl_map_zip(AccRedDep);
      AccRedDep = isl_set_unwrap(isl_map_domain(AccRedDep));
      setReductionDependences(MA, AccRedDep);
    }
  }

  assert(isl_union_map_is_equal(RED_SIN, TC_RED));
  isl_union_map_free(RED_SIN);

  RAW = isl_union_map_zip(RAW);
  WAW = isl_union_map_zip(WAW);
  WAR = isl_union_map_zip(WAR);
  RED = isl_union_map_zip(RED);
  TC_RED = isl_union_map_zip(TC_RED);

  RAW = isl_union_set_unwrap(isl_union_map_domain(RAW));
  WAW = isl_union_set_unwrap(isl_union_map_domain(WAW));
  WAR = isl_union_set_unwrap(isl_union_map_domain(WAR));
  RED = isl_union_set_unwrap(isl_union_map_domain(RED));
  TC_RED = isl_union_set_unwrap(isl_union_map_domain(TC_RED));

  RAW = isl_union_map_union(RAW, STMT_RAW);
  WAW = isl_union_map_union(WAW, STMT_WAW);
  WAR = isl_union_map_union(WAR, STMT_WAR);

  RAW = isl_union_map_coalesce(RAW);
  WAW = isl_union_map_coalesce(WAW);
  WAR = isl_union_map_coalesce(WAR);
  RED = isl_union_map_coalesce(RED);
  TC_RED = isl_union_map_coalesce(TC_RED);
}

bool Dependences::isValidSchedule(Scop &S, isl::schedule NewSched) const {
  StatementToIslMapTy NewSchedules;
  for (auto NewMap : NewSched.get_map().get_map_list()) {
    auto *Stmt = reinterpret_cast<ScopStmt *>(
        NewMap.get_tuple_id(isl::dim::in).get_user());
    NewSchedules[Stmt] = NewMap;
  }

  return isValidSchedule(S, NewSchedules);
}

bool Dependences::isValidSchedule(
    Scop &S, const StatementToIslMapTy &NewSchedule) const {
  if (LegalityCheckDisabled)
    return true;

  isl::union_map Dependences = getDependences(TYPE_RAW | TYPE_WAW | TYPE_WAR);
  isl::union_map Schedule = isl::union_map::empty(S.getIslCtx());

  isl::space ScheduleSpace;

  for (ScopStmt &Stmt : S) {
    isl::map StmtScat;

    auto Lookup = NewSchedule.find(&Stmt);
    if (Lookup == NewSchedule.end())
      StmtScat = Stmt.getSchedule();
    else
      StmtScat = Lookup->second;
    assert(!StmtScat.is_null() &&
           "Schedules that contain extension nodes require special handling.");

    if (ScheduleSpace.is_null())
      ScheduleSpace = StmtScat.get_space().range();

    Schedule = Schedule.unite(StmtScat);
  }

  Dependences = Dependences.apply_domain(Schedule);
  Dependences = Dependences.apply_range(Schedule);

  isl::set Zero = isl::set::universe(ScheduleSpace);
  for (auto i : polly::rangeIslSize(0, Zero.tuple_dim()))
    Zero = Zero.fix_si(isl::dim::set, i, 0);

  isl::union_set UDeltas = Dependences.deltas();
  isl::set Deltas = polly::singleton(UDeltas, ScheduleSpace);

  isl::space Space = Deltas.get_space();
  isl::map NonPositive = isl::map::universe(Space.map_from_set());
  NonPositive =
      NonPositive.lex_le_at(isl::multi_pw_aff::identity_on_domain(Space));
  NonPositive = NonPositive.intersect_domain(Deltas);
  NonPositive = NonPositive.intersect_range(Zero);

  return NonPositive.is_empty();
}

bool Dependences::isParallel(__isl_keep isl_union_map *Schedule,
                             __isl_take isl_union_map *Deps,
                             __isl_give isl_pw_aff **MinDistancePtr) const {
  isl_set *Deltas, *Distance;
  isl_map *ScheduleDeps;
  unsigned Dimension;
  bool IsParallel;

  Deps = isl_union_map_apply_range(Deps, isl_union_map_copy(Schedule));
  Deps = isl_union_map_apply_domain(Deps, isl_union_map_copy(Schedule));

  if (isl_union_map_is_empty(Deps)) {
    isl_union_map_free(Deps);
    return true;
  }

  ScheduleDeps = isl_map_from_union_map(Deps);
  Dimension = isl_map_dim(ScheduleDeps, isl_dim_out) - 1;

  for (unsigned i = 0; i < Dimension; i++)
    ScheduleDeps = isl_map_equate(ScheduleDeps, isl_dim_out, i, isl_dim_in, i);

  Deltas = isl_map_deltas(ScheduleDeps);
  Distance = isl_set_universe(isl_set_get_space(Deltas));

  for (unsigned i = 0; i < Dimension; i++)
    Distance = isl_set_fix_si(Distance, isl_dim_set, i, 0);

  Distance = isl_set_lower_bound_si(Distance, isl_dim_set, Dimension, 1);
  Distance = isl_set_intersect(Distance, Deltas);

  IsParallel = isl_set_is_empty(Distance);
  if (IsParallel || !MinDistancePtr) {
    isl_set_free(Distance);
    return IsParallel;
  }

  Distance = isl_set_project_out(Distance, isl_dim_set, 0, Dimension);
  Distance = isl_set_coalesce(Distance);

  *MinDistancePtr = isl_pw_aff_coalesce(isl_set_dim_min(Distance, 0));

  return false;
}

static void printDependencyMap(raw_ostream &OS, __isl_keep isl_union_map *DM) {
  if (DM) {
    printIslStr(OS, isl_union_map_to_str(DM));
    OS << "\n";
  } else {
    OS << "n/a\n";
  }
}

void Dependences::print(raw_ostream &OS) const {
  OS << "\tRAW dependences:\n\t\t";
  printDependencyMap(OS, RAW);
  OS << "\tWAR dependences:\n\t\t";
  printDependencyMap(OS, WAR);
  OS << "\tWAW dependences:\n\t\t";
  printDependencyMap(OS, WAW);
  OS << "\tReduction dependences:\n\t\t";
  printDependencyMap(OS, RED);
  OS << "\tTransitive closure of reduction dependences:\n\t\t";
  printDependencyMap(OS, TC_RED);
}

void Dependences::dump() const { print(dbgs()); }

void Dependences::releaseMemory() {
  isl_union_map_free(RAW);
  isl_union_map_free(WAR);
  isl_union_map_free(WAW);
  isl_union_map_free(RED);
  isl_union_map_free(TC_RED);

  RED = RAW = WAR = WAW = TC_RED = nullptr;

  for (auto &ReductionDeps : ReductionDependences)
    isl_map_free(ReductionDeps.second);
  ReductionDependences.clear();
}

isl::union_map Dependences::getDependences(int Kinds) const {
  assert(hasValidDependences() && "No valid dependences available");
  isl::space Space = isl::manage_copy(RAW).get_space();
  isl::union_map Deps = Deps.empty(Space.ctx());

  if (Kinds & TYPE_RAW)
    Deps = Deps.unite(isl::manage_copy(RAW));

  if (Kinds & TYPE_WAR)
    Deps = Deps.unite(isl::manage_copy(WAR));

  if (Kinds & TYPE_WAW)
    Deps = Deps.unite(isl::manage_copy(WAW));

  if (Kinds & TYPE_RED)
    Deps = Deps.unite(isl::manage_copy(RED));

  if (Kinds & TYPE_TC_RED)
    Deps = Deps.unite(isl::manage_copy(TC_RED));

  Deps = Deps.coalesce();
  Deps = Deps.detect_equalities();
  return Deps;
}

bool Dependences::hasValidDependences() const {
  return (RAW != nullptr) && (WAR != nullptr) && (WAW != nullptr);
}

__isl_give isl_map *
Dependences::getReductionDependences(MemoryAccess *MA) const {
  return isl_map_copy(ReductionDependences.lookup(MA));
}

void Dependences::setReductionDependences(MemoryAccess *MA,
                                          __isl_take isl_map *D) {
  assert(ReductionDependences.count(MA) == 0 &&
         "Reduction dependences set twice!");
  ReductionDependences[MA] = D;
}

static __isl_give isl_union_map *project_out_tags(
    __isl_take isl_union_map *umap) {
  return isl_union_map_domain_factor_domain(umap);
}

static void compute_tagger(struct ppcg_scop *ps) {
  isl_union_map *tagged;
  isl_union_pw_multi_aff *tagger;

  tagged = isl_union_map_copy(ps->tagged_reads);
  tagged = isl_union_map_union(tagged,
                               isl_union_map_copy(ps->tagged_may_writes));
  tagged = isl_union_map_union(tagged,
                               isl_union_map_copy(ps->tagged_must_kills));
  tagged = isl_union_map_universe(tagged);
  tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));

  tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);

  ps->tagger = tagger;
}

/// Remove intra-iteration tagged pairs `[Stmt[x] -> MA_a] -> [Stmt[x] -> MA_b]`
/// (where statement and iteration vector `x` are identical) so intra-statement
/// write-then-read kills cross-iteration writes without introducing reflexive
/// `Stmt[x] -> Stmt[x]` self-dependences.
static __isl_give isl_union_map *
subtractSameIterationTaggedDeps(__isl_take isl_union_map *deps,
                                __isl_keep isl_union_pw_multi_aff *tagger) {
  if (!deps || !tagger)
    return deps;
  isl_union_map *T =
      isl_union_map_from_union_pw_multi_aff(isl_union_pw_multi_aff_copy(tagger));
  isl_union_map *revT = isl_union_map_reverse(isl_union_map_copy(T));
  isl_union_map *sameIter = isl_union_map_apply_range(T, revT);
  return isl_union_map_subtract(deps, sameIter);
}

static __isl_give isl_schedule *getTaggedSchedule(struct ppcg_scop *ps) {
  if (ps->tagged_schedule)
    return isl_schedule_copy(ps->tagged_schedule);
  return isl_schedule_pullback_union_pw_multi_aff(
      isl_schedule_copy(ps->schedule), isl_union_pw_multi_aff_copy(ps->tagger));
}

/// Build an access-ordered schedule tree for `AL_Access` by pulling back
/// `ps->schedule` through `ps->tagger` and inserting an
/// `isl_schedule_node_sequence` at each multi-access leaf in exact `ScopStmt`
/// program order so `isl_flow` distinguishes read-before-write (reductions)
/// from write-before-read (intra-statement temporaries).
static __isl_give isl_schedule *
buildAccessOrderedTaggedSchedule(Scop &S, struct ppcg_scop *ps) {
  isl_schedule *sched = isl_schedule_pullback_union_pw_multi_aff(
      isl_schedule_copy(ps->schedule), isl_union_pw_multi_aff_copy(ps->tagger));
  if (!sched)
    return nullptr;

  struct OrderedAccess {
    isl_union_set *univ;
  };
  std::vector<OrderedAccess> ordered;
  DenseSet<isl_id *> seenIds;
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      if (!seenIds.insert(MA->getId().get()).second)
        continue;
      isl_set *stmtUniv = isl_set_universe(Stmt.getDomainSpace().release());
      isl_map *idMap = isl_set_identity(stmtUniv);
      isl_map *taggedMap = tag(idMap, MA->getId().release());
      isl_set *maSet = isl_map_domain(taggedMap);
      ordered.push_back({isl_union_set_from_set(maSet)});
    }
  }

  sched = isl_schedule_map_schedule_node_bottom_up(
      sched,
      [](__isl_take isl_schedule_node *node, void *user) -> isl_schedule_node * {
        if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
          return node;
        isl_union_set *leafDom = isl_schedule_node_get_universe_domain(node);
        if (isl_union_set_n_set(leafDom) <= 1) {
          isl_union_set_free(leafDom);
          return node;
        }
        auto *orderedPtr = static_cast<std::vector<OrderedAccess> *>(user);
        isl_ctx *ctx = isl_schedule_node_get_ctx(node);
        isl_union_set_list *filters =
            isl_union_set_list_alloc(ctx, orderedPtr->size());
        isl_union_set *covered =
            isl_union_set_empty(isl_union_set_get_space(leafDom));

        for (const OrderedAccess &oa : *orderedPtr) {
          if (isl_union_set_is_disjoint(leafDom, oa.univ) == isl_bool_false) {
            filters = isl_union_set_list_add(filters,
                                             isl_union_set_copy(oa.univ));
            covered =
                isl_union_set_union(covered, isl_union_set_copy(oa.univ));
          }
        }
        isl_union_set *rem = isl_union_set_subtract(leafDom, covered);
        if (isl_union_set_is_empty(rem) == isl_bool_false)
          filters = isl_union_set_list_add(filters, rem);
        else
          isl_union_set_free(rem);

        if (isl_union_set_list_n_union_set(filters) > 1)
          node = isl_schedule_node_insert_sequence(node, filters);
        else
          isl_union_set_list_free(filters);
        return node;
      },
      &ordered);

  for (OrderedAccess &oa : ordered)
    isl_union_set_free(oa.univ);
  return sched;
}

static void compute_live_out(struct ppcg_scop *ps) {
  isl_schedule *schedule;
  isl_union_map *kills;
  isl_union_map *exposed;
  isl_union_map *covering;
  isl_union_access_info *access;
  isl_union_flow *flow;

  schedule = isl_schedule_copy(ps->schedule);
  kills = isl_union_map_union(isl_union_map_copy(ps->must_writes),
                              isl_union_map_copy(ps->must_kills));
  access = isl_union_access_info_from_sink(kills);
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(ps->may_writes));
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  covering = isl_union_flow_get_full_may_dependence(flow);
  isl_union_flow_free(flow);

  covering = isl_union_map_range_factor_range(covering);
  exposed = isl_union_map_copy(ps->may_writes);
  exposed = isl_union_map_subtract(exposed, covering);
  ps->live_out = exposed;
}

static void compute_tagged_flow_dep_only(struct ppcg_scop *ps) {
  isl_schedule *schedule;
  isl_union_map *live_in;
  isl_union_access_info *access;
  isl_union_flow *flow;
  isl_union_map *must_source;
  isl_union_map *kills;
  isl_union_map *tagged_flow;

  schedule = getTaggedSchedule(ps);
  kills = isl_union_map_copy(ps->tagged_must_kills);
  must_source = isl_union_map_copy(ps->tagged_must_writes);
  kills = isl_union_map_union(kills, must_source);
  access = isl_union_access_info_from_sink(
      isl_union_map_copy(ps->tagged_reads));
  access = isl_union_access_info_set_kill(access, kills);
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(ps->tagged_may_writes));
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  tagged_flow = isl_union_flow_get_may_dependence(flow);
  tagged_flow = subtractSameIterationTaggedDeps(tagged_flow, ps->tagger);
  ps->tagged_dep_flow = tagged_flow;
  live_in = isl_union_flow_get_may_no_source(flow);
  ps->live_in = project_out_tags(live_in);
  isl_union_flow_free(flow);
}

static void derive_flow_dep_from_tagged_flow_dep(struct ppcg_scop *ps) {
  ps->dep_flow = isl_union_map_copy(ps->tagged_dep_flow);
  ps->dep_flow = isl_union_map_factor_domain(ps->dep_flow);
}

static void compute_tagged_flow_dep(struct ppcg_scop *ps) {
  compute_tagged_flow_dep_only(ps);
  derive_flow_dep_from_tagged_flow_dep(ps);
}

static void compute_order_dependences(struct ppcg_scop *ps) {
  isl_union_map *reads;
  isl_union_map *shared_access;
  isl_union_set *matched;
  isl_union_map *unmatched;
  isl_schedule *schedule;
  isl_union_access_info *access;
  isl_union_flow *flow;

  schedule = getTaggedSchedule(ps);
  reads = isl_union_map_copy(ps->tagged_reads);
  matched = isl_union_map_domain(isl_union_map_copy(ps->tagged_dep_flow));
  unmatched = isl_union_map_copy(ps->tagged_may_writes);
  unmatched = isl_union_map_subtract_domain(unmatched, matched);
  reads = isl_union_map_union(reads, unmatched);
  access = isl_union_access_info_from_sink(
      isl_union_map_copy(ps->tagged_may_writes));
  access = isl_union_access_info_set_may_source(access, reads);
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  shared_access = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);
  shared_access = subtractSameIterationTaggedDeps(shared_access, ps->tagger);

  ps->tagged_dep_order = isl_union_map_copy(shared_access);
  ps->dep_order = isl_union_map_factor_domain(shared_access);
}

static void compute_forced_dependences(struct ppcg_scop *ps) {
  isl_union_map *shared_access;
  isl_union_map *exposed;
  isl_union_map *live_in;
  isl_union_map *sink_access;
  isl_union_map *shared_sink;
  isl_union_access_info *access;
  isl_union_flow *flow;
  isl_schedule *schedule;

  exposed = isl_union_map_copy(ps->live_out);
  schedule = isl_schedule_copy(ps->schedule);
  access = isl_union_access_info_from_sink(exposed);
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(ps->may_writes));
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  shared_access = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);
  ps->dep_forced = shared_access;

  schedule = isl_schedule_copy(ps->schedule);
  access = isl_union_access_info_from_sink(
      isl_union_map_copy(ps->may_writes));
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(ps->live_in));
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  live_in = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);

  ps->dep_forced = isl_union_map_union(ps->dep_forced, live_in);
  ps->dep_forced = isl_union_map_subtract(
      ps->dep_forced, isl_union_map_copy(ps->independence));

  schedule = isl_schedule_copy(ps->schedule);
  sink_access = isl_union_map_copy(ps->tagged_dep_flow);
  sink_access = isl_union_map_range_product(
      sink_access, isl_union_map_copy(ps->tagged_may_writes));
  sink_access = isl_union_map_domain_factor_domain(sink_access);
  access = isl_union_access_info_from_sink(isl_union_map_copy(sink_access));
  access = isl_union_access_info_set_may_source(access, sink_access);
  access = isl_union_access_info_set_schedule(access, schedule);
  flow = isl_union_access_info_compute_flow(access);
  shared_sink = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);
  ps->dep_forced = isl_union_map_union(ps->dep_forced, shared_sink);
}

static void remove_independences_from_tagged_flow(struct ppcg_scop *ps) {
  isl_union_map *tf;
  isl_union_set *indep;
  isl_union_set *mw;

  tf = isl_union_map_copy(ps->tagged_dep_flow);
  tf = isl_union_map_zip(tf);
  indep = isl_union_map_wrap(isl_union_map_copy(ps->independence));
  tf = isl_union_map_intersect_domain(tf, indep);
  tf = isl_union_map_zip(tf);
  mw = isl_union_map_domain(isl_union_map_copy(ps->tagged_must_writes));
  tf = isl_union_map_subtract_domain(tf, mw);
  ps->tagged_dep_flow = isl_union_map_subtract(ps->tagged_dep_flow, tf);
}

static void compute_live_range_reordering_dependences(struct ppcg_scop *ps) {
  compute_tagged_flow_dep_only(ps);
  remove_independences_from_tagged_flow(ps);
  derive_flow_dep_from_tagged_flow_dep(ps);
  compute_order_dependences(ps);
  compute_forced_dependences(ps);
}

static void compute_flow_dep(struct ppcg_scop *ps) {
  isl_union_access_info *access;
  isl_union_flow *flow;
  isl_union_map *kills, *must_writes;

  access = isl_union_access_info_from_sink(isl_union_map_copy(ps->reads));
  kills = isl_union_map_copy(ps->must_kills);
  must_writes = isl_union_map_copy(ps->must_writes);
  kills = isl_union_map_union(kills, must_writes);
  access = isl_union_access_info_set_kill(access, kills);
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(ps->may_writes));
  access = isl_union_access_info_set_schedule(access,
                                              isl_schedule_copy(ps->schedule));
  flow = isl_union_access_info_compute_flow(access);

  ps->dep_flow = isl_union_flow_get_may_dependence(flow);
  ps->live_in = isl_union_flow_get_may_no_source(flow);
  isl_union_flow_free(flow);
}

static void compute_async_dependences(struct ppcg_scop *scop) {
  isl_union_access_info *access;
  access = isl_union_access_info_from_sink(isl_union_map_copy(scop->reads));
  access = isl_union_access_info_set_kill(
      access, isl_union_map_union(isl_union_map_copy(scop->must_writes),
                                  isl_union_map_copy(scop->must_kills)));
  access = isl_union_access_info_set_may_source(
      access, isl_union_map_copy(scop->async_must_writes));
  access = isl_union_access_info_set_schedule(
      access, isl_schedule_copy(scop->schedule));
  isl_union_flow *flow = isl_union_access_info_compute_flow(access);

  scop->dep_async = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);
}

static void compute_dependences(struct ppcg_scop *scop) {
  isl_union_map *may_source;
  isl_union_access_info *access;
  isl_union_flow *flow;

  if (!scop)
    return;

  compute_async_dependences(scop);
  compute_live_out(scop);

  if (scop->options->live_range_reordering)
    compute_live_range_reordering_dependences(scop);
  else if (scop->options->target != PPCG_TARGET_C)
    compute_tagged_flow_dep(scop);
  else
    compute_flow_dep(scop);

  if (scop->tagged_schedule) {
    may_source = isl_union_map_union(isl_union_map_copy(scop->tagged_may_writes),
                                     isl_union_map_copy(scop->tagged_reads));
    access = isl_union_access_info_from_sink(
        isl_union_map_copy(scop->tagged_may_writes));
    access = isl_union_access_info_set_kill(
        access, isl_union_map_copy(scop->tagged_must_writes));
    access = isl_union_access_info_set_may_source(access, may_source);
    access = isl_union_access_info_set_schedule(
        access, isl_schedule_copy(scop->tagged_schedule));
    flow = isl_union_access_info_compute_flow(access);
    isl_union_map *tagged_false = isl_union_flow_get_may_dependence(flow);
    isl_union_flow_free(flow);
    tagged_false = subtractSameIterationTaggedDeps(tagged_false, scop->tagger);
    scop->dep_false =
        isl_union_map_coalesce(isl_union_map_factor_domain(tagged_false));
  } else {
    may_source = isl_union_map_union(isl_union_map_copy(scop->may_writes),
                                     isl_union_map_copy(scop->reads));
    access = isl_union_access_info_from_sink(
        isl_union_map_copy(scop->may_writes));
    access = isl_union_access_info_set_kill(
        access, isl_union_map_copy(scop->must_writes));
    access = isl_union_access_info_set_may_source(access, may_source);
    access = isl_union_access_info_set_schedule(
        access, isl_schedule_copy(scop->schedule));
    flow = isl_union_access_info_compute_flow(access);

    scop->dep_false = isl_union_flow_get_may_dependence(flow);
    scop->dep_false = isl_union_map_coalesce(scop->dep_false);
    isl_union_flow_free(flow);
  }
}

static void report_dead_code(struct ppcg_scop *ps,
                             __isl_keep isl_union_set *live) {
  if (!ps->options->debug->verbose)
    return;
  if (isl_union_set_is_equal(ps->domain, live))
    return;

  isl_ctx *ctx = isl_union_set_get_ctx(live);
  isl_union_set *dead = isl_union_set_subtract(isl_union_set_copy(ps->domain),
                                               isl_union_set_copy(live));

  isl_printer *p = isl_printer_to_file(ctx, stderr);
  p = isl_printer_print_str(p, "Eliminated dead instances: ");
  p = isl_printer_print_union_set(p, dead);
  p = isl_printer_end_line(p);
  isl_printer_free(p);

  isl_union_set_free(dead);
}

static void eliminate_dead_code(struct ppcg_scop *ps) {
  isl_union_set *live;
  isl_union_map *dep;
  isl_union_pw_multi_aff *tagger;

  live = isl_union_map_domain(isl_union_map_copy(ps->live_out));
  if (ps->call && !isl_union_set_is_empty(ps->call)) {
    live = isl_union_set_union(live, isl_union_set_copy(ps->call));
    live = isl_union_set_coalesce(live);
  }

  dep = isl_union_map_copy(ps->dep_flow);
  dep = isl_union_map_reverse(dep);

  for (;;) {
    isl_union_set *extra =
        isl_union_set_apply(isl_union_set_copy(live), isl_union_map_copy(dep));
    if (isl_union_set_is_subset(extra, live)) {
      isl_union_set_free(extra);
      break;
    }

    live = isl_union_set_union(live, extra);
    live = isl_union_set_affine_hull(live);
    live = isl_union_set_intersect(live, isl_union_set_copy(ps->domain));
  }

  isl_union_map_free(dep);

  report_dead_code(ps, live);

  ps->domain = isl_union_set_intersect(ps->domain, isl_union_set_copy(live));
  ps->schedule =
      isl_schedule_intersect_domain(ps->schedule, isl_union_set_copy(live));
  ps->dep_flow =
      isl_union_map_intersect_range(ps->dep_flow, isl_union_set_copy(live));
  tagger = isl_union_pw_multi_aff_copy(ps->tagger);
  live = isl_union_set_preimage_union_pw_multi_aff(live, tagger);
  ps->tagged_dep_flow = isl_union_map_intersect_range(ps->tagged_dep_flow, live);
}

static bool gpu_array_can_be_private(ScopArrayInfo &sai) {
  auto ty = dyn_cast<mlir::MemRefType>(sai.val.getType());
  if (!ty)
    return false;
  if (!ty.getMemorySpace())
    return true;
  if (auto intAttr = dyn_cast<mlir::IntegerAttr>(ty.getMemorySpace())) {
    int64_t space = intAttr.getInt();
    return space == 0 || space == 1 || space == 5;
  }
  return false;
}

static void collect_order_dependences(Scop &S, ppcg_scop *scop) {
  isl_space *space = isl_union_map_get_space(scop->reads);
  isl_union_map *array_order = isl_union_map_empty(space);

  isl_union_map *accesses = isl_union_map_copy(scop->tagged_reads);
  accesses = isl_union_map_union(accesses,
                                 isl_union_map_copy(scop->tagged_may_writes));
  accesses = isl_union_map_universe(accesses);

  for (auto &array : S.arrays) {
    isl_set *set = isl_set_universe(array.space.copy());
    isl_union_set *uset = isl_union_set_from_set(set);
    uset = isl_union_map_domain(
        isl_union_map_intersect_range(isl_union_map_copy(accesses), uset));
    isl_union_map *order = isl_union_map_copy(scop->tagged_dep_order);
    order = isl_union_map_intersect_domain(order, uset);
    order = isl_union_map_zip(order);
    order = isl_union_set_unwrap(isl_union_map_domain(order));
    order = isl_union_map_subtract(order, S.getIndependence().release());
    array.dep_order = isl::manage(order);

    if (gpu_array_can_be_private(array))
      continue;

    array_order = isl_union_map_union(array_order, array.dep_order.copy());
  }

  isl_union_map_free(accesses);
  scop->array_order = array_order;
}

static ppcg_scop *computeDeps(Scop &S,
                              polymer::Dependences::AnalysisLevel level) {
  isl_union_map *ReductionTagMap1 = nullptr;
  isl_union_set *TaggedStmtDomain1 = nullptr;
  isl_union_map *ReductionTagMap2 = nullptr;
  isl_union_set *TaggedStmtDomain2 = nullptr;

  bool verbose = false;
  LLVM_DEBUG(verbose = true);
  ppcg_debug_options *debug_options = new ppcg_debug_options{verbose};
  ppcg_options *options =
      new ppcg_options{1, PPCG_TARGET_CUDA, debug_options};
  ppcg_scop *ps = new ppcg_scop{};
  ps->options = options;

  collectInfo(S, ps->reads, ps->must_writes, ps->may_writes, ps->must_kills,
              ReductionTagMap1, TaggedStmtDomain1,
              polymer::Dependences::AL_Statement);
  isl_union_map_free(ReductionTagMap1);
  isl_union_set_free(TaggedStmtDomain1);

  collectInfo(S, ps->tagged_reads, ps->tagged_must_writes,
              ps->tagged_may_writes, ps->tagged_must_kills, ReductionTagMap2,
              TaggedStmtDomain2, level);
  isl_union_map_free(ReductionTagMap2);
  isl_union_set_free(TaggedStmtDomain2);

  ps->may_writes =
      isl_union_map_union(isl_union_map_copy(ps->must_writes), ps->may_writes);
  ps->tagged_may_writes = isl_union_map_union(
      isl_union_map_copy(ps->tagged_must_writes), ps->tagged_may_writes);

  collectAsyncCopyInfo(S, ps->async_must_writes, ps->async_reads);

  ps->schedule = S.getScheduleTree().release();
  ps->domain = S.getScheduleTree().get_domain().release();
  ps->call = isl_union_set_empty(isl_union_set_get_space(ps->domain));
  ps->context =
      isl_set_universe(S.getScheduleTree().get_domain().get_space().release());
  ps->independence = S.getIndependence().release();

  compute_tagger(ps);
  if (level == polymer::Dependences::AL_Access)
    ps->tagged_schedule = buildAccessOrderedTaggedSchedule(S, ps);
  compute_dependences(ps);
  eliminate_dead_code(ps);
  collect_order_dependences(S, ps);

  return ps;
}

namespace polymer {

void freePpcgScop(ppcg_scop *ps) {
  if (!ps)
    return;
  if (ps->options) {
    delete ps->options->debug;
    delete ps->options;
  }
  isl_set_free(ps->context);
  isl_union_set_free(ps->domain);
  isl_union_set_free(ps->call);
  isl_union_map_free(ps->tagged_reads);
  isl_union_map_free(ps->atagged_reads);
  isl_union_map_free(ps->reads);
  isl_union_map_free(ps->live_in);
  isl_union_map_free(ps->tagged_may_writes);
  isl_union_map_free(ps->atagged_may_writes);
  isl_union_map_free(ps->may_writes);
  isl_union_map_free(ps->tagged_must_writes);
  isl_union_map_free(ps->atagged_must_writes);
  isl_union_map_free(ps->must_writes);
  isl_union_map_free(ps->live_out);
  isl_union_map_free(ps->tagged_must_kills);
  isl_union_map_free(ps->atagged_must_kills);
  isl_union_map_free(ps->must_kills);
  isl_union_map_free(ps->async_reads);
  isl_union_map_free(ps->async_must_writes);
  isl_union_pw_multi_aff_free(ps->tagger);
  isl_union_pw_multi_aff_free(ps->atagger);
  isl_union_map_free(ps->independence);
  isl_union_map_free(ps->dep_flow);
  isl_union_map_free(ps->tagged_dep_flow);
  isl_union_map_free(ps->atagged_dep_flow);
  isl_union_map_free(ps->dep_false);
  isl_union_map_free(ps->dep_forced);
  isl_union_map_free(ps->dep_order);
  isl_union_map_free(ps->tagged_dep_order);
  isl_union_map_free(ps->dep_async);
  isl_schedule_free(ps->schedule);
  isl_schedule_free(ps->tagged_schedule);
  isl_union_map_free(ps->array_order);
  delete ps;
}

ppcg_scop *computeDeps(Scop &S) {
  ppcg_scop *ps = ::computeDeps(S, polymer::Dependences::AL_Access);

  isl_union_map *redTagMap = nullptr;
  isl_union_set *taggedStmtDom = nullptr;
  collectInfo(S, ps->atagged_reads, ps->atagged_must_writes,
              ps->atagged_may_writes, ps->atagged_must_kills, redTagMap,
              taggedStmtDom, polymer::Dependences::AL_Reference);
  isl_union_map_free(redTagMap);
  isl_union_set_free(taggedStmtDom);
  ps->atagged_may_writes = isl_union_map_union(
      isl_union_map_copy(ps->atagged_must_writes), ps->atagged_may_writes);

  isl_union_map *atagged = isl_union_map_copy(ps->atagged_reads);
  atagged =
      isl_union_map_union(atagged, isl_union_map_copy(ps->atagged_may_writes));
  atagged =
      isl_union_map_union(atagged, isl_union_map_copy(ps->atagged_must_kills));
  atagged = isl_union_map_universe(atagged);
  atagged = isl_union_set_unwrap(isl_union_map_domain(atagged));
  ps->atagger = isl_union_map_domain_map_union_pw_multi_aff(atagged);

  // Derive `atagged_dep_flow` (`AL_Reference`) directly from `tagged_dep_flow`
  // (`AL_Access`) by retagging each access ID `[Stmt[x] -> MA_id]` to its array
  // ID `[Stmt[x] -> Arr_id]` so intra-statement read+write reductions preserve
  // their cross-iteration array live-range span (`lrs`).
  isl_union_map *retag = isl_union_map_empty(S.getParamSpace().release());
  for (ScopStmt &Stmt : S) {
    for (MemoryAccess *MA : Stmt) {
      isl_set *stmtUniv = isl_set_universe(Stmt.getDomainSpace().release());
      isl_map *idStmt = isl_set_identity(stmtUniv);
      isl_map *mAccess = tag(isl_map_copy(idStmt), MA->getId().release());
      isl_map *mArray = tag(idStmt, MA->getArrayId().release());
      isl_map *mapAccToArr =
          isl_map_apply_range(mAccess, isl_map_reverse(mArray));
      retag = isl_union_map_add_map(retag, mapAccToArr);
    }
  }
  retag = isl_union_map_coalesce(retag);
  ps->atagged_dep_flow = isl_union_map_apply_domain(
      isl_union_map_copy(ps->tagged_dep_flow), isl_union_map_copy(retag));
  ps->atagged_dep_flow =
      isl_union_map_apply_range(ps->atagged_dep_flow, retag);
  ps->atagged_dep_flow = isl_union_map_coalesce(ps->atagged_dep_flow);

  return ps;
}

} // namespace polymer
