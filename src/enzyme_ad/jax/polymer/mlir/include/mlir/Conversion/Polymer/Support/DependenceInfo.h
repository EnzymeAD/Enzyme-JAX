//===--- DependenceInfo.h - Polyhedral dependency analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYMER_SUPPORT_DEPENDENCE_INFO_H
#define POLYMER_SUPPORT_DEPENDENCE_INFO_H

#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "llvm/ADT/DenseMap.h"
#include "isl/ctx.h"
#include "isl/isl-noexceptions.h"

namespace polymer {

typedef IslScop Scop;

class Dependences final {
public:
  enum AnalysisLevel {
    AL_Statement = 0,
    AL_Reference,
    AL_Access,
    NumAnalysisLevels
  };

  using ReductionDependencesMapTy = llvm::DenseMap<MemoryAccess *, isl_map *>;
  using StatementToIslMapTy = llvm::DenseMap<ScopStmt *, isl::map>;

  enum Type {
    TYPE_WAR = 1 << 0,
    TYPE_RAW = 1 << 1,
    TYPE_WAW = 1 << 2,
    TYPE_RED = 1 << 3,
    TYPE_TC_RED = 1 << 4,
  };

  const std::shared_ptr<isl_ctx> &getSharedIslCtx() const { return IslCtx; }

  isl::union_map getDependences(int Kinds) const;
  bool hasValidDependences() const;
  __isl_give isl_map *getReductionDependences(MemoryAccess *MA) const;

  const ReductionDependencesMapTy &getReductionDependences() const {
    return ReductionDependences;
  }

  bool isParallel(__isl_keep isl_union_map *Schedule,
                  __isl_take isl_union_map *Deps,
                  __isl_give isl_pw_aff **MinDistancePtr = nullptr) const;

  bool isValidSchedule(Scop &S, const StatementToIslMapTy &NewSchedules) const;
  bool isValidSchedule(Scop &S, isl::schedule NewSched) const;

  void print(llvm::raw_ostream &OS) const;
  void dump() const;

  AnalysisLevel getDependenceLevel() const { return Level; }

  ~Dependences() { releaseMemory(); }

  explicit Dependences(const std::shared_ptr<isl_ctx> &IslCtx,
                       AnalysisLevel Level)
      : RAW(nullptr), WAR(nullptr), WAW(nullptr), RED(nullptr), TC_RED(nullptr),
        IslCtx(IslCtx), Level(Level) {}

  void addPrivatizationDependences();
  void calculateDependences(Scop &S);

private:
  void setReductionDependences(MemoryAccess *MA, __isl_take isl_map *Deps);
  void releaseMemory();

  isl_union_map *RAW;
  isl_union_map *WAR;
  isl_union_map *WAW;
  isl_union_map *RED;
  isl_union_map *TC_RED;
  ReductionDependencesMapTy ReductionDependences;
  std::shared_ptr<isl_ctx> IslCtx;
  const AnalysisLevel Level;
};

} // namespace polymer

#define PPCG_TARGET_C 0
#define PPCG_TARGET_CUDA 1
#define PPCG_TARGET_OPENCL 2

struct ppcg_debug_options {
  int verbose;
};

struct pet_scop {};
struct ppcg_options {
  int live_range_reordering;
  int target;
  struct ppcg_debug_options *debug;
};

struct ppcg_scop {
  struct ppcg_options *options;

  unsigned start;
  unsigned end;

  isl_set *context;
  isl_union_set *domain;
  isl_union_set *call;
  isl_union_map *tagged_reads;
  isl_union_map *atagged_reads;
  isl_union_map *reads;
  isl_union_map *live_in;
  isl_union_map *tagged_may_writes;
  isl_union_map *atagged_may_writes;
  isl_union_map *may_writes;
  isl_union_map *tagged_must_writes;
  isl_union_map *atagged_must_writes;
  isl_union_map *must_writes;
  isl_union_map *live_out;
  isl_union_map *tagged_must_kills;
  isl_union_map *atagged_must_kills;
  isl_union_map *must_kills;
  isl_union_map *async_reads;
  isl_union_map *async_must_writes;

  isl_union_pw_multi_aff *tagger;
  isl_union_pw_multi_aff *atagger;

  isl_union_map *independence;

  isl_union_map *dep_flow;
  isl_union_map *tagged_dep_flow;
  isl_union_map *atagged_dep_flow;
  isl_union_map *dep_false;
  isl_union_map *dep_forced;
  isl_union_map *dep_order;
  isl_union_map *tagged_dep_order;
  isl_union_map *dep_async;
  isl_schedule *schedule;
  isl_schedule *tagged_schedule;

  isl_id_to_ast_expr *names;

  struct pet_scop *pet;

  isl_union_map *array_order;
};

namespace polymer {
ppcg_scop *computeDeps(Scop &S);
void freePpcgScop(ppcg_scop *ps);
} // namespace polymer

#endif // POLYMER_SUPPORT_DEPENDENCE_INFO_H
