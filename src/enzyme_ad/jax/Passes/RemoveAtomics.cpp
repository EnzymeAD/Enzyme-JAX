//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements a pass to raise operations to arith dialect.
//===---------------------------------------------------------------------===//

#ifndef DISABLE_POLYMER
#include "../polymer/mlir/include/mlir/Conversion/Polymer/Support/IslScop.h"
#include "../polymer/mlir/include/mlir/Conversion/Polymer/Target/ISL.h"
#endif
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "polly/Support/GICHelper.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "llvm/Support/DebugLog.h"
#include <isl/isl-noexceptions.h>
#include <isl/union_map.h>
#include <isl/union_map_type.h>
#include <llvm/ADT/SmallVector.h>

#define DEBUG_TYPE "remove-atomics"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_REMOVEATOMICSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#ifndef DISABLE_POLYMER
using namespace polymer;

#define LDBG_ISL_DUMP(OBJ)                                                     \
  do {                                                                         \
    LDBG() << #OBJ;                                                            \
    LLVM_DEBUG(polly::dumpIslObj(OBJ));                                        \
  } while (0)

static inline void islAssert(const isl_size &size) {
  assert(size != isl_size_error);
}
[[maybe_unused]] static inline unsigned
unsignedFromIslSize(const isl::size &size) {
  if (size.is_error())
    llvm_unreachable("unrecoverable isl error");
  return static_cast<unsigned>(size);
}
[[maybe_unused]] static inline unsigned
unsignedFromIslSize(const isl_size &size) {
  islAssert(size);
  return static_cast<unsigned>(size);
}

// Checks for any racy execution of two statements.
bool isAnyInstanceRacy(isl::schedule_node node, isl::union_set domA,
                       isl::union_set domB) {
  LDBG_ISL_DUMP(node);

  if (node.domain().intersect(domA).is_empty())
    return false;
  if (node.domain().intersect(domB).is_empty())
    return false;

  if (node.isa<isl::schedule_node_band>()) {
    auto band = node.as<isl::schedule_node_band>();

    if (band.get_permutable())
      return true;

    unsigned members = unsignedFromIslSize(band.n_member());
    for (unsigned i = 0; i < members; i++) {
      isl::schedule_node child = band.child(i);
      if (isAnyInstanceRacy(child, domA, domB))
        return true;
    }
    return false;
  } else if (node.isa<isl::schedule_node_sequence>()) {
    auto seq = node.as<isl::schedule_node_sequence>();
    unsigned members = unsignedFromIslSize(seq.n_children());
    for (unsigned i = 0; i < members; i++) {
      isl::schedule_node child = seq.child(i);
      if (isAnyInstanceRacy(child, domA, domB))
        return true;
    }
    return false;
  } else if (node.isa<isl::schedule_node_filter>()) {
    auto filter = node.as<isl::schedule_node_filter>();
    return isAnyInstanceRacy(filter.child(0),
                             domA.intersect(filter.get_filter()),
                             domB.intersect(filter.get_filter()));
  } else if (node.isa<isl::schedule_node_leaf>()) {
    return false;
  }
  llvm_unreachable("unexpected schedule node type");
}

bool isAnyInstanceRacy(ScopStmt &stmtA, ScopStmt &stmtB) {
  LDBG() << "isAnyInstanceRacy";
  LLVM_DEBUG(polly::dumpIslObj(stmtA.getSchedule()));
  LLVM_DEBUG(polly::dumpIslObj(stmtB.getSchedule()));
  isl::schedule schedule = stmtA.getParent()->getScheduleTree();
  LDBG_ISL_DUMP(schedule);
  return isAnyInstanceRacy(schedule.get_root().child(0), stmtA.getDomain(),
                           stmtB.getDomain());
}

// This is copied from polly::Dependences::isParallel:
//
// Check if the current scheduling dimension is parallel.
//
// We check for parallelism by verifying that the loop does not carry any
// dependences.
//
// Parallelism test: if the distance is zero in all outer dimensions, then it
// has to be zero in the current dimension as well.
//
// Implementation: first, translate dependences into time space, then force
// outer dimensions to be equal. If the distance is zero in the current
// dimension, then the loop is parallel. The distance is zero in the current
// dimension if it is a subset of a map with equal values for the current
// dimension.
bool isParallel(__isl_keep isl_union_map *Schedule,
                __isl_take isl_union_map *Deps) {
  LDBG_ISL_DUMP(Schedule);
  LDBG_ISL_DUMP(Deps);
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

  // [0, ..., 0, +] - All zeros and last dimension larger than zero
  for (unsigned i = 0; i < Dimension; i++)
    Distance = isl_set_fix_si(Distance, isl_dim_set, i, 0);

  Distance = isl_set_lower_bound_si(Distance, isl_dim_set, Dimension, 1);
  Distance = isl_set_intersect(Distance, Deltas);

  IsParallel = isl_set_is_empty(Distance);
  isl_set_free(Distance);
  return IsParallel;
}

// Gather read-after-write, write-after-write and write-after-read dependencies
// of two statements.
// TODO check for leaks
isl::union_map getDeps(ScopStmt &stmtA, MemoryAccess *accessA, ScopStmt &stmtB,
                       MemoryAccess *accessB) {
  isl_space *Space = stmtA.getParent()->getParamSpace().release();
  isl_union_map *Read = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *MayWrite = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *MustWrite = isl_union_map_empty(isl_space_copy(Space));
  isl_union_map *Kill = isl_union_map_empty(isl_space_copy(Space));
  // isl_union_map *StmtSchedule = isl_union_map_empty(Space);

  for (auto [MA, Stmt] :
       llvm::zip(SmallVector<MemoryAccess *>{accessA, accessB},
                 SmallVector<ScopStmt *>{&stmtA, &stmtB})) {
    isl_set *domcp = Stmt->getDomain().release();
    isl_map *accdom = MA->getAccessRelation().release();

    accdom = isl_map_intersect_domain(accdom, domcp);
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

  Read = isl_union_map_coalesce(Read);
  MustWrite = isl_union_map_coalesce(MustWrite);
  MayWrite = isl_union_map_coalesce(MayWrite);
  isl_union_map *Write = isl_union_map_union(isl_union_map_copy(MustWrite),
                                             isl_union_map_copy(MayWrite));
  Write = isl_union_map_coalesce(Write);

  isl_schedule *Schedule = stmtA.getParent()->getScheduleTree().release();

  auto buildFlow = [&](isl_union_map *Sink, isl_union_map *MaySource,
                       isl_union_map *MustSource, isl_union_map *Kill) {
    isl_union_access_info *AI =
        isl_union_access_info_from_sink(isl_union_map_copy(Sink));
    if (MaySource)
      AI = isl_union_access_info_set_may_source(AI,
                                                isl_union_map_copy(MaySource));
    if (MustSource)
      AI = isl_union_access_info_set_must_source(
          AI, isl_union_map_copy(MustSource));
    if (Kill)
      AI = isl_union_access_info_set_kill(AI, isl_union_map_copy(Kill));
    AI = isl_union_access_info_set_schedule(AI, isl_schedule_copy(Schedule));
    isl_union_flow *Flow = isl_union_access_info_compute_flow(AI);
    LLVM_DEBUG(if (!Flow) llvm::dbgs()
                   << "last error: "
                   << isl_ctx_last_error(isl_schedule_get_ctx(Schedule)) << " "
                   << isl_ctx_last_error_msg(isl_schedule_get_ctx(Schedule))
                   << '\n';);
    return Flow;
  };

  LDBG_ISL_DUMP(MayWrite);
  LDBG_ISL_DUMP(MustWrite);
  LDBG_ISL_DUMP(Read);
  LDBG_ISL_DUMP(Kill);

  isl_union_flow *Flow = buildFlow(Read, MayWrite, MustWrite, nullptr);
  isl_union_map *raw = isl_union_flow_get_may_dependence(Flow);
  isl_union_flow_free(Flow);

  Flow = buildFlow(Write, MayWrite, MustWrite, nullptr);
  isl_union_map *waw = isl_union_flow_get_may_dependence(Flow);
  isl_union_flow_free(Flow);

  Flow = buildFlow(Write, Read, nullptr, MustWrite);
  isl_union_map *war = isl_union_flow_get_may_dependence(Flow);
  isl_union_flow_free(Flow);

  LDBG() << "RAW\n";
  LLVM_DEBUG(isl_union_map_dump(raw));
  LDBG() << "WAW\n";
  LLVM_DEBUG(isl_union_map_dump(waw));
  LDBG() << "WAR\n";
  LLVM_DEBUG(isl_union_map_dump(war));

  isl_union_map *deps = isl_union_map_union(raw, waw);
  deps = isl_union_map_union(deps, war);
  deps = isl_union_map_coalesce(deps);

  isl_union_map_free(Read);
  isl_union_map_free(MayWrite);
  isl_union_map_free(MustWrite);
  isl_union_map_free(Kill);
  isl_union_map_free(Write);
  isl_schedule_free(Schedule);

  return isl::manage(deps);
}

// Determine whether the accesses with dependency `deps` are racy. This is done
// by progressively checking for each shared parent parallel loop, from outer to
// inner, whether there are any carried dependencies using the `isParallel`
// function.
bool isAccessRacy(isl::schedule_node node, isl::union_map deps) {
  LDBG_ISL_DUMP(node);

  if (deps.is_empty())
    return false;

  if (node.isa<isl::schedule_node_band>()) {
    auto band = node.as<isl::schedule_node_band>();

    bool isPermutable = band.get_permutable();
    if (isPermutable) {
      auto prefix = node.child(0).get_prefix_schedule_union_map();
      if (!isParallel(prefix.get(), deps.copy())) {
        LDBG() << "Detected race!";
        return true;
      }
    }

    unsigned members = unsignedFromIslSize(band.n_member());
    for (unsigned i = 0; i < members; i++) {
      isl::schedule_node child = band.child(i);
      if (isAccessRacy(child, deps))
        return true;
    }
    return false;
  } else if (node.isa<isl::schedule_node_sequence>()) {
    auto seq = node.as<isl::schedule_node_sequence>();
    unsigned members = unsignedFromIslSize(seq.n_children());
    for (unsigned i = 0; i < members; i++) {
      isl::schedule_node child = seq.child(i);
      if (isAccessRacy(child, deps))
        return true;
    }
    return false;
  } else if (node.isa<isl::schedule_node_filter>()) {
    auto filter = node.as<isl::schedule_node_filter>();
    return isAccessRacy(filter.child(0),
                        deps.intersect_domain(filter.get_filter()));
  } else if (node.isa<isl::schedule_node_leaf>()) {
    return false;
  }
  llvm_unreachable("unexpected schedule node type");
}

bool isAccessRacy(ScopStmt &stmtA, MemoryAccess *accessA, ScopStmt &stmtB,
                  MemoryAccess *accessB) {
  LDBG() << "isAccessRacy";
  LLVM_DEBUG(polly::dumpIslObj(stmtA.getSchedule()));
  LLVM_DEBUG(polly::dumpIslObj(stmtB.getSchedule()));

  isl::union_map deps = getDeps(stmtA, accessA, stmtB, accessB);

  isl::schedule schedule = stmtA.getParent()->getScheduleTree();
  LDBG_ISL_DUMP(schedule);

  LLVM_DEBUG({
    isl::union_set domain = stmtA.getDomain().to_union_set().unite(
        stmtB.getDomain().to_union_set());
    isl::schedule intersectedSchedule = schedule.intersect_domain(domain);
    LDBG_ISL_DUMP(intersectedSchedule);
  });

  return isAccessRacy(schedule.get_root().child(0), deps);
}

ScopStmt *getStatementContaining(IslScop &scop, Operation *op) {
  for (ScopStmt &stmt : scop)
    if (stmt.getOperation() == op || stmt.getOperation()->isAncestor(op))
      return &stmt;
  return nullptr;
}

MemoryAccess *getAtomicArrayWrite(enzyme::AffineAtomicRMWOp rmw,
                                  ScopStmt &stmt) {
  MemoryAccess *write = nullptr;
  for (MemoryAccess *ma : stmt) {
    if (ma->Kind != MemoryAccess::MT_Array || !ma->isWrite())
      continue;
    if (ma->AI->val != rmw.getMemref())
      continue;
    if (write)
      return nullptr;
    write = ma;
  }
  return write;
}

TypedValue<MemRefType> getMemrefAtomicMemref(Operation *op) {
  if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op))
    return rmw.getMemref();
  if (auto rmw = dyn_cast<memref::GenericAtomicRMWOp>(op))
    return rmw.getMemref();
  return nullptr;
}

Value stripMemrefCasts(Value value) {
  while (auto cast = value.getDefiningOp<memref::CastOp>())
    value = cast.getSource();
  return value;
}

bool areDistinctNoAliasFunctionArgs(Value lhs, Value rhs) {
  lhs = stripMemrefCasts(lhs);
  rhs = stripMemrefCasts(rhs);
  if (lhs == rhs)
    return false;

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (!lhsArg || !rhsArg || lhsArg.getOwner() != rhsArg.getOwner())
    return false;

  auto function =
      dyn_cast<FunctionOpInterface>(lhsArg.getOwner()->getParentOp());
  if (!function)
    return false;

  StringRef noAliasAttr = LLVM::LLVMDialect::getNoAliasAttrName();
  return function.getArgAttr(lhsArg.getArgNumber(), noAliasAttr) &&
         function.getArgAttr(rhsArg.getArgNumber(), noAliasAttr);
}

bool mayAliasForAtomicRemoval(Value lhs, Value rhs) {
  if (areDistinctNoAliasFunctionArgs(lhs, rhs))
    return false;
  return mayAlias(lhs, rhs);
}

// Determines whether it is safe to "remove", i.e. convert an atomic rmw to
// non-atomic read-modify-store.
//
// This is safe when there is no other racy atomic rmw w.r.t. this rmw. Regular
// load/store races are ignored because those races already exist in the input.
bool isSafeToRemoveAtomicImpl(enzyme::AffineAtomicRMWOp rmw, IslScop &scop,
                              ArrayRef<enzyme::AffineAtomicRMWOp> rmws,
                              ArrayRef<Operation *> memrefAtomics) {
  LDBG() << "Handling rmw: " << rmw;
  ScopStmt *rmwStmt = getStatementContaining(scop, rmw);
  if (!rmwStmt) {
    LDBG() << "Failed to find statement for rmw";
    return false;
  }
  MemoryAccess *theArrayWrite = getAtomicArrayWrite(rmw, *rmwStmt);
  if (!theArrayWrite) {
    LDBG() << "Failed to identify atomic write for rmw";
    return false;
  }
  assert(theArrayWrite);
  LDBG() << "Found array write";
  LLVM_DEBUG(polly::dumpIslObj(theArrayWrite->getAccessRelation()));

  TypedValue<MemRefType> theArray = rmw.getMemref();
  assert(theArray == theArrayWrite->AI->val);

  for (Operation *memrefAtomic : memrefAtomics) {
    TypedValue<MemRefType> otherArray = getMemrefAtomicMemref(memrefAtomic);
    assert(otherArray);
    if (!mayAliasForAtomicRemoval(otherArray, theArray))
      continue;

    LDBG() << "Found atomic operation to an aliasing memref: " << *memrefAtomic;
    return false;
  }

  for (auto otherRmw : rmws) {
    if (otherRmw == rmw)
      continue;

    TypedValue<MemRefType> otherArray = otherRmw.getMemref();
    if (!mayAliasForAtomicRemoval(otherArray, theArray))
      continue;

    ScopStmt *otherStmt = getStatementContaining(scop, otherRmw);
    if (!otherStmt) {
      LDBG() << "Failed to find statement for other rmw";
      return false;
    }

    MemoryAccess *otherArrayWrite = getAtomicArrayWrite(otherRmw, *otherStmt);
    if (!otherArrayWrite) {
      LDBG() << "Failed to identify atomic write for other rmw";
      return false;
    }

    if (otherArray != theArray) {
      if (isAnyInstanceRacy(*rmwStmt, *otherStmt)) {
        LDBG() << "Found racy atomic to an aliasing array: " << otherRmw;
        return false;
      }
    } else {
      if (isAccessRacy(*rmwStmt, theArrayWrite, *otherStmt, otherArrayWrite)) {
        LDBG() << "Found racy atomic to the same array: " << otherRmw;
        return false;
      }
    }
  }
  return true;
}

bool isSafeToRemoveAtomic(enzyme::AffineAtomicRMWOp rmw, IslScop &scop,
                          ArrayRef<enzyme::AffineAtomicRMWOp> rmws,
                          ArrayRef<Operation *> memrefAtomics) {
  if (isSafeToRemoveAtomicImpl(rmw, scop, rmws, memrefAtomics)) {
    LDBG("remove-atomics-decision") << "Legal to remove atomic from " << rmw;
    return true;
  } else {
    LDBG("remove-atomics-decision") << "Illegal to remove atomic from " << rmw;
    return false;
  }
}

void convertRmw(enzyme::AffineAtomicRMWOp rmw) {
  OpBuilder b(rmw);
  auto read = affine::AffineLoadOp::create(b, rmw.getLoc(), rmw.getMemref(),
                                           rmw.getMap(), rmw.getIndices());

  mlir::Value modify;
  // TODO fast math flags?
  switch (rmw.getKind()) {
  case mlir::arith::AtomicRMWKind::addf:
    modify = arith::AddFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::mulf:
    modify = arith::MulFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::muli:
    modify = arith::MulIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::addi:
    modify = arith::AddIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::andi:
    modify = arith::AndIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::maximumf:
    modify = arith::MaximumFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::minimumf:
    modify = arith::MinimumFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::xori:
    modify = arith::XOrIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::ori:
    modify = arith::OrIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::maxs:
    modify = arith::MaxSIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::mins:
    modify = arith::MinSIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::assign:
    modify = rmw.getValue();
    break;
  case mlir::arith::AtomicRMWKind::maxnumf:
    modify = arith::MaxNumFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::minnumf:
    modify = arith::MinNumFOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::maxu:
    modify = arith::MaxUIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  case mlir::arith::AtomicRMWKind::minu:
    modify = arith::MinUIOp::create(b, rmw.getLoc(), read, rmw.getValue());
    break;
  }

  affine::AffineStoreOp::create(b, rmw.getLoc(), modify, rmw.getMemref(),
                                rmw.getMap(), rmw.getIndices());
  rmw.getResult().replaceAllUsesWith(modify);
  rmw.erase();
}

// We are working under the assumption that atomic rmw is only atomic within the
// GPU kernel it is in, thus, we can assume there is no other concurrent code
// that could modify it.
void handleGPUWrapper(enzymexla::GPUWrapperOp wrapperOp) {
  LDBG() << "Processing " << wrapperOp;

  llvm::SmallVector<enzyme::AffineAtomicRMWOp> rmws;
  wrapperOp->walk([&](enzyme::AffineAtomicRMWOp rmw) { rmws.push_back(rmw); });
  SmallVector<Operation *> memrefAtomics;
  wrapperOp->walk([&](Operation *op) {
    if (getMemrefAtomicMemref(op))
      memrefAtomics.push_back(op);
  });
  if (rmws.empty()) {
    LDBG() << "No RMWs";
    return;
  }

  std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(
      wrapperOp, /*allowScfIfConditionalWritesAsMay=*/true);
  if (!scop) {
    LDBG() << "Failed to build scop";
    return;
  }
  if (scop->buildSchedule().failed()) {
    LDBG() << "Failed to build schedule\n";
    return;
  }
  LDBG() << "Schedule:";
  LDBG_ISL_DUMP(scop->getScheduleTree().get());
  LDBG() << "Accesses:";
  LLVM_DEBUG(scop->dumpAccesses(llvm::dbgs()));

  SmallVector<enzyme::AffineAtomicRMWOp> toConvert;
  for (auto rmw : rmws)
    if (isSafeToRemoveAtomic(rmw, *scop, rmws, memrefAtomics))
      toConvert.push_back(rmw);

  for (auto rmw : toConvert)
    convertRmw(rmw);
}
#endif

struct RemoveAtomicsPass
    : public enzyme::impl::RemoveAtomicsPassBase<RemoveAtomicsPass> {
  using RemoveAtomicsPassBase::RemoveAtomicsPassBase;
  void runOnOperation() override {
#ifndef DISABLE_POLYMER
    getOperation()->walk([](enzymexla::GPUWrapperOp op) {
      handleGPUWrapper(op);
      return WalkResult::skip();
    });
#endif
  }
};
} // namespace
