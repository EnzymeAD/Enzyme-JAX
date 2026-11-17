//===- GPUPolyhedralOpt.cpp - Polyhedral GPU Scheduling & Shmem -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Polyhedral GPU optimization pass integrating ISL schedule construction (with
// live-range reordering, maximal live-range span, anti-proximity, and cache
// size bounds), automatic GPU grid/thread-block tiling, PPCG-style shared and
// thread-private register memory promotion, and block-loop undistribution with
// minimal barrier insertion.
//
//===----------------------------------------------------------------------===//

#ifndef DISABLE_POLYMER
#include "mlir/Conversion/Polymer/Support/DependenceInfo.h"
#include "mlir/Conversion/Polymer/Support/ISLUtils.h"
#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Conversion/Polymer/Transforms/LoopUndistribute.h"
#include "mlir/Conversion/Polymer/Transforms/SharedMemoryPromotion.h"
#endif

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#ifndef DISABLE_POLYMER
#include "isl/aff.h"
#include "isl/ast.h"
#include "isl/ast_build.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/id.h"
#include "isl/id_to_id.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/set.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"
#endif

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "gpu-polyhedral-opt"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_GPUPOLYHEDRALOPTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

#ifndef DISABLE_POLYMER
namespace {

static constexpr char kBlockParallelMark[] = "block_parallel";
static constexpr char kStmtGroupAttr[] = "polymer.stmt_group";

struct BlockParallelMarkInfo {
  SmallVector<int64_t, 3> blockSizes;
};

static inline unsigned unsignedFromIslSize(isl_size sz) {
  assert(sz >= 0);
  return static_cast<unsigned>(sz);
}

static bool extractIslInt(__isl_take isl_val *v, int64_t &out) {
  if (!v)
    return false;
  if (!isl_val_is_int(v)) {
    isl_val_free(v);
    return false;
  }
  out = isl_val_get_num_si(v);
  isl_val_free(v);
  return true;
}

/// Construct ISL schedule constraints from `ppcg_scop` (ported from Ivan's
/// `GPUAffineOpt.cpp:construct_schedule_constraints`, extended to handle
/// dynamic-shaped arrays gracefully without aborting).
static __isl_give isl_schedule_constraints *
constructScheduleConstraints(ppcg_scop *ps, polymer::IslScop &S,
                             int64_t maxSharedMemBytes) {
  isl_union_set *domain = isl_union_set_copy(ps->domain);
  isl_schedule_constraints *sc = isl_schedule_constraints_on_domain(domain);
  sc = isl_schedule_constraints_set_context(sc, isl_set_copy(ps->context));

  isl_union_map *validity = nullptr;
  isl_union_map *proximity = nullptr;
  isl_union_map *coincidence = nullptr;

  if (ps->options->live_range_reordering) {
    sc = isl_schedule_constraints_set_conditional_validity(
        sc, isl_union_map_copy(ps->tagged_dep_flow),
        isl_union_map_copy(ps->tagged_dep_order));
    proximity = isl_union_map_copy(ps->dep_flow);
    validity = isl_union_map_union(isl_union_map_copy(proximity),
                                   isl_union_map_copy(ps->dep_forced));
    isl_union_map *allCoincDeps = isl_union_map_union(
        isl_union_map_copy(validity), isl_union_map_copy(ps->dep_false));
    coincidence = isl_union_map_subtract(allCoincDeps,
                                         isl_union_map_copy(ps->independence));
  } else {
    isl_union_map *depRaw = isl_union_map_copy(ps->dep_flow);
    isl_union_map *dep =
        isl_union_map_union(isl_union_map_copy(ps->dep_false), depRaw);
    dep = isl_union_map_coalesce(dep);
    proximity = isl_union_map_copy(dep);
    coincidence = isl_union_map_copy(dep);
    validity = dep;
  }

  isl_union_map *antiProximity = isl_union_map_copy(ps->dep_async);
  sc = isl_schedule_constraints_set_validity(sc, validity);
  sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
  sc = isl_schedule_constraints_set_proximity(sc, proximity);
  sc = isl_schedule_constraints_set_anti_proximity(sc, antiProximity);

  isl_union_map *lrs = isl_union_map_copy(ps->atagged_dep_flow);
  sc = isl_schedule_constraints_set_live_range_span(sc, isl_union_map_copy(lrs));
  isl_union_map *maxLrs =
      polymer::get_maximal_paths(isl::manage(lrs)).release();
  sc = isl_schedule_constraints_set_live_range_maximal_span(sc, maxLrs);

  isl_union_set *arrays = isl_union_map_range(isl_union_set_unwrap(
      isl_union_map_domain(isl_union_map_copy(ps->atagged_dep_flow))));
  arrays = isl_union_set_universe(arrays);
  isl::union_map arraySizes =
      isl::manage(isl_union_map_empty(isl_union_set_get_space(arrays)));

  bool allStaticSizes = true;
  isl::manage(arrays).foreach_set([&](isl::set set) -> isl::stat {
    Value v = Value::getFromOpaquePointer(set.get_tuple_id().get_user());
    auto *sai = S.getArray(v);
    if (!sai || !sai->getSize()) {
      allStaticSizes = false;
      return isl::stat::ok();
    }
    unsigned byteSize = *sai->getSize();
    isl::space sizeSpace(set.ctx(), 0, 1);
    isl::set sizeSet = isl::set::universe(sizeSpace);
    isl::constraint cst =
        isl::constraint::alloc_equality(isl::local_space(sizeSet.get_space()));
    cst = cst.set_constant_si(-static_cast<int64_t>(byteSize));
    cst = cst.set_coefficient_si(isl::dim::out, 0, 1);
    sizeSet = sizeSet.add_constraint(cst);
    isl::map map = isl::map::from_domain_and_range(set, sizeSet);
    arraySizes = arraySizes.unite(isl::union_map(map));
    return isl::stat::ok();
  });

  if (allStaticSizes && maxSharedMemBytes > 0) {
    int cacheBytes = static_cast<int>(
        std::min<int64_t>(maxSharedMemBytes, INT32_MAX));
    sc = isl_schedule_constraints_set_caches(sc, 1, &cacheBytes);
    sc = isl_schedule_constraints_set_array_sizes(sc, arraySizes.release());
  }

  return sc;
}

/// Count leading coincident (`parallel`) members of `band`.
static unsigned countLeadingCoincident(__isl_keep isl_schedule_node *band) {
  unsigned n = unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  unsigned p = 0;
  for (; p < n; ++p) {
    if (isl_schedule_node_band_member_get_coincident(band, p) != isl_bool_true)
      break;
  }
  return p;
}

/// Check if `node`'s subtree contains a `sequence` whose children EVERY one
/// start with (or contain) a band having at least 1 coincident member.
static bool hasMultiStageParallelSequence(__isl_keep isl_schedule_node *node) {
  isl_schedule_node_type ty = isl_schedule_node_get_type(node);
  if (ty == isl_schedule_node_sequence || ty == isl_schedule_node_set) {
    unsigned nCh = unsignedFromIslSize(isl_schedule_node_n_children(node));
    if (nCh == 0)
      return false;
    for (unsigned i = 0; i < nCh; ++i) {
      isl_schedule_node *filter = isl_schedule_node_get_child(node, i);
      isl_schedule_node *inner = isl_schedule_node_get_child(filter, 0);
      isl_schedule_node_free(filter);
      bool branchHasPar = false;
      while (isl_schedule_node_get_type(inner) == isl_schedule_node_mark) {
        isl_schedule_node *next = isl_schedule_node_get_child(inner, 0);
        isl_schedule_node_free(inner);
        inner = next;
      }
      if (isl_schedule_node_get_type(inner) == isl_schedule_node_band &&
          countLeadingCoincident(inner) >= 1) {
        branchHasPar = true;
      }
      isl_schedule_node_free(inner);
      if (!branchHasPar)
        return false;
    }
    return true;
  }
  if (ty == isl_schedule_node_mark || ty == isl_schedule_node_filter) {
    if (isl_schedule_node_n_children(node) != 1)
      return false;
    isl_schedule_node *ch = isl_schedule_node_get_child(node, 0);
    bool res = hasMultiStageParallelSequence(ch);
    isl_schedule_node_free(ch);
    return res;
  }
  return false;
}

/// Reverse the first `numParallel` members of `band` (`[0 .. P-1]` -> `[P-1 .. 0]`)
/// so member 0 (`blockIdx.x` / `threadIdx.x`) corresponds to the innermost
/// contiguous parallel loop in row-major MLIR memrefs.
static __isl_give isl_schedule_node *
reverseLeadingParallelMembers(__isl_take isl_schedule_node *band,
                              unsigned numParallel) {
  if (numParallel <= 1)
    return band;
  unsigned nMembers =
      unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  SmallVector<isl_bool, 8> coinc(nMembers);
  for (unsigned i = 0; i < nMembers; ++i)
    coinc[i] = isl_schedule_node_band_member_get_coincident(band, i);

  isl_multi_union_pw_aff *mupa =
      isl_schedule_node_band_get_partial_schedule(band);
  isl_multi_union_pw_aff *perm = isl_multi_union_pw_aff_copy(mupa);
  for (unsigned k = 0; k < numParallel; ++k) {
    isl_union_pw_aff *upa =
        isl_multi_union_pw_aff_get_union_pw_aff(mupa, numParallel - 1 - k);
    perm = isl_multi_union_pw_aff_set_union_pw_aff(perm, k, upa);
  }
  isl_multi_union_pw_aff_free(mupa);

  isl_schedule_node *child = isl_schedule_node_delete(band);
  isl_schedule_node *newBand =
      isl_schedule_node_insert_partial_schedule(child, perm);
  if (permutable == isl_bool_true)
    newBand = isl_schedule_node_band_set_permutable(newBand, 1);
  for (unsigned i = 0; i < nMembers; ++i) {
    unsigned origIdx = (i < numParallel) ? (numParallel - 1 - i) : i;
    newBand = isl_schedule_node_band_member_set_coincident(
        newBand, i, coinc[origIdx] == isl_bool_true ? 1 : 0);
  }
  return newBand;
}

/// If `band` consists solely of `nPar` parallel dimensions (`totalMembers == nPar`),
/// `seqTileSize > 1`, and its single child is a sequential band `seqBand`
/// (`countLeadingCoincident(seqBand) == 0`, with `nSeq == 1` or `permutable`),
/// merge `seqBand` into `band` as a permutable `(nPar + nSeq)`-member band
/// whenever all dependences in `allDeps` active at `band` have zero dependence
/// distance along `band` (`p(x) == p(y)`).
static __isl_give isl_schedule_node *
tryMergeSequentialChildBand(__isl_take isl_schedule_node *band, unsigned nPar,
                            int64_t seqTileSize,
                            __isl_keep isl_union_map *allDeps) {
  if (seqTileSize <= 1 || !allDeps)
    return band;
  unsigned totalMembers =
      unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  if (totalMembers != nPar || isl_schedule_node_n_children(band) != 1)
    return band;

  isl_schedule_node *child = isl_schedule_node_get_child(band, 0);
  if (isl_schedule_node_get_type(child) != isl_schedule_node_band) {
    isl_schedule_node_free(child);
    return band;
  }
  unsigned nSeq =
      unsignedFromIslSize(isl_schedule_node_band_n_member(child));
  bool childIsSeq = (nSeq > 0 && countLeadingCoincident(child) == 0 &&
                     (nSeq == 1 || isl_schedule_node_band_get_permutable(
                                       child) == isl_bool_true));
  if (!childIsSeq) {
    isl_schedule_node_free(child);
    return band;
  }

  isl_union_map *prefix = isl_schedule_node_get_prefix_schedule_relation(band);
  isl_union_map *samePrefix = isl_union_map_apply_range(
      isl_union_map_copy(prefix), isl_union_map_reverse(prefix));
  isl_union_map_free(prefix);
  isl_union_map *subDeps =
      isl_union_map_intersect(isl_union_map_copy(allDeps), samePrefix);

  isl_union_map *bandMap =
      isl_schedule_node_band_get_partial_schedule_union_map(band);
  isl_union_map *schedPairs =
      isl_union_map_apply_domain(subDeps, isl_union_map_copy(bandMap));
  schedPairs = isl_union_map_apply_range(schedPairs, bandMap);

  isl_space *bandSpace = isl_schedule_node_band_get_space(band);
  isl_map *idMap = isl_map_identity(isl_space_map_from_set(bandSpace));
  idMap = isl_map_align_params(idMap, isl_union_map_get_space(schedPairs));
  isl_union_map *idUmap = isl_union_map_from_map(idMap);
  isl_bool zeroDelta = isl_union_map_is_subset(schedPairs, idUmap);
  isl_union_map_free(schedPairs);
  isl_union_map_free(idUmap);

  if (zeroDelta != isl_bool_true) {
    isl_schedule_node_free(child);
    return band;
  }

  isl_multi_union_pw_aff *mupaPar =
      isl_schedule_node_band_get_partial_schedule(band);
  isl_multi_union_pw_aff *mupaSeq =
      isl_schedule_node_band_get_partial_schedule(child);
  isl_schedule_node_free(child);
  isl_multi_union_pw_aff *merged =
      isl_multi_union_pw_aff_flat_range_product(mupaPar, mupaSeq);

  isl_schedule_node *seqNode = isl_schedule_node_delete(band);
  isl_schedule_node *grandChild = isl_schedule_node_delete(seqNode);
  isl_schedule_node *newBand =
      isl_schedule_node_insert_partial_schedule(grandChild, merged);
  newBand = isl_schedule_node_band_set_permutable(newBand, 1);
  for (unsigned i = 0; i < nPar + nSeq; ++i)
    newBand = isl_schedule_node_band_member_set_coincident(
        newBand, i, i < nPar ? 1 : 0);
  return newBand;
}

/// Tile an inner band for `block_parallel` (used in multi-stage fusion where an
/// outer band is already mapped to `grid_parallel`).
static __isl_give isl_schedule_node *
tileBandForBlockOnly(__isl_take isl_schedule_node *band,
                     ArrayRef<int64_t> blockSizes, int64_t seqTileSize,
                     __isl_keep isl_union_map *allDeps,
                     std::vector<std::unique_ptr<BlockParallelMarkInfo>> &marks) {
  isl_ctx *ctx = isl_schedule_node_get_ctx(band);
  unsigned totalMembers =
      unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  unsigned nPar = std::min<unsigned>(
      countLeadingCoincident(band),
      std::min<unsigned>(3, blockSizes.size()));
  if (nPar == 0)
    return band;

  // If there are > 3 parallel members, split off excess before reversing/tiling.
  unsigned allCoinc = countLeadingCoincident(band);
  if (allCoinc > nPar) {
    band = isl_schedule_node_band_split(band, nPar);
    totalMembers = nPar;
  }

  band = reverseLeadingParallelMembers(band, nPar);
  if (totalMembers == nPar && seqTileSize > 1) {
    band = tryMergeSequentialChildBand(band, nPar, seqTileSize, allDeps);
    totalMembers = unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  }
  isl_bool permutable = isl_schedule_node_band_get_permutable(band);

  // If not permutable and `totalMembers > nPar`, split off sequential suffix so
  // we only tile the parallel prefix.
  if ((permutable != isl_bool_true || seqTileSize <= 1) &&
      totalMembers > nPar) {
    band = isl_schedule_node_band_split(band, nPar);
    totalMembers = nPar;
  }

  isl_space *bandSpace = isl_schedule_node_band_get_space(band);
  isl_multi_val *tileMV = isl_multi_val_zero(bandSpace);
  SmallVector<int64_t, 3> chosenBlockSizes;
  for (unsigned i = 0; i < totalMembers; ++i) {
    int64_t sz = (i < nPar) ? blockSizes[i] : std::max<int64_t>(1, seqTileSize);
    if (i < nPar)
      chosenBlockSizes.push_back(sz);
    tileMV = isl_multi_val_set_val(tileMV, i, isl_val_int_from_si(ctx, sz));
  }

  isl_options_set_tile_scale_tile_loops(ctx, 0);
  isl_options_set_tile_shift_point_loops(ctx, 1);
  band = isl_schedule_node_band_tile(band, tileMV);

  // `band` is now the serial tile loop band; its child is the point band.
  isl_schedule_node *pointBand = isl_schedule_node_get_child(band, 0);
  isl_schedule_node_free(band);
  if (totalMembers > nPar)
    pointBand = isl_schedule_node_band_split(pointBand, nPar);
  for (unsigned i = 0; i < nPar; ++i)
    pointBand = isl_schedule_node_band_member_set_ast_loop_type(
        pointBand, i, isl_ast_loop_atomic);

  auto info = std::make_unique<BlockParallelMarkInfo>();
  info->blockSizes = chosenBlockSizes;
  BlockParallelMarkInfo *rawInfo = info.get();
  marks.push_back(std::move(info));

  isl_id *bpMark = isl_id_alloc(ctx, kBlockParallelMark, rawInfo);
  pointBand = isl_schedule_node_insert_mark(pointBand, bpMark);
  return isl_schedule_node_parent(pointBand);
}

/// Map a top-level band to `grid_parallel` + `block_parallel` (either by tiling
/// a single parallel band, or by mapping an outer band to `grid_parallel` and
/// inner sequence bands to `block_parallel`).
static __isl_give isl_schedule_node *
mapBandToGPU(__isl_take isl_schedule_node *band, ArrayRef<int64_t> blockSizes,
             int64_t seqTileSize, __isl_keep isl_union_map *allDeps,
             std::vector<std::unique_ptr<BlockParallelMarkInfo>> &marks) {
  isl_ctx *ctx = isl_schedule_node_get_ctx(band);
  unsigned leadingCoinc = countLeadingCoincident(band);
  if (leadingCoinc == 0)
    return band;

  unsigned nPar = std::min<unsigned>(
      leadingCoinc, std::min<unsigned>(3, blockSizes.size()));
  if (nPar == 0)
    return band;

  // Check Case B: multi-stage sequence of inner parallel bands beneath `band`.
  if (leadingCoinc ==
      unsignedFromIslSize(isl_schedule_node_band_n_member(band))) {
    isl_schedule_node *ch = isl_schedule_node_get_child(band, 0);
    bool isMultiStage = hasMultiStageParallelSequence(ch);
    isl_schedule_node_free(ch);
    if (isMultiStage) {
      if (leadingCoinc > nPar)
        band = isl_schedule_node_band_split(band, nPar);
      band = reverseLeadingParallelMembers(band, nPar);
      for (unsigned i = 0; i < nPar; ++i)
        band = isl_schedule_node_band_member_set_ast_loop_type(
            band, i, isl_ast_loop_atomic);

      // Transform each child branch of the sequence beneath `band`.
      isl_schedule_node *seqNode = isl_schedule_node_get_child(band, 0);
      isl_schedule_node_free(band);
      unsigned nBranches =
          unsignedFromIslSize(isl_schedule_node_n_children(seqNode));
      for (unsigned b = 0; b < nBranches; ++b) {
        isl_schedule_node *filt = isl_schedule_node_get_child(seqNode, b);
        isl_schedule_node_free(seqNode);
        isl_schedule_node *inner = isl_schedule_node_get_child(filt, 0);
        isl_schedule_node_free(filt);
        inner =
            tileBandForBlockOnly(inner, blockSizes, seqTileSize, allDeps, marks);
        filt = isl_schedule_node_parent(inner);
        seqNode = isl_schedule_node_parent(filt);
      }
      band = isl_schedule_node_parent(seqNode);
      isl_id *gridMark = isl_id_alloc(
          ctx, polymer::gridParallelMark,
          reinterpret_cast<void *>(static_cast<uintptr_t>(nPar)));
      return isl_schedule_node_insert_mark(band, gridMark);
    }
  }

  // Case A: Tile `band` into `grid_parallel` + optional serial tile loops +
  // `block_parallel` + optional serial point loops!
  unsigned totalMembers =
      unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  if (leadingCoinc > nPar) {
    // Keep extra parallel loops inside by splitting after `nPar`.
    band = isl_schedule_node_band_split(band, nPar);
    totalMembers = nPar;
  }

  band = reverseLeadingParallelMembers(band, nPar);
  if (totalMembers == nPar && seqTileSize > 1) {
    band = tryMergeSequentialChildBand(band, nPar, seqTileSize, allDeps);
    totalMembers = unsignedFromIslSize(isl_schedule_node_band_n_member(band));
  }
  isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  if (permutable != isl_bool_true && totalMembers > nPar) {
    band = isl_schedule_node_band_split(band, nPar);
    totalMembers = nPar;
  }
  if (seqTileSize <= 1 && totalMembers > nPar) {
    band = isl_schedule_node_band_split(band, nPar);
    totalMembers = nPar;
  }

  isl_space *bandSpace = isl_schedule_node_band_get_space(band);
  isl_multi_val *tileMV = isl_multi_val_zero(bandSpace);
  SmallVector<int64_t, 3> chosenBlockSizes;
  for (unsigned i = 0; i < totalMembers; ++i) {
    int64_t sz = (i < nPar) ? blockSizes[i] : seqTileSize;
    if (i < nPar)
      chosenBlockSizes.push_back(sz);
    tileMV = isl_multi_val_set_val(tileMV, i, isl_val_int_from_si(ctx, sz));
  }

  isl_options_set_tile_scale_tile_loops(ctx, 0);
  isl_options_set_tile_shift_point_loops(ctx, 1);
  band = isl_schedule_node_band_tile(band, tileMV);

  // Split outer tile band into `grid_parallel` (`0..nPar`) + serial tile band.
  if (totalMembers > nPar)
    band = isl_schedule_node_band_split(band, nPar);
  for (unsigned i = 0; i < nPar; ++i)
    band = isl_schedule_node_band_member_set_ast_loop_type(
        band, i, isl_ast_loop_atomic);

  // Navigate to point band (`child` if `totalMembers == nPar`, else `child.child`).
  isl_schedule_node *pointBand = isl_schedule_node_get_child(band, 0);
  isl_schedule_node_free(band);
  if (totalMembers > nPar) {
    isl_schedule_node *next = isl_schedule_node_get_child(pointBand, 0);
    isl_schedule_node_free(pointBand);
    pointBand = next;
  }

  if (totalMembers > nPar)
    pointBand = isl_schedule_node_band_split(pointBand, nPar);
  for (unsigned i = 0; i < nPar; ++i)
    pointBand = isl_schedule_node_band_member_set_ast_loop_type(
        pointBand, i, isl_ast_loop_atomic);

  auto info = std::make_unique<BlockParallelMarkInfo>();
  info->blockSizes = chosenBlockSizes;
  BlockParallelMarkInfo *rawInfo = info.get();
  marks.push_back(std::move(info));

  isl_id *bpMark = isl_id_alloc(ctx, kBlockParallelMark, rawInfo);
  pointBand = isl_schedule_node_insert_mark(pointBand, bpMark);

  // Walk back up to `band` (grid band) and insert `grid_parallel` mark.
  isl_schedule_node *cur = isl_schedule_node_parent(pointBand);
  if (totalMembers > nPar)
    cur = isl_schedule_node_parent(cur);
  isl_id *gridMark = isl_id_alloc(
      ctx, polymer::gridParallelMark,
      reinterpret_cast<void *>(static_cast<uintptr_t>(nPar)));
  return isl_schedule_node_insert_mark(cur, gridMark);
}

static isl::schedule
tileScheduleForGPU(isl::schedule schedule, ArrayRef<int64_t> blockSizes,
                   int64_t seqTileSize, bool alreadyHasBlockPar,
                   __isl_keep isl_union_map *allDeps,
                   std::vector<std::unique_ptr<BlockParallelMarkInfo>> &marks) {
  isl_schedule_node *root = isl_schedule_get_root(schedule.get());
  if (isl_schedule_node_n_children(root) != 1) {
    isl_schedule_node_free(root);
    return {};
  }

  isl_schedule_node *child = isl_schedule_node_get_child(root, 0);
  isl_schedule_node_free(root);

  // Skip context node if present.
  bool steppedIntoContext = false;
  if (isl_schedule_node_get_type(child) == isl_schedule_node_context) {
    isl_schedule_node *next = isl_schedule_node_get_child(child, 0);
    isl_schedule_node_free(child);
    child = next;
    steppedIntoContext = true;
  }

  auto processBand = [&](isl_schedule_node *bNode) -> isl_schedule_node * {
    if (isl_schedule_node_get_type(bNode) != isl_schedule_node_band)
      return bNode;
    if (alreadyHasBlockPar) {
      // Statements are already `gpu.par.block` ops; mark outer coincident band
      // directly as `grid_parallel`.
      unsigned nPar = std::min<unsigned>(countLeadingCoincident(bNode), 3);
      if (nPar == 0)
        return bNode;
      unsigned total =
          unsignedFromIslSize(isl_schedule_node_band_n_member(bNode));
      if (total > nPar)
        bNode = isl_schedule_node_band_split(bNode, nPar);
      for (unsigned i = 0; i < nPar; ++i)
        bNode = isl_schedule_node_band_member_set_ast_loop_type(
            bNode, i, isl_ast_loop_atomic);
      isl_id *gridMark = isl_id_alloc(
          schedule.ctx().get(), polymer::gridParallelMark,
          reinterpret_cast<void *>(static_cast<uintptr_t>(nPar)));
      return isl_schedule_node_insert_mark(bNode, gridMark);
    }
    return mapBandToGPU(bNode, blockSizes, seqTileSize, allDeps, marks);
  };

  isl_schedule_node_type ty = isl_schedule_node_get_type(child);
  if (ty == isl_schedule_node_band) {
    child = processBand(child);
  } else if (ty == isl_schedule_node_sequence || ty == isl_schedule_node_set) {
    unsigned nBranches =
        unsignedFromIslSize(isl_schedule_node_n_children(child));
    for (unsigned i = 0; i < nBranches; ++i) {
      isl_schedule_node *filt = isl_schedule_node_get_child(child, i);
      isl_schedule_node_free(child);
      isl_schedule_node *bNode = isl_schedule_node_get_child(filt, 0);
      isl_schedule_node_free(filt);
      bNode = processBand(bNode);
      filt = isl_schedule_node_parent(bNode);
      child = isl_schedule_node_parent(filt);
    }
  }

  if (steppedIntoContext)
    child = isl_schedule_node_parent(child);
  isl_schedule *res = isl_schedule_node_get_schedule(child);
  isl_schedule_node_free(child);
  return isl::manage(res);
}

/// Pure `AffineDialect` ISL AST code generator emitting `affine.parallel`,
/// `affine.for`, `affine.if`, and `affine.apply` (subsequently composed into
/// `affine.load` / `affine.store`).
class IslAffineBuilder {
public:
  IslAffineBuilder(OpBuilder &b, IRMapping &funcMapping, polymer::IslScop &scop)
      : b(b), ctx(b.getContext()), loc(b.getUnknownLoc()),
        funcMapping(funcMapping), scop(scop) {}

  void mapParams(__isl_take isl_union_set *domain) {
    isl_space *space = isl_union_set_get_space(domain);
    int nparams = isl_space_dim(space, isl_dim_param);
    for (int i = 0; i < nparams; ++i) {
      isl_id *id = isl_space_get_dim_id(space, isl_dim_param, i);
      Value origVal = Value::getFromOpaquePointer(isl_id_get_user(id));
      Value mapped = funcMapping.lookupOrDefault(origVal);
      if (!isa<IndexType>(mapped.getType())) {
        mapped =
            arith::IndexCastOp::create(b, loc, b.getIndexType(), mapped);
      }
      idToValue[id] = mapped;
      paramIds.insert(id);
      isl_id_free(id);
    }
    isl_space_free(space);
    isl_union_set_free(domain);
  }

  bool emit(__isl_take isl_ast_node *node) {
    bool ok = createNode(node);
    return ok;
  }

private:
  OpBuilder &b;
  MLIRContext *ctx;
  Location loc;
  IRMapping &funcMapping;
  polymer::IslScop &scop;
  DenseMap<isl_id *, Value> idToValue;
  DenseSet<isl_id *> paramIds;

  struct BuiltAffineExpr {
    AffineExpr expr;
    SmallVector<Value, 8> dims;
    SmallVector<Value, 8> syms;

    AffineMap toMap(MLIRContext *c) const {
      return AffineMap::get(dims.size(), syms.size(), expr, c);
    }
    SmallVector<Value, 8> allOperands() const {
      SmallVector<Value, 8> ops = dims;
      ops.append(syms.begin(), syms.end());
      return ops;
    }
  };

  /// Merge operands of `rhs` into `lhs` and return remapped `(lhsExpr, rhsExpr)`.
  std::pair<AffineExpr, AffineExpr>
  unifyOperands(BuiltAffineExpr &lhs, const BuiltAffineExpr &rhs) {
    SmallVector<AffineExpr, 8> rhsDimRepl, rhsSymRepl;
    for (Value dVal : rhs.dims) {
      auto it = llvm::find(lhs.dims, dVal);
      unsigned pos = 0;
      if (it == lhs.dims.end()) {
        pos = lhs.dims.size();
        lhs.dims.push_back(dVal);
      } else {
        pos = std::distance(lhs.dims.begin(), it);
      }
      rhsDimRepl.push_back(getAffineDimExpr(pos, ctx));
    }
    for (Value sVal : rhs.syms) {
      auto it = llvm::find(lhs.syms, sVal);
      unsigned pos = 0;
      if (it == lhs.syms.end()) {
        pos = lhs.syms.size();
        lhs.syms.push_back(sVal);
      } else {
        pos = std::distance(lhs.syms.begin(), it);
      }
      rhsSymRepl.push_back(getAffineSymbolExpr(pos, ctx));
    }
    AffineExpr rMapped = rhs.expr.replaceDimsAndSymbols(rhsDimRepl, rhsSymRepl);
    return {lhs.expr, rMapped};
  }

  /// Build an `AffineExpr` from an `isl_ast_expr *` (materializing `min`/`max`
  /// via `affine.min`/`affine.max` if nested inside arithmetic).
  std::optional<BuiltAffineExpr> buildAffineExpr(__isl_keep isl_ast_expr *expr) {
    if (!expr)
      return std::nullopt;
    switch (isl_ast_expr_get_type(expr)) {
    case isl_ast_expr_error:
      return std::nullopt;
    case isl_ast_expr_int: {
      int64_t val = 0;
      if (!extractIslInt(isl_ast_expr_get_val(expr), val))
        return std::nullopt;
      BuiltAffineExpr res;
      res.expr = getAffineConstantExpr(val, ctx);
      return res;
    }
    case isl_ast_expr_id: {
      isl_id *id = isl_ast_expr_get_id(expr);
      auto it = idToValue.find(id);
      bool isSym = paramIds.contains(id);
      isl_id_free(id);
      if (it == idToValue.end())
        return std::nullopt;
      BuiltAffineExpr res;
      if (isSym || (!affine::isValidDim(it->second) &&
                    affine::isValidSymbol(it->second))) {
        res.expr = getAffineSymbolExpr(0, ctx);
        res.syms.push_back(it->second);
      } else {
        res.expr = getAffineDimExpr(0, ctx);
        res.dims.push_back(it->second);
      }
      return res;
    }
    case isl_ast_expr_op: {
      isl_ast_op_type opTy = isl_ast_expr_get_op_type(expr);
      int nArgs = isl_ast_expr_get_op_n_arg(expr);
      if (opTy == isl_ast_op_minus && nArgs == 1) {
        isl_ast_expr *a0 = isl_ast_expr_get_op_arg(expr, 0);
        auto sub = buildAffineExpr(a0);
        isl_ast_expr_free(a0);
        if (!sub)
          return std::nullopt;
        sub->expr = -sub->expr;
        return sub;
      }
      if (opTy == isl_ast_op_min || opTy == isl_ast_op_max) {
        auto listOpt = buildBoundExprList(expr, opTy);
        if (!listOpt)
          return std::nullopt;
        AffineMap map = AffineMap::get(listOpt->dims.size(),
                                       listOpt->syms.size(), listOpt->exprs, ctx);
        SmallVector<Value, 8> ops = listOpt->dims;
        ops.append(listOpt->syms.begin(), listOpt->syms.end());
        Value matVal =
            (opTy == isl_ast_op_min)
                ? affine::AffineMinOp::create(b, loc, map, ops).getResult()
                : affine::AffineMaxOp::create(b, loc, map, ops).getResult();
        BuiltAffineExpr res;
        if (affine::isValidSymbol(matVal)) {
          res.expr = getAffineSymbolExpr(0, ctx);
          res.syms.push_back(matVal);
        } else {
          res.expr = getAffineDimExpr(0, ctx);
          res.dims.push_back(matVal);
        }
        return res;
      }
      if (nArgs != 2)
        return std::nullopt;
      isl_ast_expr *a0 = isl_ast_expr_get_op_arg(expr, 0);
      isl_ast_expr *a1 = isl_ast_expr_get_op_arg(expr, 1);
      auto lhs = buildAffineExpr(a0);
      auto rhs = buildAffineExpr(a1);
      isl_ast_expr_free(a0);
      isl_ast_expr_free(a1);
      if (!lhs || !rhs)
        return std::nullopt;

      auto [lExpr, rExpr] = unifyOperands(*lhs, *rhs);
      switch (opTy) {
      case isl_ast_op_add:
        lhs->expr = lExpr + rExpr;
        return lhs;
      case isl_ast_op_sub:
        lhs->expr = lExpr - rExpr;
        return lhs;
      case isl_ast_op_mul:
        lhs->expr = lExpr * rExpr;
        return lhs;
      case isl_ast_op_fdiv_q:
      case isl_ast_op_pdiv_q:
      case isl_ast_op_div: {
        auto rCst = dyn_cast<AffineConstantExpr>(rExpr);
        if (!rCst || rCst.getValue() <= 0)
          return std::nullopt;
        lhs->expr = lExpr.floorDiv(rCst.getValue());
        return lhs;
      }
      case isl_ast_op_pdiv_r:
      case isl_ast_op_zdiv_r: {
        auto rCst = dyn_cast<AffineConstantExpr>(rExpr);
        if (!rCst || rCst.getValue() <= 0)
          return std::nullopt;
        lhs->expr = lExpr % rCst.getValue();
        return lhs;
      }
      default:
        return std::nullopt;
      }
    }
    }
    return std::nullopt;
  }

  struct BoundExprList {
    SmallVector<AffineExpr, 4> exprs;
    SmallVector<Value, 8> dims;
    SmallVector<Value, 8> syms;
  };

  /// Flatten top-level `isl_ast_op_max` (for lower bounds) or `isl_ast_op_min`
  /// (for upper bounds) into a multi-result `AffineMap`.
  std::optional<BoundExprList> buildBoundExprList(__isl_keep isl_ast_expr *expr,
                                                  isl_ast_op_type flattenOp) {
    if (!expr)
      return std::nullopt;
    if (isl_ast_expr_get_type(expr) == isl_ast_expr_op &&
        isl_ast_expr_get_op_type(expr) == flattenOp) {
      int nArgs = isl_ast_expr_get_op_n_arg(expr);
      BoundExprList combined;
      BuiltAffineExpr acc;
      acc.expr = getAffineConstantExpr(0, ctx);
      for (int i = 0; i < nArgs; ++i) {
        isl_ast_expr *arg = isl_ast_expr_get_op_arg(expr, i);
        auto subList = buildBoundExprList(arg, flattenOp);
        isl_ast_expr_free(arg);
        if (!subList)
          return std::nullopt;
        for (AffineExpr subE : subList->exprs) {
          BuiltAffineExpr item{subE, subList->dims, subList->syms};
          auto [_, mappedE] = unifyOperands(acc, item);
          (void)_;
          combined.exprs.push_back(mappedE);
        }
      }
      combined.dims = std::move(acc.dims);
      combined.syms = std::move(acc.syms);
      return combined;
    }

    auto single = buildAffineExpr(expr);
    if (!single)
      return std::nullopt;
    BoundExprList list;
    list.exprs.push_back(single->expr);
    list.dims = std::move(single->dims);
    list.syms = std::move(single->syms);
    return list;
  }

  /// Extract `[lbMap, ubMap, step]` (half-open `[lb, ub)`) from `For`.
  bool extractLoopBounds(__isl_keep isl_ast_node *forNode, AffineMap &lbMap,
                         SmallVectorImpl<Value> &lbOps, AffineMap &ubMap,
                         SmallVectorImpl<Value> &ubOps, int64_t &stepOut,
                         isl_id *&iterIdOut) {
    isl_ast_expr *iterExpr = isl_ast_node_for_get_iterator(forNode);
    iterIdOut = isl_ast_expr_get_id(iterExpr);
    isl_ast_expr_free(iterExpr);

    if (isl_ast_node_for_is_degenerate(forNode) == isl_bool_true) {
      isl_ast_expr *initExpr = isl_ast_node_for_get_init(forNode);
      auto initBuilt = buildAffineExpr(initExpr);
      isl_ast_expr_free(initExpr);
      if (!initBuilt) {
        isl_id_free(iterIdOut);
        return false;
      }
      lbMap = initBuilt->toMap(ctx);
      lbOps = initBuilt->allOperands();
      BuiltAffineExpr ubBuilt = *initBuilt;
      ubBuilt.expr = ubBuilt.expr + 1;
      ubMap = ubBuilt.toMap(ctx);
      ubOps = ubBuilt.allOperands();
      stepOut = 1;
      return true;
    }

    isl_ast_expr *initExpr = isl_ast_node_for_get_init(forNode);
    isl_ast_expr *incExpr = isl_ast_node_for_get_inc(forNode);
    isl_ast_expr *condExpr = isl_ast_node_for_get_cond(forNode);

    int64_t stepVal = 1;
    bool stepOk = (isl_ast_expr_get_type(incExpr) == isl_ast_expr_int) &&
                  extractIslInt(isl_ast_expr_get_val(incExpr), stepVal) &&
                  stepVal > 0;
    isl_ast_expr_free(incExpr);
    if (!stepOk) {
      isl_ast_expr_free(initExpr);
      isl_ast_expr_free(condExpr);
      isl_id_free(iterIdOut);
      return false;
    }
    stepOut = stepVal;

    auto lbList = buildBoundExprList(initExpr, isl_ast_op_max);
    isl_ast_expr_free(initExpr);
    if (!lbList) {
      isl_ast_expr_free(condExpr);
      isl_id_free(iterIdOut);
      return false;
    }
    lbMap =
        AffineMap::get(lbList->dims.size(), lbList->syms.size(), lbList->exprs, ctx);
    lbOps = lbList->dims;
    lbOps.append(lbList->syms.begin(), lbList->syms.end());

    if (isl_ast_expr_get_type(condExpr) != isl_ast_expr_op) {
      isl_ast_expr_free(condExpr);
      isl_id_free(iterIdOut);
      return false;
    }
    isl_ast_op_type cmpOp = isl_ast_expr_get_op_type(condExpr);
    if (cmpOp != isl_ast_op_le && cmpOp != isl_ast_op_lt) {
      isl_ast_expr_free(condExpr);
      isl_id_free(iterIdOut);
      return false;
    }
    isl_ast_expr *rhsCond = isl_ast_expr_get_op_arg(condExpr, 1);
    auto ubList = buildBoundExprList(rhsCond, isl_ast_op_min);
    isl_ast_expr_free(rhsCond);
    isl_ast_expr_free(condExpr);
    if (!ubList) {
      isl_id_free(iterIdOut);
      return false;
    }
    if (cmpOp == isl_ast_op_le) {
      for (AffineExpr &e : ubList->exprs)
        e = e + 1;
    }
    ubMap =
        AffineMap::get(ubList->dims.size(), ubList->syms.size(), ubList->exprs, ctx);
    ubOps = ubList->dims;
    ubOps.append(ubList->syms.begin(), ubList->syms.end());
    return true;
  }

  bool createGridParallel(__isl_take isl_ast_node *forNode, unsigned nLoops) {
    SmallVector<AffineMap, 3> lbMaps, ubMaps;
    SmallVector<int64_t, 3> steps;
    SmallVector<isl_id *, 3> iterIds;

    // Unify all bound operands into common `(lbDims, lbSyms)` and `(ubDims, ubSyms)`.
    BuiltAffineExpr lbUnifier{getAffineConstantExpr(0, ctx), {}, {}};
    BuiltAffineExpr ubUnifier{getAffineConstantExpr(0, ctx), {}, {}};
    SmallVector<SmallVector<AffineExpr, 2>, 3> perLoopLbExprs(nLoops);
    SmallVector<SmallVector<AffineExpr, 2>, 3> perLoopUbExprs(nLoops);

    isl_ast_node *cur = forNode;
    for (unsigned i = 0; i < nLoops; ++i) {
      if (!cur || isl_ast_node_get_type(cur) != isl_ast_node_for) {
        // ISL eliminated a single-value affine tile loop (substituting its
        // value directly into all inner expressions); represent it as `[0, 1)`.
        perLoopLbExprs[i].push_back(getAffineConstantExpr(0, ctx));
        perLoopUbExprs[i].push_back(getAffineConstantExpr(1, ctx));
        steps.push_back(1);
        iterIds.push_back(nullptr);
        continue;
      }
      AffineMap lMap, uMap;
      SmallVector<Value, 8> lOps, uOps;
      int64_t step = 1;
      isl_id *iterId = nullptr;
      if (!extractLoopBounds(cur, lMap, lOps, uMap, uOps, step, iterId)) {
        isl_ast_node_free(cur);
        for (isl_id *id : iterIds)
          if (id)
            isl_id_free(id);
        return false;
      }
      iterIds.push_back(iterId);
      steps.push_back(step);

      SmallVector<Value, 8> lDims(lOps.begin(), lOps.begin() + lMap.getNumDims());
      SmallVector<Value, 8> lSyms(lOps.begin() + lMap.getNumDims(), lOps.end());
      for (AffineExpr e : lMap.getResults()) {
        BuiltAffineExpr item{e, lDims, lSyms};
        auto [_, mapped] = unifyOperands(lbUnifier, item);
        (void)_;
        perLoopLbExprs[i].push_back(mapped);
      }

      SmallVector<Value, 8> uDims(uOps.begin(), uOps.begin() + uMap.getNumDims());
      SmallVector<Value, 8> uSyms(uOps.begin() + uMap.getNumDims(), uOps.end());
      for (AffineExpr e : uMap.getResults()) {
        BuiltAffineExpr item{e, uDims, uSyms};
        auto [_, mapped] = unifyOperands(ubUnifier, item);
        (void)_;
        perLoopUbExprs[i].push_back(mapped);
      }

      isl_ast_node *nextBody = isl_ast_node_for_get_body(cur);
      isl_ast_node_free(cur);
      cur = nextBody;
    }
    isl_ast_node *body = cur;

    for (unsigned i = 0; i < nLoops; ++i) {
      lbMaps.push_back(AffineMap::get(lbUnifier.dims.size(),
                                      lbUnifier.syms.size(), perLoopLbExprs[i],
                                      ctx));
      ubMaps.push_back(AffineMap::get(ubUnifier.dims.size(),
                                      ubUnifier.syms.size(), perLoopUbExprs[i],
                                      ctx));
    }

    SmallVector<Value, 8> lbOperands = lbUnifier.allOperands();
    SmallVector<Value, 8> ubOperands = ubUnifier.allOperands();
    auto gridPar = affine::AffineParallelOp::create(
        b, loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), lbMaps,
        lbOperands, ubMaps, ubOperands, steps);
    gridPar->setAttr(polymer::kGpuGridParAttr, b.getUnitAttr());

    for (unsigned i = 0; i < nLoops; ++i) {
      if (iterIds[i]) {
        idToValue[iterIds[i]] = gridPar.getIVs()[i];
        isl_id_free(iterIds[i]);
      }
    }

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(gridPar.getBody());
    return createNode(body);
  }

  bool createBlockParallel(__isl_take isl_ast_node *forNode,
                           ArrayRef<int64_t> blockSizes) {
    unsigned nLoops = blockSizes.size();
    SmallVector<AffineMap, 3> constLbMaps, constUbMaps;
    SmallVector<int64_t, 3> constSteps(nLoops, 1);
    for (int64_t sz : blockSizes) {
      constLbMaps.push_back(AffineMap::getConstantMap(0, ctx));
      constUbMaps.push_back(AffineMap::getConstantMap(sz, ctx));
    }

    auto blockPar = affine::AffineParallelOp::create(
        b, loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), constLbMaps,
        ValueRange(), constUbMaps, ValueRange(), constSteps);
    blockPar->setAttr(polymer::kGpuBlockParAttr, b.getUnitAttr());

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(blockPar.getBody());

    // Extract actual AST loop bounds for each point loop `k = 0 .. nLoops-1`;
    // any non-trivial bound (`lb > 0` or `ub < B_k`) becomes an `affine.if`
    // guard INSIDE `blockPar` so `blockPar` always has uniform `(0..B_k)` bounds!
    BuiltAffineExpr guardUnifier{getAffineConstantExpr(0, ctx), {}, {}};
    SmallVector<AffineExpr, 6> guardExprs;
    SmallVector<bool, 6> guardEqs;

    isl_ast_node *cur = forNode;
    for (unsigned k = 0; k < nLoops; ++k) {
      Value tIV = blockPar.getIVs()[k];
      if (!cur || isl_ast_node_get_type(cur) != isl_ast_node_for) {
        // ISL eliminated a single-value point loop (already substituting its
        // affine value into inner expressions); restrict to `tIV == 0`.
        BuiltAffineExpr ivBuilt{getAffineDimExpr(0, ctx), {tIV}, {}};
        auto [_, ivMapped] = unifyOperands(guardUnifier, ivBuilt);
        (void)_;
        guardExprs.push_back(ivMapped);
        guardEqs.push_back(true);
        continue;
      }
      bool isDegenerate =
          (isl_ast_node_for_is_degenerate(cur) == isl_bool_true);
      AffineMap lMap, uMap;
      SmallVector<Value, 8> lOps, uOps;
      int64_t step = 1;
      isl_id *iterId = nullptr;
      if (!extractLoopBounds(cur, lMap, lOps, uMap, uOps, step, iterId) ||
          step != 1) {
        if (iterId)
          isl_id_free(iterId);
        isl_ast_node_free(cur);
        return false;
      }

      idToValue[iterId] = tIV;
      isl_id_free(iterId);

      BuiltAffineExpr ivBuilt{getAffineDimExpr(0, ctx), {tIV}, {}};
      auto [_, ivExprInGuard] = unifyOperands(guardUnifier, ivBuilt);
      (void)_;
      (void)ivExprInGuard;

      SmallVector<Value, 8> lDims(lOps.begin(), lOps.begin() + lMap.getNumDims());
      SmallVector<Value, 8> lSyms(lOps.begin() + lMap.getNumDims(), lOps.end());

      if (isDegenerate) {
        BuiltAffineExpr initItem{lMap.getResult(0), lDims, lSyms};
        auto [curIv, initMapped] = unifyOperands(guardUnifier, initItem);
        (void)curIv;
        // Re-lookup `tIV` dim position in `guardUnifier`.
        unsigned ivPos =
            std::distance(guardUnifier.dims.begin(),
                          llvm::find(guardUnifier.dims, tIV));
        guardExprs.push_back(getAffineDimExpr(ivPos, ctx) - initMapped);
        guardEqs.push_back(true);
      } else {
        for (AffineExpr le : lMap.getResults()) {
          if (auto cst = dyn_cast<AffineConstantExpr>(le)) {
            if (cst.getValue() <= 0)
              continue;
          }
          BuiltAffineExpr lItem{le, lDims, lSyms};
          auto [_, lMapped] = unifyOperands(guardUnifier, lItem);
          (void)_;
          unsigned ivPos =
              std::distance(guardUnifier.dims.begin(),
                            llvm::find(guardUnifier.dims, tIV));
          // `tIV >= lMapped`  <=>  `tIV - lMapped >= 0`
          guardExprs.push_back(getAffineDimExpr(ivPos, ctx) - lMapped);
          guardEqs.push_back(false);
        }

        SmallVector<Value, 8> uDims(uOps.begin(),
                                    uOps.begin() + uMap.getNumDims());
        SmallVector<Value, 8> uSyms(uOps.begin() + uMap.getNumDims(),
                                    uOps.end());
        for (AffineExpr ue : uMap.getResults()) {
          if (auto cst = dyn_cast<AffineConstantExpr>(ue)) {
            if (cst.getValue() >= blockSizes[k])
              continue;
          }
          BuiltAffineExpr uItem{ue, uDims, uSyms};
          auto [_, uMapped] = unifyOperands(guardUnifier, uItem);
          (void)_;
          unsigned ivPos =
              std::distance(guardUnifier.dims.begin(),
                            llvm::find(guardUnifier.dims, tIV));
          // `tIV < uMapped`  <=>  `uMapped - 1 - tIV >= 0`
          guardExprs.push_back(uMapped - 1 - getAffineDimExpr(ivPos, ctx));
          guardEqs.push_back(false);
        }
      }

      isl_ast_node *nextBody = isl_ast_node_for_get_body(cur);
      isl_ast_node_free(cur);
      cur = nextBody;
    }
    isl_ast_node *body = cur;

    if (!guardExprs.empty()) {
      IntegerSet gSet = IntegerSet::get(guardUnifier.dims.size(),
                                        guardUnifier.syms.size(), guardExprs,
                                        guardEqs);
      SmallVector<Value, 8> gOps = guardUnifier.allOperands();
      auto ifOp = affine::AffineIfOp::create(b, loc, TypeRange(), gSet, gOps,
                                             /*withElseRegion=*/false);
      b.setInsertionPointToStart(ifOp.getThenBlock());
    }

    return createNode(body);
  }

  bool createFor(__isl_take isl_ast_node *forNode) {
    AffineMap lbMap, ubMap;
    SmallVector<Value, 8> lbOps, ubOps;
    int64_t step = 1;
    isl_id *iterId = nullptr;
    if (!extractLoopBounds(forNode, lbMap, lbOps, ubMap, ubOps, step, iterId)) {
      isl_ast_node_free(forNode);
      return false;
    }
    isl_ast_node *body = isl_ast_node_for_get_body(forNode);
    isl_ast_node_free(forNode);

    auto forOp =
        affine::AffineForOp::create(b, loc, lbOps, lbMap, ubOps, ubMap, step);
    idToValue[iterId] = forOp.getInductionVar();
    isl_id_free(iterId);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(forOp.getBody());
    return createNode(body);
  }

  bool appendConjunctiveCond(__isl_keep isl_ast_expr *cond,
                             BuiltAffineExpr &unifier,
                             SmallVectorImpl<AffineExpr> &exprs,
                             SmallVectorImpl<bool> &eqFlags) {
    if (!cond || isl_ast_expr_get_type(cond) != isl_ast_expr_op)
      return false;
    isl_ast_op_type opTy = isl_ast_expr_get_op_type(cond);
    if (opTy == isl_ast_op_and || opTy == isl_ast_op_and_then) {
      isl_ast_expr *c0 = isl_ast_expr_get_op_arg(cond, 0);
      isl_ast_expr *c1 = isl_ast_expr_get_op_arg(cond, 1);
      bool ok = appendConjunctiveCond(c0, unifier, exprs, eqFlags) &&
                appendConjunctiveCond(c1, unifier, exprs, eqFlags);
      isl_ast_expr_free(c0);
      isl_ast_expr_free(c1);
      return ok;
    }

    if (isl_ast_expr_get_op_n_arg(cond) != 2)
      return false;
    isl_ast_expr *a0 = isl_ast_expr_get_op_arg(cond, 0);
    isl_ast_expr *a1 = isl_ast_expr_get_op_arg(cond, 1);
    auto lhs = buildAffineExpr(a0);
    auto rhs = buildAffineExpr(a1);
    isl_ast_expr_free(a0);
    isl_ast_expr_free(a1);
    if (!lhs || !rhs)
      return false;

    auto [_, lMap] = unifyOperands(unifier, *lhs);
    (void)_;
    auto [__, rMap] = unifyOperands(unifier, *rhs);
    (void)__;
    // Re-unify `lhs` in case `rhs` appended dims to `unifier`? Note `unifyOperands`
    // only appends to the end of `unifier.dims` / `unifier.syms`, so existing
    // dim/sym indices in `lMap` remain unchanged!
    switch (opTy) {
    case isl_ast_op_eq:
      exprs.push_back(lMap - rMap);
      eqFlags.push_back(true);
      return true;
    case isl_ast_op_ge:
      exprs.push_back(lMap - rMap);
      eqFlags.push_back(false);
      return true;
    case isl_ast_op_gt:
      exprs.push_back(lMap - rMap - 1);
      eqFlags.push_back(false);
      return true;
    case isl_ast_op_le:
      exprs.push_back(rMap - lMap);
      eqFlags.push_back(false);
      return true;
    case isl_ast_op_lt:
      exprs.push_back(rMap - lMap - 1);
      eqFlags.push_back(false);
      return true;
    default:
      return false;
    }
  }

  bool emitCondBranch(__isl_keep isl_ast_expr *cond,
                      __isl_keep isl_ast_node *thenNode,
                      __isl_keep isl_ast_node *elseNode) {
    if (isl_ast_expr_get_type(cond) == isl_ast_expr_op) {
      isl_ast_op_type opTy = isl_ast_expr_get_op_type(cond);
      if (opTy == isl_ast_op_or || opTy == isl_ast_op_or_else) {
        // `if (c0 || c1) T else E`  <=>  `if (c0) T else { if (c1) T else E }`
        isl_ast_expr *c0 = isl_ast_expr_get_op_arg(cond, 0);
        isl_ast_expr *c1 = isl_ast_expr_get_op_arg(cond, 1);
        BuiltAffineExpr u0{getAffineConstantExpr(0, ctx), {}, {}};
        SmallVector<AffineExpr, 4> e0;
        SmallVector<bool, 4> eq0;
        if (!appendConjunctiveCond(c0, u0, e0, eq0)) {
          isl_ast_expr_free(c0);
          isl_ast_expr_free(c1);
          return false;
        }
        isl_ast_expr_free(c0);
        IntegerSet set0 =
            IntegerSet::get(u0.dims.size(), u0.syms.size(), e0, eq0);
        auto if0 = affine::AffineIfOp::create(
            b, loc, TypeRange(), set0, u0.allOperands(), /*withElseRegion=*/true);
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(if0.getThenBlock());
          if (!createNode(isl_ast_node_copy(thenNode))) {
            isl_ast_expr_free(c1);
            return false;
          }
          b.setInsertionPointToStart(if0.getElseBlock());
          bool okElse = emitCondBranch(c1, thenNode, elseNode);
          isl_ast_expr_free(c1);
          return okElse;
        }
      }
    }

    BuiltAffineExpr unifier{getAffineConstantExpr(0, ctx), {}, {}};
    SmallVector<AffineExpr, 4> exprs;
    SmallVector<bool, 4> eqFlags;
    if (!appendConjunctiveCond(cond, unifier, exprs, eqFlags))
      return false;

    bool hasElse = (elseNode != nullptr);
    IntegerSet iset = IntegerSet::get(unifier.dims.size(), unifier.syms.size(),
                                      exprs, eqFlags);
    auto ifOp = affine::AffineIfOp::create(
        b, loc, TypeRange(), iset, unifier.allOperands(), hasElse);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(ifOp.getThenBlock());
    if (!createNode(isl_ast_node_copy(thenNode)))
      return false;
    if (hasElse) {
      b.setInsertionPointToStart(ifOp.getElseBlock());
      if (!createNode(isl_ast_node_copy(elseNode)))
        return false;
    }
    return true;
  }

  bool createIf(__isl_take isl_ast_node *ifNode) {
    isl_ast_expr *cond = isl_ast_node_if_get_cond(ifNode);
    isl_ast_node *thenNode = isl_ast_node_if_get_then(ifNode);
    isl_ast_node *elseNode =
        (isl_ast_node_if_has_else(ifNode) == isl_bool_true)
            ? isl_ast_node_if_get_else(ifNode)
            : nullptr;

    bool ok = emitCondBranch(cond, thenNode, elseNode);
    isl_ast_expr_free(cond);
    isl_ast_node_free(thenNode);
    if (elseNode)
      isl_ast_node_free(elseNode);
    isl_ast_node_free(ifNode);
    return ok;
  }

  bool createBlock(__isl_take isl_ast_node *blockNode) {
    isl_ast_node_list *list = isl_ast_node_block_get_children(blockNode);
    int n = isl_ast_node_list_n_ast_node(list);
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
      ok = createNode(isl_ast_node_list_get_ast_node(list, i));
    isl_ast_node_list_free(list);
    isl_ast_node_free(blockNode);
    return ok;
  }

  bool createMark(__isl_take isl_ast_node *markNode) {
    isl_id *id = isl_ast_node_mark_get_id(markNode);
    isl_ast_node *child = isl_ast_node_mark_get_node(markNode);
    isl_ast_node_free(markNode);

    llvm::StringRef name = isl_id_get_name(id);
    if (name == polymer::gridParallelMark) {
      auto nMembers =
          static_cast<unsigned>(reinterpret_cast<uintptr_t>(isl_id_get_user(id)));
      isl_id_free(id);
      return createGridParallel(child, nMembers);
    }
    if (name == kBlockParallelMark) {
      auto *info = static_cast<BlockParallelMarkInfo *>(isl_id_get_user(id));
      isl_id_free(id);
      return createBlockParallel(child, info->blockSizes);
    }
    if (name == "parallel") {
      isl_id_free(id);
      return createNode(child);
    }
    isl_id_free(id);
    return createNode(child);
  }

  bool createUser(__isl_take isl_ast_node *userNode) {
    isl_ast_expr *callExpr = isl_ast_node_user_get_expr(userNode);
    isl_ast_expr *calleeExpr = isl_ast_expr_get_op_arg(callExpr, 0);
    isl_id *calleeId = isl_ast_expr_get_id(calleeExpr);
    llvm::StringRef calleeName = isl_id_get_name(calleeId);

    SmallVector<Value, 8> ivValues;
    int nArgs = isl_ast_expr_get_op_n_arg(callExpr);
    for (int i = 1; i < nArgs; ++i) {
      isl_ast_expr *sub = isl_ast_expr_get_op_arg(callExpr, i);
      if (isl_ast_expr_get_type(sub) == isl_ast_expr_op &&
          isl_ast_expr_get_op_type(sub) == isl_ast_op_call) {
        isl_ast_expr_free(sub);
        break;
      }
      auto built = buildAffineExpr(sub);
      isl_ast_expr_free(sub);
      if (!built) {
        isl_id_free(calleeId);
        isl_ast_expr_free(calleeExpr);
        isl_ast_expr_free(callExpr);
        isl_ast_node_free(userNode);
        return false;
      }
      // Shortcut identity dim/sym without creating a redundant `affine.apply`.
      if (auto dE = dyn_cast<AffineDimExpr>(built->expr)) {
        ivValues.push_back(built->dims[dE.getPosition()]);
      } else if (auto sE = dyn_cast<AffineSymbolExpr>(built->expr)) {
        ivValues.push_back(built->syms[sE.getPosition()]);
      } else {
        Value v = affine::AffineApplyOp::create(
                      b, loc, built->toMap(ctx), built->allOperands())
                      .getResult();
        ivValues.push_back(v);
      }
    }

    // Look up matching `ScopStmt` in `scop`.
    polymer::ScopStmt *foundStmt = nullptr;
    for (polymer::ScopStmt &s : scop) {
      if (s.getName() == calleeName) {
        foundStmt = &s;
        break;
      }
    }
    isl_id_free(calleeId);
    isl_ast_expr_free(calleeExpr);
    isl_ast_expr_free(callExpr);
    isl_ast_node_free(userNode);
    if (!foundStmt)
      return false;

    Operation *origOp = foundStmt->getOperation();
    llvm::SetVector<Value> extValues;
    origOp->walk([&](Operation *nested) {
      for (Value opr : nested->getOperands()) {
        if (opr.getParentRegion()->getParentOp()->isProperAncestor(origOp))
          extValues.insert(opr);
      }
    });

    SmallVector<Operation *, 8> enclosing;
    foundStmt->getEnclosingAffineOps(enclosing);

    IRMapping stmtMapping = funcMapping;
    for (Value origArg : extValues) {
      if (auto ba = dyn_cast<BlockArgument>(origArg)) {
        Operation *owner = ba.getOwner()->getParentOp();
        if (isa<affine::AffineForOp, affine::AffineParallelOp>(owner)) {
          unsigned ivIdx = 0;
          for (Operation *enc : enclosing) {
            if (isa<affine::AffineIfOp>(enc))
              continue;
            if (auto par = dyn_cast<affine::AffineParallelOp>(enc)) {
              if (owner == enc) {
                ivIdx += ba.getArgNumber();
                break;
              }
              ivIdx += par.getNumDims();
            } else if (isa<affine::AffineForOp>(enc)) {
              if (owner == enc)
                break;
              ivIdx += 1;
            }
          }
          if (ivIdx >= ivValues.size())
            return false;
          stmtMapping.map(origArg, ivValues[ivIdx]);
        }
      }
    }

    SmallVector<Operation *, 8> clonedOps;
    if (auto groupIf = dyn_cast<affine::AffineIfOp>(origOp);
        groupIf && groupIf->hasAttr(kStmtGroupAttr)) {
      for (Operation &inner : groupIf.getThenBlock()->without_terminator()) {
        Operation *cloned = b.clone(inner, stmtMapping);
        clonedOps.push_back(cloned);
      }
    } else if (origOp->hasTrait<OpTrait::IsTerminator>()) {
      return true;
    } else {
      Operation *cloned = b.clone(*origOp, stmtMapping);
      cloned->removeAttr("polymer.stmt.name");
      cloned->removeAttr("polymer.stmt.async.copy");
      clonedOps.push_back(cloned);
      for (auto [oldR, newR] :
           llvm::zip(origOp->getResults(), cloned->getResults()))
        funcMapping.map(oldR, newR);
    }

    // Fold any newly generated `affine.apply` ops directly into cloned affine
    // loads/stores/loops/ifs!
    for (Operation *cOp : clonedOps) {
      cOp->walk([&](Operation *nested) {
        nested->removeAttr("polymer.stmt.name");
        nested->removeAttr("polymer.stmt.async.copy");
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(nested)) {
          AffineMap m = loadOp.getAffineMap();
          SmallVector<Value, 8> ops(loadOp.getMapOperands().begin(),
                                    loadOp.getMapOperands().end());
          affine::fullyComposeAffineMapAndOperands(&m, &ops);
          affine::canonicalizeMapAndOperands(&m, &ops);
          m = simplifyAffineMap(m);
          affine::canonicalizeMapAndOperands(&m, &ops);
          OpBuilder lb(loadOp);
          auto repl =
              affine::AffineLoadOp::create(lb, loadOp.getLoc(), loadOp.getMemRef(), m, ops);
          loadOp.replaceAllUsesWith(repl.getResult());
          loadOp.erase();
        } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(nested)) {
          AffineMap m = storeOp.getAffineMap();
          SmallVector<Value, 8> ops(storeOp.getMapOperands().begin(),
                                    storeOp.getMapOperands().end());
          affine::fullyComposeAffineMapAndOperands(&m, &ops);
          affine::canonicalizeMapAndOperands(&m, &ops);
          m = simplifyAffineMap(m);
          affine::canonicalizeMapAndOperands(&m, &ops);
          OpBuilder sb(storeOp);
          affine::AffineStoreOp::create(sb, storeOp.getLoc(),
                                        storeOp.getValueToStore(),
                                        storeOp.getMemRef(), m, ops);
          storeOp.erase();
        }
      });
    }

    return true;
  }

  bool createNode(__isl_take isl_ast_node *node) {
    if (!node)
      return false;
    switch (isl_ast_node_get_type(node)) {
    case isl_ast_node_error:
      isl_ast_node_free(node);
      return false;
    case isl_ast_node_mark:
      return createMark(node);
    case isl_ast_node_for:
      return createFor(node);
    case isl_ast_node_if:
      return createIf(node);
    case isl_ast_node_block:
      return createBlock(node);
    case isl_ast_node_user:
      return createUser(node);
    }
    isl_ast_node_free(node);
    return false;
  }
};

/// Hoist loop-invariant constants & top-level `index_cast`s to `entryBlock` top,
/// demote any cross-slice SSA scalars inside loops to 0-D `memref.alloca`s, and
/// wrap maximal loop-free statement slices in `{polymer.stmt_group}` `affine.if`
/// ops so `IslScop::rescopeStatements` groups intra-block SSA chains cleanly.
static void wrapStatementGroups(Operation *rootOp,
                                SmallVectorImpl<affine::AffineIfOp> &wrappers) {
  Block &entryBlock = rootOp->getRegion(0).front();
  OpBuilder builder(rootOp->getContext());

  // 1. Hoist `ConstantLike` ops and `arith.index_cast` of entry block args to
  // the start of `entryBlock`.
  SmallVector<Operation *, 16> toHoist;
  rootOp->walk([&](Operation *op) {
    if (op->getBlock() == &entryBlock)
      return;
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      toHoist.push_back(op);
    } else if (auto ic = dyn_cast<arith::IndexCastOp>(op)) {
      if (auto ba = dyn_cast<BlockArgument>(ic.getIn())) {
        if (ba.getOwner() == &entryBlock)
          toHoist.push_back(op);
      }
    }
  });
  for (Operation *op : toHoist)
    op->moveBefore(&entryBlock.front());

  // 2. Demote any SSA value defined inside a nested block that is used in a
  // different block to a 0-D `memref.alloca` in `entryBlock`.
  SmallVector<Value, 8> crossBlockVals;
  rootOp->walk([&](Operation *op) {
    if (op->getBlock() == &entryBlock ||
        isa<affine::AffineForOp, affine::AffineParallelOp, affine::AffineIfOp>(
            op))
      return;
    for (Value res : op->getResults()) {
      if (!res.getType().isIntOrIndexOrFloat())
        continue;
      bool escapesBlock = llvm::any_of(res.getUsers(), [&](Operation *user) {
        return user->getBlock() != op->getBlock();
      });
      if (escapesBlock)
        crossBlockVals.push_back(res);
    }
  });

  AffineMap emptyMap = AffineMap::get(rootOp->getContext());
  for (Value val : crossBlockVals) {
    Operation *defOp = val.getDefiningOp();
    builder.setInsertionPointToStart(&entryBlock);
    auto slot = memref::AllocaOp::create(
        builder, defOp->getLoc(), MemRefType::get({}, val.getType()));
    builder.setInsertionPointAfter(defOp);
    SmallVector<OpOperand *, 8> externalUses;
    for (OpOperand &use : val.getUses()) {
      if (use.getOwner()->getBlock() != defOp->getBlock())
        externalUses.push_back(&use);
    }
    affine::AffineStoreOp::create(builder, defOp->getLoc(), val,
                                  slot.getMemref(), emptyMap, ValueRange{});
    for (OpOperand *use : externalUses) {
      builder.setInsertionPoint(use->getOwner());
      auto ld = affine::AffineLoadOp::create(builder, use->getOwner()->getLoc(),
                                             slot.getMemref(), emptyMap,
                                             ValueRange{});
      use->set(ld.getResult());
    }
  }

  // 3. Wrap maximal contiguous slices of leaf operations inside affine loops/ifs
  // in `{polymer.stmt_group}` trivial `affine.if (0 == 0)` ops.
  IntegerSet trivialTrueSet = IntegerSet::get(
      0, 0, {getAffineConstantExpr(0, rootOp->getContext())}, {true});

  std::function<void(Block &, bool)> wrapInBlock = [&](Block &blk,
                                                       bool insideLoop) {
    SmallVector<Operation *, 16> ops;
    for (Operation &op : blk.without_terminator())
      ops.push_back(&op);

    size_t i = 0;
    while (i < ops.size()) {
      Operation *cur = ops[i];
      if (auto forOp = dyn_cast<affine::AffineForOp>(cur)) {
        wrapInBlock(*forOp.getBody(), /*insideLoop=*/true);
        ++i;
        continue;
      }
      if (auto parOp = dyn_cast<affine::AffineParallelOp>(cur)) {
        if (!polymer::isBlockParallel(parOp))
          wrapInBlock(*parOp.getBody(), /*insideLoop=*/true);
        ++i;
        continue;
      }
      if (auto ifOp = dyn_cast<affine::AffineIfOp>(cur)) {
        wrapInBlock(*ifOp.getThenBlock(), insideLoop);
        if (ifOp.hasElse())
          wrapInBlock(*ifOp.getElseBlock(), insideLoop);
        ++i;
        continue;
      }
      if (!insideLoop) {
        ++i;
        continue;
      }

      // Collect maximal slice `[i .. j)` of leaf ops whose results aren't used
      // by structured ops in the same block after `j`.
      size_t j = i;
      while (j < ops.size() &&
             !isa<affine::AffineForOp, affine::AffineParallelOp,
                  affine::AffineIfOp>(ops[j])) {
        ++j;
      }

      // Check that no op in `[i .. j)` defines an SSA value used outside `[i .. j)`
      // (e.g. a loop bound operand of a sibling inner `affine.for` in `blk`);
      // if one does, split before/after it or demote it.
      DenseSet<Operation *> sliceSet(ops.begin() + i, ops.begin() + j);
      bool hasEscapingSSA = false;
      for (size_t k = i; k < j; ++k) {
        for (Value res : ops[k]->getResults()) {
          for (Operation *u : res.getUsers()) {
            if (!sliceSet.contains(u))
              hasEscapingSSA = true;
          }
        }
      }

      if (!hasEscapingSSA) {
        builder.setInsertionPoint(ops[i]);
        auto groupIf = affine::AffineIfOp::create(
            builder, ops[i]->getLoc(), TypeRange(), trivialTrueSet,
            ValueRange{}, /*withElseRegion=*/false);
        groupIf->setAttr(kStmtGroupAttr, builder.getUnitAttr());
        Operation *yieldOp = groupIf.getThenBlock()->getTerminator();
        for (size_t k = i; k < j; ++k)
          ops[k]->moveBefore(yieldOp);
        wrappers.push_back(groupIf);
      }
      i = j;
    }
  };

  wrapInBlock(entryBlock, /*insideLoop=*/false);
}

static void unwrapStatementGroups(ArrayRef<affine::AffineIfOp> wrappers) {
  for (affine::AffineIfOp groupIf : wrappers) {
    if (!groupIf || !groupIf->getBlock())
      continue;
    for (Operation &op :
         llvm::make_early_inc_range(groupIf.getThenBlock()->without_terminator())) {
      op.moveBefore(groupIf);
    }
    groupIf.erase();
  }
}

static bool canAnalyzePolyhedrally(Operation *rootOp) {
  if (rootOp->getNumRegions() != 1 ||
      !llvm::hasSingleElement(rootOp->getRegion(0)))
    return false;
  bool hasAffineLoop = false;
  bool hasUnsupportedControl = false;
  rootOp->getRegion(0).walk([&](Operation *op) {
    if (op == rootOp)
      return WalkResult::advance();
    if (isa<affine::AffineForOp, affine::AffineParallelOp>(op))
      hasAffineLoop = true;
    if (isa<scf::ForOp, scf::ParallelOp, scf::WhileOp, scf::IfOp>(op)) {
      hasUnsupportedControl = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasAffineLoop && !hasUnsupportedControl;
}

/// Run ISL scheduling + GPU tiling + Affine AST code generation on `rootOp`.
static LogicalResult scheduleAndTileRegion(Operation *rootOp,
                                           ArrayRef<int64_t> blockSizes,
                                           int64_t seqTileSize,
                                           int64_t maxSharedMemBytes) {
  if (!canAnalyzePolyhedrally(rootOp))
    return success();

  bool hasGridPar = false;
  unsigned numBlockPars = 0;
  rootOp->walk([&](Operation *op) {
    if (polymer::isGridParallel(op))
      hasGridPar = true;
    if (polymer::isBlockParallel(op))
      ++numBlockPars;
  });

  // If already tiled into a single `gpu.par.grid` + single `gpu.par.block`,
  // preserve its schedule and proceed directly to shared memory promotion.
  if (hasGridPar && numBlockPars <= 1)
    return success();

  SmallVector<affine::AffineIfOp, 8> wrappers;
  if (!hasGridPar)
    wrapStatementGroups(rootOp, wrappers);

  auto cleanupOnExit = [&]() {
    polymer::IslScop *dummy = nullptr;
    (void)dummy;
    rootOp->walk([](enzymexla::AffineStoreVar sv) { sv->erase(); });
    rootOp->walk([](Operation *op) {
      op->removeAttr("polymer.stmt.name");
      op->removeAttr("polymer.stmt.async.copy");
    });
    unwrapStatementGroups(wrappers);
  };

  std::unique_ptr<polymer::IslScop> scop = polymer::createIslFromFuncOp(rootOp);
  if (!scop || failed(scop->buildSchedule())) {
    cleanupOnExit();
    return success();
  }

  if (hasGridPar) {
    scop->rescopeStatements(
        [](Operation *op) { return polymer::getAffineBlockParallel(op) != nullptr; },
        [](Operation *) { return false; });
  } else {
    scop->rescopeStatements(
        [](Operation *op) { return op->hasAttr(kStmtGroupAttr); },
        [](Operation *) { return false; });
  }

  if (failed(scop->buildSchedule())) {
    cleanupOnExit();
    return success();
  }

  ppcg_scop *ps = polymer::computeDeps(*scop);
  if (!ps || isl_union_set_is_empty(ps->domain) == isl_bool_true) {
    polymer::freePpcgScop(ps);
    cleanupOnExit();
    return success();
  }

  isl_schedule_constraints *sc =
      constructScheduleConstraints(ps, *scop, maxSharedMemBytes);
  if (!sc) {
    polymer::freePpcgScop(ps);
    cleanupOnExit();
    return success();
  }

  isl_ctx *ctx = scop->getIslCtx();
  isl_options_set_schedule_whole_component(ctx, 1);
  isl_options_set_schedule_serialize_sccs(ctx, 0);
  isl_options_set_schedule_maximize_band_depth(ctx, 1);
  isl_options_set_schedule_maximize_coincidence(ctx, 1);
  isl_options_set_schedule_outer_coincidence(ctx, 1);
  isl_options_set_ast_build_atomic_upper_bound(ctx, 1);
  isl_options_set_ast_build_exploit_nested_bounds(ctx, 1);
  isl_options_set_ast_build_group_coscheduled(ctx, 1);
  isl_options_set_ast_build_allow_else(ctx, 1);

  isl::union_map allDeps = isl::manage(isl_union_map_union(
      isl_union_map_copy(ps->dep_flow),
      isl_union_map_union(isl_union_map_copy(ps->dep_false),
                          isl_union_map_copy(ps->dep_forced))));
  isl::schedule newSchedule =
      isl::manage(isl_schedule_constraints_compute_schedule(sc));
  polymer::freePpcgScop(ps);

  if (newSchedule.is_null()) {
    cleanupOnExit();
    return success();
  }

  std::vector<std::unique_ptr<BlockParallelMarkInfo>> bpMarks;
  isl::schedule gpuSchedule = tileScheduleForGPU(
      newSchedule, blockSizes, seqTileSize, hasGridPar, allDeps.get(), bpMarks);
  if (gpuSchedule.is_null()) {
    cleanupOnExit();
    return success();
  }

  // Verify the tiled schedule actually introduced at least one `grid_parallel`
  // mark (i.e. at least 1 parallel dimension existed).
  bool foundGridMark = false;
  isl_schedule_foreach_schedule_node_top_down(
      gpuSchedule.get(),
      [](__isl_keep isl_schedule_node *n, void *user) -> isl_bool {
        if (isl_schedule_node_get_type(n) == isl_schedule_node_mark) {
          isl_id *id = isl_schedule_node_mark_get_id(n);
          if (llvm::StringRef(isl_id_get_name(id)) == polymer::gridParallelMark)
            *static_cast<bool *>(user) = true;
          isl_id_free(id);
        }
        return isl_bool_true;
      },
      &foundGridMark);
  if (!foundGridMark) {
    cleanupOnExit();
    return success();
  }

  // Clone `rootOp` shell & non-affine preamble ops into a temporary sibling op;
  // only replace `rootOp`'s body if AST emission succeeds 100%!
  IRMapping oldToNew;
  OpBuilder moduleBuilder(rootOp);
  Operation *newRoot = moduleBuilder.cloneWithoutRegions(*rootOp, oldToNew);
  Block &origEntry = rootOp->getRegion(0).front();
  OpBuilder bodyBuilder(rootOp->getContext());
  Block *newEntry = bodyBuilder.createBlock(
      &newRoot->getRegion(0), newRoot->getRegion(0).begin(),
      origEntry.getArgumentTypes(),
      llvm::map_to_vector(origEntry.getArguments(),
                          [](BlockArgument a) { return a.getLoc(); }));
  oldToNew.map(origEntry.getArguments(), newEntry->getArguments());

  for (Operation &op : origEntry.getOperations()) {
    if (isa<enzymexla::AffineStoreVar>(&op))
      continue;
    if (!isa<affine::AffineDialect>(op.getDialect()))
      bodyBuilder.clone(op, oldToNew);
  }

  OpBuilder astBuilder(newEntry->getTerminator());
  IslAffineBuilder affineCodegen(astBuilder, oldToNew, *scop);
  isl_union_set *schedDomain = isl_schedule_get_domain(gpuSchedule.get());
  affineCodegen.mapParams(schedDomain);

  isl_ast_build *build = isl_ast_build_alloc(ctx);
  isl_ast_node *astRoot =
      isl_ast_build_node_from_schedule(build, gpuSchedule.copy());
  isl_ast_build_free(build);

  bool emittedOk = affineCodegen.emit(astRoot);
  if (!emittedOk) {
    newRoot->erase();
    cleanupOnExit();
    return success();
  }

  // Erase dead `affine.apply` ops left after map composition in `newRoot`.
  SmallVector<affine::AffineApplyOp, 16> deadApplies;
  newRoot->walk([&](affine::AffineApplyOp applyOp) {
    if (applyOp.use_empty())
      deadApplies.push_back(applyOp);
  });
  for (affine::AffineApplyOp applyOp : deadApplies)
    applyOp.erase();

  cleanupOnExit();

  // Splice `newRoot`'s transformed region into `rootOp` and erase `newRoot`.
  origEntry.erase();
  rootOp->getRegion(0).getBlocks().splice(rootOp->getRegion(0).begin(),
                                          newRoot->getRegion(0).getBlocks());
  newRoot->erase();
  return success();
}

static LogicalResult lowerRegionAffineToSCF(Operation *rootOp) {
  MLIRContext *ctx = rootOp->getContext();
  RewritePatternSet patterns(ctx);
  populateAffineToStdConversionPatterns(patterns);
  populateAffineToVectorConversionPatterns(patterns);
  affine::populateAffineExpandIndexOpsPatterns(patterns);

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect, vector::VectorDialect,
                         enzymexla::EnzymeXLADialect, func::FuncDialect>();
  target.addIllegalDialect<affine::AffineDialect>();

  if (failed(applyPartialConversion(rootOp, target, std::move(patterns))))
    return failure();

  // Hoist `ConstantLike` ops (such as `arith.constant` bounds materialized for
  // inner `scf.parallel` block loops) out of `scf.parallel` to `rootOp` entry
  // so `ConvertParallelToGPU1` sees only `memref.alloca` ops before `blockPop`.
  Block &entry = rootOp->getRegion(0).front();
  SmallVector<Operation *, 16> constantsToHoist;
  rootOp->walk([&](Operation *op) {
    if (op->getBlock() != &entry && op->hasTrait<OpTrait::ConstantLike>())
      constantsToHoist.push_back(op);
  });
  for (Operation *cst : constantsToHoist)
    cst->moveBefore(&entry.front());

  return success();
}

} // namespace
#endif // DISABLE_POLYMER

namespace {
struct GPUPolyhedralOptPass
    : public enzyme::impl::GPUPolyhedralOptPassBase<GPUPolyhedralOptPass> {
  using GPUPolyhedralOptPassBase::GPUPolyhedralOptPassBase;

  void runOnOperation() override {
#ifndef DISABLE_POLYMER
    Operation *moduleOrRoot = getOperation();

    SmallVector<Operation *, 8> targets;
    auto considerOp = [&](Operation *op) {
      if (isa<enzymexla::GPUWrapperOp>(op)) {
        targets.push_back(op);
        return WalkResult::skip();
      }
      if (auto fn = dyn_cast<func::FuncOp>(op)) {
        if (fn.isDeclaration())
          return WalkResult::skip();
        // If `fn` contains `enzymexla.gpu_wrapper` ops, process each wrapper
        // individually rather than the outer host function.
        bool hasWrapper = false;
        fn.walk([&](enzymexla::GPUWrapperOp) { hasWrapper = true; });
        if (hasWrapper)
          return WalkResult::advance();
        if (processFuncs || fn->hasAttr("gpu.polyhedral") ||
            fn->hasAttr("gpu.par.kernel")) {
          targets.push_back(fn);
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    };

    if (isa<enzymexla::GPUWrapperOp, func::FuncOp>(moduleOrRoot))
      considerOp(moduleOrRoot);
    else
      moduleOrRoot->walk(considerOp);

    SmallVector<int64_t, 3> blockSizes;
    if (blockSizeX > 0)
      blockSizes.push_back(blockSizeX);
    if (blockSizeY > 1)
      blockSizes.push_back(blockSizeY);
    if (blockSizeZ > 1)
      blockSizes.push_back(blockSizeZ);
    if (blockSizes.empty())
      blockSizes.push_back(16);

    polymer::SharedMemoryPromotionOptions shmemOpts;
    shmemOpts.maxSharedMemBytes = maxSharedMemBytes;
    shmemOpts.enableSharedMem = enableSharedMem;
    shmemOpts.enablePrivateMem = enablePrivateMem;

    for (Operation *target : targets) {
      if (reschedule) {
        if (failed(scheduleAndTileRegion(target, blockSizes, tileSize,
                                         maxSharedMemBytes))) {
          signalPassFailure();
          return;
        }
      }

      if (enableSharedMem || enablePrivateMem) {
        if (failed(polymer::promoteSharedMemory(target, shmemOpts))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(polymer::undistributeFunctions(target))) {
        signalPassFailure();
        return;
      }

      if (enablePrivateMem) {
        if (failed(polymer::reducePrivateAllocas(target))) {
          signalPassFailure();
          return;
        }
      }

      if (lowerToSCF) {
        if (failed(lowerRegionAffineToSCF(target))) {
          signalPassFailure();
          return;
        }
      }
    }
#endif
  }
};
} // namespace
