//===- SharedMemoryPromotion.cpp - Polyhedral GPU Shared Mem ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Polyhedral shared-memory and thread-private register opportunity analysis and
// code emission adapted from PPCG (gpu_group.c / gpu.c) for MLIR Affine GPU
// kernels.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Transforms/SharedMemoryPromotion.h"

#include "mlir/Conversion/Polymer/Support/ISLUtils.h"
#include "mlir/Conversion/Polymer/Transforms/LoopUndistribute.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "isl/aff.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/fixed_box.h"
#include "isl/id.h"
#include "isl/isl-noexceptions.h"
#include "isl/local_space.h"
#include "isl/map.h"
#include "isl/mat.h"
#include "isl/options.h"
#include "isl/set.h"
#include "isl/space.h"
#include "isl/stride_info.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "polymer-shared-mem"

using namespace mlir;

namespace polymer {
namespace {

static constexpr char kPrivateAttr[] = "polymer.private";

static bool extractInt64(__isl_take isl_val *v, int64_t &out) {
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

/// Extract constant `[0 .. B_k)` step-1 bounds from a `gpu.par.block` op.
static bool getConstantZeroBasedBounds(affine::AffineParallelOp par,
                                       SmallVectorImpl<int64_t> &upperBounds) {
  upperBounds.clear();
  unsigned nDims = par.getNumDims();
  if (nDims == 0 || nDims > 3)
    return false;
  SmallVector<int64_t, 8> steps = par.getSteps();
  if (steps.size() != nDims)
    return false;
  for (unsigned i = 0; i < nDims; ++i) {
    if (steps[i] != 1)
      return false;
    AffineMap lbMap = par.getLowerBoundMap(i);
    AffineMap ubMap = par.getUpperBoundMap(i);
    if (lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1)
      return false;
    auto lbCst = dyn_cast<AffineConstantExpr>(lbMap.getResult(0));
    auto ubCst = dyn_cast<AffineConstantExpr>(ubMap.getResult(0));
    if (!lbCst || !ubCst || lbCst.getValue() != 0 || ubCst.getValue() <= 0)
      return false;
    upperBounds.push_back(ubCst.getValue());
  }
  return true;
}

static bool isEligibleGlobalMemRef(Value memref,
                                   affine::AffineParallelOp gridPar) {
  auto mt = dyn_cast<MemRefType>(memref.getType());
  if (!mt || mt.getRank() == 0)
    return false;
  Type elemTy = mt.getElementType();
  if (!elemTy.isIntOrFloat())
    return false;
  if (mt.getMemorySpace() && mt.getMemorySpaceAsInt() != 0)
    return false;
  if (Operation *def = memref.getDefiningOp()) {
    if (gridPar->isAncestor(def))
      return false;
  } else if (auto ba = dyn_cast<BlockArgument>(memref)) {
    if (gridPar->isAncestor(ba.getOwner()->getParentOp()))
      return false;
  }
  return true;
}

static isl_mat *createConstraintRows(isl_ctx *ctx,
                                     affine::FlatAffineValueConstraints &cst,
                                     bool isEq) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimVars();
  unsigned numLocalIds = cst.getNumLocalVars();
  unsigned numSymbolIds = cst.getNumSymbolVars();
  unsigned numCols = cst.getNumCols();
  isl_mat *mat = isl_mat_alloc(ctx, numRows, numCols);

  for (unsigned i = 0; i < numRows; ++i) {
    auto row = isEq ? cst.getEquality(i) : cst.getInequality(i);
    for (unsigned j = 0; j < numDimIds; ++j)
      mat = isl_mat_set_element_si(mat, i, j, (int64_t)row[j]);
    for (unsigned j = 0; j < numLocalIds; ++j)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds,
                                   (int64_t)row[j + numDimIds + numSymbolIds]);
    for (unsigned j = 0; j < numSymbolIds; ++j)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds + numLocalIds,
                                   (int64_t)row[j + numDimIds]);
    mat =
        isl_mat_set_element_si(mat, i, numCols - 1, (int64_t)row[numCols - 1]);
  }
  return mat;
}

static void alignSymbolVars(affine::FlatAffineValueConstraints &cst,
                            ArrayRef<Value> paramValues) {
  for (Value sym : paramValues) {
    unsigned pos = 0;
    if (!cst.findVar(sym, &pos))
      cst.appendSymbolVar(sym);
  }
  for (unsigned i = 0; i < paramValues.size(); ++i) {
    unsigned pos = 0;
    bool found = cst.findVar(paramValues[i], &pos);
    (void)found;
    assert(found && "missing symbol var");
    unsigned targetPos = cst.getNumDimVars() + i;
    if (pos != targetPos)
      cst.swapVar(targetPos, pos);
  }
}

/// Construct `[d_0 .. d_{n-1}] -> [d_0 .. d_{pos}+1 .. d_{n-1}]`.
static __isl_give isl_map *makeNextMap(__isl_take isl_space *setSpace,
                                       unsigned pos) {
  isl_space *mapSpace = isl_space_map_from_set(setSpace);
  isl_multi_aff *ma = isl_multi_aff_identity(mapSpace);
  isl_aff *aff = isl_multi_aff_get_aff(ma, pos);
  aff = isl_aff_add_constant_si(aff, 1);
  ma = isl_multi_aff_set_aff(ma, pos, aff);
  return isl_map_from_multi_aff(ma);
}

/// Fix the first `numFixed` input dimensions of `map` to fresh ISL parameters
/// so injectivity/bijectivity/coalescing checks treat outer dimensions as fixed.
static __isl_give isl_map *parameterizeOuterDims(__isl_take isl_map *map,
                                                 unsigned numFixed,
                                                 llvm::StringRef prefix) {
  if (!map || numFixed == 0)
    return map;
  isl_ctx *ctx = isl_map_get_ctx(map);
  unsigned baseParam = isl_map_dim(map, isl_dim_param);
  map = isl_map_add_dims(map, isl_dim_param, numFixed);
  for (unsigned i = 0; i < numFixed; ++i) {
    std::string name = (prefix + llvm::Twine(i)).str();
    isl_id *id = isl_id_alloc(ctx, name.c_str(), nullptr);
    map = isl_map_set_dim_id(map, isl_dim_param, baseParam + i, id);
    isl_constraint *c =
        isl_constraint_alloc_equality(isl_local_space_from_space(
            isl_map_get_space(map)));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_param, baseParam + i, -1);
    map = isl_map_add_constraint(map, c);
  }
  return map;
}

/// Lexicographic `<` map on `setSpace` that equates the first `numEqual` dims
/// and orders the remaining dimensions strictly `<_lex`.
static __isl_give isl_map *makeLexLtAfterPrefix(__isl_take isl_space *setSpace,
                                                unsigned numEqual) {
  unsigned total = isl_space_dim(setSpace, isl_dim_set);
  if (numEqual >= total) {
    isl_space *ms = isl_space_map_from_set(setSpace);
    return isl_map_empty(ms);
  }
  isl_ctx *ctx = isl_space_get_ctx(setSpace);
  isl_map *result = isl_map_empty(isl_space_map_from_set(isl_space_copy(setSpace)));
  for (unsigned diffPos = numEqual; diffPos < total; ++diffPos) {
    isl_map *term =
        isl_map_universe(isl_space_map_from_set(isl_space_copy(setSpace)));
    for (unsigned k = 0; k < diffPos; ++k)
      term = isl_map_equate(term, isl_dim_in, k, isl_dim_out, k);
    isl_constraint *lt = isl_constraint_alloc_inequality(
        isl_local_space_from_space(isl_map_get_space(term)));
    lt = isl_constraint_set_coefficient_si(lt, isl_dim_out, diffPos, 1);
    lt = isl_constraint_set_coefficient_si(lt, isl_dim_in, diffPos, -1);
    lt = isl_constraint_set_constant_si(lt, -1);
    term = isl_map_add_constraint(term, lt);
    result = isl_map_union(result, term);
  }
  (void)ctx;
  isl_space_free(setSpace);
  return isl_map_coalesce(result);
}

/// Map `s -> s'` on grid block coordinates where `s != s'`.
static __isl_give isl_map *makeNeqMap(__isl_take isl_space *setSpace) {
  isl_map *lt = makeLexLtAfterPrefix(setSpace, 0);
  isl_map *gt = isl_map_reverse(isl_map_copy(lt));
  return isl_map_coalesce(isl_map_union(lt, gt));
}

struct TileDimBound {
  int64_t size = 0;
  int64_t stride = 1;
  isl::aff lb;
  isl::aff shift;
};

struct ArrayTile {
  unsigned depth = 0;
  SmallVector<TileDimBound, 4> bounds;
  int64_t numElements = 0;
  int64_t sizeBytes = 0;
};

/// Adapted from PPCG `detect_strides`, `remove_strides`, and `can_tile`
/// (`gpu_group.c:138-260`).
static std::optional<ArrayTile> computeArrayTile(isl::map access,
                                                 unsigned elemBytes) {
  if (access.is_null() || access.is_empty())
    return std::nullopt;

  isl_map *acc = isl_map_detect_equalities(access.copy());
  acc = isl_map_reset_tuple_id(acc, isl_dim_in);
  acc = isl_map_reset_tuple_id(acc, isl_dim_out);
  int nIdx = isl_map_dim(acc, isl_dim_out);
  int inDim = isl_map_dim(acc, isl_dim_in);
  if (nIdx <= 0 || inDim < 0) {
    isl_map_free(acc);
    return std::nullopt;
  }

  ArrayTile tile;
  tile.depth = static_cast<unsigned>(inDim);
  tile.bounds.resize(nIdx);

  bool hasStrides = false;
  for (int i = 0; i < nIdx; ++i) {
    isl_stride_info *si = isl_map_get_range_stride_info(acc, i);
    if (!si) {
      isl_map_free(acc);
      return std::nullopt;
    }
    isl_val *strideVal = isl_stride_info_get_stride(si);
    isl_aff *shiftAff = isl_aff_neg(isl_stride_info_get_offset(si));
    isl_stride_info_free(si);

    int64_t strideInt = 1;
    if (!extractInt64(strideVal, strideInt) || strideInt < 0 || !shiftAff) {
      isl_aff_free(shiftAff);
      isl_map_free(acc);
      return std::nullopt;
    }
    if (strideInt == 0)
      strideInt = 1;
    if (strideInt > 1)
      hasStrides = true;
    tile.bounds[i].stride = strideInt;
    tile.bounds[i].shift = isl::manage(shiftAff);
  }

  if (hasStrides) {
    isl_space *mapSpace = isl_map_get_space(acc);
    isl_multi_aff *shiftMA = isl_multi_aff_zero(isl_space_copy(mapSpace));
    isl_space *rangeSpace = isl_space_range(mapSpace);
    isl_multi_val *strideMV = isl_multi_val_zero(isl_space_copy(rangeSpace));
    isl_multi_aff *scaleMA =
        isl_multi_aff_identity(isl_space_map_from_set(rangeSpace));
    for (int i = 0; i < nIdx; ++i) {
      shiftMA = isl_multi_aff_set_aff(shiftMA, i, tile.bounds[i].shift.copy());
      strideMV = isl_multi_val_set_val(
          strideMV, i, isl_val_int_from_si(access.ctx().get(), tile.bounds[i].stride));
    }
    scaleMA = isl_multi_aff_scale_multi_val(scaleMA, strideMV);
    acc = isl_map_sum(acc, isl_map_from_multi_aff(shiftMA));
    acc = isl_map_preimage_range_multi_aff(acc, scaleMA);
  }

  // Drop tuple IDs before computing fixed box so offsets live in unnamed space.
  acc = isl_map_reset_tuple_id(acc, isl_dim_in);
  acc = isl_map_reset_tuple_id(acc, isl_dim_out);

  isl_fixed_box *box = isl_map_get_range_simple_fixed_box_hull(acc);
  isl_map_free(acc);
  if (!box)
    return std::nullopt;

  isl_bool valid = isl_fixed_box_is_valid(box);
  if (valid != isl_bool_true) {
    isl_fixed_box_free(box);
    return std::nullopt;
  }

  isl_multi_aff *offsetMA = isl_fixed_box_get_offset(box);
  isl_multi_val *sizeMV = isl_fixed_box_get_size(box);
  isl_fixed_box_free(box);

  int64_t totalElems = 1;
  for (int i = 0; i < nIdx; ++i) {
    int64_t sz = 0;
    if (!extractInt64(isl_multi_val_get_val(sizeMV, i), sz) || sz <= 0) {
      isl_multi_aff_free(offsetMA);
      isl_multi_val_free(sizeMV);
      return std::nullopt;
    }
    isl_aff *lbAff = isl_multi_aff_get_aff(offsetMA, i);
    if (!lbAff || isl_aff_is_nan(lbAff) != isl_bool_false) {
      isl_aff_free(lbAff);
      isl_multi_aff_free(offsetMA);
      isl_multi_val_free(sizeMV);
      return std::nullopt;
    }
    tile.bounds[i].size = sz;
    tile.bounds[i].lb = isl::manage(lbAff);
    if (__builtin_mul_overflow(totalElems, sz, &totalElems)) {
      isl_multi_aff_free(offsetMA);
      isl_multi_val_free(sizeMV);
      return std::nullopt;
    }
  }
  isl_multi_aff_free(offsetMA);
  isl_multi_val_free(sizeMV);

  tile.numElements = totalElems;
  tile.sizeBytes = totalElems * static_cast<int64_t>(elemBytes);
  return tile;
}

static unsigned computeTileDepth(const ArrayTile &tile, unsigned minDepth) {
  for (int j = static_cast<int>(tile.depth) - 1;
       j >= static_cast<int>(minDepth); --j) {
    for (const TileDimBound &b : tile.bounds) {
      if (isl_aff_involves_dims(b.lb.get(), isl_dim_in, j, 1) == isl_bool_true)
        return static_cast<unsigned>(j + 1);
      if (!b.shift.is_null() &&
          isl_aff_involves_dims(b.shift.get(), isl_dim_in, j, 1) ==
              isl_bool_true)
        return static_cast<unsigned>(j + 1);
    }
  }
  return minDepth;
}

static void adjustTileDepth(ArrayTile &tile, unsigned newDepth) {
  if (tile.depth == newDepth)
    return;
  assert(newDepth <= tile.depth);
  unsigned drop = tile.depth - newDepth;
  for (TileDimBound &b : tile.bounds) {
    b.lb = isl::manage(
        isl_aff_drop_dims(b.lb.release(), isl_dim_in, newDepth, drop));
    if (!b.shift.is_null()) {
      b.shift = isl::manage(
          isl_aff_drop_dims(b.shift.release(), isl_dim_in, newDepth, drop));
    }
  }
  tile.depth = newDepth;
}

/// Adapted from PPCG `access_is_coalesced` (`gpu_group.c:342-393`), adjusted so
/// `threadIdx.x` is at `threadDepth + 0` (`blockPar` argument 0 in Enzyme-JAX).
static bool accessIsCoalesced(isl::map fullSchedAccess, unsigned gridDims,
                              unsigned threadDepth) {
  isl_map *localAcc =
      parameterizeOuterDims(fullSchedAccess.copy(), gridDims, "__grid_");
  localAcc = isl_map_reset_tuple_id(localAcc, isl_dim_in);
  localAcc = isl_map_reset_tuple_id(localAcc, isl_dim_out);

  isl_space *rangeSpace = isl_space_range(isl_map_get_space(localAcc));
  int rank = isl_space_dim(rangeSpace, isl_dim_set);
  if (rank <= 0) {
    isl_space_free(rangeSpace);
    isl_map_free(localAcc);
    return true;
  }

  isl_map *nextElem = makeNextMap(rangeSpace, static_cast<unsigned>(rank - 1));
  isl_set *accessed = isl_map_range(isl_map_copy(localAcc));
  isl_map *consecutive = isl_map_copy(nextElem);
  consecutive = isl_map_intersect_domain(consecutive, isl_set_copy(accessed));
  consecutive = isl_map_intersect_range(consecutive, accessed);
  isl_bool empty = isl_map_is_empty(consecutive);
  isl_map_free(consecutive);
  if (empty == isl_bool_true) {
    isl_map_free(nextElem);
    isl_map_free(localAcc);
    return true;
  }

  isl_space *domSpace = isl_space_domain(isl_map_get_space(localAcc));
  // In Enzyme-JAX ConvertParallelToGPU1, blockPar arg 0 is gpu::Dimension::x.
  isl_map *nextThreadX = makeNextMap(domSpace, threadDepth);
  isl_map *stepMap = isl_map_apply_domain(nextThreadX, isl_map_copy(localAcc));
  stepMap = isl_map_apply_range(stepMap, localAcc);

  isl_bool coalesced = isl_map_is_subset(stepMap, nextElem);
  isl_map_free(nextElem);
  isl_map_free(stepMap);
  return coalesced == isl_bool_true;
}

/// Adapted from PPCG `access_is_bijective` (`gpu_group.c:428-448`).
static bool accessIsBijectiveAtDepth(isl::map threadAccess,
                                     unsigned threadDepth) {
  isl_map *acc =
      parameterizeOuterDims(threadAccess.copy(), threadDepth, "__outer_");
  // Project out the fixed outer dims so we test bijectivity of `T -> A`.
  acc = isl_map_project_out(acc, isl_dim_in, 0, threadDepth);
  isl_bool bij = isl_map_is_bijective(acc);
  isl_map_free(acc);
  return bij == isl_bool_true;
}

/// Adapted from PPCG `compute_accessed_by_single_thread_depth`
/// (`gpu_group.c:524-568`).
static int computeAccessedBySingleThreadDepth(isl::map threadAccess,
                                              unsigned kernelDepth,
                                              unsigned threadDepth,
                                              unsigned nThread) {
  if (threadDepth == kernelDepth)
    return static_cast<int>(threadDepth);

  isl_map *acc = isl_map_copy(threadAccess.get());
  acc = isl_map_reset_tuple_id(acc, isl_dim_in);
  acc = isl_map_reset_tuple_id(acc, isl_dim_out);

  isl_space *space = isl_space_params(isl_map_get_space(acc));
  space = isl_space_set_from_params(space);
  space = isl_space_add_dims(space, isl_dim_set, threadDepth);
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out, nThread);
  space = isl_space_wrap(space);
  isl_map *flatten = isl_set_flatten_map(isl_set_universe(space));
  acc = isl_map_apply_range(flatten, acc);

  isl_space *domSpace = isl_space_domain(isl_map_get_space(acc));
  isl_map *rangeMap =
      isl_map_range_map(isl_map_universe(isl_space_unwrap(domSpace)));
  acc = isl_map_range_product(acc, rangeMap);
  acc = isl_map_domain_factor_domain(acc);
  acc = isl_map_uncurry(acc);

  int i = static_cast<int>(threadDepth) - 1;
  for (; i >= static_cast<int>(kernelDepth); --i) {
    acc = isl_map_project_out(acc, isl_dim_in, i, 1);
    isl_bool sv = isl_map_is_single_valued(acc);
    if (sv < 0) {
      isl_map_free(acc);
      return -1;
    }
    if (!sv)
      break;
  }
  isl_map_free(acc);
  return i + 1;
}

struct AccessRef {
  Operation *op = nullptr;
  Value memref;
  bool isWrite = false;
  affine::AffineParallelOp blockPar;
  SmallVector<affine::AffineForOp, 4> outerFors;
  SmallVector<affine::AffineForOp, 4> innerFors;
  AffineMap composedMap;
  SmallVector<Value, 8> composedOperands;
  /// `[g..., outerFors..., t..., innerFors...] -> A[...]`
  isl::map loopAccessMap;
  /// Uniform execution-ordered schedule space `[exec...] -> A[...]`
  isl::map execAccessMap;
};

enum class PromotionKind { None, Shared, Private };

struct RefGroup {
  Value memref;
  MemRefType memrefType;
  unsigned elemBytes = 0;
  SmallVector<unsigned, 8> refIndices;
  bool hasReads = false;
  bool hasWrites = false;

  SmallVector<affine::AffineForOp, 4> commonOuterFors;
  unsigned copyDepth = 0;
  unsigned minDepth = 0;
  bool allSameOuterFors = false;
  bool hasReuse = false;
  bool isCoalesced = true;

  /// Access relation projected to `copyDepth` (`[g..., commonOuterFors...] -> A`).
  isl::map copyAccess;

  std::optional<ArrayTile> sharedTile;
  std::optional<ArrayTile> privateTile;

  PromotionKind chosenKind() const {
    if (privateTile && sharedTile && sharedTile->depth < privateTile->depth)
      return PromotionKind::Shared;
    if (privateTile)
      return PromotionKind::Private;
    if (sharedTile)
      return PromotionKind::Shared;
    return PromotionKind::None;
  }
};

static void computeGroupCommonOuterFors(RefGroup &g,
                                        ArrayRef<AccessRef> allRefs,
                                        unsigned gridDims) {
  assert(!g.refIndices.empty());
  const auto &first = allRefs[g.refIndices.front()].outerFors;
  size_t prefixLen = first.size();
  bool sameLen = true;
  for (unsigned idx : g.refIndices) {
    const auto &cur = allRefs[idx].outerFors;
    if (cur.size() != first.size())
      sameLen = false;
    prefixLen = std::min(prefixLen, cur.size());
    for (size_t k = 0; k < prefixLen; ++k) {
      if (cur[k] != first[k]) {
        prefixLen = k;
        sameLen = false;
        break;
      }
    }
  }
  g.commonOuterFors.assign(first.begin(), first.begin() + prefixLen);
  g.copyDepth = gridDims + static_cast<unsigned>(prefixLen);
  g.allSameOuterFors = sameLen && (prefixLen == first.size());

  isl::map unionMap;
  for (unsigned idx : g.refIndices) {
    const AccessRef &ref = allRefs[idx];
    unsigned refDim = isl_map_dim(ref.loopAccessMap.get(), isl_dim_in);
    assert(refDim >= g.copyDepth);
    isl::map proj = isl::manage(isl_map_project_out(
        ref.loopAccessMap.copy(), isl_dim_in, g.copyDepth, refDim - g.copyDepth));
    proj = isl::manage(isl_map_reset_tuple_id(proj.release(), isl_dim_in));
    if (unionMap.is_null())
      unionMap = proj;
    else
      unionMap = unionMap.unite(proj);
  }
  g.copyAccess = isl::manage(isl_map_coalesce(unionMap.release()));
}

static RefGroup mergeTwoGroups(const RefGroup &a, const RefGroup &b,
                               ArrayRef<AccessRef> allRefs, unsigned gridDims) {
  RefGroup merged;
  merged.memref = a.memref;
  merged.memrefType = a.memrefType;
  merged.elemBytes = a.elemBytes;
  merged.refIndices = a.refIndices;
  merged.refIndices.append(b.refIndices.begin(), b.refIndices.end());
  llvm::sort(merged.refIndices);
  merged.hasReads = a.hasReads || b.hasReads;
  merged.hasWrites = a.hasWrites || b.hasWrites;
  computeGroupCommonOuterFors(merged, allRefs, gridDims);
  return merged;
}

static bool groupsOverlapAtDepth(const RefGroup &a, const RefGroup &b,
                                 unsigned depth) {
  unsigned dimA = isl_map_dim(a.copyAccess.get(), isl_dim_in);
  unsigned dimB = isl_map_dim(b.copyAccess.get(), isl_dim_in);
  unsigned d = std::min({depth, dimA, dimB});
  isl_map *ma =
      isl_map_project_out(a.copyAccess.copy(), isl_dim_in, d, dimA - d);
  isl_map *mb =
      isl_map_project_out(b.copyAccess.copy(), isl_dim_in, d, dimB - d);
  ma = isl_map_reset_tuple_id(ma, isl_dim_in);
  mb = isl_map_reset_tuple_id(mb, isl_dim_in);
  isl_bool disj = isl_map_is_disjoint(ma, mb);
  isl_map_free(ma);
  isl_map_free(mb);
  return disj != isl_bool_true;
}

static void evaluateGroupBounds(RefGroup &g, ArrayRef<AccessRef> allRefs,
                                unsigned gridDims, unsigned nThread,
                                const SharedMemoryPromotionOptions &options) {
  g.sharedTile.reset();
  g.privateTile.reset();
  g.minDepth = g.copyDepth;

  // Build union execution-scheduled access map for reuse and write-safety checks.
  isl::map execAll;
  isl::map gridAll;
  isl::map gridWrites;
  for (unsigned idx : g.refIndices) {
    const AccessRef &ref = allRefs[idx];
    execAll = execAll.is_null() ? ref.execAccessMap
                                : execAll.unite(ref.execAccessMap);
    unsigned refDim = isl_map_dim(ref.loopAccessMap.get(), isl_dim_in);
    isl::map gProj = isl::manage(isl_map_project_out(
        ref.loopAccessMap.copy(), isl_dim_in, gridDims, refDim - gridDims));
    gProj = isl::manage(isl_map_reset_tuple_id(gProj.release(), isl_dim_in));
    gridAll = gridAll.is_null() ? gProj : gridAll.unite(gProj);
    if (ref.isWrite)
      gridWrites = gridWrites.is_null() ? gProj : gridWrites.unite(gProj);
  }

  // Write safety: distinct grid blocks must never write an element accessed
  // (written or read) by another grid block.
  if (g.hasWrites) {
    isl_space *gridSetSpace = isl_space_domain(isl_map_get_space(gridWrites.get()));
    isl_map *neqGrid = makeNeqMap(gridSetSpace);
    isl_map *otherBlockWrites =
        isl_map_apply_domain(isl_map_copy(gridWrites.get()), neqGrid);
    isl_map *conflict =
        isl_map_intersect(otherBlockWrites, isl_map_copy(gridAll.get()));
    isl_bool safe = isl_map_is_empty(conflict);
    isl_map_free(conflict);
    if (safe != isl_bool_true)
      return;
  }

  // Reuse check within a single grid block (fixing gridDims as parameters).
  {
    isl_map *localExec =
        parameterizeOuterDims(execAll.copy(), gridDims, "__g_");
    isl_bool injective = isl_map_is_injective(localExec);
    isl_map_free(localExec);
    g.hasReuse = (injective == isl_bool_false);
  }

  // Coalescing check when there is no reuse and all refs share the same outer
  // loop nest.
  g.isCoalesced = true;
  if (!g.hasReuse && g.allSameOuterFors) {
    unsigned maxInner = 0;
    for (unsigned idx : g.refIndices)
      maxInner = std::max<unsigned>(maxInner, allRefs[idx].innerFors.size());
    isl::map paddedLoopMap;
    for (unsigned idx : g.refIndices) {
      const AccessRef &ref = allRefs[idx];
      unsigned curInner = ref.innerFors.size();
      isl_map *m = ref.loopAccessMap.copy();
      if (curInner < maxInner) {
        unsigned pad = maxInner - curInner;
        unsigned inDim = isl_map_dim(m, isl_dim_in);
        m = isl_map_add_dims(m, isl_dim_in, pad);
        for (unsigned p = 0; p < pad; ++p)
          m = isl_map_fix_si(m, isl_dim_in, inDim + p, 0);
      }
      m = isl_map_reset_tuple_id(m, isl_dim_in);
      isl::map managed = isl::manage(m);
      paddedLoopMap = paddedLoopMap.is_null() ? managed
                                              : paddedLoopMap.unite(managed);
    }
    g.isCoalesced =
        accessIsCoalesced(paddedLoopMap, gridDims, g.copyDepth);
  }

  // Candidate 1: Shared memory tile.
  if (options.enableSharedMem && (g.hasReuse || !g.isCoalesced)) {
    if (auto tile = computeArrayTile(g.copyAccess, g.elemBytes)) {
      if (tile->numElements > 1 &&
          tile->sizeBytes <= options.maxSharedMemBytes) {
        unsigned depth = computeTileDepth(*tile, gridDims);
        adjustTileDepth(*tile, depth);
        g.sharedTile = std::move(*tile);
      }
    }
  }

  // Candidate 2: Thread-private register tile (PPCG `compute_group_bounds_core`).
  if (options.enablePrivateMem && g.hasReuse && g.allSameOuterFors) {
    isl::map threadAcc;
    unsigned threadEnd = g.copyDepth + nThread;
    for (unsigned idx : g.refIndices) {
      const AccessRef &ref = allRefs[idx];
      unsigned refDim = isl_map_dim(ref.loopAccessMap.get(), isl_dim_in);
      assert(refDim >= threadEnd);
      isl_map *proj = isl_map_project_out(ref.loopAccessMap.copy(), isl_dim_in,
                                          threadEnd, refDim - threadEnd);
      proj = isl_map_reset_tuple_id(proj, isl_dim_in);
      isl::map managed = isl::manage(proj);
      threadAcc =
          threadAcc.is_null() ? managed : threadAcc.unite(managed);
    }
    threadAcc = isl::manage(isl_map_coalesce(threadAcc.release()));

    if (accessIsBijectiveAtDepth(threadAcc, g.copyDepth)) {
      int uniqueDepth = computeAccessedBySingleThreadDepth(
          threadAcc, gridDims, g.copyDepth, nThread);
      if (uniqueDepth >= 0) {
        // Parameterize thread dims `t_0 .. t_{nThread-1}` at the end of params,
        // project them out of domain, and check for a 1-element per-thread tile.
        isl_ctx *ctx = threadAcc.ctx().get();
        isl_map *privAcc = threadAcc.copy();
        unsigned baseParams = isl_map_dim(privAcc, isl_dim_param);
        privAcc = isl_map_add_dims(privAcc, isl_dim_param, nThread);
        for (unsigned t = 0; t < nThread; ++t) {
          std::string tName = ("__priv_t" + llvm::Twine(t)).str();
          isl_id *tid = isl_id_alloc(ctx, tName.c_str(), nullptr);
          privAcc =
              isl_map_set_dim_id(privAcc, isl_dim_param, baseParams + t, tid);
          isl_constraint *eq = isl_constraint_alloc_equality(
              isl_local_space_from_space(isl_map_get_space(privAcc)));
          eq = isl_constraint_set_coefficient_si(eq, isl_dim_in,
                                                 g.copyDepth + t, 1);
          eq = isl_constraint_set_coefficient_si(eq, isl_dim_param,
                                                 baseParams + t, -1);
          privAcc = isl_map_add_constraint(privAcc, eq);
        }
        privAcc = isl_map_project_out(privAcc, isl_dim_in, g.copyDepth, nThread);
        if (auto pTile =
                computeArrayTile(isl::manage(privAcc), g.elemBytes)) {
          if (pTile->numElements == 1) {
            unsigned pDepth = std::max<unsigned>(
                computeTileDepth(*pTile, gridDims),
                static_cast<unsigned>(uniqueDepth));
            adjustTileDepth(*pTile, pDepth);
            // Convert trailing `__priv_t*` params back into input dimensions at
            // `pDepth .. pDepth + nThread - 1` so `lb` is over `[outer..., t...]`.
            for (TileDimBound &b : pTile->bounds) {
              isl_aff *lb = b.lb.release();
              lb = isl_aff_move_dims(lb, isl_dim_in, pDepth, isl_dim_param,
                                     baseParams, nThread);
              b.lb = isl::manage(lb);
              if (!b.shift.is_null()) {
                isl_aff *sh = b.shift.release();
                sh = isl_aff_move_dims(sh, isl_dim_in, pDepth, isl_dim_param,
                                       baseParams, nThread);
                b.shift = isl::manage(sh);
              }
            }
            g.privateTile = std::move(*pTile);
          }
        }
      }
    }
  }

  if (g.privateTile)
    g.minDepth = std::min(g.minDepth, g.privateTile->depth);
  if (g.sharedTile)
    g.minDepth = std::min(g.minDepth, g.sharedTile->depth);
}

/// Convert a guarded `isl_set` (over `numDims` set dims and `numSyms` params)
/// simplified against `knownDomain` into zero (if universe) or more disjoint
/// `IntegerSet`s. Returns `false` if the set cannot be represented in MLIR.
static bool convertGuardSetToIntegerSets(isl::set guardSet,
                                         isl::set knownDomain,
                                         MLIRContext *ctx,
                                         bool &isEmptyOut,
                                         SmallVectorImpl<IntegerSet> &setsOut) {
  setsOut.clear();
  isEmptyOut = false;

  isl_set *g = guardSet.copy();
  g = isl_set_gist(g, knownDomain.copy());
  g = isl_set_coalesce(g);
  if (!g)
    return false;

  if (isl_set_is_empty(g) == isl_bool_true) {
    isl_set_free(g);
    isEmptyOut = true;
    return true;
  }
  if (isl_set_plain_is_universe(g) == isl_bool_true) {
    isl_set_free(g);
    return true;
  }

  g = isl_set_compute_divs(g);
  g = isl_set_make_disjoint(g);
  isl_basic_set_list *bList = isl_set_get_basic_set_list(g);
  isl_set_free(g);
  if (!bList)
    return false;

  int nBasic = isl_basic_set_list_n_basic_set(bList);
  if (nBasic < 0) {
    isl_basic_set_list_free(bList);
    return false;
  }

  for (int i = 0; i < nBasic; ++i) {
    isl_basic_set *bset = isl_basic_set_list_get_basic_set(bList, i);
    isl_bool bUniv = isl_basic_set_plain_is_universe(bset);
    if (bUniv == isl_bool_true) {
      isl_basic_set_free(bset);
      setsOut.clear();
      isl_basic_set_list_free(bList);
      return true;
    }
    IntegerSet iset = islBasicSetToIntegerSet(bset, ctx);
    isl_basic_set_free(bset);
    if (!iset) {
      isl_basic_set_list_free(bList);
      return false;
    }
    setsOut.push_back(iset);
  }
  isl_basic_set_list_free(bList);
  return true;
}

/// Find the direct child operation of `scopeBlock` that contains `op`.
static Operation *findAncestorInBlock(Operation *op, Block *scopeBlock) {
  while (op && op->getBlock() != scopeBlock)
    op = op->getParentOp();
  return op;
}

/// Create a `gpu.par.block` `affine.parallel` with constant `(0, ...) to (B...)`.
static affine::AffineParallelOp
createBlockParallel(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockSizes) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<AffineMap, 3> lbMaps, ubMaps;
  SmallVector<int64_t, 3> steps(blockSizes.size(), 1);
  for (int64_t bSize : blockSizes) {
    lbMaps.push_back(AffineMap::getConstantMap(0, ctx));
    ubMaps.push_back(AffineMap::getConstantMap(bSize, ctx));
  }
  auto par = affine::AffineParallelOp::create(
      builder, loc, TypeRange(), ArrayRef<arith::AtomicRMWKind>(), lbMaps,
      ValueRange(), ubMaps, ValueRange(), steps);
  par->setAttr(kGpuBlockParAttr, builder.getUnitAttr());
  return par;
}

/// Promote shared and private memory candidates inside a single `gridPar`.
static LogicalResult
promoteGridParallel(affine::AffineParallelOp gridPar,
                    const SharedMemoryPromotionOptions &options) {
  unsigned gridDims = gridPar.getNumDims();
  if (gridDims == 0 || gridDims > 3)
    return success();

  // Discover all `gpu.par.block` ops and verify consistent constant block sizes
  // and affine-only enclosing loops.
  SmallVector<affine::AffineParallelOp, 8> blockPars;
  SmallVector<int64_t, 3> blockSizes;
  bool validStructure = true;

  gridPar.getBody()->walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::ParallelOp, scf::WhileOp>(op)) {
      validStructure = false;
      return WalkResult::interrupt();
    }
    if (auto bp = getAffineBlockParallel(op)) {
      SmallVector<int64_t, 3> curSizes;
      if (!getConstantZeroBasedBounds(bp, curSizes)) {
        validStructure = false;
        return WalkResult::interrupt();
      }
      if (blockSizes.empty()) {
        blockSizes = curSizes;
      } else if (blockSizes != curSizes) {
        validStructure = false;
        return WalkResult::interrupt();
      }
      blockPars.push_back(bp);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  if (!validStructure || blockPars.empty())
    return success();

  unsigned nThread = blockSizes.size();

  // Identify any outer memrefs that are accessed non-affinely inside `gridPar`
  // so we never partially promote an array with raw/atomic/unknown uses.
  DenseSet<Value> disqualifiedMemRefs;
  gridPar.getBody()->walk([&](Operation *op) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineForOp,
            affine::AffineIfOp, affine::AffineParallelOp, affine::AffineYieldOp,
            affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>(op))
      return;
    for (Value opr : op->getOperands()) {
      if (isa<MemRefType>(opr.getType()))
        disqualifiedMemRefs.insert(opr);
    }
  });

  // Collect all affine load/store references inside `blockPar`s, along with
  // their enclosing loop hierarchy and block positions for exact execution order.
  struct RawRef {
    Operation *op;
    Value memref;
    bool isWrite;
    affine::AffineParallelOp blockPar;
    SmallVector<Operation *, 8> enclosingAffineOps;
    SmallVector<affine::AffineForOp, 4> outerFors;
    SmallVector<affine::AffineForOp, 4> innerFors;
    SmallVector<int64_t, 6> outerPos;
    SmallVector<int64_t, 6> innerPos;
    AffineMap composedMap;
    SmallVector<Value, 8> composedOperands;
  };

  SmallVector<RawRef, 16> rawRefs;
  unsigned maxOuterFors = 0;
  unsigned maxInnerFors = 0;

  // Walk AST recursively to record exact syntactic position chains.
  SmallVector<Operation *, 8> curEnclosing;
  curEnclosing.push_back(gridPar.getOperation());
  SmallVector<affine::AffineForOp, 4> curOuterFors;
  SmallVector<affine::AffineForOp, 4> curInnerFors;
  SmallVector<int64_t, 6> curOuterPos;
  SmallVector<int64_t, 6> curInnerPos;
  affine::AffineParallelOp curBlockPar = nullptr;

  std::function<void(Block &, bool)> walkBlock = [&](Block &blk,
                                                     bool flattenIf) {
    int64_t posCounter = 0;
    for (Operation &op : blk.without_terminator()) {
      int64_t myPos = posCounter++;
      if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // Inside an affine.if we keep advancing sequential positions in the
        // parent loop level while recording `ifOp` in `curEnclosing`.
        if (!curBlockPar)
          curOuterPos.back() = myPos;
        else
          curInnerPos.back() = myPos;
        curEnclosing.push_back(&op);
        walkBlock(*ifOp.getThenBlock(), /*flattenIf=*/true);
        if (ifOp.hasElse())
          walkBlock(*ifOp.getElseBlock(), /*flattenIf=*/true);
        curEnclosing.pop_back();
        // Bump counter so ops after `ifOp` strictly follow ops inside `ifOp`.
        posCounter += 1024;
        continue;
      }
      if (!flattenIf) {
        if (!curBlockPar)
          curOuterPos.back() = myPos;
        else
          curInnerPos.back() = myPos;
      } else {
        if (!curBlockPar)
          curOuterPos.back() += 1;
        else
          curInnerPos.back() += 1;
      }

      if (auto bp = getAffineBlockParallel(&op)) {
        curBlockPar = bp;
        curEnclosing.push_back(&op);
        curInnerPos.push_back(0);
        walkBlock(*bp.getBody(), /*flattenIf=*/false);
        curInnerPos.pop_back();
        curEnclosing.pop_back();
        curBlockPar = nullptr;
        continue;
      }

      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        curEnclosing.push_back(&op);
        if (!curBlockPar) {
          curOuterFors.push_back(forOp);
          maxOuterFors = std::max<unsigned>(maxOuterFors, curOuterFors.size());
          curOuterPos.push_back(0);
          walkBlock(*forOp.getBody(), /*flattenIf=*/false);
          curOuterPos.pop_back();
          curOuterFors.pop_back();
        } else {
          curInnerFors.push_back(forOp);
          maxInnerFors = std::max<unsigned>(maxInnerFors, curInnerFors.size());
          curInnerPos.push_back(0);
          walkBlock(*forOp.getBody(), /*flattenIf=*/false);
          curInnerPos.pop_back();
          curInnerFors.pop_back();
        }
        curEnclosing.pop_back();
        continue;
      }

      if (!curBlockPar) {
        // Memory access outside `blockPar` disqualifies that memref.
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(&op))
          disqualifiedMemRefs.insert(loadOp.getMemRef());
        else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(&op))
          disqualifiedMemRefs.insert(storeOp.getMemRef());
        continue;
      }

      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(&op)) {
        Value memref = loadOp.getMemRef();
        if (!disqualifiedMemRefs.contains(memref) &&
            isEligibleGlobalMemRef(memref, gridPar)) {
          AffineMap map = loadOp.getAffineMap();
          SmallVector<Value, 8> operands(loadOp.getMapOperands().begin(),
                                         loadOp.getMapOperands().end());
          affine::fullyComposeAffineMapAndOperands(&map, &operands);
          affine::canonicalizeMapAndOperands(&map, &operands);
          rawRefs.push_back(RawRef{&op, memref, /*isWrite=*/false, curBlockPar,
                                   curEnclosing, curOuterFors, curInnerFors,
                                   curOuterPos, curInnerPos, map, operands});
        }
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(&op)) {
        Value memref = storeOp.getMemRef();
        if (!disqualifiedMemRefs.contains(memref) &&
            isEligibleGlobalMemRef(memref, gridPar)) {
          AffineMap map = storeOp.getAffineMap();
          SmallVector<Value, 8> operands(storeOp.getMapOperands().begin(),
                                         storeOp.getMapOperands().end());
          affine::fullyComposeAffineMapAndOperands(&map, &operands);
          affine::canonicalizeMapAndOperands(&map, &operands);
          rawRefs.push_back(RawRef{&op, memref, /*isWrite=*/true, curBlockPar,
                                   curEnclosing, curOuterFors, curInnerFors,
                                   curOuterPos, curInnerPos, map, operands});
        }
      }
    }
  };

  curOuterPos.push_back(0);
  walkBlock(*gridPar.getBody(), /*flattenIf=*/false);
  curOuterPos.pop_back();

  // Filter out any memrefs that were disqualified later during the walk.
  llvm::erase_if(rawRefs, [&](const RawRef &r) {
    return disqualifiedMemRefs.contains(r.memref);
  });
  if (rawRefs.empty())
    return success();

  // Build ISL context and gather all symbol parameters across domains & accesses.
  std::unique_ptr<isl_ctx, decltype(&isl_ctx_free)> islCtx(isl_ctx_alloc(),
                                                           &isl_ctx_free);
  isl_options_set_on_error(islCtx.get(), ISL_ON_ERROR_CONTINUE);

  SmallVector<affine::FlatAffineValueConstraints, 16> rawDomains(rawRefs.size());
  llvm::SetVector<Value> paramSet;

  for (size_t i = 0; i < rawRefs.size(); ++i) {
    if (failed(affine::getIndexSet(rawRefs[i].enclosingAffineOps,
                                   &rawDomains[i]))) {
      disqualifiedMemRefs.insert(rawRefs[i].memref);
      continue;
    }
    SmallVector<Value, 8> domSyms;
    rawDomains[i].getValues(rawDomains[i].getNumDimVars(),
                            rawDomains[i].getNumDimAndSymbolVars(), &domSyms);
    for (Value s : domSyms)
      paramSet.insert(s);
    unsigned nDims = rawRefs[i].composedMap.getNumDims();
    for (unsigned k = nDims; k < rawRefs[i].composedOperands.size(); ++k)
      paramSet.insert(rawRefs[i].composedOperands[k]);
    // Also check if any map "dim" operand is actually a symbol outside `gridPar`.
    for (unsigned k = 0; k < nDims; ++k) {
      Value v = rawRefs[i].composedOperands[k];
      unsigned pos = 0;
      if (!rawDomains[i].findVar(v, &pos))
        paramSet.insert(v);
    }
  }

  SmallVector<Value, 8> paramValues(paramSet.begin(), paramSet.end());
  unsigned nParams = paramValues.size();

  auto setupParamSpace = [&](isl_space *sp) -> isl_space * {
    for (unsigned p = 0; p < nParams; ++p) {
      std::string pName = ("P" + llvm::Twine(p)).str();
      isl_id *pid =
          isl_id_alloc(islCtx.get(), pName.c_str(), paramValues[p].getAsOpaquePointer());
      sp = isl_space_set_dim_id(sp, isl_dim_param, p, pid);
    }
    return sp;
  };

  // Uniform execution-order space dimension:
  // [g_0..g_{G-1}, opos_0, ofor_0, opos_1, ..., opos_maxO,
  //  t_0..t_{T-1}, ipos_0, ifor_0, ipos_1, ..., ipos_maxI]
  unsigned execDim =
      gridDims + (2 * maxOuterFors + 1) + nThread + (2 * maxInnerFors + 1);

  DenseMap<Value, isl::id> arrayIds;
  unsigned arrayCounter = 0;
  auto getArrayId = [&](Value memref) -> isl::id {
    auto it = arrayIds.find(memref);
    if (it != arrayIds.end())
      return it->second;
    std::string name = ("Arr" + llvm::Twine(arrayCounter++)).str();
    isl::id id = isl::manage(
        isl_id_alloc(islCtx.get(), name.c_str(), memref.getAsOpaquePointer()));
    arrayIds.insert({memref, id});
    return id;
  };

  SmallVector<AccessRef, 16> refs;
  refs.reserve(rawRefs.size());

  for (size_t i = 0; i < rawRefs.size(); ++i) {
    RawRef &rr = rawRefs[i];
    if (disqualifiedMemRefs.contains(rr.memref))
      continue;

    affine::FlatAffineValueConstraints domain = rawDomains[i];
    alignSymbolVars(domain, paramValues);

    // Build `domainSet` in ISL.
    unsigned domDims = domain.getNumDimVars();
    unsigned expectedDims =
        gridDims + rr.outerFors.size() + nThread + rr.innerFors.size();
    if (domDims != expectedDims) {
      disqualifiedMemRefs.insert(rr.memref);
      continue;
    }

    isl_space *domSpace =
        setupParamSpace(isl_space_set_alloc(islCtx.get(), nParams, domDims));
    isl_mat *domEq = createConstraintRows(islCtx.get(), domain, /*isEq=*/true);
    isl_mat *domIneq =
        createConstraintRows(islCtx.get(), domain, /*isEq=*/false);
    isl_set *domSet = isl_set_from_basic_set(
        isl_basic_set_from_constraint_matrices(domSpace, domEq, domIneq,
                                               isl_dim_set, isl_dim_div,
                                               isl_dim_param, isl_dim_cst));

    // Compose `AffineValueMap` into an access relation over `domain`'s variables.
    affine::AffineValueMap vMap(rr.composedMap, rr.composedOperands);
    affine::FlatAffineValueConstraints accCst;
    accCst.mergeAndAlignVarsWithOther(0, &domain);
    if (failed(accCst.composeMap(&vMap))) {
      isl_set_free(domSet);
      disqualifiedMemRefs.insert(rr.memref);
      continue;
    }
    alignSymbolVars(accCst, paramValues);

    unsigned rank = vMap.getNumResults();
    isl_space *accSpace = setupParamSpace(
        isl_space_alloc(islCtx.get(), nParams, domDims, rank));
    accSpace = isl_space_set_tuple_id(accSpace, isl_dim_out,
                                      getArrayId(rr.memref).release());
    isl_mat *accEq = createConstraintRows(islCtx.get(), accCst, /*isEq=*/true);
    isl_mat *accIneq =
        createConstraintRows(islCtx.get(), accCst, /*isEq=*/false);
    isl_map *accMap =
        isl_map_from_basic_map(isl_basic_map_from_constraint_matrices(
            accSpace, accEq, accIneq, isl_dim_out, isl_dim_in, isl_dim_div,
            isl_dim_param, isl_dim_cst));
    accMap = isl_map_intersect_domain(accMap, domSet);
    accMap = isl_map_coalesce(accMap);

    // Embed `[domDims] -> [execDim]` for exact sequential execution ordering.
    isl_space *embedSpace = setupParamSpace(
        isl_space_alloc(islCtx.get(), nParams, domDims, execDim));
    isl_multi_aff *embedMA = isl_multi_aff_zero(embedSpace);
    isl_local_space *ls =
        isl_local_space_from_space(isl_multi_aff_get_domain_space(embedMA));
    auto setDimAff = [&](unsigned outPos, unsigned inPos) {
      isl_aff *a =
          isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, inPos);
      embedMA = isl_multi_aff_set_aff(embedMA, outPos, a);
    };
    auto setConstAff = [&](unsigned outPos, int64_t val) {
      isl_aff *a = isl_aff_val_on_domain(isl_local_space_copy(ls),
                                         isl_val_int_from_si(islCtx.get(), val));
      embedMA = isl_multi_aff_set_aff(embedMA, outPos, a);
    };

    unsigned inCur = 0;
    unsigned outCur = 0;
    for (unsigned g = 0; g < gridDims; ++g)
      setDimAff(outCur++, inCur++);

    for (unsigned o = 0; o < maxOuterFors; ++o) {
      int64_t posVal = (o < rr.outerPos.size()) ? rr.outerPos[o] : 0;
      setConstAff(outCur++, posVal);
      if (o < rr.outerFors.size())
        setDimAff(outCur++, inCur++);
      else
        setConstAff(outCur++, 0);
    }
    int64_t lastOuterPos =
        (rr.outerFors.size() < rr.outerPos.size())
            ? rr.outerPos[rr.outerFors.size()]
            : rr.outerPos.back();
    setConstAff(outCur++, lastOuterPos);

    for (unsigned t = 0; t < nThread; ++t)
      setDimAff(outCur++, inCur++);

    for (unsigned m = 0; m < maxInnerFors; ++m) {
      int64_t posVal = (m < rr.innerPos.size()) ? rr.innerPos[m] : 0;
      setConstAff(outCur++, posVal);
      if (m < rr.innerFors.size())
        setDimAff(outCur++, inCur++);
      else
        setConstAff(outCur++, 0);
    }
    int64_t lastInnerPos =
        (rr.innerFors.size() < rr.innerPos.size())
            ? rr.innerPos[rr.innerFors.size()]
            : rr.innerPos.back();
    setConstAff(outCur++, lastInnerPos);
    assert(inCur == domDims && outCur == execDim);
    isl_local_space_free(ls);

    isl_map *embedMap = isl_map_from_multi_aff(embedMA);
    isl_map *execMap = isl_map_apply_domain(isl_map_copy(accMap), embedMap);
    execMap = isl_map_coalesce(execMap);

    refs.push_back(AccessRef{rr.op, rr.memref, rr.isWrite, rr.blockPar,
                             rr.outerFors, rr.innerFors, rr.composedMap,
                             rr.composedOperands, isl::manage(accMap),
                             isl::manage(execMap)});
  }

  // Remove any refs whose memref was disqualified during constraint building.
  llvm::erase_if(refs, [&](const AccessRef &r) {
    return disqualifiedMemRefs.contains(r.memref);
  });
  if (refs.empty())
    return success();

  // Partition references by base `memref` in deterministic encounter order.
  llvm::SetVector<Value> candidateMemRefs;
  for (const AccessRef &r : refs)
    candidateMemRefs.insert(r.memref);

  SmallVector<RefGroup, 8> allCandidateGroups;

  for (Value memref : candidateMemRefs) {
    auto mt = cast<MemRefType>(memref.getType());
    unsigned bitWidth = mt.getElementType().getIntOrFloatBitWidth();
    unsigned elemBytes = std::max(1u, (bitWidth + 7) / 8);

    std::vector<RefGroup> groups;
    for (unsigned idx = 0; idx < refs.size(); ++idx) {
      if (refs[idx].memref != memref)
        continue;
      RefGroup g;
      g.memref = memref;
      g.memrefType = mt;
      g.elemBytes = elemBytes;
      g.refIndices.push_back(idx);
      g.hasReads = !refs[idx].isWrite;
      g.hasWrites = refs[idx].isWrite;
      computeGroupCommonOuterFors(g, refs, gridDims);
      groups.push_back(std::move(g));
    }

    // Step 1: `group_overlapping_writes` at `min(copyDepth_i, copyDepth_j)`.
    for (size_t i = 0; i < groups.size();) {
      bool mergedAny = false;
      for (size_t j = groups.size(); j-- > i + 1;) {
        if (!groups[i].hasWrites && !groups[j].hasWrites)
          continue;
        unsigned d = std::min(groups[i].copyDepth, groups[j].copyDepth);
        if (!groupsOverlapAtDepth(groups[i], groups[j], d))
          continue;
        groups[i] = mergeTwoGroups(groups[i], groups[j], refs, gridDims);
        groups.erase(groups.begin() + j);
        mergedAny = true;
      }
      if (!mergedAny)
        ++i;
    }

    // Step 2: Evaluate initial tile bounds & depths.
    for (RefGroup &g : groups)
      evaluateGroupBounds(g, refs, gridDims, nThread, options);

    // Step 3: `group_depth_overlapping_writes` at `min(minDepth_i, minDepth_j)`.
    for (size_t i = 0; i < groups.size();) {
      bool mergedAny = false;
      for (size_t j = groups.size(); j-- > i + 1;) {
        if (!groups[i].hasWrites && !groups[j].hasWrites)
          continue;
        unsigned d = std::min(groups[i].minDepth, groups[j].minDepth);
        if (!groupsOverlapAtDepth(groups[i], groups[j], d))
          continue;
        groups[i] = mergeTwoGroups(groups[i], groups[j], refs, gridDims);
        evaluateGroupBounds(groups[i], refs, gridDims, nThread, options);
        groups.erase(groups.begin() + j);
        mergedAny = true;
      }
      if (!mergedAny)
        ++i;
    }

    // Step 4: `group_common_shared_memory_tile` — merge read-shared groups on
    // the same array when a combined shared tile has size <= sum of individual
    // tiles (e.g. halo stencil references `in[i-1]`, `in[i]`, `in[i+1]`).
    if (options.enableSharedMem && groups.size() > 1) {
      for (size_t i = 0; i < groups.size();) {
        bool mergedAny = false;
        if (groups[i].chosenKind() == PromotionKind::Shared ||
            (!groups[i].hasWrites && !groups[i].sharedTile)) {
          for (size_t j = groups.size(); j-- > i + 1;) {
            if (groups[i].hasWrites || groups[j].hasWrites)
              continue;
            RefGroup trial =
                mergeTwoGroups(groups[i], groups[j], refs, gridDims);
            evaluateGroupBounds(trial, refs, gridDims, nThread, options);
            if (trial.chosenKind() != PromotionKind::Shared)
              continue;
            int64_t prevBytes =
                (groups[i].sharedTile ? groups[i].sharedTile->sizeBytes : 0) +
                (groups[j].sharedTile ? groups[j].sharedTile->sizeBytes : 0);
            if (prevBytes == 0 ||
                trial.sharedTile->sizeBytes <= prevBytes) {
              groups[i] = std::move(trial);
              groups.erase(groups.begin() + j);
              mergedAny = true;
            }
          }
        }
        if (!mergedAny)
          ++i;
      }
    }

    for (RefGroup &g : groups) {
      if (g.chosenKind() != PromotionKind::None)
        allCandidateGroups.push_back(std::move(g));
    }
  }

  if (allCandidateGroups.empty())
    return success();

  // Select all private-register groups (0 shared memory bytes) and greedily
  // select shared-memory groups within `options.maxSharedMemBytes`.
  SmallVector<unsigned, 8> selectedIndices;
  SmallVector<std::pair<double, unsigned>, 8> sharedCandidates;
  for (unsigned idx = 0; idx < allCandidateGroups.size(); ++idx) {
    const RefGroup &g = allCandidateGroups[idx];
    if (g.chosenKind() == PromotionKind::Private) {
      selectedIndices.push_back(idx);
    } else if (g.chosenKind() == PromotionKind::Shared) {
      double score = static_cast<double>(g.refIndices.size()) * 16.0;
      if (g.hasReuse)
        score += 100.0;
      if (!g.isCoalesced)
        score += 50.0;
      // Prefer hoisting to shallower loop depths (e.g. outside inner loops).
      score += 10.0 / static_cast<double>(1 + g.sharedTile->depth);
      score /= static_cast<double>(std::max<int64_t>(1, g.sharedTile->sizeBytes));
      sharedCandidates.push_back({score, idx});
    }
  }

  llvm::stable_sort(sharedCandidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first > rhs.first;
  });

  int64_t usedSharedBytes = 0;
  for (const auto &[score, idx] : sharedCandidates) {
    (void)score;
    int64_t bytes = allCandidateGroups[idx].sharedTile->sizeBytes;
    if (usedSharedBytes + bytes <= options.maxSharedMemBytes) {
      usedSharedBytes += bytes;
      selectedIndices.push_back(idx);
    }
  }

  if (selectedIndices.empty())
    return success();

  // Emit promotions for each selected group.
  MLIRContext *mlirCtx = gridPar.getContext();
  OpBuilder builder(mlirCtx);

  // Build grid-level domain set `[g_0..g_{G-1}, f_0..f_{k-1}]` helper so
  // `gist(tileBoxDomain)` knows the outer loop bounds at `tileDepth`!
  auto buildOuterDomainAtDepth = [&](const RefGroup &g,
                                     unsigned tileDepth) -> isl::set {
    SmallVector<Operation *, 6> outerOps;
    outerOps.push_back(gridPar.getOperation());
    unsigned numOuterFors = tileDepth - gridDims;
    for (unsigned k = 0; k < numOuterFors; ++k) {
      affine::AffineForOp outerFor = g.commonOuterFors[k];
      outerOps.push_back(outerFor.getOperation());
    }
    affine::FlatAffineValueConstraints outerCst;
    if (failed(affine::getIndexSet(outerOps, &outerCst)))
      return {};
    alignSymbolVars(outerCst, paramValues);
    isl_space *sp =
        setupParamSpace(isl_space_set_alloc(islCtx.get(), nParams, tileDepth));
    isl_mat *eq = createConstraintRows(islCtx.get(), outerCst, /*isEq=*/true);
    isl_mat *ineq =
        createConstraintRows(islCtx.get(), outerCst, /*isEq=*/false);
    return isl::manage(isl_set_from_basic_set(
        isl_basic_set_from_constraint_matrices(
            sp, eq, ineq, isl_dim_set, isl_dim_div, isl_dim_param,
            isl_dim_cst)));
  };

  for (unsigned selIdx : selectedIndices) {
    const RefGroup &g = allCandidateGroups[selIdx];
    PromotionKind kind = g.chosenKind();
    const ArrayTile &tile =
        (kind == PromotionKind::Shared) ? *g.sharedTile : *g.privateTile;
    unsigned tileDepth = tile.depth;
    unsigned rank = g.memrefType.getRank();

    // Locate `scopeBlock` and `[firstTopOp, lastTopOp]` at `tileDepth`.
    Block *scopeBlock = nullptr;
    if (tileDepth == gridDims) {
      scopeBlock = gridPar.getBody();
    } else {
      affine::AffineForOp scopeFor =
          g.commonOuterFors[tileDepth - gridDims - 1];
      scopeBlock = scopeFor.getBody();
    }

    SmallVector<Value, 6> outerIVs;
    for (Value iv : gridPar.getIVs())
      outerIVs.push_back(iv);
    for (unsigned k = 0; k < tileDepth - gridDims; ++k) {
      affine::AffineForOp outerFor = g.commonOuterFors[k];
      outerIVs.push_back(outerFor.getInductionVar());
    }
    assert(outerIVs.size() == tileDepth);

    Operation *firstTopOp = nullptr;
    Operation *lastTopOp = nullptr;
    DenseSet<Operation *> groupTopOps;
    for (unsigned refIdx : g.refIndices) {
      Operation *top = findAncestorInBlock(refs[refIdx].op, scopeBlock);
      if (!top)
        continue;
      groupTopOps.insert(top);
    }
    for (Operation &op : scopeBlock->without_terminator()) {
      if (groupTopOps.contains(&op)) {
        if (!firstTopOp)
          firstTopOp = &op;
        lastTopOp = &op;
      }
    }
    if (!firstTopOp || !lastTopOp)
      continue;

    // Determine exact `exposedReads` (for copy-in) and `mustWrites` (for copy-out)
    // within the same outer `tileDepth` iteration.
    unsigned execPrefix =
        gridDims + 2 * (tileDepth - gridDims); // [g..., pos_0, f_0, ..., f_{k-1}]
    isl::map execReads;
    isl::map execWrites;
    for (unsigned refIdx : g.refIndices) {
      const AccessRef &ref = refs[refIdx];
      if (ref.isWrite)
        execWrites = execWrites.is_null() ? ref.execAccessMap
                                          : execWrites.unite(ref.execAccessMap);
      else
        execReads = execReads.is_null() ? ref.execAccessMap
                                        : execReads.unite(ref.execAccessMap);
    }

    bool needsCopyIn = false;
    isl::map exposedReadsExec;
    if (!execReads.is_null()) {
      if (execWrites.is_null()) {
        exposedReadsExec = execReads;
      } else {
        isl_space *execSetSpace =
            isl_space_domain(isl_map_get_space(execWrites.get()));
        isl_map *ltSameTile = makeLexLtAfterPrefix(execSetSpace, execPrefix);
        isl::map covered = isl::manage(
            isl_map_apply_domain(execWrites.copy(), ltSameTile));
        exposedReadsExec = execReads.subtract(covered);
      }
      needsCopyIn = !exposedReadsExec.is_empty();
    }
    bool needsCopyOut = g.hasWrites && !execWrites.is_null();

    // Helper to project an `execDim -> A[...]` map down to `[outerIVs...] -> A[...]`.
    auto projectExecToTileDepth = [&](isl::map execMap) -> isl::map {
      isl_map *m = execMap.copy();
      // Keep `gridDims` dims `0..gridDims-1` and `f_0..f_{L-1}` at `gridDims + 2*k + 1`.
      // Drop from right to left so indices stay valid!
      unsigned curDim = isl_map_dim(m, isl_dim_in);
      unsigned afterOuterFors = gridDims + 2 * (tileDepth - gridDims);
      if (curDim > afterOuterFors)
        m = isl_map_project_out(m, isl_dim_in, afterOuterFors,
                                curDim - afterOuterFors);
      for (int k = static_cast<int>(tileDepth - gridDims) - 1; k >= 0; --k) {
        // Drop `opos_k` at `gridDims + 2 * k`.
        m = isl_map_project_out(m, isl_dim_in, gridDims + 2 * k, 1);
      }
      m = isl_map_reset_tuple_id(m, isl_dim_in);
      m = isl_map_reset_tuple_id(m, isl_dim_out);
      return isl::manage(isl_map_coalesce(m));
    };

    isl::set outerDom = buildOuterDomainAtDepth(g, tileDepth);
    if (outerDom.is_null())
      continue;

    if (kind == PromotionKind::Shared) {
      // Pre-build `tileToGlobal` map: `(d_0..d_{tileDepth-1}, c_0..c_{R-1}) -> A[...]`
      // where `a_m = stride_m * (lb_m(d) + c_m) - shift_m(d)`.
      isl_space *copyMapSpace = setupParamSpace(
          isl_space_alloc(islCtx.get(), nParams, tileDepth + rank, rank));
      isl_multi_aff *tileToGlobalMA =
          isl_multi_aff_zero(isl_space_copy(copyMapSpace));
      isl_local_space *copyLS = isl_local_space_from_space(
          isl_multi_aff_get_domain_space(tileToGlobalMA));

      for (unsigned m = 0; m < rank; ++m) {
        isl_aff *lb = isl_aff_add_dims(tile.bounds[m].lb.copy(), isl_dim_in, rank);
        isl_aff *cm = isl_aff_var_on_domain(isl_local_space_copy(copyLS),
                                            isl_dim_set, tileDepth + m);
        isl_aff *sum = isl_aff_add(lb, cm);
        sum = isl_aff_scale_val(
            sum, isl_val_int_from_si(islCtx.get(), tile.bounds[m].stride));
        if (!tile.bounds[m].shift.is_null()) {
          isl_aff *sh =
              isl_aff_add_dims(tile.bounds[m].shift.copy(), isl_dim_in, rank);
          sum = isl_aff_sub(sum, sh);
        }
        tileToGlobalMA = isl_multi_aff_set_aff(tileToGlobalMA, m, sum);
      }
      isl_local_space_free(copyLS);
      isl_space_free(copyMapSpace);

      AffineMap tileToGlobalAffMap =
          islMultiAffToAffineMap(tileToGlobalMA, mlirCtx);
      if (!tileToGlobalAffMap) {
        isl_multi_aff_free(tileToGlobalMA);
        continue;
      }

      // Build `tileBoxDomain = { [d..., c...] : d in outerDom and 0 <= c_m < S_m }`.
      isl_set *tileBox = isl_set_add_dims(outerDom.copy(), isl_dim_set, rank);
      for (unsigned m = 0; m < rank; ++m) {
        isl_constraint *ge = isl_constraint_alloc_inequality(
            isl_local_space_from_space(isl_set_get_space(tileBox)));
        ge = isl_constraint_set_coefficient_si(ge, isl_dim_set, tileDepth + m, 1);
        tileBox = isl_set_add_constraint(tileBox, ge);

        isl_constraint *lt = isl_constraint_alloc_inequality(
            isl_local_space_from_space(isl_set_get_space(tileBox)));
        lt = isl_constraint_set_coefficient_si(lt, isl_dim_set, tileDepth + m,
                                               -1);
        lt = isl_constraint_set_constant_si(lt, tile.bounds[m].size - 1);
        tileBox = isl_set_add_constraint(tileBox, lt);
      }
      isl::set tileBoxSet = isl::manage(tileBox);

      // Compute copy-in and copy-out `IntegerSet` guards BEFORE mutating IR.
      // Build `[d..., c...] -> [d..., a...]` so we can take exact preimages!
      isl_space *idOuterSpace = setupParamSpace(
          isl_space_alloc(islCtx.get(), nParams, tileDepth + rank, tileDepth));
      isl_multi_aff *outerProjMA = isl_multi_aff_zero(idOuterSpace);
      isl_local_space *projLS = isl_local_space_from_space(
          isl_multi_aff_get_domain_space(outerProjMA));
      for (unsigned d = 0; d < tileDepth; ++d) {
        outerProjMA = isl_multi_aff_set_aff(
            outerProjMA, d,
            isl_aff_var_on_domain(isl_local_space_copy(projLS), isl_dim_set, d));
      }
      isl_local_space_free(projLS);
      isl_multi_aff *combinedMA =
          isl_multi_aff_range_product(outerProjMA, tileToGlobalMA);
      // `combinedMA` maps `[d..., c...] -> [d... -> a...]`.

      auto computeCopyGuards = [&](isl::map flatMap, bool &isEmpty,
                                   SmallVectorImpl<IntegerSet> &intSets) -> bool {
        isl_set *wrapped = isl_map_wrap(flatMap.copy());
        isl_set *preimg =
            isl_set_preimage_multi_aff(wrapped, isl_multi_aff_copy(combinedMA));
        preimg = isl_set_intersect(preimg, tileBoxSet.copy());
        return convertGuardSetToIntegerSets(isl::manage(preimg), tileBoxSet,
                                            mlirCtx, isEmpty, intSets);
      };

      bool copyInEmpty = false;
      SmallVector<IntegerSet, 2> copyInGuards;
      if (needsCopyIn) {
        isl::map exposedFlat = projectExecToTileDepth(exposedReadsExec);
        if (!computeCopyGuards(exposedFlat, copyInEmpty, copyInGuards)) {
          isl_multi_aff_free(combinedMA);
          continue;
        }
      }

      bool copyOutEmpty = false;
      SmallVector<IntegerSet, 2> copyOutGuards;
      if (needsCopyOut) {
        isl::map writesFlat = projectExecToTileDepth(execWrites);
        if (!computeCopyGuards(writesFlat, copyOutEmpty, copyOutGuards)) {
          isl_multi_aff_free(combinedMA);
          continue;
        }
      }
      isl_multi_aff_free(combinedMA);

      // Also pre-build `lb` and `shift` `AffineExpr`s for rewriting original
      // accesses: `t_m = (a_m + shift_m(d)) floordiv stride_m - lb_m(d)`.
      SmallVector<AffineExpr, 4> lbExprs(rank), shiftExprs(rank);
      bool exprsOk = true;
      for (unsigned m = 0; m < rank; ++m) {
        lbExprs[m] = islAffToAffineExpr(tile.bounds[m].lb.get(), mlirCtx);
        if (!lbExprs[m]) {
          exprsOk = false;
          break;
        }
        if (!tile.bounds[m].shift.is_null()) {
          shiftExprs[m] =
              islAffToAffineExpr(tile.bounds[m].shift.get(), mlirCtx);
          if (!shiftExprs[m]) {
            exprsOk = false;
            break;
          }
        } else {
          shiftExprs[m] = getAffineConstantExpr(0, mlirCtx);
        }
      }
      if (!exprsOk)
        continue;

      // All analysis and conversions succeeded — emit shared `memref.alloca` at
      // the very top of `gridPar.getBody()`.
      SmallVector<int64_t, 4> shShape;
      for (unsigned m = 0; m < rank; ++m)
        shShape.push_back(tile.bounds[m].size);
      auto shMemRefType =
          MemRefType::get(shShape, g.memrefType.getElementType());

      builder.setInsertionPointToStart(gridPar.getBody());
      IntegerAttr alignAttr =
          (tile.sizeBytes >= 16) ? builder.getI64IntegerAttr(16) : IntegerAttr();
      auto shAlloca = memref::AllocaOp::create(
          builder, gridPar.getLoc(), shMemRefType, ValueRange(), ValueRange(),
          alignAttr);
      Value shMemRef = shAlloca.getMemref();

      auto emitSharedCopyStage = [&](bool isCopyIn,
                                     ArrayRef<IntegerSet> guards) {
        Location loc = firstTopOp->getLoc();
        auto copyPar = createBlockParallel(builder, loc, blockSizes);
        builder.setInsertionPointToStart(copyPar.getBody());

        // Guard unmapped thread dimensions (`k >= rank`) with `iv[k] == 0`.
        if (nThread > rank) {
          SmallVector<AffineExpr, 2> unmappedExprs;
          SmallVector<bool, 2> unmappedEqs;
          SmallVector<Value, 2> unmappedOps;
          for (unsigned k = rank; k < nThread; ++k) {
            unmappedExprs.push_back(
                getAffineDimExpr(unmappedOps.size(), mlirCtx));
            unmappedEqs.push_back(true);
            unmappedOps.push_back(copyPar.getIVs()[k]);
          }
          auto unmappedSet = IntegerSet::get(unmappedOps.size(), 0,
                                             unmappedExprs, unmappedEqs);
          auto unmappedIf = affine::AffineIfOp::create(
              builder, loc, TypeRange(), unmappedSet, unmappedOps,
              /*withElseRegion=*/false);
          builder.setInsertionPointToStart(unmappedIf.getThenBlock());
        }

        // Map tile dims `d = 0 .. rank-1` so `d = rank - 1` uses `copyPar` arg 0
        // (`threadIdx.x`) for coalesced global accesses and bank-conflict-free
        // shared memory accesses.
        SmallVector<Value, 4> tileCoords(rank);
        SmallVector<AffineExpr, 4> subBlockGuards;
        SmallVector<Value, 4> subBlockOps;

        for (unsigned d = 0; d < rank; ++d) {
          int64_t sDim = tile.bounds[d].size;
          unsigned revIdx = rank - 1 - d;
          if (revIdx < nThread) {
            Value tIV = copyPar.getIVs()[revIdx];
            int64_t bDim = blockSizes[revIdx];
            if (sDim == bDim) {
              tileCoords[d] = tIV;
            } else if (sDim < bDim) {
              tileCoords[d] = tIV;
              subBlockGuards.push_back(
                  getAffineConstantExpr(sDim - 1, mlirCtx) -
                  getAffineDimExpr(subBlockOps.size(), mlirCtx));
              subBlockOps.push_back(tIV);
            } else {
              // `sDim > bDim`: strided cooperative loop `for c = tIV to sDim step bDim`.
              AffineMap lbMap = AffineMap::getMultiDimIdentityMap(1, mlirCtx);
              AffineMap ubMap = AffineMap::getConstantMap(sDim, mlirCtx);
              auto strideFor = affine::AffineForOp::create(
                  builder, loc, ValueRange{tIV}, lbMap, ValueRange{}, ubMap,
                  bDim);
              tileCoords[d] = strideFor.getInductionVar();
              builder.setInsertionPointToStart(strideFor.getBody());
            }
          } else {
            // More tile dims than thread dims: serial loop `0 .. sDim`.
            AffineMap lbMap = AffineMap::getConstantMap(0, mlirCtx);
            AffineMap ubMap = AffineMap::getConstantMap(sDim, mlirCtx);
            auto serFor = affine::AffineForOp::create(
                builder, loc, ValueRange{}, lbMap, ValueRange{}, ubMap, 1);
            tileCoords[d] = serFor.getInductionVar();
            builder.setInsertionPointToStart(serFor.getBody());
          }
        }

        if (!subBlockGuards.empty()) {
          SmallVector<bool, 4> eqFlags(subBlockGuards.size(), false);
          auto sbSet = IntegerSet::get(subBlockOps.size(), 0, subBlockGuards,
                                       eqFlags);
          auto sbIf = affine::AffineIfOp::create(builder, loc, TypeRange(),
                                                 sbSet, subBlockOps,
                                                 /*withElseRegion=*/false);
          builder.setInsertionPointToStart(sbIf.getThenBlock());
        }

        SmallVector<Value, 12> fullMapOperands;
        fullMapOperands.append(outerIVs.begin(), outerIVs.end());
        fullMapOperands.append(tileCoords.begin(), tileCoords.end());
        fullMapOperands.append(paramValues.begin(), paramValues.end());

        AffineMap shIdentityMap =
            AffineMap::getMultiDimIdentityMap(rank, mlirCtx);

        auto emitLeafTransfer = [&](OpBuilder &b) {
          AffineMap globMap = tileToGlobalAffMap;
          SmallVector<Value, 12> globOps = fullMapOperands;
          affine::canonicalizeMapAndOperands(&globMap, &globOps);
          globMap = simplifyAffineMap(globMap);
          affine::canonicalizeMapAndOperands(&globMap, &globOps);
          if (isCopyIn) {
            auto val = affine::AffineLoadOp::create(b, loc, g.memref, globMap,
                                                    globOps);
            affine::AffineStoreOp::create(b, loc, val, shMemRef, shIdentityMap,
                                          tileCoords);
          } else {
            auto val = affine::AffineLoadOp::create(b, loc, shMemRef,
                                                    shIdentityMap, tileCoords);
            affine::AffineStoreOp::create(b, loc, val, g.memref, globMap,
                                          globOps);
          }
        };

        if (guards.empty()) {
          emitLeafTransfer(builder);
        } else {
          for (IntegerSet gSet : guards) {
            OpBuilder::InsertionGuard ig(builder);
            auto guardIf = affine::AffineIfOp::create(
                builder, loc, TypeRange(), gSet, fullMapOperands,
                /*withElseRegion=*/false);
            builder.setInsertionPointToStart(guardIf.getThenBlock());
            emitLeafTransfer(builder);
          }
        }
      };

      if (needsCopyIn && !copyInEmpty) {
        builder.setInsertionPoint(firstTopOp);
        emitSharedCopyStage(/*isCopyIn=*/true, copyInGuards);
      }
      if (needsCopyOut && !copyOutEmpty) {
        builder.setInsertionPointAfter(lastTopOp);
        emitSharedCopyStage(/*isCopyIn=*/false, copyOutGuards);
      }

      // Rewrite all group references to index `shMemRef`.
      for (unsigned refIdx : g.refIndices) {
        const AccessRef &ref = refs[refIdx];
        AffineMap origMap = ref.composedMap;
        unsigned origNumDims = origMap.getNumDims();

        // Build combined operand list:
        // dims = `[origDimOperands..., outerIVs...]`
        // syms = `[origSymOperands..., paramValues...]`
        SmallVector<Value, 12> newDims(
            ref.composedOperands.begin(),
            ref.composedOperands.begin() + origNumDims);
        unsigned outerDimOffset = newDims.size();
        newDims.append(outerIVs.begin(), outerIVs.end());

        SmallVector<Value, 12> newSyms(
            ref.composedOperands.begin() + origNumDims,
            ref.composedOperands.end());
        unsigned paramSymOffset = newSyms.size();
        newSyms.append(paramValues.begin(), paramValues.end());

        // Remap `lbExprs[m]` and `shiftExprs[m]` (which have dims `0..tileDepth-1`
        // and syms `0..nParams-1`) into the combined `(newDims, newSyms)` space.
        SmallVector<AffineExpr, 6> dimReplacements, symReplacements;
        for (unsigned d = 0; d < tileDepth; ++d)
          dimReplacements.push_back(
              getAffineDimExpr(outerDimOffset + d, mlirCtx));
        for (unsigned s = 0; s < nParams; ++s)
          symReplacements.push_back(
              getAffineSymbolExpr(paramSymOffset + s, mlirCtx));

        SmallVector<AffineExpr, 4> shResults;
        for (unsigned m = 0; m < rank; ++m) {
          AffineExpr origExpr = origMap.getResult(m);
          AffineExpr lbRemapped = lbExprs[m].replaceDimsAndSymbols(
              dimReplacements, symReplacements);
          AffineExpr shiftRemapped = shiftExprs[m].replaceDimsAndSymbols(
              dimReplacements, symReplacements);
          AffineExpr num = origExpr + shiftRemapped;
          if (tile.bounds[m].stride > 1)
            num = num.floorDiv(tile.bounds[m].stride);
          shResults.push_back(num - lbRemapped);
        }

        AffineMap shMap = AffineMap::get(newDims.size(), newSyms.size(),
                                         shResults, mlirCtx);
        SmallVector<Value, 16> shOperands(newDims.begin(), newDims.end());
        shOperands.append(newSyms.begin(), newSyms.end());
        affine::fullyComposeAffineMapAndOperands(&shMap, &shOperands);
        affine::canonicalizeMapAndOperands(&shMap, &shOperands);
        shMap = simplifyAffineMap(shMap);
        affine::canonicalizeMapAndOperands(&shMap, &shOperands);

        builder.setInsertionPoint(ref.op);
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(ref.op)) {
          auto newLoad = affine::AffineLoadOp::create(
              builder, loadOp.getLoc(), shMemRef, shMap, shOperands);
          loadOp.replaceAllUsesWith(newLoad.getResult());
          loadOp.erase();
        } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(ref.op)) {
          affine::AffineStoreOp::create(builder, storeOp.getLoc(),
                                        storeOp.getValueToStore(), shMemRef,
                                        shMap, shOperands);
          storeOp.erase();
        }
      }
    } else if (kind == PromotionKind::Private) {
      // Private register promotion:
      // `tile.bounds[m].lb` has `tileDepth + nThread` input dims `[outerIVs..., t...]`,
      // `size == 1`, so global index for thread `t` is `stride_m * lb_m - shift_m`.
      isl_space *privSpace = setupParamSpace(
          isl_space_alloc(islCtx.get(), nParams, tileDepth + nThread, rank));
      isl_multi_aff *privToGlobalMA =
          isl_multi_aff_zero(isl_space_copy(privSpace));
      for (unsigned m = 0; m < rank; ++m) {
        isl_aff *a = tile.bounds[m].lb.copy();
        a = isl_aff_scale_val(
            a, isl_val_int_from_si(islCtx.get(), tile.bounds[m].stride));
        if (!tile.bounds[m].shift.is_null())
          a = isl_aff_sub(a, tile.bounds[m].shift.copy());
        privToGlobalMA = isl_multi_aff_set_aff(privToGlobalMA, m, a);
      }
      isl_space_free(privSpace);

      AffineMap privToGlobalMap =
          islMultiAffToAffineMap(privToGlobalMA, mlirCtx);
      if (!privToGlobalMap) {
        isl_multi_aff_free(privToGlobalMA);
        continue;
      }

      // Build `threadBoxDomain = { [d..., t...] : d in outerDom and 0 <= t_k < B_k }`.
      isl_set *threadBox =
          isl_set_add_dims(outerDom.copy(), isl_dim_set, nThread);
      for (unsigned k = 0; k < nThread; ++k) {
        isl_constraint *ge = isl_constraint_alloc_inequality(
            isl_local_space_from_space(isl_set_get_space(threadBox)));
        ge = isl_constraint_set_coefficient_si(ge, isl_dim_set, tileDepth + k, 1);
        threadBox = isl_set_add_constraint(threadBox, ge);

        isl_constraint *lt = isl_constraint_alloc_inequality(
            isl_local_space_from_space(isl_set_get_space(threadBox)));
        lt = isl_constraint_set_coefficient_si(lt, isl_dim_set, tileDepth + k,
                                               -1);
        lt = isl_constraint_set_constant_si(lt, blockSizes[k] - 1);
        threadBox = isl_set_add_constraint(threadBox, lt);
      }
      isl::set threadBoxSet = isl::manage(threadBox);

      isl_space *idOuterSpace = setupParamSpace(
          isl_space_alloc(islCtx.get(), nParams, tileDepth + nThread, tileDepth));
      isl_multi_aff *outerProjMA = isl_multi_aff_zero(idOuterSpace);
      isl_local_space *projLS = isl_local_space_from_space(
          isl_multi_aff_get_domain_space(outerProjMA));
      for (unsigned d = 0; d < tileDepth; ++d) {
        outerProjMA = isl_multi_aff_set_aff(
            outerProjMA, d,
            isl_aff_var_on_domain(isl_local_space_copy(projLS), isl_dim_set, d));
      }
      isl_local_space_free(projLS);
      isl_multi_aff *combinedMA =
          isl_multi_aff_range_product(outerProjMA, privToGlobalMA);

      auto computePrivGuards = [&](isl::map flatMap, bool &isEmpty,
                                   SmallVectorImpl<IntegerSet> &intSets) -> bool {
        isl_set *wrapped = isl_map_wrap(flatMap.copy());
        isl_set *preimg =
            isl_set_preimage_multi_aff(wrapped, isl_multi_aff_copy(combinedMA));
        preimg = isl_set_intersect(preimg, threadBoxSet.copy());
        return convertGuardSetToIntegerSets(isl::manage(preimg), threadBoxSet,
                                            mlirCtx, isEmpty, intSets);
      };

      bool copyInEmpty = false;
      SmallVector<IntegerSet, 2> copyInGuards;
      if (needsCopyIn) {
        isl::map exposedFlat = projectExecToTileDepth(exposedReadsExec);
        if (!computePrivGuards(exposedFlat, copyInEmpty, copyInGuards)) {
          isl_multi_aff_free(combinedMA);
          continue;
        }
      }

      bool copyOutEmpty = false;
      SmallVector<IntegerSet, 2> copyOutGuards;
      if (needsCopyOut) {
        isl::map writesFlat = projectExecToTileDepth(execWrites);
        if (!computePrivGuards(writesFlat, copyOutEmpty, copyOutGuards)) {
          isl_multi_aff_free(combinedMA);
          continue;
        }
      }
      isl_multi_aff_free(combinedMA);

      // Allocate thread-indexed `{polymer.private}` memref at `gridPar` top;
      // `reducePrivateAllocas` shrinks it to scalar `memref<elemTy>` inside
      // `blockPar` after `LoopUndistribute` fuses the `blockPar` loops.
      auto privType =
          MemRefType::get(blockSizes, g.memrefType.getElementType());
      builder.setInsertionPointToStart(gridPar.getBody());
      auto privAlloca = memref::AllocaOp::create(builder, gridPar.getLoc(),
                                                 privType);
      privAlloca->setAttr(kPrivateAttr, builder.getUnitAttr());
      Value privMemRef = privAlloca.getMemref();

      AffineMap privIdentityMap =
          AffineMap::getMultiDimIdentityMap(nThread, mlirCtx);

      auto emitPrivCopyStage = [&](bool isCopyIn, ArrayRef<IntegerSet> guards) {
        Location loc = firstTopOp->getLoc();
        auto copyPar = createBlockParallel(builder, loc, blockSizes);
        builder.setInsertionPointToStart(copyPar.getBody());

        SmallVector<Value, 3> threadIVs(copyPar.getIVs().begin(),
                                        copyPar.getIVs().end());
        SmallVector<Value, 12> fullOps;
        fullOps.append(outerIVs.begin(), outerIVs.end());
        fullOps.append(threadIVs.begin(), threadIVs.end());
        fullOps.append(paramValues.begin(), paramValues.end());

        auto emitTransfer = [&](OpBuilder &b) {
          AffineMap globMap = privToGlobalMap;
          SmallVector<Value, 12> globOps = fullOps;
          affine::canonicalizeMapAndOperands(&globMap, &globOps);
          globMap = simplifyAffineMap(globMap);
          affine::canonicalizeMapAndOperands(&globMap, &globOps);
          if (isCopyIn) {
            auto val =
                affine::AffineLoadOp::create(b, loc, g.memref, globMap, globOps);
            affine::AffineStoreOp::create(b, loc, val, privMemRef,
                                          privIdentityMap, threadIVs);
          } else {
            auto val = affine::AffineLoadOp::create(
                b, loc, privMemRef, privIdentityMap, threadIVs);
            affine::AffineStoreOp::create(b, loc, val, g.memref, globMap,
                                          globOps);
          }
        };

        if (guards.empty()) {
          emitTransfer(builder);
        } else {
          for (IntegerSet gSet : guards) {
            OpBuilder::InsertionGuard ig(builder);
            auto guardIf = affine::AffineIfOp::create(
                builder, loc, TypeRange(), gSet, fullOps,
                /*withElseRegion=*/false);
            builder.setInsertionPointToStart(guardIf.getThenBlock());
            emitTransfer(builder);
          }
        }
      };

      if (needsCopyIn && !copyInEmpty) {
        builder.setInsertionPoint(firstTopOp);
        emitPrivCopyStage(/*isCopyIn=*/true, copyInGuards);
      }
      if (needsCopyOut && !copyOutEmpty) {
        builder.setInsertionPointAfter(lastTopOp);
        emitPrivCopyStage(/*isCopyIn=*/false, copyOutGuards);
      }

      // Rewrite all group references to `privMemRef[tx, ty, ...]`.
      for (unsigned refIdx : g.refIndices) {
        AccessRef &ref = refs[refIdx];
        SmallVector<Value, 3> tIVs(ref.blockPar.getIVs().begin(),
                                   ref.blockPar.getIVs().end());
        builder.setInsertionPoint(ref.op);
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(ref.op)) {
          auto newLoad = affine::AffineLoadOp::create(
              builder, loadOp.getLoc(), privMemRef, privIdentityMap, tIVs);
          loadOp.replaceAllUsesWith(newLoad.getResult());
          loadOp.erase();
        } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(ref.op)) {
          affine::AffineStoreOp::create(builder, storeOp.getLoc(),
                                        storeOp.getValueToStore(), privMemRef,
                                        privIdentityMap, tIVs);
          storeOp.erase();
        }
      }
    }
  }

  return success();
}

} // namespace

LogicalResult
promoteSharedMemory(Operation *root,
                    const SharedMemoryPromotionOptions &options) {
  SmallVector<affine::AffineParallelOp, 4> gridPars;
  root->walk([&](Operation *op) {
    if (auto gp = getAffineGridParallel(op)) {
      gridPars.push_back(gp);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  for (affine::AffineParallelOp gp : gridPars) {
    if (failed(promoteGridParallel(gp, options)))
      return failure();
  }
  return success();
}

LogicalResult reducePrivateAllocas(Operation *root) {
  SmallVector<memref::AllocaOp, 8> privAllocas;
  root->walk([&](memref::AllocaOp allocaOp) {
    if (allocaOp->hasAttr(kPrivateAttr))
      privAllocas.push_back(allocaOp);
  });

  OpBuilder builder(root->getContext());
  AffineMap emptyMap = AffineMap::get(root->getContext());

  for (memref::AllocaOp allocaOp : privAllocas) {
    allocaOp->removeAttr(kPrivateAttr);
    affine::AffineParallelOp singleBlockPar = nullptr;
    bool canReduce = true;
    SmallVector<affine::AffineLoadOp, 8> loads;
    SmallVector<affine::AffineStoreOp, 8> stores;

    for (OpOperand &use : allocaOp.getMemref().getUses()) {
      Operation *user = use.getOwner();
      affine::AffineParallelOp enclosingBlockPar = nullptr;
      for (Operation *p = user->getParentOp(); p; p = p->getParentOp()) {
        if (auto bp = getAffineBlockParallel(p)) {
          enclosingBlockPar = bp;
          break;
        }
      }
      if (!enclosingBlockPar) {
        canReduce = false;
        break;
      }
      if (!singleBlockPar)
        singleBlockPar = enclosingBlockPar;
      else if (singleBlockPar != enclosingBlockPar) {
        canReduce = false;
        break;
      }

      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(user)) {
        if (loadOp.getMemRef() != allocaOp.getMemref() ||
            !loadOp.getAffineMap().isIdentity() ||
            ValueRange(loadOp.getMapOperands()) !=
                ValueRange(singleBlockPar.getIVs())) {
          canReduce = false;
          break;
        }
        loads.push_back(loadOp);
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
        if (storeOp.getMemRef() != allocaOp.getMemref() ||
            !storeOp.getAffineMap().isIdentity() ||
            ValueRange(storeOp.getMapOperands()) !=
                ValueRange(singleBlockPar.getIVs())) {
          canReduce = false;
          break;
        }
        stores.push_back(storeOp);
      } else {
        canReduce = false;
        break;
      }
    }

    if (!canReduce || !singleBlockPar)
      continue;

    auto scalarType =
        MemRefType::get({}, allocaOp.getType().getElementType());
    builder.setInsertionPointToStart(singleBlockPar.getBody());
    auto scalarAlloca =
        memref::AllocaOp::create(builder, allocaOp.getLoc(), scalarType);
    Value scalarMemRef = scalarAlloca.getMemref();

    for (affine::AffineLoadOp loadOp : loads) {
      builder.setInsertionPoint(loadOp);
      auto newLoad = affine::AffineLoadOp::create(
          builder, loadOp.getLoc(), scalarMemRef, emptyMap, ValueRange{});
      loadOp.replaceAllUsesWith(newLoad.getResult());
      loadOp.erase();
    }
    for (affine::AffineStoreOp storeOp : stores) {
      builder.setInsertionPoint(storeOp);
      affine::AffineStoreOp::create(builder, storeOp.getLoc(),
                                    storeOp.getValueToStore(), scalarMemRef,
                                    emptyMap, ValueRange{});
      storeOp.erase();
    }
    allocaOp.erase();
  }

  return success();
}

} // namespace polymer
