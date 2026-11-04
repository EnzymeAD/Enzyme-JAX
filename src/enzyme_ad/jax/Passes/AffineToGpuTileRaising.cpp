//===- AffineToGpuTileRaising.cpp - Raise affine kernels to tile dialects -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements --raise-affine-to-triton, --raise-affine-to-cutile, and
// --raise-affine-to-mosaic, raising affine parallel kernel representations
// (func.func + jit_call, standalone memref func.func, enzymexla.jit_region, and
// enzymexla.gpu_wrapper) into Triton (tt + enzymexla_tt_ext), CUDA Tile
// (cuda_tile + enzymexla_ct_ext), and Mosaic TPU (tpu + enzymexla_mosaic_ext)
// modules and calls.
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/AffineUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "src/enzyme_ad/jax/Dialect/CudaTileExt/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/MosaicExt/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#ifndef ENZYME_JAX_ENABLE_TRITON
#define ENZYME_JAX_ENABLE_TRITON 0
#endif

#if ENZYME_JAX_ENABLE_TRITON
#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#endif

#define DEBUG_TYPE "raise-affine-to-gpu-tile"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AFFINETOTRITONRAISING
#define GEN_PASS_DEF_AFFINETOCUTILERAISING
#define GEN_PASS_DEF_AFFINETOMOSAICRAISING
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

enum class TileBackend { Triton, CuTile, Mosaic };

static StringRef getBackendDisplayName(TileBackend backend) {
  switch (backend) {
  case TileBackend::Triton:
    return "Triton";
  case TileBackend::CuTile:
    return "cuTile";
  case TileBackend::Mosaic:
    return "Mosaic";
  }
  llvm_unreachable("unhandled TileBackend");
}

struct GpuTileRaisingOptions {
  TileBackend backend = TileBackend::Triton;
  bool err_if_not_fully_raised = true;
  bool strip_llvm_debuginfo = false;
  int64_t default_tile_size = 128;
  int64_t max_tile_elements = 1024;
  int64_t default_grid_size = 128;
  int32_t num_warps = 4;
  int32_t num_stages = 3;
};

static Location sanitizeLocation(Location loc, bool stripDebugInfo) {
  if (auto csl = dyn_cast<CallSiteLoc>(loc)) {
    auto callee = sanitizeLocation(csl.getCallee(), stripDebugInfo);
    auto caller = sanitizeLocation(csl.getCaller(), stripDebugInfo);
    if (isa<UnknownLoc>(callee))
      return caller;
    return CallSiteLoc::get(callee, caller);
  }
  if (auto fl = dyn_cast<FusedLoc>(loc)) {
    if (isa_and_nonnull<LLVM::DISubprogramAttr, LLVM::DILocalScopeAttr>(
            fl.getMetadata())) {
      if (stripDebugInfo)
        return UnknownLoc::get(fl.getContext());
      if (auto sp = dyn_cast<LLVM::DISubprogramAttr>(fl.getMetadata())) {
        auto fileLoc = FileLineColLoc::get(
            sp.getContext(), sp.getFile().getName(), sp.getLine(), 0);
        return Location(NameLoc::get(sp.getName(), fileLoc));
      }
      return UnknownLoc::get(fl.getContext());
    }
    SmallVector<Location> locs;
    for (Location sub : fl.getLocations())
      locs.push_back(sanitizeLocation(sub, stripDebugInfo));
    return FusedLoc::get(fl.getContext(), locs);
  }
  return loc;
}

static bool isSupportedScalarElemType(Type ty, TileBackend backend) {
  if (isa<IndexType>(ty))
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned w = it.getWidth();
    return w == 1 || w == 8 || w == 16 || w == 32 || w == 64;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    if (ft.isF16() || ft.isBF16() || ft.isF32() || ft.isF64())
      return true;
    if (backend == TileBackend::CuTile &&
        (isa<Float8E4M3FNType, Float8E5M2Type>(ft)))
      return true;
  }
  return false;
}

static Type normalizeScalarType(Type ty) {
  if (isa<IndexType>(ty))
    return IntegerType::get(ty.getContext(), 64);
  return ty;
}

// In Triton, !tt.ptr<i1> loads/stores are problematic in downstream lowering,
// whereas boolean XLA buffers are byte-addressed (i8). Represent i1 buffer
// pointees as i8 in Triton and convert on load/store.
static Type getStoragePointeeType(Type logicalElemTy, TileBackend backend) {
  Type norm = normalizeScalarType(logicalElemTy);
  if (backend == TileBackend::Triton && norm.isInteger(1))
    return IntegerType::get(norm.getContext(), 8);
  return norm;
}

static int64_t prevPowerOfTwo(int64_t n) {
  if (n <= 1)
    return 1;
  uint64_t u = static_cast<uint64_t>(n);
  return static_cast<int64_t>(llvm::bit_floor(u));
}

static int64_t nextPowerOfTwo(int64_t n) {
  if (n <= 1)
    return 1;
  uint64_t u = static_cast<uint64_t>(n);
  return static_cast<int64_t>(llvm::bit_ceil(u));
}

static ArrayAttr buildOutputOperandAliases(MLIRContext *ctx,
                                           ArrayRef<unsigned> aliasedOperands) {
  SmallVector<Attribute> attrs;
  bool singleResult = (aliasedOperands.size() == 1);
  for (auto [resIdx, opIdx] : llvm::enumerate(aliasedOperands)) {
    SmallVector<int64_t> outTupleIndices;
    if (!singleResult)
      outTupleIndices.push_back(static_cast<int64_t>(resIdx));
    attrs.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, outTupleIndices, static_cast<int64_t>(opIdx), {}));
  }
  return ArrayAttr::get(ctx, attrs);
}

//===----------------------------------------------------------------------===//
// Preprocessing on cloned kernel region
//===----------------------------------------------------------------------===//

static void stripMemorySpaceCastsInRegion(Region &region) {
  SmallVector<memref::MemorySpaceCastOp> casts;
  region.walk([&](memref::MemorySpaceCastOp op) { casts.push_back(op); });
  for (auto castOp : casts) {
    Value src = castOp.getSource();
    auto srcTy = cast<MemRefType>(src.getType());
    SmallVector<OpOperand *> uses;
    for (OpOperand &u : castOp.getDest().getUses())
      uses.push_back(&u);
    for (OpOperand *u : uses) {
      Operation *owner = u->getOwner();
      if (isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp,
              memref::StoreOp, enzyme::AffineAtomicRMWOp, memref::AtomicRMWOp,
              memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp>(
              owner)) {
        u->set(src);
      } else if (auto subCast = dyn_cast<memref::MemorySpaceCastOp>(owner)) {
        if (subCast.getDest().getType() == srcTy) {
          subCast.getDest().replaceAllUsesWith(src);
          subCast.erase();
        }
      }
    }
    if (castOp.use_empty())
      castOp.erase();
  }

  // Fold pointer2memref(memref2pointer(x)) with identical shape & element type.
  SmallVector<enzymexla::Pointer2MemrefOp> p2ms;
  region.walk([&](enzymexla::Pointer2MemrefOp op) { p2ms.push_back(op); });
  for (auto p2m : p2ms) {
    Value ptr = p2m.getSource();
    while (auto asc = ptr.getDefiningOp<LLVM::AddrSpaceCastOp>())
      ptr = asc.getArg();
    if (auto m2p = ptr.getDefiningOp<enzymexla::Memref2PointerOp>()) {
      auto srcMt = cast<MemRefType>(m2p.getSource().getType());
      auto dstMt = p2m.getType();
      if (srcMt.getShape() == dstMt.getShape() &&
          srcMt.getElementType() == dstMt.getElementType() &&
          srcMt.getLayout() == dstMt.getLayout()) {
        p2m.replaceAllUsesWith(m2p.getSource());
        p2m.erase();
      }
    }
  }
  SmallVector<Operation *> deadPtrCasts;
  region.walk([&](Operation *op) {
    if (isa<LLVM::AddrSpaceCastOp, enzymexla::Memref2PointerOp>(op) &&
        op->use_empty())
      deadPtrCasts.push_back(op);
  });
  for (Operation *op : deadPtrCasts)
    op->erase();
}

static void inlineTrivialScopes(Region &region) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<memref::AllocaScopeOp> scopes;
    region.walk([&](memref::AllocaScopeOp s) { scopes.push_back(s); });
    for (auto scope : scopes) {
      if (!scope.getBodyRegion().hasOneBlock())
        continue;
      Block &blk = scope.getBodyRegion().front();
      auto ret = dyn_cast<memref::AllocaScopeReturnOp>(blk.getTerminator());
      if (!ret)
        continue;
      SmallVector<Value> retVals(ret.getResults());
      ret.erase();
      Block *parentBlk = scope->getBlock();
      parentBlk->getOperations().splice(Block::iterator(scope),
                                        blk.getOperations());
      scope.replaceAllUsesWith(retVals);
      scope.erase();
      changed = true;
    }

    SmallVector<scf::ExecuteRegionOp> execs;
    region.walk([&](scf::ExecuteRegionOp e) { execs.push_back(e); });
    for (auto exec : execs) {
      if (!exec.getRegion().hasOneBlock())
        continue;
      Block &blk = exec.getRegion().front();
      auto yieldOp = dyn_cast<scf::YieldOp>(blk.getTerminator());
      if (!yieldOp)
        continue;
      SmallVector<Value> retVals(yieldOp.getResults());
      yieldOp.erase();
      Block *parentBlk = exec->getBlock();
      parentBlk->getOperations().splice(Block::iterator(exec),
                                        blk.getOperations());
      exec.replaceAllUsesWith(retVals);
      exec.erase();
      changed = true;
    }
  }
}

// Promote single-block local scalar/constant-indexed memref.alloca / alloc
// scratch variables to SSA values when all accesses are in the same block or
// structured affine/scf loops without escaping pointers.
static std::optional<int64_t> getConstFlatScratchIndex(Operation *user,
                                                       MemRefType mt) {
  if (mt.getRank() == 0)
    return 0;
  if (!mt.hasStaticShape())
    return std::nullopt;
  SmallVector<int64_t, 4> coords;
  if (auto ld = dyn_cast<affine::AffineLoadOp>(user)) {
    affine::AffineValueMap avm(ld.getAffineMap(), ld.getIndices());
    avm.composeSimplifyAndCanonicalize();
    for (AffineExpr e : avm.getAffineMap().getResults()) {
      auto ce = dyn_cast<AffineConstantExpr>(e);
      if (!ce)
        return std::nullopt;
      coords.push_back(ce.getValue());
    }
  } else if (auto st = dyn_cast<affine::AffineStoreOp>(user)) {
    affine::AffineValueMap avm(st.getAffineMap(), st.getIndices());
    avm.composeSimplifyAndCanonicalize();
    for (AffineExpr e : avm.getAffineMap().getResults()) {
      auto ce = dyn_cast<AffineConstantExpr>(e);
      if (!ce)
        return std::nullopt;
      coords.push_back(ce.getValue());
    }
  } else if (auto ld = dyn_cast<memref::LoadOp>(user)) {
    for (Value idx : ld.getIndices()) {
      APInt c;
      if (!matchPattern(idx, m_ConstantInt(&c)))
        return std::nullopt;
      coords.push_back(c.getSExtValue());
    }
  } else if (auto st = dyn_cast<memref::StoreOp>(user)) {
    for (Value idx : st.getIndices()) {
      APInt c;
      if (!matchPattern(idx, m_ConstantInt(&c)))
        return std::nullopt;
      coords.push_back(c.getSExtValue());
    }
  } else {
    return std::nullopt;
  }
  if (static_cast<int64_t>(coords.size()) != mt.getRank())
    return std::nullopt;
  int64_t flat = 0;
  for (int64_t d = 0; d < mt.getRank(); ++d) {
    if (coords[d] < 0 || coords[d] >= mt.getDimSize(d))
      return std::nullopt;
    flat = flat * mt.getDimSize(d) + coords[d];
  }
  return flat;
}

static bool promoteScratchInBlock(Block &block, Value scratch, MemRefType mt,
                                  MutableArrayRef<Value> slotVals) {
  SmallVector<Operation *> toErase;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (op.getNumRegions() == 0) {
      bool touchesScratch =
          llvm::any_of(op.getOperands(), [&](Value v) { return v == scratch; });
      if (!touchesScratch)
        continue;
      if (isa<memref::DeallocOp>(op)) {
        toErase.push_back(&op);
        continue;
      }
      auto flatIdx = getConstFlatScratchIndex(&op, mt);
      if (!flatIdx)
        return false;
      if (isa<affine::AffineLoadOp, memref::LoadOp>(op)) {
        Value cur = slotVals[*flatIdx];
        if (!cur) {
          OpBuilder b(&op);
          Type elemTy = mt.getElementType();
          cur = arith::ConstantOp::create(b, op.getLoc(), elemTy,
                                          b.getZeroAttr(elemTy));
          slotVals[*flatIdx] = cur;
        }
        op.getResult(0).replaceAllUsesWith(cur);
        toErase.push_back(&op);
      } else if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
        if (st.getValueToStore() == scratch)
          return false;
        slotVals[*flatIdx] = st.getValueToStore();
        toErase.push_back(&op);
      } else if (auto st = dyn_cast<memref::StoreOp>(op)) {
        if (st.getValueToStore() == scratch)
          return false;
        slotVals[*flatIdx] = st.getValueToStore();
        toErase.push_back(&op);
      } else {
        return false;
      }
      continue;
    }

    // Check if nested regions touch `scratch`.
    bool nestedTouch = false;
    op.walk([&](Operation *inner) {
      if (inner == &op)
        return;
      for (Value v : inner->getOperands())
        if (v == scratch)
          nestedTouch = true;
    });
    if (!nestedTouch)
      continue;

    int64_t numSlots = static_cast<int64_t>(slotVals.size());
    OpBuilder b(&op);
    Type elemTy = mt.getElementType();
    for (int64_t s = 0; s < numSlots; ++s) {
      if (!slotVals[s]) {
        slotVals[s] = arith::ConstantOp::create(b, op.getLoc(), elemTy,
                                                b.getZeroAttr(elemTy));
      }
    }

    if (auto afor = dyn_cast<affine::AffineForOp>(op)) {
      SmallVector<Value> initArgs(afor.getInits());
      unsigned baseIdx = initArgs.size();
      initArgs.append(slotVals.begin(), slotVals.end());
      auto newFor = affine::AffineForOp::create(
          b, afor.getLoc(), afor.getLowerBoundOperands(),
          afor.getLowerBoundMap(), afor.getUpperBoundOperands(),
          afor.getUpperBoundMap(), afor.getStepAsInt(), initArgs);
      Block *oldBody = afor.getBody();
      Block *newBody = newFor.getBody();
      SmallVector<Value> bodySlotVals;
      for (int64_t s = 0; s < numSlots; ++s)
        bodySlotVals.push_back(newBody->getArgument(1 + baseIdx + s));

      IRMapping map;
      map.map(oldBody->getArgument(0), newBody->getArgument(0));
      for (unsigned i = 0; i < baseIdx; ++i)
        map.map(oldBody->getArgument(1 + i), newBody->getArgument(1 + i));

      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newBody);
      SmallVector<Value> origYields;
      for (Operation &innerOp : oldBody->without_terminator())
        bodyBuilder.clone(innerOp, map);
      auto oldYield = cast<affine::AffineYieldOp>(oldBody->getTerminator());
      for (Value y : oldYield.getOperands())
        origYields.push_back(map.lookupOrDefault(y));
      // Remove auto-created empty terminator if present before promoting.
      if (!newBody->empty() && isa<affine::AffineYieldOp>(newBody->back()))
        newBody->back().erase();

      if (!promoteScratchInBlock(*newBody, scratch, mt, bodySlotVals)) {
        newFor.erase();
        return false;
      }
      bodyBuilder.setInsertionPointToEnd(newBody);
      origYields.append(bodySlotVals.begin(), bodySlotVals.end());
      affine::AffineYieldOp::create(bodyBuilder, afor.getLoc(), origYields);

      for (unsigned i = 0; i < baseIdx; ++i)
        afor.getResult(i).replaceAllUsesWith(newFor.getResult(i));
      for (int64_t s = 0; s < numSlots; ++s)
        slotVals[s] = newFor.getResult(baseIdx + s);
      toErase.push_back(&op);
      continue;
    }

    if (auto sfor = dyn_cast<scf::ForOp>(op)) {
      SmallVector<Value> initArgs(sfor.getInitArgs());
      unsigned baseIdx = initArgs.size();
      initArgs.append(slotVals.begin(), slotVals.end());
      auto newFor =
          scf::ForOp::create(b, sfor.getLoc(), sfor.getLowerBound(),
                             sfor.getUpperBound(), sfor.getStep(), initArgs);
      Block *oldBody = sfor.getBody();
      Block *newBody = newFor.getBody();
      SmallVector<Value> bodySlotVals;
      for (int64_t s = 0; s < numSlots; ++s)
        bodySlotVals.push_back(newBody->getArgument(1 + baseIdx + s));

      IRMapping map;
      map.map(oldBody->getArgument(0), newBody->getArgument(0));
      for (unsigned i = 0; i < baseIdx; ++i)
        map.map(oldBody->getArgument(1 + i), newBody->getArgument(1 + i));

      if (!newBody->empty() && isa<scf::YieldOp>(newBody->back()))
        newBody->back().erase();
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newBody);
      SmallVector<Value> origYields;
      for (Operation &innerOp : oldBody->without_terminator())
        bodyBuilder.clone(innerOp, map);
      auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
      for (Value y : oldYield.getResults())
        origYields.push_back(map.lookupOrDefault(y));

      if (!promoteScratchInBlock(*newBody, scratch, mt, bodySlotVals)) {
        newFor.erase();
        return false;
      }
      bodyBuilder.setInsertionPointToEnd(newBody);
      origYields.append(bodySlotVals.begin(), bodySlotVals.end());
      scf::YieldOp::create(bodyBuilder, sfor.getLoc(), origYields);

      for (unsigned i = 0; i < baseIdx; ++i)
        sfor.getResult(i).replaceAllUsesWith(newFor.getResult(i));
      for (int64_t s = 0; s < numSlots; ++s)
        slotVals[s] = newFor.getResult(baseIdx + s);
      toErase.push_back(&op);
      continue;
    }

    if (auto sif = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> resTypes(sif.getResultTypes());
      unsigned baseIdx = resTypes.size();
      for (int64_t s = 0; s < numSlots; ++s)
        resTypes.push_back(elemTy);
      auto newIf =
          scf::IfOp::create(b, sif.getLoc(), resTypes, sif.getCondition(),
                            /*withElseRegion=*/true);
      Block *thenBlk = &newIf.getThenRegion().front();
      if (!thenBlk->empty() && isa<scf::YieldOp>(thenBlk->back()))
        thenBlk->back().erase();
      IRMapping thenMap;
      OpBuilder thenB = OpBuilder::atBlockBegin(thenBlk);
      for (Operation &innerOp : sif.thenBlock()->without_terminator())
        thenB.clone(innerOp, thenMap);
      SmallVector<Value> thenYields;
      for (Value y : sif.thenYield().getResults())
        thenYields.push_back(thenMap.lookupOrDefault(y));
      SmallVector<Value> thenSlots(slotVals.begin(), slotVals.end());
      if (!promoteScratchInBlock(*thenBlk, scratch, mt, thenSlots)) {
        newIf.erase();
        return false;
      }
      thenB.setInsertionPointToEnd(thenBlk);
      thenYields.append(thenSlots.begin(), thenSlots.end());
      scf::YieldOp::create(thenB, sif.getLoc(), thenYields);

      Block *elseBlk = &newIf.getElseRegion().front();
      if (!elseBlk->empty() && isa<scf::YieldOp>(elseBlk->back()))
        elseBlk->back().erase();
      OpBuilder elseB = OpBuilder::atBlockBegin(elseBlk);
      SmallVector<Value> elseYields;
      SmallVector<Value> elseSlots(slotVals.begin(), slotVals.end());
      if (!sif.getElseRegion().empty()) {
        IRMapping elseMap;
        for (Operation &innerOp : sif.elseBlock()->without_terminator())
          elseB.clone(innerOp, elseMap);
        for (Value y : sif.elseYield().getResults())
          elseYields.push_back(elseMap.lookupOrDefault(y));
        if (!promoteScratchInBlock(*elseBlk, scratch, mt, elseSlots)) {
          newIf.erase();
          return false;
        }
      }
      elseB.setInsertionPointToEnd(elseBlk);
      elseYields.append(elseSlots.begin(), elseSlots.end());
      scf::YieldOp::create(elseB, sif.getLoc(), elseYields);

      for (unsigned i = 0; i < baseIdx; ++i)
        sif.getResult(i).replaceAllUsesWith(newIf.getResult(i));
      for (int64_t s = 0; s < numSlots; ++s)
        slotVals[s] = newIf.getResult(baseIdx + s);
      toErase.push_back(&op);
      continue;
    }

    return false;
  }

  for (Operation *op : toErase)
    op->erase();
  return true;
}

static void promoteLocalScratchAllocas(Region &region) {
  SmallVector<Operation *> allocs;
  region.walk([&](Operation *op) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op))
      allocs.push_back(op);
  });
  for (Operation *allocOp : allocs) {
    Value scratch = allocOp->getResult(0);
    auto mt = dyn_cast<MemRefType>(scratch.getType());
    if (!mt || !mt.hasStaticShape())
      continue;
    int64_t numElems = mt.getNumElements();
    if (numElems <= 0 || numElems > 32)
      continue;
    if (!isa<IntegerType, FloatType, IndexType>(mt.getElementType()))
      continue;

    // Verify all uses lie inside allocOp->getBlock() (or regions nested in it)
    // and only via constant-indexed load/store/dealloc.
    Block *defBlock = allocOp->getBlock();
    bool legal = true;
    for (Operation *user : scratch.getUsers()) {
      if (!defBlock->findAncestorOpInBlock(*user)) {
        legal = false;
        break;
      }
      if (isa<memref::DeallocOp>(user))
        continue;
      if (!getConstFlatScratchIndex(user, mt).has_value()) {
        legal = false;
        break;
      }
    }
    if (!legal)
      continue;

    SmallVector<Value> slots(numElems, Value());
    if (promoteScratchInBlock(*defBlock, scratch, mt, slots) &&
        scratch.use_empty()) {
      allocOp->erase();
    }
  }
}

// Coalesce tightly nested `affine.parallel` ops when the outer body contains
// only pure scalar setup ops and a single inner `affine.parallel` whose bounds
// do not depend on outer parallel IVs (and with 0 results).
static void coalesceNestedAffineParallels(Region &region) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<affine::AffineParallelOp> pars;
    region.walk([&](affine::AffineParallelOp p) { pars.push_back(p); });
    for (auto outer : pars) {
      if (outer.getNumResults() != 0)
        continue;
      Block *outerBody = outer.getBody();
      affine::AffineParallelOp inner = nullptr;
      bool canMerge = true;
      for (Operation &op : outerBody->without_terminator()) {
        if (auto p = dyn_cast<affine::AffineParallelOp>(&op)) {
          if (inner || p.getNumResults() != 0) {
            canMerge = false;
            break;
          }
          inner = p;
          continue;
        }
        if (!isPure(&op)) {
          canMerge = false;
          break;
        }
      }
      if (!canMerge || !inner)
        continue;

      // Ensure inner bounds do not depend on outer IVs or ops inside outerBody.
      bool boundsIndependent = true;
      for (Value v : inner.getOperands()) {
        Operation *parentOp = v.getParentRegion()->getParentOp();
        if (v.getParentBlock() == outerBody ||
            (parentOp && outer->isProperAncestor(parentOp))) {
          boundsIndependent = false;
          break;
        }
      }
      if (!boundsIndependent)
        continue;

      // Hoist pure setup ops before `outer` if they don't use outer IVs, or
      // move them into the merged body! Moving them into the merged body works
      // even when they DO use outer IVs!
      SmallVector<Value> lbOperands, ubOperands;
      SmallVector<int64_t> steps;

      unsigned outerDims = outer.getNumDims();
      unsigned innerDims = inner.getNumDims();
      for (unsigned d = 0; d < outerDims; ++d)
        steps.push_back(outer.getSteps()[d]);
      // Simpler & exact: use the AffineParallelOp builder taking
      // ArrayRef<AffineMap> where each map takes the full Group of operands, or
      // build ValueBounds when each axis has single-result map:
      bool simpleMaps = true;
      SmallVector<AffineMap> perDimLb, perDimUb;
      SmallVector<SmallVector<Value>> perDimLbOps, perDimUbOps;
      auto collectAxisBounds = [&](affine::AffineParallelOp p) {
        for (unsigned d = 0; d < p.getNumDims(); ++d) {
          AffineMap lbm = p.getLowerBoundMap(d);
          AffineMap ubm = p.getUpperBoundMap(d);
          if (lbm.getNumResults() != 1 || ubm.getNumResults() != 1) {
            simpleMaps = false;
            return;
          }
          perDimLb.push_back(lbm);
          perDimUb.push_back(ubm);
          auto lbGroup = p.getLowerBoundsOperands();
          auto ubGroup = p.getUpperBoundsOperands();
          perDimLbOps.emplace_back(lbGroup.begin(), lbGroup.end());
          perDimUbOps.emplace_back(ubGroup.begin(), ubGroup.end());
        }
      };
      collectAxisBounds(outer);
      collectAxisBounds(inner);
      if (!simpleMaps)
        continue;

      for (int64_t s : inner.getSteps())
        steps.push_back(s);

      // Re-index each single-result map onto concatenated lbOperands /
      // ubOperands.
      SmallVector<AffineMap> finalLbMaps, finalUbMaps;
      for (size_t i = 0; i < perDimLb.size(); ++i) {
        AffineMap m = perDimLb[i];
        unsigned baseDim = lbOperands.size();
        // Convert all symbols to dims so concatenated ValueRange is pure dims.
        SmallVector<AffineExpr> dimRepls, symRepls;
        for (unsigned j = 0; j < m.getNumDims(); ++j)
          dimRepls.push_back(getAffineDimExpr(baseDim + j, m.getContext()));
        for (unsigned j = 0; j < m.getNumSymbols(); ++j)
          symRepls.push_back(
              getAffineDimExpr(baseDim + m.getNumDims() + j, m.getContext()));
        lbOperands.append(perDimLbOps[i].begin(), perDimLbOps[i].end());
        finalLbMaps.push_back(AffineMap::get(
            /*dimCount=*/0, /*symbolCount=*/0,
            m.getResult(0).replaceDimsAndSymbols(dimRepls, symRepls),
            m.getContext()));
      }
      for (auto &m : finalLbMaps)
        m = AffineMap::get(lbOperands.size(), 0, m.getResult(0));

      for (size_t i = 0; i < perDimUb.size(); ++i) {
        AffineMap m = perDimUb[i];
        unsigned baseDim = ubOperands.size();
        SmallVector<AffineExpr> dimRepls, symRepls;
        for (unsigned j = 0; j < m.getNumDims(); ++j)
          dimRepls.push_back(getAffineDimExpr(baseDim + j, m.getContext()));
        for (unsigned j = 0; j < m.getNumSymbols(); ++j)
          symRepls.push_back(
              getAffineDimExpr(baseDim + m.getNumDims() + j, m.getContext()));
        ubOperands.append(perDimUbOps[i].begin(), perDimUbOps[i].end());
        finalUbMaps.push_back(AffineMap::get(
            /*dimCount=*/0, /*symbolCount=*/0,
            m.getResult(0).replaceDimsAndSymbols(dimRepls, symRepls),
            m.getContext()));
      }
      for (auto &m : finalUbMaps)
        m = AffineMap::get(ubOperands.size(), 0, m.getResult(0));

      OpBuilder b(outer);
      auto merged = affine::AffineParallelOp::create(
          b, outer.getLoc(), TypeRange{}, ArrayRef<arith::AtomicRMWKind>{},
          finalLbMaps, lbOperands, finalUbMaps, ubOperands, steps);
      Block *mergedBody = merged.getBody();
      if (!mergedBody->empty() &&
          isa<affine::AffineYieldOp>(mergedBody->back()))
        mergedBody->back().erase();

      IRMapping map;
      for (unsigned d = 0; d < outerDims; ++d)
        map.map(outerBody->getArgument(d), mergedBody->getArgument(d));
      for (unsigned d = 0; d < innerDims; ++d)
        map.map(inner.getBody()->getArgument(d),
                mergedBody->getArgument(outerDims + d));

      OpBuilder mb = OpBuilder::atBlockBegin(mergedBody);
      for (Operation &op : outerBody->without_terminator()) {
        if (&op == inner.getOperation())
          continue;
        mb.clone(op, map);
      }
      for (Operation &op : inner.getBody()->without_terminator())
        mb.clone(op, map);
      affine::AffineYieldOp::create(mb, outer.getLoc());

      outer.erase();
      changed = true;
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// MemRef View / Subview / Cast Descriptor Tracking
//===----------------------------------------------------------------------===//

struct MemRefViewDesc {
  unsigned argIndex = 0;
  MemRefType baseType;
  Type logicalElemType;
  Type storageElemType;
  Value baseOffsetI64; // null means 0
  SmallVector<Value, 4> stridesI64;
};

//===----------------------------------------------------------------------===//
// Backend-Parameterized Tile Emitter
//===----------------------------------------------------------------------===//

class TileKernelEmitter {
public:
  TileKernelEmitter(MLIRContext *ctx, const GpuTileRaisingOptions &opts)
      : ctx(ctx), opts(opts) {}

  TileBackend getBackend() const { return opts.backend; }

  ArrayRef<int64_t> getShape(Value v) const {
    if (!v)
      return {};
    if (opts.backend == TileBackend::CuTile)
      return cast<cuda_tile::TileType>(v.getType()).getShape();
    if (auto vt = dyn_cast<VectorType>(v.getType()))
      return vt.getShape();
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return rtt.getShape();
    return {};
  }

  Type getElemType(Value v) const {
    if (opts.backend == TileBackend::CuTile)
      return cast<cuda_tile::TileType>(v.getType()).getElementType();
    if (auto vt = dyn_cast<VectorType>(v.getType()))
      return vt.getElementType();
    if (auto rtt = dyn_cast<RankedTensorType>(v.getType()))
      return rtt.getElementType();
    return v.getType();
  }

  bool isScalar(Value v) const { return getShape(v).empty(); }

  Type makeType(ArrayRef<int64_t> shape, Type elemTy) const {
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::TileType::get(shape, elemTy);
    if (shape.empty())
      return elemTy;
    if (opts.backend == TileBackend::Mosaic)
      return VectorType::get(shape, elemTy);
    return RankedTensorType::get(shape, elemTy);
  }

  Type makePointerElemType(Type storagePointeeTy) const {
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::PointerType::get(storagePointeeTy);
#if ENZYME_JAX_ENABLE_TRITON
    return triton::PointerType::get(storagePointeeTy);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  Value emitIntConst(OpBuilder &b, Location loc, int64_t val, unsigned width,
                     ArrayRef<int64_t> shape = {}) {
    Type intTy = IntegerType::get(ctx, width);
    Attribute elemAttr =
        (width == 1) ? static_cast<Attribute>(b.getBoolAttr(val != 0))
                     : static_cast<Attribute>(b.getIntegerAttr(intTy, val));
    if (opts.backend == TileBackend::CuTile) {
      auto tileTy = cuda_tile::TileType::get(shape, intTy);
      auto denseAttr = cast<DenseTypedElementsAttr>(
          DenseElementsAttr::get(tileTy, elemAttr));
      return cuda_tile::ConstantOp::create(b, loc, tileTy, denseAttr);
    }
    if (shape.empty())
      return arith::ConstantOp::create(b, loc, intTy,
                                       cast<TypedAttr>(elemAttr));
    if (opts.backend == TileBackend::Mosaic) {
      auto vt = VectorType::get(shape, intTy);
      return arith::ConstantOp::create(b, loc, vt,
                                       DenseElementsAttr::get(vt, elemAttr));
    }
    auto rtt = RankedTensorType::get(shape, intTy);
    return arith::ConstantOp::create(b, loc, rtt,
                                     DenseElementsAttr::get(rtt, elemAttr));
  }

  Value emitFloatConst(OpBuilder &b, Location loc, APFloat val, FloatType ft,
                       ArrayRef<int64_t> shape = {}) {
    bool losesInfo = false;
    val.convert(ft.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
    auto fAttr = b.getFloatAttr(ft, val);
    if (opts.backend == TileBackend::CuTile) {
      auto tileTy = cuda_tile::TileType::get(shape, ft);
      auto denseAttr =
          cast<DenseTypedElementsAttr>(DenseElementsAttr::get(tileTy, fAttr));
      return cuda_tile::ConstantOp::create(b, loc, tileTy, denseAttr);
    }
    if (shape.empty())
      return arith::ConstantOp::create(b, loc, ft, fAttr);
    if (opts.backend == TileBackend::Mosaic) {
      auto vt = VectorType::get(shape, ft);
      return arith::ConstantOp::create(b, loc, vt,
                                       DenseElementsAttr::get(vt, fAttr));
    }
    auto rtt = RankedTensorType::get(shape, ft);
    return arith::ConstantOp::create(b, loc, rtt,
                                     DenseElementsAttr::get(rtt, fAttr));
  }

  Value emitZero(OpBuilder &b, Location loc, Type scalarElemTy,
                 ArrayRef<int64_t> shape = {}) {
    scalarElemTy = normalizeScalarType(scalarElemTy);
    if (auto it = dyn_cast<IntegerType>(scalarElemTy))
      return emitIntConst(b, loc, 0, it.getWidth(), shape);
    auto ft = cast<FloatType>(scalarElemTy);
    return emitFloatConst(b, loc, APFloat::getZero(ft.getFloatSemantics()), ft,
                          shape);
  }

  Value broadcastToShape(OpBuilder &b, Location loc, Value v,
                         ArrayRef<int64_t> targetShape) {
    ArrayRef<int64_t> curShape = getShape(v);
    if (curShape == targetShape)
      return v;
    assert(!targetShape.empty() && "cannot broadcast tile down to scalar");
    Type elemTy = getElemType(v);

    if (opts.backend == TileBackend::CuTile) {
      Value cur = v;
      if (curShape.empty()) {
        SmallVector<int64_t> ones(targetShape.size(), 1);
        auto oneTy = cuda_tile::TileType::get(ones, elemTy);
        cur = cuda_tile::ReshapeOp::create(b, loc, oneTy, cur);
        curShape = getShape(cur);
      }
      assert(curShape.size() == targetShape.size());
      if (curShape != targetShape) {
        auto dstTy = cuda_tile::TileType::get(targetShape, elemTy);
        cur = cuda_tile::BroadcastOp::create(b, loc, dstTy, cur);
      }
      return cur;
    }

    if (opts.backend == TileBackend::Mosaic) {
      auto dstTy = VectorType::get(targetShape, elemTy);
      return vector::BroadcastOp::create(b, loc, dstTy, v);
    }

#if ENZYME_JAX_ENABLE_TRITON
    Value cur = v;
    if (curShape.empty()) {
      auto dstTy = RankedTensorType::get(targetShape, elemTy);
      return triton::SplatOp::create(b, loc, dstTy, cur);
    }
    assert(curShape.size() == targetShape.size());
    if (curShape != targetShape) {
      auto dstTy = RankedTensorType::get(targetShape, elemTy);
      cur = triton::BroadcastOp::create(b, loc, dstTy, cur);
    }
    return cur;
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  std::pair<Value, Value> matchShapes(OpBuilder &b, Location loc, Value lhs,
                                      Value rhs) {
    ArrayRef<int64_t> s0 = getShape(lhs);
    ArrayRef<int64_t> s1 = getShape(rhs);
    if (s0 == s1)
      return {lhs, rhs};
    if (s0.empty())
      return {broadcastToShape(b, loc, lhs, s1), rhs};
    if (s1.empty())
      return {lhs, broadcastToShape(b, loc, rhs, s0)};
    assert(s0.size() == s1.size() && "mismatched tile ranks");
    SmallVector<int64_t> common(s0.size());
    for (size_t i = 0; i < s0.size(); ++i)
      common[i] = std::max(s0[i], s1[i]);
    return {broadcastToShape(b, loc, lhs, common),
            broadcastToShape(b, loc, rhs, common)};
  }

  std::tuple<Value, Value, Value> matchShapes3(OpBuilder &b, Location loc,
                                               Value a, Value c, Value d) {
    auto [ab, cb] = matchShapes(b, loc, a, c);
    auto [ab2, db] = matchShapes(b, loc, ab, d);
    auto [cb2, db2] = matchShapes(b, loc, cb, db);
    return {ab2, cb2, db2};
  }

  // Emit 1D range [0, tileSize) as i64, then reshape + broadcast along axis `d`
  // of `tileShape`.
  Value emitAxisIotaI64(OpBuilder &b, Location loc, unsigned axis,
                        ArrayRef<int64_t> tileShape) {
    int64_t n = tileShape[axis];
    Value range1D;
    if (opts.backend == TileBackend::CuTile) {
      auto iotaTy = cuda_tile::TileType::get({n}, b.getI64Type());
      range1D = cuda_tile::IotaOp::create(b, loc, iotaTy);
      if (tileShape.size() > 1) {
        SmallVector<int64_t> reshapedDims(tileShape.size(), 1);
        reshapedDims[axis] = n;
        auto reshapedTy =
            cuda_tile::TileType::get(reshapedDims, b.getI64Type());
        range1D = cuda_tile::ReshapeOp::create(b, loc, reshapedTy, range1D);
      }
      return broadcastToShape(b, loc, range1D, tileShape);
    }

    if (opts.backend == TileBackend::Mosaic) {
      if (n == 1)
        return emitIntConst(b, loc, 0, 64, tileShape);
      auto iotaTy = VectorType::get(tileShape, b.getI64Type());
      return tpu::IotaOp::create(b, loc, iotaTy,
                                 ArrayRef<int32_t>{static_cast<int32_t>(axis)});
    }

#if ENZYME_JAX_ENABLE_TRITON
    Value rangeI32;
    if (n == 1) {
      rangeI32 = emitIntConst(b, loc, 0, 32, {1});
    } else {
      auto r32Ty = RankedTensorType::get({n}, b.getI32Type());
      rangeI32 = triton::MakeRangeOp::create(b, loc, r32Ty, 0,
                                             static_cast<int32_t>(n));
    }
    auto r64Ty = RankedTensorType::get({n}, b.getI64Type());
    range1D = arith::ExtSIOp::create(b, loc, r64Ty, rangeI32);
    Value cur = range1D;
    for (unsigned d = 0; d < axis; ++d)
      cur = triton::ExpandDimsOp::create(b, loc, cur, 0);
    for (unsigned d = axis + 1; d < tileShape.size(); ++d)
      cur = triton::ExpandDimsOp::create(
          b, loc, cur, static_cast<int>(getShape(cur).size()));
    return broadcastToShape(b, loc, cur, tileShape);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  // Emit block/program ID along axis (0=x, 1=y, 2=z) as scalar i64.
  Value emitBlockIdI64(OpBuilder &b, Location loc, int axis) {
    if (opts.backend == TileBackend::CuTile) {
      auto ids = cuda_tile::GetTileBlockIdOp::create(b, loc);
      Value id32 = (axis == 0)   ? ids.getBlockIdX()
                   : (axis == 1) ? ids.getBlockIdY()
                                 : ids.getBlockIdZ();
      return emitIntCast(b, loc, id32, b.getI64Type(), /*isSigned=*/false);
    }
    if (opts.backend == TileBackend::Mosaic) {
      return arith::ExtSIOp::create(b, loc, b.getI64Type(),
                                    mosaicBlockIdsI32[axis]);
    }
#if ENZYME_JAX_ENABLE_TRITON
    Value id32 = triton::GetProgramIdOp::create(b, loc, axis);
    return arith::ExtUIOp::create(b, loc, b.getI64Type(), id32);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  Value emitNumBlocksI64(OpBuilder &b, Location loc, int axis) {
    if (opts.backend == TileBackend::CuTile) {
      auto sz = cuda_tile::GetNumTileBlocksOp::create(b, loc);
      Value n32 = (axis == 0)   ? sz.getGridSizeX()
                  : (axis == 1) ? sz.getGridSizeY()
                                : sz.getGridSizeZ();
      return emitIntCast(b, loc, n32, b.getI64Type(), /*isSigned=*/false);
    }
    if (opts.backend == TileBackend::Mosaic) {
      Value n32 = tpu::GetIterationBoundOp::create(b, loc, b.getI32Type(),
                                                   static_cast<uint32_t>(axis));
      return arith::ExtSIOp::create(b, loc, b.getI64Type(), n32);
    }
#if ENZYME_JAX_ENABLE_TRITON
    Value n32 = triton::GetNumProgramsOp::create(b, loc, axis);
    return arith::ExtUIOp::create(b, loc, b.getI64Type(), n32);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  // Integer binary operations
  Value emitAddI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::AddIOp::create(b, loc, lhs, rhs);
    return arith::AddIOp::create(b, loc, lhs, rhs);
  }

  Value emitSubI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::SubIOp::create(b, loc, lhs, rhs);
    return arith::SubIOp::create(b, loc, lhs, rhs);
  }

  Value emitMulI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::MulIOp::create(b, loc, lhs, rhs);
    return arith::MulIOp::create(b, loc, lhs, rhs);
  }

  Value emitDivI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::DivIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned,
                                       cuda_tile::RoundingMode::ZERO);
    }
    if (isSigned)
      return arith::DivSIOp::create(b, loc, lhs, rhs);
    return arith::DivUIOp::create(b, loc, lhs, rhs);
  }

  Value emitFloorDivSI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::DivIOp::create(b, loc, lhs, rhs,
                                       cuda_tile::Signedness::Signed,
                                       cuda_tile::RoundingMode::NEGATIVE_INF);
    }
    // Euclidean/floor signed division using standard integer ops (since Triton
    // lowers arith.divsi/remsi directly without arith.floordivsi).
    Value q = arith::DivSIOp::create(b, loc, lhs, rhs);
    Value r = arith::RemSIOp::create(b, loc, lhs, rhs);
    unsigned w = cast<IntegerType>(getElemType(lhs)).getWidth();
    Value zero = emitIntConst(b, loc, 0, w, getShape(lhs));
    Value one = emitIntConst(b, loc, 1, w, getShape(lhs));
    Value rNonZero =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, r, zero);
    Value diffSign =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                              arith::XOrIOp::create(b, loc, lhs, rhs), zero);
    Value adjust = arith::AndIOp::create(b, loc, rNonZero, diffSign);
    Value qMinus1 = arith::SubIOp::create(b, loc, q, one);
    return arith::SelectOp::create(b, loc, adjust, qMinus1, q);
  }

  Value emitCeilDivI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                     bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::DivIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned,
                                       cuda_tile::RoundingMode::POSITIVE_INF);
    }
    unsigned w = cast<IntegerType>(getElemType(lhs)).getWidth();
    Value zero = emitIntConst(b, loc, 0, w, getShape(lhs));
    Value one = emitIntConst(b, loc, 1, w, getShape(lhs));
    if (!isSigned) {
      Value q = arith::DivUIOp::create(b, loc, lhs, rhs);
      Value r = arith::RemUIOp::create(b, loc, lhs, rhs);
      Value hasRem =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, r, zero);
      Value qPlus1 = arith::AddIOp::create(b, loc, q, one);
      return arith::SelectOp::create(b, loc, hasRem, qPlus1, q);
    }
    Value q = arith::DivSIOp::create(b, loc, lhs, rhs);
    Value r = arith::RemSIOp::create(b, loc, lhs, rhs);
    Value rNonZero =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, r, zero);
    Value sameSign =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                              arith::XOrIOp::create(b, loc, lhs, rhs), zero);
    Value adjust = arith::AndIOp::create(b, loc, rNonZero, sameSign);
    Value qPlus1 = arith::AddIOp::create(b, loc, q, one);
    return arith::SelectOp::create(b, loc, adjust, qPlus1, q);
  }

  Value emitRemI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::RemIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned);
    }
    if (isSigned)
      return arith::RemSIOp::create(b, loc, lhs, rhs);
    return arith::RemUIOp::create(b, loc, lhs, rhs);
  }

  Value emitEuclideanModSI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    Value rem = emitRemI(b, loc, lhs, rhs, /*isSigned=*/true);
    unsigned w = cast<IntegerType>(getElemType(rem)).getWidth();
    Value zero = emitIntConst(b, loc, 0, w, getShape(rem));
    Value isNeg = emitCmpI(b, loc, arith::CmpIPredicate::slt, rem, zero);
    Value adjusted = emitAddI(b, loc, rem, rhs);
    return emitSelect(b, loc, isNeg, adjusted, rem);
  }

  Value emitAndI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::AndIOp::create(b, loc, lhs, rhs);
    return arith::AndIOp::create(b, loc, lhs, rhs);
  }

  Value emitOrI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::OrIOp::create(b, loc, lhs, rhs);
    return arith::OrIOp::create(b, loc, lhs, rhs);
  }

  Value emitXorI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::XOrIOp::create(b, loc, lhs, rhs);
    return arith::XOrIOp::create(b, loc, lhs, rhs);
  }

  Value emitNotI1(OpBuilder &b, Location loc, Value v) {
    Value ones = emitIntConst(b, loc, 1, 1, getShape(v));
    return emitXorI(b, loc, v, ones);
  }

  Value emitShLI(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::ShLIOp::create(b, loc, lhs, rhs);
    return arith::ShLIOp::create(b, loc, lhs, rhs);
  }

  Value emitShRI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::ShRIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned);
    }
    if (isSigned)
      return arith::ShRSIOp::create(b, loc, lhs, rhs);
    return arith::ShRUIOp::create(b, loc, lhs, rhs);
  }

  Value emitMaxI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::MaxIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned);
    }
    if (isSigned)
      return arith::MaxSIOp::create(b, loc, lhs, rhs);
    return arith::MaxUIOp::create(b, loc, lhs, rhs);
  }

  Value emitMinI(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool isSigned) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::MinIOp::create(b, loc, lhs, rhs,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned);
    }
    if (isSigned)
      return arith::MinSIOp::create(b, loc, lhs, rhs);
    return arith::MinUIOp::create(b, loc, lhs, rhs);
  }

  Value emitCmpI(OpBuilder &b, Location loc, arith::CmpIPredicate pred,
                 Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      cuda_tile::ComparisonPredicate cp = cuda_tile::ComparisonPredicate::EQUAL;
      cuda_tile::Signedness sign = cuda_tile::Signedness::Signed;
      switch (pred) {
      case arith::CmpIPredicate::eq:
        cp = cuda_tile::ComparisonPredicate::EQUAL;
        break;
      case arith::CmpIPredicate::ne:
        cp = cuda_tile::ComparisonPredicate::NOT_EQUAL;
        break;
      case arith::CmpIPredicate::slt:
        cp = cuda_tile::ComparisonPredicate::LESS_THAN;
        sign = cuda_tile::Signedness::Signed;
        break;
      case arith::CmpIPredicate::sle:
        cp = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
        sign = cuda_tile::Signedness::Signed;
        break;
      case arith::CmpIPredicate::sgt:
        cp = cuda_tile::ComparisonPredicate::GREATER_THAN;
        sign = cuda_tile::Signedness::Signed;
        break;
      case arith::CmpIPredicate::sge:
        cp = cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
        sign = cuda_tile::Signedness::Signed;
        break;
      case arith::CmpIPredicate::ult:
        cp = cuda_tile::ComparisonPredicate::LESS_THAN;
        sign = cuda_tile::Signedness::Unsigned;
        break;
      case arith::CmpIPredicate::ule:
        cp = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
        sign = cuda_tile::Signedness::Unsigned;
        break;
      case arith::CmpIPredicate::ugt:
        cp = cuda_tile::ComparisonPredicate::GREATER_THAN;
        sign = cuda_tile::Signedness::Unsigned;
        break;
      case arith::CmpIPredicate::uge:
        cp = cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
        sign = cuda_tile::Signedness::Unsigned;
        break;
      }
      return cuda_tile::CmpIOp::create(b, loc, cp, lhs, rhs, sign);
    }
    return arith::CmpIOp::create(b, loc, pred, lhs, rhs);
  }

  Value emitCmpF(OpBuilder &b, Location loc, arith::CmpFPredicate pred,
                 Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile) {
      using CP = cuda_tile::ComparisonPredicate;
      using CO = cuda_tile::ComparisonOrdering;
      switch (pred) {
      case arith::CmpFPredicate::AlwaysFalse:
        return emitIntConst(b, loc, 0, 1, getShape(lhs));
      case arith::CmpFPredicate::AlwaysTrue:
        return emitIntConst(b, loc, 1, 1, getShape(lhs));
      case arith::CmpFPredicate::OEQ:
        return cuda_tile::CmpFOp::create(b, loc, CP::EQUAL, CO::ORDERED, lhs,
                                         rhs);
      case arith::CmpFPredicate::OGT:
        return cuda_tile::CmpFOp::create(b, loc, CP::GREATER_THAN, CO::ORDERED,
                                         lhs, rhs);
      case arith::CmpFPredicate::OGE:
        return cuda_tile::CmpFOp::create(b, loc, CP::GREATER_THAN_OR_EQUAL,
                                         CO::ORDERED, lhs, rhs);
      case arith::CmpFPredicate::OLT:
        return cuda_tile::CmpFOp::create(b, loc, CP::LESS_THAN, CO::ORDERED,
                                         lhs, rhs);
      case arith::CmpFPredicate::OLE:
        return cuda_tile::CmpFOp::create(b, loc, CP::LESS_THAN_OR_EQUAL,
                                         CO::ORDERED, lhs, rhs);
      case arith::CmpFPredicate::ONE:
        return cuda_tile::CmpFOp::create(b, loc, CP::NOT_EQUAL, CO::ORDERED,
                                         lhs, rhs);
      case arith::CmpFPredicate::ORD: {
        Value lOrd =
            cuda_tile::CmpFOp::create(b, loc, CP::EQUAL, CO::ORDERED, lhs, lhs);
        Value rOrd =
            cuda_tile::CmpFOp::create(b, loc, CP::EQUAL, CO::ORDERED, rhs, rhs);
        return emitAndI(b, loc, lOrd, rOrd);
      }
      case arith::CmpFPredicate::UEQ:
        return cuda_tile::CmpFOp::create(b, loc, CP::EQUAL, CO::UNORDERED, lhs,
                                         rhs);
      case arith::CmpFPredicate::UGT:
        return cuda_tile::CmpFOp::create(b, loc, CP::GREATER_THAN,
                                         CO::UNORDERED, lhs, rhs);
      case arith::CmpFPredicate::UGE:
        return cuda_tile::CmpFOp::create(b, loc, CP::GREATER_THAN_OR_EQUAL,
                                         CO::UNORDERED, lhs, rhs);
      case arith::CmpFPredicate::ULT:
        return cuda_tile::CmpFOp::create(b, loc, CP::LESS_THAN, CO::UNORDERED,
                                         lhs, rhs);
      case arith::CmpFPredicate::ULE:
        return cuda_tile::CmpFOp::create(b, loc, CP::LESS_THAN_OR_EQUAL,
                                         CO::UNORDERED, lhs, rhs);
      case arith::CmpFPredicate::UNE:
        return cuda_tile::CmpFOp::create(b, loc, CP::NOT_EQUAL, CO::UNORDERED,
                                         lhs, rhs);
      case arith::CmpFPredicate::UNO: {
        Value lUno = cuda_tile::CmpFOp::create(b, loc, CP::NOT_EQUAL,
                                               CO::UNORDERED, lhs, lhs);
        Value rUno = cuda_tile::CmpFOp::create(b, loc, CP::NOT_EQUAL,
                                               CO::UNORDERED, rhs, rhs);
        return emitOrI(b, loc, lUno, rUno);
      }
      }
      llvm_unreachable("unhandled CmpFPredicate");
    }
    return arith::CmpFOp::create(b, loc, pred, lhs, rhs);
  }

  // Float arithmetic
  Value emitAddF(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::AddFOp::create(
          b, loc, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN, false);
    return arith::AddFOp::create(b, loc, lhs, rhs);
  }

  Value emitSubF(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::SubFOp::create(
          b, loc, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN, false);
    return arith::SubFOp::create(b, loc, lhs, rhs);
  }

  Value emitMulF(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::MulFOp::create(
          b, loc, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN, false);
    return arith::MulFOp::create(b, loc, lhs, rhs);
  }

  Value emitDivF(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::DivFOp::create(
          b, loc, lhs, rhs, cuda_tile::RoundingMode::NEAREST_EVEN, false);
    return arith::DivFOp::create(b, loc, lhs, rhs);
  }

  Value emitRemF(OpBuilder &b, Location loc, Value lhs, Value rhs) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::RemFOp::create(b, loc, lhs, rhs);
    return arith::RemFOp::create(b, loc, lhs, rhs);
  }

  Value emitNegF(OpBuilder &b, Location loc, Value v) {
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::NegFOp::create(b, loc, v);
    return arith::NegFOp::create(b, loc, v);
  }

  Value emitMaxF(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool propagateNan) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::MaxFOp::create(b, loc, lhs, rhs, propagateNan, false);
    if (propagateNan)
      return arith::MaximumFOp::create(b, loc, lhs, rhs);
    return arith::MaxNumFOp::create(b, loc, lhs, rhs);
  }

  Value emitMinF(OpBuilder &b, Location loc, Value lhs, Value rhs,
                 bool propagateNan) {
    std::tie(lhs, rhs) = matchShapes(b, loc, lhs, rhs);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::MinFOp::create(b, loc, lhs, rhs, propagateNan, false);
    if (propagateNan)
      return arith::MinimumFOp::create(b, loc, lhs, rhs);
    return arith::MinNumFOp::create(b, loc, lhs, rhs);
  }

  Value emitFma(OpBuilder &b, Location loc, Value a, Value c, Value d) {
    std::tie(a, c, d) = matchShapes3(b, loc, a, c, d);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::FmaOp::create(
          b, loc, a, c, d, cuda_tile::RoundingMode::NEAREST_EVEN, false);
    return math::FmaOp::create(b, loc, a, c, d);
  }

  Value emitSelect(OpBuilder &b, Location loc, Value cond, Value tVal,
                   Value fVal) {
    std::tie(cond, tVal, fVal) = matchShapes3(b, loc, cond, tVal, fVal);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::SelectOp::create(b, loc, tVal.getType(), cond, tVal,
                                         fVal);
    return arith::SelectOp::create(b, loc, cond, tVal, fVal);
  }

  // Casts
  Value emitIntCast(OpBuilder &b, Location loc, Value v, Type dstScalarIntTy,
                    bool isSigned) {
    dstScalarIntTy = normalizeScalarType(dstScalarIntTy);
    auto srcIntTy = cast<IntegerType>(getElemType(v));
    auto dstIntTy = cast<IntegerType>(dstScalarIntTy);
    if (srcIntTy.getWidth() == dstIntTy.getWidth())
      return v;
    Type resTy = makeType(getShape(v), dstIntTy);
    if (srcIntTy.getWidth() < dstIntTy.getWidth()) {
      if (opts.backend == TileBackend::CuTile) {
        return cuda_tile::ExtIOp::create(b, loc, resTy, v,
                                         isSigned
                                             ? cuda_tile::Signedness::Signed
                                             : cuda_tile::Signedness::Unsigned);
      }
      if (isSigned)
        return arith::ExtSIOp::create(b, loc, resTy, v);
      return arith::ExtUIOp::create(b, loc, resTy, v);
    }
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::TruncIOp::create(b, loc, resTy, v);
    return arith::TruncIOp::create(b, loc, resTy, v);
  }

  Value emitFloatCast(OpBuilder &b, Location loc, Value v,
                      FloatType dstScalarFloatTy) {
    auto srcFloatTy = cast<FloatType>(getElemType(v));
    if (srcFloatTy == dstScalarFloatTy)
      return v;
    Type resTy = makeType(getShape(v), dstScalarFloatTy);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::FToFOp::create(b, loc, resTy, v,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    if (srcFloatTy.getWidth() < dstScalarFloatTy.getWidth())
      return arith::ExtFOp::create(b, loc, resTy, v);
    return arith::TruncFOp::create(b, loc, resTy, v);
  }

  Value emitIToF(OpBuilder &b, Location loc, Value v, FloatType dstFloatTy,
                 bool isSigned) {
    Type resTy = makeType(getShape(v), dstFloatTy);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::IToFOp::create(b, loc, resTy, v,
                                       isSigned
                                           ? cuda_tile::Signedness::Signed
                                           : cuda_tile::Signedness::Unsigned,
                                       cuda_tile::RoundingMode::NEAREST_EVEN);
    }
    if (isSigned)
      return arith::SIToFPOp::create(b, loc, resTy, v);
    return arith::UIToFPOp::create(b, loc, resTy, v);
  }

  Value emitFToI(OpBuilder &b, Location loc, Value v, IntegerType dstIntTy,
                 bool isSigned) {
    Type resTy = makeType(getShape(v), dstIntTy);
    if (opts.backend == TileBackend::CuTile) {
      return cuda_tile::FToIOp::create(
          b, loc, resTy, v,
          isSigned ? cuda_tile::Signedness::Signed
                   : cuda_tile::Signedness::Unsigned,
          cuda_tile::RoundingMode::NEAREST_INT_TO_ZERO, false);
    }
    if (isSigned)
      return arith::FPToSIOp::create(b, loc, resTy, v);
    return arith::FPToUIOp::create(b, loc, resTy, v);
  }

  Value emitBitcast(OpBuilder &b, Location loc, Value v, Type dstScalarTy) {
    dstScalarTy = normalizeScalarType(dstScalarTy);
    if (getElemType(v) == dstScalarTy)
      return v;
    Type resTy = makeType(getShape(v), dstScalarTy);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::BitcastOp::create(b, loc, resTy, v);
    if (opts.backend == TileBackend::Mosaic)
      return arith::BitcastOp::create(b, loc, resTy, v);
#if ENZYME_JAX_ENABLE_TRITON
    return triton::BitcastOp::create(b, loc, resTy, v);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  struct Linear1DIotaForm {
    int64_t iotaCoeff = 0;
    int64_t constOffset = 0;
    SmallVector<std::pair<Value, int64_t>, 4> terms;
  };

  std::optional<Linear1DIotaForm> matchLinear1DIota(Value v) const {
    if (auto iota = v.getDefiningOp<tpu::IotaOp>()) {
      if (iota.getDimensions().size() == 1 && iota.getDimensions()[0] == 0) {
        Linear1DIotaForm res;
        res.iotaCoeff = 1;
        return res;
      }
      return std::nullopt;
    }
    SplatElementsAttr splat;
    if (matchPattern(v, m_Constant(&splat)) && splat.isSplat()) {
      Linear1DIotaForm res;
      res.constOffset = splat.getSplatValue<APInt>().getSExtValue();
      return res;
    }
    if (auto bcast = v.getDefiningOp<vector::BroadcastOp>()) {
      Value src = bcast.getSource();
      if (!src.getType().isInteger(64))
        return std::nullopt;
      APInt c;
      Linear1DIotaForm res;
      if (matchPattern(src, m_ConstantInt(&c)))
        res.constOffset = c.getSExtValue();
      else
        res.terms.push_back({src, 1});
      return res;
    }
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      auto lhs = matchLinear1DIota(add.getLhs());
      auto rhs = matchLinear1DIota(add.getRhs());
      if (!lhs || !rhs)
        return std::nullopt;
      lhs->iotaCoeff += rhs->iotaCoeff;
      lhs->constOffset += rhs->constOffset;
      lhs->terms.append(rhs->terms.begin(), rhs->terms.end());
      return lhs;
    }
    if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
      auto lhs = matchLinear1DIota(sub.getLhs());
      auto rhs = matchLinear1DIota(sub.getRhs());
      if (!lhs || !rhs)
        return std::nullopt;
      lhs->iotaCoeff -= rhs->iotaCoeff;
      lhs->constOffset -= rhs->constOffset;
      for (auto &[val, scale] : rhs->terms)
        lhs->terms.push_back({val, -scale});
      return lhs;
    }
    if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
      auto lhs = matchLinear1DIota(mul.getLhs());
      auto rhs = matchLinear1DIota(mul.getRhs());
      if (!lhs || !rhs)
        return std::nullopt;
      if (rhs->iotaCoeff == 0 && rhs->terms.empty()) {
        int64_t k = rhs->constOffset;
        lhs->iotaCoeff *= k;
        lhs->constOffset *= k;
        for (auto &t : lhs->terms)
          t.second *= k;
        return lhs;
      }
      if (lhs->iotaCoeff == 0 && lhs->terms.empty()) {
        int64_t k = lhs->constOffset;
        rhs->iotaCoeff *= k;
        rhs->constOffset *= k;
        for (auto &t : rhs->terms)
          t.second *= k;
        return rhs;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  Value extractContiguous1DBaseI64(OpBuilder &b, Location loc,
                                   Value elemOffsetI64) {
    auto form = matchLinear1DIota(elemOffsetI64);
    if (!form || form->iotaCoeff != 1)
      return nullptr;
    Value acc;
    for (auto &[scalarVal, scale] : form->terms) {
      if (scale == 0)
        continue;
      Value term = scalarVal;
      if (scale != 1) {
        Value sc = emitIntConst(b, loc, scale, 64);
        term = arith::MulIOp::create(b, loc, term, sc);
      }
      acc = acc ? Value(arith::AddIOp::create(b, loc, acc, term)) : term;
    }
    if (!acc)
      return emitIntConst(b, loc, form->constOffset, 64);
    if (form->constOffset != 0) {
      Value co = emitIntConst(b, loc, form->constOffset, 64);
      acc = arith::AddIOp::create(b, loc, acc, co);
    }
    return acc;
  }

  // Pointer offset + global memory load/store/atomic
  Value emitPtrOffset(OpBuilder &b, Location loc, Value basePtrScalar,
                      Value elemOffsetI64) {
    ArrayRef<int64_t> shape = getShape(elemOffsetI64);
    Value ptr = shape.empty() ? basePtrScalar
                              : broadcastToShape(b, loc, basePtrScalar, shape);
    if (opts.backend == TileBackend::CuTile)
      return cuda_tile::OffsetOp::create(b, loc, ptr.getType(), ptr,
                                         elemOffsetI64);
#if ENZYME_JAX_ENABLE_TRITON
    return triton::AddPtrOp::create(b, loc, ptr.getType(), ptr, elemOffsetI64);
#else
    llvm_unreachable("Triton disabled");
#endif
  }

  Value emitLoad(OpBuilder &b, Location loc, Value basePtrScalar,
                 Value elemOffsetI64, Type logicalElemTy, Value maskOrNull,
                 Value &cuTileToken) {
    logicalElemTy = normalizeScalarType(logicalElemTy);
    Type storageElemTy = getStoragePointeeType(logicalElemTy, opts.backend);
    if (maskOrNull && getShape(maskOrNull) != getShape(elemOffsetI64)) {
      std::tie(elemOffsetI64, maskOrNull) =
          matchShapes(b, loc, elemOffsetI64, maskOrNull);
    }
    if (opts.backend == TileBackend::Mosaic) {
      ArrayRef<int64_t> shape = getShape(elemOffsetI64);
      if (shape.empty()) {
        if (!maskOrNull) {
          Value idx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                 elemOffsetI64);
          return memref::LoadOp::create(b, loc, basePtrScalar, ValueRange{idx});
        }
        auto ifOp = scf::IfOp::create(b, loc, TypeRange{storageElemTy},
                                      maskOrNull, /*withElseRegion=*/true);
        Block *tBlk = ifOp.thenBlock();
        if (!tBlk->empty() && isa<scf::YieldOp>(tBlk->back()))
          tBlk->back().erase();
        {
          OpBuilder tb = OpBuilder::atBlockBegin(tBlk);
          Value idx = arith::IndexCastOp::create(tb, loc, tb.getIndexType(),
                                                 elemOffsetI64);
          Value ldVal =
              memref::LoadOp::create(tb, loc, basePtrScalar, ValueRange{idx});
          scf::YieldOp::create(tb, loc, ValueRange{ldVal});
        }
        Block *eBlk = ifOp.elseBlock();
        if (!eBlk->empty() && isa<scf::YieldOp>(eBlk->back()))
          eBlk->back().erase();
        {
          OpBuilder eb = OpBuilder::atBlockBegin(eBlk);
          Value z = emitZero(eb, loc, storageElemTy);
          scf::YieldOp::create(eb, loc, ValueRange{z});
        }
        return ifOp.getResult(0);
      }

      auto vecTy = VectorType::get(shape, storageElemTy);
      unsigned bitwidth = storageElemTy.getIntOrFloatBitWidth();
      if (shape.size() == 1 && (!maskOrNull || bitwidth == 32)) {
        if (Value scalarBaseI64 =
                extractContiguous1DBaseI64(b, loc, elemOffsetI64)) {
          Value baseIdx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                     scalarBaseI64);
          Value raw = tpu::VectorLoadOp::create(
              b, loc, vecTy, basePtrScalar, ValueRange{baseIdx}, maskOrNull);
          if (maskOrNull) {
            Value z = emitZero(b, loc, storageElemTy, shape);
            raw = arith::SelectOp::create(b, loc, maskOrNull, raw, z);
          }
          return raw;
        }
      }
      if (bitwidth == 32) {
        Value idxI32 = emitIntCast(b, loc, elemOffsetI64, b.getI32Type(),
                                   /*isSigned=*/true);
        Value raw = tpu::VectorLoadIdxOp::create(
            b, loc, vecTy, basePtrScalar, ValueRange{idxI32}, maskOrNull);
        if (maskOrNull) {
          Value z = emitZero(b, loc, storageElemTy, shape);
          raw = arith::SelectOp::create(b, loc, maskOrNull, raw, z);
        }
        return raw;
      }
      Value c0Idx = arith::ConstantIndexOp::create(b, loc, 0);
      Value maskVec =
          maskOrNull ? maskOrNull : emitIntConst(b, loc, 1, 1, shape);
      Value zeroVec = emitZero(b, loc, storageElemTy, shape);
      return vector::GatherOp::create(b, loc, vecTy, basePtrScalar,
                                      ValueRange{c0Idx}, elemOffsetI64, maskVec,
                                      zeroVec);
    }
    Value ptr = emitPtrOffset(b, loc, basePtrScalar, elemOffsetI64);
    ArrayRef<int64_t> shape = getShape(ptr);

    Value rawVal;
    if (opts.backend == TileBackend::CuTile) {
      auto valTy = cuda_tile::TileType::get(shape, storageElemTy);
      auto tokTy = cuda_tile::TokenType::get(ctx);
      Value padOrNull =
          maskOrNull ? emitZero(b, loc, storageElemTy, shape) : Value();
      auto ld = cuda_tile::LoadPtrTkoOp::create(
          b, loc, valTy, tokTy, cuda_tile::MemoryOrderingSemantics::WEAK,
          cuda_tile::MemoryScopeAttr(), ptr, maskOrNull, padOrNull, cuTileToken,
          cuda_tile::OptimizationHintsAttr());
      cuTileToken = ld.getResultToken();
      rawVal = ld.getResult();
    } else {
#if ENZYME_JAX_ENABLE_TRITON
      if (maskOrNull) {
        Value pad = emitZero(b, loc, storageElemTy, shape);
        rawVal = triton::LoadOp::create(b, loc, ptr, maskOrNull, pad,
                                        /*isVolatile=*/false);
      } else {
        rawVal = triton::LoadOp::create(b, loc, ptr, /*isVolatile=*/false);
      }
#else
      llvm_unreachable("Triton disabled");
#endif
    }

    if (storageElemTy != logicalElemTy) {
      // i8 -> i1 conversion
      Value zero8 = emitIntConst(b, loc, 0, 8, shape);
      return emitCmpI(b, loc, arith::CmpIPredicate::ne, rawVal, zero8);
    }
    return rawVal;
  }

  void emitStore(OpBuilder &b, Location loc, Value basePtrScalar,
                 Value elemOffsetI64, Value storedVal, Type logicalElemTy,
                 Value maskOrNull, Value &cuTileToken) {
    logicalElemTy = normalizeScalarType(logicalElemTy);
    Type storageElemTy = getStoragePointeeType(logicalElemTy, opts.backend);
    Value valToStore = storedVal;
    if (storageElemTy != logicalElemTy) {
      valToStore =
          emitIntCast(b, loc, valToStore, storageElemTy, /*isSigned=*/false);
    }
    std::tie(elemOffsetI64, valToStore) =
        matchShapes(b, loc, elemOffsetI64, valToStore);
    if (maskOrNull) {
      std::tie(elemOffsetI64, maskOrNull) =
          matchShapes(b, loc, elemOffsetI64, maskOrNull);
      valToStore =
          broadcastToShape(b, loc, valToStore, getShape(elemOffsetI64));
    }
    if (opts.backend == TileBackend::Mosaic) {
      ArrayRef<int64_t> shape = getShape(elemOffsetI64);
      if (shape.empty()) {
        if (!maskOrNull) {
          Value idx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                 elemOffsetI64);
          memref::StoreOp::create(b, loc, valToStore, basePtrScalar,
                                  ValueRange{idx});
          return;
        }
        auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, maskOrNull,
                                      /*withElseRegion=*/false);
        Block *tBlk = ifOp.thenBlock();
        if (!tBlk->empty() && isa<scf::YieldOp>(tBlk->back()))
          tBlk->back().erase();
        OpBuilder tb = OpBuilder::atBlockBegin(tBlk);
        Value idx = arith::IndexCastOp::create(tb, loc, tb.getIndexType(),
                                               elemOffsetI64);
        memref::StoreOp::create(tb, loc, valToStore, basePtrScalar,
                                ValueRange{idx});
        scf::YieldOp::create(tb, loc);
        return;
      }

      unsigned bitwidth = storageElemTy.getIntOrFloatBitWidth();
      if (shape.size() == 1 && (!maskOrNull || bitwidth == 32)) {
        if (Value scalarBaseI64 =
                extractContiguous1DBaseI64(b, loc, elemOffsetI64)) {
          Value baseIdx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                     scalarBaseI64);
          tpu::VectorStoreOp::create(b, loc, valToStore, basePtrScalar,
                                     ValueRange{baseIdx}, maskOrNull,
                                     /*add=*/false);
          return;
        }
      }
      if (bitwidth == 32) {
        Value idxI32 = emitIntCast(b, loc, elemOffsetI64, b.getI32Type(),
                                   /*isSigned=*/true);
        tpu::VectorStoreIdxOp::create(b, loc, valToStore, basePtrScalar,
                                      ValueRange{idxI32}, maskOrNull,
                                      /*add=*/false);
        return;
      }
      Value c0Idx = arith::ConstantIndexOp::create(b, loc, 0);
      Value maskVec =
          maskOrNull ? maskOrNull : emitIntConst(b, loc, 1, 1, shape);
      Value flatOff = elemOffsetI64;
      Value flatMask = maskVec;
      Value flatVal = valToStore;
      if (shape.size() > 1) {
        int64_t numElems = 1;
        for (int64_t dimSz : shape)
          numElems *= dimSz;
        flatOff = vector::ShapeCastOp::create(
            b, loc, VectorType::get({numElems}, b.getI64Type()), flatOff);
        flatMask = vector::ShapeCastOp::create(
            b, loc, VectorType::get({numElems}, b.getI1Type()), flatMask);
        flatVal = vector::ShapeCastOp::create(
            b, loc, VectorType::get({numElems}, storageElemTy), flatVal);
      }
      vector::ScatterOp::create(b, loc, /*resultType=*/Type(), basePtrScalar,
                                ValueRange{c0Idx}, flatOff, flatMask, flatVal);
      return;
    }
    Value ptr = emitPtrOffset(b, loc, basePtrScalar, elemOffsetI64);
    if (opts.backend == TileBackend::CuTile) {
      auto tokTy = cuda_tile::TokenType::get(ctx);
      auto st = cuda_tile::StorePtrTkoOp::create(
          b, loc, tokTy, cuda_tile::MemoryOrderingSemantics::WEAK,
          cuda_tile::MemoryScopeAttr(), ptr, valToStore, maskOrNull,
          cuTileToken, cuda_tile::OptimizationHintsAttr());
      cuTileToken = st.getResultToken();
      return;
    }
#if ENZYME_JAX_ENABLE_TRITON
    if (maskOrNull)
      triton::StoreOp::create(b, loc, ptr, valToStore, maskOrNull);
    else
      triton::StoreOp::create(b, loc, ptr, valToStore);
#endif
  }

  FailureOr<Value> emitAtomicRMW(OpBuilder &b, Location loc,
                                 arith::AtomicRMWKind kind, Value basePtrScalar,
                                 Value elemOffsetI64, Value val,
                                 Type logicalElemTy, Value maskOrNull,
                                 Value &cuTileToken, bool resultUsed = true) {
    logicalElemTy = normalizeScalarType(logicalElemTy);
    if (logicalElemTy.isInteger(1))
      return failure();

    std::tie(elemOffsetI64, val) = matchShapes(b, loc, elemOffsetI64, val);
    if (maskOrNull) {
      std::tie(elemOffsetI64, maskOrNull) =
          matchShapes(b, loc, elemOffsetI64, maskOrNull);
      val = broadcastToShape(b, loc, val, getShape(elemOffsetI64));
    }
    if (opts.backend == TileBackend::Mosaic) {
      ArrayRef<int64_t> shape = getShape(elemOffsetI64);
      if (shape.empty()) {
        if (!maskOrNull) {
          Value idx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                 elemOffsetI64);
          auto rmwOp = memref::AtomicRMWOp::create(
              b, loc, kind, val, basePtrScalar, ValueRange{idx});
          return rmwOp.getResult();
        }
        auto ifOp = scf::IfOp::create(b, loc, TypeRange{logicalElemTy},
                                      maskOrNull, /*withElseRegion=*/true);
        Block *tBlk = ifOp.thenBlock();
        if (!tBlk->empty() && isa<scf::YieldOp>(tBlk->back()))
          tBlk->back().erase();
        {
          OpBuilder tb = OpBuilder::atBlockBegin(tBlk);
          Value idx = arith::IndexCastOp::create(tb, loc, tb.getIndexType(),
                                                 elemOffsetI64);
          auto rmwOp = memref::AtomicRMWOp::create(
              tb, loc, kind, val, basePtrScalar, ValueRange{idx});
          Value r = rmwOp.getResult();
          scf::YieldOp::create(tb, loc, ValueRange{r});
        }
        Block *eBlk = ifOp.elseBlock();
        if (!eBlk->empty() && isa<scf::YieldOp>(eBlk->back()))
          eBlk->back().erase();
        {
          OpBuilder eb = OpBuilder::atBlockBegin(eBlk);
          Value z = emitZero(eb, loc, logicalElemTy);
          scf::YieldOp::create(eb, loc, ValueRange{z});
        }
        return ifOp.getResult(0);
      }

      if (!resultUsed &&
          (kind == arith::AtomicRMWKind::addf ||
           kind == arith::AtomicRMWKind::addi) &&
          (logicalElemTy.isF32() || logicalElemTy.isInteger(32))) {
        if (shape.size() == 1) {
          if (Value scalarBaseI64 =
                  extractContiguous1DBaseI64(b, loc, elemOffsetI64)) {
            Value baseIdx = arith::IndexCastOp::create(b, loc, b.getIndexType(),
                                                       scalarBaseI64);
            tpu::VectorStoreOp::create(b, loc, val, basePtrScalar,
                                       ValueRange{baseIdx}, maskOrNull,
                                       /*add=*/true);
            return emitZero(b, loc, logicalElemTy, shape);
          }
        }
        Value idxI32 = emitIntCast(b, loc, elemOffsetI64, b.getI32Type(),
                                   /*isSigned=*/true);
        tpu::VectorStoreIdxOp::create(b, loc, val, basePtrScalar,
                                      ValueRange{idxI32}, maskOrNull,
                                      /*add=*/true);
        return emitZero(b, loc, logicalElemTy, shape);
      }
      if (!resultUsed && kind == arith::AtomicRMWKind::assign) {
        emitStore(b, loc, basePtrScalar, elemOffsetI64, val, logicalElemTy,
                  maskOrNull, cuTileToken);
        return emitZero(b, loc, logicalElemTy, shape);
      }
      return failure();
    }
    Value ptr = emitPtrOffset(b, loc, basePtrScalar, elemOffsetI64);

    if (opts.backend == TileBackend::CuTile) {
      cuda_tile::AtomicRMWMode mode;
      switch (kind) {
      case arith::AtomicRMWKind::addf:
        mode = cuda_tile::AtomicRMWMode::ADDF;
        break;
      case arith::AtomicRMWKind::addi:
        mode = cuda_tile::AtomicRMWMode::ADD;
        break;
      case arith::AtomicRMWKind::maxs:
        mode = cuda_tile::AtomicRMWMode::MAX;
        break;
      case arith::AtomicRMWKind::mins:
        mode = cuda_tile::AtomicRMWMode::MIN;
        break;
      case arith::AtomicRMWKind::maxu:
        mode = cuda_tile::AtomicRMWMode::UMAX;
        break;
      case arith::AtomicRMWKind::minu:
        mode = cuda_tile::AtomicRMWMode::UMIN;
        break;
      case arith::AtomicRMWKind::andi:
        mode = cuda_tile::AtomicRMWMode::AND;
        break;
      case arith::AtomicRMWKind::ori:
        mode = cuda_tile::AtomicRMWMode::OR;
        break;
      case arith::AtomicRMWKind::assign:
        mode = cuda_tile::AtomicRMWMode::XCHG;
        break;
      default:
        return failure();
      }
      auto tokTy = cuda_tile::TokenType::get(ctx);
      auto rmw = cuda_tile::AtomicRMWTkoOp::create(
          b, loc, val.getType(), tokTy,
          cuda_tile::MemoryOrderingSemantics::ACQ_REL,
          cuda_tile::MemoryScope::DEVICE, ptr, mode, val, maskOrNull,
          cuTileToken);
      cuTileToken = rmw.getResultToken();
      return rmw.getResult();
    }

#if ENZYME_JAX_ENABLE_TRITON
    triton::RMWOp mode;
    switch (kind) {
    case arith::AtomicRMWKind::addf:
      mode = triton::RMWOp::FADD;
      break;
    case arith::AtomicRMWKind::addi:
      mode = triton::RMWOp::ADD;
      break;
    case arith::AtomicRMWKind::maxs:
    case arith::AtomicRMWKind::maximumf:
    case arith::AtomicRMWKind::maxnumf:
      mode = triton::RMWOp::MAX;
      break;
    case arith::AtomicRMWKind::mins:
    case arith::AtomicRMWKind::minimumf:
    case arith::AtomicRMWKind::minnumf:
      mode = triton::RMWOp::MIN;
      break;
    case arith::AtomicRMWKind::maxu:
      mode = triton::RMWOp::UMAX;
      break;
    case arith::AtomicRMWKind::minu:
      mode = triton::RMWOp::UMIN;
      break;
    case arith::AtomicRMWKind::andi:
      mode = triton::RMWOp::AND;
      break;
    case arith::AtomicRMWKind::ori:
      mode = triton::RMWOp::OR;
      break;
    case arith::AtomicRMWKind::assign:
      mode = triton::RMWOp::XCHG;
      break;
    default:
      return failure();
    }
    auto rmw = triton::AtomicRMWOp::create(
        b, loc, val.getType(), mode, ptr, val, maskOrNull,
        triton::MemSemantic::ACQUIRE_RELEASE, triton::MemSyncScope::GPU);
    return rmw.getResult();
#else
    return failure();
#endif
  }

  Value mosaicBlockIdsI32[3] = {nullptr, nullptr, nullptr};
  MLIRContext *ctx;
  const GpuTileRaisingOptions &opts;
};

//===----------------------------------------------------------------------===//
// Kernel Segment Raising Context
//===----------------------------------------------------------------------===//

struct KernelSegment {
  affine::AffineParallelOp parOp = nullptr; // non-null for parallel segment
  SmallVector<Operation *> scalarOps;       // non-empty for scalar segment
};

struct LoweredKernelLaunch {
  SymbolRefAttr calleeAttr;
  int64_t gridX = 1;
  int64_t gridY = 1;
  int64_t gridZ = 1;
  SmallVector<unsigned> argIndices; // subset of candidate memref args passed
  SmallVector<unsigned> writtenPositionsInArgs; // indices into `argIndices`
};

struct SegmentRaisingState {
  TileKernelEmitter &emitter;
  const GpuTileRaisingOptions &opts;
  IRMapping valMap;
  DenseMap<Value, MemRefViewDesc> memrefViews;
  DenseMap<unsigned, Value> kernelPtrArgs; // candidate argIdx -> ptr Value
  ArrayRef<int64_t> activeTileShape;
  Value activeMask;  // null when unmasked; i1 scalar or tile otherwise
  Value cuTileToken; // threaded token in CuTile backend

  SegmentRaisingState(TileKernelEmitter &emitter,
                      const GpuTileRaisingOptions &opts)
      : emitter(emitter), opts(opts) {}

  Location cleanLoc(Location loc) const {
    return sanitizeLocation(loc, opts.strip_llvm_debuginfo);
  }

  Value combineMask(OpBuilder &b, Location loc, Value extraMask) {
    if (!activeMask)
      return extraMask;
    if (!extraMask)
      return activeMask;
    return emitter.emitAndI(b, loc, activeMask, extraMask);
  }

  Value lookupI64(OpBuilder &b, Location loc, Value origVal) {
    Value mapped = valMap.lookupOrNull(origVal);
    if (!mapped)
      return nullptr;
    Type elemTy = emitter.getElemType(mapped);
    if (!isa<IntegerType>(elemTy))
      return nullptr;
    return emitter.emitIntCast(b, loc, mapped, b.getI64Type(),
                               /*isSigned=*/true);
  }

  FailureOr<Value> evalAffineExpr(OpBuilder &b, Location loc, AffineExpr expr,
                                  unsigned numDims, ValueRange operands) {
    if (auto c = dyn_cast<AffineConstantExpr>(expr))
      return emitter.emitIntConst(b, loc, c.getValue(), 64);
    if (auto d = dyn_cast<AffineDimExpr>(expr)) {
      if (d.getPosition() >= numDims || d.getPosition() >= operands.size())
        return failure();
      Value v = lookupI64(b, loc, operands[d.getPosition()]);
      if (!v)
        return failure();
      return v;
    }
    if (auto s = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned pos = numDims + s.getPosition();
      if (pos >= operands.size())
        return failure();
      Value v = lookupI64(b, loc, operands[pos]);
      if (!v)
        return failure();
      return v;
    }
    auto bin = cast<AffineBinaryOpExpr>(expr);
    auto lhs = evalAffineExpr(b, loc, bin.getLHS(), numDims, operands);
    auto rhs = evalAffineExpr(b, loc, bin.getRHS(), numDims, operands);
    if (failed(lhs) || failed(rhs))
      return failure();
    switch (expr.getKind()) {
    case AffineExprKind::Add:
      return emitter.emitAddI(b, loc, *lhs, *rhs);
    case AffineExprKind::Mul:
      return emitter.emitMulI(b, loc, *lhs, *rhs);
    case AffineExprKind::FloorDiv:
      return emitter.emitFloorDivSI(b, loc, *lhs, *rhs);
    case AffineExprKind::CeilDiv:
      return emitter.emitCeilDivI(b, loc, *lhs, *rhs, /*isSigned=*/true);
    case AffineExprKind::Mod:
      return emitter.emitEuclideanModSI(b, loc, *lhs, *rhs);
    default:
      return failure();
    }
  }

  FailureOr<SmallVector<Value>> evalAffineMap(OpBuilder &b, Location loc,
                                              AffineMap map,
                                              ValueRange operands) {
    SmallVector<Value> results;
    for (AffineExpr e : map.getResults()) {
      auto v = evalAffineExpr(b, loc, e, map.getNumDims(), operands);
      if (failed(v))
        return failure();
      results.push_back(*v);
    }
    return results;
  }

  FailureOr<std::pair<MemRefViewDesc, Value>>
  computeLinearOffset(OpBuilder &b, Location loc, Value memrefVal,
                      ArrayRef<Value> indicesI64) {
    auto it = memrefViews.find(memrefVal);
    if (it == memrefViews.end())
      return failure();
    const MemRefViewDesc &desc = it->second;
    if (indicesI64.size() != desc.stridesI64.size())
      return failure();
    Value linear = desc.baseOffsetI64 ? desc.baseOffsetI64
                                      : emitter.emitIntConst(b, loc, 0, 64);
    for (size_t d = 0; d < indicesI64.size(); ++d) {
      Value term = emitter.emitMulI(b, loc, indicesI64[d], desc.stridesI64[d]);
      linear = emitter.emitAddI(b, loc, linear, term);
    }
    return std::make_pair(desc, linear);
  }

  // Check if any op inside `region` updates an iter_arg to a non-scalar tile,
  // or uses a tile value inside its computation that reaches the yield.
  bool regionProducesTileResult(Region &region) const {
    if (activeTileShape.empty())
      return false;
    bool hasTile = false;
    region.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (Value mapped = valMap.lookupOrNull(operand)) {
          if (!emitter.isScalar(mapped))
            hasTile = true;
        }
      }
    });
    return hasTile;
  }

  LogicalResult raiseBlockOps(OpBuilder &b, Block &block) {
    for (Operation &op : block.without_terminator()) {
      if (failed(raiseOp(b, &op)))
        return failure();
    }
    return success();
  }

  LogicalResult tryRaiseMosaicVectorReduction(
      OpBuilder &b, Location loc, Value lbI64, Value ubI64, Value stepI64,
      Value origIv, Type origIvType, Block::BlockArgListType origIterArgs,
      ArrayRef<Value> initVals, Block &origBody, ValueRange origYieldOperands,
      ResultRange origResults) {
    if (opts.backend != TileBackend::Mosaic || !activeTileShape.empty() ||
        activeMask || origIterArgs.size() != 1 || origResults.size() != 1)
      return failure();

    APInt lbAP, ubAP, stepAP;
    if (!matchPattern(lbI64, m_ConstantInt(&lbAP)) ||
        !matchPattern(ubI64, m_ConstantInt(&ubAP)) ||
        !matchPattern(stepI64, m_ConstantInt(&stepAP)))
      return failure();
    int64_t lbC = lbAP.getSExtValue();
    int64_t ubC = ubAP.getSExtValue();
    int64_t stepC = stepAP.getSExtValue();
    if (stepC <= 0 || ubC <= lbC)
      return failure();

    int64_t trips = (ubC - lbC + stepC - 1) / stepC;
    int64_t vecLen = nextPowerOfTwo(trips);
    if (trips < 2 || vecLen > std::max<int64_t>(1, opts.max_tile_elements))
      return failure();

    BlockArgument accArg = origIterArgs[0];
    if (!accArg.hasOneUse())
      return failure();
    Operation *combOp = *accArg.getUsers().begin();
    if (combOp != origYieldOperands[0].getDefiningOp() ||
        combOp->getBlock() != &origBody || combOp->getNumOperands() != 2)
      return failure();

    Value elemOrig = (combOp->getOperand(0) == accArg) ? combOp->getOperand(1)
                                                       : combOp->getOperand(0);
    if (elemOrig == accArg)
      return failure();

    tpu::ReductionKind redKind;
    bool propagateNan = true;
    if (isa<arith::AddFOp, arith::AddIOp>(combOp)) {
      redKind = tpu::ReductionKind::kSum;
    } else if (isa<arith::MaximumFOp>(combOp)) {
      redKind = tpu::ReductionKind::kMaxF;
      propagateNan = true;
    } else if (isa<arith::MaxNumFOp>(combOp)) {
      redKind = tpu::ReductionKind::kMaxF;
      propagateNan = false;
    } else if (isa<arith::MinimumFOp>(combOp)) {
      redKind = tpu::ReductionKind::kMinF;
      propagateNan = true;
    } else if (isa<arith::MinNumFOp>(combOp)) {
      redKind = tpu::ReductionKind::kMinF;
      propagateNan = false;
    } else if (isa<arith::MaxSIOp>(combOp)) {
      redKind = tpu::ReductionKind::kMaxSI;
    } else if (isa<arith::MinSIOp>(combOp)) {
      redKind = tpu::ReductionKind::kMinSI;
    } else if (isa<arith::MaxUIOp>(combOp)) {
      redKind = tpu::ReductionKind::kMaxUI;
    } else if (isa<arith::MinUIOp>(combOp)) {
      redKind = tpu::ReductionKind::kMinUI;
    } else {
      return failure();
    }

    if (trips != vecLen && redKind != tpu::ReductionKind::kSum)
      return failure();

    bool hasUnsupportedInner = false;
    origBody.walk([&](Operation *inner) {
      if (isa<affine::AffineStoreOp, memref::StoreOp, enzyme::AffineAtomicRMWOp,
              memref::AtomicRMWOp, affine::AffineForOp, scf::ForOp>(inner))
        hasUnsupportedInner = true;
    });
    if (hasUnsupportedInner)
      return failure();

    struct RollbackListener : public OpBuilder::Listener {
      SmallVector<Operation *> ops;
      void notifyOperationInserted(Operation *op,
                                   OpBuilder::InsertPoint) override {
        ops.push_back(op);
      }
    } listener;
    OpBuilder tb(b);
    tb.setListener(&listener);
    auto rollbackCreated = [&]() {
      for (Operation *created : llvm::reverse(listener.ops)) {
        created->dropAllUses();
        created->erase();
      }
    };
    SmallVector<int64_t, 1> redShape = {vecLen};
    Value laneOff = emitter.emitAxisIotaI64(tb, loc, 0, redShape);
    Value tailMask;
    if (trips != vecLen) {
      Value tripVec = emitter.emitIntConst(tb, loc, trips, 64, redShape);
      tailMask = emitter.emitCmpI(tb, loc, arith::CmpIPredicate::slt, laneOff,
                                  tripVec);
    }
    Value ivVal = laneOff;
    if (stepC != 1) {
      Value stConst = emitter.emitIntConst(tb, loc, stepC, 64);
      ivVal = emitter.emitMulI(tb, loc, ivVal, stConst);
    }
    if (lbC != 0) {
      Value lbConst = emitter.emitIntConst(tb, loc, lbC, 64);
      ivVal = emitter.emitAddI(tb, loc, ivVal, lbConst);
    }
    if (!isa<IndexType>(origIvType)) {
      ivVal =
          emitter.emitIntCast(tb, loc, ivVal, origIvType, /*isSigned=*/true);
    }

    IRMapping savedMap = valMap;
    ArrayRef<int64_t> savedShape = activeTileShape;
    Value savedMask = activeMask;
    valMap.map(origIv, ivVal);
    activeTileShape = redShape;
    activeMask = tailMask;

    bool ok = true;
    for (Operation &op : origBody.without_terminator()) {
      if (&op == combOp)
        continue;
      if (failed(raiseOp(tb, &op))) {
        ok = false;
        break;
      }
    }
    Value elemVal = ok ? valMap.lookupOrNull(elemOrig) : Value();
    valMap = std::move(savedMap);
    activeTileShape = savedShape;
    activeMask = savedMask;
    if (!elemVal) {
      rollbackCreated();
      return failure();
    }

    if (emitter.getShape(elemVal) != ArrayRef<int64_t>(redShape))
      elemVal = emitter.broadcastToShape(tb, loc, elemVal, redShape);
    Type scalarTy = emitter.getElemType(elemVal);
    if (tailMask) {
      Value z = emitter.emitZero(tb, loc, scalarTy, redShape);
      elemVal = emitter.emitSelect(tb, loc, tailMask, elemVal, z);
    }

    auto outVecTy = VectorType::get({1}, scalarTy);
    Value redVec = tpu::ReduceOp::create(tb, loc, outVecTy, elemVal,
                                         ArrayRef<int64_t>{0}, redKind);
    Value redScalar =
        vector::ExtractOp::create(tb, loc, redVec, ArrayRef<int64_t>{0});

    Value initScalar = initVals[0];
    Value combined;
    if (isa<arith::AddFOp>(combOp))
      combined = emitter.emitAddF(tb, loc, initScalar, redScalar);
    else if (isa<arith::AddIOp>(combOp))
      combined = emitter.emitAddI(tb, loc, initScalar, redScalar);
    else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(combOp))
      combined = emitter.emitMaxF(tb, loc, initScalar, redScalar, propagateNan);
    else if (isa<arith::MinimumFOp, arith::MinNumFOp>(combOp))
      combined = emitter.emitMinF(tb, loc, initScalar, redScalar, propagateNan);
    else if (isa<arith::MaxSIOp>(combOp))
      combined = emitter.emitMaxI(tb, loc, initScalar, redScalar, true);
    else if (isa<arith::MinSIOp>(combOp))
      combined = emitter.emitMinI(tb, loc, initScalar, redScalar, true);
    else if (isa<arith::MaxUIOp>(combOp))
      combined = emitter.emitMaxI(tb, loc, initScalar, redScalar, false);
    else
      combined = emitter.emitMinI(tb, loc, initScalar, redScalar, false);

    b.setInsertionPoint(tb.getBlock(), tb.getInsertionPoint());
    valMap.map(origResults[0], combined);
    return success();
  }

  LogicalResult raiseForLike(OpBuilder &b, Location loc, Value lbI64,
                             Value ubI64, Value stepI64, Value origIv,
                             Type origIvType,
                             Block::BlockArgListType origIterArgs,
                             ValueRange origInits, Block &origBody,
                             ValueRange origYieldOperands,
                             ResultRange origResults) {
    // Loop bounds must be uniform across the CTA.
    if (!emitter.isScalar(lbI64) || !emitter.isScalar(ubI64) ||
        !emitter.isScalar(stepI64))
      return failure();

    SmallVector<Value> initVals;
    bool promoteInitsToTile = !activeTileShape.empty() &&
                              regionProducesTileResult(*origBody.getParent());
    for (Value init : origInits) {
      Value mapped = valMap.lookupOrNull(init);
      if (!mapped)
        return failure();
      if (promoteInitsToTile && emitter.isScalar(mapped))
        mapped = emitter.broadcastToShape(b, loc, mapped, activeTileShape);
      initVals.push_back(mapped);
    }

    if (succeeded(tryRaiseMosaicVectorReduction(
            b, loc, lbI64, ubI64, stepI64, origIv, origIvType, origIterArgs,
            initVals, origBody, origYieldOperands, origResults))) {
      return success();
    }

    auto raiseLoopBody =
        [&](OpBuilder &bodyB, Value iv64, ValueRange bodyIterArgs,
            SmallVectorImpl<Value> &yieldedOut) -> LogicalResult {
      IRMapping savedMap = valMap;
      Value mappedIv = iv64;
      if (!isa<IndexType>(origIvType)) {
        mappedIv = emitter.emitIntCast(bodyB, loc, iv64, origIvType,
                                       /*isSigned=*/true);
      }
      valMap.map(origIv, mappedIv);
      for (auto [origArg, newArg] : llvm::zip_equal(origIterArgs, bodyIterArgs))
        valMap.map(origArg, newArg);

      if (failed(raiseBlockOps(bodyB, origBody))) {
        valMap = std::move(savedMap);
        return failure();
      }

      for (auto [idx, y] : llvm::enumerate(origYieldOperands)) {
        Value yVal = valMap.lookupOrNull(y);
        if (!yVal) {
          valMap = std::move(savedMap);
          return failure();
        }
        ArrayRef<int64_t> expectedShape = emitter.getShape(initVals[idx]);
        if (emitter.getShape(yVal) != expectedShape) {
          if (expectedShape.empty()) {
            // Body produced a tile while init was kept scalar; caller retries
            // with tile-promoted initVals.
            valMap = std::move(savedMap);
            return failure();
          }
          yVal = emitter.broadcastToShape(bodyB, loc, yVal, expectedShape);
        }
        // Under lane predication, preserve previous iteration's iter_arg on
        // inactive lanes!
        if (activeMask && !expectedShape.empty()) {
          yVal = emitter.emitSelect(bodyB, loc, activeMask, yVal,
                                    bodyIterArgs[idx]);
        } else if (activeMask && expectedShape.empty()) {
          yVal = emitter.emitSelect(bodyB, loc, activeMask, yVal,
                                    bodyIterArgs[idx]);
        }
        yieldedOut.push_back(yVal);
      }
      valMap = std::move(savedMap);
      return success();
    };

    auto attemptBuild = [&](bool forceTileInits) -> LogicalResult {
      if (forceTileInits && !activeTileShape.empty()) {
        for (Value &iv : initVals) {
          if (emitter.isScalar(iv))
            iv = emitter.broadcastToShape(b, loc, iv, activeTileShape);
        }
      }
      if (opts.backend == TileBackend::CuTile) {
        SmallVector<Value> fullInits(initVals.begin(), initVals.end());
        fullInits.push_back(cuTileToken);
        auto forOp = cuda_tile::ForOp::create(
            b, loc, lbI64, ubI64, stepI64, fullInits,
            /*bodyBuilderFn=*/nullptr, /*unsignedCmp=*/false);
        Block *bodyBlk = forOp.getBody();
        OpBuilder bodyB = OpBuilder::atBlockBegin(bodyBlk);
        Value iv64 = bodyBlk->getArgument(0);
        auto allIterArgs = bodyBlk->getArguments().drop_front(1);
        Value prevTok = cuTileToken;
        cuTileToken = allIterArgs.back();
        SmallVector<Value> yielded;
        if (failed(raiseLoopBody(bodyB, iv64, allIterArgs.drop_back(1),
                                 yielded))) {
          cuTileToken = prevTok;
          forOp.erase();
          return failure();
        }
        yielded.push_back(cuTileToken);
        cuda_tile::ContinueOp::create(bodyB, loc, yielded);
        cuTileToken = forOp.getResult(initVals.size());
        for (size_t i = 0; i < origResults.size(); ++i)
          valMap.map(origResults[i], forOp.getResult(i));
        return success();
      }

      auto forOp = scf::ForOp::create(b, loc, lbI64, ubI64, stepI64, initVals);
      Block *bodyBlk = forOp.getBody();
      if (!bodyBlk->empty() && isa<scf::YieldOp>(bodyBlk->back()))
        bodyBlk->back().erase();
      OpBuilder bodyB = OpBuilder::atBlockBegin(bodyBlk);
      Value iv64 = bodyBlk->getArgument(0);
      auto bodyIterArgs = bodyBlk->getArguments().drop_front(1);
      SmallVector<Value> yielded;
      if (failed(raiseLoopBody(bodyB, iv64, bodyIterArgs, yielded))) {
        forOp.erase();
        return failure();
      }
      scf::YieldOp::create(bodyB, loc, yielded);
      for (size_t i = 0; i < origResults.size(); ++i)
        valMap.map(origResults[i], forOp.getResult(i));
      return success();
    };

    if (succeeded(attemptBuild(/*forceTileInits=*/false)))
      return success();
    if (!activeTileShape.empty() && !origInits.empty())
      return attemptBuild(/*forceTileInits=*/true);
    return failure();
  }

  LogicalResult raiseIfLike(OpBuilder &b, Location loc, Value cond,
                            Block *thenBlock, ValueRange thenYields,
                            Block *elseBlock, ValueRange elseYields,
                            ResultRange origResults) {
    // Uniform scalar condition with no lane divergence: emit structured if.
    if (emitter.isScalar(cond) &&
        (!activeMask || emitter.isScalar(activeMask))) {
      // Raise both branches cleanly; in CuTile we always thread cuTileToken.
      if (opts.backend == TileBackend::CuTile) {
        // Determine result types by raising into temporary blocks or matching
        // shapes if needed. First inspect whether any result becomes a tile.
        SmallVector<Type> resTypes;
        bool useTileResults =
            !activeTileShape.empty() &&
            ((thenBlock && regionProducesTileResult(*thenBlock->getParent())) ||
             (elseBlock && regionProducesTileResult(*elseBlock->getParent())));
        for (Value r : origResults) {
          Type scalarTy = normalizeScalarType(r.getType());
          resTypes.push_back(emitter.makeType(
              useTileResults ? activeTileShape : ArrayRef<int64_t>{},
              scalarTy));
        }
        resTypes.push_back(cuda_tile::TokenType::get(b.getContext()));
        auto ifOp = cuda_tile::IfOp::create(b, loc, resTypes, cond);
        Block &tBlk = ifOp.getThenRegion().emplaceBlock();
        Block &eBlk = ifOp.getElseRegion().emplaceBlock();

        Value entryTok = cuTileToken;
        {
          IRMapping saved = valMap;
          cuTileToken = entryTok;
          OpBuilder tb = OpBuilder::atBlockBegin(&tBlk);
          if (thenBlock && failed(raiseBlockOps(tb, *thenBlock)))
            return failure();
          SmallVector<Value> yVals;
          for (auto [idx, y] : llvm::enumerate(thenYields)) {
            Value v = valMap.lookupOrNull(y);
            if (!v)
              return failure();
            auto targetShape =
                cast<cuda_tile::TileType>(resTypes[idx]).getShape();
            if (emitter.getShape(v) != targetShape) {
              if (targetShape.empty())
                return failure();
              v = emitter.broadcastToShape(tb, loc, v, targetShape);
            }
            yVals.push_back(v);
          }
          yVals.push_back(cuTileToken);
          cuda_tile::YieldOp::create(tb, loc, yVals);
          valMap = std::move(saved);
        }
        {
          IRMapping saved = valMap;
          cuTileToken = entryTok;
          OpBuilder eb = OpBuilder::atBlockBegin(&eBlk);
          if (elseBlock && failed(raiseBlockOps(eb, *elseBlock)))
            return failure();
          SmallVector<Value> yVals;
          for (auto [idx, y] : llvm::enumerate(elseYields)) {
            Value v = valMap.lookupOrNull(y);
            if (!v)
              return failure();
            auto targetShape =
                cast<cuda_tile::TileType>(resTypes[idx]).getShape();
            if (emitter.getShape(v) != targetShape) {
              if (targetShape.empty())
                return failure();
              v = emitter.broadcastToShape(eb, loc, v, targetShape);
            }
            yVals.push_back(v);
          }
          yVals.push_back(cuTileToken);
          cuda_tile::YieldOp::create(eb, loc, yVals);
          valMap = std::move(saved);
        }
        cuTileToken = ifOp.getResult(origResults.size());
        for (size_t i = 0; i < origResults.size(); ++i)
          valMap.map(origResults[i], ifOp.getResult(i));
        return success();
      }

      // Triton structured scf.if
      bool useTileResults =
          !activeTileShape.empty() &&
          ((thenBlock && regionProducesTileResult(*thenBlock->getParent())) ||
           (elseBlock && regionProducesTileResult(*elseBlock->getParent())));
      SmallVector<Type> resTypes;
      for (Value r : origResults) {
        Type scalarTy = normalizeScalarType(r.getType());
        resTypes.push_back(emitter.makeType(
            useTileResults ? activeTileShape : ArrayRef<int64_t>{}, scalarTy));
      }
      bool hasElse = (elseBlock != nullptr) || !resTypes.empty();
      auto ifOp = scf::IfOp::create(b, loc, resTypes, cond, hasElse);
      Block *tBlk = ifOp.thenBlock();
      if (!tBlk->empty() && isa<scf::YieldOp>(tBlk->back()))
        tBlk->back().erase();
      {
        IRMapping saved = valMap;
        OpBuilder tb = OpBuilder::atBlockBegin(tBlk);
        if (thenBlock && failed(raiseBlockOps(tb, *thenBlock)))
          return failure();
        SmallVector<Value> yVals;
        for (auto [idx, y] : llvm::enumerate(thenYields)) {
          Value v = valMap.lookupOrNull(y);
          if (!v)
            return failure();
          ArrayRef<int64_t> targetShape =
              useTileResults ? activeTileShape : ArrayRef<int64_t>{};
          if (emitter.getShape(v) != targetShape) {
            if (targetShape.empty())
              return failure();
            v = emitter.broadcastToShape(tb, loc, v, targetShape);
          }
          yVals.push_back(v);
        }
        scf::YieldOp::create(tb, loc, yVals);
        valMap = std::move(saved);
      }
      if (hasElse) {
        Block *eBlk = ifOp.elseBlock();
        if (!eBlk->empty() && isa<scf::YieldOp>(eBlk->back()))
          eBlk->back().erase();
        IRMapping saved = valMap;
        OpBuilder eb = OpBuilder::atBlockBegin(eBlk);
        if (elseBlock && failed(raiseBlockOps(eb, *elseBlock)))
          return failure();
        SmallVector<Value> yVals;
        for (auto [idx, y] : llvm::enumerate(elseYields)) {
          Value v = valMap.lookupOrNull(y);
          if (!v)
            return failure();
          ArrayRef<int64_t> targetShape =
              useTileResults ? activeTileShape : ArrayRef<int64_t>{};
          if (emitter.getShape(v) != targetShape) {
            if (targetShape.empty())
              return failure();
            v = emitter.broadcastToShape(eb, loc, v, targetShape);
          }
          yVals.push_back(v);
        }
        scf::YieldOp::create(eb, loc, yVals);
        valMap = std::move(saved);
      }
      for (size_t i = 0; i < origResults.size(); ++i)
        valMap.map(origResults[i], ifOp.getResult(i));
      return success();
    }

    // Tile-divergent condition (or activeMask is already a tile): predicate
    // `then` and `else` regions inline with masked memory ops and `select` on
    // yielded SSA values!
    Value savedMask = activeMask;
    SmallVector<Value> thenVals, elseVals;
    if (thenBlock) {
      IRMapping savedMap = valMap;
      activeMask = combineMask(b, loc, cond);
      if (failed(raiseBlockOps(b, *thenBlock))) {
        activeMask = savedMask;
        return failure();
      }
      for (Value y : thenYields) {
        Value v = valMap.lookupOrNull(y);
        if (!v) {
          activeMask = savedMask;
          return failure();
        }
        thenVals.push_back(v);
      }
      valMap = std::move(savedMap);
    }
    if (elseBlock) {
      IRMapping savedMap = valMap;
      Value notCond = emitter.emitNotI1(b, loc, cond);
      activeMask = combineMask(b, loc, notCond);
      if (failed(raiseBlockOps(b, *elseBlock))) {
        activeMask = savedMask;
        return failure();
      }
      for (Value y : elseYields) {
        Value v = valMap.lookupOrNull(y);
        if (!v) {
          activeMask = savedMask;
          return failure();
        }
        elseVals.push_back(v);
      }
      valMap = std::move(savedMap);
    }
    activeMask = savedMask;
    for (size_t i = 0; i < origResults.size(); ++i) {
      Value merged = emitter.emitSelect(b, loc, cond, thenVals[i], elseVals[i]);
      valMap.map(origResults[i], merged);
    }
    return success();
  }

  LogicalResult raiseOp(OpBuilder &b, Operation *op) {
    Location loc = cleanLoc(op->getLoc());

    // Benign hints / barriers / scope declarations
    if (isa<LLVM::AssumeOp, LLVM::NoAliasScopeDeclOp, memref::DeallocOp>(op))
      return success();
    if (isa<enzymexla::BarrierOp>(op)) {
      if (opts.backend == TileBackend::CuTile && cuTileToken) {
        cuTileToken =
            cuda_tile::MemoryFenceAliasTkoOp::create(b, loc, cuTileToken);
      }
      return success();
    }

    // Constants & undef/poison
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      Type resTy = cst.getType();
      if (!isSupportedScalarElemType(resTy, opts.backend))
        return failure();
      if (isa<IndexType>(resTy)) {
        int64_t val = cast<IntegerAttr>(cst.getValue()).getInt();
        valMap.map(cst.getResult(), emitter.emitIntConst(b, loc, val, 64));
        return success();
      }
      if (auto it = dyn_cast<IntegerType>(resTy)) {
        int64_t val =
            cast<IntegerAttr>(cst.getValue()).getValue().getSExtValue();
        valMap.map(cst.getResult(),
                   emitter.emitIntConst(b, loc, val, it.getWidth()));
        return success();
      }
      if (auto ft = dyn_cast<FloatType>(resTy)) {
        APFloat val = cast<FloatAttr>(cst.getValue()).getValue();
        valMap.map(cst.getResult(), emitter.emitFloatConst(b, loc, val, ft));
        return success();
      }
      return failure();
    }

    if (isa<ub::PoisonOp, LLVM::UndefOp, LLVM::PoisonOp, LLVM::ZeroOp>(op)) {
      Type resTy = op->getResult(0).getType();
      if (!isSupportedScalarElemType(resTy, opts.backend))
        return failure();
      valMap.map(op->getResult(0), emitter.emitZero(b, loc, resTy));
      return success();
    }

    // MemRef view/cast/dim operations
    if (auto dimOp = dyn_cast<memref::DimOp>(op)) {
      auto constIdx = dimOp.getConstantIndex();
      auto mt = dyn_cast<MemRefType>(dimOp.getSource().getType());
      if (!constIdx || !mt || *constIdx < 0 ||
          *constIdx >= static_cast<int64_t>(mt.getRank()) ||
          mt.isDynamicDim(*constIdx))
        return failure();
      valMap.map(dimOp.getResult(),
                 emitter.emitIntConst(b, loc, mt.getDimSize(*constIdx), 64));
      return success();
    }
    if (auto castOp = dyn_cast<memref::CastOp>(op)) {
      auto it = memrefViews.find(castOp.getSource());
      if (it == memrefViews.end())
        return failure();
      memrefViews[castOp.getDest()] = it->second;
      return success();
    }
    if (auto msc = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      auto it = memrefViews.find(msc.getSource());
      if (it == memrefViews.end())
        return failure();
      memrefViews[msc.getDest()] = it->second;
      return success();
    }
    if (auto sub = dyn_cast<memref::SubViewOp>(op)) {
      auto it = memrefViews.find(sub.getSource());
      if (it == memrefViews.end())
        return failure();
      MemRefViewDesc parent = it->second;
      SmallVector<OpFoldResult> mixedOffsets = sub.getMixedOffsets();
      SmallVector<OpFoldResult> mixedStrides = sub.getMixedStrides();
      if (mixedOffsets.size() != parent.stridesI64.size())
        return failure();

      Value newBaseOffset = parent.baseOffsetI64
                                ? parent.baseOffsetI64
                                : emitter.emitIntConst(b, loc, 0, 64);
      SmallVector<Value, 4> fullStrides;
      for (size_t d = 0; d < mixedOffsets.size(); ++d) {
        Value offI64;
        if (auto attr = dyn_cast<Attribute>(mixedOffsets[d]))
          offI64 = emitter.emitIntConst(b, loc,
                                        cast<IntegerAttr>(attr).getInt(), 64);
        else
          offI64 = lookupI64(b, loc, cast<Value>(mixedOffsets[d]));
        if (!offI64)
          return failure();
        Value stI64;
        if (auto attr = dyn_cast<Attribute>(mixedStrides[d]))
          stI64 = emitter.emitIntConst(b, loc, cast<IntegerAttr>(attr).getInt(),
                                       64);
        else
          stI64 = lookupI64(b, loc, cast<Value>(mixedStrides[d]));
        if (!stI64)
          return failure();
        newBaseOffset = emitter.emitAddI(
            b, loc, newBaseOffset,
            emitter.emitMulI(b, loc, offI64, parent.stridesI64[d]));
        fullStrides.push_back(
            emitter.emitMulI(b, loc, parent.stridesI64[d], stI64));
      }
      llvm::SmallBitVector dropped = sub.getDroppedDims();
      MemRefViewDesc child = parent;
      child.baseOffsetI64 = newBaseOffset;
      child.stridesI64.clear();
      for (size_t d = 0; d < fullStrides.size(); ++d) {
        if (!dropped.test(d))
          child.stridesI64.push_back(fullStrides[d]);
      }
      memrefViews[sub.getResult()] = std::move(child);
      return success();
    }
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(op)) {
      auto it = memrefViews.find(rc.getSource());
      if (it == memrefViews.end())
        return failure();
      MemRefViewDesc child = it->second;
      OpFoldResult offOfr = rc.getMixedOffsets().front();
      Value offI64;
      if (auto attr = dyn_cast<Attribute>(offOfr))
        offI64 =
            emitter.emitIntConst(b, loc, cast<IntegerAttr>(attr).getInt(), 64);
      else
        offI64 = lookupI64(b, loc, cast<Value>(offOfr));
      if (!offI64)
        return failure();
      child.baseOffsetI64 = offI64;
      child.stridesI64.clear();
      for (OpFoldResult stOfr : rc.getMixedStrides()) {
        Value stI64;
        if (auto attr = dyn_cast<Attribute>(stOfr))
          stI64 = emitter.emitIntConst(b, loc, cast<IntegerAttr>(attr).getInt(),
                                       64);
        else
          stI64 = lookupI64(b, loc, cast<Value>(stOfr));
        if (!stI64)
          return failure();
        child.stridesI64.push_back(stI64);
      }
      memrefViews[rc.getResult()] = std::move(child);
      return success();
    }

    // affine.apply / affine.min / affine.max
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
      affine::AffineValueMap avm(apply.getAffineMap(), apply.getMapOperands());
      avm.composeSimplifyAndCanonicalize();
      auto res = evalAffineMap(b, loc, avm.getAffineMap(), avm.getOperands());
      if (failed(res) || res->size() != 1)
        return failure();
      valMap.map(apply.getResult(), (*res)[0]);
      return success();
    }
    if (auto amin = dyn_cast<affine::AffineMinOp>(op)) {
      auto res = evalAffineMap(b, loc, amin.getAffineMap(), amin.getOperands());
      if (failed(res) || res->empty())
        return failure();
      Value acc = (*res)[0];
      for (size_t i = 1; i < res->size(); ++i)
        acc = emitter.emitMinI(b, loc, acc, (*res)[i], /*isSigned=*/true);
      valMap.map(amin.getResult(), acc);
      return success();
    }
    if (auto amax = dyn_cast<affine::AffineMaxOp>(op)) {
      auto res = evalAffineMap(b, loc, amax.getAffineMap(), amax.getOperands());
      if (failed(res) || res->empty())
        return failure();
      Value acc = (*res)[0];
      for (size_t i = 1; i < res->size(); ++i)
        acc = emitter.emitMaxI(b, loc, acc, (*res)[i], /*isSigned=*/true);
      valMap.map(amax.getResult(), acc);
      return success();
    }

    // Memory loads
    if (auto ld = dyn_cast<affine::AffineLoadOp>(op)) {
      affine::AffineValueMap avm(ld.getAffineMap(), ld.getMapOperands());
      avm.composeSimplifyAndCanonicalize();
      auto idxVals =
          evalAffineMap(b, loc, avm.getAffineMap(), avm.getOperands());
      if (failed(idxVals))
        return failure();
      auto addr = computeLinearOffset(b, loc, ld.getMemRef(), *idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      Value loaded = emitter.emitLoad(b, loc, basePtr, addr->second,
                                      addr->first.logicalElemType, activeMask,
                                      cuTileToken);
      valMap.map(ld.getResult(), loaded);
      return success();
    }
    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      SmallVector<Value> idxVals;
      for (Value idx : ld.getIndices()) {
        Value v = lookupI64(b, loc, idx);
        if (!v)
          return failure();
        idxVals.push_back(v);
      }
      auto addr = computeLinearOffset(b, loc, ld.getMemRef(), idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      Value loaded = emitter.emitLoad(b, loc, basePtr, addr->second,
                                      addr->first.logicalElemType, activeMask,
                                      cuTileToken);
      valMap.map(ld.getResult(), loaded);
      return success();
    }

    // Memory stores
    if (auto st = dyn_cast<affine::AffineStoreOp>(op)) {
      Value val = valMap.lookupOrNull(st.getValueToStore());
      if (!val)
        return failure();
      affine::AffineValueMap avm(st.getAffineMap(), st.getMapOperands());
      avm.composeSimplifyAndCanonicalize();
      auto idxVals =
          evalAffineMap(b, loc, avm.getAffineMap(), avm.getOperands());
      if (failed(idxVals))
        return failure();
      auto addr = computeLinearOffset(b, loc, st.getMemRef(), *idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      emitter.emitStore(b, loc, basePtr, addr->second, val,
                        addr->first.logicalElemType, activeMask, cuTileToken);
      return success();
    }
    if (auto st = dyn_cast<memref::StoreOp>(op)) {
      Value val = valMap.lookupOrNull(st.getValueToStore());
      if (!val)
        return failure();
      SmallVector<Value> idxVals;
      for (Value idx : st.getIndices()) {
        Value v = lookupI64(b, loc, idx);
        if (!v)
          return failure();
        idxVals.push_back(v);
      }
      auto addr = computeLinearOffset(b, loc, st.getMemRef(), idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      emitter.emitStore(b, loc, basePtr, addr->second, val,
                        addr->first.logicalElemType, activeMask, cuTileToken);
      return success();
    }

    // Atomic RMW ops
    if (auto rmw = dyn_cast<enzyme::AffineAtomicRMWOp>(op)) {
      Value val = valMap.lookupOrNull(rmw.getValue());
      if (!val)
        return failure();
      affine::AffineValueMap avm(rmw.getMap(), rmw.getIndices());
      avm.composeSimplifyAndCanonicalize();
      auto idxVals =
          evalAffineMap(b, loc, avm.getAffineMap(), avm.getOperands());
      if (failed(idxVals))
        return failure();
      auto addr = computeLinearOffset(b, loc, rmw.getMemref(), *idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      auto oldVal =
          emitter.emitAtomicRMW(b, loc, rmw.getKind(), basePtr, addr->second,
                                val, addr->first.logicalElemType, activeMask,
                                cuTileToken, !rmw.getResult().use_empty());
      if (failed(oldVal))
        return failure();
      valMap.map(rmw.getResult(), *oldVal);
      return success();
    }
    if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op)) {
      Value val = valMap.lookupOrNull(rmw.getValue());
      if (!val)
        return failure();
      SmallVector<Value> idxVals;
      for (Value idx : rmw.getIndices()) {
        Value v = lookupI64(b, loc, idx);
        if (!v)
          return failure();
        idxVals.push_back(v);
      }
      auto addr = computeLinearOffset(b, loc, rmw.getMemref(), idxVals);
      if (failed(addr))
        return failure();
      Value basePtr = kernelPtrArgs.lookup(addr->first.argIndex);
      if (!basePtr)
        return failure();
      auto oldVal =
          emitter.emitAtomicRMW(b, loc, rmw.getKind(), basePtr, addr->second,
                                val, addr->first.logicalElemType, activeMask,
                                cuTileToken, !rmw.getResult().use_empty());
      if (failed(oldVal))
        return failure();
      valMap.map(rmw.getResult(), *oldVal);
      return success();
    }

    // Loops: affine.for and scf.for
    if (auto afor = dyn_cast<affine::AffineForOp>(op)) {
      auto lbs = evalAffineMap(b, loc, afor.getLowerBoundMap(),
                               afor.getLowerBoundOperands());
      auto ubs = evalAffineMap(b, loc, afor.getUpperBoundMap(),
                               afor.getUpperBoundOperands());
      if (failed(lbs) || failed(ubs) || lbs->empty() || ubs->empty())
        return failure();
      Value lb = (*lbs)[0];
      for (size_t i = 1; i < lbs->size(); ++i)
        lb = emitter.emitMaxI(b, loc, lb, (*lbs)[i], /*isSigned=*/true);
      Value ub = (*ubs)[0];
      for (size_t i = 1; i < ubs->size(); ++i)
        ub = emitter.emitMinI(b, loc, ub, (*ubs)[i], /*isSigned=*/true);
      Value step = emitter.emitIntConst(b, loc, afor.getStepAsInt(), 64);
      auto yieldOp =
          cast<affine::AffineYieldOp>(afor.getBody()->getTerminator());
      return raiseForLike(b, loc, lb, ub, step, afor.getInductionVar(),
                          afor.getInductionVar().getType(),
                          afor.getRegionIterArgs(), afor.getInits(),
                          *afor.getBody(), yieldOp.getOperands(),
                          afor.getResults());
    }
    if (auto sfor = dyn_cast<scf::ForOp>(op)) {
      Value lb = lookupI64(b, loc, sfor.getLowerBound());
      Value ub = lookupI64(b, loc, sfor.getUpperBound());
      Value step = lookupI64(b, loc, sfor.getStep());
      if (!lb || !ub || !step)
        return failure();
      auto yieldOp = cast<scf::YieldOp>(sfor.getBody()->getTerminator());
      return raiseForLike(b, loc, lb, ub, step, sfor.getInductionVar(),
                          sfor.getInductionVar().getType(),
                          sfor.getRegionIterArgs(), sfor.getInitArgs(),
                          *sfor.getBody(), yieldOp.getResults(),
                          sfor.getResults());
    }

    // Conditionals: affine.if and scf.if
    if (auto aif = dyn_cast<affine::AffineIfOp>(op)) {
      IntegerSet set = aif.getIntegerSet();
      Value cond;
      Value zero = emitter.emitIntConst(b, loc, 0, 64);
      for (unsigned i = 0; i < set.getNumConstraints(); ++i) {
        auto lhs = evalAffineExpr(b, loc, set.getConstraint(i),
                                  set.getNumDims(), aif.getOperands());
        if (failed(lhs))
          return failure();
        Value c = emitter.emitCmpI(b, loc,
                                   set.isEq(i) ? arith::CmpIPredicate::eq
                                               : arith::CmpIPredicate::sge,
                                   *lhs, zero);
        cond = cond ? emitter.emitAndI(b, loc, cond, c) : c;
      }
      if (!cond)
        cond = emitter.emitIntConst(b, loc, 1, 1);
      Block *thenBlk = aif.getThenBlock();
      ValueRange thenYields =
          cast<affine::AffineYieldOp>(thenBlk->getTerminator()).getOperands();
      Block *elseBlk = aif.hasElse() ? aif.getElseBlock() : nullptr;
      ValueRange elseYields =
          elseBlk ? cast<affine::AffineYieldOp>(elseBlk->getTerminator())
                        .getOperands()
                  : ValueRange{};
      return raiseIfLike(b, loc, cond, thenBlk, thenYields, elseBlk, elseYields,
                         aif.getResults());
    }
    if (auto sif = dyn_cast<scf::IfOp>(op)) {
      Value cond = valMap.lookupOrNull(sif.getCondition());
      if (!cond)
        return failure();
      Block *thenBlk = sif.thenBlock();
      ValueRange thenYields = sif.thenYield().getResults();
      Block *elseBlk = sif.getElseRegion().empty() ? nullptr : sif.elseBlock();
      ValueRange elseYields =
          elseBlk ? sif.elseYield().getResults() : ValueRange{};
      return raiseIfLike(b, loc, cond, thenBlk, thenYields, elseBlk, elseYields,
                         sif.getResults());
    }

    // Integer binary arith ops
    auto lookup2 = [&](Operation *o) -> std::optional<std::pair<Value, Value>> {
      Value l = valMap.lookupOrNull(o->getOperand(0));
      Value r = valMap.lookupOrNull(o->getOperand(1));
      if (!l || !r)
        return std::nullopt;
      return std::make_pair(l, r);
    };

    if (isa<arith::AddIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitAddI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::SubIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitSubI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::MulIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMulI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::DivSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitDivI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::DivUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitDivI(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::FloorDivSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitFloorDivSI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::CeilDivSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitCeilDivI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::CeilDivUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitCeilDivI(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::RemSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitRemI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::RemUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitRemI(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::AndIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitAndI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::OrIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitOrI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::XOrIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitXorI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::ShLIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitShLI(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::ShRSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitShRI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::ShRUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitShRI(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::MaxSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMaxI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::MinSIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMinI(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::MaxUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMaxI(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::MinUIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMinI(b, loc, p->first, p->second, false));
      return success();
    }
    if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(cmpi.getResult(), emitter.emitCmpI(b, loc, cmpi.getPredicate(),
                                                    p->first, p->second));
      return success();
    }

    // Float binary/unary arith ops
    if (isa<arith::AddFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitAddF(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::SubFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitSubF(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::MulFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMulF(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::DivFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitDivF(b, loc, p->first, p->second));
      return success();
    }
    if (isa<arith::RemFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitRemF(b, loc, p->first, p->second));
      return success();
    }
    if (auto negf = dyn_cast<arith::NegFOp>(op)) {
      Value v = valMap.lookupOrNull(negf.getOperand());
      if (!v)
        return failure();
      valMap.map(negf.getResult(), emitter.emitNegF(b, loc, v));
      return success();
    }
    if (isa<arith::MaximumFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMaxF(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::MaxNumFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMaxF(b, loc, p->first, p->second, false));
      return success();
    }
    if (isa<arith::MinimumFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMinF(b, loc, p->first, p->second, true));
      return success();
    }
    if (isa<arith::MinNumFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitMinF(b, loc, p->first, p->second, false));
      return success();
    }
    if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      valMap.map(cmpf.getResult(), emitter.emitCmpF(b, loc, cmpf.getPredicate(),
                                                    p->first, p->second));
      return success();
    }
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      Value c = valMap.lookupOrNull(sel.getCondition());
      Value t = valMap.lookupOrNull(sel.getTrueValue());
      Value f = valMap.lookupOrNull(sel.getFalseValue());
      if (!c || !t || !f)
        return failure();
      valMap.map(sel.getResult(), emitter.emitSelect(b, loc, c, t, f));
      return success();
    }

    // Cast ops
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      Type dstTy = op->getResult(0).getType();
      if (!v || !isSupportedScalarElemType(dstTy, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitIntCast(b, loc, v, dstTy, /*isSigned=*/true));
      return success();
    }
    if (isa<arith::IndexCastUIOp, arith::ExtUIOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      Type dstTy = op->getResult(0).getType();
      if (!v || !isSupportedScalarElemType(dstTy, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitIntCast(b, loc, v, dstTy, /*isSigned=*/false));
      return success();
    }
    if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      auto dstFt = dyn_cast<FloatType>(op->getResult(0).getType());
      if (!v || !dstFt || !isSupportedScalarElemType(dstFt, opts.backend))
        return failure();
      valMap.map(op->getResult(0), emitter.emitFloatCast(b, loc, v, dstFt));
      return success();
    }
    if (isa<arith::SIToFPOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      auto dstFt = dyn_cast<FloatType>(op->getResult(0).getType());
      if (!v || !dstFt || !isSupportedScalarElemType(dstFt, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitIToF(b, loc, v, dstFt, /*isSigned=*/true));
      return success();
    }
    if (isa<arith::UIToFPOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      auto dstFt = dyn_cast<FloatType>(op->getResult(0).getType());
      if (!v || !dstFt || !isSupportedScalarElemType(dstFt, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitIToF(b, loc, v, dstFt, /*isSigned=*/false));
      return success();
    }
    if (isa<arith::FPToSIOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      auto dstIt = dyn_cast<IntegerType>(
          normalizeScalarType(op->getResult(0).getType()));
      if (!v || !dstIt || !isSupportedScalarElemType(dstIt, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitFToI(b, loc, v, dstIt, /*isSigned=*/true));
      return success();
    }
    if (isa<arith::FPToUIOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      auto dstIt = dyn_cast<IntegerType>(
          normalizeScalarType(op->getResult(0).getType()));
      if (!v || !dstIt || !isSupportedScalarElemType(dstIt, opts.backend))
        return failure();
      valMap.map(op->getResult(0),
                 emitter.emitFToI(b, loc, v, dstIt, /*isSigned=*/false));
      return success();
    }
    if (isa<arith::BitcastOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      Type dstTy = op->getResult(0).getType();
      if (!v || !isSupportedScalarElemType(dstTy, opts.backend))
        return failure();
      valMap.map(op->getResult(0), emitter.emitBitcast(b, loc, v, dstTy));
      return success();
    }

    // Math ops + enzymexla.fmuladd
    if (isa<math::FmaOp, enzymexla::FMulAddOp>(op)) {
      Value a = valMap.lookupOrNull(op->getOperand(0));
      Value c = valMap.lookupOrNull(op->getOperand(1));
      Value d = valMap.lookupOrNull(op->getOperand(2));
      if (!a || !c || !d)
        return failure();
      valMap.map(op->getResult(0), emitter.emitFma(b, loc, a, c, d));
      return success();
    }
    if (isa<math::AbsFOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::AbsFOp::create(b, loc, v))
                      : Value(math::AbsFOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::AbsIOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::AbsIOp::create(b, loc, v))
                      : Value(math::AbsIOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::CeilOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::CeilOp::create(b, loc, v))
                      : Value(math::CeilOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::FloorOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::FloorOp::create(b, loc, v))
                      : Value(math::FloorOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::CosOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::CosOp::create(b, loc, v))
                      : Value(math::CosOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::SinOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::SinOp::create(b, loc, v))
                      : Value(math::SinOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::ExpOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::ExpOp::create(b, loc, v))
                      : Value(math::ExpOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::Exp2Op>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::Exp2Op::create(
                            b, loc, v, /*flush_to_zero=*/false))
                      : Value(math::Exp2Op::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::LogOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::LogOp::create(b, loc, v))
                      : Value(math::LogOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::Log2Op>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::Log2Op::create(b, loc, v))
                      : Value(math::Log2Op::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::SqrtOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res = (opts.backend == TileBackend::CuTile)
                      ? Value(cuda_tile::SqrtOp::create(
                            b, loc, v, cuda_tile::RoundingMode::NEAREST_EVEN,
                            /*flush_to_zero=*/false))
                      : Value(math::SqrtOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::RsqrtOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      Value res =
          (opts.backend == TileBackend::CuTile)
              ? Value(cuda_tile::RsqrtOp::create(b, loc, v,
                                                 /*flush_to_zero=*/false))
              : Value(math::RsqrtOp::create(b, loc, v));
      valMap.map(op->getResult(0), res);
      return success();
    }
    if (isa<math::TanOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0), cuda_tile::TanOp::create(b, loc, v));
      } else {
        Value s = math::SinOp::create(b, loc, v);
        Value c = math::CosOp::create(b, loc, v);
        valMap.map(op->getResult(0), arith::DivFOp::create(b, loc, s, c));
      }
      return success();
    }
    if (isa<math::TanhOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0), cuda_tile::TanHOp::create(b, loc, v));
      } else {
        auto ft = cast<FloatType>(emitter.getElemType(v));
        ArrayRef<int64_t> sh = emitter.getShape(v);
        Value negTwo = emitter.emitFloatConst(b, loc, APFloat(-2.0), ft, sh);
        Value one = emitter.emitFloatConst(b, loc, APFloat(1.0), ft, sh);
        Value zero = emitter.emitFloatConst(b, loc, APFloat(0.0), ft, sh);
        Value absV = math::AbsFOp::create(b, loc, v);
        Value scaled = arith::MulFOp::create(b, loc, negTwo, absV);
        Value e = math::ExpOp::create(b, loc, scaled);
        Value num = arith::SubFOp::create(b, loc, one, e);
        Value den = arith::AddFOp::create(b, loc, one, e);
        Value mag = arith::DivFOp::create(b, loc, num, den);
        Value isNeg =
            arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, v, zero);
        Value negMag = arith::NegFOp::create(b, loc, mag);
        valMap.map(op->getResult(0),
                   arith::SelectOp::create(b, loc, isNeg, negMag, mag));
      }
      return success();
    }
    if (isa<math::SinhOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0), cuda_tile::SinHOp::create(b, loc, v));
      } else {
        auto ft = cast<FloatType>(emitter.getElemType(v));
        Value half = emitter.emitFloatConst(b, loc, APFloat(0.5), ft,
                                            emitter.getShape(v));
        Value ep = math::ExpOp::create(b, loc, v);
        Value em =
            math::ExpOp::create(b, loc, arith::NegFOp::create(b, loc, v));
        valMap.map(op->getResult(0),
                   arith::MulFOp::create(
                       b, loc, half, arith::SubFOp::create(b, loc, ep, em)));
      }
      return success();
    }
    if (isa<math::CoshOp>(op)) {
      Value v = valMap.lookupOrNull(op->getOperand(0));
      if (!v)
        return failure();
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0), cuda_tile::CosHOp::create(b, loc, v));
      } else {
        auto ft = cast<FloatType>(emitter.getElemType(v));
        Value half = emitter.emitFloatConst(b, loc, APFloat(0.5), ft,
                                            emitter.getShape(v));
        Value ep = math::ExpOp::create(b, loc, v);
        Value em =
            math::ExpOp::create(b, loc, arith::NegFOp::create(b, loc, v));
        valMap.map(op->getResult(0),
                   arith::MulFOp::create(
                       b, loc, half, arith::AddFOp::create(b, loc, ep, em)));
      }
      return success();
    }
    if (isa<math::PowFOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      auto [lhs, rhs] = emitter.matchShapes(b, loc, p->first, p->second);
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0),
                   cuda_tile::FPowFOp::create(b, loc, lhs, rhs));
      } else {
        Value lg = math::Log2Op::create(b, loc, lhs);
        valMap.map(op->getResult(0),
                   math::Exp2Op::create(
                       b, loc, arith::MulFOp::create(b, loc, rhs, lg)));
      }
      return success();
    }
    if (isa<math::FPowIOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      auto [lhs, rhs] = emitter.matchShapes(b, loc, p->first, p->second);
      if (opts.backend == TileBackend::CuTile) {
        valMap.map(op->getResult(0),
                   cuda_tile::FPowIOp::create(b, loc, lhs, rhs));
      } else {
        auto ft = cast<FloatType>(emitter.getElemType(lhs));
        Value rhsF = emitter.emitIToF(b, loc, rhs, ft, /*isSigned=*/true);
        Value lg = math::Log2Op::create(b, loc, lhs);
        valMap.map(op->getResult(0),
                   math::Exp2Op::create(
                       b, loc, arith::MulFOp::create(b, loc, rhsF, lg)));
      }
      return success();
    }
    if (isa<math::CopySignOp>(op)) {
      auto p = lookup2(op);
      if (!p)
        return failure();
      auto [mag, sgn] = emitter.matchShapes(b, loc, p->first, p->second);
      auto ft = cast<FloatType>(emitter.getElemType(mag));
      Value absMag = (opts.backend == TileBackend::CuTile)
                         ? Value(cuda_tile::AbsFOp::create(b, loc, mag))
                         : Value(math::AbsFOp::create(b, loc, mag));
      Value zero = emitter.emitFloatConst(b, loc, APFloat(0.0), ft,
                                          emitter.getShape(mag));
      Value isNeg =
          emitter.emitCmpF(b, loc, arith::CmpFPredicate::OLT, sgn, zero);
      Value negMag = emitter.emitNegF(b, loc, absMag);
      valMap.map(op->getResult(0),
                 emitter.emitSelect(b, loc, isNeg, negMag, absMag));
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Outlining & Raising a Candidate Region into Kernel Launch(es)
//===----------------------------------------------------------------------===//

static std::optional<int64_t> evalConstAffineBound(AffineMap map,
                                                   ValueRange operands) {
  if (map.getNumResults() != 1)
    return std::nullopt;
  if (map.isSingleConstant())
    return map.getSingleConstantResult();
  SmallVector<Attribute> constOps;
  for (Value v : operands) {
    Attribute a;
    if (!matchPattern(v, m_Constant(&a)))
      return std::nullopt;
    constOps.push_back(a);
  }
  SmallVector<Attribute> folded;
  if (failed(map.constantFold(constOps, folded)) || folded.size() != 1)
    return std::nullopt;
  if (auto ia = dyn_cast_or_null<IntegerAttr>(folded[0]))
    return ia.getValue().getSExtValue();
  return std::nullopt;
}

// Collect pure prefix definitions from `regionBody` that are used inside the
// ops of `segment` so each outlined kernel can rematerialize needed constants
// or scalar index expressions.
static void collectRequiredPrefixDefs(Block &regionBody,
                                      ArrayRef<Operation *> segmentOps,
                                      SmallVectorImpl<Operation *> &prefixOut) {
  DenseSet<Operation *> inSegment(segmentOps.begin(), segmentOps.end());
  DenseSet<Operation *> needed;
  SmallVector<Value> worklist;

  auto enqueueExternalOperands = [&](Operation *root) {
    root->walk([&](Operation *inner) {
      for (Value v : inner->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (!def)
          continue;
        if (def->getBlock() == &regionBody && !inSegment.contains(def))
          worklist.push_back(v);
      }
    });
  };
  for (Operation *op : segmentOps)
    enqueueExternalOperands(op);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != &regionBody || inSegment.contains(def))
      continue;
    if (!needed.insert(def).second)
      continue;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  for (Operation &op : regionBody.without_terminator()) {
    if (needed.contains(&op))
      prefixOut.push_back(&op);
  }
}

static LogicalResult raiseCandidateRegionToModuleAndLaunches(
    ModuleOp parentMod, StringRef baseName, Region &origRegion,
    const GpuTileRaisingOptions &opts,
    SmallVectorImpl<LoweredKernelLaunch> &launches,
    SmallVectorImpl<Operation *> &createdExtModsOut) {
  createdExtModsOut.clear();
  auto cleanupExtMods = [&]() {
    for (Operation *m : createdExtModsOut)
      m->erase();
    createdExtModsOut.clear();
  };
#if !ENZYME_JAX_ENABLE_TRITON
  if (opts.backend == TileBackend::Triton)
    return failure();
#endif

  if (!origRegion.hasOneBlock())
    return failure();

  // Work on an isolated clone of `origRegion` so preprocessing never mutates
  // the caller's IR if raising later bails out.
  Region workRegion;
  IRMapping cloneMap;
  origRegion.cloneInto(&workRegion, cloneMap);

  stripMemorySpaceCastsInRegion(workRegion);
  inlineTrivialScopes(workRegion);
  promoteLocalScratchAllocas(workRegion);
  coalesceNestedAffineParallels(workRegion);

  Block &body = workRegion.front();
  unsigned numCandArgs = body.getNumArguments();

  // Validate all candidate block arguments are MemRefTypes with supported
  // element types and static strides/offset (dynamic outer dim is supported
  // when inner strides are static).
  SmallVector<MemRefType> argMemRefTypes;
  for (BlockArgument arg : body.getArguments()) {
    auto mt = dyn_cast<MemRefType>(arg.getType());
    if (!mt || !isSupportedScalarElemType(mt.getElementType(), opts.backend))
      return failure();
    int64_t offset = 0;
    SmallVector<int64_t> strides;
    if (failed(mt.getStridesAndOffset(strides, offset)) ||
        ShapedType::isDynamic(offset) ||
        llvm::any_of(strides, ShapedType::isDynamic))
      return failure();
    argMemRefTypes.push_back(mt);
  }

  // Partition `body` into sequential KernelSegments.
  SmallVector<KernelSegment> segments;
  for (Operation &op : body.without_terminator()) {
    if (isa<LLVM::AssumeOp, LLVM::NoAliasScopeDeclOp, memref::DeallocOp>(op))
      continue;
    if (auto par = dyn_cast<affine::AffineParallelOp>(&op)) {
      if (par.getNumResults() != 0)
        return failure();
      KernelSegment seg;
      seg.parOp = par;
      segments.push_back(std::move(seg));
      continue;
    }
    // Pure scalar/view setup ops that only feed later segments do not need
    // their own GPU launch unless no parallel segment follows or they perform
    // memory writes.
    bool hasWrite = false;
    op.walk([&](Operation *inner) {
      if (isa<affine::AffineStoreOp, memref::StoreOp, enzyme::AffineAtomicRMWOp,
              memref::AtomicRMWOp>(inner))
        hasWrite = true;
    });
    if (!hasWrite &&
        (isPure(&op) ||
         isa<memref::CastOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::MemorySpaceCastOp, affine::AffineLoadOp, memref::LoadOp>(
             op))) {
      // Check if all users are in `body`; if unused or pure/read-only scalar
      // setup, `collectRequiredPrefixDefs` will clone it into each segment!
      bool usedByTerminator = false;
      for (Operation *u : op.getUsers())
        if (u == body.getTerminator())
          usedByTerminator = true;
      if (!usedByTerminator)
        continue;
    }
    if (!segments.empty() && !segments.back().parOp) {
      segments.back().scalarOps.push_back(&op);
    } else {
      KernelSegment seg;
      seg.scalarOps.push_back(&op);
      segments.push_back(std::move(seg));
    }
  }

  // If the region is completely empty / pure no-op, no launches are needed.
  if (segments.empty())
    return success();

  MLIRContext *ctx = parentMod.getContext();
  SymbolTable parentSymTab(parentMod);
  OpBuilder topBuilder(ctx);
  Location modLoc =
      sanitizeLocation(parentMod.getLoc(), opts.strip_llvm_debuginfo);

  // Sanitize base symbol prefix. Each emitted kernel launch gets its own
  // `enzymexla_tt_ext.module` / `enzymexla_ct_ext.module` so downstream
  // serialization (`--lower-triton`) sees a single kernel per module.
  std::string CleanBase = baseName.str();
  for (char &ch : CleanBase) {
    if (ch == '$')
      ch = '_';
  }

  TileKernelEmitter emitter(ctx, opts);

  for (auto [segIdx, seg] : llvm::enumerate(segments)) {
    SmallVector<Operation *> segOps;
    if (seg.parOp)
      segOps.push_back(seg.parOp);
    else
      segOps.append(seg.scalarOps.begin(), seg.scalarOps.end());

    SmallVector<Operation *> prefixDefs;
    collectRequiredPrefixDefs(body, segOps, prefixDefs);

    // Determine which candidate memref arguments are referenced and which are
    // written in `prefixDefs + segOps`.
    llvm::SmallBitVector usedArgBits(numCandArgs, false);
    llvm::SmallBitVector writtenArgBits(numCandArgs, false);

    DenseMap<Value, unsigned> aliasToArg;
    for (unsigned i = 0; i < numCandArgs; ++i)
      aliasToArg[body.getArgument(i)] = i;

    auto scanOpForArgs = [&](Operation *root) {
      root->walk([&](Operation *op) {
        if (auto c = dyn_cast<memref::CastOp>(op)) {
          if (aliasToArg.count(c.getSource()))
            aliasToArg[c.getDest()] = aliasToArg[c.getSource()];
        } else if (auto c = dyn_cast<memref::MemorySpaceCastOp>(op)) {
          if (aliasToArg.count(c.getSource()))
            aliasToArg[c.getDest()] = aliasToArg[c.getSource()];
        } else if (auto s = dyn_cast<memref::SubViewOp>(op)) {
          if (aliasToArg.count(s.getSource()))
            aliasToArg[s.getResult()] = aliasToArg[s.getSource()];
        } else if (auto r = dyn_cast<memref::ReinterpretCastOp>(op)) {
          if (aliasToArg.count(r.getSource()))
            aliasToArg[r.getResult()] = aliasToArg[r.getSource()];
        }

        for (Value operand : op->getOperands()) {
          auto it = aliasToArg.find(operand);
          if (it != aliasToArg.end())
            usedArgBits.set(it->second);
        }
        Value writtenMemref;
        if (auto st = dyn_cast<affine::AffineStoreOp>(op))
          writtenMemref = st.getMemRef();
        else if (auto st = dyn_cast<memref::StoreOp>(op))
          writtenMemref = st.getMemRef();
        else if (auto rmw = dyn_cast<enzyme::AffineAtomicRMWOp>(op))
          writtenMemref = rmw.getMemref();
        else if (auto rmw = dyn_cast<memref::AtomicRMWOp>(op))
          writtenMemref = rmw.getMemref();
        if (writtenMemref && aliasToArg.count(writtenMemref))
          writtenArgBits.set(aliasToArg[writtenMemref]);
      });
    };

    for (Operation *p : prefixDefs)
      scanOpForArgs(p);
    for (Operation *s : segOps)
      scanOpForArgs(s);

    // If a segment performs zero global writes, skip it.
    if (writtenArgBits.none())
      continue;

    LoweredKernelLaunch launch;
    for (unsigned i = 0; i < numCandArgs; ++i) {
      if (usedArgBits.test(i)) {
        if (writtenArgBits.test(i))
          launch.writtenPositionsInArgs.push_back(launch.argIndices.size());
        launch.argIndices.push_back(i);
      }
    }

    std::string segBase = segments.size() == 1
                              ? CleanBase
                              : CleanBase + "_" + std::to_string(segIdx);
    StringRef extSuffix = (opts.backend == TileBackend::CuTile)   ? "_ct"
                          : (opts.backend == TileBackend::Mosaic) ? "_mosaic"
                                                                  : "_tt";
    std::string extModBase = segBase + extSuffix.str();
    std::string extModName = extModBase;
    for (unsigned suffix = 0; parentSymTab.lookup(extModName) != nullptr;
         ++suffix)
      extModName = extModBase + "_" + std::to_string(suffix);
    std::string innerModName = segBase + "_inner";
    std::string kernelName = segBase + "_kernel";

    Block *kernelContainerBlock = nullptr;
    Operation *segExtMod = nullptr;
    if (opts.backend == TileBackend::CuTile) {
      topBuilder.setInsertionPointToStart(parentMod.getBody());
      auto ctExtMod = enzymexla::cuda_tile_ext::CudaTileModuleOp::create(
          topBuilder, modLoc, StringRef(extModName),
          /*sym_visibility=*/StringAttr());
      ctExtMod.getBodyRegion().emplaceBlock();
      OpBuilder innerB =
          OpBuilder::atBlockBegin(&ctExtMod.getBodyRegion().front());
      auto ctMod = cuda_tile::ModuleOp::create(innerB, modLoc, innerModName);
      kernelContainerBlock = &ctMod.getBody().front();
      segExtMod = ctExtMod;
      createdExtModsOut.push_back(ctExtMod);
    } else if (opts.backend == TileBackend::Mosaic) {
      topBuilder.setInsertionPointToStart(parentMod.getBody());
      auto mosExtMod = enzymexla::mosaic_ext::MosaicModuleOp::create(
          topBuilder, modLoc, StringRef(extModName),
          /*sym_visibility=*/StringAttr());
      mosExtMod.getBodyRegion().emplaceBlock();
      OpBuilder innerB =
          OpBuilder::atBlockBegin(&mosExtMod.getBodyRegion().front());
      auto innerMod = ModuleOp::create(innerB, modLoc, innerModName);
      kernelContainerBlock = innerMod.getBody();
      segExtMod = mosExtMod;
      createdExtModsOut.push_back(mosExtMod);
    } else {
#if ENZYME_JAX_ENABLE_TRITON
      topBuilder.setInsertionPointToStart(parentMod.getBody());
      auto ttExtMod = enzymexla::triton_ext::TritonModuleOp::create(
          topBuilder, modLoc, StringRef(extModName),
          /*sym_visibility=*/StringAttr());
      ttExtMod.getBodyRegion().emplaceBlock();
      OpBuilder innerB =
          OpBuilder::atBlockBegin(&ttExtMod.getBodyRegion().front());
      auto innerMod = ModuleOp::create(innerB, modLoc, innerModName);
      innerMod->setAttr("enzymexla.num_warps",
                        innerB.getI32IntegerAttr(opts.num_warps));
      innerMod->setAttr("enzymexla.num_stages",
                        innerB.getI32IntegerAttr(opts.num_stages));
      kernelContainerBlock = innerMod.getBody();
      segExtMod = ttExtMod;
      createdExtModsOut.push_back(ttExtMod);
#endif
    }

    launch.calleeAttr =
        SymbolRefAttr::get(ctx, extModName,
                           {FlatSymbolRefAttr::get(ctx, innerModName),
                            FlatSymbolRefAttr::get(ctx, kernelName)});

    // Build kernel function signature.
    SmallVector<Type> kernelArgTypes;
    if (opts.backend == TileBackend::Mosaic) {
      for (unsigned axis = 0; axis < 3; ++axis)
        kernelArgTypes.push_back(IntegerType::get(ctx, 32));
      auto vmemAttr = tpu::MemorySpaceAttr::get(ctx, tpu::MemorySpace::kVmem);
      for (unsigned candIdx : launch.argIndices) {
        MemRefType mt = argMemRefTypes[candIdx];
        Type storageTy =
            getStoragePointeeType(mt.getElementType(), opts.backend);
        MemRefType vmemTy;
        if (!mt.getLayout().isIdentity()) {
          vmemTy = MemRefType::get({ShapedType::kDynamic}, storageTy,
                                   AffineMap(), vmemAttr);
        } else if (mt.getRank() == 0) {
          vmemTy = MemRefType::get({1}, storageTy, AffineMap(), vmemAttr);
        } else {
          vmemTy =
              MemRefType::get(mt.getShape(), storageTy, AffineMap(), vmemAttr);
        }
        kernelArgTypes.push_back(vmemTy);
      }
    } else {
      for (unsigned candIdx : launch.argIndices) {
        Type storageTy = getStoragePointeeType(
            argMemRefTypes[candIdx].getElementType(), opts.backend);
        Type ptrElemTy = emitter.makePointerElemType(storageTy);
        kernelArgTypes.push_back(emitter.makeType({}, ptrElemTy));
      }
    }
    auto kernelFnType = FunctionType::get(ctx, kernelArgTypes, {});

    OpBuilder kb = OpBuilder::atBlockEnd(kernelContainerBlock);
    Location segLoc =
        sanitizeLocation(segOps.front()->getLoc(), opts.strip_llvm_debuginfo);
    Block *entryBlock = nullptr;
    func::FuncOp mosaicFuncOp = nullptr;
    if (opts.backend == TileBackend::CuTile) {
      auto entryOp = cuda_tile::EntryOp::create(
          kb, segLoc, kernelName, kernelFnType, ArrayAttr(), ArrayAttr(),
          cuda_tile::OptimizationHintsAttr());
      entryBlock = &entryOp.getBody().emplaceBlock();
      for (Type argTy : kernelArgTypes)
        entryBlock->addArgument(argTy, segLoc);
    } else if (opts.backend == TileBackend::Mosaic) {
      mosaicFuncOp = func::FuncOp::create(kb, segLoc, kernelName, kernelFnType);
      mosaicFuncOp.setVisibility(SymbolTable::Visibility::Public);
      entryBlock = mosaicFuncOp.addEntryBlock();
      for (unsigned axis = 0; axis < 3; ++axis)
        emitter.mosaicBlockIdsI32[axis] = entryBlock->getArgument(axis);
    } else {
#if ENZYME_JAX_ENABLE_TRITON
      auto funcOp =
          triton::FuncOp::create(kb, segLoc, kernelName, kernelFnType);
      funcOp.setVisibility(SymbolTable::Visibility::Public);
      entryBlock = funcOp.addEntryBlock();
#endif
    }

    OpBuilder b = OpBuilder::atBlockBegin(entryBlock);
    SegmentRaisingState state(emitter, opts);
    if (opts.backend == TileBackend::CuTile)
      state.cuTileToken = cuda_tile::MakeTokenOp::create(b, segLoc);

    // Populate MemRefViewDesc and pointer/buffer arguments.
    unsigned argBaseOffset = (opts.backend == TileBackend::Mosaic) ? 3 : 0;
    for (auto [pos, candIdx] : llvm::enumerate(launch.argIndices)) {
      Value rawArg = entryBlock->getArgument(argBaseOffset + pos);
      if (opts.backend == TileBackend::Mosaic) {
        auto vmemTy = cast<MemRefType>(rawArg.getType());
        if (vmemTy.getRank() >= 2) {
          ReassociationIndices group;
          for (int64_t d = 0; d < vmemTy.getRank(); ++d)
            group.push_back(d);
          rawArg = memref::CollapseShapeOp::create(
              b, segLoc, rawArg, ArrayRef<ReassociationIndices>{group});
        }
      }
      state.kernelPtrArgs[candIdx] = rawArg;
      MemRefType mt = argMemRefTypes[candIdx];
      int64_t offset = 0;
      SmallVector<int64_t> strides;
      (void)mt.getStridesAndOffset(strides, offset);

      MemRefViewDesc desc;
      desc.argIndex = candIdx;
      desc.baseType = mt;
      desc.logicalElemType = normalizeScalarType(mt.getElementType());
      desc.storageElemType =
          getStoragePointeeType(mt.getElementType(), opts.backend);
      if (offset != 0)
        desc.baseOffsetI64 = emitter.emitIntConst(b, segLoc, offset, 64);
      for (int64_t st : strides)
        desc.stridesI64.push_back(emitter.emitIntConst(b, segLoc, st, 64));
      state.memrefViews[body.getArgument(candIdx)] = std::move(desc);
    }

    // Raise required scalar/load prefix definitions at top of kernel.
    for (Operation *prefOp : prefixDefs) {
      if (failed(state.raiseOp(b, prefOp))) {
        cleanupExtMods();
        return failure();
      }
    }

    if (!seg.parOp) {
      // Scalar (1, 1, 1) segment.
      launch.gridX = 1;
      launch.gridY = 1;
      launch.gridZ = 1;
      for (Operation *sop : seg.scalarOps) {
        if (failed(state.raiseOp(b, sop))) {
          cleanupExtMods();
          return failure();
        }
      }
    } else {
      affine::AffineParallelOp par = seg.parOp;
      unsigned numDims = par.getNumDims();
      if (numDims == 0) {
        launch.gridX = 1;
        launch.gridY = 1;
        launch.gridZ = 1;
        if (failed(state.raiseBlockOps(b, *par.getBody()))) {
          cleanupExtMods();
          return failure();
        }
      } else {
        SmallVector<int64_t> steps = par.getSteps();
        for (int64_t s : steps) {
          if (s <= 0) {
            cleanupExtMods();
            return failure();
          }
        }

        // Check if all bounds are compile-time constants.
        SmallVector<int64_t> constLbs, constUbs, constTrips;
        bool allStatic = true;
        for (unsigned d = 0; d < numDims; ++d) {
          AffineMap lbm = par.getLowerBoundMap(d);
          AffineMap ubm = par.getUpperBoundMap(d);
          auto lb = evalConstAffineBound(lbm, par.getLowerBoundsOperands());
          auto ub = evalConstAffineBound(ubm, par.getUpperBoundsOperands());
          if (!lb || !ub) {
            allStatic = false;
            break;
          }
          int64_t trips =
              (*ub <= *lb) ? 0 : ((*ub - *lb + steps[d] - 1) / steps[d]);
          constLbs.push_back(*lb);
          constUbs.push_back(*ub);
          constTrips.push_back(trips);
        }

        bool hasBarrier = false;
        par.getBody()->walk([&](enzymexla::BarrierOp) { hasBarrier = true; });

        if (allStatic) {
          // If any axis has 0 iterations, the entire parallel loop is a no-op!
          if (llvm::any_of(constTrips, [](int64_t t) { return t <= 0; })) {
            // Erase the just-created ext module and skip this launch.
            createdExtModsOut.pop_back();
            if (segExtMod)
              segExtMod->erase();
            continue;
          }

          SmallVector<int64_t> tileShape(numDims, 1);
          SmallVector<int64_t> gridShape(numDims, 1);
          int64_t remainingBudget =
              std::max<int64_t>(1, opts.max_tile_elements);
          int64_t maxAxisTile = std::max<int64_t>(1, opts.default_tile_size);

          for (int d = static_cast<int>(numDims) - 1; d >= 0; --d) {
            int64_t trips = constTrips[d];
            if (hasBarrier && d == static_cast<int>(numDims) - 1) {
              // Keep synchronized innermost axis within a single CTA.
              int64_t reqTile = nextPowerOfTwo(trips);
              if (reqTile > remainingBudget) {
                cleanupExtMods();
                return failure();
              }
              tileShape[d] = reqTile;
              gridShape[d] = 1;
              remainingBudget = std::max<int64_t>(1, remainingBudget / reqTile);
              continue;
            }
            int64_t cap =
                prevPowerOfTwo(std::min(maxAxisTile, remainingBudget));
            int64_t want = nextPowerOfTwo(trips);
            int64_t ts = std::max<int64_t>(1, std::min(want, cap));
            tileShape[d] = ts;
            gridShape[d] = (trips + ts - 1) / ts;
            remainingBudget = std::max<int64_t>(1, remainingBudget / ts);
          }

          // Map `gridShape` onto (gridX, gridY, gridZ). Put innermost axis on
          // gridX for contiguous memory coalescing across neighboring CTAs, or
          // pack excess axes (>3) onto gridZ.
          SmallVector<Value> axisBlockIdI64(numDims);
          if (numDims == 1) {
            launch.gridX = gridShape[0];
            launch.gridY = 1;
            launch.gridZ = 1;
            axisBlockIdI64[0] = emitter.emitBlockIdI64(b, segLoc, 0);
          } else if (numDims == 2) {
            launch.gridX = gridShape[1];
            launch.gridY = gridShape[0];
            launch.gridZ = 1;
            axisBlockIdI64[1] = emitter.emitBlockIdI64(b, segLoc, 0);
            axisBlockIdI64[0] = emitter.emitBlockIdI64(b, segLoc, 1);
          } else {
            launch.gridX = gridShape[numDims - 1];
            launch.gridY = gridShape[numDims - 2];
            int64_t packedZ = 1;
            for (int d = 0; d <= static_cast<int>(numDims) - 3; ++d)
              packedZ *= gridShape[d];
            launch.gridZ = packedZ;
            axisBlockIdI64[numDims - 1] = emitter.emitBlockIdI64(b, segLoc, 0);
            axisBlockIdI64[numDims - 2] = emitter.emitBlockIdI64(b, segLoc, 1);
            Value remZ = emitter.emitBlockIdI64(b, segLoc, 2);
            for (int d = static_cast<int>(numDims) - 3; d >= 0; --d) {
              if (d == 0) {
                axisBlockIdI64[0] = remZ;
              } else {
                Value gConst =
                    emitter.emitIntConst(b, segLoc, gridShape[d], 64);
                axisBlockIdI64[d] = emitter.emitRemI(b, segLoc, remZ, gConst,
                                                     /*isSigned=*/false);
                remZ = emitter.emitDivI(b, segLoc, remZ, gConst,
                                        /*isSigned=*/false);
              }
            }
          }

          Value boundsMask;
          for (unsigned d = 0; d < numDims; ++d) {
            Value tsConst = emitter.emitIntConst(b, segLoc, tileShape[d], 64);
            Value blockBase =
                emitter.emitMulI(b, segLoc, axisBlockIdI64[d], tsConst);
            Value laneOff = emitter.emitAxisIotaI64(b, segLoc, d, tileShape);
            Value logicalIdx = emitter.emitAddI(b, segLoc, blockBase, laneOff);

            if (gridShape[d] * tileShape[d] != constTrips[d]) {
              Value tripConst =
                  emitter.emitIntConst(b, segLoc, constTrips[d], 64, tileShape);
              Value inBounds = emitter.emitCmpI(
                  b, segLoc, arith::CmpIPredicate::slt, logicalIdx, tripConst);
              boundsMask =
                  boundsMask ? emitter.emitAndI(b, segLoc, boundsMask, inBounds)
                             : inBounds;
            }

            Value ivVal = logicalIdx;
            if (steps[d] != 1) {
              Value stepConst = emitter.emitIntConst(b, segLoc, steps[d], 64);
              ivVal = emitter.emitMulI(b, segLoc, ivVal, stepConst);
            }
            if (constLbs[d] != 0) {
              Value lbConst = emitter.emitIntConst(b, segLoc, constLbs[d], 64);
              ivVal = emitter.emitAddI(b, segLoc, ivVal, lbConst);
            }
            state.valMap.map(par.getBody()->getArgument(d), ivVal);
          }

          state.activeTileShape = tileShape;
          state.activeMask = boundsMask;
          if (failed(state.raiseBlockOps(b, *par.getBody()))) {
            cleanupExtMods();
            return failure();
          }
        } else {
          // Dynamic bounds: emit static 1D grid (`default_grid_size`) + 1D tile
          // with a grid-stride loop over `totalTrips`.
          if (hasBarrier) {
            cleanupExtMods();
            return failure();
          }
          SmallVector<Value> dynLbs(numDims), dynTrips(numDims);
          Value zero64 = emitter.emitIntConst(b, segLoc, 0, 64);
          Value totalTrips = emitter.emitIntConst(b, segLoc, 1, 64);
          for (unsigned d = 0; d < numDims; ++d) {
            auto lbs = state.evalAffineMap(b, segLoc, par.getLowerBoundMap(d),
                                           par.getLowerBoundsOperands());
            auto ubs = state.evalAffineMap(b, segLoc, par.getUpperBoundMap(d),
                                           par.getUpperBoundsOperands());
            if (failed(lbs) || failed(ubs) || lbs->empty() || ubs->empty()) {
              cleanupExtMods();
              return failure();
            }
            Value lb = (*lbs)[0];
            for (size_t i = 1; i < lbs->size(); ++i)
              lb = emitter.emitMaxI(b, segLoc, lb, (*lbs)[i], true);
            Value ub = (*ubs)[0];
            for (size_t i = 1; i < ubs->size(); ++i)
              ub = emitter.emitMinI(b, segLoc, ub, (*ubs)[i], true);
            if (!emitter.isScalar(lb) || !emitter.isScalar(ub)) {
              cleanupExtMods();
              return failure();
            }
            Value span = emitter.emitMaxI(
                b, segLoc, emitter.emitSubI(b, segLoc, ub, lb), zero64, true);
            Value stepVal = emitter.emitIntConst(b, segLoc, steps[d], 64);
            Value trips = emitter.emitCeilDivI(b, segLoc, span, stepVal, true);
            dynLbs[d] = lb;
            dynTrips[d] = trips;
            totalTrips = emitter.emitMulI(b, segLoc, totalTrips, trips);
          }

          int64_t tileSize = prevPowerOfTwo(
              std::min(std::max<int64_t>(1, opts.default_tile_size),
                       std::max<int64_t>(1, opts.max_tile_elements)));
          int64_t gridSize = std::max<int64_t>(1, opts.default_grid_size);
          launch.gridX = gridSize;
          launch.gridY = 1;
          launch.gridZ = 1;
          SmallVector<int64_t> tileShape = {tileSize};

          Value pid = emitter.emitBlockIdI64(b, segLoc, 0);
          Value tsConst = emitter.emitIntConst(b, segLoc, tileSize, 64);
          Value startBase = emitter.emitMulI(b, segLoc, pid, tsConst);
          Value stride =
              emitter.emitIntConst(b, segLoc, gridSize * tileSize, 64);

          auto populateGridStrideBody = [&](OpBuilder &loopB,
                                            Value baseVal) -> LogicalResult {
            Value laneOff =
                emitter.emitAxisIotaI64(loopB, segLoc, 0, tileShape);
            Value linearIdx = emitter.emitAddI(loopB, segLoc, baseVal, laneOff);
            Value inBounds =
                emitter.emitCmpI(loopB, segLoc, arith::CmpIPredicate::slt,
                                 linearIdx, totalTrips);

            Value rem = linearIdx;
            for (int d = static_cast<int>(numDims) - 1; d >= 0; --d) {
              Value coord;
              if (d == 0) {
                coord = rem;
              } else {
                coord =
                    emitter.emitRemI(loopB, segLoc, rem, dynTrips[d], false);
                rem = emitter.emitDivI(loopB, segLoc, rem, dynTrips[d], false);
              }
              if (steps[d] != 1) {
                Value stVal = emitter.emitIntConst(loopB, segLoc, steps[d], 64);
                coord = emitter.emitMulI(loopB, segLoc, coord, stVal);
              }
              coord = emitter.emitAddI(loopB, segLoc, coord, dynLbs[d]);
              state.valMap.map(par.getBody()->getArgument(d), coord);
            }
            state.activeTileShape = tileShape;
            state.activeMask = inBounds;
            return state.raiseBlockOps(loopB, *par.getBody());
          };

          if (opts.backend == TileBackend::CuTile) {
            SmallVector<Value> initTok = {state.cuTileToken};
            auto gFor = cuda_tile::ForOp::create(
                b, segLoc, startBase, totalTrips, stride, initTok,
                /*bodyBuilderFn=*/nullptr, /*unsignedCmp=*/false);
            Block *loopBlk = gFor.getBody();
            OpBuilder loopB = OpBuilder::atBlockBegin(loopBlk);
            state.cuTileToken = loopBlk->getArgument(1);
            if (failed(
                    populateGridStrideBody(loopB, loopBlk->getArgument(0)))) {
              cleanupExtMods();
              return failure();
            }
            cuda_tile::ContinueOp::create(loopB, segLoc,
                                          ValueRange{state.cuTileToken});
            state.cuTileToken = gFor.getResult(0);
          } else {
            auto gFor =
                scf::ForOp::create(b, segLoc, startBase, totalTrips, stride);
            Block *loopBlk = gFor.getBody();
            if (!loopBlk->empty() && isa<scf::YieldOp>(loopBlk->back()))
              loopBlk->back().erase();
            OpBuilder loopB = OpBuilder::atBlockBegin(loopBlk);
            if (failed(
                    populateGridStrideBody(loopB, loopBlk->getArgument(0)))) {
              cleanupExtMods();
              return failure();
            }
            scf::YieldOp::create(loopB, segLoc);
          }
        }
      }
    }

    // Finalize Mosaic TPU metadata attributes and emit kernel terminator.
    if (opts.backend == TileBackend::Mosaic && mosaicFuncOp) {
      mosaicFuncOp->setAttr("tpu.core_type",
                            tpu::CoreTypeAttr::get(ctx, tpu::CoreType::kTc));
      mosaicFuncOp->setAttr(
          "iteration_bounds",
          b.getDenseI64ArrayAttr({launch.gridX, launch.gridY, launch.gridZ}));
      auto parSem = tpu::DimensionSemanticsAttr::get(
          ctx, tpu::DimensionSemantics::parallel);
      mosaicFuncOp->setAttr("dimension_semantics",
                            b.getArrayAttr({parSem, parSem, parSem}));
    }
    b.setInsertionPointToEnd(entryBlock);
    if (opts.backend == TileBackend::CuTile) {
      cuda_tile::ReturnOp::create(b, segLoc);
    } else if (opts.backend == TileBackend::Mosaic) {
      func::ReturnOp::create(b, segLoc);
    } else {
#if ENZYME_JAX_ENABLE_TRITON
      triton::ReturnOp::create(b, segLoc);
#endif
    }

    launches.push_back(std::move(launch));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Host Launch Call Sequence Emission
//===----------------------------------------------------------------------===//

static void
emitHostLaunchSequence(OpBuilder &b, Location loc, TileBackend backend,
                       ArrayRef<LoweredKernelLaunch> launches,
                       MutableArrayRef<Value> curTensors,
                       ArrayAttr singleLaunchOperandLayouts = nullptr,
                       UnitAttr singleLaunchSideEffectFree = nullptr) {
  MLIRContext *ctx = b.getContext();
  auto scalarI64TenTy = RankedTensorType::get({}, b.getI64Type());
  DenseMap<int64_t, Value> gridConstCache;
  auto getGridConst = [&](int64_t val) -> Value {
    auto it = gridConstCache.find(val);
    if (it != gridConstCache.end())
      return it->second;
    Value c = stablehlo::ConstantOp::create(
        b, loc, scalarI64TenTy,
        DenseElementsAttr::get(scalarI64TenTy, b.getI64IntegerAttr(val)));
    gridConstCache[val] = c;
    return c;
  };

  Value c1 = getGridConst(1);
  bool preserveAttrs = (launches.size() == 1);

  for (const LoweredKernelLaunch &launch : launches) {
    Value gx = getGridConst(launch.gridX);
    Value gy = getGridConst(launch.gridY);
    Value gz = getGridConst(launch.gridZ);

    SmallVector<Value> callInputs;
    for (unsigned candIdx : launch.argIndices)
      callInputs.push_back(curTensors[candIdx]);

    SmallVector<Type> resTypes;
    for (unsigned pos : launch.writtenPositionsInArgs)
      resTypes.push_back(callInputs[pos].getType());

    ArrayAttr aliases =
        buildOutputOperandAliases(ctx, launch.writtenPositionsInArgs);
    ArrayAttr layouts =
        preserveAttrs ? singleLaunchOperandLayouts : ArrayAttr();
    UnitAttr seFree = preserveAttrs ? singleLaunchSideEffectFree : UnitAttr();

    Operation *callOp = nullptr;
    if (backend == TileBackend::CuTile) {
      callOp = enzymexla::cuda_tile_ext::CudaTileCallOp::create(
          b, loc, resTypes, launch.calleeAttr, gx, gy, gz, c1, c1, c1,
          callInputs, /*backend_config=*/StringAttr(), layouts,
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(), aliases,
          seFree);
    } else if (backend == TileBackend::Mosaic) {
      callOp = enzymexla::mosaic_ext::MosaicCallOp::create(
          b, loc, resTypes, launch.calleeAttr, gx, gy, gz, c1, c1, c1,
          callInputs, /*backend_config=*/StringAttr(), layouts,
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(), aliases,
          seFree);
    } else {
#if ENZYME_JAX_ENABLE_TRITON
      callOp = enzymexla::triton_ext::TritonCallOp::create(
          b, loc, resTypes, launch.calleeAttr, gx, gy, gz, c1, c1, c1,
          callInputs, /*backend_config=*/StringAttr(), layouts,
          /*arg_attrs=*/ArrayAttr(), /*res_attrs=*/ArrayAttr(), aliases,
          seFree);
#else
      llvm_unreachable("Triton disabled");
#endif
    }

    for (auto [resIdx, posInArgs] :
         llvm::enumerate(launch.writtenPositionsInArgs)) {
      unsigned candIdx = launch.argIndices[posInArgs];
      curTensors[candIdx] = callOp->getResult(resIdx);
    }
  }
}

// Replace a JITCallOp or JITRegionOp using its `output_operand_aliases`.
static LogicalResult mapAliasedOutputsToResults(ArrayAttr callSiteAliases,
                                                ArrayRef<Value> updatedTensors,
                                                ResultRange oldResults) {
  if (!callSiteAliases) {
    return oldResults.empty() ? success() : failure();
  }
  if (callSiteAliases.size() != oldResults.size())
    return failure();
  for (auto [resIdx, attr] : llvm::enumerate(callSiteAliases)) {
    auto alias = dyn_cast<stablehlo::OutputOperandAliasAttr>(attr);
    if (!alias)
      return failure();
    int64_t opIdx = alias.getOperandIndex();
    if (opIdx < 0 || static_cast<size_t>(opIdx) >= updatedTensors.size())
      return failure();
    oldResults[resIdx].replaceAllUsesWith(updatedTensors[opIdx]);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Candidate Drivers (func.func + jit_call, standalone func.func, jit_region,
// gpu_wrapper)
//===----------------------------------------------------------------------===//

static bool tryRaiseKernelFunc(func::FuncOp func, ArrayRef<Operation *> users,
                               const GpuTileRaisingOptions &opts) {
  if (func.isDeclaration() || !func.getBody().hasOneBlock())
    return false;

  // Validate all users (if any) are `enzymexla.jit_call` with matching arity.
  for (Operation *u : users) {
    auto jc = dyn_cast<enzymexla::JITCallOp>(u);
    if (!jc || jc.getInputs().size() != func.getNumArguments())
      return false;
    if (jc.getOutputOperandAliasesAttr() &&
        jc.getOutputOperandAliasesAttr().size() != jc.getNumResults())
      return false;
  }

  auto parentMod = func->getParentOfType<ModuleOp>();
  if (!parentMod)
    return false;

  SmallVector<LoweredKernelLaunch> launches;
  SmallVector<Operation *> createdExtMods;
  if (failed(raiseCandidateRegionToModuleAndLaunches(
          parentMod, func.getSymName(), func.getBody(), opts, launches,
          createdExtMods))) {
    return false;
  }

  Location funcLoc = sanitizeLocation(func.getLoc(), opts.strip_llvm_debuginfo);
  if (users.empty()) {
    // Pattern 2: Standalone memref kernel function (matching
    // --raise-affine-to-stablehlo's `@<name>_raised(%tensors...) ->
    // (%tensors...)` signature).
    SmallVector<Type> tensorTypes;
    for (Type argTy : func.getFunctionType().getInputs()) {
      auto mt = cast<MemRefType>(argTy);
      tensorTypes.push_back(
          RankedTensorType::get(mt.getShape(), mt.getElementType()));
    }
    auto raisedFnType =
        FunctionType::get(func.getContext(), tensorTypes, tensorTypes);
    std::string raisedName = (func.getSymName() + "_raised").str();
    OpBuilder modB(func);
    auto newFunc =
        func::FuncOp::create(modB, funcLoc, raisedName, raisedFnType);
    newFunc.setVisibility(SymbolTable::Visibility::Private);
    Block *entry = newFunc.addEntryBlock();
    OpBuilder fb = OpBuilder::atBlockBegin(entry);
    SmallVector<Value> curTensors(entry->getArguments().begin(),
                                  entry->getArguments().end());
    emitHostLaunchSequence(fb, funcLoc, opts.backend, launches, curTensors);
    func::ReturnOp::create(fb, funcLoc, curTensors);
    func.erase();
    return true;
  }

  // Pattern 1: Replace every `enzymexla.jit_call` user directly with
  // `enzymexla_tt_ext.call` / `enzymexla_ct_ext.call`.
  for (Operation *u : users) {
    auto jc = cast<enzymexla::JITCallOp>(u);
    Location callLoc = sanitizeLocation(jc.getLoc(), opts.strip_llvm_debuginfo);
    OpBuilder cb(jc);
    SmallVector<Value> curTensors(jc.getInputs().begin(), jc.getInputs().end());
    ArrayAttr layouts =
        (launches.size() == 1 &&
         launches[0].argIndices.size() == curTensors.size())
            ? dyn_cast_or_null<ArrayAttr>(jc.getOperandLayoutsAttr())
            : ArrayAttr();
    emitHostLaunchSequence(cb, callLoc, opts.backend, launches, curTensors,
                           layouts, jc.getXlaSideEffectFreeAttr());
    if (failed(mapAliasedOutputsToResults(jc.getOutputOperandAliasesAttr(),
                                          curTensors, jc.getResults()))) {
      for (Operation *m : createdExtMods)
        m->erase();
      return false;
    }
    jc.erase();
  }
  func.erase();
  return true;
}

static bool tryRaiseJITRegion(enzymexla::JITRegionOp regionOp,
                              unsigned regionCounter,
                              const GpuTileRaisingOptions &opts) {
  auto parentMod = regionOp->getParentOfType<ModuleOp>();
  if (!parentMod || !regionOp.getBody().hasOneBlock())
    return false;
  if (regionOp.getBody().getNumArguments() != regionOp.getInputs().size())
    return false;

  std::string baseName = "jit_region_" + std::to_string(regionCounter);
  SmallVector<LoweredKernelLaunch> launches;
  SmallVector<Operation *> createdExtMods;
  if (failed(raiseCandidateRegionToModuleAndLaunches(
          parentMod, baseName, regionOp.getBody(), opts, launches,
          createdExtMods))) {
    return false;
  }

  Location loc = sanitizeLocation(regionOp.getLoc(), opts.strip_llvm_debuginfo);
  OpBuilder b(regionOp);
  SmallVector<Value> curTensors(regionOp.getInputs().begin(),
                                regionOp.getInputs().end());
  ArrayAttr layouts =
      (launches.size() == 1 &&
       launches[0].argIndices.size() == curTensors.size())
          ? dyn_cast_or_null<ArrayAttr>(regionOp.getOperandLayoutsAttr())
          : ArrayAttr();
  emitHostLaunchSequence(b, loc, opts.backend, launches, curTensors, layouts,
                         regionOp.getXlaSideEffectFreeAttr());
  if (failed(mapAliasedOutputsToResults(regionOp.getOutputOperandAliasesAttr(),
                                        curTensors, regionOp.getResults()))) {
    for (Operation *m : createdExtMods)
      m->erase();
    return false;
  }
  regionOp.erase();
  return true;
}

static bool tryRaiseGPUWrapper(enzymexla::GPUWrapperOp g,
                               unsigned wrapperCounter,
                               const GpuTileRaisingOptions &opts) {
  auto parentMod = g->getParentOfType<ModuleOp>();
  if (!parentMod || !g.getRegion().hasOneBlock())
    return false;

  // Inspect values defined above `g.getRegion()` without mutating `g` unless
  // all external operands are MemRefTypes or scalar constants!
  SetVector<Value> externalVals;
  getUsedValuesDefinedAbove(g.getRegion(), externalVals);

  SmallVector<Value> capturedMemrefs;
  for (Value ext : externalVals) {
    Attribute cstAttr;
    if (isa<IntegerType, FloatType, IndexType>(ext.getType()) &&
        matchPattern(ext, m_Constant(&cstAttr))) {
      continue;
    }
    auto mt = dyn_cast<MemRefType>(ext.getType());
    if (!mt || !mt.hasStaticShape() ||
        !isSupportedScalarElemType(mt.getElementType(), opts.backend)) {
      return false;
    }
    capturedMemrefs.push_back(ext);
  }

  // Build a synthetic single-block region parameterized by `capturedMemrefs`,
  // cloning scalar constants into the top of the block.
  Region parameterizedRegion;
  Block *paramBlk = &parameterizedRegion.emplaceBlock();
  IRMapping map;
  for (Value mr : capturedMemrefs)
    map.map(mr, paramBlk->addArgument(mr.getType(), mr.getLoc()));

  OpBuilder pb(g.getContext());
  pb.setInsertionPointToStart(paramBlk);
  for (Value ext : externalVals) {
    if (map.contains(ext))
      continue;
    Operation *def = ext.getDefiningOp();
    Operation *cloned = pb.clone(*def);
    map.map(ext, cloned->getResult(0));
  }
  for (Operation &op : g.getRegion().front().without_terminator())
    pb.clone(op, map);
  enzymexla::PolygeistYieldOp::create(pb, g.getLoc());

  std::string baseName = "rxla_gpu_" + std::to_string(wrapperCounter);
  SmallVector<LoweredKernelLaunch> launches;
  SmallVector<Operation *> createdExtMods;
  if (failed(raiseCandidateRegionToModuleAndLaunches(
          parentMod, baseName, parameterizedRegion, opts, launches,
          createdExtMods))) {
    return false;
  }

  SmallVector<Type> tensorTypes;
  for (Value mr : capturedMemrefs) {
    auto mt = cast<MemRefType>(mr.getType());
    tensorTypes.push_back(
        RankedTensorType::get(mt.getShape(), mt.getElementType()));
  }

  Location loc = sanitizeLocation(g.getLoc(), opts.strip_llvm_debuginfo);
  SymbolTable symTab(parentMod);
  std::string wrapperFnName = "rxla$raised_" + std::to_string(wrapperCounter);
  for (unsigned s = 0; symTab.lookup(wrapperFnName) != nullptr; ++s)
    wrapperFnName = "rxla$raised_" + std::to_string(wrapperCounter) + "_" +
                    std::to_string(s);

  auto fnType = FunctionType::get(g.getContext(), tensorTypes, tensorTypes);
  auto newFunc = func::FuncOp::create(loc, wrapperFnName, fnType);
  newFunc.setVisibility(SymbolTable::Visibility::Private);
  Block *entry = newFunc.addEntryBlock();
  OpBuilder fb = OpBuilder::atBlockBegin(entry);
  SmallVector<Value> curTensors(entry->getArguments().begin(),
                                entry->getArguments().end());
  emitHostLaunchSequence(fb, loc, opts.backend, launches, curTensors);
  func::ReturnOp::create(fb, loc, curTensors);
  parentMod.getBody()->push_back(newFunc);

  OpBuilder gb(g);
  enzymexla::XLAWrapperOp::create(gb, loc, SymbolRefAttr::get(newFunc),
                                  capturedMemrefs, nullptr, nullptr);
  if (g->getNumResults() > 0) {
    Value zero = arith::ConstantIndexOp::create(gb, loc, 0);
    g->getResult(0).replaceAllUsesWith(zero);
  }
  g.erase();
  return true;
}

static LogicalResult
runGpuTileRaisingOnModule(ModuleOp mod, const GpuTileRaisingOptions &opts,
                          bool &anyRaisedOut) {
  anyRaisedOut = false;
#if !ENZYME_JAX_ENABLE_TRITON
  if (opts.backend == TileBackend::Triton) {
    mod.emitError("Triton support is disabled in this build");
    return failure();
  }
#endif

  SmallVector<func::FuncOp> candidateFuncs;
  mod->walk([&](func::FuncOp func) {
    FunctionType ft = func.getFunctionType();
    if (!func.isDeclaration() && ft.getNumResults() == 0 &&
        ft.getNumInputs() != 0 && llvm::all_of(ft.getInputs(), [](Type t) {
          return isa<MemRefType>(t);
        })) {
      bool hasHostWrapper = false;
      func.walk([&](Operation *inner) {
        if (isa<enzymexla::GPUWrapperOp, enzymexla::XLAWrapperOp>(inner))
          hasHostWrapper = true;
      });
      if (!hasHostWrapper)
        candidateFuncs.push_back(func);
    }
  });

  SymbolTableCollection symbolTable;
  SymbolUserMap userMap(symbolTable, mod);

  for (func::FuncOp fn : llvm::reverse(candidateFuncs)) {
    ArrayRef<Operation *> users = userMap.getUsers(fn);
    if (tryRaiseKernelFunc(fn, users, opts)) {
      anyRaisedOut = true;
    } else if (opts.err_if_not_fully_raised) {
      fn.emitError("failed to raise affine function to ")
          << getBackendDisplayName(opts.backend);
      return failure();
    }
  }

  SmallVector<enzymexla::JITRegionOp> jitRegions;
  mod->walk([&](enzymexla::JITRegionOp r) { jitRegions.push_back(r); });
  unsigned regionCounter = 0;
  for (enzymexla::JITRegionOp r : jitRegions) {
    if (tryRaiseJITRegion(r, regionCounter++, opts)) {
      anyRaisedOut = true;
    } else if (opts.err_if_not_fully_raised) {
      r.emitError("failed to raise enzymexla.jit_region to ")
          << getBackendDisplayName(opts.backend);
      return failure();
    }
  }

  SmallVector<enzymexla::GPUWrapperOp> gpuWrappers;
  mod->walk([&](enzymexla::GPUWrapperOp g) { gpuWrappers.push_back(g); });
  unsigned wrapperCounter = 0;
  for (enzymexla::GPUWrapperOp g : gpuWrappers) {
    if (tryRaiseGPUWrapper(g, wrapperCounter++, opts)) {
      anyRaisedOut = true;
    } else if (opts.err_if_not_fully_raised) {
      g.emitError("failed to raise enzymexla.gpu_wrapper to ")
          << getBackendDisplayName(opts.backend);
      return failure();
    }
  }

  return success();
}

struct AffineToTritonRaisingPass
    : public mlir::enzyme::impl::AffineToTritonRaisingBase<
          AffineToTritonRaisingPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
#if ENZYME_JAX_ENABLE_TRITON
    registry.insert<enzymexla::triton_ext::TritonExtDialect,
                    triton::TritonDialect>();
#endif
  }

  void runOnOperation() override {
    GpuTileRaisingOptions opts;
    opts.backend = TileBackend::Triton;
    opts.err_if_not_fully_raised = err_if_not_fully_raised;
    opts.strip_llvm_debuginfo = strip_llvm_debuginfo;
    opts.default_tile_size = default_tile_size;
    opts.max_tile_elements = max_tile_elements;
    opts.default_grid_size = default_grid_size;
    opts.num_warps = num_warps;
    opts.num_stages = num_stages;

    bool anyRaised = false;
    if (failed(runGpuTileRaisingOnModule(getOperation(), opts, anyRaised))) {
      signalPassFailure();
      return;
    }
    if (!anyRaised)
      markAllAnalysesPreserved();
  }
};

struct AffineToCuTileRaisingPass
    : public mlir::enzyme::impl::AffineToCuTileRaisingBase<
          AffineToCuTileRaisingPass> {
  using Base::Base;

  void runOnOperation() override {
    GpuTileRaisingOptions opts;
    opts.backend = TileBackend::CuTile;
    opts.err_if_not_fully_raised = err_if_not_fully_raised;
    opts.strip_llvm_debuginfo = strip_llvm_debuginfo;
    opts.default_tile_size = default_tile_size;
    opts.max_tile_elements = max_tile_elements;
    opts.default_grid_size = default_grid_size;

    bool anyRaised = false;
    if (failed(runGpuTileRaisingOnModule(getOperation(), opts, anyRaised))) {
      signalPassFailure();
      return;
    }
    if (!anyRaised)
      markAllAnalysesPreserved();
  }
};

struct AffineToMosaicRaisingPass
    : public mlir::enzyme::impl::AffineToMosaicRaisingBase<
          AffineToMosaicRaisingPass> {
  using Base::Base;

  void runOnOperation() override {
    GpuTileRaisingOptions opts;
    opts.backend = TileBackend::Mosaic;
    opts.err_if_not_fully_raised = err_if_not_fully_raised;
    opts.strip_llvm_debuginfo = strip_llvm_debuginfo;
    opts.default_tile_size = default_tile_size;
    opts.max_tile_elements = max_tile_elements;
    opts.default_grid_size = default_grid_size;

    bool anyRaised = false;
    if (failed(runGpuTileRaisingOnModule(getOperation(), opts, anyRaised))) {
      signalPassFailure();
      return;
    }
    if (!anyRaised)
      markAllAnalysesPreserved();
  }
};

} // namespace
