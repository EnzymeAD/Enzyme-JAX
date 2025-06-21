//===- PolygeistOps.cpp - BFV dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "stablehlo/dialect/StablehloOps.h"

#include <set>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::arith;

namespace mlir {
namespace enzyme {
/// Collect the memory effects of the given op in 'effects'. Returns 'true' it
/// could extract the effect information from the op, otherwise returns 'false'
/// and conservatively populates the list with all possible effects.
bool collectEffects(Operation *op,
                    SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                    bool ignoreBarriers) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<enzymexla::BarrierOp>(op))
    return true;

  // Ignore CacheLoads as they are already guaranteed to not have side effects
  // in the context of a parallel op, these only exist while we are in the
  // CPUifyPass
  // if (isa<CacheLoad>(op))
  //  return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  if (auto cop = dyn_cast<LLVM::CallOp>(op)) {
    if (auto callee = cop.getCallee()) {
      if (*callee == "scanf" || *callee == "__isoc99_scanf") {
        // Global read
        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());

        bool first = true;
        for (auto &arg : cop.getArgOperandsMutable()) {
          if (first)
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg);
          else
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          first = false;
        }

        return true;
      }
      if (*callee == "fscanf" || *callee == "__isoc99_fscanf") {
        // Global read
        effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());

        for (auto &&[idx, arg] : llvm::enumerate(cop.getArgOperandsMutable())) {
          if (idx == 0) {
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          } else if (idx == 1) {
            effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
          } else
            effects.emplace_back(::mlir::MemoryEffects::Write::get(), &arg,
                                 ::mlir::SideEffects::DefaultResource::get());
        }

        return true;
      }
      if (*callee == "printf") {
        // Global read
        effects.emplace_back(
            MemoryEffects::Effect::get<MemoryEffects::Write>());
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
      if (*callee == "free") {
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Free::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
      if (*callee == "strlen") {
        for (auto &arg : cop.getArgOperandsMutable()) {
          effects.emplace_back(::mlir::MemoryEffects::Read::get(), &arg,
                               ::mlir::SideEffects::DefaultResource::get());
        }
        return true;
      }
    }
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
  return false;
}

// Rethrns if we are non-conservative whether we have filled with all possible
// effects.
bool getEffectsBefore(Operation *op,
                      SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                      bool stopAtBarrier) {
  if (op != &op->getBlock()->front())
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<enzymexla::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects, /* ignoreBarriers */ true))
        return false;
    }

  bool conservative = false;

  if (isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp()))
    return true;

  // As we didn't hit another barrier, we must check the predecessors of this
  // operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  if (!isa<scf::IfOp, affine::AffineIfOp, memref::AllocaScopeOp>(
          op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects, /* ignoreBarriers */ true)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}
bool getEffectsAfter(Operation *op,
                     SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                     bool stopAtBarrier) {
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<enzymexla::BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects, /* ignoreBarriers */ true))
        return false;
    }

  bool conservative = false;

  if (isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp()))
    return true;

  // As we didn't hit another barrier, we must check the predecessors of this
  // operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  if (!isa<scf::IfOp, affine::AffineIfOp, memref::AllocaScopeOp>(
          op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects, /* ignoreBarriers */ true)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

bool isReadOnly(Operation *op) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadOnly(&nestedOp))
            return false;
      }
    }
    return true;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (!llvm::all_of(effects, [op](const MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Read>(it.getEffect());
        })) {
      return false;
    }
    return true;
  }
  return false;
}

bool isReadNone(Operation *op) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (!isReadNone(&nestedOp))
            return false;
      }
    }
    return true;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    if (llvm::any_of(effects, [op](const MemoryEffects::EffectInstance &it) {
          return isa<MemoryEffects::Read>(it.getEffect()) ||
                 isa<MemoryEffects::Write>(it.getEffect());
        })) {
      return false;
    }
    return true;
  }
  return false;
}

const std::set<std::string> &getNonCapturingFunctions() {
  static std::set<std::string> NonCapturingFunctions = {
      "free",           "printf",       "fprintf",       "scanf",
      "fscanf",         "gettimeofday", "clock_gettime", "getenv",
      "strrchr",        "strlen",       "sprintf",       "sscanf",
      "mkdir",          "fwrite",       "fread",         "memcpy",
      "cudaMemcpy",     "memset",       "cudaMemset",    "__isoc99_scanf",
      "__isoc99_fscanf"};
  return NonCapturingFunctions;
}

bool isCaptured(Value v, Operation *potentialUser = nullptr,
                bool *seenuse = nullptr) {
  SmallVector<Value> todo = {v};
  while (todo.size()) {
    Value v = todo.pop_back_val();
    for (auto u : v.getUsers()) {
      if (seenuse && u == potentialUser)
        *seenuse = true;
      if (isa<memref::LoadOp, LLVM::LoadOp, affine::AffineLoadOp>(u))
        continue;
      // if (isa<polygeist::CacheLoad>(u)) continue
      if (auto s = dyn_cast<memref::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<affine::AffineStoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto s = dyn_cast<LLVM::StoreOp>(u)) {
        if (s.getValue() == v)
          return true;
        continue;
      }
      if (auto sub = dyn_cast<LLVM::GEPOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::BitcastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<LLVM::AddrSpaceCastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<func::ReturnOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemsetOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemcpyOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<LLVM::MemmoveOp>(u)) {
        continue;
      }
      if (auto sub = dyn_cast<memref::CastOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<memref::DeallocOp>(u)) {
        continue;
      }
      // if (auto sub = dyn_cast<polygeist::SubIndexOp>(u)) {
      //   todo.push_back(sub);
      // }
      if (auto sub = dyn_cast<enzymexla::Memref2PointerOp>(u)) {
        todo.push_back(sub);
      }
      if (auto sub = dyn_cast<enzymexla::Pointer2MemrefOp>(u)) {
        todo.push_back(sub);
      }
      if (auto cop = dyn_cast<LLVM::CallOp>(u)) {
        if (auto callee = cop.getCallee()) {
          if (getNonCapturingFunctions().count(callee->str()))
            continue;
        }
      }
      if (auto cop = dyn_cast<func::CallOp>(u)) {
        if (getNonCapturingFunctions().count(cop.getCallee().str()))
          continue;
      }
      return true;
    }
  }

  return false;
}

Value getBase(Value v) {
  while (true) {
    // if (auto s = v.getDefiningOp<SubIndexOp>()) {
    //   v = s.getSource();
    //   continue;
    // }
    if (auto s = v.getDefiningOp<enzymexla::Memref2PointerOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
      v = s.getSource();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::GEPOp>()) {
      v = s.getBase();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::BitcastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      v = s.getArg();
      continue;
    }
    if (auto s = v.getDefiningOp<memref::CastOp>()) {
      v = s.getSource();
      continue;
    }
    break;
  }
  return v;
}

bool isStackAlloca(Value v) {
  return v.getDefiningOp<memref::AllocaOp>() ||
         v.getDefiningOp<memref::AllocOp>() ||
         v.getDefiningOp<LLVM::AllocaOp>();
}
static bool mayAlias(Value v, Value v2) {
  v = getBase(v);
  v2 = getBase(v2);
  if (v == v2)
    return true;

  // We may now assume neither v1 nor v2 are subindices

  if (auto glob = v.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto Aglob = v2.getDefiningOp<memref::GetGlobalOp>()) {
      return glob.getName() == Aglob.getName();
    }
  }

  if (auto glob = v.getDefiningOp<LLVM::AddressOfOp>()) {
    if (auto Aglob = v2.getDefiningOp<LLVM::AddressOfOp>()) {
      return glob.getGlobalName() == Aglob.getGlobalName();
    }
  }

  bool isAlloca[2];
  bool isGlobal[2];

  isAlloca[0] = isStackAlloca(v);
  isGlobal[0] = v.getDefiningOp<memref::GetGlobalOp>() ||
                v.getDefiningOp<LLVM::AddressOfOp>();

  isAlloca[1] = isStackAlloca(v2);

  isGlobal[1] = v2.getDefiningOp<memref::GetGlobalOp>() ||
                v2.getDefiningOp<LLVM::AddressOfOp>();

  // Non-equivalent allocas/global's cannot conflict with each other
  if ((isAlloca[0] || isGlobal[0]) && (isAlloca[1] || isGlobal[1]))
    return false;

  bool isArg[2];
  isArg[0] = isa<BlockArgument>(v) &&
             isa<FunctionOpInterface>(
                 cast<BlockArgument>(v).getOwner()->getParentOp());

  isArg[1] = isa<BlockArgument>(v) &&
             isa<FunctionOpInterface>(
                 cast<BlockArgument>(v).getOwner()->getParentOp());

  // Stack allocations cannot have been passed as an argument.
  if ((isAlloca[0] && isArg[1]) || (isAlloca[1] && isArg[0]))
    return false;

  // Non captured base allocas cannot conflict with another base value.
  if (isAlloca[0] && !isCaptured(v))
    return false;

  if (isAlloca[1] && !isCaptured(v2))
    return false;

  return true;
}

bool mayAlias(MemoryEffects::EffectInstance a,
              MemoryEffects::EffectInstance b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  if (Value v2 = b.getValue()) {
    return mayAlias(a, v2);
  } else if (Value v = a.getValue()) {
    return mayAlias(b, v);
  }
  return true;
}

bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}
bool mayWriteTo(Operation *op, Value val, bool ignoreBarrier) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (mayWriteTo(&nestedOp, val, ignoreBarrier))
            return true;
      }
    }
    return false;
  }

  if (ignoreBarrier && isa<enzymexla::BarrierOp>(op))
    return false;

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    for (auto it : effects) {
      if (!isa<MemoryEffects::Write>(it.getEffect()))
        continue;
      if (mayAlias(it, val))
        return true;
    }
    return false;
  }

  // Calls which do not use a derived pointer of a known alloca, which is not
  // captured can not write to said memory.
  if (isa<LLVM::CallOp, func::CallOp>(op)) {
    auto base = getBase(val);
    bool seenuse = false;
    if (isStackAlloca(base) && !isCaptured(base, op, &seenuse) && !seenuse) {
      return false;
    }
  }
  return true;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty) {
  Ty = getElementTypeOrSelf(Ty);
  if (Ty.isInteger())
    return true;
  return allowOnFloatingPointMath;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy,
                          Type inTy) {
  outTy = getElementTypeOrSelf(outTy);
  inTy = getElementTypeOrSelf(inTy);
  if (outTy.isInteger() && inTy.isInteger())
    return true;
  return allowOnFloatingPointMath;
}

bool anyOperandIsConstant(mlir::Operation *op) {
  DenseElementsAttr attr;
  for (auto operand : op->getOperands()) {
    if (matchPattern(operand, m_Constant(&attr)))
      return true;
  }
  return false;
}

bool allOperandsAreConstant(mlir::Operation *op) {
  DenseElementsAttr attr;
  for (auto operand : op->getOperands()) {
    if (!matchPattern(operand, m_Constant(&attr)))
      return false;
  }
  return true;
}

} // namespace enzyme

namespace stablehlo {

stablehlo::GatherDimensionNumbersAttr
getGatherDims(mlir::MLIRContext *ctx,
              stablehlo::ScatterDimensionNumbersAttr scatterDimNumbers) {
  return stablehlo::GatherDimensionNumbersAttr::get(
      ctx, scatterDimNumbers.getUpdateWindowDims(),
      scatterDimNumbers.getInsertedWindowDims(),
      scatterDimNumbers.getInputBatchingDims(),
      scatterDimNumbers.getScatterIndicesBatchingDims(),
      scatterDimNumbers.getScatterDimsToOperandDims(),
      scatterDimNumbers.getIndexVectorDim());
}

bool isScatterSetindexOp(stablehlo::ScatterOp &scatterOp) {
  auto &updateComputation = scatterOp.getUpdateComputation();

  if (!updateComputation.hasOneBlock())
    return false;

  auto &block = updateComputation.front();
  if (block.getNumArguments() != 2)
    return false;

  auto originalValue = block.getArgument(0);
  auto updateValue = block.getArgument(1);

  // The block should have exactly one operation (the return)
  if (block.getOperations().size() != 1)
    return false;

  auto &returnOp = block.front();
  auto stablehloReturn = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturn)
    return false;

  if (stablehloReturn.getNumOperands() != 1)
    return false;

  // The returned value should be the update value (second argument)
  return stablehloReturn.getOperand(0) == updateValue;
}

SmallVector<int64_t> computeGatherSliceSizes(stablehlo::ScatterOp &scatterOp) {
  auto inputType = cast<ShapedType>(scatterOp.getInputs()[0].getType());
  auto updateType = cast<ShapedType>(scatterOp.getUpdates()[0].getType());
  auto scatterDimNumbers = scatterOp.getScatterDimensionNumbers();

  auto inputShape = inputType.getShape();
  auto updateShape = updateType.getShape();

  SmallVector<int64_t> sliceSizes(inputShape.size(), 1);

  auto updateWindowDims = scatterDimNumbers.getUpdateWindowDims();
  auto scatterIndicesBatchingDims =
      scatterDimNumbers.getScatterIndicesBatchingDims();

  // Map update window dimensions to their corresponding input dimensions
  for (int64_t i = 0; i < updateWindowDims.size(); ++i) {
    int64_t inputDim = updateWindowDims[i];

    // Calculate the corresponding dimension in the update tensor
    // Update tensor layout: [scatter_indices_batching_dims...,
    // update_window_dims...]
    int64_t updateDimIndex = scatterIndicesBatchingDims.size() + i;

    if (updateDimIndex < updateShape.size()) {
      sliceSizes[inputDim] = updateShape[updateDimIndex];
    }
  }

  return sliceSizes;
}

} // namespace stablehlo

} // namespace mlir
