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

#include "stablehlo/dialect/ChloOps.h"
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
  // if (isa<enzymexla::CacheLoadOp>(op))
  //   return true;

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
      if (!collectEffects(it, effects, /* ignoreBarriers */ true)) {
        return false;
      }
    }

  bool conservative = false;

  if (isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp()))
    return true;

  // As we didn't hit another barrier, we must check the predecessors of this
  // operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier)) {
    return false;
  }
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
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
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

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type Ty,
                          mlir::Operation *op, PatternRewriter &rewriter) {
  Ty = getElementTypeOrSelf(Ty);
  if (Ty.isInteger())
    return true;
  return allowOnFloatingPointMath || guaranteedNoNanResult(op, rewriter);
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy,
                          Type inTy) {
  outTy = getElementTypeOrSelf(outTy);
  inTy = getElementTypeOrSelf(inTy);
  if (outTy.isInteger() && inTy.isInteger())
    return true;
  return allowOnFloatingPointMath;
}

bool canApplyNoNanPattern(bool allowOnFloatingPointMath, Type outTy, Type inTy,
                          mlir::Operation *op, PatternRewriter &rewriter) {
  outTy = getElementTypeOrSelf(outTy);
  inTy = getElementTypeOrSelf(inTy);
  if (outTy.isInteger() && inTy.isInteger())
    return true;
  return allowOnFloatingPointMath || guaranteedNoNanResult(op, rewriter);
}

bool canApplySymmetricPattern(mlir::Operation *op, PatternRewriter &rewriter) {
  return guaranteedSymmetricResult(op, rewriter);
}

SymmetricResultAnalysis initSymmetricResultAnalysis() {
  return SymmetricResultAnalysis();
}

bool SymmetricResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  if (!attr)
    return false;

  auto type = dyn_cast<RankedTensorType>(attr.getType());
  if (!type)
    return false;
  if (type.getRank() == 0)
    return true;
  if (type.getRank() != 2)
    return false;

  auto shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  if (rows != cols)
    return false;

  if (attr.isSplat())
    return true;

  auto values = attr.getValues<APInt>();
  auto it = values.begin();

  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = i + 1; j < cols; j++) {
      auto a = *(it + i * cols + j);
      auto b = *(it + j * cols + i);
      if (a != b)
        return false;
    }
  }

  return true;
}

bool SymmetricResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  if (!attr)
    return false;

  auto type = dyn_cast<RankedTensorType>(attr.getType());

  if (!type)
    return false;
  if (type.getRank() == 0)
    return true;
  if (type.getRank() != 2)
    return false;

  auto shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  if (rows != cols)
    return false;

  if (attr.isSplat())
    return true;

  auto values = attr.getValues<APFloat>();
  auto it = values.begin();

  for (int64_t i = 0; i < rows; i++) {
    for (int64_t j = i + 1; j < cols; j++) {
      auto a = *(it + i * cols + j);
      auto b = *(it + j * cols + i);
      if (a.compare(b) != llvm::APFloat::cmpEqual)
        return false;
    }
  }

  return true;
}

SymmetricResultAnalysis::State SymmetricResultAnalysis::localGuaranteed(
    Operation *op, SmallVectorImpl<Operation *> &localtodo,
    PatternRewriter &rewriter) {
  assert(op);

  auto outTy = cast<RankedTensorType>(op->getResult(0).getType());
  if (outTy.getRank() != 2)
    return State::NOTGUARANTEED; // this pass only checks for symmetric matrices
  if (outTy.getDimSize(0) != outTy.getDimSize(1))
    return State::NOTGUARANTEED; // quick check and exit

  SplatElementsAttr splatAttr;
  if (matchPattern(op, m_Constant(&splatAttr))) {
    return State::GUARANTEED;
  }

  DenseElementsAttr denseAttr;
  if (matchPattern(op, m_Constant(&denseAttr))) {
    if (guaranteedConstantOp(op, denseAttr, rewriter)) {
      return State::GUARANTEED;
    } else {
      return State::NOTGUARANTEED;
    }
  }

  // check that transpose dimensions are [1,0]
  auto isTrueTranspose = [](stablehlo::TransposeOp tOp) -> bool {
    auto perm = tOp.getPermutation();
    return perm.size() == 2 && perm[0] == 1 && perm[1] == 0;
  };

  // TODO: check for dot_general as well

  if (auto broadcastOp = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    auto operand = broadcastOp.getOperand();
    auto operandTy = cast<RankedTensorType>(operand.getType());
    auto dims = broadcastOp.getBroadcastDimensions();
    if (operandTy.getRank() == 0 && dims.empty()) {
      return State::GUARANTEED;
    }
  }

  // commutative operation with A and A^T will always be symmetric
  // op(A, A^T) will also always be symmetric
  if (stablehlo::hasTraitElementwise(op) &&
      (op->hasTrait<OpTrait::IsCommutative>() ||
       op->hasTrait<hlo::OpTrait::IsCommutative>())) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);

    // op(A, A^T)
    if (auto rhsT = rhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(rhsT)) {
        if (lhs == rhsT.getOperand()) {
          return State::GUARANTEED;
        }
      }
    }

    // op(A^T, A)
    if (auto lhsT = lhs.getDefiningOp<stablehlo::TransposeOp>()) {
      if (isTrueTranspose(lhsT)) {
        if (rhs == lhsT.getOperand()) {
          return State::GUARANTEED;
        }
      }
    }
  }

  bool recursiveCheck = false;

  // elementwise ops
  if (stablehlo::hasTraitElementwise(op)) {
    recursiveCheck = true;
  }

  /**
   * TODO
   * - check if its * 0 -> symmetric
   */

  if (recursiveCheck) {
    bool allOperandsGuaranteed = true;
    for (auto operand : op->getOperands()) {
      {
        auto found = valueCache.find(operand);
        if (found != valueCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }
      auto dop = operand.getDefiningOp();
      if (!dop)
        return State::NOTGUARANTEED;

      {
        auto found = opCache.find(dop);
        if (found != opCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      localtodo.push_back(dop);
      allOperandsGuaranteed = false;
    }

    if (allOperandsGuaranteed)
      return State::GUARANTEED;
    else
      return State::PENDING;
  } else {
    return State::NOTGUARANTEED;
  }
}

NoNanResultAnalysis initNoNanResultAnalysis() {
  auto finiteAnalysis = std::make_shared<FiniteResultAnalysis>();
  auto noNanAnalysis = std::make_shared<NoNanResultAnalysis>();
  noNanAnalysis->setFiniteResultAnalysis(finiteAnalysis);
  finiteAnalysis->setNoNanResultAnalysis(noNanAnalysis);
  return *noNanAnalysis;
}

bool NoNanResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  return true;
}

bool NoNanResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (elem.isNaN())
      return false;
  }
  return true;
}

NoNanResultAnalysis::State
NoNanResultAnalysis::localGuaranteed(Operation *op,
                                     SmallVectorImpl<Operation *> &localtodo,
                                     PatternRewriter &rewriter) {
  assert(op);

  if (auto boolAttr = op->getAttrOfType<BoolAttr>(getAttrName())) {
    if (boolAttr.getValue())
      return State::GUARANTEED;
    else
      return State::NOTGUARANTEED;
  }

  DenseElementsAttr denseAttr;
  if (matchPattern(op, m_Constant(&denseAttr))) {
    if (guaranteedConstantOp(op, denseAttr, rewriter)) {
      return State::GUARANTEED;
    } else {
      return State::NOTGUARANTEED;
    }
  }

  // integer ops
  if (isa<stablehlo::AndOp, stablehlo::OrOp, stablehlo::XorOp, stablehlo::NotOp,
          stablehlo::IotaOp>(op)) {
    return State::GUARANTEED;
  }

  bool recursiveCheck = false;

  if (isa<stablehlo::SliceOp, stablehlo::ConcatenateOp,
          stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp>(op)) {
    // data movement ops
    recursiveCheck = true;
  } else if (isa<stablehlo::AbsOp, stablehlo::ExpOp, stablehlo::ConvertOp,
                 stablehlo::CompareOp, stablehlo::TanhOp, stablehlo::LogisticOp,
                 stablehlo::FloorOp, stablehlo::CeilOp>(op)) {
    // elementwise ops that are no-nan if all operands are not nan
    recursiveCheck = true;
  } else if (isa<stablehlo::AddOp, stablehlo::SubtractOp>(op)) {

    // If any one of the operands is a Inf, the result is Inf. If both are Inf,
    // the result is NaN.
    auto lhsFinite =
        finiteResultAnalysis->guaranteed(op->getOperand(0), rewriter);
    auto rhsFinite =
        finiteResultAnalysis->guaranteed(op->getOperand(1), rewriter);

    if (lhsFinite && rhsFinite) {
      return State::GUARANTEED;
    }

    recursiveCheck = true;
  } else if (isa<stablehlo::SineOp, stablehlo::CosineOp>(op)) {

    if (!finiteResultAnalysis->guaranteed(op->getOperand(0), rewriter)) {
      return State::NOTGUARANTEED;
    }

    recursiveCheck = true;
  } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    // if lhs is Inf & rhs is 0 or the other way around, mul is going to be NaN

    // TODO: If one is inf check if the other is zero. We can significantly
    // relax this check if we can prove that the other is not zero.
    if (!finiteResultAnalysis->guaranteed(mulOp.getLhs(), rewriter) ||
        !finiteResultAnalysis->guaranteed(mulOp.getRhs(), rewriter)) {
      return State::NOTGUARANTEED;
    }

    recursiveCheck = true;
  } else if (isa<mlir::stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
  }

  if (recursiveCheck) {
    bool allOperandsGuaranteed = true;
    for (auto operand : op->getOperands()) {
      if (auto TT = dyn_cast<TensorType>(operand.getType())) {
        if (TT.getElementType().isInteger())
          continue;
      }

      {
        auto found = valueCache.find(operand);
        if (found != valueCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }
      auto dop = operand.getDefiningOp();
      if (!dop)
        return State::NOTGUARANTEED;

      {
        auto found = opCache.find(dop);
        if (found != opCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      localtodo.push_back(dop);
      allOperandsGuaranteed = false;
    }

    if (allOperandsGuaranteed) {
      return State::GUARANTEED;
    } else {
      return State::PENDING;
    }
  } else {
    return State::NOTGUARANTEED;
  }
}

FiniteResultAnalysis initFiniteResultAnalysis() {
  auto noNanAnalysis = std::make_shared<NoNanResultAnalysis>();
  auto finiteAnalysis = std::make_shared<FiniteResultAnalysis>();
  noNanAnalysis->setFiniteResultAnalysis(finiteAnalysis);
  finiteAnalysis->setNoNanResultAnalysis(noNanAnalysis);
  return *finiteAnalysis;
}

bool FiniteResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (!elem.isFinite())
      return false;
  }
  return true;
}

bool FiniteResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  return true;
}

FiniteResultAnalysis::State
FiniteResultAnalysis::localGuaranteed(Operation *op,
                                      SmallVectorImpl<Operation *> &localtodo,
                                      PatternRewriter &rewriter) {
  assert(op);

  if (auto boolAttr = op->getAttrOfType<BoolAttr>(getAttrName())) {
    if (boolAttr.getValue())
      return State::GUARANTEED;
    else
      return State::NOTGUARANTEED;
  }

  DenseElementsAttr denseAttr;
  if (matchPattern(op, m_Constant(&denseAttr))) {
    if (guaranteedConstantOp(op, denseAttr, rewriter)) {
      return State::GUARANTEED;
    } else {
      return State::NOTGUARANTEED;
    }
  }

  // integer ops
  if (isa<stablehlo::AndOp, stablehlo::OrOp, stablehlo::XorOp, stablehlo::NotOp,
          stablehlo::IotaOp>(op)) {
    return State::GUARANTEED;
  }

  bool recursiveCheck = false;

  if (isa<stablehlo::SliceOp, stablehlo::ConcatenateOp,
          stablehlo::BroadcastInDimOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp>(op)) {
    // data movement ops
    recursiveCheck = true;
  } else if (isa<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp,
                 stablehlo::AbsOp, stablehlo::ExpOp, stablehlo::ConvertOp,
                 stablehlo::CompareOp>(op)) {
    // if both finite [but possibly nan], the result is finite, or nan

    recursiveCheck = true;
  } else if (isa<stablehlo::TanhOp, stablehlo::LogisticOp, stablehlo::SineOp,
                 stablehlo::CosineOp>(op)) {
    // guaranteed finite or nan result, always
    return State::GUARANTEED;
  } else if (isa<mlir::stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
  }

  if (recursiveCheck) {
    bool allOperandsGuaranteed = true;
    for (auto operand : op->getOperands()) {
      if (auto TT = dyn_cast<TensorType>(operand.getType())) {
        if (TT.getElementType().isInteger())
          continue;
      }

      {
        auto found = valueCache.find(operand);
        if (found != valueCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      auto dop = operand.getDefiningOp();
      if (!dop) {
        return State::NOTGUARANTEED;
      }

      {
        auto found = opCache.find(dop);
        if (found != opCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      localtodo.push_back(dop);
      allOperandsGuaranteed = false;
    }

    if (allOperandsGuaranteed) {
      return State::GUARANTEED;
    } else {
      return State::PENDING;
    }
  } else {
    return State::NOTGUARANTEED;
  }
}

bool NonNegativeResultAnalysis::constantIntCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APInt>()) {
    if (elem.isNegative())
      return false;
  }
  return true;
}

bool NonNegativeResultAnalysis::constantFloatCheck(DenseElementsAttr attr) {
  for (auto elem : attr.getValues<APFloat>()) {
    if (elem.isNegative())
      return false;
  }
  return true;
}

NonNegativeResultAnalysis::State NonNegativeResultAnalysis::localGuaranteed(
    Operation *op, SmallVectorImpl<Operation *> &localtodo,
    PatternRewriter &rewriter) {
  assert(op);

  if (auto boolAttr = op->getAttrOfType<BoolAttr>(getAttrName())) {
    if (boolAttr.getValue())
      return State::GUARANTEED;
    else
      return State::NOTGUARANTEED;
  }

  DenseElementsAttr denseAttr;
  if (matchPattern(op, m_Constant(&denseAttr))) {
    if (guaranteedConstantOp(op, denseAttr, rewriter)) {
      return State::GUARANTEED;
    } else {
      return State::NOTGUARANTEED;
    }
  }

  // integer ops
  if (isa<stablehlo::AbsOp, stablehlo::SqrtOp, stablehlo::ExpOp,
          stablehlo::IotaOp, stablehlo::AndOp, stablehlo::OrOp,
          stablehlo::XorOp, stablehlo::NotOp>(op)) {
    return State::GUARANTEED;
  }

  if (isa<chlo::ErfInvOp>(op)) {
    return State::NOTGUARANTEED;
  }

  // Any non-negative operation that produces a non-negative result
  // Here we recur on the rhs, as that is more likely to be a constant.
  if (isa<stablehlo::MaxOp>(op)) {
    if (guaranteed(op->getOperand(1), rewriter)) {
      return State::GUARANTEED;
    }

    auto operand = op->getOperand(0);

    {
      auto found = valueCache.find(operand);
      if (found != valueCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    auto dop = operand.getDefiningOp();
    if (!dop) {
      return State::NOTGUARANTEED;
    }

    {
      auto found = opCache.find(dop);
      if (found != opCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    localtodo.push_back(dop);
    return State::PENDING;
  }

  // (mul a a) is always non-negative
  if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    auto lhsOp = mulOp.getLhs().getDefiningOp();
    auto rhsOp = mulOp.getRhs().getDefiningOp();

    if (lhsOp == rhsOp) {
      return State::GUARANTEED;
    }
  }

  if (auto clampOp = dyn_cast<stablehlo::ClampOp>(op)) {
    // Clamp is non-negative if the min operand is non-negative
    auto operand = clampOp.getMin();

    {
      auto found = valueCache.find(operand);
      if (found != valueCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    auto dop = operand.getDefiningOp();
    if (!dop) {
      return State::NOTGUARANTEED;
    }

    {
      auto found = opCache.find(dop);
      if (found != opCache.end()) {
        if (found->second) {
          return State::GUARANTEED;
        } else {
          return State::NOTGUARANTEED;
        }
      }
    }

    localtodo.push_back(dop);
    return State::PENDING;
  }

  bool recursiveCheck = false;

  if (isa<stablehlo::MinOp, stablehlo::AddOp, stablehlo::MulOp,
          stablehlo::ConcatenateOp, stablehlo::ReshapeOp,
          stablehlo::TransposeOp, stablehlo::SliceOp,
          stablehlo::DynamicUpdateSliceOp, stablehlo::BroadcastInDimOp>(op)) {
    // All non-negative operations that produce a non-negative result
    recursiveCheck = true;
  } else if (isa<mlir::stablehlo::SelectOp>(op)) {
    recursiveCheck = true;
  }

  if (recursiveCheck) {
    bool allOperandsGuaranteed = true;
    size_t idx = 0;
    for (auto operand : op->getOperands()) {
      if (idx == 0 && isa<mlir::stablehlo::SelectOp>(op))
        continue;
      idx++;

      {
        auto found = valueCache.find(operand);
        if (found != valueCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      auto dop = operand.getDefiningOp();
      if (!dop) {
        return State::NOTGUARANTEED;
      }

      {
        auto found = opCache.find(dop);
        if (found != opCache.end()) {
          if (found->second) {
            continue;
          } else {
            return State::NOTGUARANTEED;
          }
        }
      }

      localtodo.push_back(dop);
      allOperandsGuaranteed = false;
    }

    if (allOperandsGuaranteed) {
      return State::GUARANTEED;
    } else {
      return State::PENDING;
    }
  } else {
    return State::NOTGUARANTEED;
  }
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

SmallVector<int64_t> findReshapeInsertionDims(RankedTensorType inputType,
                                              RankedTensorType outputType) {
  return findReshapeInsertionDims(inputType.getShape(), outputType.getShape());
}

SmallVector<int64_t> findReshapeInsertionDims(ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> outputShape) {
  if (inputShape.size() >= outputShape.size())
    return {}; // trivial no insertion case

  SmallVector<int64_t> insertionDims;
  size_t inputDimIndex = 0;

  for (size_t i = 0; i < outputShape.size(); ++i) {
    auto dim = outputShape[i];
    if (inputDimIndex < inputShape.size() && dim == inputShape[inputDimIndex]) {
      ++inputDimIndex;
    } else if (dim == 1 && (inputDimIndex >= inputShape.size() ||
                            dim != inputShape[inputDimIndex])) {
      // Singleton dimension inserted by reshape.
      insertionDims.push_back(i);
    } else {
      // Reshape modifies existing dimensions, which we don't handle here.
      return {};
    }
  }

  // If we haven't seen all of the input dimensions, we don't have a valid
  // insertion point.
  if (inputDimIndex != inputShape.size())
    return {};

  return insertionDims;
}

bool areValidInsertionDims(RankedTensorType inputType,
                           RankedTensorType outputType,
                           SmallVector<int64_t> insertionDims) {
  if (insertionDims.size() != outputType.getRank() - inputType.getRank())
    return false;

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  for (auto dim : insertionDims) {
    if (dim >= outputType.getRank() || dim < 0 || outputShape[dim] != 1)
      return false;
  }

  SmallVector<int64_t> outShapeAfterDeletion;
  outShapeAfterDeletion.reserve(outputShape.size() - insertionDims.size());
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (llvm::is_contained(insertionDims, i))
      continue;
    outShapeAfterDeletion.push_back(outputShape[i]);
  }

  for (auto [inDim, outDim] : llvm::zip(inputShape, outShapeAfterDeletion)) {
    if (inDim != outDim)
      return false;
  }

  return true;
}

bool isOnlyUsedInOperation(Operation *operation, Operation *parentOp) {
  if (!operation || !parentOp)
    return false;

  for (Operation *user : operation->getUsers()) {
    if (user != parentOp)
      return false;
  }
  return true;
}

bool mayReadFrom(Operation *op, Value val) {
  bool hasRecursiveEffects = op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block)
          if (mayReadFrom(&nestedOp, val))
            return true;
      }
    }
    return false;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    effectInterface.getEffects(effects);
    for (auto it : effects) {
      if (!isa<MemoryEffects::Read>(it.getEffect()))
        continue;
      if (mayAlias(it, val))
        return true;
    }
    return false;
  }
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    auto base = getBase(val);
    bool seenuse = false;
    if (isStackAlloca(base) && !isCaptured(base, op, &seenuse) && !seenuse) {
      return false;
    }
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

bool isSetindexBlock(mlir::Block *block) {
  if (block->getNumArguments() != 2)
    return false;

  auto updateValue = block->getArgument(1);

  // The block should have exactly one operation (the return)
  if (block->getOperations().size() != 1)
    return false;

  auto &returnOp = block->front();
  auto stablehloReturnOp = dyn_cast<stablehlo::ReturnOp>(returnOp);
  if (!stablehloReturnOp)
    return false;

  if (stablehloReturnOp.getNumOperands() != 1)
    return false;

  // The returned value should be the update value (second argument)
  return stablehloReturnOp.getOperand(0) == updateValue;
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

stablehlo::ComparisonDirection
reversedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GE;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GT;
  }
}

stablehlo::ComparisonDirection
negatedComparisonDirection(stablehlo::ComparisonDirection direction) {
  switch (direction) {
  case stablehlo::ComparisonDirection::EQ:
    return stablehlo::ComparisonDirection::NE;
  case stablehlo::ComparisonDirection::NE:
    return stablehlo::ComparisonDirection::EQ;
  case stablehlo::ComparisonDirection::GE:
    return stablehlo::ComparisonDirection::LT;
  case stablehlo::ComparisonDirection::GT:
    return stablehlo::ComparisonDirection::LE;
  case stablehlo::ComparisonDirection::LE:
    return stablehlo::ComparisonDirection::GT;
  case stablehlo::ComparisonDirection::LT:
    return stablehlo::ComparisonDirection::GE;
  }
}

bool reshapeIsTranspose(stablehlo::ReshapeOp reshapeOp) {
  auto input = reshapeOp.getOperand();
  auto output = reshapeOp.getResult();

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto outputType = dyn_cast<RankedTensorType>(output.getType());

  if (!inputType || !outputType)
    return false;

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  // If the number of dimensions is different, it cannot be a simple transpose
  if (inputShape.size() != outputShape.size())
    return false;

  // for each input dim find the next output dim that is of same size.
  // if there are any non-1 dims, it cannot be a simple transpose
  size_t j = 0, i = 0;
  for (; i < inputShape.size(); ++i) {
    auto insz = inputShape[i];
    if (insz == 1)
      continue;

    bool found = false;
    while (j < outputShape.size()) {
      auto outsz = outputShape[j];
      ++j;
      if (outsz == 1)
        continue;
      if (insz == outsz) {
        found = true;
        break;
      }
      if (insz != outsz)
        return false;
    }

    if (!found)
      return false;
  }

  for (; j < outputShape.size(); ++j) {
    if (outputShape[j] != 1) {
      return false;
    }
  }

  return true;
}

Value reshapeAxisInto(OpBuilder &builder, Value input,
                      ArrayRef<int64_t> &batchSizes, int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  SmallVector<int64_t> permutation(inputShape.size());
  for (size_t i = 0; i < dim; i++)
    permutation[i] = i + batchSizes.size(); // left shift
  for (size_t i = 0; i < batchSizes.size(); i++)
    permutation[dim + i] = i; // move the batch dims
  for (size_t i = batchSizes.size() + dim; i < permutation.size(); i++)
    permutation[i] = i; // keep the rest

  auto transposedInput =
      stablehlo::TransposeOp::create(builder, input.getLoc(), input,
                                     builder.getDenseI64ArrayAttr(permutation));

  SmallVector<int64_t> newShape(inputShape.begin() + batchSizes.size(),
                                inputShape.end());
  newShape[dim] =
      newShape[dim] * std::accumulate(batchSizes.begin(), batchSizes.end(), 1,
                                      std::multiplies<int64_t>());
  return stablehlo::ReshapeOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(newShape, inputType.getElementType()),
      transposedInput);
}

Value reshapeAxisOutOf(OpBuilder &builder, Value input,
                       ArrayRef<int64_t> &batchSizes, int64_t dim) {
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = llvm::to_vector(inputType.getShape());
  auto batchSize = std::accumulate(batchSizes.begin(), batchSizes.end(), 1,
                                   std::multiplies<int64_t>());
  inputShape[dim] = inputShape[dim] / batchSize;
  for (size_t i = 0; i < batchSizes.size(); i++)
    inputShape.insert(inputShape.begin() + dim + i, batchSizes[i]);

  auto reshapedInput = stablehlo::ReshapeOp::create(
      builder, input.getLoc(),
      RankedTensorType::get(inputShape, inputType.getElementType()), input);

  SmallVector<int64_t> permutation(inputShape.size());
  for (size_t i = 0; i < batchSizes.size(); i++)
    permutation[i] = dim + i;
  for (size_t i = 0; i < dim; i++)
    permutation[batchSizes.size() + i] = i;
  for (size_t i = batchSizes.size() + dim; i < permutation.size(); i++)
    permutation[i] = i;

  return stablehlo::TransposeOp::create(
      builder, input.getLoc(), reshapedInput,
      builder.getDenseI64ArrayAttr(permutation));
}

bool hasTraitElementwise(Operation *op) {
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;

  if (op->hasTrait<hlo::OpTrait::BroadcastingElementwise>()) {
    // Check sizes (shapes) match across operands, not the exact types.
    auto refShapedTy = dyn_cast<ShapedType>(op->getOperand(0).getType());
    if (!refShapedTy)
      return false;

    for (auto operand : op->getOperands()) {
      auto curShapedTy = dyn_cast<ShapedType>(operand.getType());
      if (!curShapedTy)
        return false;

      if (curShapedTy.getRank() != refShapedTy.getRank())
        return false;

      for (int64_t i = 0; i < curShapedTy.getRank(); ++i) {
        int64_t a = curShapedTy.getDimSize(i);
        int64_t b = refShapedTy.getDimSize(i);
        // If both are static and different, sizes don't match.
        if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
          return false;
      }
    }
    return true;
  }

  return false;
}

bool isAssociativeOp(Operation *op) {
  return isa<stablehlo::AddOp, stablehlo::MulOp, stablehlo::MinOp,
             stablehlo::MaxOp, stablehlo::AndOp, stablehlo::OrOp,
             stablehlo::XorOp>(op);
}

} // namespace stablehlo

} // namespace mlir
