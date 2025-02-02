//===- PropagateConstantBounds.cpp - Remove duplicate fund def ------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROPAGATECONSTANTBOUNDSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {
struct PropagateConstantBoundsPass
    : public enzyme::impl::PropagateConstantBoundsPassBase<
          PropagateConstantBoundsPass> {

  // If we know that this range is constant, we attach the LLVM range attribute
  // to the target. If the target already has a range, we update it by taking
  // the maximum between the current value and the old value to be conservative.
  static void attachConstantRangeIfConstant(MLIRContext *ctx,
                                            Operation *maybeCst,
                                            Operation *target) {
    APInt intValue;
    if (matchPattern(maybeCst, m_ConstantInt(&intValue))) {
      std::string constantRangeAttrName = "range";
      Attribute maybeRange = target->getAttr(constantRangeAttrName);
      if (!maybeRange) {
        target->setAttr(
            constantRangeAttrName,
            LLVM::ConstantRangeAttr::get(ctx, 32, 0, intValue.getSExtValue()));
      } else {
        LLVM::ConstantRangeAttr range =
            dyn_cast<LLVM::ConstantRangeAttr>(maybeRange);
        int64_t high = range.getUpper().getSExtValue();
        high = std::max(high, intValue.getSExtValue());
        target->setAttr(constantRangeAttrName,
                        LLVM::ConstantRangeAttr::get(ctx, 32, 0, high));
      }
    }
  }

  // Replace the target with a constant if the target is a constant value.
  static void replaceWithConstantIfConstant(OpBuilder &builder,
                                            Operation *maybeCst,
                                            Operation *target) {
    APInt intValue;
    auto loc = target->getLoc();
    if (matchPattern(maybeCst, m_ConstantInt(&intValue))) {
      builder.setInsertionPoint(target);
      auto newCst = builder.create<LLVM::ConstantOp>(
          loc, builder.getI32Type(),
          builder.getIntegerAttr(builder.getI32Type(),
                                 intValue.getSExtValue()));
      target->getResult(0).replaceAllUsesWith(newCst.getResult());
    }
  }

  static int32_t getSizeInBytes(Type ty) {
    int32_t bitWidth = 0;
    if (auto inType = dyn_cast<IntegerType>(ty)) {
      bitWidth = inType.getWidth();
      if (bitWidth == 1)
        return 1;
    }
    if (auto floatType = dyn_cast<FloatType>(ty))
      bitWidth = floatType.getWidth();
    assert(bitWidth != 0);
    return bitWidth / 8;
  }

  static int32_t getMemRefSizeInBytes(Value operand) {
    auto ty = operand.getType();
    int32_t numberOfElems = 0;
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
      numberOfElems = tensorTy.getNumElements();
    }
    return numberOfElems * getSizeInBytes(getElementTypeOrSelf(ty));
  }

  FailureOr<Attribute> getAttributeWithName(ArrayRef<NamedAttribute> attrs,
                                            StringRef name) {
    for (const NamedAttribute &attr : attrs) {
      if (attr.getName() == name)
        return attr.getValue();
    }
    return failure();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp->getContext();
    OpBuilder builder(ctx);
    SymbolTable symTable(moduleOp);

    moduleOp->walk([&](enzymexla::KernelCallOp callOp) {
      auto symbolName = callOp.getFn();
      auto callee = symTable.lookup<LLVM::LLVMFuncOp>(symbolName);
      if (!callee)
        return;
      Region *reg = callee.getCallableRegion();
      // thread idx
      reg->walk([&](NVVM::ThreadIdXOp idxOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlockx().getDefiningOp(),
                                      idxOp.getOperation());
      });
      reg->walk([&](NVVM::ThreadIdYOp idyOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlocky().getDefiningOp(),
                                      idyOp.getOperation());
      });
      reg->walk([&](NVVM::ThreadIdZOp idzOp) {
        attachConstantRangeIfConstant(ctx, callOp.getBlockz().getDefiningOp(),
                                      idzOp.getOperation());
      });
      // thread range
      reg->walk([&](NVVM::BlockDimXOp blockIdxOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getBlockx().getDefiningOp(),
                                      blockIdxOp.getOperation());
      });
      reg->walk([&](NVVM::BlockDimYOp blockIdyOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getBlocky().getDefiningOp(),
                                      blockIdyOp.getOperation());
      });
      reg->walk([&](NVVM::BlockDimZOp blockIdzOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getBlockz().getDefiningOp(),
                                      blockIdzOp.getOperation());
      });
      // block index
      reg->walk([&](NVVM::BlockIdXOp blkIdxOp) {
        attachConstantRangeIfConstant(ctx, callOp.getGridx().getDefiningOp(),
                                      blkIdxOp.getOperation());
      });
      reg->walk([&](NVVM::BlockIdYOp blkIdyOp) {
        attachConstantRangeIfConstant(ctx, callOp.getGridy().getDefiningOp(),
                                      blkIdyOp.getOperation());
      });
      reg->walk([&](NVVM::BlockIdZOp blkIdzOp) {
        attachConstantRangeIfConstant(ctx, callOp.getGridz().getDefiningOp(),
                                      blkIdzOp.getOperation());
      });
      // block range
      reg->walk([&](NVVM::GridDimXOp gridIdxOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridx().getDefiningOp(),
                                      gridIdxOp.getOperation());
      });
      reg->walk([&](NVVM::GridDimYOp gridIdyOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridy().getDefiningOp(),
                                      gridIdyOp.getOperation());
      });
      reg->walk([&](NVVM::GridDimZOp gridIdzOp) {
        replaceWithConstantIfConstant(builder,
                                      callOp.getGridz().getDefiningOp(),
                                      gridIdzOp.getOperation());
      });
    });

    auto result = moduleOp->walk([&](enzymexla::KernelCallOp callOp) {
      auto symbolName = callOp.getFn();
      auto callee = symTable.lookup<FunctionOpInterface>(symbolName);
      if (!callee)
        return WalkResult::advance();
      MLIRContext *ctx = callee->getContext();
      for (auto [index, valTy] : llvm::enumerate(callee.getArgumentTypes())) {
        ArrayRef<NamedAttribute> operandAttrs = callee.getArgAttrs(index);
        if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(valTy)) {

          FailureOr<Attribute> noAliasAttr = getAttributeWithName(
              operandAttrs, LLVM::LLVMDialect::getNoAliasAttrName());
          FailureOr<Attribute> alignAttr = getAttributeWithName(
              operandAttrs, LLVM::LLVMDialect::getAlignAttrName());
          FailureOr<Attribute> dereferenceableAttr = getAttributeWithName(
              operandAttrs, LLVM::LLVMDialect::getDereferenceableAttrName());

          if (failed(noAliasAttr)) {
            callee.setArgAttr(index, LLVM::LLVMDialect::getNoAliasAttrName(),
                              UnitAttr::get(ctx));
          }

          if (failed(alignAttr)) {
            callee.setArgAttr(index, LLVM::LLVMDialect::getAlignAttrName(),
                              IntegerAttr::get(IntegerType::get(ctx, 32), 128));
          }

          if (failed(dereferenceableAttr)) {
            callee.setArgAttr(index,
                              LLVM::LLVMDialect::getDereferenceableAttrName(),
                              IntegerAttr::get(IntegerType::get(ctx, 32),
                                               getMemRefSizeInBytes(
                                                   callOp.getInputs()[index])));
          } else {
            // Conservatively update the dereferenceable attribute if the
            // current value is less than we already have.
            IntegerAttr intAttr = cast<IntegerAttr>(*dereferenceableAttr);
            int64_t oldVal = intAttr.getInt();
            int64_t currentVal =
                getMemRefSizeInBytes(callOp.getInputs()[index]);
            if (currentVal < oldVal) {
              callee.setArgAttr(
                  index, LLVM::LLVMDialect::getDereferenceableAttrName(),
                  IntegerAttr::get(IntegerType::get(ctx, 32), currentVal));
            }
          }
        }
      }
      return WalkResult::advance();
    });
  }
};
} // end namespace