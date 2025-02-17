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

  static int32_t getSizeInBytes(Type ty) {
    int32_t bitWidth = 0;
    if (auto inType = dyn_cast<IntegerType>(ty)) {
      bitWidth = inType.getWidth();
      if (bitWidth == 1)
        return 1;
    }
    if (auto floatType = dyn_cast<FloatType>(ty))
      bitWidth = floatType.getWidth();
    if (auto complexType = dyn_cast<ComplexType>(ty)) {
      int32_t sizeInByte = getSizeInBytes(complexType.getElementType());
      return sizeInByte * 2;
    }
    assert(bitWidth != 0);
    return bitWidth / 8;
  }

  static int64_t getMemRefSizeInBytes(Value operand) {
    auto ty = operand.getType();
    int64_t numberOfElems = 0;
    if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
      numberOfElems = tensorTy.getNumElements();
    }
    return numberOfElems * getSizeInBytes(getElementTypeOrSelf(ty));
  }

  struct ThreadAndBlockInfo {
    int64_t threadIdX = std::numeric_limits<int64_t>::min();
    int64_t threadIdY = std::numeric_limits<int64_t>::min();
    int64_t threadIdZ = std::numeric_limits<int64_t>::min();

    int64_t blockIdX = std::numeric_limits<int64_t>::min();
    int64_t blockIdY = std::numeric_limits<int64_t>::min();
    int64_t blockIdZ = std::numeric_limits<int64_t>::min();

    int64_t blockDimX = std::numeric_limits<int64_t>::min();
    int64_t blockDimY = std::numeric_limits<int64_t>::min();
    int64_t blockDimZ = std::numeric_limits<int64_t>::min();

    int64_t gridDimX = std::numeric_limits<int64_t>::min();
    int64_t gridDimY = std::numeric_limits<int64_t>::min();
    int64_t gridDimZ = std::numeric_limits<int64_t>::min();
  };

  static ThreadAndBlockInfo getRange(FunctionOpInterface callee,
                                     CallOpInterface caller) {
    ThreadAndBlockInfo info;

    if (auto kernelCall =
            dyn_cast<enzymexla::KernelCallOp>(caller.getOperation())) {
      APInt intValue;
      if (matchPattern(kernelCall.getBlockx(), m_ConstantInt(&intValue)))
        info.threadIdX = intValue.getSExtValue();
      if (matchPattern(kernelCall.getBlocky(), m_ConstantInt(&intValue)))
        info.threadIdY = intValue.getSExtValue();
      if (matchPattern(kernelCall.getBlockz(), m_ConstantInt(&intValue)))
        info.threadIdZ = intValue.getSExtValue();

      if (matchPattern(kernelCall.getGridx(), m_ConstantInt(&intValue)))
        info.blockIdX = intValue.getSExtValue();
      if (matchPattern(kernelCall.getGridy(), m_ConstantInt(&intValue)))
        info.blockIdY = intValue.getSExtValue();
      if (matchPattern(kernelCall.getGridz(), m_ConstantInt(&intValue)))
        info.blockIdZ = intValue.getSExtValue();

      if (matchPattern(kernelCall.getBlockx(), m_ConstantInt(&intValue)))
        info.blockDimX = intValue.getSExtValue();
      if (matchPattern(kernelCall.getBlocky(), m_ConstantInt(&intValue)))
        info.blockDimY = intValue.getSExtValue();
      if (matchPattern(kernelCall.getBlockz(), m_ConstantInt(&intValue)))
        info.blockDimZ = intValue.getSExtValue();

      if (matchPattern(kernelCall.getGridx(), m_ConstantInt(&intValue)))
        info.gridDimX = intValue.getSExtValue();
      if (matchPattern(kernelCall.getGridy(), m_ConstantInt(&intValue)))
        info.gridDimY = intValue.getSExtValue();
      if (matchPattern(kernelCall.getGridz(), m_ConstantInt(&intValue)))
        info.gridDimZ = intValue.getSExtValue();
    }
    return info;
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = moduleOp->getContext();
    OpBuilder builder(ctx);
    SymbolTable symTable(moduleOp);

    DenseMap<FunctionOpInterface, SetVector<CallOpInterface>> funcToKernelMap;
    moduleOp->walk([&](enzymexla::KernelCallOp callOp) {
      auto symbolName =
          dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
      if (!symbolName)
        return;
      auto callee =
          symTable.lookup<FunctionOpInterface>(symbolName.getLeafReference());
      if (!callee)
        return;
      funcToKernelMap[callee].insert(callOp);
    });
    moduleOp->walk([&](enzymexla::JITCallOp callOp) {
      auto symbolName =
          dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
      if (!symbolName)
        return;
      auto callee =
          symTable.lookup<FunctionOpInterface>(symbolName.getLeafReference());
      if (!callee)
        return;
      funcToKernelMap[callee].insert(callOp);
    });
    if (funcToKernelMap.empty())
      return;

    // Iterate over all the callees.
    for (auto [callee, callers] : funcToKernelMap) {
      SmallVector<int64_t> dereferenceable(callee.getNumArguments(),
                                           std::numeric_limits<int64_t>::max());
      ThreadAndBlockInfo maxRange;

      // Iterate over all the callers for this callee. Compute the max over all
      // callers for thread/blocks related things while min for dereferenceable
      // attribute.
      for (auto caller : callers) {
        SmallVector<Value> operands = llvm::to_vector(caller.getArgOperands());
        assert(operands.size() == callee.getNumArguments());

        ThreadAndBlockInfo currentRange = getRange(callee, caller);
        maxRange.threadIdX =
            std::max(maxRange.threadIdX, currentRange.threadIdX);
        maxRange.threadIdY =
            std::max(maxRange.threadIdY, currentRange.threadIdY);
        maxRange.threadIdZ =
            std::max(maxRange.threadIdZ, currentRange.threadIdZ);

        maxRange.blockIdX = std::max(maxRange.blockIdX, currentRange.blockIdX);
        maxRange.blockIdY = std::max(maxRange.blockIdY, currentRange.blockIdY);
        maxRange.blockIdZ = std::max(maxRange.blockIdZ, currentRange.blockIdZ);

        maxRange.blockDimX =
            std::max(maxRange.blockDimX, currentRange.blockDimX);
        maxRange.blockDimY =
            std::max(maxRange.blockDimY, currentRange.blockDimY);
        maxRange.blockDimZ =
            std::max(maxRange.blockDimZ, currentRange.blockDimZ);

        maxRange.gridDimX = std::max(maxRange.gridDimX, currentRange.gridDimX);
        maxRange.gridDimY = std::max(maxRange.gridDimY, currentRange.gridDimY);
        maxRange.gridDimZ = std::max(maxRange.gridDimZ, currentRange.gridDimZ);

        for (auto [index, operand] : llvm::enumerate(callee.getArguments())) {
          dereferenceable[index] = std::min(
              dereferenceable[index], getMemRefSizeInBytes(operands[index]));
        }
      }
      Region *reg = callee.getCallableRegion();
      assert(reg);
      std::string constantRangeAttrName = "range";

      auto setConstantRangeAttrIfConstant = [&](auto op, int64_t maxValue) {
        if (maxValue == std::numeric_limits<int64_t>::min())
          return;
        op->setAttr(constantRangeAttrName,
                    LLVM::ConstantRangeAttr::get(ctx, 32, 0, maxValue));
      };

      reg->walk([&](NVVM::ThreadIdXOp threadIdXOp) {
        setConstantRangeAttrIfConstant(threadIdXOp, maxRange.threadIdX);
      });
      reg->walk([&](NVVM::ThreadIdYOp threadIdYOp) {
        setConstantRangeAttrIfConstant(threadIdYOp, maxRange.threadIdY);
      });
      reg->walk([&](NVVM::ThreadIdZOp threadIdZOp) {
        setConstantRangeAttrIfConstant(threadIdZOp, maxRange.threadIdZ);
      });
      reg->walk([&](NVVM::BlockIdXOp blkIdxOp) {
        setConstantRangeAttrIfConstant(blkIdxOp, maxRange.blockIdX);
      });
      reg->walk([&](NVVM::BlockIdYOp blkIdyOp) {
        setConstantRangeAttrIfConstant(blkIdyOp, maxRange.blockIdY);
      });
      reg->walk([&](NVVM::BlockIdZOp blkIdzOp) {
        setConstantRangeAttrIfConstant(blkIdzOp, maxRange.blockIdZ);
      });

      // Attempt to replace BlockDim and GridDim with constants if we have a
      // single caller for this callee and BlockDim and GridDim are known
      // constants. Otherwise set an interval range, if BlockDim and
      // GridDim are known constants.
      auto replaceWithConstantOrSetConstantRangeAttr =
          [&](auto op, int64_t maxValue, bool hasSingleCaller) {
            if (maxValue == std::numeric_limits<int64_t>::min())
              return;
            if (hasSingleCaller) {
              builder.setInsertionPoint(op);
              auto newCst = builder.create<LLVM::ConstantOp>(
                  op->getLoc(), builder.getI32Type(),
                  builder.getIntegerAttr(builder.getI32Type(), maxValue));
              op->getResult(0).replaceAllUsesWith(newCst.getResult());
            } else {
              setConstantRangeAttrIfConstant(op, maxValue);
            }
          };

      bool hasSingleCaller = callers.size() == 1;
      reg->walk([&](NVVM::BlockDimXOp blockDimIdXOp) {
        replaceWithConstantOrSetConstantRangeAttr(
            blockDimIdXOp, maxRange.blockDimX, hasSingleCaller);
      });
      reg->walk([&](NVVM::BlockDimYOp blockDimIdYOp) {
        replaceWithConstantOrSetConstantRangeAttr(
            blockDimIdYOp, maxRange.blockDimY, hasSingleCaller);
      });
      reg->walk([&](NVVM::BlockDimZOp blockDimIdZOp) {
        replaceWithConstantOrSetConstantRangeAttr(
            blockDimIdZOp, maxRange.blockDimZ, hasSingleCaller);
      });
      reg->walk([&](NVVM::GridDimXOp gridDimXOp) {
        replaceWithConstantOrSetConstantRangeAttr(gridDimXOp, maxRange.gridDimX,
                                                  hasSingleCaller);
      });
      reg->walk([&](NVVM::GridDimYOp gridDimYOp) {
        replaceWithConstantOrSetConstantRangeAttr(gridDimYOp, maxRange.gridDimY,
                                                  hasSingleCaller);
      });
      reg->walk([&](NVVM::GridDimZOp gridDimZOp) {
        replaceWithConstantOrSetConstantRangeAttr(gridDimZOp, maxRange.gridDimZ,
                                                  hasSingleCaller);
      });

      // Set no-alias for each callee arguments. No-alias is guaranteed.
      // Set alignment to 128. An alignment of 128 is guaranteed.
      // Set dereferenceable for each operand in callee.
      MLIRContext *ctx = callee->getContext();
      for (auto [index, valTy] : llvm::enumerate(callee.getArgumentTypes())) {
        if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(valTy)) {
          callee.setArgAttr(index, LLVM::LLVMDialect::getNoAliasAttrName(),
                            UnitAttr::get(ctx));
          callee.setArgAttr(index, LLVM::LLVMDialect::getAlignAttrName(),
                            IntegerAttr::get(IntegerType::get(ctx, 32), 128));
          if (dereferenceable[index] != std::numeric_limits<int64_t>::max()) {
            callee.setArgAttr(index,
                              LLVM::LLVMDialect::getDereferenceableAttrName(),
                              IntegerAttr::get(IntegerType::get(ctx, 64),
                                               dereferenceable[index]));
          }
        }
      }
    }
  }
};
} // end namespace
