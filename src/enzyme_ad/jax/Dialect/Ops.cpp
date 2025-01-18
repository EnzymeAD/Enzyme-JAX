//===- EnzymeXLAOps.cpp - EnzymeXLA dialect ops -----------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ops.h"
#include "Dialect.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "enzymexla"

using namespace mlir;
using namespace enzymexla;
using namespace mlir::arith;

using namespace mlir::LLVM;
using namespace mlir::stablehlo;

LogicalResult
KernelCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // TODO: Verify that the result type is same as the type of the referenced
  // func.func op.
  auto global = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
class ReadOnlyKernelArg final
    : public OpRewritePattern<enzymexla::KernelCallOp> {
public:
  using OpRewritePattern<enzymexla::KernelCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::KernelCallOp launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(launchOp->getParentOfType<ModuleOp>());
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    auto operand_aliases = launchOp.getOutputOperandAliases();
    assert(operand_aliases.size() == launchOp.getNumResults());
    bool changed = false;
    size_t outputs = launchOp.getNumResults();
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<OutputOperandAliasAttr>(alias_attr);
      auto outputTupleIndices = alias.getOutputTupleIndices();
      auto operandIndex = alias.getOperandIndex();
      auto operandTupleIndices = alias.getOperandTupleIndices();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {

        changed = true;
        outputs--;
      }
    }
    if (!changed)
      return failure();
    SmallVector<Attribute> outputAliases;
    SmallVector<Type> resTys;
    size_t out_idx = 0;
    for (auto en : llvm::enumerate(operand_aliases)) {
      auto idx = en.index();
      auto alias = cast<OutputOperandAliasAttr>(en.value());
      auto outputTupleIndices = alias.getOutputTupleIndices();
      auto operandIndex = alias.getOperandIndex();
      auto operandTupleIndices = alias.getOperandTupleIndices();

      auto operand = fn.front().getArgument(operandIndex);
      assert(launchOp.getInputs()[operandIndex].getType() ==
             launchOp.getResultTypes()[idx]);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {
        continue;
      }
      resTys.push_back(launchOp.getResultTypes()[idx]);
      if (outputs == 1) {
        outputAliases.push_back(OutputOperandAliasAttr::get(
            launchOp->getContext(), {}, operandIndex, {}));
      } else {
        outputAliases.push_back(OutputOperandAliasAttr::get(
            launchOp->getContext(), {(long)out_idx}, operandIndex, {}));
      }
      out_idx++;
    }

    auto newOp = rewriter.create<enzymexla::KernelCallOp>(
        launchOp.getLoc(), resTys, launchOp.getFn(), launchOp.getGridx(),
        launchOp.getGridy(), launchOp.getGridz(), launchOp.getBlockx(),
        launchOp.getBlocky(), launchOp.getBlockz(), launchOp.getShmem(),
        launchOp.getInputs(), launchOp.getBackendConfigAttr(),
        launchOp.getOperandLayoutsAttr(), /*resultLayouts*/ nullptr,
        ArrayAttr::get(launchOp->getContext(), outputAliases));

    assert(outputAliases.size() == newOp.getNumResults());
    SmallVector<Value> replacements;
    out_idx = 0;
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<OutputOperandAliasAttr>(alias_attr);
      auto outputTupleIndices = alias.getOutputTupleIndices();
      auto operandIndex = alias.getOperandIndex();
      auto operandTupleIndices = alias.getOperandTupleIndices();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());

      if (readonly) {
        replacements.push_back(launchOp.getInputs()[operandIndex]);
        continue;
      } else {
        replacements.push_back(newOp.getResult(out_idx));
        out_idx++;
      }
    }
    rewriter.replaceOp(launchOp, replacements);
    return success();
  }
};

class ReadNoneKernelArg final
    : public OpRewritePattern<enzymexla::KernelCallOp> {
public:
  using OpRewritePattern<enzymexla::KernelCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::KernelCallOp launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    auto mod = launchOp->getParentOfType<ModuleOp>();
    symbolTable.getSymbolTable(mod);
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    bool changed = false;

    SmallVector<enzymexla::KernelCallOp> calls;
    auto use_opt = symbolTable.getSymbolTable(mod).getSymbolUses(fn, mod);
    if (!use_opt)
      return failure();
    for (auto u : *use_opt) {
      auto launch2 = dyn_cast<enzymexla::KernelCallOp>(u.getUser());
      if (!launch2)
        return failure();
      calls.push_back(launch2);
      auto operand_aliases2 = launchOp.getOutputOperandAliases();
      assert(operand_aliases2.size() == launchOp.getNumResults());
    }

    BitVector deadArgs(fn.front().getNumArguments(), false);
    for (auto arg : fn.front().getArguments()) {
      auto operandIndex = arg.getArgNumber();
      bool readnone = arg.use_empty();
      //    fn.getArgAttr(operandIndex, LLVMDialect::getReadnoneAttrName());
      if (!readnone)
        continue;

      for (auto call : calls) {
        auto operand_aliases = call.getOutputOperandAliases();
        for (auto alias_attr : operand_aliases) {
          auto alias = cast<OutputOperandAliasAttr>(alias_attr);
          auto aliasOperandIndex = alias.getOperandIndex();
          if (aliasOperandIndex == operandIndex) {
            return failure();
          }
        }
      }
      changed = true;
      deadArgs[operandIndex] = true;
    }

    if (!changed)
      return failure();

    rewriter.modifyOpInPlace(fn, [&]() {
      // fn.eraseArguments(deadArgs);
      if (auto T = dyn_cast<LLVMFunctionType>(fn.getFunctionType())) {
        SmallVector<Type> argStorage;
        mlir::filterTypesOut(fn.getArgumentTypes(), deadArgs, argStorage);
        auto fty2 =
            LLVMFunctionType::get(T.getReturnType(), argStorage, T.getVarArg());
        mlir::function_interface_impl::eraseFunctionArguments(fn, deadArgs,
                                                              fty2);
      } else {
        fn.eraseArguments(deadArgs);
      }
    });

    for (auto call : calls) {
      BitVector nonLiveCallOperands(call.getNumOperands(), false);
      for (int index : deadArgs.set_bits())
        nonLiveCallOperands.set(call.getInputs().getBeginOperandIndex() +
                                index);

      SmallVector<Attribute> outputAliases;
      auto operand_aliases = call.getOutputOperandAliases();

      for (auto alias_attr : operand_aliases) {
        auto alias = cast<OutputOperandAliasAttr>(alias_attr);
        auto operandIndex = alias.getOperandIndex();
        size_t nextIndex = operandIndex;
        for (int index : deadArgs.set_bits()) {
          if (index <= operandIndex)
            nextIndex--;
        }
        outputAliases.push_back(OutputOperandAliasAttr::get(
            call->getContext(), alias.getOutputTupleIndices(), nextIndex,
            alias.getOperandTupleIndices()));
      }

      rewriter.modifyOpInPlace(call, [&]() {
        call->eraseOperands(nonLiveCallOperands);
        call.setOutputOperandAliasesAttr(
            ArrayAttr::get(call->getContext(), outputAliases));
      });
    }
    return success();
  }
};

void KernelCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ReadOnlyKernelArg, ReadNoneKernelArg>(context);
}
