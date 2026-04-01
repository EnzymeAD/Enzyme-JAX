#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzymexla {

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
template <typename OpTy>
class ReadOnlyArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  OpTy create(PatternRewriter &rewriter, OpTy launchOp, ArrayRef<Type> resTys,
              ArrayAttr outputAliases) const;

  LogicalResult matchAndRewrite(OpTy launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(
        ((Operation *)launchOp)->getParentOfType<ModuleOp>());
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    auto operand_aliases = launchOp.getOutputOperandAliases();
    assert(operand_aliases.size() == launchOp.getNumResults());
    bool changed = false;
    size_t outputs = launchOp.getNumResults();
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex,
                        LLVM::LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVM::LLVMDialect::getReadnoneAttrName());

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
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(en.value());
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      assert(launchOp.getInputs()[operandIndex].getType() ==
             launchOp.getResultTypes()[idx]);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex,
                        LLVM::LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVM::LLVMDialect::getReadnoneAttrName());

      if (readonly) {
        continue;
      }
      resTys.push_back(launchOp.getResultTypes()[idx]);
      if (outputs == 1) {
        outputAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            launchOp->getContext(), {}, operandIndex, {}));
      } else {
        outputAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            launchOp->getContext(), {(long)out_idx}, operandIndex, {}));
      }
      out_idx++;
    }

    auto newOp = create(rewriter, launchOp, resTys,
                        ArrayAttr::get(launchOp->getContext(), outputAliases));

    assert(outputAliases.size() == newOp.getNumResults());
    SmallVector<Value> replacements;
    out_idx = 0;
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
      auto operandIndex = alias.getOperandIndex();

      auto operand = fn.front().getArgument(operandIndex);
      bool readonly =
          operand.use_empty() ||
          fn.getArgAttr(operandIndex,
                        LLVM::LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(operandIndex, LLVM::LLVMDialect::getReadnoneAttrName());

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

template <typename OpTy>
class ReadNoneArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  void updateOperandSegmentSizes(OpTy call, int32_t numLiveOperands,
                                 PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(OpTy launchOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    auto mod = ((Operation *)launchOp)->getParentOfType<ModuleOp>();
    symbolTable.getSymbolTable(mod);
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));

    // Early error if no arg is read none
    {
      bool potentialReadNone = false;
      for (auto arg : fn.front().getArguments()) {
        bool readnone = arg.use_empty();
        if (!readnone)
          continue;
        potentialReadNone = true;
        break;
      }
      if (!potentialReadNone)
        return failure();
    }
    bool changed = false;

    SmallVector<OpTy> calls;
    auto use_opt = symbolTable.getSymbolTable(mod).getSymbolUses(fn, mod);
    if (!use_opt)
      return failure();
    for (auto u : *use_opt) {
      auto launch2 = dyn_cast<OpTy>(u.getUser());
      if (!launch2)
        return failure();
      calls.push_back(launch2);
      auto operand_aliases2 = launchOp.getOutputOperandAliases();
      (void)operand_aliases2;
      assert(operand_aliases2.size() == launchOp.getNumResults());
    }

    BitVector deadArgs(fn.front().getNumArguments(), false);
    for (auto arg : fn.front().getArguments()) {
      auto operandIndex = arg.getArgNumber();
      bool readnone = arg.use_empty();
      if (!readnone)
        continue;

      for (auto call : calls) {
        auto operand_aliases = call.getOutputOperandAliases();
        for (auto alias_attr : operand_aliases) {
          auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
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
      if (auto T = dyn_cast<LLVM::LLVMFunctionType>(fn.getFunctionType())) {
        SmallVector<Type> argStorage;
        mlir::filterTypesOut(fn.getArgumentTypes(), deadArgs, argStorage);
        auto fty2 = LLVM::LLVMFunctionType::get(T.getReturnType(), argStorage,
                                                T.getVarArg());
        mlir::function_interface_impl::eraseFunctionArguments(fn, deadArgs,
                                                              fty2);
      } else {
        (void)fn.eraseArguments(deadArgs);
      }
    });

    for (auto call : calls) {
      BitVector nonLiveCallOperands(call.getNumOperands(), false);
      for (int index : deadArgs.set_bits())
        nonLiveCallOperands.set(call.getInputs().getBeginOperandIndex() +
                                index);

      int32_t numLiveOperands = 0;
      for (int32_t idx = call.getInputs().getBeginOperandIndex();
           idx < nonLiveCallOperands.size(); idx++) {
        if (nonLiveCallOperands[idx])
          continue;
        numLiveOperands++;
      }

      SmallVector<Attribute> outputAliases;
      auto operand_aliases = call.getOutputOperandAliases();

      for (auto alias_attr : operand_aliases) {
        auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
        auto operandIndex = alias.getOperandIndex();
        size_t nextIndex = operandIndex;
        for (int index : deadArgs.set_bits()) {
          if (index <= operandIndex)
            nextIndex--;
        }
        outputAliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            call->getContext(), alias.getOutputTupleIndices(), nextIndex,
            alias.getOperandTupleIndices()));
      }

      rewriter.modifyOpInPlace(call, [&]() {
        call->eraseOperands(nonLiveCallOperands);
        updateOperandSegmentSizes(call, numLiveOperands, rewriter);
        call.setOutputOperandAliasesAttr(
            ArrayAttr::get(call->getContext(), outputAliases));
      });
    }
    return success();
  }
};

} // namespace enzymexla
} // namespace mlir
