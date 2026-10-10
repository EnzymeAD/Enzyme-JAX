#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzymexla {

inline LogicalResult verifyResultsAliasOperands(Operation *op,
                                                ArrayAttr outputOperandAliases,
                                                unsigned numInputs) {
  unsigned numResults = op->getNumResults();
  if (numResults == 0)
    return success();
  SmallVector<bool> aliased(numResults, false);
  for (auto attr : outputOperandAliases) {
    auto alias = dyn_cast<stablehlo::OutputOperandAliasAttr>(attr);
    if (!alias)
      return op->emitOpError("output_operand_aliases must contain "
                             "#stablehlo.output_operand_alias attributes");
    if (alias.getOperandIndex() < 0 ||
        alias.getOperandIndex() >= (int64_t)numInputs)
      return op->emitOpError("output_operand_alias refers to operand ")
             << alias.getOperandIndex() << ", but there are only " << numInputs
             << " inputs";
    auto outIdxs = alias.getOutputTupleIndices();
    if (numResults == 1 ? !outIdxs.empty() : outIdxs.size() != 1)
      return op->emitOpError("output_operand_alias must index the result "
                             "with `output_tuple_indices = [<result>]` "
                             "(empty only for a single result)");
    int64_t idx = numResults == 1 ? 0 : outIdxs[0];
    if (idx < 0 || idx >= (int64_t)numResults)
      return op->emitOpError("output_operand_alias refers to result ")
             << idx << ", but there are only " << numResults << " results";
    aliased[idx] = true;
  }
  for (unsigned i = 0; i < numResults; i++)
    if (!aliased[i])
      return op->emitOpError("result #")
             << i << " has no matching operand in output_operand_aliases";
  return success();
}

/// Replace cast(subindex(x, InterimType), FinalType) with subindex(x,
/// FinalType)
template <typename OpTy>
class ReadOnlyArg final : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  OpTy create(PatternRewriter &rewriter, OpTy launchOp, ArrayRef<Type> resTys,
              ArrayAttr outputAliases) const;

  // Whether each operand of `launchOp` is never written by it.
  BitVector getReadOnlyOperands(OpTy launchOp) const {
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(
        ((Operation *)launchOp)->getParentOfType<ModuleOp>());
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(launchOp, launchOp.getFnAttr()));
    BitVector readonly(launchOp.getInputs().size(), false);
    for (unsigned i = 0, e = readonly.size(); i < e; ++i)
      readonly[i] =
          fn.front().getArgument(i).use_empty() ||
          fn.getArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName()) ||
          fn.getArgAttr(i, LLVM::LLVMDialect::getReadnoneAttrName());
    return readonly;
  }

  LogicalResult matchAndRewrite(OpTy launchOp,
                                PatternRewriter &rewriter) const override {
    BitVector readonly = getReadOnlyOperands(launchOp);

    auto operand_aliases = launchOp.getOutputOperandAliases();
    assert(operand_aliases.size() == launchOp.getNumResults());
    bool changed = false;
    size_t outputs = launchOp.getNumResults();
    for (auto alias_attr : operand_aliases) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
      if (readonly[alias.getOperandIndex()]) {
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

      assert(launchOp.getInputs()[operandIndex].getType() ==
             launchOp.getResultTypes()[idx]);
      if (readonly[operandIndex])
        continue;
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
      if (readonly[operandIndex]) {
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

template <typename T>
static std::optional<SmallVector<T>> getSymbolUsers(FunctionOpInterface fn,
                                                    Operation *op) {
  SmallVector<T> users;
  SmallVector<Region *, 1> todo;
  for (auto &reg : op->getRegions())
    todo.push_back(&reg);
  while (!todo.empty()) {
    for (Operation &op : todo.pop_back_val()->getOps()) {
      if (auto elem = dyn_cast<CallOpInterface>(&op)) {
        auto callable = elem.getCallableForCallee();
        if (auto sym = callable.dyn_cast<SymbolRefAttr>()) {

          if (sym == FlatSymbolRefAttr::get(op.getContext(), fn.getName())) {
            if (auto ET = dyn_cast<T>(&op)) {
              users.push_back(ET);
              continue;
            }
            return {};
          } else {
            continue;
          }
        }
        return {};
      }
      // jitcall and kernelcall ops are only allowed in stablehlo-compatible
      // func ops
      if (isa<LLVM::LLVMFuncOp>(&op))
        continue;
      if (op.hasTrait<OpTrait::SymbolTable>())
        continue;
      for (auto &reg : op.getRegions())
        todo.push_back(&reg);
    }
  }
  return std::move(users);
}

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

    // Do it first on the current launch to avoid costly collection of all
    // symbol users if already we can early prove there are no changes.
    {
      bool potentialChanged = false;
      for (auto arg : fn.front().getArguments()) {
        auto operandIndex = arg.getArgNumber();
        bool readnone = arg.use_empty();
        if (!readnone)
          continue;

        auto operand_aliases = launchOp.getOutputOperandAliases();
        for (auto alias_attr : operand_aliases) {
          auto alias = cast<stablehlo::OutputOperandAliasAttr>(alias_attr);
          auto aliasOperandIndex = alias.getOperandIndex();
          if (aliasOperandIndex == operandIndex) {
            goto notlegal0;
          }
        }
        potentialChanged = true;
        break;
      notlegal0:;
      }
      if (!potentialChanged)
        return failure();
    }

    bool changed = false;

    SmallVector<OpTy> calls;
    auto use_opt = getSymbolUsers<OpTy>(fn, mod);
    if (!use_opt)
      return failure();
    calls = std::move(*use_opt);
    for (auto launchOp : calls) {
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
            goto notlegal;
          }
        }
      }
      changed = true;
      deadArgs[operandIndex] = true;
    notlegal:;
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
