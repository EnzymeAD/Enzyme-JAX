#include "src/enzyme_ad/jax/Passes/JITCallOrdering.h"

#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;
using namespace mlir;

LogicalResult orderJITCalls(ArrayRef<enzymexla::JITCallOp> callOps) {
  if (callOps.empty())
    return success();

  mlir::DenseMap<Operation *, Operation *> funcToSlot;

  OpBuilder builder(callOps.front());

  auto tokenType = stablehlo::TokenType::get(builder.getContext());

  for (auto jitCall : callOps) {
    auto func = jitCall->getParentOfType<FunctionOpInterface>();
    if (!func) {
      jitCall->emitError("jit call is not in a function");
      return failure();
    }

    Value slot = nullptr;

    auto it = funcToSlot.find(func);
    if (it == funcToSlot.end()) {
      auto slotType =
          enzyme::GradientType::get(jitCall.getContext(), tokenType);

      builder.setInsertionPointToStart(&func.getBlocks().front());
      auto slotOp = enzyme::InitOp::create(builder, func.getLoc(), slotType);
      funcToSlot[func] = slotOp;
      slot = slotOp;
    } else {
      slot = it->second->getResult(0);
    }

    builder.setInsertionPoint(jitCall);
    auto token =
        enzyme::GetOp::create(builder, jitCall.getLoc(), tokenType, slot);

    SmallVector<Value> operands(jitCall.getOperands());
    SmallVector<Type> resultTypes(jitCall.getResultTypes());

    operands.push_back(token);
    resultTypes.push_back(tokenType);

    SmallVector<Attribute> aliases;
    for (Attribute attr : jitCall.getOutputOperandAliases()) {
      auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
      if (jitCall.getNumResults() == 1 && alias.getOutputTupleIndices().empty())
        alias = stablehlo::OutputOperandAliasAttr::get(
            builder.getContext(), {0}, alias.getOperandIndex(),
            alias.getOperandTupleIndices());
      aliases.push_back(alias);
    }

    SmallVector<int64_t> tokenResultIndices;
    if (resultTypes.size() != 1)
      tokenResultIndices.push_back(jitCall.getNumResults());
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        builder.getContext(), tokenResultIndices, jitCall.getNumOperands(),
        /*operandTupleIndices=*/{}));

    auto appendAttribute = [&](ArrayAttr attrs, Attribute attr) -> ArrayAttr {
      if (!attrs)
        return nullptr;
      SmallVector<Attribute> extended(attrs.getValue());
      extended.push_back(attr);
      return builder.getArrayAttr(extended);
    };

    Attribute operandLayouts = jitCall.getOperandLayoutsAttr();
    if (operandLayouts)
      operandLayouts = appendAttribute(cast<ArrayAttr>(operandLayouts),
                                       builder.getIndexTensorAttr({}));

    auto newJitCall = enzymexla::JITCallOp::create(
        builder, jitCall.getLoc(), resultTypes, jitCall.getFnAttr(), operands,
        jitCall.getBackendConfigAttr(), operandLayouts,
        appendAttribute(jitCall.getArgAttrsAttr(),
                        builder.getDictionaryAttr({})),
        appendAttribute(jitCall.getResAttrsAttr(),
                        builder.getDictionaryAttr({})),
        builder.getArrayAttr(aliases), jitCall.getXlaSideEffectFreeAttr());
    enzyme::SetOp::create(builder, jitCall.getLoc(), slot,
                          newJitCall.getResults().back());

    for (auto [oldRes, newRes] :
         llvm::zip(jitCall.getResults(), newJitCall.getResults())) {
      oldRes.replaceAllUsesWith(newRes);
    }

    jitCall->erase();
  }

  SmallVector<PromotableAllocationOpInterface> allocators;
  for (auto &entry : funcToSlot)
    allocators.push_back(cast<PromotableAllocationOpInterface>(entry.second));

  DataLayout dataLayout = DataLayout::closest(allocators.front());
  DominanceInfo dominance;
  if (failed(
          tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance)))
    return funcToSlot.begin()->first->emitError(
        "failed to promote JIT call token slots");

  return success();
}
