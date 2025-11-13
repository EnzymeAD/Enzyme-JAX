//===- TritonExtAutoDiffOpInterfaceImpl.cpp - Interface external model ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the MLIR triton_ext dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"

#include "src/enzyme_ad/jax/Dialect/TritonExt/Dialect.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;

namespace {

static std::optional<unsigned>
findAliasedOperand(ArrayAttr outputOperandAliases, unsigned outputIndex) {
  for (auto attr : outputOperandAliases) {
    auto alias = cast<stablehlo::OutputOperandAliasAttr>(attr);
    if (alias.getOutputTupleIndices()[0] != outputIndex)
      continue;
    assert(alias.getOutputTupleIndices().size() == 1);
    assert(alias.getOperandTupleIndices().empty());
    return alias.getOperandIndex();
  }
  return std::nullopt;
}

class AutoDiffTritonCallFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffTritonCallFwd,
                                                triton_ext::TritonCallOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    DerivativeMode mode = DerivativeMode::ForwardMode;

    auto callOp = cast<triton_ext::TritonCallOp>(orig);

    for (auto [i, res] : llvm::enumerate(callOp->getResults())) {
      if (!isa<TensorType>(res.getType())) {
        orig->emitError() << "unsupported triton kernel call with non array "
                             "return at return #"
                          << i << " of type " << res.getType() << ".";
        return failure();
      }
    }

    auto output_operand_aliases = callOp.getOutputOperandAliases();
    auto operandLayouts = callOp.getOperandLayouts();
    auto resultLayouts = callOp.getResultLayouts();

    // if (!output_operand_aliases.empty()) {
    //   orig->emitError() << "TODO: support output operand aliases";
    //   return failure();
    // }

    Operation *callee =
        SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getFn());
    auto fn = cast<FunctionOpInterface>(callee);

    size_t width = gutils->width;

    int numInputs = callOp.getInputs().size();
    int narg = numInputs + orig->getNumResults();
    int nret = 0;

    SmallVector<Type> retTypes;

    std::vector<DIFFE_TYPE> RetActivity;

    // Unless there is aliasing, returns values arguments are assumed to
    // appended to the argument list in the triton kernel.
    SmallVector<unsigned> operandIndexMap;
    SmallVector<unsigned> resultIndexMap;

    unsigned argCnt = 0;

    std::vector<DIFFE_TYPE> ArgActivity;
    for (auto arg : callOp.getInputs()) {
      auto act = gutils->isConstantValue(arg) ? DIFFE_TYPE::CONSTANT
                                              : DIFFE_TYPE::DUP_ARG;
      operandIndexMap.push_back(argCnt);
      ArgActivity.push_back(act);
      argCnt++;
      if (act == DIFFE_TYPE::DUP_ARG)
        argCnt++;
    }

    for (auto [i, res] : llvm::enumerate(callOp.getResults())) {
      auto aliasedOperandIndex = findAliasedOperand(output_operand_aliases, i);
      if (!aliasedOperandIndex.has_value()) {
        auto act = gutils->isConstantValue(res) ? DIFFE_TYPE::CONSTANT
                                                : DIFFE_TYPE::DUP_ARG;
        ArgActivity.push_back(act);

        resultIndexMap.push_back(argCnt);

        argCnt++;
        if (act == DIFFE_TYPE::DUP_ARG)
          argCnt++;
      } else {
        resultIndexMap.push_back(operandIndexMap[*aliasedOperandIndex]);
        narg--;
      }
    }

    std::vector<bool> returnPrimal(nret, true);
    std::vector<bool> returnShadow(nret, false);

    auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

    bool freeMemory = true;

    std::vector<bool> volatile_args(narg, false);

    auto forwardFn = gutils->Logic.CreateForwardDiff(
        fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, mode,
        freeMemory, width,
        /* addedType */ nullptr, type_args, volatile_args,
        /* augmented */ nullptr, gutils->omp, gutils->postpasses,
        gutils->verifyPostPasses, gutils->strongZero);

    SmallVector<Value> fwdArguments;
    SmallVector<Type> returnTypes;

    for (auto &&[arg, act] : llvm::zip(callOp.getOperands(), ArgActivity)) {
      fwdArguments.push_back(gutils->getNewFromOriginal(arg));
      if (act == DIFFE_TYPE::DUP_ARG)
        fwdArguments.push_back(gutils->invertPointerM(arg, builder));
    }

    SmallVector<Attribute> newOutputOperandAliases;

    unsigned naliased = 0;
    for (auto &&[i, res] : llvm::enumerate(callOp->getResults())) {
      auto aliasedOperandIndex = findAliasedOperand(output_operand_aliases, i);

      DIFFE_TYPE act;
      if (aliasedOperandIndex.has_value()) {
        naliased++;

        act = ArgActivity[*aliasedOperandIndex];

        auto newOperandIndex = operandIndexMap[*aliasedOperandIndex];
        int64_t newResultIndex = returnTypes.size();
        newOutputOperandAliases.push_back(
            stablehlo::OutputOperandAliasAttr::get(
                callOp.getContext(), ArrayRef<int64_t>{newResultIndex},
                newOperandIndex, ArrayRef<int64_t>{}));

        if (act == DIFFE_TYPE::DUP_ARG) {
          newOutputOperandAliases.push_back(
              stablehlo::OutputOperandAliasAttr::get(
                  callOp.getContext(), ArrayRef<int64_t>{newResultIndex + 1},
                  newOperandIndex + 1, ArrayRef<int64_t>{}));
        }
      } else {
        act = ArgActivity[i - naliased + numInputs];
      }

      returnTypes.push_back(res.getType());
      if (act == DIFFE_TYPE::DUP_ARG)
        returnTypes.push_back(
            cast<AutoDiffTypeInterface>(res.getType()).getShadowType(width));
    }

    SmallVector<FlatSymbolRefAttr, 2> nestedRefs = {
        FlatSymbolRefAttr::get(
            forwardFn->getParentOfType<mlir::ModuleOp>().getSymNameAttr()),
        FlatSymbolRefAttr::get(
            StringAttr::get(callOp.getContext(), forwardFn.getName()))};
    auto fnRef = SymbolRefAttr::get(
        callOp.getContext(),
        forwardFn->getParentOfType<triton_ext::TritonModuleOp>().getSymName(),
        nestedRefs);

    Value gridx = gutils->getNewFromOriginal(callOp.getGridx()),
          gridy = gutils->getNewFromOriginal(callOp.getGridy()),
          gridz = gutils->getNewFromOriginal(callOp.getGridz());

    Value clusterx = gutils->getNewFromOriginal(callOp.getClusterx()),
          clustery = gutils->getNewFromOriginal(callOp.getClustery()),
          clusterz = gutils->getNewFromOriginal(callOp.getClusterz());

    auto fwdCallOp = triton_ext::TritonCallOp::create(
        builder, callOp.getLoc(), TypeRange(returnTypes),
        /*fn*/ fnRef,

        gridx, gridy, gridz,

        clusterx, clustery, clusterz,

        ValueRange(fwdArguments),
        /* backendConfig */ StringAttr::get(callOp.getContext(), ""),
        callOp.getOperandLayoutsAttr(), callOp.getResultLayoutsAttr(),
        /* argAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
        /* resAttrs */ mlir::ArrayAttr::get(callOp.getContext(), {}),
        ArrayAttr::get(callOp.getContext(), newOutputOperandAliases),
        /* xla_side_effect_free */ nullptr);

    SmallVector<Value> primals;
    primals.reserve(callOp->getNumResults());

    naliased = 0;
    int fwdIndex = 0;
    for (auto &&[i, ret] : llvm::enumerate(callOp.getResults())) {
      auto fwdRet = fwdCallOp.getResult(fwdIndex);
      primals.push_back(fwdRet);

      fwdIndex++;

      auto aliasedOperandIndex = findAliasedOperand(output_operand_aliases, i);

      DIFFE_TYPE act;
      if (aliasedOperandIndex.has_value()) {
        act = ArgActivity[*aliasedOperandIndex];
        naliased++;
      } else {
        act = ArgActivity[i - naliased + numInputs];
      }

      if (act == DIFFE_TYPE::DUP_ARG) {
        gutils->setDiffe(ret, fwdCallOp.getResult(fwdIndex), builder);
        fwdIndex++;
      }
    }

    auto newOp = gutils->getNewFromOriginal(orig);
    gutils->replaceOrigOpWith(orig, primals);
    gutils->erase(newOp);

    return success();
  }
};

} // end anonymous namespace

void mlir::enzyme::registerTritonExtDialectAutoDiffInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            triton_ext::TritonExtDialect *) {
    triton_ext::TritonCallOp::attachInterface<AutoDiffTritonCallFwd>(*context);
  });
}
