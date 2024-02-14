//===- EnzymeWrapPass.cpp - Replace calls with their derivatives ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to create wrapper functions which differentiate
// ops.
//===----------------------------------------------------------------------===//

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace {
struct ArithRaisingPass
    : public ArithRaisingPassBase<ArithRaisingPass> {

  void runOnOperation() override {

    Operation *symbolOp = nullptr;
    if (infn != "")
      symbolOp = symbolTable.lookupSymbolIn<Operation *>(
          getOperation(), StringAttr::get(getOperation()->getContext(), infn));
    else {
      for (auto &op : getOperation()->getRegion(0).front()) {
        auto fn = dyn_cast<FunctionOpInterface>(symbolOp);
        if (!fn)
          continue;
        assert(symbolOp == nullptr);
        symbolOp = &op;
      }
    }
    auto fn = cast<FunctionOpInterface>(symbolOp);
    SmallVector<StringRef, 1> split;
    StringRef(argTys.getValue().data(), argTys.getValue().size())
        .split(split, ',');
    std::vector<DIFFE_TYPE> constants;
    for (auto &str : split) {
      if (str == "enzyme_dup")
        constants.push_back(DIFFE_TYPE::DUP_ARG);
      else if (str == "enzyme_const")
        constants.push_back(DIFFE_TYPE::CONSTANT);
      else if (str == "enzyme_dupnoneed")
        constants.push_back(DIFFE_TYPE::DUP_NONEED);
      else if (str == "enzyme_out")
        constants.push_back(DIFFE_TYPE::OUT_DIFF);
      else {
        llvm::errs() << "unknown argument activity to parse, found: '" << str
                     << "'\n";
        assert(0 && " unknown constant");
      }
    }

    DIFFE_TYPE retType = retTy.getValue();
    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);

    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getFunctionBody().getArguments()) {
      (void)a;
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    FunctionOpInterface newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr);
    if (outfn == "") {
      fn->erase();
    } else {
      SymbolTable::setSymbolName(cast<FunctionOpInterface>(newFunc),
                                 (std::string)outfn);
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createArithRaisingPass() {
  return std::make_unique<ArithRaisingPass>();
}
} // namespace enzyme
} // namespace mlir

