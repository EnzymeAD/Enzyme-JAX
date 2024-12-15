//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#define DEBUG_TYPE "lower-kernel"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;
using namespace mlir::enzymexla;
using namespace enzymexla;

using namespace stablehlo;

typedef size_t KernelContext[7];
typedef void XlaCustomCallStatus;

// See API details at https://github.com/openxla/xla/blob/37fb0612d36ac3d08ff984b1d61e4bc4dedf4809/xla/service/hlo.proto#L73
extern "C" void EnzymeGPUCustomCall(void* __restrict__ stream, void** __restrict__ buffers, KernelContext* __restrict__ opaqueptr,
                               size_t opaque_len, XlaCustomCallStatus* __restrict__ status) {
  auto ptr = (void(*)(void*, void**, size_t, size_t, size_t, size_t, size_t, size_t)) (opaqueptr[0][0]);

  size_t gridx = opaqueptr[0][1];
  size_t gridy = opaqueptr[0][2];
  size_t gridz = opaqueptr[0][3];

  size_t blockx = opaqueptr[0][4];
  size_t blocky = opaqueptr[0][5];
  size_t blockz = opaqueptr[0][6];

  ptr(stream, buffers, gridx, gridy, gridz, blockx, blocky, blockz);
}

void* CompileKernel(FunctionOpInterface op, bool jit) {
  if (!jit)
      return nullptr;

  op->emitError() << "JIT compilation of kernels not yet implemented";
  return nullptr;
};

namespace {

struct LowerKernelPass : public LowerKernelPassBase<LowerKernelPass> {

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());


    getOperation()->walk([&](KernelCallOp op) {
        mlir::ArrayAttr operand_layouts = op.getOperandLayouts() ? cast<mlir::ArrayAttr>(*op.getOperandLayouts()) : nullptr;
        mlir::ArrayAttr result_layouts = op.getResultLayouts() ? cast<mlir::ArrayAttr>(*op.getResultLayouts()) : nullptr;
        mlir::ArrayAttr output_operand_aliases = op.getOutputOperandAliases();

        KernelContext data;


        auto *symbolOp = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
        auto fn = cast<FunctionOpInterface>(symbolOp);

        // Compiled kernel goes here once ready
        data[0] = (size_t)CompileKernel(fn, jit);

        Value vals[6] = {op.getGridx(), op.getGridy(), op.getGridz(), op.getBlockx(), op.getBlocky(), op.getBlockz()};
        for (auto en : llvm::enumerate(vals)) {
          DenseIntElementsAttr stepAttr;
          if (!matchPattern(en.value(), m_Constant(&stepAttr))) {
            op->emitError() << "Cannot lower kernel with a grid/block size which is not a constant integer tensor";
            return;
          }
          if (stepAttr.size() != 1) {
            op->emitError() << "Cannot lower kernel with a grid/block size which is not a constant integer tensor of size 1";
            return;
          }
          auto val = (*stepAttr.begin()).getZExtValue();
          data[1+en.index()] = val;
        }

        std::string backendinfo((char*)&data, sizeof(data));
       
        OpBuilder rewriter(op);
        auto replacement = rewriter.create<stablehlo::CustomCallOp>(op.getLoc(),
            op.getResultTypes(),
            op.getInputs(),
            rewriter.getStringAttr("enzymexla_gpu"),
            /* has_side_effect*/rewriter.getBoolAttr(false),
            /*backend_config*/rewriter.getStringAttr(backendinfo),
            /* api_version*/CustomCallApiVersionAttr::get(
                                 rewriter.getContext(),mlir::stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
            /*calledcomputations*/nullptr,
            operand_layouts,
            result_layouts,
            output_operand_aliases
            );

        op.replaceAllUsesWith(replacement);
        op.erase();
    });

  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createLowerKernelPass() {
  return std::make_unique<LowerKernelPass>();
}
} // namespace enzyme
} // namespace mlir
