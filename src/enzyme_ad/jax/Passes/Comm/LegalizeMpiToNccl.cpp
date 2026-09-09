#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
#include "src/enzyme_ad/jax/Passes/Comm/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LEGALIZECOMMMPITONCCLPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

struct LegalizeCommMpiToNcclPass
    : public comm::impl::LegalizeCommMpiToNcclPassBase<LegalizeCommMpiToNcclPass> {
  using Base::Base;

  void runOnOperation() override {
  }
};
