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
#define GEN_PASS_DEF_LEGALIZEMPITONCCLPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

struct LegalizeMpiCommRankOpToNccl
    : public OpConversionPattern<comm::MpiCommRankOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();
  }
}; // struct LegalizeMpiCommRankOpToNccl


struct LegalizeMpiToNcclPass
    : public comm::impl::LegalizeMpiToNcclPassBase<LegalizeMpiToNcclPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    ConversionTarget target(*context);

    target.addLegalDialect<stablehlo::StablehloDialect,
                           enzymexla::EnzymeXLADialect,
                           mlir::LLVM::LLVMDialect>();
    target.addIllegalDialect<comm::CommDialect>();
    target.addLegalOp<comm::NcclCommSplitOp, comm::NcclCommFinalizeOp,
                      comm::NcclCommDestroyOp, comm::NcclCommAbortOp,
                      comm::NcclCommCountOp, comm::NcclCommCuDeviceOp,
                      comm::NcclCommUserRankOp, comm::NcclAllReduceOp,
                      comm::NcclBroadcastOp, comm::NcclSendOp,
                      comm::NcclRecvOp>();

    comm::MpiToNcclTypeConverter converter;

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    RewritePatternSet patterns(context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, converter);

    patterns.add<LegalizeMpiCommRankOpToNccl>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
