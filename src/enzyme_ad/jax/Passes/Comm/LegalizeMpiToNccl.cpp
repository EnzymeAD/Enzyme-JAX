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

struct LegalizeMpiConstantOpToNccl
    : public OpConversionPattern<comm::MpiConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.constant is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiConstantOpToNccl

struct LegalizeMpiCommSplitOpToNccl
    : public OpConversionPattern<comm::MpiCommSplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.comm_split is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiCommSplitOpToNccl

struct LegalizeMpiBarrierOpToNccl
    : public OpConversionPattern<comm::MpiBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.barrier is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiBarrierOpToNccl

struct LegalizeMpiSendOpToNccl : public OpConversionPattern<comm::MpiSendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclSendOp>(
        op, adaptor.getBuffer(), adaptor.getDest(), adaptor.getComm());
    return success();
  }
}; // struct LegalizeMpiSendOpToNccl

struct LegalizeMpiIsendOpToNccl : public OpConversionPattern<comm::MpiIsendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIsendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.isend is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiIsendOpToNccl

struct LegalizeMpiRecvOpToNccl : public OpConversionPattern<comm::MpiRecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclRecvOp>(
        op, op.getType(), adaptor.getSource(), adaptor.getComm());
    return success();
  }
}; // struct LegalizeMpiRecvOpToNccl

struct LegalizeMpiIrecvOpToNccl : public OpConversionPattern<comm::MpiIrecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIrecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.irecv is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiIrecvOpToNccl

struct LegalizeMpiWaitOpToNccl : public OpConversionPattern<comm::MpiWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.wait is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiWaitOpToNccl

struct LegalizeMpiWaitallOpToNccl
    : public OpConversionPattern<comm::MpiWaitallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.waitall is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiWaitallOpToNccl

struct LegalizeMpiAllreduceOpToNccl
    : public OpConversionPattern<comm::MpiAllreduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiAllreduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.allreduce is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiAllreduceOpToNccl

struct LegalizeMpiBcastOpToNccl : public OpConversionPattern<comm::MpiBcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto context = op->getContext();

    op.emitError(
        "MPI-to-NCCL lowering for comm.mpi.bcast is not yet implemented");
    return failure();
  }
}; // struct LegalizeMpiBcastOpToNccl

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

    // These always run on the host, so a NCCL lowering is not needed
    target.addLegalOp<comm::MpiCommRankOp, comm::MpiCommSizeOp>();

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

    patterns.add<LegalizeMpiConstantOpToNccl, LegalizeMpiCommSplitOpToNccl,
                 LegalizeMpiBarrierOpToNccl, LegalizeMpiSendOpToNccl,
                 LegalizeMpiIsendOpToNccl, LegalizeMpiRecvOpToNccl,
                 LegalizeMpiIrecvOpToNccl, LegalizeMpiWaitOpToNccl,
                 LegalizeMpiWaitallOpToNccl, LegalizeMpiAllreduceOpToNccl,
                 LegalizeMpiBcastOpToNccl>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
