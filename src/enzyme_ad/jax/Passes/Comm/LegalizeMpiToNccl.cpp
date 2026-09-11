#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    auto commAttr = dyn_cast<comm::MpiCommAttr>(op.getValueAttr());
    if (!commAttr || commAttr.getValue() != comm::MpiCommEnum::MPI_COMM_WORLD)
      return rewriter.notifyMatchFailure(
          op, "only MPI_COMM_WORLD can be lowered to an NCCL communicator");

    rewriter.replaceOpWithNewOp<comm::NcclConstantOp>(
        op, comm::NcclCommType::get(op.getContext()));
    return success();
  }
};

struct LegalizeMpiCommSplitOpToNccl
    : public OpConversionPattern<comm::MpiCommSplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclCommSplitOp>(
        op, comm::NcclCommType::get(op.getContext()), adaptor.getComm(),
        adaptor.getColor(), adaptor.getKey());
    return success();
  }
};

struct LegalizeMpiBarrierOpToNccl
    : public OpConversionPattern<comm::MpiBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclBarrierOp>(op, adaptor.getComm());
    return success();
  }
};

struct LegalizeMpiSendOpToNccl : public OpConversionPattern<comm::MpiSendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclSendOp>(
        op, adaptor.getBuffer(), adaptor.getDest(), adaptor.getComm());
    return success();
  }
};

struct LegalizeMpiRecvOpToNccl : public OpConversionPattern<comm::MpiRecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclRecvOp>(
        op, op.getType(), adaptor.getSource(), adaptor.getComm());
    return success();
  }
};

struct FoldMpiWaitOp : public OpRewritePattern<comm::MpiWaitOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comm::MpiWaitOp op,
                                PatternRewriter &rewriter) const override {
    Value request = op.getRequest();
    if (!request.hasOneUse())
      return rewriter.notifyMatchFailure(
          op, "expected an MPI request used only by this mpi.wait");

    if (auto isend = request.getDefiningOp<comm::MpiIsendOp>()) {
      rewriter.eraseOp(op);
      rewriter.replaceOpWithNewOp<comm::MpiSendOp>(
          isend, isend.getBuffer(), isend.getDest(), isend.getTag(),
          isend.getComm());
      return success();
    }

    if (auto irecv = request.getDefiningOp<comm::MpiIrecvOp>()) {
      rewriter.setInsertionPoint(irecv);
      auto recv = rewriter.create<comm::MpiRecvOp>(
          irecv.getLoc(), irecv.getBuffer().getType(), irecv.getSource(),
          irecv.getTag(), irecv.getComm());

      rewriter.eraseOp(op);
      rewriter.replaceAllUsesWith(irecv.getBuffer(), recv.getBuffer());
      rewriter.eraseOp(irecv);
      return success();
    }

    return rewriter.notifyMatchFailure(
        op, "expected an MPI request produced by mpi.isend or mpi.irecv");
  }
};

struct FoldMpiWaitallOp : public OpRewritePattern<comm::MpiWaitallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(comm::MpiWaitallOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> producers;
    Block *block = op->getBlock();

    for (Value request : op.getRequests()) {
      if (!request.hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "expected every MPI request to be used only by this "
                "mpi.waitall");

      Operation *producer = request.getDefiningOp();
      if (!isa_and_nonnull<comm::MpiIsendOp, comm::MpiIrecvOp>(producer))
        return rewriter.notifyMatchFailure(
            op, "expected every MPI request to be produced by mpi.isend or "
                "mpi.irecv");
      if (producer->getBlock() != block || !producer->isBeforeInBlock(op))
        return rewriter.notifyMatchFailure(
            op, "expected every MPI request producer to precede this "
                "mpi.waitall in the same block");
      producers.push_back(producer);
    }
    if (producers.empty())
      return rewriter.notifyMatchFailure(
          op, "expected mpi.waitall to have at least one associated mpi.isend "
              "or mpi.irecv");

    SmallVector<Operation *> orderedProducers;
    for (Operation &candidate : *block) {
      for (Operation *producer : producers) {
        if (&candidate == producer) {
          orderedProducers.push_back(producer);
          break;
        }
      }
    }

    // Don't allow other unrelated mpi ops within the scope of a request.
    // Could relax this to only disallow partially overlapping request
    // scopes since these semantics are impossible to represent in nccl
    // (nested request scopes should be ok however)
    for (Operation *candidate = orderedProducers.front();;
         candidate = candidate->getNextNode()) {
      bool isProducer = false;
      for (Operation *producer : orderedProducers) {
        if (candidate == producer) {
          isProducer = true;
          break;
        }
      }
      if (!isProducer && candidate->getName().getDialectNamespace() == "comm")
        return rewriter.notifyMatchFailure(
            op, "cannot group request producers interleaved with other "
                "communication operations");
      if (candidate == orderedProducers.back())
        break;
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(orderedProducers.front());
    rewriter.create<comm::NcclGroupStartOp>(loc);
    rewriter.eraseOp(op);

    Operation *lastReplacement = nullptr;
    for (Operation *producer : orderedProducers) {
      if (auto isend = dyn_cast<comm::MpiIsendOp>(producer)) {
        auto send = rewriter.replaceOpWithNewOp<comm::MpiSendOp>(
            isend, isend.getBuffer(), isend.getDest(), isend.getTag(),
            isend.getComm());
        lastReplacement = send;
        continue;
      }

      auto irecv = cast<comm::MpiIrecvOp>(producer);
      rewriter.setInsertionPoint(irecv);
      auto recv = rewriter.create<comm::MpiRecvOp>(
          irecv.getLoc(), irecv.getBuffer().getType(), irecv.getSource(),
          irecv.getTag(), irecv.getComm());
      rewriter.replaceAllUsesWith(irecv.getBuffer(), recv.getResult());
      rewriter.eraseOp(irecv);
      lastReplacement = recv;
    }

    rewriter.setInsertionPointAfter(lastReplacement);
    rewriter.create<comm::NcclGroupEndOp>(loc);
    return success();
  }
};

struct LegalizeMpiAllreduceOpToNccl
    : public OpConversionPattern<comm::MpiAllreduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiAllreduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    comm::NcclRedOpAttr reduceOp;
    switch (op.getReduceOp().getValue()) {
    case comm::MpiOpEnum::MPI_SUM:
      reduceOp = comm::NcclRedOpAttr::get(op.getContext(),
                                          comm::NcclRedOpEnum::ncclSum);
      break;
    case comm::MpiOpEnum::MPI_PROD:
      reduceOp = comm::NcclRedOpAttr::get(op.getContext(),
                                          comm::NcclRedOpEnum::ncclProd);
      break;
    case comm::MpiOpEnum::MPI_MIN:
      reduceOp = comm::NcclRedOpAttr::get(op.getContext(),
                                          comm::NcclRedOpEnum::ncclMin);
      break;
    case comm::MpiOpEnum::MPI_MAX:
      reduceOp = comm::NcclRedOpAttr::get(op.getContext(),
                                          comm::NcclRedOpEnum::ncclMax);
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "MPI reduction is not supported by NCCL");
    }

    rewriter.replaceOpWithNewOp<comm::NcclAllReduceOp>(
        op, op.getType(), adaptor.getSendbuf(), reduceOp, adaptor.getComm());
    return success();
  }
};

struct LegalizeMpiBcastOpToNccl : public OpConversionPattern<comm::MpiBcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<comm::NcclBroadcastOp>(
        op, op.getType(), adaptor.getInBuffer(), adaptor.getRoot(),
        adaptor.getComm());
    return success();
  }
};

struct LegalizeMpiToNcclPass
    : public comm::impl::LegalizeMpiToNcclPassBase<LegalizeMpiToNcclPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    // Normalize nonblocking MPI ops:
    // A matching mpi.wait folds it's associated mpi.isend/mpi.irecv into an
    // mpi.send/recv; a matching mpi.waitall does the same and creates NCCL
    // group boundaries. This leaves independent mpi.send/recv's for the
    // conversion to nccl.send/recv below.
    RewritePatternSet waitPatterns(context);
    waitPatterns.add<FoldMpiWaitOp, FoldMpiWaitallOp>(context);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(waitPatterns)))) {
      signalPassFailure();
      return;
    }

    if (getOperation()
            ->walk([&](Operation *op) -> WalkResult {
              if (auto wait = dyn_cast<comm::MpiWaitOp>(op)) {
                wait.emitError("expected a single-use request produced by "
                               "comm.mpi.isend or "
                               "comm.mpi.irecv");
                return WalkResult::interrupt();
              }
              if (auto waitall = dyn_cast<comm::MpiWaitallOp>(op)) {
                waitall.emitError("expected single-use requests produced by "
                                  "comm.mpi.isend or "
                                  "comm.mpi.irecv in the same "
                                  "communication-safe block range");
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(*context);

    target.addLegalDialect<stablehlo::StablehloDialect,
                           enzymexla::EnzymeXLADialect,
                           mlir::LLVM::LLVMDialect>();

    target.addIllegalDialect<comm::CommDialect>();

    // These always run on the host, so a NCCL lowering is not needed
    target.addLegalOp<comm::MpiCommRankOp, comm::MpiCommSizeOp>();

    target.addLegalOp<
        comm::NcclConstantOp, comm::NcclGroupStartOp, comm::NcclGroupEndOp,
        comm::NcclBarrierOp, comm::NcclCommSplitOp, comm::NcclCommFinalizeOp,
        comm::NcclCommDestroyOp, comm::NcclCommAbortOp, comm::NcclCommCountOp,
        comm::NcclCommCuDeviceOp, comm::NcclCommUserRankOp,
        comm::NcclAllReduceOp, comm::NcclBroadcastOp, comm::NcclSendOp,
        comm::NcclRecvOp>();

    comm::MpiToNcclTypeConverter converter;

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });

    RewritePatternSet patterns(context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);

    patterns.add<LegalizeMpiConstantOpToNccl, LegalizeMpiCommSplitOpToNccl,
                 LegalizeMpiBarrierOpToNccl, LegalizeMpiSendOpToNccl,
                 LegalizeMpiRecvOpToNccl, LegalizeMpiAllreduceOpToNccl,
                 LegalizeMpiBcastOpToNccl>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
