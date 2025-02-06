//===- MPIToStableHLO.cpp - Convert MPI ops to StableHLO custom_call ops --===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MPI ops to StableHLO custom_call ops.
//
//===----------------------------------------------------------------------===//

// NOTE we should be targetting libmpitrampoline ABI, since XLA already adds it
// as a dependency and it fix some issues with MPI ABI compatibility. In
// particular, MPI types defined in the standard are not ABI-stable, so we must
// use `uintptr_t` instead of `MPI_Comm`, `MPI_Request`, etc...
// TODO or can we use libmpitrampoline's ABI directly? i.e. `MPIABI_Comm`, ...

#include "Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/MPI/IR/MPI.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERMPITOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::mpi;
using namespace stablehlo;

namespace {
struct InitOpLowering : public OpRewritePattern<mpi::InitOp> {
    using OpRewritePattern<mpi::InitOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::InitOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_init"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

struct FinalizeOpLowering : public OpRewritePattern<mpi::FinalizeOp> {
    using OpRewritePattern<mpi::FinalizeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::FinalizeOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_finalize"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

// struct CommWorldOpLowering : public OpRewritePattern<mpi::CommWorldOp> {
//     using OpRewritePattern<mpi::CommWorldOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::CommWorldOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_comm_world"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

struct CommRankOpLowering : public OpRewritePattern<mpi::CommRankOp> {
    using OpRewritePattern<mpi::CommRankOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::CommRankOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_comm_rank"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

// struct CommSizeOpLowering : public OpRewritePattern<mpi::CommSizeOp> {
//     using OpRewritePattern<mpi::CommSizeOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::CommSizeOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_comm_size"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

// struct CommSplitOpLowering : public OpRewritePattern<mpi::CommSplitOp> {
//     using OpRewritePattern<mpi::CommSplitOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::CommSplitOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_comm_split"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

struct SendOpLowering : public OpRewritePattern<mpi::SendOp> {
    using OpRewritePattern<mpi::SendOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::SendOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_send"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

struct RecvOpLowering : public OpRewritePattern<mpi::RecvOp> {
    using OpRewritePattern<mpi::RecvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::RecvOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_recv"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

// struct ISendOpLowering : public OpRewritePattern<mpi::ISendOp> {
//     using OpRewritePattern<mpi::ISendOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::ISendOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_isend"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

// struct IRecvOpLowering : public OpRewritePattern<mpi::IRecvOp> {
//     using OpRewritePattern<mpi::IRecvOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::IRecvOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_irecv"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

// struct BarrierOpLowering : public OpRewritePattern<mpi::BarrierOp> {
//     using OpRewritePattern<mpi::BarrierOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::BarrierOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_barrier"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

// struct WaitOpLowering : public OpRewritePattern<mpi::WaitOp> {
//     using OpRewritePattern<mpi::WaitOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::WaitOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_wait"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

// struct AllReduceOpLowering : public OpRewritePattern<mpi::AllReduceOp> {
//     using OpRewritePattern<mpi::AllReduceOp>::OpRewritePattern;

//     LogicalResult matchAndRewrite(mpi::AllReduceOp op, PatternRewriter &rewriter) const override {
//         rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
//             op, op.getResultTypes(), op.getOperands(),
//             rewriter.getStringAttr("mpi_allreduce"),
//             rewriter.getBoolAttr(false),
//             rewriter.getDictionaryAttr({}),
//             CustomCallApiVersionAttr::get(
//                 rewriter.getContext(),
//                 mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
//             nullptr, ValueRange(), ValueRange(), ValueRange());
//         return success();
//     }
// };

struct RetvalCheckOpLowering : public OpRewritePattern<mpi::RetvalCheckOp> {
    using OpRewritePattern<mpi::RetvalCheckOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::RetvalCheckOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_retval_check"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

struct ErrorClassOpLowering : public OpRewritePattern<mpi::ErrorClassOp> {
    using OpRewritePattern<mpi::ErrorClassOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::ErrorClassOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
            op, op.getResultTypes(), op.getOperands(),
            rewriter.getStringAttr("mpi_error_class"),
            rewriter.getBoolAttr(false),
            rewriter.getDictionaryAttr({}),
            CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            nullptr, ValueRange(), ValueRange(), ValueRange());
        return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct LowerMPIToStableHLOPass : public LowerMPIToStableHLOPassBase<LowerMPIToStableHLOPass> {
using LowerMPIToStableHLOPassBase::LowerMPIToStableHLOPassBase;
    void runOnOperation() override {
        ConversionTarget target(getContext());

        // XLA can't handle MPI ops, so we must convert all MPI ops to `stablehlo.custom_call` ops
        target.addIllegalDialect<MPI::MPIDialect>();

        RewritePatternSet patterns(&getContext());
        patterns.add<
            InitOpLowering,
            FinalizeOpLowering,
            CommRankOpLowering,
            SendOpLowering,
            RecvOpLowering,
            RetvalCheckOpLowering,
            ErrorClassOpLowering
        >(&getContext());

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
}
} // namespace
