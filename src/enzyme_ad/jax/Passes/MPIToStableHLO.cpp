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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"


namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERMPITOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::mpi;
using namespace stablehlo;
using namespace mlir::enzymexla;

namespace {



struct InitOpLowering : public OpRewritePattern<mpi::InitOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::InitOp op, PatternRewriter &rewriter) const override {
        return failure();
        // ::mlir::ValueRange inputs{};
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op.getResultTypes(), inputs,
        //     rewriter.getStringAttr("mpi_init"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
    }
};

struct FinalizeOpLowering : public OpRewritePattern<mpi::FinalizeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::FinalizeOp op, PatternRewriter &rewriter) const override {
        return failure();
        // ::mlir::ValueRange inputs{};
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op.getResultTypes(), inputs,
        //     rewriter.getStringAttr("mpi_finalize"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
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
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::CommRankOp op, PatternRewriter &rewriter) const override {
        return failure();
        // ::mlir::ValueRange inputs{};
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op.getResultTypes(), inputs,
        //     rewriter.getStringAttr("mpi_comm_rank"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
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
    using OpRewritePattern::OpRewritePattern;
    mutable bool func_written = false;

    LogicalResult matchAndRewrite(mpi::SendOp op, PatternRewriter &rewriter) const override {
        auto jit_op = rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(op, op.getResultTypes(),
            rewriter.getStringAttr("mpi_send_func"),
            op.getOperands(),
            "",
            ::mlir::Attribute{},
            ::mlir::Attribute{},
            ::mlir::ArrayAttr{});

        if (!func_written) {
            auto ctx = rewriter.getContext();
            auto op_types = op->getOperandTypes();

            SmallVector<mlir::Type> types;
            types.push_back(mlir::LLVM::LLVMPointerType::get(ctx));
            for (auto itr = ++op_types.begin(); itr != op_types.end(); ++itr) 
                types.push_back(*itr);


            const auto func_type = FunctionType::get(ctx,
                op.getOperandTypes(),
                op->getResultTypes()
            );
            const auto mpi_func_type = LLVM::LLVMFunctionType::get(
                mlir::IntegerType::get(ctx, 32),
                types,
                false
            );
            
            auto module = ([op]() {
                auto h = op->getParentOp();
                while (auto parent = h->getParentOp())
                    h = parent;
                return llvm::dyn_cast<ModuleOp>(h);
            })();
            assert(module);
            auto module_block = &module.getBodyRegion().getBlocks().front();

            rewriter.setInsertionPoint( module_block, module_block->end());

            // first, create the MPI_Send symbol
            auto mpi_send = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(),
                "MPI_Send",
                mpi_func_type
            );

            // then create the wrapper function
            auto func_op = rewriter.create<func::FuncOp>(op.getLoc(),
                "mpi_send_func",
                func_type);

            auto entry_block = func_op.addEntryBlock();
            assert(entry_block);
            rewriter.setInsertionPoint(entry_block, entry_block->begin());

            auto operands = entry_block->getArguments();
            auto extract_op = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(),
                operands[0]);
            auto cast_op = rewriter.create<arith::IndexCastOp>(op.getLoc(), mlir::IntegerType::get(ctx, 64), extract_op);
            auto memref_ptr_op = rewriter.create<LLVM::IntToPtrOp>(op.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx), cast_op);

            SmallVector<mlir::Value> values;
            values.push_back(memref_ptr_op);
            values.insert(values.end(), operands.begin()+1, operands.end());

            
            rewriter.create<LLVM::CallOp>(op.getLoc(), 
                mpi_send,
                mlir::ValueRange{values}
            );
            rewriter.create<func::ReturnOp>(op.getLoc());

            func_written = true;
        }

        return success();
    }
};

struct RecvOpLowering : public OpRewritePattern<mpi::RecvOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::RecvOp op, PatternRewriter &rewriter) const override {
        return failure();
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op.getResultTypes(), op.getOperands(),
        //     rewriter.getStringAttr("mpi_recv"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
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
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::RetvalCheckOp op, PatternRewriter &rewriter) const override {
        return failure();
        // ::mlir::ValueRange inputs{};
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op->getResultTypes(), inputs,
        //     rewriter.getStringAttr("mpi_retval_check"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
    }
};

struct ErrorClassOpLowering : public OpRewritePattern<mpi::ErrorClassOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(mpi::ErrorClassOp op, PatternRewriter &rewriter) const override {
        return failure();
        // ::mlir::ValueRange inputs{};
        // rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        //     op, op->getResultTypes(), inputs,
        //     rewriter.getStringAttr("mpi_error_class"),
        //     rewriter.getBoolAttr(false),
        //     rewriter.getDictionaryAttr({}),
        //     CustomCallApiVersionAttr::get(
        //         rewriter.getContext(),
        //         mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        //     nullptr, ValueRange(), ValueRange(), ValueRange());
        // return success();
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct LowerMPIToStableHLOPass : public mlir::enzyme::impl::LowerMPIToStableHLOPassBase<LowerMPIToStableHLOPass> {
using LowerMPIToStableHLOPassBase::LowerMPIToStableHLOPassBase;
    void runOnOperation() override {
        using namespace mlir::enzyme::impl;
        auto& ctx = getContext();
        ctx.loadDialect<mlir::LLVM::LLVMDialect>();
        ctx.loadDialect<mpi::MPIDialect>();
        ctx.loadDialect<enzymexla::EnzymeXLADialect>();
        ctx.loadDialect<memref::MemRefDialect>();

        mlir::ConversionTarget target(getContext());
        // XLA can't handle MPI ops, so we must convert all MPI ops to `stablehlo.custom_call` ops
        target.template addIllegalDialect<mpi::MPIDialect>();
        target.addLegalDialect("enzymexla");
        target.addLegalDialect("llvm");
        target.addLegalDialect("memref");
        target.addLegalDialect("arith");
        target.addLegalDialect("func");

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
};
  
} // namespace

void mlir::enzyme::registerLowerMPIToStableHLOPassHere() {
    PassRegistration<LowerMPIToStableHLOPass>();
}