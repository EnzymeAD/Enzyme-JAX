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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
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
using namespace mlir::enzymexla;

namespace {

struct SendOpLowering : public OpRewritePattern<mpi::SendOp> {
  using OpRewritePattern::OpRewritePattern;

  mutable bool func_written = false;

  LogicalResult matchAndRewrite(mpi::SendOp op,
                                PatternRewriter &rewriter) const override {
    auto jit_op = rewriter.replaceOpWithNewOp<enzymexla::JITCallOp>(
        op, op.getResultTypes(), rewriter.getStringAttr("mpi_send_func"),
        op.getOperands(), "", ::mlir::Attribute{}, ::mlir::Attribute{},
        ::mlir::ArrayAttr{});

    if (!func_written) {
      auto ctx = rewriter.getContext();
      auto op_types = op->getOperandTypes();

      SmallVector<mlir::Type> mpi_send_operand_types{
          mlir::LLVM::LLVMPointerType::get(ctx), // const void *buf
          mlir::IntegerType::get(ctx, 32),       // int count
          mlir::IntegerType::get(ctx, 32),       // MPI_Datatype datatype
          mlir::IntegerType::get(ctx, 32),       // int dest
          mlir::IntegerType::get(ctx, 32),       // int tag
          mlir::IntegerType::get(ctx, 32),       // MPI_Comm comm
      };

      const auto func_type =
          FunctionType::get(ctx, op.getOperandTypes(), op->getResultTypes());
      const auto mpi_func_type = LLVM::LLVMFunctionType::get(
          mlir::IntegerType::get(ctx, 32), mpi_send_operand_types, false);

      auto module = ([op]() {
        auto h = op->getParentOp();
        while (auto parent = h->getParentOp())
          h = parent;
        return llvm::dyn_cast<ModuleOp>(h);
      })();
      assert(module);
      auto module_block = &module.getBodyRegion().getBlocks().front();

      rewriter.setInsertionPoint(module_block, module_block->end());

      // we create the MPI symbols here
      auto global = rewriter.create<LLVM::GlobalOp>(
          op.getLoc(), mlir::IntegerType::get(ctx, 32), true,
          LLVM::Linkage::Internal, "MPI_DataType",
          rewriter.getIntegerAttr(mlir::IntegerType::get(ctx, 32), 0xffff));
      // first, create the MPI_Send symbol
      auto mpi_send = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), "MPI_Send",
                                                        mpi_func_type);

      // then create the wrapper function
      auto func_op = rewriter.create<func::FuncOp>(op.getLoc(), "mpi_send_func",
                                                   func_type);

      auto entry_block = func_op.addEntryBlock();
      assert(entry_block);
      rewriter.setInsertionPoint(entry_block, entry_block->begin());

      auto operands = entry_block->getArguments();
      auto memref_ptr_op = rewriter.create<enzymexla::Memref2PointerOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx), operands[0]);
      auto global_addr =
          rewriter.create<LLVM::AddressOfOp>(op.getLoc(), global);
      auto global_load = rewriter.create<LLVM::LoadOp>(
          op.getLoc(), global.getType(), global_addr);

      SmallVector<mlir::Value> values{{
          memref_ptr_op,
          global_load,
          global_load,
          global_load,
          global_load,
          global_load,
      }};

      rewriter.create<LLVM::CallOp>(op.getLoc(), mpi_send,
                                    mlir::ValueRange{values});
      rewriter.create<func::ReturnOp>(op.getLoc());

      func_written = true;
    }

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct LowerMPIToStableHLOPass
    : public mlir::enzyme::impl::LowerMPIToStableHLOPassBase<
          LowerMPIToStableHLOPass> {
  using LowerMPIToStableHLOPassBase::LowerMPIToStableHLOPassBase;
  void runOnOperation() override {
    using namespace mlir::enzyme::impl;
    auto &ctx = getContext();
    SymbolTable symTable(getOperation());

    ctx.loadDialect<mlir::LLVM::LLVMDialect>();
    ctx.loadDialect<mpi::MPIDialect>();
    ctx.loadDialect<enzymexla::EnzymeXLADialect>();
    ctx.loadDialect<memref::MemRefDialect>();

    mlir::ConversionTarget target(getContext());
    // XLA can't handle MPI ops, so we must convert all MPI ops to
    // `stablehlo.custom_call` ops
    target.template addIllegalDialect<mpi::MPIDialect>();
    target.addLegalDialect("enzymexla");
    target.addLegalDialect("llvm");
    target.addLegalDialect("memref");
    target.addLegalDialect("arith");
    target.addLegalDialect("func");

    RewritePatternSet patterns(&getContext());
    patterns.add<SendOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::enzyme::registerLowerMPIToStableHLOPassHere() {
  PassRegistration<LowerMPIToStableHLOPass>();
}