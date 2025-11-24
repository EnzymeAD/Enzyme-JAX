// #include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
// #include "stablehlo/dialect/StablehloOps.h"
// #include "llvm/ADT/DynamicAPInt.h"
// #include "llvm/ADT/SetVector.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/Support/ErrorHandling.h"
// #include "llvm/Support/LogicalResult.h"
// #include "llvm/Support/MathExtras.h"
// #include <algorithm>
// #include <cstdint>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMPIPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

struct MPICommRankOpLowering
    : public OpRewritePattern<enzymexla::MPICommRankOp> {

  std::string backend;
  MPICommRankOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPICommRankOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto comm = op.getOperand();
    if (backend == "cpu") {
      // auto moduleOp = op->getParentOfType<ModuleOp>();
      // static int64_t fnNum = 0;

      // // auto blasIntType = rewriter.getIntegerType(blasIntWidth);
      // // auto llvmBlasIntType = typeConverter.convertType(blasIntType);
      // auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      // auto llvmVoidPtrType = LLVM::LLVMVoidType::get(ctx);

      // std::string fn = "MPI_Comm_Rank";
      
      // std::string fnName = fn + "wrapper_" + std::to_string(fnNum);
      // fnNum++;
      // {
      // }
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }




};

// //===----------------------------------------------------------------------===//
// // CommRankOpLowering
// //===----------------------------------------------------------------------===//

// struct CommRankOpLowering : public ConvertOpToLLVMPattern<mpi::CommRankOp> {
//   using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

//   LogicalResult
//   matchAndRewrite(mpi::CommRankOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     // get some helper vars
//     Location loc = op.getLoc();
//     MLIRContext *context = rewriter.getContext();
//     Type i32 = rewriter.getI32Type();

//     // ptrType `!llvm.ptr`
//     Type ptrType = LLVM::LLVMPointerType::get(context);

//     // grab a reference to the global module op:
//     auto moduleOp = op->getParentOfType<ModuleOp>();

//     auto mpiTraits = MPIImplTraits::get(moduleOp);
//     // get communicator
//     Value comm = mpiTraits->castComm(loc, rewriter, adaptor.getComm());

//     // LLVM Function type representing `i32 MPI_Comm_rank(ptr, ptr)`
//     auto rankFuncType =
//         LLVM::LLVMFunctionType::get(i32, {comm.getType(), ptrType});
//     // get or create function declaration:
//     LLVM::LLVMFuncOp initDecl = getOrDefineFunction(
//         moduleOp, loc, rewriter, "MPI_Comm_rank", rankFuncType);

//     // replace with function call
//     auto one = LLVM::ConstantOp::create(rewriter, loc, i32, 1);
//     auto rankptr = LLVM::AllocaOp::create(rewriter, loc, ptrType, i32, one);
//     auto callOp = LLVM::CallOp::create(rewriter, loc, initDecl,
//                                        ValueRange{comm, rankptr.getRes()});

//     // load the rank into a register
//     auto loadedRank =
//         LLVM::LoadOp::create(rewriter, loc, i32, rankptr.getResult());

//     // if retval is checked, replace uses of retval with the results from the
//     // call op
//     SmallVector<Value> replacements;
//     if (op.getRetval())
//       replacements.push_back(callOp.getResult());

//     // replace all uses, then erase op
//     replacements.push_back(loadedRank.getRes());
//     rewriter.replaceOp(op, replacements);

//     return success();
//   }
// };

// //===----------------------------------------------------------------------===//


struct LowerEnzymeXLAMPIPass
    : public enzyme::impl::LowerEnzymeXLAMPIPassBase<
          LowerEnzymeXLAMPIPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<MPICommRankOpLowering>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
