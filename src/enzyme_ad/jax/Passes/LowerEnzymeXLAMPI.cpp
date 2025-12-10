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

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidPtrType = LLVM::LLVMVoidType::get(ctx);

      auto i32Type = IntegerType::get(rewriter.getContext(), 32);

      std::string fn;
      fn = "MPI_Comm_rank";

      // For now we just hard code MPI_COMM_WORLD as the communicator. 
      // TODO make this more flexible
      std::string comm;
      comm = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper_MPI_Comm_rank LLVM function body
      std::string fnName = "enzymexla_wrapper_" + fn;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidPtrType, // void return type
            {llvmPtrType},   // parameter types TODO how to add {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}
            false);          // is variadic: false

        auto func =
            LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), fnName, funcType);

        Block *entryBlock = func.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // auto ptrSize =
        //     LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
        //                              rewriter.getIntegerAttr(blasIntType, 1));
        // auto mPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(), llvmPtrType,
        //                                    llvmBlasIntType, ptrSize, 0);
        // auto nPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(), llvmPtrType,
        //                                    llvmBlasIntType, ptrSize, 0);

        // auto mVal =
        //     LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
        //                              rewriter.getIntegerAttr(blasIntType, m));
        // auto nVal =
        //     LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
        //                              rewriter.getIntegerAttr(blasIntType, n));

        // LLVM::StoreOp::create(rewriter, op.getLoc(), mVal, mPtr);
        // LLVM::StoreOp::create(rewriter, op.getLoc(), nVal, nPtr);

        // Get the first (and only) argument of the function
        Value funcArg = entryBlock->getArgument(0);

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          comm
        );

        auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(ctx, fn),
                             ValueRange{addressOfComm, funcArg});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Comm_rank function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(fn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, 
            {llvmPtrType, llvmPtrType}, 
            false
        );

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), fn, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(comm)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          comm,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Get the result type (it's a tensor<i32>)
      auto resultType = op.getResult().getType();
      
      // Create a dense tensor constant with value 0
      auto elementType = cast<RankedTensorType>(resultType).getElementType();
      auto attr = DenseElementsAttr::get(
          cast<RankedTensorType>(resultType),
          rewriter.getIntegerAttr(elementType, 0));
      
      auto placeholderValue = rewriter.create<arith::ConstantOp>(
          op.getLoc(), attr);
      
      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, 
          op.getLoc(),
          TypeRange{resultType},
          mlir::FlatSymbolRefAttr::get(ctx, fnName),
          ValueRange{placeholderValue},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/nullptr,
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
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
      printf("FAILEEEEDDDDDD\n");
      signalPassFailure();
    }
  }
};
