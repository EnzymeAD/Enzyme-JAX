#pragma once

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-blas"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLABLASPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::stablehlo;

struct SymmOpLowering : public OpRewritePattern<enzymexla::SymmOp> {

  using OpRewritePattern<enzymexla::SymmOp>::OpRewritePattern;

  std::string backend;
  int64_t blasIntWidth;
  SymmOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::SymmOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return matchAndRewriteCPU(op, rewriter);

    // else if (backend == "cuda")
    //   return matchAndRewriteCUDA(op, rewriter);

    // else if (backend == "tpu")
    //   return matchAndRewriteTPU(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  LogicalResult matchAndRewriteCPU(enzymexla::SymmOp op,
                                    PatternRewriter &rewriter) const {
    llvm::errs() << "1\n";

    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);
    llvm::errs() << "2\n";
    
    Value a = op.getOperand(0);
    Value b = op.getOperand(1);
    Value c = op.getOperand(2);
    Value alpha_value = op.getAlpha();
    Value beta_value = op.getBeta();
    auto side_value = op.getSide() == enzymexla::LapackSide::left ? 'L' : 'R';
    auto uplo_value = op.getUplo() == enzymexla::LapackUplo::L ? 'L' : 'U';
    llvm::errs() << "3\n";
    
    auto aType = cast<RankedTensorType>(a.getType());
    auto bType = cast<RankedTensorType>(b.getType());
    auto cType = cast<RankedTensorType>(c.getType());
    llvm::errs() << "4\n";
    if (!aType || !bType || !cType)
{      llvm::errs() << "operand types not ranked tensor types\n";
      return rewriter.notifyMatchFailure(op, "operand types not ranked tensor types");

}    if (!aType.hasRank() || !bType.hasRank() || !cType.hasRank())
{      llvm::errs() << "expected ranked tensor types\n";
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
}
    if (aType.getRank() != 2 || bType.getRank() > 2 || cType.getRank() > 2)
{      llvm::errs() << "only 2D matrices supported for symm\n";
      return rewriter.notifyMatchFailure(op, "only 2D matrices supported for symm");
}

    llvm::errs() << "passed type checks\n";

    Type elementType = aType.getElementType();
    auto blasIntType = rewriter.getIntegerType(blasIntWidth);
    auto intType = RankedTensorType::get({}, blasIntType);
    auto uint8Type = RankedTensorType::get({}, rewriter.getIntegerType(8, false));
    auto llvmIntType = typeConverter.convertType(blasIntType);
    auto llvmElmType = typeConverter.convertType(elementType);
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);

    llvm::errs() << "5\n";

    std::string blasFn;
    if (auto prefix = lapackPrecisionPrefix(elementType)) {
      blasFn = "enzymexla_blas_" + *prefix + "symm_";
    } else {
      op->emitOpError() << "Unsupported element type: " << elementType;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    std::string blasFnWrapper = blasFn + "wrapper";
    llvm::errs() << "6\n";

    auto moduleOp = op->getParentOfType<ModuleOp>();

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(blasFn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidType,
                                      {
                                          llvmPtrType, // side
                                          llvmPtrType, // uplo
                                          llvmPtrType, // m
                                          llvmPtrType, // n
                                          llvmPtrType, // alpha
                                          llvmPtrType, // A
                                          llvmPtrType, // lda
                                          llvmPtrType, // B
                                          llvmPtrType, // ldb
                                          llvmPtrType, // beta
                                          llvmPtrType, // C
                                          llvmPtrType, // ldc
                                          llvmIntType,
                                          llvmIntType
                                      },
                                      false);
      rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), blasFn, funcType,
                                        LLVM::Linkage::External);
    }

    llvm::errs() << "7\n";


    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(blasFnWrapper)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          llvmVoidType,
          {
            llvmPtrType, // side
            llvmPtrType, // uplo
            llvmPtrType, // m
            llvmPtrType, // n
            llvmPtrType, // alpha
            llvmPtrType, // A
            llvmPtrType, // lda
            llvmPtrType, // B
            llvmPtrType, // ldb
            llvmPtrType, // beta
            llvmPtrType, // C
            llvmPtrType, // ldc
          },
          false);

    llvm::errs() << "8\n";
      

      auto funcOp =
          LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), blasFnWrapper,
                                   funcType, LLVM::Linkage::Private);
      rewriter.setInsertionPointToStart(funcOp.addEntryBlock(rewriter));

      SmallVector<Value> args(funcOp.getArguments().begin(),
                              funcOp.getArguments().end());
      auto const1 =
          LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmIntType,
                                   rewriter.getIntegerAttr(llvmIntType, 1));
      args.push_back(const1);
      args.push_back(const1);

      auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                                         SymbolRefAttr::get(ctx, blasFn), args);
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    llvm::errs() << "9\n";


    static int64_t fn_counter = 0;
    blasFnWrapper += "_" + std::to_string(fn_counter++);

    SmallVector<bool> isColMajorArr(12, true);
    SmallVector<int64_t> operandRanks = {0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 0};
    SmallVector<int64_t> outputRanks = {2};
    auto operandLayouts = getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);
    llvm::errs() << "12323\n";


    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, 10, {})); /*C*/

    func::FuncOp shloFunc;
    
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      SmallVector<Type> argTypes = {
        op.getA().getType(), // A
        op.getB().getType(), // B
        op.getC().getType(), // C
        op.getAlpha().getType(), // alpha
        op.getBeta().getType(),  // beta
      };
      SmallVector<Type> retTypes = {op.getC().getType()};

      auto calleeType = rewriter.getFunctionType(argTypes, retTypes);
      auto shloFunc = func::FuncOp::create(rewriter, op.getLoc(), blasFnWrapper, calleeType); 
      shloFunc.setPrivate();

      auto &entryBlock = *shloFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);
    llvm::errs() << "10\n";


      auto A = entryBlock.getArgument(0);
      auto B = entryBlock.getArgument(1);
      auto C = entryBlock.getArgument(2);
      auto alpha = entryBlock.getArgument(3);
      auto beta = entryBlock.getArgument(4);

      auto side = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(uint8Type, side_value)));
      auto uplo = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(uint8Type, uplo_value)));
      
      auto lda = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), intType,
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), A, 0));
      auto ldb = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), intType,
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), B, 0));
      auto ldc = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), intType,
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), C, 0));
      auto mSize = ldc;
      auto nSize = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), intType,
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), C, 1));
    llvm::errs() << "11\n";

      
      auto jitCall = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{op.getC().getType()},
        mlir::FlatSymbolRefAttr::get(ctx, blasFnWrapper), // TODO CHECK blasFnWrapper vs fn
        ValueRange{side, uplo, mSize, nSize, alpha, A, lda, B, ldb, beta, C, ldc},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

      func::ReturnOp::create(rewriter, op.getLoc(),
                        ValueRange{jitCall.getResult(0)}); // could be empty?
    }
    llvm::errs() << "12\n";

    assert(op.getA() && "A is null");
    assert(op.getB() && "B is null");
    assert(op.getC() && "C is null");
    assert(op.getAlpha() && "alpha is null");
    assert(op.getBeta() && "beta is null");

    moduleOp.verify();

    auto callOp = func::CallOp::create(
    rewriter, op.getLoc(), shloFunc,
    ValueRange{op.getA(), op.getB(), op.getC(), op.getAlpha(), op.getBeta()});
    llvm::errs() << "13\n";


    auto result = callOp.getResult(0);
    llvm::errs() << "14\n";

    rewriter.replaceAllUsesWith(op.getResult(), result);
    // rewriter.eraseOp(op); // remove?

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::SymmOp op,
                                    PatternRewriter &rewriter) const {
    return failure();
  }
  LogicalResult matchAndRewriteTPU(enzymexla::SymmOp op,
                                   PatternRewriter &rewriter) const {
    return failure();
  }
};

struct SyrkOpLowering : public OpRewritePattern<enzymexla::SyrkOp> {
  using OpRewritePattern<enzymexla::SyrkOp>::OpRewritePattern;

  SyrkOpLowering(std::string backend, int64_t blasIntWidth,
                 MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth){};

  LogicalResult matchAndRewrite(enzymexla::SyrkOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return matchAndRewriteCPU(op, rewriter);

    return matchAndRewriteFallback(op, rewriter);
  }

  LogicalResult matchAndRewriteCPU(enzymexla::SyrkOp op,
                                   PatternRewriter &rewriter) const {
    auto nBatchDims = cast<RankedTensorType>(op.getA().getType()).getRank() - 2;
    if (nBatchDims != 0) {
      return matchAndRewriteFallback(op, rewriter);
    }

    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto AType = cast<RankedTensorType>(op.getA().getType());

    bool isComplex = false;
    if (auto complexType = dyn_cast<ComplexType>(AType.getElementType())) {
      isComplex = true;
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto blasIntType = rewriter.getIntegerType(blasIntWidth);
    auto intType = RankedTensorType::get({}, blasIntType);
    auto uint8Type =
        RankedTensorType::get({}, rewriter.getIntegerType(8, false));
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
    auto llvmIntType = typeConverter.convertType(blasIntType);

    std::string blasFn;
    auto prefix = lapackPrecisionPrefix(AType.getElementType());
    if (prefix) {
      blasFn = "enzymexla_blas_" + *prefix + "syrk_";
    } else {
      op->emitOpError() << "Unsupported element type: "
                        << AType.getElementType();
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }
    std::string blasFnWrapper = blasFn + "wrapper";

    // declare BLAS function declarations if not present
    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(blasFn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          llvmVoidType,
          {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
           llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
           llvmIntType, llvmIntType},
          false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), blasFn, funcType,
                               LLVM::Linkage::External);
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(blasFnWrapper)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(
          llvmVoidType,
          {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType,
           llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
          false);

      auto funcOp =
          LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), blasFnWrapper,
                                   funcType, LLVM::Linkage::Private);
      rewriter.setInsertionPointToStart(funcOp.addEntryBlock(rewriter));

      SmallVector<Value> args(funcOp.getArguments().begin(),
                              funcOp.getArguments().end());
      auto const1 =
          LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmIntType,
                                   rewriter.getIntegerAttr(llvmIntType, 1));
      args.push_back(const1);
      args.push_back(const1);

      auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                                         SymbolRefAttr::get(ctx, blasFn), args);
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    enzymexla::LapackUplo uplo2 = op.getUplo(); // drop `F` uplo attribute
    char uploValue;
    switch (op.getUplo()) {
    case enzymexla::LapackUplo::U:
      uploValue = 'U';
      break;
    case enzymexla::LapackUplo::L:
      uploValue = 'L';
      break;
    case enzymexla::LapackUplo::F:
      uploValue = 'U';
      uplo2 = enzymexla::LapackUplo::U;
      break;
    }

    char transValue;
    switch (op.getTranspose()) {
    case enzymexla::LapackTranspose::none:
      transValue = 'N';
      break;
    case enzymexla::LapackTranspose::adjoint:
      if (isComplex) {
        llvm_unreachable("adjoint is not supported for complex matrices");
      }
      // adjoint for real matrices is the same as transpose
    case enzymexla::LapackTranspose::transpose:
      transValue = 'T';
      break;
    }

    // generate the func.funcOp that calls the blas function
    static int64_t fn_counter = 0;
    std::string funcFnName = blasFnWrapper + "_" + std::to_string(fn_counter++);

    SmallVector<bool> isColMajorArr(10, true);
    SmallVector<int64_t> operandRanks = {0, 0, 0, 0, 0, 2, 0, 0, 2, 0};
    SmallVector<int64_t> outputRanks = {2};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{}, 8, std::vector<int64_t>{}));

    func::FuncOp shloFunc;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      SmallVector<Type> argTypes = {AType, op.getC().getType(),
                                    op.getAlpha().getType(),
                                    op.getBeta().getType()};
      SmallVector<Type> retTypes = {op.getC().getType()};

      FunctionType calleeType = rewriter.getFunctionType(argTypes, retTypes);
      shloFunc =
          func::FuncOp::create(rewriter, op.getLoc(), funcFnName, calleeType);
      shloFunc.setPrivate();

      auto &entryBlock = *shloFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);

      auto A = entryBlock.getArgument(0);
      auto C = entryBlock.getArgument(1);
      auto alpha = entryBlock.getArgument(2);
      auto beta = entryBlock.getArgument(3);

      auto nSize = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), A,
                                                transValue == 'N' ? 0 : 1));
      auto kSize = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), A,
                                                transValue == 'N' ? 1 : 0));

      auto lda = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), A, 0));
      auto ldc = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), C, 0));

      auto uploConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(uint8Type, uploValue)));
      auto transConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(uint8Type, transValue)));

      // {uplo, trans, n, k, alpha, A, lda, beta, C, ldc}
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{op.getC().getType()},
          mlir::FlatSymbolRefAttr::get(ctx, blasFnWrapper),
          ValueRange{uploConst, transConst, nSize, kSize, alpha, A, lda, beta,
                     C, ldc},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/operandLayouts,
          /*result_layouts=*/resultLayouts,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/rewriter.getUnitAttr());

      func::ReturnOp::create(rewriter, op.getLoc(),
                             ValueRange{jitCall.getResult(0)});
    }

    auto callOp = func::CallOp::create(
        rewriter, op.getLoc(), shloFunc,
        ValueRange{op.getA(), op.getC(), op.getAlpha(), op.getBeta()});

    auto result = callOp.getResult(0);
    if (op.getFill()) {
      result = stablehlo::copyTriangularPart(rewriter, result, uplo2);
    }
    rewriter.replaceAllUsesWith(op.getResult(), result);

    return success();
  }

  // TODO: gpu lowering after we register the cublas functions via XLA FFI

  LogicalResult matchAndRewriteFallback(enzymexla::SyrkOp op,
                                        PatternRewriter &rewriter) const {
    auto AType = cast<RankedTensorType>(op.getA().getType());
    auto nBatchDims = AType.getRank() - 2;
    SmallVector<int64_t> batchDims(nBatchDims, 0);
    std::iota(batchDims.begin(), batchDims.end(), 0);

    Value C = op.getC();
    if (!matchPattern(C, m_Constant())) {
      // for safety we need to copy the uplo part into the other half of the
      // matrix
      C = stablehlo::copyTriangularPart(rewriter, C, op.getUplo());
      if (!C)
        return failure();
    }

    bool isComplex = false;
    if (auto complexType = dyn_cast<ComplexType>(AType.getElementType())) {
      isComplex = true;
    }

    // fallback to emitting a stablehlo.dot_general that computes:
    //   alpha * A * A^T + beta * C
    //   alpha * A^T * A + beta * C
    stablehlo::DotDimensionNumbersAttr dotDims;
    switch (op.getTranspose()) {
    case enzymexla::LapackTranspose::none:
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims + 1},
          {nBatchDims + 1});
      break;
    case enzymexla::LapackTranspose::adjoint:
      if (isComplex) {
        llvm_unreachable("adjoint is not supported for complex matrices");
      }
      // adjoint for real matrices is the same as transpose
    case enzymexla::LapackTranspose::transpose:
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims}, {nBatchDims});
      break;
    }

    auto AAT = stablehlo::DotGeneralOp::create(
        rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
        op.getA(), op.getA(), dotDims, nullptr, nullptr);

    auto alpha = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), cast<RankedTensorType>(AAT.getType()),
        op.getAlpha(), rewriter.getDenseI64ArrayAttr({}));

    auto lhs = stablehlo::MulOp::create(rewriter, op.getLoc(), alpha, AAT);

    auto beta = stablehlo::BroadcastInDimOp::create(
        rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
        op.getBeta(), rewriter.getDenseI64ArrayAttr({}));

    rewriter.replaceOpWithNewOp<stablehlo::AddOp>(
        op, lhs, stablehlo::MulOp::create(rewriter, op.getLoc(), beta, C));
    return success();
  }

private:
  std::string backend;
  int64_t blasIntWidth;
};

struct LowerEnzymeXLABLASPass
    : public enzyme::impl::LowerEnzymeXLABLASPassBase<LowerEnzymeXLABLASPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<SyrkOpLowering, SymmOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
