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

// Helper function to extract constant scalar value (real/imag parts)
static bool extractConstantScalar(Value val, double &realPart,
                                  double &imagPart) {
  DenseElementsAttr attr;
  if (!matchPattern(val, m_Constant(&attr)))
    return false;

  auto valType = cast<RankedTensorType>(val.getType());
  auto elemType = valType.getElementType();

  if (auto complexType = dyn_cast<ComplexType>(elemType)) {
    // Complex scalar
    auto complexVal = attr.getSplatValue<std::complex<APFloat>>();
    realPart = complexVal.real().convertToDouble();
    imagPart = complexVal.imag().convertToDouble();
    return true;
  } else if (isa<FloatType>(elemType)) {
    // Real scalar
    realPart = attr.getSplatValue<APFloat>().convertToDouble();
    imagPart = 0.0;
    return true;
  }
  return false;
}

// Helper function to create operand and rank for scalar value
// Returns the operand to use and the rank (1 for empty placeholder, 0 for
// scalar)
static std::pair<Value, int64_t> createScalarOperand(PatternRewriter &rewriter,
                                                     Location loc,
                                                     Value originalVal,
                                                     bool useAttribute) {
  if (useAttribute) {
    // Create an empty 0-element tensor as placeholder
    auto emptyType = RankedTensorType::get(
        {0}, cast<RankedTensorType>(originalVal.getType()).getElementType());
    auto emptyTensor = stablehlo::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(emptyType, ArrayRef<Attribute>{}));
    return {emptyTensor, 1};
  }
  return {originalVal, 0};
}

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
    if (backend == "cuda")
      return matchAndRewriteCUDA(op, rewriter);

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

      LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
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
      LLVM_FALLTHROUGH;
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

  LogicalResult matchAndRewriteCUDA(enzymexla::SyrkOp op,
                                    PatternRewriter &rewriter) const {
    auto CType = cast<RankedTensorType>(op.getC().getType());
    auto rank = CType.getRank();

    bool isComplex = false;
    if (auto complex_type = dyn_cast<ComplexType>(CType.getElementType())) {
      isComplex = true;
    }

    if (isComplex && op.getTranspose() == enzymexla::LapackTranspose::adjoint) {
      return rewriter.notifyMatchFailure(
          op, "Complex matrix not supported for complex transpose");
    }

    bool transpose = op.getTranspose() != enzymexla::LapackTranspose::none;

    // Try to extract alpha and beta as constants
    double alphaReal = 0.0, alphaImag = 0.0;
    double betaReal = 0.0, betaImag = 0.0;
    bool useAlphaAttr =
        extractConstantScalar(op.getAlpha(), alphaReal, alphaImag);
    bool useBetaAttr = extractConstantScalar(op.getBeta(), betaReal, betaImag);

    // Build operands list - use empty tensors for constant alpha/beta
    SmallVector<Value> operands;
    operands.push_back(op.getA());
    operands.push_back(op.getC());

    SmallVector<int64_t> operandRanks = {rank, rank};

    auto [alphaOperand, alphaRank] =
        createScalarOperand(rewriter, op.getLoc(), op.getAlpha(), useAlphaAttr);
    operands.push_back(alphaOperand);
    operandRanks.push_back(alphaRank);

    auto [betaOperand, betaRank] =
        createScalarOperand(rewriter, op.getLoc(), op.getBeta(), useBetaAttr);
    operands.push_back(betaOperand);
    operandRanks.push_back(betaRank);

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{CType}, operands,
        rewriter.getStringAttr("reactant_cublas_syrk_ffi"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/
        rewriter.getDictionaryAttr({
            rewriter.getNamedAttr("transpose", rewriter.getBoolAttr(transpose)),
            rewriter.getNamedAttr(
                "uplo",
                rewriter.getBoolAttr(op.getUplo() == enzymexla::LapackUplo::U)),
            rewriter.getNamedAttr("use_alpha_attribute",
                                  rewriter.getBoolAttr(useAlphaAttr)),
            rewriter.getNamedAttr("use_beta_attribute",
                                  rewriter.getBoolAttr(useBetaAttr)),
            rewriter.getNamedAttr("alpha_real",
                                  rewriter.getF64FloatAttr(alphaReal)),
            rewriter.getNamedAttr("alpha_imag",
                                  rewriter.getF64FloatAttr(alphaImag)),
            rewriter.getNamedAttr("beta_real",
                                  rewriter.getF64FloatAttr(betaReal)),
            rewriter.getNamedAttr("beta_imag",
                                  rewriter.getF64FloatAttr(betaImag)),
        }),
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, operandRanks, SmallVector<bool>(4, true), rank),
        /*result_layouts*/
        getSHLOLayout(rewriter, {rank}, SmallVector<bool>(rank, true), rank),
        /*output_operand_aliases*/
        rewriter.getArrayAttr({
            stablehlo::OutputOperandAliasAttr::get(op.getContext(), {}, 1, {}),
        }));

    auto result = customCall.getResult(0);
    if (op.getFill() || op.getUplo() == enzymexla::LapackUplo::L) {
      result = stablehlo::copyTriangularPart(rewriter, result,
                                             enzymexla::LapackUplo::U);
    }
    rewriter.replaceAllUsesWith(op.getResult(), result);

    return success();
  }

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
      LLVM_FALLTHROUGH;
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

    patterns.add<SyrkOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzymexla::SyrkOp>(op)) {
        op->emitError("Failed to lower enzymexla::SyrkOp");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
