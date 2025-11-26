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
#include <mlir/IR/BuiltinAttributes.h>

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
    auto AType = cast<RankedTensorType>(op.getA().getType());
    auto nBatchDims = AType.getRank() - 2;

    if (nBatchDims == 0) {
      if (backend == "cpu") {
        return matchAndRewriteCPU(op, rewriter);
      } else if (backend == "cuda") {
        return matchAndRewriteCUDA(op, rewriter);
      }
    }

    return matchAndRewriteFallback(op, rewriter);
  }

  enum CopyMode { NOT_NEEDED, COPY, TRANSPOSE };

  void resolveUplo(enzymexla::SyrkOp op, enzymexla::LapackUplo &customCallUplo,
                   CopyMode &needsCopy) const {
    switch (op.getUplo()) {
    case enzymexla::LapackUplo::F:
      customCallUplo = standardizeUplo(op.getOutputUplo());
      needsCopy = op.getOutputUplo() == enzymexla::LapackUplo::F
                      ? CopyMode::COPY
                      : CopyMode::NOT_NEEDED;
      break;
    case enzymexla::LapackUplo::L:
      customCallUplo = op.getUplo();
      switch (op.getOutputUplo()) {
      case enzymexla::LapackUplo::F:
        needsCopy = CopyMode::COPY;
        break;
      case enzymexla::LapackUplo::L:
        needsCopy = CopyMode::NOT_NEEDED;
        break;
      case enzymexla::LapackUplo::U:
        needsCopy = CopyMode::TRANSPOSE;
        break;
      }
      break;
    case enzymexla::LapackUplo::U:
      customCallUplo = op.getUplo();
      switch (op.getOutputUplo()) {
      case enzymexla::LapackUplo::F:
        needsCopy = CopyMode::COPY;
        break;
      case enzymexla::LapackUplo::L:
        needsCopy = CopyMode::TRANSPOSE;
        break;
      case enzymexla::LapackUplo::U:
        needsCopy = CopyMode::NOT_NEEDED;
        break;
      }
      break;
    }
  }

  LogicalResult matchAndRewriteCPU(enzymexla::SyrkOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto AType = cast<RankedTensorType>(op.getA().getType());

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

    CopyMode needsCopy;
    enzymexla::LapackUplo customCallUplo;
    resolveUplo(op, customCallUplo, needsCopy);

    // generate the func.funcOp that calls the blas function
    static int64_t fn_counter = 0;
    std::string funcFnName = blasFnWrapper + "_" + std::to_string(fn_counter++);

    // Pass A and C in row-major format (isColMajor = false) to avoid layout
    // transforms. This requires flipping uplo and transpose parameters below,
    // similar to the CUDA FFI implementation in xla_ffi.cpp.
    // operandRanks: {uplo, trans, n, k, alpha, A, lda, beta, C, ldc}
    SmallVector<bool> isColMajorArr = {false, false, false, false, false,
                                       false, false, false, false, false};
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
          stablehlo::GetDimensionSizeOp::create(
              rewriter, op.getLoc(), A,
              op.getTranspose() != enzymexla::LapackTranspose::none));
      auto kSize = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(
              rewriter, op.getLoc(), A,
              op.getTranspose() == enzymexla::LapackTranspose::none));

      // For row-major format, lda is the trailing dimension (columns in shape)
      // This is dimension 1 for a 2D matrix
      auto lda = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), A, 1));
      auto ldc = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), C, 1));

      // We flip uplo here because C is passed in row-major format.
      // Row-major C is equivalent to C^T in column-major, and since C is
      // symmetric, this means we need to swap upper/lower triangular.
      auto uploConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(
              uint8Type,
              customCallUplo == enzymexla::LapackUplo::U ? 'L' : 'U')));
      // We intentionally flip transpose here, this allows us to pass in
      // the data as a row-major format without paying the cost of
      // layout transformation to a col-major (which CPU BLAS uses)
      auto transConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), uint8Type,
          cast<ElementsAttr>(makeAttr(
              uint8Type, op.getTranspose() == enzymexla::LapackTranspose::none
                             ? 'T'
                             : 'N')));

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

    Value result = callOp.getResult(0);
    switch (needsCopy) {
    case CopyMode::COPY:
      result = stablehlo::copyTriangularPart(rewriter, result, customCallUplo);
      break;
    case CopyMode::TRANSPOSE:
      result = stablehlo::TransposeOpCreate(rewriter, op.getLoc(), result,
                                            ArrayRef<int64_t>{1, 0});
      break;
    case CopyMode::NOT_NEEDED:
      break;
    }
    rewriter.replaceAllUsesWith(op.getResult(), result);

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::SyrkOp op,
                                    PatternRewriter &rewriter) const {
    auto CType = cast<RankedTensorType>(op.getC().getType());
    auto rank = CType.getRank();

    // Try to extract alpha and beta as constants
    double alphaReal = 0.0, alphaImag = 0.0;
    double betaReal = 0.0, betaImag = 0.0;
    bool useAlphaAttr =
        extractConstantScalar(op.getAlpha(), alphaReal, alphaImag);
    bool useBetaAttr = extractConstantScalar(op.getBeta(), betaReal, betaImag);

    CopyMode needsCopy;
    enzymexla::LapackUplo customCallUplo;
    resolveUplo(op, customCallUplo, needsCopy);

    // Build operands list - use empty tensors for constant alpha/beta
    SmallVector<Value> operands;
    SmallVector<int64_t> operandRanks;
    SmallVector<bool> areColMajor;

    auto A = op.getA();
    auto C = op.getC();

    auto [alphaOperand, alphaRank] =
        createScalarOperand(rewriter, op.getLoc(), op.getAlpha(), useAlphaAttr);
    operands.push_back(alphaOperand);
    operandRanks.push_back(alphaRank);

    auto [betaOperand, betaRank] =
        createScalarOperand(rewriter, op.getLoc(), op.getBeta(), useBetaAttr);
    operands.push_back(betaOperand);
    operandRanks.push_back(betaRank);

    StringAttr customCallTarget;
    ArrayAttr aliases;

    SmallVector<NamedAttribute> configAttrs = {
        rewriter.getNamedAttr(
            "transpose",
            rewriter.getBoolAttr(op.getTranspose() !=
                                 enzymexla::LapackTranspose::none)),
        rewriter.getNamedAttr(
            "uplo",
            rewriter.getBoolAttr(customCallUplo == enzymexla::LapackUplo::U)),
        rewriter.getNamedAttr("use_alpha_attribute",
                              rewriter.getBoolAttr(useAlphaAttr)),
        rewriter.getNamedAttr("alpha_real",
                              rewriter.getF64FloatAttr(alphaReal)),
        rewriter.getNamedAttr("alpha_imag",
                              rewriter.getF64FloatAttr(alphaImag))};

    if (matchPattern(betaOperand, m_AnyZeroFloat()) ||
        matchPattern(betaOperand, m_Zero()) ||
        matchPattern(op.getC(), m_AnyZeroFloat()) ||
        matchPattern(op.getC(), m_Zero())) {
      customCallTarget =
          rewriter.getStringAttr("reactant_cublas_syrk_no_c_ffi");
      operands = {A, alphaOperand};
      operandRanks = {rank, alphaRank};
      aliases = rewriter.getArrayAttr({});
      areColMajor = {false, false};
    } else {
      customCallTarget = rewriter.getStringAttr("reactant_cublas_syrk_ffi");
      operands = {A, C, alphaOperand, betaOperand};
      operandRanks = {rank, rank, alphaRank, betaRank};

      configAttrs.push_back(rewriter.getNamedAttr(
          "use_beta_attribute", rewriter.getBoolAttr(useBetaAttr)));
      configAttrs.push_back(rewriter.getNamedAttr(
          "beta_real", rewriter.getF64FloatAttr(betaReal)));
      configAttrs.push_back(rewriter.getNamedAttr(
          "beta_imag", rewriter.getF64FloatAttr(betaImag)));

      aliases = rewriter.getArrayAttr(
          {stablehlo::OutputOperandAliasAttr::get(op.getContext(), {}, 1, {})});
      areColMajor = {false, false, true, true};
    }

    DictionaryAttr backendConfig = rewriter.getDictionaryAttr(configAttrs);

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{CType}, operands, customCallTarget,
        /*has_side_effect*/ nullptr,
        /*backend_config*/ backendConfig,
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, operandRanks, areColMajor, rank),
        /*result_layouts*/
        getSHLOLayout(rewriter, {rank}, SmallVector<bool>(rank, false), rank),
        /*output_operand_aliases*/ aliases);

    Value result = customCall.getResult(0);
    switch (needsCopy) {
    case CopyMode::COPY:
      result = stablehlo::copyTriangularPart(rewriter, result, customCallUplo);
      break;
    case CopyMode::TRANSPOSE:
      result = stablehlo::TransposeOpCreate(rewriter, op.getLoc(), result,
                                            ArrayRef<int64_t>{1, 0});
      break;
    case CopyMode::NOT_NEEDED:
      break;
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
    if (!stablehlo::IsTensorFilled(C)) {
      // If the tensor is not filled, we copy to the non-uplo region for safety
      C = stablehlo::copyTriangularPart(rewriter, C, op.getUplo());
      if (!C) {
        return failure();
      }
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
      LLVM_FALLTHROUGH;
    case enzymexla::LapackTranspose::transpose:
      dotDims = stablehlo::DotDimensionNumbersAttr::get(
          op.getContext(), batchDims, batchDims, {nBatchDims}, {nBatchDims});
      break;
    }

    auto AAT = stablehlo::DotGeneralOp::create(
        rewriter, op.getLoc(), cast<RankedTensorType>(op.getC().getType()),
        op.getA(), op.getA(), dotDims, nullptr, nullptr);

    auto aop =
        stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getAlpha(), AAT);
    auto bop = stablehlo::MulOpCreate(rewriter, op->getLoc(), op.getBeta(), C);

    auto res = stablehlo::AddOpCreate(rewriter, op->getLoc(), aop, bop);
    rewriter.replaceOp(op, res);
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
    config.setUseTopDownTraversal(true);
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
