#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "lower-factorization"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERFACTORIZATIONPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

// https://github.com/jax-ml/jax/blob/48001a24cb74f311b51d8bcf0891437069db6b95/jax/_src/lax/linalg.py#L2792
SmallVector<int64_t> columnMajorMatrixLayout(int64_t ndim) {
  SmallVector<int64_t> layout = {ndim - 2, ndim - 1};
  for (int64_t i = ndim - 3; i >= 0; i--) {
    layout.push_back(i);
  }
  return layout;
}

SmallVector<int64_t> rowMajorMatrixLayout(int64_t ndim) {
  SmallVector<int64_t> layout;
  for (int64_t i = ndim - 1; i >= 0; i--) {
    layout.push_back(i);
  }
  return layout;
}

mlir::Attribute getSHLOLayout(PatternRewriter &rewriter, int64_t ndim,
                              bool isColMajor, int64_t maxNumDims) {
  if (isColMajor && ndim == maxNumDims) {
    return rewriter.getIndexTensorAttr(columnMajorMatrixLayout(ndim));
  }
  return rewriter.getIndexTensorAttr(rowMajorMatrixLayout(ndim));
}

mlir::ArrayAttr getSHLOLayout(PatternRewriter &rewriter,
                              SmallVector<int64_t> ndims,
                              SmallVector<bool> isColMajorArr,
                              int64_t maxNumDims) {
  SmallVector<mlir::Attribute> attrs;
  for (auto [ndim, isColMajor] : llvm::zip(ndims, isColMajorArr)) {
    attrs.push_back(getSHLOLayout(rewriter, ndim, isColMajor, maxNumDims));
  }
  return rewriter.getArrayAttr(attrs);
}

struct LUFactorizationOpLowering
    : public OpRewritePattern<enzymexla::LUFactorizationOp> {

  std::string backend;
  int64_t blasIntWidth;
  LUFactorizationOpLowering(std::string backend, int64_t blasIntWidth,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::LUFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    auto inputRank = inputShape.size();
    auto inputElementType =
        cast<RankedTensorType>(input.getType()).getElementType();

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;
    auto inputType = input.getType();

    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto pivotRank = pivotType.getRank();
    auto infoType = cast<RankedTensorType>(op.getResult(2).getType());
    auto infoRank = infoType.getRank();

    if (backend == "cpu") {
      if (numBatchDims > 0) {
        // TODO: Implement batched LU factorizations
        // If we are already linking against MKL we can call
        // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/getrf-batch-strided.html.
        // Or assume this as the call signature and rely on the downstream user
        // to correctly set the function pointers. JAX currently lowers to a
        // loop for CPU
        return rewriter.notifyMatchFailure(
            op, "Batched LU factorizations not yet implemented on CPU.");
      }

      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto blasIntType = rewriter.getIntegerType(blasIntWidth);
      auto llvmBlasIntType = typeConverter.convertType(blasIntType);
      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidPtrType = LLVM::LLVMVoidType::get(ctx);

      std::string lapackFn;
      if (inputElementType.isF32()) {
        lapackFn = "sgetrf_"; // single-precision float
      } else if (inputElementType.isF64()) {
        lapackFn = "dgetrf_"; // double-precision float
      } else if (auto complexType = dyn_cast<ComplexType>(inputElementType)) {
        auto elem = complexType.getElementType();
        if (elem.isF32()) {
          lapackFn = "cgetrf_"; // single-precision complex
        } else if (elem.isF64()) {
          lapackFn = "zgetrf_"; // double-precision complex
        } else {
          op->emitOpError() << "Unsupported complex element type: " << elem;
          return rewriter.notifyMatchFailure(
              op, "unsupported complex element type");
        }
      } else {
        op->emitOpError() << "Unsupported input element type: "
                          << inputElementType;
        return rewriter.notifyMatchFailure(op,
                                           "unsupported input element type");
      }
      lapackFn = "enzymexla_lapack_" + lapackFn;

      // Generate the LLVM function body
      std::string fnName = lapackFn + "wrapper_" + std::to_string(fnNum);
      fnNum++;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidPtrType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto ptrSize = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmBlasIntType,
            rewriter.getIntegerAttr(blasIntType, 1));
        auto mPtr = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmBlasIntType, ptrSize, 0);
        auto nPtr = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), llvmPtrType, llvmBlasIntType, ptrSize, 0);

        auto mVal = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmBlasIntType,
            rewriter.getIntegerAttr(blasIntType, m));
        auto nVal = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), llvmBlasIntType,
            rewriter.getIntegerAttr(blasIntType, n));

        auto mStore = rewriter.create<LLVM::StoreOp>(op.getLoc(), mVal, mPtr);
        auto nStore = rewriter.create<LLVM::StoreOp>(op.getLoc(), nVal, nPtr);

        rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                      SymbolRefAttr::get(ctx, lapackFn),
                                      ValueRange{
                                          mPtr,
                                          nPtr,
                                          func.getArgument(0),
                                          mPtr,
                                          func.getArgument(1),
                                          func.getArgument(2),
                                      });

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(lapackFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidPtrType,
                                        {llvmPtrType, llvmPtrType, llvmPtrType,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), lapackFn, funcType,
                                          LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      for (int i = 0; i < 3; ++i) {
        aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
            ctx, std::vector<int64_t>{i}, i, std::vector<int64_t>{}));
      }

      auto blasPivotType = RankedTensorType::get(
          pivotType.getShape(), rewriter.getIntegerType(blasIntWidth));
      auto blasInfoType = RankedTensorType::get(
          infoType.getShape(), rewriter.getIntegerType(blasIntWidth));

      auto pivot = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), blasPivotType,
          cast<ElementsAttr>(makeAttr(blasPivotType, -1)));
      auto info = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), blasInfoType,
          cast<ElementsAttr>(makeAttr(blasInfoType, -1)));

      SmallVector<bool> isColMajorArr = {true, true, true};
      SmallVector<int64_t> operandRanks = {inputRank, pivotRank, infoRank};
      SmallVector<int64_t> outputRanks = {inputRank, pivotRank, infoRank};
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{inputType, blasPivotType, blasInfoType},
          mlir::FlatSymbolRefAttr::get(ctx, fnName),
          ValueRange{input, pivot, info}, rewriter.getStringAttr(""),
          /*operand_layouts=*/
          getSHLOLayout(rewriter, operandRanks, isColMajorArr, inputRank),
          /*result_layouts=*/
          getSHLOLayout(rewriter, outputRanks, isColMajorArr, inputRank),
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/rewriter.getUnitAttr());

      rewriter.replaceAllUsesWith(op.getResult(0), jitCall.getResult(0));
      rewriter.replaceAllUsesWith(
          op.getResult(1), rewriter.create<stablehlo::ConvertOp>(
                               op.getLoc(), pivotType, jitCall.getResult(1)));
      rewriter.replaceAllUsesWith(
          op.getResult(2), rewriter.create<stablehlo::ConvertOp>(
                               op.getLoc(), infoType, jitCall.getResult(2)));

      return success();
    } else if (backend == "cuda") {
      SmallVector<Attribute> aliases = {stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{0}, 0, std::vector<int64_t>{})};

      SmallVector<bool> isColMajorArrOperands = {true};
      SmallVector<int64_t> operandRanks = {inputRank};
      SmallVector<bool> isColMajorArrOutputs = {true, true, true};
      SmallVector<int64_t> outputRanks = {inputRank, pivotRank, infoRank};

      rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
          op, TypeRange{inputType, pivotType, infoType}, ValueRange{input},
          rewriter.getStringAttr("cusolver_getrf_ffi"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ nullptr,
          /*api_version*/ nullptr,
          /*calledcomputations*/ nullptr,
          /*operand_layouts*/
          getSHLOLayout(rewriter, operandRanks, isColMajorArrOperands,
                        inputRank),
          /*result_layouts*/
          getSHLOLayout(rewriter, outputRanks, isColMajorArrOutputs, inputRank),
          /*output_operand_aliases*/ rewriter.getArrayAttr(aliases));

      return success();
    } else if (backend == "tpu") {
      SmallVector<int64_t> permutationShape;
      for (int i = 0; i < numBatchDims; i++) {
        permutationShape.push_back(inputShape[i]);
      }
      permutationShape.push_back(m);
      auto permutationType =
          RankedTensorType::get(permutationShape, pivotType.getElementType());

      // TPU returns (LU, pivots, permutation). info isn't returned. based on
      // how JAX operates, I am assuming info = 0 when there is a nan in the
      // output.
      auto customCall = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(), TypeRange{inputType, pivotType, permutationType},
          ValueRange{input}, rewriter.getStringAttr("LUFactorization"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ nullptr,
          /*api_version*/ nullptr,
          /*calledcomputations*/ nullptr,
          /*operand_layouts*/ nullptr,
          /*result_layouts*/ nullptr,
          /*output_operand_aliases*/ nullptr);

      // LAPACK returns 1-indexed pivots, while XLA returns 0-indexed pivots. We
      // make it consistent with LAPACK by adding 1 to the pivots.
      auto pivots1Indexed = rewriter.create<stablehlo::AddOp>(
          op.getLoc(),
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), pivotType,
              cast<ElementsAttr>(makeAttr(pivotType, 1))),
          customCall.getResult(1));

      auto isFinite = rewriter.create<stablehlo::IsFiniteOp>(
          op.getLoc(), customCall.getResult(0));

      SmallVector<int64_t> reductionDims;
      for (int i = numBatchDims; i < inputRank; i++) {
        reductionDims.push_back(i);
      }
      auto initValType = RankedTensorType::get({}, rewriter.getI1Type());
      auto initVal = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), initValType,
          cast<ElementsAttr>(makeAttr(initValType, 1)));

      auto allFinite = rewriter.create<stablehlo::ReduceOp>(
          op.getLoc(),
          RankedTensorType::get(infoType.getShape(), rewriter.getI1Type()),
          ValueRange{isFinite.getResult()}, ValueRange{initVal},
          rewriter.getDenseI64ArrayAttr(reductionDims));

      {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = allFinite.getBody();
        auto *block =
            rewriter.createBlock(&region, {}, {initValType, initValType},
                                 {op.getLoc(), op.getLoc()});

        rewriter.setInsertionPointToStart(block);
        auto lhs = block->getArgument(0);
        auto rhs = block->getArgument(1);
        auto andOp = rewriter.create<stablehlo::AndOp>(op.getLoc(), lhs, rhs);

        rewriter.create<stablehlo::ReturnOp>(op.getLoc(),
                                             ValueRange{andOp.getResult()});
      }

      // info == 0 if all finite (success)
      auto info = rewriter.create<stablehlo::ConvertOp>(
          op.getLoc(), infoType,
          rewriter.create<stablehlo::NotOp>(op.getLoc(),
                                            allFinite.getResult(0)));

      rewriter.replaceAllUsesWith(op.getResult(0), customCall.getResult(0));
      rewriter.replaceAllUsesWith(op.getResult(1), pivots1Indexed);
      rewriter.replaceAllUsesWith(op.getResult(2), info);
      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct LowerFactorizationPass
    : public enzyme::impl::LowerFactorizationPassBase<LowerFactorizationPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
