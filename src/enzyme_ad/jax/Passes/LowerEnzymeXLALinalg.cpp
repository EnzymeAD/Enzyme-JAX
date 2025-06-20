#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "lower-enzymexla-linalg"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLALINALGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

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

std::optional<std::string> lapack_precision_prefix(Type elementType) {

  // single-precision float
  if (elementType.isF32()) {
    return "s";

    // double-precision float
  } else if (elementType.isF64()) {
    return "d";

  } else if (auto complexType = dyn_cast<ComplexType>(elementType)) {
    auto elem = complexType.getElementType();

    // single-precision complex
    if (elem.isF32()) {
      return "c";

      // double-precision complex
    } else if (elem.isF64()) {
      return "z";

    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
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
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;

    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto pivotRank = pivotType.getRank();
    auto permutationType = cast<RankedTensorType>(op.getResult(2).getType());
    auto permutationRank = permutationType.getRank();
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());
    auto infoRank = infoType.getRank();

    if (backend == "cpu") {
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

      SmallVector<bool> isColMajorArr = {true, true, true};
      SmallVector<int64_t> operandRanks = {2, 1, 0};
      SmallVector<int64_t> outputRanks = {2, 1, 0};
      auto operandLayouts =
          getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
      auto resultLayouts =
          getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

      auto iterType = RankedTensorType::get({}, rewriter.getI32Type());
      auto iter = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), iterType, cast<ElementsAttr>(makeAttr(iterType, 0)));
      auto zeroConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), iterType, cast<ElementsAttr>(makeAttr(iterType, 0)));

      Value factorizedResult, pivotResult, infoResult;

      if (numBatchDims > 0) {
        // TODO: Implement batched LU factorizations by directly calling MKL
        //       https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/getrf-batch-strided.html.

        int64_t batchSize = 1;
        for (int i = 0; i < numBatchDims; i++) {
          batchSize *= inputShape[i];
        }
        SmallVector<int64_t> flattenedInput = {batchSize, m, n};

        auto flatInputType =
            RankedTensorType::get(flattenedInput, inputElementType);
        auto flatInput = rewriter.create<stablehlo::ReshapeOp>(
            op.getLoc(), flatInputType, input);

        auto flatPivotType = RankedTensorType::get(
            {batchSize, pivotType.getShape()[pivotRank - 1]}, blasIntType);
        auto flatPivot = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), flatPivotType,
            cast<ElementsAttr>(makeAttr(flatPivotType, -1)));

        auto flatInfoType = RankedTensorType::get({batchSize}, blasIntType);
        auto flatInfo = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), flatInfoType,
            cast<ElementsAttr>(makeAttr(flatInfoType, -1)));

        auto whileReturnTypes = {iterType, flatInputType, flatPivotType,
                                 flatInfoType};
        auto whileOp = rewriter.create<stablehlo::WhileOp>(
            op.getLoc(),
            TypeRange{iterType, flatInputType, flatPivotType, flatInfoType},
            ValueRange{iter, flatInput, flatPivot, flatInfo});

        {
          OpBuilder::InsertionGuard guard(rewriter);

          Block *block = rewriter.createBlock(&whileOp.getCond());
          rewriter.setInsertionPointToStart(block);

          for (auto type : whileReturnTypes) {
            block->addArgument(type, whileOp.getLoc());
          }

          auto batchSizeConst = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), iterType,
              cast<ElementsAttr>(makeAttr(iterType, batchSize)));

          auto comparison = rewriter.create<stablehlo::CompareOp>(
              op.getLoc(), block->getArgument(0), batchSizeConst,
              stablehlo::ComparisonDirection::LT);

          rewriter.create<stablehlo::ReturnOp>(
              op.getLoc(), ValueRange{comparison.getResult()});
        }

        {
          OpBuilder::InsertionGuard guard(rewriter);

          Block *block = rewriter.createBlock(&whileOp.getBody());
          rewriter.setInsertionPointToStart(block);

          for (auto type : whileReturnTypes) {
            block->addArgument(type, whileOp.getLoc());
          }

          auto iterArg = block->getArgument(0);

          auto inputSliceType = RankedTensorType::get({m, n}, inputElementType);
          auto inputSlice = rewriter.create<stablehlo::ReshapeOp>(
              op.getLoc(), inputSliceType,
              rewriter.create<stablehlo::DynamicSliceOp>(
                  op.getLoc(), block->getArgument(1),
                  ValueRange{iterArg, zeroConst, zeroConst},
                  rewriter.getDenseI64ArrayAttr({1, m, n})));

          auto pivotSliceType =
              RankedTensorType::get({std::min(m, n)}, blasIntType);
          auto pivotSlice = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), pivotSliceType,
              cast<ElementsAttr>(makeAttr(pivotSliceType, -1)));

          auto infoSliceType = RankedTensorType::get({}, blasIntType);
          auto infoSlice = rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), infoSliceType,
              cast<ElementsAttr>(makeAttr(infoSliceType, -1)));

          auto jitCall = rewriter.create<enzymexla::JITCallOp>(
              op.getLoc(),
              TypeRange{inputSliceType, pivotSliceType, infoSliceType},
              mlir::FlatSymbolRefAttr::get(ctx, fnName),
              ValueRange{inputSlice, pivotSlice, infoSlice},
              rewriter.getStringAttr(""),
              /*operand_layouts=*/operandLayouts,
              /*result_layouts=*/resultLayouts,
              /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
              /*xla_side_effect_free=*/rewriter.getUnitAttr());

          auto inputUpdated = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
              op.getLoc(), block->getArgument(1),
              rewriter.create<stablehlo::ReshapeOp>(
                  op.getLoc(),
                  RankedTensorType::get({1, m, n}, inputElementType),
                  jitCall.getResult(0)),
              ValueRange{iterArg, zeroConst, zeroConst});
          auto pivotUpdated = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
              op.getLoc(), block->getArgument(2),
              rewriter.create<stablehlo::ReshapeOp>(
                  op.getLoc(),
                  RankedTensorType::get({1, std::min(m, n)}, blasIntType),
                  jitCall.getResult(1)),
              ValueRange{iterArg, zeroConst});
          auto infoUpdated = rewriter.create<stablehlo::DynamicUpdateSliceOp>(
              op.getLoc(), block->getArgument(3),
              rewriter.create<stablehlo::ReshapeOp>(
                  op.getLoc(), RankedTensorType::get({1}, blasIntType),
                  jitCall.getResult(2)),
              ValueRange{iterArg});

          auto updatedIter = rewriter.create<stablehlo::AddOp>(
              op.getLoc(), block->getArgument(0),
              rewriter.create<stablehlo::ConstantOp>(
                  op.getLoc(), iterType,
                  cast<ElementsAttr>(makeAttr(iterType, 1))));

          rewriter.create<stablehlo::ReturnOp>(
              op.getLoc(),
              ValueRange{updatedIter, inputUpdated, pivotUpdated, infoUpdated});
        }

        factorizedResult = rewriter.create<stablehlo::ReshapeOp>(
            op.getLoc(), inputType, whileOp.getResult(1));
        pivotResult = rewriter.create<stablehlo::ReshapeOp>(
            op.getLoc(), blasPivotType, whileOp.getResult(2));
        infoResult = rewriter.create<stablehlo::ReshapeOp>(
            op.getLoc(), blasInfoType, whileOp.getResult(3));
      } else {
        auto pivot = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), blasPivotType,
            cast<ElementsAttr>(makeAttr(blasPivotType, -1)));
        auto info = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), blasInfoType,
            cast<ElementsAttr>(makeAttr(blasInfoType, -1)));

        auto jitCall = rewriter.create<enzymexla::JITCallOp>(
            op.getLoc(), TypeRange{inputType, blasPivotType, blasInfoType},
            mlir::FlatSymbolRefAttr::get(ctx, fnName),
            ValueRange{input, pivot, info}, rewriter.getStringAttr(""),
            /*operand_layouts=*/operandLayouts,
            /*result_layouts=*/resultLayouts,
            /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
            /*xla_side_effect_free=*/rewriter.getUnitAttr());

        factorizedResult = jitCall.getResult(0);
        pivotResult = jitCall.getResult(1);
        infoResult = jitCall.getResult(2);
      }

      auto pivots0indexed = rewriter.create<stablehlo::SubtractOp>(
          op.getLoc(), pivotResult,
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), blasPivotType,
              cast<ElementsAttr>(makeAttr(blasPivotType, 1))));

      auto permutation = rewriter.create<stablehlo::IotaOp>(
          op.getLoc(), blasPivotType,
          rewriter.getI64IntegerAttr(blasPivotType.getRank() - 1));

      auto pivotToPermReturnTypes = {iterType, blasPivotType};
      auto pivotToPermWhileOp = rewriter.create<stablehlo::WhileOp>(
          op.getLoc(), TypeRange{iterType, blasPivotType},
          ValueRange{iter, permutation});

      {
        OpBuilder::InsertionGuard guard(rewriter);

        Block *block = rewriter.createBlock(&pivotToPermWhileOp.getCond());
        rewriter.setInsertionPointToStart(block);

        for (auto type : pivotToPermReturnTypes)
          block->addArgument(type, pivotToPermWhileOp.getLoc());

        auto pivotShapeConst = rewriter.create<stablehlo::ConstantOp>(
            op.getLoc(), iterType,
            cast<ElementsAttr>(makeAttr(
                iterType, pivotType.getShape()[pivotType.getRank() - 1])));

        auto comparison = rewriter.create<stablehlo::CompareOp>(
            op.getLoc(), block->getArgument(0), pivotShapeConst,
            stablehlo::ComparisonDirection::LT);

        rewriter.create<stablehlo::ReturnOp>(
            op.getLoc(), ValueRange{comparison.getResult()});
      }

      {
        OpBuilder::InsertionGuard guard(rewriter);

        Block *block = rewriter.createBlock(&pivotToPermWhileOp.getBody());
        rewriter.setInsertionPointToStart(block);

        for (auto type : pivotToPermReturnTypes)
          block->addArgument(type, pivotToPermWhileOp.getLoc());

        auto iterArg = block->getArgument(0);

        auto updatedIter = rewriter.create<stablehlo::AddOp>(
            op.getLoc(), iterArg,
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), iterType,
                cast<ElementsAttr>(makeAttr(iterType, 1))));

        /*
        for i in range(pivot.shape[-1]):
          j = pivot[..., i]        # dynamic slice
          x = permutation[..., i]  # dynamic slice
          y = permutation[j]       # gather
          permutation[..., i] = y  # dynamic update slice
          permutation[j] = x       # scatter
        */

        SmallVector<Value> indices;
        SmallVector<int64_t> sliceShape, batchDims;
        for (int i = 0; i < numBatchDims; i++) {
          indices.push_back(zeroConst);
          sliceShape.push_back(pivotType.getShape()[i]);
          batchDims.push_back(i);
        }
        indices.push_back(iterArg);
        sliceShape.push_back(1);
        SmallVector<int64_t> gatherSliceSizes(numBatchDims + 1, 1);

        auto pivotJ = rewriter.create<stablehlo::DynamicSliceOp>(
            op.getLoc(), pivots0indexed, indices, sliceShape);
        auto permutationX = rewriter.create<stablehlo::DynamicSliceOp>(
            op.getLoc(), block->getArgument(1), indices, sliceShape);

        auto gatherDims = stablehlo::GatherDimensionNumbersAttr::get(
            op.getContext(),
            /*offsetDims=*/{numBatchDims},
            /*collapsedSliceDims=*/{},
            /*operandBatchingDims=*/batchDims,
            /*startIndicesBatchingDims=*/batchDims,
            /*startIndexMap=*/{numBatchDims},
            /*indexVectorDim=*/numBatchDims);
        auto permutationY = rewriter.create<stablehlo::GatherOp>(
            op.getLoc(),
            RankedTensorType::get(
                sliceShape,
                cast<RankedTensorType>(block->getArgument(1).getType())
                    .getElementType()),
            block->getArgument(1), pivotJ.getResult(), gatherDims,
            gatherSliceSizes);

        auto permutationUpdate1 =
            rewriter.create<stablehlo::DynamicUpdateSliceOp>(
                op.getLoc(), block->getArgument(1), permutationY->getResult(0),
                indices);

        auto scatterDims = stablehlo::ScatterDimensionNumbersAttr::get(
            op.getContext(),
            /*updateWindowDims=*/{},
            /*insertedWindowDims=*/{numBatchDims},
            /*inputBatchingDims=*/batchDims,
            /*scatterIndicesBatchingDims=*/batchDims,
            /*scatterDimsToOperandDims=*/{numBatchDims},
            /*indexVectorDim=*/numBatchDims);
        SmallVector<int64_t> scatterShape(sliceShape.begin(),
                                          sliceShape.end() - 1);
        auto permutationUpdate2 = rewriter.create<stablehlo::ScatterOp>(
            op.getLoc(), TypeRange{permutationUpdate1->getResult(0).getType()},
            ValueRange(permutationUpdate1->getResult(0)), pivotJ,
            ValueRange(rewriter.create<stablehlo::ReshapeOp>(
                op.getLoc(),
                RankedTensorType::get(scatterShape,
                                      permutationX.getType().getElementType()),
                permutationX)),
            scatterDims);

        {
          OpBuilder::InsertionGuard guard(rewriter);
          auto *block =
              rewriter.createBlock(&permutationUpdate2.getUpdateComputation());
          block->addArgument(RankedTensorType::get({}, blasIntType),
                             op.getLoc());
          block->addArgument(RankedTensorType::get({}, blasIntType),
                             op.getLoc());
          rewriter.setInsertionPointToStart(block);

          rewriter.create<stablehlo::ReturnOp>(
              op.getLoc(), ValueRange{block->getArgument(1)});
        }

        rewriter.create<stablehlo::ReturnOp>(
            op.getLoc(),
            ValueRange{updatedIter, permutationUpdate2->getResult(0)});
      }

      auto finalPermutation = rewriter.create<stablehlo::AddOp>(
          op.getLoc(), pivotToPermWhileOp.getResult(1),
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), blasPivotType,
              cast<ElementsAttr>(makeAttr(blasPivotType, 1))));

      rewriter.replaceAllUsesWith(op.getResult(0), factorizedResult);
      rewriter.replaceAllUsesWith(op.getResult(1),
                                  rewriter.create<stablehlo::ConvertOp>(
                                      op.getLoc(), pivotType, pivotResult));
      rewriter.replaceAllUsesWith(
          op.getResult(2), rewriter.create<stablehlo::ConvertOp>(
                               op.getLoc(), pivotType, finalPermutation));
      rewriter.replaceAllUsesWith(op.getResult(3),
                                  rewriter.create<stablehlo::ConvertOp>(
                                      op.getLoc(), infoType, infoResult));

      return success();
    } else if (backend == "cuda") {
      SmallVector<Attribute> aliases = {stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{0}, 0, std::vector<int64_t>{})};

      SmallVector<bool> isColMajorArrOperands = {true};
      SmallVector<int64_t> operandRanks = {inputRank};
      SmallVector<bool> isColMajorArrOutputs = {true, true, true};
      SmallVector<int64_t> outputRanks = {inputRank, pivotRank, infoRank};

      auto pivotCuSolverType =
          RankedTensorType::get(pivotType.getShape(), rewriter.getI32Type());
      auto infoCuSolverType =
          RankedTensorType::get(infoType.getShape(), rewriter.getI32Type());

      auto cusolverffi = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(),
          TypeRange{inputType, pivotCuSolverType, infoCuSolverType},
          ValueRange{input}, rewriter.getStringAttr("cusolver_getrf_ffi"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ nullptr,
          /*api_version*/
          stablehlo::CustomCallApiVersionAttr::get(
              rewriter.getContext(),
              mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
          /*calledcomputations*/ nullptr,
          /*operand_layouts*/
          getSHLOLayout(rewriter, operandRanks, isColMajorArrOperands,
                        inputRank),
          /*result_layouts*/
          getSHLOLayout(rewriter, outputRanks, isColMajorArrOutputs, inputRank),
          /*output_operand_aliases*/ rewriter.getArrayAttr(aliases));

      // unused custom call not getting optimized away. so adding a manual check
      if (!op.getResult(2).getUses().empty()) {
        auto pivots0indexed = rewriter.create<stablehlo::SubtractOp>(
            op.getLoc(), cusolverffi.getResult(1),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), pivotCuSolverType,
                cast<ElementsAttr>(makeAttr(pivotCuSolverType, 1))));

        SmallVector<bool> isColMajorArrOperandsPermutation = {true};
        SmallVector<int64_t> operandRanksPermutation = {pivotRank};
        SmallVector<bool> isColMajorArrOutputsPermutation = {true};
        SmallVector<int64_t> outputRanksPermutation = {pivotRank};

        auto permutation = rewriter.create<stablehlo::CustomCallOp>(
            op.getLoc(), TypeRange{pivotCuSolverType},
            ValueRange{pivots0indexed.getResult()},
            rewriter.getStringAttr("cu_lu_pivots_to_permutation"),
            /*has_side_effect*/ nullptr,
            /*backend_config*/ nullptr,
            /*api_version*/
            stablehlo::CustomCallApiVersionAttr::get(
                rewriter.getContext(),
                mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
            /*calledcomputations*/ nullptr,
            /*operand_layouts*/
            getSHLOLayout(rewriter, operandRanksPermutation,
                          isColMajorArrOperandsPermutation, inputRank),
            /*result_layouts*/
            getSHLOLayout(rewriter, outputRanksPermutation,
                          isColMajorArrOutputsPermutation, inputRank),
            /*output_operand_aliases*/ nullptr);
        auto permutation1Indexed = rewriter.create<stablehlo::AddOp>(
            op.getLoc(),
            rewriter.create<stablehlo::ConstantOp>(
                op.getLoc(), pivotCuSolverType,
                cast<ElementsAttr>(makeAttr(pivotCuSolverType, 1))),
            permutation.getResult(0));

        rewriter.replaceAllUsesWith(
            op.getResult(2), rewriter.create<stablehlo::ConvertOp>(
                                 op.getLoc(), pivotType, permutation1Indexed));
      }

      rewriter.replaceAllUsesWith(op.getResult(0), cusolverffi.getResult(0));
      rewriter.replaceAllUsesWith(
          op.getResult(1),
          rewriter.create<stablehlo::ConvertOp>(op.getLoc(), pivotType,
                                                cusolverffi.getResult(1)));
      rewriter.replaceAllUsesWith(
          op.getResult(3),
          rewriter.create<stablehlo::ConvertOp>(op.getLoc(), infoType,
                                                cusolverffi.getResult(2)));

      return success();
    } else if (backend == "tpu") {
      SmallVector<int64_t> permutationShape;
      for (int i = 0; i < numBatchDims; i++) {
        permutationShape.push_back(inputShape[i]);
      }
      permutationShape.push_back(m);
      auto permutationType =
          RankedTensorType::get(permutationShape, rewriter.getI32Type());

      auto pivotTPUType =
          RankedTensorType::get(pivotType.getShape(), rewriter.getI32Type());

      // TPU returns (LU, pivots, permutation). info isn't returned. based on
      // how JAX operates, I am assuming info != 0 when there is a nan in the
      // output.
      auto customCall = rewriter.create<stablehlo::CustomCallOp>(
          op.getLoc(), TypeRange{inputType, pivotTPUType, permutationType},
          ValueRange{input}, rewriter.getStringAttr("LuDecomposition"),
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
          rewriter.create<stablehlo::ConvertOp>(op.getLoc(), pivotType,
                                                customCall.getResult(1)));

      auto permutation1Indexed = rewriter.create<stablehlo::AddOp>(
          op.getLoc(),
          rewriter.create<stablehlo::ConstantOp>(
              op.getLoc(), permutationType,
              cast<ElementsAttr>(makeAttr(permutationType, 1))),
          rewriter.create<stablehlo::ConvertOp>(op.getLoc(), permutationType,
                                                customCall.getResult(2)));

      auto isFinite = rewriter.create<stablehlo::AndOp>(
          op.getLoc(),
          rewriter.create<stablehlo::IsFiniteOp>(
              op.getLoc(), rewriter.create<stablehlo::RealOp>(
                               op.getLoc(), customCall.getResult(0))),
          rewriter.create<stablehlo::IsFiniteOp>(
              op.getLoc(), rewriter.create<stablehlo::ImagOp>(
                               op.getLoc(), customCall.getResult(0))));

      SmallVector<int64_t> reductionDims;
      for (int i = numBatchDims; i < inputRank; i++)
        reductionDims.push_back(i);
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
      rewriter.replaceAllUsesWith(op.getResult(2), permutation1Indexed);
      rewriter.replaceAllUsesWith(op.getResult(3), info);
      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct QRFactorizationOpLowering
    : public OpRewritePattern<enzymexla::QRFactorizationOp> {
  std::string backend;
  int64_t blasIntWidth;

  QRFactorizationOpLowering(std::string backend, int64_t blasIntWidth,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::QRFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return this->matchAndRewrite_cpu(op, rewriter);

    else if (backend == "cuda")
      return this->matchAndRewrite_cuda(op, rewriter);

    else if (backend == "tpu")
      return this->matchAndRewrite_tpu(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewrite_cpu(enzymexla::QRFactorizationOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputElementType = inputType.getElementType();

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;

    if (numBatchDims > 0) {
      return rewriter.notifyMatchFailure(
          op, "QR factorization with batch dimensions is not yet supported");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);

    // TODO change QR method with attributes
    std::string fn = "geqrf_";
    if (auto prefix = lapack_precision_prefix(inputElementType)) {
      fn = *prefix + fn;
    } else {
      op->emitOpError() << "Unsupported complex element type: "
                        << inputElementType;
      return rewriter.notifyMatchFailure(op,
                                         "unsupported complex element type");
    }

    std::string bind_fn = "enzymexla_lapacke_" + fn;
    std::string wrapper_fn = "enzymexla_wrapper_lapacke_" + fn;

    // declare LAPACKE function declarations if not present
    auto moduleOp = op->getParentOfType<ModuleOp>();

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(bind_fn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto func_type =
          LLVM::LLVMFunctionType::get(type_llvm_lapack_int,
                                      {
                                          type_llvm_lapack_int, // matrix_layout
                                          type_llvm_lapack_int, // m
                                          type_llvm_lapack_int, // n
                                          type_llvm_ptr,        // A
                                          type_llvm_lapack_int, // lda
                                          type_llvm_ptr         // tau
                                      },
                                      false);
      rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), bind_fn, func_type,
                                        LLVM::Linkage::External);
    }

    // WARN probably will need another function name encoding if we call to
    // `geqrf`, `orgqr` or `ungqr` in other op insert wrapper function for
    // `geqrf`
    static int64_t fn_counter = 0;
    fn_counter++;

    wrapper_fn += std::to_string(fn_counter);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto func_type = LLVM::LLVMFunctionType::get(type_llvm_void,
                                                   {
                                                       type_llvm_ptr, // A
                                                       type_llvm_ptr, // tau
                                                       type_llvm_ptr, // info
                                                   },
                                                   false);

      auto func =
          rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapper_fn, func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto m = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[0]));
      auto n = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[1]));
      auto lda = m;

      // call to `lapacke_*geqrf*`
      auto res = rewriter.create<LLVM::CallOp>(op.getLoc(),
                                               TypeRange{type_llvm_lapack_int},
                                               SymbolRefAttr::get(ctx, bind_fn),
                                               ValueRange{
                                                   layout.getResult(),
                                                   m.getResult(),
                                                   n.getResult(),
                                                   func.getArgument(0),
                                                   lda.getResult(),
                                                   func.getArgument(1),
                                               });

      rewriter.create<LLVM::StoreOp>(op.getLoc(), res.getResult(),
                                     func.getArgument(2));
      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `geqrf` wrapper
    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto info = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, -1)));

    auto tsize = std::min(inputShape.front(), inputShape.back());
    auto type_tau = RankedTensorType::get({tsize}, inputElementType);
    auto tau = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_tau, cast<ElementsAttr>(makeAttr(type_tau, 0)));

    SmallVector<bool> isColMajorArr = {true, true, true};
    SmallVector<int64_t> operandRanks = {2, 1, 0};
    SmallVector<int64_t> outputRanks = {2, 1, 0};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    for (int i = 0; i < 3; ++i) {
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{i}, i, std::vector<int64_t>{}));
    }

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{inputType, type_tau, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, tau.getResult(), info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts, // TODO
        /*result_layouts=*/resultLayouts,   // TODO
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.linalg.qr with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(0), jit_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), jit_call_op.getResult(1));
    rewriter.replaceAllUsesWith(op.getResult(2), jit_call_op.getResult(2));
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewrite_cuda(enzymexla::QRFactorizationOp op,
                                     PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

    const int64_t m = shape_input[rank_input - 2];
    const int64_t n = shape_input[rank_input - 1];
    const int64_t numBatchDims = rank_input - 2;

    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());
    auto rank_tau = type_tau.getRank();

    // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{0}, 0, std::vector<int64_t>{})};
    SmallVector<int64_t> ranks_operands = {rank_input};
    SmallVector<int64_t> ranks_results = {rank_input, rank_tau};
    SmallVector<bool> isColMajorArrOperands = {true};
    SmallVector<bool> isColMajorArrOutputs = {true, true};

    auto cusolver_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input, type_tau}, ValueRange{input},
        rewriter.getStringAttr("cusolver_geqrf_ffi"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, ranks_operands, isColMajorArrOperands,
                      rank_input),
        /*result_layouts*/
        getSHLOLayout(rewriter, ranks_results, isColMajorArrOutputs,
                      rank_input),
        /*output_operand_aliases*/ rewriter.getArrayAttr(aliases));

    rewriter.replaceAllUsesWith(op.getResult(0), cusolver_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), cusolver_call_op.getResult(1));

    // Netlib's LAPACK returns `info`, but cuSOLVER doesn't
    // TODO what does JAX do?
    auto type_info =
        RankedTensorType::get({}, rewriter.getIntegerType(blasIntWidth));
    auto info_op = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, 0)));
    rewriter.replaceAllUsesWith(op.getResult(2), info_op.getResult());

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewrite_tpu(enzymexla::QRFactorizationOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

    const int64_t m = shape_input[rank_input - 2];
    const int64_t n = shape_input[rank_input - 1];
    const int64_t numBatchDims = rank_input - 2;

    // emit `stablehlo.custom_call` to `@QrDecomposition` kernel from XLA
    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());

    auto custom_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input, type_tau}, ValueRange{input},
        rewriter.getStringAttr("QrDecomposition"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(0), custom_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), custom_call_op.getResult(1));

    // Netlib's LAPACK returns `info`, but TPU kernel doesn't
    auto type_info =
        RankedTensorType::get({}, rewriter.getIntegerType(blasIntWidth));
    auto info_op = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, 0)));
    rewriter.replaceAllUsesWith(op.getResult(2), info_op.getResult());

    return success();
  }
};

struct SVDFactorizationOpLowering
    : public OpRewritePattern<enzymexla::SVDFactorizationOp> {
  std::string backend;
  int64_t blasIntWidth;

  SVDFactorizationOpLowering(std::string backend, int64_t blasIntWidth,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::SVDFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return this->matchAndRewrite_cpu(op, rewriter);

    else if (backend == "cuda")
      return this->matchAndRewrite_cuda(op, rewriter);

    else if (backend == "tpu")
      return this->matchAndRewrite_tpu(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  // TODO support more SVD algorithms (e.g. `gesdd`, `gesvj`)
  LogicalResult matchAndRewrite_cpu(enzymexla::SVDFactorizationOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputElementType = inputType.getElementType();

    auto isfull = op.getFull();

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;

    if (numBatchDims > 0) {
      return rewriter.notifyMatchFailure(
          op, "SVD factorization with batch dimensions is not yet supported");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_lapack_char = rewriter.getIntegerType(sizeof(char) * 8);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);

    // TODO change SVD method with attributes
    std::string fn = "gesvd_";
    if (auto prefix = lapack_precision_prefix(inputElementType)) {
      fn = *prefix + fn;
    } else {
      op->emitOpError() << "Unsupported complex element type: "
                        << inputElementType;
      return rewriter.notifyMatchFailure(op,
                                         "unsupported complex element type");
    }

    std::string bind_fn = "enzymexla_lapacke_" + fn;
    std::string wrapper_fn = "enzymexla_wrapper_lapacke_" + fn;

    // declare LAPACKE function declarations if not present
    auto moduleOp = op->getParentOfType<ModuleOp>();

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(bind_fn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto func_type =
          LLVM::LLVMFunctionType::get(type_llvm_lapack_int,
                                      {
                                          type_llvm_lapack_int, // matrix_layout
                                          type_lapack_char,     // jobu
                                          type_lapack_char,     // jobvt
                                          type_llvm_lapack_int, // m
                                          type_llvm_lapack_int, // n
                                          type_llvm_ptr,        // a
                                          type_llvm_lapack_int, // lda
                                          type_llvm_ptr,        // s
                                          type_llvm_ptr,        // u
                                          type_llvm_lapack_int, // ldu
                                          type_llvm_ptr,        // v
                                          type_llvm_lapack_int, // ldvt
                                          type_llvm_ptr,        // superb
                                      },
                                      false);
      rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), bind_fn, func_type,
                                        LLVM::Linkage::External);
    }

    // emit wrapper function
    static int64_t fn_counter = 0;
    fn_counter++;

    wrapper_fn += std::to_string(fn_counter);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto func_type = LLVM::LLVMFunctionType::get(type_llvm_void,
                                                   {
                                                       type_llvm_ptr, // a
                                                       type_llvm_ptr, // u
                                                       type_llvm_ptr, // s
                                                       type_llvm_ptr, // vt
                                                       type_llvm_ptr, // superb
                                                       type_llvm_ptr, // info
                                                   },
                                                   false);

      auto op_func =
          rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapper_fn, func_type);
      rewriter.setInsertionPointToStart(op_func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto op_layout = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto op_jobu = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, op.getFull() ? 'A' : 'S')
      );
      auto op_jobvt = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, op.getFull() ? 'A' : 'S')
      );
      auto op_m = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[0]));
      auto op_n = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[1]));
      auto op_lda = op_m;
      auto op_ldu = op_m;
      auto op_ldvt = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int,
            std::min(inputShape[0], inputShape[1])));

      // call to `lapacke_*gesvd`
      auto llvm_call_op = rewriter.create<LLVM::CallOp>(op.getLoc(),
                                               TypeRange{type_llvm_lapack_int},
                                               SymbolRefAttr::get(ctx, bind_fn),
                                               ValueRange{
                                                   op_layout.getResult(),
                                                   op_jobu.getResult(),
                                                   op_jobvt.getResult(),
                                                   op_m.getResult(),
                                                   op_n.getResult(),
                                                   op_func.getArgument(0), // a
                                                   op_lda.getResult(),
                                                   op_func.getArgument(2), // s
                                                   op_func.getArgument(1), // u
                                                   op_ldu.getResult(),
                                                   op_func.getArgument(3), // vt
                                                   op_ldvt.getResult(),
                                                   op_func.getArgument(4), // superb
                                               });

      rewriter.create<LLVM::StoreOp>(op.getLoc(), llvm_call_op.getResult(), op_func.getArgument(5));
      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `gesvd` wrapper
    auto type_u = RankedTensorType::get({inputShape[0], std::min(inputShape[0], inputShape[1])}, inputElementType);
    auto op_u = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_u, cast<ElementsAttr>(makeAttr(type_u, 0))
    );

    auto type_s = RankedTensorType::get({std::min(inputShape[0], inputShape[1])}, inputElementType);
    auto op_s = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_s, cast<ElementsAttr>(makeAttr(type_s, 0))
    );

    auto type_vt = RankedTensorType::get({std::min(inputShape[0], inputShape[1]), inputShape[1]}, inputElementType);
    auto op_vt = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_vt, cast<ElementsAttr>(makeAttr(type_vt, 0))
    );

    auto type_superb = RankedTensorType::get({std::min(inputShape[0], inputShape[1]) - 1}, inputElementType);
    auto op_superb = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_superb, cast<ElementsAttr>(makeAttr(type_superb, 0))
    );

    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto op_info = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, -1)));

    SmallVector<bool> isColMajorArr = {true, true, true, true, true, true};
    SmallVector<int64_t> operandRanks = {2, 2, 1, 2, 1, 0};
    SmallVector<int64_t> outputRanks = {2, 1, 2, 0};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    // alias for u
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, std::vector<int64_t>{0}, 1, std::vector<int64_t>{}));
    // alias for s
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, std::vector<int64_t>{1}, 2, std::vector<int64_t>{}));
    // alias for vt
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, std::vector<int64_t>{2}, 3, std::vector<int64_t>{}));
    // alias for info
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, std::vector<int64_t>{3}, 5, std::vector<int64_t>{}));

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{type_u, type_s, type_vt, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, op_u.getResult(), op_s.getResult(), op_vt.getResult(), op_superb.getResult(), op_info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.linalg.svd with enzymexla.jit_call
    rewriter.replaceAllUsesWith(op.getResult(0), jit_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), jit_call_op.getResult(1));
    rewriter.replaceAllUsesWith(op.getResult(2), jit_call_op.getResult(2));
    rewriter.replaceAllUsesWith(op.getResult(3), jit_call_op.getResult(3));
    rewriter.eraseOp(op);

    return success();
  }

  // TODO revise this is ok
  LogicalResult matchAndRewrite_cuda(enzymexla::SVDFactorizationOp op,
                                     PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

    const int64_t m = shape_input[rank_input - 2];
    const int64_t n = shape_input[rank_input - 1];
    const int64_t numBatchDims = rank_input - 2;

    auto type_u = cast<RankedTensorType>(op.getResult(0).getType());
    auto rank_u = type_u.getRank();

    auto type_s = cast<RankedTensorType>(op.getResult(1).getType());
    auto rank_s = type_s.getRank();

    auto type_vt = cast<RankedTensorType>(op.getResult(2).getType());
    auto rank_vt = type_vt.getRank();

    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto rank_info = type_info.getRank();
    auto op_info = rewriter.create<stablehlo::ConstantOp>(
      op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, -1)));

    // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {};
    SmallVector<int64_t> ranks_operands = {rank_input};
    SmallVector<int64_t> ranks_results = {rank_input, rank_s, rank_u, rank_vt, rank_info};
    SmallVector<bool> isColMajorArrOperands = {true};
    SmallVector<bool> isColMajorArrOutputs = {true, true, true, true, true};

    // TODO pass `full_matrices`, `compute_uv` and `transposed` attrs from 
    // https://github.com/jax-ml/jax/blob/22f7b7b5cc2cfb8ed43b15fdad491b2268f4f3de/jaxlib/gpu/solver_kernels_ffi.cc#L864-L877
    auto cusolver_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input, type_s, type_u, type_vt, type_info}, ValueRange{input},
        rewriter.getStringAttr("cusolver_gesvd_ffi"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ getSHLOLayout(rewriter, ranks_operands, isColMajorArrOperands, rank_input),
        /*result_layouts*/ getSHLOLayout(rewriter, ranks_results, isColMajorArrOutputs, rank_input),
        /*output_operand_aliases*/ rewriter.getArrayAttr(aliases));
    cusolver_call_op->setAttr(rewriter.getStringAttr("full_matrices"), rewriter.getBoolAttr(op.getFull()));
    cusolver_call_op->setAttr(rewriter.getStringAttr("compute_uv"), rewriter.getBoolAttr(true));
    cusolver_call_op->setAttr(rewriter.getStringAttr("transposed"), rewriter.getBoolAttr(false));

    // replace enzymexla.linalg.svd with stablehlo.custom_call
    rewriter.replaceAllUsesWith(op.getResult(0), cusolver_call_op.getResult(2));
    rewriter.replaceAllUsesWith(op.getResult(1), cusolver_call_op.getResult(1));
    rewriter.replaceAllUsesWith(op.getResult(2), cusolver_call_op.getResult(3));
    rewriter.replaceAllUsesWith(op.getResult(3), cusolver_call_op.getResult(4));
    rewriter.eraseOp(op);

    return success();
  }

  // TODO find registered TPU kernel
  LogicalResult matchAndRewrite_tpu(enzymexla::SVDFactorizationOp op,
                                    PatternRewriter &rewriter) const {


    return rewriter.notifyMatchFailure(op, "We don't know yet to which SVD TPU kernel to lower to :_(");

    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

    const int64_t m = shape_input[rank_input - 2];
    const int64_t n = shape_input[rank_input - 1];
    const int64_t numBatchDims = rank_input - 2;

    // emit `stablehlo.custom_call` to `@Svd` kernel from XLA
    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());

    auto custom_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input, type_tau}, ValueRange{input},
        rewriter.getStringAttr("Svd"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(0), custom_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), custom_call_op.getResult(1));

    // Netlib's LAPACK returns `info`, but TPU kernel doesn't
    auto type_info =
        RankedTensorType::get({}, rewriter.getIntegerType(blasIntWidth));
    auto info_op = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, 0)));
    rewriter.replaceAllUsesWith(op.getResult(2), info_op.getResult());

    return success();
  }
};

struct LowerEnzymeXLALinalgPass
    : public enzyme::impl::LowerEnzymeXLALinalgPassBase<
          LowerEnzymeXLALinalgPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<LUFactorizationOpLowering>(backend, blasIntWidth, context);
    patterns.add<QRFactorizationOpLowering>(backend, blasIntWidth, context);
    patterns.add<SVDFactorizationOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
