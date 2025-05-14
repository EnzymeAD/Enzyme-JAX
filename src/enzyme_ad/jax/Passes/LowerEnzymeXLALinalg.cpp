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

#define DEBUG_TYPE "lower-enzymexla-linalg"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLALINALGPASS
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
          /*api_version*/ nullptr,
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

        auto permutation = rewriter.create<stablehlo::CustomCallOp>(
            op.getLoc(), TypeRange{pivotCuSolverType},
            ValueRange{pivots0indexed.getResult()},
            rewriter.getStringAttr("cu_lu_pivots_to_permutation"),
            /*has_side_effect*/ nullptr,
            /*backend_config*/ nullptr,
            /*api_version*/ nullptr,
            /*calledcomputations*/ nullptr,
            /*operand_layouts*/ nullptr,
            /*result_layouts*/ nullptr,
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

struct LowerEnzymeXLALinalgPass
    : public enzyme::impl::LowerEnzymeXLALinalgPassBase<
          LowerEnzymeXLALinalgPass> {
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
