#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
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
            LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto ptrSize =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
                                     rewriter.getIntegerAttr(blasIntType, 1));
        auto mPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(), llvmPtrType,
                                           llvmBlasIntType, ptrSize, 0);
        auto nPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(), llvmPtrType,
                                           llvmBlasIntType, ptrSize, 0);

        auto mVal =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
                                     rewriter.getIntegerAttr(blasIntType, m));
        auto nVal =
            LLVM::ConstantOp::create(rewriter, op.getLoc(), llvmBlasIntType,
                                     rewriter.getIntegerAttr(blasIntType, n));

        LLVM::StoreOp::create(rewriter, op.getLoc(), mVal, mPtr);
        LLVM::StoreOp::create(rewriter, op.getLoc(), nVal, nPtr);

        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                             SymbolRefAttr::get(ctx, lapackFn),
                             ValueRange{
                                 mPtr,
                                 nPtr,
                                 func.getArgument(0),
                                 mPtr,
                                 func.getArgument(1),
                                 func.getArgument(2),
                             });

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
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

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), lapackFn, funcType,
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
      auto iter = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), iterType,
          cast<ElementsAttr>(makeAttr(iterType, 0)));
      auto zeroConst = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), iterType,
          cast<ElementsAttr>(makeAttr(iterType, 0)));

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
        auto flatInput = stablehlo::ReshapeOp::create(rewriter, op.getLoc(),
                                                      flatInputType, input);

        auto flatPivotType = RankedTensorType::get(
            {batchSize, pivotType.getShape()[pivotRank - 1]}, blasIntType);
        auto flatPivot = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), flatPivotType,
            cast<ElementsAttr>(makeAttr(flatPivotType, -1)));

        auto flatInfoType = RankedTensorType::get({batchSize}, blasIntType);
        auto flatInfo = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), flatInfoType,
            cast<ElementsAttr>(makeAttr(flatInfoType, -1)));

        auto whileReturnTypes = {iterType, flatInputType, flatPivotType,
                                 flatInfoType};
        auto whileOp = stablehlo::WhileOp::create(
            rewriter, op.getLoc(),
            TypeRange{iterType, flatInputType, flatPivotType, flatInfoType},
            ValueRange{iter, flatInput, flatPivot, flatInfo});

        {
          OpBuilder::InsertionGuard guard(rewriter);

          Block *block = rewriter.createBlock(&whileOp.getCond());
          rewriter.setInsertionPointToStart(block);

          for (auto type : whileReturnTypes) {
            block->addArgument(type, whileOp.getLoc());
          }

          auto batchSizeConst = stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), iterType,
              cast<ElementsAttr>(makeAttr(iterType, batchSize)));

          auto comparison = stablehlo::CompareOp::create(
              rewriter, op.getLoc(), block->getArgument(0), batchSizeConst,
              stablehlo::ComparisonDirection::LT);

          stablehlo::ReturnOp::create(rewriter, op.getLoc(),
                                      ValueRange{comparison.getResult()});
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
          auto inputSlice = stablehlo::ReshapeOp::create(
              rewriter, op.getLoc(), inputSliceType,
              stablehlo::DynamicSliceOp::create(
                  rewriter, op.getLoc(), block->getArgument(1),
                  ValueRange{iterArg, zeroConst, zeroConst},
                  rewriter.getDenseI64ArrayAttr({1, m, n})));

          auto pivotSliceType =
              RankedTensorType::get({std::min(m, n)}, blasIntType);
          auto pivotSlice = stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), pivotSliceType,
              cast<ElementsAttr>(makeAttr(pivotSliceType, -1)));

          auto infoSliceType = RankedTensorType::get({}, blasIntType);
          auto infoSlice = stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), infoSliceType,
              cast<ElementsAttr>(makeAttr(infoSliceType, -1)));

          auto jitCall = enzymexla::JITCallOp::create(
              rewriter, op.getLoc(),
              TypeRange{inputSliceType, pivotSliceType, infoSliceType},
              mlir::FlatSymbolRefAttr::get(ctx, fnName),
              ValueRange{inputSlice, pivotSlice, infoSlice},
              rewriter.getStringAttr(""),
              /*operand_layouts=*/operandLayouts,
              /*result_layouts=*/resultLayouts,
              /*arg_attrs=*/nullptr,
              /*res_attrs=*/nullptr,
              /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
              /*xla_side_effect_free=*/rewriter.getUnitAttr());

          auto inputUpdated = stablehlo::DynamicUpdateSliceOp::create(
              rewriter, op.getLoc(), block->getArgument(1),
              stablehlo::ReshapeOp::create(
                  rewriter, op.getLoc(),
                  RankedTensorType::get({1, m, n}, inputElementType),
                  jitCall.getResult(0)),
              ValueRange{iterArg, zeroConst, zeroConst});
          auto pivotUpdated = stablehlo::DynamicUpdateSliceOp::create(
              rewriter, op.getLoc(), block->getArgument(2),
              stablehlo::ReshapeOp::create(
                  rewriter, op.getLoc(),
                  RankedTensorType::get({1, std::min(m, n)}, blasIntType),
                  jitCall.getResult(1)),
              ValueRange{iterArg, zeroConst});
          auto infoUpdated = stablehlo::DynamicUpdateSliceOp::create(
              rewriter, op.getLoc(), block->getArgument(3),
              stablehlo::ReshapeOp::create(
                  rewriter, op.getLoc(),
                  RankedTensorType::get({1}, blasIntType),
                  jitCall.getResult(2)),
              ValueRange{iterArg});

          auto updatedIter = stablehlo::AddOp::create(
              rewriter, op.getLoc(), block->getArgument(0),
              stablehlo::ConstantOp::create(
                  rewriter, op.getLoc(), iterType,
                  cast<ElementsAttr>(makeAttr(iterType, 1))));

          stablehlo::ReturnOp::create(
              rewriter, op.getLoc(),
              ValueRange{updatedIter, inputUpdated, pivotUpdated, infoUpdated});
        }

        factorizedResult = stablehlo::ReshapeOp::create(
            rewriter, op.getLoc(), inputType, whileOp.getResult(1));
        pivotResult = stablehlo::ReshapeOp::create(
            rewriter, op.getLoc(), blasPivotType, whileOp.getResult(2));
        infoResult = stablehlo::ReshapeOp::create(
            rewriter, op.getLoc(), blasInfoType, whileOp.getResult(3));
      } else {
        auto pivot = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), blasPivotType,
            cast<ElementsAttr>(makeAttr(blasPivotType, -1)));
        auto info = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), blasInfoType,
            cast<ElementsAttr>(makeAttr(blasInfoType, -1)));

        auto jitCall = enzymexla::JITCallOp::create(
            rewriter, op.getLoc(),
            TypeRange{inputType, blasPivotType, blasInfoType},
            mlir::FlatSymbolRefAttr::get(ctx, fnName),
            ValueRange{input, pivot, info}, rewriter.getStringAttr(""),
            /*operand_layouts=*/operandLayouts,
            /*result_layouts=*/resultLayouts,
            /*arg_attrs=*/nullptr,
            /*res_attrs=*/nullptr,
            /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
            /*xla_side_effect_free=*/rewriter.getUnitAttr());

        factorizedResult = jitCall.getResult(0);
        pivotResult = jitCall.getResult(1);
        infoResult = jitCall.getResult(2);
      }

      auto pivots0indexed = stablehlo::SubtractOp::create(
          rewriter, op.getLoc(), pivotResult,
          stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), blasPivotType,
              cast<ElementsAttr>(makeAttr(blasPivotType, 1))));

      auto permutation = stablehlo::IotaOp::create(
          rewriter, op.getLoc(), blasPivotType,
          rewriter.getI64IntegerAttr(blasPivotType.getRank() - 1));

      auto pivotToPermReturnTypes = {iterType, blasPivotType};
      auto pivotToPermWhileOp = stablehlo::WhileOp::create(
          rewriter, op.getLoc(), TypeRange{iterType, blasPivotType},
          ValueRange{iter, permutation});

      {
        OpBuilder::InsertionGuard guard(rewriter);

        Block *block = rewriter.createBlock(&pivotToPermWhileOp.getCond());
        rewriter.setInsertionPointToStart(block);

        for (auto type : pivotToPermReturnTypes)
          block->addArgument(type, pivotToPermWhileOp.getLoc());

        auto pivotShapeConst = stablehlo::ConstantOp::create(
            rewriter, op.getLoc(), iterType,
            cast<ElementsAttr>(makeAttr(
                iterType, pivotType.getShape()[pivotType.getRank() - 1])));

        auto comparison = stablehlo::CompareOp::create(
            rewriter, op.getLoc(), block->getArgument(0), pivotShapeConst,
            stablehlo::ComparisonDirection::LT);

        stablehlo::ReturnOp::create(rewriter, op.getLoc(),
                                    ValueRange{comparison.getResult()});
      }

      {
        OpBuilder::InsertionGuard guard(rewriter);

        Block *block = rewriter.createBlock(&pivotToPermWhileOp.getBody());
        rewriter.setInsertionPointToStart(block);

        for (auto type : pivotToPermReturnTypes)
          block->addArgument(type, pivotToPermWhileOp.getLoc());

        auto iterArg = block->getArgument(0);

        auto updatedIter = stablehlo::AddOp::create(
            rewriter, op.getLoc(), iterArg,
            stablehlo::ConstantOp::create(
                rewriter, op.getLoc(), iterType,
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

        auto pivotJ = stablehlo::DynamicSliceOp::create(
            rewriter, op.getLoc(), pivots0indexed, indices, sliceShape);
        auto permutationX = stablehlo::DynamicSliceOp::create(
            rewriter, op.getLoc(), block->getArgument(1), indices, sliceShape);

        auto gatherDims = stablehlo::GatherDimensionNumbersAttr::get(
            op.getContext(),
            /*offsetDims=*/{numBatchDims},
            /*collapsedSliceDims=*/{},
            /*operandBatchingDims=*/batchDims,
            /*startIndicesBatchingDims=*/batchDims,
            /*startIndexMap=*/{numBatchDims},
            /*indexVectorDim=*/numBatchDims);
        auto permutationY = stablehlo::GatherOp::create(
            rewriter, op.getLoc(),
            RankedTensorType::get(
                sliceShape,
                cast<RankedTensorType>(block->getArgument(1).getType())
                    .getElementType()),
            block->getArgument(1), pivotJ.getResult(), gatherDims,
            gatherSliceSizes);

        auto permutationUpdate1 = stablehlo::DynamicUpdateSliceOp::create(
            rewriter, op.getLoc(), block->getArgument(1),
            permutationY->getResult(0), indices);

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
        auto permutationUpdate2 = stablehlo::ScatterOp::create(
            rewriter, op.getLoc(),
            TypeRange{permutationUpdate1->getResult(0).getType()},
            ValueRange(permutationUpdate1->getResult(0)), pivotJ,
            ValueRange(stablehlo::ReshapeOp::create(
                rewriter, op.getLoc(),
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

          stablehlo::ReturnOp::create(rewriter, op.getLoc(),
                                      ValueRange{block->getArgument(1)});
        }

        stablehlo::ReturnOp::create(
            rewriter, op.getLoc(),
            ValueRange{updatedIter, permutationUpdate2->getResult(0)});
      }

      auto finalPermutation = stablehlo::AddOp::create(
          rewriter, op.getLoc(), pivotToPermWhileOp.getResult(1),
          stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), blasPivotType,
              cast<ElementsAttr>(makeAttr(blasPivotType, 1))));

      rewriter.replaceAllUsesWith(op.getResult(0), factorizedResult);
      rewriter.replaceAllUsesWith(
          op.getResult(1), stablehlo::ConvertOp::create(
                               rewriter, op.getLoc(), pivotType, pivotResult));
      rewriter.replaceAllUsesWith(
          op.getResult(2),
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                       finalPermutation));
      rewriter.replaceAllUsesWith(
          op.getResult(3), stablehlo::ConvertOp::create(rewriter, op.getLoc(),
                                                        infoType, infoResult));

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

      auto cusolverffi = stablehlo::CustomCallOp::create(
          rewriter, op.getLoc(),
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
        auto pivots0indexed = stablehlo::SubtractOp::create(
            rewriter, op.getLoc(), cusolverffi.getResult(1),
            stablehlo::ConstantOp::create(
                rewriter, op.getLoc(), pivotCuSolverType,
                cast<ElementsAttr>(makeAttr(pivotCuSolverType, 1))));

        SmallVector<bool> isColMajorArrOperandsPermutation = {true};
        SmallVector<int64_t> operandRanksPermutation = {pivotRank};
        SmallVector<bool> isColMajorArrOutputsPermutation = {true};
        SmallVector<int64_t> outputRanksPermutation = {pivotRank};

        auto permutation = stablehlo::CustomCallOp::create(
            rewriter, op.getLoc(), TypeRange{pivotCuSolverType},
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
        auto permutation1Indexed = stablehlo::AddOp::create(
            rewriter, op.getLoc(),
            stablehlo::ConstantOp::create(
                rewriter, op.getLoc(), pivotCuSolverType,
                cast<ElementsAttr>(makeAttr(pivotCuSolverType, 1))),
            permutation.getResult(0));

        rewriter.replaceAllUsesWith(
            op.getResult(2),
            stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                         permutation1Indexed));
      }

      rewriter.replaceAllUsesWith(op.getResult(0), cusolverffi.getResult(0));
      rewriter.replaceAllUsesWith(
          op.getResult(1),
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                       cusolverffi.getResult(1)));
      rewriter.replaceAllUsesWith(
          op.getResult(3),
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), infoType,
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
      auto customCall = stablehlo::CustomCallOp::create(
          rewriter, op.getLoc(),
          TypeRange{inputType, pivotTPUType, permutationType},
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
      auto pivots1Indexed = stablehlo::AddOp::create(
          rewriter, op.getLoc(),
          stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), pivotType,
              cast<ElementsAttr>(makeAttr(pivotType, 1))),
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                       customCall.getResult(1)));

      auto permutation1Indexed = stablehlo::AddOp::create(
          rewriter, op.getLoc(),
          stablehlo::ConstantOp::create(
              rewriter, op.getLoc(), permutationType,
              cast<ElementsAttr>(makeAttr(permutationType, 1))),
          stablehlo::ConvertOp::create(rewriter, op.getLoc(), permutationType,
                                       customCall.getResult(2)));

      auto isFinite = stablehlo::AndOp::create(
          rewriter, op.getLoc(),
          stablehlo::IsFiniteOp::create(
              rewriter, op.getLoc(),
              stablehlo::RealOp::create(rewriter, op.getLoc(),
                                        customCall.getResult(0))),
          stablehlo::IsFiniteOp::create(
              rewriter, op.getLoc(),
              stablehlo::ImagOp::create(rewriter, op.getLoc(),
                                        customCall.getResult(0))));

      SmallVector<int64_t> reductionDims;
      for (int i = numBatchDims; i < inputRank; i++)
        reductionDims.push_back(i);
      auto initValType = RankedTensorType::get({}, rewriter.getI1Type());
      auto initVal = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), initValType,
          cast<ElementsAttr>(makeAttr(initValType, 1)));

      auto allFinite = stablehlo::ReduceOp::create(
          rewriter, op.getLoc(),
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
        auto andOp = stablehlo::AndOp::create(rewriter, op.getLoc(), lhs, rhs);

        stablehlo::ReturnOp::create(rewriter, op.getLoc(),
                                    ValueRange{andOp.getResult()});
      }

      // info == 0 if all finite (success)
      auto info = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), infoType,
          stablehlo::NotOp::create(rewriter, op.getLoc(),
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

    const int64_t m = inputShape[inputRank - 2];
    const int64_t n = inputShape[inputRank - 1];
    const int64_t numBatchDims = inputRank - 2;
    const bool isfull = op.getFull();

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
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
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

      auto op_func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                              func_type);
      rewriter.setInsertionPointToStart(op_func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto op_layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto op_jobu = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, isfull ? 'A' : 'S'));
      auto op_jobvt = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, isfull ? 'A' : 'S'));
      auto op_m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[0]));
      auto op_n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[1]));
      auto op_lda = op_m;
      auto op_ldu = op_m;
      auto op_ldvt = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int,
                                  isfull ? n : std::min(m, n)));

      // call to `lapacke_*gesvd`
      auto llvm_call_op = LLVM::CallOp::create(
          rewriter, op.getLoc(), TypeRange{type_llvm_lapack_int},
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

      LLVM::StoreOp::create(rewriter, op.getLoc(), llvm_call_op.getResult(),
                            op_func.getArgument(5));
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `gesvd` wrapper
    auto type_u = RankedTensorType::get({m, isfull ? m : std::min(m, n)},
                                        inputElementType);
    auto op_u = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_u, cast<ElementsAttr>(makeAttr(type_u, 0)));

    auto type_input_element_real = inputElementType;
    if (auto complex_type = dyn_cast<ComplexType>(inputElementType)) {
      type_input_element_real = complex_type.getElementType();
    }
    auto type_s =
        RankedTensorType::get({std::min(m, n)}, type_input_element_real);
    auto op_s = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_s, cast<ElementsAttr>(makeAttr(type_s, 0)));

    auto type_vt = RankedTensorType::get({isfull ? n : std::min(m, n), n},
                                         inputElementType);
    auto op_vt =
        stablehlo::ConstantOp::create(rewriter, op.getLoc(), type_vt,
                                      cast<ElementsAttr>(makeAttr(type_vt, 0)));

    auto type_superb =
        RankedTensorType::get({std::min(m, n) - 1}, inputElementType);
    auto op_superb = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_superb,
        cast<ElementsAttr>(makeAttr(type_superb, 0)));

    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto op_info = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, -1)));

    SmallVector<bool> isColMajorArr = {true, true, true, true, true, true};
    SmallVector<int64_t> operandRanks = {2, 2, 1, 2, 1, 0};
    SmallVector<int64_t> outputRanks = {2, 1, 2, 0};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    // alias for u
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{0}, 1, std::vector<int64_t>{}));
    // alias for s
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{1}, 2, std::vector<int64_t>{}));
    // alias for vt
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{2}, 3, std::vector<int64_t>{}));
    // alias for info
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{3}, 5, std::vector<int64_t>{}));

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_u, type_s, type_vt, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, op_u.getResult(), op_s.getResult(), op_vt.getResult(),
                   op_superb.getResult(), op_info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
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

    auto type_input_element_real = type_input.getElementType();
    if (auto complex_type = dyn_cast<ComplexType>(type_input_element_real)) {
      type_input_element_real = complex_type.getElementType();
    }
    auto type_s = cast<RankedTensorType>(RankedTensorType::get(
        {std::min(shape_input[0], shape_input[1])}, type_input_element_real));
    auto rank_s = type_s.getRank();

    auto type_vt = cast<RankedTensorType>(op.getResult(2).getType());
    auto rank_vt = type_vt.getRank();

    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto rank_info = type_info.getRank();
    auto op_info = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, -1)));

    // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {};
    SmallVector<int64_t> ranks_operands = {rank_input};
    SmallVector<int64_t> ranks_results = {rank_input, rank_s, rank_u, rank_vt,
                                          rank_info};
    SmallVector<bool> isColMajorArrOperands = {true};
    SmallVector<bool> isColMajorArrOutputs = {true, true, true, true, true};

    auto cusolver_call_op = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(),
        TypeRange{type_input, type_s, type_u, type_vt, type_info},
        ValueRange{input}, rewriter.getStringAttr("cusolver_gesvd_ffi"),
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
    cusolver_call_op->setAttr(rewriter.getStringAttr("full_matrices"),
                              rewriter.getBoolAttr(op.getFull()));
    cusolver_call_op->setAttr(rewriter.getStringAttr("compute_uv"),
                              rewriter.getBoolAttr(true));
    cusolver_call_op->setAttr(rewriter.getStringAttr("transposed"),
                              rewriter.getBoolAttr(false));

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

    return rewriter.notifyMatchFailure(
        op, "We don't know yet to which SVD TPU kernel to lower to :_(");

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

    auto custom_call_op = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_input, type_tau},
        ValueRange{input}, rewriter.getStringAttr("Svd"),
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
    auto info_op = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, 0)));
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
    patterns.add<SVDFactorizationOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
