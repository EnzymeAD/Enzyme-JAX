#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
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
    if (backend == "cpu") {
      return matchAndRewriteCPU(op, rewriter);
    } else if (backend == "cuda") {
      return matchAndRewriteCUDA(op, rewriter);
    } else if (backend == "tpu") {
      return matchAndRewriteTPU(op, rewriter);
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }

private:
  func::FuncOp createWrapperFuncOpCPULapack(
      PatternRewriter &rewriter, const std::string &lapackFn,
      RankedTensorType inputType, RankedTensorType blasPivotType,
      RankedTensorType blasInfoType, Type blasIntType,
      const std::string &fnName, enzymexla::LUFactorizationOp op,
      ArrayAttr operandLayouts, ArrayAttr resultLayouts,
      ArrayAttr outputOperandAliases) const {
    auto ctx = op->getContext();

    OpBuilder::InsertionGuard guard(rewriter);
    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return nullptr;
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    SmallVector<Type> argTypes = {inputType};
    SmallVector<Type> retTypes = {inputType, blasPivotType, blasInfoType};

    FunctionType calleeType = rewriter.getFunctionType(argTypes, retTypes);
    func::FuncOp func =
        func::FuncOp::create(rewriter, op.getLoc(), fnName, calleeType);
    func.setPrivate();

    auto &entryBlock = *func.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    auto input = entryBlock.getArgument(0);
    auto mSize = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), RankedTensorType::get({}, blasIntType),
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), input, 0));
    auto nSize = stablehlo::ConvertOp::create(
        rewriter, op.getLoc(), RankedTensorType::get({}, blasIntType),
        stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), input, 1));
    auto pivot = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), blasPivotType,
        cast<ElementsAttr>(makeAttr(blasPivotType, -1)));
    auto info = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), blasInfoType,
        cast<ElementsAttr>(makeAttr(blasInfoType, -1)));

    auto jitCall = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(),
        TypeRange{inputType, blasPivotType, blasInfoType},
        mlir::FlatSymbolRefAttr::get(ctx, lapackFn),
        ValueRange{mSize, nSize, input, mSize, pivot, info},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/outputOperandAliases,
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    func::ReturnOp::create(rewriter, op.getLoc(),
                           ValueRange{jitCall.getResult(0),
                                      jitCall.getResult(1),
                                      jitCall.getResult(2)});

    return func;
  }

  LogicalResult matchAndRewriteCPU(enzymexla::LUFactorizationOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();

    auto input = op.getOperand();
    auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputType = cast<RankedTensorType>(input.getType());
    auto unbatchedInputType = RankedTensorType::get(
        SmallVector<int64_t>(inputType.getShape().end() - 2,
                             inputType.getShape().end()),
        inputType.getElementType());
    auto inputElementType = inputType.getElementType();

    const int64_t numBatchDims = inputRank - 2;

    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto pivotRank = pivotType.getRank();
    auto unbatchedPivotType = RankedTensorType::get(
        SmallVector<int64_t>(pivotType.getShape().end() - 1,
                             pivotType.getShape().end()),
        pivotType.getElementType());

    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());
    auto infoRank = infoType.getRank();
    auto unbatchedInfoType =
        RankedTensorType::get({}, infoType.getElementType());

    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto blasIntType = rewriter.getIntegerType(blasIntWidth);
    auto intType = RankedTensorType::get({}, blasIntType);
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidPtrType = LLVM::LLVMVoidType::get(ctx);

    std::string lapackFn;
    auto prefix = lapackPrecisionPrefix(inputElementType);
    if (prefix) {
      lapackFn = "enzymexla_lapack_" + *prefix + "getrf_";
    } else {
      op->emitOpError() << "Unsupported input element type: "
                        << inputElementType;
      return rewriter.notifyMatchFailure(op, "unsupported input element type");
    }
    std::string lapackFnWrapper = lapackFn + "wrapper";

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

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(lapackFnWrapper)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidPtrType,
                                      {llvmPtrType, llvmPtrType, llvmPtrType,
                                       llvmPtrType, llvmPtrType, llvmPtrType},
                                      false);

      auto funcOp =
          LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), lapackFnWrapper,
                                   funcType, LLVM::Linkage::Private);
      rewriter.setInsertionPointToStart(funcOp.addEntryBlock(rewriter));

      funcOp.setArgAttr(0, LLVM::LLVMDialect::getReadonlyAttrName(),
                        rewriter.getUnitAttr());
      funcOp.setArgAttr(1, LLVM::LLVMDialect::getReadonlyAttrName(),
                        rewriter.getUnitAttr());
      // 2 is read + write
      funcOp.setArgAttr(3, LLVM::LLVMDialect::getReadonlyAttrName(),
                        rewriter.getUnitAttr());
      funcOp.setArgAttr(4, LLVM::LLVMDialect::getWriteOnlyAttrName(),
                        rewriter.getUnitAttr());
      funcOp.setArgAttr(5, LLVM::LLVMDialect::getWriteOnlyAttrName(),
                        rewriter.getUnitAttr());
      for (int i = 0; i < 6; i++) {
        funcOp.setArgAttr(i, LLVM::LLVMDialect::getNoFreeAttrName(),
                          rewriter.getUnitAttr());
      }

      auto callOp = LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                                         SymbolRefAttr::get(ctx, lapackFn),
                                         ValueRange{
                                             funcOp.getArgument(0),
                                             funcOp.getArgument(1),
                                             funcOp.getArgument(2),
                                             funcOp.getArgument(3),
                                             funcOp.getArgument(4),
                                             funcOp.getArgument(5),
                                         });
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // Call the LLVM function with enzymexla.jit_call
    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{0}, 2, std::vector<int64_t>{}));
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{1}, 4, std::vector<int64_t>{}));
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
        ctx, std::vector<int64_t>{2}, 5, std::vector<int64_t>{}));

    auto unbatchedBLASPivotType = RankedTensorType::get(
        unbatchedPivotType.getShape(), rewriter.getIntegerType(blasIntWidth));
    auto blasPivotType = RankedTensorType::get(
        pivotType.getShape(), rewriter.getIntegerType(blasIntWidth));
    auto unbatchedBLASInfoType = RankedTensorType::get(
        unbatchedInfoType.getShape(), rewriter.getIntegerType(blasIntWidth));
    auto blasInfoType = RankedTensorType::get(
        infoType.getShape(), rewriter.getIntegerType(blasIntWidth));

    auto operandLayouts =
        getSHLOLayout(rewriter, SmallVector<int64_t, 6>{0, 0, 2, 0, 1, 0},
                      SmallVector<bool, 6>(6, true), 2);
    auto resultLayouts =
        getSHLOLayout(rewriter, SmallVector<int64_t, 3>{2, 1, 0},
                      SmallVector<bool, 3>(3, true), 2);

    Value factorizedResult, pivotResult, infoResult;
    static int64_t fnNum = 0;
    std::string wrapperFnName = lapackFn + std::to_string(fnNum++);

    func::FuncOp func = createWrapperFuncOpCPULapack(
        rewriter, lapackFnWrapper, unbatchedInputType, unbatchedBLASPivotType,
        unbatchedBLASInfoType, blasIntType, wrapperFnName, op, operandLayouts,
        resultLayouts, rewriter.getArrayAttr(aliases));
    if (!func)
      return rewriter.notifyMatchFailure(op,
                                         "failed to create wrapper function");

    SmallVector<enzyme::BatchOp> batchOps;
    SmallVector<FunctionOpInterface> batchFunctions;

    if (numBatchDims > 0) {
      // TODO: Implement batched LU factorizations by directly calling MKL
      //       https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/getrf-batch-strided.html.
      SmallVector<int64_t> batchShape(inputShape.begin(),
                                      inputShape.begin() + numBatchDims);

      auto batchOp = enzyme::BatchOp::create(
          rewriter, op.getLoc(),
          TypeRange{inputType, blasPivotType, blasInfoType},
          mlir::FlatSymbolRefAttr::get(op.getContext(), wrapperFnName),
          ValueRange{input}, rewriter.getDenseI64ArrayAttr(batchShape));

      factorizedResult = batchOp.getResult(0);
      pivotResult = batchOp.getResult(1);
      infoResult = batchOp.getResult(2);

      batchOps.push_back(batchOp);
      batchFunctions.push_back(cast<FunctionOpInterface>(func.getOperation()));
    } else {
      auto callOp =
          func::CallOp::create(rewriter, op.getLoc(), func, ValueRange{input});

      factorizedResult = callOp.getResult(0);
      pivotResult = callOp.getResult(1);
      infoResult = callOp.getResult(2);
    }

    auto iterType = RankedTensorType::get({}, rewriter.getI32Type());
    auto iter = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), iterType,
        cast<ElementsAttr>(makeAttr(iterType, 0)));
    auto zeroConst = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), iterType,
        cast<ElementsAttr>(makeAttr(iterType, 0)));

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
          RankedTensorType::get(sliceShape, cast<RankedTensorType>(
                                                block->getArgument(1).getType())
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
        block->addArgument(RankedTensorType::get({}, blasIntType), op.getLoc());
        block->addArgument(RankedTensorType::get({}, blasIntType), op.getLoc());
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
        op.getResult(1), stablehlo::ConvertOp::create(rewriter, op.getLoc(),
                                                      pivotType, pivotResult));
    rewriter.replaceAllUsesWith(
        op.getResult(2),
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                     finalPermutation));
    rewriter.replaceAllUsesWith(
        op.getResult(3), stablehlo::ConvertOp::create(rewriter, op.getLoc(),
                                                      infoType, infoResult));

    std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
        batchedFunctionCache;
    for (auto [batchOp, func] : llvm::zip(batchOps, batchFunctions)) {
      if (failed(enzyme::batchutils::batchOperation(rewriter, batchOp, func,
                                                    batchedFunctionCache))) {
        return rewriter.notifyMatchFailure(op, "failed to batch operation");
      }
    }

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::LUFactorizationOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();

    auto input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();
    auto numBatchDims = inputRank - 2;

    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto pivotRank = pivotType.getRank();
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());
    auto infoRank = infoType.getRank();

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
        getSHLOLayout(rewriter, operandRanks, isColMajorArrOperands, inputRank),
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
          getSHLOLayout(rewriter, SmallVector<int64_t>{pivotRank},
                        SmallVector<bool>{true}, inputRank),
          /*result_layouts*/
          getSHLOLayout(rewriter, SmallVector<int64_t>{pivotRank},
                        SmallVector<bool>{true}, inputRank),
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
  }

  LogicalResult matchAndRewriteTPU(enzymexla::LUFactorizationOp op,
                                   PatternRewriter &rewriter) const {
    auto input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();

    auto inputRank = inputType.getRank();
    auto numBatchDims = inputRank - 2;

    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());

    SmallVector<int64_t> permutationShape(inputShape.begin(),
                                          inputShape.end() - 2);
    permutationShape.push_back(inputShape[inputRank - 2]);
    auto permutationType =
        RankedTensorType::get(permutationShape, rewriter.getI32Type());

    auto pivotTPUType =
        RankedTensorType::get(pivotType.getShape(), rewriter.getI32Type());

    // TPU returns (LU, pivots, permutation). info isn't returned. based on
    // how JAX operates, I am assuming info != 0 when there is a nan in the
    // output.
    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(),
        TypeRange{inputType, pivotTPUType, permutationType}, ValueRange{input},
        rewriter.getStringAttr("LuDecomposition"),
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
      auto *block = rewriter.createBlock(
          &region, {}, {initValType, initValType}, {op.getLoc(), op.getLoc()});

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
      return matchAndRewriteCPU(op, rewriter);
    else if (backend == "cuda")
      return matchAndRewriteCUDA(op, rewriter);
    else if (backend == "tpu")
      return matchAndRewriteTPU(op, rewriter);
    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  // TODO support more SVD algorithms (e.g. `gesdd`, `gesvj`)
  LogicalResult matchAndRewriteCPU(enzymexla::SVDFactorizationOp op,
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
    if (auto prefix = lapackPrecisionPrefix(inputElementType)) {
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
              op_layout.getResult(), op_jobu.getResult(), op_jobvt.getResult(),
              op_m.getResult(), op_n.getResult(),
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

  LogicalResult matchAndRewriteCUDA(enzymexla::SVDFactorizationOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();

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

  LogicalResult matchAndRewriteTPU(enzymexla::SVDFactorizationOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();

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
        ValueRange{input}, rewriter.getStringAttr("SVD"),
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

    patterns.add<LUFactorizationOpLowering, SVDFactorizationOpLowering>(
        backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
