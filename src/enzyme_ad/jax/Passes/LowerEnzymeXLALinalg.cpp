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
using namespace mlir::enzymexla;

struct TridiagonalSolveOpLowering
    : public OpRewritePattern<enzymexla::TridiagonalSolveOp> {
  using OpRewritePattern::OpRewritePattern;

  std::string backend;
  int64_t blasIntWidth;
  TridiagonalSolveOpLowering(std::string backend, int64_t blasIntWidth,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::TridiagonalSolveOp op,
                                PatternRewriter &rewriter) const override {

    auto BType = cast<RankedTensorType>(op.getB().getType());
    auto nBatchDims = BType.getRank() - 2;

    if (backend == "cpu" && nBatchDims == 0) {
      return matchAndRewriteCPU(op, rewriter);
    } else if (backend == "cuda") {
      return matchAndRewriteCUDA(op, rewriter);
    }
    return matchAndRewriteFallback(op, rewriter);
  }

  LogicalResult matchAndRewriteCPU(enzymexla::TridiagonalSolveOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    Value dl = op.getDL();
    Value d = op.getD();
    Value du = op.getDU();
    Value B = op.getB();

    auto dlType = cast<RankedTensorType>(dl.getType());
    auto dType = cast<RankedTensorType>(d.getType());
    auto duType = cast<RankedTensorType>(dl.getType());
    auto bType = cast<RankedTensorType>(B.getType());

    if (!dlType || !dType || !duType || !bType)
      return rewriter.notifyMatchFailure(
          op, "operand types not ranked tensor types");

    if (!dlType.hasRank() || !dType.hasRank() || !duType.hasRank() ||
        !bType.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    if (dlType.getRank() != 1 || dType.getRank() != 1 ||
        duType.getRank() != 1 || bType.getRank() > 2)
      return rewriter.notifyMatchFailure(
          op, "only 2D matrices supported for gtsv on CPU");

    Type elementType = dlType.getElementType();
    auto blasIntType = rewriter.getIntegerType(blasIntWidth);
    auto intType = RankedTensorType::get({}, blasIntType);
    auto uint8Type =
        RankedTensorType::get({}, rewriter.getIntegerType(8, false));
    auto llvmIntType = typeConverter.convertType(blasIntType);
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);

    std::string lapackFn;
    if (auto prefix = lapackPrecisionPrefix(elementType)) {
      lapackFn = "enzymexla_lapack_" + *prefix + "gtsv_";
    } else {
      op->emitOpError() << "Unsupported element type: " << elementType;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(lapackFn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType,
                                                  {
                                                      llvmPtrType, // n
                                                      llvmPtrType, // nrhs
                                                      llvmPtrType, // dl
                                                      llvmPtrType, // d
                                                      llvmPtrType, // du
                                                      llvmPtrType, // b
                                                      llvmPtrType, // ldb
                                                      llvmPtrType, // info
                                                  },
                                                  false);
      rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), lapackFn, funcType,
                                        LLVM::Linkage::External);
    }

    static int64_t fn_counter = 0;
    std::string funcFnName =
        blasFnWrapper + "wrapper_" + std::to_string(fn_counter++);

    SmallVector<bool> isColMajorArrOperands(8, true);
    SmallVector<int64_t> operandRanks = {0, 0, 1, 1, 1, 2, 0, 0};

    SmallVector<bool> isColMajorArrResults(5, true);
    SmallVector<int64_t> outputRanks = {1, 1, 1, 2, 0};

    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArrOperands, 2);
    auto resultLayouts =
        getSHLOLayout(rewriter, outputRanks, isColMajorArrResults, 2);

    SmallVector<Attribute> aliases = {
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 2, {}), // dl
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 3, {}), // d
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 4, {}), // du
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 5, {}), // b
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 7, {}), // info
    };

    func::FuncOp shloFunc;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      SmallVector<Type> argTypes = {
          op.getDl().getType(),
          op.getD().getType(),
          op.getDu().getType(),
          op.getB().getType(),
      };
      SmallVector<Type> retTypes = {op.getB().getType()};

      auto calleeType = rewriter.getFunctionType(argTypes, retTypes);
      shloFunc =
          func::FuncOp::create(rewriter, op.getLoc(), funcFnName, calleeType);
      shloFunc.setPrivate();

      auto &entryBlock = *shloFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(&entryBlock);

      auto dl = entryBlock.getArgument(0);
      auto d = entryBlock.getArgument(1);
      auto du = entryBlock.getArgument(2);
      auto b = entryBlock.getArgument(3);

      auto n = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), d, 0));
      auto nrhs = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), B, 1));
      auto ldb = stablehlo::ConvertOp::create(
          rewriter, op.getLoc(), intType,
          stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), B, 0));
      auto info = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), intType,
          cast<ElementsAttr>(makeAttr(intType, -1)));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(),
          TypeRange{op.getDl().getType(), op.getD().getType(),
                    op.getDu().getType(), op.getB().getType(), intType},
          mlir::FlatSymbolRefAttr::get(ctx, lapackFn),
          ValueRange{n, nrhs, dl, d, du, b, ldb, info},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/operandLayouts,
          /*result_layouts=*/resultLayouts,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/rewriter.getUnitAttr());

      func::ReturnOp::create(rewriter, op.getLoc(),
                             ValueRange{jitCall.getResult(3)}); // B
    }
    auto callOp = func::CallOp::create(rewriter, op.getLoc(), shloFunc,
                                       ValueRange{dl, d, du, B});

    rewriter.replaceOp(op, callOp);

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::TridiagonalSolveOp op,
                                    PatternRewriter &rewriter) const {
    auto dlType = cast<RankedTensorType>(op.getDl().getType());
    auto dType = cast<RankedTensorType>(op.getD().getType());
    auto dRank = dType.getRank();
    auto bType = cast<RankedTensorType>(op.getBl().getType());
    auto bRank = bType.getRank();

    if (!dlType || !dType || !duType || !bType)
      return rewriter.notifyMatchFailure(
          op, "operand types not ranked tensor types");

    if (!dlType.hasRank() || !dType.hasRank() || !duType.hasRank() ||
        !bType.hasRank())
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    if (dlType.getRank() != 1 || dType.getRank() != 1 ||
        duType.getRank() != 1 || bType.getRank() > 2)
      return rewriter.notifyMatchFailure(
          op, "only 2D matrices supported for gtsv on CUDA");

    SmallVector<Value> operands;
    SmallVector<int64_t> operandRanks;
    SmallVector<bool> areColMajor;

    auto dl_in = op.getDl();
    auto d = op.getD();
    auto du_in = op.getDu();
    auto B = op.getB();

    // In TridiagonalSolveOp, dl and du are of length n-1, but cuBLAS requires
    // them to be padded to length n, where dl[0] = 0 and du[n-1] = 0.
    int64_t n = dType.getShape()[0]; // target size

    auto elemTy = dlType.getElementType();
    auto outType = RankedTensorType::get({n}, elemTy);

    // zero vector
    auto zeroAttr = rewriter.getZeroAttr(elemTy);
    auto zero_du = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({}, elemTy), zeroAttr));
    auto zero_dl = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(),
        DenseElementsAttr::get(RankedTensorType::get({}, elemTy), zeroAttr));

    auto zeroTensorDl = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, outType, zero,
        DenseI64ArrayAttr::get(rewriter.getContext(), {}));
    // insert dl_in at index 1
    llvm::SmallVector<int64_t> dl_start(rank, 0);
    start[rank - 1] = 1;

    auto dl = stablehlo::DynamicUpdateSliceOp::create(
        rewriter, loc, outType, zeroTensorDl, dl_in, dl_start);

    auto zeroTensorDu = stablehlo::BroadcastInDimOp::create(
        rewriter, loc, outType, zero,
        DenseI64ArrayAttr::get(rewriter.getContext(), {}));
    // insert du_in at index 0
    llvm::SmallVector<int64_t> du_start(rank, 0);

    auto du = stablehlo::DynamicUpdateSliceOp::create(
        rewriter, loc, outType, zeroTensorDu, du_in, du_start);

    StringAttr customCallTarget;
    ArrayAttr aliases;

    customCallTarget = rewriter.getStringAttr("cusparse_gtsv2_ffi");
    operands = {dl, d, du, B};
    operandRanks = {dRank, dRank, dRank, bRank};
    aliases = rewriter.getArrayAttr(
        {stablehlo::OutputOperandAliasAttr::get(op.getContext(), {}, 3, {})});
    areColMajor = {true, true, true, true};

    DictionaryAttr backendConfig = rewriter.getDictionaryAttr({});

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{bType}, operands, customCallTarget,
        /*has_side_effect*/ nullptr,
        /*backend_config*/ backendConfig,
        /*api_version*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/
        getSHLOLayout(rewriter, operandRanks, areColMajor, bRank),
        /*result_layouts*/
        getSHLOLayout(rewriter, {bRank}, SmallVector<bool>(bRank, true), bRank),
        /*output_operand_aliases*/ aliases);

    Value result = customCall.getResult(0);
    rewriter.replaceAllUsesWith(op.getResult(), result);

    return success();
  }

  LogicalResult matchAndRewriteFallback(enzymexla::TridiagonalSolveOp op,
                                        PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    auto dl = op.getDl();
    auto d = op.getD();
    auto du = op.getDu();

    Value B = op.getB();

    auto dType = cast<RankedTensorType>(d.getType());
    int64_t n = dType.getShape()[0];

    auto elemTy = dType.getElementType();

    // build scatter index tensor
    SmallVector<int64_t> indicesData;
    indicesData.reserve(3 * n * 2);

    // diagonal (i,i)
    for (int64_t i = 0; i < n; i++) {
      indicesData.push_back(i);
      indicesData.push_back(i);
    }
    // lower (i+1,i)
    for (int64_t i = 0; i < n - 1; i++) {
      indicesData.push_back(i + 1);
      indicesData.push_back(i);
    }
    // upper (i,i+1)
    for (int64_t i = 0; i < n - 1; i++) {
      indicesData.push_back(i);
      indicesData.push_back(i + 1);
    }

    auto indicesType =
        RankedTensorType::get({3 * n - 2, 2}, rewriter.getI64Type());
    auto indicesAttr = DenseElementsAttr::get(indicesType, indicesData);
    Value indices = stablehlo::ConstantOp::create(rewriter, loc, indicesAttr);

    auto matrixType = RankedTensorType::get({n, n}, elemTy);

    auto zeroAttr =
        DenseElementsAttr::get(matrixType, rewriter.getZeroAttr(elemTy));

    Value zero = stablehlo::ConstantOp::create(rewriter, loc, zeroAttr);

    Value values = stablehlo::ConcatenateOp::create(
        rewriter, loc, RankedTensorType::get({3 * n - 2}, elemTy),
        ValueRange{d, dl, du},
        /*dimension=*/0);

    auto scatterDims = stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*update_window_dims=*/ArrayRef<int64_t>{},
        /*inserted_window_dims=*/ArrayRef<int64_t>{0, 1},
        /*scatter_dims_to_operand_dims=*/ArrayRef<int64_t>{0, 1},
        /*index_vector_dim=*/1);

    // reconstructed A
    auto scatterOp = stablehlo::ScatterOp::create(
        rewriter, loc, matrixType,
        /*inputs=*/ValueRange{zero},
        /*scatter_indices=*/indices,
        /*updates=*/ValueRange{values}, scatterDims,
        /*indices_are_sorted=*/false,
        /*unique_indices=*/true);
    Value A = scatterOp.getResult();

    auto lu =
        enzymexla::LUFactorizationOp::create(rewriter, loc, A.getType(), A);

    Value LU = lu.getResult(0);
    Value perm = lu.getResult(2);

    auto permType = cast<RankedTensorType>(perm.getType());
    auto idxType = RankedTensorType::get({n, 1}, permType.getElementType());
    Value indices = stablehlo::ReshapeOp::create(rewriter, loc, idxType, perm);

    auto dnums = stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offset_dims=*/ArrayRef<int64_t>{1}, // result keeps column dim
        /*collapsed_slice_dims=*/ArrayRef<int64_t>{0}, // collapse row dim
        /*start_index_map=*/ArrayRef<int64_t>{0},      // index maps to row dim
        /*index_vector_dim=*/1);
    SmallVector<int64_t> sliceSizes = {1, BType.getShape()[1]};

    Value B_perm = stablehlo::GatherOp::create(rewriter, loc, B.getType(), B,
                                               indices, dnums, sliceSizes);

    // solve Ly = B_perm
    Value y = stablehlo::TriangularSolveOp::create(
        rewriter, loc, B.getType(),
        LU, // we can directly use the LU buffer and indicate unit diagonal is
            // true
        B_perm,
        /*left_side=*/true,
        /*lower=*/true,
        /*unit_diagonal=*/true,
        /*transpose_a=*/false);

    // solve Ux = y
    Value x =
        stablehlo::TriangularSolveOp::create(rewriter, loc, B.getType(), LU, y,
                                             /*left_side=*/true,
                                             /*lower=*/false,
                                             /*unit_diagonal=*/false,
                                             /*transpose_a=*/false);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LUFactorizationOpLowering
    : public OpRewritePattern<enzymexla::LUFactorizationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::LUFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    auto getrfOp = enzymexla::GetrfOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), op.getInput());
    rewriter.replaceOp(op, getrfOp);
    return success();
  }
};

struct SVDFactorizationOpLowering
    : public OpRewritePattern<enzymexla::SVDFactorizationOp> {
  std::string backend;

  SVDFactorizationOpLowering(std::string backend, MLIRContext *context)
      : OpRewritePattern(context), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::SVDFactorizationOp op,
                                PatternRewriter &rewriter) const override {
    SVDAlgorithm algorithm = op.getAlgorithm();
    if (algorithm == SVDAlgorithm::DEFAULT) {
      if (backend == "cpu") {
        algorithm = SVDAlgorithm::DivideAndConquer;
      } else if (backend == "cuda" || backend == "tpu" || backend == "rocm") {
        algorithm = SVDAlgorithm::Jacobi;
      } else {
        op->emitOpError() << "Unsupported backend: " << backend;
        return failure();
      }
    }

    bool computeUv =
        !op.getResult(0).use_empty() || !op.getResult(2).use_empty();

    switch (algorithm) {
    case SVDAlgorithm::QRIteration: {
      auto gesddOp = enzymexla::GesvdOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesddOp);
      break;
    }
    case SVDAlgorithm::DivideAndConquer: {
      auto gesvjOp = enzymexla::GesddOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesvjOp);
      break;
    }
    case SVDAlgorithm::Jacobi: {
      auto gesvdOp = enzymexla::GesvjOp::create(
          rewriter, op.getLoc(), op->getResultTypes(), op.getInput(),
          /*full=*/op.getFull(), computeUv);
      rewriter.replaceOp(op, gesvdOp);
      break;
    }
    case SVDAlgorithm::DEFAULT:
      llvm_unreachable("Default should have already been handled");
    }

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

    patterns.add<LUFactorizationOpLowering>(context);
    patterns.add<SVDFactorizationOpLowering>(backend, context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // Verify that all illegal ops have been lowered
    auto walkResult = getOperation()->walk([&](Operation *op) {
      if (isa<enzymexla::LUFactorizationOp, enzymexla::SVDFactorizationOp>(
              op)) {
        op->emitError("Failed to lower enzymexla linalg operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
