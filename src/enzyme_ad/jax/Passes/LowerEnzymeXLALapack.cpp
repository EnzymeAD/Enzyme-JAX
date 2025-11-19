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

#define DEBUG_TYPE "lower-enzymexla-lapack"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLALAPACKPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

struct GeqrfOpLowering : public OpRewritePattern<enzymexla::GeqrfOp> {
  std::string backend;
  int64_t blasIntWidth;

  GeqrfOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GeqrfOp op,
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
  LogicalResult matchAndRewriteCPU(enzymexla::GeqrfOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputElementType = inputType.getElementType();

    const int64_t numBatchDims = inputRank - 2;

    if (numBatchDims > 0) {
      return rewriter.notifyMatchFailure(
          op, "QR factorization with batch dimensions is not yet supported");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);

    std::string fn = "geqrf_";
    if (auto prefix = lapackPrecisionPrefix(inputElementType)) {
      fn = *prefix + fn;
    } else {
      op->emitOpError() << "Unsupported element type: " << inputElementType;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
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
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
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

      auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[0]));
      auto n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[1]));
      auto lda = m;

      // call to `lapacke_*geqrf*`
      auto res = LLVM::CallOp::create(rewriter, op.getLoc(),
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

      LLVM::StoreOp::create(rewriter, op.getLoc(), res.getResult(),
                            func.getArgument(2));
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `geqrf` wrapper
    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto info = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, -1)));

    auto tsize = std::min(inputShape.front(), inputShape.back());
    auto type_tau = RankedTensorType::get({tsize}, inputElementType);
    auto tau = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_tau,
        cast<ElementsAttr>(makeAttr(type_tau, 0)));

    SmallVector<bool> isColMajorArr = {true, true, true};
    SmallVector<int64_t> operandRanks = {2, 1, 0};
    SmallVector<int64_t> outputRanks = {2, 1, 0};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    for (int i = 0; i < 3; ++i) {
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, i, {}));
    }

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{inputType, type_tau, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, tau.getResult(), info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.lapack.geqrf with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(0), jit_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), jit_call_op.getResult(1));
    rewriter.replaceAllUsesWith(op.getResult(2), jit_call_op.getResult(2));
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::GeqrfOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto rank_input = type_input.getRank();

    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());
    auto rank_tau = type_tau.getRank();

    // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {
        stablehlo::OutputOperandAliasAttr::get(ctx, {0}, 0, {})};
    SmallVector<int64_t> ranks_operands = {rank_input};
    SmallVector<int64_t> ranks_results = {rank_input, rank_tau};
    SmallVector<bool> isColMajorArrOperands = {true};
    SmallVector<bool> isColMajorArrOutputs = {true, true};

    auto cusolver_call_op = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_input, type_tau},
        ValueRange{input}, rewriter.getStringAttr("cusolver_geqrf_ffi"),
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
    auto info_op = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, 0)));
    rewriter.replaceAllUsesWith(op.getResult(2), info_op.getResult());

    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewriteTPU(enzymexla::GeqrfOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());

    // emit `stablehlo.custom_call` to `@Qr` kernel from XLA
    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_input, type_tau},
        ValueRange{input}, rewriter.getStringAttr("Qr"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(0), customCall.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), customCall.getResult(1));

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

// NOTE CUDA (cuSOLVER) and TPU (XLA) do not have specific implementations :(
// but could work if we lower directly to StableHLO
struct GeqrtOpLowering : public OpRewritePattern<enzymexla::GeqrtOp> {
  std::string backend;
  int64_t blasIntWidth;

  GeqrtOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GeqrtOp op,
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

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewriteCPU(enzymexla::GeqrtOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputElementType = inputType.getElementType();

    const int64_t numBatchDims = inputRank - 2;

    if (numBatchDims > 0) {
      return rewriter.notifyMatchFailure(
          op, "QR factorization with batch dimensions is not yet supported");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);

    std::string fn = "geqrt_";
    if (auto prefix = lapackPrecisionPrefix(inputElementType)) {
      fn = *prefix + fn;
    } else {
      op->emitOpError() << "Unsupported element type: " << inputElementType;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
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
                                          type_llvm_lapack_int, // nb
                                          type_llvm_ptr,        // *A
                                          type_llvm_lapack_int, // lda
                                          type_llvm_ptr,        // *T
                                          type_llvm_lapack_int  // ldt
                                      },
                                      false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
                               LLVM::Linkage::External);
    }

    // WARN probably will need another function name encoding if we call to
    // `geqrf`, `orgqr` or `ungqr` in other op insert wrapper function for
    // `geqrf`
    static int64_t fn_counter = 0;
    fn_counter++;

    wrapper_fn += std::to_string(fn_counter);
    int64_t ldt_value = 0;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto func_type = LLVM::LLVMFunctionType::get(type_llvm_void,
                                                   {
                                                       type_llvm_ptr, // A
                                                       type_llvm_ptr, // T
                                                       type_llvm_ptr, // info
                                                   },
                                                   false);

      auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[0]));
      auto n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, inputShape[1]));
      auto lda = m;

      int64_t nb_value = 0;
      if (op.getBlocksize()) {
        nb_value = op.getBlocksize().value();
        assert(std::min(inputShape[0], inputShape[1]) >= nb_value &&
               "Block size must be less than or equal to min(m, n)");
        assert(nb_value >= 1 &&
               "Block size must be greater than or equal to 1");
      } else {
        // default block size is min(m, n)
        nb_value = std::min(inputShape[0], inputShape[1]);
      }
      auto nb_attr = rewriter.getI64IntegerAttr(nb_value);
      auto nb_op = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                            type_llvm_lapack_int, nb_attr);
      // can reuse nb = ldt
      ldt_value = nb_value;
      auto ldt_op = nb_op;

      auto A = func.getArgument(0);
      auto T = func.getArgument(1);
      auto info = func.getArgument(2);

      // call to `lapacke_*geqrt*`
      auto res = LLVM::CallOp::create(rewriter, op.getLoc(),
                                      TypeRange{type_llvm_lapack_int},
                                      SymbolRefAttr::get(ctx, bind_fn),
                                      ValueRange{
                                          layout.getResult(),
                                          m.getResult(),
                                          n.getResult(),
                                          nb_op.getResult(),
                                          A,
                                          lda.getResult(),
                                          T,
                                          ldt_op.getResult(),
                                      });

      LLVM::StoreOp::create(rewriter, op.getLoc(), res.getResult(), info);
      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `geqrt` wrapper
    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto info = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_info,
        cast<ElementsAttr>(makeAttr(type_info, -1)));

    auto type_T = RankedTensorType::get(
        {ldt_value, std::min(inputShape[0], inputShape[1])}, inputElementType);
    auto T = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_T, cast<ElementsAttr>(makeAttr(type_T, 0)));

    SmallVector<bool> isColMajorArr = {true, true, true};
    SmallVector<int64_t> operandRanks = {2, 1, 0};
    SmallVector<int64_t> outputRanks = {2, 1, 0};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    for (int i = 0; i < 3; ++i) {
      aliases.push_back(
          stablehlo::OutputOperandAliasAttr::get(ctx, {i}, i, {}));
    }

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{inputType, type_T, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, T.getResult(), info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.lapack.geqrf with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(0), jit_call_op.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), jit_call_op.getResult(1));
    rewriter.replaceAllUsesWith(op.getResult(2), jit_call_op.getResult(2));
    rewriter.eraseOp(op);

    return success();
  }
};

struct OrgqrOpLowering : public OpRewritePattern<enzymexla::OrgqrOp> {
  std::string backend;
  int64_t blasIntWidth;

  OrgqrOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::OrgqrOp op,
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
  LogicalResult matchAndRewriteCPU(enzymexla::OrgqrOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand(0);
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto inputRank = static_cast<int64_t>(inputShape.size());
    auto inputElementType = inputType.getElementType();

    auto tau = op.getOperand(1);
    auto type_tau = cast<RankedTensorType>(tau.getType());
    auto rank_tau = type_tau.getRank();

    const int64_t numBatchDims = inputRank - 2;

    if (numBatchDims > 0) {
      return rewriter.notifyMatchFailure(
          op, "`enzymexla.lapack.orgqr` with batch dimensions on CPU is not "
              "yet supported");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);

    std::string fn = "gqr_";
    if (auto prefix = lapackPrecisionPrefix(inputElementType)) {
      if (prefix == "s" || prefix == "d")
        fn = *prefix + "or" + fn;
      else
        fn = *prefix + "un" + fn;
    } else {
      op->emitOpError() << "Unsupported element type: " << inputElementType;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
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
                                          type_llvm_lapack_int, // k
                                          type_llvm_ptr,        // A
                                          type_llvm_lapack_int, // lda
                                          type_llvm_ptr         // tau
                                      },
                                      false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
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
                                                   },
                                                   false);

      auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto mC = inputShape[0];
      auto m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, mC));
      auto nC = inputShape[1];
      auto n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nC));
      auto k_value = nC;
      auto k = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));
      auto lda = m;

      // call to `lapacke_*(or|un)gqr*`
      auto res = LLVM::CallOp::create(rewriter, op.getLoc(),
                                      TypeRange{type_llvm_lapack_int},
                                      SymbolRefAttr::get(ctx, bind_fn),
                                      ValueRange{
                                          layout.getResult(),
                                          m.getResult(),
                                          n.getResult(),
                                          k.getResult(),
                                          func.getArgument(0),
                                          lda.getResult(),
                                          func.getArgument(1),
                                      });

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `(or|un)gqr` wrapper
    SmallVector<bool> isColMajorArr = {true, true};
    SmallVector<int64_t> operandRanks = {2, 1};
    SmallVector<int64_t> outputRanks = {2};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {0}, 0, {}));

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{inputType},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn), ValueRange{input, tau},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.lapack.geqrf with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(), jit_call_op.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::OrgqrOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand(0);
    auto type_input = cast<RankedTensorType>(input.getType());
    auto rank_input = type_input.getRank();

    auto tau = op.getOperand(1);
    auto type_tau = cast<RankedTensorType>(tau.getType());
    auto rank_tau = type_tau.getRank();

    // emit `stablehlo.custom_call` to `@cusolver_orgqr_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {
        stablehlo::OutputOperandAliasAttr::get(ctx, {}, 0, {})};
    SmallVector<int64_t> ranks_operands = {rank_input, rank_tau};
    SmallVector<int64_t> ranks_results = {rank_input};
    SmallVector<bool> isColMajorArrOperands = {true, true};
    SmallVector<bool> isColMajorArrOutputs = {true};

    auto cusolver_call_op = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_input}, ValueRange{input, tau},
        rewriter.getStringAttr("cusolver_orgqr_ffi"),
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

    rewriter.replaceAllUsesWith(op.getResult(), cusolver_call_op.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewriteTPU(enzymexla::OrgqrOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand(0);
    auto type_input = cast<RankedTensorType>(input.getType());
    auto tau = op.getOperand(1);

    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{type_input}, ValueRange{input, tau},
        rewriter.getStringAttr("ProductOfElementaryHouseholderReflectors"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(), customCall.getResult(0));

    return success();
  }
};

struct OrmqrOpLowering : public OpRewritePattern<enzymexla::OrmqrOp> {
  std::string backend;
  int64_t blasIntWidth;

  OrmqrOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::OrmqrOp op,
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

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewriteCPU(enzymexla::OrmqrOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto A = op.getOperand(0);
    auto A_type = cast<RankedTensorType>(A.getType());
    auto A_shape = A_type.getShape();
    auto A_rank = static_cast<int64_t>(A_shape.size());
    auto A_eltype = A_type.getElementType();

    auto tau = op.getOperand(1);
    auto tau_type = cast<RankedTensorType>(tau.getType());
    auto tau_shape = tau_type.getShape();
    auto tau_rank = tau_type.getRank();
    auto tau_eltype = tau_type.getElementType();

    auto C = op.getOperand(2);
    auto C_type = cast<RankedTensorType>(C.getType());
    auto C_shape = C_type.getShape();
    auto C_rank = static_cast<int64_t>(C_shape.size());
    auto C_eltype = C_type.getElementType();

    auto output = op.getResult();
    auto output_type = cast<RankedTensorType>(output.getType());
    auto output_shape = output_type.getShape();
    auto output_rank = static_cast<int64_t>(output_shape.size());
    auto output_eltype = output_type.getElementType();

    auto side_value = op.getSide() == enzymexla::LapackSide::left ? 'L' : 'R';
    char trans_value = 'N';
    switch (op.getTranspose()) {
    case enzymexla::LapackTranspose::none:
      trans_value = 'N';
      break;
    case enzymexla::LapackTranspose::transpose:
      trans_value = 'T';
      break;
    case enzymexla::LapackTranspose::adjoint:
      trans_value = 'C';
      break;
    }

    assert(output_shape == C_shape && "`enzymexla.lapack.ormqr` requires `C` "
                                      "and `output` to have the same shape");
    assert(A_eltype == C_eltype && A_eltype == tau_eltype &&
           "`enzymexla.lapack.ormqr` requires the same element type for all "
           "operands");

    auto mA = A_shape[0];
    auto mC = C_shape[0];
    auto nC = C_shape[1];
    auto k_value = tau_shape[0];

    if (A_rank - 2 > 0 || C_rank - 2 > 0) {
      return rewriter.notifyMatchFailure(
          op, "`enzymexla.lapack.orgqr` with batch dimensions on CPU is not "
              "yet supported");
    }

    assert(A_shape[0] >= A_shape[1] &&
           "`lapack.ormqr` with wide QR not yet supported. use "
           "`stablehlo.dynamic_update_slice` first");
    assert(A_shape[1] == k_value &&
           "second dimension of A and dimension of tau must match");

    if (side_value == 'L') {
      assert(mC == mA && "for a left-sided multiplication, the first dimension "
                         "of C, must equal the first dimension of A");
      assert(mC >= k_value && "invalid number of reflectors: k should be <= m");
    } else { // side_value == 'R'
      assert(nC == mA && "for a right-sided multiplication, the second "
                         "dimension of C, must equal the first dimension of A");
      assert(nC >= k_value && "invalid number of reflectors: k should be <= n");
    }

    auto lda_value = A_shape[0];
    auto ldc_value = C_shape[0];

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);
    auto type_llvm_char = rewriter.getIntegerType(8);

    std::string fn = "mqr_";
    if (auto prefix = lapackPrecisionPrefix(A_eltype)) {
      if (prefix == "s" || prefix == "d")
        fn = *prefix + "or" + fn;
      else
        fn = *prefix + "un" + fn;
    } else {
      op->emitOpError() << "Unsupported element type: " << A_eltype;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
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
                                          type_llvm_char,       // side
                                          type_llvm_char,       // trans
                                          type_llvm_lapack_int, // m
                                          type_llvm_lapack_int, // n
                                          type_llvm_lapack_int, // k
                                          type_llvm_ptr,        // A
                                          type_llvm_lapack_int, // lda
                                          type_llvm_ptr,        // tau
                                          type_llvm_ptr,        // C
                                          type_llvm_lapack_int, // ldc
                                      },
                                      false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
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
                                                       type_llvm_ptr, // C
                                                   },
                                                   false);

      auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));

      auto side = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, side_value));

      auto trans = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, trans_value));

      auto m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, mC));

      auto n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nC));

      auto k = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));

      auto lda = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, lda_value));

      auto ldc = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldc_value));

      // call to `lapacke_*(or|un)mqr*`
      auto res = LLVM::CallOp::create(rewriter, op.getLoc(),
                                      TypeRange{type_llvm_lapack_int},
                                      SymbolRefAttr::get(ctx, bind_fn),
                                      ValueRange{
                                          layout.getResult(),
                                          side.getResult(),
                                          trans.getResult(),
                                          m.getResult(),
                                          n.getResult(),
                                          k.getResult(),
                                          func.getArgument(0),
                                          lda.getResult(),
                                          func.getArgument(1),
                                          func.getArgument(2),
                                          ldc.getResult(),
                                      });

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `(or|un)mqr` wrapper
    SmallVector<bool> isColMajorArr = {true, true, true};
    SmallVector<int64_t> operandRanks = {2, 1, 2};
    SmallVector<int64_t> outputRanks = {2};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, 2, {}));

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{C_type},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn), ValueRange{A, tau, C},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.lapack.geqrf with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(), jit_call_op.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

struct GemqrtOpLowering : public OpRewritePattern<enzymexla::GemqrtOp> {
  std::string backend;
  int64_t blasIntWidth;

  GemqrtOpLowering(std::string backend, int64_t blasIntWidth,
                   MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GemqrtOp op,
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

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewriteCPU(enzymexla::GemqrtOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto V = op.getOperand(0);
    auto V_type = cast<RankedTensorType>(V.getType());
    auto V_shape = V_type.getShape();
    auto V_rank = static_cast<int64_t>(V_shape.size());
    auto V_eltype = V_type.getElementType();

    auto T = op.getOperand(1);
    auto T_type = cast<RankedTensorType>(T.getType());
    auto T_shape = T_type.getShape();
    auto T_rank = T_type.getRank();
    auto T_eltype = T_type.getElementType();

    auto C = op.getOperand(2);
    auto C_type = cast<RankedTensorType>(C.getType());
    auto C_shape = C_type.getShape();
    auto C_rank = static_cast<int64_t>(C_shape.size());
    auto C_eltype = C_type.getElementType();

    auto output = op.getResult();
    auto output_type = cast<RankedTensorType>(output.getType());
    auto output_shape = output_type.getShape();
    auto output_rank = static_cast<int64_t>(output_shape.size());
    auto output_eltype = output_type.getElementType();

    auto side_value = op.getSide() == enzymexla::LapackSide::left ? 'L' : 'R';
    char trans_value = 'N';
    switch (op.getTranspose()) {
    case enzymexla::LapackTranspose::none:
      trans_value = 'N';
      break;
    case enzymexla::LapackTranspose::transpose:
      trans_value = 'T';
      break;
    case enzymexla::LapackTranspose::adjoint:
      trans_value = 'C';
      break;
    }

    assert(V_rank == 2 &&
           "`enzymexla.lapack.gemqrt` requires `V` to be a matrix");
    assert(T_rank == 2 &&
           "`enzymexla.lapack.gemqrt` requires `T` to be a matrix");
    assert(C_rank == 2 &&
           "`enzymexla.lapack.gemqrt` requires `C` to be a matrix");
    assert(output_shape == C_shape && "`enzymexla.lapack.gemqrt` requires `C` "
                                      "and `output` to have the same shape");

    assert(V_eltype == C_eltype && V_eltype == T_eltype &&
           "`enzymexla.lapack.gemqrt` requires the same element type for all "
           "operands");

    if (V_rank - 2 > 0 || T_rank - 2 > 0 || C_rank - 2 > 0) {
      return rewriter.notifyMatchFailure(
          op, "`enzymexla.lapack.orgqr` with batch dimensions on CPU is not "
              "yet supported");
    }

    auto nb_value = T_shape[0];
    auto k_value = T_shape[1];
    assert(k_value >= nb_value &&
           "Block size must be less than or equal to min(m, n)");
    assert(nb_value >= 1 && "Block size must be greater than or equal to 1");
    assert(V_shape[1] == k_value && "invalid number of reflectors (k) on T");

    auto ldv_value = V_shape[0];
    auto ldt_value = T_shape[0];
    auto ldc_value = C_shape[0];

    assert(ldt_value >= nb_value && "ldt must be >= nb");
    if (side_value == 'L') {
      assert(ldv_value == C_shape[0] &&
             "on left-sided muliplication, the first dimension "
             "of V must equal the first dimension of C");
      assert(C_shape[0] >= k_value &&
             "invalid number of reflectors: k should be <= m");
    } else { // side_value == 'R'
      assert(ldv_value == C_shape[1] &&
             "on right-sided multiplication, the first dimension"
             "of V must equal the second dimension of C");
      assert(C_shape[1] >= k_value &&
             "invalid number of reflectors: k should be <= n");
    }

    auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
    auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
    auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
    auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);
    auto type_llvm_char = rewriter.getIntegerType(8);

    std::string fn = "gemqrt_";
    if (auto prefix = lapackPrecisionPrefix(C_eltype)) {
      fn = *prefix + fn;
    } else {
      op->emitOpError() << "Unsupported element type: " << C_eltype;
      return rewriter.notifyMatchFailure(op, "unsupported element type");
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
                                          type_llvm_char,       // side
                                          type_llvm_char,       // trans
                                          type_llvm_lapack_int, // m
                                          type_llvm_lapack_int, // n
                                          type_llvm_lapack_int, // k
                                          type_llvm_lapack_int, // nb
                                          type_llvm_ptr,        // V
                                          type_llvm_lapack_int, // ldv
                                          type_llvm_ptr,        // T
                                          type_llvm_lapack_int, // ldt
                                          type_llvm_ptr,        // C
                                          type_llvm_lapack_int, // ldc
                                      },
                                      false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
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
                                                       type_llvm_ptr, // C
                                                   },
                                                   false);

      auto func = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));

      auto side = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, side_value));

      auto trans = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, trans_value));

      auto m = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, C_shape[0]));

      auto n = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, C_shape[1]));

      auto k = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));

      auto nb = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nb_value));

      auto ldv = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldv_value));

      auto ldt = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldt_value));

      auto ldc = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldc_value));

      // call to `lapacke_*(or|un)mqr*`
      auto res = LLVM::CallOp::create(rewriter, op.getLoc(),
                                      TypeRange{type_llvm_lapack_int},
                                      SymbolRefAttr::get(ctx, bind_fn),
                                      ValueRange{
                                          layout.getResult(),
                                          side.getResult(),
                                          trans.getResult(),
                                          m.getResult(),
                                          n.getResult(),
                                          k.getResult(),
                                          nb.getResult(),
                                          func.getArgument(0), // V
                                          ldv.getResult(),
                                          func.getArgument(1), // T
                                          ldt.getResult(),
                                          func.getArgument(2), // C
                                          ldc.getResult(),
                                      });

      LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `(or|un)mqr` wrapper
    SmallVector<bool> isColMajorArr = {true, true, true};
    SmallVector<int64_t> operandRanks = {2, 2, 2};
    SmallVector<int64_t> outputRanks = {2};
    auto operandLayouts =
        getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
    auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

    SmallVector<Attribute> aliases;
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, 2, {}));

    auto jit_call_op = enzymexla::JITCallOp::create(
        rewriter, op.getLoc(), TypeRange{C_type},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn), ValueRange{V, T, C},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
        /*xla_side_effect_free=*/rewriter.getUnitAttr());

    // replace enzymexla.lapack.geqrf with the jit_call
    rewriter.replaceAllUsesWith(op.getResult(), jit_call_op.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

Value anyNonFiniteValue(PatternRewriter &rewriter, Location loc, Type outType,
                        Value input, int64_t inputRank) {
  auto areFinite = stablehlo::AndOp::create(
      rewriter, loc,
      stablehlo::IsFiniteOp::create(
          rewriter, loc, stablehlo::RealOp::create(rewriter, loc, input)),
      stablehlo::IsFiniteOp::create(
          rewriter, loc, stablehlo::ImagOp::create(rewriter, loc, input)));

  SmallVector<int64_t> reductionDims;
  for (int i = inputRank - 2; i < inputRank; i++)
    reductionDims.push_back(i);

  auto initValType = RankedTensorType::get({}, rewriter.getI1Type());
  auto initVal = stablehlo::ConstantOp::create(
      rewriter, loc, initValType, cast<ElementsAttr>(makeAttr(initValType, 1)));

  auto allFinite = stablehlo::ReduceOp::create(
      rewriter, loc, ValueRange{areFinite.getResult()}, ValueRange{initVal},
      rewriter.getDenseI64ArrayAttr(reductionDims));

  {
    OpBuilder::InsertionGuard guard(rewriter);
    auto &region = allFinite.getBody();
    auto *block = rewriter.createBlock(&region, {}, {initValType, initValType},
                                       {loc, loc});

    rewriter.setInsertionPointToStart(block);
    stablehlo::ReturnOp::create(
        rewriter, loc,
        ValueRange{stablehlo::AndOp::create(rewriter, loc,
                                            block->getArgument(0),
                                            block->getArgument(1))
                       .getResult()});
  }

  // 0 is success, 1 is failure
  return stablehlo::ConvertOp::create(
      rewriter, loc, outType,
      stablehlo::NotOp::create(rewriter, loc, allFinite.getResult(0)));
}

struct GetrfOpLowering : public OpRewritePattern<enzymexla::GetrfOp> {
  std::string backend;
  int64_t blasIntWidth;

  GetrfOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GetrfOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return matchAndRewriteCPU(op, rewriter);
    else if (backend == "cuda")
      return matchAndRewriteCUDA(op, rewriter);
    else if (backend == "tpu")
      return matchAndRewriteTPU(op, rewriter);

    op->emitOpError() << "Unsupported backend: " << backend;
    return failure();
  }

private:
  func::FuncOp createWrapperFuncOpCPULapack(
      PatternRewriter &rewriter, const std::string &lapackFn,
      RankedTensorType inputType, RankedTensorType blasPivotType,
      RankedTensorType blasInfoType, Type blasIntType,
      const std::string &fnName, enzymexla::GetrfOp op,
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

  LogicalResult matchAndRewriteCPU(enzymexla::GetrfOp op,
                                   PatternRewriter &rewriter) const {
    auto ctx = op->getContext();

    auto input = op.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());

    auto inputElementType = inputType.getElementType();
    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();
    auto infoRank = infoType.getRank();

    auto numBatchDims = inputRank - 2;

    auto unbatchedInputType = RankedTensorType::get(
        SmallVector<int64_t>(inputType.getShape().end() - 2,
                             inputType.getShape().end()),
        inputType.getElementType());
    auto unbatchedPivotType = RankedTensorType::get(
        SmallVector<int64_t>(pivotType.getShape().end() - 1,
                             pivotType.getShape().end()),
        pivotType.getElementType());
    auto unbatchedInfoType =
        RankedTensorType::get({}, infoType.getElementType());

    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto blasIntType = rewriter.getIntegerType(blasIntWidth);
    auto intType = RankedTensorType::get({}, blasIntType);
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);

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

    // declare LAPACK function declarations if not present
    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(lapackFn)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType =
          LLVM::LLVMFunctionType::get(llvmVoidType,
                                      {llvmPtrType, llvmPtrType, llvmPtrType,
                                       llvmPtrType, llvmPtrType, llvmPtrType},
                                      false);
      LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), lapackFn, funcType,
                               LLVM::Linkage::External);
    }

    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(lapackFnWrapper)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType,
                                                  {
                                                      llvmPtrType,
                                                      llvmPtrType,
                                                      llvmPtrType,
                                                      llvmPtrType,
                                                      llvmPtrType,
                                                      llvmPtrType,
                                                  },
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

    // Call tge LLVM function with enzymexla.jit_call
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
      // https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/getrf-batch-strided.html.
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

  LogicalResult matchAndRewriteCUDA(enzymexla::GetrfOp op,
                                    PatternRewriter &rewriter) const {
    auto input = op.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());

    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();
    auto pivotRank = pivotType.getRank();
    auto infoRank = infoType.getRank();

    auto pivotCuSolverType =
        RankedTensorType::get(pivotType.getShape(), rewriter.getI32Type());
    auto infoCuSolverType =
        RankedTensorType::get(infoType.getShape(), rewriter.getI32Type());

    auto cusolverGetrfFFI = stablehlo::CustomCallOp::create(
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
        getSHLOLayout(rewriter, {inputRank}, SmallVector<bool>{true},
                      inputRank),
        /*result_layouts*/
        getSHLOLayout(rewriter, {inputRank, pivotRank, infoRank},
                      SmallVector<bool>(3, true), inputRank),
        /*output_operand_aliases*/
        rewriter.getArrayAttr({stablehlo::OutputOperandAliasAttr::get(
            op.getContext(), std::vector<int64_t>{0}, 0,
            std::vector<int64_t>{})}));

    // unused custom call not getting optimized away. so adding a manual
    // check
    if (!op.getResult(2).getUses().empty()) {
      auto pivotOnes = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), pivotCuSolverType,
          cast<ElementsAttr>(makeAttr(pivotCuSolverType, 1)));

      auto pivots0Indexed = stablehlo::SubtractOp::create(
          rewriter, op.getLoc(), cusolverGetrfFFI.getResult(1), pivotOnes);

      auto permutation = stablehlo::CustomCallOp::create(
          rewriter, op.getLoc(), TypeRange{pivotCuSolverType},
          ValueRange{pivots0Indexed.getResult()},
          rewriter.getStringAttr("cu_lu_pivots_to_permutation"),
          /*has_side_effect*/ nullptr,
          /*backend_config*/ nullptr,
          /*api_version*/
          stablehlo::CustomCallApiVersionAttr::get(
              rewriter.getContext(),
              mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
          /*calledcomputations*/ nullptr,
          /*operand_layouts*/
          getSHLOLayout(rewriter, {pivotRank}, SmallVector<bool>{true},
                        inputRank),
          /*result_layouts*/
          getSHLOLayout(rewriter, {pivotRank}, SmallVector<bool>{true},
                        inputRank),
          /*output_operand_aliases*/ nullptr);
      auto permutation1Indexed = stablehlo::AddOp::create(
          rewriter, op.getLoc(), permutation.getResult(0), pivotOnes);
      rewriter.replaceAllUsesWith(op.getResult(2), permutation1Indexed);
    }

    rewriter.replaceAllUsesWith(op.getResult(0), cusolverGetrfFFI.getResult(0));
    rewriter.replaceAllUsesWith(
        op.getResult(1),
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), pivotType,
                                     cusolverGetrfFFI.getResult(1)));
    rewriter.replaceAllUsesWith(
        op.getResult(3),
        stablehlo::ConvertOp::create(rewriter, op.getLoc(), infoType,
                                     cusolverGetrfFFI.getResult(2)));
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewriteTPU(enzymexla::GetrfOp op,
                                   PatternRewriter &rewriter) const {
    auto input = op.getInput();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto pivotType = cast<RankedTensorType>(op.getResult(1).getType());
    auto infoType = cast<RankedTensorType>(op.getResult(3).getType());

    auto inputShape = inputType.getShape();
    auto inputRank = inputType.getRank();

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

    // LAPACK returns 1-indexed pivots, while XLA returns 0-indexed pivots.
    // We make it consistent with LAPACK by adding 1 to the pivots.
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

    auto info = anyNonFiniteValue(rewriter, op.getLoc(), infoType,
                                  customCall.getResult(0), inputRank);

    rewriter.replaceAllUsesWith(op.getResult(0), customCall.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), pivots1Indexed);
    rewriter.replaceAllUsesWith(op.getResult(2), permutation1Indexed);
    rewriter.replaceAllUsesWith(op.getResult(3), info);
    rewriter.eraseOp(op);

    return success();
  }
};

struct GetriOpLowering : public OpRewritePattern<enzymexla::GetriOp> {
  std::string backend;
  int64_t blasIntWidth;

  GetriOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GetriOp op,
                                PatternRewriter &rewriter) const override {
    op->emitOpError() << "Unsupported backend: " << backend;
    return failure();
  }
};

template <typename OpTy>
func::FuncOp createSVDAlgorithmWrapperFuncOpCPULapack(
    PatternRewriter &rewriter, const std::string &lapackFn,
    RankedTensorType inputType, RankedTensorType UType, RankedTensorType SType,
    RankedTensorType VType, Type infoType, Type blasIntType,
    const std::string &fnName, OpTy op, ArrayAttr operandLayouts,
    ArrayAttr resultLayouts, ArrayAttr outputOperandAliases) {
  auto ctx = op->getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  auto moduleOp = op->template getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;

  rewriter.setInsertionPointToStart(moduleOp.getBody());

  SmallVector<Type> argTypes = {inputType};
  SmallVector<Type> retTypes = {UType, SType, VType, infoType};

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

  auto U = stablehlo::ConstantOp::create(
      rewriter, op.getLoc(), UType, cast<ElementsAttr>(makeAttr(UType, 0)));
  auto S = stablehlo::ConstantOp::create(
      rewriter, op.getLoc(), SType, cast<ElementsAttr>(makeAttr(SType, 0)));
  auto VT = stablehlo::ConstantOp::create(
      rewriter, op.getLoc(), VType, cast<ElementsAttr>(makeAttr(VType, 0)));
  auto info =
      stablehlo::ConstantOp::create(rewriter, op.getLoc(), infoType,
                                    cast<ElementsAttr>(makeAttr(infoType, 0)));

  auto ldu = stablehlo::ConvertOp::create(
      rewriter, op.getLoc(), RankedTensorType::get({}, blasIntType),
      stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), U, 0));
  auto ldvt = stablehlo::ConvertOp::create(
      rewriter, op.getLoc(), RankedTensorType::get({}, blasIntType),
      stablehlo::GetDimensionSizeOp::create(rewriter, op.getLoc(), VT, 0));

  auto jitCall = enzymexla::JITCallOp::create(
      rewriter, op.getLoc(), TypeRange{UType, SType, VType, infoType},
      mlir::FlatSymbolRefAttr::get(ctx, lapackFn),
      ValueRange{mSize, nSize, input, mSize, S, U, ldu, VT, ldvt, info},
      rewriter.getStringAttr(""),
      /*operand_layouts=*/operandLayouts,
      /*result_layouts=*/resultLayouts,
      /*arg_attrs=*/nullptr,
      /*res_attrs=*/nullptr,
      /*output_operand_aliases=*/outputOperandAliases,
      /*xla_side_effect_free=*/rewriter.getUnitAttr());

  func::ReturnOp::create(rewriter, op.getLoc(),
                         ValueRange{jitCall.getResult(0), jitCall.getResult(1),
                                    jitCall.getResult(2),
                                    jitCall.getResult(3)});

  return func;
}

RankedTensorType getVType(RankedTensorType VtType) {
  auto inputShape = VtType.getShape();
  auto inputRank = VtType.getRank();

  SmallVector<int64_t> VShape(inputShape.begin(), inputShape.end() - 2);
  VShape.push_back(inputShape[inputRank - 1]);
  VShape.push_back(inputShape[inputRank - 2]);

  return RankedTensorType::get(VShape, VtType.getElementType());
}

template <typename OpTy>
LogicalResult lowerSVDAlgorithmCPU(OpTy op, PatternRewriter &rewriter,
                                   int64_t blasIntWidth) {
  enzymexla::SVDAlgorithm algorithm;
  std::string fn;
  if constexpr (std::is_same_v<OpTy, enzymexla::GesvdOp>) {
    algorithm = enzymexla::SVDAlgorithm::QRIteration;
    fn = "gesvd_";
  } else if constexpr (std::is_same_v<OpTy, enzymexla::GesddOp>) {
    algorithm = enzymexla::SVDAlgorithm::DivideAndConquer;
    fn = "gesdd_";
  } else {
    op->emitOpError() << "Unsupported algorithm";
    return failure();
  }

  auto ctx = op->getContext();
  LLVMTypeConverter typeConverter(ctx);

  auto input = op.getOperand();
  auto inputType = cast<RankedTensorType>(input.getType());
  auto inputRank = inputType.getRank();
  auto inputElementType = inputType.getElementType();

  const int64_t numBatchDims = inputRank - 2;
  const bool isfull = op.getFull();

  auto type_lapack_int = rewriter.getIntegerType(blasIntWidth);
  auto type_lapack_int64 = rewriter.getIntegerType(64);
  auto type_lapack_char = rewriter.getIntegerType(sizeof(char) * 8);
  auto type_llvm_lapack_int = typeConverter.convertType(type_lapack_int);
  auto type_llvm_input = typeConverter.convertType(inputElementType);
  auto type_llvm_int64 = typeConverter.convertType(type_lapack_int64);
  auto type_llvm_ptr = LLVM::LLVMPointerType::get(ctx);
  auto type_llvm_void = LLVM::LLVMVoidType::get(ctx);
  auto type_input_element_real = inputElementType;
  bool isComplex = false;
  if (auto complex_type = dyn_cast<ComplexType>(type_input_element_real)) {
    isComplex = true;
    type_input_element_real = complex_type.getElementType();
  }

  if (auto prefix = lapackPrecisionPrefix(inputElementType)) {
    fn = *prefix + fn;
  } else {
    op->emitOpError() << "Unsupported element type: " << inputElementType;
    return rewriter.notifyMatchFailure(op, "unsupported input element type");
  }

  std::string bind_fn = "enzymexla_lapack_" + fn;
  std::string wrapper_fn = "enzymexla_wrapper_lapack_" + fn;

  if (isfull) {
    wrapper_fn += "_full";
  }

  // declare LAPACK function declarations if not present
  auto moduleOp = op->template getParentOfType<ModuleOp>();

  if (!moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(bind_fn)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    SmallVector<Type> argTypes(14 + isComplex, type_llvm_ptr);
    if (algorithm == enzymexla::SVDAlgorithm::QRIteration) {
      argTypes.push_back(type_llvm_int64);
      argTypes.push_back(type_llvm_int64);
    } else if (algorithm == enzymexla::SVDAlgorithm::DivideAndConquer) {
      argTypes.push_back(type_llvm_int64);
    }

    auto func_type =
        LLVM::LLVMFunctionType::get(type_llvm_void, argTypes, false);
    LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), bind_fn, func_type,
                             LLVM::Linkage::External);
  }

  if (!moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(wrapper_fn)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func_type = LLVM::LLVMFunctionType::get(type_llvm_void,
                                                 {
                                                     type_llvm_ptr, // m
                                                     type_llvm_ptr, // n
                                                     type_llvm_ptr, // a
                                                     type_llvm_ptr, // lda
                                                     type_llvm_ptr, // s
                                                     type_llvm_ptr, // u
                                                     type_llvm_ptr, // ldu
                                                     type_llvm_ptr, // vt
                                                     type_llvm_ptr, // ldvt
                                                     type_llvm_ptr, // info
                                                 },
                                                 false);
    auto funcOp = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), wrapper_fn,
                                           func_type, LLVM::Linkage::Private);
    rewriter.setInsertionPointToStart(funcOp.addEntryBlock(rewriter));

    auto const1 =
        LLVM::ConstantOp::create(rewriter, op.getLoc(), type_llvm_lapack_int,
                                 rewriter.getIntegerAttr(type_lapack_int, 1));
    auto i64_1 =
        LLVM::ConstantOp::create(rewriter, op.getLoc(), type_lapack_int64,
                                 rewriter.getIntegerAttr(type_lapack_int64, 1));
    auto constM1 =
        LLVM::ConstantOp::create(rewriter, op.getLoc(), type_llvm_lapack_int,
                                 rewriter.getIntegerAttr(type_lapack_int, -1));

    SmallVector<Value> args;

    auto lworkptr = LLVM::AllocaOp::create(rewriter, op.getLoc(), type_llvm_ptr,
                                           type_lapack_int, const1);
    LLVM::StoreOp::create(rewriter, op.getLoc(), constM1, lworkptr);

    // first call extracts the optimal size for the workspace
    auto workBuffer1 = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), type_llvm_ptr, type_input_element_real, const1);

    if (algorithm == enzymexla::SVDAlgorithm::QRIteration) {
      auto jobuptr = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), type_llvm_ptr, type_lapack_char, const1);
      auto jobvtptr = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), type_llvm_ptr, type_lapack_char, const1);

      auto jobu = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, isfull ? 'A' : 'S'));
      auto jobvt = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, isfull ? 'A' : 'S'));

      LLVM::StoreOp::create(rewriter, op.getLoc(), jobu, jobuptr);
      LLVM::StoreOp::create(rewriter, op.getLoc(), jobvt, jobvtptr);

      args = {
          jobuptr,
          jobvtptr,
          funcOp.getArgument(0),
          funcOp.getArgument(1),
          funcOp.getArgument(2),
          funcOp.getArgument(3),
          funcOp.getArgument(4),
          funcOp.getArgument(5),
          funcOp.getArgument(6),
          funcOp.getArgument(7),
          funcOp.getArgument(8),
          workBuffer1,
          lworkptr,
          funcOp.getArgument(9),
          i64_1,
          i64_1,
      };

      if (isComplex) {
        auto MVal = LLVM::LoadOp::create(
            rewriter, op.getLoc(), type_llvm_lapack_int, funcOp.getArgument(0));
        auto NVal = LLVM::LoadOp::create(
            rewriter, op.getLoc(), type_llvm_lapack_int, funcOp.getArgument(1));

        auto const5minMN = LLVM::MulOp::create(
            rewriter, op.getLoc(),
            LLVM::ConstantOp::create(
                rewriter, op.getLoc(), type_llvm_lapack_int,
                rewriter.getIntegerAttr(type_llvm_lapack_int, 5)),
            arith::MinSIOp::create(rewriter, op.getLoc(), MVal, NVal));

        auto rworkptr =
            LLVM::AllocaOp::create(rewriter, op.getLoc(), type_llvm_ptr,
                                   type_input_element_real, const5minMN);

        args.insert(args.begin() + 13, rworkptr);
      }
    } else if (algorithm == enzymexla::SVDAlgorithm::DivideAndConquer) {
      auto jobptr = LLVM::AllocaOp::create(rewriter, op.getLoc(), type_llvm_ptr,
                                           type_lapack_char, const1);

      auto job = LLVM::ConstantOp::create(
          rewriter, op.getLoc(), type_lapack_char,
          rewriter.getIntegerAttr(type_lapack_char, isfull ? 'A' : 'S'));

      auto MVal = LLVM::LoadOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int, funcOp.getArgument(0));
      auto NVal = LLVM::LoadOp::create(
          rewriter, op.getLoc(), type_llvm_lapack_int, funcOp.getArgument(1));
      auto minMN = arith::MinSIOp::create(rewriter, op.getLoc(), MVal, NVal);

      auto const8minMN = LLVM::MulOp::create(
          rewriter, op.getLoc(),
          LLVM::ConstantOp::create(
              rewriter, op.getLoc(), type_llvm_lapack_int,
              rewriter.getIntegerAttr(type_llvm_lapack_int, 8)),
          minMN);

      auto iworkptr = LLVM::AllocaOp::create(
          rewriter, op.getLoc(), type_llvm_ptr, type_lapack_int, const8minMN);

      LLVM::StoreOp::create(rewriter, op.getLoc(), job, jobptr);

      args = {
          jobptr,
          funcOp.getArgument(0),
          funcOp.getArgument(1),
          funcOp.getArgument(2),
          funcOp.getArgument(3),
          funcOp.getArgument(4),
          funcOp.getArgument(5),
          funcOp.getArgument(6),
          funcOp.getArgument(7),
          funcOp.getArgument(8),
          workBuffer1,
          lworkptr,
          iworkptr,
          funcOp.getArgument(9),
          i64_1,
      };

      if (isComplex) {
        // minmn*max(5*minmn+7, 2*max(m,n)+2*minm
        auto maxMN = arith::MaxSIOp::create(rewriter, op.getLoc(), MVal, NVal);

        // 5 * minmn
        auto c5 = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), type_llvm_lapack_int,
            rewriter.getIntegerAttr(type_llvm_lapack_int, 5));
        auto fiveMin = arith::MulIOp::create(rewriter, op.getLoc(), c5, minMN);

        // 5*minmn + 7
        auto c7 = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), type_llvm_lapack_int,
            rewriter.getIntegerAttr(type_llvm_lapack_int, 7));
        auto termA = arith::AddIOp::create(rewriter, op.getLoc(), fiveMin, c7);

        // 2 * max(m,n)
        auto c2 = LLVM::ConstantOp::create(
            rewriter, op.getLoc(), type_llvm_lapack_int,
            rewriter.getIntegerAttr(type_llvm_lapack_int, 2));
        auto twoMax = arith::MulIOp::create(rewriter, op.getLoc(), c2, maxMN);

        // 2*minmn
        auto twoMin = arith::MulIOp::create(rewriter, op.getLoc(), c2, minMN);

        // 2*max(m,n) + 2*minmn
        auto termB =
            arith::AddIOp::create(rewriter, op.getLoc(), twoMax, twoMin);

        // max(termA, termB)
        auto maxTerm =
            arith::MaxSIOp::create(rewriter, op.getLoc(), termA, termB);

        auto rworkSize =
            arith::MulIOp::create(rewriter, op.getLoc(), minMN, maxTerm);

        auto rworkptr =
            LLVM::AllocaOp::create(rewriter, op.getLoc(), type_llvm_ptr,
                                   type_input_element_real, rworkSize);

        args.insert(args.begin() + 13, rworkptr);
      }
    }

    LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                         SymbolRefAttr::get(ctx, bind_fn), ValueRange(args));

    // load and allocate the optimal size for the workspace
    auto workSpaceSizeFloat = LLVM::LoadOp::create(
        rewriter, op.getLoc(), type_input_element_real, workBuffer1);
    auto workSpaceSize = LLVM::FPToSIOp::create(
        rewriter, op.getLoc(), type_llvm_lapack_int, workSpaceSizeFloat);
    auto workspace =
        LLVM::AllocaOp::create(rewriter, op.getLoc(), type_llvm_ptr,
                               type_input_element_real, workSpaceSize);

    LLVM::StoreOp::create(rewriter, op.getLoc(), workSpaceSize, lworkptr);

    if (algorithm == enzymexla::SVDAlgorithm::QRIteration) {
      args[11] = workspace;
    } else if (algorithm == enzymexla::SVDAlgorithm::DivideAndConquer) {
      args[10] = workspace;
    }

    LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{},
                         SymbolRefAttr::get(ctx, bind_fn), ValueRange(args));

    LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
  }

  auto blasIntType = rewriter.getIntegerType(blasIntWidth);

  SmallVector<bool> isColMajorArr(10, true);
  SmallVector<int64_t> operandRanks = {0, 0, 2, 0, 1, 2, 0, 2, 0, 0};
  SmallVector<int64_t> outputRanks = {2, 1, 2, 0};
  auto operandLayouts = getSHLOLayout(rewriter, operandRanks, isColMajorArr, 2);
  auto resultLayouts = getSHLOLayout(rewriter, outputRanks, isColMajorArr, 2);

  SmallVector<Attribute> aliases;
  aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
      ctx, std::vector<int64_t>{0}, 5, std::vector<int64_t>{}));
  aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
      ctx, std::vector<int64_t>{1}, 4, std::vector<int64_t>{}));
  aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
      ctx, std::vector<int64_t>{2}, 7, std::vector<int64_t>{}));
  aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
      ctx, std::vector<int64_t>{3}, 9, std::vector<int64_t>{}));

  Value UResult, SResult, VTResult, infoResult;
  static int64_t fn_counter = 0;
  std::string shlo_wrapper_fn =
      "shlo_" + wrapper_fn + "_wrapper_" + std::to_string(fn_counter++);

  auto inputShape = inputType.getShape();

  auto unbatchedInputType = RankedTensorType::get(
      SmallVector<int64_t>(inputShape.end() - 2, inputShape.end()),
      inputType.getElementType());

  auto UResultType = cast<RankedTensorType>(op.getResult(0).getType());
  auto unbatchedUResultType = RankedTensorType::get(
      SmallVector<int64_t>(UResultType.getShape().end() - 2,
                           UResultType.getShape().end()),
      UResultType.getElementType());

  auto SResultType = cast<RankedTensorType>(op.getResult(1).getType());
  auto unbatchedSResultType = RankedTensorType::get(
      SmallVector<int64_t>(SResultType.getShape().end() - 1,
                           SResultType.getShape().end()),
      SResultType.getElementType());

  auto VTResultType = cast<RankedTensorType>(op.getResult(2).getType());
  auto unbatchedVTResultType = RankedTensorType::get(
      SmallVector<int64_t>(VTResultType.getShape().end() - 2,
                           VTResultType.getShape().end()),
      VTResultType.getElementType());

  SmallVector<int64_t> batchShape(inputType.getShape().begin(),
                                  inputType.getShape().end() - 2);
  auto infoType =
      RankedTensorType::get(batchShape, rewriter.getIntegerType(blasIntWidth));
  auto unbatchedInfoType =
      RankedTensorType::get({}, rewriter.getIntegerType(blasIntWidth));

  func::FuncOp func = createSVDAlgorithmWrapperFuncOpCPULapack(
      rewriter, wrapper_fn, unbatchedInputType, unbatchedUResultType,
      unbatchedSResultType, unbatchedVTResultType, unbatchedInfoType,
      blasIntType, shlo_wrapper_fn, op, operandLayouts, resultLayouts,
      rewriter.getArrayAttr(aliases));
  if (!func)
    return rewriter.notifyMatchFailure(op, "failed to create wrapper function");

  SmallVector<enzyme::BatchOp> batchOps;
  SmallVector<FunctionOpInterface> batchFunctions;

  if (numBatchDims > 0) {
    SmallVector<int64_t> batchShape(inputShape.begin(),
                                    inputShape.begin() + numBatchDims);

    auto batchOp = enzyme::BatchOp::create(
        rewriter, op.getLoc(),
        TypeRange{UResultType, SResultType, VTResultType, infoType},
        mlir::FlatSymbolRefAttr::get(op.getContext(), wrapper_fn),
        ValueRange{input}, rewriter.getDenseI64ArrayAttr(batchShape));

    UResult = batchOp.getResult(0);
    SResult = batchOp.getResult(1);
    VTResult = batchOp.getResult(2);
    infoResult = batchOp.getResult(3);

    batchOps.push_back(batchOp);
    batchFunctions.push_back(cast<FunctionOpInterface>(func.getOperation()));
  } else {
    auto callOp =
        func::CallOp::create(rewriter, op.getLoc(), func, ValueRange{input});

    UResult = callOp.getResult(0);
    SResult = callOp.getResult(1);
    VTResult = callOp.getResult(2);
    infoResult = callOp.getResult(3);
  }

  // replace enzymexla.linalg.svd with enzymexla.jit_call
  auto infoFinal = stablehlo::ConvertOp::create(
      rewriter, op.getLoc(), op.getResult(3).getType(), infoResult);

  rewriter.replaceAllUsesWith(op.getResult(0), UResult);
  rewriter.replaceAllUsesWith(op.getResult(1), SResult);
  rewriter.replaceAllUsesWith(op.getResult(2), VTResult);
  rewriter.replaceAllUsesWith(op.getResult(3), infoFinal);
  rewriter.eraseOp(op);

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

template <typename OpTy>
LogicalResult lowerSVDAlgorithmCUDA(OpTy op, PatternRewriter &rewriter,
                                    DictionaryAttr &options,
                                    bool skipUVAllocation,
                                    std::string targetName) {
  auto input = op.getOperand();
  auto type_input = cast<RankedTensorType>(input.getType());
  auto rank_input = type_input.getRank();

  auto type_u = cast<RankedTensorType>(op.getResult(0).getType());
  auto rank_u = type_u.getRank();

  auto type_input_element_real = type_input.getElementType();
  if (auto complex_type = dyn_cast<ComplexType>(type_input_element_real)) {
    type_input_element_real = complex_type.getElementType();
  }
  auto type_s = cast<RankedTensorType>(op.getResult(1).getType());
  auto rank_s = type_s.getRank();

  auto type_vt = cast<RankedTensorType>(op.getResult(2).getType());
  auto rank_vt = type_vt.getRank();
  auto type_v = getVType(type_vt);

  auto type_info_out = cast<RankedTensorType>(op.getResult(3).getType());
  auto type_info = RankedTensorType::get(type_info_out.getShape(),
                                         rewriter.getIntegerType(32));
  auto rank_info = type_info.getRank();

  // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
  SmallVector<Attribute> aliases = {};
  SmallVector<int64_t> ranks_operands = {rank_input};
  SmallVector<bool> isColMajorArrOperands = {true};

  SmallVector<int64_t> ranks_results;
  SmallVector<Type> outTypes;
  if (skipUVAllocation) {
    auto empty_type_u = RankedTensorType::get({}, type_u.getElementType());
    auto empty_type_vt = RankedTensorType::get({}, type_vt.getElementType());

    outTypes = {type_input, type_s, empty_type_u, empty_type_vt, type_info};
    ranks_results = {rank_input, rank_s, 0, 0, rank_info};
  } else {
    outTypes = {type_input, type_s, type_u, type_v, type_info};
    ranks_results = {rank_input, rank_s, rank_u, rank_vt, rank_info};
  }
  SmallVector<bool> isColMajorArrOutputs = {true, true, true, true, true};

  auto cusolverCallOp = stablehlo::CustomCallOp::create(
      rewriter, op.getLoc(), outTypes, ValueRange{input},
      rewriter.getStringAttr(targetName),
      /*has_side_effect*/ nullptr,
      /*backend_config*/ options,
      /*api_version*/
      stablehlo::CustomCallApiVersionAttr::get(
          rewriter.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
      /*calledcomputations*/ nullptr,
      /*operand_layouts*/
      getSHLOLayout(rewriter, ranks_operands, isColMajorArrOperands,
                    rank_input),
      /*result_layouts*/
      getSHLOLayout(rewriter, ranks_results, isColMajorArrOutputs, rank_input),
      /*output_operand_aliases*/ rewriter.getArrayAttr(aliases));

  auto info = stablehlo::ConvertOp::create(rewriter, op.getLoc(),
                                           op.getResult(3).getType(),
                                           cusolverCallOp.getResult(4));

  // replace enzymexla.linalg.svd with stablehlo.custom_call
  rewriter.replaceAllUsesWith(op.getResult(1), cusolverCallOp.getResult(1));

  Value Ures, Vtres;
  if (skipUVAllocation) {
    // return empty tensors for U and Vt
    Ures = stablehlo::ConstantOp::create(
        rewriter, op.getLoc(), type_u, cast<ElementsAttr>(makeAttr(type_u, 0)));
    Vtres =
        stablehlo::ConstantOp::create(rewriter, op.getLoc(), type_vt,
                                      cast<ElementsAttr>(makeAttr(type_vt, 0)));
  } else {
    Ures = cusolverCallOp.getResult(2);

    // cuda_customcall returns `U` and `V`. We need to transpose `V` to match the
    // return convention of `enzymexla.linalg.svd`.
    SmallVector<int64_t> permutation(rank_input);
    std::iota(permutation.begin(), permutation.end() - 2, 0);
    permutation[rank_input - 1] = rank_input - 2;
    permutation[rank_input - 2] = rank_input - 1;
    Vtres = stablehlo::TransposeOp::create(
        rewriter, op.getLoc(), cusolverCallOp.getResult(3), permutation);
  }
  rewriter.replaceAllUsesWith(op.getResult(0), Ures);
  rewriter.replaceAllUsesWith(op.getResult(2), Vtres);

  rewriter.replaceAllUsesWith(op.getResult(3), info.getResult());
  rewriter.eraseOp(op);

  return success();
}

struct GesvdOpLowering : public OpRewritePattern<enzymexla::GesvdOp> {
  std::string backend;
  int64_t blasIntWidth;

  GesvdOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GesvdOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return matchAndRewriteCPU(op, rewriter);
    else if (backend == "cuda")
      return matchAndRewriteCUDA(op, rewriter);

    op->emitOpError() << "Unsupported backend: " << backend;
    return failure();
  }

  LogicalResult matchAndRewriteCPU(enzymexla::GesvdOp op,
                                   PatternRewriter &rewriter) const {
    return lowerSVDAlgorithmCPU(op, rewriter, blasIntWidth);
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::GesvdOp op,
                                    PatternRewriter &rewriter) const {
    auto backend_config = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("full_matrices", op.getFullAttr()),
        rewriter.getNamedAttr("compute_uv", op.getComputeUvAttr()),
        rewriter.getNamedAttr("transposed", rewriter.getBoolAttr(false)),
    });
    return lowerSVDAlgorithmCUDA(op, rewriter, backend_config,
                                 !op.getComputeUv(), "cusolver_gesvd_ffi");
  }
};

struct GesddOpLowering : public OpRewritePattern<enzymexla::GesddOp> {
  std::string backend;
  int64_t blasIntWidth;

  GesddOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GesddOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cpu")
      return matchAndRewriteCPU(op, rewriter);

    op->emitOpError() << "Unsupported backend: " << backend;
    return failure();
  }

  LogicalResult matchAndRewriteCPU(enzymexla::GesddOp op,
                                   PatternRewriter &rewriter) const {
    return lowerSVDAlgorithmCPU(op, rewriter, blasIntWidth);
  }
};

struct GesvjOpLowering : public OpRewritePattern<enzymexla::GesvjOp> {
  std::string backend;
  int64_t blasIntWidth;

  GesvjOpLowering(std::string backend, int64_t blasIntWidth,
                  MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        blasIntWidth(blasIntWidth) {}

  LogicalResult matchAndRewrite(enzymexla::GesvjOp op,
                                PatternRewriter &rewriter) const override {
    if (backend == "cuda")
      return matchAndRewriteCUDA(op, rewriter);
    else if (backend == "tpu")
      return matchAndRewriteTPU(op, rewriter);

    op->emitOpError() << "Unsupported backend: " << backend;
    return failure();
  }

  LogicalResult matchAndRewriteCUDA(enzymexla::GesvjOp op,
                                    PatternRewriter &rewriter) const {
    auto backend_config = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr("full_matrices", op.getFullAttr()),
        rewriter.getNamedAttr("compute_uv", op.getComputeUvAttr()),
    });
    return lowerSVDAlgorithmCUDA(
        op, rewriter, backend_config,
        false, // https://github.com/jax-ml/jax/blob/43d14afad3e6bc321309054c512c5ffe1d1bca86/jaxlib/gpu/solver_kernels_ffi.cc#L1117
        "cusolver_gesvdj_ffi");
  }

  LogicalResult matchAndRewriteTPU(enzymexla::GesvjOp op,
                                   PatternRewriter &rewriter) const {
    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto rank_input = type_input.getRank();

    // emit `stablehlo.custom_call` to `@SVD` kernel from XLA
    auto customCall = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(),
        TypeRange{op.getResult(0).getType(), op.getResult(1).getType(),
                  getVType(cast<RankedTensorType>(op.getResult(2).getType()))},
        ValueRange{input}, rewriter.getStringAttr("SVD"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(0), customCall.getResult(0));
    rewriter.replaceAllUsesWith(op.getResult(1), customCall.getResult(1));

    // @SVD returns `U` and `V`. We need to transpose `V` to match the
    // return convention of `enzymexla.linalg.svd`.
    SmallVector<int64_t> permutation(rank_input);
    std::iota(permutation.begin(), permutation.end() - 2, 0);
    permutation[rank_input - 1] = rank_input - 2;
    permutation[rank_input - 2] = rank_input - 1;
    auto Vt = stablehlo::TransposeOp::create(
        rewriter, op.getLoc(), customCall.getResult(2), permutation);
    rewriter.replaceAllUsesWith(op.getResult(2), Vt);

    // Netlib's LAPACK returns `info`, but TPU kernel doesn't
    auto info =
        anyNonFiniteValue(rewriter, op.getLoc(),
                          cast<RankedTensorType>(op.getResult(3).getType()),
                          customCall.getResult(0), type_input.getRank());
    rewriter.replaceAllUsesWith(op.getResult(3), info);

    return success();
  }
};

struct LowerEnzymeXLALapackPass
    : public enzyme::impl::LowerEnzymeXLALapackPassBase<
          LowerEnzymeXLALapackPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns
        .add<GeqrfOpLowering, GeqrtOpLowering, OrgqrOpLowering, OrmqrOpLowering,
             GemqrtOpLowering, GetrfOpLowering, GetriOpLowering,
             GesvdOpLowering, GesddOpLowering, GesvjOpLowering>(
            backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
