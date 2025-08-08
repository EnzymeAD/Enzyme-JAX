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
  LogicalResult matchAndRewrite_cpu(enzymexla::GeqrfOp op,
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
    if (auto prefix = lapack_precision_prefix(inputElementType)) {
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
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, i, {}));
    }

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{inputType, type_tau, type_info},
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

  LogicalResult matchAndRewrite_cuda(enzymexla::GeqrfOp op,
                                     PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());
    auto rank_tau = type_tau.getRank();

    // emit `stablehlo.custom_call` to `@cusolver_geqrf_ffi` kernel from jaxlib
    SmallVector<Attribute> aliases = {
        stablehlo::OutputOperandAliasAttr::get(ctx, {0}, 0, {})};
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

  LogicalResult matchAndRewrite_tpu(enzymexla::GeqrfOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand();
    auto type_input = cast<RankedTensorType>(input.getType());

    // emit `stablehlo.custom_call` to `@Qr` kernel from XLA
    auto type_tau = cast<RankedTensorType>(op.getResult(1).getType());

    auto custom_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input, type_tau}, ValueRange{input},
        rewriter.getStringAttr("Qr"),
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
      return this->matchAndRewrite_cpu(op, rewriter);

    // else if (backend == "cuda")
    //   return this->matchAndRewrite_cuda(op, rewriter);

    // else if (backend == "tpu")
    //   return this->matchAndRewrite_tpu(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewrite_cpu(enzymexla::GeqrtOp op,
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
    if (auto prefix = lapack_precision_prefix(inputElementType)) {
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
      rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), bind_fn, func_type,
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
      auto nb_op = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int, nb_attr);
      // can reuse nb = ldt
      ldt_value = nb_value;
      auto ldt_op = nb_op;

      auto A = func.getArgument(0);
      auto T = func.getArgument(1);
      auto info = func.getArgument(2);

      // call to `lapacke_*geqrt*`
      auto res = rewriter.create<LLVM::CallOp>(op.getLoc(),
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

      rewriter.create<LLVM::StoreOp>(op.getLoc(), res.getResult(), info);
      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `geqrt` wrapper
    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto info = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, -1)));

    auto type_T = RankedTensorType::get(
        {ldt_value, std::min(inputShape[0], inputShape[1])}, inputElementType);
    auto T = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_T, cast<ElementsAttr>(makeAttr(type_T, 0)));

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

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{inputType, type_T, type_info},
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
  LogicalResult matchAndRewrite_cpu(enzymexla::OrgqrOp op,
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
    if (auto prefix = lapack_precision_prefix(inputElementType)) {
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
                                                   },
                                                   false);

      auto func =
          rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapper_fn, func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));
      auto mC = inputShape[0];
      auto m = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, mC));
      auto nC = inputShape[1];
      auto n = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nC));
      auto k_value = nC;
      auto k = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));
      auto lda = m;

      // call to `lapacke_*(or|un)gqr*`
      auto res = rewriter.create<LLVM::CallOp>(op.getLoc(),
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

      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
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

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{inputType},
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

  LogicalResult matchAndRewrite_cuda(enzymexla::OrgqrOp op,
                                     PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand(0);
    auto type_input = cast<RankedTensorType>(input.getType());
    auto shape_input = type_input.getShape();
    auto rank_input = static_cast<int64_t>(shape_input.size());

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

    auto cusolver_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input}, ValueRange{input, tau},
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

  LogicalResult matchAndRewrite_tpu(enzymexla::OrgqrOp op,
                                    PatternRewriter &rewriter) const {
    auto ctx = op->getContext();
    LLVMTypeConverter typeConverter(ctx);

    auto input = op.getOperand(0);
    auto type_input = cast<RankedTensorType>(input.getType());
    auto tau = op.getOperand(1);

    auto custom_call_op = rewriter.create<stablehlo::CustomCallOp>(
        op.getLoc(), TypeRange{type_input}, ValueRange{input, tau},
        rewriter.getStringAttr("ProductOfElementaryHouseholderReflectors"),
        /*has_side_effect*/ nullptr,
        /*backend_config*/ nullptr,
        /*api_version*/ nullptr,
        /*calledcomputations*/ nullptr,
        /*operand_layouts*/ nullptr,
        /*result_layouts*/ nullptr,
        /*output_operand_aliases*/ nullptr);

    rewriter.replaceAllUsesWith(op.getResult(), custom_call_op.getResult(0));

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
      return this->matchAndRewrite_cpu(op, rewriter);

    // else if (backend == "cuda")
    //   return this->matchAndRewrite_cuda(op, rewriter);

    // else if (backend == "tpu")
    //   return this->matchAndRewrite_tpu(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewrite_cpu(enzymexla::OrmqrOp op,
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
    if (auto prefix = lapack_precision_prefix(A_eltype)) {
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
                                                       type_llvm_ptr, // C
                                                   },
                                                   false);

      auto func =
          rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapper_fn, func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));

      auto side = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, side_value));

      auto trans = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, trans_value));

      auto m = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, mC));

      auto n = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nC));

      auto k = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));

      auto lda = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, lda_value));

      auto ldc = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldc_value));

      // call to `lapacke_*(or|un)mqr*`
      auto res = rewriter.create<LLVM::CallOp>(op.getLoc(),
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

      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
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

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{C_type},
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
      return this->matchAndRewrite_cpu(op, rewriter);

    // else if (backend == "cuda")
    //   return this->matchAndRewrite_cuda(op, rewriter);

    // else if (backend == "tpu")
    //   return this->matchAndRewrite_tpu(op, rewriter);

    else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO get matrix sizes dynamically so that we don't need to create a
  // function wrapper for each op instance
  LogicalResult matchAndRewrite_cpu(enzymexla::GemqrtOp op,
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
    if (auto prefix = lapack_precision_prefix(C_eltype)) {
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
                                                       type_llvm_ptr, // C
                                                   },
                                                   false);

      auto func =
          rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapper_fn, func_type);
      rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

      // `101` for row-major, `102` for col-major
      auto layout = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, 101));

      auto side = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, side_value));

      auto trans = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_char,
          rewriter.getIntegerAttr(type_llvm_char, trans_value));

      auto m = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, C_shape[0]));

      auto n = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, C_shape[1]));

      auto k = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, k_value));

      auto nb = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, nb_value));

      auto ldv = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldv_value));

      auto ldt = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldt_value));

      auto ldc = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), type_llvm_lapack_int,
          rewriter.getIntegerAttr(type_lapack_int, ldc_value));

      // call to `lapacke_*(or|un)mqr*`
      auto res = rewriter.create<LLVM::CallOp>(op.getLoc(),
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

      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
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

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{C_type},
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

struct LowerEnzymeXLALapackPass
    : public enzyme::impl::LowerEnzymeXLALapackPassBase<
          LowerEnzymeXLALapackPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<GeqrfOpLowering>(backend, blasIntWidth, context);
    patterns.add<GeqrtOpLowering>(backend, blasIntWidth, context);
    patterns.add<OrgqrOpLowering>(backend, blasIntWidth, context);
    patterns.add<OrmqrOpLowering>(backend, blasIntWidth, context);
    patterns.add<GemqrtOpLowering>(backend, blasIntWidth, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
