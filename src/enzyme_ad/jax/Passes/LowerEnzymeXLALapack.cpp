#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
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
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
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

// TODO implement backends
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

  // TODO rewrite for geqrt
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

      IntegerAttr nb_attr;
      if (op.getBlocksize()) {
        auto nb_value = op.getBlocksize().value();
        assert(std::min(inputShape[0], inputShape[1]) >= nb_value &&
               "Block size must be less than or equal to min(m, n)");
        assert(nb_value >= 1 && "Block size must be greater than or equal to 1");
        nb_attr = rewriter.getI64IntegerAttr(nb_value);
      } else {
        // default block size is min(m, n)
        nb_attr = rewriter.getI64IntegerAttr(std::min(inputShape[0], inputShape[1]));
      }
      auto nb = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), type_llvm_lapack_int, nb_attr);
      auto ldt = nb;

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
                                                   nb.getResult(),
                                                   A,
                                                   lda.getResult(),
                                                   T,
                                                   ldt.getResult(),
                                               });

      rewriter.create<LLVM::StoreOp>(op.getLoc(), res.getResult(), info);
      rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
    }

    // emit the `enzymexla.jit_call` op to `geqrt` wrapper
    auto type_info = RankedTensorType::get({}, type_lapack_int);
    auto info = rewriter.create<stablehlo::ConstantOp>(
        op.getLoc(), type_info, cast<ElementsAttr>(makeAttr(type_info, -1)));

    auto tsize = std::min(inputShape.front(), inputShape.back());
    auto type_T = RankedTensorType::get({tsize}, inputElementType);
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
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          ctx, std::vector<int64_t>{i}, i, std::vector<int64_t>{}));
    }

    auto jit_call_op = rewriter.create<enzymexla::JITCallOp>(
        op.getLoc(), TypeRange{inputType, type_T, type_info},
        mlir::FlatSymbolRefAttr::get(ctx, wrapper_fn),
        ValueRange{input, T.getResult(), info.getResult()},
        rewriter.getStringAttr(""),
        /*operand_layouts=*/operandLayouts,
        /*result_layouts=*/resultLayouts,
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
    // if (backend == "cpu")
    //   return this->matchAndRewrite_cpu(op, rewriter);

    // else if (backend == "cuda")
    //   return this->matchAndRewrite_cuda(op, rewriter);

    // else if (backend == "tpu")
    //   return this->matchAndRewrite_tpu(op, rewriter);

    // else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
  }

  // TODO for TPU, use `@ProductOfElementaryHouseholderReflectors`
  // https://github.com/openxla/xla/blob/6b3ac21e936757fe8073073bef5ad4145d5e2c06/xla/hlo/builder/lib/qr.h#L44C7-L44C47
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
    // if (backend == "cpu")
    //   return this->matchAndRewrite_cpu(op, rewriter);

    // else if (backend == "cuda")
    //   return this->matchAndRewrite_cuda(op, rewriter);

    // else if (backend == "tpu")
    //   return this->matchAndRewrite_tpu(op, rewriter);

    // else
      return rewriter.notifyMatchFailure(op, "Unknown backend: \"" + backend +
                                                 "\"");
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

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
