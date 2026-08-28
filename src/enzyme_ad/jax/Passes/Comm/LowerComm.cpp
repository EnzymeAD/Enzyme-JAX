#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LOWERCOMMTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

struct LowerCommMpiCommRankOp
    : public OpConversionPattern<comm::MpiCommRankOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    SmallVector<Type> converted_res_types;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types, ValueRange{adaptor.getComm()},
        rewriter.getStringAttr("MpiCommRank"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiCommSizeOp
    : public OpConversionPattern<comm::MpiCommSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    SmallVector<Type> converted_res_types;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types, ValueRange{adaptor.getComm()},
        rewriter.getStringAttr("MpiCommSize"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiCommSplitOp
    : public OpConversionPattern<comm::MpiCommSplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiCommSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    SmallVector<Type> converted_res_types;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getComm(), adaptor.getColor(), adaptor.getKey()},
        rewriter.getStringAttr("MpiCommSplit"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiBarrierOp : public OpConversionPattern<comm::MpiBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{adaptor.getComm()},
        rewriter.getStringAttr("MpiBarrier"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiSendOp : public OpConversionPattern<comm::MpiSendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{},
        ValueRange{adaptor.getBuffer(), adaptor.getDest(), adaptor.getTag(),
                   adaptor.getComm()},
        rewriter.getStringAttr("MpiSend"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiIsendOp : public OpConversionPattern<comm::MpiIsendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIsendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> converted_res_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getBuffer(), adaptor.getDest(), adaptor.getTag(),
                   adaptor.getComm()},
        rewriter.getStringAttr("MpiIsend"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiRecvOp : public OpConversionPattern<comm::MpiRecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> converted_res_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getSource(), adaptor.getTag(), adaptor.getComm()},
        rewriter.getStringAttr("MpiRecv"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiIrecvOp : public OpConversionPattern<comm::MpiIrecvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiIrecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> converted_res_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getSource(), adaptor.getTag(), adaptor.getComm()},
        rewriter.getStringAttr("MpiIrecv"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiWaitOp : public OpConversionPattern<comm::MpiWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{adaptor.getRequest()},
        rewriter.getStringAttr("MpiWait"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiWaitallOp : public OpConversionPattern<comm::MpiWaitallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{adaptor.getRequests()},
        rewriter.getStringAttr("MpiWaitall"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiAllreduceOp
    : public OpConversionPattern<comm::MpiAllreduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiAllreduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> converted_res_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_res_types))) {
      return failure();
    }

    auto backend_config = rewriter.getDictionaryAttr({
        rewriter.getNamedAttr(
            "op",
            rewriter.getStringAttr(comm::stringifyMpiOps(op.getReduceOp()))),
    });

    // TODO pass attributes: reduceOp
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getSendbuf(), adaptor.getComm()},
        rewriter.getStringAttr("MpiAllreduce"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/backend_config,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiBcastOp : public OpConversionPattern<comm::MpiBcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> converted_res_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_res_types))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getInBuffer(), adaptor.getRoot(), adaptor.getComm()},
        rewriter.getStringAttr("MpiBcast"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/nullptr,
        /*api_version=*/nullptr,
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommToStablehloPass
    : public mlir::comm::impl::LowerCommToStablehloPassBase<
          LowerCommToStablehloPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addIllegalDialect<comm::CommDialect>();

    // defaults to no conversion for other types
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // !comm.mpi.comm, !comm.mpi.request are pointer-like, so lower to
    // tensor<i64>
    auto ptr_tensor_type =
        RankedTensorType::get({}, IntegerType::get(context, 64));
    converter.addConversion(
        [&](comm::MpiCommType type) { return ptr_tensor_type; });
    converter.addConversion(
        [&](comm::MpiRequestType type) { return ptr_tensor_type; });

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return converter.isSignatureLegal(op.getCalleeType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });

    // lower comm.mpi ops to stablehlo.custom_call ops
    RewritePatternSet patterns(context);

    mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, converter);
    mlir::populateCallOpTypeConversionPattern(patterns, converter);
    mlir::populateReturnOpTypeConversionPattern(patterns, converter);

    patterns.add<LowerCommMpiCommRankOp, LowerCommMpiCommSizeOp,
                 LowerCommMpiCommSplitOp, LowerCommMpiBarrierOp,
                 LowerCommMpiSendOp, LowerCommMpiIsendOp, LowerCommMpiRecvOp,
                 LowerCommMpiIrecvOp, LowerCommMpiWaitOp, LowerCommMpiWaitallOp,
                 LowerCommMpiAllreduceOp, LowerCommMpiBcastOp>(converter,
                                                               context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
