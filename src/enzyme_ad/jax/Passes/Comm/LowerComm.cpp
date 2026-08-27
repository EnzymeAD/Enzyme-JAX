#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
// #include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LOWERCOMMTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

struct CommMpiCommRankOpLowering
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
        op, converted_res_types, ValueRange{op.getComm()},
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

struct CommMpiCommSizeOpLowering
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
        op, converted_res_types, ValueRange{op.getComm()},
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

struct CommMpiCommSplitOpLowering
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
        ValueRange{op.getComm(), op.getColor(), op.getKey()},
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

struct CommMpiBarrierOpLowering
    : public OpConversionPattern<comm::MpiBarrierOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{op.getComm()},
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

struct CommMpiSendOpLowering : public OpConversionPattern<comm::MpiSendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{},
        ValueRange{op.getBuffer(), op.getDest(), op.getTag(), op.getComm()},
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

struct CommMpiIsendOpLowering : public OpConversionPattern<comm::MpiIsendOp> {
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
        ValueRange{op.getBuffer(), op.getDest(), op.getTag(), op.getComm()},
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

struct CommMpiRecvOpLowering : public OpConversionPattern<comm::MpiRecvOp> {
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
        ValueRange{op.getBuffer(), op.getSource(), op.getTag(), op.getComm()},
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

struct CommMpiIrecvOpLowering : public OpConversionPattern<comm::MpiIrecvOp> {
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
        ValueRange{op.getBuffer(), op.getSource(), op.getTag(), op.getComm()},
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

struct CommMpiWaitOpLowering : public OpConversionPattern<comm::MpiWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{op.getRequest()},
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

struct CommMpiWaitallOpLowering
    : public OpConversionPattern<comm::MpiWaitallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiWaitallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{}, ValueRange{op.getRequests()},
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

struct CommMpiAllreduceOpLowering
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

    // TODO pass attributes: reduceOp
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types, ValueRange{op.getSendbuf(), op.getComm()},
        rewriter.getStringAttr("MpiAllreduce"),
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

struct CommMpiBcastOpLowering : public OpConversionPattern<comm::MpiBcastOp> {
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
        ValueRange{op.getInBuffer(), op.getRoot(), op.getComm()},
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

    // !comm.mpi.comm, !comm.mpi.request are pointer-like, so lower to
    // tensor<i64>
    auto ptr_tensor_type =
        RankedTensorType::get({}, IntegerType::get(context, 64));
    converter.addConversion(
        [&](comm::MpiCommType type) { return ptr_tensor_type; });
    converter.addConversion(
        [&](comm::MpiRequestType type) { return ptr_tensor_type; });

    // lower comm.mpi ops to stablehlo.custom_call ops
    RewritePatternSet patterns(context);
    patterns.add<CommMpiCommRankOpLowering, CommMpiCommSizeOpLowering,
                 CommMpiCommSplitOpLowering, CommMpiBarrierOpLowering,
                 CommMpiSendOpLowering, CommMpiIsendOpLowering,
                 CommMpiRecvOpLowering, CommMpiIrecvOpLowering,
                 CommMpiWaitOpLowering, CommMpiWaitallOpLowering,
                 CommMpiAllreduceOpLowering, CommMpiBcastOpLowering>(converter,
                                                                     context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
