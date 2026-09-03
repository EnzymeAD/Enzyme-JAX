#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
#include "src/enzyme_ad/jax/Passes/Comm/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LOWERCOMMTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

// from LowerJIT
extern "C" void *EnzymeJaXLookupSymbol(const char *name);

struct LowerCommMpiConstantOpToStablehlo
    : public OpConversionPattern<comm::MpiConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(comm::MpiConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *converter = getTypeConverter();

    auto restype = converter->convertType(op.getResult().getType());
    if (restype == nullptr)
      return failure();

    llvm::StringRef name;
    auto value_attr = op.getValue();
    if (auto attr = cast<comm::MpiCommAttr>(value_attr)) {
      name = comm::stringifyMpiCommEnum(attr.getValue());
    } else if (auto attr = cast<comm::MpiOpAttr>(value_attr)) {
      name = comm::stringifyMpiOpEnum(attr.getValue());
    } else {
      return rewriter.notifyMatchFailure(
          op, "MPI constant is not a valid attribute");
    }

    void *value_abi = EnzymeJaXLookupSymbol(name.data());
    if (value_abi == nullptr) {
      return rewriter.notifyMatchFailure(op, "MPI constant `" + name +
                                                 "` not found");
    }

    uint64_t value = reinterpret_cast<uint64_t>(value_abi);
    auto constant_attr = SplatElementsAttr::get(
        RankedTensorType::get({}, rewriter.getIntegerType(64)),
        ArrayRef(APInt(64, value)));

    rewriter.replaceOpWithNewOp<stablehlo::ConstantOp>(
        op, restype, cast<ElementsAttr>(constant_attr));

    return success();
  }
};

struct LowerCommMpiCommRankOpToStablehlo
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

struct LowerCommMpiCommSizeOpToStablehlo
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

struct LowerCommMpiCommSplitOpToStablehlo
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

struct LowerCommMpiBarrierOpToStablehlo
    : public OpConversionPattern<comm::MpiBarrierOp> {
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

struct LowerCommMpiSendOpToStablehlo
    : public OpConversionPattern<comm::MpiSendOp> {
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

struct LowerCommMpiIsendOpToStablehlo
    : public OpConversionPattern<comm::MpiIsendOp> {
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

struct LowerCommMpiRecvOpToStablehlo
    : public OpConversionPattern<comm::MpiRecvOp> {
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

struct LowerCommMpiIrecvOpToStablehlo
    : public OpConversionPattern<comm::MpiIrecvOp> {
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

struct LowerCommMpiWaitOpToStablehlo
    : public OpConversionPattern<comm::MpiWaitOp> {
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

struct LowerCommMpiWaitallOpToStablehlo
    : public OpConversionPattern<comm::MpiWaitallOp> {
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

struct LowerCommMpiAllreduceOpToStablehlo
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
            "op", rewriter.getStringAttr(
                      comm::stringifyMpiOpEnum(op.getReduceOp().getValue()))),
    });

    // TODO pass attributes: reduceOp
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, converted_res_types,
        ValueRange{adaptor.getSendbuf(), adaptor.getComm()},
        rewriter.getStringAttr("MpiAllreduce"),
        /*has_side_effect=*/rewriter.getBoolAttr(true),
        /*backend_config=*/backend_config,
        /*api_version=*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*called_computations=*/nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings*/ nullptr);

    return success();
  }
};

struct LowerCommMpiBcastOpToStablehlo
    : public OpConversionPattern<comm::MpiBcastOp> {
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

    StablehloTypeConverter converter(context);

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

    patterns.add<
        LowerCommMpiConstantOpToStablehlo, LowerCommMpiCommRankOpToStablehlo,
        LowerCommMpiCommSizeOpToStablehlo, LowerCommMpiCommSplitOpToStablehlo,
        LowerCommMpiBarrierOpToStablehlo, LowerCommMpiSendOpToStablehlo,
        LowerCommMpiIsendOpToStablehlo, LowerCommMpiRecvOpToStablehlo,
        LowerCommMpiIrecvOpToStablehlo, LowerCommMpiWaitOpToStablehlo,
        LowerCommMpiWaitallOpToStablehlo, LowerCommMpiAllreduceOpToStablehlo,
        LowerCommMpiBcastOpToStablehlo>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
