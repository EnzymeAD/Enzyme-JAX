#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h"
#include "src/enzyme_ad/jax/Passes/Comm/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::comm {
#define GEN_PASS_DEF_LOWERCOMMTOJITPASS
#include "src/enzyme_ad/jax/Passes/Comm/Passes.h.inc"
} // namespace mlir::comm

using namespace mlir;

extern "C" void *EnzymeJaXLookupSymbol(const char *name);

struct LowerCommMpiConstantOpToJIT
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

struct LowerCommToJITPass
    : public mlir::comm::impl::LowerCommToJITPassBase<LowerCommToJITPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addIllegalDialect<comm::CommDialect>();

    comm::StablehloTypeConverter converter;

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

    patterns.add<LowerCommMpiConstantOpToJIT>(converter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
