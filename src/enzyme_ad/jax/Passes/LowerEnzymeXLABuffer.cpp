#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/JITCallOrdering.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLABUFFERPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::stablehlo;
using namespace mlir::enzymexla;

static JITCallOp createIdentityJITCall(PatternRewriter &rewriter, Operation *op,
                                       ValueRange operands) {
  auto ctx = rewriter.getContext();
  auto N = operands.size();
  std::string wrapperName = "enzymexla_jitwrap_identity_" + std::to_string(N);

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto llvmPointerType = LLVM::LLVMPointerType::get(ctx);
    auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
    SmallVector<Type> funcOperandTypes(N, llvmPointerType);
    auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType, funcOperandTypes);
    auto funcOp =
        LLVM::LLVMFuncOp::create(rewriter, op->getLoc(), wrapperName, funcType);

    rewriter.setInsertionPointToStart(funcOp.addEntryBlock(rewriter));
    LLVM::ReturnOp::create(rewriter, op->getLoc(), ValueRange{});
  }

  SmallVector<Attribute> aliases;
  if (N == 1) {
    aliases.push_back(stablehlo::OutputOperandAliasAttr::get(ctx, {}, 0, {}));
  } else {
    for (unsigned i = 0; i < N; ++i) {
      auto attr = stablehlo::OutputOperandAliasAttr::get(ctx, {i}, i, {});
      aliases.push_back(attr);
    }
  }

  auto jitCall = JITCallOp::create(
      rewriter, op->getLoc(), operands.getTypes(),
      rewriter.getAttr<mlir::FlatSymbolRefAttr>(wrapperName), operands,
      rewriter.getStringAttr(""),
      /*operand_layouts=*/nullptr,
      /*arg_attrs=*/nullptr,
      /*res_attrs=*/nullptr,
      /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
      /*xla_side_effect_free=*/nullptr);
  jitCall->setAttr(kRequireJITOrderingAttrName, rewriter.getUnitAttr());
  return jitCall;
}

namespace {

struct LowerCreateBufferOp : public OpConversionPattern<CreateBufferOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // this op doesn't require a jit_call ordering
    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, TypeRange{op.getType()}, ValueRange{},
        rewriter.getStringAttr("CreateBuffer"),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings=*/nullptr);

    return success();
  }
};

struct LowerPinOp : public OpConversionPattern<PinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto customCallOp = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{op.getType()},
        ValueRange{adaptor.getTensor()}, rewriter.getStringAttr("Pin"),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings=*/nullptr);
    auto jitCallOp =
        createIdentityJITCall(rewriter, op, customCallOp.getResults());
    rewriter.replaceOp(op, jitCallOp.getResults());

    return success();
  }
};

struct LowerUnpinOp : public OpConversionPattern<UnpinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnpinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto jitCallOp = createIdentityJITCall(rewriter, op, adaptor.getBuffer());
    auto customCallOp = stablehlo::CustomCallOp::create(
        rewriter, op.getLoc(), TypeRange{op.getType()}, jitCallOp.getResults(),
        rewriter.getStringAttr("Unpin"),
        /*has_side_effect=*/nullptr,
        /*backend_config=*/nullptr,
        /*api_version=*/
        stablehlo::CustomCallApiVersionAttr::get(
            rewriter.getContext(),
            mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI),
        /*calledcomputations*/ nullptr,
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr,
        /*output_operand_aliases=*/nullptr,
        /*result_tilings=*/nullptr);
    rewriter.replaceOp(op, customCallOp.getResults());

    return success();
  }
};

struct LowerEnzymeXLABufferPass
    : public enzyme::impl::LowerEnzymeXLABufferPassBase<
          LowerEnzymeXLABufferPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<stablehlo::StablehloDialect, LLVM::LLVMDialect,
                           enzymexla::EnzymeXLADialect>();
    target.addIllegalOp<CreateBufferOp, PinOp, UnpinOp>();

    RewritePatternSet patterns(context);
    patterns.add<LowerCreateBufferOp, LowerPinOp, LowerUnpinOp>(context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
