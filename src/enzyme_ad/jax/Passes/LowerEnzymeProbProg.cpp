#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "lower-enzyme-probprog"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEPROBPROGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

struct addSampleToTraceOpConversion
    : public OpConversionPattern<enzyme::AddSampleToTraceOp> {
  using OpConversionPattern::OpConversionPattern;

  std::string backend;
  addSampleToTraceOpConversion(std::string backend,
                               TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit), backend(backend) {
  }

  LogicalResult
  matchAndRewrite(enzyme::AddSampleToTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    auto sample = adaptor.getSample();

    auto traceAttr = op->getAttrOfType<IntegerAttr>("trace");
    auto symbolAttr = op->getAttrOfType<IntegerAttr>("symbol");

    if (!traceAttr || !symbolAttr) {
      return rewriter.notifyMatchFailure(op,
                                         "Missing trace or symbol attribute");
    }

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto llvmVoidType = LLVM::LLVMVoidType::get(ctx);
      auto llvmI64Type = IntegerType::get(ctx, 64);

      std::string addSampleToTraceFn = "enzyme_probprog_add_sample_to_trace";

      auto traceConstType = RankedTensorType::get({}, llvmI64Type);
      auto symbolConstType = RankedTensorType::get({}, llvmI64Type);

      auto traceConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), traceConstType,
          cast<ElementsAttr>(
              makeAttr(traceConstType, traceAttr.getValue().getZExtValue())));

      auto symbolConst = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), symbolConstType,
          cast<ElementsAttr>(
              makeAttr(symbolConstType, symbolAttr.getValue().getZExtValue())));

      // Generate the LLVM function body
      std::string fnName =
          addSampleToTraceFn + "_wrapper_" + std::to_string(fnNum);
      fnNum++;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto callResult = rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{},
            SymbolRefAttr::get(ctx, addSampleToTraceFn),
            ValueRange{
                func.getArgument(0),
                func.getArgument(1),
                func.getArgument(2),
            });

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(addSampleToTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), addSampleToTraceFn,
                                          funcType, LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Value> operands{traceConst, symbolConst};
      operands.append(sample.begin(), sample.end());
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{}, mlir::FlatSymbolRefAttr::get(ctx, fnName),
          operands, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      // Replace the addSampleToTraceOp with the result of the JIT call
      rewriter.replaceOp(op, jitCall.getResults());

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Unknown backend " + backend);
    }
  }
};

struct LowerEnzymeProbProgPass
    : public enzyme::impl::LowerEnzymeProbProgPassBase<
          LowerEnzymeProbProgPass> {
  using LowerEnzymeProbProgPassBase::LowerEnzymeProbProgPassBase;

  void runOnOperation() override {
    auto context = getOperation()->getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<enzymexla::EnzymeXLADialect>();
    target.addLegalDialect<stablehlo::StablehloDialect>();
    target.addIllegalOp<enzyme::AddSampleToTraceOp>();

    RewritePatternSet patterns(context);
    patterns.add<addSampleToTraceOpConversion>(backend, typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
