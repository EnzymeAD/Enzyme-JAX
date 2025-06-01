#include "Enzyme/MLIR/Dialect/Ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

struct InitTraceOpLowering : public OpRewritePattern<enzyme::initTraceOp> {

  std::string backend;
  InitTraceOpLowering(std::string backend, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzyme::initTraceOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();

    if (backend == "cpu") {
      auto moduleOp = op->getParentOfType<ModuleOp>();
      static int64_t fnNum = 0;

      auto llvmPtrType = LLVM::LLVMPointerType::get(ctx);
      auto loweredTraceType = RankedTensorType::get({1}, IntegerType::get(ctx, 64, IntegerType::Unsigned));

      std::string initTraceFn = "enzyme_probprog_init_trace";

      // Generate the LLVM function body
      std::string fnName = initTraceFn + "_wrapper_" + std::to_string(fnNum);
      fnNum++;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(llvmPtrType, {}, false);

        auto func =
            rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), fnName, funcType);
        rewriter.setInsertionPointToStart(func.addEntryBlock(rewriter));

        auto callResult = rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{llvmPtrType},
            SymbolRefAttr::get(ctx, initTraceFn), ValueRange{});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), callResult.getResults());
      }

      // Insert function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(initTraceFn)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(llvmPtrType, {}, false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), initTraceFn, funcType,
                                          LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), TypeRange{loweredTraceType},
          mlir::FlatSymbolRefAttr::get(ctx, fnName), ValueRange{},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/rewriter.getArrayAttr({}),
          /*result_layouts=*/rewriter.getArrayAttr({}),
          /*output_operand_aliases=*/rewriter.getArrayAttr({}),
          /*xla_side_effect_free=*/nullptr);

      // Replace the initTraceOp with the result of the JIT call
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
    RewritePatternSet patterns(context);

    patterns.add<InitTraceOpLowering>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
