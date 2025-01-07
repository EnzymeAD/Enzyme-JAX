//===- LibDeviceFuncsRaisingPass.cpp - Raise libdevice.bc func calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/PassDetails.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
template <typename TargetOp>
class CallToOpRaising : public OpRewritePattern<LLVM::CallOp> {
public:
  CallToOpRaising(MLIRContext *context, StringRef funcNameStr)
      : OpRewritePattern<LLVM::CallOp>(context),
        funcName(StringAttr::get(context, funcNameStr)) {}

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    CallInterfaceCallable callable = op.getCallableForCallee();
    auto callee = callable.dyn_cast<SymbolRefAttr>();
    if (!callee)
      return failure();

    if (callee.getLeafReference() != funcName)
      return failure();

    rewriter.replaceOpWithNewOp<TargetOp>(op, op->getResults().getTypes(),
                                          op->getOperands());
    return success();
  }

private:
  StringAttr funcName;
};
} // namespace

template <typename TargetOp>
static void populateOpPatterns(MLIRContext *context,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func, StringRef f32ApproxFunc = "",
                               StringRef f16Func = "") {
  patterns.add<CallToOpRaising<TargetOp>>(context, f32Func);
  patterns.add<CallToOpRaising<TargetOp>>(context, f64Func);
  if (!f32ApproxFunc.empty())
    patterns.add<CallToOpRaising<TargetOp>>(context, f32ApproxFunc);
  if (!f16Func.empty())
    patterns.add<CallToOpRaising<TargetOp>>(context, f16Func);
}

void mlir::enzyme::populateLibDeviceFuncsToOpsPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  // XXX: Keep in sync with
  // mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp.

  auto *converter = context;
  populateOpPatterns<arith::RemFOp>(converter, patterns, "__nv_fmodf",
                                    "__nv_fmod");
  populateOpPatterns<math::AbsFOp>(converter, patterns, "__nv_fabsf",
                                   "__nv_fabs");
  populateOpPatterns<math::AcosOp>(converter, patterns, "__nv_acosf",
                                   "__nv_acos");
  populateOpPatterns<math::AcoshOp>(converter, patterns, "__nv_acoshf",
                                    "__nv_acosh");
  populateOpPatterns<math::AsinOp>(converter, patterns, "__nv_asinf",
                                   "__nv_asin");
  populateOpPatterns<math::AsinhOp>(converter, patterns, "__nv_asinhf",
                                    "__nv_asinh");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__nv_atanf",
                                   "__nv_atan");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__nv_atan2f",
                                    "__nv_atan2");
  populateOpPatterns<math::AtanhOp>(converter, patterns, "__nv_atanhf",
                                    "__nv_atanh");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__nv_cbrtf",
                                   "__nv_cbrt");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__nv_ceilf",
                                   "__nv_ceil");
  populateOpPatterns<math::CopySignOp>(converter, patterns, "__nv_copysignf",
                                       "__nv_copysign");
  populateOpPatterns<math::CosOp>(converter, patterns, "__nv_cosf", "__nv_cos",
                                  "__nv_fast_cosf");
  populateOpPatterns<math::CoshOp>(converter, patterns, "__nv_coshf",
                                   "__nv_cosh");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__nv_erff", "__nv_erf");
  populateOpPatterns<math::ExpOp>(converter, patterns, "__nv_expf", "__nv_exp",
                                  "__nv_fast_expf");
  populateOpPatterns<math::Exp2Op>(converter, patterns, "__nv_exp2f",
                                   "__nv_exp2");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__nv_expm1f",
                                    "__nv_expm1");
  populateOpPatterns<math::FloorOp>(converter, patterns, "__nv_floorf",
                                    "__nv_floor");
  populateOpPatterns<math::FmaOp>(converter, patterns, "__nv_fmaf", "__nv_fma");
  populateOpPatterns<math::LogOp>(converter, patterns, "__nv_logf", "__nv_log",
                                  "__nv_fast_logf");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__nv_log10f",
                                    "__nv_log10", "__nv_fast_log10f");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__nv_log1pf",
                                    "__nv_log1p");
  populateOpPatterns<math::Log2Op>(converter, patterns, "__nv_log2f",
                                   "__nv_log2", "__nv_fast_log2f");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__nv_powf", "__nv_pow",
                                   "__nv_fast_powf");
  populateOpPatterns<math::RoundOp>(converter, patterns, "__nv_roundf",
                                    "__nv_round");
  populateOpPatterns<math::RoundEvenOp>(converter, patterns, "__nv_rintf",
                                        "__nv_rint");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__nv_rsqrtf",
                                    "__nv_rsqrt");
  populateOpPatterns<math::SinOp>(converter, patterns, "__nv_sinf", "__nv_sin",
                                  "__nv_fast_sinf");
  populateOpPatterns<math::SinhOp>(converter, patterns, "__nv_sinhf",
                                   "__nv_sinh");
  populateOpPatterns<math::SqrtOp>(converter, patterns, "__nv_sqrtf",
                                   "__nv_sqrt");
  populateOpPatterns<math::TanOp>(converter, patterns, "__nv_tanf", "__nv_tan",
                                  "__nv_fast_tanf");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__nv_tanhf",
                                   "__nv_tanh");
}

namespace {
class LibDeviceFuncsRaisingPass
    : public LibDeviceFuncsRaisingPassBase<LibDeviceFuncsRaisingPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LibDeviceFuncsRaisingPass)

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    populateLibDeviceFuncsToOpsPatterns(getOperation()->getContext(), patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc()) << "failed to raise __nv functions";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::enzyme::createLibDeviceFuncsRaisingPass() {
  return std::make_unique<LibDeviceFuncsRaisingPass>();
}
