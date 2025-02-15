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
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LIBDEVICEFUNCSRAISINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename SourceOp, typename TargetOp,
          template <typename, typename> typename AttrConvert =
              AttrConvertPassThrough>
class VectorConvertFromLLVMPattern : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");
    // Determine attributes for the target op
    AttrConvert<SourceOp, TargetOp> attrConvert(op);

    auto operands = op->getOperands();
    auto llvmNDVectorTy = operands[0].getType();
    if (isa<LLVM::LLVMArrayType, LLVM::LLVMFixedVectorType,
            LLVM::LLVMScalableVectorType>(llvmNDVectorTy)) {
      return failure();
    }
    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(TargetOp::getOperationName()),
        operands, op->getResultTypes(), attrConvert.getAttrs());

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

arith::IntegerOverflowFlags
convertArithOverflowFlagsFromLLVM(LLVM::IntegerOverflowFlags llvmFlags) {
  arith::IntegerOverflowFlags arithFlags{};
  const std::pair<arith::IntegerOverflowFlags, LLVM::IntegerOverflowFlags>
      flags[] = {
          {arith::IntegerOverflowFlags::nsw, LLVM::IntegerOverflowFlags::nsw},
          {arith::IntegerOverflowFlags::nuw, LLVM::IntegerOverflowFlags::nuw}};
  for (auto [arithFlag, llvmFlag] : flags) {
    if (bitEnumContainsAny(llvmFlags, llvmFlag))
      arithFlags = arithFlags | arithFlag;
  }
  return arithFlags;
}

template <typename SourceOp, typename TargetOp>
class AttrConvertOverflowFromLLVM {
public:
  AttrConvertOverflowFromLLVM(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith overflow attribute.
    StringRef arithAttrName = SourceOp::getIntegerOverflowAttrName();
    // Remove the source overflow attribute.
    if (auto arithAttr = dyn_cast_if_present<LLVM::IntegerOverflowFlagsAttr>(
            convertedAttr.erase(arithAttrName))) {
      if (arithAttr.getValue() != LLVM::IntegerOverflowFlags::none) {
        StringRef targetAttrName = TargetOp::getOverflowFlagsAttrName();
        convertedAttr.set(targetAttrName, arith::IntegerOverflowFlagsAttr::get(
                                              srcOp->getContext(),
                                              convertArithOverflowFlagsFromLLVM(
                                                  arithAttr.getValue())));
      }
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};

arith::FastMathFlags
convertArithFastMathFlagsFromLLVM(LLVM::FastmathFlags llvmFMF) {
  arith::FastMathFlags arithFMF{};
  const std::pair<arith::FastMathFlags, LLVM::FastmathFlags> flags[] = {
      {arith::FastMathFlags::nnan, LLVM::FastmathFlags::nnan},
      {arith::FastMathFlags::ninf, LLVM::FastmathFlags::ninf},
      {arith::FastMathFlags::nsz, LLVM::FastmathFlags::nsz},
      {arith::FastMathFlags::arcp, LLVM::FastmathFlags::arcp},
      {arith::FastMathFlags::contract, LLVM::FastmathFlags::contract},
      {arith::FastMathFlags::afn, LLVM::FastmathFlags::afn},
      {arith::FastMathFlags::reassoc, LLVM::FastmathFlags::reassoc}};
  for (auto [arithFlag, llvmFlag] : flags) {
    if (bitEnumContainsAny(llvmFMF, llvmFlag))
      arithFMF = arithFMF | arithFlag;
  }
  return arithFMF;
}

arith::FastMathFlagsAttr
convertArithFastMathAttrFromLLVM(LLVM::FastmathFlagsAttr fmfAttr) {
  auto arithFMF = fmfAttr.getValue();
  return arith::FastMathFlagsAttr::get(
      fmfAttr.getContext(), convertArithFastMathFlagsFromLLVM(arithFMF));
}

// Attribute converter that populates a NamedAttrList by removing the fastmath
// attribute from the source operation attributes, and replacing it with an
// equivalent LLVM fastmath attribute.
template <typename SourceOp, typename TargetOp>
class AttrConvertFastMathFromLLVM {
public:
  AttrConvertFastMathFromLLVM(SourceOp srcOp) {
    // Copy the source attributes.
    convertedAttr = NamedAttrList{srcOp->getAttrs()};
    // Get the name of the arith fastmath attribute.
    StringRef arithFMFAttrName = SourceOp::getFastmathAttrName();
    // Remove the source fastmath attribute.
    auto arithFMFAttr = dyn_cast_if_present<LLVM::FastmathFlagsAttr>(
        convertedAttr.erase(arithFMFAttrName));
    if (arithFMFAttr &&
        arithFMFAttr.getValue() != mlir::LLVM::FastmathFlags::none) {
      StringRef targetAttrName = TargetOp::getFastMathAttrName();
      convertedAttr.set(targetAttrName,
                        convertArithFastMathAttrFromLLVM(arithFMFAttr));
    }
  }

  ArrayRef<NamedAttribute> getAttrs() const { return convertedAttr.getAttrs(); }

private:
  NamedAttrList convertedAttr;
};

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

class IsFPClassRaising : public OpRewritePattern<LLVM::CallOp> {
public:
  IsFPClassRaising(MLIRContext *context)
      : OpRewritePattern<LLVM::CallOp>(context) {}

  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rewriter) const override {
    CallInterfaceCallable callable = op.getCallableForCallee();
    auto callee = callable.dyn_cast<SymbolRefAttr>();
    if (!callee)
      return failure();

    if (callee.getLeafReference() == "__nv_isnand" ||
        callee.getLeafReference() == "__nv_isnan") {
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(
          op, op->getResultTypes(),
          rewriter.create<LLVM::IsFPClass>(op.getLoc(), rewriter.getI1Type(),
                                           op->getOperands()[0], 3));
      return success();
    }

    // https://llvm.org/docs/LangRef.html#llvm-is-fpclass
    /*
     *
Bit #

floating-point class

0 Signaling NaN

1 Quiet NaN

2 Negative infinity

3 Negative normal

4 Negative subnormal

5 Negative zero

6 Positive zero

7 Positive subnormal

8 Positive normal

9 Positive infinity

2**3 + 2**4 + 2**5 + 2**6 + 2**7 + 2**8
*/
    if (callee.getLeafReference() == "__nv_isfinited" ||
        callee.getLeafReference() == "__nv_isfinite") {
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(
          op, op->getResultTypes(),
          rewriter.create<LLVM::IsFPClass>(op.getLoc(), rewriter.getI1Type(),
                                           op->getOperands()[0],
                                           ((1 << 3) | (1 << 4) | (1 << 5) |
                                            (1 << 6) | (1 << 7) | (1 << 8))));
      return success();
    }

    return failure();
  }
};
} // namespace

template <typename TargetOp, typename Arg, typename... Args>
static void populateOpPatterns(MLIRContext *context,
                               RewritePatternSet &patterns, Arg &&arg,
                               Args &&...args) {
  patterns.add<CallToOpRaising<TargetOp>>(context, std::forward<Arg>(arg));
  if constexpr (sizeof...(Args) != 0)
    populateOpPatterns<TargetOp>(context, patterns,
                                 std::forward<Args>(args)...);
}

namespace {

// From
// https://github.com/llvm/llvm-project/blob/7d8b4eb0ead277f41ff69525ed807f9f6e227f37/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp#L31
// except we invert source and target
template <typename SourceOp, typename TargetOp>
using ConvertFastMath = AttrConvertFastMathFromLLVM<SourceOp, TargetOp>;

template <typename SourceOp, typename TargetOp,
          template <typename, typename> typename AttrConvert =
              AttrConvertPassThrough>
using InvVectorConvertFromLLVMPattern =
    VectorConvertFromLLVMPattern<TargetOp, SourceOp, AttrConvertPassThrough>;

template <typename SourceOp, typename TargetOp>
using ConvertFMFMathFromLLVMPattern =
    VectorConvertFromLLVMPattern<TargetOp, SourceOp, ConvertFastMath>;

using AbsFOpLowering =
    ConvertFMFMathFromLLVMPattern<math::AbsFOp, LLVM::FAbsOp>;
using CeilOpLowering =
    ConvertFMFMathFromLLVMPattern<math::CeilOp, LLVM::FCeilOp>;
using CopySignOpLowering =
    ConvertFMFMathFromLLVMPattern<math::CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = ConvertFMFMathFromLLVMPattern<math::CosOp, LLVM::CosOp>;
using CtPopFOpLowering =
    VectorConvertFromLLVMPattern<LLVM::CtPopOp, math::CtPopOp>;
using Exp2OpLowering =
    ConvertFMFMathFromLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using ExpOpLowering = ConvertFMFMathFromLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using FloorOpLowering =
    ConvertFMFMathFromLLVMPattern<math::FloorOp, LLVM::FFloorOp>;
using FmaOpLowering = ConvertFMFMathFromLLVMPattern<math::FmaOp, LLVM::FMAOp>;
using Log10OpLowering =
    ConvertFMFMathFromLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering =
    ConvertFMFMathFromLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = ConvertFMFMathFromLLVMPattern<math::LogOp, LLVM::LogOp>;
using PowFOpLowering = ConvertFMFMathFromLLVMPattern<math::PowFOp, LLVM::PowOp>;
using FPowIOpLowering =
    ConvertFMFMathFromLLVMPattern<math::FPowIOp, LLVM::PowIOp>;
using RoundEvenOpLowering =
    ConvertFMFMathFromLLVMPattern<math::RoundEvenOp, LLVM::RoundEvenOp>;
using RoundOpLowering =
    ConvertFMFMathFromLLVMPattern<math::RoundOp, LLVM::RoundOp>;
using SinOpLowering = ConvertFMFMathFromLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering =
    ConvertFMFMathFromLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;
using FTruncOpLowering =
    ConvertFMFMathFromLLVMPattern<math::TruncOp, LLVM::FTruncOp>;

using AddFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::AddFOp, LLVM::FAddOp,
                                    AttrConvertFastMathFromLLVM>;
using AddIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::AddIOp, LLVM::AddOp,
                                    AttrConvertOverflowFromLLVM>;
using AndIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::AndIOp, LLVM::AndOp>;
using BitcastOpLowering =
    InvVectorConvertFromLLVMPattern<arith::BitcastOp, LLVM::BitcastOp>;
using DivFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::DivFOp, LLVM::FDivOp,
                                    AttrConvertFastMathFromLLVM>;
using DivSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::DivSIOp, LLVM::SDivOp>;
using DivUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::DivUIOp, LLVM::UDivOp>;
using ExtFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ExtFOp, LLVM::FPExtOp>;
using ExtSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ExtSIOp, LLVM::SExtOp>;
using ExtUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ExtUIOp, LLVM::ZExtOp>;
using FPToSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::FPToSIOp, LLVM::FPToSIOp>;
using FPToUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::FPToUIOp, LLVM::FPToUIOp>;
using MaximumFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MaximumFOp, LLVM::MaximumOp,
                                    AttrConvertFastMathFromLLVM>;
using MaxNumFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MaxNumFOp, LLVM::MaxNumOp,
                                    AttrConvertFastMathFromLLVM>;
using MaxSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MaxSIOp, LLVM::SMaxOp>;
using MaxUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MaxUIOp, LLVM::UMaxOp>;
using MinimumFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MinimumFOp, LLVM::MinimumOp,
                                    AttrConvertFastMathFromLLVM>;
using MinNumFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MinNumFOp, LLVM::MinNumOp,
                                    AttrConvertFastMathFromLLVM>;
using MinSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MinSIOp, LLVM::SMinOp>;
using MinUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MinUIOp, LLVM::UMinOp>;
using MulFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MulFOp, LLVM::FMulOp,
                                    AttrConvertFastMathFromLLVM>;
using MulIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::MulIOp, LLVM::MulOp,
                                    AttrConvertOverflowFromLLVM>;
using NegFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::NegFOp, LLVM::FNegOp,
                                    AttrConvertFastMathFromLLVM>;
using OrIOpLowering = InvVectorConvertFromLLVMPattern<arith::OrIOp, LLVM::OrOp>;
using RemFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::RemFOp, LLVM::FRemOp,
                                    AttrConvertFastMathFromLLVM>;
using RemSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::RemSIOp, LLVM::SRemOp>;
using RemUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::RemUIOp, LLVM::URemOp>;
using SelectOpLowering =
    InvVectorConvertFromLLVMPattern<arith::SelectOp, LLVM::SelectOp>;
using ShLIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ShLIOp, LLVM::ShlOp,
                                    AttrConvertOverflowFromLLVM>;
using ShRSIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ShRSIOp, LLVM::AShrOp>;
using ShRUIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::ShRUIOp, LLVM::LShrOp>;
using SIToFPOpLowering =
    InvVectorConvertFromLLVMPattern<arith::SIToFPOp, LLVM::SIToFPOp>;
using SubFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::SubFOp, LLVM::FSubOp,
                                    AttrConvertFastMathFromLLVM>;
using SubIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::SubIOp, LLVM::SubOp,
                                    AttrConvertOverflowFromLLVM>;
// using TruncFOpLowering =
//    ConstrainedVectorConvertFromLLVMPattern<arith::TruncFOp, LLVM::FPTruncOp,
//                                          false>;
// using ConstrainedTruncFOpLowering = ConstrainedVectorConvertFromLLVMPattern<
//    arith::TruncFOp, LLVM::ConstrainedFPTruncIntr, true,
//    arith::AttrConverterConstrainedFPFromLLVM>;
using TruncIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::TruncIOp, LLVM::TruncOp>;
using UIToFPOpLowering =
    InvVectorConvertFromLLVMPattern<arith::UIToFPOp, LLVM::UIToFPOp>;
using XOrIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::XOrIOp, LLVM::XOrOp>;
using CmpIOpLowering =
    InvVectorConvertFromLLVMPattern<arith::CmpIOp, LLVM::ICmpOp>;
using CmpFOpLowering =
    InvVectorConvertFromLLVMPattern<arith::CmpFOp, LLVM::FCmpOp,
                                    AttrConvertFastMathFromLLVM>;

struct ConstantOpLowering : public OpRewritePattern<LLVM::ConstantOp> {
  using OpRewritePattern<LLVM::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<mlir::IntegerType, mlir::FloatType>(op.getResult().getType())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, op->getResultTypes(), op->getOperands(), op->getAttrs());
      return success();
    }
    return failure();
  }
};

} // namespace

void mlir::enzyme::populateLibDeviceFuncsToOpsPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  // XXX: Keep in sync with
  // mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp.

  auto *converter = context;

  patterns.add<IsFPClassRaising>(context);

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
  populateOpPatterns<math::FPowIOp>(converter, patterns, "__nv_powif",
                                    "__nv_powi");
  populateOpPatterns<math::AbsIOp>(converter, patterns, "__nv_abs",
                                   "__nv_llabs");
  populateOpPatterns<arith::MaxNumFOp>(converter, patterns, "__nv_fmax",
                                       "__nv_fmaxf");
  populateOpPatterns<arith::MinNumFOp>(converter, patterns, "__nv_fmin",
                                       "__nv_fminf");
}

void populateLLVMToMathPatterns(MLIRContext *context,
                                RewritePatternSet &patterns) {
  auto *converter = context;
  // From
  // https://github.com/llvm/llvm-project/blob/7d8b4eb0ead277f41ff69525ed807f9f6e227f37/mlir/lib/Conversion/MathToLLVM/MathToLLVM.cpp#L306
  // patterns.add<FTruncOpLowering>(converter);
  patterns.add<AbsFOpLowering,
               // AbsIOpLowering,
               CeilOpLowering, CopySignOpLowering, CosOpLowering,
               // CountLeadingZerosOpLowering,
               // CountTrailingZerosOpLowering,
               // CtPopFOpLowering,
               Exp2OpLowering,
               // ExpM1OpLowering,
               ExpOpLowering, FPowIOpLowering, FloorOpLowering, FmaOpLowering,
               Log10OpLowering, Log2OpLowering, LogOpLowering, PowFOpLowering,
               RoundEvenOpLowering, RoundOpLowering,
               // RsqrtOpLowering,
               SinOpLowering, SqrtOpLowering, FTruncOpLowering>(converter);

  patterns.add<CmpFOpLowering, CmpIOpLowering>(converter);

  patterns
      .add<AddFOpLowering, AddIOpLowering, AndIOpLowering,
           // AddUIExtendedOpLowering,
           BitcastOpLowering, ConstantOpLowering, DivFOpLowering,
           DivSIOpLowering, DivUIOpLowering, ExtFOpLowering, ExtSIOpLowering,
           ExtUIOpLowering, FPToSIOpLowering, FPToUIOpLowering,
           // IndexCastOpSILowering,
           // IndexCastOpUILowering,
           MaximumFOpLowering, MaxNumFOpLowering, MaxSIOpLowering,
           MaxUIOpLowering, MinimumFOpLowering, MinNumFOpLowering,
           MinSIOpLowering, MinUIOpLowering, MulFOpLowering, MulIOpLowering,
           // MulSIExtendedOpLowering,
           // MulUIExtendedOpLowering,
           NegFOpLowering, OrIOpLowering, RemFOpLowering, RemSIOpLowering,
           RemUIOpLowering, SelectOpLowering, ShLIOpLowering, ShRSIOpLowering,
           ShRUIOpLowering, SIToFPOpLowering, SubFOpLowering, SubIOpLowering,
           // TruncFOpLowering,
           // ConstrainedTruncFOpLowering,
           TruncIOpLowering, UIToFPOpLowering, XOrIOpLowering>(converter);
}

namespace {
struct LibDeviceFuncsRaisingPass
    : public enzyme::impl::LibDeviceFuncsRaisingPassBase<
          LibDeviceFuncsRaisingPass> {
  using LibDeviceFuncsRaisingPassBase::LibDeviceFuncsRaisingPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation()->getContext());
    populateLLVMToMathPatterns(getOperation()->getContext(), patterns);
    populateLibDeviceFuncsToOpsPatterns(getOperation()->getContext(), patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      emitError(getOperation()->getLoc()) << "failed to raise __nv functions";
      return signalPassFailure();
    }
  }
};
} // namespace
