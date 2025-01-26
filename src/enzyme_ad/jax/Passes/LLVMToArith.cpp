#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLLVMTOARITHPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-llvm-to-arith"

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

template <typename SourceOp, typename TargetOp>
class AttrConvertPassThrough {
public:
  AttrConvertPassThrough(SourceOp srcOp) : srcAttrs(srcOp->getAttrs()) {}

  ArrayRef<NamedAttribute> getAttrs() const { return srcAttrs; }

private:
  ArrayRef<NamedAttribute> srcAttrs;
};

namespace {
LogicalResult oneToOneRewrite(Operation *op, StringRef targetOp,
                              ValueRange operands,
                              ArrayRef<NamedAttribute> targetAttrs,
                              PatternRewriter &rewriter) {
  unsigned numResults = op->getNumResults();

  // Create the operation through state since we don't know its C++ type.
  Operation *newOp =
      rewriter.create(op->getLoc(), rewriter.getStringAttr(targetOp), operands,
                      op->getResultTypes(), targetAttrs);

  // If the operation produced 0 or 1 result, return them immediately.
  if (numResults == 0)
    return rewriter.eraseOp(op), success();
  if (numResults == 1)
    return rewriter.replaceOp(op, newOp->getResult(0)), success();

  return failure();
}
} // namespace

template <typename TargetOp, typename SourceOp,
          template <typename, typename> typename AttrConvert =
              AttrConvertPassThrough>
class VectorConvertToArithPattern : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  using Super = VectorConvertToArithPattern<SourceOp, TargetOp>;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    SmallVector<Value> operands;
    if constexpr (std::is_same<SourceOp, LLVM::ICmpOp>::value) {
      for (auto opr : op->getOperands()) {
        if (isa<LLVM::LLVMPointerType>(opr.getType()))
          // TODO not always i64
          operands.push_back(rewriter.create<LLVM::PtrToIntOp>(
              op->getLoc(), rewriter.getI64Type(), opr));
        else
          operands.push_back(opr);
      }
    } else {
      for (auto opr : op->getOperands())
        operands.push_back(opr);
    }

    // Determine attributes for the target op
    AttrConvert<SourceOp, TargetOp> attrConvert(op);

    // TODO overflow flags
    return oneToOneRewrite(op, TargetOp::getOperationName(), operands,
                           attrConvert.getAttrs(), rewriter);
  }
};
} // namespace

#define AttrConvertOverflowToArith AttrConvertPassThrough

using CmpIOpConversion =
    VectorConvertToArithPattern<arith::CmpIOp, LLVM::ICmpOp>;
// using AddFOpConversion =
//     VectorConvertToArithPattern<arith::AddFOp, LLVM::FAddOp,
//                                arith::AttrConvertFastMathToLLVM>;
using AddIOpConversion =
    VectorConvertToArithPattern<arith::AddIOp, LLVM::AddOp,
                                AttrConvertOverflowToArith>;
using AndIOpConversion =
    VectorConvertToArithPattern<arith::AndIOp, LLVM::AndOp>;
// using BitcastOpConversion =
//     VectorConvertToArithPattern<arith::BitcastOp, LLVM::BitcastOp>;
// using DivFOpConversion =
//     VectorConvertToArithPattern<arith::DivFOp, LLVM::FDivOp,
//                                arith::AttrConvertFastMathToLLVM>;
using DivSIOpConversion =
    VectorConvertToArithPattern<arith::DivSIOp, LLVM::SDivOp>;
using DivUIOpConversion =
    VectorConvertToArithPattern<arith::DivUIOp, LLVM::UDivOp>;
// using ExtFOpConversion = VectorConvertToArithPattern<arith::ExtFOp,
// LLVM::FPExtOp>;
using ExtSIOpConversion =
    VectorConvertToArithPattern<arith::ExtSIOp, LLVM::SExtOp>;
using ExtUIOpConversion =
    VectorConvertToArithPattern<arith::ExtUIOp, LLVM::ZExtOp>;
using FPToSIOpConversion =
    VectorConvertToArithPattern<arith::FPToSIOp, LLVM::FPToSIOp>;
using FPToUIOpConversion =
    VectorConvertToArithPattern<arith::FPToUIOp, LLVM::FPToUIOp>;
// using MaximumFOpConversion =
//     VectorConvertToArithPattern<arith::MaximumFOp, LLVM::MaximumOp,
//                                arith::AttrConvertFastMathToLLVM>;
// using MaxNumFOpConversion =
//     VectorConvertToArithPattern<arith::MaxNumFOp, LLVM::MaxNumOp,
//                                arith::AttrConvertFastMathToLLVM>;
using MaxSIOpConversion =
    VectorConvertToArithPattern<arith::MaxSIOp, LLVM::SMaxOp>;
using MaxUIOpConversion =
    VectorConvertToArithPattern<arith::MaxUIOp, LLVM::UMaxOp>;
// using MinimumFOpConversion =
//     VectorConvertToArithPattern<arith::MinimumFOp, LLVM::MinimumOp,
//                                arith::AttrConvertFastMathToLLVM>;
// using MinNumFOpConversion =
//     VectorConvertToArithPattern<arith::MinNumFOp, LLVM::MinNumOp,
//                                arith::AttrConvertFastMathToLLVM>;
using MinSIOpConversion =
    VectorConvertToArithPattern<arith::MinSIOp, LLVM::SMinOp>;
using MinUIOpConversion =
    VectorConvertToArithPattern<arith::MinUIOp, LLVM::UMinOp>;
// using MulFOpConversion =
//     VectorConvertToArithPattern<arith::MulFOp, LLVM::FMulOp,
//                                arith::AttrConvertFastMathToLLVM>;
using MulIOpConversion =
    VectorConvertToArithPattern<arith::MulIOp, LLVM::MulOp,
                                AttrConvertOverflowToArith>;
// using NegFOpConversion =
//     VectorConvertToArithPattern<arith::NegFOp, LLVM::FNegOp,
//                                arith::AttrConvertFastMathToLLVM>;
using OrIOpConversion = VectorConvertToArithPattern<arith::OrIOp, LLVM::OrOp>;
// using RemFOpConversion =
//     VectorConvertToArithPattern<arith::RemFOp, LLVM::FRemOp,
//                                arith::AttrConvertFastMathToLLVM>;
using RemSIOpConversion =
    VectorConvertToArithPattern<arith::RemSIOp, LLVM::SRemOp>;
using RemUIOpConversion =
    VectorConvertToArithPattern<arith::RemUIOp, LLVM::URemOp>;
using SelectOpConversion =
    VectorConvertToArithPattern<arith::SelectOp, LLVM::SelectOp>;
using ShLIOpConversion =
    VectorConvertToArithPattern<arith::ShLIOp, LLVM::ShlOp,
                                AttrConvertOverflowToArith>;
using ShRSIOpConversion =
    VectorConvertToArithPattern<arith::ShRSIOp, LLVM::AShrOp>;
using ShRUIOpConversion =
    VectorConvertToArithPattern<arith::ShRUIOp, LLVM::LShrOp>;
using SIToFPOpConversion =
    VectorConvertToArithPattern<arith::SIToFPOp, LLVM::SIToFPOp>;
// using SubFOpConversion =
//     VectorConvertToArithPattern<arith::SubFOp, LLVM::FSubOp,
//                                arith::AttrConvertFastMathToLLVM>;
using SubIOpConversion =
    VectorConvertToArithPattern<arith::SubIOp, LLVM::SubOp,
                                AttrConvertOverflowToArith>;
// using TruncFOpConversion =
//     ConstrainedVectorConvertToLLVMPattern<arith::TruncFOp, LLVM::FPTruncOp,
//                                           false>;
// using ConstrainedTruncFOpConversion = ConstrainedVectorConvertToLLVMPattern<
//     arith::TruncFOp, LLVM::ConstrainedFPTruncIntr, true,
//     arith::AttrConverterConstrainedFPToLLVM>;
using TruncIOpConversion =
    VectorConvertToArithPattern<arith::TruncIOp, LLVM::TruncOp>;
// using UIToFPOpConversion =
//     VectorConvertToArithPattern<arith::UIToFPOp, LLVM::UIToFPOp>;
using XOrIOpConversion =
    VectorConvertToArithPattern<arith::XOrIOp, LLVM::XOrOp>;

struct ConstantOpConversion : public OpRewritePattern<LLVM::ConstantOp> {
  using OpRewritePattern<LLVM::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getResult().getType().isInteger())
      return failure();
    return oneToOneRewrite(op, arith::ConstantIntOp::getOperationName(),
                           op->getOperands(), op->getAttrs(), rewriter);
  }
};

void mlir::arith::populateLLVMToArithConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    //AddFOpConversion,
    AddIOpConversion,
    AndIOpConversion,
    //AddUIExtendedOpConversion,
    //BitcastOpConversion,
    ConstantOpConversion,
    //CmpFOpConversion,
    CmpIOpConversion,
    //DivFOpConversion,
    DivSIOpConversion,
    DivUIOpConversion,
    //ExtFOpConversion,
    ExtSIOpConversion,
    ExtUIOpConversion,
    FPToSIOpConversion,
    FPToUIOpConversion,
    //IndexCastOpSIConversion,
    //IndexCastOpUIConversion,
    //MaximumFOpConversion,
    //MaxNumFOpConversion,
    MaxSIOpConversion,
    MaxUIOpConversion,
    //MinimumFOpConversion,
    //MinNumFOpConversion,
    MinSIOpConversion,
    MinUIOpConversion,
    //MulFOpConversion,
    MulIOpConversion,
    //MulSIExtendedOpConversion,
    //MulUIExtendedOpConversion,
    //NegFOpConversion,
    OrIOpConversion,
    //RemFOpConversion,
    RemSIOpConversion,
    RemUIOpConversion,
    SelectOpConversion,
    ShLIOpConversion,
    ShRSIOpConversion,
    ShRUIOpConversion,
    SIToFPOpConversion,
    //SubFOpConversion,
    SubIOpConversion,
    //TruncFOpConversion,
    //ConstrainedTruncFOpConversion,
    TruncIOpConversion,
    //UIToFPOpConversion,
    XOrIOpConversion
      >(patterns.getContext());
  // clang-format on
}

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct ConvertLLVMToArith
    : public impl::ConvertLLVMToArithPassBase<ConvertLLVMToArith> {
  using Base::Base;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::arith::populateLLVMToArithConversionPatterns(patterns);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    getOperation()->walk([](LLVM::DbgValueOp op) { op->erase(); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createConvertLLVMToArithPass() {
  return std::make_unique<ConvertLLVMToArith>();
}
