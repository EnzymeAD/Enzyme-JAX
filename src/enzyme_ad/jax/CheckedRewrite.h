#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
namespace enzyme {

static constexpr StringRef kDisablePatternAttrName =
    "enzymexla.disable_hlo_opts";

static LogicalResult failIfDynamicShape(Operation *op,
                                        PatternRewriter &rewriter) {
  for (auto type : op->getResultTypes()) {
    auto rType = dyn_cast<RankedTensorType>(type);
    if (!rType || !rType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "unsupported dynamic shape for output.");
  }

  for (auto type : op->getOperandTypes()) {
    auto rType = dyn_cast<RankedTensorType>(type);
    if (!rType || !rType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "unsupported dynamic shape for input.");
  }

  return success();
}

static LogicalResult failIfFuncOpInterfaceHasAttr(Operation *op,
                                                  StringRef attrName,
                                                  PatternRewriter &rewriter) {
  if (auto func = op->getParentOfType<FunctionOpInterface>()) {
    if (func->hasAttrOfType<UnitAttr>(attrName))
      return rewriter.notifyMatchFailure(op, "disabled by attribute.");
  }

  return success();
}

template <typename OpTy, typename Child>
struct CheckedOpRewritePattern : public OpRewritePattern<OpTy> {
  using Base = OpRewritePattern<OpTy>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override final {
    LogicalResult res =
        failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter);
    if (res.failed())
      return res;

    if (!((Child *)this)->supportsDynamicShapes()) {
      LogicalResult res = failIfDynamicShape(op, rewriter);
      if (res.failed())
        return res;
    }

    return ((Child *)this)->matchAndRewriteImpl(op, rewriter);
  }

  bool supportsDynamicShapes() const { return false; }
};

template <template <typename> class TraitType, typename Child>
struct CheckedOpTraitRewritePattern : public OpTraitRewritePattern<TraitType> {
  using Base = OpTraitRewritePattern<TraitType>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op,
                  PatternRewriter &rewriter) const override final {
    LogicalResult res =
        failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter);
    if (res.failed())
      return res;

    if (!((Child *)this)->supportsDynamicShapes()) {
      auto res = failIfDynamicShape(op, rewriter);
      if (res.failed())
        return res;
    }

    return ((Child *)this)->matchAndRewriteImpl(op, rewriter);
  }

  bool supportsDynamicShapes() const { return false; }
};

} // namespace enzyme
} // namespace mlir
