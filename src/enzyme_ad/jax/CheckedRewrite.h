#pragma once

#include <type_traits>

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

static LogicalResult checkPreconditions(Operation *op,
                                        PatternRewriter &rewriter,
                                        bool supportsDynamicShapes) {
  if (op->hasAttr(kDisablePatternAttrName))
    return rewriter.notifyMatchFailure(op, "disabled by attribute.");

  if (failIfFuncOpInterfaceHasAttr(op, kDisablePatternAttrName, rewriter)
          .failed())
    return failure();

  if (!supportsDynamicShapes) {
    if (failIfDynamicShape(op, rewriter).failed())
      return failure();
  }

  return success();
}

template <typename OpTy, typename Child>
struct CheckedOpRewritePattern : public OpRewritePattern<OpTy> {
  using Base = OpRewritePattern<OpTy>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override final {
    if (checkPreconditions(op, rewriter,
                           ((Child *)this)->supportsDynamicShapes())
            .failed())
      return failure();

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
    if (checkPreconditions(op, rewriter,
                           ((Child *)this)->supportsDynamicShapes())
            .failed())
      return failure();

    return ((Child *)this)->matchAndRewriteImpl(op, rewriter);
  }

  bool supportsDynamicShapes() const { return false; }
};

template <typename T, typename = void>
struct has_supports_dynamic_shapes : std::false_type {};

template <typename T>
struct has_supports_dynamic_shapes<
    T, std::void_t<decltype(std::declval<T>().supportsDynamicShapes())>>
    : std::true_type {};

template <typename PatternTy> struct CheckedPattern : public PatternTy {
  using PatternTy::PatternTy;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool supportsDynamic = false;
    if constexpr (has_supports_dynamic_shapes<PatternTy>::value) {
      supportsDynamic = this->supportsDynamicShapes();
    }

    if (checkPreconditions(op, rewriter, supportsDynamic).failed())
      return failure();

    return PatternTy::matchAndRewrite(op, rewriter);
  }
};

} // namespace enzyme
} // namespace mlir
