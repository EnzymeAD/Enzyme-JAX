//===- Passes.h - Enzyme pass include header  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ENZYMEXLA_PASSES_H
#define ENZYMEXLA_PASSES_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PatternRewriter;
class AffineMap;
class DominanceInfo;

namespace enzyme {

void populateAffineCFGPatterns(RewritePatternSet &rpl);

#define GEN_PASS_DECL
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

void populateLibDeviceFuncsToOpsPatterns(MLIRContext *context,
                                         RewritePatternSet &patterns);

void addSingleIter(mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);

mlir::AffineExpr recreateExpr(mlir::AffineExpr expr);
mlir::AffineMap recreateExpr(mlir::AffineMap expr);
mlir::IntegerSet recreateExpr(mlir::IntegerSet expr);
} // namespace enzyme

namespace cf {
void populateLLVMToControlFlowConversionPatterns(RewritePatternSet &patterns);
} // namespace cf

} // end namespace mlir

void fully2ComposeAffineMapAndOperands(
    mlir::PatternRewriter &rewriter, mlir::AffineMap *map,
    llvm::SmallVectorImpl<mlir::Value> *operands, mlir::DominanceInfo &DI,
    mlir::Region *scope,
    llvm::SmallVectorImpl<mlir::Operation *> *insertedOps = nullptr);
bool isValidIndex(mlir::Value val, mlir::Region *scope);
mlir::Region *getLocalAffineScope(mlir::Operation *op);

struct ValueOrInt {
  bool isValue;
  mlir::Value v_val;
  llvm::APInt i_val;
  ValueOrInt(mlir::Value v) { initValue(v); }
  void initValue(mlir::Value v) {
    using namespace mlir;
    if (v) {
      IntegerAttr iattr;
      if (matchPattern(v, m_Constant(&iattr))) {
        i_val = iattr.getValue();
        v_val = nullptr;
        isValue = false;
        return;
      }
    }
    isValue = true;
    v_val = v;
  }

  ValueOrInt(llvm::APInt i) : isValue(false), v_val(), i_val(i) {}

  bool operator>=(int64_t v) {
    if (isValue)
      return false;
    return i_val.sge(v);
  }
  bool operator>(int64_t v) {
    if (isValue)
      return false;
    return i_val.sgt(v);
  }
  bool operator==(int64_t v) {
    if (isValue)
      return false;
    return i_val == v;
  }
  bool operator<(int64_t v) {
    if (isValue)
      return false;
    return i_val.slt(v);
  }
  bool operator<=(int64_t v) {
    if (isValue)
      return false;
    return i_val.sle(v);
  }
  bool operator>=(llvm::APInt v) {
    if (isValue)
      return false;
    if (v.getBitWidth() != i_val.getBitWidth()) {
      return operator>=(v.getSExtValue());
    }
    return i_val.sge(v);
  }
  bool operator>(llvm::APInt v) {
    if (isValue)
      return false;
    if (v.getBitWidth() != i_val.getBitWidth()) {
      return operator>(v.getSExtValue());
    }
    return i_val.sgt(v);
  }
  bool operator==(llvm::APInt v) {
    if (isValue)
      return false;
    if (v.getBitWidth() != i_val.getBitWidth()) {
      return operator==(v.getSExtValue());
    }
    return i_val == v;
  }
  bool operator<(llvm::APInt v) {
    if (isValue)
      return false;
    if (v.getBitWidth() != i_val.getBitWidth()) {
      return operator<(v.getSExtValue());
    }
    return i_val.slt(v);
  }
  bool operator<=(llvm::APInt v) {
    if (isValue)
      return false;
    if (v.getBitWidth() != i_val.getBitWidth()) {
      return operator<=(v.getSExtValue());
    }
    return i_val.sle(v);
  }
};

enum class Cmp { EQ, LT, LE, GT, GE };

bool valueCmp(Cmp cmp, mlir::AffineExpr expr, size_t numDim,
              mlir::ValueRange operands, ValueOrInt val);

bool valueCmp(Cmp cmp, mlir::AffineExpr expr, size_t numDim,
              mlir::ValueRange operands, int64_t val);

bool valueCmp(Cmp cmp, ValueOrInt bval, ValueOrInt val);
bool valueCmp(Cmp cmp, ValueOrInt bval, int64_t val);

bool valueCmp(Cmp cmp, mlir::Value bval, ValueOrInt val);

bool valueCmp(Cmp cmp, llvm::APInt bval, ValueOrInt val);

#endif // ENZYMEXLA_PASSES_H
