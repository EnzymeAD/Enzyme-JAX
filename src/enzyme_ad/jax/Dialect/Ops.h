//===- EnzymeXLAOps.h - EnzymeXLA dialect ops -------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLAOPS_H
#define ENZYMEXLAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "stablehlo/dialect/Base.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> BarrierOpt;

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/EnzymeXLAOps.h.inc"

struct ValueOrInt_Polygeist {
  bool isValue;
  mlir::Value v_val;
  int64_t i_val;
  ValueOrInt_Polygeist(mlir::Value v) { initValue(v); }
  void initValue(mlir::Value v) {
    using namespace mlir;
    if (v) {
      IntegerAttr iattr;
      if (matchPattern(v, m_Constant(&iattr))) {
        i_val = iattr.getValue().getSExtValue();
        v_val = nullptr;
        isValue = false;
        return;
      }
    }
    isValue = true;
    v_val = v;
  }

  ValueOrInt_Polygeist(size_t i) : isValue(false), v_val(), i_val(i) {}

  bool operator>=(int64_t v) {
    if (isValue)
      return false;
    return i_val >= v;
  }
  bool operator>(int64_t v) {
    if (isValue)
      return false;
    return i_val > v;
  }
  bool operator==(int64_t v) {
    if (isValue)
      return false;
    return i_val == v;
  }
  bool operator<(int64_t v) {
    if (isValue)
      return false;
    return i_val < v;
  }
  bool operator<=(int64_t v) {
    if (isValue)
      return false;
    return i_val <= v;
  }
  bool operator>=(llvm::APInt v) {
    if (isValue)
      return false;
    return i_val >= v.getSExtValue();
  }
  bool operator>(llvm::APInt v) {
    if (isValue)
      return false;
    return i_val > v.getSExtValue();
  }
  bool operator==(llvm::APInt v) {
    if (isValue)
      return false;
    return i_val == v.getSExtValue();
  }
  bool operator<(llvm::APInt v) {
    if (isValue)
      return false;
    return i_val < v.getSExtValue();
  }
  bool operator<=(llvm::APInt v) {
    if (isValue)
      return false;
    return i_val <= v.getSExtValue();
  }
};

enum class Cmp_Polygeist { EQ, LT, LE, GT, GE };

bool valueCmp_Polygeist(Cmp_Polygeist cmp, mlir::AffineExpr expr, size_t numDim,
              mlir::ValueRange operands, ValueOrInt_Polygeist val);

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, mlir::AffineExpr expr, size_t numDim,
//               mlir::ValueRange operands, int64_t val);

bool valueCmp_Polygeist(Cmp_Polygeist cmp, ValueOrInt_Polygeist bval, ValueOrInt_Polygeist val);
// bool valueCmp_Polygeist(Cmp_Polygeist cmp, ValueOrInt_Polygeist bval, int64_t val);

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, mlir::Value bval, ValueOrInt_Polygeist val);

// bool valueCmp_Polygeist(Cmp_Polygeist cmp, llvm::APInt bval, ValueOrInt_Polygeist val);
#endif // ENZYMEXLAOPS_H
