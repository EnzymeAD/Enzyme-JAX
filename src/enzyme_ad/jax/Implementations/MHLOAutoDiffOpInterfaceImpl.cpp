//===- ArithAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/ChloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"
#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::mhlo;

static int64_t to_i64(int64_t x) { return x; }
static int64_t to_i64(llvm::APInt x) { return x.getSExtValue(); }

static mlir::DenseIntElementsAttr getI64Attr(OpBuilder &builder,
                                             llvm::ArrayRef<int64_t> vals) {
  return builder.getI64VectorAttr(vals);
}

static int64_t getI64Value(mlir::DenseIntElementsAttr attr, size_t pos) {
  return attr.getValues<int64_t>()[pos];
}

static mlir::DenseElementsAttr getBoolAttr(OpBuilder &builder,
                                           llvm::ArrayRef<bool> vals) {
  return builder.getBoolVectorAttr(vals);
}

static auto getBoolIter(mlir::DenseElementsAttr attr) {
  return attr.getValues<bool>();
}
static inline Operation *createAddRegion(Operation *op) {
  mlir::OpBuilder builder(op->getContext());
  mlir::Block *block = new Block();
  op->getRegion(0).push_back(block);
  auto elemType = cast<ShapedType>(op->getResult(0).getType()).getElementType();
  auto tensorType = RankedTensorType::get({}, elemType);
  block->addArguments({tensorType, tensorType}, {op->getLoc(), op->getLoc()});
  builder.setInsertionPointToEnd(block);
  mlir::mhlo::ReturnOp::create(builder, op->getLoc(),
                               mlir::mhlo::AddOp::create(builder, op->getLoc(),
                                                         block->getArgument(0),
                                                         block->getArgument(1))
                                   ->getResult(0));
  return op;
}

static inline DenseIntElementsAttr
getBroadcastInDimsAttr(OpBuilder &builder, ArrayRef<int64_t> dims) {
  return builder.getI64VectorAttr(dims);
}

static inline SmallVector<int64_t> shiftDimensions(ArrayRef<int64_t> dims,
                                                   SmallVector<int64_t> newDims,
                                                   int64_t addFactor) {
  SmallVector<int64_t> shiftedDims(newDims.begin(), newDims.end());
  for (auto dim : dims) {
    shiftedDims.push_back(dim + addFactor);
  }
  return shiftedDims;
}

namespace {
#include "src/enzyme_ad/jax/Implementations/MHLODerivatives.inc"
} // namespace

void mlir::enzyme::registerMHLODialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, mhlo::MhloDialect *) {
    registerInterfaces(context);
  });
}
