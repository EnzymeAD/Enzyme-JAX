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

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "src/enzyme_ad/jax/Implementations/MHLODerivatives.inc"

#ifndef DISABLE_MHLO_TENSOR_INTERFACE
class MHLOTensorTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<MHLOTensorTypeInterface,
                                                  TensorType> {
public:
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto tenType = self.cast<TensorType>();
    auto attr = DenseElementsAttr::get(tenType, 0);
    return builder.create<mhlo::ConstantOp>(loc, tenType, attr);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    return builder.create<mhlo::AddOp>(loc, a, b);
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool requiresShadow(Type self) const { return false; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
};
#endif
} // namespace

void mlir::enzyme::registerMHLODialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, mhlo::MhloDialect *) {
    registerInterfaces(context);

#ifndef DISABLE_MHLO_TENSOR_INTERFACE
    UnrankedTensorType::attachInterface<MHLOTensorTypeInterface>(*context);
    RankedTensorType::attachInterface<MHLOTensorTypeInterface>(*context);
#endif
  });
}
