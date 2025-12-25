//===- TritonAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the MLIR tt dialect.
//
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::triton;

namespace {

#include "src/enzyme_ad/jax/Implementations/TritonDerivatives.inc"

class TritonPointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<TritonPointerTypeInterface,
                                                  triton::PointerType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm_unreachable("TODO");
  }
  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    llvm_unreachable("TODO");
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    // TODO inspect val and memset corresponding size
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

class AutoDiffTritonFuncFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffTritonFuncFunctionInterface, triton::FuncOp> {
public:
  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &returnTypes) const {}

  Operation *createCall(Operation *self, OpBuilder &builder, Location loc,
                        ValueRange args) const {
    return triton::CallOp::create(builder, loc, cast<triton::FuncOp>(self),
                                  args);
  }

  Operation *createReturn(Operation *self, OpBuilder &builder, Location loc,
                          ValueRange retArgs) const {
    return triton::ReturnOp::create(builder, loc, retArgs);
  }
};

} // end anonymous namespace

void mlir::enzyme::registerTritonDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, triton::TritonDialect *) {
    registerInterfaces(context);
    triton::FuncOp::attachInterface<AutoDiffTritonFuncFunctionInterface>(
        *context);
    triton::PointerType::attachInterface<TritonPointerTypeInterface>(*context);
  });
}
