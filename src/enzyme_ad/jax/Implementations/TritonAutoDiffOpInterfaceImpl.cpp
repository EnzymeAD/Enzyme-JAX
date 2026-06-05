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
#include "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"

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

/// Forward mode autodiff for Triton ReduceOp.
///
/// Triton reduce syntax:
///   %out = "tt.reduce"(%input) <{axis = 0 : i32}> ({
///     ^bb0(%arg0: f32, %arg1: f32):
///       %res = arith.addf %arg0, %arg1 : f32
///       tt.reduce.return %res : f32
///   }) : (tensor<64xf32>) -> f32
///
/// Key differences from StableHLO:
///   - No init_value operand (Triton infers identity from combiner)
///   - Block args are scalars (f32), not rank-0 tensors
///   - Single axis attribute, not array of dimensions
///   - Result can be scalar when reducing all dimensions
///
/// For sum reduction: d(sum(x)) = sum(dx)
/// For max/min: derivative flows through the selected element
class AutoDiffTritonReduceFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffTritonReduceFwd,
                                                triton::ReduceOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto reduce = cast<triton::ReduceOp>(orig);
    auto &body = reduce.getCombineOp().front();

    // Get the inner combiner operation (skip terminator)
    if (body.getOperations().size() != 2) {
      return orig->emitError()
             << "Triton reduce autodiff only supports single-op combiners";
    }
    Operation &innerOp = body.front();

    // For max/min reductions, use control flow handler
    if (isa<arith::MaxNumFOp, arith::MinNumFOp, arith::MaximumFOp,
            arith::MinimumFOp>(innerOp)) {
      llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
      llvm::SmallDenseSet<unsigned> resultPositionsToShadow;
      for (unsigned i = 0; i < orig->getNumOperands(); i++) {
        if (!gutils->isConstantValue(orig->getOperand(i)))
          operandPositionsToShadow.insert(i);
      }
      for (unsigned i = 0; i < orig->getNumResults(); i++) {
        if (!gutils->isConstantValue(orig->getResult(i)))
          resultPositionsToShadow.insert(i);
      }
      return mlir::enzyme::detail::controlFlowForwardHandler(
          orig, builder, gutils, operandPositionsToShadow,
          resultPositionsToShadow);
    }

    // Only support sum (addf/addi) reductions
    if (!isa<arith::AddFOp, arith::AddIOp>(innerOp)) {
      return orig->emitError()
             << "Triton reduce autodiff only supports add/max/min combiners, "
                "got: "
             << innerOp.getName();
    }

    // For sum: d(sum(x)) = sum(dx)
    // Get the primal operation
    Operation *primal = gutils->getNewFromOriginal(orig);

    // Clone the reduce with shadow inputs
    IRMapping map;
    for (Value input : reduce.getSrcs()) {
      if (gutils->isConstantValue(input)) {
        // Create zero shadow for constant inputs
        auto iface = dyn_cast<AutoDiffTypeInterface>(input.getType());
        if (!iface) {
          return orig->emitError()
                 << "Cannot differentiate reduce with non-differentiable type";
        }
        map.map(input, iface.createNullValue(builder, input.getLoc()));
      } else {
        map.map(input, gutils->invertPointerM(input, builder));
      }
    }

    // Clone entire op with remapped inputs (this correctly handles the region)
    Operation *shadowReduce = builder.clone(*orig, map);

    // Clean up: Enzyme may have added shadow block arguments to the primal.
    // For sum reduction, we don't need shadows in the combiner body.
    // Remove inverted pointers for the inner op result and block arguments.
    auto invInnerResult = gutils->invertedPointers.lookup(innerOp.getResult(0));
    gutils->invertedPointers.erase(innerOp.getResult(0));
    if (invInnerResult && invInnerResult.getDefiningOp())
      gutils->erase(invInnerResult.getDefiningOp());

    // Remove shadow block arguments from primal
    auto primalReduce = cast<triton::ReduceOp>(primal);
    auto &primalBody = primalReduce.getCombineOp().front();
    BitVector baToErase(primalBody.getNumArguments());
    for (auto ba : body.getArguments()) {
      auto invBA = gutils->invertedPointers.lookup(ba);
      if (invBA) {
        gutils->invertedPointers.erase(ba);
        if (auto invBABlock = dyn_cast<BlockArgument>(invBA)) {
          baToErase.set(invBABlock.getArgNumber());
        }
      }
    }
    primalBody.eraseArguments(baToErase);

    // Set derivatives for results
    for (auto [origRes, shadowRes] :
         llvm::zip(orig->getResults(), shadowReduce->getResults())) {
      if (!gutils->isConstantValue(origRes)) {
        gutils->setDiffe(origRes, shadowRes, builder);
      }
    }

    gutils->eraseIfUnused(orig);
    return success();
  }
};

/// Forward mode autodiff for Triton DotOp.
///
/// tt.dot computes: result = a @ b + c
/// Forward mode:    d(result) = da @ b + a @ db + dc
///
/// We chain dot operations using the accumulator:
///   shadow = dc (or zero if c is constant)
///   if a is active: shadow = dot(da, b, shadow)
///   if b is active: shadow = dot(a, db, shadow)
class AutoDiffTritonDotFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffTritonDotFwd,
                                                triton::DotOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto dot = cast<triton::DotOp>(orig);
    Value a = dot.getA();
    Value b = dot.getB();
    Value c = dot.getC();

    Value newA = gutils->getNewFromOriginal(a);
    Value newB = gutils->getNewFromOriginal(b);

    Value shadow;
    if (!gutils->isConstantValue(c)) {
      shadow = gutils->invertPointerM(c, builder);
    } else {
      auto iface = cast<AutoDiffTypeInterface>(orig->getResult(0).getType());
      shadow = iface.createNullValue(builder, orig->getLoc());
    }

    if (!gutils->isConstantValue(a)) {
      Value shadowA = gutils->invertPointerM(a, builder);
      IRMapping map;
      map.map(a, shadowA);
      map.map(b, newB);
      map.map(c, shadow);
      shadow = builder.clone(*orig, map)->getResult(0);
    }

    if (!gutils->isConstantValue(b)) {
      Value shadowB = gutils->invertPointerM(b, builder);
      IRMapping map;
      map.map(a, newA);
      map.map(b, shadowB);
      map.map(c, shadow);
      shadow = builder.clone(*orig, map)->getResult(0);
    }

    Value origResult = orig->getResult(0);
    if (!gutils->isConstantValue(origResult)) {
      gutils->setDiffe(origResult, shadow, builder);
    }

    gutils->eraseIfUnused(orig);
    return success();
  }
};

/// Control flow interface for Triton ReduceOp (needed for max/min)
class AutoDiffTritonReduceCF
    : public ControlFlowAutoDiffOpInterface::ExternalModel<
          AutoDiffTritonReduceCF, triton::ReduceOp> {
public:
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands,
                               TypeRange rettys) const {
    auto reduce = cast<triton::ReduceOp>(original);
    auto newOp = builder.create<triton::ReduceOp>(
        original->getLoc(), remappedOperands, reduce.getAxis());
    IRMapping mapping;
    reduce.getCombineOp().cloneInto(&newOp.getCombineOp(), mapping);
    return newOp;
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
    triton::DotOp::attachInterface<AutoDiffTritonDotFwd>(*context);
    triton::ReduceOp::attachInterface<AutoDiffTritonReduceFwd>(*context);
    triton::ReduceOp::attachInterface<AutoDiffTritonReduceCF>(*context);
  });
}
