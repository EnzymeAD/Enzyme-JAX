/* Copyright 2024 The StableHLO Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {

#define GEN_PASS_DEF_ENZYMEREFINEARGUMENTSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"

namespace {

const char* kCustomCallOperandBarrierTarget = "stablehlo.shape_refinement_operand_wrapper";

stablehlo::CustomCallOp makeShapeRefinementOperandWrapper(OpBuilder& builder,
                                               Value operand,
                                               RankedTensorType refinedType) {
  auto constant = stablehlo::ConstantOp::create(
      builder, operand.getLoc(),
      builder.getI64TensorAttr(refinedType.getShape()));
  return stablehlo::CustomCallOp::create(
      builder, operand.getLoc(), operand.getType(),
      ValueRange{operand, constant},
      llvm::SmallVector<NamedAttribute>{
          builder.getNamedAttr(
              "call_target_name",
              builder.getStringAttr(kCustomCallOperandBarrierTarget)),
          builder.getNamedAttr("indices_of_shape_operands",
                               builder.getI64TensorAttr({1}))});
}

ParseResult parseRefinedTypes(ModuleOp module,
                              ArrayRef<std::string> shapeString,
                              SmallVector<Type>& refinedTypes) {
  MLIRContext* context = module.getContext();
  for (const auto& shape : shapeString) {
    Type type = mlir::parseType(shape, context);
    if (!type) return module->emitOpError("Invalid type string: ") << shape;
    refinedTypes.push_back(type);
  }
  return success();
}

struct EnzymeRefineArgumentsPass
    : public impl::EnzymeRefineArgumentsPassBase<EnzymeRefineArgumentsPass> {
  using EnzymeRefineArgumentsPassBase::EnzymeRefineArgumentsPassBase;

  EnzymeRefineArgumentsPass(TypeRange refinedTypes_) {
    refinedTypes = llvm::to_vector(refinedTypes_);
  }

  void runOnOperation() override {
    auto module = getOperation();
    
    // Parse if string specified as option
    if (!refinedTypesOption.empty() &&
        failed(parseRefinedTypes(module, refinedTypesOption,
                                 refinedTypes))) {
      return signalPassFailure();
    }

    func::FuncOp func;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main") {
        func = op;
        break;
      }
    }
    if (!func && !module.getOps<func::FuncOp>().empty()) {
      func = *module.getOps<func::FuncOp>().begin();
    }
    if (!func) return;

    if (failed(refineArguments(func, refinedTypes))) return signalPassFailure();
  }

 private:
  SmallVector<Type> refinedTypes;

  LogicalResult refineArguments(func::FuncOp func, TypeRange refinedTypes);
};

LogicalResult refinementError(Operation* op, int64_t idx, Type argType,
                              Type refinedType, StringRef msg) {
  return op->emitOpError() << "invalid refinement for argument " << idx
                           << ", refinement " << msg << " in "
                           << mlir::debugString(argType) << " -> "
                           << mlir::debugString(refinedType);
}

LogicalResult validateRefinedTypes(Operation* op, TypeRange argTypes,
                                   TypeRange refinedTypes) {
  if (argTypes.size() != refinedTypes.size()) {
    return op->emitOpError(
               "number of refinements must match number of op operands ")
           << refinedTypes.size() << " vs " << argTypes.size();
  }

  for (size_t i = 0; i < argTypes.size(); ++i) {
    Type type = argTypes[i];
    Type refinedType = refinedTypes[i];

    if (type == refinedType) continue;

    auto tensorType = dyn_cast<TensorType>(type);
    auto refinedTensorType = dyn_cast<TensorType>(refinedType);
    if (!tensorType || !refinedTensorType) {
      return refinementError(op, i, type, refinedType, "must be a tensor");
    }

    if (tensorType.getElementType() != refinedTensorType.getElementType()) {
      if (tensorType.getElementType().isInteger(8) || refinedTensorType.getElementType().isInteger(8)) {
        // Allowed for bitcast handling
      } else {
        return refinementError(op, i, type, refinedType,
                               "element types must match or one must be i8");
      }
    }

    if (isa<UnrankedTensorType>(refinedType)) {
      return refinementError(op, i, type, refinedType, "must be ranked");
    }

    if (!tensorType.hasRank()) continue;

    if (tensorType.getElementType() != refinedTensorType.getElementType()) {
       // Allow rank mismatch for bitcast
    } else if (tensorType.getRank() != refinedTensorType.getRank()) {
      return refinementError(op, i, type, refinedType,
                             "rank must match operand rank");
    }

    if (tensorType.getElementType() == refinedTensorType.getElementType()) {
      for (auto [dimSize, refinedDimSize] :
           llvm::zip(tensorType.getShape(), refinedTensorType.getShape())) {
        if (!ShapedType::isDynamic(dimSize) && dimSize != refinedDimSize) {
          return refinementError(
              op, i, type, refinedType,
              "dimension sizes must match for static dimensions");
        }
      }
    }
  }
  return success();
}

LogicalResult EnzymeRefineArgumentsPass::refineArguments(func::FuncOp func, TypeRange refinedTypes) {
  if (failed(validateRefinedTypes(func, func.getArgumentTypes(), refinedTypes)))
    return failure();

  Region& body = func.getBody();
  OpBuilder builder(body);
  builder.setInsertionPointToStart(&body.front());

  SmallVector<stablehlo::CustomCallOp> wrappers(func.getNumArguments(), nullptr);

  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    BlockArgument arg = body.getArgument(i);
    Type argType = arg.getType();
    Type refinedType = refinedTypes[i];
    if (argType != refinedType) {
      auto rankedRefinedType = cast<RankedTensorType>(refinedType);
      auto customCall =
          makeShapeRefinementOperandWrapper(builder, arg, rankedRefinedType);
      wrappers[i] = customCall;
      auto callResult = customCall.getResult(0);
      arg.replaceAllUsesExcept(callResult, customCall);
    }
  }

  // Update signature
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedTypes[i]);
  }
  // Update return types to match argument types as requested by user.
  func.setType(builder.getFunctionType(refinedTypes, refinedTypes));

  // Fix up wrappers if element types mismatch
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    stablehlo::CustomCallOp wrapper = wrappers[i];
    if (!wrapper) continue;

    Type originalType = wrapper.getResult(0).getType();
    Type refinedType = refinedTypes[i];

    auto originalTensorType = cast<RankedTensorType>(originalType);
    auto refinedTensorType = cast<RankedTensorType>(refinedType);

    if (originalTensorType.getElementType() != refinedTensorType.getElementType()) {
      builder.setInsertionPoint(wrapper); // Insert before wrapper

      Type srcElType = refinedTensorType.getElementType();
      Type dstElType = originalTensorType.getElementType();
      
      Value input = wrapper.getOperand(0);
      
      if (srcElType.isInteger(8) && !dstElType.isInteger(8)) {
        int64_t dstSize = dstElType.getIntOrFloatBitWidth() / 8;
        SmallVector<int64_t> reshapeShape = llvm::to_vector(refinedTensorType.getShape());
        assert(!reshapeShape.empty() && reshapeShape.back() % dstSize == 0);
        reshapeShape.back() /= dstSize;
        reshapeShape.push_back(dstSize);
        
        auto reshapeType = RankedTensorType::get(reshapeShape, srcElType);
        Value reshapedInput = builder.create<stablehlo::ReshapeOp>(wrapper.getLoc(), reshapeType, input);
        
        SmallVector<int64_t> bitcastShape = llvm::to_vector(refinedTensorType.getShape());
        bitcastShape.back() /= dstSize;
        auto bitcastType = RankedTensorType::get(bitcastShape, dstElType);
        
        Value convertedVal = builder.create<stablehlo::BitcastConvertOp>(wrapper.getLoc(), bitcastType, reshapedInput);
        
        // Update wrapper to use convertedVal
        wrapper.setOperand(0, convertedVal);
        
        // Update shape operand of wrapper
        auto newShapeAttr = builder.getI64TensorAttr(bitcastType.getShape());
        auto newShapeConst = builder.create<stablehlo::ConstantOp>(wrapper.getLoc(), newShapeAttr);
        wrapper.setOperand(1, newShapeConst);
      }
    }
  }

  // Inverse conversion for return operands
  func.walk([&](func::ReturnOp returnOp) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(returnOp);
    
    SmallVector<Value> newReturnOperands;
    for (auto [idx, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (idx < refinedTypes.size()) {
        Type refinedType = refinedTypes[idx];
        Type originalType = operand.getType();
        
        auto originalTensorType = cast<RankedTensorType>(originalType);
        auto refinedTensorType = cast<RankedTensorType>(refinedType);
        
        if (originalTensorType.getElementType() != refinedTensorType.getElementType()) {
          Type srcElType = originalTensorType.getElementType();
          Type dstElType = refinedTensorType.getElementType();
          
          Value val = operand;
          
          if (!srcElType.isInteger(8) && dstElType.isInteger(8)) {
             int64_t srcSize = srcElType.getIntOrFloatBitWidth() / 8;
             
             // Assume we can reshape from dynamic to static if we know the static shape!
             SmallVector<int64_t> staticShape = llvm::to_vector(refinedTensorType.getShape());
             assert(!staticShape.empty() && staticShape.back() % srcSize == 0);
             staticShape.back() /= srcSize;
             
             auto staticType = RankedTensorType::get(staticShape, srcElType);
             val = builder.create<stablehlo::ReshapeOp>(returnOp.getLoc(), staticType, val);
             
             SmallVector<int64_t> bitcastShape = llvm::to_vector(refinedTensorType.getShape());
             bitcastShape.back() /= srcSize;
             bitcastShape.push_back(srcSize);
             
             auto bitcastType = RankedTensorType::get(bitcastShape, dstElType);
             val = builder.create<stablehlo::BitcastConvertOp>(returnOp.getLoc(), bitcastType, val);
             
             val = builder.create<stablehlo::ReshapeOp>(returnOp.getLoc(), refinedType, val);
             
             newReturnOperands.push_back(val);
          } else {
             newReturnOperands.push_back(operand);
          }
        } else {
          newReturnOperands.push_back(operand);
        }
      } else {
        newReturnOperands.push_back(operand);
      }
    }
    returnOp.getOperation()->setOperands(newReturnOperands);
  });

  return success();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEnzymeRefineArgumentsPass(
    TypeRange refinedTypes) {
  return std::make_unique<EnzymeRefineArgumentsPass>(refinedTypes);
}

} // namespace enzyme
} // namespace mlir
