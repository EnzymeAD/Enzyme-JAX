//===- KernelCastPass.cpp - Cast GPU kernel floating point type -----------===//
//
// This file implements a pass that rewrites a GPU kernel compiled for one
// floating-point type to use a different floating-point type.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_KERNELCASTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static Type parseFloatTypeAttr(StringRef name, MLIRContext *ctx) {
  return StringSwitch<Type>(name)
      .Case("f16", Float16Type::get(ctx))
      .Case("f32", Float32Type::get(ctx))
      .Case("f64", Float64Type::get(ctx))
      .Case("bf16", BFloat16Type::get(ctx))
      .Case("f8E4M3FN", Float8E4M3FNType::get(ctx))
      .Case("f8E5M2", Float8E5M2Type::get(ctx))
      .Case("f8E4M3FNUZ", Float8E4M3FNUZType::get(ctx))
      .Case("f8E5M2FNUZ", Float8E5M2FNUZType::get(ctx))
      .Default(Type{});
}

// Recursively clone `src` into `dst`, converting all floating-point types
// (scalar and memref element) to `targetFloat`.
static void cloneWithTypeConversion(Region &src, Region &dst,
                                    IRMapping &mapping, MLIRContext *ctx,
                                    function_ref<Type(Type)> convertType,
                                    Type targetFloat) {
  for (Block &srcBlock : src.getBlocks()) {
    auto *dstBlock = new Block();
    dst.push_back(dstBlock);
    mapping.map(&srcBlock, dstBlock);
    for (BlockArgument srcArg : srcBlock.getArguments()) {
      BlockArgument dstArg =
          dstBlock->addArgument(convertType(srcArg.getType()), srcArg.getLoc());
      mapping.map(srcArg, dstArg);
    }
  }

  for (Block &srcBlock : src.getBlocks()) {
    Block *dstBlock = mapping.lookup(&srcBlock);

    for (Operation &op : srcBlock.getOperations()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto fa = dyn_cast<FloatAttr>(constOp.getValue());
            fa && isa<FloatType>(fa.getType())) {
          // Full clone: result type matches attribute, mapper updated.
          Operation *newConst = op.clone(mapping);
          dstBlock->push_back(newConst);
          if (constOp.getType() != targetFloat) {
            OpBuilder b(ctx);
            b.setInsertionPointAfter(newConst);
            Value cast;
            if (constOp.getType().getIntOrFloatBitWidth() <
                targetFloat.getIntOrFloatBitWidth())
              cast = arith::ExtFOp::create(b, newConst->getLoc(), targetFloat,
                                           newConst->getResult(0));
            else
              cast = arith::TruncFOp::create(b, newConst->getLoc(), targetFloat,
                                             newConst->getResult(0));
            // Override the mapper entry set by clone() to point to the cast.
            mapping.map(op.getResult(0), cast);
          }
          continue;
        }
      }

      SmallVector<Type> resultTypes =
          llvm::map_to_vector(op.getResultTypes(), convertType);
      Operation *newOp = op.clone(mapping, Operation::CloneOptions()
                                               .withResultTypes(resultTypes)
                                               .cloneRegions(false)
                                               .cloneOperands(true));
      dstBlock->push_back(newOp);
      for (auto [srcRes, dstRes] :
           llvm::zip(op.getResults(), newOp->getResults()))
        mapping.map(srcRes, dstRes);

      for (auto [srcReg, dstReg] :
           llvm::zip(op.getRegions(), newOp->getRegions()))
        cloneWithTypeConversion(srcReg, dstReg, mapping, ctx, convertType,
                                targetFloat);

      if (isa<arith::ExtFOp, arith::TruncFOp>(newOp) &&
          newOp->getOperand(0).getType() == newOp->getResult(0).getType()) {
        mapping.map(op.getResult(0), newOp->getOperand(0));
        newOp->erase();
      }
    }
  }
}

void castKernelTypes(func::FuncOp func, mlir::Type targetFloat) {
  MLIRContext *ctx = func.getContext();

  auto convertType = [&](Type t) -> Type {
    if (isa<FloatType>(t))
      return targetFloat;
    if (auto mt = dyn_cast<MemRefType>(t))
      if (isa<FloatType>(mt.getElementType()))
        return MemRefType::get(mt.getShape(), targetFloat, mt.getLayout(),
                               mt.getMemorySpace());
    return t;
  };

  auto oldFT = func.getFunctionType();
  SmallVector<Type> newInputs, newOutputs;
  for (Type t : oldFT.getInputs())
    newInputs.push_back(convertType(t));
  for (Type t : oldFT.getResults())
    newOutputs.push_back(convertType(t));

  Region oldBody;
  oldBody.takeBody(func.getBody());

  func.setFunctionType(FunctionType::get(ctx, newInputs, newOutputs));

  IRMapping mapping;
  cloneWithTypeConversion(oldBody, func.getBody(), mapping, ctx, convertType,
                          targetFloat);
}

namespace {

struct KernelCastPass
    : public enzyme::impl::KernelCastPassBase<KernelCastPass> {
  using KernelCastPassBase::KernelCastPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto attr = func->getAttrOfType<StringAttr>("enzymexla.float_type");
    if (!attr)
      return;

    Type targetFloat = parseFloatTypeAttr(attr.getValue(), func.getContext());
    if (!targetFloat) {
      func.emitError("enzymexla.float_type: unrecognised float type '")
          << attr.getValue() << "'";
      return signalPassFailure();
    }

    castKernelTypes(func, targetFloat);
    func->removeAttr("enzymexla.float_type");
  }
};

} // namespace
