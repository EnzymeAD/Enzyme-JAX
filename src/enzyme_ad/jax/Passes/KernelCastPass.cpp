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
#include "mlir/IR/Matchers.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_KERNELCASTPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static Type convertType(Type t, Type targetFloat) {
  if (isa<FloatType>(t))
    return targetFloat;
  if (auto shapedType = dyn_cast<ShapedType>(t))
    return shapedType.clone(
        shapedType.getShape(),
        convertType(shapedType.getElementType(), targetFloat));
  return t;
};

// Recursively clone `src` into `dst`, converting all floating-point types
// (scalar and memref element) to `targetFloat`.
static void cloneWithTypeConversion(Region &src, Region &dst,
                                    IRMapping &mapping, MLIRContext *ctx,
                                    Type targetFloat) {
  for (Block &srcBlock : src.getBlocks()) {
    auto *dstBlock = new Block();
    dst.push_back(dstBlock);
    mapping.map(&srcBlock, dstBlock);
    for (BlockArgument srcArg : srcBlock.getArguments()) {
      BlockArgument dstArg = dstBlock->addArgument(
          convertType(srcArg.getType(), targetFloat), srcArg.getLoc());
      mapping.map(srcArg, dstArg);
    }
  }

  for (Block &srcBlock : src.getBlocks()) {
    Block *dstBlock = mapping.lookup(&srcBlock);

    for (Operation &op : srcBlock.getOperations()) {
      APFloat apf(0.0);
      if (op.getNumResults() == 1 &&
          isa<FloatType>(op.getResult(0).getType()) &&
          matchPattern(op.getResult(0), m_ConstantFloat(&apf))) {
        Operation *newConst = op.clone(mapping);
        dstBlock->push_back(newConst);
        auto constType = cast<FloatType>(op.getResult(0).getType());
        if (constType != targetFloat) {
          OpBuilder b(ctx);
          b.setInsertionPointAfter(newConst);
          Value cast;
          if (constType.getIntOrFloatBitWidth() <
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

      SmallVector<Type> resultTypes =
          llvm::map_to_vector(op.getResultTypes(), [&targetFloat](Type ty) {
            return convertType(ty, targetFloat);
          });
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
        cloneWithTypeConversion(srcReg, dstReg, mapping, ctx, targetFloat);

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

  auto oldFT = func.getFunctionType();
  SmallVector<Type> newInputs, newOutputs;
  for (Type t : oldFT.getInputs())
    newInputs.push_back(convertType(t, targetFloat));
  for (Type t : oldFT.getResults())
    newOutputs.push_back(convertType(t, targetFloat));

  Region oldBody;
  oldBody.takeBody(func.getBody());

  func.setFunctionType(FunctionType::get(ctx, newInputs, newOutputs));

  IRMapping mapping;
  cloneWithTypeConversion(oldBody, func.getBody(), mapping, ctx, targetFloat);
}

namespace {

struct KernelCastPass
    : public enzyme::impl::KernelCastPassBase<KernelCastPass> {
  using KernelCastPassBase::KernelCastPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto attr = func->getAttrOfType<TypeAttr>("enzymexla.float_type");
    if (!attr)
      return;

    Type targetFloat = attr.getValue();
    if (!isa<FloatType>(targetFloat)) {
      func.emitError("enzymexla.float_type: expected a float type, got '")
          << targetFloat << "'";
      return signalPassFailure();
    }

    castKernelTypes(func, targetFloat);
    func->removeAttr("enzymexla.float_type");
  }
};

} // namespace
