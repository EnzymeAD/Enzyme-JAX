//===- CHLOAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
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
#include "src/enzyme_ad/jax/Implementations/SHLOGenericBatchOpInterface.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;
using namespace mlir::stablehlo;

static int64_t to_i64(int64_t x) { return x; }
static int64_t to_i64(llvm::APInt x) { return x.getSExtValue(); }

static mlir::DenseI64ArrayAttr getI64Attr(OpBuilder &builder,
                                          llvm::ArrayRef<int64_t> vals) {
  return builder.getDenseI64ArrayAttr(vals);
}

namespace {
#include "src/enzyme_ad/jax/Implementations/EnzymeXLADerivatives.inc"

struct GPUWrapperOpEnzymeOpsRemover
    : public EnzymeOpsRemoverOpInterface::ExternalModel<
          GPUWrapperOpEnzymeOpsRemover, GPUWrapperOp> {

  LogicalResult removeEnzymeOps(Operation *op,
                                PatternRewriter &rewriter) const {

    auto wrapOp = cast<GPUWrapperOp>(op);

    // Gradients whose value is set in either branches.
    llvm::SetVector<Value> gradients;

    // We assume pushes are exclusive.
    llvm::MapVector<Value, CacheInfo> pushedCaches;

    // Grad to value
    IRMapping mapping;

    removalBlockExplore(&wrapOp.getRegion().front(), mapping, rewriter,
                        gradients, pushedCaches);

    if (gradients.empty() && pushedCaches.empty())
      return success();

    if (gradients.size())
      return failure();

    if (pushedCaches.size())
      return failure();

    // TODO need to convert to gpu allocations and conversion/copy

    /*
    for (auto grad : gradients) {
      auto trueValue = trueMapping.lookupOrNull(grad);
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));

    }
    */

    /*
    for (auto &[pushedValue, info] : pushedCaches) {
      trueTerm->insertOperands(trueTerm->getNumOperands(),
                               ValueRange(trueValue));
    }
    */

    /*
    size_t idx = ifOp->getNumResults();
    for (auto grad : gradients) {
      enzyme::SetOp::create(rewriter, grad.getLoc(), grad,
                                     newIf->getResult(idx));
      idx++;
    }

    for (auto &[pushedValue, info] : pushedCaches) {
      enzyme::PushOp::create(rewriter, info.pushOp->getLoc(),
                                      info.initOp.getResult(),
                                      newIf->getResult(idx));
      rewriter.eraseOp(info.pushOp);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(info.popOp->getParentOp());

      auto newPop = enzyme::PopOp::create(rewriter,
          info.popOp->getLoc(), info.popOp.getResult().getType(),
          info.popOp.getCache());
      rewriter.replaceAllUsesWith(info.popOp.getResult(), newPop);
      rewriter.eraseOp(info.popOp);

      idx++;
    }

    rewriter.replaceAllUsesWith(
        ifOp->getResults(),
        newIf->getResults().slice(0, ifOp->getNumResults()));
    rewriter.eraseOp(ifOp);
    */
    return success();
  }
};

// Reverse-mode adjoint for pure view-cast ops (Pointer2Memref /
// Memref2Pointer). We only need to materialize the corresponding shadow view
// in the augmented primal and register it via setInvertedPointer so downstream
// memref.load / memref.store consumers can locate it through invertPointerM.
// This mirrors the pattern used by LLVM::GEPOp and memref::SubViewOp.
template <typename OpTy>
struct ViewCastOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ViewCastOpInterfaceReverse<OpTy>, OpTy> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto castOp = cast<OpTy>(op);
    auto newCastOp = cast<OpTy>(gutils->getNewFromOriginal(op));
    Value source = castOp.getSource();
    if (!gutils->isConstantValue(source)) {
      Value sourceShadow = gutils->invertPointerM(source, builder);
      auto shadowCast = cast<OpTy>(builder.clone(*newCastOp));
      shadowCast.getSourceMutable().assign(sourceShadow);
      gutils->setInvertedPointer(castOp.getResult(), shadowCast->getResult(0));
    }
  }
};

struct GPUWrapperOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GPUWrapperOpInterfaceReverse, GPUWrapperOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {

    auto wrapOp = cast<GPUWrapperOp>(op);

    SmallVector<Value> operands;
    for (auto v : caches) {
      operands.push_back(gutils->popCache(v, builder));
    }

    auto repFor = GPUWrapperOp::create(builder, wrapOp.getLoc(), operands);

    bool valid = true;
    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), repFor->getRegions())) {
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        OpBuilder bodyBuilder(&revBB, revBB.end());

        // Create implicit terminator if not present (when num results > 0)
        if (revBB.empty()) {
          YieldOp::create(bodyBuilder, repFor->getLoc(), ValueRange());
        }
        bodyBuilder.setInsertionPoint(revBB.getTerminator());

        auto first = oBB.rbegin();
        first++; // skip terminator

        auto last = oBB.rend();

        for (auto it = first; it != last; ++it) {
          Operation *op = &*it;
          valid &=
              gutils->Logic.visitChild(op, bodyBuilder, gutils).succeeded();
        }
      }
    }

    return success(valid);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto wrapOp = cast<GPUWrapperOp>(op);

    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    SmallVector<Value> caches;

    for (auto val : wrapOp.getBlockDims()) {
      Value cacheLB = gutils->initAndPushCache(gutils->getNewFromOriginal(val),
                                               cacheBuilder);
      caches.push_back(cacheLB);
    }

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct QRFactorizationOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          QRFactorizationOpInterfaceReverse, QRFactorizationOp> {
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<QRFactorizationOp>(orig);
    auto Q = op.getQ();
    auto R = op.getR();
    auto Qbar = gutils->getNewFromOriginal(Q);
    auto Rbar = gutils->getNewFromOriginal(R);

    // create M = R̄ / R^dag - Q̄^dag * Q
    auto alpha = stablehlo::ConstantOp::create(op.getLoc(),
                                               builder.getF64FloatAttr(1.0));
    auto RbarDivR = enzymexla::TrsmOp::create(
        op.getLoc(), alpha, R, Rbar,
        /*side=*/enzymexla::LapackSide::right,
        /*uplo=*/enzymexla::LapackUplo::upper,
        /*transa=*/enzymexla::LapackTranspose::adjoint,
        /*unit_diagonal=*/false);

    SmallVector<int64_t> perm;
    auto rank_q = R.getType().cast<ShapedType>().getRank();
    for (auto i = 0; i < rank_q; i++) {
      perm.push_back(i);
    }
    perm[rank_q - 1] = rank_q - 2;
    perm[rank_q - 2] = rank_q - 1;
    auto QbarConj = chlo::ConjugateOp::create(op.getLoc(), Qbar);
    auto QbarDag = stablehlo::TransposeOp::create(
        op.getLoc(), QbarConj.getType(), QbarConj.getResult(),
        getI64Attr(builder, perm));

    // TODO add contracting dims and batching dims
    auto QbarDagMulQ =
        stablehlo::DotGeneralOp::create(op.getLoc(), QbarDag.getResult(), Q);

    auto M = stablehlo::SubOp::create(op.getLoc(), RbarDivR.getResult(),
                                      QbarDagMulQ.getResult());

    // X = Q̄ + Q * copyltu(M)
    // do not copy triangular lower to upper part... just do symm
    auto beta = stablehlo::ConstantOp::create(op.getLoc(),
                                              builder.getF64FloatAttr(0.0));
    auto C = stablehlo::ConstantOp::create(op.getLoc(),
                                           builder.getZeroAttr(Q.getType()));
    auto QMulM =
        enzymexla::SymmOp::create(op.getLoc(), M, Q, C, alpha, beta,
                                  /*side=*/enzymexla::LapackSide::right,
                                  /*uplo=*/enzymexla::LapackUplo::lower);

    auto QbarAddQMulM =
        stablehlo::AddOp::create(op.getLoc(), Qbar, QMulM.getResult());

    // X / R^dag
    auto Abar = enzymexla::TrsmOp::create(
        op.getLoc(), alpha, R, QbarAddQMulM.getResult(),
        /*side=*/enzymexla::LapackSide::right,
        /*uplo=*/enzymexla::LapackUplo::upper,
        /*transa=*/enzymexla::LapackTranspose::adjoint,
        /*unit_diagonal=*/false);
    gutils->setOutputGradient(Abar.getResult(), 0);
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }
};

} // namespace

void mlir::enzyme::registerEnzymeXLADialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, EnzymeXLADialect *) {
    registerInterfaces(context);
    GPUWrapperOp::attachInterface<GPUWrapperOpInterfaceReverse>(*context);
    GPUWrapperOp::attachInterface<GPUWrapperOpEnzymeOpsRemover>(*context);

    Pointer2MemrefOp::attachInterface<
        ViewCastOpInterfaceReverse<Pointer2MemrefOp>>(*context);
    Memref2PointerOp::attachInterface<
        ViewCastOpInterfaceReverse<Memref2PointerOp>>(*context);

    // linalg diff interfaces
    QRFactorizationOp::attachInterface<QRFactorizationOpInterfaceReverse>(
        *context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
