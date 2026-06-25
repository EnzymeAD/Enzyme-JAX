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

struct QRFactorizationOpFwdDerivative
    : public AutoDiffOpInterface::ExternalModel<QRFactorizationOpFwdDerivative,
                                                enzymexla::QRFactorizationOp> {
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<enzymexla::QRFactorizationOp>(orig);
    gutils->eraseIfUnused(op);
    if (gutils->isConstantInstruction(op))
      return success();

    if (gutils->isConstantValue(op.getInput())) {
      gutils->setDiffe(op.getQ(), nullptr, builder);
      gutils->setDiffe(op.getR(), nullptr, builder);
      return success();
    }

    auto A = op.getInput();
    auto dA = gutils->invertPointerM(op.getInput(), builder);
    auto Q = gutils->getNewFromOriginal(op.getQ());
    auto R = gutils->getNewFromOriginal(op.getR());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(op);

    auto type_a = cast<RankedTensorType>(A.getType());
    auto rank = type_a.getRank();

    // B = dA * R^-1 = dA / R
    auto B = enzymexla::TrsmOp::create(builder, op.getLoc(), R, dA,
                                       enzymexla::LapackSide::right,
                                       enzymexla::LapackUplo::U);

    // E = Q^dag * B
    SmallVector<int64_t> perm;
    for (int64_t i = 0; i < rank - 2; i++)
      perm.push_back(i);
    perm.push_back(rank - 1);
    perm.push_back(rank - 2);

    auto Qconj = chlo::ConjOp::create(builder, op.getLoc(), Q);
    auto Qdag =
        stablehlo::TransposeOp::create(builder, op.getLoc(), Qconj, perm);

    auto E = enzymexla::GemmOp::create(builder, op.getLoc(), Qdag, B);

    // Psi = (U .* E) + (L^hat .* E)^dag = U .* (E + E^dag)
    // NOTE we avoid doing the lower-to-upper triangular part copying by using
    // `TrmmOp` instead of `GemmOp` on its uses
    auto Econj = chlo::ConjOp::create(builder, op.getLoc(), E);
    auto Edag =
        stablehlo::TransposeOp::create(builder, op.getLoc(), Econj, perm);
    auto Psi = stablehlo::AddOp::create(builder, op.getLoc(), E, Edag);

    // dQ = B - Q * Psi
    auto QxPsi = enzymexla::TrmmOp::create(builder, op.getLoc(), Psi, Q,
                                           enzymexla::LapackSide::right,
                                           enzymexla::LapackUplo::U);
    auto dQ = stablehlo::SubtractOp::create(builder, op.getLoc(), B, QxPsi);

    // dR = Psi * R
    // NOTE we use TrmmOp on Psi to avoid issues of not lower triangular part,
    // as R is truly upper triangular
    auto dR = enzymexla::TrmmOp::create(builder, op.getLoc(), Psi, R,
                                        enzymexla::LapackSide::left,
                                        enzymexla::LapackUplo::U);

    gutils->setDiffe(op.getQ(), dQ.getResult(), builder);
    gutils->setDiffe(op.getR(), dR.getResult(), builder);

    return success();
  }
};

struct QRFactorizationOpRevDerivative
    : public ReverseAutoDiffOpInterface::ExternalModel<
          QRFactorizationOpRevDerivative, QRFactorizationOp> {
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<QRFactorizationOp>(orig);
    auto Q = gutils->getNewFromOriginal(op.getQ());
    auto R = gutils->getNewFromOriginal(op.getR());
    auto Qbar = gutils->diffe(op.getResult(0), builder);
    auto Rbar = gutils->diffe(op.getResult(1), builder);

    // create M = R̄ / R^dag - Q̄^dag * Q
    auto RbarDivR = enzymexla::TrsmOp::create(
        builder, op.getLoc(), R, Rbar,
        /*side=*/enzymexla::LapackSide::right,
        /*uplo=*/enzymexla::LapackUplo::U,
        /*transa=*/enzymexla::LapackTranspose::adjoint,
        /*unit_diagonal=*/false);

    auto QbarDagMulQ = enzymexla::GemmOp::create(
        builder, op.getLoc(), Qbar, Q, enzymexla::LapackTranspose::adjoint);

    auto M = stablehlo::SubtractOp::create(builder, op.getLoc(), RbarDivR,
                                           QbarDagMulQ);

    // X = Q̄ + Q * copyltu(M)
    // do not copy triangular lower to upper part... just do symm
    // TODO check whether this is numerically correct or we need to copy
    auto QMulM =
        enzymexla::SymmOp::create(builder, op.getLoc(), M, Q,
                                  /*side=*/enzymexla::LapackSide::right,
                                  /*uplo=*/enzymexla::LapackUplo::L);

    auto X = stablehlo::AddOp::create(builder, op.getLoc(), Qbar, QMulM);

    // Ā = X / R^dag
    auto Abar = enzymexla::TrsmOp::create(
        builder, op.getLoc(), R, X,
        /*side=*/enzymexla::LapackSide::right,
        /*uplo=*/enzymexla::LapackUplo::U,
        /*transa=*/enzymexla::LapackTranspose::adjoint,
        /*unit_diagonal=*/false);

    gutils->addToDiffe(op.getOperand(), Abar, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op->getResult(0)))
      return {};
    auto neededArgs = cachedArguments(op, gutils);
    SmallVector<Value> toret;
    OpBuilder builder(gutils->getNewFromOriginal(op));
    for (auto en : llvm::enumerate(neededArgs))
      if (en.value()) {
        Value cache = gutils->initAndPushCache(
            gutils->getNewFromOriginal(op->getOperand(en.index())), builder);
        toret.push_back(cache);
      }
    return toret;
  }

  SmallVector<bool> cachedArguments(Operation *op,
                                    MGradientUtilsReverse *gutils) const {
    SmallVector<bool> toret(op->getNumOperands(), false);
    return toret;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
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
    QRFactorizationOp::attachInterface<QRFactorizationOpFwdDerivative>(
        *context);
    QRFactorizationOp::attachInterface<QRFactorizationOpRevDerivative>(
        *context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
