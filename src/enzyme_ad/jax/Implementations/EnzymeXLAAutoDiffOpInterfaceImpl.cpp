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

struct SVDFactorizationOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          SVDFactorizationOpInterfaceReverse, SVDFactorizationOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<SVDFactorizationOp>(op);
    auto A = op.getA();
    auto U = gutils->getNewFromOriginal(op.getU());
    auto S = gutils->getNewFromOriginal(op.getS());
    auto Vt = gutils->getNewFromOriginal(op.getVt());
    auto Ubar = gutils->diffe(op.getResult(0), builder);
    auto Sbar = gutils->diffe(op.getResult(1), builder);
    auto Vtbar = gutils->diffe(op.getResult(2), builder);

    auto type_A = cast<RankedTensorType>(A.getType());
    auto type_elem = type_A.getElementType();
    auto rank = type_A.getRank();

    SmallVector<int64_t> perm;
    for (int64_t i = 0; i < rank - 2; ++i) {
      perm.push_back(i);
    }
    perm.push_back(rank - 1);
    perm.push_back(rank - 2);

    auto transpose = [&](Value x) -> Value {
      return stablehlo::TransposeOp::create(
          builder, x.getLoc(), chlo::ConjOp::create(builder, x.getLoc(), x),
          perm);
    };
    auto matmul = [&](Value a, Value b) -> Value {
      return enzymexla::GemmOp::create(builder, a.getLoc(), a, b);
    };
    auto add = [&](Value a, Value b) -> Value {
      return stablehlo::AddOp::create(builder, a.getLoc(), a, b);
    };
    auto subtract = [&](Value a, Value b) -> Value {
      return stablehlo::SubtractOp::create(builder, a.getLoc(), a, b);
    };
    auto mul = [&](Value a, Value b) -> Value {
      return stablehlo::MulOp::create(builder, a.getLoc(), a, b);
    };

    // X = UᵀŪ - ŪᵀU
    auto X = subtract(matmul(transpose(U), Ubar), matmul(transpose(Ubar), U));

    // Y = VᵀV̄ - V̄ᵀV
    // NOTE don't worry about extra transpose/conj here. they will be opt away
    auto V = transpose(Vt);
    auto Vbar = transpose(Vtbar);
    auto Y = subtract(matmul(transpose(V), Vbar), matmul(transpose(Vbar), V));

    // F₊[i,j] = 1/(s[j]-s[i]) + 1/(s[j]+s[i]) if i != j else 0
    // F₋[i,j] = 1/(s[j]-s[i]) - 1/(s[j]+s[i]) if i != j else 0
    SmallVector<int64_t> shape;
    shape.append(cast<RankedTensorType>(S.getType()).getShape());
    shape.push_back(shape.back());

    SmallVector<int64_t> broadcastDims_j;
    for (int64_t i = 0; i < rank - 2; i++)
      broadcastDims_j.push_back(i);
    broadcastDims_j.push_back(rank - 1);
    auto sj = stablehlo::BroadcastInDimOp::create(builder, op.getLoc(), shape,
                                                  S, broadcastDims_j);

    SmallVector<int64_t> broadcastDims_i;
    for (int64_t i = 0; i < rank - 1; i++)
      broadcastDims_i.push_back(i);
    auto si = stablehlo::BroadcastInDimOp::create(builder, op.getLoc(), shape,
                                                  S, broadcastDims_i);

    auto sj_sub_si = subtract(sj, si);
    auto sj_add_si = add(sj, si);

    auto one = stablehlo::ConstantOp::create(
        builder, op.getLoc(), sj_sub_si.getType(),
        cast<ElementsAttr>(makeAttr(sj_sub_si.getType(), 1)));
    auto reciprocal_sj_sub_si =
        stablehlo::DivOp::create(builder, op.getLoc(), one, sj_sub_si);
    auto reciprocal_sj_add_si =
        stablehlo::DivOp::create(builder, op.getLoc(), one, sj_add_si);

    auto iota_j =
        stablehlo::IotaOp::create(builder, op.getLoc(), shape, rank - 1);
    auto iota_i =
        stablehlo::IotaOp::create(builder, op.getLoc(), shape, rank - 2);
    auto mask =
        stablehlo::CompareOp::create(builder, op.getLoc(), iota_j, iota_i,
                                     stablehlo::ComparisonDirection::EQ);
    auto zero = stablehlo::ConstantOp::create(
        builder, op.getLoc(), RankedTensorType::get(shape, type_elem),
        cast<ElementsAttr>(
            makeAttr(RankedTensorType::get(shape, type_elem), 0)));
    auto Fplus = stablehlo::SelectOp::create(
        builder, op.getLoc(), mask, zero,
        add(reciprocal_sj_sub_si, reciprocal_sj_add_si));
    auto Fminus = stablehlo::SelectOp::create(
        builder, op.getLoc(), mask, zero,
        subtract(reciprocal_sj_sub_si, reciprocal_sj_add_si));

    // Z = F₊ .* X + F₋ .* Y
    auto Z = add(mul(Fplus, X), mul(Fminus, Y));

    // Ā = 1/2 * U * Z * Vᵀ
    auto half = stablehlo::ConstantOp::create(
        builder, op.getLoc(), U.getType(),
        cast<ElementsAttr>(makeAttr(U.getType(), 0.5)));
    auto Abar1 = matmul(matmul(mul(half, U), Z), transpose(V));

    // Ā += U * S̄ * Vᵀ
    auto dotDimsAttr = stablehlo::DotDimensionNumbersAttr::get(
        builder.getContext(), {rank - 1}, {rank - 2}, {}, {});
    auto USbar = stablehlo::DotGeneralOp::create(builder, op.getLoc(), U, Sbar,
                                                 dotDimsAttr, nullptr, nullptr);
    auto Abar2 = add(Abar1, matmul(USbar, transpose(V)));

    // Ā += (I - U * Uᵀ) * Ū * inv(S) * Vᵀ
    // TODO auto identity = ...;
    // TODO auto invS = ...;
    auto Abar3 = add(
        Abar2,
        matmul(matmul(matmul(subtract(identity, matmul(U, transpose(U))), Ubar),
                      invS),
               transpose(V)));

    // TODO Ā += U * inv(S) * V̄ᵀ * (I - V * Vᵀ)
    auto Abar4 = add(
        Abar3,
        matmul(U, matmul(invS,
                         matmul(transpose(Vbar),
                                subtract(identity, matmul(V, transpose(V)))))));

    gutils->addToDiffe(op.getOperand(), Abar4, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
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
    SVDFactorizationOp::attachInterface<SVDFactorizationOpInterfaceReverse>(
        *context);

    Pointer2MemrefOp::attachInterface<
        ViewCastOpInterfaceReverse<Pointer2MemrefOp>>(*context);
    Memref2PointerOp::attachInterface<
        ViewCastOpInterfaceReverse<Memref2PointerOp>>(*context);

    // Register batching interfaces
    JITCallOp::attachInterface<SHLOGenericBatchOpInterface<JITCallOp>>(
        *context);

    context->loadDialect<stablehlo::StablehloDialect>();
  });
}
