
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/Region.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROPAGATECONSTANTBOUNDSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

#define DEBUG_TYPE "delinearize-indexing"

namespace {

struct AccessInfo {
  affine::MemRefAccess access;
  AffineMap map;

  AccessInfo(affine::MemRefAccess access) : access(access) {}
};

LogicalResult getOpIndexSet(Operation *op,
                            affine::FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  affine::getEnclosingAffineOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

LogicalResult
reshapeMemref(Value memref,
              std::function<void(RewriterBase &, unsigned, unsigned)>
                  rewriteMemrefCallback) {

  MLIRContext *ctx = memref.getContext();
  using namespace mlir::affine;

  SmallVector<AccessInfo> accesses;
  bool foundAllUses = true;
  for (auto user : memref.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(user)) {
      accesses.push_back({MemRefAccess(load)});
    } else if (auto store = dyn_cast<AffineStoreOp>(user)) {
      accesses.push_back({MemRefAccess(store)});
    } else {
      foundAllUses = false;
      break;
    }
  }

  if (!foundAllUses)
    return failure();

  llvm::SmallSetVector<int64_t, 16> constantsSet;
  llvm::SmallSetVector<Value, 16> symbols;
  for (auto ainfo : accesses) {
    auto access = ainfo.access;
    AffineValueMap valueMap;
    access.getAccessMap(&valueMap);
    AffineMap map = valueMap.getAffineMap();
    for (AffineExpr result : map.getResults()) {
      result.walk([&](AffineExpr expr) {
        if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
          if (cst.getValue() > 0)
            constantsSet.insert(cst.getValue());
          else
            constantsSet.insert(-cst.getValue());
        }
        if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
          symbols.insert(
              valueMap.getOperand(valueMap.getNumDims() + sym.getPosition()));
      });
    }
  }

  SmallVector<int64_t> constants(constantsSet.begin(), constantsSet.end());
  llvm::sort(constants);

  auto checkShapeCandidate = [&](int64_t cst, unsigned resultId) {
    LLVM_DEBUG(llvm::dbgs() << "Checking shape candidate " << cst << " at "
                            << resultId << "\n");
    bool allValid = true;
    for (auto &ainfo : accesses) {
      auto access = ainfo.access;
      AffineValueMap accessAvm;
      access.getAccessMap(&accessAvm);
      AffineExpr expr = accessAvm.getResult(resultId);
      LLVM_DEBUG(llvm::dbgs() << "For access " << *access.opInst
                              << " with expr " << expr << "\n");
      auto mod = expr % cst;
      auto floor = expr.floorDiv(cst);
      LLVM_DEBUG(llvm::dbgs() << "Mod: " << mod << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Floor: " << floor << "\n");

      DenseMap<AffineExpr, AffineExpr> toReplace;
      auto res = mod.walk([&](AffineExpr expr) {
        AffineBinaryOpExpr binexpr = dyn_cast<AffineBinaryOpExpr>(expr);
        if (!binexpr)
          return WalkResult::advance();
        if (binexpr.getKind() != AffineExprKind::Mod)
          return WalkResult::advance();
        if (binexpr.getRHS() != getAffineConstantExpr(cst, ctx))
          return WalkResult::advance();
        auto lhs = binexpr.getLHS();
        auto lhsMap = AffineMap::get(accessAvm.getNumDims(),
                                     accessAvm.getNumSymbols(), lhs);
        AffineValueMap lhsAvm(lhsMap, accessAvm.getOperands());
        lhsAvm.composeSimplifyAndCanonicalize();
        LLVM_DEBUG(llvm::dbgs()
                   << "Nested mod: " << lhsAvm.getAffineMap() << "\n");
        affine::FlatAffineValueConstraints domain;
        if (failed(getOpIndexSet(access.opInst, &domain)))
          return WalkResult::interrupt();
        if (failed(domain.composeMap(&lhsAvm)))
          return WalkResult::interrupt();
        LLVM_DEBUG(llvm::dbgs() << "Composed domain: ");
        LLVM_DEBUG(domain.dump());
        domain.setDimSymbolSeparation(domain.getNumDimAndSymbolVars() - 1);
        domain.simplify();
        SmallVector<Value, 4> vars;
        domain.getValues(domain.getNumDimVars(),
                         domain.getNumDimAndSymbolVars(), &vars);
        for (Value var : vars)
          if ((affine::isAffineInductionVar(var)))
            domain.projectOut(var);
        domain.constantFoldVarRange(
            /*pos=*/1,
            /*num=*/domain.getNumDimAndSymbolVars() - 1);
        domain.removeTrivialRedundancy();
        auto bounds = domain.getLowerAndUpperBound(
            0, 0, 1, domain.getNumDimVars(), {}, ctx);
        auto lbExpr = bounds.first.getResult(0);
        auto ubExpr = bounds.second.getResult(0);
        LLVM_DEBUG(llvm::dbgs() << "LB: " << lbExpr << "\n");
        LLVM_DEBUG(llvm::dbgs() << "UB: " << ubExpr << "\n");
        auto cLb = dyn_cast<AffineConstantExpr>(lbExpr);
        auto cUb = dyn_cast<AffineConstantExpr>(ubExpr);
        if (!cLb || !cUb)
          return WalkResult::interrupt();
        auto lb = cLb.getValue();
        auto ub = cUb.getValue();

        if (!(lb >= 0 && lb <= cst && ub >= 0 && ub <= cst))
          return WalkResult::interrupt();

        toReplace.insert({expr, lhs});
        return WalkResult::advance();
      });

      bool isValid = !res.wasInterrupted();

      if (!isValid) {
        allValid = false;
        break;
      }

      mod = simplifyAffineExpr(mod.replace(toReplace), accessAvm.getNumDims(),
                               accessAvm.getNumSymbols());

      toReplace.clear();
      auto zero = getAffineConstantExpr(0, ctx);
      floor.walk([&](AffineExpr expr) {
        AffineBinaryOpExpr binexpr = dyn_cast<AffineBinaryOpExpr>(expr);
        if (!binexpr)
          return;
        if (binexpr.getKind() != AffineExprKind::FloorDiv)
          return;
        if (binexpr.getRHS() != getAffineConstantExpr(cst, ctx))
          return;

        // Theoretically all the checks above for the `mod` should succeed
        // here too.
        toReplace.insert({expr, zero});
      });

      floor =
          simplifyAffineExpr(floor.replace(toReplace), accessAvm.getNumDims(),
                             accessAvm.getNumSymbols());

      SmallVector<AffineExpr> exprs(
          accessAvm.getAffineMap().getResults().begin(),
          accessAvm.getAffineMap().getResults().end());
      exprs.erase(std::next(exprs.begin(), resultId));
      exprs.insert(std::next(exprs.begin(), resultId), mod);
      exprs.insert(std::next(exprs.begin(), resultId), floor);
      ainfo.map = AffineMap::get(accessAvm.getNumDims(),
                                 accessAvm.getNumSymbols(), exprs, ctx);
      LLVM_DEBUG(llvm::dbgs() << "New map: " << ainfo.map << "\n");
    }

    if (!allValid)
      return false;

    LLVM_DEBUG(llvm::dbgs() << "Found valid shape candidate " << cst << "\n");

    IRRewriter rewriter(ctx);

    rewriteMemrefCallback(rewriter, resultId, cst);

    for (auto &ainfo : accesses) {
      if (auto load = dyn_cast<AffineLoadOp>(ainfo.access.opInst)) {
        rewriter.setInsertionPoint(load);
        rewriter.replaceOpWithNewOp<AffineLoadOp>(
            load, load.getMemref(), ainfo.map, load.getMapOperands());
      } else if (auto store = dyn_cast<AffineStoreOp>(ainfo.access.opInst)) {
        rewriter.setInsertionPoint(store);
        rewriter.replaceOpWithNewOp<AffineStoreOp>(store, store.getValue(),
                                                   store.getMemref(), ainfo.map,
                                                   store.getMapOperands());
      } else {
        llvm_unreachable("unexpected");
      }
    }

    return true;
  };

  for (auto cst : constants) {
    if (cst == 1 || cst == 0)
      continue;
    assert(cst > 0);
    if (checkShapeCandidate(cst, 0))
      return success();
  }
  return failure();
}

LogicalResult reshapeAtAddr(enzymexla::Pointer2MemrefOp &atAddr) {
  auto memref = atAddr.getResult();
  return reshapeMemref(memref, [&](RewriterBase &rewriter, unsigned resultId,
                                   unsigned cst) {
    rewriter.setInsertionPoint(atAddr);
    auto oldMt = atAddr.getResult().getType();
    SmallVector<int64_t> shape(oldMt.getShape().begin(),
                               oldMt.getShape().end());
    shape.insert(std::next(shape.begin(), resultId + 1), cst);
    if (shape[resultId] != ShapedType::kDynamic) {
      shape[resultId] = llvm::divideCeil(shape[resultId], cst);
    } else {
      shape[resultId] = ShapedType::kDynamic;
    }
    // TODO should we use the existing alloca layout interface in here
    // somehow?
    auto newMt =
        MemRefType::get(shape, oldMt.getElementType(),
                        MemRefLayoutAttrInterface{}, oldMt.getMemorySpace());

    atAddr = rewriter.replaceOpWithNewOp<enzymexla::Pointer2MemrefOp>(
        atAddr, atAddr.getSource(), newMt);
  });
}

LogicalResult reshapeAlloca(memref::AllocaOp &alloca) {
  auto memref = alloca.getResult();
  return reshapeMemref(memref, [&](RewriterBase &rewriter, unsigned resultId,
                                   unsigned cst) {
    rewriter.setInsertionPoint(alloca);
    auto oldMt = alloca.getMemref().getType();
    SmallVector<Value> dynSizes = alloca.getDynamicSizes();
    SmallVector<int64_t> shape(oldMt.getShape().begin(),
                               oldMt.getShape().end());
    shape.insert(std::next(shape.begin(), resultId + 1), cst);
    if (shape[resultId] != ShapedType::kDynamic) {
      shape[resultId] = llvm::divideCeil(shape[resultId], cst);
    } else {
      llvm_unreachable("Unsupported dymanic memref.alloca");
    }
    // TODO should we use the existing alloca layout interface in here
    // somehow?
    auto newMt =
        MemRefType::get(shape, oldMt.getElementType(),
                        MemRefLayoutAttrInterface{}, oldMt.getMemorySpace());

    alloca = rewriter.replaceOpWithNewOp<memref::AllocaOp>(
        alloca, newMt, alloca.getDynamicSizes(), alloca.getSymbolOperands(),
        alloca.getAlignmentAttr());
  });
}

} // namespace

struct DelinearizeIndexingPass
    : public enzyme::impl::DelinearizeIndexingPassBase<DelinearizeIndexingPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    {
      SmallVector<memref::AllocaOp> toHandle;
      op->walk([&](memref::AllocaOp alloca) { toHandle.push_back(alloca); });
      for (auto alloca : toHandle)
        while (succeeded(reshapeAlloca(alloca)))
          ;
    }
    {
      SmallVector<enzymexla::Pointer2MemrefOp> toHandle;
      op->walk([&](enzymexla::Pointer2MemrefOp atAddr) {
        toHandle.push_back(atAddr);
      });
      for (auto atAddr : toHandle)
        while (succeeded(reshapeAtAddr(atAddr)))
          ;
    }
  }
};

std::unique_ptr<Pass> mlir::createDelinearizeIndexingPass() {
  return std::make_unique<DelinearizeIndexingPass>();
}
