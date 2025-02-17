
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
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
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

#include "mlir/IR/Region.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DELINEARIZEINDEXINGPASS
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

  AccessInfo(affine::MemRefAccess access, AffineMap map = AffineMap())
      : access(access), map(map) {}
};

LogicalResult getOpIndexSet(Operation *op,
                            affine::FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  affine::getEnclosingAffineOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

static int64_t floorDiv(int64_t a, int64_t b) {
  int64_t div = a / b;
  if (a % b != 0 && (a < 0) != (b < 0))
    return div - 1;
  return div;
}

static int64_t ceilDiv(int64_t a, int64_t b) {
  return floorDiv((a - 1), b) + 1;
}

AffineMap simplifyExprs(affine::AffineValueMap accessAvm,
                        affine::MemRefAccess access) {
  using namespace affine;
  auto ctx = access.opInst->getContext();

  SmallVector<AffineExpr> exprs;
  for (auto resultId : llvm::seq<unsigned>(accessAvm.getNumResults())) {
    AffineExpr expr = accessAvm.getResult(resultId);
    LLVM_DEBUG(llvm::dbgs() << "Simplifying expr " << expr << "\n");

    DenseMap<AffineExpr, AffineExpr> toReplace;
    while (expr.walk([&](AffineExpr expr) {
                 LLVM_DEBUG(llvm::dbgs() << "Walking expr " << expr << "\n");
                 AffineBinaryOpExpr binexpr =
                     dyn_cast<AffineBinaryOpExpr>(expr);
                 if (!binexpr) {
                   LLVM_DEBUG(llvm::dbgs() << "Not binexpr\n");
                   return WalkResult::advance();
                 }
                 if (binexpr.getKind() != AffineExprKind::Mod &&
                     binexpr.getKind() != AffineExprKind::FloorDiv &&
                     binexpr.getKind() != AffineExprKind::CeilDiv) {
                   LLVM_DEBUG(llvm::dbgs() << (unsigned)binexpr.getKind()
                                           << " not mod or div\n");
                   return WalkResult::advance();
                 }

                 auto rhs = binexpr.getRHS();
                 auto rhsMap = AffineMap::get(accessAvm.getNumDims(),
                                              accessAvm.getNumSymbols(), rhs);

                 auto rhsCstExpr = dyn_cast<AffineConstantExpr>(rhs);
                 if (!rhsCstExpr) {
                   LLVM_DEBUG(llvm::dbgs() << "RHS not const\n");
                   return WalkResult::advance();
                 }
                 int64_t cst = rhsCstExpr.getValue();
                 LLVM_DEBUG(llvm::dbgs() << "RHS cst " << cst << "\n");

                 auto lhs = binexpr.getLHS();
                 auto lhsMap = AffineMap::get(accessAvm.getNumDims(),
                                              accessAvm.getNumSymbols(), lhs);

                 AffineValueMap lhsAvm(lhsMap, accessAvm.getOperands());
                 lhsAvm.composeSimplifyAndCanonicalize();
                 LLVM_DEBUG(llvm::dbgs()
                            << "Nested mod: " << lhsAvm.getAffineMap() << "\n");
                 affine::FlatAffineValueConstraints domain;
                 if (failed(getOpIndexSet(access.opInst, &domain))) {
                   LLVM_DEBUG(llvm::dbgs() << "Could not get op index set\n");
                   return WalkResult::advance();
                 }
                 if (failed(domain.composeMap(&lhsAvm))) {
                   LLVM_DEBUG(llvm::dbgs() << "Could compose map\n");
                   return WalkResult::advance();
                 }
                 LLVM_DEBUG(llvm::dbgs() << "Composed domain: ");
                 LLVM_DEBUG(domain.dump());
                 domain.setDimSymbolSeparation(domain.getNumDimAndSymbolVars() -
                                               1);
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
                 if (domain.getNumLocalVars() > 0) {
                   LLVM_DEBUG(
                       llvm::dbgs()
                       << "We don't know what to do with local vars yet.\n");
                   // TODO they need to be passed into the getLowerAndUpperBound
                   // call
                   return WalkResult::advance();
                 }
                 auto bounds = domain.getLowerAndUpperBound(
                     0, 0, 1, domain.getNumDimVars(), {}, ctx);
                 auto lbExpr = bounds.first.getResult(0);
                 auto ubExpr = bounds.second.getResult(0);
                 LLVM_DEBUG(llvm::dbgs() << "LB: " << lbExpr << "\n");
                 LLVM_DEBUG(llvm::dbgs() << "UB: " << ubExpr << "\n");
                 auto cLb = dyn_cast<AffineConstantExpr>(lbExpr);
                 auto cUb = dyn_cast<AffineConstantExpr>(ubExpr);
                 if (!cLb || !cUb) {
                   LLVM_DEBUG(llvm::dbgs() << "Could not get cLb cUb\n");
                   return WalkResult::advance();
                 }
                 // Get the range [lb, ub] from [cLb, cUb)
                 int64_t lb = cLb.getValue();
                 int64_t ub = cUb.getValue() - 1;
                 LLVM_DEBUG(llvm::dbgs() << "LB: " << lb << "\n");
                 LLVM_DEBUG(llvm::dbgs() << "UB: " << ub << "\n");

                 if (ub - lb >= cst) {
                   LLVM_DEBUG(llvm::dbgs() << "Range bigger than cst\n");
                   return WalkResult::advance();
                 }

                 if (binexpr.getKind() == AffineExprKind::Mod ||
                     binexpr.getKind() == AffineExprKind::FloorDiv) {
                   int64_t ubd = floorDiv(ub, cst);
                   int64_t lbd = floorDiv(lb, cst);
                   if (ubd != lbd) {
                     LLVM_DEBUG(llvm::dbgs()
                                << "Unequal div " << lbd << " " << ubd << "\n");
                     return WalkResult::advance();
                   }

                   if (binexpr.getKind() == AffineExprKind::FloorDiv) {
                     AffineExpr simplified =
                         getAffineConstantExpr(floorDiv(ub, cst), ctx);
                     toReplace.insert({expr, simplified});
                     return WalkResult::interrupt();
                   } else if (binexpr.getKind() == AffineExprKind::Mod) {
                     AffineExpr simplified = lhs - (floorDiv(ub, cst) * cst);
                     toReplace.insert({expr, simplified});
                     return WalkResult::interrupt();
                   } else {
                     llvm_unreachable("?");
                   }

                 } else if (binexpr.getKind() == AffineExprKind::CeilDiv) {
                   int64_t ubd = ceilDiv(ub, cst);
                   int64_t lbd = ceilDiv(lb, cst);
                   if (ubd != lbd) {
                     LLVM_DEBUG(llvm::dbgs() << "Unequal cdiv " << lbd << " "
                                             << ubd << "\n");
                     return WalkResult::advance();
                   }
                   AffineExpr simplified =
                       getAffineConstantExpr(ceilDiv(ub, cst), ctx);
                   toReplace.insert({expr, simplified});
                   return WalkResult::interrupt();
                 } else {
                   llvm_unreachable("?");
                 }
                 return WalkResult::advance();
               })
               .wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "REPLACING " << toReplace.begin()->getFirst() << " WITH "
                 << toReplace.begin()->getSecond() << "\n");
      expr = simplifyAffineExpr(expr.replace(toReplace), accessAvm.getNumDims(),
                                accessAvm.getNumSymbols());
    }
    LLVM_DEBUG(llvm::dbgs() << "Simplified expr: " << expr << "\n");
    exprs.push_back(expr);
  }
  AffineMap map = AffineMap::get(accessAvm.getNumDims(),
                                 accessAvm.getNumSymbols(), exprs, ctx);
  LLVM_DEBUG(llvm::dbgs() << "New map: " << map << "\n");
  return map;
}

LogicalResult
reshapeMemref2(Value memref, ArrayRef<int64_t> shape,
               std::function<void(RewriterBase &)> rewriteMemrefCallback) {

  MLIRContext *ctx = memref.getContext();
  using namespace mlir::affine;

  SmallVector<AccessInfo> accesses;
  bool foundAllUses = true;
  for (auto user : memref.getUsers()) {
    if (auto load = dyn_cast<AffineLoadOp>(user)) {
      accesses.push_back({MemRefAccess(load), load.getAffineMap()});
    } else if (auto store = dyn_cast<AffineStoreOp>(user)) {
      accesses.push_back({MemRefAccess(store), store.getAffineMap()});
    } else {
      foundAllUses = false;
      break;
    }
  }

  if (!foundAllUses)
    return failure();

  IRRewriter rewriter(ctx);

  for (unsigned shapeIdx :
       llvm::reverse(llvm::seq<unsigned>(1, shape.size()))) {
    int64_t cst = shape[shapeIdx];
    unsigned resultId = 0;
    for (auto &ainfo : accesses) {
      auto access = ainfo.access;
      AffineMap map = ainfo.map;
      AffineExpr expr = map.getResult(resultId);
      LLVM_DEBUG(llvm::dbgs() << "For access " << *access.opInst
                              << " with expr " << expr << "\n");
      auto mod = expr % cst;
      auto floor = expr.floorDiv(cst);
      LLVM_DEBUG(llvm::dbgs() << "Mod: " << mod << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Floor: " << floor << "\n");

      SmallVector<AffineExpr> exprs(map.getResults().begin(),
                                    map.getResults().end());
      exprs.erase(std::next(exprs.begin(), resultId));
      exprs.insert(std::next(exprs.begin(), resultId), mod);
      exprs.insert(std::next(exprs.begin(), resultId), floor);
      ainfo.map =
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx);
      LLVM_DEBUG(llvm::dbgs() << "New map: " << ainfo.map << "\n");
      AffineValueMap avm;
      access.getAccessMap(&avm);
      avm.reset(ainfo.map, avm.getOperands());
      ainfo.map = simplifyExprs(avm, ainfo.access);
    }
  }

  rewriteMemrefCallback(rewriter);

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
  return success();
}

LogicalResult reshapeAtAddr(enzymexla::Pointer2MemrefOp &atAddr) {
  auto source = atAddr.getSource();
  auto m2p = source.getDefiningOp<enzymexla::Memref2PointerOp>();
  if (!m2p)
    return failure();
  MemRefType newMt = m2p.getSource().getType();
  auto shape = newMt.getShape();

  // Only the first rank can be dynamic
  if (llvm::any_of(llvm::drop_begin(shape),
                   [](int64_t size) { return size == ShapedType::kDynamic; }))
    return failure();

  if (shape.size() <= 1)
    return failure();

  auto memref = atAddr.getResult();
  return reshapeMemref2(memref, shape, [&](RewriterBase &rewriter) {
    rewriter.setInsertionPoint(atAddr);
    auto oldMt = atAddr.getResult().getType();

    atAddr = rewriter.replaceOpWithNewOp<enzymexla::Pointer2MemrefOp>(
        atAddr, newMt, atAddr.getSource());
  });
}

} // namespace

struct DelinearizeIndexingPass
    : public enzyme::impl::DelinearizeIndexingPassBase<
          DelinearizeIndexingPass> {
  using Base::Base;
  void runOnOperation() override {
    Operation *op = getOperation();
    {
      SmallVector<enzymexla::Pointer2MemrefOp> toHandle;
      op->walk([&](enzymexla::Pointer2MemrefOp atAddr) {
        toHandle.push_back(atAddr);
      });
      for (auto atAddr : toHandle)
        succeeded(reshapeAtAddr(atAddr));
    }
  }
};
