
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
#include <algorithm>
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
  bool is_affine;
  // memref loads and stores
  Operation *mOpInst;
  Value linear_key;
  SmallVector<Value> linear_vals;

  AccessInfo(affine::MemRefAccess access, AffineMap map = AffineMap())
      : is_affine(true), access(access), map(map) {}

  AccessInfo(memref::LoadOp load)
      : is_affine(false), access(nullptr), map(nullptr) {
    linear_key = load.getIndices()[0]; // there's only one index (for now)
    mOpInst = load;
  }
  AccessInfo(memref::StoreOp store)
      : is_affine(false), access(nullptr), map(nullptr) {
    linear_key = store.getIndices()[0]; // there's only one index(for now)
    mOpInst = store;
  }
};

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
    } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
      accesses.push_back({load});
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      accesses.push_back({store});
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
      if (ainfo.is_affine) {
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
      } else {
        // either memref ld/st (emit new index calc ops)
        Value last_dim_key = ainfo.linear_key;
        rewriter.setInsertionPoint(ainfo.mOpInst);
        auto dim_size = rewriter.create<arith::ConstantIndexOp>(
            ainfo.mOpInst->getLoc(), cst);
        auto mod = rewriter.create<arith::RemSIOp>(ainfo.mOpInst->getLoc(),
                                                   last_dim_key, dim_size);
        auto floor = rewriter.create<arith::FloorDivSIOp>(
            ainfo.mOpInst->getLoc(), last_dim_key, dim_size);
        ainfo.linear_vals.push_back(mod);

        // floor is the new last dim key
        ainfo.linear_key = floor;
      }
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
    } else if (auto load = dyn_cast<memref::LoadOp>(ainfo.mOpInst)) {
      ainfo.linear_vals.push_back(ainfo.linear_key);
      std::reverse(ainfo.linear_vals.begin(), ainfo.linear_vals.end());
      rewriter.setInsertionPoint(load);
      rewriter.replaceOpWithNewOp<memref::LoadOp>(load, load.getMemref(),
                                                  ainfo.linear_vals);
    } else if (auto store = dyn_cast<memref::StoreOp>(ainfo.mOpInst)) {

      ainfo.linear_vals.push_back(ainfo.linear_key);
      std::reverse(ainfo.linear_vals.begin(), ainfo.linear_vals.end());
      rewriter.setInsertionPoint(load);
      rewriter.replaceOpWithNewOp<memref::StoreOp>(store, store.getMemref(),
                                                   ainfo.linear_vals);
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
