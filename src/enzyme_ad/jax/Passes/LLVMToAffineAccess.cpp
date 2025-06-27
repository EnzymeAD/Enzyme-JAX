#include "Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "Utils.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>

#define DEBUG_TYPE "llvm-to-affine-access"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LLVMTOAFFINEACCESSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

using PtrVal = TypedValue<LLVM::LLVMPointerType>;
using MemRefVal = MemrefValue;

static mlir::Value createConstantInt(RewriterBase &rewriter, Location loc,
                                     Type ty, int64_t v) {
  if (ty.isIndex())
    return rewriter.create<arith::ConstantIndexOp>(loc, v);
  else
    return rewriter.create<arith::ConstantIntOp>(loc, v, ty);
}

static std::optional<int64_t> getConstant(Operation *op) {
  if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<LLVM::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return {};
}

static std::optional<int64_t> getConstant(Value v) {
  Operation *op = v.getDefiningOp();
  if (op)
    return getConstant(op);
  return {};
}

template <typename FromAlloc, bool inPlace = false>
static LogicalResult
convertLLVMAllocaToMemrefAlloca(FromAlloc alloc, RewriterBase &rewriter,
                                const DataLayout &dataLayout) {
  SmallVector<enzymexla::Pointer2MemrefOp> p2ms;
  SmallVector<Operation *> others;
  for (auto op : alloc->getResult(0).getUsers()) {
    if (inPlace) {
      if (auto m2p = dyn_cast<enzymexla::Memref2PointerOp>(op)) {
        for (auto &op2u : m2p.getResult().getUses()) {
          auto op2 = op2u.getOwner();
          if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op2)) {
            p2ms.push_back(p2m);
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "unknown user: " << *op << "\n");
          LLVM_DEBUG(llvm::dbgs() << " - sub user : " << *op2 << "\n");
          return rewriter.notifyMatchFailure(alloc,
                                             "unknown user of allocation");
        }
        others.push_back(m2p);
        continue;
      }
    } else {
      if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(op)) {
        p2ms.push_back(p2m);
        continue;
      }
      if (isa<LLVM::LifetimeStartOp>(op) || isa<LLVM::LifetimeEndOp>(op)) {
        others.push_back(op);
        continue;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "unknown user: " << *op << "\n");
    return rewriter.notifyMatchFailure(alloc, "Unknown user of allocation");
  }
  if (p2ms.size() == 0)
    return failure();
  for (int i = 1; i < p2ms.size(); i++) {
    if (p2ms[i].getType().getElementType() !=
        p2ms[0].getType().getElementType()) {
      if (p2ms[0].getType().getElementType().isInteger(8)) {
        std::swap(p2ms[0], p2ms[i]);
      }
      if (p2ms[i].getType().getElementType().isInteger(8)) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "p2ms[0]:" << p2ms[0] << "\n");
      LLVM_DEBUG(llvm::dbgs() << "p2ms[i]:" << p2ms[i] << "\n");
      return rewriter.notifyMatchFailure(alloc,
                                         "Non matching tpye of multi p2m");
    }
  }

  auto ptr2memref = p2ms[0];
  int64_t elNum;
  if constexpr (!inPlace) {
    auto sizeVal = getConstant(alloc.getArraySize());
    if (!sizeVal)
      return failure();

    elNum = dataLayout.getTypeSize(alloc.getElemType()) * (*sizeVal);
  } else {
    if (ptr2memref.getResult().getType().getElementType() ==
        alloc.getType().getElementType())
      return failure();
    if (alloc.getType().getShape().size() != 1)
      return failure();
    elNum = alloc.getType().getShape()[0];
    if (elNum == ShapedType::kDynamic) {
      auto sizeVal = getConstant(alloc.getDynamicSizes()[0]);
      if (!sizeVal) {
        size_t num = 1;
        size_t den = 1;
        Value op = alloc.getDynamicSizes()[0];
        while (true) {
          if (auto icast = op.getDefiningOp<arith::IndexCastOp>()) {
            op = icast.getOperand();
            continue;
          }
          if (auto icast = op.getDefiningOp<arith::IndexCastUIOp>()) {
            op = icast.getOperand();
            continue;
          }
          if (auto shr = op.getDefiningOp<arith::ShRSIOp>()) {
            if (auto cst = getConstant(shr.getRhs())) {
              auto val = 1ULL << *cst;
              if (num % val == 0) {
                num /= val;
                op = shr.getLhs();
                continue;
              } else if (val != 0 && val % num == 0) {
                den *= (val / num);
                num = 1;
                op = shr.getLhs();
                continue;
              }
            }
          }
          if (auto shr = op.getDefiningOp<arith::ShRUIOp>()) {
            if (auto cst = getConstant(shr.getRhs())) {
              auto val = 1ULL << *cst;
              if (num % val == 0) {
                num /= val;
                op = shr.getLhs();
                continue;
              } else if (val != 0 && val % num == 0) {
                den *= (val / num);
                num = 1;
                op = shr.getLhs();
                continue;
              }
            }
          }
          if (auto shl = op.getDefiningOp<arith::ShLIOp>()) {
            if (auto cst = getConstant(shl.getRhs())) {
              auto val = 1ULL << *cst;
              if (den % val == 0) {
                den /= val;
                op = shl.getLhs();
                continue;
              } else if (val != 0 && val % den == 0) {
                num *= (val / den);
                den = 1;
                op = shl.getLhs();
                continue;
              }
            }
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "could not deduce size of copy due to " << op
                     << " num=" << num << " den=" << den << "\n");
          break;
        }
        if (den == 1) {
          elNum = num;
        } else {
          return failure();
        }
      } else {
        elNum = *sizeVal;
      }
    }
    elNum *= dataLayout.getTypeSize(
        cast<MemRefType>(alloc->getResult(0).getType()).getElementType());
  }

  auto newElSize =
      dataLayout.getTypeSize(ptr2memref.getResult().getType().getElementType());
  int64_t newElnum = elNum / newElSize;
  if (newElSize * newElnum != elNum)
    return failure();

  MemRefType memrefType;
  if constexpr (!inPlace) {
    SmallVector<int64_t, 1> sizes = {newElnum};
    memrefType = MemRefType::get(
        sizes, ptr2memref.getResult().getType().getElementType(),
        MemRefLayoutAttrInterface{},
        ptr2memref.getResult().getType().getMemorySpace());
  } else {
    SmallVector<int64_t, 1> sizes = {newElnum};
    if (alloc.getDynamicSizes().size()) {
      sizes[0] = ShapedType::kDynamic;
    }
    memrefType = MemRefType::get(
        sizes, ptr2memref.getResult().getType().getElementType(),
        MemRefLayoutAttrInterface{}, alloc.getType().getMemorySpace());
  }
  Value newAlloc;
  if constexpr (!inPlace)
    newAlloc = rewriter.create<memref::AllocaOp>(alloc->getLoc(), memrefType);
  else {

    auto tys = llvm::to_vector(alloc->getResultTypes());
    tys[0] = memrefType;

    auto dyn = llvm::to_vector(alloc.getDynamicSizes());
    if (alloc.getDynamicSizes().size()) {
      dyn[dyn.size() - 1] = rewriter.create<arith::MulIOp>(
          alloc->getLoc(), dyn[dyn.size() - 1],
          rewriter.create<arith::ConstantIndexOp>(
              alloc->getLoc(),
              dataLayout.getTypeSize(
                  cast<MemRefType>(alloc->getResult(0).getType())
                      .getElementType())));
      dyn[dyn.size() - 1] = rewriter.create<arith::DivUIOp>(
          alloc->getLoc(), dyn[dyn.size() - 1],
          rewriter.create<arith::ConstantIndexOp>(
              alloc->getLoc(),
              dataLayout.getTypeSize(
                  ptr2memref.getResult().getType().getElementType())));
    }

    auto newOp = cast<FromAlloc>(rewriter.create(
        alloc->getLoc(), alloc->getName().getIdentifier(), alloc->getOperands(),
        tys, alloc->getAttrs(), alloc->getSuccessors()));

    if (alloc.getDynamicSizes().size()) {
      newOp.getDynamicSizesMutable().assign(dyn);
    }

    newAlloc = newOp->getResult(0);
  }

  for (auto p2m : p2ms) {
    Value replacement = newAlloc;
    if (memrefType.getElementType() != p2m.getType().getElementType()) {
      replacement = rewriter.create<enzymexla::Memref2PointerOp>(
          alloc->getLoc(), p2m.getOperand().getType(), replacement);
      rewriter.modifyOpInPlace(
          p2m, [&]() { p2m.getSourceMutable().set(replacement); });
      continue;
    }
    if (replacement.getType() != p2m.getType())
      replacement = rewriter.create<memref::CastOp>(p2m.getLoc(), p2m.getType(),
                                                    replacement);
    rewriter.replaceOp(p2m, replacement);
  }

  for (auto other : others) {
    rewriter.eraseOp(other);
  }

  rewriter.eraseOp(alloc);
  return success();
}

namespace {

struct ConvertToTypedMemref
    : public OpRewritePattern<enzymexla::Pointer2MemrefOp> {
  using OpRewritePattern<enzymexla::Pointer2MemrefOp>::OpRewritePattern;
  const DataLayoutAnalysis &dl;
  ConvertToTypedMemref(MLIRContext *context, const DataLayoutAnalysis &dl)
      : OpRewritePattern<enzymexla::Pointer2MemrefOp>(context), dl(dl) {}

  LogicalResult matchAndRewrite(enzymexla::Pointer2MemrefOp p2m,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Checking " << p2m << "\n");
    TypedValue<MemRefType> memref = p2m.getResult();
    bool allGood = true;
    Type type = nullptr;
    int64_t allSize = 0;
    TypedValue<MemRefType> newMemref = nullptr;
    auto getNewMemref = [&]() {
      OpBuilder::InsertionGuard g(rewriter);
      if (!newMemref) {
        rewriter.setInsertionPoint(p2m);
        auto newp2m = rewriter.create<enzymexla::Pointer2MemrefOp>(
            p2m.getLoc(),
            MemRefType::get({ShapedType::kDynamic}, type,
                            MemRefLayoutAttrInterface{},
                            memref.getType().getMemorySpace()),
            p2m.getSource());
        newMemref = newp2m.getResult();
      }
      return newMemref;
    };

    SmallVector<Operation *> toErase;

    IRMapping mapping;
    for (auto &use : memref.getUses()) {
      auto checkTypeAndAlignment = [&](int64_t size, Type t, AffineExpr expr) {
        allSize = size;
        if (!expr.isMultipleOf(size))
          return failure();
        if (!type) {
          type = t;
          return success();
        }
        if (type == t) {
          return success();
        }
        return failure();
      };
      if (auto load = dyn_cast<affine::AffineVectorLoadOp>(use.getOwner())) {
        assert(load.getValue().hasOneUse());
        Operation *user = *load.getValue().user_begin();
        assert(user->getNumResults() == 1);
        assert(load.getType().getRank() == 1);
        assert(load.getMemRefType().getRank() == 1);
        auto size = load.getType().getShape()[0];
        assert(size != ShapedType::kDynamic);
        auto map = load.getMap();
        auto expr = map.getResults()[0];
        auto value = user->getResult(0);
        if (checkTypeAndAlignment(size, value.getType(), expr).failed()) {
          allGood = false;
          break;
        }
        rewriter.setInsertionPoint(load);
        auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                     {expr.floorDiv(size)}, load.getContext());
        auto newLoad = rewriter.create<affine::AffineLoadOp>(
            load.getLoc(), getNewMemref(), newMap, load.getMapOperands());
        // Copy attributes from the old load to the new one
        for (auto attr : load->getAttrs()) {
          newLoad->setAttr(attr.getName(), attr.getValue());
        }
        mapping.map(value, newLoad.getValue());
        toErase.push_back(user);
        toErase.push_back(load);
      } else if (auto store =
                     dyn_cast<affine::AffineVectorStoreOp>(use.getOwner())) {
        Operation *user = store.getValue().getDefiningOp();
        auto size = store.getValue().getType().getShape()[0];
        assert(size != ShapedType::kDynamic);
        auto map = store.getMap();
        auto expr = map.getResults()[0];
        auto value = user->getOperand(0);
        if (checkTypeAndAlignment(size, value.getType(), expr).failed()) {
          allGood = false;
          break;
        }
        rewriter.setInsertionPoint(store);
        auto newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                     {expr.floorDiv(size)}, store.getContext());
        auto newStore = rewriter.create<affine::AffineStoreOp>(
            store.getLoc(), value, getNewMemref(), newMap,
            store.getMapOperands());
        // Copy attributes from the old store to the new one
        for (auto attr : store->getAttrs()) {
          newStore->setAttr(attr.getName(), attr.getValue());
        }
        toErase.push_back(store);
        toErase.push_back(user);
      } else {
        allGood = false;
        break;
      }
    }

    if (!allGood)
      return failure();

    if (type == rewriter.getI8Type())
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "all good " << allGood << "\n");

    for (auto &m : mapping.getValueMap())
      rewriter.replaceAllUsesWith(m.getFirst(), m.getSecond());

    for (Operation *op : toErase)
      rewriter.eraseOp(op);

    return failure();
  }
};

struct ConvertLLVMAllocaToMemrefAlloca
    : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;
  const DataLayoutAnalysis &dl;
  ConvertLLVMAllocaToMemrefAlloca(MLIRContext *context,
                                  const DataLayoutAnalysis &dl)
      : OpRewritePattern<LLVM::AllocaOp>(context), dl(dl) {}

  LogicalResult matchAndRewrite(LLVM::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto dataLayout = dl.getAtOrAbove(alloc);
    return convertLLVMAllocaToMemrefAlloca(alloc, rewriter, dataLayout);
  }
};

template <typename T> struct SimplifyInPlaceAlloc : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  const DataLayoutAnalysis &dl;
  SimplifyInPlaceAlloc(MLIRContext *context, const DataLayoutAnalysis &dl)
      : OpRewritePattern<T>(context), dl(dl) {}

  LogicalResult matchAndRewrite(T alloc,
                                PatternRewriter &rewriter) const override {
    auto dataLayout = dl.getAtOrAbove(alloc);
    return convertLLVMAllocaToMemrefAlloca<T, true>(alloc, rewriter,
                                                    dataLayout);
  }
};

static Value convertToIndex(Value v) {
  OpBuilder builder(v.getContext());
  if (v.getType() == builder.getIndexType())
    return v;
  if (auto ba = dyn_cast<BlockArgument>(v))
    builder.setInsertionPointToStart(ba.getOwner());
  else
    builder.setInsertionPointAfter(v.getDefiningOp());
  return builder
      .create<arith::IndexCastOp>(v.getLoc(), builder.getIndexType(), v)
      .getResult();
}

struct GEPOfMemRefLoad : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;
  const DataLayoutAnalysis &dl;
  GEPOfMemRefLoad(MLIRContext *context, const DataLayoutAnalysis &dl)
      : OpRewritePattern<memref::LoadOp>(context), dl(dl) {}

  LogicalResult matchAndRewrite(memref::LoadOp ld,
                                PatternRewriter &rewriter) const override {
    auto dataLayout = dl.getAtOrAbove(ld);
    if (cast<MemRefType>(ld.getMemRef().getType()).getShape().size() != 1)
      return failure();
    auto ptr = ld.getMemRef().getDefiningOp<enzymexla::Pointer2MemrefOp>();
    if (!ptr)
      return failure();
    auto gep = ptr.getOperand().getDefiningOp<LLVM::GEPOp>();
    if (!gep)
      return failure();
    Type currentType = gep.getElemType();
    if (currentType != ld.getType())
      return failure();

    auto ptr2 = rewriter.create<enzymexla::Pointer2MemrefOp>(
        ptr.getLoc(), ptr.getType(), gep.getBase());
    SmallVector<Value, 1> indices;
    for (auto gidx : gep.getIndices()) {
      if (auto intv = dyn_cast<IntegerAttr>(gidx)) {
        auto val = rewriter.create<arith::ConstantOp>(gep.getLoc(),
                                                      intv.getType(), intv);
        indices.push_back(convertToIndex(val));
      } else if (auto val = dyn_cast<Value>(gidx)) {
        indices.push_back(convertToIndex(val));
      }
    }
    if (indices.size() != 1)
      return failure();
    SmallVector<Value, 1> idx2;
    for (auto midx : ld.getIndices())
      idx2.push_back(midx);
    idx2[0] = rewriter.create<arith::AddIOp>(gep.getLoc(), idx2[0], indices[0]);
    rewriter.modifyOpInPlace(ld, [&]() {
      ld.setMemRef(ptr2);
      ld.getIndicesMutable().assign(idx2);
    });
    return success();
  }
};

struct IndexCastAddSub : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp cst,
                                PatternRewriter &rewriter) const override {
    auto op = cst.getOperand().getDefiningOp();
    if (!op)
      return failure();
    if (!isa<arith::AddIOp, arith::SubIOp>(op))
      return failure();
    if (!isa<IndexType>(cst.getType()))
      return failure();
    if (cast<IntegerType>(cst.getOperand().getType()).getWidth() < 32)
      return failure();

    auto lhs = rewriter.create<arith::IndexCastOp>(cst.getLoc(), cst.getType(),
                                                   op->getOperand(0));
    auto rhs = rewriter.create<arith::IndexCastOp>(cst.getLoc(), cst.getType(),
                                                   op->getOperand(1));
    if (isa<arith::AddIOp>(op)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(cst, lhs, rhs);
    } else {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(cst, lhs, rhs);
    }
    return success();
  }
};

struct MemrefLoadAffineApply : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp ld,
                                PatternRewriter &rewriter) const override {
    if (cast<MemRefType>(ld.getMemRef().getType()).getShape().size() != 1)
      return failure();
    SmallVector<std::pair<bool, Value>> todo;
    todo.emplace_back(false, ld.getIndices()[0]);
    SmallVector<std::pair<bool, Value>> irreducible;

    SmallVector<Value> operands;
    AffineExpr expr = rewriter.getAffineConstantExpr(0);
    while (todo.size()) {
      auto cur = todo.pop_back_val();
      if (isValidIndex(cur.second)) {
        auto d2 = rewriter.getAffineSymbolExpr(operands.size());
        operands.push_back(cur.second);
        if (cur.first)
          expr = expr - d2;
        else
          expr = expr + d2;
        continue;
      }
      if (auto add = cur.second.getDefiningOp<arith::AddIOp>()) {
        todo.emplace_back(cur.first, add.getLhs());
        todo.emplace_back(cur.first, add.getRhs());
        continue;
      }
      if (auto add = cur.second.getDefiningOp<arith::SubIOp>()) {
        todo.emplace_back(cur.first, add.getLhs());
        todo.emplace_back(!cur.first, add.getRhs());
        continue;
      }
      irreducible.push_back(cur);
    }
    if (operands.size() == 0)
      return failure();
    SmallVector<Value> preoperands = operands;

    AffineExpr exprs[1] = {expr};
    auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/operands.size(),
                              exprs, rewriter.getContext());

    auto scope = ld->getParentOfType<FunctionOpInterface>();
    DominanceInfo DI(scope);
    assert(map.getNumInputs() == operands.size());
    fully2ComposeAffineMapAndOperands(rewriter, &map, &operands, DI);
    assert(map.getNumInputs() == operands.size());
    affine::canonicalizeMapAndOperands(&map, &operands);
    map = mlir::enzyme::recreateExpr(map);

    if (preoperands.size() == 1) {
      Attribute attr;
      if (matchPattern(preoperands[0], m_Constant(&attr)))
        return failure();
      if (preoperands == operands)
        return failure();
      if (auto app = preoperands[0].getDefiningOp<affine::AffineApplyOp>()) {
        if (app.getAffineMap() == map && app.getMapOperands() == operands)
          return failure();
      }
    }

    Value app;
    if (auto cst = dyn_cast<AffineConstantExpr>(map.getResult(0)))
      app =
          rewriter.create<arith::ConstantIndexOp>(ld.getLoc(), cst.getValue());
    else
      app = rewriter.create<affine::AffineApplyOp>(ld.getLoc(), map, operands);

    for (auto &&[negated, val] : irreducible) {
      if (negated)
        app = rewriter.create<arith::SubIOp>(ld.getLoc(), app, val);
      else
        app = rewriter.create<arith::AddIOp>(ld.getLoc(), app, val);
    }
    Value idx2[] = {app};
    rewriter.modifyOpInPlace(ld,
                             [&]() { ld.getIndicesMutable().assign(idx2); });
    return success();
  }
};

struct SelectCSE : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    auto lhs = sel.getTrueValue().getDefiningOp();
    if (!lhs)
      return failure();
    auto rhs = sel.getFalseValue().getDefiningOp();
    if (!rhs)
      return failure();

    if (lhs->getName() != rhs->getName())
      return failure();
    if (!isa<arith::AddIOp, arith::SubIOp>(lhs))
      return failure();

    auto op0 = rewriter.create<arith::SelectOp>(
        lhs->getLoc(), sel.getCondition(), lhs->getOperand(0),
        rhs->getOperand(0));
    auto op1 = rewriter.create<arith::SelectOp>(
        rhs->getLoc(), sel.getCondition(), rhs->getOperand(1),
        rhs->getOperand(1));
    if (isa<arith::AddIOp>(lhs)) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(sel, op0, op1);
    } else {
      rewriter.replaceOpWithNewOp<arith::SubIOp>(sel, op0, op1);
    }
    return success();
  }
};

struct SelectAddrCast : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp sel,
                                PatternRewriter &rewriter) const override {
    auto lhs = sel.getTrueValue().getDefiningOp<LLVM::AddrSpaceCastOp>();
    if (!lhs)
      return failure();
    auto rhs = sel.getFalseValue().getDefiningOp<LLVM::AddrSpaceCastOp>();
    if (!rhs)
      return failure();

    if (lhs.getOperand().getType() != rhs.getOperand().getType())
      return failure();

    auto sel0 = rewriter.create<arith::SelectOp>(
        sel.getLoc(), sel.getCondition(), lhs.getOperand(), rhs.getOperand());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(sel, sel.getType(),
                                                       sel0);
    return success();
  }
};

struct Pointer2MemrefSelect
    : public OpRewritePattern<enzymexla::Pointer2MemrefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::Pointer2MemrefOp p2m,
                                PatternRewriter &rewriter) const override {
    auto sel = p2m.getOperand().getDefiningOp<arith::SelectOp>();
    if (!sel)
      return failure();

    auto tval = rewriter.create<enzymexla::Pointer2MemrefOp>(
        p2m.getLoc(), p2m.getType(), sel.getTrueValue());
    auto fval = rewriter.create<enzymexla::Pointer2MemrefOp>(
        p2m.getLoc(), p2m.getType(), sel.getFalseValue());
    rewriter.replaceOpWithNewOp<arith::SelectOp>(p2m, sel.getCondition(), tval,
                                                 fval);
    return success();
  }
};

struct LoadSelect : public OpRewritePattern<affine::AffineLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp ld,
                                PatternRewriter &rewriter) const override {
    auto sel = ld.getMemRef().getDefiningOp<arith::SelectOp>();
    if (!sel)
      return failure();

    SmallVector<Type> resultTypes = {ld.getType()};

    auto newIfOp = rewriter.create<scf::IfOp>(sel.getLoc(), resultTypes,
                                              sel.getCondition(),
                                              /*hasElse=*/true);

    // Create new yield in then block
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
      auto ld2 = cast<affine::AffineLoadOp>(rewriter.clone(*ld));
      ld2.getMemrefMutable().set(sel.getTrueValue());
      rewriter.create<scf::YieldOp>(sel.getLoc(), ld2->getResults());
    }

    // Create new yield in else block
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
      auto ld2 = cast<affine::AffineLoadOp>(rewriter.clone(*ld));
      ld2.getMemrefMutable().set(sel.getFalseValue());
      rewriter.create<scf::YieldOp>(sel.getLoc(), ld2->getResults());
    }
    rewriter.replaceOp(ld, newIfOp);
    return success();
  }
};

} // namespace

static MemRefVal convertToMemref(PtrVal addr) {
  OpBuilder builder(addr.getContext());
  if (auto ba = dyn_cast<BlockArgument>(addr))
    builder.setInsertionPointToStart(ba.getOwner());
  else
    builder.setInsertionPointAfter(addr.getDefiningOp());
  Attribute addrSpace;
  if (addr.getType().getAddressSpace() == 0)
    addrSpace = nullptr;
  else
    addrSpace = IntegerAttr::get(IntegerType::get(addr.getContext(), 64),
                                 addr.getType().getAddressSpace());
  // TODO we can actually plug in the size of the memref here if `addr` is
  // defined by an llvm.alloca

  auto ptr2memref = builder.create<enzymexla::Pointer2MemrefOp>(
      addr.getLoc(),
      MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                      MemRefLayoutAttrInterface{}, Attribute(addrSpace)),
      addr);
  return cast<MemRefVal>(ptr2memref.getResult());
}

template <typename From, typename To, auto F> struct ConverterBase {
  DenseMap<From, To> map;
  To operator()(From p) {
    auto it = map.find(p);
    if (it != map.end())
      return it->getSecond();
    auto converted = F(p);
    map.insert({p, converted});
    return converted;
  }
  SmallVector<To> operator()(ValueRange range) {
    return llvm::map_to_vector(range, [&](From v) { return (*this)(v); });
  }
};

using MemrefConverter = ConverterBase<PtrVal, MemRefVal, convertToMemref>;
using IndexConverter = ConverterBase<Value, Value, convertToIndex>;

static BlockArgument getScopeRemap(enzymexla::AffineScopeOp scope, Value v) {
  for (unsigned i = 0; i < scope->getNumOperands(); i++)
    if (scope->getOperand(i) == v)
      return scope.getRegion().begin()->getArgument(i);
  return nullptr;
}

/// See llvm/Support/Alignment.h
static AffineExpr alignTo(AffineExpr expr, uint64_t a) {
  return (expr + a - 1).floorDiv(a) * a;
}

// TODO To preserve correctness, we need to keep track of values for which
// converting indexing to the index type preserves the semantics, i.e. no
// overflows or underflows or trucation etc and insert a runtime guard against
// that
struct AffineExprBuilder {
  AffineExprBuilder(Operation *user, bool legalizeSymbols)
      : user(user), legalizeSymbols(legalizeSymbols) {}
  Operation *user;

  SmallPtrSet<Value, 4> illegalSymbols;

  DenseMap<Value, unsigned> symToPos;
  DenseMap<Value, unsigned> dimToPos;
  SmallVector<Value> symbolOperands;
  SmallVector<Value> dimOperands;

  // Options
  bool legalizeSymbols;

  SmallVector<Value> symbolsForScope;
  unsigned scopedIllegalSymbols = 0;
  bool scoped = false;

  bool isLegal() {
    return illegalSymbols.size() == 0 ||
           (illegalSymbols.size() == scopedIllegalSymbols && scoped);
  }

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    assert(region->getBlocks().size() == 1);
    SmallVector<AffineExpr> newExprs;
    if (!region->isAncestor(user->getParentRegion()))
      return;
    // An illegal symbol will be legalized either by defining in at the top
    // level in a region, or by remapping it in the scope
    for (auto sym : illegalSymbols) {
      assert(sym.getParentRegion()->isAncestor(region));
      bool isOutsideRegion = sym.getParentRegion()->isProperAncestor(region);
      auto ba = dyn_cast<BlockArgument>(sym);
      bool isTopLevelBlockArg = ba && ba.getOwner()->getParent() == region;
      [[maybe_unused]] bool isTopLevelOp =
          !ba && sym.getParentRegion() == region;
      assert((unsigned)isOutsideRegion + (unsigned)isTopLevelBlockArg +
                 (unsigned)isTopLevelOp ==
             1);
      scopedIllegalSymbols++;
      if (isOutsideRegion || isTopLevelBlockArg)
        symbols.insert(sym);
    }
    if (!region->isProperAncestor(user->getParentRegion()))
      return;
    // We redefine dims to be symbols in this scope
    for (auto dim : dimOperands) {
      if (dim.getParentRegion()->isProperAncestor(region)) {
        symbols.insert(dim);
        symbolsForScope.push_back(dim);
      }
    }
    // TODO we may have a state like this:
    //
    // func.func () {
    //   %sym = ...
    //   region: {
    //     ...
    //   }
    // }
    //
    // and `sym` was mot marked illegal because func.func is an affine scope.
    // Should we rescope it to the new scope?
  }

  AffineExpr rescopeExprImpl(AffineExpr expr, enzymexla::AffineScopeOp scope) {
    auto newExpr = expr;
    for (auto sym : symbolsForScope) {
      unsigned dimPos = getDimPosition(sym);
      assert(dimOperands[dimPos] == sym);
      BlockArgument newSym = getScopeRemap(scope, sym);
      assert(newSym);
      unsigned newSymPos = getSymbolPosition(newSym);
      AffineExpr dimExpr = getAffineDimExpr(dimPos, user->getContext());
      AffineExpr newSymExpr = getAffineDimExpr(newSymPos, user->getContext());
      newExpr = newExpr.replace(dimExpr, newSymExpr);
    }
    for (auto sym : illegalSymbols) {
      if (sym.getParentRegion() == &scope.getRegion())
        continue;
      BlockArgument newSym = getScopeRemap(scope, sym);
      assert(newSym);
      auto it = llvm::find(symbolOperands, sym);
      assert(it != symbolOperands.end());
      *it = newSym;
    }
    return newExpr;
  }

  void rescopeExpr(enzymexla::AffineScopeOp scope) {
    expr = rescopeExprImpl(expr, scope);
    assert(!scoped);
    scoped = true;
  }

  unsigned getPosition(Value v, SmallVectorImpl<Value> &operands,
                       DenseMap<Value, unsigned> toPos) {
    auto it = toPos.find(v);
    if (it != toPos.end())
      return it->getSecond();
    unsigned newPos = operands.size();
    toPos.insert({v, newPos});
    operands.push_back(v);
    return newPos;
  }

  unsigned getSymbolPosition(Value v) {
    return getPosition(v, symbolOperands, symToPos);
  }
  unsigned getDimPosition(Value v) {
    return getPosition(v, dimOperands, dimToPos);
  }

  FailureOr<AffineExpr> buildExpr(Value v) {
    auto context = v.getContext();
    Operation *op = v.getDefiningOp();
    auto cst = getConstant(op);
    if (cst)
      return getAffineConstantExpr(*cst, context);
    bool isIndexTy = v.getType().isIndex();
    Value oldV = v;
    if (!isIndexTy)
      v = convertToIndex(v);
    if (affine::isValidSymbol(v)) {
      return getAffineSymbolExpr(getSymbolPosition(v), v.getContext());
    } else if (affine::isValidDim(v)) {
      return getAffineDimExpr(getDimPosition(v), v.getContext());
    }
    if (!isIndexTy) {
      v.getDefiningOp()->erase();
      v = oldV;
    }

    if (op) {
      if (op->getNumOperands() == 2 &&
          isa<LLVM::AddOp, arith::AddIOp, LLVM::SubOp, arith::SubIOp,
              LLVM::MulOp, arith::MulIOp, LLVM::UDivOp, LLVM::SDivOp,
              arith::DivUIOp, arith::DivSIOp, LLVM::URemOp, arith::RemSIOp,
              LLVM::SRemOp, arith::RemUIOp, arith::ShRUIOp, arith::ShRSIOp,
              LLVM::LShrOp, LLVM::AShrOp, arith::OrIOp>(op)) {
        auto lhs = buildExpr(op->getOperand(0));
        if (failed(lhs))
          return failure();
        auto rhs = buildExpr(op->getOperand(1));
        if (failed(rhs))
          return failure();

        if (isa<LLVM::AddOp, arith::AddIOp>(op))
          return (*lhs) + (*rhs);
        else if (isa<LLVM::SubOp, arith::SubIOp>(op))
          return (*lhs) - (*rhs);
        else if (isa<LLVM::MulOp, arith::MulIOp>(op))
          return (*lhs) * (*rhs);
        else if (isa<LLVM::UDivOp, LLVM::SDivOp, arith::DivUIOp,
                     arith::DivSIOp>(op))
          return (*lhs).floorDiv(*rhs);
        else if (isa<arith::ShRUIOp, arith::ShRSIOp, LLVM::LShrOp,
                     LLVM::AShrOp>(op)) {
          auto cexpr = dyn_cast<AffineConstantExpr>(*rhs);
          if (!cexpr)
            return failure();
          if (isa<arith::ShLIOp, LLVM::ShlOp>(op)) {
            return (*lhs) * getAffineConstantExpr(1 << cexpr.getValue(),
                                                  op->getContext());
          } else if (isa<arith::ShRUIOp, arith::ShRSIOp, LLVM::LShrOp,
                         LLVM::AShrOp>(op)) {
            return (*lhs).floorDiv(
                getAffineConstantExpr(1 << cexpr.getValue(), op->getContext()));
          } else {
            llvm_unreachable("unknown operation");
          }
        } else if (isa<LLVM::URemOp, arith::RemSIOp, LLVM::SRemOp,
                       arith::RemUIOp>(op)) {
          return (*lhs) % (*rhs);
        } else if (isa<arith::OrIOp>(op)) {
          auto cexpr = dyn_cast<AffineConstantExpr>(*rhs);
          if (!cexpr)
            return failure();
          if (!lhs->isMultipleOf(2))
            return failure();
          if (cexpr.getValue() != 1)
            return failure();
          return *lhs + 1;
        } else {
          llvm_unreachable("unknown operation");
        }
      } else if (isa<LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp, arith::ExtSIOp,
                     arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
                     arith::IndexCastUIOp>(op)) {
        return getExpr(op->getOperand(0));
      }
    }

    // TODO We may find an affine op reduction block arg - we may be able to
    // handle them

    for (auto &use : v.getUses()) {
      if (auto affineScope =
              dyn_cast<enzymexla::AffineScopeOp>(use.getOwner())) {
        if (affineScope->isAncestor(user))
          // TODO should we try to find the inner-most one?
          return getAffineSymbolExpr(
              getSymbolPosition(affineScope.getRegion().front().getArgument(
                  use.getOperandNumber())),
              v.getContext());
      }
    }

    if (legalizeSymbols) {
      illegalSymbols.insert(v);
      return getAffineSymbolExpr(getSymbolPosition(v), context);
    }

    return failure();
  }

  FailureOr<AffineExpr> getExpr(llvm::PointerUnion<IntegerAttr, Value> index) {
    auto constIndex = dyn_cast<IntegerAttr>(index);
    if (constIndex) {
      return getAffineConstantExpr(constIndex.getInt(), user->getContext());
    } else {
      auto expr = buildExpr(cast<Value>(index));
      LLVM_DEBUG(if (succeeded(expr)) expr->dump());
      return expr;
    }
  }

  AffineExpr expr;
  LogicalResult build(llvm::PointerUnion<IntegerAttr, Value> index) {
    auto mexpr = getExpr(index);
    if (failed(mexpr))
      return failure();
    expr = *mexpr;
    return success();
  }

  struct MapAndOperands {
    AffineMap map;
    SmallVector<Value> operands;
  };
  AffineExpr getExpr() {
    assert(isLegal());
    return expr;
  }
  MapAndOperands getMap() {
    assert(isLegal());
    AffineMap map = AffineMap::get(dimOperands.size(), symbolOperands.size(),
                                   expr, user->getContext());
    auto concat = llvm::concat<Value>(dimOperands, symbolOperands);
    SmallVector<Value> operands =
        SmallVector<Value>(concat.begin(), concat.end());
    affine::canonicalizeMapAndOperands(&map, &operands);
    map = simplifyAffineMap(map);
    return {map, operands};
  }
};

struct AffineAccessBuilder : AffineExprBuilder {
private:
  struct AffineAccess {
    PtrVal base;
    AffineExpr expr;
  };

public:
  AffineAccessBuilder(Operation *accessOp, bool legalizeSymbols)
      : AffineExprBuilder(accessOp, legalizeSymbols) {}

  PtrVal base = nullptr;

  LogicalResult build(const DataLayout &dataLayout, PtrVal addr) {
    auto aa = buildAffineAccess(dataLayout, addr);
    if (failed(aa))
      return failure();
    expr = aa->expr;
    base = aa->base;

    LLVM_DEBUG(llvm::dbgs() << "Built expr: " << expr << "\n");
    return success();
  }

  AffineExprBuilder::MapAndOperands getMap() {
    return AffineExprBuilder::getMap();
  }

  PtrVal getBase() {
    assert(base);
    return base;
  }

  void maybeReplaceBase(Value before, Value after) {
    if (base == before)
      base = cast<PtrVal>(after);
  }

  void rescope(enzymexla::AffineScopeOp scope) {
    if (!scope->isAncestor(user))
      return;
    rescopeExpr(scope);
  }

private:
  std::optional<AffineExpr> getGepAffineExpr(const DataLayout &dataLayout,
                                             LLVM::GEPOp gep) {
    // TODO what happens if we get a negative index
    auto indicesRange = gep.getIndices();
    auto indices = SmallVector<LLVM::GEPIndicesAdaptor<ValueRange>::value_type>(
        indicesRange.begin(), indicesRange.end());
    assert(indices.size() > 0);
    Type currentType = gep.getElemType();
    auto expr = getExpr(indices[0]);
    if (failed(expr))
      return std::nullopt;
    AffineExpr offset = (*expr) * dataLayout.getTypeSize(currentType);

    for (auto index : llvm::drop_begin(indices)) {
      bool shouldCancel =
          TypeSwitch<Type, bool>(currentType)
              .Case([&](LLVM::LLVMArrayType arrayType) {
                auto expr = getExpr(index);
                if (failed(expr))
                  return true;
                offset = offset + (*expr) * dataLayout.getTypeSize(
                                                arrayType.getElementType());
                currentType = arrayType.getElementType();
                return false;
              })
              .Case([&](LLVM::LLVMStructType structType) {
                ArrayRef<Type> body = structType.getBody();
                int64_t indexInt;
                auto constIndex = dyn_cast<IntegerAttr>(index);
                if (constIndex)
                  indexInt = constIndex.getInt();
                else
                  return true;

                for (uint32_t i : llvm::seq(indexInt)) {
                  if (!structType.isPacked())
                    offset = alignTo(offset,
                                     dataLayout.getTypeABIAlignment(body[i]));
                  offset = offset + dataLayout.getTypeSize(body[i]);
                }

                // Align for the current type as well.
                if (!structType.isPacked())
                  offset = alignTo(
                      offset, dataLayout.getTypeABIAlignment(body[indexInt]));
                currentType = body[indexInt];
                return false;
              })
              .Default([&](Type type) {
                LLVM_DEBUG(llvm::dbgs()
                           << "Unsupported type for offset computations" << type
                           << "\n");
                return true;
              });

      if (shouldCancel)
        return std::nullopt;
    }

    LLVM_DEBUG(llvm::dbgs() << "offset " << offset << "\n");

    return offset;
  }

  FailureOr<AffineAccess> buildAffineAccess(const DataLayout &dataLayout,
                                            PtrVal addr) {
    if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(addr.getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "gep " << gep << "\n");
      auto base = cast<PtrVal>(gep.getBase());

      auto gepExpr = getGepAffineExpr(dataLayout, gep);
      if (!gepExpr)
        return failure();

      auto aa = buildAffineAccess(dataLayout, base);
      if (failed(aa))
        return failure();

      AffineAccess newAA;
      newAA.base = aa->base;
      newAA.expr = aa->expr + *gepExpr;
      LLVM_DEBUG(llvm::dbgs() << "added " << newAA.expr << "\n");
      return newAA;
    } else if (auto addrSpaceCast = dyn_cast_or_null<LLVM::AddrSpaceCastOp>(
                   addr.getDefiningOp())) {
      return buildAffineAccess(dataLayout,
                               cast<PtrVal>(addrSpaceCast.getArg()));
    }

    AffineAccess aa;
    aa.base = addr;
    aa.expr = getAffineConstantExpr(0, addr.getContext());
    LLVM_DEBUG(llvm::dbgs() << "base " << aa.expr << "\n");
    return aa;
  }
};

struct AffineForBuilder {
public:
  AffineForBuilder(scf::ForOp forOp, bool legalizeSymbols)
      : lbBuilder(forOp, legalizeSymbols), ubBuilder(forOp, legalizeSymbols),
        forOp(forOp) {}

  AffineExprBuilder lbBuilder;
  AffineExprBuilder ubBuilder;

  scf::ForOp forOp;
  int64_t step;

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    lbBuilder.collectSymbolsForScope(region, symbols);
    ubBuilder.collectSymbolsForScope(region, symbols);
  }

  SmallPtrSet<Value, 4> getIllegalSymbols() {
    auto set = lbBuilder.illegalSymbols;
    set.insert(ubBuilder.illegalSymbols.begin(),
               ubBuilder.illegalSymbols.end());
    return set;
  }

  LogicalResult build() {
    auto cstStep = getConstant(forOp.getStep());
    if (!cstStep)
      return failure();
    step = *cstStep;

    if (failed(ubBuilder.build(forOp.getUpperBound())) ||
        failed(lbBuilder.build(forOp.getLowerBound())))
      return failure();

    return success();
  }

  AffineExprBuilder::MapAndOperands getUbMap() { return ubBuilder.getMap(); }

  AffineExprBuilder::MapAndOperands getLbMap() { return lbBuilder.getMap(); }

  int64_t getStep() { return step; }

  void rescope(enzymexla::AffineScopeOp scope) {
    if (!scope->isAncestor(forOp))
      return;
    SmallVector<AffineExpr> newExprs;

    lbBuilder.rescopeExpr(scope);
    ubBuilder.rescopeExpr(scope);
  }
};

struct AffineIfBuilder {
public:
  scf::IfOp ifOp;
  bool legalizeSymbols;
  AffineIfBuilder(scf::IfOp ifOp, bool legalizeSymbols)
      : ifOp(ifOp), legalizeSymbols(legalizeSymbols) {}

  struct Constraint {
    arith::CmpIPredicate pred;
    struct Side {
      Value val;
      AffineExprBuilder builder;
    };
    Side rhs, lhs;
  };

  struct SetAndOperands {
    IntegerSet set;
    SmallVector<Value> operands;
  } sao;

  SmallVector<Constraint, 0> constraints;

  LogicalResult build() {
    Value cond = ifOp.getCondition();

    if (failed(getConstraints(cond, constraints)))
      return failure();

    for (auto &c : constraints) {
      for (auto side : {&c.lhs, &c.rhs}) {
        auto &builder = side->builder;
        if (failed(builder.build(side->val)))
          return failure();
      }
    }

    return success();
  }

  void collectSymbolsForScope(Region *region, SmallPtrSetImpl<Value> &symbols) {
    for (auto &c : constraints) {
      c.lhs.builder.collectSymbolsForScope(region, symbols);
      c.rhs.builder.collectSymbolsForScope(region, symbols);
    }
  }

  SmallPtrSet<Value, 4> getIllegalSymbols() {
    SmallPtrSet<Value, 4> set;
    for (auto &c : constraints) {
      set.insert(c.lhs.builder.illegalSymbols.begin(),
                 c.lhs.builder.illegalSymbols.end());
      set.insert(c.rhs.builder.illegalSymbols.begin(),
                 c.rhs.builder.illegalSymbols.end());
    }
    return set;
  }

  void rescope(enzymexla::AffineScopeOp scope) {
    if (!scope->isAncestor(ifOp))
      return;
    SmallVector<AffineExpr> newExprs;

    for (auto &c : constraints) {
      c.lhs.builder.rescopeExpr(scope);
      c.rhs.builder.rescopeExpr(scope);
    }
  }

  SetAndOperands getSet() {
    SmallVector<bool> eqs;
    SmallVector<AffineExpr> exprs;
    unsigned numDims = 0;
    unsigned numSymbols = 0;
    SmallVector<Value> dimOperands;
    SmallVector<Value> symbolOperands;

    auto getExpr = [&](AffineExprBuilder &builder) {
      auto lhs = builder.getExpr();
      lhs = lhs.shiftDims(builder.dimOperands.size(), numDims);
      lhs = lhs.shiftSymbols(builder.symbolOperands.size(), numSymbols);
      numDims += builder.dimOperands.size();
      numSymbols += builder.symbolOperands.size();
      dimOperands.append(builder.dimOperands);
      symbolOperands.append(builder.symbolOperands);
      return lhs;
    };

    for (auto &c : constraints) {

      auto lhs = getExpr(c.lhs.builder);
      auto rhs = getExpr(c.rhs.builder);

      AffineExpr expr = getAffineConstantExpr(0, ifOp->getContext());
      switch (c.pred) {
      case arith::CmpIPredicate::eq:
        exprs.push_back(rhs - lhs);
        eqs.push_back(true);
        break;
      case arith::CmpIPredicate::ne:
        llvm_unreachable("no ne");
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        expr = expr - 1;
        [[fallthrough]];
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        expr = expr + lhs - rhs;
        exprs.push_back(expr);
        eqs.push_back(false);
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        expr = expr - 1;
        [[fallthrough]];
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        expr = expr + rhs - lhs;
        exprs.push_back(expr);
        eqs.push_back(false);
        break;
      }
    }
    sao.set = IntegerSet::get(numDims, numSymbols, exprs, eqs);
    sao.operands = dimOperands;
    sao.operands.append(symbolOperands);
    affine::canonicalizeSetAndOperands(&sao.set, &sao.operands);
    return sao;
  }

  LogicalResult getConstraints(Value conjunction,
                               SmallVectorImpl<Constraint> &constraints) {
    Operation *op = conjunction.getDefiningOp();
    if (!op)
      return failure();
    if (isa<LLVM::AndOp, arith::AndIOp>(op)) {
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      if (succeeded(getConstraints(lhs, constraints)) &&
          succeeded(getConstraints(rhs, constraints)))
        return success();
      else
        return failure();
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      // TODO there is a way to make this work with ne, but it is annoying to
      // think through, ingore for now.
      if (cmp.getPredicate() == arith::CmpIPredicate::ne)
        return failure();
      constraints.emplace_back(
          Constraint{cmp.getPredicate(),
                     {cmp.getLhs(), AffineExprBuilder(ifOp, legalizeSymbols)},
                     {cmp.getRhs(), AffineExprBuilder(ifOp, legalizeSymbols)}});
      return success();
    }
    return failure();
  }
};

// TODO this works for single-block regions where SSA values are not used across
// blocks but will fail when a value defined in `block` is used in another
// block.
static enzymexla::AffineScopeOp appendToScope(enzymexla::AffineScopeOp oldScope,
                                              ValueRange operands) {
  IRRewriter rewriter(oldScope);
  assert(llvm::all_of(operands, [&](Value a) {
    return llvm::all_of(oldScope->getOperands(),
                        [&](Value b) { return a != b; });
  }));
  SmallVector<Value> newOperands(oldScope->getOperands());
  Block *b = &oldScope.getRegion().front();
  for (Value v : operands) {
    if (llvm::find(newOperands, v) == newOperands.end()) {
      b->addArgument(v.getType(), v.getLoc());
      newOperands.push_back(v);
    }
  }
  auto scope = rewriter.create<enzymexla::AffineScopeOp>(
      oldScope.getLoc(), oldScope->getResultTypes(), newOperands);
  rewriter.inlineRegionBefore(oldScope.getRegion(), scope.getRegion(),
                              scope.getRegion().begin());
  rewriter.replaceOp(oldScope, scope);
  return scope;
}

template <typename T> SmallVector<Location> getLocs(T values) {
  return llvm::map_to_vector(values, [](Value v) { return v.getLoc(); });
}

static enzymexla::AffineScopeOp insertAffineScope(Block *block,
                                                  ValueRange operands) {
  assert(block->getParent()->getBlocks().size() == 1);

  assert(!isa<enzymexla::AffineScopeOp>(block->getParentOp()));
  if (auto scope = dyn_cast<enzymexla::AffineScopeOp>(block->front())) {
    assert(scope->getNextNode() == scope->getBlock()->getTerminator());
    return appendToScope(scope, operands);
  }

  IRRewriter rewriter(block->getParentOp()->getContext());
  rewriter.setInsertionPointToStart(block);
  auto scope = rewriter.create<enzymexla::AffineScopeOp>(
      block->getParentOp()->getLoc(), block->getTerminator()->getOperandTypes(),
      operands);
  Block *innerBlock = rewriter.createBlock(
      &scope.getRegion(), {}, operands.getTypes(), getLocs(operands));
  while (scope->getNextNode() != &block->back())
    rewriter.moveOpBefore(scope->getNextNode(), innerBlock, innerBlock->end());
  rewriter.setInsertionPointToEnd(innerBlock);
  Operation *terminator = block->getTerminator();
  rewriter.create<affine::AffineYieldOp>(terminator->getLoc(),
                                         terminator->getOperands());
  terminator->setOperands(scope->getResults());
  return scope;
}

static constexpr bool useVectorLoadStore = true;

static Value createVectorLoad(OpBuilder &b, Location loc, Type ty,
                              VectorType vty, MemRefVal m, AffineMap map,
                              ValueRange mapOperands) {
  if (useVectorLoadStore) {
    auto vl =
        b.create<affine::AffineVectorLoadOp>(loc, vty, m, map, mapOperands);
    vl->setAttr("polymer.access.type", TypeAttr::get(ty));
    return vl;
  }
  llvm_unreachable("");
}

/// Fold constant dimensions into an alloc like operation.
template <typename AllocLikeOp, bool gpu = false>
struct SimplifyAllocConst : public OpRewritePattern<AllocLikeOp> {
  using OpRewritePattern<AllocLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AllocLikeOp alloc,
                                PatternRewriter &rewriter) const override {
    // Check to see if any dimensions operands are constants.  If so, we can
    // substitute and drop them.
    if (llvm::none_of(alloc.getDynamicSizes(), [](Value operand) {
          APInt constSizeArg;
          if (!matchPattern(operand, m_ConstantInt(&constSizeArg)))
            return false;
          return constSizeArg.isNonNegative();
        }))
      return failure();

    auto memrefType = alloc.getType();

    // Ok, we have one or more constant operands.  Collect the non-constant ones
    // and keep track of the resultant memref type to build.
    SmallVector<int64_t, 4> newShapeConstants;
    newShapeConstants.reserve(memrefType.getRank());
    SmallVector<Value, 4> dynamicSizes;

    unsigned dynamicDimPos = 0;
    for (unsigned dim = 0, e = memrefType.getRank(); dim < e; ++dim) {
      int64_t dimSize = memrefType.getDimSize(dim);
      // If this is already static dimension, keep it.
      if (!ShapedType::isDynamic(dimSize)) {
        newShapeConstants.push_back(dimSize);
        continue;
      }
      auto dynamicSize = alloc.getDynamicSizes()[dynamicDimPos];
      APInt constSizeArg;
      if (matchPattern(dynamicSize, m_ConstantInt(&constSizeArg)) &&
          constSizeArg.isNonNegative()) {
        // Dynamic shape dimension will be folded.
        newShapeConstants.push_back(constSizeArg.getZExtValue());
      } else {
        // Dynamic shape dimension not folded; copy dynamicSize from old memref.
        newShapeConstants.push_back(ShapedType::kDynamic);
        dynamicSizes.push_back(dynamicSize);
      }
      dynamicDimPos++;
    }

    // Create new memref type (which will have fewer dynamic dimensions).
    MemRefType newMemRefType =
        MemRefType::Builder(memrefType).setShape(newShapeConstants);
    assert(dynamicSizes.size() == newMemRefType.getNumDynamicDims());

    // Create and insert the alloc op for the new memref.
    if constexpr (gpu) {
      auto newAlloc = rewriter.create<AllocLikeOp>(
          alloc.getLoc(), newMemRefType,
          alloc->getNumResults() == 1 ? nullptr : alloc->getResultTypes()[1],
          alloc.getAsyncDependencies(), dynamicSizes, alloc.getSymbolOperands(),
          alloc.getHostShared());
      // Insert a cast so we have the same type as the old alloc.
      rewriter.replaceOpWithNewOp<memref::CastOp>(alloc, alloc.getType(),
                                                  newAlloc->getResult(0));
    } else {
      auto newAlloc = rewriter.create<AllocLikeOp>(
          alloc.getLoc(), newMemRefType, dynamicSizes,
          alloc.getSymbolOperands(), alloc.getAlignmentAttr());
      // Insert a cast so we have the same type as the old alloc.
      rewriter.replaceOpWithNewOp<memref::CastOp>(alloc, alloc.getType(),
                                                  newAlloc);
    }
    return success();
  }
};

template <typename T, bool gpu = false>
struct SimplifyDeadAlloc : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T alloc,
                                PatternRewriter &rewriter) const override {
    for (auto op : alloc->getResult(0).getUsers()) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (storeOp.getValue() == alloc->getResult(0))
          return failure();
        continue;
      }
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        if (storeOp.getValue() == alloc->getResult(0))
          return failure();
        continue;
      }
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getValue() == alloc->getResult(0))
          return failure();
        continue;
      }

      if (auto cpy = dyn_cast<enzymexla::MemcpyOp>(op)) {
        if (cpy.getTarget() == alloc->getResult(0))
          return failure();
        continue;
      }

      if (isa<memref::DeallocOp>(op))
        continue;

      if (isa<LLVM::LifetimeStartOp>(op))
        continue;

      if (isa<LLVM::LifetimeEndOp>(op))
        continue;

      LLVM_DEBUG(llvm::errs() << "Alloc is not dead due to unknown user, alloc="
                              << *alloc << " user = " << *op << "\n");

      return failure();
    }

    if constexpr (gpu) {
      if (alloc.getAsyncToken()) {
        alloc->getResult(1).replaceAllUsesWith(alloc.getAsyncToken());
      }
    }

    for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
      rewriter.eraseOp(user);

    rewriter.eraseOp(alloc);
    return success();
  }
};

static bool definedOutsideOrAt(Value v, Operation *op) {
  return !op->isProperAncestor(v.getParentBlock()->getParentOp());
}

template <typename T> struct SimpleMem2Reg : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T alloc,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> stored;
    SmallVector<Operation *> loads;
    for (auto op : alloc->getUsers()) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (storeOp.getValue() == alloc)
          return failure();
        stored.push_back(storeOp.getValue());
        continue;
      }
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getValue() == alloc)
          return failure();
        stored.push_back(storeOp.getValue());
        continue;
      }

      if (isa<memref::DeallocOp>(op))
        continue;

      if (isa<LLVM::LifetimeStartOp>(op))
        continue;

      if (isa<LLVM::LifetimeEndOp>(op))
        continue;

      if (isa<affine::AffineLoadOp>(op) || isa<memref::LoadOp>(op)) {
        loads.push_back(op);
        continue;
      }

      LLVM_DEBUG(llvm::errs()
                 << "Alloc is not simple mem2reg due to unknown user, alloc="
                 << *alloc << " user = " << *op << "\n");

      return failure();
    }

    if (loads.size() != 1 || stored.size() != 1)
      return failure();

    DominanceInfo DI(alloc->getParentOp());
    bool changed = false;
    for (auto load : loads) {
      if (definedOutsideOrAt(stored[0], alloc->getParentOp()) ||
          DI.dominates(stored[0], load)) {
        rewriter.replaceOp(load, stored);
        changed = true;
        continue;
      }
    }

    return success(changed);
  }
};

namespace mlir {
LogicalResult
convertLLVMToAffineAccess(Operation *op,
                          const DataLayoutAnalysis &dataLayoutAnalysis,
                          bool legalizeSymbols) {
  if (!legalizeSymbols && !op->hasTrait<OpTrait::AffineScope>()) {
    LLVM_DEBUG(llvm::errs() << "Must be called with an affine scope root when "
                               "not legalizing symbols\n");
    return failure();
  }

  MLIRContext *context = op->getContext();

  MemrefConverter mc;
  IndexConverter ic;

  SmallVector<std::unique_ptr<AffineAccessBuilder>> accessBuilders;
  auto handleOp = [&](Operation *op, PtrVal addr) {
    LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << op
                            << " for address " << addr << "\n");
    accessBuilders.push_back(
        std::make_unique<AffineAccessBuilder>(op, legalizeSymbols));
    AffineAccessBuilder &aab = *accessBuilders.back();
    auto dl = dataLayoutAnalysis.getAtOrAbove(op);
    (void)aab.build(dl, addr);
  };
  op->walk([&](LLVM::StoreOp store) {
    PtrVal addr = store.getAddr();
    handleOp(store, addr);
  });
  op->walk([&](LLVM::LoadOp load) {
    PtrVal addr = load.getAddr();
    handleOp(load, addr);
  });
  // TODO should also gather other mem operations such as memory intrinsics
  // TODO should we shrink the scope to where no other memory operations
  // exist?

  if (legalizeSymbols) {
    SetVector<Block *> blocksToScope;
    for (auto &aabp : accessBuilders)
      for (auto illegalSym : aabp->illegalSymbols)
        blocksToScope.insert(illegalSym.getParentBlock());
    SetVector<Block *> innermostBlocks;
    for (Block *b : blocksToScope) {
      SmallVector<Block *> toRemove;
      bool isInnermost = true;
      for (Block *existing : innermostBlocks) {
        if (existing->getParent()->isProperAncestor(b->getParent()))
          toRemove.push_back(existing);
        if (b->getParent()->isAncestor(existing->getParent()))
          isInnermost = false;
      }
      for (Block *r : toRemove)
        innermostBlocks.remove(r);
      if (isInnermost)
        innermostBlocks.insert(b);
    }

    // TODO this looks terribly slow
    for (Block *b : innermostBlocks) {
      SmallPtrSet<Value, 6> symbols;
      for (auto &aabp : accessBuilders)
        aabp->collectSymbolsForScope(b->getParent(), symbols);
      SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
      auto scope = insertAffineScope(b, symbolsVec);
      for (auto &aabp : accessBuilders) {
        aabp->rescope(scope);
      }
    }
  }

  IRMapping mapping;
  for (auto &aabp : accessBuilders) {
    AffineAccessBuilder &aab = *aabp;
    // TODO add a test where some operations are left illegal

    auto isAligned = [&](auto op, llvm::TypeSize tySize) {
      if (auto alignment = op.getAlignment())
        return *alignment % tySize == 0;
      return false;
    };

    auto dl = dataLayoutAnalysis.getAtOrAbove(aab.user);
    if (auto load = dyn_cast<LLVM::LoadOp>(aab.user)) {
      IRRewriter rewriter(load);

      Type ty = load.getType();
      auto tySize = dl.getTypeSize(ty);
      if (MemRefType::isValidElementType(ty) && aab.isLegal() && aab.base) {
        auto memref0 = mc(aab.base);
        Value memref = memref0;
        auto memrefTy = memref0.getType();
        if (memrefTy.getElementType() != ty) {
          if (auto p2m = memref.getDefiningOp<enzymexla::Pointer2MemrefOp>())
            memref = p2m.getOperand();
          else
            memref = rewriter.create<enzymexla::Memref2PointerOp>(
                load.getLoc(), LLVM::LLVMPointerType::get(ty.getContext()),
                memref);
          memref = rewriter
                       .create<enzymexla::Pointer2MemrefOp>(
                           load.getLoc(),
                           MemRefType::get(memrefTy.getShape(), ty,
                                           MemRefLayoutAttrInterface{},
                                           memrefTy.getMemorySpace()),
                           memref)
                       .getResult();
        }

        auto mao = aab.getMap();
        if (mao.map.getResult(0).isMultipleOf(tySize) ||
            isAligned(load, tySize)) {
          auto expr = mao.map.getResult(0).floorDiv(tySize);
          SmallVector<NamedAttribute> attrs(load->getAttrs().begin(),
                                            load->getAttrs().end());
          auto newLoad = rewriter.create<affine::AffineLoadOp>(
              load.getLoc(), memref,
              AffineMap::get(mao.map.getNumDims(), mao.map.getNumSymbols(),
                             expr),
              ic(mao.operands));
          for (auto &a2 : accessBuilders) {
            a2->maybeReplaceBase(load, newLoad);
          }
          rewriter.replaceOp(load, newLoad);
          for (auto attr : attrs) {
            newLoad->setAttr(attr.getName(), attr.getValue());
          }
          continue;
        }
      }

      SmallVector<NamedAttribute> attrs(load->getAttrs().begin(),
                                        load->getAttrs().end());
      Value idxs[1] = {
          rewriter.create<arith::ConstantIndexOp>(load.getLoc(), 0)};
      auto newLoad = rewriter.create<memref::LoadOp>(
          load.getLoc(),
          rewriter.create<enzymexla::Pointer2MemrefOp>(
              load.getLoc(),
              MemRefType::get({ShapedType::kDynamic}, ty,
                              MemRefLayoutAttrInterface{},
                              rewriter.getIndexAttr(
                                  load.getAddr().getType().getAddressSpace())),
              load.getAddr()),
          idxs);
      for (auto &a2 : accessBuilders) {
        a2->maybeReplaceBase(load, newLoad);
      }
      rewriter.replaceOp(load, newLoad);
      for (auto attr : attrs) {
        newLoad->setAttr(attr.getName(), attr.getValue());
      }

    } else if (auto store = dyn_cast<LLVM::StoreOp>(aab.user)) {
      Type ty = store.getValue().getType();
      IRRewriter rewriter(store);
      auto tySize = dl.getTypeSize(ty);
      if (MemRefType::isValidElementType(ty) && aab.isLegal() && aab.base) {
        auto memref0 = mc(aab.base);
        Value memref = memref0;
        auto memrefTy = memref0.getType();
        if (memrefTy.getElementType() != ty) {
          if (auto p2m = memref.getDefiningOp<enzymexla::Pointer2MemrefOp>())
            memref = p2m.getOperand();
          else
            memref = rewriter.create<enzymexla::Memref2PointerOp>(
                store.getLoc(), LLVM::LLVMPointerType::get(ty.getContext()),
                memref);
          memref = rewriter
                       .create<enzymexla::Pointer2MemrefOp>(
                           store.getLoc(),
                           MemRefType::get(memrefTy.getShape(), ty,
                                           MemRefLayoutAttrInterface{},
                                           memrefTy.getMemorySpace()),
                           memref)
                       .getResult();
        }
        auto mao = aab.getMap();
        if (mao.map.getResult(0).isMultipleOf(tySize) ||
            isAligned(store, tySize)) {
          assert(cast<MemRefType>(memref.getType()).getElementType() ==
                 store.getValue().getType());
          auto expr = mao.map.getResult(0).floorDiv(tySize);
          SmallVector<NamedAttribute> attrs(store->getAttrs().begin(),
                                            store->getAttrs().end());
          auto newStore = rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              store, store.getValue(), memref,
              AffineMap::get(mao.map.getNumDims(), mao.map.getNumSymbols(),
                             expr),
              ic(mao.operands));
          for (auto attr : attrs) {
            newStore->setAttr(attr.getName(), attr.getValue());
          }
        }
        continue;
      }

      SmallVector<NamedAttribute> attrs(store->getAttrs().begin(),
                                        store->getAttrs().end());
      Value idxs[1] = {
          rewriter.create<arith::ConstantIndexOp>(store.getLoc(), 0)};
      auto newStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
          store, store.getValue(),
          rewriter.create<enzymexla::Pointer2MemrefOp>(
              store.getLoc(),
              MemRefType::get({ShapedType::kDynamic}, ty,
                              MemRefLayoutAttrInterface{},
                              rewriter.getIndexAttr(
                                  store.getAddr().getType().getAddressSpace())),
              store.getAddr()),
          idxs);
      for (auto attr : attrs) {
        newStore->setAttr(attr.getName(), attr.getValue());
      }
    } else {
      llvm_unreachable("Unknown operation to raise");
    }
  }

  {
    RewritePatternSet patterns(context);
    patterns.insert<ConvertLLVMAllocaToMemrefAlloca, GEPOfMemRefLoad,
                    SimplifyInPlaceAlloc<memref::AllocOp>,
                    SimplifyInPlaceAlloc<memref::AllocaOp>,
                    SimplifyInPlaceAlloc<gpu::AllocOp>>(context,
                                                        dataLayoutAnalysis);
    patterns.insert<IndexCastAddSub, MemrefLoadAffineApply, SelectCSE,
                    SelectAddrCast>(context);
    patterns.insert<SimplifyAllocConst<memref::AllocOp>,
                    SimplifyAllocConst<memref::AllocaOp>,
                    SimplifyAllocConst<gpu::AllocOp, true>>(context);
    patterns.insert<SimplifyDeadAlloc<memref::AllocaOp>,
                    SimplifyDeadAlloc<memref::AllocOp>,
                    SimplifyDeadAlloc<LLVM::AllocaOp>,
                    SimplifyDeadAlloc<gpu::AllocOp, true>, Pointer2MemrefSelect,
                    LoadSelect, SimpleMem2Reg<memref::AllocaOp>>(context);
    GreedyRewriteConfig config;
    if (applyPatternsAndFoldGreedily(op, std::move(patterns), config).failed())
      return failure();
  }
  return success();
}
} // namespace mlir

namespace mlir {
void populateRemoveIVPatterns(RewritePatternSet &patterns) {
  patterns.insert<mlir::transform::NormalizeLoop, mlir::transform::RemoveIVs>(
      patterns.getContext());
}
} // namespace mlir

// This should be scheduled on individual functions
struct LLVMToAffineAccessPass
    : public enzyme::impl::LLVMToAffineAccessPassBase<LLVMToAffineAccessPass> {
  using LLVMToAffineAccessPassBase::LLVMToAffineAccessPassBase;

  void runOnOperation() override {
    auto context = &getContext();
    RewritePatternSet patterns(context);
    populateRemoveIVPatterns(patterns);
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
      return;
    }
    Operation *op = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    // TODO in order to enable legalization we need to add an enzymexla.yield op
    // to terminate it with
    if (failed(convertLLVMToAffineAccess(op, dataLayoutAnalysis, false))) {
      signalPassFailure();
      return;
    }
  }
};
