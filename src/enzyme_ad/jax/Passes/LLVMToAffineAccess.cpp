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

static LogicalResult
convertLLVMAllocaToMemrefAlloca(LLVM::AllocaOp alloc, RewriterBase &rewriter,
                                const DataLayout &dataLayout) {
  if (!alloc.getRes().hasOneUse())
    return failure();

  auto sizeVal = getConstant(alloc.getArraySize());
  if (!sizeVal)
    return failure();

  Type elType = rewriter.getI8Type();
  int64_t elNum = dataLayout.getTypeSize(alloc.getElemType()) * (*sizeVal);

  auto ptr2memref = dyn_cast<enzymexla::Pointer2MemrefOp>(
      alloc.getRes().use_begin()->getOwner());
  if (!ptr2memref)
    return failure();

  assert(elType == ptr2memref.getResult().getType().getElementType());

  SmallVector<int64_t, 1> sizes = {elNum};
  auto memrefType =
      MemRefType::get(sizes, elType, MemRefLayoutAttrInterface{},
                      ptr2memref.getResult().getType().getMemorySpace());
  auto newAlloca =
      rewriter.create<memref::AllocaOp>(alloc->getLoc(), memrefType);
  rewriter.replaceAllUsesWith(ptr2memref.getResult(), newAlloca.getResult());
  rewriter.eraseOp(ptr2memref);
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
        rewriter.create<affine::AffineStoreOp>(store.getLoc(), value,
                                               getNewMemref(), newMap,
                                               store.getMapOperands());
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
} // namespace

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

  template <typename... Ts>
  inline FailureOr<AffineExpr> buildPassthrough(Operation *op) {
    if (isa<Ts...>(op)) {
      assert(op->getNumOperands() == 1);
      return buildExpr(op->getOperand(0));
    }
    return failure();
  }

  template <typename... Ts>
  inline FailureOr<AffineExpr>
  buildBinOpExpr(Operation *op,
                 AffineExpr (AffineExpr::*handler)(AffineExpr) const) {
    if (isa<Ts...>(op)) {
      assert(op->getNumOperands() == 2);
      auto lhs = buildExpr(op->getOperand(0));
      auto rhs = buildExpr(op->getOperand(1));
      if (failed(lhs) || failed(rhs))
        return failure();
      return ((*lhs).*handler)(*rhs);
    }
    return failure();
  }

  inline FailureOr<AffineExpr> buildOriAddOne(Operation *op) {
    auto ori = dyn_cast<arith::OrIOp>(op);
    if (!ori)
      return failure();
    auto lhs = buildExpr(op->getOperand(0));
    auto rhs = buildExpr(op->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!lhs->isMultipleOf(2))
      return failure();
    auto cstExpr = dyn_cast<AffineConstantExpr>(*rhs);
    if (!cstExpr || cstExpr.getValue() != 1)
      return failure();
    return *lhs + 1;
  }

  // TODO test this
  FailureOr<AffineExpr> buildShift(Operation *op) {
    if (op->getNumOperands() != 2)
      return failure();
    auto rhs = getConstant(op->getOperand(1));
    if (!rhs)
      return failure();
    auto lhs = buildExpr(op->getOperand(0));
    if (failed(lhs))
      return failure();
    if (isa<arith::ShLIOp, LLVM::ShlOp>(op)) {
      return (*lhs) * getAffineConstantExpr(1 << (*rhs), op->getContext());
    } else if (isa<arith::ShRUIOp, arith::ShRSIOp, LLVM::LShrOp, LLVM::AShrOp>(
                   op)) {
      return (*lhs).floorDiv(
          getAffineConstantExpr(1 << (*rhs), op->getContext()));
    }
    return failure();
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
      // clang-format off
#define RIS(X) do { auto res = X; if (succeeded(res)) return *res; } while (0)
      RIS((buildBinOpExpr<LLVM::AddOp, arith::AddIOp>(
               op, &AffineExpr::operator+)));
      RIS((buildBinOpExpr<LLVM::SubOp, arith::SubIOp>(
               op, &AffineExpr::operator-)));
      RIS((buildBinOpExpr<LLVM::URemOp, arith::RemSIOp, LLVM::SRemOp, arith::RemUIOp>(
               op, &AffineExpr::operator%)));
      // TODO need to check that we dont end up with dim * dim or other invalid
      // expression
      RIS((buildBinOpExpr<LLVM::MulOp, arith::MulIOp>(
               op, &AffineExpr::operator*)));
      RIS((buildBinOpExpr<LLVM::UDivOp, LLVM::SDivOp, arith::DivUIOp, arith::DivSIOp>(
               op, &AffineExpr::floorDiv)));
      RIS((buildPassthrough<
           LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp,
           arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
           arith::IndexCastOp, arith::IndexCastUIOp>(op)));
      RIS((buildShift(op)));
      RIS((buildOriAddOne(op)));
#undef RIS
      // clang-format on
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

static Operation *createVectorStore(OpBuilder &b, Location loc, Type ty,
                                    TypedValue<VectorType> v, MemRefVal m,
                                    AffineMap map, ValueRange mapOperands) {
  if (useVectorLoadStore) {
    auto vs =
        b.create<affine::AffineVectorStoreOp>(loc, v, m, map, mapOperands);
    vs->setAttr("polymer.access.type", TypeAttr::get(ty));
    return vs;
  }
  llvm_unreachable("");
}

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

namespace mlir {
LogicalResult
convertLLVMToAffineAccess(Operation *op,
                          const DataLayoutAnalysis &dataLayoutAnalysis,
                          bool legalizeSymbols) {
  if (!legalizeSymbols && !op->hasTrait<OpTrait::AffineScope>()) {
    LLVM_DEBUG(llvm::errs() << "Must be called with an affine scope root when "
                               "not legelizing symbols\n");
    return failure();
  }

  MLIRContext *context = op->getContext();

  MemrefConverter mc;
  IndexConverter ic;

  // TODO Pretty slow but annoying to implement as we wrap the operation in
  // the callback
  while (true) {
    auto res = op->walk<WalkOrder::PreOrder>([&](scf::ForOp forOp) {
      AffineForBuilder forBuilder(forOp, legalizeSymbols);
      if (failed(forBuilder.build()))
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "Converting\n" << forOp << "\n");
      if (legalizeSymbols) {
        SmallPtrSet<Block *, 8> blocksToScope;
        for (auto illegalSym : forBuilder.getIllegalSymbols())
          blocksToScope.insert(illegalSym.getParentBlock());
        for (Block *b : blocksToScope) {
          SmallPtrSet<Value, 6> symbols;
          forBuilder.collectSymbolsForScope(b->getParent(), symbols);
          SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
          auto scope = insertAffineScope(b, symbolsVec);
          forBuilder.rescope(scope);
        }
      }
      IRRewriter rewriter(forOp);
      auto lb = forBuilder.getLbMap();
      auto ub = forBuilder.getUbMap();
      auto affineForOp = rewriter.create<affine::AffineForOp>(
          forOp.getLoc(), ic(lb.operands), lb.map, ic(ub.operands), ub.map,
          forBuilder.getStep(), forOp.getInitArgs());
      if (!affineForOp.getRegion().empty())
        affineForOp.getRegion().front().erase();
      Block *block = forOp.getBody();
      SmallVector<Type> blockArgTypes = {rewriter.getIndexType()};
      auto iterArgTypes = forOp.getInitArgs().getTypes();
      blockArgTypes.insert(blockArgTypes.end(), iterArgTypes.begin(),
                           iterArgTypes.end());
      SmallVector<Location> blockArgLocs =
          getLocs(forOp.getBody()->getArguments());
      auto newBlock = rewriter.createBlock(&affineForOp.getRegion(), {},
                                           blockArgTypes, blockArgLocs);
      SmallVector<Value> newBlockArgs(newBlock->getArguments());
      auto origIVType = forOp.getInductionVar().getType();
      if (origIVType != rewriter.getIndexType()) {
        rewriter.setInsertionPointToStart(newBlock);
        newBlockArgs[0] = rewriter.create<arith::IndexCastOp>(
            newBlockArgs[0].getLoc(), origIVType, newBlockArgs[0]);
      }
      rewriter.inlineBlockBefore(block, newBlock, newBlock->end(),
                                 newBlockArgs);
      rewriter.replaceOp(forOp, affineForOp);
      auto yield = cast<scf::YieldOp>(newBlock->getTerminator());
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yield,
                                                         yield.getOperands());
      return WalkResult::interrupt();
    });
    if (!res.wasInterrupted())
      break;
  }

  while (true) {
    auto res = op->walk<WalkOrder::PreOrder>([&](scf::IfOp ifOp) {
      AffineIfBuilder ifBuilder(ifOp, legalizeSymbols);
      if (failed(ifBuilder.build()))
        return WalkResult::advance();
      LLVM_DEBUG(llvm::dbgs() << "Converting\n" << ifOp << "\n");
      if (legalizeSymbols) {
        SmallPtrSet<Block *, 8> blocksToScope;
        for (auto illegalSym : ifBuilder.getIllegalSymbols())
          blocksToScope.insert(illegalSym.getParentBlock());
        for (Block *b : blocksToScope) {
          SmallPtrSet<Value, 6> symbols;
          ifBuilder.collectSymbolsForScope(b->getParent(), symbols);
          SmallVector<Value, 6> symbolsVec(symbols.begin(), symbols.end());
          auto scope = insertAffineScope(b, symbolsVec);
          ifBuilder.rescope(scope);
        }
      }
      IRRewriter rewriter(ifOp);
      auto sao = ifBuilder.getSet();
      auto affineIfOp = rewriter.create<affine::AffineIfOp>(
          ifOp.getLoc(), ifOp.getResultTypes(), sao.set, ic(sao.operands),
          ifOp.elseBlock());
      for (auto [newRegion, oldRegion] :
           llvm::zip(affineIfOp.getRegions(), ifOp.getRegions())) {
        if (!newRegion->empty())
          newRegion->front().erase();
        if (oldRegion->empty())
          continue;
        Block *block = &oldRegion->front();
        auto newBlock = rewriter.createBlock(newRegion);
        rewriter.inlineBlockBefore(block, newBlock, newBlock->end(), {});
        auto yield = cast<scf::YieldOp>(newBlock->getTerminator());
        rewriter.setInsertionPoint(yield);
        rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(yield,
                                                           yield.getOperands());
      }
      rewriter.replaceOp(ifOp, affineIfOp);
      return WalkResult::interrupt();
    });
    if (!res.wasInterrupted())
      break;
  }

  SmallVector<std::unique_ptr<AffineAccessBuilder>> accessBuilders;
  auto handleOp = [&](Operation *op, PtrVal addr) {
    LLVM_DEBUG(llvm::dbgs() << "Building affine access for " << op
                            << " for address " << addr << "\n");
    accessBuilders.push_back(
        std::make_unique<AffineAccessBuilder>(op, legalizeSymbols));
    AffineAccessBuilder &aab = *accessBuilders.back();
    auto dl = dataLayoutAnalysis.getAtOrAbove(op);
    auto res = aab.build(dl, addr);
    if (failed(res))
      accessBuilders.pop_back();
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
    SmallPtrSet<Block *, 8> blocksToScope;
    for (auto &aabp : accessBuilders)
      for (auto illegalSym : aabp->illegalSymbols)
        blocksToScope.insert(illegalSym.getParentBlock());
    SmallPtrSet<Block *, 8> innermostBlocks;
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
        innermostBlocks.erase(r);
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
    if (!aab.isLegal())
      continue;

    auto mao = aab.getMap();

    auto dl = dataLayoutAnalysis.getAtOrAbove(aab.user);
    if (auto load = dyn_cast<LLVM::LoadOp>(aab.user)) {
      IRRewriter rewriter(load);
      auto vty = VectorType::get({(int64_t)dl.getTypeSize(load.getType())},
                                 rewriter.getI8Type());
      auto vecLoad =
          createVectorLoad(rewriter, load.getLoc(), load.getType(), vty,
                           mc(aab.getBase()), mao.map, ic(mao.operands));
      Operation *newLoad;
      if (isa<LLVM::LLVMPointerType>(load.getType())) {
        Type intTy = rewriter.getIntegerType(
            (int64_t)dl.getTypeSize(load.getType()) * 8);
        auto cast =
            rewriter.create<LLVM::BitcastOp>(load.getLoc(), intTy, vecLoad);
        newLoad = rewriter.create<LLVM::IntToPtrOp>(load.getLoc(),
                                                    load.getType(), cast);
      } else {
        newLoad = rewriter.create<LLVM::BitcastOp>(load.getLoc(),
                                                   load.getType(), vecLoad);
      }
      mapping.map(load, newLoad);
    } else if (auto store = dyn_cast<LLVM::StoreOp>(aab.user)) {
      Type ty = store.getValue().getType();
      IRRewriter rewriter(store);
      auto vty =
          VectorType::get({(int64_t)dl.getTypeSize(ty)}, rewriter.getI8Type());
      Value v;
      if (isa<LLVM::LLVMPointerType>(ty)) {
        Type intTy = rewriter.getIntegerType((int64_t)dl.getTypeSize(ty) * 8);
        v = rewriter.create<LLVM::PtrToIntOp>(store.getLoc(), intTy,
                                              store.getValue());
        v = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty, v);
      } else {
        v = rewriter.create<LLVM::BitcastOp>(store.getLoc(), vty,
                                             store.getValue());
      }
      Operation *newStore = createVectorStore(
          rewriter, store.getLoc(), ty, cast<TypedValue<VectorType>>(v),
          mc(aab.base), mao.map, ic(mao.operands));
      mapping.map(store.getOperation(), newStore);
    } else {
      llvm_unreachable("");
    }
  }

  IRRewriter rewriter(context);
  for (auto &&[oldOp, newOp] : mapping.getOperationMap()) {
    rewriter.replaceOp(oldOp, newOp);
  }

  {
    RewritePatternSet patterns(context);
    patterns.insert<ConvertToTypedMemref>(context, dataLayoutAnalysis);
    GreedyRewriteConfig config;
    config.fold = false;
    if (applyPatternsGreedily(op, std::move(patterns), config).failed())
      return failure();
  }
  {
    RewritePatternSet patterns(context);
    patterns.insert<ConvertLLVMAllocaToMemrefAlloca>(context,
                                                     dataLayoutAnalysis);
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
