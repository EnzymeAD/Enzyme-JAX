#include "src/enzyme_ad/jax/Passes/Julia/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Julia/Ops.h"

namespace mlir::enzymexla::julia {

#define GEN_PASS_DEF_JULIAFUSEOPSPASS
#include "src/enzyme_ad/jax/Passes/Julia/Passes.h.inc"

namespace {

static Value createIndexConst(OpBuilder &b, Location loc, int64_t v) {
  return ConstantOp::create(b, loc, b.getIndexType(), b.getIndexAttr(v));
}

static bool isScalarOrLoadFusibleOp(Operation *op) {
  return isa<ConstantOp, UnaryOp, BinaryOp, CmpOp, SelectOp, ClampOp, CastOp,
             BitcastOp, MathLibmOp, SizeOp, LoadOp>(op);
}

static bool isExpensiveUnary(UnaryOpKind kind) {
  switch (kind) {
  case UnaryOpKind::neg:
  case UnaryOpKind::abs:
  case UnaryOpKind::not_op:
  case UnaryOpKind::real:
  case UnaryOpKind::imag:
  case UnaryOpKind::conj:
  case UnaryOpKind::sign:
  case UnaryOpKind::is_finite:
  case UnaryOpKind::floor:
  case UnaryOpKind::ceil:
  case UnaryOpKind::round_nearest_afz:
  case UnaryOpKind::round_nearest_even:
  case UnaryOpKind::leading_zeros:
  case UnaryOpKind::popcnt:
    return false;
  default:
    return true;
  }
}

/// Returns true if `prod` is cheap enough to rematerialize into multiple
/// consumers (e.g. iota, scalar cast/scale/offset, comparison mask).
static bool isCheapRematerializableKernel(KernelOp prod) {
  if (prod.getNumResults() != 1)
    return false;
  int arithCount = 0;
  for (Operation &op : prod.getBody().front().without_terminator()) {
    if (!isScalarOrLoadFusibleOp(&op))
      return false;
    if (isa<MathLibmOp>(&op))
      return false;
    if (auto u = dyn_cast<UnaryOp>(&op)) {
      if (isExpensiveUnary(u.getOp()))
        return false;
    }
    if (!isa<ConstantOp>(&op))
      ++arithCount;
    if (arithCount > 4)
      return false;
  }
  return true;
}

static bool hasNoInterveningWrites(Operation *earlier, Operation *later) {
  Block *blk = earlier->getBlock();
  Operation *laterInBlk = blk->findAncestorOpInBlock(*later);
  if (!laterInBlk)
    return false;
  for (Operation *cur = earlier->getNextNode(); cur && cur != laterInBlk;
       cur = cur->getNextNode()) {
    if (!isMemoryEffectFree(cur))
      return false;
  }
  return true;
}

static SmallVector<int64_t> stripUnitDims(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> res;
  for (int64_t s : shape)
    if (s != 1)
      res.push_back(s);
  return res;
}

/// Local common-subexpression + dead-op cleanup within a single `KernelOp` body.
static void simplifyKernelBody(KernelOp kernel) {
  Block &body = kernel.getBody().front();

  auto makeKey = [](Operation *op) {
    llvm::SmallVector<void *, 8> key;
    key.push_back(const_cast<void *>(op->getName().getAsOpaquePointer()));
    key.push_back(
        const_cast<void *>(op->getAttrDictionary().getAsOpaquePointer()));
    for (Type t : op->getResultTypes())
      key.push_back(const_cast<void *>(t.getAsOpaquePointer()));
    for (Value v : op->getOperands())
      key.push_back(v.getAsOpaquePointer());
    return key;
  };

  SmallVector<std::pair<llvm::SmallVector<void *, 8>, Operation *>> seen;
  for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
    if (!isScalarOrLoadFusibleOp(&op) || op.getNumResults() != 1)
      continue;
    auto key = makeKey(&op);
    Operation *match = nullptr;
    for (auto &[prevKey, prevOp] : seen) {
      if (prevKey == key) {
        match = prevOp;
        break;
      }
    }
    if (match) {
      op.getResult(0).replaceAllUsesWith(match->getResult(0));
      op.erase();
    } else {
      seen.emplace_back(std::move(key), &op);
    }
  }

  // Reverse sweep to erase dead pure ops inside the kernel body.
  for (Operation *cur = body.getTerminator()->getPrevNode(); cur;) {
    Operation *prev = cur->getPrevNode();
    if (cur->use_empty() && isScalarOrLoadFusibleOp(cur))
      cur->erase();
    cur = prev;
  }
}

/// Fold `LoadOp(src, indices)` inside `consumer` when `src` is a pure
/// index/value producer (`FillOp`, `BroadcastInDimOp`, `TransposeOp`,
/// `SliceOp`, `ReverseOp`, unit-dim `ReshapeOp`, or fusible `KernelOp`).
static bool foldLoadsInKernel(KernelOp consumer) {
  bool anyChanged = false;
  Block &body = consumer.getBody().front();

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op :
         llvm::make_early_inc_range(body.without_terminator())) {
      auto loadOp = dyn_cast<LoadOp>(&op);
      if (!loadOp)
        continue;

      Value src = loadOp.getSource();
      Operation *prodOp = src.getDefiningOp();
      if (!prodOp || body.findAncestorOpInBlock(*prodOp))
        continue;

      OpBuilder b(loadOp);
      Location loc = loadOp.getLoc();
      SmallVector<Value> idx(loadOp.getIndices().begin(),
                             loadOp.getIndices().end());

      // 1. FillOp -> scalar value
      if (auto fillOp = dyn_cast<FillOp>(prodOp)) {
        loadOp.replaceAllUsesWith(fillOp.getValue());
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }

      // 2. BroadcastInDimOp -> scalar or projected indices
      if (auto bcast = dyn_cast<BroadcastInDimOp>(prodOp)) {
        Value operand = bcast.getOperand();
        auto opShapeTy = dyn_cast<ShapedType>(operand.getType());
        if (!opShapeTy) {
          loadOp.replaceAllUsesWith(operand);
          loadOp.erase();
          changed = anyChanged = true;
          continue;
        }
        ArrayRef<int64_t> bDims = bcast.getBroadcastDimensions();
        auto resTy = bcast.getType();
        SmallVector<Value> inIdx(opShapeTy.getRank());
        Value oneVal;
        for (int64_t d = 0; d < opShapeTy.getRank(); ++d) {
          int64_t outDim = bDims[d];
          bool isBroadcastedUnit =
              !opShapeTy.isDynamicDim(d) && opShapeTy.getDimSize(d) == 1 &&
              (resTy.isDynamicDim(outDim) || resTy.getDimSize(outDim) != 1);
          if (isBroadcastedUnit || opShapeTy.getDimSize(d) == 1) {
            if (!oneVal)
              oneVal = createIndexConst(b, loc, 1);
            inIdx[d] = oneVal;
          } else {
            inIdx[d] = idx[outDim];
          }
        }
        Value newLoad =
            LoadOp::create(b, loc, loadOp.getType(), operand, inIdx);
        loadOp.replaceAllUsesWith(newLoad);
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }

      // 3. TransposeOp (`out_dim[d] = in_dim[perm[d]]` -> `inIdx[perm[d]] = idx[d]`)
      if (auto tr = dyn_cast<TransposeOp>(prodOp)) {
        ArrayRef<int64_t> perm = tr.getPermutation();
        SmallVector<Value> inIdx(perm.size());
        for (size_t d = 0; d < perm.size(); ++d)
          inIdx[perm[d]] = idx[d];
        Value newLoad =
            LoadOp::create(b, loc, loadOp.getType(), tr.getOperand(), inIdx);
        loadOp.replaceAllUsesWith(newLoad);
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }

      // 4. SliceOp (`inIdx[d] = (idx[d] - 1) * stride[d] + (start[d] + 1)`)
      if (auto sl = dyn_cast<SliceOp>(prodOp)) {
        ArrayRef<int64_t> starts = sl.getStartIndices();
        ArrayRef<int64_t> strides = sl.getStrides();
        SmallVector<Value> inIdx(idx.size());
        for (size_t d = 0; d < idx.size(); ++d) {
          if (strides[d] == 1 && starts[d] == 0) {
            inIdx[d] = idx[d];
          } else if (strides[d] == 1) {
            Value off = createIndexConst(b, loc, starts[d]);
            inIdx[d] = BinaryOp::create(b, loc, b.getIndexType(),
                                        BinaryOpKind::add, idx[d], off);
          } else {
            Value one = createIndexConst(b, loc, 1);
            Value zeroBased = BinaryOp::create(b, loc, b.getIndexType(),
                                               BinaryOpKind::sub, idx[d], one);
            Value st = createIndexConst(b, loc, strides[d]);
            Value scaled = BinaryOp::create(b, loc, b.getIndexType(),
                                            BinaryOpKind::mul, zeroBased, st);
            Value base1 = createIndexConst(b, loc, starts[d] + 1);
            inIdx[d] = BinaryOp::create(b, loc, b.getIndexType(),
                                        BinaryOpKind::add, scaled, base1);
          }
        }
        Value newLoad =
            LoadOp::create(b, loc, loadOp.getType(), sl.getOperand(), inIdx);
        loadOp.replaceAllUsesWith(newLoad);
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }

      // 5. ReverseOp (`inIdx[d] = (size(op, d) + 1) - idx[d]` for `d in dims`)
      if (auto rev = dyn_cast<ReverseOp>(prodOp)) {
        Value operand = rev.getOperand();
        auto opTy = cast<ShapedType>(operand.getType());
        llvm::DenseSet<int64_t> revDims(rev.getDimensions().begin(),
                                        rev.getDimensions().end());
        SmallVector<Value> inIdx = idx;
        for (int64_t d : revDims) {
          Value nPlus1;
          if (!opTy.isDynamicDim(d)) {
            nPlus1 = createIndexConst(b, loc, opTy.getDimSize(d) + 1);
          } else {
            OpBuilder outerB(consumer);
            Value dimSz =
                SizeOp::create(outerB, loc, b.getIndexType(), operand, d);
            Value one = createIndexConst(b, loc, 1);
            nPlus1 = BinaryOp::create(b, loc, b.getIndexType(),
                                      BinaryOpKind::add, dimSz, one);
          }
          inIdx[d] = BinaryOp::create(b, loc, b.getIndexType(),
                                      BinaryOpKind::sub, nPlus1, idx[d]);
        }
        Value newLoad =
            LoadOp::create(b, loc, loadOp.getType(), operand, inIdx);
        loadOp.replaceAllUsesWith(newLoad);
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }

      // 6. Unit-dimension-only static ReshapeOp
      if (auto resh = dyn_cast<ReshapeOp>(prodOp)) {
        auto inTy = cast<ShapedType>(resh.getOperand().getType());
        auto outTy = resh.getType();
        if (inTy.hasStaticShape() && outTy.hasStaticShape() &&
            stripUnitDims(inTy.getShape()) == stripUnitDims(outTy.getShape())) {
          SmallVector<Value> nonUnitOut;
          for (int64_t d = 0; d < outTy.getRank(); ++d)
            if (outTy.getDimSize(d) != 1)
              nonUnitOut.push_back(idx[d]);
          SmallVector<Value> inIdx(inTy.getRank());
          Value oneVal;
          size_t pos = 0;
          for (int64_t d = 0; d < inTy.getRank(); ++d) {
            if (inTy.getDimSize(d) == 1) {
              if (!oneVal)
                oneVal = createIndexConst(b, loc, 1);
              inIdx[d] = oneVal;
            } else {
              inIdx[d] = nonUnitOut[pos++];
            }
          }
          Value newLoad = LoadOp::create(b, loc, loadOp.getType(),
                                         resh.getOperand(), inIdx);
          loadOp.replaceAllUsesWith(newLoad);
          loadOp.erase();
          changed = anyChanged = true;
          continue;
        }
      }

      // 7. Producer KernelOp vertical fusion
      if (auto prod = dyn_cast<KernelOp>(prodOp)) {
        if (!llvm::all_of(prod.getBody().front().without_terminator(),
                          [](Operation &inner) {
                            return isScalarOrLoadFusibleOp(&inner);
                          })) {
          continue;
        }
        if (prod->getBlock() == consumer->getBlock() &&
            !hasNoInterveningWrites(prod, consumer)) {
          continue;
        }

        // Fuse if all uses of `prod` are `LoadOp`s inside `consumer` (makes
        // `prod` completely dead!) OR `prod` is a cheap single-result kernel.
        bool allUsesInConsumer = llvm::all_of(
            prod->getUsers(), [&](Operation *user) {
              return isa<LoadOp>(user) &&
                     user->getParentOp() == consumer.getOperation();
            });
        if (!allUsesInConsumer && !isCheapRematerializableKernel(prod))
          continue;

        unsigned resIdx = cast<OpResult>(src).getResultNumber();
        Block &prodBody = prod.getBody().front();
        if (prodBody.getNumArguments() != idx.size())
          continue;

        IRMapping map;
        for (auto [arg, iv] : llvm::zip(prodBody.getArguments(), idx))
          map.map(arg, iv);
        for (Operation &inner : prodBody.without_terminator())
          b.clone(inner, map);
        auto prodYield = cast<YieldOp>(prodBody.getTerminator());
        Value inlinedVal = map.lookup(prodYield.getOperand(resIdx));
        loadOp.replaceAllUsesWith(inlinedVal);
        loadOp.erase();
        changed = anyChanged = true;
        continue;
      }
    }
  }

  if (anyChanged)
    simplifyKernelBody(consumer);
  return anyChanged;
}

/// Erase dead pure value-domain array operations in reverse order.
static void eraseDeadValueArrayOps(Block &block) {
  for (Operation *cur = block.getTerminator()->getPrevNode(); cur;) {
    Operation *prev = cur->getPrevNode();
    if (cur->use_empty() && isPure(cur))
      cur->erase();
    cur = prev;
  }
}

/// Promote remaining live `BroadcastInDimOp`, `TransposeOp`, and `ReverseOp`
/// operations into `FillOp` (for scalar broadcast) or `KernelOp` (for array
/// remappings) so they fuse with upstream producers, merge horizontally, and
/// bufferize in-place.
static bool promoteStandaloneRemapOps(Block &block) {
  bool promoted = false;
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    Location loc = op.getLoc();
    OpBuilder b(&op);

    if (auto bcast = dyn_cast<BroadcastInDimOp>(&op)) {
      if (bcast.use_empty())
        continue;
      Value operand = bcast.getOperand();
      auto resTy = bcast.getType();
      if (!isa<ShapedType>(operand.getType())) {
        Value filled =
            FillOp::create(b, loc, resTy, operand, bcast.getDynamicDims());
        bcast.replaceAllUsesWith(filled);
        bcast.erase();
        promoted = true;
        continue;
      }
      int64_t outRank = resTy.getRank();
      auto kern =
          KernelOp::create(b, loc, TypeRange{resTy}, bcast.getDynamicDims());
      Block *kBlk = b.createBlock(&kern.getBody());
      SmallVector<Value> ivs;
      for (int64_t d = 0; d < outRank; ++d)
        ivs.push_back(kBlk->addArgument(b.getIndexType(), loc));
      auto inTy = cast<ShapedType>(operand.getType());
      ArrayRef<int64_t> bDims = bcast.getBroadcastDimensions();
      SmallVector<Value> inIdx(inTy.getRank());
      Value oneVal;
      for (int64_t d = 0; d < inTy.getRank(); ++d) {
        if (!inTy.isDynamicDim(d) && inTy.getDimSize(d) == 1) {
          if (!oneVal)
            oneVal = createIndexConst(b, loc, 1);
          inIdx[d] = oneVal;
        } else {
          inIdx[d] = ivs[bDims[d]];
        }
      }
      Value val = LoadOp::create(b, loc, resTy.getElementType(), operand, inIdx);
      YieldOp::create(b, loc, ValueRange{val});
      bcast.replaceAllUsesWith(kern.getResult(0));
      bcast.erase();
      promoted = true;
      continue;
    }

    if (auto tr = dyn_cast<TransposeOp>(&op)) {
      if (tr.use_empty())
        continue;
      Value operand = tr.getOperand();
      auto resTy = tr.getType();
      int64_t rank = resTy.getRank();
      ArrayRef<int64_t> perm = tr.getPermutation();
      SmallVector<Value> dynDims;
      for (int64_t d = 0; d < rank; ++d) {
        if (resTy.isDynamicDim(d))
          dynDims.push_back(
              SizeOp::create(b, loc, b.getIndexType(), operand, perm[d]));
      }
      auto kern = KernelOp::create(b, loc, TypeRange{resTy}, dynDims);
      Block *kBlk = b.createBlock(&kern.getBody());
      SmallVector<Value> ivs;
      for (int64_t d = 0; d < rank; ++d)
        ivs.push_back(kBlk->addArgument(b.getIndexType(), loc));
      SmallVector<Value> inIdx(rank);
      for (int64_t d = 0; d < rank; ++d)
        inIdx[perm[d]] = ivs[d];
      Value val = LoadOp::create(b, loc, resTy.getElementType(), operand, inIdx);
      YieldOp::create(b, loc, ValueRange{val});
      tr.replaceAllUsesWith(kern.getResult(0));
      tr.erase();
      promoted = true;
      continue;
    }

    if (auto rev = dyn_cast<ReverseOp>(&op)) {
      if (rev.use_empty())
        continue;
      Value operand = rev.getOperand();
      auto resTy = rev.getType();
      int64_t rank = resTy.getRank();
      SmallVector<Value> dynDims;
      for (int64_t d = 0; d < rank; ++d) {
        if (resTy.isDynamicDim(d))
          dynDims.push_back(
              SizeOp::create(b, loc, b.getIndexType(), operand, d));
      }
      llvm::DenseSet<int64_t> revDims(rev.getDimensions().begin(),
                                      rev.getDimensions().end());
      llvm::DenseMap<int64_t, Value> dynBounds;
      for (int64_t d : revDims) {
        if (resTy.isDynamicDim(d))
          dynBounds[d] = SizeOp::create(b, loc, b.getIndexType(), operand, d);
      }
      auto kern = KernelOp::create(b, loc, TypeRange{resTy}, dynDims);
      Block *kBlk = b.createBlock(&kern.getBody());
      SmallVector<Value> ivs;
      for (int64_t d = 0; d < rank; ++d)
        ivs.push_back(kBlk->addArgument(b.getIndexType(), loc));
      SmallVector<Value> inIdx = ivs;
      for (int64_t d : revDims) {
        Value nPlus1;
        if (!resTy.isDynamicDim(d)) {
          nPlus1 = createIndexConst(b, loc, resTy.getDimSize(d) + 1);
        } else {
          Value one = createIndexConst(b, loc, 1);
          nPlus1 = BinaryOp::create(b, loc, b.getIndexType(),
                                    BinaryOpKind::add, dynBounds[d], one);
        }
        inIdx[d] = BinaryOp::create(b, loc, b.getIndexType(), BinaryOpKind::sub,
                                    nPlus1, ivs[d]);
      }
      Value val = LoadOp::create(b, loc, resTy.getElementType(), operand, inIdx);
      YieldOp::create(b, loc, ValueRange{val});
      rev.replaceAllUsesWith(kern.getResult(0));
      rev.erase();
      promoted = true;
      continue;
    }
  }
  return promoted;
}

//===----------------------------------------------------------------------===//
// Matmul Epilogue Fusion (`alpha * A * B + beta * C` -> 5-arg `mul!`)
//===----------------------------------------------------------------------===//

static Value createScalarNumericConst(OpBuilder &b, Location loc, Type elemTy,
                                      int64_t signVal) {
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    return ConstantOp::create(b, loc, ft,
                              b.getFloatAttr(ft, static_cast<double>(signVal)));
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    return ConstantOp::create(b, loc, it, b.getIntegerAttr(it, signVal));
  }
  if (auto ct = dyn_cast<ComplexType>(elemTy)) {
    Type sub = ct.getElementType();
    Value re = createScalarNumericConst(b, loc, sub, signVal);
    Value im = createScalarNumericConst(b, loc, sub, 0);
    return BinaryOp::create(b, loc, ct, BinaryOpKind::complex_make, re, im);
  }
  return Value();
}

static bool isDefinedOutsideKernel(Value v, KernelOp k) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return v.getParentRegion() != &k.getBody();
  return !k.getBody().isAncestor(def->getParentRegion());
}

static bool canMaterializeOutsideKernel(Value v, KernelOp k) {
  if (isDefinedOutsideKernel(v, k))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def || isa<LoadOp>(def) || def->getNumRegions() > 0 || !isPure(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value opnd) {
    return canMaterializeOutsideKernel(opnd, k);
  });
}

static Value materializeOutsideKernel(OpBuilder &b, Value v, KernelOp k,
                                      IRMapping &hoisted) {
  if (isDefinedOutsideKernel(v, k))
    return v;
  if (Value mapped = hoisted.lookupOrNull(v))
    return mapped;
  Operation *def = v.getDefiningOp();
  for (Value opnd : def->getOperands())
    materializeOutsideKernel(b, opnd, k, hoisted);
  Operation *cloned = b.clone(*def, hoisted);
  return cloned->getResult(0);
}

static bool matchIdentityLoad(Value v, KernelOp k, Value &arrOut) {
  auto load = v.getDefiningOp<LoadOp>();
  if (!load)
    return false;
  if (!isDefinedOutsideKernel(load.getSource(), k))
    return false;
  Block &body = k.getBody().front();
  if (load.getIndices().size() != body.getNumArguments())
    return false;
  for (auto [idx, arg] : llvm::zip(load.getIndices(), body.getArguments())) {
    if (idx != arg)
      return false;
  }
  arrOut = load.getSource();
  return true;
}

static bool matchScaledTerm(Value expr, KernelOp k, Value expectedArr,
                            bool requireExactArr, Value &matchedArr,
                            Value &scaleInside, bool &negated) {
  negated = false;
  scaleInside = Value();

  if (auto negOp = expr.getDefiningOp<UnaryOp>();
      negOp && negOp.getOp() == UnaryOpKind::neg) {
    Value innerArr, innerScale;
    bool innerNeg = false;
    if (matchScaledTerm(negOp.getInput(), k, expectedArr, requireExactArr,
                        innerArr, innerScale, innerNeg)) {
      matchedArr = innerArr;
      scaleInside = innerScale;
      negated = !innerNeg;
      return true;
    }
    return false;
  }

  Value arr;
  if (matchIdentityLoad(expr, k, arr)) {
    if (requireExactArr && arr != expectedArr)
      return false;
    if (!requireExactArr && arr == expectedArr)
      return false;
    matchedArr = arr;
    return true;
  }

  if (auto bin = expr.getDefiningOp<BinaryOp>();
      bin && bin.getOp() == BinaryOpKind::mul) {
    Value lhs = bin.getLhs();
    Value rhs = bin.getRhs();
    if (matchIdentityLoad(lhs, k, arr) && canMaterializeOutsideKernel(rhs, k)) {
      if (requireExactArr && arr != expectedArr)
        return false;
      if (!requireExactArr && arr == expectedArr)
        return false;
      matchedArr = arr;
      scaleInside = rhs;
      return true;
    }
    if (matchIdentityLoad(rhs, k, arr) && canMaterializeOutsideKernel(lhs, k)) {
      if (requireExactArr && arr != expectedArr)
        return false;
      if (!requireExactArr && arr == expectedArr)
        return false;
      matchedArr = arr;
      scaleInside = lhs;
      return true;
    }
  }
  return false;
}

static void fuseMatmulEpiloguesInBlock(Block &block) {
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    auto kern = dyn_cast<KernelOp>(&op);
    if (!kern || kern.getNumResults() != 1)
      continue;

    auto kernTy = cast<ArrayType>(kern.getResult(0).getType());
    Type elemTy = kernTy.getElementType();
    if (!isa<FloatType, IntegerType, ComplexType>(elemTy) ||
        elemTy.isInteger(1))
      continue;

    // Look for a single-consumer `MatmulOp` in the same block loaded inside `kern`.
    MatmulOp targetMm;
    for (Operation &inner : kern.getBody().front().without_terminator()) {
      auto load = dyn_cast<LoadOp>(&inner);
      if (!load)
        continue;
      auto mm = load.getSource().getDefiningOp<MatmulOp>();
      if (!mm || mm->getBlock() != &block)
        continue;
      if (!isa<ArrayType>(mm.getType()) || mm.getAcc() || mm.getAlpha() ||
          mm.getBeta())
        continue;
      if (!mm->hasOneUse())
        continue;
      if (!hasNoInterveningWrites(mm, kern))
        continue;
      targetMm = mm;
      break;
    }
    if (!targetMm)
      continue;

    Value mmVal = targetMm.getResult();
    auto yieldOp = cast<YieldOp>(kern.getBody().front().getTerminator());
    Value y = yieldOp.getOperand(0);

    Value matchedMm, matchedAcc;
    Value alphaIn, betaIn;
    bool negAlpha = false, negBeta = false;
    bool matched = false;

    // Case A: pure scaled matmul `alpha * M` or `-M`
    if (matchScaledTerm(y, kern, mmVal, /*requireExactArr=*/true, matchedMm,
                        alphaIn, negAlpha)) {
      matched = true;
    } else if (auto bin = y.getDefiningOp<BinaryOp>()) {
      if (bin.getOp() == BinaryOpKind::add ||
          bin.getOp() == BinaryOpKind::sub) {
        bool isSub = (bin.getOp() == BinaryOpKind::sub);
        Value lhs = bin.getLhs();
        Value rhs = bin.getRhs();
        if (matchScaledTerm(lhs, kern, mmVal, /*requireExactArr=*/true,
                            matchedMm, alphaIn, negAlpha) &&
            matchScaledTerm(rhs, kern, mmVal, /*requireExactArr=*/false,
                            matchedAcc, betaIn, negBeta)) {
          if (isSub)
            negBeta = !negBeta;
          matched = true;
        } else if (matchScaledTerm(rhs, kern, mmVal, /*requireExactArr=*/true,
                                   matchedMm, alphaIn, negAlpha) &&
                   matchScaledTerm(lhs, kern, mmVal, /*requireExactArr=*/false,
                                   matchedAcc, betaIn, negBeta)) {
          if (isSub)
            negAlpha = !negAlpha;
          matched = true;
        }
      }
    }

    if (!matched)
      continue;

    if (matchedAcc) {
      auto accShapeTy = dyn_cast<ShapedType>(matchedAcc.getType());
      if (!accShapeTy || !accShapeTy.hasStaticShape() || !kernTy.hasStaticShape() ||
          accShapeTy.getShape() != kernTy.getShape() ||
          accShapeTy.getElementType() != elemTy)
        continue;
    }

    OpBuilder b(kern);
    Location loc = kern.getLoc();
    IRMapping hoisted;

    auto buildScale = [&](Value scaleVal, bool neg, int64_t defaultSign) {
      if (!scaleVal)
        return createScalarNumericConst(b, loc, elemTy,
                                        neg ? -defaultSign : defaultSign);
      Value s = materializeOutsideKernel(b, scaleVal, kern, hoisted);
      if (neg)
        s = UnaryOp::create(b, loc, elemTy, UnaryOpKind::neg, s);
      return s;
    };

    Value alphaOut = buildScale(alphaIn, negAlpha, 1);
    Value betaOut = matchedAcc ? buildScale(betaIn, negBeta, 1)
                               : createScalarNumericConst(b, loc, elemTy, 0);
    if (!alphaOut || !betaOut)
      continue;

    auto newMm = MatmulOp::create(
        b, loc, kernTy, targetMm.getLhs(), targetMm.getRhs(), matchedAcc,
        alphaOut, betaOut, targetMm.getLhsTranspose(),
        targetMm.getRhsTranspose(), targetMm.getBatchDimsLhs(),
        targetMm.getBatchDimsRhs(), targetMm.getBatchDimsOut());
    kern.getResult(0).replaceAllUsesWith(newMm.getResult());
    kern.erase();
    targetMm.erase();
  }
}

//===----------------------------------------------------------------------===//
// Horizontal Sibling Kernel Fusion
//===----------------------------------------------------------------------===//

static void collectExternalValuesUsedInKernel(
    KernelOp kern, SmallVectorImpl<Value> &externals) {
  Region &bodyReg = kern.getBody();
  kern.walk([&](Operation *inner) {
    if (inner == kern.getOperation())
      return;
    for (Value v : inner->getOperands()) {
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        if (ba.getParentRegion() != &bodyReg)
          externals.push_back(v);
      } else if (Operation *def = v.getDefiningOp()) {
        if (!bodyReg.isAncestor(def->getParentRegion()))
          externals.push_back(v);
      }
    }
  });
}

static void fuseHorizontalKernelsInBlock(Block &block) {
  static constexpr unsigned kMaxCombinedOutputs = 8;

  // Hoist pure scalar constants to the top of the block so they never block
  // horizontal merging of sibling kernels.
  Operation *firstOp = &block.front();
  for (Operation &op : llvm::make_early_inc_range(block.without_terminator())) {
    if (isa<ConstantOp>(&op) && &op != firstOp)
      op.moveBefore(firstOp);
  }

  bool mergedAny = true;
  while (mergedAny) {
    mergedAny = false;
    SmallVector<KernelOp> kernels;
    for (Operation &op : block.without_terminator())
      if (auto k = dyn_cast<KernelOp>(&op))
        kernels.push_back(k);

    for (size_t i = 0; i < kernels.size() && !mergedAny; ++i) {
      KernelOp k1 = kernels[i];
      if (k1.getNumResults() == 0)
        continue;
      auto k1Ty = cast<ArrayType>(k1.getResult(0).getType());
      if (!k1Ty.hasStaticShape())
        continue;

      for (size_t j = i + 1; j < kernels.size(); ++j) {
        KernelOp k2 = kernels[j];
        if (k2.getNumResults() == 0 ||
            k1.getNumResults() + k2.getNumResults() > kMaxCombinedOutputs)
          continue;
        auto k2Ty = cast<ArrayType>(k2.getResult(0).getType());
        if (!k2Ty.hasStaticShape() || k2Ty.getShape() != k1Ty.getShape())
          continue;
        if (!hasNoInterveningWrites(k1, k2))
          continue;

        // Check that every external SSA value used inside `k2` dominates `k1`
        // (or is a pure zero-region scalar op in `block` whose operands
        // dominate `k1`, which we can hoist before `k1`).
        SmallVector<Value> ext2;
        collectExternalValuesUsedInKernel(k2, ext2);
        SmallVector<Operation *> hoistBeforeK1;
        llvm::DenseSet<Operation *> hoistSet;
        bool canDominate = true;
        for (Value v : ext2) {
          Operation *def = v.getDefiningOp();
          if (!def || def->getBlock() != &block || def->isBeforeInBlock(k1))
            continue;
          if (isa<ConstantOp, SizeOp>(def) &&
              llvm::all_of(def->getOperands(), [&](Value opnd) {
                Operation *od = opnd.getDefiningOp();
                return !od || od->getBlock() != &block ||
                       od->isBeforeInBlock(k1);
              })) {
            if (hoistSet.insert(def).second)
              hoistBeforeK1.push_back(def);
          } else {
            canDominate = false;
            break;
          }
        }
        if (!canDominate)
          continue;

        for (Operation *h : hoistBeforeK1)
          h->moveBefore(k1);

        // Merge `k1` and `k2` into a combined `KernelOp` at `k1`!
        OpBuilder b(k1);
        Location loc = k1.getLoc();
        SmallVector<Type> combinedResTypes;
        combinedResTypes.append(k1.getResultTypes().begin(),
                                k1.getResultTypes().end());
        combinedResTypes.append(k2.getResultTypes().begin(),
                                k2.getResultTypes().end());

        auto fused = KernelOp::create(b, loc, combinedResTypes, ValueRange{});
        Block *fBlk = b.createBlock(&fused.getBody());
        SmallVector<Value> ivs;
        for (int64_t d = 0; d < k1Ty.getRank(); ++d)
          ivs.push_back(fBlk->addArgument(b.getIndexType(), loc));

        SmallVector<Value> yielded;
        auto cloneBodyInto = [&](KernelOp srcK) {
          IRMapping map;
          Block &srcB = srcK.getBody().front();
          for (auto [arg, iv] : llvm::zip(srcB.getArguments(), ivs))
            map.map(arg, iv);
          for (Operation &inner : srcB.without_terminator())
            b.clone(inner, map);
          auto yOp = cast<YieldOp>(srcB.getTerminator());
          for (Value yv : yOp.getOperands())
            yielded.push_back(map.lookup(yv));
        };
        cloneBodyInto(k1);
        cloneBodyInto(k2);
        YieldOp::create(b, loc, yielded);
        simplifyKernelBody(fused);

        unsigned n1 = k1.getNumResults();
        for (unsigned r = 0; r < n1; ++r)
          k1.getResult(r).replaceAllUsesWith(fused.getResult(r));
        for (unsigned r = 0; r < k2.getNumResults(); ++r)
          k2.getResult(r).replaceAllUsesWith(fused.getResult(n1 + r));
        k2.erase();
        k1.erase();
        mergedAny = true;
        break;
      }
    }
  }
}

static void processBlock(Block &block, bool fuseHorizontal,
                         bool fuseMatmulEpilogue) {
  // Recursively optimize nested regions first (IfOp, WhileOp, etc.).
  for (Operation &op : block.without_terminator()) {
    if (isa<KernelOp>(&op))
      continue;
    for (Region &reg : op.getRegions())
      for (Block &sub : reg)
        processBlock(sub, fuseHorizontal, fuseMatmulEpilogue);
  }

  auto runVerticalSweep = [&]() {
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation &op : block.without_terminator()) {
        if (auto kern = dyn_cast<KernelOp>(&op)) {
          if (foldLoadsInKernel(kern))
            changed = true;
        }
      }
      eraseDeadValueArrayOps(block);
    }
  };

  // Phase 1: Vertical producer-into-consumer fusion.
  runVerticalSweep();

  // Phase 2: Promote remaining standalone BroadcastInDim / Transpose / Reverse
  // into FillOp / KernelOp and run vertical sweep again so e.g.
  // `transpose(sin(x) + y)` fuses into a single loop!
  if (promoteStandaloneRemapOps(block))
    runVerticalSweep();

  // Phase 3: Matmul epilogue fusion (`alpha * A * B + beta * C` -> 5-arg `mul!`).
  if (fuseMatmulEpilogue) {
    fuseMatmulEpiloguesInBlock(block);
    eraseDeadValueArrayOps(block);
  }

  // Phase 4: Horizontal sibling kernel fusion + intra-kernel CSE.
  if (fuseHorizontal) {
    fuseHorizontalKernelsInBlock(block);
    eraseDeadValueArrayOps(block);
  }

  for (Operation &op : block.without_terminator())
    if (auto kern = dyn_cast<KernelOp>(&op))
      simplifyKernelBody(kern);
}

struct JuliaFuseOpsPass
    : public impl::JuliaFuseOpsPassBase<JuliaFuseOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (FuncOp fn : mod.getOps<FuncOp>()) {
      if (fn.isExternal())
        continue;
      processBlock(fn.getBody().front(), fuse_horizontal, fuse_matmul_epilogue);
    }
  }
};

} // namespace

} // namespace mlir::enzymexla::julia
