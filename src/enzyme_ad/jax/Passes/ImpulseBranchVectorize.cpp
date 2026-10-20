//===- ImpulseBranchVectorize.cpp - Vectorize families of branch sites ----===//

#include "Enzyme/MLIR/Dialect/Impulse/Impulse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "impulse-vectorize"

using namespace mlir;

namespace mlir {
namespace impulse {
bool vectorizeBranchFamilies(InferRegionOp regionOp);
} // namespace impulse
} // namespace mlir

namespace {

using impulse::IfOp;
using impulse::InferRegionOp;
using impulse::YieldOp;

bool isOneElement(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt && rt.hasStaticShape() && rt.getNumElements() == 1;
}

bool isShapeIdentityOp(Operation *op) {
  if (!isa<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp,
           impulse::ReshapeOp>(op))
    return false;
  return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
         isOneElement(op->getOperand(0).getType()) &&
         isOneElement(op->getResult(0).getType());
}

// Both branches run for every element; these ops have finite derivatives at
// finite inputs, so the untaken branch's zero cotangent stays zero.
bool isSafeElementwise(Operation *op) {
  return isa<stablehlo::AddOp, stablehlo::SubtractOp, stablehlo::MulOp,
             stablehlo::NegOp, stablehlo::AbsOp, stablehlo::MaxOp,
             stablehlo::MinOp, stablehlo::SelectOp, stablehlo::CompareOp,
             stablehlo::AndOp, stablehlo::OrOp, stablehlo::NotOp,
             stablehlo::ConvertOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
             arith::NegFOp>(op);
}

struct Member {
  IfOp op;
  SmallVector<Value> externals;
};

bool signatureOf(IfOp ifOp, std::string &sig, SmallVector<Value> &externals) {
  llvm::raw_string_ostream os(sig);
  DenseMap<Value, std::string> local;
  DenseMap<Value, unsigned> slotOf;
  auto ref = [&](Value v) -> std::string {
    auto it = local.find(v);
    if (it != local.end())
      return it->second;
    auto [sit, inserted] = slotOf.try_emplace(v, externals.size());
    if (inserted)
      externals.push_back(v);
    return "E" + std::to_string(sit->second);
  };
  os << ifOp.getPredicate().getType() << "|";
  for (Type t : ifOp.getResultTypes())
    os << t << ",";
  unsigned n = 0;
  for (Region *r : {&ifOp.getTrueBranch(), &ifOp.getFalseBranch()}) {
    os << "{";
    for (Operation &op : r->front()) {
      if (op.getNumRegions() != 0)
        return false;
      os << op.getName() << "(";
      for (Value v : op.getOperands())
        os << ref(v) << ",";
      os << ")" << op.getDiscardableAttrDictionary()
         << op.getPropertiesAsAttribute() << ":";
      for (Value res : op.getResults()) {
        os << res.getType() << ";";
        local[res] = "L" + std::to_string(n++);
      }
    }
    os << "}";
  }
  return true;
}

class FamilyRewriter {
public:
  FamilyRewriter(Block &block, ArrayRef<Member> members)
      : block(block), members(members.begin(), members.end()),
        K(members.size()) {}

  bool run();

private:
  Block &block;
  SmallVector<Member> members;
  int64_t K;
  SmallVector<bool> varyingSlot;

  RankedTensorType vecType(Type t) {
    return RankedTensorType::get({K},
                                 cast<RankedTensorType>(t).getElementType());
  }
  bool check();
  Value stack(OpBuilder &b, Location loc, ArrayRef<Value> vals);
  Value toScalar(OpBuilder &b, Location loc, Value v);
  Value broadcastToVec(OpBuilder &b, Location loc, Value uniform);
};

Value FamilyRewriter::stack(OpBuilder &b, Location loc, ArrayRef<Value> vals) {
  if (auto first = vals[0].getDefiningOp<stablehlo::SliceOp>()) {
    Value src = first.getOperand();
    auto srcType = cast<RankedTensorType>(src.getType());
    bool consecutive = srcType.getRank() == 1;
    int64_t start = consecutive ? first.getStartIndices()[0] : 0;
    for (auto [k, v] : llvm::enumerate(vals)) {
      auto s = v.getDefiningOp<stablehlo::SliceOp>();
      if (!consecutive || !s || s.getOperand() != src ||
          s.getStartIndices()[0] != start + static_cast<int64_t>(k) ||
          s.getLimitIndices()[0] != start + static_cast<int64_t>(k) + 1 ||
          s.getStrides()[0] != 1) {
        consecutive = false;
        break;
      }
    }
    if (consecutive && start == 0 && srcType.getDimSize(0) == K)
      return src;
    if (consecutive)
      return stablehlo::SliceOp::create(
          b, loc, vecType(src.getType()), src, b.getDenseI64ArrayAttr({start}),
          b.getDenseI64ArrayAttr({start + K}), b.getDenseI64ArrayAttr({1}));
  }
  SmallVector<Value> ones;
  for (Value v : vals) {
    auto t = cast<RankedTensorType>(v.getType());
    auto one = RankedTensorType::get({1}, t.getElementType());
    ones.push_back(
        t == one ? v
                 : stablehlo::ReshapeOp::create(b, loc, one, v).getResult());
  }
  return stablehlo::ConcatenateOp::create(b, loc, ones, 0);
}

Value FamilyRewriter::toScalar(OpBuilder &b, Location loc, Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getRank() == 0)
    return v;
  return stablehlo::ReshapeOp::create(
      b, loc, RankedTensorType::get({}, t.getElementType()), v);
}

Value FamilyRewriter::broadcastToVec(OpBuilder &b, Location loc,
                                     Value uniform) {
  Value s = toScalar(b, loc, uniform);
  return stablehlo::BroadcastInDimOp::create(b, loc, vecType(s.getType()), s,
                                             b.getDenseI64ArrayAttr({}));
}

bool FamilyRewriter::check() {
  unsigned numSlots = members[0].externals.size();
  varyingSlot.assign(numSlots, false);
  for (unsigned j = 0; j < numSlots; ++j)
    for (Member &m : members)
      if (m.externals[j] != members[0].externals[j])
        varyingSlot[j] = true;
  // Members are isomorphic, so checking the first one checks them all.
  IfOp ref = members[0].op;
  DenseMap<Value, unsigned> slotOf;
  for (auto [j, v] : llvm::enumerate(members[0].externals))
    slotOf[v] = j;
  DenseSet<Value> varying;
  auto isVarying = [&](Value v) {
    auto it = slotOf.find(v);
    if (it != slotOf.end())
      return bool(varyingSlot[it->second]);
    return varying.contains(v);
  };
  for (unsigned j = 0; j < numSlots; ++j)
    if (varyingSlot[j] && !isOneElement(members[0].externals[j].getType()))
      return false;
  if (!isOneElement(ref.getPredicate().getType()))
    return false;
  for (Region *r : {&ref.getTrueBranch(), &ref.getFalseBranch()})
    for (Operation &op : r->front().without_terminator()) {
      bool dep = llvm::any_of(op.getOperands(), isVarying);
      if (!dep) {
        if (!isMemoryEffectFree(&op))
          return false;
        continue;
      }
      for (Value res : op.getResults()) {
        if (!isOneElement(res.getType()))
          return false;
        varying.insert(res);
      }
      if (isShapeIdentityOp(&op))
        continue;
      if (auto div = dyn_cast<stablehlo::DivOp>(&op)) {
        if (isVarying(div.getRhs()))
          return false;
        continue;
      }
      if (!isSafeElementwise(&op))
        return false;
    }
  for (unsigned r = 0; r < ref.getNumResults(); ++r)
    if (llvm::any_of(
            members,
            [&](Member &m) { return !m.op.getResult(r).use_empty(); }) &&
        !isOneElement(ref.getResult(r).getType()))
      return false;
  return true;
}

bool FamilyRewriter::run() {
  if (!check())
    return false;
  Operation *first = members[0].op;
  for (Member &m : members)
    if (m.op->isBeforeInBlock(first))
      first = m.op;
  DenseSet<Operation *> memberOps;
  for (Member &m : members)
    memberOps.insert(m.op);

  SetVector<Operation *> toMove;
  SmallVector<Value> work;
  for (Member &m : members) {
    work.push_back(m.op.getPredicate());
    for (auto [j, v] : llvm::enumerate(m.externals))
      if (varyingSlot[j])
        work.push_back(v);
  }
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != &block || def->isBeforeInBlock(first) ||
        toMove.contains(def))
      continue;
    if (memberOps.contains(def) || def->getNumRegions() != 0 ||
        !isMemoryEffectFree(def))
      return false;
    toMove.insert(def);
    for (Value o : def->getOperands())
      work.push_back(o);
  }
  SmallVector<Operation *> moveList(toMove.begin(), toMove.end());
  llvm::sort(moveList,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : moveList)
    op->moveBefore(first);

  OpBuilder b(first);
  Location loc = first->getLoc();
  SmallVector<Value> preds;
  for (Member &m : members)
    preds.push_back(m.op.getPredicate());
  Value P = stack(b, loc, preds);

  IfOp ref = members[0].op;
  DenseMap<Value, unsigned> slotOf;
  for (auto [j, v] : llvm::enumerate(members[0].externals))
    slotOf[v] = j;
  SmallVector<Value> slotVec(varyingSlot.size());
  for (unsigned j = 0; j < varyingSlot.size(); ++j) {
    if (!varyingSlot[j])
      continue;
    SmallVector<Value> vals;
    for (Member &m : members)
      vals.push_back(m.externals[j]);
    slotVec[j] = stack(b, loc, vals);
  }

  auto lowerBranch = [&](Region &r, DenseMap<Value, Value> &vec,
                         IRMapping &uni) {
    auto get = [&](Value v, bool &isVec) -> Value {
      auto sit = slotOf.find(v);
      if (sit != slotOf.end()) {
        isVec = varyingSlot[sit->second];
        return isVec ? slotVec[sit->second] : v;
      }
      auto vit = vec.find(v);
      if (vit != vec.end()) {
        isVec = true;
        return vit->second;
      }
      isVec = false;
      return uni.lookupOrDefault(v);
    };
    for (Operation &op : r.front().without_terminator()) {
      bool anyVec = false;
      SmallVector<Value> ops;
      SmallVector<bool> opIsVec;
      for (Value o : op.getOperands()) {
        bool iv;
        ops.push_back(get(o, iv));
        opIsVec.push_back(iv);
        anyVec |= iv;
      }
      if (!anyVec) {
        b.clone(op, uni);
        continue;
      }
      if (isShapeIdentityOp(&op)) {
        vec[op.getResult(0)] = ops[0];
        continue;
      }
      for (unsigned i = 0; i < ops.size(); ++i)
        if (!opIsVec[i])
          ops[i] = broadcastToVec(b, loc, ops[i]);
      OperationState st(op.getLoc(), op.getName());
      st.addOperands(ops);
      for (Value res : op.getResults())
        st.addTypes(vecType(res.getType()));
      st.addAttributes(op.getDiscardableAttrDictionary().getValue());
      st.propertiesAttr = op.getPropertiesAsAttribute();
      Operation *nv = b.create(st);
      for (auto [res, nres] : llvm::zip(op.getResults(), nv->getResults()))
        vec[res] = nres;
    }
    auto yield = cast<YieldOp>(r.front().getTerminator());
    SmallVector<std::pair<Value, bool>> out;
    for (Value y : yield.getOperands()) {
      bool iv;
      Value v = get(y, iv);
      out.push_back({v, iv});
    }
    return out;
  };
  DenseMap<Value, Value> vecT, vecF;
  IRMapping uniT, uniF;
  auto outT = lowerBranch(ref.getTrueBranch(), vecT, uniT);
  auto outF = lowerBranch(ref.getFalseBranch(), vecF, uniF);

  for (unsigned r = 0; r < ref.getNumResults(); ++r) {
    if (llvm::all_of(members,
                     [&](Member &m) { return m.op.getResult(r).use_empty(); }))
      continue;
    Value tv =
        outT[r].second ? outT[r].first : broadcastToVec(b, loc, outT[r].first);
    Value fv =
        outF[r].second ? outF[r].first : broadcastToVec(b, loc, outF[r].first);
    Value sel = stablehlo::SelectOp::create(b, loc, P, tv, fv);
    for (auto [k, m] : llvm::enumerate(members)) {
      Value res = m.op.getResult(r);
      if (res.use_empty())
        continue;
      auto elemType = cast<RankedTensorType>(res.getType()).getElementType();
      Value e = stablehlo::SliceOp::create(
          b, loc, RankedTensorType::get({1}, elemType), sel,
          b.getDenseI64ArrayAttr({static_cast<int64_t>(k)}),
          b.getDenseI64ArrayAttr({static_cast<int64_t>(k) + 1}),
          b.getDenseI64ArrayAttr({1}));
      if (e.getType() != res.getType())
        e = stablehlo::ReshapeOp::create(b, loc, res.getType(), e);
      res.replaceAllUsesWith(e);
    }
  }
  for (Member &m : members)
    m.op->erase();
  return true;
}

} // namespace

bool mlir::impulse::vectorizeBranchFamilies(InferRegionOp regionOp) {
  // MH and Gibbs read per-site terms, which batching would tie to every flag.
  if (!regionOp.getHmcConfigAttr() && !regionOp.getNutsConfigAttr())
    return false;
  if (regionOp.getLogpdf().empty())
    return false;
  Block &block = regionOp.getLogpdf().front();

  for (Operation &op : block) {
    auto ifOp = dyn_cast<IfOp>(&op);
    if (!ifOp)
      continue;
    auto yt = cast<YieldOp>(ifOp.getTrueBranch().front().getTerminator());
    auto yf = cast<YieldOp>(ifOp.getFalseBranch().front().getTerminator());
    for (unsigned r = 0; r < ifOp.getNumResults(); ++r) {
      Value a = yt.getOperand(r), c = yf.getOperand(r);
      if (a != c || ifOp->isAncestor(a.getParentRegion()->getParentOp()) ||
          (a.getDefiningOp() && ifOp->isAncestor(a.getDefiningOp())))
        continue;
      ifOp.getResult(r).replaceAllUsesWith(a);
    }
  }

  // Recollect after each rewrite: other families may read replaced results.
  bool changed = false;
  while (true) {
    llvm::MapVector<StringRef, SmallVector<Member>> families;
    std::vector<std::unique_ptr<std::string>> keys;
    for (Operation &op : block) {
      auto ifOp = dyn_cast<IfOp>(&op);
      if (!ifOp)
        continue;
      auto sig = std::make_unique<std::string>();
      Member m{ifOp, {}};
      if (!signatureOf(ifOp, *sig, m.externals))
        continue;
      keys.push_back(std::move(sig));
      families[*keys.back()].push_back(std::move(m));
    }
    bool rewrote = false;
    for (auto &[key, members] : families) {
      if (members.size() < 2)
        continue;
      if (FamilyRewriter(block, members).run()) {
        rewrote = true;
        break;
      }
    }
    if (!rewrote)
      return changed;
    changed = true;
  }
}

namespace mlir {
namespace impulse {
bool vectorizeSiteFamilies(InferRegionOp regionOp);
} // namespace impulse
} // namespace mlir

namespace {

bool isFamilyOp(Operation *op) {
  if (op->getNumRegions() != 0 || !isMemoryEffectFree(op) ||
      op->getNumResults() != 1)
    return false;
  if (isShapeIdentityOp(op))
    return true;
  // stablehlo.select lacks the Elementwise trait (scalar predicate form).
  if (!op->hasTrait<OpTrait::Elementwise>() && !isa<stablehlo::SelectOp>(op))
    return false;
  for (Type t : op->getOperandTypes())
    if (!isOneElement(t))
      return false;
  return isOneElement(op->getResult(0).getType());
}

struct UnitSlice {
  Value src;
  SmallVector<int64_t> start;
};

std::optional<UnitSlice> asUnitSlice(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (isa<stablehlo::ReshapeOp, impulse::ReshapeOp>(op) &&
        isOneElement(op->getOperand(0).getType())) {
      v = op->getOperand(0);
      continue;
    }
    break;
  }
  ArrayRef<int64_t> start, limit, strides;
  Value src;
  if (auto s = v.getDefiningOp<stablehlo::SliceOp>()) {
    src = s.getOperand();
    start = s.getStartIndices();
    limit = s.getLimitIndices();
    strides = s.getStrides();
  } else if (auto s = v.getDefiningOp<impulse::SliceOp>()) {
    src = s->getOperand(0);
    start = s.getStartIndices();
    limit = s.getLimitIndices();
    strides = s.getStrides();
  } else {
    return std::nullopt;
  }
  for (auto [a, l, st] : llvm::zip(start, limit, strides))
    if (l != a + 1 || st != 1)
      return std::nullopt;
  return UnitSlice{src, SmallVector<int64_t>(start)};
}

Value stackValues(OpBuilder &b, Location loc, ArrayRef<Value> vals) {
  int64_t K = vals.size();
  auto elemType = cast<RankedTensorType>(vals[0].getType()).getElementType();
  auto vecTy = RankedTensorType::get({K}, elemType);
  SmallVector<UnitSlice> us;
  for (Value v : vals) {
    auto u = asUnitSlice(v);
    if (!u || (!us.empty() && u->src != us[0].src))
      break;
    us.push_back(*u);
  }
  if ((int64_t)us.size() == K && K >= 2) {
    int64_t rank = us[0].start.size(), dim = -1, stride = 0;
    bool ok = true;
    for (int64_t d = 0; d < rank && ok; ++d) {
      int64_t s = us[1].start[d] - us[0].start[d];
      if (s == 0)
        continue;
      if (dim != -1 || s < 1)
        ok = false;
      dim = d;
      stride = s;
    }
    for (int64_t k = 0; ok && dim != -1 && k < K; ++k)
      for (int64_t d = 0; d < rank; ++d)
        if (us[k].start[d] != us[0].start[d] + (d == dim ? stride * k : 0))
          ok = false;
    if (ok && dim != -1) {
      SmallVector<int64_t> start(us[0].start), limit, strides(rank, 1);
      for (int64_t d = 0; d < rank; ++d)
        limit.push_back(start[d] + 1);
      limit[dim] = start[dim] + stride * (K - 1) + 1;
      strides[dim] = stride;
      Value s = stablehlo::SliceOp::create(
          b, loc, us[0].src, b.getDenseI64ArrayAttr(start),
          b.getDenseI64ArrayAttr(limit), b.getDenseI64ArrayAttr(strides));
      return s.getType() == vecTy
                 ? s
                 : stablehlo::ReshapeOp::create(b, loc, vecTy, s).getResult();
    }
  }
  SmallVector<Value> ones;
  for (Value v : vals) {
    auto t = cast<RankedTensorType>(v.getType());
    auto one = RankedTensorType::get({1}, t.getElementType());
    ones.push_back(
        t == one ? v
                 : stablehlo::ReshapeOp::create(b, loc, one, v).getResult());
  }
  return stablehlo::ConcatenateOp::create(b, loc, ones, 0);
}

struct SiteMember {
  unsigned yieldIdx;
  SmallVector<Operation *> ops; // block order
  SmallVector<Value> externals;
  std::string sig;
};

/// Rewrites one family; false (and no change) when a precondition fails.
bool rewriteSiteFamily(InferRegionOp regionOp, Block &block,
                       ArrayRef<SiteMember> members) {
  int64_t K = members.size();
  unsigned numSlots = members[0].externals.size();
  SmallVector<bool> varying(numSlots, false);
  for (unsigned j = 0; j < numSlots; ++j)
    for (const SiteMember &m : members)
      if (m.externals[j] != members[0].externals[j])
        varying[j] = true;
  for (unsigned j = 0; j < numSlots; ++j)
    if (varying[j] && !isOneElement(members[0].externals[j].getType()))
      return false;

  DenseSet<Operation *> memberOps;
  Operation *first = nullptr;
  for (const SiteMember &m : members)
    for (Operation *op : m.ops) {
      memberOps.insert(op);
      if (!first || op->isBeforeInBlock(first))
        first = op;
    }

  SetVector<Operation *> toMove;
  SmallVector<Value> work;
  for (const SiteMember &m : members)
    for (Value v : m.externals)
      work.push_back(v);
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || def->getBlock() != &block || def->isBeforeInBlock(first) ||
        toMove.contains(def))
      continue;
    if (memberOps.contains(def) || def->getNumRegions() != 0 ||
        !isMemoryEffectFree(def))
      return false;
    toMove.insert(def);
    for (Value o : def->getOperands())
      work.push_back(o);
  }
  SmallVector<Operation *> moveList(toMove.begin(), toMove.end());
  llvm::sort(moveList,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : moveList)
    op->moveBefore(first);

  OpBuilder outer(regionOp);
  OpBuilder b(first);
  Location loc = first->getLoc();
  SmallVector<Value> slotVec(numSlots);
  for (unsigned j = 0; j < numSlots; ++j) {
    if (!varying[j])
      continue;
    SmallVector<Value> vals;
    bool outside = true;
    for (const SiteMember &m : members) {
      Value v = m.externals[j];
      vals.push_back(v);
      if (regionOp->isAncestor(v.getParentRegion()->getParentOp()))
        outside = false;
    }
    slotVec[j] = stackValues(outside ? outer : b, loc, vals);
  }

  auto vecType = [&](Type t) {
    return RankedTensorType::get({K},
                                 cast<RankedTensorType>(t).getElementType());
  };
  auto broadcastToVec = [&](Value u) -> Value {
    auto t = cast<RankedTensorType>(u.getType());
    Value s =
        t.getRank() == 0
            ? u
            : stablehlo::ReshapeOp::create(
                  b, loc, RankedTensorType::get({}, t.getElementType()), u)
                  .getResult();
    return stablehlo::BroadcastInDimOp::create(b, loc, vecType(s.getType()), s,
                                               b.getDenseI64ArrayAttr({}));
  };

  const SiteMember &ref = members[0];
  DenseMap<Value, unsigned> slotOf;
  for (auto [j, v] : llvm::enumerate(ref.externals))
    slotOf.try_emplace(v, j);
  DenseMap<Value, Value> vec; // vector or uniform value
  DenseSet<Value> isVec;
  IRMapping uni;
  for (Operation *op : ref.ops) {
    SmallVector<Value> ops;
    SmallVector<bool> opVec;
    bool anyVec = false;
    for (Value o : op->getOperands()) {
      Value r;
      bool iv = false;
      auto sit = slotOf.find(o);
      if (sit != slotOf.end()) {
        iv = varying[sit->second];
        r = iv ? slotVec[sit->second] : o;
      } else if (vec.count(o)) {
        r = vec[o];
        iv = isVec.contains(o);
      } else {
        r = uni.lookupOrDefault(o);
      }
      ops.push_back(r);
      opVec.push_back(iv);
      anyVec |= iv;
    }
    Value res = op->getResult(0);
    if (!anyVec) {
      Operation *c = b.clone(*op, uni);
      vec[res] = c->getResult(0);
      continue;
    }
    if (isShapeIdentityOp(op)) {
      vec[res] = ops[0];
      isVec.insert(res);
      continue;
    }
    for (unsigned i = 0; i < ops.size(); ++i)
      if (!opVec[i])
        ops[i] = broadcastToVec(ops[i]);
    OperationState st(op->getLoc(), op->getName());
    st.addOperands(ops);
    st.addTypes(vecType(res.getType()));
    st.addAttributes(op->getDiscardableAttrDictionary().getValue());
    st.propertiesAttr = op->getPropertiesAsAttribute();
    Operation *nv = b.create(st);
    vec[res] = nv->getResult(0);
    isVec.insert(res);
  }

  for (auto [k, m] : llvm::enumerate(members))
    for (auto [i, op] : llvm::enumerate(m.ops)) {
      Value r = op->getResult(0);
      bool external = llvm::any_of(
          r.getUsers(), [&](Operation *u) { return !memberOps.contains(u); });
      if (!external)
        continue;
      Value refVal = ref.ops[i]->getResult(0);
      Value v = vec[refVal];
      Value e;
      if (isVec.contains(refVal)) {
        // The MH lowering recognizes family yields by impulse slice/reshape.
        auto et = cast<RankedTensorType>(r.getType()).getElementType();
        e = impulse::SliceOp::create(
            b, loc, RankedTensorType::get({1}, et), v,
            b.getDenseI64ArrayAttr({static_cast<int64_t>(k)}),
            b.getDenseI64ArrayAttr({static_cast<int64_t>(k) + 1}),
            b.getDenseI64ArrayAttr({1}));
        if (e.getType() != r.getType())
          e = impulse::ReshapeOp::create(b, loc, r.getType(), e);
      } else {
        e = v;
      }
      r.replaceUsesWithIf(e, [&](OpOperand &use) {
        return !memberOps.contains(use.getOwner());
      });
    }
  SmallVector<Operation *> all(memberOps.begin(), memberOps.end());
  llvm::sort(all,
             [](Operation *a, Operation *b) { return b->isBeforeInBlock(a); });
  for (Operation *op : all)
    op->erase();
  return true;
}

} // namespace

bool mlir::impulse::vectorizeSiteFamilies(InferRegionOp regionOp) {
  if (!regionOp.getMhConfigAttr() || regionOp.getLogpdf().empty())
    return false;
  Block &block = regionOp.getLogpdf().front();
  Operation *term = block.getTerminator();

  DenseSet<Value> tainted(block.args_begin(), block.args_end());
  for (Operation &op : block.without_terminator())
    if (llvm::any_of(op.getOperands(),
                     [&](Value v) { return tainted.contains(v); }))
      tainted.insert(op.result_begin(), op.result_end());

  unsigned numYields = term->getNumOperands();
  SmallVector<SetVector<Operation *>> slices(numYields);
  DenseMap<Operation *, unsigned> count;
  for (unsigned j = 1; j < numYields; ++j) {
    SmallVector<Value> work = {term->getOperand(j)};
    while (!work.empty()) {
      Value v = work.pop_back_val();
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != &block || slices[j].contains(def))
        continue;
      slices[j].insert(def);
      for (Value o : def->getOperands())
        work.push_back(o);
    }
    for (Operation *op : slices[j])
      ++count[op];
  }

  llvm::MapVector<StringRef, SmallVector<SiteMember>> families;
  std::vector<std::unique_ptr<std::string>> keys;
  DenseSet<Value> seenYield;
  for (unsigned j = 1; j < numYields; ++j) {
    Value y = term->getOperand(j);
    if (!tainted.contains(y) || !seenYield.insert(y).second)
      continue;
    SiteMember m{j, {}, {}, {}};
    for (Operation *op : slices[j])
      if (count[op] == 1)
        m.ops.push_back(op);
    if (m.ops.size() < 2 || !llvm::is_contained(m.ops, y.getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "site yield " << j << ": " << m.ops.size()
                 << " own ops, yield op own: "
                 << llvm::is_contained(m.ops, y.getDefiningOp()) << "\n");
      continue;
    }
    llvm::sort(m.ops, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    if (!llvm::all_of(m.ops, isFamilyOp)) {
      LLVM_DEBUG({
        for (Operation *op : m.ops)
          if (!isFamilyOp(op))
            llvm::dbgs() << "site yield " << j << ": not a family op: " << *op
                         << "\n";
      });
      continue;
    }
    DenseSet<Operation *> own(m.ops.begin(), m.ops.end());
    DenseMap<Value, std::string> local;
    DenseMap<Value, unsigned> slotOf;
    llvm::raw_string_ostream os(m.sig);
    unsigned n = 0;
    for (Operation *op : m.ops) {
      os << op->getName() << "(";
      for (Value v : op->getOperands()) {
        auto it = local.find(v);
        if (it != local.end()) {
          os << it->second << ",";
          continue;
        }
        auto [sit, inserted] = slotOf.try_emplace(v, m.externals.size());
        if (inserted)
          m.externals.push_back(v);
        os << "E" << sit->second << ":" << v.getType() << ",";
      }
      os << ")" << op->getDiscardableAttrDictionary()
         << op->getPropertiesAsAttribute() << ":" << op->getResult(0).getType()
         << ";";
      local[op->getResult(0)] = "L" + std::to_string(n++);
    }
    os << "->" << local[y];
    os.flush();
    keys.push_back(std::make_unique<std::string>(m.sig));
    families[*keys.back()].push_back(std::move(m));
  }
  bool changed = false;
  for (auto &[key, members] : families) {
    LLVM_DEBUG(llvm::dbgs()
               << "site family of " << members.size()
               << " members, first yield " << members[0].yieldIdx << "\n");
    if (members.size() < 4)
      continue;
    bool ok = rewriteSiteFamily(regionOp, block, members);
    LLVM_DEBUG(llvm::dbgs() << "  rewritten: " << ok << "\n");
    changed |= ok;
  }
  return changed;
}
