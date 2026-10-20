//===- ImpulseBranchVectorize.cpp - Vectorize families of branch sites ----===//

#include "Enzyme/MLIR/Dialect/Impulse/Impulse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

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
