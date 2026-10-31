//===- Widener.h - Rebuild a value on more rows along a dimension ---------===//
//
// A value computed on a window of rows is rebuilt on `lo` more rows below
// and `hi` more rows above along a dimension, over the same sources:
// slices move their windows (recomputing or rotating/padding a source that
// lacks the rows), elementwise ops, pads, broadcasts, iotas, unit-dimension
// reshapes, concatenates and dynamic update slices are rebuilt at the wider
// extent, and multislice/rotate/wrap custom calls follow their operand.
// Every op created is recorded so a failed attempt can be undone, and an
// op equivalent to one that already exists is reused. Used by the
// communication optimizer (DUSSinkSlicesComm, PadSinkSlicesComm: sink the
// slices of a DUS update down to the DUS so the partitioner moves no data)
// and by the late concat_pieces_to_select / pad_computed_to_select
// rewrites (rebuild a stencil's split boundary pieces over every row so
// the backend fuses one kernel).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {

// Defined in OptimizeCommunication.cpp.
int64_t getNumDevicesAlongDimension(const sdy::TensorShardingAttr &shardingAttr,
                                    int dimension, Operation *op);
std::optional<int64_t> parseMultiSliceConfigInt(StringRef cfg, StringRef key);
std::optional<SmallVector<int64_t>> parseMultiSliceConfigList(StringRef cfg,
                                                              StringRef key);
bool shiftedReadMovesData(int64_t m, int64_t a, int64_t s, int64_t d);
bool getConstantDUSIndex(Value idx, int64_t &out);

// Builds values whose dimension `dim` is `lo + size + hi` long and whose
// entry i along it equals entry i - lo of the original wherever both
// exist; the other entries are unspecified. `lo` and `hi` are usually
// non-negative (the value is widened), but either may be negative when a
// slice has to leave its operand on one side only: the operand is then
// recomputed on exactly the window needed, shifted, rather than widened
// on both sides and sliced again. Every op it creates is recorded so that
// a failed attempt can be undone.
struct RowWidener {
  PatternRewriter &rewriter;
  // The DUS being rewritten.
  Operation *anchor;
  // A window of a source that cannot be rebuilt, exactly the source's
  // size but shifted, is read through a rotation of the source (the
  // cheapest halo exchange when it is sharded) when set; through a pad
  // otherwise.
  bool rotateWholeWindow = true;
  SmallVector<Operation *> created;
  DenseSet<Operation *> createdSet;
  DenseMap<std::tuple<Value, int64_t, int64_t, int64_t>, Value> cache;
  // The other results of a widened multi-result call: reused by later
  // widenings, but their narrow versions keep their other users (the
  // narrow call stays for them, so a redirect would only add a shift).
  DenseMap<std::tuple<Value, int64_t, int64_t, int64_t>, Value> aliases;
  // The DUS window along the dimension being widened: the target's extent,
  // the window's start and size. Rows of the widened values outside the
  // window are never read, so only the window decides whether a fallback
  // may move data: it may when the original program had to move data to
  // bring the source's rows into the window's layout.
  int64_t winExtent = 0, winOff = 0, winSize = 0;

  // Does placing source rows shifted by `shift` (target row i reads source
  // row i + shift) into the window move data between the `d` devices?
  bool windowReadMovesData(int64_t m, int64_t shift, int64_t d) const {
    if (d <= 1)
      return false;
    int64_t cm = (m + d - 1) / d, ct = (winExtent + d - 1) / d;
    for (int64_t k = 0; k < d; k++) {
      int64_t t0 = std::max(winOff, k * ct);
      int64_t t1 = std::min(winOff + winSize, (k + 1) * ct);
      if (t0 >= t1)
        continue;
      if (t0 + shift < k * cm || t1 + shift > std::min(m, (k + 1) * cm))
        return true;
    }
    return false;
  }

  RowWidener(PatternRewriter &rewriter, Operation *anchor)
      : rewriter(rewriter), anchor(anchor) {}

  void rollback() {
    for (Operation *op : llvm::reverse(created))
      rewriter.eraseOp(op);
    created.clear();
    createdSet.clear();
  }

  Value track(Value v) {
    created.push_back(v.getDefiningOp());
    createdSet.insert(v.getDefiningOp());
    return v;
  }

  // An op equivalent to `cand` that already exists ahead of it, if any.
  // Looked for among the users of the first non-constant operand. One
  // placed after `cand` cannot be reused (the ops being built sit between
  // the two), but `cand` can take its place once the widening has
  // succeeded: it is remembered for `dedupe`. (The anchors are visited
  // from the last up, so an earlier anchor's copy is usually below.)
  Operation *findExisting(Operation *cand) {
    for (Value operand : cand->getOperands()) {
      if (matchPattern(operand, m_Constant()) ||
          operand.getDefiningOp<sdy::ConstantOp>())
        continue;
      int seen = 0;
      SmallVector<Operation *> below;
      for (Operation *user : operand.getUsers()) {
        if (++seen > 256)
          break;
        if (user == cand || user == anchor ||
            user->getBlock() != cand->getBlock())
          continue;
        if (!OperationEquivalence::isEquivalentTo(
                user, cand, OperationEquivalence::IgnoreLocations))
          continue;
        if (user->isBeforeInBlock(cand))
          return user;
        if (!createdSet.contains(user))
          below.push_back(user);
      }
      for (Operation *user : below)
        duplicates.emplace_back(user, cand);
      break;
    }
    return nullptr;
  }

  // Equivalent ops below a created op, replaced by it once the widening
  // has succeeded (a failed one erases the created op instead).
  SmallVector<std::pair<Operation *, Operation *>> duplicates;
  void dedupe() {
    DenseSet<Operation *> gone;
    for (auto [below, cand] : duplicates) {
      if (gone.contains(below) || !createdSet.contains(cand) ||
          !cand->isBeforeInBlock(below))
        continue;
      rewriter.replaceOp(below, cand->getResults());
      gone.insert(below);
    }
    duplicates.clear();
  }

  // Records a freshly created op, or swaps it for an existing equivalent.
  Value finish(Value v) {
    Operation *op = v.getDefiningOp();
    if (Operation *existing = findExisting(op)) {
      rewriter.eraseOp(op);
      return existing->getResult(0);
    }
    return track(v);
  }

  static RankedTensorType widenedType(RankedTensorType ty, int64_t dim,
                                      int64_t lo, int64_t hi) {
    SmallVector<int64_t> shape(ty.getShape());
    shape[dim] += lo + hi;
    return RankedTensorType::get(shape, ty.getElementType());
  }

  static void copySharding(Value from, Value to) {
    if (auto sharding = sdy::getSharding(from))
      sdy::setSharding(to, sharding);
  }

  Value splat(Location loc, RankedTensorType ty, int64_t val, Value like) {
    Value res = stablehlo::ConstantOp::create(
        rewriter, loc, cast<ElementsAttr>(makeAttr(ty, val)));
    copySharding(like, res);
    return finish(res);
  }

  // Moves the window of a slice of `x`. When the new window leaves `x`,
  // `x` is recomputed on exactly that window.
  Value widenSlice(Operation *op, Value x, int64_t dim,
                   SmallVector<int64_t> starts, SmallVector<int64_t> limits,
                   ArrayRef<int64_t> strides, int64_t lo, int64_t hi) {
    if (strides[dim] != 1)
      return nullptr;
    auto xTy = cast<RankedTensorType>(x.getType());
    int64_t a = starts[dim], b = limits[dim], m = xTy.getDimSize(dim);
    int64_t na = a - lo, nb = b + hi;
    int64_t devices = 1;
    if (auto xs = sdy::getSharding(x))
      devices = getNumDevicesAlongDimension(xs, dim, op);
    // Target row i of the widened value reads source row i + na.
    const bool originMoves = windowReadMovesData(m, na, devices);
    // When `x` itself is read shifted (the window moved, or a fallback
    // moves a halo of it), first take what the slice keeps along the
    // unsharded dimensions (the levels of a field), so that the shift is
    // of that slab and not of the whole array: the partitioner exchanges
    // the halo of its operand before slicing the other dimensions.
    auto preSlice = [&]() {
      SmallVector<int64_t> preStarts(xTy.getRank(), 0),
          preLimits(xTy.getShape()), preStrides(xTy.getRank(), 1);
      bool pre = false;
      for (int64_t d = 0; d < xTy.getRank(); d++) {
        if (d == dim || strides[d] != 1 ||
            (starts[d] == 0 && limits[d] == xTy.getDimSize(d)))
          continue;
        int64_t dev = 1;
        if (auto xs = sdy::getSharding(x))
          dev = getNumDevicesAlongDimension(xs, d, op);
        if (dev > 1)
          continue;
        preStarts[d] = starts[d];
        preLimits[d] = limits[d];
        starts[d] = 0;
        limits[d] = limits[d] - preStarts[d];
        pre = true;
      }
      if (!pre)
        return;
      Value xs = stablehlo::SliceOp::create(rewriter, op->getLoc(), x,
                                            preStarts, preLimits, preStrides);
      copySharding(x, xs);
      x = finish(xs);
      xTy = cast<RankedTensorType>(x.getType());
    };
    Value xw = x;
    if (na < 0 || nb > m) {
      xw = widen(x, dim, -na, nb - m);
      if (xw) {
        nb -= na;
        na = 0;
      } else {
        preSlice();
        xw = x;
        // A pad whose operand is that slab would only be rebuilt as itself
        // (the slab is whole along `dim`): nothing to sink.
        if (isa<stablehlo::PadOp>(anchor) && anchor->getOperand(0) == x)
          return nullptr;
        if (nb - na == m && rotateWholeWindow) {
          // `x` cannot be recomputed on the wider window (an argument, or an
          // op the widener does not know), but the window has its size: the
          // entries that would come from outside `x` are never read, so a
          // rotation of `x` serves, and is the cheapest exchange of a halo.
          // Rotations of one value by several amounts are later grouped.
          // Never trade a read that moved no data for one that does.
          int64_t amount = ((na % m) + m) % m;
          if (amount != 0 && !originMoves) {
            return nullptr;
          }
          xw = enzymexla::RotateOp::create(rewriter, op->getLoc(), x,
                                           static_cast<int32_t>(amount),
                                           static_cast<int32_t>(dim));
          copySharding(x, xw);
          xw = finish(xw);
          na = 0;
          nb = m;
        } else {
          // Otherwise pad: the padded entries are never read either. Again
          // only if that does not move data where the original did not.
          {
            int64_t padLo = std::max<int64_t>(0, -na),
                    padHi = std::max<int64_t>(0, nb - m);
            int64_t padded = m + padLo + padHi;
            // The padded tensor's own layout against the source's, then the
            // window read out of it.
            bool moves =
                shiftedReadMovesData(padded, na + padLo, nb - na, devices);
            int64_t cp = (padded + devices - 1) / devices,
                    cm = (m + devices - 1) / devices;
            for (int64_t k = 0; k < devices && !moves; k++) {
              int64_t r0 = std::max<int64_t>(0, k * cp - padLo);
              int64_t r1 = std::min(m, (k + 1) * cp - padLo);
              if (r0 < r1 && (r0 < k * cm || r1 > std::min(m, (k + 1) * cm)))
                moves = true;
            }
            if (moves && !originMoves) {
              return nullptr;
            }
          }
          Value zero = finish(stablehlo::ConstantOp::create(
              rewriter, op->getLoc(),
              cast<ElementsAttr>(makeAttr(
                  RankedTensorType::get({}, xTy.getElementType()), 0))));
          SmallVector<int64_t> low(xTy.getRank(), 0), high(xTy.getRank(), 0),
              interior(xTy.getRank(), 0);
          low[dim] = std::max<int64_t>(0, -na);
          high[dim] = std::max<int64_t>(0, nb - m);
          xw = stablehlo::PadOp::create(rewriter, op->getLoc(), x, zero, low,
                                        high, interior);
          copySharding(x, xw);
          xw = finish(xw);
          na += low[dim];
          nb += low[dim];
        }
      }
    } else if (originMoves) {
      preSlice();
      xw = x;
    }
    starts[dim] = na;
    limits[dim] = nb;
    auto xwTy = cast<RankedTensorType>(xw.getType());
    bool whole = llvm::all_of(strides, [](int64_t s) { return s == 1; });
    for (int64_t d = 0; d < xwTy.getRank() && whole; d++)
      whole = starts[d] == 0 && limits[d] == xwTy.getDimSize(d);
    if (whole)
      return xw;
    Value res = stablehlo::SliceOp::create(rewriter, op->getLoc(), xw, starts,
                                           limits, strides);
    copySharding(op->getResult(0), res);
    return finish(res);
  }

  Value widen(Value v, int64_t dim, int64_t lo, int64_t hi) {
    if (lo == 0 && hi == 0)
      return v;
    auto key = std::make_tuple(v, dim, lo, hi);
    if (auto it = cache.find(key); it != cache.end())
      return it->second;
    if (auto it = aliases.find(key); it != aliases.end())
      return it->second;
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || !ty.hasStaticShape())
      return nullptr;
    if (ty.getDimSize(dim) + lo + hi <= 0)
      return nullptr;
    Operation *op = v.getDefiningOp();
    if (!op)
      return nullptr;
    auto loc = op->getLoc();
    auto newTy = widenedType(ty, dim, lo, hi);
    // The wide value goes right after the narrow one, so that it is
    // available wherever the narrow one was.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);

    Value res;
    if (isa<stablehlo::ConstantOp, sdy::ConstantOp>(op)) {
      auto attr = cast<ElementsAttr>(op->getAttr("value"));
      if (auto splat = dyn_cast<SplatElementsAttr>(attr)) {
        auto newAttr =
            SplatElementsAttr::get(newTy, splat.getSplatValue<Attribute>());
        if (isa<sdy::ConstantOp>(op))
          res = sdy::ConstantOp::create(rewriter, loc, newAttr);
        else
          res = stablehlo::ConstantOp::create(rewriter, loc, newAttr);
        copySharding(v, res);
        res = finish(res);
      } else {
        // Drop the rows that leave the window, and pad zeros into the rows
        // that are never read; both fold on a constant.
        int64_t size = ty.getDimSize(dim);
        if (-lo >= size || -hi >= size)
          return nullptr;
        Value cur = v;
        if (lo < 0 || hi < 0) {
          SmallVector<int64_t> starts(ty.getRank(), 0), limits(ty.getShape()),
              strides(ty.getRank(), 1);
          starts[dim] = std::max<int64_t>(0, -lo);
          limits[dim] = size - std::max<int64_t>(0, -hi);
          cur = stablehlo::SliceOp::create(rewriter, loc, cur, starts, limits,
                                           strides);
          copySharding(v, cur);
          cur = finish(cur);
        }
        if (lo > 0 || hi > 0) {
          SmallVector<int64_t> low(ty.getRank(), 0), high(ty.getRank(), 0),
              interior(ty.getRank(), 0);
          low[dim] = std::max<int64_t>(0, lo);
          high[dim] = std::max<int64_t>(0, hi);
          auto zero = track(stablehlo::ConstantOp::create(
              rewriter, loc,
              cast<ElementsAttr>(makeAttr(
                  RankedTensorType::get({}, ty.getElementType()), 0))));
          cur = stablehlo::PadOp::create(rewriter, loc, cur, zero, low, high,
                                         interior);
          copySharding(v, cur);
          cur = finish(cur);
        }
        res = cur;
      }
    } else if (auto slice = dyn_cast<stablehlo::SliceOp>(op)) {
      res = widenSlice(
          op, slice.getOperand(), dim, llvm::to_vector(slice.getStartIndices()),
          llvm::to_vector(slice.getLimitIndices()), slice.getStrides(), lo, hi);
    } else if (auto ms = dyn_cast<enzymexla::MultiSliceOp>(op)) {
      // Result i is the slice shifted by i along the multislice dimension.
      int64_t shift = cast<OpResult>(v).getResultNumber();
      SmallVector<int64_t> starts(ms.getStartIndices()),
          limits(ms.getLimitIndices());
      starts[ms.getDimension()] += shift;
      limits[ms.getDimension()] += shift;
      res = widenSlice(op, ms.getOperand(), dim, starts, limits,
                       ms.getStrides(), lo, hi);
    } else if (auto cc = dyn_cast<stablehlo::CustomCallOp>(op);
               cc && cc.getNumOperands() == 1 &&
               (cc.getCallTargetName() == "_SPMDInternalOp_MultiSlice" ||
                cc.getCallTargetName() == "_SPMDInternalOp_MultiRotate" ||
                cc.getCallTargetName() == "_SPMDInternalOp_Wrap")) {
      // The lowered forms of a multislice, multirotate or wrap.
      auto cfg = cc->getAttrOfType<StringAttr>("backend_config");
      if (!cfg)
        return nullptr;
      auto ccDim = parseMultiSliceConfigInt(cfg.getValue(), "dimension");
      if (!ccDim)
        return nullptr;
      bool isMultiSlice =
          cc.getCallTargetName() == "_SPMDInternalOp_MultiSlice";
      int64_t idx = cast<OpResult>(v).getResultNumber();
      if (isMultiSlice && *ccDim == dim) {
        // Result i is the slice shifted by i along the multislice dimension.
        auto starts =
            parseMultiSliceConfigList(cfg.getValue(), "start_indices");
        auto limits =
            parseMultiSliceConfigList(cfg.getValue(), "limit_indices");
        auto strides = parseMultiSliceConfigList(cfg.getValue(), "strides");
        if (!starts || !limits || !strides || *ccDim < 0 ||
            *ccDim >= ty.getRank() || (int64_t)starts->size() != ty.getRank() ||
            (int64_t)limits->size() != ty.getRank() ||
            (int64_t)strides->size() != ty.getRank())
          return nullptr;
        (*starts)[*ccDim] += idx;
        (*limits)[*ccDim] += idx;
        res = widenSlice(op, cc.getOperand(0), dim, *starts, *limits, *strides,
                         lo, hi);
      } else if (*ccDim == dim) {
        // A rotation or wrap along the widened dimension itself.
        return nullptr;
      } else {
        // Along another dimension: the same call on the widened operand.
        // Every result widens with it; a multislice's window grows along
        // the widened dimension by what was added.
        Value x = widen(cc.getOperand(0), dim, lo, hi);
        if (!x)
          return nullptr;
        std::string newCfg = cfg.getValue().str();
        if (isMultiSlice) {
          auto limits =
              parseMultiSliceConfigList(cfg.getValue(), "limit_indices");
          if (!limits || dim >= (int64_t)limits->size())
            return nullptr;
          (*limits)[dim] += lo + hi;
          size_t pos = newCfg.find("limit_indices=[");
          size_t end = newCfg.find(']', pos);
          if (pos == std::string::npos || end == std::string::npos)
            return nullptr;
          std::string list;
          for (auto [i, l] : llvm::enumerate(*limits))
            list += (i ? ", " : "") + std::to_string(l);
          newCfg.replace(pos + strlen("limit_indices=["),
                         end - pos - strlen("limit_indices=["), list);
        }
        SmallVector<Type> resTys;
        for (Value r : cc.getResults())
          resTys.push_back(
              widenedType(cast<RankedTensorType>(r.getType()), dim, lo, hi));
        OperationState state(loc, cc->getName());
        state.addOperands(x);
        state.addTypes(resTys);
        for (NamedAttribute attr : cc->getAttrs())
          if (attr.getName() == "backend_config")
            state.addAttribute(attr.getName(), rewriter.getStringAttr(newCfg));
          else
            state.addAttribute(attr.getName(), attr.getValue());
        Operation *ncc = rewriter.create(state);
        if (Operation *existing = findExisting(ncc)) {
          rewriter.eraseOp(ncc);
          ncc = existing;
        } else {
          track(ncc->getResult(0));
        }
        // The other results are widened by the same call.
        for (auto [j, r] : llvm::enumerate(cc.getResults()))
          if ((int64_t)j != idx)
            aliases[std::make_tuple(r, dim, lo, hi)] = ncc->getResult(j);
        res = ncc->getResult(idx);
      }
    } else if (auto pad = dyn_cast<stablehlo::PadOp>(op)) {
      if (pad.getInteriorPadding()[dim] != 0)
        return nullptr;
      SmallVector<int64_t> low(pad.getEdgePaddingLow()),
          high(pad.getEdgePaddingHigh());
      low[dim] += lo;
      high[dim] += hi;
      res = stablehlo::PadOp::create(rewriter, loc, pad.getOperand(),
                                     pad.getPaddingValue(), low, high,
                                     pad.getInteriorPadding());
      copySharding(v, res);
      res = finish(res);
    } else if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
      Value x = bcast.getOperand();
      auto xTy = cast<RankedTensorType>(x.getType());
      int64_t srcDim = -1;
      for (auto [i, d] : llvm::enumerate(bcast.getBroadcastDimensions()))
        if (d == dim)
          srcDim = i;
      if (srcDim >= 0 && xTy.getDimSize(srcDim) != 1) {
        x = widen(x, srcDim, lo, hi);
        if (!x)
          return nullptr;
      }
      res = stablehlo::BroadcastInDimOp::create(rewriter, loc, newTy, x,
                                                bcast.getBroadcastDimensions());
      copySharding(v, res);
      res = finish(res);
    } else if (auto iota = dyn_cast<stablehlo::IotaOp>(op)) {
      res = stablehlo::IotaOp::create(rewriter, loc, newTy,
                                      iota.getIotaDimension());
      copySharding(v, res);
      res = finish(res);
      if ((int64_t)iota.getIotaDimension() == dim && lo != 0) {
        res = stablehlo::SubtractOp::create(rewriter, loc, res,
                                            splat(loc, newTy, lo, v));
        copySharding(v, res);
        res = finish(res);
      }
    } else if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
      // A reshape that only inserts or removes unit dimensions: the
      // widened dimension maps to the operand's dimension of the same
      // size, in order among the non-unit dimensions.
      auto inTy = cast<RankedTensorType>(reshape.getOperand().getType());
      int64_t srcDim = -1;
      {
        int64_t j = 0;
        bool ok = true;
        for (int64_t i = 0; i < ty.getRank() && ok; i++) {
          if (ty.getDimSize(i) == 1)
            continue;
          while (j < inTy.getRank() && inTy.getDimSize(j) == 1)
            j++;
          if (j >= inTy.getRank() || inTy.getDimSize(j) != ty.getDimSize(i))
            ok = false;
          else if (i == dim)
            srcDim = j;
          j++;
        }
        for (; j < inTy.getRank() && ok; j++)
          if (inTy.getDimSize(j) != 1)
            ok = false;
        if (!ok || srcDim < 0)
          return nullptr;
      }
      Value x = widen(reshape.getOperand(), srcDim, lo, hi);
      if (!x)
        return nullptr;
      res = stablehlo::ReshapeOp::create(rewriter, loc, newTy, x);
      copySharding(v, res);
      res = finish(res);
    } else if (auto concat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
      bool along = (int64_t)concat.getDimension() == dim;
      if (along && (lo < 0 || hi < 0))
        return nullptr;
      SmallVector<Value> operands;
      int64_t last = concat.getNumOperands() - 1;
      for (auto [i, operand] : llvm::enumerate(concat.getOperands())) {
        Value w = widen(operand, dim, (!along || i == 0) ? lo : 0,
                        (!along || (int64_t)i == last) ? hi : 0);
        if (!w)
          return nullptr;
        operands.push_back(w);
      }
      res = stablehlo::ConcatenateOp::create(rewriter, loc, operands,
                                             concat.getDimension());
      copySharding(v, res);
      res = finish(res);
    } else if (auto dus = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op)) {
      // The widened DUS writes the same update at a shifted window; it is
      // then a candidate for this pattern itself.
      int64_t off;
      if (!getConstantDUSIndex(dus.getStartIndices()[dim], off))
        return nullptr;
      off = std::clamp<int64_t>(off, 0,
                                ty.getDimSize(dim) -
                                    dus.getUpdate().getType().getDimSize(dim));
      if (off + lo < 0 || off + lo + dus.getUpdate().getType().getDimSize(dim) >
                              newTy.getDimSize(dim))
        return nullptr;
      Value base = widen(dus.getOperand(), dim, lo, hi);
      if (!base)
        return nullptr;
      SmallVector<Value> starts(dus.getStartIndices());
      auto idxTy = cast<RankedTensorType>(starts[dim].getType());
      starts[dim] = track(stablehlo::ConstantOp::create(
          rewriter, loc, cast<ElementsAttr>(makeAttr(idxTy, off + lo))));
      res = stablehlo::DynamicUpdateSliceOp::create(rewriter, loc, base,
                                                    dus.getUpdate(), starts);
      copySharding(v, res);
      res = finish(res);
    } else if (op->getNumResults() == 1 && op->getNumRegions() == 0 &&
               (op->hasTrait<OpTrait::Elementwise>() ||
                isa<stablehlo::SelectOp, stablehlo::ClampOp>(op))) {
      SmallVector<Value> operands;
      for (Value operand : op->getOperands()) {
        auto oTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!oTy)
          return nullptr;
        if (oTy.getRank() == 0) {
          operands.push_back(operand);
          continue;
        }
        if (oTy.getShape() != ty.getShape())
          return nullptr;
        Value w = widen(operand, dim, lo, hi);
        if (!w)
          return nullptr;
        operands.push_back(w);
      }
      OperationState state(loc, op->getName());
      state.addOperands(operands);
      state.addTypes(newTy);
      state.addAttributes(op->getAttrs());
      res = finish(rewriter.create(state)->getResult(0));
    } else {
      return nullptr;
    }
    // A failed attempt is not cached: a caller may still succeed by other
    // means (the rotate or pad of a source that could not be widened).
    if (res)
      cache[key] = res;
    return res;
  }

  // Redirects the users of a narrow intermediate outside the widened chain
  // to a slice of its wide version, so that the narrow chain dies and
  // nothing is computed twice. Called once widening has succeeded.
  void sinkOtherUses() {
    DenseSet<Operation *> narrowOps;
    for (auto &[key, wide] : cache)
      if (Operation *op = std::get<0>(key).getDefiningOp())
        narrowOps.insert(op);
    auto inChain = [&](Operation *user) {
      return user == anchor || createdSet.contains(user) ||
             narrowOps.contains(user);
    };
    for (auto &[key, wide] : cache) {
      auto [v, dim, lo, hi] = key;
      Operation *op = v.getDefiningOp();
      // A shifted version does not contain the narrow one.
      if (!wide || wide == v || lo < 0 || hi < 0 || !op ||
          isa<stablehlo::ConstantOp, sdy::ConstantOp>(op))
        continue;
      if (llvm::none_of(v.getUsers(), std::not_fn(inChain)))
        continue;
      auto ty = cast<RankedTensorType>(v.getType());
      SmallVector<int64_t> starts(ty.getRank(), 0), limits(ty.getShape()),
          strides(ty.getRank(), 1);
      starts[dim] = lo;
      limits[dim] += lo;
      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *wideOp = wide.getDefiningOp())
        rewriter.setInsertionPointAfter(wideOp);
      else
        rewriter.setInsertionPointToStart(cast<BlockArgument>(wide).getOwner());
      Value narrow = stablehlo::SliceOp::create(rewriter, op->getLoc(), wide,
                                                starts, limits, strides);
      copySharding(v, narrow);
      track(narrow);
      rewriter.replaceUsesWithIf(
          v, narrow, [&](OpOperand &use) { return !inChain(use.getOwner()); });
    }
  }
};

} // namespace enzyme
} // namespace mlir
