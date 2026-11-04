//===- ElementChunk.cpp - run a raised kernel's elements in chunks --------===//
//
// A raised kernel computes every element on its own: each tensor of the
// computation has the element dimension, and nothing combines two elements.
// The stages of a sum-factorized contraction each read the tensor the stage
// before wrote, for every element, and XLA runs each stage as one kernel over
// all of them: once a stage's tensor outgrows the GPU's L2 cache, every stage
// is a round trip through DRAM (MFEM's 3D applies at 110592 elements write
// 100-500 MB per stage, against 96 MB of L2 on an RTX 5090). This pass runs
// the computation as a while loop over chunks of the elements, so that each
// chunk's stages hand their tensors on through the cache.
//
// The element dimension of a tensor is the one dimension whose size is a
// multiple of the element count (a byte buffer or a flattened view counts the
// elements times the rest). A function is rewritten when every op keeps the
// elements apart: elementwise ops, broadcasts, transposes, reshapes that keep
// the dimensions before the element dimension, and slices, pads, reductions
// and concatenations that leave the element dimension whole. The views of the
// arguments (their reshapes and bitcast conversions) are read at the chunk's
// offset with dynamic_slice; the values returned are written into the loop's
// results with dynamic_update_slice, and their trailing views applied after
// the loop. With `barriers`, the chunks are unrolled instead, and ordered by
// optimization barriers: a hint to XLA's scheduler, for the end of the
// pipeline, where nothing optimizes across them any more.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEHLOELEMENTCHUNKPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

constexpr int64_t kNoAxis = -1;
constexpr int64_t kInvalid = -2;

// The element dimension of a type: the one dimension whose size is a nonzero
// multiple of `elements`, kNoAxis when there is none, kInvalid when there are
// several or the type is not a static tensor.
static int64_t elementAxis(Type type, int64_t elements) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return isa<TensorType>(type) ? kInvalid : kNoAxis;
  if (!tensor.hasStaticShape())
    return kInvalid;
  int64_t axis = kNoAxis;
  for (auto [i, size] : llvm::enumerate(tensor.getShape())) {
    if (size == 0 || size % elements)
      continue;
    if (axis != kNoAxis)
      return kInvalid;
    axis = i;
  }
  return axis;
}

// The number of positions before the element dimension: a reshape keeps the
// elements apart when it keeps this.
static int64_t leadingProduct(RankedTensorType type, int64_t axis) {
  int64_t product = 1;
  for (int64_t i = 0; i < axis; i++)
    product *= type.getDimSize(i);
  return product;
}

// The type of a chunk of `part` of the `elements` of a tensor whose element
// dimension is `axis`.
static RankedTensorType chunkType(RankedTensorType type, int64_t axis,
                                  int64_t part, int64_t elements) {
  SmallVector<int64_t> shape(type.getShape());
  shape[axis] = shape[axis] / elements * part;
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

// A view of its operand: XLA reads it in place.
static bool isView(Operation *op) {
  return isa<stablehlo::ReshapeOp, stablehlo::BitcastConvertOp>(op);
}

// The argument a view reads, through the reshapes and bitcasts that give a
// byte buffer its element type, or null.
static Value viewedArgument(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isView(op))
      return nullptr;
    v = op->getOperand(0);
  }
  return isa<BlockArgument>(v) ? v : nullptr;
}

// The element count: of the dimensions of the largest argument view, the size
// that leads the views of the most arguments (an argument viewed in several
// shapes counts once). A byte buffer's reshapes to bytes are not views.
static int64_t elementCount(func::FuncOp func) {
  RankedTensorType largest;
  DenseMap<int64_t, DenseSet<Value>> leads;
  func.walk([&](stablehlo::ReshapeOp reshape) {
    auto type = reshape.getType();
    if (type.getRank() < 2 || !type.hasStaticShape() ||
        type.getElementType().isInteger(8))
      return;
    Value argument = viewedArgument(reshape.getResult());
    if (!argument)
      return;
    leads[type.getDimSize(0)].insert(argument);
    if (!largest || type.getNumElements() > largest.getNumElements())
      largest = type;
  });
  if (!largest)
    return kNoAxis;
  int64_t elements = kNoAxis;
  size_t best = 0;
  for (int64_t size : largest.getShape()) {
    size_t count = leads.lookup(size).size();
    if (count > best) {
      best = count;
      elements = size;
    }
  }
  return elements;
}

struct Chunking {
  int64_t elements;
  DenseMap<Value, int64_t> axes;

  int64_t axis(Value v) const {
    auto it = axes.find(v);
    return it == axes.end() ? kNoAxis : it->second;
  }

  // Whether `op` keeps the elements apart, given the element dimensions of
  // its operands and results.
  bool keepsElementsApart(Operation *op) const {
    auto ty = [](Value v) { return cast<RankedTensorType>(v.getType()); };
    if (isa<func::ReturnOp>(op))
      return true;
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
      for (Value input : reduce.getInputs())
        if (axis(input) != kNoAxis &&
            llvm::is_contained(reduce.getDimensions(), axis(input)))
          return false;
      return true;
    }
    if (op->getNumRegions())
      return false;
    if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
      int64_t a = axis(bcast.getOperand());
      return a == kNoAxis ||
             (bcast.getBroadcastDimensions()[a] == axis(bcast.getResult()) &&
              ty(bcast.getOperand()).getDimSize(a) ==
                  bcast.getType().getDimSize(axis(bcast.getResult())));
    }
    if (isa<stablehlo::ReshapeOp, stablehlo::BitcastConvertOp>(op)) {
      Value in = op->getOperand(0), out = op->getResult(0);
      if (axis(in) == kNoAxis || axis(out) == kNoAxis)
        return axis(in) == axis(out);
      return leadingProduct(ty(in), axis(in)) ==
             leadingProduct(ty(out), axis(out));
    }
    if (isa<stablehlo::TransposeOp>(op))
      return true;
    if (auto slice = dyn_cast<stablehlo::SliceOp>(op)) {
      int64_t a = axis(slice.getOperand());
      return a == kNoAxis ||
             (slice.getStartIndices()[a] == 0 && slice.getStrides()[a] == 1 &&
              slice.getLimitIndices()[a] ==
                  ty(slice.getOperand()).getDimSize(a));
    }
    if (auto pad = dyn_cast<stablehlo::PadOp>(op)) {
      int64_t a = axis(pad.getOperand());
      return a == kNoAxis || (pad.getEdgePaddingLow()[a] == 0 &&
                              pad.getEdgePaddingHigh()[a] == 0 &&
                              pad.getInteriorPadding()[a] == 0);
    }
    if (auto concat = dyn_cast<stablehlo::ConcatenateOp>(op))
      return axis(concat.getResult()) != (int64_t)concat.getDimension();
    if (auto iota = dyn_cast<stablehlo::IotaOp>(op))
      return axis(iota.getResult()) != (int64_t)iota.getIotaDimension();
    if (auto constant = dyn_cast<stablehlo::ConstantOp>(op))
      return axis(constant.getResult()) == kNoAxis ||
             isa<SplatElementsAttr>(constant.getValue());
    if (isa<stablehlo::OptimizationBarrierOp>(op) ||
        op->hasTrait<OpTrait::Elementwise>() ||
        isa<stablehlo::SelectOp, stablehlo::ClampOp, stablehlo::CompareOp>(
            op)) {
      // Same shapes: the element dimension stays where it is.
      for (auto [i, result] : llvm::enumerate(op->getResults())) {
        Value operand =
            op->getOperand(isa<stablehlo::OptimizationBarrierOp>(op) ? i : 0);
        if (axis(result) != kNoAxis && axis(operand) != axis(result) &&
            !isa<stablehlo::SelectOp, stablehlo::ClampOp>(op))
          return false;
      }
      return true;
    }
    return false;
  }
};

struct ElementChunkPass
    : public enzyme::impl::EnzymeHLOElementChunkPassBase<ElementChunkPass> {
  using EnzymeHLOElementChunkPassBase::EnzymeHLOElementChunkPassBase;

  void runOnOperation() override {
    getOperation().walk([&](func::FuncOp func) {
      if (func.getBody().hasOneBlock())
        chunk(func);
    });
  }

  void chunk(func::FuncOp func) {
    Block &block = func.getBody().front();
    Chunking c;
    c.elements = elements > 0 ? elements : elementCount(func);
    if (c.elements <= 1)
      return;
    // The element dimension of every value.
    auto record = [&](Value v) {
      int64_t a = elementAxis(v.getType(), c.elements);
      c.axes[v] = a;
      return a != kInvalid;
    };
    for (BlockArgument arg : block.getArguments())
      if (!record(arg))
        return;
    for (Operation &op : block) {
      for (Value result : op.getResults())
        if (!record(result))
          return;
      if (!c.keepsElementsApart(&op))
        return;
    }
    // The ops that run per chunk: those computing a value with the element
    // dimension, except the views of the arguments, read in place.
    SetVector<Operation *> chunked;
    DenseSet<Value> inPlace;
    for (BlockArgument arg : block.getArguments())
      inPlace.insert(arg);
    for (Operation &op : block.without_terminator()) {
      bool perChunk = llvm::any_of(
          op.getResults(), [&](Value v) { return c.axis(v) != kNoAxis; });
      if (isView(&op) && inPlace.contains(op.getOperand(0)))
        perChunk = false;
      if (llvm::any_of(op.getOperands(),
                       [&](Value v) { return chunkedValue(chunked, v); }))
        perChunk = true;
      if (!perChunk) {
        for (Value result : op.getResults())
          inPlace.insert(result);
        continue;
      }
      chunked.insert(&op);
    }
    // The values returned, through the views applied to them after the loop.
    Operation *ret = block.getTerminator();
    SmallVector<Value> roots;
    SetVector<Operation *> trailing;
    for (Value v : ret->getOperands()) {
      while (Operation *def = v.getDefiningOp()) {
        if (!chunked.contains(def) || !isView(def) || !v.hasOneUse() ||
            c.axis(def->getOperand(0)) == kNoAxis)
          break;
        trailing.insert(def);
        v = def->getOperand(0);
      }
      if (chunkedValue(chunked, v) && !llvm::is_contained(roots, v))
        roots.push_back(v);
    }
    for (Operation *op : trailing)
      chunked.remove(op);
    if (roots.empty())
      return;
    // The chunk count: the fewest chunks in which no tensor of a chunk exceeds
    // chunkBytes. The loop's chunks are all one size, so they divide the
    // elements; the unrolled chunks need not (their sizes differ by one
    // element at most), which keeps an element count with no divisor near the
    // count needed (a prime one, 67759 in dg_amr's refined mesh) from
    // unrolling into a chunk per element.
    int64_t largest = 0;
    for (Operation *op : chunked)
      for (Value v : op->getResults()) {
        auto type = cast<RankedTensorType>(v.getType());
        largest = std::max(largest, type.getNumElements() *
                                        type.getElementTypeBitWidth() / 8);
      }
    int64_t chunks = this->chunks;
    if (chunks <= 0 && barriers) {
      int64_t perElement = std::max<int64_t>(1, largest / c.elements);
      int64_t perChunk = std::max<int64_t>(1, chunkBytes / perElement);
      chunks = (c.elements + perChunk - 1) / perChunk;
    } else if (chunks <= 0) {
      chunks = 1;
      while (chunks < c.elements &&
             (largest > chunkBytes * chunks || c.elements % chunks))
        chunks++;
    }
    if (chunks <= 1 || chunks > c.elements ||
        (!barriers && c.elements % chunks))
      return;
    if (barriers)
      unroll(func, c, chunked, roots, trailing, chunks);
    else
      rewrite(func, c, chunked, roots, trailing, chunks);
  }

  static bool chunkedValue(const SetVector<Operation *> &chunked, Value v) {
    Operation *def = v.getDefiningOp();
    return def && chunked.contains(def);
  }

  void rewrite(func::FuncOp func, const Chunking &c,
               const SetVector<Operation *> &chunked, ArrayRef<Value> roots,
               const SetVector<Operation *> &trailing, int64_t chunks) {
    int64_t part = c.elements / chunks;
    Block &block = func.getBody().front();
    Operation *ret = block.getTerminator();
    Location loc = func.getLoc();
    OpBuilder builder(ret);
    auto i64 = RankedTensorType::get({}, builder.getI64Type());
    auto scalar = [&](OpBuilder &b, int64_t value) {
      return stablehlo::ConstantOp::create(
                 b, loc,
                 SplatElementsAttr::get(i64, b.getI64IntegerAttr(value)))
          .getResult();
    };
    // The values run once that the chunks read: carried through the loop,
    // except the constants, which each chunk makes again (a reduction's
    // initial value stays a constant to XLA).
    SetVector<Value> carried;
    SetVector<Operation *> constants;
    for (Operation *op : chunked)
      for (Value v : op->getOperands()) {
        if (chunkedValue(chunked, v))
          continue;
        if (auto constant = v.getDefiningOp<stablehlo::ConstantOp>())
          constants.insert(constant);
        else
          carried.insert(v);
      }
    // Loop state: the chunk index, the results being written, the carried.
    SmallVector<Value> inits{scalar(builder, 0)};
    for (Value root : roots) {
      auto type = cast<RankedTensorType>(root.getType());
      auto zero = stablehlo::ConstantOp::create(
          builder, loc,
          SplatElementsAttr::get(type,
                                 builder.getZeroAttr(type.getElementType())));
      inits.push_back(zero);
    }
    inits.append(carried.begin(), carried.end());
    SmallVector<Type> types;
    for (Value v : inits)
      types.push_back(v.getType());
    SmallVector<Location> locs(types.size(), loc);
    auto loop = stablehlo::WhileOp::create(builder, loc, types, inits);
    {
      Block *cond = builder.createBlock(&loop.getCond(), {}, types, locs);
      auto limit = scalar(builder, chunks);
      auto less = stablehlo::CompareOp::create(
          builder, loc, cond->getArgument(0), limit,
          stablehlo::ComparisonDirection::LT);
      stablehlo::ReturnOp::create(builder, loc, less.getResult());
    }
    Block *body = builder.createBlock(&loop.getBody(), {}, types, locs);
    Value index = body->getArgument(0);
    // The offsets of a chunk of a tensor whose element dimension is `axis`.
    auto offsets = [&](RankedTensorType type, int64_t axis) {
      auto step = scalar(builder, type.getDimSize(axis) / chunks);
      auto offset = stablehlo::MulOp::create(builder, loc, index, step);
      SmallVector<Value> result;
      for (int64_t i = 0; i < type.getRank(); i++)
        result.push_back(i == axis ? offset.getResult() : scalar(builder, 0));
      return result;
    };
    IRMapping map;
    for (Operation *constant : constants)
      builder.clone(*constant, map);
    for (auto [i, v] : llvm::enumerate(carried)) {
      Value arg = body->getArgument(1 + roots.size() + i);
      int64_t a = c.axis(v);
      if (a == kNoAxis) {
        map.map(v, arg);
        continue;
      }
      auto type = cast<RankedTensorType>(v.getType());
      auto piece = stablehlo::DynamicSliceOp::create(
          builder, loc, chunkType(type, a, part, c.elements), arg,
          offsets(type, a), chunkType(type, a, part, c.elements).getShape());
      map.map(v, piece.getResult());
    }
    cloneChunk(builder, map, c, chunked, part);
    SmallVector<Value> yields{
        stablehlo::AddOp::create(builder, loc, index, scalar(builder, 1))
            .getResult()};
    for (auto [i, root] : llvm::enumerate(roots)) {
      auto type = cast<RankedTensorType>(root.getType());
      auto update = stablehlo::DynamicUpdateSliceOp::create(
          builder, loc, body->getArgument(1 + i), map.lookup(root),
          offsets(type, c.axis(root)));
      yields.push_back(update.getResult());
    }
    for (size_t i = 0; i < carried.size(); i++)
      yields.push_back(body->getArgument(1 + roots.size() + i));
    stablehlo::ReturnOp::create(builder, loc, yields);
    builder.setInsertionPoint(ret);
    finish(builder, block, roots, loop.getResults().slice(1, roots.size()),
           chunked, trailing);
  }

  // The chunk of `part` elements the ops in `chunked` compute, cloned at
  // `builder`, given the chunks of the values they read in `map`.
  static void cloneChunk(OpBuilder &builder, IRMapping &map, const Chunking &c,
                         const SetVector<Operation *> &chunked, int64_t part) {
    for (Operation *op : chunked) {
      Operation *clone = builder.clone(*op, map);
      for (auto [o, n] : llvm::zip(op->getResults(), clone->getResults())) {
        int64_t a = c.axis(o);
        if (a == kNoAxis)
          continue;
        n.setType(chunkType(cast<RankedTensorType>(o.getType()), a, part,
                            c.elements));
      }
      if (auto slice = dyn_cast<stablehlo::SliceOp>(clone)) {
        int64_t a = c.axis(cast<stablehlo::SliceOp>(op).getOperand());
        if (a != kNoAxis) {
          SmallVector<int64_t> limits(slice.getLimitIndices());
          limits[a] = limits[a] / c.elements * part;
          slice.setLimitIndices(limits);
        }
      }
      if (auto constant = dyn_cast<stablehlo::ConstantOp>(clone))
        if (auto splat = dyn_cast<SplatElementsAttr>(constant.getValue()))
          constant.setValueAttr(
              SplatElementsAttr::get(cast<ShapedType>(constant.getType()),
                                     splat.getSplatValue<Attribute>()));
      for (auto [o, n] : llvm::zip(op->getResults(), clone->getResults()))
        map.map(o, n);
    }
  }

  // The chunks unrolled: each reads the values run once, and the results the
  // chunks before it wrote, through an optimization barrier with them, so XLA
  // runs the chunks one after another (the barriers, made after the rest of
  // the pipeline, only order XLA's kernels).
  void unroll(func::FuncOp func, const Chunking &c,
              const SetVector<Operation *> &chunked, ArrayRef<Value> roots,
              const SetVector<Operation *> &trailing, int64_t chunks) {
    Block &block = func.getBody().front();
    Location loc = func.getLoc();
    OpBuilder builder(block.getTerminator());
    auto i64 = RankedTensorType::get({}, builder.getI64Type());
    // (the constants are read as they are: a reduction's initial value stays
    // a constant to XLA)
    SetVector<Value> carried;
    for (Operation *op : chunked)
      for (Value v : op->getOperands())
        if (!chunkedValue(chunked, v) &&
            !v.getDefiningOp<stablehlo::ConstantOp>())
          carried.insert(v);
    SmallVector<Value> results;
    for (Value root : roots) {
      auto type = cast<RankedTensorType>(root.getType());
      auto zero = stablehlo::ConstantOp::create(
          builder, loc,
          SplatElementsAttr::get(type,
                                 builder.getZeroAttr(type.getElementType())));
      results.push_back(zero);
    }
    SmallVector<Value> state(carried.begin(), carried.end());
    for (int64_t k = 0; k < chunks; k++) {
      if (k > 0) {
        SmallVector<Value> operands(results);
        operands.append(state);
        auto barrier =
            stablehlo::OptimizationBarrierOp::create(builder, loc, operands);
        results.assign(barrier.result_begin(),
                       barrier.result_begin() + roots.size());
        state.assign(barrier.result_begin() + roots.size(),
                     barrier.result_end());
      }
      // (the elements [begin, end) of the chunk)
      int64_t begin = k * c.elements / chunks;
      int64_t end = (k + 1) * c.elements / chunks;
      IRMapping map;
      for (auto [v, read] : llvm::zip(carried, state)) {
        int64_t a = c.axis(v);
        if (a == kNoAxis) {
          map.map(v, read);
          continue;
        }
        auto type = cast<RankedTensorType>(v.getType());
        int64_t size = type.getDimSize(a) / c.elements;
        SmallVector<int64_t> start(type.getRank(), 0);
        SmallVector<int64_t> limit(type.getShape());
        SmallVector<int64_t> strides(type.getRank(), 1);
        start[a] = begin * size;
        limit[a] = end * size;
        auto piece = stablehlo::SliceOp::create(builder, loc, read, start,
                                                limit, strides);
        map.map(v, piece.getResult());
      }
      cloneChunk(builder, map, c, chunked, end - begin);
      for (auto [i, root] : llvm::enumerate(roots)) {
        auto type = cast<RankedTensorType>(root.getType());
        int64_t a = c.axis(root);
        SmallVector<Value> offsets;
        for (int64_t d = 0; d < type.getRank(); d++) {
          int64_t offset =
              d == a ? begin * (type.getDimSize(a) / c.elements) : 0;
          auto constant = stablehlo::ConstantOp::create(
              builder, loc,
              SplatElementsAttr::get(i64, builder.getI64IntegerAttr(offset)));
          offsets.push_back(constant);
        }
        auto update = stablehlo::DynamicUpdateSliceOp::create(
            builder, loc, results[i], map.lookup(root), offsets);
        results[i] = update.getResult();
      }
    }
    finish(builder, block, roots, results, chunked, trailing);
  }

  // The values returned, written from `results` through the trailing views
  // of `roots`; the ops the chunks replace erased.
  static void finish(OpBuilder &builder, Block &block, ArrayRef<Value> roots,
                     ValueRange results, const SetVector<Operation *> &chunked,
                     const SetVector<Operation *> &trailing) {
    Operation *ret = block.getTerminator();
    IRMapping after;
    for (auto [root, result] : llvm::zip(roots, results))
      after.map(root, result);
    for (Operation &op : block) {
      if (!trailing.contains(&op))
        continue;
      Operation *clone = builder.clone(op, after);
      for (auto [o, n] : llvm::zip(op.getResults(), clone->getResults()))
        after.map(o, n);
    }
    for (OpOperand &operand : ret->getOpOperands())
      operand.set(after.lookupOrDefault(operand.get()));
    // (trailing lists each chain of views from the return back, users first)
    for (Operation *op : trailing)
      op->erase();
    for (Operation *op : llvm::reverse(chunked))
      op->erase();
  }
};

} // namespace
