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
// the loop.
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

static RankedTensorType chunkType(RankedTensorType type, int64_t axis,
                                  int64_t chunks) {
  SmallVector<int64_t> shape(type.getShape());
  shape[axis] /= chunks;
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
    // The chunk count: the fewest chunks, dividing the elements, in which no
    // tensor of a chunk exceeds chunkBytes.
    int64_t largest = 0;
    for (Operation *op : chunked)
      for (Value v : op->getResults()) {
        auto type = cast<RankedTensorType>(v.getType());
        largest = std::max(largest, type.getNumElements() *
                                        type.getElementTypeBitWidth() / 8);
      }
    int64_t chunks = this->chunks;
    if (chunks <= 0) {
      chunks = 1;
      while (chunks < c.elements &&
             (largest > chunkBytes * chunks || c.elements % chunks))
        chunks++;
    }
    if (chunks <= 1 || c.elements % chunks)
      return;
    rewrite(func, c, chunked, roots, trailing, chunks);
  }

  static bool chunkedValue(const SetVector<Operation *> &chunked, Value v) {
    Operation *def = v.getDefiningOp();
    return def && chunked.contains(def);
  }

  void rewrite(func::FuncOp func, const Chunking &c,
               const SetVector<Operation *> &chunked, ArrayRef<Value> roots,
               const SetVector<Operation *> &trailing, int64_t chunks) {
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
          builder, loc, chunkType(type, a, chunks), arg, offsets(type, a),
          chunkType(type, a, chunks).getShape());
      map.map(v, piece.getResult());
    }
    for (Operation *op : chunked) {
      Operation *clone = builder.clone(*op, map);
      for (auto [o, n] : llvm::zip(op->getResults(), clone->getResults())) {
        int64_t a = c.axis(o);
        if (a == kNoAxis)
          continue;
        n.setType(chunkType(cast<RankedTensorType>(o.getType()), a, chunks));
      }
      if (auto slice = dyn_cast<stablehlo::SliceOp>(clone)) {
        int64_t a = c.axis(cast<stablehlo::SliceOp>(op).getOperand());
        if (a != kNoAxis) {
          SmallVector<int64_t> limits(slice.getLimitIndices());
          limits[a] /= chunks;
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
    // After the loop: the results, through their trailing views.
    builder.setInsertionPoint(ret);
    IRMapping after;
    for (auto [i, root] : llvm::enumerate(roots))
      after.map(root, loop.getResult(1 + i));
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
