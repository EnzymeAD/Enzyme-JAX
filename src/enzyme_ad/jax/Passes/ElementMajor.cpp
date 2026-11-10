//===- ElementMajor.cpp - the element dimension leads every tensor --------===//
//
// A raised kernel reads and writes its memory per element: every argument
// view is (elements, ...). Batching stacks the pieces of a computation along
// new leading dimensions, so its intermediates come out (stack, elements, ...)
// and XLA's layout assignment, deriving physical layouts from the logical
// shapes, iterates the fusions stack-major: each fusion then reads the
// element-major arguments and intermediates strided, and reads them once per
// stack entry. This pass moves the element dimension to the front of every
// tensor that has it (the dimensions after it, the minor ones, keep their
// order), rewriting the ops in between; where an op cannot be rewritten its
// operands are transposed back and its result stays as it was.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEHLOELEMENTMAJORPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

// The permutation (old dimension -> new dimension) moving dimension `at` of
// a rank-`rank` tensor to the front; identity when `at` is 0 or out of range.
static SmallVector<int64_t> moveToFront(int64_t rank, int64_t at) {
  SmallVector<int64_t> perm(rank);
  for (int64_t i = 0; i < rank; i++)
    perm[i] =
        (at <= 0 || at >= rank) ? i : (i == at ? 0 : (i < at ? i + 1 : i));
  return perm;
}

static bool isIdentity(ArrayRef<int64_t> perm) {
  return llvm::equal(perm, llvm::seq<int64_t>(0, perm.size()));
}

// The permutation moving the element dimension of `type` (the first
// dimension of size `elements` that is neither the leading nor the last one)
// to the front; identity when there is none. The root of the layout of a
// value computed from values that have none of their own.
static SmallVector<int64_t> rootPermutation(Type type, int64_t elements) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return {};
  ArrayRef<int64_t> shape = tensor.getShape();
  auto it = llvm::find(shape, elements);
  int64_t at = it - shape.begin();
  if (it == shape.end() || at == (int64_t)shape.size() - 1)
    return moveToFront(shape.size(), 0);
  return moveToFront(shape.size(), at);
}

static RankedTensorType permuted(RankedTensorType type,
                                 ArrayRef<int64_t> perm) {
  SmallVector<int64_t> shape(type.getRank());
  for (auto [i, p] : llvm::enumerate(perm))
    shape[p] = type.getDimSize(i);
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

// The argument a memory view reads: a reshape of a function argument
// (through the reshapes and bitcasts that give a byte buffer its element
// type), or null.
static Value viewedArgument(stablehlo::ReshapeOp reshape) {
  Value v = reshape.getOperand();
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<stablehlo::BitcastConvertOp, stablehlo::ReshapeOp>(op))
      return nullptr;
    v = op->getOperand(0);
  }
  return isa<BlockArgument>(v) ? v : nullptr;
}

// The element dimension: of the dimensions of the largest argument view, the
// size that leads the views of the most arguments (an argument viewed in
// several shapes counts once), the earlier dimension of that view among
// equals. A byte buffer's reshapes to bytes are not views.
static int64_t elementDimension(func::FuncOp func) {
  RankedTensorType largest;
  DenseMap<int64_t, DenseSet<Value>> leads;
  func.walk([&](stablehlo::ReshapeOp reshape) {
    auto type = reshape.getType();
    if (type.getRank() < 2 || !type.hasStaticShape() ||
        type.getElementType().isInteger(8))
      return;
    Value argument = viewedArgument(reshape);
    if (!argument)
      return;
    leads[type.getDimSize(0)].insert(argument);
    if (!largest || type.getNumElements() > largest.getNumElements())
      largest = type;
  });
  if (!largest)
    return -1;
  int64_t elements = -1, best = 0;
  for (int64_t size : largest.getShape()) {
    int64_t count = leads.lookup(size).size();
    if (count > best) {
      best = count;
      elements = size;
    }
  }
  return elements;
}

struct Relayout {
  int64_t elements;
  // The layout each rewritten value is stored in (old dimension -> new
  // dimension), and the value in that layout.
  DenseMap<Value, SmallVector<int64_t>> perms;
  DenseMap<Value, Value> mapped;
  DenseSet<Operation *> created;
  SmallVector<Operation *> replaced;

  // The layout of `v`: its own when rewritten, else the identity.
  SmallVector<int64_t> layoutOf(Value v) {
    if (auto found = perms.find(v); found != perms.end())
      return found->second;
    auto type = dyn_cast<RankedTensorType>(v.getType());
    return type ? moveToFront(type.getRank(), 0) : SmallVector<int64_t>{};
  }

  // The index the moved dimension (new dimension 0) had, -1 for the identity.
  static int64_t movedDim(ArrayRef<int64_t> perm) {
    if (perm.empty() || isIdentity(perm))
      return -1;
    return llvm::find(perm, 0) - perm.begin();
  }

  // The layout a reshape reads its operand in: the operand's own, else the
  // root one of its type (an operand nothing rewrote still has its element
  // dimension to lead).
  SmallVector<int64_t> reshapeInputLayout(stablehlo::ReshapeOp reshape) {
    SmallVector<int64_t> po = layoutOf(reshape.getOperand());
    if (movedDim(po) < 0)
      po = rootPermutation(reshape.getOperand().getType(), elements);
    return po;
  }

  // `v` in layout `perm`, transposed into it where it is stored otherwise.
  Value inLayout(Value v, ArrayRef<int64_t> perm, OpBuilder &builder) {
    SmallVector<int64_t> have = layoutOf(v);
    Value stored = mapped.count(v) ? mapped[v] : v;
    if (llvm::equal(have, perm))
      return stored;
    // result dimension perm[i] holds old dimension i, stored at have[i]
    SmallVector<int64_t> dims(perm.size());
    for (auto [i, p] : llvm::enumerate(perm))
      dims[p] = have[i];
    auto transpose =
        stablehlo::TransposeOp::create(builder, v.getLoc(), stored, dims);
    created.insert(transpose);
    return transpose.getResult();
  }

  // The layout of `op`'s result: derived from its operands' when they have
  // one, else the root one of its type. Empty when `op` is not rewritten.
  SmallVector<int64_t> resultLayout(Operation *op, bool &handled) {
    handled = false;
    if (op->getNumResults() == 0)
      return {};
    auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || !type.hasStaticShape())
      return {};
    handled = true;
    if (stablehlo::hasTraitElementwise(op) ||
        isa<stablehlo::OptimizationBarrierOp, stablehlo::ConcatenateOp>(op)) {
      // the operands of the result's shape agree on a layout, or take the
      // root one (a barrier's result i follows its operand i, below)
      for (Value operand : op->getOperands())
        if (operand.getType() == type && perms.count(operand))
          return perms[operand];
      return rootPermutation(type, elements);
    }
    if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
      int64_t a = movedDim(layoutOf(bcast.getOperand()));
      if (a < 0)
        return rootPermutation(type, elements);
      return moveToFront(type.getRank(), bcast.getBroadcastDimensions()[a]);
    }
    if (auto transpose = dyn_cast<stablehlo::TransposeOp>(op)) {
      int64_t a = movedDim(layoutOf(transpose.getOperand()));
      if (a < 0)
        return rootPermutation(type, elements);
      auto perm = transpose.getPermutation();
      return moveToFront(type.getRank(), llvm::find(perm, a) - perm.begin());
    }
    if (auto slice = dyn_cast<stablehlo::SliceOp>(op))
      return perms.count(slice.getOperand()) ? perms[slice.getOperand()]
                                             : rootPermutation(type, elements);
    if (auto constant = dyn_cast<stablehlo::ConstantOp>(op)) {
      handled = cast<DenseElementsAttr>(constant.getValue()).isSplat();
      return rootPermutation(type, elements);
    }
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
      auto in = cast<RankedTensorType>(reduce.getInputs()[0].getType());
      SmallVector<int64_t> perm = layoutOf(reduce.getInputs()[0]);
      int64_t a = movedDim(perm);
      if (a < 0) {
        // no layout of its own: the input's root one, unless reduced
        perm = rootPermutation(in, elements);
        a = movedDim(perm);
        if (a < 0)
          return moveToFront(type.getRank(), 0);
      }
      if (llvm::is_contained(reduce.getDimensions(), a)) {
        handled = false;
        return {};
      }
      int64_t kept = 0;
      for (int64_t d = 0; d < a; d++)
        if (!llvm::is_contained(reduce.getDimensions(), d))
          kept++;
      return moveToFront(type.getRank(), kept);
    }
    if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
      // the moved dimension stays a dimension of its own: the dimensions
      // before it reshape among themselves (and so do those after it)
      auto in = cast<RankedTensorType>(reshape.getOperand().getType());
      int64_t a = movedDim(reshapeInputLayout(reshape));
      if (a < 0) {
        if (in.getRank() == 0 || in.getDimSize(0) != elements)
          return moveToFront(type.getRank(), 0);
        a = 0; // the element dimension leads the input: keep it leading
      }
      int64_t before = 1;
      for (int64_t d = 0; d < a; d++)
        before *= in.getDimSize(d);
      int64_t prefix = 1;
      for (int64_t b = 0; b < type.getRank(); b++) {
        if (prefix == before && type.getDimSize(b) == in.getDimSize(a))
          return moveToFront(type.getRank(), b);
        prefix *= type.getDimSize(b);
        if (prefix > before)
          break;
      }
      handled = false;
      return {};
    }
    handled = false;
    return {};
  }

  void rewrite(Operation *op) {
    bool handled;
    SmallVector<int64_t> layout = resultLayout(op, handled);
    bool any = !isIdentity(layout);
    for (Value operand : op->getOperands())
      any |= perms.count(operand) != 0;
    if (!handled || !any)
      return;
    OpBuilder builder(op);
    Location loc = op->getLoc();
    SmallVector<Value> operands;
    SmallVector<Type> types;
    Operation *created = nullptr;
    if (auto bcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
      SmallVector<int64_t> po = layoutOf(bcast.getOperand());
      Value operand = inLayout(bcast.getOperand(), po, builder);
      auto dims = bcast.getBroadcastDimensions();
      SmallVector<int64_t> newDims(dims.size());
      for (auto [k, d] : llvm::enumerate(dims))
        newDims[po[k]] = layout[d];
      created = stablehlo::BroadcastInDimOp::create(
          builder, loc, permuted(bcast.getType(), layout), operand, newDims);
    } else if (auto transpose = dyn_cast<stablehlo::TransposeOp>(op)) {
      SmallVector<int64_t> po = layoutOf(transpose.getOperand());
      Value operand = inLayout(transpose.getOperand(), po, builder);
      auto perm = transpose.getPermutation();
      SmallVector<int64_t> newPerm(perm.size());
      for (auto [j, pj] : llvm::enumerate(perm))
        newPerm[layout[j]] = po[pj];
      if (isIdentity(newPerm)) {
        // the two layouts agree: nothing to transpose
        mapped[transpose.getResult()] = operand;
        perms[transpose.getResult()] = layout;
        replaced.push_back(op);
        return;
      }
      created = stablehlo::TransposeOp::create(builder, loc, operand, newPerm);
    } else if (auto slice = dyn_cast<stablehlo::SliceOp>(op)) {
      Value operand = inLayout(slice.getOperand(), layout, builder);
      SmallVector<int64_t> starts(layout.size()), limits(layout.size()),
          strides(layout.size());
      for (auto [i, p] : llvm::enumerate(layout)) {
        starts[p] = slice.getStartIndices()[i];
        limits[p] = slice.getLimitIndices()[i];
        strides[p] = slice.getStrides()[i];
      }
      created = stablehlo::SliceOp::create(builder, loc, operand, starts,
                                           limits, strides);
    } else if (auto concat = dyn_cast<stablehlo::ConcatenateOp>(op)) {
      for (Value operand : op->getOperands())
        operands.push_back(inLayout(operand, layout, builder));
      created = stablehlo::ConcatenateOp::create(builder, loc, operands,
                                                 layout[concat.getDimension()]);
    } else if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
      SmallVector<int64_t> po = layoutOf(reduce.getInputs()[0]);
      if (movedDim(po) < 0)
        po = rootPermutation(reduce.getInputs()[0].getType(), elements);
      for (Value input : reduce.getInputs())
        operands.push_back(inLayout(input, po, builder));
      SmallVector<int64_t> dims;
      for (int64_t d : reduce.getDimensions())
        dims.push_back(po[d]);
      llvm::sort(dims);
      for (Type t : op->getResultTypes())
        types.push_back(permuted(cast<RankedTensorType>(t), layout));
      auto newReduce = stablehlo::ReduceOp::create(
          builder, loc, types, operands, reduce.getInitValues(), dims);
      IRMapping mapping;
      reduce.getBody().cloneInto(&newReduce.getBody(), mapping);
      created = newReduce;
    } else if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
      SmallVector<int64_t> po = reshapeInputLayout(reshape);
      Value operand = inLayout(reshape.getOperand(), po, builder);
      created = stablehlo::ReshapeOp::create(
          builder, loc, permuted(reshape.getType(), layout), operand);
    } else if (auto constant = dyn_cast<stablehlo::ConstantOp>(op)) {
      created = stablehlo::ConstantOp::create(
          builder, loc,
          cast<DenseElementsAttr>(constant.getValue())
              .resizeSplat(permuted(constant.getType(), layout)));
    } else if (auto barrier = dyn_cast<stablehlo::OptimizationBarrierOp>(op)) {
      // each result follows its operand
      SmallVector<SmallVector<int64_t>> layouts;
      for (auto [operand, result] :
           llvm::zip(op->getOperands(), op->getResults())) {
        SmallVector<int64_t> p = layoutOf(operand);
        if (movedDim(p) < 0)
          p = rootPermutation(operand.getType(), elements);
        layouts.push_back(p);
        operands.push_back(inLayout(operand, p, builder));
        types.push_back(permuted(cast<RankedTensorType>(result.getType()), p));
      }
      created =
          stablehlo::OptimizationBarrierOp::create(builder, loc, operands);
      this->created.insert(created);
      for (auto [i, result] : llvm::enumerate(op->getResults())) {
        mapped[result] = created->getResult(i);
        perms[result] = layouts[i];
      }
      replaced.push_back(op);
      return;
    } else {
      // elementwise: the same op on the operands in the result's layout
      for (Value operand : op->getOperands())
        operands.push_back(
            cast<RankedTensorType>(operand.getType()).getRank() ==
                    (int64_t)layout.size()
                ? inLayout(operand, layout, builder)
                : operand);
      OperationState state(loc, op->getName());
      state.addOperands(operands);
      for (Type t : op->getResultTypes())
        state.addTypes(permuted(cast<RankedTensorType>(t), layout));
      state.addAttributes(op->getAttrs());
      created = builder.create(state);
    }
    this->created.insert(created);
    for (auto [old, result] :
         llvm::zip(op->getResults(), created->getResults())) {
      mapped[old] = result;
      perms[old] = layout;
    }
    replaced.push_back(op);
  }

  // After the walk: the ops that were not rewritten read the rewritten
  // values transposed back, and the rewritten ops go.
  void finish() {
    for (Operation *op : replaced) {
      for (Value old : op->getResults()) {
        Value result = mapped.lookup(old);
        SmallVector<int64_t> perm = perms.lookup(old);
        if (isIdentity(perm)) {
          old.replaceAllUsesWith(result);
          continue;
        }
        if (llvm::all_of(old.getUsers(),
                         [&](Operation *user) { return created.count(user); }))
          continue;
        OpBuilder builder(result.getContext());
        builder.setInsertionPointAfterValue(result);
        auto back =
            stablehlo::TransposeOp::create(builder, old.getLoc(), result, perm);
        old.replaceUsesWithIf(back, [&](OpOperand &use) {
          return !created.count(use.getOwner()) && use.getOwner() != back;
        });
      }
    }
    for (Operation *op : llvm::reverse(replaced))
      if (op->use_empty())
        op->erase();
  }
};

struct ElementMajorPass
    : public enzyme::impl::EnzymeHLOElementMajorPassBase<ElementMajorPass> {
  using Base::Base;

  void runOnOperation() override {
    getOperation()->walk([&](func::FuncOp func) {
      if (func.getBody().empty() || !func.getBody().hasOneBlock())
        return;
      int64_t elements = dim >= 0 ? dim : elementDimension(func);
      if (elements <= 1)
        return;
      Relayout relayout{elements};
      SmallVector<Operation *> ops;
      for (Operation &op : func.getBody().front())
        ops.push_back(&op);
      for (Operation *op : ops)
        relayout.rewrite(op);
      relayout.finish();
    });
  }
};

} // namespace
