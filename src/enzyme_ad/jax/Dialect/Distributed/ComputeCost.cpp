#include "ComputeCost.h"

#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>

namespace mlir::enzyme::distributed {

namespace {

enum class Kind { Arithmetic, Transcendental, Movement, Free, Collective };

// The op kinds of the table in ComputeCost.h, by name without the dialect
// prefix. dot_general, convolution, reduce and reduce_window are handled
// separately because their flops are not a function of the result size.
const llvm::StringMap<Kind> &kindTable() {
  static const llvm::StringMap<Kind> table = [] {
    llvm::StringMap<Kind> t;
    for (const char *name : {"add",
                             "subtract",
                             "multiply",
                             "divide",
                             "remainder",
                             "negate",
                             "abs",
                             "sign",
                             "floor",
                             "ceil",
                             "maximum",
                             "minimum",
                             "and",
                             "or",
                             "xor",
                             "not",
                             "shift_left",
                             "shift_right_arithmetic",
                             "shift_right_logical",
                             "popcnt",
                             "compare",
                             "select",
                             "clamp",
                             "is_finite",
                             "real",
                             "imag",
                             "complex",
                             "round_nearest_afz",
                             "round_nearest_even"})
      t[name] = Kind::Arithmetic;
    for (const char *name :
         {"exponential", "exponential_minus_one", "log", "log_plus_one", "tanh",
          "logistic", "sqrt", "rsqrt", "cbrt", "power", "sine", "cosine", "tan",
          "atan2"})
      t[name] = Kind::Transcendental;
    for (const char *name :
         {"transpose", "broadcast_in_dim", "slice", "dynamic_slice",
          "dynamic_update_slice", "concatenate", "pad", "gather", "scatter",
          "convert", "reverse", "copy"})
      t[name] = Kind::Movement;
    for (const char *name :
         {"reshape", "bitcast_convert", "constant", "iota", "tuple",
          "get_tuple_element", "optimization_barrier"})
      t[name] = Kind::Free;
    for (const char *name :
         {"all_reduce", "all_gather", "reduce_scatter", "all_to_all",
          "collective_permute", "collective_broadcast"})
      t[name] = Kind::Collective;
    return t;
  }();
  return table;
}

// Size in bytes of a value of `type`, or nullopt if it is not a static tensor
// of a sized element type (R6, R7).
std::optional<double> staticBytes(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape())
    return std::nullopt;
  Type element = tensor.getElementType();
  unsigned bits;
  if (auto complex = dyn_cast<ComplexType>(element))
    bits = 2 * complex.getElementType().getIntOrFloatBitWidth();
  else if (element.isIntOrFloat())
    bits = element.getIntOrFloatBitWidth();
  else if (element.isIndex())
    bits = 64;
  else
    return std::nullopt;
  return static_cast<double>(tensor.getNumElements()) * ((bits + 7) / 8);
}

double numelOf(Type type) {
  return static_cast<double>(cast<RankedTensorType>(type).getNumElements());
}

} // namespace

double OpCost::time(const DeviceCostParams &device) const {
  return std::max(flops / device.flops, bytes / device.memBandwidth);
}

std::optional<OpCost> costOp(Operation *op) {
  OpCost cost;
  for (Type type :
       llvm::concat<const Type>(op->getOperandTypes(), op->getResultTypes())) {
    std::optional<double> bytes = staticBytes(type);
    if (!bytes)
      return std::nullopt;
    cost.bytes += *bytes;
  }

  StringRef name = op->getName().getStringRef();
  if (!name.consume_front("stablehlo.")) {
    cost.unknown = true;
    return cost;
  }

  double resultNumel =
      op->getNumResults() ? numelOf(op->getResult(0).getType()) : 0;

  if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    auto lhs = cast<RankedTensorType>(dot.getLhs().getType());
    double contracting = 1;
    for (int64_t dim :
         dot.getDotDimensionNumbers().getLhsContractingDimensions())
      contracting *= lhs.getDimSize(dim);
    cost.flops = 2 * resultNumel * contracting;
    return cost;
  }
  if (auto conv = dyn_cast<stablehlo::ConvolutionOp>(op)) {
    auto kernel = cast<RankedTensorType>(conv.getRhs().getType());
    int64_t outFeatures = kernel.getDimSize(
        conv.getDimensionNumbers().getKernelOutputFeatureDimension());
    cost.flops =
        2 * resultNumel * (numelOf(kernel) / static_cast<double>(outFeatures));
    return cost;
  }
  if (isa<stablehlo::ReduceOp, stablehlo::ReduceWindowOp>(op)) {
    for (Type type : op->getOperandTypes())
      cost.flops += numelOf(type);
    return cost;
  }

  auto it = kindTable().find(name);
  if (it == kindTable().end()) {
    cost.unknown = true;
    return cost;
  }
  switch (it->second) {
  case Kind::Arithmetic:
    cost.flops = resultNumel;
    break;
  case Kind::Transcendental:
    cost.flops = kTranscendentalWeight * resultNumel;
    break;
  case Kind::Movement:
    break;
  case Kind::Free:
    cost.bytes = 0;
    break;
  case Kind::Collective:
    cost.unknown = true;
    cost.collective = true;
    break;
  }
  return cost;
}

ComputeCostSummary summarizeKernelCompute(
    ModuleOp module, const DeviceCostParams &device,
    llvm::function_ref<void(Operation *, const OpCost &)> visitor) {
  ComputeCostSummary summary;
  module.walk([&](DistributedKernelOp kernel) {
    ++summary.numKernels;
    assert(kernel.getPartitioningAxes().empty() &&
           "kernel bodies are costed at local shapes");
    for (Operation &op : kernel.getBody().front()) {
      if (isa<DistributedYieldOp>(op))
        continue;
      std::optional<OpCost> cost = costOp(&op);
      if (!cost) {
        ++summary.numSkippedOps;
        continue;
      }
      ++summary.numOps;
      summary.numUnknownOps += cost->unknown;
      summary.numCollectiveOps += cost->collective;
      summary.total += cost->time(device);
      if (visitor)
        visitor(&op, *cost);
    }
  });
  return summary;
}

} // namespace mlir::enzyme::distributed
