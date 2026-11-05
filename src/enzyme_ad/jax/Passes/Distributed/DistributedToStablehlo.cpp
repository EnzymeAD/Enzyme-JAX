// Exports a finalized distributed schedule as plain StableHLO that an XLA
// pipeline compiles with SPMD partitioning and runs on global inputs.
//
// The per-device program keeps its local shapes and explicit collectives.
// How each local value maps to its global tensor is recorded in each
// DistributedFunction's declared argument and result shardings, so the export
// turns those into XLA's own notation: a global-shaped public @main in the
// form jax.shard_map lowers to, which enters the manual per-device program
// through @SPMDFullToShardShape and leaves it through @SPMDShardToFullShape,
// each argument and result carrying its tiled HloSharding. The boundary
// casts become the function signatures' local types and are erased.

#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Axis/Utilities.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DISTRIBUTEDTOSTABLEHLOPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

constexpr StringLiteral kShardingAttrName = "mhlo.sharding";
constexpr StringLiteral kManualSharding = "{manual}";
constexpr StringLiteral kReplicatedSharding = "{replicated}";
constexpr StringLiteral kEntryName = "main";
constexpr StringLiteral kPerDeviceEntryName = "main_per_device";

// The device-id digits each tensor dimension is split over, one list per
// dimension, its first digit the most significant.
using DimensionDigits = SmallVector<SmallVector<axis::AxisAtom>>;

// How one function argument or result crosses the local/global boundary: the
// per-device type, and the HloSharding of the global value.
struct BoundaryLayout {
  Type localType;
  std::string sharding;

  bool operator==(const BoundaryLayout &other) const {
    return localType == other.localType && sharding == other.sharding;
  }
  bool operator!=(const BoundaryLayout &other) const {
    return !(*this == other);
  }
};

// Global-scope ops at the entry's boundary, which move into the global @main
// so the per-device function's boundary is exactly its casts' layouts.
// CanonicalizeShardedFactorOrder relays out strided entry boundaries this way.
// - For an argument, `from` is the argument and `to` the value its
//   CastGlobalToLocals read.
// - For a result, `from` is a CastLocalToGlobal's output and `to` the
//   returned value.
// Operand-free ops they read (constants, iotas) may also feed per-device code,
// so those are cloned into @main instead of moved.
struct HoistedOps {
  SmallVector<Operation *> ops; // in program order
  SmallVector<Operation *> clonedOps;
  Value from;
  Value to;
};

// How one argument or result crosses the boundary: the layout entering or
// leaving the per-device function, the HloSharding of the caller's global
// value, and the ops moving into @main, if any. Without hoisted ops the
// caller's sharding is the layout's, and the declared layout must not be
// strided (SplitDigits).
struct BoundaryValue {
  BoundaryLayout layout;
  std::string callerSharding;
  bool strided = false;
  std::optional<HoistedOps> hoisted;
};

// The boundary of one DistributedFunction: each argument and result, and the
// boundary casts the export erases.
struct FunctionBoundary {
  SmallVector<BoundaryValue> arguments;
  SmallVector<BoundaryValue> results;
  SmallVector<Operation *> casts;
};

bool isExportedMetadata(Operation *op) {
  return isa_and_present<axis::AxisDialect>(op->getDialect()) ||
         isa<DeviceLocalAxisOp, ReplicationAxisOp, GetPhysicalMeshAxesOp,
             PhysicalMeshOp, sdy::MeshOp>(op);
}

bool isFromExportedDialect(Operation *op) {
  return isa_and_present<DistributedDialect, axis::AxisDialect,
                         sdy::SdyDialect>(op->getDialect());
}

bool isExportedAttrName(StringRef name) {
  return name.starts_with("sdy.") || name.starts_with("distributed.");
}

DictionaryAttr withoutExportedAttrs(DictionaryAttr attrs) {
  SmallVector<NamedAttribute> kept;
  for (NamedAttribute attr : attrs)
    if (!isExportedAttrName(attr.getName()))
      kept.push_back(attr);
  return DictionaryAttr::get(attrs.getContext(), kept);
}

// The device-id digits of a value's dimensions, and whether some dimension
// interleaves them with device-local factors. A device-local factor major to
// a physical one gives each device strided elements, which no tiled
// HloSharding expresses.
struct SplitDigits {
  DimensionDigits digits;
  bool interleaved = false;
};

// The device-id digits each dimension is split over, given the factor groups
// that split each dimension, major first. A physical factor of extent e and
// stride s on an axis of id stride t is the device-id digit of extent e and
// stride s * t. Device-local factors stay within a device's tile.
FailureOr<SplitDigits>
getDimensionDigits(Operation *diagnosticOp,
                   ArrayRef<SmallVector<Value>> groupsPerDimension) {
  SplitDigits split;
  for (ArrayRef<Value> groups : groupsPerDimension) {
    SmallVector<axis::AxisAtom> &dimension = split.digits.emplace_back();
    bool afterDeviceLocal = false;
    for (Value group : groups) {
      auto factors = axis::getProductProvenanceFactors(
          cast<TypedValue<axis::FactorGroupType>>(group));
      assert(succeeded(factors) &&
             "axis.product is the only op producing a factor group");
      for (TypedValue<axis::AxisFactorType> factor : *factors) {
        uint64_t extent = axis::getFactorExtent(factor);
        if (extent == 1)
          continue;
        Type axisType = factor.getType().getAxisType();
        if (isa<DeviceLocalAxisType>(axisType)) {
          afterDeviceLocal = true;
          continue;
        }
        auto physicalAxis = dyn_cast<PhysicalCommAxisType>(axisType);
        if (!physicalAxis)
          return diagnosticOp->emitError()
                 << "expected partitioning axes over physical mesh and "
                    "device-local axes, found "
                 << axisType;
        split.interleaved |= afterDeviceLocal;
        dimension.push_back(
            {extent, static_cast<uint64_t>(axis::getFactorStride(factor)) *
                         physicalAxis.getIdStride()});
      }
    }
  }
  return split;
}

LogicalResult emitInterleavedError(Operation *op) {
  return op->emitError() << "expected the physical factors of each dimension "
                            "to be major to its device-local factors";
}

// The HloSharding of a global tensor tiled over `digits` on `deviceCount`
// densely numbered devices, in XLA's iota notation:
//
//   {devices=[tile extents, replicas]<=[atom extents]T(permutation)
//    last_tile_dim_replicate}
//
// - Cutting the device ids at every digit's borders gives the atoms,
//   major-first, so an iota reshaped to the atom extents holds each device at
//   its own coordinates.
// - The permutation lists each dimension's atoms, in that dimension's digit
//   order, then the atoms no dimension uses, which become the replicated last
//   tile dimension.
std::string formatHloSharding(const DimensionDigits &digits,
                              uint64_t deviceCount) {
  SmallVector<uint64_t> cuts;
  for (ArrayRef<axis::AxisAtom> dimension : digits)
    for (axis::AxisAtom digit : dimension) {
      axis::FactorCuts borders =
          axis::getFactorCuts(digit.extent, digit.stride);
      cuts.append({borders.low, borders.high});
    }
  FailureOr<SmallVector<axis::AxisAtom>> atoms =
      axis::applyCutsSorted(deviceCount, cuts);
  assert(
      succeeded(atoms) &&
      "distinct digits of the device ids cut them into a divisibility chain");

  SmallVector<uint64_t> tileExtents;
  SmallVector<size_t> permutation;
  SmallVector<bool> used(atoms->size(), false);
  for (ArrayRef<axis::AxisAtom> dimension : digits) {
    uint64_t tileExtent = 1;
    for (axis::AxisAtom digit : dimension) {
      axis::AtomRange range =
          axis::getFactorAtomRange(*atoms, digit.extent, digit.stride);
      for (size_t atom = range.first; atom < range.first + range.count;
           ++atom) {
        assert(!used[atom] && "a device-id digit splits two dimensions");
        used[atom] = true;
        permutation.push_back(atom);
      }
      tileExtent *= digit.extent;
    }
    tileExtents.push_back(tileExtent);
  }
  uint64_t replicas = 1;
  for (auto [atom, isUsed] : llvm::enumerate(used))
    if (!isUsed) {
      permutation.push_back(atom);
      replicas *= (*atoms)[atom].extent;
    }
  if (replicas == deviceCount)
    return kReplicatedSharding.str();
  if (replicas > 1)
    tileExtents.push_back(replicas);

  std::string sharding;
  llvm::raw_string_ostream os(sharding);
  os << "{devices=[";
  llvm::interleave(tileExtents, os, ",");
  os << "]<=[";
  if (llvm::is_sorted(permutation)) {
    os << deviceCount << "]";
  } else {
    llvm::interleave(
        *atoms, os, [&](axis::AxisAtom atom) { os << atom.extent; }, ",");
    os << "]T(";
    llvm::interleave(permutation, os, ",");
    os << ")";
  }
  if (replicas > 1)
    os << " last_tile_dim_replicate";
  os << "}";
  return sharding;
}

// The layout of a value of `globalType` whose dimensions are split over
// `digits`.
BoundaryLayout getLayout(Type globalType, const DimensionDigits &digits,
                         uint64_t deviceCount) {
  auto tensorType = cast<RankedTensorType>(globalType);
  SmallVector<int64_t> localShape(tensorType.getShape());
  for (auto [size, dimension] : llvm::zip_equal(localShape, digits))
    for (axis::AxisAtom digit : dimension) {
      assert(size % static_cast<int64_t>(digit.extent) == 0 &&
             "a declared sharding splits its dimension evenly");
      size /= static_cast<int64_t>(digit.extent);
    }
  return {tensorType.clone(localShape), formatHloSharding(digits, deviceCount)};
}

// The boundary value `function` declares for a value of `globalType` with
// `sharding`: its layout over the physical digits, which is also the caller's
// sharding.
FailureOr<BoundaryValue> getDeclaredValue(DistributedFunctionOp function,
                                          Type globalType,
                                          IndexedTensorShardingAttr sharding,
                                          uint64_t deviceCount) {
  SmallVector<SmallVector<Value>> groupsPerDimension;
  for (DenseI64ArrayAttr groupIndices : sharding.getDimPartitioningAxes())
    groupsPerDimension.push_back(llvm::to_vector(
        llvm::map_range(groupIndices.asArrayRef(), [&](int64_t index) {
          return function.getPartitioningAxes()[index];
        })));
  FailureOr<SplitDigits> split =
      getDimensionDigits(function, groupsPerDimension);
  if (failed(split))
    return failure();
  BoundaryLayout layout = getLayout(globalType, split->digits, deviceCount);
  return BoundaryValue{layout, layout.sharding, split->interleaved,
                       std::nullopt};
}

// The layout of a boundary cast with `localType` and `partitioningAxes`, one
// factor group per dimension.
FailureOr<BoundaryLayout> getCastLayout(Operation *castOp, Type localType,
                                        ValueRange partitioningAxes,
                                        uint64_t deviceCount) {
  SmallVector<SmallVector<Value>> groupsPerDimension;
  for (Value group : partitioningAxes)
    groupsPerDimension.push_back({group});
  FailureOr<SplitDigits> split = getDimensionDigits(castOp, groupsPerDimension);
  if (failed(split))
    return failure();
  if (split->interleaved)
    return emitInterleavedError(castOp);
  return BoundaryLayout{localType,
                        formatHloSharding(split->digits, deviceCount)};
}

// Ops that can move from the entry's body into the global @main: StableHLO
// ops without regions or memory effects whose results don't depend on the
// device, so @main computes the same global values.
bool isHoistable(Operation *op) {
  return isa_and_present<stablehlo::StablehloDialect>(op->getDialect()) &&
         op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         !isa<stablehlo::PartitionIdOp, stablehlo::ReplicaIdOp,
              stablehlo::AllGatherOp, stablehlo::AllToAllOp,
              stablehlo::CollectivePermuteOp, stablehlo::CollectiveBroadcastOp,
              stablehlo::CustomCallOp, stablehlo::RngOp>(op);
}

bool isBefore(Operation *lhs, Operation *rhs) {
  return lhs->isBeforeInBlock(rhs);
}

// The hoistable ops at the top level of a block that a value is computed from.
struct BackwardSlice {
  SmallVector<Operation *> ops;           // with operands, in program order
  SmallVector<Operation *> clonedOps;     // operand-free, in program order
  llvm::SmallSetVector<Value, 2> sources; // values outside the slice it reads
};

// The slice of the hoistable ops in `body` that `root` is computed from.
BackwardSlice getHoistableSlice(Value root, Block &body) {
  BackwardSlice slice;
  llvm::SmallPtrSet<Operation *, 8> seen;
  SmallVector<Value> worklist{root};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *producer = value.getDefiningOp();
    if (!producer || producer->getBlock() != &body || !isHoistable(producer)) {
      slice.sources.insert(value);
      continue;
    }
    if (!seen.insert(producer).second)
      continue;
    if (producer->getNumOperands() == 0) {
      slice.clonedOps.push_back(producer);
      continue;
    }
    slice.ops.push_back(producer);
    llvm::append_range(worklist, producer->getOperands());
  }
  llvm::sort(slice.ops, isBefore);
  llvm::sort(slice.clonedOps, isBefore);
  return slice;
}

// Whether every user of `value` is in `members` or is `allowed`.
bool usedOnlyBy(Value value, const llvm::SmallPtrSetImpl<Operation *> &members,
                Operation *allowed = nullptr) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return members.contains(user) || user == allowed;
  });
}

// The entry's argument prologues, by argument number: for each
// CastGlobalToLocal that reads a computed value instead of an argument, the
// hoistable ops computing it from one argument. Casts of anything else are
// left to the stray-cast check. Fails with a diagnostic when an argument
// reaches casts through two different values, or when a prologue op or the
// argument itself is also used outside the prologue.
FailureOr<DenseMap<unsigned, HoistedOps>>
findPrologues(DistributedFunctionOp function,
              SmallVectorImpl<Operation *> &casts) {
  Block &body = function.getBody().front();
  DenseMap<unsigned, HoistedOps> prologues;
  DenseMap<unsigned, SmallVector<Operation *>> castsOf;
  for (auto castOp : body.getOps<DistributedCastGlobalToLocalOp>()) {
    if (isa<BlockArgument>(castOp.getInput()))
      continue;
    // Casts of one relaid-out value share its prologue.
    if (llvm::any_of(prologues, [&](const auto &entry) {
          return entry.second.to == castOp.getInput();
        })) {
      for (auto &entry : prologues)
        if (entry.second.to == castOp.getInput())
          castsOf[entry.first].push_back(castOp);
      continue;
    }
    BackwardSlice slice = getHoistableSlice(castOp.getInput(), body);
    if (slice.sources.size() != 1 || !isa<BlockArgument>(slice.sources[0]))
      continue;
    auto argument = cast<BlockArgument>(slice.sources[0]);
    HoistedOps &prologue = prologues[argument.getArgNumber()];
    if (prologue.to)
      return function.emitError()
             << "argument #" << argument.getArgNumber()
             << " reaches its casts through more than one value";
    prologue = {std::move(slice.ops), std::move(slice.clonedOps), argument,
                castOp.getInput()};
    castsOf[argument.getArgNumber()].push_back(castOp);
  }
  for (auto &entry : prologues) {
    unsigned index = entry.first;
    const HoistedOps &prologue = entry.second;
    llvm::SmallPtrSet<Operation *, 8> members(prologue.ops.begin(),
                                              prologue.ops.end());
    members.insert(castsOf[index].begin(), castsOf[index].end());
    if (llvm::any_of(prologue.from.getUsers(),
                     llvm::IsaPred<DistributedCastGlobalToLocalOp>))
      return function.emitError()
             << "argument #" << index
             << " reaches its casts through more than one value";
    bool usedOnlyInPrologue =
        usedOnlyBy(prologue.from, members) &&
        llvm::all_of(prologue.ops, [&](Operation *op) {
          return llvm::all_of(op->getResults(), [&](Value result) {
            return usedOnlyBy(result, members);
          });
        });
    if (!usedOnlyInPrologue)
      return function.emitError()
             << "argument #" << index
             << " is used at its global type outside the ops that compute "
                "its casts' value";
    llvm::append_range(casts, castsOf[index]);
  }
  return prologues;
}

// The entry's result epilogues, by result number: for each CastLocalToGlobal
// whose output feeds hoistable ops instead of only the yield, those ops up to
// the one value they return. A cast feeding anything else is left to the
// stray-cast check. Fails with a diagnostic when the ops read values other
// than the cast's output and operand-free ops, or when the cast's value feeds
// anything besides one result's computation.
FailureOr<DenseMap<unsigned, HoistedOps>>
findEpilogues(DistributedFunctionOp function,
              SmallVectorImpl<Operation *> &casts) {
  Block &body = function.getBody().front();
  Operation *yield = body.getTerminator();
  DenseMap<unsigned, HoistedOps> epilogues;
  for (auto castOp : body.getOps<DistributedCastLocalToGlobalOp>()) {
    Value castOutput = castOp.getOutput();
    if (llvm::none_of(castOutput.getUsers(), isHoistable))
      continue;
    llvm::SetVector<Operation *> ops;
    SmallVector<OpOperand *> returns;
    bool escapes = false;
    SmallVector<Value> worklist{castOutput};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (user == yield)
          returns.push_back(&use);
        else if (!isHoistable(user))
          escapes = true;
        else if (ops.insert(user))
          llvm::append_range(worklist, user->getResults());
      }
    }
    // A global value that never reaches the yield is a relayout inside the
    // body, which the stray-cast check reports.
    if (returns.empty())
      continue;
    if (escapes || returns.size() != 1 || returns.front()->get() == castOutput)
      return castOp.emitError()
             << "expected this cast's global value to feed exactly one "
                "result's computation and nothing else";
    unsigned index = returns.front()->getOperandNumber();
    SmallVector<Operation *> clonedOps;
    for (Operation *op : ops)
      for (Value operand : op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (operand == castOutput || (producer && ops.contains(producer)))
          continue;
        if (!producer || producer->getNumOperands() != 0 ||
            !isHoistable(producer))
          return function.emitError()
                 << "result #" << index
                 << " is computed at its global type from values other "
                    "than one CastLocalToGlobal";
        if (!llvm::is_contained(clonedOps, producer))
          clonedOps.push_back(producer);
      }
    SmallVector<Operation *> sorted = ops.takeVector();
    llvm::sort(sorted, isBefore);
    llvm::sort(clonedOps, isBefore);
    epilogues[index] = {std::move(sorted), std::move(clonedOps), castOutput,
                        returns.front()->get()};
    casts.push_back(castOp);
  }
  return epilogues;
}

// Reads `function`'s boundary from its argument_shardings and
// output_shardings. Canonicalization folds away the casts of a value that
// passes straight through, so the declared shardings are the layouts, and
// the casts that survive only cross-check them.
//
// In the entry, an argument or result relaid out at global scope (see
// HoistedOps) instead takes its layout from its casts, and its declared
// layout only gives the caller's sharding, which may be strided.
//
// Fails with a diagnostic when the body doesn't match the declared layouts:
// - a boundary cast whose layout differs from its value's declared one;
// - a sharded argument used at its global type, other than returned as a
//   result of the same layout;
// - a sharded result without a CastLocalToGlobal;
// - a strided declared layout on a value without hoisted ops.
FailureOr<FunctionBoundary> analyzeBoundary(DistributedFunctionOp function,
                                            uint64_t deviceCount,
                                            bool isEntry) {
  FunctionBoundary boundary;
  FunctionType type = function.getFunctionType();
  auto declare = [&](TypeRange globalTypes,
                     ArrayRef<IndexedTensorShardingAttr> shardings,
                     SmallVectorImpl<BoundaryValue> &values) -> LogicalResult {
    for (auto [globalType, sharding] :
         llvm::zip_equal(globalTypes, shardings)) {
      FailureOr<BoundaryValue> declared =
          getDeclaredValue(function, globalType, sharding, deviceCount);
      if (failed(declared))
        return failure();
      values.push_back(std::move(*declared));
    }
    return success();
  };
  if (failed(declare(type.getInputs(),
                     function.getArgumentShardings().getShardings(),
                     boundary.arguments)) ||
      failed(declare(type.getResults(),
                     function.getOutputShardings().getShardings(),
                     boundary.results)))
    return failure();

  DenseMap<unsigned, HoistedOps> prologues, epilogues;
  if (isEntry) {
    FailureOr<DenseMap<unsigned, HoistedOps>> foundPrologues =
        findPrologues(function, boundary.casts);
    if (failed(foundPrologues))
      return failure();
    FailureOr<DenseMap<unsigned, HoistedOps>> foundEpilogues =
        findEpilogues(function, boundary.casts);
    if (failed(foundEpilogues))
      return failure();
    prologues = std::move(*foundPrologues);
    epilogues = std::move(*foundEpilogues);
  }

  Block &body = function.getBody().front();
  auto yield = cast<DistributedYieldOp>(body.getTerminator());
  for (auto [argument, value] :
       llvm::zip_equal(body.getArguments(), boundary.arguments)) {
    if (auto it = prologues.find(argument.getArgNumber());
        it != prologues.end()) {
      SmallVector<Operation *> cutCasts(it->second.to.getUsers());
      llvm::sort(cutCasts, isBefore);
      std::optional<BoundaryLayout> cutLayout;
      for (Operation *user : cutCasts) {
        auto castOp = cast<DistributedCastGlobalToLocalOp>(user);
        FailureOr<BoundaryLayout> castLayout =
            getCastLayout(castOp, castOp.getType(),
                          castOp.getPartitioningAxes(), deviceCount);
        if (failed(castLayout))
          return failure();
        if (cutLayout && *castLayout != *cutLayout)
          return castOp.emitError() << "argument #" << argument.getArgNumber()
                                    << " has casts to different layouts";
        cutLayout = *castLayout;
      }
      value.layout = *cutLayout;
      value.hoisted = std::move(it->second);
      continue;
    }
    if (value.strided)
      return emitInterleavedError(function);
    bool isSharded = value.layout.localType != argument.getType();
    for (Operation *user : argument.getUsers()) {
      auto castOp = dyn_cast<DistributedCastGlobalToLocalOp>(user);
      if (!castOp) {
        // A returned argument is checked against its result's layout below.
        if (isSharded && user != yield)
          return function.emitError()
                 << "argument #" << argument.getArgNumber()
                 << " is sharded but used at its global type";
        continue;
      }
      FailureOr<BoundaryLayout> castLayout = getCastLayout(
          castOp, castOp.getType(), castOp.getPartitioningAxes(), deviceCount);
      if (failed(castLayout))
        return failure();
      if (*castLayout != value.layout)
        return castOp.emitError()
               << "argument #" << argument.getArgNumber() << " has a cast to "
               << castLayout->localType << " " << castLayout->sharding
               << ", but its declared layout is " << value.layout.localType
               << " " << value.layout.sharding;
      boundary.casts.push_back(castOp);
    }
  }

  for (auto [index, returned, value] :
       llvm::enumerate(yield.getReturns(), boundary.results)) {
    if (auto it = epilogues.find(index); it != epilogues.end()) {
      auto castOp =
          it->second.from.getDefiningOp<DistributedCastLocalToGlobalOp>();
      FailureOr<BoundaryLayout> castLayout =
          getCastLayout(castOp, castOp.getInput().getType(),
                        castOp.getPartitioningAxes(), deviceCount);
      if (failed(castLayout))
        return failure();
      value.layout = *castLayout;
      value.hoisted = std::move(it->second);
      continue;
    }
    if (value.strided)
      return emitInterleavedError(function);
    if (auto argument = dyn_cast<BlockArgument>(returned)) {
      const BoundaryValue &argumentValue =
          boundary.arguments[argument.getArgNumber()];
      assert(!argumentValue.hoisted &&
             "findPrologues rejects a prologue's argument used elsewhere");
      if (argumentValue.layout != value.layout)
        return function.emitError()
               << "result #" << index << " returns argument #"
               << argument.getArgNumber() << ", whose declared layout differs";
      continue;
    }
    auto castOp = returned.getDefiningOp<DistributedCastLocalToGlobalOp>();
    if (!castOp || !llvm::all_of(castOp->getUsers(), [&](Operation *user) {
          return user == yield;
        })) {
      if (value.layout.localType != returned.getType())
        return function.emitError()
               << "result #" << index
               << " is sharded but has no CastLocalToGlobal";
      continue;
    }
    FailureOr<BoundaryLayout> castLayout =
        getCastLayout(castOp, castOp.getInput().getType(),
                      castOp.getPartitioningAxes(), deviceCount);
    if (failed(castLayout))
      return failure();
    if (*castLayout != value.layout)
      return castOp.emitError()
             << "result #" << index << " has a cast from "
             << castLayout->localType << " " << castLayout->sharding
             << ", but its declared layout is " << value.layout.localType << " "
             << value.layout.sharding;
    boundary.casts.push_back(castOp);
  }
  return boundary;
}

// Fails with a diagnostic if a result of the entry function is a partial
// value: unreduced over a partitioning axis that actually splits devices.
LogicalResult verifyEntryResultsReduced(DistributedFunctionOp entry) {
  for (auto [index, sharding] :
       llvm::enumerate(entry.getOutputShardings().getShardings())) {
    for (int64_t axisIndex : sharding.getUnreducedAxes().asArrayRef()) {
      auto groupType = cast<axis::FactorGroupType>(
          entry.getPartitioningAxes()[axisIndex].getType());
      if (groupType.getExtent() > 1)
        return entry.emitError()
               << "result #" << index
               << " is unreduced; exporting needs reduced entry results";
    }
  }
  return success();
}

// Fails with a diagnostic on anything the export can't translate. Casts are
// checked against the function boundaries afterwards.
LogicalResult verifyExportable(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (isa<DistributedCollectiveOp, DistributedAwait>(op)) {
      op->emitError() << "unlowered " << op->getName()
                      << "; export a schedule finalized by "
                         "distributed-search-lowering-pipeline";
      return WalkResult::interrupt();
    }
    if (isExportedMetadata(op) ||
        isa<DistributedFunctionOp, DistributedYieldOp, DistributedCallOp,
            DistributedCastGlobalToLocalOp, DistributedCastLocalToGlobalOp>(
            op) ||
        !isFromExportedDialect(op))
      return WalkResult::advance();
    op->emitError() << "unexpected " << op->getName()
                    << " in a schedule to export";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

// Replaces `function` by a func.func named `name` whose signature is its
// boundary's local types, with the boundary casts erased. Hoisted ops are
// detached from the body, still reading their `from` values, for
// buildGlobalEntry to insert into @main.
func::FuncOp exportFunction(DistributedFunctionOp function,
                            const FunctionBoundary &boundary, StringRef name,
                            StringAttr visibility) {
  Block &body = function.getBody().front();
  for (auto [argument, value] :
       llvm::zip_equal(body.getArguments(), boundary.arguments)) {
    if (value.hoisted) {
      // findPrologues accepted only casts as users of the relaid-out value.
      for (Operation *user : llvm::to_vector(value.hoisted->to.getUsers())) {
        cast<DistributedCastGlobalToLocalOp>(user)
            .getResult()
            .replaceAllUsesWith(argument);
        user->erase();
      }
      for (Operation *op : value.hoisted->ops)
        op->remove();
    }
    // Snapshot the users, since each replacement adds uses to the argument.
    for (Operation *user : llvm::to_vector(argument.getUsers())) {
      if (!isa<DistributedCastGlobalToLocalOp>(user))
        continue;
      user->getResult(0).replaceAllUsesWith(argument);
      user->erase();
    }
    argument.setType(value.layout.localType);
  }

  // analyzeBoundary accepted a CastLocalToGlobal feeding the yield, or an
  // epilogue, only as a boundary cast.
  auto yield = cast<DistributedYieldOp>(body.getTerminator());
  SmallVector<Value> returned;
  llvm::SmallSetVector<Operation *, 4> resultCasts;
  SmallVector<Operation *> detached;
  for (auto [value, result] :
       llvm::zip_equal(yield.getReturns(), boundary.results)) {
    if (result.hoisted) {
      Operation *castOp = result.hoisted->from.getDefiningOp();
      returned.push_back(castOp->getOperand(0));
      detached.push_back(castOp);
      llvm::append_range(detached, result.hoisted->ops);
      continue;
    }
    auto castOp = value.getDefiningOp<DistributedCastLocalToGlobalOp>();
    if (!castOp) {
      returned.push_back(value);
      continue;
    }
    returned.push_back(castOp.getInput());
    resultCasts.insert(castOp);
  }
  OpBuilder builder(yield);
  func::ReturnOp::create(builder, yield.getLoc(), returned);
  yield.erase();
  for (Operation *castOp : resultCasts)
    castOp->erase();
  for (Operation *op : detached)
    op->remove();

  auto convertAttrs = [](ArrayAttr attrs) -> ArrayAttr {
    if (!attrs)
      return nullptr;
    return ArrayAttr::get(
        attrs.getContext(),
        llvm::to_vector(llvm::map_range(attrs, [](Attribute dict) -> Attribute {
          return withoutExportedAttrs(cast<DictionaryAttr>(dict));
        })));
  };

  builder.setInsertionPoint(function);
  auto type = builder.getFunctionType(body.getArgumentTypes(),
                                      ValueRange(returned).getTypes());
  auto exported = func::FuncOp::create(builder, function.getLoc(), name, type);
  exported.setSymVisibilityAttr(visibility);
  if (ArrayAttr argAttrs = convertAttrs(function.getArgAttrsAttr()))
    exported.setArgAttrsAttr(argAttrs);
  if (ArrayAttr resAttrs = convertAttrs(function.getResAttrsAttr()))
    exported.setResAttrsAttr(resAttrs);
  exported.getBody().takeBody(function.getBody());
  function.erase();
  return exported;
}

Value createShardingCustomCall(OpBuilder &builder, Location loc,
                               StringRef target, Value operand, Type resultType,
                               StringRef sharding) {
  NamedAttribute attrs[] = {
      builder.getNamedAttr("call_target_name", builder.getStringAttr(target)),
      builder.getNamedAttr("backend_config", builder.getStringAttr("")),
      builder.getNamedAttr(kShardingAttrName, builder.getStringAttr(sharding)),
  };
  return stablehlo::CustomCallOp::create(builder, loc, TypeRange{resultType},
                                         ValueRange{operand}, attrs)
      .getResult(0);
}

// Inserts the detached `hoisted` ops at `builder`'s insertion point, reading
// `replacement` where they read `from`, and returns their `to` value.
Value insertHoisted(OpBuilder &builder, const HoistedOps &hoisted,
                    Value replacement) {
  IRMapping mapping;
  mapping.map(hoisted.from, replacement);
  // Clones stand in for the operand-free ops, which per-device code may also
  // read.
  for (Operation *op : hoisted.clonedOps)
    builder.clone(*op, mapping);
  for (Operation *op : hoisted.ops) {
    builder.insert(op);
    for (OpOperand &operand : op->getOpOperands())
      if (Value mapped = mapping.lookupOrNull(operand.get()))
        operand.set(mapped);
  }
  // A result's `from` is its CastLocalToGlobal, now unread.
  if (Operation *castOp = hoisted.from.getDefiningOp())
    castOp->erase();
  // Originals no per-device code reads are dead.
  for (Operation *op : hoisted.clonedOps)
    if (op->use_empty())
      op->erase();
  return hoisted.to;
}

// Builds the public, global-shaped @main over `perDevice`, in the form
// jax.shard_map lowers to:
//
//   %s = custom_call @Sharding(%global) {mhlo.sharding = <layout>}
//   %l = custom_call @SPMDFullToShardShape(%s) {mhlo.sharding = "{manual}"}
//   %r = call @main_per_device(%l, ...)
//   %m = custom_call @Sharding(%r) {mhlo.sharding = "{manual}"}
//   %g = custom_call @SPMDShardToFullShape(%m) {mhlo.sharding = <layout>}
//
// `globalType` is the entry DistributedFunction's signature, and each
// argument and result carries its caller's sharding as mhlo.sharding. Hoisted
// ops run on the global values: an argument's between %global and @Sharding,
// a result's after @SPMDShardToFullShape.
void buildGlobalEntry(func::FuncOp perDevice, FunctionType globalType,
                      const FunctionBoundary &boundary) {
  OpBuilder builder(perDevice);
  Location loc = perDevice.getLoc();
  auto entry = func::FuncOp::create(builder, loc, kEntryName, globalType);

  auto withSharding = [&](DictionaryAttr attrs, StringRef sharding) {
    NamedAttrList list(attrs);
    list.set(kShardingAttrName, builder.getStringAttr(sharding));
    return list.getDictionary(builder.getContext());
  };
  for (auto [index, value] : llvm::enumerate(boundary.arguments))
    entry.setArgAttrs(index, withSharding(perDevice.getArgAttrDict(index),
                                          value.callerSharding));
  for (auto [index, value] : llvm::enumerate(boundary.results))
    entry.setResultAttrs(index, withSharding(perDevice.getResultAttrDict(index),
                                             value.callerSharding));

  Block *block = entry.addEntryBlock();
  builder.setInsertionPointToStart(block);
  SmallVector<Value> localArguments;
  for (auto [global, localType, value] : llvm::zip_equal(
           block->getArguments(), perDevice.getFunctionType().getInputs(),
           boundary.arguments)) {
    Value relaidOut =
        value.hoisted ? insertHoisted(builder, *value.hoisted, global) : global;
    Value annotated =
        createShardingCustomCall(builder, loc, "Sharding", relaidOut,
                                 relaidOut.getType(), value.layout.sharding);
    localArguments.push_back(
        createShardingCustomCall(builder, loc, "SPMDFullToShardShape",
                                 annotated, localType, kManualSharding));
  }
  auto call = func::CallOp::create(builder, loc, perDevice, localArguments);
  SmallVector<Value> globalResults;
  for (auto [local, globalResultType, value] : llvm::zip_equal(
           call.getResults(), globalType.getResults(), boundary.results)) {
    Value manual = createShardingCustomCall(builder, loc, "Sharding", local,
                                            local.getType(), kManualSharding);
    Type fullType =
        value.hoisted ? value.hoisted->from.getType() : globalResultType;
    Value full =
        createShardingCustomCall(builder, loc, "SPMDShardToFullShape", manual,
                                 fullType, value.layout.sharding);
    globalResults.push_back(
        value.hoisted ? insertHoisted(builder, *value.hoisted, full) : full);
  }
  func::ReturnOp::create(builder, loc, globalResults);
}

void replaceCall(DistributedCallOp call) {
  OpBuilder builder(call);
  auto exported =
      func::CallOp::create(builder, call.getLoc(), call.getCalleeAttr(),
                           call.getResultTypes(), call.getArguments());
  call.replaceAllUsesWith(exported.getResults());
  call.erase();
}

void eraseMetadata(ModuleOp module) {
  SmallVector<Operation *> metadata;
  module.walk([&](Operation *op) {
    if (isExportedMetadata(op))
      metadata.push_back(op);
  });
  for (Operation *op : metadata)
    op->dropAllReferences();
  for (Operation *op : metadata)
    op->erase();
}

void stripExportedAttrs(ModuleOp module) {
  module.walk([](Operation *op) {
    for (NamedAttribute attr :
         llvm::to_vector(op->getDiscardableAttrDictionary()))
      if (isExportedAttrName(attr.getName()))
        op->removeDiscardableAttr(attr.getName());
  });
}

struct DistributedToStablehloPass
    : public impl::DistributedToStablehloPassBase<DistributedToStablehloPass> {
  using DistributedToStablehloPassBase::DistributedToStablehloPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyExportable(module)))
      return signalPassFailure();

    FailureOr<PhysicalMeshOp> mesh = findUniquePhysicalMesh(module);
    if (failed(mesh))
      return signalPassFailure();
    SmallVector<PhysicalCommAxisType> physicalAxes;
    for (Attribute axisAttr : mesh->getAxesAttr())
      physicalAxes.push_back(
          cast<PhysicalCommAxisType>(cast<TypeAttr>(axisAttr).getValue()));
    if (!numbersDevicesDensely(physicalAxes)) {
      mesh->emitError() << "expected the physical axes to number devices "
                           "densely as 0..N-1";
      return signalPassFailure();
    }
    uint64_t deviceCount = getDenseDeviceCount(physicalAxes);

    DistributedFunctionOp entry;
    std::vector<std::pair<DistributedFunctionOp, FunctionBoundary>> functions;
    llvm::SmallPtrSet<Operation *, 16> boundaryCasts;
    for (auto function : module.getOps<DistributedFunctionOp>()) {
      bool isEntry = function.getSymName() == kEntryName;
      FailureOr<FunctionBoundary> boundary =
          analyzeBoundary(function, deviceCount, isEntry);
      if (failed(boundary))
        return signalPassFailure();
      boundaryCasts.insert(boundary->casts.begin(), boundary->casts.end());
      if (isEntry)
        entry = function;
      functions.emplace_back(function, std::move(*boundary));
    }
    if (!entry) {
      module.emitError() << "expected a distributed.DistributedFunction named @"
                         << kEntryName;
      return signalPassFailure();
    }
    if (failed(verifyEntryResultsReduced(entry)))
      return signalPassFailure();
    WalkResult strayCast = module.walk([&](Operation *op) {
      if (!isa<DistributedCastGlobalToLocalOp, DistributedCastLocalToGlobalOp>(
              op) ||
          boundaryCasts.contains(op))
        return WalkResult::advance();
      op->emitError() << "expected a cast only at a function boundary: on an "
                         "argument, or feeding only the function's results";
      return WalkResult::interrupt();
    });
    if (strayCast.wasInterrupted())
      return signalPassFailure();

    MLIRContext *context = &getContext();
    FunctionType globalEntryType = entry.getFunctionType();
    StringAttr privateVisibility = StringAttr::get(context, "private");
    for (auto &[function, boundary] : functions) {
      if (function != entry) {
        exportFunction(function, boundary, function.getSymName(),
                       function.getSymVisibilityAttr());
        continue;
      }
      func::FuncOp perDevice = exportFunction(
          function, boundary, kPerDeviceEntryName, privateVisibility);
      buildGlobalEntry(perDevice, globalEntryType, boundary);
    }
    module.walk([](DistributedCallOp call) { replaceCall(call); });

    eraseMetadata(module);
    stripExportedAttrs(module);
    Builder builder(context);
    module->setAttr("mhlo.num_partitions",
                    builder.getI32IntegerAttr(deviceCount));
    module->setAttr("mhlo.num_replicas", builder.getI32IntegerAttr(1));
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
