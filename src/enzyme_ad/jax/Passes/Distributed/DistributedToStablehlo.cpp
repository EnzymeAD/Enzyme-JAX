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

// The boundary of one DistributedFunction: each argument's and result's
// declared layout, and the boundary casts the export erases.
struct FunctionBoundary {
  SmallVector<BoundaryLayout> arguments;
  SmallVector<BoundaryLayout> results;
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

// The device-id digits each dimension is split over, given the factor groups
// that split each dimension, major first. A physical factor of extent e and
// stride s on an axis of id stride t is the device-id digit of extent e and
// stride s * t. Device-local factors stay within a device's tile, so they
// must be minor to every physical factor of their dimension.
FailureOr<DimensionDigits>
getDimensionDigits(Operation *diagnosticOp,
                   ArrayRef<SmallVector<Value>> groupsPerDimension) {
  DimensionDigits digits;
  for (ArrayRef<Value> groups : groupsPerDimension) {
    SmallVector<axis::AxisAtom> &dimension = digits.emplace_back();
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
        if (afterDeviceLocal)
          return diagnosticOp->emitError()
                 << "expected the physical factors of each dimension to be "
                    "major to its device-local factors";
        dimension.push_back(
            {extent, static_cast<uint64_t>(axis::getFactorStride(factor)) *
                         physicalAxis.getIdStride()});
      }
    }
  }
  return digits;
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

// The layout `function` declares for a value of `globalType` with `sharding`.
FailureOr<BoundaryLayout> getDeclaredLayout(DistributedFunctionOp function,
                                            Type globalType,
                                            IndexedTensorShardingAttr sharding,
                                            uint64_t deviceCount) {
  SmallVector<SmallVector<Value>> groupsPerDimension;
  for (DenseI64ArrayAttr groupIndices : sharding.getDimPartitioningAxes())
    groupsPerDimension.push_back(llvm::to_vector(
        llvm::map_range(groupIndices.asArrayRef(), [&](int64_t index) {
          return function.getPartitioningAxes()[index];
        })));
  FailureOr<DimensionDigits> digits =
      getDimensionDigits(function, groupsPerDimension);
  if (failed(digits))
    return failure();
  return getLayout(globalType, *digits, deviceCount);
}

// The layout of a boundary cast with `localType` and `partitioningAxes`, one
// factor group per dimension.
FailureOr<BoundaryLayout> getCastLayout(Operation *castOp, Type localType,
                                        ValueRange partitioningAxes,
                                        uint64_t deviceCount) {
  SmallVector<SmallVector<Value>> groupsPerDimension;
  for (Value group : partitioningAxes)
    groupsPerDimension.push_back({group});
  FailureOr<DimensionDigits> digits =
      getDimensionDigits(castOp, groupsPerDimension);
  if (failed(digits))
    return failure();
  return BoundaryLayout{localType, formatHloSharding(*digits, deviceCount)};
}

// Reads `function`'s boundary from its argument_shardings and
// output_shardings. Canonicalization folds away the casts of a value that
// passes straight through, so the declared shardings are the layouts, and
// the casts that survive only cross-check them. Fails with a diagnostic when
// the body doesn't match the declared layouts:
// - a boundary cast whose layout differs from its value's declared one;
// - a sharded argument used at its global type, other than returned as a
//   result of the same layout;
// - a sharded result without a CastLocalToGlobal.
FailureOr<FunctionBoundary> analyzeBoundary(DistributedFunctionOp function,
                                            uint64_t deviceCount) {
  FunctionBoundary boundary;
  FunctionType type = function.getFunctionType();
  for (auto [globalType, sharding] : llvm::zip_equal(
           type.getInputs(), function.getArgumentShardings().getShardings())) {
    FailureOr<BoundaryLayout> layout =
        getDeclaredLayout(function, globalType, sharding, deviceCount);
    if (failed(layout))
      return failure();
    boundary.arguments.push_back(*layout);
  }
  for (auto [globalType, sharding] : llvm::zip_equal(
           type.getResults(), function.getOutputShardings().getShardings())) {
    FailureOr<BoundaryLayout> layout =
        getDeclaredLayout(function, globalType, sharding, deviceCount);
    if (failed(layout))
      return failure();
    boundary.results.push_back(*layout);
  }

  Block &body = function.getBody().front();
  auto yield = cast<DistributedYieldOp>(body.getTerminator());
  for (auto [argument, declared] :
       llvm::zip_equal(body.getArguments(), boundary.arguments)) {
    bool isSharded = declared.localType != argument.getType();
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
      if (*castLayout != declared)
        return castOp.emitError()
               << "argument #" << argument.getArgNumber() << " has a cast to "
               << castLayout->localType << " " << castLayout->sharding
               << ", but its declared layout is " << declared.localType << " "
               << declared.sharding;
      boundary.casts.push_back(castOp);
    }
  }

  for (auto [index, returned, declared] :
       llvm::enumerate(yield.getReturns(), boundary.results)) {
    if (auto argument = dyn_cast<BlockArgument>(returned)) {
      if (boundary.arguments[argument.getArgNumber()] != declared)
        return function.emitError()
               << "result #" << index << " returns argument #"
               << argument.getArgNumber() << ", whose declared layout differs";
      continue;
    }
    auto castOp = returned.getDefiningOp<DistributedCastLocalToGlobalOp>();
    if (!castOp || !llvm::all_of(castOp->getUsers(), [&](Operation *user) {
          return user == yield;
        })) {
      if (declared.localType != returned.getType())
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
    if (*castLayout != declared)
      return castOp.emitError()
             << "result #" << index << " has a cast from "
             << castLayout->localType << " " << castLayout->sharding
             << ", but its declared layout is " << declared.localType << " "
             << declared.sharding;
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
// boundary's local types, with the boundary casts erased.
func::FuncOp exportFunction(DistributedFunctionOp function,
                            const FunctionBoundary &boundary, StringRef name,
                            StringAttr visibility) {
  Block &body = function.getBody().front();
  for (auto [argument, layout] :
       llvm::zip_equal(body.getArguments(), boundary.arguments)) {
    // Snapshot the users, since each replacement adds uses to the argument.
    for (Operation *user : llvm::to_vector(argument.getUsers())) {
      if (!isa<DistributedCastGlobalToLocalOp>(user))
        continue;
      user->getResult(0).replaceAllUsesWith(argument);
      user->erase();
    }
    argument.setType(layout.localType);
  }

  // analyzeBoundary accepted a CastLocalToGlobal feeding the yield only as a
  // boundary cast.
  auto yield = cast<DistributedYieldOp>(body.getTerminator());
  SmallVector<Value> returned;
  llvm::SmallSetVector<Operation *, 4> resultCasts;
  for (Value value : yield.getReturns()) {
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
// argument and result carries its layout as mhlo.sharding.
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
  for (auto [index, layout] : llvm::enumerate(boundary.arguments))
    entry.setArgAttrs(
        index, withSharding(perDevice.getArgAttrDict(index), layout.sharding));
  for (auto [index, layout] : llvm::enumerate(boundary.results))
    entry.setResultAttrs(index, withSharding(perDevice.getResultAttrDict(index),
                                             layout.sharding));

  Block *block = entry.addEntryBlock();
  builder.setInsertionPointToStart(block);
  SmallVector<Value> localArguments;
  for (auto [global, localType, layout] : llvm::zip_equal(
           block->getArguments(), perDevice.getFunctionType().getInputs(),
           boundary.arguments)) {
    Value annotated = createShardingCustomCall(
        builder, loc, "Sharding", global, global.getType(), layout.sharding);
    localArguments.push_back(
        createShardingCustomCall(builder, loc, "SPMDFullToShardShape",
                                 annotated, localType, kManualSharding));
  }
  auto call = func::CallOp::create(builder, loc, perDevice, localArguments);
  SmallVector<Value> globalResults;
  for (auto [local, globalResultType, layout] : llvm::zip_equal(
           call.getResults(), globalType.getResults(), boundary.results)) {
    Value manual = createShardingCustomCall(builder, loc, "Sharding", local,
                                            local.getType(), kManualSharding);
    globalResults.push_back(
        createShardingCustomCall(builder, loc, "SPMDShardToFullShape", manual,
                                 globalResultType, layout.sharding));
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
    SmallVector<std::pair<DistributedFunctionOp, FunctionBoundary>> functions;
    llvm::SmallPtrSet<Operation *, 16> boundaryCasts;
    for (auto function : module.getOps<DistributedFunctionOp>()) {
      FailureOr<FunctionBoundary> boundary =
          analyzeBoundary(function, deviceCount);
      if (failed(boundary))
        return signalPassFailure();
      boundaryCasts.insert(boundary->casts.begin(), boundary->casts.end());
      if (function.getSymName() == kEntryName)
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
