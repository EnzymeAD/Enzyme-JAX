#include "Utilities.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <functional>

namespace mlir::enzyme::distributed {

using namespace ::mlir::enzyme::axis;

namespace {

::mlir::Region *
buildSyntheticReductionBody(::mlir::stablehlo::ReduceOpKind reductionKind,
                            ::mlir::Type elementType,
                            std::shared_ptr<::mlir::Region> &storage) {
  if (!elementType ||
      reductionKind == ::mlir::stablehlo::ReduceOpKind::Unknown) {
    return nullptr;
  }

  auto context = elementType.getContext();
  storage = std::make_shared<::mlir::Region>();
  ::mlir::Block *block = new ::mlir::Block();
  storage->push_back(block);

  auto scalarTensorType = ::mlir::RankedTensorType::get({}, elementType);
  auto loc = ::mlir::UnknownLoc::get(context);
  block->addArgument(scalarTensorType, loc);
  block->addArgument(scalarTensorType, loc);

  ::mlir::OpBuilder builder(context);
  builder.setInsertionPointToStart(block);
  auto lhs = block->getArgument(0);
  auto rhs = block->getArgument(1);
  ::mlir::Value result = ::mlir::stablehlo::CreateReductionOpGeneral(
      builder, loc, reductionKind, lhs, rhs);

  builder.create<::mlir::stablehlo::ReturnOp>(loc, result);
  return storage.get();
}

} // namespace

Operation *lookupSymbolInEnclosingScopes(Operation *from,
                                         FlatSymbolRefAttr symRef) {
  if (!from || !symRef) {
    return nullptr;
  }

  for (auto *scope = from; scope; scope = scope->getParentOp()) {
    if (!scope->hasTrait<OpTrait::SymbolTable>()) {
      continue;
    }
    if (auto *op = SymbolTable::lookupSymbolIn(scope, symRef)) {
      return op;
    }
  }

  return nullptr;
}

bool numbersDevicesDensely(ArrayRef<PhysicalCommAxisType> physicalAxes) {
  SmallVector<PhysicalCommAxisType> byIdStride(physicalAxes);
  llvm::sort(byIdStride,
             [](PhysicalCommAxisType lhs, PhysicalCommAxisType rhs) {
               return lhs.getIdStride() < rhs.getIdStride();
             });
  uint64_t count = 1;
  for (PhysicalCommAxisType axis : byIdStride) {
    if (axis.getExtent() == 1)
      continue;
    if (axis.getIdStride() != count)
      return false;
    count *= axis.getExtent();
  }
  return true;
}

uint64_t getDenseDeviceCount(ArrayRef<PhysicalCommAxisType> physicalAxes) {
  assert(numbersDevicesDensely(physicalAxes) &&
         "expected the physical axes to number devices densely");
  uint64_t count = 1;
  for (PhysicalCommAxisType axis : physicalAxes)
    count *= axis.getExtent();
  return count;
}

FailureOr<PhysicalMeshOp> findUniquePhysicalMesh(ModuleOp moduleOp) {
  if (!moduleOp) {
    return failure();
  }

  unsigned physicalMeshCount = 0;
  PhysicalMeshOp physicalMesh;
  for (PhysicalMeshOp meshOp : moduleOp.getOps<PhysicalMeshOp>()) {
    ++physicalMeshCount;
    if (physicalMeshCount == 1) {
      physicalMesh = meshOp;
    }
    if (physicalMeshCount > 1) {
      moduleOp.emitError()
          << "expected exactly one distributed physical mesh in module, found "
          << physicalMeshCount;
      return failure();
    }
  }

  if (physicalMeshCount == 0) {
    moduleOp.emitError()
        << "expected exactly one distributed physical mesh in module, found 0";
    return failure();
  }

  return physicalMesh;
}

::llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>>
filterOutReplicationFactors(
    TypedValueArrayRef<::mlir::enzyme::axis::AxisFactorType> factors) {
  llvm::SmallVector<TypedValue<::mlir::enzyme::axis::AxisFactorType>>
      filteredFactors;
  for (auto factor : factors) {
    // type of factor should wrap replication axis if it is a replication factor
    auto factorType =
        cast<::mlir::enzyme::axis::AxisFactorType>(factor.getType());
    ::mlir::Type axisType = factorType.getAxisType();
    if (!isa<::mlir::enzyme::distributed::ReplicationAxisType>(axisType)) {
      filteredFactors.push_back(factor);
    }
  }
  return filteredFactors;
}

OpShardingRuleAndReductionKind
getOrSynthesizeOpShardingRule(::mlir::Operation *op) {
  // Calls, collectives and local/global boundary ops (casts, anchors,
  // kernels) have no rule of their own: their axes are carried explicitly, or
  // come from the callee. They are never sharded or clustered as computation.
  if (!op ||
      isa<::mlir::CallOpInterface, PartitioningAnchorOpInterface,
          DistributedCollectiveOp, DistributedAwait>(op) ||
      getReshardingOp(op)) {
    return {};
  }

  // First check the Shardy rule registry. Shardy's rule only says which
  // factors are reductions; how partial results are combined comes from the
  // op itself. A reduce combines with its own body (max, min, ...), every
  // other op with a reduction factor (dot_general's contraction) sums.
  if (auto shardingRule = ::mlir::sdy::getOrCreateShardingRule(op)) {
    ::mlir::stablehlo::ReduceOpKind reductionKind =
        ::mlir::stablehlo::ReduceOpKind::Add;
    std::shared_ptr<::mlir::Region> reductionBody;
    if (auto reduceOp = dyn_cast<::mlir::stablehlo::ReduceOp>(op)) {
      reductionKind = ::mlir::stablehlo::CheckCommonReduceOp(reduceOp).kind;
      reductionBody = std::make_shared<::mlir::Region>();
      ::mlir::IRMapping regionMapper;
      reduceOp.getRegion().cloneInto(reductionBody.get(), regionMapper);
    }
    return {shardingRule, reductionKind, std::move(reductionBody)};
  }

  // Rule registry missing some cases for us, so we construct our own
  ::mlir::sdy::OpShardingRuleAttr synthesizedRule;
  ::mlir::stablehlo::ReduceOpKind reductionKind =
      ::mlir::stablehlo::ReduceOpKind::Add;
  std::shared_ptr<::mlir::Region> reductionBody;
  if (isa<::mlir::stablehlo::ConstantOp, ::mlir::sdy::ConstantOp>(op)) {
    // Constants are pointwise over their single result tensor shape.
    Value result = op->getResult(0);
    synthesizedRule = ::mlir::sdy::OpShardingRuleBuilder(op)
                          .addPointwise(::mlir::sdy::getTensorShape(result))
                          .build();
  } else if (auto iotaOp = dyn_cast<::mlir::stablehlo::IotaOp>(op)) {
    // TODO: once pattern-based lowering can shard an iota with device offsets,
    // make the iota dimension pass-through like the others.
    //
    // Shardy registers no rule for iota. Sharding along the iota dimension
    // isn't supported yet, so it is a need-replication factor, which
    // ShardyLogicalAxisAnalysis localizes as a device-local axis together with
    // every dimension it unifies with. Every other dimension is pointwise.
    int64_t iotaDimension = iotaOp.getIotaDimension();
    auto factorType = [&](int64_t dimension) {
      return dimension == iotaDimension
                 ? ::mlir::sdy::FactorType::kNeedReplication
                 : ::mlir::sdy::FactorType::kPassThrough;
    };
    synthesizedRule =
        ::mlir::sdy::OpShardingRuleBuilder(op)
            .addPointwise(::mlir::sdy::getTensorShape(iotaOp.getResult()),
                          factorType)
            .build();
  } else if (auto reduceOp = dyn_cast<::mlir::stablehlo::ReduceOp>(op)) {
    // Reductions can be split on any non-reduce dimension and require a
    // reduction factor on the dimensions being reduced.
    reductionKind = ::mlir::stablehlo::CheckCommonReduceOp(reduceOp).kind;
    reductionBody = std::make_shared<::mlir::Region>();
    ::mlir::IRMapping regionMapper;
    reduceOp.getRegion().cloneInto(reductionBody.get(), regionMapper);
    auto inputType =
        dyn_cast<RankedTensorType>(reduceOp.getOperand(0).getType());
    auto resultType =
        dyn_cast<RankedTensorType>(reduceOp.getResult(0).getType());
    if (!inputType || !resultType) {
      return {};
    }

    auto builder = ::mlir::sdy::OpShardingRuleBuilder(op);
    int64_t resultDimIdx = 0;
    for (int64_t inputDimIdx = 0; inputDimIdx < inputType.getRank();
         ++inputDimIdx) {
      if (inputType.isDynamicDim(inputDimIdx)) {
        return {};
      }

      bool isReductionDim =
          llvm::is_contained(reduceOp.getDimensions(), inputDimIdx);
      int64_t resultDim = ::mlir::sdy::kNullDim;
      if (!isReductionDim) {
        if (resultDimIdx >= resultType.getRank() ||
            resultType.isDynamicDim(resultDimIdx)) {
          return {};
        }
        resultDim = resultDimIdx++;
      }

      builder.addFactor({inputDimIdx, ::mlir::sdy::kNullDim}, {resultDim},
                        inputType.getDimSize(inputDimIdx),
                        isReductionDim ? ::mlir::sdy::FactorType::kReduction
                                       : ::mlir::sdy::FactorType::kPassThrough);
    }

    if (resultDimIdx != resultType.getRank()) {
      return {};
    }

    synthesizedRule = builder.build();
  }
  if (!synthesizedRule) {
    return {};
  }
  return {synthesizedRule, reductionKind, std::move(reductionBody)};
}

::mlir::Region *OpShardingRuleAndReductionKind::getReductionBody(
    ::mlir::Type elementType) const {
  if (reductionBody) {
    return reductionBody.get();
  }
  return buildSyntheticReductionBody(reductionKind, elementType, reductionBody);
}

bool isReshardPoint(::mlir::Operation *op) {
  auto customCall = dyn_cast<::mlir::stablehlo::CustomCallOp>(op);
  return customCall && customCall.getCallTargetName() == kReshardPointTarget;
}

::mlir::FailureOr<::llvm::SmallVector<bool>>
getReshardPointBrokenDims(::mlir::Operation *op) {
  assert(isReshardPoint(op) && "expected a reshard point");
  auto fail = [&](const ::llvm::Twine &message) {
    op->emitError() << kReshardPointTarget << ": " << message;
    return ::mlir::failure();
  };
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return fail("expected exactly one operand and one result");
  }
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape() || op->getOperand(0).getType() != type) {
    return fail("expected a statically shaped tensor operand whose type "
                "equals the result's");
  }
  auto rule = op->getAttrOfType<::mlir::sdy::OpShardingRuleAttr>(
      ::mlir::sdy::kShardingRuleAttr);
  if (!rule) {
    return fail("expected an sdy.sharding_rule naming the broken dimensions");
  }
  // Each dimension maps to one factor per side. A pass-through factor sits on
  // the same dimension of both sides and a broken dimension's two factors
  // appear nowhere else, so Shardy couples exactly the pass-through
  // dimensions across the op.
  auto operandDims = rule.getOperandMapping(0).getDimMappings();
  auto resultDims = rule.getResultMapping(0).getDimMappings();
  ::llvm::SmallVector<int64_t> operandFactors, resultFactors;
  ::llvm::DenseMap<int64_t, int64_t> factorUses;
  for (auto [operandDim, resultDim] :
       llvm::zip_equal(operandDims, resultDims)) {
    if (operandDim.getFactorIndices().size() != 1 ||
        resultDim.getFactorIndices().size() != 1) {
      return fail("expected each dimension to map to exactly one factor");
    }
    operandFactors.push_back(operandDim.getFactorIndices().front());
    resultFactors.push_back(resultDim.getFactorIndices().front());
    ++factorUses[operandFactors.back()];
    ++factorUses[resultFactors.back()];
  }
  ::llvm::SmallVector<bool> broken;
  for (auto [operandFactor, resultFactor] :
       llvm::zip_equal(operandFactors, resultFactors)) {
    bool isBroken = operandFactor != resultFactor;
    bool wellPlaced = isBroken ? factorUses[operandFactor] == 1 &&
                                     factorUses[resultFactor] == 1
                               : factorUses[operandFactor] == 2;
    if (!wellPlaced) {
      return fail("expected each factor on one dimension, either shared by "
                  "both sides or private to one side");
    }
    broken.push_back(isBroken);
  }
  if (!rule.getReductionFactors().empty() ||
      !rule.getNeedReplicationFactors().empty() ||
      !rule.getPermutationFactors().empty()) {
    return fail("expected only pass-through factors");
  }
  return broken;
}

std::optional<ReshardingOp> getReshardingOp(::mlir::Operation *op) {
  if (auto reshard = dyn_cast<::mlir::sdy::ReshardOp>(op)) {
    return ReshardingOp{op, reshard.getInput(), reshard.getResult()};
  }
  if (isReshardPoint(op)) {
    return ReshardingOp{op, op->getOperand(0), op->getResult(0)};
  }
  return std::nullopt;
}

CollectiveAndAwait
createCollectiveAndAwait(::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::Value inputObject, ::mlir::Value inputMesh,
                         ::mlir::Value outputMesh,
                         ::mlir::ValueRange reductionGroups,
                         ::mlir::Value mapping, ::mlir::Type outputType) {
  OperationState state(loc, DistributedCollectiveOp::getOperationName());
  state.addOperands(inputObject);
  state.addOperands(inputMesh);
  state.addOperands(outputMesh);
  state.addOperands(reductionGroups);
  state.addOperands(mapping);
  state.addTypes(::mlir::enzyme::distributed::AsynchHandleType::get(
      builder.getContext(), outputType));
  state.addAttribute("output_type", ::mlir::TypeAttr::get(outputType));
  for (size_t i = 0; i < reductionGroups.size(); ++i) {
    state.addRegion();
  }
  auto collective = cast<DistributedCollectiveOp>(builder.create(state));
  auto await = builder.create<DistributedAwait>(loc, outputType,
                                                collective.getAsyncHandle());
  return {collective, await};
}

bool isTriviallyLocalKernel(DistributedKernelOp kernelOp) {
  Block &body = kernelOp.getBody().front();
  for (auto [operand, blockArg] :
       llvm::zip(kernelOp.getArguments(), body.getArguments())) {
    if (operand.getType() != blockArg.getType()) {
      return false;
    }
  }
  auto yieldOp = cast<DistributedYieldOp>(body.getTerminator());
  for (auto [result, yieldOperand] :
       llvm::zip(kernelOp.getResults(), yieldOp.getReturns())) {
    if (result.getType() != yieldOperand.getType()) {
      return false;
    }
  }
  return true;
}

IndexedTensorShardingAttr buildEmptyShardingForType(::mlir::MLIRContext *ctx,
                                                    ::mlir::Type type) {
  auto emptyAxes = DenseI64ArrayAttr::get(ctx, ArrayRef<int64_t>{});
  SmallVector<DenseI64ArrayAttr> dimPartitioningAxes;
  if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
    dimPartitioningAxes.append(rankedType.getRank(), emptyAxes);
  }
  return IndexedTensorShardingAttr::get(ctx, dimPartitioningAxes, emptyAxes);
}

FailureOr<ModuleOp> buildKernelBodyModule(DistributedKernelOp kernelOp) {
  OpBuilder builder(kernelOp.getContext());
  auto module = ModuleOp::create(kernelOp.getLoc());

  Block &kernelBody = kernelOp.getBody().front();
  auto yield = cast<DistributedYieldOp>(kernelBody.getTerminator());

  auto fnType = builder.getFunctionType(kernelBody.getArgumentTypes(),
                                        yield.getOperandTypes());
  auto func =
      func::FuncOp::create(builder, kernelOp.getLoc(), "kernel", fnType);
  module.push_back(func);

  Block *entry = func.addEntryBlock();
  IRMapping mapping;
  mapping.map(kernelBody.getArguments(), entry->getArguments());
  builder.setInsertionPointToStart(entry);

  // A kernel body need not be isolated from above: e.g. CSE commons up a
  // constant used by several sibling kernels and hoists the single copy
  // just outside all of them. The standalone module built here must be
  // self-contained regardless, so any operand not already mapped (i.e. not
  // one of the body's own block args or an already-cloned op's result) is
  // materialized by cloning its defining op first, recursively -- always
  // safe for the kind of side-effect-free op CSE would have hoisted this
  // way. Plain recursion over defining ops (rather than a region-ancestor
  // utility like makeRegionIsolatedFromAbove) is what makes this work at
  // all here: by the time a capture is discovered, the func body has
  // already been detached into a brand new, disconnected module, so any
  // check relying on the two actually sharing a region tree would vacuously
  // find nothing to capture.
  bool ok = true;
  std::function<void(Value)> materialize =
      [&](Value value) {
        if (!ok || mapping.contains(value)) {
          return;
        }
        Operation *definingOp = value.getDefiningOp();
        if (!definingOp) {
          kernelOp.emitError()
              << "kernel body captures a value with no defining op (an outer "
                 "block argument), which cannot be inlined into a standalone "
                 "module";
          ok = false;
          return;
        }
        for (Value operand : definingOp->getOperands()) {
          materialize(operand);
        }
        if (ok) {
          builder.clone(*definingOp, mapping);
        }
      };

  for (Operation &op : kernelBody.without_terminator()) {
    for (Value operand : op.getOperands()) {
      materialize(operand);
    }
    if (!ok) {
      break;
    }
    builder.clone(op, mapping);
  }

  SmallVector<Value> results;
  if (ok) {
    results.reserve(yield.getReturns().size());
    for (Value v : yield.getReturns()) {
      materialize(v);
      if (ok) {
        results.push_back(mapping.lookup(v));
      }
    }
  }
  if (!ok) {
    module.erase();
    return failure();
  }

  builder.create<func::ReturnOp>(kernelOp.getLoc(), results);
  return module;
}

LogicalResult spliceKernelBodyModule(DistributedKernelOp kernelOp,
                                     ModuleOp patchedModule) {
  SymbolTable patchedSymbols(patchedModule);
  auto patchedKernelFn = patchedSymbols.lookup<func::FuncOp>("kernel");
  if (!patchedKernelFn || patchedKernelFn.isDeclaration()) {
    return kernelOp.emitError() << "patched kernel module has no defined "
                                   "'kernel' function to splice in";
  }

  Block &kernelBody = kernelOp.getBody().front();
  auto oldYield = cast<DistributedYieldOp>(kernelBody.getTerminator());
  Block &patchedBody = patchedKernelFn.getBody().front();
  auto patchedReturn = cast<func::ReturnOp>(patchedBody.getTerminator());

  auto kernelArgTypes = llvm::to_vector(kernelBody.getArgumentTypes());
  auto patchedArgTypes = llvm::to_vector(patchedBody.getArgumentTypes());
  if (kernelArgTypes != patchedArgTypes) {
    return kernelOp.emitError()
           << "patched 'kernel' function's argument types do not match "
              "this kernel's body: expected ("
           << kernelArgTypes << "), got (" << patchedArgTypes << ")";
  }
  auto yieldTypes = llvm::to_vector(oldYield.getOperandTypes());
  auto patchedReturnTypes = llvm::to_vector(patchedReturn.getOperandTypes());
  if (yieldTypes != patchedReturnTypes) {
    return kernelOp.emitError()
           << "patched 'kernel' function's result types do not match "
              "this kernel's body: expected ("
           << yieldTypes << "), got (" << patchedReturnTypes << ")";
  }

  // Block arguments are the kernel body's own -- they stay put -- while the
  // patched function's block arguments are only a mapping key to reach them.
  IRMapping mapping;
  mapping.map(patchedBody.getArguments(), kernelBody.getArguments());

  // A value can only be used by ops later in the same straight-line block,
  // so erasing back-to-front never leaves a dangling use behind.
  while (!kernelBody.empty()) {
    kernelBody.back().erase();
  }

  OpBuilder builder(kernelOp.getContext());
  builder.setInsertionPointToStart(&kernelBody);
  for (Operation &op : patchedBody.without_terminator()) {
    builder.clone(op, mapping);
  }
  SmallVector<Value> yieldOperands = llvm::map_to_vector(
      patchedReturn.getOperands(), [&](Value v) { return mapping.lookup(v); });
  builder.create<DistributedYieldOp>(kernelOp.getLoc(), yieldOperands);

  auto siblings = llvm::make_filter_range(
      patchedModule.getOps<func::FuncOp>(),
      [&](func::FuncOp fn) { return fn != patchedKernelFn; });
  if (siblings.empty())
    return success();

  // DistributedKernelOp is only ever created inside a module in this
  // pipeline (buildDistributedSearchLoweringPipeline runs on a ModuleOp),
  // so this is an invariant to assert, not a case to fail on.
  auto containingModule = kernelOp->getParentOfType<ModuleOp>();
  assert(containingModule &&
         "DistributedKernelOp expected to live inside a module");
  SymbolTable containingSymbols(containingModule);

  // Siblings are inserted first and references rewritten afterwards in one
  // simultaneous pass: an inserted sibling's new name (@x_1) may equal another
  // sibling's original name, so rewriting per sibling would redirect uses
  // twice.
  llvm::DenseMap<StringAttr, StringAttr> newNames;
  SmallVector<Operation *> clones;
  for (func::FuncOp patchedFn : siblings) {
    Operation *clone = patchedFn->clone();
    newNames[patchedFn.getSymNameAttr()] = containingSymbols.insert(clone);
    clones.push_back(clone);
  }

  // Only a reference's root names a sibling. Skipping a reference's sub
  // elements keeps a nested reference (@other::@x) from being rewritten.
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](SymbolRefAttr ref)
          -> std::optional<std::pair<Attribute, WalkResult>> {
        Attribute replacement = ref;
        auto it = newNames.find(ref.getRootReference());
        if (it != newNames.end())
          replacement =
              SymbolRefAttr::get(it->second, ref.getNestedReferences());
        return std::make_pair(replacement, WalkResult::skip());
      });
  for (Operation &op : kernelBody)
    replacer.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/false);
  for (Operation *clone : clones)
    replacer.recursivelyReplaceElementsIn(clone, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/false);
  return success();
}

llvm::Expected<llvm::SmallVector<PhysicalFactorRef>>
projectDeviceWindow(int64_t extent, int64_t deviceStride,
                    llvm::ArrayRef<PhysicalAxisSpec> physicalAxes) {
  llvm::SmallVector<PhysicalFactorRef> minorFirst;
  int64_t remaining = extent;
  int64_t position = deviceStride;
  while (remaining > 1) {
    // The physical axis whose digit range contains `position`.
    int64_t found = -1;
    for (auto [i, axis] : llvm::enumerate(physicalAxes)) {
      if (axis.extent > 1 && position % axis.idStride == 0 &&
          position / axis.idStride < axis.extent) {
        found = i;
        break;
      }
    }
    if (found < 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "device-id stride %lld does not fall on a physical axis digit",
          (long long)position);
    const PhysicalAxisSpec &axis = physicalAxes[found];
    int64_t digit = position / axis.idStride;
    int64_t available = axis.extent / digit;
    if (axis.extent % digit != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "device-id stride %lld splits physical axis #%lld "
          "(extent %lld) non-divisibly",
          (long long)position, (long long)found, (long long)axis.extent);
    int64_t take = remaining <= available ? remaining : available;
    if (available % take != 0 || remaining % take != 0)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "extent %lld straddles the boundary of physical axis #%lld "
          "(extent %lld, %lld digits available) without divisibility",
          (long long)remaining, (long long)found, (long long)axis.extent,
          (long long)available);
    minorFirst.push_back({found, take, digit});
    remaining /= take;
    position = axis.idStride * axis.extent;
  }
  return llvm::SmallVector<PhysicalFactorRef>(llvm::reverse(minorFirst));
}

llvm::Expected<llvm::SmallVector<PhysicalFactorRef>>
projectShardySubAxis(int64_t fullAxisSize, int64_t fullAxisDeviceStride,
                     int64_t preSize, int64_t size,
                     llvm::ArrayRef<PhysicalAxisSpec> physicalAxes) {
  // Shardy guarantees preSize * size divides the full axis size.
  assert(fullAxisSize % (preSize * size) == 0);
  return projectDeviceWindow(
      size, fullAxisDeviceStride * (fullAxisSize / (preSize * size)),
      physicalAxes);
}

llvm::Expected<llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>>>
projectShardyMesh(llvm::ArrayRef<::mlir::sdy::MeshAxisAttr> meshAxes,
                  llvm::ArrayRef<PhysicalAxisSpec> physicalAxes) {
  int64_t meshDevices = 1, physicalDevices = 1;
  for (auto axis : meshAxes)
    meshDevices *= axis.getSize();
  for (auto axis : physicalAxes)
    physicalDevices *= axis.extent;
  if (meshDevices != physicalDevices)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mesh has %lld devices but the physical mesh has %lld",
        (long long)meshDevices, (long long)physicalDevices);

  llvm::SmallVector<llvm::SmallVector<PhysicalFactorRef>> result(
      meshAxes.size());
  int64_t stride = 1;
  for (size_t i = meshAxes.size(); i-- > 0;) {
    auto factors =
        projectDeviceWindow(meshAxes[i].getSize(), stride, physicalAxes);
    if (!factors)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(), "axis \"%s\": %s",
          meshAxes[i].getName().str().c_str(),
          llvm::toString(factors.takeError()).c_str());
    result[i] = std::move(*factors);
    stride *= meshAxes[i].getSize();
  }
  return result;
}

} // namespace mlir::enzyme::distributed
