#include "Utilities.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  // Calls and local/global boundary ops (casts, anchors, kernels) have no
  // rule of their own: their axes are carried explicitly, or come from the
  // callee.
  if (!op || isa<::mlir::CallOpInterface, PartitioningAnchorOpInterface>(op)) {
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

} // namespace mlir::enzyme::distributed