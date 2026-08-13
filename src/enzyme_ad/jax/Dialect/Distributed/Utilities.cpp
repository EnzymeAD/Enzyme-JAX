#include "Utilities.h"

#include "src/enzyme_ad/jax/Utils.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme::distributed {

using namespace ::mlir::enzyme::axis;

namespace {

::mlir::Region *buildSyntheticReductionBody(
    ::mlir::stablehlo::ReduceOpKind reductionKind, ::mlir::Type elementType,
    std::shared_ptr<::mlir::Region> &storage) {
  if (!elementType || reductionKind == ::mlir::stablehlo::ReduceOpKind::Unknown) {
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
  if (!op) {
    return {};
  }

  // First check the Shardy rule registry
  if (auto shardingRule = ::mlir::sdy::getOrCreateShardingRule(op)) {
    return {shardingRule, ::mlir::stablehlo::ReduceOpKind::Add};
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
    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getOperand(0).getType());
    auto resultType = dyn_cast<RankedTensorType>(reduceOp.getResult(0).getType());
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

      bool isReductionDim = llvm::is_contained(reduceOp.getDimensions(),
                                               inputDimIdx);
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
                        isReductionDim
                            ? ::mlir::sdy::FactorType::kReduction
                            : ::mlir::sdy::FactorType::kPassThrough);
    }

    if (resultDimIdx != resultType.getRank()) {
      return {};
    }

    synthesizedRule = builder.build();
  } else if (isa<::mlir::func::ReturnOp,
                 ::mlir::enzyme::distributed::DistributedYieldOp>(op)) {
    // Return-like terminators should not couple partition axes across
    // operands. Give each tensor axis of each operand its own independent
    // pass-through factor.
    auto builder = ::mlir::sdy::OpShardingRuleBuilder(op);
    int64_t numOperands = op->getNumOperands();
    for (int64_t operandIdx = 0; operandIdx < numOperands; ++operandIdx) {
      auto tensorType =
          dyn_cast<RankedTensorType>(op->getOperand(operandIdx).getType());
      if (!tensorType) {
        continue;
      }
      for (int64_t dimIdx = 0; dimIdx < tensorType.getRank(); ++dimIdx) {
        if (tensorType.isDynamicDim(dimIdx)) {
          return {};
        }

        SmallVector<int64_t> lhsDims(numOperands, ::mlir::sdy::kNullDim);
        lhsDims[operandIdx] = dimIdx;
        builder.addFactor(lhsDims, {}, tensorType.getDimSize(dimIdx),
                          ::mlir::sdy::FactorType::kPassThrough);
      }
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

CollectiveAndAwait createCollectiveAndAwait(
    ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value inputObject,
    ::mlir::Value inputMesh, ::mlir::Value outputMesh,
    ::mlir::ValueRange reductionGroups, ::mlir::Value mapping,
    ::mlir::Type outputType) {
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

} // namespace mlir::enzyme::distributed