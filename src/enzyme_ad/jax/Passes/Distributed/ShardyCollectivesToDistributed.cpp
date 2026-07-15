#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace enzyme {
namespace distributed {

static constexpr llvm::StringLiteral kShardyAxisNamesAttr =
		"enzyme.shardy_axis_names";

#define GEN_PASS_DEF_SHARDYCOLLECTIVESTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

/**
 * Some collectives (the reduces) have computational behavior that isn't
 * fully captured just by the sharding annotations, but most of the other
 * ops are capturable just by the sharding annotations and can be rewritten
 * as a group. This function identifies the latter.
 */
bool sharding_encapsulates_behavior(Operation *op) {
	bool captured = isa<sdy::AllGatherOp>(op) || isa<sdy::AllSliceOp>(op) ||
									isa<sdy::AllToAllOp>(op) || isa<sdy::CollectivePermuteOp>(op);
	return captured;
}

// TODO: hacky horrible quick prototype! Should use shardy op interfaces /
// getter methods or should record sharding for values before rewriting.
FailureOr<sdy::TensorShardingAttr> getTensorShardingFromValue(Value value) {
	if (auto blockArg = dyn_cast<BlockArgument>(value)) {
		Operation *parentOp = blockArg.getOwner()->getParentOp();
		if (auto meshComputation = dyn_cast<MeshComputationOp>(parentOp)) {
			unsigned argIndex = blockArg.getArgNumber();
			auto inputTensors = meshComputation.getInputTensors();
			if (argIndex < inputTensors.size()) {
				return sdy::getSharding(inputTensors[argIndex]);
			}
		}
		return sdy::getSharding(value);
	}

	if (isa<OpResult>(value)) {
		// can now call into the shardy api since we have skipped the
		// mesh region
		return sdy::getSharding(value);
	}

	return failure();
}

FailureOr<llvm::StringMap<Value>>
buildShardyAxisToDistributedAxisMap(MeshComputationOp meshComputation,
									SmallVectorImpl<Value> &collectiveAxes) {
	auto shardyAxisNames =
			meshComputation->getAttrOfType<ArrayAttr>(kShardyAxisNamesAttr);
	if (!shardyAxisNames) {
		return failure();
	}

	SmallVector<Value> distributedAxes;
	distributedAxes.reserve(meshComputation.getSpmdAxes().size() +
											meshComputation.getMpmdAxes().size());
	for (Value axis : meshComputation.getSpmdAxes()) {
		distributedAxes.push_back(axis);
	}
	for (Value axis : meshComputation.getMpmdAxes()) {
		distributedAxes.push_back(axis);
	}
	collectiveAxes.assign(distributedAxes.begin(), distributedAxes.end());
	if (shardyAxisNames.size() != distributedAxes.size()) {
		return failure();
	}

	llvm::StringMap<Value> axisMap;
	for (auto [index, shardyAxisNameAttr] : llvm::enumerate(shardyAxisNames)) {
		auto shardyAxisName = dyn_cast<StringAttr>(shardyAxisNameAttr);
		if (!shardyAxisName) {
			return failure();
		}
		axisMap[shardyAxisName.getValue()] = distributedAxes[index];
	}

	return axisMap;
}

FailureOr<SmallVector<Value>>
mapShardyAxisRefsToDistributedAxes(ArrayRef<sdy::AxisRefAttr> shardyAxes,
																	 const llvm::StringMap<Value> &axisMap) {
	SmallVector<Value> mappedAxes;
	mappedAxes.reserve(shardyAxes.size());
	for (sdy::AxisRefAttr shardyAxis : shardyAxes) {
		if (shardyAxis.getSubAxisInfo()) {
			return failure();
		}

		auto axisIt = axisMap.find(shardyAxis.getName());
		if (axisIt == axisMap.end()) {
			return failure();
		}
		mappedAxes.push_back(axisIt->second);
	}
	return mappedAxes;
}

FailureOr<SmallVector<Value>>
buildDistributedShardingForTensor(sdy::TensorShardingAttr sharding,
																	const llvm::StringMap<Value> &axisMap,
																	PatternRewriter &rewriter, Location loc) {
	if (!sharding.getUnreducedAxes().empty()) {
		return failure();
	}

	SmallVector<Value> distributedShardings;
	distributedShardings.reserve(sharding.getDimShardings().size());
	for (sdy::DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
		FailureOr<SmallVector<Value>> logicalAxes =
				mapShardyAxisRefsToDistributedAxes(dimSharding.getAxes(), axisMap);
		if (failed(logicalAxes)) {
			return failure();
		}

		auto shardingOp = rewriter.create<ShardingOp>(loc, *logicalAxes);
		distributedShardings.push_back(shardingOp.getSharding());
	}

	return distributedShardings;
}

struct LowerSdyCollectiveToDistributedPattern : public RewritePattern {
	explicit LowerSdyCollectiveToDistributedPattern(MLIRContext *context)
			: RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, context) {}

	LogicalResult matchAndRewrite(Operation *op,
																PatternRewriter &rewriter) const override {
		// Find a shardy collective op in scope, child to a MeshComputationOp.
		auto sdyCollective = dyn_cast<sdy::CollectiveOpInterface>(op);
		if (!sdyCollective) {
			return failure();
		}

		auto meshComputation = op->getParentOfType<MeshComputationOp>();
		if (!meshComputation) {
			return failure();
		}

		if (op->getNumResults() != 1) {
			return rewriter.notifyMatchFailure(
					op, "expected single-result Shardy collective op");
		}

		if (!sharding_encapsulates_behavior(op)) {
			// TODO add cases for reductions
			return rewriter.notifyMatchFailure(
					op, "Shardy collective op behavior not fully captured by sharding");
		}

		Value tensorIn = sdyCollective.getTensor();
		Type tensorOutType = op->getResult(0).getType();

		FailureOr<sdy::TensorShardingAttr> inputSharding =
				getTensorShardingFromValue(tensorIn);
		if (failed(inputSharding)) {
			return rewriter.notifyMatchFailure(
					op, "expected input tensor to have explicit Shardy sharding");
		}

		sdy::TensorShardingAttr outputSharding = sdyCollective.getOutSharding();

		SmallVector<Value> collectiveAxes;
		FailureOr<llvm::StringMap<Value>> axisMap =
			buildShardyAxisToDistributedAxisMap(meshComputation, collectiveAxes);
		if (failed(axisMap)) {
			return rewriter.notifyMatchFailure(
					op, "failed to build Shardy-to-distributed axis mapping");
		}

		OpBuilder::InsertionGuard guard(rewriter);
		rewriter.setInsertionPoint(meshComputation);

		FailureOr<SmallVector<Value>> inputShardingAxes =
				buildDistributedShardingForTensor(*inputSharding, *axisMap, rewriter,
																		op->getLoc());
		if (failed(inputShardingAxes)) {
			return rewriter.notifyMatchFailure(
					op, "failed to lower input Shardy sharding to distributed sharding "
							"(common non-reduction case)");
		}

		FailureOr<SmallVector<Value>> outputShardingAxes =
				buildDistributedShardingForTensor(outputSharding, *axisMap, rewriter,
																					op->getLoc());
		if (failed(outputShardingAxes)) {
			return rewriter.notifyMatchFailure(
					op, "failed to lower output Shardy sharding to distributed sharding "
							"(common non-reduction case)");
		}

		auto globalInputTensorTypeAttr = TypeAttr::get(tensorIn.getType());
		auto globalOutputTensorTypeAttr = TypeAttr::get(tensorOutType);

		// TODO: set local tensor type attrs if/when they differ from global types.
		auto localInputTensorTypeAttr = globalInputTensorTypeAttr;
		auto localOutputTensorTypeAttr = globalOutputTensorTypeAttr;

		auto collective = rewriter.create<CollectiveOp>(
				op->getLoc(), MessageTokenType::get(getContext()),
			collectiveAxes, globalInputTensorTypeAttr,
				globalOutputTensorTypeAttr, localInputTensorTypeAttr,
				localOutputTensorTypeAttr, *inputShardingAxes, *outputShardingAxes);

		rewriter.setInsertionPoint(op);
		rewriter.create<SendOp>(op->getLoc(), collective.getToken(), tensorIn);
		auto recv =
			rewriter.create<RecvOp>(op->getLoc(), tensorOutType, collective.getToken());
		sdy::setSharding(recv.getMessage(), outputSharding);

		if (meshComputation.getNumCommunicationBodies() == 0) {
			return rewriter.notifyMatchFailure(
				op, "expected at least one communication region for transfer op");
		}

		Value transferAxis;
		if (!collectiveAxes.empty()) {
			transferAxis = collectiveAxes.front();
		}
		if (!outputShardingAxes->empty()) {
			if (auto shardingOp =
						outputShardingAxes->front().getDefiningOp<ShardingOp>()) {
				if (!shardingOp.getAxes().empty()) {
					transferAxis = shardingOp.getAxes().front();
				}
			}
		}

		unsigned communicationBodyIndex = 0;
		if (transferAxis) {
			auto communicationBodyIndexOr =
					meshComputation.findCommunicationBodyIndexForAxis(transferAxis);
			if (succeeded(communicationBodyIndexOr)) {
				communicationBodyIndex = *communicationBodyIndexOr;
			}
		}

		Region &communicationBody =
				meshComputation.getCommunicationBody(communicationBodyIndex);
		Block &communicationBlock = communicationBody.front();
		auto transferInsertIt = communicationBlock.end();
		if (!communicationBlock.empty()) {
			transferInsertIt = Block::iterator(communicationBlock.getTerminator());
		}
		rewriter.setInsertionPoint(&communicationBlock, transferInsertIt);
		rewriter.create<TransferOp>(op->getLoc(), collective.getToken());

		rewriter.replaceOp(op, recv.getMessage());
		return success();
	}
};

struct ShardyCollectivesToDistributedPass
		: public enzyme::distributed::impl::ShardyCollectivesToDistributedPassBase<
					ShardyCollectivesToDistributedPass> {
	using ShardyCollectivesToDistributedPassBase::
			ShardyCollectivesToDistributedPassBase;

	void runOnOperation() override {
		if (failed(rewriteShardyCollectivesInFunction(getOperation()))) {
			signalPassFailure();
		}
	}
};

} // namespace

llvm::LogicalResult rewriteShardyCollectivesInFunction(func::FuncOp funcOp) {
	RewritePatternSet patterns(funcOp.getContext());
	patterns.add<LowerSdyCollectiveToDistributedPattern>(funcOp.getContext());
	return applyPatternsGreedily(funcOp, std::move(patterns));
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
