#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/FindShardyFunctionsAnalysis.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SHARDYTODISTRIBUTEDPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct FuncToDistributedFunctionPattern
    : public OpRewritePattern<func::FuncOp> {
  FuncToDistributedFunctionPattern(
      MLIRContext *context, TypedValue<axis::FactorGroupType> contextGroup)
      : OpRewritePattern<func::FuncOp>(context), contextGroup(contextGroup) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (!funcOp->getParentOfType<distributed::MeshComputationOp>()) {
      return failure();
    }

    SmallVector<Attribute> argumentTypes;
    argumentTypes.reserve(funcOp.getFunctionType().getNumInputs());
    for (Type inputType : funcOp.getFunctionType().getInputs()) {
      argumentTypes.push_back(TypeAttr::get(inputType));
    }

    SmallVector<Attribute> returnTypes;
    returnTypes.reserve(funcOp.getFunctionType().getNumResults());
    for (Type resultType : funcOp.getFunctionType().getResults()) {
      returnTypes.push_back(TypeAttr::get(resultType));
    }

    auto distributedFunc = rewriter.create<distributed::DistributedFunctionOp>(
        funcOp.getLoc(), funcOp.getSymNameAttr(), contextGroup,
        rewriter.getArrayAttr(argumentTypes),
        rewriter.getArrayAttr(returnTypes));

    for (NamedAttribute attr : funcOp->getAttrs()) {
      if (attr.getName() == rewriter.getStringAttr("sym_name") ||
          attr.getName() == rewriter.getStringAttr("function_type")) {
        continue;
      }
      distributedFunc->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.inlineRegionBefore(funcOp.getBody(), distributedFunc.getBody(),
                                distributedFunc.getBody().end());
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  TypedValue<axis::FactorGroupType> contextGroup;
};

struct FuncCallToDistributedCallPattern
    : public OpRewritePattern<func::CallOp> {
  FuncCallToDistributedCallPattern(
      MLIRContext *context,
      const llvm::DenseSet<StringAttr> &distributedFunctionNames,
      TypedValue<axis::FactorGroupType> replicateOver)
      : OpRewritePattern<func::CallOp>(context),
        distributedFunctionNames(distributedFunctionNames),
        replicateOver(replicateOver) {}

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (!callOp->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    StringRef callee = callOp.getCallee();
    if (!distributedFunctionNames.contains(rewriter.getStringAttr(callee))) {
      return failure();
    }

    auto distributedCall = rewriter.create<distributed::DistributedCallOp>(
        callOp.getLoc(), callOp.getResultTypes(), callOp.getCalleeAttr(),
        replicateOver, callOp.getOperands());
    rewriter.replaceOp(callOp, distributedCall->getResults());
    return success();
  }

private:
  llvm::DenseSet<StringAttr> distributedFunctionNames;
  TypedValue<axis::FactorGroupType> replicateOver;
};

struct FuncReturnToDistributedYieldPattern
    : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    if (!returnOp->getParentOfType<distributed::DistributedFunctionOp>()) {
      return failure();
    }

    rewriter.create<distributed::DistributedYieldOp>(returnOp.getLoc(),
                                                     returnOp.getOperands());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

struct ShardyToDistributedPass
    : public impl::ShardyToDistributedPassBase<ShardyToDistributedPass> {
  using ShardyToDistributedPassBase::ShardyToDistributedPassBase;

  // For now we simplify our life by requiring all shardy functions to have the
  // same mesh. This will be relaxed at need as implementation progresses.
  sdy::MeshAttr getCommonMesh(const FindShardyFunctionsAnalysis &analysis) {
    auto shardyFunctions = analysis.getShardyFunctions();
    sdy::MeshAttr commonMesh = nullptr;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         shardyFunctions) {
      if (info.meshes.size() != 1) {
        getOperation()->emitError()
            << "expected shardy function to have exactly one mesh, found "
            << info.meshes.size();
        signalPassFailure();
        return nullptr;
      }
      if (!commonMesh) {
        commonMesh = info.meshes[0];
      } else if (commonMesh != info.meshes[0]) {
        getOperation()->emitError()
            << "expected all shardy functions to have the same mesh, found "
            << commonMesh << " and " << info.meshes[0];
        signalPassFailure();
        return nullptr;
      }
    }
    return commonMesh;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    FailureOr<distributed::PhysicalMeshOp> physicalMesh =
        distributed::findUniquePhysicalMesh(moduleOp);
    if (failed(physicalMesh)) {
      signalPassFailure();
      return;
    }

    const FindShardyFunctionsAnalysis &analysis =
        getAnalysis<FindShardyFunctionsAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }
    if (analysis.getShardyFunctions().empty()) {
      moduleOp.emitRemark() << "no shardy functions found, skipping pass";
      return;
    }

    sdy::MeshAttr commonMesh = getCommonMesh(analysis);
    if (!commonMesh) {
      return;
    }

    // Create a new mesh computation using the modules pysical mesh
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointAfter(
        *physicalMesh); // graph region, doesn't matter
    distributed::MeshComputationOp meshComputation =
        builder.create<distributed::MeshComputationOp>(
            moduleOp.getLoc(), builder.getStringAttr("mesh_computation"),
            FlatSymbolRefAttr::get(physicalMesh->getSymNameAttr()));
    Region &meshComputationBody = meshComputation.getBody();
    if (meshComputationBody.empty()) {
      meshComputationBody.emplaceBlock();
    }
    builder.setInsertionPointToStart(&meshComputationBody.front());

    // Create a new logical mesh inside of the mesh computation using the common
    // shardy mesh for axis sizes. Internally record a mapping from shardy axis
    // names to distributed axis values.
    llvm::SmallVector<int32_t> logicalAxisExtents;
    llvm::SmallVector<StringAttr> shardyAxisNames;
    for (sdy::MeshAxisAttr axis : commonMesh.getAxes()) {
      int64_t axisSize = axis.getSize();
      if (axisSize <= 0 || axisSize > std::numeric_limits<int32_t>::max()) {
        moduleOp.emitError() << "unsupported shardy mesh axis size " << axisSize
                             << " for axis " << axis.getName();
        signalPassFailure();
        return;
      }
      logicalAxisExtents.push_back(static_cast<int32_t>(axisSize));
      shardyAxisNames.push_back(builder.getStringAttr(axis.getName()));
    }

    distributed::LogicalMeshAxesOp logicalMeshAxes =
        builder.create<distributed::LogicalMeshAxesOp>(
            moduleOp.getLoc(),
            builder.getDenseI32ArrayAttr(logicalAxisExtents));

    llvm::DenseMap<StringAttr, Value> shardyToDistributedAxis;
    for (auto [idx, distributedAxis] :
         llvm::enumerate(logicalMeshAxes.getAxes())) {
      shardyToDistributedAxis[shardyAxisNames[idx]] = distributedAxis;
    }

    // Also map the canonical logical mesh axis to factor
    // types. Shardy works on full axes so we produce only one factor
    // per axis, which we can subdivide in transformations later if needed.
    // Mapping here determined by arg number.
    auto factors = axis::viewAxesAsFactors(logicalMeshAxes.getAxes(), builder,
                                           moduleOp.getLoc());

    auto executionContext =
        builder.create<axis::AxisProductOp>(moduleOp.getLoc(), factors)
            .getProduct();
    auto replicateOver =
        builder.create<axis::AxisProductOp>(moduleOp.getLoc(), ValueRange())
            .getProduct();

    llvm::DenseSet<StringAttr> distributedFunctionNames;
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      distributedFunctionNames.insert(info.symName);
    }

    // Now we copy the shardy functions into the mesh computation,
    // rewrite them to use the distributed dialect ops for functions,
    // yield, and call, and raise their collectives to our collective
    // abstractions. Normal HLO ops can be left as-is, since they are
    // supposed to be local sizes anyway.
    builder.setInsertionPointToEnd(&meshComputationBody.front());
    for (const FindShardyFunctionsAnalysis::FunctionInfo &info :
         analysis.getShardyFunctions()) {
      func::FuncOp shardyFunction = info.funcOp;
      builder.clone(*shardyFunction.getOperation());
    }

    RewritePatternSet patterns(moduleOp.getContext());
    patterns.add<FuncToDistributedFunctionPattern>(moduleOp.getContext(),
                                                   executionContext);
    patterns.add<FuncCallToDistributedCallPattern>(
        moduleOp.getContext(), distributedFunctionNames, replicateOver);
    patterns.add<FuncReturnToDistributedYieldPattern>(moduleOp.getContext());

    if (failed(applyPatternsGreedily(meshComputation, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    (void)shardyToDistributedAxis;
    (void)logicalMeshAxes;
    (void)factors;
  }
};

} // namespace

} // namespace mlir::enzyme::distributed