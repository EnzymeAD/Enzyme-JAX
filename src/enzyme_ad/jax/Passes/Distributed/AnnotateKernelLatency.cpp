#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/CostBackend.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ANNOTATEKERNELLATENCYPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Writes `kernelModule`, the module buildKernelBodyModule built from
// `kernelOp`, under `directory`.
//
// This pass runs once per search candidate (it sits at the end of
// buildDistributedSearchLoweringPipeline's kernel cleanup), so a naming
// scheme keyed only on a per-invocation kernel index would collide and
// overwrite dumps across candidates, across sibling kernels in the same
// walk, and across separate enzymexlamlir-opt runs sharing a directory.
// createUniqueFile sidesteps all of that with an atomically-created,
// randomly-suffixed name; nothing here needs to map a dump back to the
// candidate or kernel that produced it.
static LogicalResult dumpKernelModule(DistributedKernelOp kernelOp,
                                      ModuleOp kernelModule,
                                      StringRef directory) {
  if (auto ec = llvm::sys::fs::create_directories(directory)) {
    return kernelOp.emitError() << "failed to create kernel dump directory '"
                                << directory << "': " << ec.message();
  }

  SmallString<128> model(directory);
  llvm::sys::path::append(model, "kernel-%%%%%%.mlir");

  int fd;
  SmallString<128> path;
  if (auto ec = llvm::sys::fs::createUniqueFile(model, fd, path)) {
    return kernelOp.emitError()
           << "failed to create a unique kernel dump file under '" << directory
           << "': " << ec.message();
  }
  llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
  kernelModule.print(os);
  os << "\n";
  return success();
}

// Whether `kernelModule` meets CostBackend::annotate's output contract for a
// @kernel whose signature was `signature` before annotation.
[[maybe_unused]] static bool satisfiesAnnotateContract(ModuleOp kernelModule,
                                                       FunctionType signature) {
  auto kernel = SymbolTable(kernelModule).lookup<func::FuncOp>("kernel");
  if (!kernel || kernel.getFunctionType() != signature)
    return false;
  return llvm::all_of(
      kernel.getBody().front().without_terminator(),
      [](Operation &op) { return getLatency(&op).has_value(); });
}

struct AnnotateKernelLatencyPass
    : public impl::AnnotateKernelLatencyPassBase<AnnotateKernelLatencyPass> {
  using AnnotateKernelLatencyPassBase::AnnotateKernelLatencyPassBase;

  void runOnOperation() override {
    SmallVector<DistributedKernelOp> kernels;
    getOperation()->walk(
        [&](DistributedKernelOp kernel) { kernels.push_back(kernel); });
    // Nothing is costed without a kernel, so a kernel-free module needs no
    // mesh and no backend.
    if (kernels.empty())
      return;

    // The mesh is user input, so an unregistered device_target is diagnosed
    // here rather than asserted.
    FailureOr<PhysicalMeshOp> mesh = findUniquePhysicalMesh(getOperation());
    if (failed(mesh))
      return signalPassFailure(); // Diagnostic already emitted.
    StringRef deviceTarget = mesh->getDeviceTarget();
    CostBackend *backend = lookupCostBackend(deviceTarget);
    if (!backend) {
      mesh->emitError() << "no cost backend registered for device_target '"
                        << deviceTarget << "' (registered: "
                        << llvm::join(registeredCostBackendNames(), ", ")
                        << ")";
      return signalPassFailure();
    }
    CostTarget target{deviceTarget.str(),
                      deviceCostParamsFromPhysicalMesh(*mesh)};

    for (DistributedKernelOp kernel : kernels)
      if (failed(annotateKernel(kernel, *backend, target)))
        return signalPassFailure();
  }

private:
  // Builds `kernel`'s standalone module, has `backend` annotate it, and
  // splices the result back into `kernel`.
  LogicalResult annotateKernel(DistributedKernelOp kernel, CostBackend &backend,
                               const CostTarget &target) {
    FailureOr<ModuleOp> built = buildKernelBodyModule(kernel);
    if (failed(built))
      return failure(); // buildKernelBodyModule already emitted a diagnostic.
    OwningOpRef<ModuleOp> kernelModule(*built);

    if (!dumpKernelModulesTo.empty() &&
        failed(dumpKernelModule(kernel, *kernelModule, dumpKernelModulesTo)))
      return failure();

    [[maybe_unused]] FunctionType signature =
        SymbolTable(*kernelModule)
            .lookup<func::FuncOp>("kernel")
            .getFunctionType();
    if (failed(backend.annotate(*kernelModule, target)))
      return failure();
    ++numBackendAnnotateCalls;
    assert(satisfiesAnnotateContract(*kernelModule, signature) &&
           "a cost backend must keep @kernel's signature and annotate every "
           "top-level op of it with distributed.latency");

    return spliceKernelBodyModule(kernel, *kernelModule);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
