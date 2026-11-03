#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <mutex>
#include <optional>
#include <string>

#include "src/enzyme_ad/jax/Dialect/Distributed/CostBackend.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_ANNOTATEKERNELLATENCYPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Writes `kernelModule`, the simplified module buildKernelBodyModule built
// from `kernelOp`, under `directory`.
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

// Annotated kernel modules, keyed by what determines a backend's output: the
// backend, the target it costs for, and the simplified kernel module printed
// without locations. Process-wide, so identical kernels across search
// candidates (each a separate run of this pass) are annotated once.
//
// - Keys are SHA-256 digests of that text, so an entry holds one copy of a
//   kernel's IR, not two.
// - Entries are printed IR without locations rather than modules, so a hit
//   never refers to IR owned by another, possibly destroyed, MLIRContext, nor
//   to the source of whichever kernel filled the entry (see
//   restoreLocations).
class KernelLatencyCache {
public:
  std::optional<std::string> lookup(llvm::StringRef key) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(key);
    if (it == entries.end())
      return std::nullopt;
    return it->second;
  }

  void insert(llvm::StringRef key, std::string annotatedModule) {
    std::lock_guard<std::mutex> lock(mutex);
    entries.try_emplace(key, std::move(annotatedModule));
  }

private:
  std::mutex mutex;
  llvm::StringMap<std::string> entries;
};

KernelLatencyCache &getKernelLatencyCache() {
  static KernelLatencyCache cache;
  return cache;
}

std::string printWithoutLocations(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os, OpPrintingFlags().enableDebugInfo(false));
  return text;
}

std::string cacheKey(const CostBackend &backend, const CostTarget &target,
                     ModuleOp kernelModule) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << backend.name() << "\n"
     << target.cacheKey() << "\n"
     << printWithoutLocations(kernelModule);
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)));
}

// Gives `annotated`, parsed from the cache, the locations of `simplified`,
// the module it was looked up with. The two walk as the same op sequence
// whenever the backend only attached attributes; when it restructured
// @kernel instead (grouping ops into calls), every op takes `fallback`, the
// kernel's own location.
void restoreLocations(ModuleOp annotated, ModuleOp simplified,
                      Location fallback) {
  SmallVector<Operation *> from, to;
  simplified.walk([&](Operation *op) { from.push_back(op); });
  annotated.walk([&](Operation *op) { to.push_back(op); });
  bool aligned =
      from.size() == to.size() &&
      llvm::all_of(llvm::zip(from, to), [](auto pair) {
        return std::get<0>(pair)->getName() == std::get<1>(pair)->getName();
      });
  for (auto [index, op] : llvm::enumerate(to))
    op->setLoc(aligned ? from[index]->getLoc() : fallback);
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

    // Canonicalization and CSE put kernels that differ only in incidental
    // ways (dead ops, duplicate subexpressions, op order canonicalization
    // fixes) into one form, so they share a cache entry. Region
    // simplification stays off, as in the lowering pipeline, so @kernel's
    // block arguments, and with them its signature, are kept.
    PassManager simplify(&getContext(), ModuleOp::getOperationName());
    GreedyRewriteConfig canonConfig;
    canonConfig.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);
    simplify.addPass(createCanonicalizerPass(canonConfig));
    simplify.addPass(createCSEPass());

    for (DistributedKernelOp kernel : kernels)
      if (failed(annotateKernel(kernel, *backend, target, simplify)))
        return signalPassFailure();
  }

private:
  // Builds `kernel`'s standalone module and simplifies it, takes the
  // annotated module from the cache or has `backend` annotate it, and splices
  // the result back into `kernel`.
  LogicalResult annotateKernel(DistributedKernelOp kernel, CostBackend &backend,
                               const CostTarget &target,
                               PassManager &simplify) {
    FailureOr<ModuleOp> built = buildKernelBodyModule(kernel);
    if (failed(built))
      return failure(); // buildKernelBodyModule already emitted a diagnostic.
    OwningOpRef<ModuleOp> kernelModule(*built);
    if (failed(simplify.run(*kernelModule)))
      return failure(); // The failing pass already emitted a diagnostic.

    if (!dumpKernelModulesTo.empty() &&
        failed(dumpKernelModule(kernel, *kernelModule, dumpKernelModulesTo)))
      return failure();

    KernelLatencyCache &cache = getKernelLatencyCache();
    std::string key = cacheKey(backend, target, *kernelModule);
    if (std::optional<std::string> cached = cache.lookup(key)) {
      OwningOpRef<ModuleOp> annotated =
          parseSourceString<ModuleOp>(*cached, ParserConfig(&getContext()));
      assert(annotated && "a cached kernel module was printed from valid IR");
      restoreLocations(*annotated, *kernelModule, kernel.getLoc());
      return spliceKernelBodyModule(kernel, *annotated);
    }

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
    cache.insert(key, printWithoutLocations(*kernelModule));

    return spliceKernelBodyModule(kernel, *kernelModule);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
