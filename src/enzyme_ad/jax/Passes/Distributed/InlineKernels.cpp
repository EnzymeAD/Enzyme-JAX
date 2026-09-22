#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_INLINEKERNELSPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Splices a DistributedKernelOp's body directly into the surrounding block:
// block arguments are replaced by the kernel's operands, and uses of the
// kernel's results are replaced by the terminator's yielded values.
//
// By the time this pass runs, distributed-lower-kernels has resolved the
// local/global scope distinction described in DistributedKernelOp's own doc
// comment, so a kernel's operand/result types are guaranteed to already
// match its body's block-arg/yield types. Inlining is therefore a pure
// splice, with no type conversion required.
void inlineKernel(DistributedKernelOp kernel) {
  Block &body = kernel.getBody().front();
  auto yield = cast<DistributedYieldOp>(body.getTerminator());

  for (auto [blockArg, operand] :
       llvm::zip_equal(body.getArguments(), kernel.getArguments())) {
    assert(blockArg.getType() == operand.getType() &&
           "kernel operand and body block-arg types must already match by "
           "the time distributed-inline-kernels runs");
    blockArg.replaceAllUsesWith(operand);
  }

  Block *parent = kernel->getBlock();
  parent->getOperations().splice(Block::iterator(kernel), body.getOperations(),
                                 body.begin(), Block::iterator(yield));

  for (auto [result, yielded] :
       llvm::zip_equal(kernel.getResults(), yield.getReturns())) {
    assert(result.getType() == yielded.getType() &&
           "kernel result and yielded value types must already match by "
           "the time distributed-inline-kernels runs");
    result.replaceAllUsesWith(yielded);
  }

  kernel.erase();
}

struct InlineKernelsPass
    : public impl::InlineKernelsPassBase<InlineKernelsPass> {
  using InlineKernelsPassBase::InlineKernelsPassBase;

  void runOnOperation() override {
    // Collect kernels before mutating: inlining splices a kernel's body
    // into its parent block and erases the kernel op, which would
    // invalidate an in-flight walk over the same region.
    //
    // DistributedKernelOp bodies never contain another DistributedKernelOp
    // today (ClusterDistributedKernels only clusters kernel-free compute
    // into a new kernel), but collecting up front rather than relying on
    // that means a nested kernel would still be handled correctly: it
    // simply gets processed later in this same loop, from its
    // by-then-updated parent block.
    SmallVector<DistributedKernelOp> kernels;
    getOperation().walk(
        [&](DistributedKernelOp kernel) { kernels.push_back(kernel); });
    for (DistributedKernelOp kernel : kernels)
      inlineKernel(kernel);
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
