#ifndef ENZYME_AD_JAX_PACT_PASSES
#define ENZYME_AD_JAX_PACT_PASSES

include "mlir/Pass/PassBase.td"

def IntrinsicRaisingPass : Pass<"pact-intrinsic-raising", "mlir::ModuleOp"> {
  let summary = "Raise LLVM NVVM intrinsics to GPU dialect ops (L5 → L4)";
  let description = [{
    Replaces LLVM calls to NVVM intrinsics (shfl.sync, barrier0, lane_id)
    with corresponding gpu dialect ops. Preserves source-side metadata
    (thread masks, shuffle modes) as discardable attributes so that the
    downstream idiom recognition pass can recover semantic contracts.

    This is the first stage of the pact raising pipeline (L5 → L4).
  }];
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect",
  ];
}

def IdiomRecognitionPass : Pass<"pact-idiom-recognition", "mlir::ModuleOp"> {
  let summary = "Recognize GPU idioms and produce pact ops with contracts (L4 -> L3)";
  let description = [{
    Recognizes GPU-dialect idioms and replaces matched subgraphs with PACT semantic operations.
  }];
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::arith::ArithDialect",
    "::mlir::enzyme::pact::PactDialect",
  ];
}

def InjectCapabilityPass : Pass<"pact-inject-capability", "mlir::ModuleOp"> {
  let summary = "Inject target capability.* attributes onto all PACT operations";
  let description = [{
    Projects target hardware capabilities and attaches them as capability.* attributes
    to all operations in the PACT dialect namespace.
  }];
  let dependentDialects = [
    "::mlir::enzyme::pact::PactDialect"
  ];
}

def SchemeAlignmentCheckPass : Pass<"pact-scheme-alignment-check", "mlir::ModuleOp"> {
  let summary = "Validate contract/capability key alignment against the scheme";
  let description = [{
    Validates that contract.* and capability.* attributes attached to PACT ops conform to the
    shared property scheme (keys, kinds and expected value shapes).
  }];
  let dependentDialects = [
    "::mlir::enzyme::pact::PactDialect"
  ];
}

def AdaptiveMeetPass : Pass<"pact-adaptive-meet", "mlir::ModuleOp"> {
  let summary = "Reconcile contracts and capabilities and attached a typed PACT plan";
  let description = [{
    Performs the adaptive meeting between source contracts (contract.*) and target capabilities (capability.*),
    producing a per-op plan attribute (pact.plan) that guides subsequent lowering.
  }];
  let dependentDialects = [
    "::mlir::enzyme::pact::PactDialect"
  ];
}

def PACTToTargetPass : Pass<"pact-to-target", "mlir::ModuleOp"> {
  let summary = "Lower PACT ops guided by pact.plan";
  let description = [{
    Consumes pact.plan and lowers PACT semantic operations into target dialect guided by plan strategies.
  }];
  let dependentDialects = [
    "::mlir::enzyme::pact::PactDialect",
    "mlir::gpu::GPUDialect",
  ];
}

def DischargePass : Pass<"pact-discharge", "mlir::ModuleOp"> {
  let summary = "Verify and discharge all PACT artifacts before standard lowering";
  let description = [{
    verifies that no PACT dialect operations and no PACT specific
    attributes remain in the module.
  }];
  let dependentDialects = [
    "::mlir::enzyme::pact::PactDialect"
  ];
}

#endif // ENZYME_AD_JAX_PACT_PASSES
