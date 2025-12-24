//===- Passes.td - EnzymeXLA pass tablegen macros  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEXLA_PASSES
#define ENZYMEXLA_PASSES

include "mlir/Pass/PassBase.td"

def TestPolymerPass : Pass<"test-polymer"> {
  let dependentDialects = [
    "enzymexla::EnzymeXLADialect",
  ];
}

def CanonicalizeLoopsPass : InterfacePass<"canonicalize-loops", 
                                          "mlir::FunctionOpInterface"> {
  let summary = "Canonicalize loops";
  let dependentDialects = ["affine::AffineDialect"];
}

def DialectCanonicalize : Pass<"dialect-canonicalize"> {
  let summary = "Canonicalize specific dialects listed in options";
  let options = [
    ListOption<"dialects", "dialects", "std::string",
               "List of dialects to fetch canonicalization patterns from">,
    Option<"maxNumRewrites", "max-num-rewrites", "uint64_t",
           /*default=*/"static_cast<uint64_t>(-1)",
           "Maximum number of rewrites to apply (default -1: unlimited)">,
    Option<"maxIterations", "max-iterations", "uint64_t",
           /*default=*/"static_cast<uint64_t>(-1)",
           "Maximum number of times a pattern can be applied (default -1: "
           "inherit from the driver)">,
    Option<"enableFolding", "enable-folding", "bool", "true",
           "Perform folding">,
    Option<"enableConstantCSE", "enable-constant-cse", "bool", "true",
           "Deduplicate constants">,
  ];
}

def RemoveDuplicateFuncDefPass
    : Pass<"remove-duplicate-func-def", "mlir::ModuleOp"> {
  let summary = "Remove duplicate function definitions";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def PropagateConstantBoundsPass
    : Pass<"propagate-constant-bounds", "ModuleOp"> {
  let summary = "Propagate constant bounds";
  let description = [{
    Propagate constant bounds information:
    1. thread index 
    2. Block dimension 
    3. Block index.
    Additionally, set all the following attributes for all kernel pointers:
    1. align 128
    2. no alias 
    3. dereferenceable = tensor.size() * sizeof(element_type)
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect"
  ];
}

def MarkFunctionMemoryEffectsPass : Pass<"mark-func-memory-effects", "ModuleOp"> {
  let summary = "Attach enzymexla.memory_effects attribute summarizing memory access";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
    "memref::MemRefDialect",
  ];
  let options = [
    Option<
      /*C++ variable name=*/"max_iterations",
      /*CLI argument=*/"max_iterations",
      /*type=*/"int32_t",
      /*default=*/"8",
      /*description=*/"propgate memory effects up to this many iterations (only for call graphs with cycles)">,
    Option<
      /*C++ variable name=*/"assume_no_memory_effects",
      /*CLI argument=*/"assume_no_memory_effects",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"assume no memory effects for ops not implementing MemoryEffectOpInterface">];
}

def ArithRaisingPass : Pass<"arith-raise"> {
  let summary = "Raise Arith to mhlo";
  let dependentDialects = [
    "arith::ArithDialect",
    "mhlo::MhloDialect",
    "stablehlo::StablehloDialect",
    "chlo::ChloDialect",
    "enzyme::EnzymeDialect"
  ];
  let options = [Option<
      /*C++ variable name=*/"use_stablehlo",
      /*CLI argument=*/"stablehlo",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to raise to stablehlo vs mhlo">];
}

def TensorEmptyOpRaisingPass : Pass<"tensor-empty-raise"> {
  let summary = "Raise tensor.empty ops";
  let dependentDialects = [
    "tensor::TensorDialect",
    "stablehlo::StablehloDialect"
  ];
}

def ConsumingInterpreterPass : Pass<"enzyme-consuming-transform-interpreter"> {
  let summary = "Run the transform interpreter and remove the script";
  let description =
      [{This pass isolates the transform script in a separate module,
        making it possible to apply the script to the anchor operation of the
            pass.}];
}

def EnzymeHLOOptPass : Pass<"enzyme-hlo-opt"> {
  let summary = "Optimize stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "tensor::TensorDialect",
    "enzyme::EnzymeDialect",
    "enzymexla::EnzymeXLADialect",
    "chlo::ChloDialect"
  ];
  let options = [
    Option<
        /*C++ variable name=*/"all_finite",
        /*CLI argument=*/"all_finite",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to raise to assume all variables are finite">,
    Option<
        /*C++ variable name=*/"licm_passes",
        /*CLI argument=*/"licm_passes",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to run LICM passes">,
    Option<
        /*C++ variable name=*/"no_nan",
        /*CLI argument=*/"no_nan",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to raise to assume no variables are nan">,
    Option<
        /*C++ variable name=*/"max_constant_expansion",
        /*CLI argument=*/"max_constant_expansion",
        /*type=*/"size_t",
        /*default=*/"1024",
        /*description=*/"Maximum size to expand constants into">,
    Option<
        /*C++ variable name=*/"max_iterations",
        /*CLI argument=*/"max_iterations",
        /*type=*/"int64_t",
        /*default=*/"100",
        /*description=*/"Maximum number of pattern iterations">,
    Option<
        /*C++ variable name=*/"top_down",
        /*CLI argument=*/"top_down",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Use top down traversal">,
    Option<
        /*C++ variable name=*/"cse",
        /*CLI argument=*/"cse",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Run CSE alongside">,
    Option<
        /*C++ variable name=*/"passses",
        /*CLI argument=*/"passses",
        /*type=*/"uint64_t",
        /*default=*/"24575",
        /*description=*/"Additional optimization passes">,
    Option<
        /*C++ variable name=*/"enable_convert_to_convolution",
        /*CLI argument=*/"enable_convert_to_convolution",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Enable convert to convolution">,
    Option<
        /*C++ variable name=*/"enable_auto_batching_passes",
        /*CLI argument=*/"enable_auto_batching_passes",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Enable auto batching passes">
  ];
}

def EnzymeHLOUnrollPass : Pass<"enzyme-hlo-unroll"> {
  let summary = "Unroll stablehlo";
  let dependentDialects =
      ["stablehlo::StablehloDialect", "tensor::TensorDialect"];
  let options = [
    Option<
        /*C++ variable name=*/"maxNumIterations",
        /*CLI argument=*/"max-num-iterations",
        /*type=*/"int",
        /*default=*/"-1",
        /*description=*/"Only unroll if total iterations is less than this value. If -1, no limit.">,
    Option<
        /*C++ variable name=*/"maxOperationThreshold",
        /*CLI argument=*/"max-operation-threshold",
        /*type=*/"int",
        /*default=*/"-1",
        /*description=*/"Only unroll if total operations is less than this value. If -1, no limit.">
  ];
}

def PrintPass : Pass<"print"> {
  let summary = "Print the module";
  let options = [Option<
      /*C++ variable name=*/"use_stdout",
      /*CLI argument=*/"stdout",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to print to stdout (vs stderr)">,
      Option</*C++ variable name=*/"debug",
      /*CLI argument=*/"debug",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether to print to stdout (vs stderr)">,
      Option</*C++ variable name=*/"generic",
      /*CLI argument=*/"generic",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether to print to generic mlir form">,
      Option</*C++ variable name=*/"filename",
      /*CLI argument=*/"filename",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"If not-empty, will print the module to a file">,
      ];
}

def SortMemory : Pass<"sort-memory"> {
  let summary = "Sort memory accesses";
}

def PrintLocationPass : Pass<"print-location"> {
  let summary = "Print locations of attributed operations";
  let options = [Option<
      /*C++ variable name=*/"shouldPrint",
      /*CLI argument=*/"print",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to print the location">,
    Option<
      /*C++ variable name=*/"shouldAttach",
      /*CLI argument=*/"attach",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to attach the location as attribute">,
    ];
}

def SROAWrappersPass : Pass<"sroa-wrappers", "mlir::ModuleOp"> {
  let summary = "Run LLVM SROA (Scalar Replacement of Aggregates)";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::DLTIDialect", "mlir::NVVM::NVVMDialect",
    "mlir::arith::ArithDialect", "mlir::math::MathDialect"
  ];
  let options = [
    Option<
        /*C++ variable name=*/"dump_prellvm",
        /*CLI argument=*/"dump_prellvm",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to dump LLVM before optimizations">,
    Option<
        /*C++ variable name=*/"dump_postllvm",
        /*CLI argument=*/"dump_postllvm",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to dump LLVM after optimizations">,
    Option<
        /*C++ variable name=*/"instcombine",
        /*CLI argument=*/"instcombine",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to run instcombine">,
    Option<
        /*C++ variable name=*/"instsimplify",
        /*CLI argument=*/"instsimplify",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to run instsimplify">,
    Option<
        /*C++ variable name=*/"sroa",
        /*CLI argument=*/"sroa",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to run instsimplify">,
    Option<
        /*C++ variable name=*/"set_private",
        /*CLI argument=*/"set_private",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to run instsimplify">,
    Option<
        /*C++ variable name=*/"attributor",
        /*CLI argument=*/"attributor",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to run instsimplify">,
  ];
}

def LibDeviceFuncsRaisingPass : Pass<"libdevice-funcs-raise"> {
  let summary = "Raise libdevice function calls to arith/math operations";
  let dependentDialects = [
    "arith::ArithDialect",
    "enzyme::EnzymeDialect",
    "math::MathDialect",
    "gpu::GPUDialect",
  ];
  let options = [
    Option<
        /*C++ variable name=*/"remove_freeze",
        /*CLI argument=*/"remove_freeze",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to remove llvm freeze operations">,
  ];
}

def ConvertPolygeistToLLVM
    : Pass<"convert-polygeist-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

        ####Input invariant

        - operations including : arithmetic on integers and floats,
        constants, direct calls, returns and branches;
    -no `tensor` types;
    -all `vector` are one - dimensional;
    -all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
        IR dialect operations,
        the pass will
                fail.Any LLVM IR operations or types already present in the IR
                    will be kept as is
                        .

            ####Output IR

                Functions converted to LLVM IR.Function arguments types are
                    converted one -
            to - one.Function results are converted one - to - one and,
        in case more than 1 value is returned,
        packed into an LLVM
            IR struct type.Function calls and returns are updated accordingly
                .Block argument types are updated to use LLVM IR types.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "gpu::GPUDialect",
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "scf::SCFDialect",
    "ROCDL::ROCDLDialect"
  ];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/ "0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">,
    Option<"useCStyleMemRef", "use-c-style-memref", "bool",
           /*default=*/"true",
           "Use C-style nested-array lowering of memref instead of "
           "the default MLIR descriptor structure">,
    Option<"use_async", "use_async", "bool",
           /*default=*/"false",
           "Use C-style nested-array lowering of memref instead of "
           "the default MLIR descriptor structure">,
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">
  ];
}

def DelinearizeIndexingPass : Pass<"delinearize-indexing"> {
  let summary = "Delinearize indexing";
  let dependentDialects = [
    "affine::AffineDialect",
    "enzymexla::EnzymeXLADialect",
  ];
}


def LowerKernelPass : Pass<"lower-kernel"> {
  let summary = "Lower kernel to custom call";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "gpu::GPUDialect", 
    "func::FuncDialect",
    "affine::AffineDialect",
    "scf::SCFDialect",
    "arith::ArithDialect",
    "tensor::TensorDialect"
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cuda\"",
        /*description=*/"HW backend">,
  ];
}

def LowerEnzymeXLALinalgPass : Pass<"lower-enzymexla-linalg"> {
  let summary = "Lower enzymexla linalg ops";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "LLVM::LLVMDialect",
    "enzyme::EnzymeDialect",
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">,
    Option<
        /*C++ variable name=*/"blasIntWidth",
        /*CLI argument=*/"blas_int_width",
        /*type=*/"int",
        /*default=*/"64",
        /*description=*/"Blas int width (32 or 64). Only used for CPU backend.">,
  ];
}

def LowerEnzymeXLALapackPass : Pass<"lower-enzymexla-lapack"> {
  let summary = "Lower enzymexla.lapack ops to stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "LLVM::LLVMDialect",
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">,
    Option<
        /*C++ variable name=*/"blasIntWidth",
        /*CLI argument=*/"blas_int_width",
        /*type=*/"int",
        /*default=*/"64",
        /*description=*/"Blas int width (32 or 64). Only used for CPU backend.">,
  ];
}

def LowerEnzymeXLABLASPass : Pass<"lower-enzymexla-blas"> {
  let summary = "Lower enzymexla.blas ops to stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "LLVM::LLVMDialect",
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">,
    Option<
        /*C++ variable name=*/"blasIntWidth",
        /*CLI argument=*/"blas_int_width",
        /*type=*/"int",
        /*default=*/"64",
        /*description=*/"Blas int width (32 or 64). Only used for CPU backend.">,
  ];
}

def LowerProbProgToStableHLOPass : Pass<"lower-probprog-to-stablehlo"> {
  let summary = "Lower computational enzyme ops to StableHLO";
  let description = [{
    Lower computational probprog ops (loop, dot, cholesky_solve, random, unflatten_slice)
    to StableHLO dialect operations.
  }];
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "chlo::ChloDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "enzyme::EnzymeDialect",
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">,
  ];
}

def LowerProbProgTraceOpsPass : Pass<"lower-probprog-trace-ops"> {
  let summary = "Lower probprog trace manipulation ops to JIT calls";
  let description = [{
    Lower probprog trace manipulation ops (initTrace, addSampleToTrace, getSampleFromTrace, etc.)
    to enzymexla.jit_call operations that invoke external runtime functions (e.g., Julia `@cfunction`).
    This pass also handles type conversions (!enzyme.Trace -> tensor<ui64>, !enzyme.Constraint -> tensor<ui64>).
  }];
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
    "LLVM::LLVMDialect",
    "func::FuncDialect",
  ];

  let options = [
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cpu\"",
        /*description=*/"HW backend">,
  ];
}

def LowerEnzymeXLAMLPass : Pass<"lower-enzymexla-ml"> {
  let summary = "Lower enzymexla mlops to stablehlo";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "chlo::ChloDialect",
    "enzymexla::EnzymeXLADialect",
  ];
}

def RaiseTritonCustomCallPass : Pass<"raise-triton-custom-call"> {
  let summary = "Raise triton custom kernel call";
  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::enzymexla::triton_ext::TritonExtDialect",
    "stablehlo::StablehloDialect"
  ];
}

def LowerJITPass : Pass<"lower-jit"> {
  let summary = "Lower jit call to custom call";
  let dependentDialects = [
    "stablehlo::StablehloDialect", 
    "gpu::GPUDialect", 
    "func::FuncDialect",
    "math::MathDialect", 
    "memref::MemRefDialect", 
    "scf::SCFDialect",
    "vector::VectorDialect", 
    "nvgpu::NVGPUDialect", 
    "NVVM::NVVMDialect",
    "LLVM::LLVMDialect", 
    "arith::ArithDialect", 
    "tensor::TensorDialect"
  ];

  let options = [
    Option<
        /*C++ variable name=*/"jit",
        /*CLI argument=*/"jit",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to jit the kernel">,
    Option<
        /*C++ variable name=*/"compileLaunch",
        /*CLI argument=*/"compileLaunch",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Whether to jit the host code">,
    Option<
        /*C++ variable name=*/"toolkitPath",
        /*CLI argument=*/"toolkitPath",
        /*type=*/"std::string",
        /*default=*/"",
        /*description=*/"The location of the cuda toolkit">,
    Option<
        /*C++ variable name=*/"linkFiles",
        /*CLI argument=*/"linkFiles",
        /*type=*/"std::string",
        /*default=*/"",
        /*description=*/"Semicolon separated list of files to link">,
    Option<
        /*C++ variable name=*/"cubinChip",
        /*CLI argument=*/"cubinChip",
        /*type=*/"std::string",
        /*default=*/"\"sm_50\"",
        /*description=*/"cubinChip">,
    Option<
        /*C++ variable name=*/"cubinFeatures",
        /*CLI argument=*/"cubinFeatures",
        /*type=*/"std::string",
        /*default=*/"\"+ptx60\"",
        /*description=*/"cubinChip">,
    Option<
        /*C++ variable name=*/"indexBitWidth",
        /*CLI argument=*/"indexBitWidth",
        /*type=*/"int",
        /*default=*/"64",
        /*description=*/"indexBitWidth">,
    Option<
        /*C++ variable name=*/"run_init",
        /*CLI argument=*/"run_init",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Run initialization of cuda module">,
    Option<
        /*C++ variable name=*/"debug",
        /*CLI argument=*/"debug",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Compile in debug prints">,
    Option<
        /*C++ variable name=*/"cuResultHandlerPtr",
        /*CLI argument=*/"cuResultHandlerPtr",
        /*type=*/"size_t",
        /*default=*/"0",
        /*description=*/"Function handler to call with result of curesult">,
    Option<
        /*C++ variable name=*/"cuStreamSynchronizePtr",
        /*CLI argument=*/"cuStreamSynchronizePtr",
        /*type=*/"size_t",
        /*default=*/"0",
        /*description=*/"Function handler to sync results">,
    Option<
        /*C++ variable name=*/"cubinFormat",
        /*CLI argument=*/"cubinFormat",
        /*type=*/"std::string",
        /*default=*/"\"bin\"",
        /*description=*/"Binary format">,
    Option<
        /*C++ variable name=*/"cuOptLevel",
        /*CLI argument=*/"cuOptLevel",
        /*type=*/"int",
        /*default=*/"2",
        /*description=*/"Opt level for ptx">,
    Option<
        /*C++ variable name=*/"cubinTriple",
        /*CLI argument=*/"cubinTriple",
        /*type=*/"std::string",
        /*default=*/"\"nvptx64-nvidia-cuda\"",
        /*description=*/"Target triple">,
    Option<
        /*C++ variable name=*/"backend",
        /*CLI argument=*/"backend",
        /*type=*/"std::string",
        /*default=*/"\"cuda\"",
        /*description=*/"HW backend">,
    Option<
        /*C++ variable name=*/"openmp",
        /*CLI argument=*/"openmp",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"whether to use openmp for lowering">,
  ];
}

//===----------------------------------------------------------------------===//
// LLVMToControlFlow
//===----------------------------------------------------------------------===//

def ConvertLLVMToControlFlowPass : Pass<"convert-llvm-to-cf"> {
  let summary = "Convert LLVM cf operations to the ControlFlow dialect";
  let dependentDialects = ["cf::ControlFlowDialect"];
}

//===----------------------------------------------------------------------===//
// ControlFlowToSCF
//===----------------------------------------------------------------------===//

def EnzymeLiftControlFlowToSCFPass : Pass<"enzyme-lift-cf-to-scf"> {
  let summary = "Lift ControlFlow dialect to SCF dialect";
  let description = [{
    Lifts ControlFlow operations to SCF dialect operations.

    This pass is prefixed with "lift" instead of "convert" as it is not always
    guaranteed to replace all ControlFlow ops.
    If a region contains only a single kind of return-like operation, all
    ControlFlow operations will be replaced successfully.
    Otherwise a single ControlFlow switch branching to one block per return-like
    operation kind remains.

    This pass may need to create unreachable terminators in case of infinite
    loops, which is only supported for 'func.func' for now. If you potentially
    have infinite loops inside CFG regions not belonging to 'func.func',
    consider using `transformCFGToSCF` function directly with corresponding
    `CFGToSCFInterface::createUnreachableTerminator` implementation.
  }];

  let dependentDialects = [
    "scf::SCFDialect", "arith::ArithDialect", "ub::UBDialect",
    // TODO: This is only necessary until we have a
    //       ub.unreachable op.
    "func::FuncDialect"
  ];
}

def LLVMToAffineAccessPass : Pass<"llvm-to-affine-access"> {
  let summary = "";
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "memref::MemRefDialect",
    "affine::AffineDialect",
    "vector::VectorDialect",
    "enzymexla::EnzymeXLADialect",
    "vector::VectorDialect",
  ];
}

def SimplifyAffineExprsPass : Pass<"simplify-affine-exprs"> {
  let summary = "";
  let dependentDialects = [
  ];
}

def LLVMToMemrefAccessPass : Pass<"llvm-to-memref-access"> {
  let summary = "";
  let dependentDialects = [
    "memref::MemRefDialect",
    "vector::VectorDialect",
  ];
}

def SCFCanonicalizeFor : Pass<"canonicalize-scf-for"> {
  let summary = "Run some additional canonicalization for scf::for";
  let dependentDialects = [
    "scf::SCFDialect",
    "math::MathDialect",
    "ub::UBDialect",
  ];
}

def AffineCFG : Pass<"affine-cfg"> {
  let summary = "Replace scf.if and similar with affine.if";
  let dependentDialects = [
    "scf::SCFDialect",
    "affine::AffineDialect",
    "enzyme::EnzymeDialect",
  ];
}

def CuDNNHLOOpt : Pass<"enzymexla-cudnn-hlo-opt"> {
  let summary = "Optimize stablehlo to emit cuDNN specific optimizations";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzymexla::EnzymeXLADialect",
  ];
}

def GPULaunchRecognition : Pass<"gpu-launch-recognition"> {
  let summary = "Optimize stablehlo to emit cuDNN specific optimizations";
  let dependentDialects = [
    "enzymexla::EnzymeXLADialect",
    "arith::ArithDialect", 
    "gpu::GPUDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::DLTIDialect"
  ];
  let options = [Option<
       /*C++ variable name=*/"use_launch_func",
       /*CLI argument=*/"use_launch_func",
       /*type=*/"bool",
       /*default=*/"false",
       /*description=*/"Convert Periodic Concat to Manual Computation with CollectivePermute">];
}

def MergeGPUModulesPass : Pass<"merge-gpu-modules", "mlir::ModuleOp"> {
  let summary = "Merge all gpu modules into one";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
}

def ConvertParallelToGPU1 : Pass<"convert-parallel-to-gpu1"> {
  let summary = "Convert parallel loops to gpu";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect", "scf::SCFDialect"];
  let options = [
  Option<"arch", "arch", "std::string", /*default=*/"\"sm_60\"", "Target GPU architecture">
  ];
}

def ConvertParallelToGPU2 : Pass<"convert-parallel-to-gpu2"> {
  let summary = "Convert parallel loops to gpu";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "memref::MemRefDialect", "gpu::GPUDialect", "mlir::NVVM::NVVMDialect"];
  let options = [Option<
       /*C++ variable name=*/"emitGPUKernelLaunchBounds",
       /*CLI argument=*/"emitGPUKernelLaunchBounds",
       /*type=*/"bool",
       /*default=*/"false",
       /*description=*/"Convert Periodic Concat to Manual Computation with CollectivePermute">];
}

def PolygeistMem2Reg : Pass<"polygeist-mem2reg"> {
  let summary = "Replace scf.if and similar with affine.if";
}

def OptimizeCommunication : Pass<"optimize-communication"> {
  let summary = "Optimize communication";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "sdy::SdyDialect",
  ];
  let options =
  [Option<
       /*C++ variable name=*/"periodic_concat",
       /*CLI argument=*/"periodic_concat",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Periodic Concat to Manual Computation with CollectivePermute">,
       Option<
       /*C++ variable name=*/"rotate_comm",
       /*CLI argument=*/"rotate_comm",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Rotate to Manual Computation with CollectivePermute">,
       Option<
       /*C++ variable name=*/"rotate_spmd",
       /*CLI argument=*/"rotate_spmd",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Rotate to Spmd custom call">,
       Option<
       /*C++ variable name=*/"rotate_to_pad_comm",
       /*CLI argument=*/"rotate_to_pad_comm",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Perform a rotate with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"wrap_comm",
       /*CLI argument=*/"wrap_comm",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Wrap to Manual Computation with CollectivePermute">,
       Option<
       /*C++ variable name=*/"extend_comm",
       /*CLI argument=*/"extend_comm",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Extend to Manual Computation with CollectivePermute">,
       Option<
       /*C++ variable name=*/"dus_to_pad_manual_comp_comm",
       /*CLI argument=*/"dus_to_pad_manual_comp_comm",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Perform a DUS with Padding with Manual Computation to optimize the communication">,
       Option</*C++ variable name=*/"dus_to_pad_comm",
       /*CLI argument=*/"dus_to_pad_comm",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Perform a DUS with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"concat_two_operands_comm",
       /*CLI argument=*/"concat_two_operands_comm",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Convert Concatenate two operands to Manual Computation with CollectivePermute">,
       Option<
       /*C++ variable name=*/"concat_to_pad_comm",
       /*CLI argument=*/"concat_to_pad_comm",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Perform a Concatenate with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"extend_to_pad_comm",
       /*CLI argument=*/"extend_to_pad_comm",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Perform an Extend with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"extend_to_pad_comm2",
       /*CLI argument=*/"extend_to_pad_comm2",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Perform an Extend with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"concat_two_dus_like",
       /*CLI argument=*/"concat_two_dus_like",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Perform a two-concat with pads and a manual select">,
       Option<
       /*C++ variable name=*/"extend_dus_like",
       /*CLI argument=*/"extend_dus_like",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Perform a two-concat with pads and a manual select">,
       Option<
       /*C++ variable name=*/"wrap_to_pad_comm",
       /*CLI argument=*/"wrap_to_pad_comm",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Perform a Wrap with Padding to optimize the communication">,
       Option<
       /*C++ variable name=*/"wrap_to_rotate",
       /*CLI argument=*/"wrap_to_rotate",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Rewrite Wrap operations in terms of Extend and Rotate">,
       Option<
       /*C++ variable name=*/"reorder_associative",
       /*CLI argument=*/"reorder_associative",
       /*type=*/"int",
       /*default=*/"1",
       /*description=*/"Reorder associative operations to minimize communication">,
       Option<
       /*C++ variable name=*/"concat_slice_optimize",
       /*CLI argument=*/"concat_slice_optimize",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Optimize concat of two slices from same source by padding and rotating">,
       Option<
       /*C++ variable name=*/"concat_largest_operand",
       /*CLI argument=*/"concat_largest_operand",
       /*type=*/"int",
       /*default=*/"0",
       /*description=*/"Optimize concat by padding largest operand and DUS others">];
}

def AffineToStableHLORaising : Pass<"raise-affine-to-stablehlo"> {
  let summary = "Raise affine parallel loops to stablehlo ops if possible";
  let dependentDialects = [
    "affine::AffineDialect",
    "stablehlo::StablehloDialect",
    "func::FuncDialect",
  ];
  let options =
      [Option<
           /*C++ variable name=*/"err_if_not_fully_raised",
           /*CLI argument=*/"err_if_not_fully_raised",
           /*type=*/"bool",
           /*default=*/"true",
           /*description=*/"Whether to throw a pass error if not fully raised">,
       Option<
           /*C++ variable name=*/"enable_lockstep_for",
           /*CLI argument=*/"enable_lockstep_for",
           /*type=*/"bool",
           /*default=*/"true",
           /*description=*/"Whether to enable the lockstep for raising">,
       Option<
           /*C++ variable name=*/"dump_failed_lockstep",
           /*CLI argument=*/"dump_failed_lockstep",
           /*type=*/"bool",
           /*default=*/"false",
           /*description=*/"Whether to enable the lockstep for raising">,
       Option<
           /*C++ variable name=*/"prefer_while_raising",
           /*CLI argument=*/"prefer_while_raising",
           /*type=*/"bool",
           /*default=*/"true",
           /*description=*/
           "Whether to prefer raising to while instead of unrolling">,
  ];
}

def ParallelLower : Pass<"parallel-lower"> {
  let summary = "Lower gpu launch op to parallel ops";
  let dependentDialects = [
    "scf::SCFDialect",
    "async::AsyncDialect",
    "enzymexla::EnzymeXLADialect",
    "cf::ControlFlowDialect",
    "memref::MemRefDialect",
    "func::FuncDialect",
    "LLVM::LLVMDialect",
  ];
  let options =
      [Option<
           /*C++ variable name=*/"wrapParallelOps",
           /*CLI argument=*/"wrapParallelOps",
           /*type=*/"bool",
           /*default=*/"false",
           /*description=*/"Whether to throw a pass error if not fully raised">,
  ];
}

def SCFParallelLoopUnroll : Pass<"scf-parallel-loop-unroll"> {
  let summary = "Unroll and interleave scf parallel loops";
  let dependentDialects = [
    "scf::SCFDialect",
    "arith::ArithDialect",
  ];
  let options = [
  Option<"unrollFactor", "unrollFactor", "int", /*default=*/"2", "Unroll factor">
  ];
}

def ParallelLICM : Pass<"parallel-licm"> {
  let summary = "Perform LICM on known parallel (and serial) loops";
}

def SCFParallelSerialization : Pass<"parallel-serialization"> {
  let summary = "Serialize SCF parallel loops";
}

def FixGPUFunc : Pass<"fix-gpu-func", "mlir::gpu::GPUModuleOp"> {
  let summary = "Fix nested calls to gpu functions we generate in the frontend";
  let dependentDialects = ["func::FuncDialect", "LLVM::LLVMDialect", "gpu::GPUDialect"];
}

def StripGPUInfo : Pass<"strip-gpu-info"> {
  let summary = "Stirng GPU Debug info";
  let dependentDialects = ["gpu::GPUDialect"];
}

def AutoBatchingPass : Pass<"auto-batching"> {
  let summary = "Check if a sequence of ops can be fused into a single batch op";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "func::FuncDialect",
    "enzyme::EnzymeDialect",
  ];
  let options = [
    Option<
      /*C++ variable name=*/"concat_insert_dim_passes",
      /*CLI argument=*/"concat_insert_dim_passes",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to run concat insert dim passes">,
    Option<
      /*C++ variable name=*/"slice_to_batch_passes",
      /*CLI argument=*/"slice_to_batch_passes",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to run slice to batch passes">,
    Option<
      /*C++ variable name=*/"while_loop_batching_mode",
      /*CLI argument=*/"while_loop_batching_mode",
      /*type=*/"std::string",
      /*default=*/"\"greedy\"",
      /*description=*/"Whether to run while loop batching passes (greedy, none)">,
    Option<
      /*C++ variable name=*/"while_elementwise_reduction_to_reduce_passes",
      /*CLI argument=*/"while_elementwise_reduction_to_reduce_passes",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to run while elementwise reduction to reduce passes">,
    Option<
      /*C++ variable name=*/"while_is_copy_simplify_passes",
      /*CLI argument=*/"while_is_copy_simplify_passes",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to run while is copy simplify passes">,
    Option<
        /*C++ variable name=*/"max_iterations",
        /*CLI argument=*/"max_iterations",
        /*type=*/"int64_t",
        /*default=*/"4",
        /*description=*/"Maximum number of pattern iterations">,
    Option<
        /*C++ variable name=*/"top_down",
        /*CLI argument=*/"top_down",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Use top down traversal">,
  ];
}

def DropUnsupportedAttributesPass : Pass<"drop-unsupported-attributes", "ModuleOp"> {
  let summary = "Drop unsupported attributes for XLA";
  let dependentDialects = [];
  let options = [
    Option<
        /*C++ variable name=*/"enzymexla_memory_effects",
        /*CLI argument=*/"enzymexla-memory-effects",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Drop enzymexla.memory_effects attribute">,
    Option<
        /*C++ variable name=*/"enzymexla_analysis_result",
        /*CLI argument=*/"enzymexla-analysis-result",
        /*type=*/"bool",
        /*default=*/"true",
        /*description=*/"Drop attribute from various analysis results">,
  ];
}

def SCFCPUify : Pass<"cpuify"> {
  let summary = "remove barrier ig";
  let dependentDialects =
      ["memref::MemRefDialect", "func::FuncDialect", "LLVM::LLVMDialect"];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def ConvertTritonToTritonGPUPreservingModuleAttributesPass : Pass<
    "convert-triton-to-triton-gpu-preserving-module-attributes", "mlir::ModuleOp"> {
  let summary = "Triton generally compiles a single kernel, so they can specify the number of ctas and warps. However, we want to be able to compile multiple kernels. This pass will use the attributes from the module and use that to lower to TritonGPU.";
  let dependentDialects = [];
  let options = [
    Option<
        /*C++ variable name=*/"target",
        /*CLI argument=*/"target",
        /*type=*/"std::string",
        /*default=*/"\"\"",
        /*description=*/"the GPU target, e.g., cuda:80, hip:gfx942"
      >];
}

def EnzymeBatchToStableHLOPass : Pass<"enzyme-batch-to-stablehlo"> {
  let summary = "Legalize batching specific enzyme ops to stablehlo dialect";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
    "enzyme::EnzymeDialect"
  ];
}

def ConvertAllConstantsToSplattedConstantPass : Pass<"convert-all-constants-to-splatted-constant", "ModuleOp"> {
  let summary = "Convert all constants to splatted constants. This is supposed to be used for debugging purposes when dumping the module.";
  let dependentDialects = [
    "stablehlo::StablehloDialect",
  ];
}

#endif
