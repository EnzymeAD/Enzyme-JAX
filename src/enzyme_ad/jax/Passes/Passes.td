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

def CanonicalizeLoopsPass : Pass<"canonicalize-loops"> {
  let summary = "Canonicalize loops";
  let dependentDialects = ["affine::AffineDialect"];
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

def ConsumingInterpreterPass : Pass<"enzyme-consuming-transform-interpreter"> {
  let summary = "Run the transform interpreter and remove the script";
  let description =
      [{This pass isolates the transform script in a separate module,
        making it possible to apply the script to the anchor operation of the
            pass.}];
}

def EnzymeHLOOptPass : Pass<"enzyme-hlo-opt"> {
  let summary = "Optimize stablehlo";
  let dependentDialects =
      ["stablehlo::StablehloDialect", "tensor::TensorDialect"];
  let options = [
    Option<
        /*C++ variable name=*/"all_finite",
        /*CLI argument=*/"all_finite",
        /*type=*/"bool",
        /*default=*/"false",
        /*description=*/"Whether to raise to assume all variables are finite">,
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
        /*description=*/"Additional optimization passes">
  ];
}

def EnzymeHLOUnrollPass : Pass<"enzyme-hlo-unroll"> {
  let summary = "Unroll stablehlo";
  let dependentDialects =
      ["stablehlo::StablehloDialect", "tensor::TensorDialect"];
}

def PrintPass : Pass<"print"> {
  let summary = "Print the module";
  let options = [Option<
      /*C++ variable name=*/"use_stdout",
      /*CLI argument=*/"stdout",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to print to stdout (vs stderr)">];
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
    "math::MathDialect",
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
           "the default MLIR descriptor structure">
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
    "memref::MemRefDialect",
    "affine::AffineDialect",
    "vector::VectorDialect",
    "enzymexla::EnzymeXLADialect",
    "vector::VectorDialect",
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
  ];
}


#endif
