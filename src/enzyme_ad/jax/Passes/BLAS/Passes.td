#ifndef ENZYME_AD_JAX_PASSES_BLAS_PASSES_TD
#define ENZYME_AD_JAX_PASSES_BLAS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerBlasToStablehloPass : Pass<"lower-blas-to-stablehlo"> {
  let summary = "Lower BLAS operations to StableHLO ops";
  let dependentDialects = [
    "mlir::enzymexla::blas::BlasDialect",
    "stablehlo::StablehloDialect"
  ];

  let options = [];
}

def LowerBlasToJitCallPass : Pass<"lower-blas-to-jit_call"> {
  let summary = "Lower BLAS operations to enzymexla.jit_call ops";
  let dependentDialects = [
    "mlir::enzymexla::blas::BlasDialect",
    "mlir::enzymexla::EnzymeXLADialect"
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

#endif // ENZYME_AD_JAX_PASSES_BLAS_PASSES_TD
