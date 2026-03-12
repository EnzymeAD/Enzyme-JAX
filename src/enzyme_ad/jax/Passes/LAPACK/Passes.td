#ifndef ENZYME_AD_JAX_PASSES_LAPACK_PASSES_TD
#define ENZYME_AD_JAX_PASSES_LAPACK_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerLapackToJitCallPass : Pass<"lower-lapack-to-jit_call"> {
  let summary = "Lower LAPACK operations to enzymexla.jit_call ops";
  let dependentDialects = [
    "mlir::lapack::LapackDialect",
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

#endif // ENZYME_AD_JAX_PASSES_LAPACK_PASSES_TD