#ifndef ENZYME_AD_JAX_DIALECT_BLAS_DIALECT_TD
#define ENZYME_AD_JAX_DIALECT_BLAS_DIALECT_TD

include "mlir/IR/DialectBase.td"

def BLAS_Dialect : Dialect {
  let name = "blas";
  let cppNamespace = "::mlir::blas";
  let description = [{
    The dialect models the Basic Linear Algebra Subprograms (BLAS) library, version 3.12. Unlike other interface dialects in upstream MLIR (e.g. MPI), the BLAS dialect is not intended to be a thin wrapper around the Fortran API of the BLAS library. Instead, it provides a slightly higher-level abstraction based on value semantics for interoperability with StableHLO, XLA and other dialects inside EnzymeXLA. The BLAS dialect is intended to be used as a common intermediate representation for linear algebra operations, and to enable optimizations and transformations that are specific to linear algebra computations.

    This dialect is under active development, and the design is subject to change.

    For an in-depth documentation of the BLAS library interface, please refer to the [official Netlib BLAS documentation](https://www.netlib.org/blas/)
  }];
  let useDefaultAttributePrinterParser = 1;
}

#endif // ENZYME_AD_JAX_DIALECT_BLAS_DIALECT_TD
