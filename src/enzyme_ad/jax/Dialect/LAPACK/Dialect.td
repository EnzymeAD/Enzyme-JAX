#ifndef ENZYMEXLA_DIALECT_LAPACK_DIALECT_TD
#define ENZYMEXLA_DIALECT_LAPACK_DIALECT_TD

include "mlir/IR/DialectBase.td"

def LAPACK_Dialect : Dialect {
  let name = "lapack";
  let cppNamespace = "::mlir::enzymexla::lapack";
  let description = [{
    The dialect models the LAPACK library, version 3.12. Unlike other interface dialects in upstream MLIR (e.g. MPI), the LAPACK dialect is not intended to be a thin wrapper around the Fortran API of the LAPACK library. Instead, it provides a slightly higher-level abstraction based on value semantics for interoperability with StableHLO, XLA and other dialects inside EnzymeXLA. The LAPACK dialect is intended to be used as a common intermediate representation for linear algebra factorizations, and to enable optimizations and transformations that are specific to linear algebra operations.

    This dialect is under active development, and the design is subject to change.

    For an in-depth documentation of the LAPACK library interface, please refer to the [official Netlib LAPACK documentation](https://www.netlib.org/lapack/)
  }];
}

#endif // ENZYMEXLA_DIALECT_LAPACK_DIALECT_TD
