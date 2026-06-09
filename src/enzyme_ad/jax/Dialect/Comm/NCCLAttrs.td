#ifndef ENZYMEXLA_COMM_DIALECT_NCCL_ATTRS_TD
#define ENZYMEXLA_COMM_DIALECT_NCCL_ATTRS_TD

include "Dialect.td"

include "mlir/IR/EnumAttr.td"

def NCCL_RedOps : I32EnumAttr<"NCCL_RedOps", "Named predefined NCCL reduction operators", [
      I32EnumAttrCase<"NCCL_SUM", 0>,
      I32EnumAttrCase<"NCCL_PROD", 1>,
      I32EnumAttrCase<"NCCL_MIN", 2>,
      I32EnumAttrCase<"NCCL_MAX", 3>,
      I32EnumAttrCase<"NCCL_AVG", 4>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::comm";
}

def NCCL_RedOpAttr : EnumAttr<Comm_Dialect, NCCL_RedOps, "nccl.red_op"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYMEXLA_COMM_DIALECT_NCCL_ATTRS_TD
