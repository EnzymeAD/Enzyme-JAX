#ifndef ENZYMEXLA_COMM_DIALECT_MPI_ATTRS_TD
#define ENZYMEXLA_COMM_DIALECT_MPI_ATTRS_TD

include "Dialect.td"

include "mlir/IR/EnumAttr.td"

def MPI_Comms : I32EnumAttr<"MPI_Comms", "Named predefined MPI communicators", [
      I32EnumAttrCase<"MPI_COMM_NULL", 256>,
      I32EnumAttrCase<"MPI_COMM_WORLD", 257>,
      I32EnumAttrCase<"MPI_COMM_SELF", 258>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::comm";
}

def MPI_CommsAttr : EnumAttr<Comm_Dialect, MPI_Comms, "communicator"> {
  let assemblyFormat = "`<` $value `>`";
}

def MPI_Datatypes : I32EnumAttr<"MPI_Datatypes", "Named predefined MPI datatypes", [
      I32EnumAttrCase<"MPI_DATATYPE_NULL", 512>,
      I32EnumAttrCase<"MPI_AINT", 513>,
      I32EnumAttrCase<"MPI_COUNT", 514>,
      I32EnumAttrCase<"MPI_OFFSET", 515>,
      I32EnumAttrCase<"MPI_PACKED", 519>,
      I32EnumAttrCase<"MPI_SHORT", 520>,
      I32EnumAttrCase<"MPI_INT", 521>,
      I32EnumAttrCase<"MPI_LONG", 522>,
      // I32EnumAttrCase<"MPI_LONG_LONG", 523>,
      I32EnumAttrCase<"MPI_LONG_LONG_INT", 523>,
      I32EnumAttrCase<"MPI_UNSIGNED_SHORT", 524>,
      I32EnumAttrCase<"MPI_UNSIGNED", 525>,
      I32EnumAttrCase<"MPI_UNSIGNED_LONG", 526>,
      I32EnumAttrCase<"MPI_UNSIGNED_LONG_LONG", 527>,
      I32EnumAttrCase<"MPI_FLOAT", 528>,
      // I32EnumAttrCase<"MPI_C_COMPLEX", 530>,
      I32EnumAttrCase<"MPI_C_FLOAT_COMPLEX", 530>,
      I32EnumAttrCase<"MPI_CXX_FLOAT_COMPLEX", 531>,
      I32EnumAttrCase<"MPI_DOUBLE", 532>,
      I32EnumAttrCase<"MPI_C_DOUBLE_COMPLEX", 534>,
      I32EnumAttrCase<"MPI_CXX_DOUBLE_COMPLEX", 535>,
      I32EnumAttrCase<"MPI_LOGICAL", 536>,
      I32EnumAttrCase<"MPI_INTEGER", 537>,
      I32EnumAttrCase<"MPI_REAL", 538>,
      I32EnumAttrCase<"MPI_COMPLEX", 539>,
      I32EnumAttrCase<"MPI_DOUBLE_PRECISION", 540>,
      I32EnumAttrCase<"MPI_DOUBLE_COMPLEX", 541>,
      I32EnumAttrCase<"MPI_CHARACTER", 542>,
      I32EnumAttrCase<"MPI_LONG_DOUBLE", 544>,
      I32EnumAttrCase<"MPI_C_LONG_DOUBLE_COMPLEX", 548>,
      I32EnumAttrCase<"MPI_CXX_LONG_DOUBLE_COMPLEX", 549>,
      I32EnumAttrCase<"MPI_FLOAT_INT", 552>,
      I32EnumAttrCase<"MPI_DOUBLE_INT", 553>,
      I32EnumAttrCase<"MPI_LONG_INT", 554>,
      I32EnumAttrCase<"MPI_2INT", 555>,
      I32EnumAttrCase<"MPI_SHORT_INT", 556>,
      I32EnumAttrCase<"MPI_LONG_DOUBLE_INT", 557>,
      I32EnumAttrCase<"MPI_2REAL", 560>,
      I32EnumAttrCase<"MPI_2DOUBLE_PRECISION", 561>,
      I32EnumAttrCase<"MPI_2INTEGER", 562>,
      I32EnumAttrCase<"MPI_C_BOOL", 568>,
      I32EnumAttrCase<"MPI_CXX_BOOL", 569>,
      I32EnumAttrCase<"MPI_WCHAR", 572>,
      I32EnumAttrCase<"MPI_INT8_T", 576>,
      I32EnumAttrCase<"MPI_UINT8_T", 577>,
      I32EnumAttrCase<"MPI_CHAR", 579>,
      I32EnumAttrCase<"MPI_SIGNED_CHAR", 580>,
      I32EnumAttrCase<"MPI_UNSIGNED_CHAR", 581>,
      I32EnumAttrCase<"MPI_BYTE", 583>,
      I32EnumAttrCase<"MPI_INT16_T", 584>,
      I32EnumAttrCase<"MPI_UINT16_T", 585>,
      I32EnumAttrCase<"MPI_INT32_T", 592>,
      I32EnumAttrCase<"MPI_UINT32_T", 593>,
      I32EnumAttrCase<"MPI_INT64_T", 600>,
      I32EnumAttrCase<"MPI_UINT64_T", 601>,
      I32EnumAttrCase<"MPI_LOGICAL1", 704>,
      I32EnumAttrCase<"MPI_INTEGER1", 705>,
      I32EnumAttrCase<"MPI_LOGICAL2", 712>,
      I32EnumAttrCase<"MPI_INTEGER2", 713>,
      I32EnumAttrCase<"MPI_REAL2", 714>,
      I32EnumAttrCase<"MPI_LOGICAL4", 720>,
      I32EnumAttrCase<"MPI_REAL4", 722>,
      I32EnumAttrCase<"MPI_INTEGER4", 721>,
      I32EnumAttrCase<"MPI_COMPLEX4", 723>,
      I32EnumAttrCase<"MPI_LOGICAL8", 728>,
      I32EnumAttrCase<"MPI_INTEGER8", 729>,
      I32EnumAttrCase<"MPI_REAL8", 730>,
      I32EnumAttrCase<"MPI_COMPLEX8", 731>,
      I32EnumAttrCase<"MPI_LOGICAL16", 736>,
      I32EnumAttrCase<"MPI_INTEGER16", 737>,
      I32EnumAttrCase<"MPI_REAL16", 738>,
      I32EnumAttrCase<"MPI_COMPLEX16", 739>,
      I32EnumAttrCase<"MPI_COMPLEX32", 747>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::comm";
}

def MPI_DatatypesAttr : EnumAttr<Comm_Dialect, MPI_Datatypes, "datatype"> {
  let assemblyFormat = "`<` $value `>`";
}

def MPI_Ops : I32EnumAttr<"MPI_Ops", "Named predefined MPI operations", [
      I32EnumAttrCase<"MPI_OP_NULL", 32>,
      I32EnumAttrCase<"MPI_SUM", 33>,
      I32EnumAttrCase<"MPI_MIN", 34>,
      I32EnumAttrCase<"MPI_MAX", 35>,
      I32EnumAttrCase<"MPI_PROD", 36>,
      I32EnumAttrCase<"MPI_BAND", 40>,
      I32EnumAttrCase<"MPI_BOR", 41>,
      I32EnumAttrCase<"MPI_BXOR", 42>,
      I32EnumAttrCase<"MPI_LAND", 48>,
      I32EnumAttrCase<"MPI_LOR", 49>,
      I32EnumAttrCase<"MPI_LXOR", 50>,
      I32EnumAttrCase<"MPI_MINLOC", 56>,
      I32EnumAttrCase<"MPI_MAXLOC", 57>,
      I32EnumAttrCase<"MPI_REPLACE", 60>,
      I32EnumAttrCase<"MPI_NO_OP", 61>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::comm";
}

def MPI_OpsAttr : EnumAttr<Comm_Dialect, MPI_Ops, "op"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYMEXLA_COMM_DIALECT_MPI_ATTRS_TD
