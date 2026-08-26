add_mlir_translation_library(PolymerTargetISL
  ISL/ConvertToISL.cc

  ADDITIONAL_HEADER_DIRS
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/ISL
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include

  LINK_COMPONENTS
  Core
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRSupport

  PolymerSupport
)


target_include_directories(PolymerTargetISL
  PRIVATE
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/ISL
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include
)
