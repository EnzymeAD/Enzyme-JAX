
set(POLYMER_TRANSFORM_SRC
  IslExternalTransform.cc
)

add_mlir_conversion_library(PolymerTransforms
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  FoldSCFIf.cc
  AnnotateScop.cc
  "${POLYMER_TRANSFORM_SRC}"

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  MLIRAffineToStandard
  MLIRAffineTransforms

  PolymerSupport
  "${POLYMER_AVAIL_TARGETS}"
)

target_include_directories(PolymerTransforms
  PRIVATE
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include
)
