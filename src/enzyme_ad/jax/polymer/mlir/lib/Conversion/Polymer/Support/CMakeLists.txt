add_mlir_library(PolymerSupport
  IslScop.cc
  ScopStmt.cc
  Utils.cc

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Support
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineAnalysis

  PollyISL
  Polly
)

target_link_libraries(PolymerSupport PUBLIC ${GMP_LIBRARY})

target_include_directories(PolymerSupport
  PRIVATE
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/Polymer/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include
)
