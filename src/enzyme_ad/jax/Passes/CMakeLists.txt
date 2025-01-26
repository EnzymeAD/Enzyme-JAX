add_flag_if_supported("-Wno-global-constructors" WNO_GLOBAL_CONSTRUCTOR_MLIR_IR)
add_mlir_conversion_library(MLIRGPULaunchToCall
  LoopDistribute.cpp
  GPULaunchToCall.cpp
  OutlineGpuJitRegions.cpp
  LLVMToMemref.cpp
  PromoteWhile.cpp
  GPUAffineOpt.cpp
  ReshapeMemrefs.cpp
  DependenceInfo.cpp
  ISLUtils.cpp
  GPULowering.cpp
  LoopUndistribute.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/GPULaunchToCall

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRPass
  PolymerTransforms
  PolymerSupport
  PolymerTargetISL
  )

target_include_directories(MLIRGPULaunchToCall
  PRIVATE
  ${POLYMER_MAIN_INCLUDE_DIR}/polymer/Target/ISL
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/include/
  ${MLIR_MAIN_INCLUDE_DIR}/../../polly/lib/External/isl/include/
  ${LLVM_BINARY_DIR}/tools/polly/lib/External/isl/include
)
