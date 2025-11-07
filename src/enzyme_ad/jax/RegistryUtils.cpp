#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "src/enzyme_ad/jax/Implementations/XLADerivatives.h"

#include "Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Passes/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVM/NVVM/Target.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

#include "mlir/Transforms/Passes.h"

#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/tests/CheckOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_export.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "src/enzyme_ad/jax/TransformOps/TransformOps.h"

namespace mlir {
namespace enzyme {
void registerEnzymeJaxTransformExtension(mlir::DialectRegistry &registry);
void registerRaisingTransformExtension(mlir::DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir

using namespace mlir;

template <typename T>
struct PermuteOperandOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          PermuteOperandOpInterface<T>, T> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation *op) const {
    bool conservativePropagation = false;
    return sdy::OpShardingRuleBuilder(op)
        .addPointwiseWithDiffTypeForMismatch(
            sdy::getTensorShape(op->getOperands()[0]),
            sdy::getTensorShape(op->getResult(0)),
            sdy::FactorType::kPermutation,
            /*mismatchFactorIsBlocked*/ conservativePropagation)
        .build();
  }
};

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

namespace mlir {
namespace enzyme {

void prepareRegistry(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
    MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
    LLVM::LLVMStructType::attachInterface<
        PtrElementModel<LLVM::LLVMStructType>>(*ctx);
    LLVM::LLVMPointerType::attachInterface<
        PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
    LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
        *ctx);
  });

  registry.addExtension(
      +[](mlir::MLIRContext *ctx, enzymexla::EnzymeXLADialect *) {
        enzymexla::WrapOp::attachInterface<
            PermuteOperandOpInterface<enzymexla::WrapOp>>(*ctx);
        enzymexla::ExtendOp::attachInterface<
            PermuteOperandOpInterface<enzymexla::ExtendOp>>(*ctx);
        enzymexla::RotateOp::attachInterface<
            PermuteOperandOpInterface<enzymexla::RotateOp>>(*ctx);
      });
}

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::complex::ComplexDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::NVVM::NVVMDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::DLTIDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect>();
  registry.insert<mlir::stablehlo::check::CheckDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<mlir::nvgpu::NVGPUDialect>();
  registry.insert<mlir::transform::TransformDialect>();
  registry.insert<mlir::ub::UBDialect>();
  registry.insert<mlir::sparse_tensor::SparseTensorDialect>();
  registry.insert<mlir::enzyme::EnzymeDialect>();
  registry.insert<mlir::enzymexla::EnzymeXLADialect>();
  registry.insert<mlir::enzyme::distributed::DistributedDialect>();
  registry.insert<mlir::enzyme::tessera::TesseraDialect>();
  registry.insert<mlir::enzyme::perfify::PerfifyDialect>();
  registry.insert<mlir::sdy::SdyDialect>();
  registry.insert<mlir::ub::UBDialect>();
  registry.insert<mlir::triton::TritonDialect>();
  registry.insert<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  registry.insert<mlir::triton::gpu::TritonGPUDialect>();
}

void loadAllRegisteredDialects(mlir::MLIRContext &context) {
  context.loadDialect<mlir::affine::AffineDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::async::AsyncDialect>();
  context.loadDialect<mlir::tensor::TensorDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::cf::ControlFlowDialect>();
  context.loadDialect<mlir::complex::ComplexDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
  context.loadDialect<mlir::gpu::GPUDialect>();
  context.loadDialect<mlir::NVVM::NVVMDialect>();
  context.loadDialect<mlir::omp::OpenMPDialect>();
  context.loadDialect<mlir::math::MathDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();
  context.loadDialect<mlir::DLTIDialect>();
  context.loadDialect<mlir::mhlo::MhloDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::stablehlo::check::CheckDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::vector::VectorDialect>();
  context.loadDialect<mlir::nvgpu::NVGPUDialect>();
  context.loadDialect<mlir::transform::TransformDialect>();
  context.loadDialect<mlir::ub::UBDialect>();
  context.loadDialect<mlir::sparse_tensor::SparseTensorDialect>();
  context.loadDialect<mlir::enzyme::EnzymeDialect>();
  context.loadDialect<mlir::enzymexla::EnzymeXLADialect>();
  context.loadDialect<mlir::sdy::SdyDialect>();
  context.loadDialect<mlir::ub::UBDialect>();
  context.loadDialect<mlir::triton::TritonDialect>();
  context.loadDialect<mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  context.loadDialect<mlir::triton::gpu::TritonGPUDialect>();
}

void registerInterfaces(mlir::DialectRegistry &registry) {
  mlir::enzyme::registerXLAAutoDiffInterfaces(registry);

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);

  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerOpenMPDialectTranslation(registry);

  mlir::registerConvertOpenMPToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);

  // Register the autodiff interface implementations for upstream dialects.
  mlir::enzyme::registerCoreDialectAutodiffInterfaces(registry);

  mlir::linalg::registerTransformDialectExtension(registry);

  mlir::enzyme::registerEnzymeJaxTransformExtension(registry);
  mlir::enzyme::registerRaisingTransformExtension(registry);

  mlir::registerLLVMDialectImport(registry);
  mlir::registerNVVMDialectImport(registry);
}

void initializePasses() {
  registerenzymePasses();
  enzyme::registerenzymexlaPasses();

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerLowerAffinePass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerStripDebugInfo();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerConvertSCFToOpenMPPass();
  mlir::affine::registerAffinePasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::enzyme::registerConvertLLVMToControlFlowPass();
  mlir::enzyme::registerEnzymeLiftControlFlowToSCFPass();
  mlir::arith::registerArithPasses();

  mlir::registerSCFToControlFlowPass();

  mlir::registerGPUPasses();

  // Transform dialect and extensions.
  mlir::transform::registerInterpreterPass();
  mlir::enzyme::registerGenerateApplyPatternsPass();
  mlir::enzyme::registerRemoveTransformPass();

  // shardy passes
  xla::sdy::registerSdyRoundTripExportPipeline();
  xla::sdy::registerSdyRoundTripImportPipeline();
  mlir::sdy::registerAllSdyPassesAndPipelines();
  xla::sdy::registerStablehloExportPipeline();
  xla::sdy::registerStablehloImportPipeline();
  xla::sdy::registerStablehloImportShardingsPass();

  // SHLO / MHLO passes
  stablehlo::registerStablehloAggressiveSimplificationPass();
  stablehlo::registerStablehloAggressiveFolderPass();
  stablehlo::registerStablehloTargetIndependentOptimizationPass();
  stablehlo::registerChloLegalizeToStablehloPass();
  stablehlo::registerShapeLegalizeToStablehloPass();
  stablehlo::registerStablehloCanonicalizeDynamismPass();
  stablehlo::registerStablehloCompatibilityExpanderPass();
  stablehlo::registerStablehloComplexMathExpanderPass();
  stablehlo::registerStablehloConvertToSignlessPass();
  stablehlo::registerStablehloLegalizeCompositeToCallPass();
  stablehlo::registerStablehloLegalizeDeprecatedOpsPass();
  stablehlo::registerStablehloLegalizeQDQToQuantizedOpPass();
  stablehlo::registerStablehloLegalizeQuantizedOpToQDQPass();
  stablehlo::registerStablehloLegalizeQuantToMathPass();
  stablehlo::registerStablehloLegalizeToVhloPass();
  stablehlo::registerStablehloRefineArgumentsPass();
  stablehlo::registerStablehloRefineShapesPass();
  stablehlo::registerVhloLegalizeToStablehloPass();
  stablehlo::registerVhloToVersionPass();
  stablehlo::registerStablehloWrapInCompositePass();
  stablehlo::registerStablehloLegalizeToLinalgPass();
  mlir::tosa::registerStablehloLegalizeToTosaPass();
  mlir::tosa::registerStablehloPrepareForTosaPass();
  mlir::tosa::registerStablehloQuantLegalizeToTosaRescalePass();
  mlir::tosa::registerTosaRescaleLegalizeToStablehloPass();
  stablehlo_ext::registerPasses();
  mlir::mhlo::registerAllMhloPasses();

  // Triton passes
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::gpu::registerTritonGPUToLLVMPasses();
  mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::registerTritonToTritonGPUPasses();
  mlir::registerLLVMDIScopePass();
}

} // namespace enzyme
} // namespace mlir
