#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include "cxxbridge/deps/tensat/src/input.rs.h"
#include "rust/cxx.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace rust::cxxbridge1;

class OperationMapInfo : public llvm::DenseMapInfo<Operation *> {
public:
  static unsigned getHashValue(const Operation *val) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(val),
        // Operands, values and locations don't matter for runtime - we just
        // need the operation, attributes and types to be the same.
        OperationEquivalence::ignoreHashValue,
        OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }

  // Adapted from llvm-project/mlir/lib/Transforms/CSE.cpp
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        lhs, rhs, OperationEquivalence::ignoreValueEquivalence, nullptr,
        OperationEquivalence::IgnoreLocations);
  }

  static bool isEqual(Operation *lhs, Operation *rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        lhs, rhs, OperationEquivalence::ignoreValueEquivalence, nullptr,
        OperationEquivalence::IgnoreLocations);
  }
};

/**
 *  Which platform the cost model is using.
 */

enum EqsatPlatform {
  CPU,
  GPU,
  // TODO: TPU
};

EqsatPlatform getPlatform() {
  auto p = getenv("EQSAT_PLATFORM");
  auto platform = p ? std::string(p) : "";

  if (platform == "cpu") {
    return EqsatPlatform::CPU;
  } else if (platform == "gpu") {
    return EqsatPlatform::GPU;
  } else {
    // TODO: TPU
    auto error_string =
        "Please set environment variable EQSAT_PLATFORM=(cpu|gpu), got: " +
        platform;
    throw std::invalid_argument(error_string);
  }
}

tensorflow::se::DeviceDescription
RTXA6000DeviceInfo(tensorflow::se::GpuComputeCapability cc =
                       tensorflow::se::CudaComputeCapability(8, 9)) {
  tensorflow::se::internal::DeviceDescriptionBuilder b;
  b.set_gpu_compute_capability(cc);
  b.set_threads_per_block_limit(1024);
  b.set_threads_per_warp(32);
  b.set_shared_memory_per_block(48 * 1024);
  b.set_shared_memory_per_block_optin(99 * 1024);
  b.set_shared_memory_per_core(100 * 1024);
  b.set_threads_per_core_limit(1536);
  b.set_core_count(84);
  b.set_fpus_per_core(128);
  b.set_block_dim_limit_x(2'147'483'647);
  b.set_block_dim_limit_y(65535);
  b.set_block_dim_limit_z(65535);
  b.set_memory_bandwidth(768'096'000'000);
  b.set_l2_cache_size(6 * 1024 * 1024);
  b.set_clock_rate_ghz(1.410);
  b.set_device_memory_size(51'050'250'240);
  return b.BuildObject();
}

/**
 * Ops to not measure cost for
 */
std::set<string> zeroCostOps = {
    "stablehlo.constant",
    "stablehlo.return",

    // these ops should just be returning a view, but measuring these will
    // make a copy since we'll end up returning the result in the module
    "stablehlo.slice",
    "stablehlo.reshape",
    "stablehlo.transpose",

    // We assume the best case, where memory is allocated smartly so that the
    // concatenate never actually happens.
    "stablehlo.concatenate",
};

bool isBlackboxed(Operation *op) {
  if (isa<stablehlo::MulOp>(op) || isa<stablehlo::SubtractOp>(op) ||
      isa<stablehlo::DivOp>(op) || isa<stablehlo::AddOp>(op) ||
      isa<stablehlo::MinOp>(op) || isa<stablehlo::MaxOp>(op) ||
      isa<stablehlo::TanhOp>(op) || isa<stablehlo::NegOp>(op) ||
      isa<stablehlo::ExpOp>(op) || isa<stablehlo::TransposeOp>(op) ||
      isa<stablehlo::ReshapeOp>(op) || isa<stablehlo::DotGeneralOp>(op) ||
      isa<stablehlo::ConcatenateOp>(op) || isa<stablehlo::SliceOp>(op) ||
      isa<stablehlo::PadOp>(op) || isa<func::ReturnOp>(op)) {
    return false;
  } else {
    return true;
  }
}

class OperationTimer {
public:
  /**
   * Measure cost of operation (execution time in microseconds) by running it
   * many times and measuring the time taken.
   */
  static uint64_t getCost(Operation *op, unsigned warmup,
                          unsigned repetitions) {
    if (!logsInitialized) {
      InitializeLogs();
      logsInitialized = true;
    }

    std::string opName = op->getName().getStringRef().str();

    // TODO: Have a whitelist instead?
    if (op->getDialect()->getNamespace() != "stablehlo" ||
        zeroCostOps.find(opName) != zeroCostOps.end() || isBlackboxed(op))
      return 0;

    if (runtimeCache.contains(op)) {
      return runtimeCache[op];
    }

    auto context = OperationTimer::getContext();

    // For some reason, not cloning the op here leads to a segfault. Maybe
    // prepareExecutable/ClientCompile consumes the op?
    auto opForMeasurement = op->clone();

    ModuleOp wrapperModule =
        createModuleFromOperation(context, opForMeasurement);

    std::unique_ptr<xla::HloModule> hloModule =
        wrapperModuleToHloModule(wrapperModule);
    tensorflow::se::DeviceDescription device_info = RTXA6000DeviceInfo();
    xla::HloCostAnalysis::ShapeSizeFunction shape_size_function =
        [](const xla::Shape &shape) {
          return xla::gpu::GetSizeOfShape(shape, 4);
        };
    xla::gpu::GpuHloCostAnalysis cost_analysis(
        xla::gpu::GpuHloCostAnalysis::Options{shape_size_function, {}, true},
        device_info);

    for (auto c : hloModule->computations()) {
      std::cout << "[computation]" << std::endl;
      for (auto i : c->instructions()) {
        std::cout << "instruction: " << i->ToString();
        std::cout
            << "  ==>  "
            << xla::gpu::GpuPerformanceModel::EstimateRunTimeForInstruction(
                   i, device_info, &cost_analysis,
                   xla::gpu::GpuPerformanceModelOptions::Default())
                   .ToString();
        std::cout << std::endl;
      }
    }

    xla::PjRtClient *client = nullptr;

    auto platform = getPlatform();

    switch (platform) {
    case CPU:
      client = MakeCPUClient(0, 1, 1);
      break;
    case GPU:
      // https://github.com/EnzymeAD/Reactant.jl/blob/65060404e19cd5a56a51e4fb2b252380477632b0/src/XLA.jl#L56
      const char *error = "";
      // TODO: is this correct?
      client = MakeGPUClient(0, 1, nullptr, 0, "gpu", &error);
      if (std::string(error) != "") {
        auto error_string =
            "Error while creating GPU client: " + std::string(error);
        throw std::invalid_argument(error_string);
      }
      break;
    }

    auto executable = prepareExecutable(client, wrapperModule);

    unsigned numArgs = op->getNumOperands();
    unsigned numResults = op->getNumResults();
    uint8_t futures = 0;

    xla::PjRtBuffer *args[numArgs];
    uint8_t isArgDonatable[numArgs];

    int numRuns = warmup + repetitions;
    xla::PjRtBuffer *res[numRuns * numResults];

    for (int i = 0; i < numArgs; i++) {
      args[i] = getRandomInput(client, op->getOperand(i).getType());
      isArgDonatable[i] = false;
    }

    std::vector<uint64_t> durations(numRuns);

    for (unsigned i = 0; i < warmup + repetitions; i++) {
      durations[i] =
          XLAExecute(executable, numArgs, args, isArgDonatable, numResults,
                     res + i * numResults, &futures, nullptr);
    }

    assert(!futures);

    // TODO: This means there's no point in warmup anymore, since we're now
    // taking individual measurements. Maybe we get rid of the parameter or do
    // something more sophisticated
    uint64_t duration = *std::min_element(durations.begin(), durations.end());

    auto indexOp = op->clone();
    runtimeCache.try_emplace(indexOp, duration);

    // Cleanup
    for (int i = 0; i < numRuns * numResults; i++) {
      PjRtBufferFree(res[i]);
    }

    FreeClient(executable->client());
    ExecutableFree(executable);
    wrapperModule.erase();

    return duration;
  }

  static Operation *getDummyOp(OpBuilder &builder, Type type) {
    // Zero-initialise inputs with same operand shape
    Attribute zeroAttr = builder.getZeroAttr(type);
    OperationState zeroState(builder.getUnknownLoc(), "stablehlo.constant");
    zeroState.addTypes(type);
    zeroState.addAttribute("value", zeroAttr);

    return builder.create(zeroState);
  }

  static MLIRContext *getContext() {
    if (!context) {
      DialectRegistry registry;
      InitializeRegistryAndPasses(wrap(&registry));
      context = new MLIRContext(registry);
      RegisterDialects(wrap(context));
    }
    return context;
  }

private:
  static llvm::DenseMap<Operation *, uint64_t, OperationMapInfo> runtimeCache;
  static MLIRContext *context;
  inline static bool logsInitialized;

  /**
   * Wrap operation into a module, where its operands are mapped to inputs of
   * main. Doesn't mutate op (instead it creates a copy).
   */
  static ModuleOp createModuleFromOperation(MLIRContext *context,
                                            Operation *op) {
    // Wrap operation into a module with dummy inputs
    OpBuilder builder(context);
    Location location = builder.getUnknownLoc();
    ModuleOp wrapperModule = ModuleOp::create(location);

    auto block = wrapperModule.getBodyRegion().begin();

    // Create a func.func to wrap newOp around
    FunctionType funcType =
        FunctionType::get(context, op->getOperandTypes(), op->getResultTypes());
    func::FuncOp funcOp =
        builder.create<func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    Block *entryBlock = funcOp.addEntryBlock();

    for (int i = 0; i < op->getNumOperands(); i++) {
      op->setOperand(i, funcOp.getArgument(i));
    }

    entryBlock->push_back(op);

    auto returnOp = builder.create<func::ReturnOp>(location, op->getResults());
    entryBlock->push_back(returnOp);

    return std::move(wrapperModule);
  }

  /**
   * Wrap and compile operation into a PjRtLoadedExecutable, to be passed into
   * XLAExecute.
   */
  static xla::PjRtLoadedExecutable *prepareExecutable(xla::PjRtClient *client,
                                                      ModuleOp &wrapperModule) {
    if (failed(verify(wrapperModule))) {
      llvm::errs() << "Module verification error\n";
    }

    xla::PjRtLoadedExecutable *executable =
        ClientCompile(client, wrap(wrapperModule));

    return executable;
  }

  /**
   * Create a PjRtBuffer from a given MLIR type with random elements.
   */
  static xla::PjRtBuffer *getRandomInput(xla::PjRtClient *client,
                                         mlir::Type type) {
    // TODO: Add support for non-ranked types
    assert(isa<RankedTensorType>(type));

    auto ranked = type.cast<RankedTensorType>();
    auto elementType = ranked.getElementType();
    auto shape = ranked.getShape();

    // Calculate the number of elements in the tensor
    int numElements = 1;
    for (auto dim : shape)
      numElements *= dim;

    // Determine the element width in bytes
    auto width = (elementType.getIntOrFloatBitWidth() + 7) /
                 8; // round up to nearest byte

    // Allocate random data based on the element type
    std::vector<uint8_t> data(width * numElements);

    // Fill the data with random values based on the type
    std::random_device rd;
    std::mt19937 gen(rd());

    if (elementType.isF32()) {
      std::uniform_real_distribution<float> dist(0.0, 1.0);
      float *typedData = reinterpret_cast<float *>(data.data());
      for (int i = 0; i < numElements; ++i) {
        typedData[i] = dist(gen);
      }
    } else if (elementType.isInteger(32)) {
      std::uniform_int_distribution<int32_t> dist(0, INT32_MAX);
      int32_t *typedData = reinterpret_cast<int32_t *>(data.data());
      for (int i = 0; i < numElements; ++i) {
        typedData[i] = dist(gen);
      }
    } else {
      // TODO: Handle other element types (e.g. integers of different widths,
      // other floating point types)
      assert(false && "Element type not supported yet");
    }

    auto device = ClientGetAddressableDevice(client, 0);
    auto buffer = ArrayFromHostBuffer(client, data.data(), wrap(elementType),
                                      shape.size(), shape.data(), device);

    return buffer;
  }

  // Analytical cost model
  static std::unique_ptr<xla::HloModule>
  wrapperModuleToHloModule(ModuleOp &wrapperModule) {
    auto context = wrapperModule.getContext();
    PassManager pm(context);
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    pm.run(wrapperModule);

    MlirToHloConversionOptions options;

    auto hloModule = ConvertMlirHloToHloModule(wrapperModule, options);
    if (!hloModule.ok()) {
      llvm::errs() << "Couldn't create hloModule: "
                   << hloModule.status().message();
      return nullptr;
    } else {
      return std::move(hloModule.value());
    }
  }
};

llvm::DenseMap<Operation *, uint64_t, OperationMapInfo>
    OperationTimer::runtimeCache;
MLIRContext *OperationTimer::context = nullptr;

/**
 * Create a new mlir::RankedTensorType based on the type of an existing
 * mlir::Value and the provided shape.
 */
mlir::RankedTensorType deriveOutputType(mlir::Value &input,
                                        llvm::ArrayRef<int64_t> shape) {
  auto inputType = input.getType();
  assert(isa<RankedTensorType>(inputType));
  auto ranked = inputType.cast<RankedTensorType>();
  RankedTensorType::Builder builder(ranked);
  return builder.setShape(shape);
}

llvm::ArrayRef<int64_t> getShape(mlir::Value &input) {
  auto inputType = input.getType();
  assert(isa<RankedTensorType>(inputType));
  auto ranked = inputType.cast<RankedTensorType>();
  return ranked.getShape();
}

/**
 * https://github.com/google/jax/blob/c08656c61d0e2460f5d902b0af808b74c76a48ca/jax/_src/lax/lax.py#L2729
 */
std::vector<int64_t> vectorExcept(llvm::ArrayRef<int64_t> &vec,
                                  std::vector<int64_t> &indices) {
  std::vector<int64_t> result;
  for (int i = 0; i < vec.size(); i++) {
    if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
      result.push_back(vec[i]);
    }
  }
  return result;
}

/**
 * https://github.com/google/jax/blob/c08656c61d0e2460f5d902b0af808b74c76a48ca/jax/_src/lax/lax.py#L2720C5-L2720C35
 */
std::vector<int64_t> dotGeneralShapeComputation(
    llvm::ArrayRef<int64_t> &lhs_shape, llvm::ArrayRef<int64_t> &rhs_shape,
    std::vector<int64_t> &lhs_batch, std::vector<int64_t> &rhs_batch,
    std::vector<int64_t> &lhs_contracting,
    std::vector<int64_t> &rhs_contracting) {
  std::vector<int64_t> batch_shape;
  for (auto i : lhs_batch)
    batch_shape.push_back(lhs_shape[i]);

  std::vector<int64_t> lhs_contract_or_batch = lhs_contracting;
  lhs_contract_or_batch.insert(lhs_contract_or_batch.end(), lhs_batch.begin(),
                               lhs_batch.end());
  std::sort(lhs_contract_or_batch.begin(), lhs_contract_or_batch.end());

  auto lhs_tensored_shape = vectorExcept(lhs_shape, lhs_contract_or_batch);

  std::vector<int64_t> rhs_contract_or_batch = rhs_contracting;
  rhs_contract_or_batch.insert(rhs_contract_or_batch.end(), rhs_batch.begin(),
                               rhs_batch.end());
  std::sort(rhs_contract_or_batch.begin(), rhs_contract_or_batch.end());

  auto rhs_tensored_shape = vectorExcept(rhs_shape, rhs_contract_or_batch);

  auto shape = batch_shape;
  shape.insert(shape.end(), lhs_tensored_shape.begin(),
               lhs_tensored_shape.end());
  shape.insert(shape.end(), rhs_tensored_shape.begin(),
               rhs_tensored_shape.end());

  return shape;
}

/**
 * Get the correct start and limiting indices of a SliceOp from a SSplit0 or
 * SSplit1.
 *
 * See comment in tensat/src/model.rs for details.
 */
std::pair<std::vector<int64_t>, std::vector<int64_t>>
getSliceIndicesFromSplit(tensat::Ops op, Value input, int axis, Value orig) {
  auto input_shape = getShape(input);
  std::vector<int64_t> start, limit;
  for (int i = 0; i < input_shape.size(); i++) {
    if (i != axis) {
      start.push_back(0);
      limit.push_back(input_shape[i]);
    } else {
      int slice_width = getShape(orig)[axis];
      if (op == tensat::Ops::SSplit0) {
        start.push_back(0);
        limit.push_back(slice_width);
      } else if (op == tensat::Ops::SSplit1) {
        int input_width = input_shape[axis];
        start.push_back(input_width - slice_width);
        limit.push_back(input_width);
      } else {
        throw std::invalid_argument("op should be either SSplit0 or SSplit1");
      }
    }
  }
  return {start, limit};
}

/**
 * Get the Reshape output type for MatchRank.
 *
 * See comment in tensat/src/model.rs for details.
 */
Type getReshapeTypeForMatchRank(Value input, Value ref) {
  auto initialShape = getShape(input);
  auto refRank = getShape(ref).size();
  int initialRank = initialShape.size();
  SmallVector<int64_t> shape;
  if (initialRank < refRank) {
    for (int i = 0; i < refRank; i++) {
      if (i < initialRank)
        shape.push_back(initialShape[i]);
      else
        shape.push_back(1);
    }
  } else {
    for (int i = 0; i < initialRank; i++) {
      if (i < refRank)
        shape.push_back(initialShape[i]);
      else
        assert(initialShape[i] == 1);
    }
  }
  return deriveOutputType(input, shape);
}

Operation *createStableHloOp(OpBuilder &builder, tensat::Ops op,
                             SmallVector<Value> &operands,
                             std::vector<std::vector<int64_t>> &other_vecs,
                             std::vector<int64_t> &int_args,
                             MLIRContext *context) {
  Operation *mlirOp = nullptr;

  switch (op) {
  case tensat::Ops::AddOp:
    mlirOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(),
                                              operands[0], operands[1]);
    break;
  case tensat::Ops::MulOp:
    mlirOp = builder.create<stablehlo::MulOp>(builder.getUnknownLoc(),
                                              operands[0], operands[1]);
    break;
  case tensat::Ops::DivOp:
    mlirOp = builder.create<stablehlo::DivOp>(builder.getUnknownLoc(),
                                              operands[0], operands[1]);
    break;
  case tensat::Ops::SubtractOp:
    mlirOp = builder.create<stablehlo::SubtractOp>(builder.getUnknownLoc(),
                                                   operands[0], operands[1]);
    break;
  case tensat::Ops::NegOp:
    mlirOp =
        builder.create<stablehlo::NegOp>(builder.getUnknownLoc(), operands[0]);
    break;
  case tensat::Ops::TanhOp:
    mlirOp =
        builder.create<stablehlo::TanhOp>(builder.getUnknownLoc(), operands[0]);
    break;
  case tensat::Ops::ExpOp:
    mlirOp =
        builder.create<stablehlo::ExpOp>(builder.getUnknownLoc(), operands[0]);
    break;
  case tensat::Ops::MinOp:
    mlirOp = builder.create<stablehlo::MinOp>(builder.getUnknownLoc(),
                                              operands[0], operands[1]);
    break;
  case tensat::Ops::MaxOp:
    mlirOp = builder.create<stablehlo::MaxOp>(builder.getUnknownLoc(),
                                              operands[0], operands[1]);
    break;
  case tensat::Ops::TransposeOp:
    mlirOp = builder.create<stablehlo::TransposeOp>(builder.getUnknownLoc(),
                                                    operands[0], other_vecs[0]);
    break;
  case tensat::Ops::ReshapeOp:
    mlirOp = builder.create<stablehlo::ReshapeOp>(
        builder.getUnknownLoc(), deriveOutputType(operands[0], other_vecs[0]),
        operands[0]);
    break;
  case tensat::Ops::MatchRank: {
    auto input = operands[0];
    auto ref = operands[1];
    auto newType = getReshapeTypeForMatchRank(input, ref);
    mlirOp = builder.create<stablehlo::ReshapeOp>(builder.getUnknownLoc(),
                                                  newType, input);
    break;
  }
  case tensat::Ops::DotGeneralOp: {
    std::vector<int64_t> lhs_batch_dim = other_vecs[0];
    std::vector<int64_t> rhs_batch_dim = other_vecs[1];
    std::vector<int64_t> lhs_contract_dim = other_vecs[2];
    std::vector<int64_t> rhs_contract_dim = other_vecs[3];
    std::vector<int64_t> precision_config = other_vecs[4];
    auto lhs_shape = getShape(operands[0]);
    auto rhs_shape = getShape(operands[1]);
    std::vector<int64_t> shape = dotGeneralShapeComputation(
        lhs_shape, rhs_shape, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim,
        rhs_contract_dim);

    std::vector<Attribute> precisionVec;

    for (auto &precision : precision_config) {
      switch (precision) {
      case 0:
        precisionVec.push_back(stablehlo::PrecisionAttr::get(
            context, stablehlo::Precision::DEFAULT));
        break;
      case 1:
        precisionVec.push_back(
            stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGH));
        break;
      case 2:
        precisionVec.push_back(stablehlo::PrecisionAttr::get(
            context, stablehlo::Precision::HIGHEST));
        break;
      }
    }

    mlirOp = builder.create<stablehlo::DotGeneralOp>(
        builder.getUnknownLoc(), deriveOutputType(operands[1], shape),
        operands[0], operands[1],
        stablehlo::DotDimensionNumbersAttr::get(context, lhs_batch_dim,
                                                rhs_batch_dim, lhs_contract_dim,
                                                rhs_contract_dim),
        mlir::ArrayAttr::get(context, llvm::ArrayRef(precisionVec)), nullptr);
    break;
  }
  case tensat::Ops::SliceOp:
    mlirOp = builder.create<stablehlo::SliceOp>(builder.getUnknownLoc(),
                                                operands[0], other_vecs[0],
                                                other_vecs[1], other_vecs[2]);
    break;
  case tensat::Ops::SSplit0:
  case tensat::Ops::SSplit1: {
    auto [startIndices, limitIndices] =
        getSliceIndicesFromSplit(op, operands[0], int_args[0], operands[1]);
    std::vector<int64_t> strides(getShape(operands[0]).size(), 1);
    mlirOp =
        builder.create<stablehlo::SliceOp>(builder.getUnknownLoc(), operands[0],
                                           startIndices, limitIndices, strides);
    break;
  }
  case tensat::Ops::ConcatenateOp:
    mlirOp = builder.create<stablehlo::ConcatenateOp>(builder.getUnknownLoc(),
                                                      operands, int_args[0]);
    break;
  case tensat::Ops::PadOp:
    mlirOp = builder.create<stablehlo::PadOp>(
        builder.getUnknownLoc(), operands[0], operands[1], other_vecs[0],
        other_vecs[1], other_vecs[2]);
    break;
  case tensat::Ops::IotaOp:
    mlirOp = builder.create<stablehlo::IotaOp>(
        builder.getUnknownLoc(),
        RankedTensorType::get(llvm::ArrayRef(other_vecs[0]),
                              builder.getF32Type()),
        int_args[0]);
    break;
  default:
    std::cout << "EGRAPH INVALID, UNSUPPORTED OP SHAPE REQUESTED" << "\n";
    assert(false);
    return nullptr;
  }

  return mlirOp;
}

// TODO: Avoid creating dummy inputs (we need them again for cost measurement,
// so duplicated)
uint64_t tensat::get_cost(tensat::Ops op, rust::Vec<tensat::Tensor> enode_args,
                          rust::Vec<tensat::Vector> other_vector_args,
                          rust::Vec<int64_t> int_args) {
  auto context = OperationTimer::getContext();
  OpBuilder builder(context);

  // Create operands and other args
  SmallVector<Value> operands;
  for (const auto &tensor : enode_args) {
    auto tensor_type = tensat::newTensorType(builder, tensor);
    operands.push_back(
        OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto &vec : other_vector_args)
    other_vecs.push_back(std::vector<int64_t>(vec.vec.begin(), vec.vec.end()));

  std::vector<int64_t> int_args_as_vec;
  for (const auto &num : int_args)
    int_args_as_vec.push_back(num);

  // Create the MLIR operation
  Operation *mlirOp = createStableHloOp(builder, op, operands, other_vecs,
                                        int_args_as_vec, context);

  int repeats = 0;
  switch (getPlatform()) {
  case CPU:
    repeats = 200;
    break;
  case GPU:
    // TODO: Review this number
    repeats = 30;
    break;
  default:
    assert(false);
  }

  if (mlirOp) {
    auto cost = OperationTimer::getCost(mlirOp, repeats, repeats);
    mlirOp->erase();
    return cost;
  }
  return 100000;
}

mlir::Type tensat::newTensorType(OpBuilder &builder, tensat::Tensor tensor) {
  auto dimsRef = llvm::ArrayRef(tensor.shape.data(), tensor.shape.size());
  auto mlirType = tensat::tensatTypeToMlirType(builder, tensor.element_type);
  return RankedTensorType::get(dimsRef, mlirType);
}

mlir::Type tensat::tensatTypeToMlirType(OpBuilder &builder, tensat::Type type) {
  switch (type) {
  case tensat::Type::i32:
    return builder.getI32Type();
  case tensat::Type::f32:
    return builder.getF32Type();
  default:
    assert(false);
  }
}

tensat::Type mlirTypeToTensatType(mlir::Type type) {
  if (type.isInteger(32)) {
    return tensat::Type::i32;
  } else if (type.isF32()) {
    return tensat::Type::f32;
  } else {
    llvm_unreachable("Unsupported MLIR type");
  }
}

tensat::Tensor mlirValueToTensatTensor(mlir::Value value) {
  auto output_tensor = value.getType().cast<TensorType>();
  auto shape_data = output_tensor.getShape();
  rust::Vec<int64_t> shape;
  for (const auto dim : shape_data) {
    shape.push_back(dim);
  }
  auto element_type = mlirTypeToTensatType(output_tensor.getElementType());
  return {shape, element_type};
}

std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (int64_t dim : shape) {
    dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

rust::Vec<int64_t> castArrayRefToRustVec(llvm::ArrayRef<int64_t> vec) {
  rust::Vec<int64_t> res;
  res.reserve(vec.size());
  for (const auto &elem : vec) {
    res.push_back(elem);
  }
  return res;
}

// SHAPE INFERENCE
rust::Vec<tensat::Tensor>
tensat::get_shape(Ops op, rust::Vec<tensat::Tensor> enode_args,
                  rust::Vec<tensat::Vector> other_vector_args,
                  rust::Vec<int64_t> int_args) {
  auto context = OperationTimer::getContext();
  OpBuilder builder(context);

  // Create operands and other args
  SmallVector<Value> operands;
  for (const auto &tensor : enode_args) {
    auto tensor_type = newTensorType(builder, tensor);
    operands.push_back(
        OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto &vec : other_vector_args)
    other_vecs.push_back(std::vector<int64_t>(vec.vec.begin(), vec.vec.end()));

  std::vector<int64_t> int_args_as_vec;
  for (const auto &num : int_args)
    int_args_as_vec.push_back(num);

  // Create the MLIR operation
  Operation *mlirOp = createStableHloOp(builder, op, operands, other_vecs,
                                        int_args_as_vec, context);
  if (mlirOp) {
    rust::Vec<tensat::Tensor> tensors;
    for (auto res : mlirOp->getResults()) {
      tensors.push_back(mlirValueToTensatTensor(res));
    }
    mlirOp->erase();
    return tensors;
  }
  return {};
}

namespace {
class EqualitySaturationPass
    : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "equality-saturation-pass"; }
  StringRef getDescription() const override {
    return "Optimizes HLO graph using a Rust-based optimizer";
  }

  int getValueIndex(Operation *definingOp, Value &value) {
    auto results = definingOp->getResults();
    for (int i = 0; i < results.size(); i++) {
      if (results[i] == value)
        return i;
    }
    return -1;
  }

  tensat::TensorInfo *handleOperand(
      Value &operand,
      std::unordered_map<Operation *, tensat::TensorInfo *> *opToTensorInfo,
      std::unordered_map<int, tensat::TensorInfo *> *blockArgToTensorInfo,
      std::vector<Operation *> *blackboxIDToTensorInfo,
      std::unordered_map<int, std::vector<Value>> *blackboxIDToCapturedValues,
      OpBuilder &builder, Box<tensat::CppGraphConverter> &graph) {
    if (auto defOp = operand.getDefiningOp()) {
      // Use existing TensorInfo if already processed
      auto convertedOperand = dfs(defOp, opToTensorInfo, blockArgToTensorInfo,
                                  blackboxIDToTensorInfo,
                                  blackboxIDToCapturedValues, builder, graph);
      int index = getValueIndex(defOp, operand);
      assert(index >= 0);
      if (index == 0) {
        return convertedOperand;
      } else {
        auto indexOperand =
            graph->new_index(index, *convertedOperand).into_raw();
        return indexOperand;
      }
    } else if (auto arg = operand.dyn_cast<BlockArgument>()) {
      // Handle BlockArguments which represent function parameters
      if (isa<TensorType>(operand.getType())) {
        int32_t block_arg_number = arg.getArgNumber();
        auto &tensorInfo = (*blockArgToTensorInfo)[block_arg_number];
        if (!tensorInfo) {
          tensorInfo = graph
                           ->new_input(block_arg_number,
                                       mlirValueToTensatTensor(operand))
                           .into_raw();
          (*blockArgToTensorInfo)[block_arg_number] = tensorInfo;
        }
        return tensorInfo;
      } else {
        std::cout
            << "EqualitySaturationPass does not support this argument type!"
            << "\n";
        operand.getType().dump();
        assert(false);
      }
    }
    std::cout << "EqualitySaturationPass: encountered operand that is neither "
                 "the result of an Op nor a BlockArgument."
              << "\n";
    assert(false);
  }

  tensat::TensorInfo *
  dfs(Operation *op,
      std::unordered_map<Operation *, tensat::TensorInfo *> *opToTensorInfo,
      std::unordered_map<int, tensat::TensorInfo *> *blockArgToTensorInfo,
      std::vector<Operation *> *blackboxIDToTensorInfo,
      std::unordered_map<int, std::vector<Value>> *blackboxIDToCapturedValues,
      OpBuilder &builder, Box<tensat::CppGraphConverter> &graph) {
    // std::cout << "DFS AT " << op->getName().getStringRef().str() << "\n";
    if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
      return opToTensorInfo->at(op);
    }
    tensat::TensorInfo *tensorInfo = nullptr;
    auto handleOperandPartial = [&](auto operand) {
      return handleOperand(operand, opToTensorInfo, blockArgToTensorInfo,
                           blackboxIDToTensorInfo, blackboxIDToCapturedValues,
                           builder, graph);
    };

    bool blackboxed = false;

    if (isa<stablehlo::MulOp>(op)) {
      auto mul = cast<stablehlo::MulOp>(op);
      tensorInfo = graph
                       ->new_mul_op(*handleOperandPartial(mul.getLhs()),
                                    *handleOperandPartial(mul.getRhs()),
                                    mlirValueToTensatTensor(mul->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::SubtractOp>(op)) {
      auto subtract = cast<stablehlo::SubtractOp>(op);
      tensorInfo =
          graph
              ->new_subtract_op(*handleOperandPartial(subtract.getLhs()),
                                *handleOperandPartial(subtract.getRhs()),
                                mlirValueToTensatTensor(subtract->getResult(0)))
              .into_raw();
    } else if (isa<stablehlo::DivOp>(op)) {
      auto div = cast<stablehlo::DivOp>(op);
      auto shape = tensorInfo =
          graph
              ->new_div_op(*handleOperandPartial(div.getLhs()),
                           *handleOperandPartial(div.getRhs()),
                           mlirValueToTensatTensor(div->getResult(0)))
              .into_raw();
    } else if (isa<stablehlo::AddOp>(op)) {
      auto add = cast<stablehlo::AddOp>(op);
      tensorInfo = graph
                       ->new_add_op(*handleOperandPartial(add.getLhs()),
                                    *handleOperandPartial(add.getRhs()),
                                    mlirValueToTensatTensor(add->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::MinOp>(op)) {
      auto min = cast<stablehlo::MinOp>(op);
      tensorInfo = graph
                       ->new_min_op(*handleOperandPartial(min.getLhs()),
                                    *handleOperandPartial(min.getRhs()),
                                    mlirValueToTensatTensor(min->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::MaxOp>(op)) {
      auto max = cast<stablehlo::MaxOp>(op);
      tensorInfo = graph
                       ->new_max_op(*handleOperandPartial(max.getLhs()),
                                    *handleOperandPartial(max.getRhs()),
                                    mlirValueToTensatTensor(max->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::TanhOp>(op)) {
      auto tanh = cast<stablehlo::TanhOp>(op);
      tensorInfo =
          graph
              ->new_tanh_op(*handleOperandPartial(tanh.getOperand()),
                            mlirValueToTensatTensor(tanh->getResult(0)))
              .into_raw();
    } else if (isa<stablehlo::NegOp>(op)) {
      auto neg = cast<stablehlo::NegOp>(op);
      tensorInfo = graph
                       ->new_neg_op(*handleOperandPartial(neg.getOperand()),
                                    mlirValueToTensatTensor(neg->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::ExpOp>(op)) {
      auto exp = cast<stablehlo::ExpOp>(op);
      tensorInfo = graph
                       ->new_exp_op(*handleOperandPartial(exp.getOperand()),
                                    mlirValueToTensatTensor(exp->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::TransposeOp>(op)) {
      auto transpose = cast<stablehlo::TransposeOp>(op);
      tensorInfo = graph
                       ->new_transpose_op(
                           *handleOperandPartial(transpose.getOperand()),
                           castArrayRefToRustVec(transpose.getPermutation()),
                           mlirValueToTensatTensor(transpose->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::ReshapeOp>(op)) {
      auto reshape = cast<stablehlo::ReshapeOp>(op);
      if (auto output_tensor =
              reshape.getResult().getType().cast<TensorType>()) {
        tensorInfo =
            graph
                ->new_reshape_op(*handleOperandPartial(reshape.getOperand()),
                                 mlirValueToTensatTensor(reshape->getResult(0)))
                .into_raw();
      } else {
        std::cout << "EqualitySaturationPass: result of stablehlo::ReshapeOp "
                     "has non-tensor type"
                  << std::endl;
      }
      // } else if (isa<stablehlo::IotaOp>(op)) {
      //   auto iota = cast<stablehlo::IotaOp>(op);
      //   int32_t iota_dimension = iota.getIotaDimension();
      //   if (auto output_tensor =
      //   iota.getResult().getType().cast<TensorType>()) {
      //     tensorInfo =
      //         graph
      //             ->new_iota_op(iota_dimension,
      //                           mlirValueToTensatTensor(iota.getResult()))
      //             .into_raw();
      //   } else {
      //     std::cout << "EqualitySaturationPass: result of stablehlo::IotaOp
      //     has "
      //                  "non-tensor type"
      //               << std::endl;
      //   }
    } else if (isa<stablehlo::DotGeneralOp>(op)) {
      // we might need more guards here
      auto dot_general = cast<stablehlo::DotGeneralOp>(op);
      auto dot_dim_attrs = dot_general.getDotDimensionNumbersAttr();

      mlir::ArrayAttr precision =
          dot_general.getPrecisionConfig().value_or(mlir::ArrayAttr());
      rust::Vec<int64_t> precision_configs;
      for (int i = 0; i < precision.size(); i++) {
        auto precisionAttr =
            precision[i].dyn_cast<mlir::stablehlo::PrecisionAttr>();
        if (!precisionAttr)
          continue; // Skip if it's not a PrecisionAttr, although such
                    // attributes should not exist here
        mlir::stablehlo::Precision val = precisionAttr.getValue();
        switch (val) {
        case mlir::stablehlo::Precision::DEFAULT:
          precision_configs.push_back(0);
          break;
        case mlir::stablehlo::Precision::HIGH:
          precision_configs.push_back(1);
          break;
        case mlir::stablehlo::Precision::HIGHEST:
          precision_configs.push_back(2);
          break;
        }
      }

      if (auto output_tensor =
              dot_general.getResult().getType().cast<TensorType>()) {
        auto shape = castArrayRefToInt32(output_tensor.getShape());
        auto output_shape_slice =
            rust::Slice<const int>{shape.data(), shape.size()};

        tensorInfo = graph
                         ->new_dot_general_op(
                             *handleOperandPartial(dot_general.getLhs()),
                             *handleOperandPartial(dot_general.getRhs()),
                             castArrayRefToRustVec(
                                 dot_dim_attrs.getLhsBatchingDimensions()),
                             castArrayRefToRustVec(
                                 dot_dim_attrs.getRhsBatchingDimensions()),
                             castArrayRefToRustVec(
                                 dot_dim_attrs.getLhsContractingDimensions()),
                             castArrayRefToRustVec(
                                 dot_dim_attrs.getRhsContractingDimensions()),
                             precision_configs,
                             mlirValueToTensatTensor(dot_general.getResult()))
                         .into_raw();
      } else {
        std::cout << "EqualitySaturationPass: result of "
                     "stablehlo::DotGeneralOp has non-tensor type"
                  << std::endl;
      }
    } else if (isa<stablehlo::ConcatenateOp>(op)) {
      auto concat = cast<stablehlo::ConcatenateOp>(op);
      auto output_tensor = concat->getResult(0).getType().cast<TensorType>();
      auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
      std::vector<tensat::TensorInfo *> inputs;
      for (auto input : concat.getInputs()) {
        inputs.push_back(handleOperandPartial(input));
      }
      int32_t dimension = concat.getDimension();
      tensorInfo = graph
                       ->new_concatenate_op(
                           {inputs.data(), inputs.size()}, dimension,
                           mlirValueToTensatTensor(concat->getResult(0)))
                       .into_raw();
    } else if (isa<stablehlo::SliceOp>(op)) {
      auto slice = cast<stablehlo::SliceOp>(op);
      auto operand = handleOperandPartial(slice.getOperand());
      tensorInfo =
          graph
              ->new_slice_op(*operand,
                             castArrayRefToRustVec(slice.getStartIndices()),
                             castArrayRefToRustVec(slice.getLimitIndices()),
                             castArrayRefToRustVec(slice.getStrides()),
                             mlirValueToTensatTensor(slice->getResult(0)))
              .into_raw();
    } else if (isa<stablehlo::PadOp>(op)) {
      auto pad = cast<stablehlo::PadOp>(op);
      auto operand = handleOperandPartial(pad.getOperand());
      auto padding_value = handleOperandPartial(pad.getPaddingValue());
      tensorInfo =
          graph
              ->new_pad_op(*operand, *padding_value,
                           castArrayRefToRustVec(pad.getEdgePaddingLow()),
                           castArrayRefToRustVec(pad.getEdgePaddingHigh()),
                           castArrayRefToRustVec(pad.getInteriorPadding()),
                           mlirValueToTensatTensor(pad->getResult(0)))
              .into_raw();
    } else if (isa<func::ReturnOp>(op)) {
      int numOperands = op->getNumOperands();
      std::vector<tensat::TensorInfo *> processedOperands;
      for (size_t i = 0; i < numOperands; i++) {
        auto operand = handleOperandPartial(op->getOperand(i));
        processedOperands.push_back(operand);
      }
      auto operandPtrsSlice = rust::Slice<tensat::TensorInfo *const>{
          processedOperands.data(),
          static_cast<size_t>(processedOperands.size())};
      tensorInfo = graph->new_return_op(operandPtrsSlice).into_raw();
    } else {
      blackboxed = true;

      int numOperands = op->getNumOperands();
      rust::Vec<tensat::Tensor> outputs;
      for (auto result : op->getResults())
        outputs.push_back(mlirValueToTensatTensor(result));

      std::vector<tensat::TensorInfo *> processedOperands;
      // We shouldn't clone operands, as those values will be invalidated
      // after a block.clear(), and we access these during reconstruction.
      auto copy = op->clone(Operation::CloneOptions(
          /* cloneRegions = */ true, /* cloneOperands = */ false));
      blackboxIDToTensorInfo->push_back(copy);
      int blackboxOpID = blackboxIDToTensorInfo->size() - 1;
      for (size_t i = 0; i < numOperands; i++) {
        auto operand = handleOperandPartial(op->getOperand(i));
        processedOperands.push_back(operand);
      }
      auto operandPtrsSlice = rust::Slice<tensat::TensorInfo *const>{
          processedOperands.data(), processedOperands.size()};

      std::vector<Value> capturedValues;
      std::vector<tensat::TensorInfo *> capturedTensorInfos;

      // Walk the operation, if any operands were originated from the current
      // block (rather than within the operation) then we should capture them
      auto outerBlock = op->getBlock();
      copy->walk([&](Operation *op) {
        if (op == copy)
          return;
        for (Value operand : op->getOperands()) {
          if (operand.getDefiningOp() != nullptr // block argument
              && operand.getDefiningOp()->getBlock() == outerBlock) {
            capturedValues.push_back(operand);
            capturedTensorInfos.push_back(handleOperandPartial(operand));
          }
        }
      });

      assert(blackboxIDToCapturedValues->find(blackboxOpID) ==
             blackboxIDToCapturedValues->end());
      (*blackboxIDToCapturedValues)[blackboxOpID] = capturedValues;

      auto capturedTensorInfosSlice = rust::Slice<tensat::TensorInfo *const>{
          capturedTensorInfos.data(), capturedTensorInfos.size()};

      tensorInfo =
          graph
              ->new_blackbox_op(operandPtrsSlice, capturedTensorInfosSlice,
                                blackboxOpID, outputs)
              .into_raw();
    }
    assert(tensorInfo != nullptr);
    // Check that isBlackboxed is up-to-date
    assert(blackboxed == isBlackboxed(op));

    opToTensorInfo->insert({op, tensorInfo});
    return tensorInfo;
  }

  Box<tensat::CppGraphConverter> createEgraph(
      std::vector<Operation *> *blackboxIDToTensorInfo,
      std::unordered_map<int, std::vector<Value>> *blackboxIDToCapturedValues,
      OpBuilder &builder, ModuleOp module) {

    auto graph = tensat::new_converter();
    // members of the class
    std::unordered_map<Operation *, tensat::TensorInfo *> opToTensorInfo;
    std::unordered_map<int, tensat::TensorInfo *> blockArgToTensorInfo;

    module.walk([&](func::ReturnOp op) {
      dfs(op, &opToTensorInfo, &blockArgToTensorInfo, blackboxIDToTensorInfo,
          blackboxIDToCapturedValues, builder, graph);
    });

    // graph->print_rec_expr();
    return graph;
  }

  template <typename T>
  Operation *createUnaryOp(OpBuilder &builder, std::vector<Value> &opVals,
                           tensat::Node &node) {
    auto location = builder.getUnknownLoc();
    return builder.create<T>(location, opVals[node.operands[0]]);
  }

  template <typename T>
  Operation *createBinaryOp(OpBuilder &builder, std::vector<Value> &opVals,
                            tensat::Node &node) {
    auto location = builder.getUnknownLoc();
    return builder.create<T>(location, opVals[node.operands[0]],
                             opVals[node.operands[1]]);
  }

  /**
   * Parse the Vec nodes with Nums (e.g Vec(Num(128), Num(128))) emitted by
   * tensat node construction.
   */
  std::vector<int64_t> parseNumVec(rust::vec<tensat::Node> &nodes,
                                   tensat::Node &seq) {
    assert(seq.name == "Vec");
    std::vector<int64_t> result;

    for (auto i : seq.operands) {
      assert(i < nodes.size());
      assert(nodes[i].name == "Num");
      result.push_back(parseNumNode(nodes, nodes[i]));
    }

    return result;
  }

  /**
   * Parse the Num nodes emitted by tensat node construction.
   * Our protocol is to encode integer values as operand indices.
   * TODO: improve this!
   */
  int64_t parseNumNode(rust::vec<tensat::Node> &nodes, tensat::Node &seq) {
    assert(seq.name == "Num");
    return seq.operands[0];
  }

  /**
   * Parse the Vec nodes with arbitrary operations (e.g Vec(Input(...),
   * AddOp(...))) emitted by tensat node construction.
   */
  std::vector<Value> parseOpVec(std::vector<Value> &opVals, tensat::Node &seq) {
    assert(seq.name == "Vec");
    std::vector<Value> result;

    for (auto i : seq.operands) {
      assert(opVals[i] != nullptr);
      result.push_back(opVals[i]);
    }

    return result;
  }

  void reconstructStablehlo(
      ModuleOp *root, std::vector<Operation *> *blackboxIDToTensorInfo,
      std::unordered_map<int, std::vector<Value>> *blackboxIDToCapturedValues,
      rust::vec<tensat::Node> &nodes, OpBuilder &builder) {
    auto context = root->getContext();
    std::vector<Value> opVals;

    // Find funcOp to get the block.
    func::FuncOp funcOp;

    for (auto &op : root->getBody()->getOperations()) {
      if (isa<func::FuncOp>(op)) {
        funcOp = cast<func::FuncOp>(op);
        break;
      }
    }

    auto &region = funcOp.getRegion();
    auto &block = funcOp.getRegion().front();

    // We don't clear the block here straight away, because this will
    // invalidate the old captured values in blackbox, and so the substitution
    // won't work. We instead put everything in a vector, and only clear the
    // block and insert everything at the very end.
    std::vector<Operation *> opsToAdd;

    auto location = builder.getUnknownLoc();

    for (auto &node : nodes) {
      Operation *newOp = nullptr;
      // Create the new operation based on the operands
      using namespace tensat;
      switch (node.op) {
      case Ops::Var:
      case Ops::Num:
      case Ops::Vec:
        /* do nothing */
        break;
      case Ops::Input: {
        int blockArgNumber = parseNumNode(nodes, nodes[node.operands[1]]);
        opVals.push_back(block.getArgument(blockArgNumber));
        continue;
      }
      case Ops::Index: {
        int index = parseNumNode(nodes, nodes[node.operands[0]]);
        int input = node.operands[1];
        opVals.push_back(opVals[input].getDefiningOp()->getResult(index));
        continue;
      }
      case Ops::NegOp:
        newOp = createUnaryOp<stablehlo::NegOp>(builder, opVals, node);
        break;
      case Ops::TanhOp:
        newOp = createUnaryOp<stablehlo::TanhOp>(builder, opVals, node);
        break;
      case Ops::ExpOp:
        newOp = createUnaryOp<stablehlo::ExpOp>(builder, opVals, node);
        break;
      case Ops::AddOp:
        newOp = createBinaryOp<stablehlo::AddOp>(builder, opVals, node);
        break;
      case Ops::SubtractOp:
        newOp = createBinaryOp<stablehlo::SubtractOp>(builder, opVals, node);
        break;
      case Ops::MulOp:
        newOp = createBinaryOp<stablehlo::MulOp>(builder, opVals, node);
        break;
      case Ops::DivOp:
        newOp = createBinaryOp<stablehlo::DivOp>(builder, opVals, node);
        break;
      case Ops::MinOp:
        newOp = createBinaryOp<stablehlo::MinOp>(builder, opVals, node);
        break;
      case Ops::MaxOp:
        newOp = createBinaryOp<stablehlo::MaxOp>(builder, opVals, node);
        break;
      case Ops::TransposeOp: {
        auto input = opVals[node.operands[0]];
        auto permutation = parseNumVec(nodes, nodes[node.operands[1]]);
        newOp = builder.create<stablehlo::TransposeOp>(location, input,
                                                       permutation);
        break;
      }
      case Ops::ReshapeOp: {
        auto input = opVals[node.operands[0]];
        auto shape = parseNumVec(nodes, nodes[node.operands[1]]);
        auto newType = deriveOutputType(input, shape);
        newOp = builder.create<stablehlo::ReshapeOp>(location, newType, input);
        break;
      }
      case Ops::MatchRank: {
        auto input = opVals[node.operands[0]];
        auto ref = opVals[node.operands[1]];
        if (getShape(input).size() == getShape(ref).size()) {
          opVals.push_back(input);
          continue;
        } else {
          auto newType = getReshapeTypeForMatchRank(input, ref);
          newOp =
              builder.create<stablehlo::ReshapeOp>(location, newType, input);
        }
        break;
      }
      case Ops::DotGeneralOp: {
        auto lhs = opVals[node.operands[0]];
        auto rhs = opVals[node.operands[1]];

        auto lhsBatchDim = parseNumVec(nodes, nodes[node.operands[2]]);
        auto rhsBatchDim = parseNumVec(nodes, nodes[node.operands[3]]);
        auto lhsContractDim = parseNumVec(nodes, nodes[node.operands[4]]);
        auto rhsContractDim = parseNumVec(nodes, nodes[node.operands[5]]);
        auto precisionConfig = parseNumVec(nodes, nodes[node.operands[6]]);
        auto lhsShape = getShape(lhs);
        auto rhsShape = getShape(rhs);
        auto shape = dotGeneralShapeComputation(lhsShape, rhsShape, lhsBatchDim,
                                                rhsBatchDim, lhsContractDim,
                                                rhsContractDim);

        auto dotDimensionNumbersAttr = stablehlo::DotDimensionNumbersAttr::get(
            context, lhsBatchDim, rhsBatchDim, lhsContractDim, rhsContractDim);

        std::vector<Attribute> precisionVec;

        for (auto &precision : precisionConfig) {
          switch (precision) {
          case 0:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(
                context, stablehlo::Precision::DEFAULT));
            break;
          case 1:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(
                context, stablehlo::Precision::HIGH));
            break;
          case 2:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(
                context, stablehlo::Precision::HIGHEST));
            break;
          }
        }

        // TODO: Is lhs correct here?
        auto newType = deriveOutputType(lhs, shape);
        newOp = builder.create<stablehlo::DotGeneralOp>(
            location, newType, lhs, rhs, dotDimensionNumbersAttr,
            mlir::ArrayAttr::get(context, llvm::ArrayRef(precisionVec)),
            nullptr);
        break;
      }
      case Ops::ConcatenateOp: {
        auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
        auto dimension = parseNumNode(nodes, nodes[node.operands[1]]);
        newOp = builder.create<stablehlo::ConcatenateOp>(location, inputs,
                                                         dimension);
        break;
      }
      case Ops::SliceOp: {
        auto operand = opVals[node.operands[0]];
        auto startIndices = parseNumVec(nodes, nodes[node.operands[1]]);
        auto limitIndices = parseNumVec(nodes, nodes[node.operands[2]]);
        auto strides = parseNumVec(nodes, nodes[node.operands[3]]);
        newOp = builder.create<stablehlo::SliceOp>(
            location, operand, startIndices, limitIndices, strides);
        break;
      }
      case Ops::SSplit0:
      case Ops::SSplit1: {
        auto operand = opVals[node.operands[0]];
        auto axis = parseNumNode(nodes, nodes[node.operands[1]]);
        auto orig = opVals[node.operands[2]];
        auto [startIndices, limitIndices] =
            getSliceIndicesFromSplit(node.op, operand, axis, orig);
        std::vector<int64_t> strides(getShape(operand).size(), 1);
        newOp = builder.create<stablehlo::SliceOp>(
            location, operand, startIndices, limitIndices, strides);
        break;
      }
      case Ops::PadOp: {
        auto operand = opVals[node.operands[0]];
        auto paddingValue = opVals[node.operands[1]];
        auto edgePaddingLow = parseNumVec(nodes, nodes[node.operands[2]]);
        auto edgePaddingHigh = parseNumVec(nodes, nodes[node.operands[3]]);
        auto interiorPadding = parseNumVec(nodes, nodes[node.operands[4]]);
        newOp = builder.create<stablehlo::PadOp>(
            location, operand, paddingValue, edgePaddingLow, edgePaddingHigh,
            interiorPadding);
        break;
      }
      case Ops::IotaOp:
        // TODO: element type handling.
        newOp = builder.create<stablehlo::IotaOp>(
            location,
            RankedTensorType::get(
                llvm::ArrayRef(parseNumVec(nodes, nodes[node.operands[1]])),
                builder.getF32Type()),
            parseNumNode(nodes, nodes[node.operands[0]]));
        break;
      case Ops::ReturnOp: {
        auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
        newOp = builder.create<func::ReturnOp>(location, inputs);
        break;
      }
      case Ops::BlackBox: {
        assert(node.operands.size() == 3);
        auto blackboxID = parseNumNode(nodes, nodes[node.operands[0]]);
        auto operands = parseOpVec(opVals, nodes[node.operands[1]]);
        std::vector<Value> capturedValues =
            parseOpVec(opVals, nodes[node.operands[2]]);
        size_t numOperands = operands.size();
        newOp = blackboxIDToTensorInfo->at(blackboxID);

        // Substitute the old captured values with the new ones
        std::vector<Value> &oldCapturedValues =
            blackboxIDToCapturedValues->at(blackboxID);
        assert(oldCapturedValues.size() == capturedValues.size());
        IRMapping subst;
        for (int i = 0; i < capturedValues.size(); i++) {
          subst.map(oldCapturedValues[i], capturedValues[i]);
        }
        newOp = newOp->clone(subst);

        assert(numOperands == newOp->getNumOperands());
        newOp->insertOperands(0, operands);
        break;
      }
      default:
        throw std::invalid_argument("unimplemented op");
      }
      if (newOp) {
        opsToAdd.push_back(newOp);
        opVals.push_back(newOp->getResult(0));
      } else {
        // This is bad practice, as we're pushing nullptr
        // to ops in case of Input, Num, or Var nodes. This
        // is unsafe, but maintains indexing. We could use
        // some llvm no-op, but that would not be much better.
        opVals.push_back(nullptr);
      }
    }
    block.clear();
    for (auto op : opsToAdd) {
      block.push_back(op);
    }
  }

  bool isUsedOutsideSegment(Value result, Block &entryBlock,
                            SmallVector<Operation *> &currentOps) {
    for (auto *user : result.getUsers()) {
      // Check if the user operation is not in the current segment
      if (std::find(currentOps.begin(), currentOps.end(), user) ==
          currentOps.end()) {
        // Check if the user operation is still within the same block
        // (entryBlock)
        if (user->getBlock() == &entryBlock) {
          return true;
        }
      }
    }
    return false;
  }

  struct SegmentationPoint {
    /// Operation at the end of a segment
    Operation *endOp;

    /// Values crossing into the segment
    SmallVector<Value> inputs;

    /// Values crossing out of the segment
    SmallVector<Value> outputs;
  };

  struct SegmentedModule {
    ModuleOp module;
    SegmentationPoint segmentPoint;
  };

  std::vector<SegmentedModule> segmentGraph(func::FuncOp funcOp,
                                            OpBuilder &builder) {
    auto context = builder.getContext();
    Block &entryBlock = funcOp.getBody().front();

    // First pass to determine segmentation points and necessary types.
    // TODO: abstract out as separate function

    const int segmentThreshold = 70;
    SmallVector<SegmentationPoint> segmentationPoints;
    SmallVector<Operation *> currentOps;
    SegmentationPoint segment;

    // We need to keep track of anything that was an output value, so that if
    // we see it in later segments then we know to add that as an input.
    DenseSet<Value> outputsBeforeCurrentSegment;

    int nonBlackboxedInCurrentSegment = 0;

    for (auto it = entryBlock.begin(); it != entryBlock.end(); ++it) {
      Operation &op = *it;
      currentOps.push_back(&op);

      for (Value operand : op.getOperands()) {
        if (operand.getDefiningOp() == nullptr ||
            operand.getDefiningOp()->getBlock() != &entryBlock ||
            outputsBeforeCurrentSegment.find(operand) !=
                outputsBeforeCurrentSegment.end()) {
          if (std::find(segment.inputs.begin(), segment.inputs.end(),
                        operand) == segment.inputs.end()) {
            segment.inputs.push_back(operand);
          }
        }
      }

      bool blackboxed = isBlackboxed(&op);

      // TODO: This ensures the last node in segment is blackboxed, but
      // ideally we actually want to reduce the number of segment outputs.
      if ((blackboxed && nonBlackboxedInCurrentSegment >= segmentThreshold) ||
          it == (--entryBlock.end())) {
        // Track outputs of this segment
        for (Operation *op : currentOps) {
          for (Value result : op->getResults()) {
            if (isUsedOutsideSegment(result, entryBlock, currentOps)) {
              segment.outputs.push_back(result);
            }
          }
        }
        segment.endOp = &op;
        segmentationPoints.push_back(segment);
        currentOps.clear();
        nonBlackboxedInCurrentSegment = 0;

        outputsBeforeCurrentSegment.insert(segment.outputs.begin(),
                                           segment.outputs.end());
        segment = SegmentationPoint();
      }

      if (!blackboxed)
        nonBlackboxedInCurrentSegment++;
    }

    std::vector<SegmentedModule> segmentedModules;

    auto opIt = entryBlock.begin();
    for (int i = 0; i < segmentationPoints.size(); i++) {
      auto segment = segmentationPoints[i];
      ModuleOp currentModule = ModuleOp::create(builder.getUnknownLoc());

      auto funcType = FunctionType::get(context, getValueTypes(segment.inputs),
                                        getValueTypes(segment.outputs));
      auto newFuncOp = builder.create<func::FuncOp>(
          builder.getUnknownLoc(), "segmented_func_" + std::to_string(i),
          funcType);
      auto newBlock = newFuncOp.addEntryBlock();

      IRMapping originalToCloned;
      // Map original block arguments to new block arguments
      for (unsigned i = 0; i < segment.inputs.size(); ++i) {
        originalToCloned.map(segment.inputs[i], newBlock->getArgument(i));
      }

      // Clone operations
      while (opIt != entryBlock.end() &&
             opIt != entryBlock.getOperations().end() &&
             &(*opIt) != segment.endOp) {
        Operation *clonedOp = builder.clone(*opIt, originalToCloned);
        newBlock->push_back(clonedOp);
        updateMapper(&(*opIt), clonedOp, originalToCloned);
        ++opIt;
      }

      // Clone the end operation
      Operation *clonedEndOp = builder.clone(*opIt, originalToCloned);
      newBlock->push_back(clonedEndOp);
      updateMapper(&(*opIt), clonedEndOp, originalToCloned);
      ++opIt;

      // We need to return all the output values as well as endOp.
      SmallVector<Value> returns;
      for (auto output : segment.outputs) {
        returns.push_back(originalToCloned.lookup(output));
      }

      if (!returns.empty()) {
        auto returnOp =
            builder.create<func::ReturnOp>(builder.getUnknownLoc(), returns);
        newBlock->push_back(returnOp);
      }
      currentModule.push_back(newFuncOp);
      SegmentedModule sm;
      sm.module = currentModule;
      sm.segmentPoint = segment;
      segmentedModules.push_back(sm);
    }

    return segmentedModules;
  }

  SmallVector<Type> getValueTypes(SmallVector<Value> &values) {
    SmallVector<Type> types;
    for (auto val : values) {
      types.push_back(val.getType());
    }
    return types;
  }

  void updateMapper(Operation *opIt, Operation *clonedOp, IRMapping &mapper) {
    for (unsigned i = 0; i < opIt->getNumResults(); ++i) {
      mapper.map(opIt->getResult(i), clonedOp->getResult(i));
    }
  }

  /// Inline the operations from each segmented module into the main function
  void recombineGraph(ModuleOp mainModule,
                      std::vector<SegmentedModule> &optimizedModules,
                      OpBuilder &builder) {
    func::FuncOp mainFunc;
    for (auto &op : mainModule.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        mainFunc = funcOp;
        break;
      }
    }

    if (!mainFunc) {
      llvm::errs() << "Error: No main FuncOp found in the main module.\n";
      return;
    }

    Block &entryBlock = mainFunc.getBody().front();
    entryBlock.clear();
    builder.setInsertionPointToEnd(&entryBlock);

    // Vector to keep track of the outputs from the previous segment
    SmallVector<Value> previousSegmentOutputs;

    DenseMap<Value, Value> availableValues;
    // Initialize with main function arguments
    for (auto arg : mainFunc.getArguments()) {
      availableValues[arg] = arg;
    }

    for (size_t i = 0; i < optimizedModules.size(); ++i) {
      SegmentedModule segmentedModule = optimizedModules[i];

      SegmentationPoint segmentPoint = segmentedModule.segmentPoint;
      func::FuncOp segmentedFunc;

      segmentedModule.module.walk([&](func::FuncOp funcOp) {
        if (segmentedFunc) {
          llvm::errs() << "Error: Segmented module contains two FuncOps.\n";
          return;
        }
        segmentedFunc = funcOp;
      });

      if (!segmentedFunc) {
        llvm::errs() << "Error: Segmented module " << i
                     << " does not contain a FuncOp.\n";
        continue;
      }

      IRMapping mapper;
      // Map inputs using availableValues
      for (unsigned argIdx = 0; argIdx < segmentedFunc.getNumArguments();
           ++argIdx) {
        Value segmentedArg = segmentedFunc.getArgument(argIdx);
        Value originalValue = segmentPoint.inputs[argIdx];
        Value correspondingMainValue = availableValues.lookup(originalValue);
        if (!correspondingMainValue) {
          llvm::errs() << "Error: Unable to find corresponding value for "
                          "segmented argument.\n";
          return;
        }
        mapper.map(segmentedArg, correspondingMainValue);
      }

      // Vector to collect the outputs of the current segment
      SmallVector<Value> currentSegmentOutputs;

      // Clone and insert the operations from the segmented function into the
      // main function
      for (auto &op : segmentedFunc.getBody().front().getOperations()) {
        if (auto retOp = dyn_cast<func::ReturnOp>(op)) {
          // Collect the return values mapped to the main function's values
          for (auto operand : retOp.getOperands()) {
            Value mappedVal = mapper.lookupOrDefault(operand);
            currentSegmentOutputs.push_back(mappedVal);
          }
          // Do not clone the `func.return` operation
          continue;
        } else {
          // Clone the operation with remapped operands
          Operation *clonedOp = builder.clone(op, mapper);
          // Map the results for subsequent operations
          for (auto result : op.getResults()) {
            mapper.map(result, clonedOp->getResult(result.getResultNumber()));
          }
        }
      }

      // Update the previousSegmentOutputs for the next segment
      previousSegmentOutputs = currentSegmentOutputs;

      for (size_t idx = 0; idx < segmentPoint.outputs.size(); ++idx) {
        Value originalOutput = segmentPoint.outputs[idx];
        Value outputValue = currentSegmentOutputs[idx];
        availableValues[originalOutput] = outputValue;
      }
    }

    // After all segments have been inlined, insert a single `func.return` at
    // the end
    if (!previousSegmentOutputs.empty()) {
      builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                     previousSegmentOutputs);
    } else {
      // If there are no outputs, insert a void return
      builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange{});
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto context = module->getContext();
    OpBuilder builder(context);

    // We assume there's only one FuncOp
    func::FuncOp funcOp;
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto foundFuncOp = dyn_cast<func::FuncOp>(op)) {
        funcOp = foundFuncOp;
        break;
      }
    }

    if (!funcOp) {
      llvm::errs() << "No FuncOp found in the module.\n";
      return;
    }

    // llvm::errs() << "Running EqualitySaturationPass on the module.\n";
    // Segment the graph
    auto segmentedModules = segmentGraph(funcOp, builder);

    // Optimize each segmented subgraph
    for (int i = 0; i < segmentedModules.size(); ++i) {
      std::vector<Operation *> blackboxIDToTensorInfo;
      std::unordered_map<int, std::vector<Value>> blackboxIDToCapturedValues;

      auto &segmentedModule = segmentedModules[i];
      // llvm::errs() << "Creating egraph for segment " << i + 1 << " of "
      //              << segmentedModules.size() << "\n";
      auto graph =
          createEgraph(&blackboxIDToTensorInfo, &blackboxIDToCapturedValues,
                       builder, segmentedModule.module);
      // llvm::errs() << "Optimizing segment " << i + 1 << " of "
      //              << segmentedModules.size() << "\n";
      auto optimized = graph->optimize();
      // llvm::errs() << "reconstructing stablehlo for segment " << i + 1 << "
      // of "
      //              << segmentedModules.size() << "\n";
      reconstructStablehlo(&segmentedModule.module, &blackboxIDToTensorInfo,
                           &blackboxIDToCapturedValues, optimized, builder);

      // llvm::errs() << "Segment " << i + 1 << " optimized successfully. \n";
    }
    // Recombine the optimized segments into the original function
    recombineGraph(module, segmentedModules, builder);
    llvm::errs() << "EqualitySaturationPass completed.\n";
  }
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createEqualitySaturationPass() {
  return std::make_unique<EqualitySaturationPass>();
}
} // end namespace enzyme
} // end namespace mlir
