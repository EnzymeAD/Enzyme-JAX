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
#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/platform_util.h"

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

class OperationTimer {
public:
  /**
   * Measure cost of operation (execution time in microseconds) by running it
   * many times and measuring the time taken.
   * TODO: Make cloning optional
   * TODO: Preserve context across runs so that we're not creating unnecessary
   * contexts
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
        opName == "stablehlo.constant" || opName == "stablehlo.return" ||
        opName == "stablehlo.compare")
      return 0;

    if (runtimeCache.contains(op)) {
      return runtimeCache[op];
    }

    auto context = OperationTimer::getContext();

    ModuleOp wrapperModule = createModuleFromOperation(context, op);

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
    xla::PjRtBuffer *res[numResults];

    for (int i = 0; i < numArgs; i++) {
      args[i] = getRandomInput(client, op->getOperand(i).getType());
      isArgDonatable[i] = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < warmup + repetitions; i++) {
      if (i == warmup)
        t1 = std::chrono::high_resolution_clock::now();
      XLAExecute(executable, numArgs, args, isArgDonatable, numResults, res,
                 &futures, nullptr);

      // Cleanup
      for (int i = 0; i < numResults; i++) {
        PjRtBufferFree(res[i]);
      }
    }

    assert(!futures);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    FreeClient(executable->client());
    ExecutableFree(executable);

    wrapperModule.erase();

    // std::cout << op->getName().getStringRef().str() << "\n";
    auto indexOp = op->clone();
    runtimeCache.try_emplace(indexOp, duration);
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

  static Operation *cloneOpInContext(OpBuilder &builder, Operation *op) {
    IRMapping mapping;
    return cloneOpInContext(builder, op, mapping);
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
   * Create a clone of the operation in the new context recursively (i.e. going
   * down to the regions). Just using op->clone() will preserve context of the
   * original operation, which poses a problem later since stablehlo -> mhlo
   * legalization pass will not match the new operation.
   *
   * Like the normal op->clone(), any operands that use values outside of the
   * operations are remapped using the map that is provided (leaving them alone
   * if no entry is present).
   *
   * TODO: Surely there's a simpler way to do this?
   */
  static Operation *cloneOpInContext(OpBuilder &builder, Operation *op,
                                     IRMapping &mapping) {
    Location location = builder.getUnknownLoc();

    // Recursively clone regions
    llvm::SmallVector<std::unique_ptr<Region>> regions;

    for (auto &region : op->getRegions()) {
      auto newRegion = std::make_unique<Region>();

      for (auto &block : region.getBlocks()) {
        auto newBlock = new Block();

        // Map from old block arguments to new ones
        for (auto &arg : block.getArguments()) {
          mapping.map(arg, newBlock->addArgument(arg.getType(), location));
        }

        for (auto &nestedOp : block.getOperations()) {
          auto *newNestedOp = cloneOpInContext(builder, &nestedOp, mapping);
          newBlock->push_back(newNestedOp);

          // Map result of old operation to that of new operation, so that
          // operations after can use it
          for (int i = 0; i < nestedOp.getNumResults(); i++) {
            mapping.map(nestedOp.getResult(i), newNestedOp->getResult(i));
          }
        }
        newRegion->push_back(newBlock);
      }
      regions.push_back(std::move(newRegion));
    }

    OperationState opState(location,
                           op->getName()
                               .getStringRef()
                               .str(), // Use string to make a new name, rather
                                       // than reusing the OperationName
                           op->getOperands(), op->getResultTypes(),
                           op->getAttrs(), {}, regions);

    auto *newOp = builder.create(opState);

    for (int i = 0; i < newOp->getNumOperands(); i++) {
      newOp->setOperand(i, mapping.lookupOrDefault(newOp->getOperand(i)));
    }

    return newOp;
  }

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

    auto *newOp = cloneOpInContext(builder, op);

    // Create a func.func to wrap newOp around
    FunctionType funcType =
        FunctionType::get(context, op->getOperandTypes(), op->getResultTypes());
    func::FuncOp funcOp =
        builder.create<func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    Block *entryBlock = funcOp.addEntryBlock();

    for (int i = 0; i < op->getNumOperands(); i++) {
      newOp->setOperand(i, funcOp.getArgument(i));
    }

    entryBlock->push_back(newOp);

    auto returnOp =
        builder.create<func::ReturnOp>(location, newOp->getResults());
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
  case tensat::Ops::SliceOp: {
    mlirOp = builder.create<stablehlo::SliceOp>(builder.getUnknownLoc(),
                                                operands[0], other_vecs[0],
                                                other_vecs[1], other_vecs[2]);
    break;
  }
  case tensat::Ops::ConcatenateOp: {
    mlirOp = builder.create<stablehlo::ConcatenateOp>(builder.getUnknownLoc(),
                                                      operands, int_args[0]);
    break;
  }
  case tensat::Ops::PadOp: {
    mlirOp = builder.create<stablehlo::PadOp>(
        builder.getUnknownLoc(), operands[0], operands[1], other_vecs[0],
        other_vecs[1], other_vecs[2]);
    break;
  }
  case tensat::Ops::IotaOp: {
    mlirOp = builder.create<stablehlo::IotaOp>(
        builder.getUnknownLoc(),
        RankedTensorType::get(llvm::ArrayRef(other_vecs[0]),
                              builder.getF32Type()),
        int_args[0]);
    break;
  }
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
    repeats = 100;
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
      int numOperands = op->getNumOperands();
      rust::Vec<tensat::Tensor> outputs;
      for (auto result : op->getResults())
        outputs.push_back(mlirValueToTensatTensor(result));

      std::vector<tensat::TensorInfo *> processedOperands;
      // We shouldn't clone operands, as those values will be invalidated after
      // a block.clear(), and we access these during reconstruction.
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
    if (tensorInfo != nullptr) {
      opToTensorInfo->insert({op, tensorInfo});
      return tensorInfo;
    }
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

    // We don't clear the block here straight away, because this will invalidate
    // the old captured values in blackbox, and so the substitution won't work.
    // We instead put everything in a vector, and only clear the block and
    // insert everything at the very end.
    std::vector<Operation *> opsToAdd;

    auto location = builder.getUnknownLoc();

    for (auto &node : nodes) {
      Operation *newOp = nullptr;
      // Create the new operation based on the operands
      if (node.name == "Var" || node.name == "Num" || node.name == "Vec") {
        /* do nothing */
      } else if (node.name == "Input") {
        int blockArgNumber = parseNumNode(nodes, nodes[node.operands[1]]);
        opVals.push_back(block.getArgument(blockArgNumber));
        continue;
      } else if (node.name == "Index") {
        int index = parseNumNode(nodes, nodes[node.operands[0]]);
        int input = node.operands[1];
        opVals.push_back(opVals[input].getDefiningOp()->getResult(index));
        continue;
      } else if (node.name == "NegOp") {
        newOp = createUnaryOp<stablehlo::NegOp>(builder, opVals, node);
      } else if (node.name == "TanhOp") {
        newOp = createUnaryOp<stablehlo::TanhOp>(builder, opVals, node);
      } else if (node.name == "ExpOp") {
        newOp = createUnaryOp<stablehlo::ExpOp>(builder, opVals, node);
      } else if (node.name == "AddOp") {
        newOp = createBinaryOp<stablehlo::AddOp>(builder, opVals, node);
      } else if (node.name == "SubtractOp") {
        newOp = createBinaryOp<stablehlo::SubtractOp>(builder, opVals, node);
      } else if (node.name == "MulOp") {
        newOp = createBinaryOp<stablehlo::MulOp>(builder, opVals, node);
      } else if (node.name == "DivOp") {
        newOp = createBinaryOp<stablehlo::DivOp>(builder, opVals, node);
      } else if (node.name == "MinOp") {
        newOp = createBinaryOp<stablehlo::MinOp>(builder, opVals, node);
      } else if (node.name == "MaxOp") {
        newOp = createBinaryOp<stablehlo::MaxOp>(builder, opVals, node);
      } else if (node.name == "TransposeOp") {
        auto input = opVals[node.operands[0]];
        auto permutation = parseNumVec(nodes, nodes[node.operands[1]]);
        newOp = builder.create<stablehlo::TransposeOp>(location, input,
                                                       permutation);
      } else if (node.name == "ReshapeOp") {
        auto input = opVals[node.operands[0]];
        auto shape = parseNumVec(nodes, nodes[node.operands[1]]);
        auto newType = deriveOutputType(input, shape);
        newOp = builder.create<stablehlo::ReshapeOp>(location, newType, input);
      } else if (node.name == "DotGeneralOp") {
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
      } else if (node.name == "ConcatenateOp") {
        auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
        auto dimension = parseNumNode(nodes, nodes[node.operands[1]]);
        newOp = builder.create<stablehlo::ConcatenateOp>(location, inputs,
                                                         dimension);
      } else if (node.name == "SliceOp") {
        auto operand = opVals[node.operands[0]];
        auto startIndices = parseNumVec(nodes, nodes[node.operands[1]]);
        auto limitIndices = parseNumVec(nodes, nodes[node.operands[2]]);
        auto strides = parseNumVec(nodes, nodes[node.operands[3]]);
        newOp = builder.create<stablehlo::SliceOp>(
            location, operand, startIndices, limitIndices, strides);
      } else if (node.name == "PadOp") {
        auto operand = opVals[node.operands[0]];
        auto paddingValue = opVals[node.operands[1]];
        auto edgePaddingLow = parseNumVec(nodes, nodes[node.operands[2]]);
        auto edgePaddingHigh = parseNumVec(nodes, nodes[node.operands[3]]);
        auto interiorPadding = parseNumVec(nodes, nodes[node.operands[4]]);
        newOp = builder.create<stablehlo::PadOp>(
            location, operand, paddingValue, edgePaddingLow, edgePaddingHigh,
            interiorPadding);
      } else if (node.name == "IotaOp") {
        // TODO: element type handling.
        newOp = builder.create<stablehlo::IotaOp>(
            location,
            RankedTensorType::get(
                llvm::ArrayRef(parseNumVec(nodes, nodes[node.operands[1]])),
                builder.getF32Type()),
            parseNumNode(nodes, nodes[node.operands[0]]));
      } else if (node.name == "ReturnOp") {
        auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
        newOp = builder.create<func::ReturnOp>(location, inputs);
      } else if (node.name == "blackbox") {
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
      } else {
        // TODO: implement other operations
        std::cout << "UNIMPLEMENTED " << node.name << "\n";
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

    const int segmentThreshold = 500;
    SmallVector<SegmentationPoint> segmentationPoints;
    SmallVector<Operation *> currentOps;
    SegmentationPoint segment;

    // We need to keep track of anything that was an output value, so that if we
    // see it in later segments then we know to add that as an input.

    // TODO: We should keep track of a mapping between function arguments
    // created as a result of an output value, so that we can use it when
    // reconstructing segments
    DenseSet<Value> outputsBeforeCurrentSegment;

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

      if (currentOps.size() >= segmentThreshold || it == (--entryBlock.end())) {
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

        outputsBeforeCurrentSegment.insert(segment.outputs.begin(),
                                           segment.outputs.end());
        segment = SegmentationPoint();
      }
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
