#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"

#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Types.h"
#include "stablehlo/reference/Value.h"
#include "stablehlo/transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include <limits>
#include <optional>
#include <random>
#include <string>
#include <variant>

namespace {
// Positional argument for the MLIR file (required)
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir file>"),
                                         llvm::cl::Required);

// --seed=<int> (defaults to 0, which we will treat as "generate a random seed")
llvm::cl::opt<uint32_t>
    seedOpt("seed",
            llvm::cl::desc("Seed for the random number generator (0 = random)"),
            llvm::cl::init(0));

// --use-custom-pool (boolean flag for your debugging needs)
llvm::cl::opt<bool> useCustomPool(
    "use-custom-pool",
    llvm::cl::desc(
        "Inject a specific custom debug pool instead of the cursed vectors"),
    llvm::cl::init(false));

// --maxElements (do not blow up CI or your machine with a large number)
llvm::cl::opt<int64_t> maxElements(
    "max-elements",
    llvm::cl::desc(
        "Maximum tensor elements to fuzz before skipping (default 100M)"),
    llvm::cl::init(10000000));
} // namespace

using namespace mlir;

// Supported types in the StableHLO specification:
// - Unsigned integers: ui4, ui8, ui16, ui32, ui64
// - Signed integers: i4, i8, i16, i32, i64
// - Boolean: i1
// - Floating-point: bf16, f16, f32, f64
// - Complex: complex<f32>, complex<f64>

// Use std::numeric_limits and templates to create most of our cursed values for
// our types except not supported types in C++ since numeric_limits does not
// help there

template <typename T> std::vector<T> createCursedFloatVector() {
  const std::vector<T> cursedValues = {(T)0.0,
                                       (T)-0.0,
                                       (T)1.0,
                                       (T)-1.0,
                                       (T)2.0,
                                       (T)-2.0,
                                       std::numeric_limits<T>::quiet_NaN(),
                                       std::numeric_limits<T>::infinity(),
                                       -std::numeric_limits<T>::infinity(),
                                       std::numeric_limits<T>::denorm_min(),
                                       std::numeric_limits<T>::max(),
                                       -std::numeric_limits<T>::max(),
                                       std::numeric_limits<T>::min(),
                                       (T)0.9999999,
                                       (T)1.0000001,
                                       static_cast<T>(M_PI),
                                       static_cast<T>(M_PI_2),
                                       static_cast<T>(M_E)};
  return cursedValues;
}

template <typename T> std::vector<T> createCursedSignedIntegerVector() {
  const std::vector<T> cursedValues = {(T)0,
                                       (T)1,
                                       (T)-1,
                                       (T)2,
                                       (T)-2,
                                       std::numeric_limits<T>::max(),
                                       std::numeric_limits<T>::min(),
                                       (T)(std::numeric_limits<T>::max() - 1),
                                       (T)(std::numeric_limits<T>::min() + 1)};
  return cursedValues;
}

template <typename T> std::vector<T> createCursedUnsignedIntegerVector() {
  const std::vector<T> cursedValues = {(T)0, (T)1, (T)2,
                                       std::numeric_limits<T>::max(),
                                       (T)(std::numeric_limits<T>::max() - 1)};
  return cursedValues;
}

template <typename T> std::vector<std::complex<T>> createCursedComplexVector() {
  const std::vector<std::complex<T>> cursedValues = {
      {(T)0.0, (T)0.0},
      {(T)-0.0, (T)0.0},
      {(T)0.0, (T)-0.0},
      {(T)-0.0, (T)-0.0},

      {(T)1.0, (T)-1.0},
      {(T)0.9999999, (T)1.0000001},

      {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()},
      {std::numeric_limits<T>::quiet_NaN(),
       -std::numeric_limits<T>::infinity()},
      {std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity()},

      {std::numeric_limits<T>::max(), std::numeric_limits<T>::denorm_min()},
      {std::numeric_limits<T>::denorm_min(), std::numeric_limits<T>::min()},

      {static_cast<T>(M_PI), static_cast<T>(M_E)}};
  return cursedValues;
}

// Packing our bits directly for non C++ types supported in the StableHLO spec
std::vector<uint16_t> createCursedF16Vector() {
  // IEEE 754 Half-Precision (1 sign bit, 5 exponent bits, 10 mantissa bits)
  return {
      0x0000, // 0.0
      0x8000, // -0.0
      0x3C00, // 1.0
      0xBC00, // -1.0
      0x4000, // 2.0
      0xC000, // -2.0
      0x7E00, // quiet_NaN
      0x7C00, // infinity
      0xFC00, // -infinity
      0x0001, // denorm_min
      0x7BFF, // max
      0xFBFF, // -max
      0x0400, // min (smallest normalized)
      0x3BFF, // 0.999... (largest value < 1.0)
      0x3C01, // 1.000... (smallest value > 1.0)
      0x4248, // M_PI (~3.140625)
      0x3E48, // M_PI_2 (~1.5703125)
      0x4170  // M_E (~2.71875)
  };
}

std::vector<uint16_t> createCursedBF16Vector() {
  // BFloat16 (1 sign bit, 8 exponent bits, 7 mantissa bits)
  return {
      0x0000, // 0.0
      0x8000, // -0.0
      0x3F80, // 1.0
      0xBF80, // -1.0
      0x4000, // 2.0
      0xC000, // -2.0
      0x7FC0, // quiet_NaN
      0x7F80, // infinity
      0xFF80, // -infinity
      0x0001, // denorm_min
      0x7F7F, // max
      0xFF7F, // -max
      0x0080, // min (smallest normalized)
      0x3F7F, // 0.999... (largest value < 1.0)
      0x3F81, // 1.000... (smallest value > 1.0)
      0x4049, // M_PI (~3.140625)
      0x3FC9, // M_PI_2 (~1.5703125)
      0x402E  // M_E (~2.71875)
  };
}

std::vector<int8_t> createCursedI4Vector() {
  // 4-bit signed integer boundaries: Max is 7, Min is -8
  return {0, 1, -1, 2, -2, 7, -8, 6, -7};
}

std::vector<uint8_t> createCursedUI4Vector() {
  // 4-bit unsigned integer boundaries: Max is 15, Min is 0
  return {0, 1, 2, 15, 14};
}

OwningOpRef<ModuleOp> loadMLIRModule(MLIRContext &context,
                                     llvm::StringRef filePath) {
  std::string errorMessage;
  auto file = mlir::openInputFile(filePath, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to open file: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

using AnyVector = std::variant<std::vector<float>,    // f32, f16, bf16
                               std::vector<double>,   // f64
                               std::vector<int8_t>,   // i8, i4
                               std::vector<int16_t>,  // i16
                               std::vector<int32_t>,  // i32
                               std::vector<int64_t>,  // i64
                               std::vector<uint8_t>,  // ui8, ui4
                               std::vector<uint16_t>, // ui16
                               std::vector<uint32_t>, // ui32
                               std::vector<uint64_t>, // ui64
                               std::vector<bool>,     // i1
                               std::vector<std::complex<float>>, // complex<f32>
                               std::vector<std::complex<double>> // complex<f64>
                               >;

// clang-format off
// Our .mlir test files contain lines like this at the start
// // RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_convert_to_convolution=true" %s | FileCheck %s 
// we want to extract the passes so we can run them on the unoptimized mlir
// function and get the optimized function to compare them
// clang-format on
// Returns a tuple: { passPipeline, allowUnregisteredDialect, splitInputFile }
std::tuple<std::string, bool, bool> parseRunLine(llvm::StringRef filePath) {
  auto bufferOrError = llvm::MemoryBuffer::getFile(filePath);
  if (!bufferOrError)
    return {"", false, false};

  llvm::StringRef content = bufferOrError.get()->getBuffer();
  llvm::StringRef runLine;

  // 1. Find the RUN line
  while (!content.empty()) {
    llvm::StringRef line;
    std::tie(line, content) = content.split('\n');
    line = line.trim();
    if (line.consume_front("// RUN:")) {
      runLine = line;
      break;
    }
  }

  if (runLine.empty())
    return {"", false, false};

  llvm::BumpPtrAllocator allocator;
  llvm::StringSaver saver(allocator);
  llvm::SmallVector<const char *, 20> args;
  llvm::cl::TokenizeGNUCommandLine(runLine, saver, args);

  llvm::SmallVector<std::string, 4> passes;
  bool allowUnreg = false;
  bool split = false;
  bool afterPipe = false;

  // 3. Process the safe tokens
  for (const char *argChar : args) {
    llvm::StringRef token(argChar);

    if (token == "|") {
      afterPipe = true;
      continue;
    }
    if (afterPipe || token == "enzymexlamlir-opt" || token == "%s")
      continue;

    // Extract environment flags
    if (token == "-allow-unregistered-dialect") {
      allowUnreg = true;
      continue;
    }
    if (token == "-split-input-file" || token == "--split-input-file") {
      split = true;
      continue;
    }

    // Format passes for the PassManager
    if (token.consume_front("--") || token.consume_front("-")) {
      auto [passName, passOpts] = token.split('=');
      if (!passOpts.empty()) {
        passes.push_back((passName + "{" + passOpts + "}").str());
      } else {
        passes.push_back(passName.str());
      }
    }
  }

  // llvm::join merges the passes with commas automatically
  return {llvm::join(passes, ","), allowUnreg, split};
}

std::optional<AnyVector> CreateCursedVector(mlir::Type elementType) {
  int64_t bitWidth = stablehlo::numBits(elementType);
  if (stablehlo::isSupportedFloatType(elementType)) {
    if (bitWidth == 16) {
      return createCursedF16Vector();
    } else if (bitWidth == 32) {
      return createCursedFloatVector<float>();
    } else if (bitWidth == 64) {
      return createCursedFloatVector<double>();
    } else {
      llvm::outs() << "Warning: Unsupported float bit-width: " << bitWidth
                   << "\n";
    }
  }

  else if (stablehlo::isSupportedSignedIntegerType(elementType)) {
    if (bitWidth == 4) {
      return createCursedI4Vector();
    } else if (bitWidth == 8) {
      return createCursedSignedIntegerVector<int8_t>();
    } else if (bitWidth == 16) {
      return createCursedSignedIntegerVector<int16_t>();
    } else if (bitWidth == 32) {
      return createCursedSignedIntegerVector<int32_t>();
    } else if (bitWidth == 64) {
      return createCursedSignedIntegerVector<int64_t>();
    } else {
      llvm::outs() << "Warning: Unsupported signed int bit-width: " << bitWidth
                   << "\n";
    }
  }

  else if (stablehlo::isSupportedUnsignedIntegerType(elementType)) {
    if (bitWidth == 4) {
      return createCursedUI4Vector();
    } else if (bitWidth == 8) {
      return createCursedUnsignedIntegerVector<uint8_t>();
    } else if (bitWidth == 16) {
      return createCursedUnsignedIntegerVector<uint16_t>();
    } else if (bitWidth == 32) {
      return createCursedUnsignedIntegerVector<uint32_t>();
    } else if (bitWidth == 64) {
      return createCursedUnsignedIntegerVector<uint64_t>();
    } else {
      llvm::outs() << "Warning: Unsupported unsigned int bit-width: "
                   << bitWidth << "\n";
    }
  }

  else if (stablehlo::isSupportedBooleanType(elementType)) {
    return std::vector<bool>{true, false};
  }

  else if (stablehlo::isSupportedComplexType(elementType)) {
    int64_t bitWidth = stablehlo::numBits(elementType);
    // Note: bitWidth for complex is 2x the base type (complex<f32> is 64
    // bits total) ((I think at least))
    if (bitWidth == 64) {
      return createCursedComplexVector<float>();
    } else if (bitWidth == 128) {
      return createCursedComplexVector<double>();
    } else {
      llvm::outs() << "Warning: Unsupported complex bit-width: " << bitWidth
                   << "\n";
    }
  }

  else {
    llvm::outs() << "Warning: Unsupported element type entirely. Skipping.\n";
  }

  return std::nullopt;
}

std::optional<mlir::DenseElementsAttr> generateCursedTensor(mlir::Type argType,
                                                            std::mt19937 &gen) {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(argType);
  if (!tensorType) {
    return std::nullopt;
  }

  Type elementType = tensorType.getElementType();
  int64_t numElements = tensorType.getNumElements();
  auto cursedNumberVector = CreateCursedVector(elementType);
  if (!cursedNumberVector)
    return std::nullopt;
  mlir::DenseElementsAttr attr;

  std::visit(
      [&](auto &&pool) {
        using VectorType = std::decay_t<decltype(pool)>;
        using T = typename VectorType::value_type;

        if (pool.empty())
          return;

        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);

        if constexpr (std::is_same_v<VectorType, std::vector<bool>>) {
          std::vector<uint8_t> data(numElements);
          for (int64_t i = 0; i < numElements; ++i) {
            data[i] = pool[dist(gen)] ? 1 : 0;
          }
          attr = mlir::DenseElementsAttr::get(tensorType,
                                              llvm::ArrayRef<uint8_t>(data));
        } else {
          std::vector<T> data(numElements);
          for (int64_t i = 0; i < numElements; ++i) {
            data[i] = pool[dist(gen)];
          }
          attr =
              mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<T>(data));
        }
      },
      *cursedNumberVector);
  if (!attr)
    return std::nullopt;
  return attr;
}

bool isClose(double unopt, double opt, double rtol = 1e-5, double atol = 1e-8) {
  // 1. Both are NaN? Match.
  if (std::isnan(unopt) && std::isnan(opt))
    return true;

  // 2. Both are Inf? Match ONLY if signs are identical.
  if (std::isinf(unopt) && std::isinf(opt))
    return unopt == opt;

  // 3. Fast-math overflow allowance:
  // If one is Inf and the other is at the absolute float boundary of the same
  // sign, allow it.
  double max_val = std::numeric_limits<double>::max();
  if (std::isinf(opt) && std::abs(unopt) >= max_val * 0.99 &&
      (opt > 0) == (unopt > 0))
    return true;
  if (std::isinf(unopt) && std::abs(opt) >= max_val * 0.99 &&
      (unopt > 0) == (opt > 0))
    return true;

  // 4. Any other mix of Inf/NaN vs finite numbers? Bug.
  if (std::isnan(unopt) || std::isnan(opt) || std::isinf(unopt) ||
      std::isinf(opt))
    return false;

  // 5. Standard tolerance check
  return std::abs(unopt - opt) <= (atol + rtol * std::abs(opt));
}

int main(int argc, char **argv) {
  bool anyMismatch = false;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "StableHLO Differential Fuzzer\n");

  uint32_t seed = seedOpt;
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    llvm::outs() << "[*] Running with random seed: " << seed << "\n";
  } else {
    llvm::outs() << "[*] Running with manual seed: " << seed << "\n";
  }

  auto [passPipeline, allowUnreg, split] = parseRunLine(inputFilename);
  if (passPipeline.empty()) {
    llvm::errs() << "No RUN line found in file!\n";
    return 1;
  }

  if (split) {
    llvm::outs() << "[!] Skipping test: Fuzzer does not yet support "
                    "--split-input-file.\n";
    return 0;
  }

  std::mt19937 gen(seed);

  MLIRContext context;
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::transform::TransformDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();
  context.loadDialect<mlir::enzyme::EnzymeDialect>();
  context.loadDialect<mlir::enzymexla::EnzymeXLADialect>();

  mlir::enzyme::registerenzymexlaPasses();
  mlir::transform::registerTransformPasses();

  OwningOpRef<ModuleOp> module = loadMLIRModule(context, inputFilename);
  if (!module)
    return 1;

  mlir::PassManager legalizationPM(&context);
  // Make it possible to run the chlo tests
  auto &funcPM = legalizationPM.nest<mlir::func::FuncOp>();
  funcPM.addPass(mlir::stablehlo::createChloLegalizeToStablehloPass());

  // Lower EnzymeXLA custom operations to StableHLO
  funcPM.addPass(mlir::enzyme::createLowerEnzymeXLAMathPass());
  funcPM.addPass(mlir::enzyme::createLowerEnzymeXLABLASPass());
  funcPM.addPass(mlir::enzyme::createLowerEnzymeXLALapackPass());
  funcPM.addPass(mlir::enzyme::createEnzymeBatchToStableHLOPass());
  funcPM.addPass(mlir::enzyme::createLowerEnzymeXLALinalgPass());
  funcPM.addPass(mlir::enzyme::createLowerEnzymeJacobianStableHLO());
  funcPM.addPass(mlir::enzyme::createLowerEnzymeXLAMPIPass());

  mlir::PassManager pm(&context);
  if (mlir::failed(mlir::parsePassPipeline(passPipeline, pm, llvm::errs()))) {
    llvm::errs() << "Failed to parse the pass pipeline: " << passPipeline
                 << "\n";
    return 1;
  }

  OwningOpRef<ModuleOp> optimizedModule = module->clone();
  if (mlir::failed(pm.run(*optimizedModule))) {
    llvm::errs() << "Pass pipeline failed to run on module!\n";
    return 1;
  }
  module->walk([&](mlir::func::FuncOp unoptFunc) {
    llvm::StringRef funcName = unoptFunc.getName();

    auto optFunc = optimizedModule->lookupSymbol<mlir::func::FuncOp>(funcName);
    if (!optFunc) {
      llvm::outs() << "Skipping: Function deleted by optimization.\n";
      return;
    }

    llvm::SmallVector<mlir::DenseElementsAttr> evalArgs;

    for (BlockArgument arg : unoptFunc.getArguments()) {
      auto tensorType = llvm::dyn_cast<RankedTensorType>(arg.getType());

      int64_t numElements = tensorType.getNumElements();
      if (numElements > maxElements) {
        llvm::outs() << "  Skipping function: argument has " << numElements
                     << " elements (exceeds --max-elements limit of "
                     << maxElements << ").\n";
        return; // Aborts this function, moves to the next
      }

      std::optional<mlir::DenseElementsAttr> attrOpt =
          generateCursedTensor(arg.getType(), gen);

      if (!attrOpt) {
        llvm::outs() << "Skipping non-ranked tensor argument.\n";
        continue;
      }
      evalArgs.push_back(*attrOpt);
    }
    stablehlo::InterpreterConfiguration config;
    // Get EnzymeXLA and CHLO into stableHLO
    OwningOpRef<ModuleOp> tempUnoptMod = ModuleOp::create(unoptFunc.getLoc());
    func::FuncOp clonedUnopt = unoptFunc.clone();
    clonedUnopt.setName("main");
    tempUnoptMod->push_back(clonedUnopt);

    if (mlir::failed(legalizationPM.run(*tempUnoptMod))) {
      llvm::outs()
          << "  [!] Legalization failed on unoptimized IR. Skipping.\n";
      return;
    }

    auto unoptResults =
        stablehlo::evalModule(tempUnoptMod.get(), evalArgs, config);
    if (mlir::failed(unoptResults)) {
      llvm::outs() << "  [!] Unoptimized evaluation failed.\n";
      return;
    }

    OwningOpRef<ModuleOp> tempOptMod = ModuleOp::create(optFunc.getLoc());
    func::FuncOp clonedOpt = optFunc.clone();
    clonedOpt.setName("main");
    tempOptMod->push_back(clonedOpt); // Push BEFORE running pass

    if (mlir::failed(legalizationPM.run(*tempOptMod))) {
      llvm::outs() << "  [!] Legalization failed on optimized IR. Skipping.\n";
      return;
    }

    auto optResults = stablehlo::evalModule(tempOptMod.get(), evalArgs, config);
    if (mlir::failed(optResults)) {
      llvm::outs() << "   Optimized evaluation failed.\n";
      return;
    }

    // Comparing results from optimized and unoptimized function
    auto &unoptVals = *unoptResults;
    auto &optVals = *optResults;
    if (unoptVals.size() != optVals.size()) {
      llvm::outs() << "   MISMATCH: Different number of return values!\n";
      return;
    }
    bool mismatch = false;
    for (size_t i = 0; i < unoptVals.size(); ++i) {
      if (unoptVals[i] == optVals[i])
        continue; // Fast path: strict bitwise match

      // If strict match fails, check if it's a floating point deviation
      auto unoptAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(unoptVals[i]);
      auto optAttr = llvm::dyn_cast<mlir::DenseElementsAttr>(optVals[i]);

      if (unoptAttr && optAttr &&
          llvm::isa<mlir::FloatType>(unoptAttr.getElementType())) {
        auto uIt = unoptAttr.getValues<llvm::APFloat>().begin();
        auto oIt = optAttr.getValues<llvm::APFloat>().begin();
        auto uEnd = unoptAttr.getValues<llvm::APFloat>().end();

        for (; uIt != uEnd; ++uIt, ++oIt) {
          double uVal = (*uIt).convertToDouble();
          double oVal = (*oIt).convertToDouble();
          if (!isClose(uVal, oVal)) {
            llvm::outs() << "  [!] FLOAT MISMATCH: Expected " << uVal
                         << " but got " << oVal << "\n";
            mismatch = true;
            anyMismatch = true;
            break;
          }
        }
      } else {
        // If it's an integer/bool type and failed strict equality, it's a
        // definitive bug.
        llvm::outs() << "  [!] MISMATCH on return value " << i << "!\n";
        mismatch = true;
        anyMismatch = false;
      }
    }
    if (!mismatch) {
      llvm::outs() << "  PASS: Outputs match exactly.\n";
    }
  });
  return anyMismatch ? 1 : 0;
}
