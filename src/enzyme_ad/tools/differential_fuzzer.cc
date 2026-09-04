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
#include "stablehlo/reference/Types.h"
#include "stablehlo/transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

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

// --max-ulps (how much floating-point drift from reassociation we tolerate)
llvm::cl::opt<unsigned> maxUlpsOpt(
    "max-ulps",
    llvm::cl::desc("Allowed deviation between optimized and unoptimized float "
                   "results, in units in the last place (default 8). log2(n) "
                   "bits are allowed to deviate so in the n=8 case the three "
                   "least signficant mantissa bits are allowed to deviate"),
    llvm::cl::init(8));

// --verbosity (how much your screen gets spammend full of stuff)
enum class Verbosity { Quiet, Normal, Verbose };

llvm::cl::opt<Verbosity> verbosity(
    "verbosity", llvm::cl::desc("Output detail level"),
    llvm::cl::values(clEnumValN(Verbosity::Quiet, "quiet", "Exit code only"),
                     clEnumValN(Verbosity::Normal, "normal",
                                "Seed, mismatches and summary"),
                     clEnumValN(Verbosity::Verbose, "verbose",
                                "Also report passing functions")),
    llvm::cl::init(Verbosity::Normal));

// --maxElements (do not blow up CI or your machine with a large number)
llvm::cl::opt<int64_t> maxElements(
    "max-elements",
    llvm::cl::desc("Maximum number of tensor elements in an argument to fuzz "
                   "before skipping (default 10M)"),
    llvm::cl::init(10000000));

llvm::cl::list<std::string> restrictInput(
    "restrict-input", llvm::cl::CommaSeparated,
    llvm::cl::desc(
        "Categories of values to omit from being part of the number pool used "
        "to generate tensors: nonzero, non_negative, noNaN, allFinite"));

} // namespace

static llvm::raw_ostream &diag() {
  return verbosity == Verbosity::Quiet ? llvm::nulls() : llvm::errs();
}

using namespace mlir;

struct PrintableComplex {
  const mlir::Complex<APFloat> v;
};

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     PrintableComplex p) {
  APFloat im = p.v.imag();
  return os << p.v.real() << (im.isNegative() ? " - " : " + ") << llvm::abs(im)
            << "i";
}

// Supported types in the StableHLO specification:
// - Unsigned integers: ui4, ui8, ui16, ui32, ui64
// - Signed integers: i4, i8, i16, i32, i64
// - Boolean: i1
// - Floating-point: bf16, f16, f32, f64
// - Complex: complex<f32>, complex<f64> (There is an RFC to extend this to
// bfloat16 and float16 complex numbers so this might need to be updated at some
// point)

SmallVector<APFloat> createCursedFloatPool(const llvm::fltSemantics &sem) {
  auto fromStr = [&](StringRef s) {
    APFloat v(sem);
    llvm::cantFail(v.convertFromString(s, APFloat::rmNearestTiesToEven));
    return v;
  };

  APFloat one(sem, 1), two(sem, 2);
  APFloat justBelowOne = one;
  justBelowOne.next(/*nextDown=*/true);
  APFloat justAboveOne = one;
  justAboveOne.next(/*nextDown=*/false);

  return {
      APFloat::getZero(sem),
      APFloat::getZero(sem, /*Negative=*/true),
      one,
      -one,
      two,
      -two,
      APFloat::getQNaN(sem),
      APFloat::getInf(sem),
      APFloat::getInf(sem, true),
      APFloat::getSmallest(sem),
      APFloat::getLargest(sem),
      APFloat::getLargest(sem, true),
      APFloat::getSmallestNormalized(sem),
      justBelowOne,
      justAboveOne,
      fromStr("3.14159265358979323846"), // pi
      fromStr("1.57079632679489661923"), // pi / 2
      fromStr("2.71828182845904523536"), // e
  };
}

SmallVector<APInt> createCursedIntPool(int64_t numBits) {
  return {
      APInt::getZero(numBits),
      APInt(numBits, 1),
      -APInt(numBits, 1),
      APInt(numBits, 2),
      -APInt(numBits, 2),
      APInt::getSignedMaxValue(numBits),
      APInt::getSignedMinValue(numBits),
      APInt::getSignedMaxValue(numBits) - 1,
      APInt::getSignedMinValue(numBits) + 1,
  };
}

SmallVector<APInt> createCursedUnsignedIntPool(int64_t numBits) {
  APInt maxValue = APInt::getMaxValue(numBits);
  return {APInt::getZero(numBits), APInt(numBits, 1), APInt(numBits, 2),
          maxValue, maxValue - 1};
}

SmallVector<mlir::Complex<APFloat>>
createCursedComplexPool(const llvm::fltSemantics &sem) {
  auto FloatPool = createCursedFloatPool(sem);
  SmallVector<mlir::Complex<APFloat>> ComplexPool;
  ComplexPool.reserve(FloatPool.size() * FloatPool.size());
  for (const APFloat &re : FloatPool)
    for (const APFloat &im : FloatPool)
      ComplexPool.emplace_back(re, im);
  return ComplexPool;
}

enum class Assume { NoNaN, Finite, NonNegative, NonZero, Unknown };

struct PoolConstraints {
  bool noNaN = false;
  bool noInf = false;
  bool noSubnormal = false;
  bool nonNegative = false;
  bool nonZero = false;
  std::optional<APInt> intLo, intHi; // from bounds
  std::optional<APFloat> floatLo, floatHi;
};

static bool allowed(const APFloat &v, const PoolConstraints &c) {
  if (c.noNaN && v.isNaN())
    return false;
  if (c.noInf && v.isInfinity())
    return false;
  if (c.noSubnormal && v.isDenormal())
    return false;
  if (c.nonNegative && v.isNegative())
    return false;
  if (c.floatLo && v < *c.floatLo)
    return false;
  if (c.floatHi && v > *c.floatHi)
    return false;
  if (c.nonZero && v.isZero())
    return false;
  return true;
}

static bool allowed(const APInt &v, const PoolConstraints &c) {
  if (c.nonZero && v.isZero())
    return false;
  if (c.nonNegative && v.isNegative())
    return false;
  return true;
}

PoolConstraints parseRestrictInput(ArrayRef<std::string> tokens) {
  PoolConstraints p;
  for (StringRef token : tokens) {
    switch (llvm::StringSwitch<Assume>(token.lower())
                .Case("nonan", Assume::NoNaN)
                .Case("no-nan", Assume::NoNaN)
                .Case("no_nan", Assume::NoNaN)
                .Case("finite", Assume::Finite)
                .Case("allfinite", Assume::Finite)
                .Case("all-finite", Assume::Finite)
                .Case("nonnegative", Assume::NonNegative)
                .Case("non-negative", Assume::NonNegative)
                .Case("non_negative", Assume::NonNegative)
                .Case("nonzero", Assume::NonZero)
                .Case("non-zero", Assume::NonZero)
                .Case("non_zero", Assume::NonZero)
                .Default(Assume::Unknown)) {
    case Assume::NoNaN:
      p.noNaN = true;
      break;
    case Assume::Finite:
      p.noNaN = p.noInf = true;
      break;
    case Assume::NonNegative:
      p.nonNegative = true;
      break;
    case Assume::NonZero:
      p.nonZero = true;
      break;
    case Assume::Unknown:
      llvm::WithColor::error(diag())
          << "--restrict-input: unknown  '" << token << "'\n";
      exit(2);
    }
  }
  return p;
}

OwningOpRef<ModuleOp> loadMLIRModule(MLIRContext &context,
                                     llvm::StringRef filePath) {
  std::string errorMessage;
  auto file = mlir::openInputFile(filePath, &errorMessage);
  if (!file) {
    llvm::WithColor::warning(diag())
        << "Failed to open file: " << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

using AnyVector =
    std::variant<SmallVector<APFloat>, SmallVector<APInt>,
                 SmallVector<mlir::Complex<APFloat>>, SmallVector<bool>>;

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
      if (passName == "pass-pipeline") {
        passes.push_back(passOpts.str());
        continue;
      }
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

void applyConstraintsFromPipeline(StringRef pipeline, PoolConstraints &c) {

  // Option form: no_nan / no_nan=true, all_finite=true. The =false spellings
  // exist, so the value has to be read rather than just the name matched.
  auto optionSet = [&](StringRef name) {
    size_t pos = 0;
    while ((pos = pipeline.find(name, pos)) != StringRef::npos) {
      StringRef rest = pipeline.substr(pos + name.size());
      pos += name.size();
      if (rest.consume_front("=")) {
        if (rest.starts_with("true"))
          return true;
        continue; // =false
      }
      // Bare form: must end at a delimiter, not be a prefix of something else.
      if (rest.empty() || rest.starts_with("}") || rest.starts_with(",") ||
          rest.starts_with(" "))
        return true;
    }
    return false;
  };

  if (optionSet("no_nan"))
    c.noNaN = true;
  if (optionSet("all_finite"))
    c.noNaN = c.noInf = true;

  // Pattern form: a pattern whose name contains no_nan, with argument 1.
  size_t pos = 0;
  while ((pos = pipeline.find("no_nan", pos)) != StringRef::npos) {
    StringRef rest = pipeline.substr(pos);
    size_t paren = rest.find('(');
    size_t delim = rest.find_first_of(";,}");
    if (paren != StringRef::npos &&
        (delim == StringRef::npos || paren < delim) &&
        rest.substr(paren).starts_with("(1)"))
      c.noNaN = true;
    pos += 6;
  }
}

PoolConstraints constraintsFromArg(Value arg, PoolConstraints &base) {
  PoolConstraints constraints = base;
  auto get = [&](StringRef name) {
    return mlir::enzyme::getAttributeFromIR<
        enzymexla::GuaranteedAnalysisResultAttr>(
        arg, name, enzymexla::GuaranteedAnalysisResult::UNKNOWN);
  };
  using G = enzymexla::GuaranteedAnalysisResult;
  if (get("enzymexla.no_nan") == G::GUARANTEED)
    constraints.noNaN = true;
  if (get("enzymexla.finite") == G::GUARANTEED)
    constraints.noNaN = constraints.noInf = true;
  if (get("enzymexla.non_negative") == G::GUARANTEED)
    constraints.nonNegative = true;
  return constraints;
}

std::optional<AnyVector> CreateCursedPool(mlir::Type elementType) {
  if (stablehlo::isSupportedFloatType(elementType)) {
    return createCursedFloatPool(
        cast<FloatType>(elementType).getFloatSemantics());
  }

  else if (stablehlo::isSupportedSignedIntegerType(elementType)) {
    return createCursedIntPool(elementType.getIntOrFloatBitWidth());
  }

  else if (stablehlo::isSupportedUnsignedIntegerType(elementType)) {
    return createCursedUnsignedIntPool(elementType.getIntOrFloatBitWidth());
  }

  else if (stablehlo::isSupportedBooleanType(elementType)) {
    return SmallVector<bool>{true, false};
  }

  else if (stablehlo::isSupportedComplexType(elementType)) {
    auto componentTy = cast<ComplexType>(elementType).getElementType();
    return createCursedComplexPool(
        cast<FloatType>(componentTy).getFloatSemantics());
  }

  else {
    llvm::WithColor::warning(diag())
        << "Unsupported element type or bitwidth. Skipping.\n";

    return std::nullopt;
  }

  return std::nullopt;
}

std::optional<mlir::DenseElementsAttr>
generateCursedTensor(mlir::Type argType, std::mt19937 &gen,
                     PoolConstraints &constraints) {
  auto tensorType = llvm::dyn_cast<RankedTensorType>(argType);
  if (!tensorType) {
    return std::nullopt;
  }

  Type elementType = tensorType.getElementType();
  int64_t numElements = tensorType.getNumElements();
  auto cursedNumberVector = CreateCursedPool(elementType);
  if (!cursedNumberVector)
    return std::nullopt;
  mlir::DenseElementsAttr attr;

  std::visit(
      [&](auto &pool) {
        using VectorType = std::decay_t<decltype(pool)>;
        using T = typename VectorType::value_type;
        if constexpr (std::is_same_v<T, APFloat> || std::is_same_v<T, APInt>)
          llvm::erase_if(pool,
                         [&](const T &v) { return !allowed(v, constraints); });
        if (pool.empty())
          return;

        std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);

        if constexpr (std::is_same_v<T, bool>) {
          SmallVector<bool> data;
          data.reserve(numElements);
          for (int64_t i = 0; i < numElements; ++i)
            data.push_back(pool[dist(gen)]);
          attr = mlir::DenseElementsAttr::get(tensorType, data);
        } else {
          SmallVector<T> data;
          data.reserve(numElements);
          for (int64_t i = 0; i < numElements; ++i)
            data.push_back(pool[dist(gen)]);
          attr =
              mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<T>(data));
        }
      },
      *cursedNumberVector);
  if (!attr)
    return std::nullopt;
  return attr;
}

bool isClose(const APFloat &unopt, const APFloat &opt, unsigned maxUlps,
             std::optional<double> rtolOverride = std::nullopt,
             std::optional<double> atolOverride = std::nullopt) {
  const llvm::fltSemantics &sem = unopt.getSemantics();
  // 1. Both are NaN? Match.
  if (unopt.isNaN() && opt.isNaN())
    return true;
  // 2. Both are Inf? Match ONLY if signs are identical.
  if (unopt.isInfinity() && opt.isInfinity())
    return unopt.isNegative() == opt.isNegative();
  // 3. Fast-math overflow allowance:
  // If one is Inf and the other is at the absolute float boundary of the same
  // sign, allow it.
  APFloat threshold = APFloat::getLargest(sem);
  threshold.next(/*nextDown=*/true); // one ulp below max
  auto nearBoundary = [&](const APFloat &v, const APFloat &i) {
    return llvm::abs(v) >= threshold && v.isNegative() == i.isNegative();
  };
  if (opt.isInfinity() && nearBoundary(unopt, opt))
    return true;
  if (unopt.isInfinity() && nearBoundary(opt, unopt))
    return true;
  // 4. Any other mix of Inf/NaN vs finite numbers? Bug.
  if (unopt.isNaN() || opt.isNaN() || unopt.isInfinity() || opt.isInfinity())
    return false;
  // 5. Standard tolerance check
  double relativeUlp = std::ldexp(1.0, 1 - APFloat::semanticsPrecision(sem));
  double rtol = rtolOverride.value_or(relativeUlp * maxUlps);
  double atol = atolOverride.value_or(
      APFloat::getSmallestNormalized(sem).convertToDouble());
  double u = unopt.convertToDouble(), o = opt.convertToDouble();
  return std::abs(u - o) <= (atol + rtol * std::abs(o));
}

int main(int argc, char **argv) {
  bool anyMismatch = false;
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "StableHLO Differential Fuzzer\n");

  uint32_t seed = seedOpt;
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    llvm::WithColor::remark(diag())
        << "Running with random seed: " << seed << "\n";
  } else {
    llvm::WithColor::remark(diag())
        << "Running with manual seed: " << seed << "\n";
  }

  auto [passPipeline, allowUnreg, split] = parseRunLine(inputFilename);
  if (passPipeline.empty()) {
    llvm::WithColor::warning(diag()) << "No RUN line found in file!\n";
    return 2;
  }

  if (split) {
    llvm::WithColor::warning(diag())
        << "Skipping test: Fuzzer does not yet support "
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
    return 2;

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
  if (mlir::failed(mlir::parsePassPipeline(passPipeline, pm, diag()))) {
    llvm::WithColor::error(diag())
        << "Failed to parse the pass pipeline: " << passPipeline << "\n";
    return 2;
  }

  auto BaseConstraints = parseRestrictInput(restrictInput);
  applyConstraintsFromPipeline(passPipeline, BaseConstraints);

  OwningOpRef<ModuleOp> optimizedModule = module->clone();
  if (mlir::failed(pm.run(*optimizedModule))) {
    llvm::WithColor::warning(diag())
        << "Pass pipeline failed to run on module!\n";
    return 1;
  }
  module->walk([&](mlir::func::FuncOp unoptFunc) {
    llvm::StringRef funcName = unoptFunc.getName();

    auto optFunc = optimizedModule->lookupSymbol<mlir::func::FuncOp>(funcName);
    if (!optFunc) {
      llvm::WithColor::warning(diag())
          << "Skipping: Function deleted by optimization.\n";
      return;
    }

    llvm::SmallVector<mlir::DenseElementsAttr> evalArgs;

    for (BlockArgument arg : unoptFunc.getArguments()) {
      auto tensorType = llvm::dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        llvm::WithColor::warning(diag())
            << funcName << ": non-ranked-tensor argument, skipping function\n";
        return;
      }

      int64_t numElements = tensorType.getNumElements();
      if (numElements > maxElements) {
        llvm::WithColor::warning(diag())
            << "Skipping function: argument has " << numElements
            << " elements (exceeds --max-elements limit of " << maxElements
            << ").\n";
        return; // Aborts this function, moves to the next
      }

      PoolConstraints UnoptConstraints = constraintsFromArg(
          optFunc.getArgument(arg.getArgNumber()), BaseConstraints);
      std::optional<mlir::DenseElementsAttr> attrOpt =
          generateCursedTensor(arg.getType(), gen, UnoptConstraints);

      if (!attrOpt) {
        llvm::WithColor::warning(diag())
            << "Skipping non-ranked tensor argument.\n";
        return;
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
      llvm::WithColor::warning(diag())
          << "Legalization failed on unoptimized IR. Skipping.\n";
      return;
    }

    auto unoptResults =
        stablehlo::evalModule(tempUnoptMod.get(), evalArgs, config);
    if (mlir::failed(unoptResults)) {
      llvm::WithColor::warning(diag()) << "Unoptimized evaluation failed.\n";
      return;
    }

    OwningOpRef<ModuleOp> tempOptMod = ModuleOp::create(optFunc.getLoc());
    func::FuncOp clonedOpt = optFunc.clone();
    clonedOpt.setName("main");
    tempOptMod->push_back(clonedOpt); // Push BEFORE running pass

    if (mlir::failed(legalizationPM.run(*tempOptMod))) {
      llvm::WithColor::error(diag())
          << "Legalization failed on optimized IR. Skipping.\n";
      anyMismatch = true;
      return;
    }

    auto optResults = stablehlo::evalModule(tempOptMod.get(), evalArgs, config);
    if (mlir::failed(optResults)) {
      llvm::WithColor::error(diag()) << "Optimized evaluation failed.\n";
      anyMismatch = true;
      return;
    }

    // Comparing results from optimized and unoptimized function
    auto &unoptVals = *unoptResults;
    auto &optVals = *optResults;
    if (unoptVals.size() != optVals.size()) {
      llvm::WithColor::warning(diag())
          << "mismatch different number of return values!\n";
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
        for (auto [u, o] :
             llvm::zip_equal(unoptAttr.getValues<llvm::APFloat>(),
                             optAttr.getValues<llvm::APFloat>())) {
          if (!isClose(u, o, maxUlpsOpt)) {
            llvm::WithColor::error(diag())
                << "float mismatch in " << funcName << " expected " << u
                << " but got " << o << "\n";
            mismatch = true;
            anyMismatch = true;
            break;
          }
        }
      } else if (unoptAttr && optAttr &&
                 llvm::isa<mlir::ComplexType>(unoptAttr.getElementType())) {
        auto uRange = unoptAttr.getValues<mlir::Complex<APFloat>>();
        auto oRange = optAttr.getValues<mlir::Complex<APFloat>>();
        auto uIt = uRange.begin(), uEnd = uRange.end();
        auto oIt = oRange.begin();
        for (; uIt != uEnd; ++uIt, ++oIt) {
          mlir::Complex<APFloat> u = *uIt, o = *oIt;
          auto uReal = u.real();
          auto oReal = o.real();
          auto uImag = u.imag();
          auto oImag = o.imag();

          auto uPrint = PrintableComplex{u};
          auto oPrint = PrintableComplex{o};

          if ((!isClose(uReal, oReal, maxUlpsOpt)) ||
              !isClose(uImag, oImag, maxUlpsOpt)) {
            llvm::WithColor::error(diag())
                << "complex mismatch in " << funcName << " expected " << uPrint
                << " but got " << oPrint << "\n";
            mismatch = true;
            anyMismatch = true;
            break;
          }
        }
      } else {
        // If it's an integer/bool type and failed strict equality, it's a
        // definitive bug.
        llvm::WithColor::error(diag())
            << "mismatch in " << funcName << " expected " << unoptVals[i]
            << " but got " << optVals[i] << "!\n";
        mismatch = true;
        anyMismatch = true;
      }
    }
    if (!mismatch && (verbosity == Verbosity::Verbose)) {
      llvm::WithColor::remark(diag())
          << "passed outputs in " << funcName << " match exactly.\n";
    }
  });
  return anyMismatch ? 1 : 0;
}
