#include "src/enzyme_ad/jax/Passes/Perfify/LeanCheck.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

/// The axioms a proof may use. Any other, `sorryAx` included, means it is
/// incomplete or assumes more than Lean's logic.
constexpr StringLiteral allowedAxioms[] = {"propext", "Classical.choice",
                                           "Quot.sound"};

/// A Lean file to check, with a `#print axioms` for each theorem.
struct CheckedFile {
  std::string text;
  /// The line of each `#print axioms`, 1-based as in Lean's messages.
  SmallVector<unsigned> printLines;
  /// Lines up to and including the last `import`. An error on one of these
  /// means the imports did not load.
  unsigned headerLines = 0;
};

/// Appends a `#print axioms` for each of `names` to `text`.
static CheckedFile appendAxiomPrints(StringRef text,
                                     ArrayRef<std::string> names) {
  CheckedFile result;
  result.text = text.str();
  if (!text.ends_with('\n'))
    result.text += "\n";

  SmallVector<StringRef> lines;
  text.split(lines, '\n');
  for (auto [index, line] : llvm::enumerate(lines))
    if (line.starts_with("import "))
      result.headerLines = index + 1;

  unsigned line = llvm::count(result.text, '\n');
  for (const std::string &name : names) {
    result.text += "#print axioms Perfify." + name + "\n";
    result.printLines.push_back(++line);
  }
  return result;
}

static std::string axiomsVerdict(StringRef message) {
  if (message.contains("does not depend on any axioms"))
    return "";
  StringRef list = message.split("depends on axioms: [").second;
  if (list.empty())
    return ("unexpected Lean output: " + message).str();

  SmallVector<StringRef> axioms;
  list.take_until([](char c) { return c == ']'; }).split(axioms, ',');
  std::string disallowed;
  for (StringRef axiom : axioms) {
    axiom = axiom.trim();
    if (llvm::is_contained(allowedAxioms, axiom))
      continue;
    disallowed += (disallowed.empty() ? "" : ", ") + axiom.str();
  }
  return disallowed.empty() ? "" : "uses " + disallowed;
}

} // namespace

llvm::Expected<SmallVector<std::string>>
mlir::enzyme::perfify::checkLeanTheorems(StringRef text,
                                         ArrayRef<std::string> names,
                                         StringRef lake, StringRef projectDir) {
  CheckedFile file = appendAxiomPrints(text, names);

  SmallString<128> leanPath;
  if (std::error_code ec =
          llvm::sys::fs::createTemporaryFile("perfify-proof", "lean", leanPath))
    return llvm::createStringError(ec, "cannot create the Lean file");
  llvm::FileRemover removeLean(leanPath);
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(leanPath, ec);
    if (ec)
      return llvm::createStringError(ec, "cannot open " + leanPath);
    os << file.text;
    os.close();
    if (os.has_error()) {
      ec = os.error();
      os.clear_error();
      return llvm::createStringError(ec, "cannot write " + leanPath);
    }
  }

  SmallString<128> outPath;
  if (std::error_code ec =
          llvm::sys::fs::createTemporaryFile("perfify-lean", "json", outPath))
    return llvm::createStringError(ec, "cannot create the Lean output file");
  llvm::FileRemover removeOut(outPath);

  llvm::ErrorOr<std::string> lakePath = llvm::sys::findProgramByName(lake);
  if (!lakePath)
    return llvm::createStringError(lakePath.getError(),
                                   "cannot find '" + lake + "'");

  StringRef args[] = {*lakePath, "-d",     projectDir, "env",
                      "lean",    "--json", leanPath};
  std::optional<StringRef> redirects[] = {StringRef(""), StringRef(outPath),
                                          std::nullopt};
  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*lakePath, args, /*Env=*/std::nullopt,
                                     redirects, /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc < 0)
    return llvm::createStringError("cannot run " + *lakePath + ": " + errMsg);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> out =
      llvm::MemoryBuffer::getFile(outPath);
  if (!out)
    return llvm::createStringError(out.getError(), "cannot read " + outPath);

  llvm::DenseMap<unsigned, std::string> verdicts;
  std::string importErrors;
  bool sawMessage = false;
  SmallVector<StringRef> outLines;
  (*out)->getBuffer().split(outLines, '\n', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
  for (StringRef outLine : outLines) {
    llvm::Expected<llvm::json::Value> value = llvm::json::parse(outLine);
    if (!value) {
      llvm::consumeError(value.takeError());
      continue;
    }
    const llvm::json::Object *msg = value->getAsObject();
    if (!msg)
      continue;
    const llvm::json::Object *pos = msg->getObject("pos");
    std::optional<int64_t> line = pos ? pos->getInteger("line") : std::nullopt;
    std::optional<StringRef> severity = msg->getString("severity");
    std::optional<StringRef> data = msg->getString("data");
    if (!line || !severity || !data)
      continue;
    sawMessage = true;

    if (*severity == "error" && *line <= file.headerLines) {
      importErrors += data->str() + "\n";
      continue;
    }
    if (!llvm::is_contained(file.printLines, *line))
      continue;
    if (*severity == "error")
      verdicts[*line] = data->str();
    else if (*severity == "information")
      verdicts[*line] = axiomsVerdict(*data);
  }

  if (!importErrors.empty())
    return llvm::createStringError("cannot load the imports:\n" + importErrors);
  if (!sawMessage && rc != 0)
    return llvm::createStringError(
        "lake exited with code " + std::to_string(rc) +
        " without running Lean; is " + projectDir + " a Lake project?");

  SmallVector<std::string> results;
  for (unsigned line : file.printLines) {
    auto it = verdicts.find(line);
    results.push_back(it != verdicts.end() ? it->second
                                           : "Lean reported nothing for it");
  }
  return results;
}
