// Tests tearing the linkable XLA runtime down at process exit, the way code
// compiled against it does.

#include "src/enzyme_ad/jax/ReactantExtra/test/reactant_extra_test.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

// The GDB JIT interface, which the library defines: LLVM's GDB registration
// listener keeps the code it is notified of in this list.
extern "C" {
struct jit_code_entry;
struct jit_descriptor {
  uint32_t version;
  uint32_t action_flag;
  jit_code_entry *relevant_entry;
  jit_code_entry *first_entry;
};
extern jit_descriptor __jit_debug_descriptor;
}

using ::reactant_test::XlaMemcpyDirection;
using ::reactant_test::XlaPrimitiveType;

namespace {

// The runtime slot and the persistent temporary of the "compiled" code.
LinkableRuntime *runtime = nullptr;
void *persistent_temporary = nullptr;
constexpr uint64_t kNumElements = 4;
// Set once the runtime holds a compiled executable.
bool compiled_executable = false;

bool HasRegisteredJITCode() {
  return __jit_debug_descriptor.first_entry != nullptr;
}

// Exit codes of the child process.
enum ExitCode : int {
  kSuccess = 0,
  kWrongResult = 1,
  kCodeNotRegistered = 2,
  kListenerDestroyedBeforeTeardown = 3,
  kCodeRegisteredAfterTeardown = 4,
};

// Exits immediately: this runs inside exit, which must not be reentered.
void FailTeardown(ExitCode code, const char *message) {
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  _exit(code);
}

void DeinitializeRuntime() {
  // The listener unregisters the code that is left when it is destroyed, so
  // code is registered here only if the listener outlives the runtime. An
  // implementation of std::recursive_mutex with a trivial destructor hides the
  // use of the destroyed listener otherwise.
  if (compiled_executable && !HasRegisteredJITCode())
    FailTeardown(kListenerDestroyedBeforeTeardown,
                 "JIT event listener destroyed before the runtime teardown");
  reactantXLADeInit(&runtime);
  if (HasRegisteredJITCode())
    FailTeardown(kCodeRegisteredAfterTeardown,
                 "runtime teardown left JIT code registered");
}

void ReleaseTemporary() {
  reactantXLAFree(&runtime, persistent_temporary);
  persistent_temporary = nullptr;
}

// These mirror the constructors that convert-polygeist-to-llvm emits for the
// XLA backends: each registers its teardown with atexit once it is done, the
// runtime first and the persistent temporaries second.
__attribute__((constructor(65534))) void InitializeRuntime() {
  reactantXLAInit(&runtime, "cpu");
  atexit(DeinitializeRuntime);
}

__attribute__((constructor(65535))) void AllocateTemporary() {
  uint64_t num_elements = kNumElements;
  persistent_temporary = reactantXLAMalloc(&runtime, XlaPrimitiveType::kF32,
                                           /*rank=*/1, &num_elements);
  atexit(ReleaseTemporary);
}

// Compiles and runs a module, so that the runtime holds an executable when
// the process exits, and exits. Exits with a nonzero code on a wrong result or
// if the executable is not registered with the listener.
[[noreturn]] void ExecuteAndExit() {
  static constexpr char kDoubleModule[] = R"mlir(
    func.func @main(%x: tensor<?xf32>) -> tensor<?xf32> {
      %sum = stablehlo.add %x, %x : tensor<?xf32>
      return %sum : tensor<?xf32>
    }
  )mlir";

  size_t size_in_bytes = kNumElements * sizeof(float);
  std::vector<float> host_input = {1.5f, 2.5f, -3.0f, 4.25f};
  reactantXLAMemcpy(&runtime, persistent_temporary, host_input.data(),
                    size_in_bytes, XlaMemcpyDirection::kMemcpyHostToDevice);

  void *arguments[1] = {persistent_temporary};
  reactantXLAExec(&runtime, kDoubleModule, /*num_arguments=*/1, arguments,
                  /*num_constants=*/0, /*constants=*/nullptr);

  std::vector<float> host_output(kNumElements, 0.0f);
  reactantXLAMemcpy(&runtime, host_output.data(), persistent_temporary,
                    size_in_bytes, XlaMemcpyDirection::kMemcpyDeviceToHost);
  std::vector<float> expected = {3.0f, 5.0f, -6.0f, 8.5f};
  if (host_output != expected)
    exit(kWrongResult);
  if (!HasRegisteredJITCode())
    exit(kCodeNotRegistered);
  compiled_executable = true;
  exit(kSuccess);
}

} // namespace

TEST(LinkableRuntimeExitTest, ConstructorsInitializeRuntimeAndTemporary) {
  EXPECT_NE(runtime, nullptr);
  EXPECT_NE(persistent_temporary, nullptr);
}

// The child process exits with a compiled executable in the runtime. It exits
// with a nonzero code, or dies, if tearing the runtime down at exit uses
// anything that was already destroyed.
TEST(LinkableRuntimeExitDeathTest, TearsDownCompiledExecutableAtExit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(ExecuteAndExit(), ::testing::ExitedWithCode(kSuccess), "");
}
