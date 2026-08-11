#pragma once

#include <cstdint>

extern "C" int32_t enzymexla_lbt_forward(const char *libname, int32_t clear,
                                         int32_t verbose,
                                         const char *suffix_hint);
extern "C" int32_t enzymexla_lbt_get_num_threads();
extern "C" void enzymexla_lbt_set_num_threads(int32_t num_threads);
