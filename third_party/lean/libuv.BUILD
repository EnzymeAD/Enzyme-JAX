# Native Bazel build of libuv for the Lean runtime, replacing its CMake build.
#
# Sources and defines follow libuv's CMakeLists.txt for Linux and macOS; when
# bumping libuv, diff them against the uv_sources/uv_defines lists there.

licenses(["notice"])

exports_files(["LICENSE"])

_UNIX_SRCS = [
    "src/unix/async.c",
    "src/unix/core.c",
    "src/unix/dl.c",
    "src/unix/fs.c",
    "src/unix/getaddrinfo.c",
    "src/unix/getnameinfo.c",
    "src/unix/loop-watcher.c",
    "src/unix/loop.c",
    "src/unix/pipe.c",
    "src/unix/poll.c",
    "src/unix/process.c",
    "src/unix/proctitle.c",
    "src/unix/random-devurandom.c",
    "src/unix/signal.c",
    "src/unix/stream.c",
    "src/unix/tcp.c",
    "src/unix/thread.c",
    "src/unix/tty.c",
    "src/unix/udp.c",
]

cc_library(
    name = "libuv",
    srcs = [
        "src/fs-poll.c",
        "src/idna.c",
        "src/inet.c",
        "src/random.c",
        "src/strscpy.c",
        "src/strtok.c",
        "src/thread-common.c",
        "src/threadpool.c",
        "src/timer.c",
        "src/uv-common.c",
        "src/uv-data-getter-setters.c",
        "src/version.c",
    ] + _UNIX_SRCS + glob([
        "src/*.h",
        "src/unix/*.h",
    ]) + select({
        "@platforms//os:macos": [
            "src/unix/bsd-ifaddrs.c",
            "src/unix/darwin-proctitle.c",
            "src/unix/darwin.c",
            "src/unix/fsevents.c",
            "src/unix/kqueue.c",
            "src/unix/random-getentropy.c",
        ],
        "//conditions:default": [
            "src/unix/linux.c",
            "src/unix/procfs-exepath.c",
            "src/unix/random-getrandom.c",
            "src/unix/random-sysctl-linux.c",
        ],
    }),
    hdrs = glob(["include/**/*.h"]),
    copts = [
        "-fno-strict-aliasing",
        "-w",
    ],
    # src/unix/*.c include "uv-common.h" from src/.
    includes = ["src"],
    linkopts = ["-pthread"] + select({
        "@platforms//os:macos": [],
        "//conditions:default": [
            "-ldl",
            "-lrt",
        ],
    }),
    local_defines = [
        "_FILE_OFFSET_BITS=64",
        "_LARGEFILE_SOURCE",
    ] + select({
        "@platforms//os:macos": [
            "_DARWIN_UNLIMITED_SELECT=1",
            "_DARWIN_USE_64_BIT_INODE=1",
        ],
        "//conditions:default": [
            "_GNU_SOURCE",
            "_POSIX_C_SOURCE=200112",
        ],
    }),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
