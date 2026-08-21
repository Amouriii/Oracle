#pragma once

// Assertions that survive a release build.
//
// The suite runs in whatever configuration CMake picked (Release by default),
// where <cassert> compiles assert() away -- taking any side effects inside it
// with it and leaving a test that passes without checking anything.  CHECK is
// always evaluated.

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "CHECK failed: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__);  \
      std::fflush(stderr);                                                                \
      std::abort();                                                                       \
    }                                                                                     \
  } while (0)

// Checks an oracle::Status and prints its message on failure.
#define CHECK_OK(expr)                                                                       \
  do {                                                                                       \
    const auto _st = (expr);                                                                 \
    if (!_st.ok()) {                                                                         \
      std::fprintf(stderr, "CHECK_OK failed: %s\n  %s\n  at %s:%d\n", #expr,                 \
                   _st.message.c_str(), __FILE__, __LINE__);                                 \
      std::fflush(stderr);                                                                   \
      std::abort();                                                                          \
    }                                                                                        \
  } while (0)
