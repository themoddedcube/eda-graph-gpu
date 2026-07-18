// egg_test.h — a tiny, dependency-free check macro for the STA test suite.
//
// We deliberately avoid <cassert>: Release builds define NDEBUG which compiles
// assert() out, so a plain assert-based test would silently pass. CHECK is
// always active and aborts with a located message (nonzero exit -> ctest FAIL).
#pragma once

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "CHECK FAILED: %s\n  at %s:%d\n", #cond,    \
                         __FILE__, __LINE__);                                \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

// Float comparison helper for the hand-computed expectations. All hand-picked
// delays/sums here are small integers exactly representable in float, so the
// default epsilon only guards against accidental rounding, never masks error.
static inline bool approxEq(float a, float b, float eps = 1e-6f) {
    float d = a - b;
    if (d < 0) d = -d;
    return d <= eps;
}
