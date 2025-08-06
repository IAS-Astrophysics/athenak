#ifndef MAYBE_UNUSED
#if __cplusplus >= 201703L
#define MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__) || defined(__NVCC__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif // END check for GCC, Clang, or NVCC
#endif // END MAYBE_UNUSED

// If compiled with AVX512F SIMD instructions enabled:
#ifdef __AVX512F__
#include <immintrin.h>

// SIMD type and width definitions
#define REAL_SIMD_ARRAY __m512d
#define simd_width 8 // 8 doubles per loop iteration

// Load and Store Operations
#define ReadSIMD(a) _mm512_loadu_pd(a)
#define WriteSIMD(a, b) _mm512_storeu_pd(a, (b))

// Constant Initialization
#define ConstSIMD(a) _mm512_set1_pd(a)

// Basic Arithmetic Operations
#define AddSIMD(a, b) _mm512_add_pd((a), (b))
#define SubSIMD(a, b) _mm512_sub_pd((a), (b))
#define MulSIMD(a, b) _mm512_mul_pd((a), (b))
#define DivSIMD(a, b) _mm512_div_pd((a), (b))

// Mathematical Functions
#define SqrtSIMD(a) _mm512_sqrt_pd((a))
#define ExpSIMD(a) _mm512_exp_pd((a))
#define SinSIMD(a) _mm512_sin_pd((a))
#define CosSIMD(a) _mm512_cos_pd((a))

// Fused Multiply-Add/Subtract Operations
#define FusedMulAddSIMD(a, b, c) _mm512_fmadd_pd((a), (b), (c))
#define FusedMulSubSIMD(a, b, c) _mm512_fmsub_pd((a), (b), (c))
#define NegFusedMulAddSIMD(a, b, c) _mm512_fnmadd_pd((a), (b), (c))
#define NegFusedMulSubSIMD(a, b, c) _mm512_fnmsub_pd((a), (b), (c))

// Absolute Value
#define AbsSIMD(a) _mm512_and_pd((a), _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF)))

// Initialize vector to zero (output is REAL_SIMD_ARRAY)
#define SetZeroSIMD _mm512_setzero_pd()
// Horizontal addition (output is a double)
#define HorizAddSIMD(vec_sum) _mm512_reduce_add_pd(vec_sum)

// Upwind Algorithm for Conditional Selection
// The result of this comparison is: result[i] = (a OP b) ? 1 : 0, stored in an 8-bit mask array.
// If result == 1, set upwind = 0 + 1; if result == 0, set upwind = 0
#define UPWIND_ALG(a) _mm512_mask_add_pd(upwind_Integer_0, _mm512_cmp_pd_mask((a), upwind_Integer_0, _CMP_GT_OQ), upwind_Integer_0, upwind_Integer_1)

// If compiled with AVX SIMD instructions enabled:
#elif __AVX__
#include <immintrin.h>
#define REAL_SIMD_ARRAY __m256d
#define simd_width 4 // 4 doubles per loop iteration

// Upwind algorithm notes for 256-bit SIMD:
// Sources: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_cmp_pd&expand=736
//          https://stackoverflow.com/questions/37099874/is-avx-intrinsic-mm256-cmp-ps-supposed-to-return-nan-when-true
// The result from _mm256_cmp_pd is 0 if a > b and NaN otherwise.
// If the comparison operator (OP) is >, then: if a > b, the result is NaN; if a <= b, the result is 0.
// We want the result to be 1 if a > b and 0 otherwise, so we perform a logical AND operation
// on the result against the number 1. This works because AND(NaN, 1) = 1, and AND(0, 1) = 0,
// where NaN is represented as 0xFFFFFFFFFFFFFFFF in double precision.
#define UPWIND_ALG(a) _mm256_and_pd(_mm256_cmp_pd((a), upwind_Integer_0, _CMP_GT_OQ), upwind_Integer_1)

// Absolute Value
#define AbsSIMD(a) _mm256_andnot_pd(_mm256_set1_pd(-0.0), (a))

// Load and Store Operations
#define ReadSIMD(a) _mm256_loadu_pd(a)
#define WriteSIMD(a, b) _mm256_storeu_pd(a, (b))

// Constant Initialization
#define ConstSIMD(a) _mm256_set1_pd(a)

// Basic Arithmetic Operations
#define AddSIMD(a, b) _mm256_add_pd((a), (b))
#define SubSIMD(a, b) _mm256_sub_pd((a), (b))
#define MulSIMD(a, b) _mm256_mul_pd((a), (b))
#define DivSIMD(a, b) _mm256_div_pd((a), (b))

// Mathematical Functions
#define SqrtSIMD(a) _mm256_sqrt_pd((a))
#define ExpSIMD(a) _mm256_exp_pd((a))
#define SinSIMD(a) _mm256_sin_pd((a))
#define CosSIMD(a) _mm256_cos_pd((a))

// Fused Multiply-Add/Subtract Operations with FMA
#ifdef __FMA__
#define FusedMulAddSIMD(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#define FusedMulSubSIMD(a, b, c) _mm256_fmsub_pd((a), (b), (c))
#define NegFusedMulAddSIMD(a, b, c) _mm256_fnmadd_pd((a), (b), (c))
#define NegFusedMulSubSIMD(a, b, c) _mm256_fnmsub_pd((a), (b), (c))

// Fused Multiply-Add/Subtract Operations without FMA
#else
#define FusedMulAddSIMD(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))    // a*b + c
#define FusedMulSubSIMD(a, b, c) _mm256_sub_pd(_mm256_mul_pd((a), (b)), (c))    // a*b - c
#define NegFusedMulAddSIMD(a, b, c) _mm256_sub_pd((c), _mm256_mul_pd((a), (b))) // c - a*b

// NegFusedMulSubSIMD(a, b, c) = -a*b - c
//                              = c - (c + a*b + c)
//                              = SubSIMD(c, AddSIMD(c, AddSIMD(MulSIMD(a, b), c)))
#define NegFusedMulSubSIMD(a, b, c) _mm256_sub_pd((c), _mm256_add_pd((c), _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))))
#endif

// Initialize vector to zero (output is REAL_SIMD_ARRAY)
#define SetZeroSIMD _mm256_setzero_pd()
// Horizontal addition (output is a double). Inpsired by https://www.ed-yang.com/worknotes/docs/manual-vectorization/sumreduce/
#define HorizAddSIMD(vec)                                                                                                                            \
  ({                                                                                                                                                 \
    __m128d low = _mm256_castpd256_pd128(vec);                                 /* Extract lower 128 bits */                                          \
    __m128d high = _mm256_extractf128_pd(vec, 1);                              /* Extract upper 128 bits */                                          \
    __m128d sum_128 = _mm_add_pd(low, high);                                   /* Add low and high parts */                                          \
    _mm_cvtsd_f64(sum_128) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum_128, sum_128)); /* Final scalar sum */                                                \
  })

// If compiled with SSE2 SIMD instructions enabled:
#elif __SSE2__
#include <emmintrin.h>
#define REAL_SIMD_ARRAY __m128d
#define simd_width 2 // 2 doubles per loop iteration

// Absolute Value
#define AbsSIMD(a) _mm_andnot_pd(_mm_set1_pd(-0.0), (a))

// Load and Store Operations
#define ReadSIMD(a) _mm_loadu_pd(a)
#define WriteSIMD(a, b) _mm_storeu_pd(a, (b))

// Constant Initialization
#define ConstSIMD(a) _mm_set1_pd(a)

// Basic Arithmetic Operations
#define AddSIMD(a, b) _mm_add_pd((a), (b))
#define SubSIMD(a, b) _mm_sub_pd((a), (b))
#define MulSIMD(a, b) _mm_mul_pd((a), (b))
#define DivSIMD(a, b) _mm_div_pd((a), (b))

// Mathematical Functions
#define SqrtSIMD(a) _mm_sqrt_pd((a))
#define ExpSIMD(a) _mm_exp_pd((a))
#define SinSIMD(a) _mm_sin_pd((a))
#define CosSIMD(a) _mm_cos_pd((a))

// Upwind Algorithm
// See the description above UPWIND_ALG for __AVX__.
// For SSE2, _mm_cmpgt_pd compares two double-precision values and returns 1.0 for true, 0.0 for false.
// We use AND to achieve the desired conditional logic, setting upwind to 1 if a > b, and 0 otherwise.
#define UPWIND_ALG(a) _mm_and_pd(_mm_cmpgt_pd((a), upwind_Integer_0), upwind_Integer_1)

#ifdef __FMA__
// Fused Multiply-Add/Subtract Operations with FMA
// Note: There are no mainstream non-AVX+ chips that have FMA, but we include these for completeness.
#define FusedMulAddSIMD(a, b, c) _mm_fmadd_pd((a), (b), (c))
#define FusedMulSubSIMD(a, b, c) _mm_fmsub_pd((a), (b), (c))
#define NegFusedMulAddSIMD(a, b, c) _mm_sub_pd((c), _mm_mul_pd((a), (b)))                                   // c - a*b
#define NegFusedMulSubSIMD(a, b, c) _mm_sub_pd((c), _mm_sub_pd((c), _mm_sub_pd(_mm_mul_pd((a), (b)), (c)))) // -a*b - c = c - (c + a*b + c)
#else
// Fused Multiply-Add/Subtract Operations without FMA
#define FusedMulAddSIMD(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))    // a*b + c
#define FusedMulSubSIMD(a, b, c) _mm_sub_pd(_mm_mul_pd((a), (b)), (c))    // a*b - c
#define NegFusedMulAddSIMD(a, b, c) _mm_sub_pd((c), _mm_mul_pd((a), (b))) // c - a*b
// NegFusedMulSubSIMD(a,b,c) = -a*b - c
//                            = c - (c + a*b + c)
//                            = SubSIMD(c, AddSIMD(c, AddSIMD(MulSIMD(a, b), c)))
#define NegFusedMulSubSIMD(a, b, c) _mm_sub_pd((c), _mm_add_pd((c), _mm_add_pd(_mm_mul_pd((a), (b)), (c))))
#endif

// Initialize vector to zero (output is REAL_SIMD_ARRAY)
#define SetZeroSIMD _mm_setzero_pd()
// Horizontal addition (output is a double)
#ifdef __SSE3__
#include <pmmintrin.h>
// SSE3-enabled horizontal addition
#define HorizAddSIMD(vec_sum)                                                                                                                        \
  ({                                                                                                                                                 \
    __m128d sum_128 = _mm_hadd_pd(vec_sum, vec_sum); /* Perform horizontal addition */                                                               \
    _mm_cvtsd_f64(sum_128);                          /* Extract final scalar sum */                                                                  \
  })

#else
// Pure SSE2 horizontal addition (without SSE3)
#define HorizAddSIMD(vec_sum)                                                                                                                        \
  ({                                                                                                                                                 \
    __m128d temp = _mm_shuffle_pd(vec_sum, vec_sum, 0x1); /* Swap the two 64-bit lanes */                                                            \
    __m128d sum_128 = _mm_add_pd(vec_sum, temp);          /* Add the swapped lanes */                                                                \
    _mm_cvtsd_f64(sum_128);                               /* Extract final scalar sum */                                                             \
  })
#endif // __SSE3__

#else
// If SIMD instructions are unavailable:
#define REAL_SIMD_ARRAY REAL
#define simd_width 1 // 1 double per loop iteration

// Basic Operations (Scalar)
#define ConstSIMD(a) (a)
#define AbsSIMD(a) (fabs(a))
#define AddSIMD(a, b) ((a) + (b))
#define SubSIMD(a, b) ((a) - (b))
#define MulSIMD(a, b) ((a) * (b))
#define DivSIMD(a, b) ((a) / (b))

// Fused Multiply-Add/Subtract Operations (Scalar)
#define FusedMulAddSIMD(a, b, c) ((a) * (b) + (c))
#define FusedMulSubSIMD(a, b, c) ((a) * (b) - (c))
#define NegFusedMulAddSIMD(a, b, c) ((c) - (a) * (b))
#define NegFusedMulSubSIMD(a, b, c) (-((a) * (b) + (c))) // -a*b - c = -(a*b + c)

// Mathematical Functions (Scalar)
#define SqrtSIMD(a) (sqrt(a))
#define ExpSIMD(a) (exp(a))
#define SinSIMD(a) (sin(a))
#define CosSIMD(a) (cos(a))

// Load and Store Operations (Scalar)
#define WriteSIMD(a, b) *(a) = (b)
#define ReadSIMD(a) *(a)

// Upwind Algorithm (Scalar Version)
// *NOTE*: This upwinding is reversed from usual upwinding algorithms,
// because the upwinding control vector in BSSN (the shift)
// acts like a *negative* velocity.
#define UPWIND_ALG(UpwindVecU) ((UpwindVecU) > 0.0 ? 1.0 : 0.0)

// Initialize vector (of size one) to zero (output is REAL_SIMD_ARRAY)
#define SetZeroSIMD 0.0
// Horizontal addition (output is a double)
#define HorizAddSIMD(a) (a) // For scalar fallback, no horizontal addition needed

#endif
