#ifndef JACOBI2D_CUH
# define JACOBI2D_CUH

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(MAX_DATASET)
#  define EXTRALARGE_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(TSTEPS) && !defined(N)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define TSTEPS 1
#   define N 4
#  endif

#  ifdef SMALL_DATASET
#   define TSTEPS 10
#   define N 500
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define TSTEPS 20
#   define N 1000
#  endif

#  ifdef LARGE_DATASET
#   define TSTEPS 20
#   define N 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define TSTEPS 100
#   define N 4000
#  endif

#  ifdef MAX_DATASET
#   define TSTEPS 100
#   define N 10000
#  endif
# endif /* !N */

# define _PB_TSTEPS POLYBENCH_LOOP_BOUND(TSTEPS,tsteps)
# define _PB_N POLYBENCH_LOOP_BOUND(N,n)

# ifndef DATA_TYPE
#  define DATA_TYPE double
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

#endif /* !JACOBI2D*/