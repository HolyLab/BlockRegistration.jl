
/* ©2012-2014 Jian Wang and Timothy E. Holy */

/* CUDA header */
#include <cuda.h>
#include <cufft.h>
#include <math_functions.h>
#include <cuComplex.h>


/* CUDA kernel_conv_components: evaluate thetaA, A and A2, where A is either the fixed or moving image */
// Pitch must be expressed in elements, not bytes!
// width (x) corresponds to the fastest dimension (the same as pitch), depth (z) to the slowest
// The array-dimension parameters must be the same for all 3 arrays.
template <typename T>
__device__ void kernel_conv_components(
  T *A,
  T *A2,
  T *thetaA,
  size_t width, size_t height, size_t depth, size_t pitch) {

  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int idxz = blockIdx.z * blockDim.z + threadIdx.z;

  for (int iz = idxz; iz < depth; iz += gridDim.z * blockDim.z) {
    int offsetz = iz;
    for (int iy = idxy; iy < height; iy += gridDim.y * blockDim.y) {
      int offsety = height*offsetz + iy;
      for (int ix = idxx; ix < width; ix += gridDim.x * blockDim.x) {
        int i = pitch*offsety + ix;
        T local_A = A[i];
        int local_thetaA = isnan(local_A);
        if (local_thetaA) {
          local_A = 0.0;
          A[i] = local_A;
        }
        thetaA[i] = !local_thetaA;
        A2[i] = local_A * local_A;
      }
    }
  }
}

__host__ __device__ static __inline__ cuComplex conj(cuComplex x) {return cuConjf(x);}
__host__ __device__ static __inline__ cuDoubleComplex conj(cuDoubleComplex x) {return cuConj(x);}
__host__ __device__ static __inline__ cuComplex cmul(cuComplex x, cuComplex y) {return cuCmulf(x,y);}
__host__ __device__ static __inline__ cuDoubleComplex cmul(cuDoubleComplex x, cuDoubleComplex y) {return cuCmul(x,y);}
__host__ __device__ static __inline__ cuComplex cadd(cuComplex x, cuComplex y) {return cuCaddf(x,y);}
__host__ __device__ static __inline__ cuDoubleComplex cadd(cuDoubleComplex x, cuDoubleComplex y) {return cuCadd(x,y);}
__host__ __device__ static __inline__ cuComplex csub(cuComplex x, cuComplex y) {return cuCsubf(x,y);}
__host__ __device__ static __inline__ cuDoubleComplex csub(cuDoubleComplex x, cuDoubleComplex y) {return cuCsub(x,y);}
//__host__ __device__ static __inline__ cuComplex make_complex(float x, float y) {return make_cuFloatComplex(x,y);}
//__host__ __device__ static __inline__ cuDoubleComplex make_complex(double x, double y) {return make_cuDoubleComplex(x,y);}

/* CUDA kernel_calcNumDenom: case INTENSITY, compute numerator and denominator before ifftn */
// See comments above about array dimensions
template <typename T>
__device__ void kernel_calcNumDenom_intensity(
  T *f_fft,
  T *f2_fft,
  T *thetaf_fft,
  T *m_fft,
  T *m2_fft,
  T *thetam_fft,
  T *numerator_fft,
  T *denominator_fft,
  size_t width, size_t height, size_t depth, size_t pitch) {

  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int idxz = blockIdx.z * blockDim.z + threadIdx.z;

  T complex2;
  complex2.x = 2;
  complex2.y = 0;

  for (int iz = idxz; iz < depth; iz += gridDim.z * blockDim.z) {
    int offsetz = iz;
    for (int iy = idxy; iy < height; iy += gridDim.y * blockDim.y) {
      int offsety = height*offsetz + iy;
      for (int ix = idxx; ix < width; ix += gridDim.x * blockDim.x) {
        int i = pitch*offsety + ix;
        T c1 = cmul(conj(f2_fft[i]), thetam_fft[i]);
        T c2 = cmul(conj(thetaf_fft[i]), m2_fft[i]);
        c1 = cadd(c1, c2);
        T c3 = cmul(conj(f_fft[i]), m_fft[i]);
        c2 = cmul(complex2, c3);
        numerator_fft[i] = csub(c1, c2);
        denominator_fft[i] = c1;
      }
    }
  }
}


/* CUDA kernel_calcNumDenom: case PIXELS, compute numerator and denominator before ifftn */
template <typename T>
__device__ void kernel_calcNumDenom_pixels(
  T *f_fft,
  T *f2_fft,
  T *thetaf_fft,
  T *m_fft,
  T *m2_fft,
  T *thetam_fft,
  T *numerator_fft,
  T *denominator_fft,
  size_t width, size_t height, size_t depth, size_t pitch) {

  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int idxz = blockIdx.z * blockDim.z + threadIdx.z;

  T complexm2;
  complexm2.x = -2;
  complexm2.y = 0;

  for (int iz = idxz; iz < depth; iz += gridDim.z * blockDim.z) {
    int offsetz = iz;
    for (int iy = idxy; iy < height; iy += gridDim.y * blockDim.y) {
      int offsety = height*offsetz + iy;
      for (int ix = idxx; ix < width; ix += gridDim.x * blockDim.x) {
        int i = pitch*offsety + ix;
        T c1 = cmul(complexm2, conj(f_fft[i]));
        c1 = cmul(c1, m_fft[i]);
        T c2 = cmul(conj(thetaf_fft[i]), m2_fft[i]);
        c1 = cadd(c1, c2);
        c2 = cmul(conj(f2_fft[i]), thetam_fft[i]);
        numerator_fft[i] = cadd(c1, c2);
        denominator_fft[i] = cmul(conj(thetaf_fft[i]), thetam_fft[i]);
      }
    }
  }
}

// Frequency-domain fftshift
// Run this before you ifft
template <typename T, typename R>
__device__ void kernel_fdshift(
  T *data1fft, T *data2fft,
  R shift1, R shift2, R shift3, size_t N1,
  size_t width, size_t height, size_t depth, size_t pitch, size_t normalize) {

  int idxx = blockIdx.x * blockDim.x + threadIdx.x;
  int idxy = blockIdx.y * blockDim.y + threadIdx.y;
  int idxz = blockIdx.z * blockDim.z + threadIdx.z;

  T phase;
  R tau = 6.283185307179586;
  R c1 = (tau*shift1)/N1;
  R c2 = (tau*shift2)/height;
  R c3 = (tau*shift3)/depth;

  for (int iz = idxz; iz < depth; iz += gridDim.z * blockDim.z) {
    int offsetz = iz;
    for (int iy = idxy; iy < height; iy += gridDim.y * blockDim.y) {
      int offsety = height*offsetz + iy;
      for (int ix = idxx; ix < width; ix += gridDim.x * blockDim.x) {
        int i = pitch*offsety + ix;
	R arg = c1*ix + c2*iy + c3*iz;
	phase.x = cos(arg)/normalize;
	phase.y = sin(arg)/normalize;
	data1fft[i] = cmul(data1fft[i], phase);
	data2fft[i] = cmul(data2fft[i], phase);
	//data2fft[i] = make_complex(c1*ix,c2*iy);
      }
    }
  }
}


// Explicit instantiation
extern "C"
{
  __global__ void kernel_conv_components_double(double *A, double *A2, double *thetaA,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_conv_components(A, A2, thetaA, width, height, depth, pitch);
  }
  __global__ void kernel_conv_components_float(float *A, float *A2, float *thetaA,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_conv_components(A, A2, thetaA, width, height, depth, pitch);
  }
  __global__ void kernel_calcNumDenom_intensity_double(
      cufftDoubleComplex *f_fft,
      cufftDoubleComplex *f2_fft,
      cufftDoubleComplex *thetaf_fft,
      cufftDoubleComplex *m_fft,
      cufftDoubleComplex *m2_fft,
      cufftDoubleComplex *thetam_fft,
      cufftDoubleComplex *numerator_fft,
      cufftDoubleComplex *denominator_fft,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_calcNumDenom_intensity(f_fft, f2_fft, thetaf_fft, m_fft, m2_fft,
                           thetam_fft, numerator_fft, denominator_fft,
                           width, height, depth, pitch);
  }
  __global__ void kernel_calcNumDenom_intensity_float(
      cufftComplex *f_fft,
      cufftComplex *f2_fft,
      cufftComplex *thetaf_fft,
      cufftComplex *m_fft,
      cufftComplex *m2_fft,
      cufftComplex *thetam_fft,
      cufftComplex *numerator_fft,
      cufftComplex *denominator_fft,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_calcNumDenom_intensity(f_fft, f2_fft, thetaf_fft, m_fft, m2_fft,
                           thetam_fft, numerator_fft, denominator_fft,
                           width, height, depth, pitch);
  }
  __global__ void kernel_calcNumDenom_pixels_double(
      cufftDoubleComplex *f_fft,
      cufftDoubleComplex *f2_fft,
      cufftDoubleComplex *thetaf_fft,
      cufftDoubleComplex *m_fft,
      cufftDoubleComplex *m2_fft,
      cufftDoubleComplex *thetam_fft,
      cufftDoubleComplex *numerator_fft,
      cufftDoubleComplex *denominator_fft,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_calcNumDenom_pixels(f_fft, f2_fft, thetaf_fft, m_fft, m2_fft,
                           thetam_fft, numerator_fft, denominator_fft,
                           width, height, depth, pitch);
  }
  __global__ void kernel_calcNumDenom_pixels_float(
      cufftComplex *f_fft,
      cufftComplex *f2_fft,
      cufftComplex *thetaf_fft,
      cufftComplex *m_fft,
      cufftComplex *m2_fft,
      cufftComplex *thetam_fft,
      cufftComplex *numerator_fft,
      cufftComplex *denominator_fft,
      size_t width, size_t height, size_t depth, size_t pitch) {
    kernel_calcNumDenom_pixels(f_fft, f2_fft, thetaf_fft, m_fft, m2_fft,
                           thetam_fft, numerator_fft, denominator_fft,
                           width, height, depth, pitch);
  }
  __global__ void kernel_fdshift_double(
      cufftDoubleComplex *numfft, cufftDoubleComplex *denomfft,
      double shift1, double shift2, double shift3, size_t N1,
      size_t width, size_t height, size_t depth, size_t pitch, size_t normalize) {
    kernel_fdshift<cufftDoubleComplex,double>(numfft, denomfft, shift1, shift2, shift3,
					      N1, width, height, depth, pitch, normalize);
  }
  __global__ void kernel_fdshift_float(
      cufftComplex *numfft, cufftComplex *denomfft,
      float shift1, float shift2, float shift3, size_t N1,
      size_t width, size_t height, size_t depth, size_t pitch, size_t normalize) {
    kernel_fdshift<cufftComplex,float>(numfft, denomfft, shift1, shift2, shift3,
				       N1, width, height, depth, pitch, normalize);
  }
}
