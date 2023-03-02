project( 'RETINAS',
         'c', 'cuda',
         default_options : ['buildtype=plain'],
         version : '0.1.0',
         license : 'BSD-2-Clause license')

cc   = meson.get_compiler('c')
# cuda = meson.get_compiler('cuda')

dep_math = cc.find_library('m', required : true)
dep_fftw = dependency('fftw3')
dep_blas = dependency('blas')
dep_cuda = dependency('cuda', version : '>=10', modules : ['cublas', 'cufft'])

subdir('retinas')

# Set the libraries' C and CUDA source files
lib_c_srcs    = [c_srcs]
lib_cuda_srcs = [cuda_srcs]

# Set the libraries' header files
install_headers(headers, subdir : 'retinas')

# The C library
c_args = ['-O3', '-march=native', '-std=gnu99']
lib_c = library('retinas', lib_c_srcs,
                include_directories : [c_inc_dir],
                implicit_include_directories : true,
                dependencies : [dep_math, dep_fftw, dep_blas],
                c_args : c_args,
                install : true)

# The CUDA library
cuda_args = ['-Xptxas=-O3', '-arch=native', '-O3', '-Xcompiler=-march=native']
lib_cuda = library('retinas_cuda', lib_cuda_srcs,
                   include_directories : [cuda_inc_dir],
                   implicit_include_directories : true,
                   dependencies : [dep_cuda],
                   cuda_args : cuda_args,
                   install : true)
