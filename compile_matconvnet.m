run matconvnet/matlab/vl_setupnn ;
cd matconvnet;
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc');
cd ..;