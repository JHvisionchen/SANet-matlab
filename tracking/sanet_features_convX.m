function [ feat ] = sanet_features_convX(net, rnet, img, boxes, opts)

n = size(boxes,1);
ims = sanet_extract_regions(img, boxes, opts);
nBatches = ceil(n/opts.batchSize_test);

for i=1:nBatches
%     fprintf('extract batch %d/%d...\n',i,nBatches);
    
    batch = ims(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i));
    if(opts.useGpu)
        batch = gpuArray(batch);
    end
    
    res = vl_simplenn(net, rnet, batch, [], [], ...
        'disableDropout', true, ...
        'conserveMemory', true, ...
        'sync', true) ;
    
    f = gather(res(end).x) ;
    if ~exist('feat','var')
        feat = zeros(size(f,1),size(f,2),size(f,3),n,'single');
    end
    feat(:,:,:,opts.batchSize_test*(i-1)+1:min(end,opts.batchSize_test*i)) = f;
end