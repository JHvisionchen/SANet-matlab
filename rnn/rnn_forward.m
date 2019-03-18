function [h, o] = rnn_forward(x, param, dropout)
% forward propagating the 3d image tensor, X is of dimension m X n X k.

[k, m, n] = size(x);
x = reshape(x, k, m*n); 

nh = param.nh;
ny = param.ny;
% the feature is column-wise
%% SE plane 
u = param.use;
w = param.wse;
b = param.bse;

hse = u*x + repmat(b, 1, m*n);
hse = reshape(hse,nh,m,n);

for i = 1 : m
    for j = 1 : n
        if i > 1
            hse(:,i,j) = hse(:,i,j) + w*hse(:,i-1,j);
        end
        if j > 1
            hse(:,i,j) = hse(:,i,j) + w*hse(:,i,j-1);
        end
        if i > 1 && j > 1
            hse(:,i,j) = hse(:,i,j) + w*hse(:,i-1,j-1);
        end
          hse(:,i,j) = relu(hse(:,i,j));
    end 
end

hse = reshape(hse, nh, m*n);
o = param.vse * hse + repmat(param.cse,1,m*n);
h.hse = hse;

%% SW plane (SW may share parameters with SE)
u = param.usw;
w = param.wsw;
b = param.bsw;

hsw = u*x + repmat(b, 1, m*n);
hsw = reshape(hsw,nh,m,n);

for i = m :-1: 1
    for j = 1 : n
        if i < m
            hsw(:,i,j) = hsw(:,i,j) + w*hsw(:,i+1,j);
        end
        if j > 1
            hsw(:,i,j) = hsw(:,i,j) + w*hsw(:,i,j-1);
        end
        if i < m && j > 1
            hsw(:,i,j) = hsw(:,i,j) + w*hsw(:,i+1,j-1);
        end
          hsw(:,i,j) = relu(hsw(:,i,j));
    end 
end

hsw = reshape(hsw, nh, m*n);
o = o + param.vsw * hsw;
h.hsw = hsw;

%% NW plane
u = param.unw;
w = param.wnw;
b = param.bnw;

hnw = u*x + repmat(b, 1, m*n);
hnw = reshape(hnw,nh,m,n);

for i = m : -1 : 1
    for j = n : -1 : 1
        if i < m
            hnw(:,i,j) = hnw(:,i,j) + w*hnw(:,i+1,j);
        end
        if j < n
            hnw(:,i,j) = hnw(:,i,j) + w*hnw(:,i,j+1); 
        end
        if i < m && j < n
            hnw(:,i,j) = hnw(:,i,j) + w*hnw(:,i+1,j+1); 
        end
        hnw(:,i,j) = relu(hnw(:,i,j));
    end
end

hnw = reshape(hnw, nh, m*n);
o = o + param.vnw * hnw;
h.hnw = hnw;

%% NE plane
u = param.une;
w = param.wne;
b = param.bne;

hne = u*x + repmat(b, 1, m*n);
hne = reshape(hne,nh,m,n);

for i = 1  : m
    for j = n : -1 : 1
        if i > 1
            hne(:,i,j) = hne(:,i,j) + w*hne(:,i-1,j);
        end
        if j < n
            hne(:,i,j) = hne(:,i,j) + w*hne(:,i,j+1); 
        end
        
        if i > 1 && j < n
            hne(:,i,j) = hne(:,i,j) + w*hne(:,i-1,j+1); 
        end
        hne(:,i,j) = relu(hne(:,i,j));
    end
end
hne = reshape(hne, nh, m*n);
o = o + param.vne * hne;
h.hne = hne;

%% Output
o = x;

% o = softmax(o);

function y = relu(x)
y = max(x, 0);

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));