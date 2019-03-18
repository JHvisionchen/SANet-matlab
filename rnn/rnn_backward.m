function [dx, dz] = rnn_backward(x, h, delta_dzds, param)

[k,m,n] = size(x);

nx = param.nx;
nh = param.nh;
ny = param.ny;

% dzds = gsoftmax(y,o);
dzds = delta_dzds;

delta = zeros(nh, m, n, 'single');

dx = zeros(k, m, n, 'single');

% One update for one image, less memory cosumption
%% SE plane
dzdh = zeros(nh, m, n, 'single');
dz.duse = zeros(nh,nx, 'single');
dz.dwse = zeros(nh,nh,'single');
dz.dvse = zeros(ny,nh,'single');
dz.dbse = zeros(nh,1,'single');
dz.dcse = zeros(ny,1,'single');

w = param.wse;
v = param.vse;
u = param.use;

hse = h.hse;
dzds = reshape(dzds, ny, m*n);
hse = reshape(hse, nh, m*n);
dz.dvse = dzds * hse';
dz.dcse = dzds * ones(m*n,1);
dzds = reshape(dzds,ny, m, n);
hse = reshape(hse, nh, m, n);

for i = m :-1: 1
    for j = n :-1: 1
        dzdh(:,i,j) = dzdh(:,i,j) + v'*dzds(:,i,j);
        dzdf =  dzdh(:,i,j) .* grelu(hse(:,i,j));
%         dzdf =  dzdh(:,i,j) .* gsigmoid(hse(:,i,j));
   
        delta(:,i,j) = delta(:,i,j) + dzdf;
 
        dz.dbse = dz.dbse + dzdf;
        dz.duse = dz.duse + dzdf*x(:,i,j)';
        
        dx(:,i,j) = dx(:,i,j) + u'*dzdf;
        
        dzdhh = w'*dzdf;
        if i > 1
            dzdh(:,i-1,j) = dzdh(:,i-1,j) + dzdhh;
            dz.dwse = dz.dwse + dzdf*hse(:,i-1,j)';
        end
        if j > 1
            dzdh(:,i,j-1) = dzdh(:,i,j-1) + dzdhh;
            dz.dwse = dz.dwse + dzdf*hse(:,i,j-1)';
        end
        if i > 1 && j > 1
            dzdh(:,i-1,j-1) = dzdh(:,i-1,j-1) + dzdhh;
            dz.dwse = dz.dwse + dzdf*hse(:,i-1,j-1)';
        end
    end
end

%% SW plane
dzdh = zeros(nh, m, n, 'single');
dz.dusw = zeros(nh,nx, 'single');
dz.dwsw = zeros(nh,nh,'single');
dz.dvsw = zeros(ny,nh,'single');
dz.dbsw = zeros(nh,1,'single');

w = param.wsw;
v = param.vsw;
u = param.usw;

hsw = h.hsw;
dzds = reshape(dzds, ny, m*n);
hsw = reshape(hsw, nh, m*n);
dz.dvsw = dzds * hsw';
dzds = reshape(dzds,ny, m, n);
hsw = reshape(hsw, nh, m, n);

for i = 1 : m
    for j = n :-1: 1
        dzdh(:,i,j) = dzdh(:,i,j) + v'*dzds(:,i,j);
        dzdf =  dzdh(:,i,j) .* grelu(hsw(:,i,j));
        
        delta(:,i,j) = delta(:,i,j) + dzdf;
        
        dz.dbsw = dz.dbsw + dzdf;
        dz.dusw = dz.dusw + dzdf*x(:,i,j)';
        
        dx(:,i,j) = dx(:,i,j) + u'*dzdf;
        
        dzdhh = w'*dzdf;
        if i < m
            dzdh(:,i+1,j) = dzdh(:,i+1,j) + dzdhh;
            dz.dwsw = dz.dwsw + dzdf*hsw(:,i+1,j)';
        end
        if j > 1
            dzdh(:,i,j-1) = dzdh(:,i,j-1) + dzdhh;
            dz.dwsw = dz.dwsw + dzdf*hsw(:,i,j-1)';
        end
        if i < m && j > 1
            dzdh(:,i+1,j-1) = dzdh(:,i+1,j-1) + dzdhh;
            dz.dwsw = dz.dwsw + dzdf*hsw(:,i+1,j-1)';
        end
    end
end

%% NW plane
dzdh = zeros(nh, m, n, 'single');
dz.dunw = zeros(nh,nx, 'single');
dz.dwnw = zeros(nh,nh,'single');
dz.dvnw = zeros(ny,nh,'single');
dz.dbnw = zeros(nh,1,'single');

w = param.wnw;
v = param.vnw;
u = param.unw;

hnw = h.hnw;
% dzds = gsoftmax(y,o);
dzds = reshape(dzds, ny, m*n);
hnw = reshape(hnw, nh, m*n);
dz.dvnw = dzds * hnw';
dzds = reshape(dzds,ny, m, n);
hnw = reshape(hnw, nh, m, n);

for i = 1 : m
    for j = 1 : n
        dzdh(:,i,j) = dzdh(:,i,j) + v'*dzds(:,i,j);
        dzdf = dzdh(:,i,j) .* grelu(hnw(:,i,j));
        
        delta(:,i,j) = delta(:,i,j) + dzdf;
        
        dz.dbnw = dz.dbnw + dzdf;
        dz.dunw = dz.dunw + dzdf*x(:,i,j)';
        
        dx(:,i,j) = dx(:,i,j) + u'*dzdf;
        
        dzdhh = w'*dzdf;
        if i < m
            dzdh(:,i+1,j) = dzdh(:,i+1,j) + dzdhh;
            dz.dwnw = dz.dwnw + dzdf*hnw(:,i+1,j)';
        end
        if j < n
            dzdh(:,i,j+1) = dzdh(:,i,j+1) + dzdhh;
            dz.dwnw = dz.dwnw + dzdf*hnw(:,i,j+1)';
        end
        if i < m && j < n
            dzdh(:,i+1,j+1) = dzdh(:,i+1,j+1) + dzdhh;
            dz.dwnw = dz.dwnw + dzdf*hnw(:,i+1,j+1)';
        end
    end 
end

%% NE plane
dzdh = zeros(nh, m, n, 'single');
dz.dune = zeros(nh,nx, 'single');
dz.dwne = zeros(nh,nh,'single');
dz.dvne = zeros(ny,nh,'single');
dz.dbne = zeros(nh,1,'single');

w = param.wne;
v = param.vne;
u = param.une;

hne = h.hne;
dzds = reshape(dzds, ny, m*n);
hne = reshape(hne, nh, m*n);
dz.dvne = dzds * hne';
dzds = reshape(dzds,ny, m, n);
hne = reshape(hne, nh, m, n);

for i = m : -1 : 1
    for j = 1 : n
        dzdh(:,i,j) = dzdh(:,i,j) + v'*dzds(:,i,j);
        dzdf = dzdh(:,i,j) .* grelu(hne(:,i,j));
        
        delta(:,i,j) = delta(:,i,j) + dzdf;
        
        dz.dbne = dz.dbne + dzdf;
        dz.dune = dz.dune + dzdf*x(:,i,j)';
        
        dx(:,i,j) = dx(:,i,j) + u'*dzdf;
        
        dzdhh = w'*dzdf;
        if i > 1
            dzdh(:,i-1,j) = dzdh(:,i-1,j) + dzdhh;
            dz.dwne = dz.dwne + dzdf*hne(:,i-1,j)';
        end
        if j < n
            dzdh(:,i,j+1) = dzdh(:,i,j+1) + dzdhh;
            dz.dwne = dz.dwne + dzdf*hne(:,i,j+1)';
        end
        if i > 1 && j < n
            dzdh(:,i-1,j+1) = dzdh(:,i-1,j+1) + dzdhh;
            dz.dwne = dz.dwne + dzdf*hne(:,i-1,j+1)';
        end
    end 
end
dx = delta_dzds;

function g = grelu(x)
g = max(0,x);
g(g>0) = 1;

function g = gsoftmax(y,yhat)
ids = y == 0;
y(ids) = 1;
y_ = y(:)' + (0:size(yhat,1):size(yhat,1)*size(yhat,2)*size(yhat,3)-1);
yhat(y_) = yhat(y_) - 1;
yhat(:,ids) = yhat(:,ids)*0;
g = yhat;
% weight for low-frequent class
% g = reshape(repmat(idf(y(:)),size(yhat,1),1),size(yhat)).*yhat;

