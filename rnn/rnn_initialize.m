function rnn = rnn_initialize(param)
rnn.nx = param.nx;
rnn.nh = param.nh;
rnn.ny = param.ny;
% SouthEast Plane
rnn.use = 1e-3*randn(param.nh, param.nx);
rnn.wse = 1e-3*randn(param.nh, param.nh);
rnn.vse = 1e-3*randn(param.ny, param.nh);
rnn.bse = 1e-3*randn(param.nh,1);
rnn.cse = 1e-3*randn(param.ny,1);
rnn.duse = zeros(param.nh, param.nx);
rnn.dwse = zeros(param.nh, param.nh);
rnn.dvse = zeros(param.ny, param.nh);
rnn.dbse = zeros(param.nh,1);
rnn.dcse = zeros(param.ny,1);

% SouthWest Plane (Can share same parameters with SE)
rnn.usw = 1e-3*randn(param.nh, param.nx);
rnn.wsw = 1e-3*randn(param.nh, param.nh);
rnn.vsw = 1e-3*randn(param.ny, param.nh);
rnn.bsw = 1e-3*randn(param.nh,1);
rnn.dusw = zeros(param.nh, param.nx);
rnn.dwsw = zeros(param.nh, param.nh);
rnn.dvsw = zeros(param.ny, param.nh);
rnn.dbsw = zeros(param.nh,1);

% NorthWest Plane
rnn.unw = 1e-3*randn(param.nh, param.nx);
rnn.wnw = 1e-3*randn(param.nh, param.nh);
rnn.vnw = 1e-3*randn(param.ny, param.nh);
rnn.bnw = 1e-3*randn(param.nh,1);
rnn.dunw = zeros(param.nh, param.nx);
rnn.dwnw = zeros(param.nh, param.nh);
rnn.dvnw = zeros(param.ny, param.nh);
rnn.dbnw = zeros(param.nh,1);

% NorthEast Plane
rnn.une = 1e-3*randn(param.nh, param.nx);
rnn.wne = 1e-3*randn(param.nh, param.nh);
rnn.vne = 1e-3*randn(param.ny, param.nh);
rnn.bne = 1e-3*randn(param.nh,1);
rnn.dune = zeros(param.nh, param.nx);
rnn.dwne = zeros(param.nh, param.nh);
rnn.dvne = zeros(param.ny, param.nh);
rnn.dbne = zeros(param.nh,1);

