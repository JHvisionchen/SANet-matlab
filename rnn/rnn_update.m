function rnn = rnn_update(rnn, dz, param)

lr = param.lr;
momentum = param.momentum;
% lambda = param.lambda;
threshold = param.threshold;

% SE plane
% gradient clipping
nm.dwse = norm(dz.dwse, 'fro');
nm.duse = norm(dz.duse, 'fro');
nm.dvse = norm(dz.dvse, 'fro');
nm.dbse = norm(dz.dbse, 'fro');
nm.dcse = norm(dz.dcse, 'fro');
if  nm.dwse > threshold
    dz.dwse = (dz.dwse / nm.dwse) *threshold;
end
if  nm.duse > threshold
    dz.duse = (dz.duse / nm.duse) *threshold;
end
if  nm.dvse > threshold
    dz.dvse = (dz.dvse / nm.dvse) *threshold;
end
if  nm.dbse > threshold
    dz.dbse = (dz.dbse / nm.dbse) *threshold;
end
if  nm.dcse > threshold
    dz.dcse = (dz.dcse / nm.dcse) *threshold;
end
rnn.duse = momentum*rnn.duse - lr*dz.duse ;
rnn.dwse = momentum*rnn.dwse -lr*dz.dwse;
rnn.dvse = momentum*rnn.dvse - lr*dz.dvse;
rnn.dbse = momentum*rnn.dbse - lr*dz.dbse;
rnn.dcse = momentum*rnn.dcse - lr*dz.dcse;

rnn.use = rnn.use + rnn.duse;
rnn.wse = rnn.wse+ rnn.dwse;
rnn.vse = rnn.vse + rnn.dvse;
rnn.bse = rnn.bse + rnn.dbse;
rnn.cse = rnn.cse + rnn.dcse;

% SW plane
nm.dwsw = norm(dz.dwsw, 'fro');
nm.dusw = norm(dz.dusw, 'fro');
nm.dvsw = norm(dz.dvsw, 'fro');
nm.dbsw = norm(dz.dbsw, 'fro');
if nm.dwsw > threshold
    dz.dwsw = (dz.dwsw / nm.dwsw)*threshold;
end
if  nm.dusw > threshold
    dz.dusw = (dz.dusw / nm.dusw) *threshold;
end
if  nm.dvsw > threshold
    dz.dvsw = (dz.dvsw / nm.dvsw) *threshold;
end
if  nm.dbsw > threshold
    dz.dbsw = (dz.dbsw / nm.dbsw) *threshold;
end
rnn.dusw = momentum*rnn.dusw - lr*dz.dusw;
rnn.dwsw = momentum*rnn.dwsw- lr*dz.dwsw;
rnn.dvsw = momentum*rnn.dvsw - lr*dz.dvsw;
rnn.dbsw = momentum*rnn.dbsw - lr*dz.dbsw;

rnn.usw = rnn.usw + rnn.dusw;
rnn.wsw = rnn.wsw+ rnn.dwsw;
rnn.vsw = rnn.vsw + rnn.dvsw;
rnn.bsw = rnn.bsw + rnn.dbsw;

% NW plane
nm.dwnw = norm(dz.dwnw, 'fro');
nm.dunw = norm(dz.dunw, 'fro');
nm.dvnw = norm(dz.dvnw, 'fro');
nm.dbnw = norm(dz.dbnw, 'fro');
if nm.dwnw > threshold
    dz.dwnw = (dz.dwnw / nm.dwnw)*threshold;
end
if  nm.dunw > threshold
    dz.dunw = (dz.dunw / nm.dunw) *threshold;
end
if  nm.dvnw > threshold
    dz.dvnw = (dz.dvnw / nm.dvnw) *threshold;
end
if  nm.dbnw > threshold
    dz.dbnw = (dz.dbnw / nm.dbnw) *threshold;
end
rnn.dunw = momentum*rnn.dunw - lr*dz.dunw;
rnn.dwnw = momentum*rnn.dwnw- lr*dz.dwnw;
rnn.dvnw = momentum*rnn.dvnw - lr*dz.dvnw;
rnn.dbnw = momentum*rnn.dbnw - lr*dz.dbnw;

rnn.unw = rnn.unw + rnn.dunw;
rnn.wnw = rnn.wnw+ rnn.dwnw;
rnn.vnw = rnn.vnw + rnn.dvnw;
rnn.bnw = rnn.bnw + rnn.dbnw;

% NE Plane
nm.dwne = norm(dz.dwne, 'fro');
nm.dune = norm(dz.dune, 'fro');
nm.dvne = norm(dz.dvne, 'fro');
nm.dbne = norm(dz.dbne, 'fro');
if nm.dwne > threshold
    dz.dwne = (dz.dwne / nm.dwne)*threshold;
end
if  nm.dune > threshold
    dz.dune = (dz.dune / nm.dune) *threshold;
end
if  nm.dvne > threshold
    dz.dvne = (dz.dvne / nm.dvne) *threshold;
end
if  nm.dbne > threshold
    dz.dbne = (dz.dbne / nm.dbne) *threshold;
end
rnn.dune = momentum*rnn.dune - lr*dz.dune;
rnn.dwne = momentum*rnn.dwne- lr*dz.dwne;
rnn.dvne = momentum*rnn.dvne - lr*dz.dvne;
rnn.dbne = momentum*rnn.dbne - lr*dz.dbne;

rnn.une = rnn.une + rnn.dune;
rnn.wne = rnn.wne+ rnn.dwne;
rnn.vne = rnn.vne + rnn.dvne;
rnn.bne = rnn.bne + rnn.dbne;
