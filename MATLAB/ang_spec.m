function SPEC = ang_spec(R, K, grid, N)
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
% This is the noise subspace UN
    [U,D] = eig(R);
    eig_the = real(diag(D));
    [~,Loc] = maxk(eig_the, K);
    Loc_noise = setdiff(1:length(eig_the), Loc);
    UN = U(:,Loc_noise);
    
    % Calculate the MUSIC spectra
    SPEC = zeros(1,numel(grid));
    for kk=1:numel(grid) 
        a = ULA_steer_vec(grid(kk), N);
        SPEC(kk) = 1/abs(a'*(UN*UN')*a); 
    end
end