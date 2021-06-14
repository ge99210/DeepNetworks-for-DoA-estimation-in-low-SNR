function [ang_est, sp_val] = l1_SVD_DoA_est(Y,ULA_N,threshold,SOURCE_K, THETA_angles)
% INPUTS:
% Ry: the sample covariance estimate
% ULA_N: the number of sensors in the array
% noise_power: the variance of the noise
% SOURCE_K: the number of sources
% THETA_angles: the grid

% OUTPUT:
% ang_est: the DoA estimate 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The grid and dictionary for the compressed sensing method
NGrids = length(THETA_angles);
A_dic = zeros(ULA_N,NGrids);
for n=1:NGrids
    A_dic(:,n) = ULA_steer_vec(THETA_angles(n),ULA_N);
end
% Calculate the \ell_2,1 SVD
[~,~,V] = svd(Y);
Dr = [eye(rank(Y)) zeros(size(Y,1),size(Y,2) -rank(Y))];
Ydr = Y*V*Dr.';

% Solve SOCP using CVX
    cvx_begin quiet
        variable S_est_dr(NGrids,size(Y,1)) complex;
        minimize( sum(norms(S_est_dr.')) ); % this is the 2,1 norm of a matrix (mixed norm) % + 0.5*norm(y - A_dic*s_est,2) );
        subject to
            norm(Ydr - A_dic*S_est_dr,'fro') <= threshold;
    cvx_end
    
% % Solve SOCP using CVX
%     cvx_begin quiet
%         variable S_est_dr(NGrids,size(Y,1)) complex;
%         minimize ( threshold*sum(norms(S_est_dr.'))+0.5*square_pos(norm(Ydr - A_dic*S_est_dr,'fro')) );% this is the 2,1 norm of a matrix (mixed norm) % + 0.5*norm(y - A_dic*s_est,2) );
%     cvx_end

S_est = S_est_dr*Dr*V';
Ps = sum(abs(S_est).^2,2);
% figure(1);
% plot(Ps);
[sp_val, spa_ind] = maxk(Ps,SOURCE_K);
ang_est = sort(THETA_angles(spa_ind))';

end