function Q = Q_mat(N)
    if rem(N,2)==0% N even
        Q = [eye(N/2) 1j*eye(N/2); flip(eye(N/2)) -1j*flip(eye(N/2))]/sqrt(2);
    else % N odd
        Q = [eye((N-1)/2) zeros((N-1)/2,1) 1j*eye((N-1)/2); ...
            zeros(1,(N-1)/2) sqrt(2) zeros(1,(N-1)/2); ...
            flip(eye((N-1)/2)) zeros((N-1)/2,1) -1j*flip(eye((N-1)/2))]/sqrt(2);    
    end
end