function P  = sub_P1(phi, W, Pr, len_pos, pos_item, n, Q, alpha, d)
% Id_col= Id_col';
 for u = 1 : n
    Iu = len_pos(u);
%     pos = find(pos_item(:,u));
    Qo = Q(:,pos_item(:,u)); %d*su
    Qu = zeros(d,(Iu*(Iu-1)/2));
    for i = 1 : Iu-1
        Qu(:,((i-1)*Iu-((i-1)*i)/2+1) : i*Iu-((i+1)*i)/2) = bsxfun(@minus,Qo(:,i),Qo(:, i+1:Iu)); % d*su
    end
    
    lam_phi = (sigm(phi{u})-1/2)./(2*phi{u});
    lam_phi(find(isnan(lam_phi)))=0;
    %    lam_phi(lam_phi==inf)=0;
    a = sparse(double(lam_phi.*W{u}));
    %     a = sparse(double(lam_phi{u}.*W{u}));
    Cu = Qu*diag(a)*Qu' + alpha*eye(d); % d*d
    du = Qu * (W{u} / 2 - (W{u}.*Pr{u})); % d*1
    %     P(:,u) = -(2*Cu)^-1*du;
    %      P(:,u) = -pinv(2*Cu)*du;    % real latent factor
    tic
    P(:,u) = bqp((Cu+Cu')/2, du/2, 200);     % discrete factor
    tp(u) = toc;
    %       P(:,u) = bqp_yalmip((Cu+Cu')/2, du/2, 200); % byYALMIP_Mosek
 end
 ttp = sum(tp)
 end


function x = bqp(A, b, L)
%%% binary quadratic problem: min x' A x + 2 b' x, s.t. x in {+1,-1}^k
[k, k1] = size(A);
assert(k == k1 && issymmetric(A), 'matrix A should be sysmmetric');
C = [A, b; b', 0]; % cast to min x' C x, s.t. x in {+1,-1}^k+1
%[~, X, ~] = psd_ip1(-C); % solve min tr(CX), st. rank(X)=1 and diag(X) = e (drop rank constraint)
% smallconstant = 10^-5;
%method 1: the original problem
    cvx_begin sdp quiet
      variable X(k+1,k+1) symmetric semidefinite 
      minimize(trace(C * X)); 
      subject to 
      diag(X) == 1; 
%       X>=eye(k+1)*smallconstant;
    cvx_end
    
Xi = sign(mvnrnd(zeros(k+1,1), X, L));
loss = sum((Xi * C) .* Xi, 2);
[~, ind] = min(loss);
x = Xi(ind,:);
t = x(k+1);
x = x(1:k) * t;
end
     