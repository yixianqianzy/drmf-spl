function Q= sub_Q1(phi, W, Pr, Train, len_pos, pos_item, P, Q, M, beta,d)
  for i = 1: M
% i=2;
    pos_user_i = Train(:,i)>0;
    ri = find(pos_user_i);
    Ii = length(ri);
    pu =zeros(d,Ii);
    su =zeros(d,Ii);
    ppu = zeros(d,Ii);
    Bs = zeros(d,d,Ii);
    len_pos_veci = len_pos(ri);   %1*Ii
    id_col = pos_item(:,ri);  % M*Ii
     for u = 1 : Ii; 
        pos_u = find(id_col(:,u));
        v = find(pos_u==i);
        s = 1:v-1;
        idn =(s-1)*len_pos_veci(u) - (s.*(s-1))/2 + v-s;
        idp = ((v-1)*len_pos_veci(u) - v*(v-1)/2 + 1) : ((v-1)*len_pos_veci(u) - v*(v-1)/2 + len_pos_veci(u) - v);
        id = [idn, idp];
        Pr_uin = 1-Pr{ri(u)}(idn);
        Pr_uip = Pr{ri(u)}(idp);
        Pr_ui =[Pr_uin; Pr_uip];
        lam_phi = (sigm(phi{ri(u)})-0.5)./(2*phi{ri(u)});
        lam_phi(find(isnan(lam_phi)))=0;
        %   lam_phi(lam_phi==inf)=0;
        lam_ui = lam_phi(id);
%         lam_ui = lam_phi{pos(u)}(id);
        w_ui = W{ri(u)}(id);
        Qui = Q(:,pos_u);  %d*(Iu)
        Qui(:,v)=[];
        piu = P(:,ri(u));
        Bu = piu*piu';
        Bs(:,:,u) = w_ui'*lam_ui*Bu;
        pu(:,u) = ones(1,len_pos_veci(u)-1)*(w_ui)*piu;  % the first term: Iu_p = sum(Wu_ij)
        Sum_qj =Qui*(lam_ui.*w_ui); 
        su(:,u) = Bu * Sum_qj; % the second term
        ppu(:, u) = (w_ui'*Pr_ui)*piu; % the third term:  without the weight term
    end
     Ci = sum(Bs,3) + beta*eye(d);
     di = (sum(pu,2)/2 -2*sum(su,2) - sum(ppu,2));
     Ci = full(Ci);
     %   Q(:,i) = -(2*Ci)^-1*di;  %% update Q: closed form of quadratic programming problem
     %     Q(:,i) = -pinv(2*Ci)*di;  % real latent factor
     tic
     Q(:,i) = bqp((Ci+Ci')/2, di/2, 200);
%       Q(:,i) = bqp_yalmip((Ci+Ci')/2, di/2, 200);  % discrete factor
     tq(i)=toc;
  end
 ttq = sum(tq)
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