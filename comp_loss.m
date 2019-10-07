function [Loss, W] = comp_loss(len_pos, pos_item, Pr, n, P, Q, k0)
% loss= cell(N, 1);
Loss = [];
parfor u = 1:n
     Iu = len_pos(u);
     ru_pre = P(:,u)' * Q(:,pos_item(:,u)); %1*Iu
     Iu_1 = Iu-1;
%      pair_pre_mat=repmat(ru_pre(1:Iu_1)',[1,Iu_1])-repmat(ru_pre(2:Iu),[Iu_1,1]);
%      intev_1 = full(triu(pair_pre_mat));
%      intev_2=tril(ones(Iu_1)*i,-1);
%      intev_3=(intev_1 + intev_2)';
%      Ou_pre=(intev_3(intev_3~=-i));  % hat(o)_uij: the predicted pairwise ratings
     Ou_pre = pair_calu(Iu_1, ru_pre);  
     loss = -Pr{u}.* Ou_pre + log(1+exp(Ou_pre)); 
     W{u}=loss < k0;  % 1/k0: initiaze k0
     Loss = [Loss;loss];
end
end
