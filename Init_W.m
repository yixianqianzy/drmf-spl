%function [gamm1, gamm2, W] = Init_W(times, I,Id_col, P, Q, N, M, Pr, alpha, beta, d, Train)
function [W, P, Q] = Init_W(len_pos,pos_item, P, Q, n, m, Pr, alpha, beta, d, Train, gamm)
W = cell(n, 1);
parfor u = 1:n
     Iu = len_pos(u);
     pair_num = Iu*(Iu-1)/2;
     W{u} = ones(pair_num,1)>0;
end
for t=1:2        
      Phi = sub_phi1(len_pos, pos_item, n, P, Q);  %update \phi
      P = sub_P1(Phi, W, Pr, len_pos, pos_item, n, Q, alpha, d);  % update P
      Phi = sub_phi1(len_pos, pos_item, n, P, Q); % update phi
      Q = sub_Q1(Phi, W, Pr, Train, len_pos, pos_item, P, Q, m, beta,d);   % update Q
end
 [~, W] = comp_loss(len_pos, pos_item, Pr, n, P, Q, gamm);
% Loss = comp_loss(len_pos, pos_item_id, Pr, n, P, Q);
%[gamm1, gamm2]  = comp_gamma_mix(Loss, 1); %mixed gamma
%W= comp_W1(I, Id_col,Pr, gamm1, gamm2, N, P, Q);
% gamm =  comp_gamma(Loss, 1);   % gamma is the middle value of Loss
% W = comp_W(len_pos, pos_item_id, Pr, gamm, n, P, Q);
% testing the result of all trainingdata
load Test
Test = Test/5;
k=10;
ndcg = rating_metric(Test, P, Q, k);
fid=fopen('C:\Users\13184888\Desktop\drmf_spl\100k\dis_all_ndcg.txt','a');
fprintf(fid,'alpha= %f, beta= %f \n', alpha , beta);
for i=1:k
    fprintf(fid,'%d, %f \n',[i, ndcg(i)]);
end
fclose(fid);  
end

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
