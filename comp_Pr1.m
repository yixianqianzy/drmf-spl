function [len_pos, pos_item, Pr]= comp_Pr1(Train, n)
Pr = cell(n, 1);  % P_uij: the groudtruth probability of the pairwise preference 
parfor u = 1:n
    pos_item_u = Train(u,:)>0; 
    ru = Train(u, pos_item_u); %r_u: postive ratings
    Iu = length(ru); %I_u
    Iu_1 = Iu-1;
%     pair_mat = repmat(ru(1:Iu_1)',[1,Iu_1]) - repmat(ru(2:Iu),[Iu_1,1]); % compute pairwise preference
%     intev_1 = full(triu(pair_mat));
%     intev_2 = tril(ones(Iu_1)*i,-1);
%     intev_3 = (intev_1+intev_2)';
%     Ou = (intev_3(intev_3 ~= -i));  % o_uij: the true pairwise ratings
    Ou = pair_calu(Iu_1, ru);  
    len_pos(u) = Iu;
    pos_item(:,u) = pos_item_u;
    Pr{u} = Ou; 
%   the groudtruth probability;
    Pr{u}(Ou<0) = 0;
    Pr{u}(Ou==0) = 0.5;
    Pr{u}(Ou>0) = 1;
end
end


function pair = pair_calu(Iu_1, ru_pre)
    pair_pre=repmat(ru_pre(1:Iu_1)',[1,Iu_1])-repmat(ru_pre(2:Iu_1+1),[Iu_1,1]);
    interv_1 = full(triu(pair_pre));
    interv_2=tril(ones(Iu_1)*i,-1);
    interv_3=(interv_1+interv_2)';
    pair=(interv_3(interv_3~=-i));
end
