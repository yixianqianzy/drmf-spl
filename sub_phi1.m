function phi= sub_phi1(len_pos, pos_item, N, P, Q)
% Id_col =Id_col';
% lam_phi= cell(N, 1);
phi= cell(N, 1);
parfor u = 1:N
    Iu = len_pos(u);
    ru_pre = P(:,u)' * Q(:,pos_item(:,u)); %1*Iu
    Iu_1=Iu-1;
    phi{u} = pair_calu(Iu_1, ru_pre);
%     lam_phi{u} = (sigm(phi{u})-1/2)./(2*phi{u}); 
%     lam_phi{u}(find(isnan(lam_phi{u})))=0;
%     lam_phi{u}(lam_phi{u}==inf)=0;
end
end

function pair = pair_calu(Iu_1, ru_pre)
    pair_pre=repmat(ru_pre(1:Iu_1)',[1,Iu_1])-repmat(ru_pre(2:Iu_1+1),[Iu_1,1]);
    interv_1 = full(triu(pair_pre));
    interv_2=tril(ones(Iu_1)*i,-1);
    interv_3=(interv_1+interv_2)';
    pair=(interv_3(interv_3~=-i));
end
