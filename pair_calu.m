function pair = pair_calu(Iu_1, ru_pre)
    pair_pre=repmat(ru_pre(1:Iu_1)',[1,Iu_1])-repmat(ru_pre(2:Iu_1+1),[Iu_1,1]);
    interv_1 = full(triu(pair_pre));
    interv_2=tril(ones(Iu_1)*i,-1);
    interv_3=(interv_1+interv_2)';
    pair=(interv_3(interv_3~=-i));
end
