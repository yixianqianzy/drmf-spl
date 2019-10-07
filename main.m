clear
%cvx_setupaddpathaddpath(genpath('/home/yzhang12/Desktop/cvx-a64/cvx'))
d = 8;
alpha= 100;   
beta = 300;
times=2;
gamm = 0.15;
tau = 1.5;
gamm_end = 3;
k=10;
s=pwd;
load([s,'\100k\Train.mat'])
Train = Train/5; %normalize ratings to [0,1]
load([s,'\100k\Test.mat'])
Test = Test./5;
[N,M] = size(Train);
[len_pos, pos_item, Pr] = comp_Pr1(Train, N);
P = single(randn(d,N));
Q = single(randn(d,M));
% sprintf('initializing P and Q ...')
% [P,Q] = main_pair(P, Q, Train, N, M, I, Id_col, Pr, d, alpha, beta, times);
% P = sign(P);
% Q = sign(Q);

% [gamm1,gamm2,W] = Init_W(times, I,Id_col, P, Q, N, M, Pr, alpha, beta, d, Train);
sprintf('initializing W and gamma...')
[W, P, Q] = Init_W(len_pos, pos_item, P, Q, N, M, Pr, alpha, beta, d, Train, gamm);


sprintf('alternating optimization ...')
% spl process
while gamm < gamm_end
         for t=1:times            
             sprintf('iteration %d / %d ...',t,times)
             sprintf('updating phi ...')
             Phi = sub_phi1(len_pos, pos_item, N, P, Q);  
             sprintf('updating P ...')
             P = sub_P1(Phi, W, Pr, len_pos,pos_item, N, Q, alpha, d);  
             sprintf('updating phi ...')
             Phi = sub_phi1(len_pos, pos_item, N, P, Q); 
             sprintf('updating Q ...')
             Q = sub_Q1(Phi, W, Pr, Train, len_pos, pos_item, P, Q, M, beta,d);   %Q全部变小
             ndcg = rating_metric(Test, P, Q, k);
             fid=fopen([s,'\100k\dis_spl_iteration.txt'],'a');
             %         fprintf(fid,'step= %d, gamm1= %f,gamm2= %f, alpha= %f, beta= %f \n', step, gamm1,gamm2, alpha , beta);
             fprintf(fid,'step= %d, gamm %f, alpha= %f, beta= %f \n', gamm,  alpha , beta);
             for i=1:k
                 fprintf(fid,'%d, %f \n',[i, ndcg(i)]);
             end
             fclose(fid);
         end
         sprintf('testing ...')
         ndcg = rating_metric(Test, P, Q, k);
         fid=fopen([s,'\100k\dis_spl.txt'],'a');
         %         fprintf(fid,'step= %d, gamm1= %f,gamm2= %f, alpha= %f, beta= %f \n', step, gamm1,gamm2, alpha , beta);
         fprintf(fid,'step= %d, gamm %f, alpha= %f, beta= %f \n', gamm, alpha , beta);
         for i=1:k
             fprintf(fid,'%d, %f \n',[i, ndcg(i)]);
         end
         fclose(fid);
         %         if step == max_intr
         %             break;
         %         else
         sprintf('updating loss, gamma, and W ...')
         gamm = gamm * tau;
         [~, W] = comp_loss(len_pos, pos_item, Pr, N, P, Q, gamm);
         
end




 
 