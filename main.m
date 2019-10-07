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
load('C:\Users\13184888\Desktop\drmf_spl\100k\\Train.mat')
Train = Train/5; %normalize ratings to [0,1]
load('C:\Users\13184888\Desktop\drmf_spl\100k\\Test.mat')
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
             Phi = sub_phi1(len_pos, pos_item, N, P, Q);  %全部变成0.125，有的变小
             sprintf('updating P ...')
             P = sub_P1(Phi, W, Pr, len_pos,pos_item, N, Q, alpha, d);  %P变小，继续变小，变大，正常化，
             sprintf('updating phi ...')
             Phi = sub_phi1(len_pos, pos_item, N, P, Q); % 全部趋向相同——变成0.12+，有的变小一点，0.1249，正常化
             sprintf('updating Q ...')
             Q = sub_Q1(Phi, W, Pr, Train, len_pos, pos_item, P, Q, M, beta,d);   %Q全部变小
             ndcg = rating_metric(Test, P, Q, k);
             fid=fopen('C:\Users\13184888\Desktop\drmf_spl\100k\\dis_spl_iteration.txt','a');
             %         fprintf(fid,'step= %d, gamm1= %f,gamm2= %f, alpha= %f, beta= %f \n', step, gamm1,gamm2, alpha , beta);
             fprintf(fid,'step= %d, gamm %f, alpha= %f, beta= %f \n', gamm,  alpha , beta);
             for i=1:k
                 fprintf(fid,'%d, %f \n',[i, ndcg(i)]);
             end
             fclose(fid);
         end
         sprintf('testing ...')
         ndcg = rating_metric(Test, P, Q, k);
         fid=fopen('C:\Users\13184888\Desktop\drmf_spl\100k\\dis_spl.txt','a');
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
         %             Loss = comp_loss(I, Id_col, Pr, N, P, Q);
         %             gamm =  comp_gamma(Loss, step + 1);
         %             W= comp_W(len_pos, pos_item_id, Pr, gamm, n, P, Q);
         
         %             if gamm > gamm_end
         %                 break;
         %             end
         %       [gamm1, gamm2]  = comp_gamma_mix(Loss, step+1);
         %             W= comp_W1(I, Id_col,Pr, gamm1, gamm2, N, P, Q);
         %         end
end
% s1 = sum(ndcg1,2);
% opt_m1=find(s1==max(s1));
% s1 = sum(ndcg1);
% opt_n1=find(s1==max(s1));
% s2 = sum(ndcg10,2);
% opt_m2=find(s2==max(s2));
% s2 = sum(ndcg10);
% opt_n2=find(s2==max(s2));
% 
%     fid=fopen('D:\桌面\AAAI18\quadratic approxi\appro\APPROX\100_optial_para1.txt','a');
%        fprintf(fid,'opt_m1 = %d, opt_n1 = %d, opt_m2 = %d, opt_n2 = %d \n', opt_m1, opt_n1, opt_m2, opt_n2);
%     fclose(fid);
% %end
% % 
% save('D:\桌面\AAAI18\quadratic approxi\appro\APPROX\ndcg1_100k','ndcg1');
% save('D:\桌面\AAAI18\quadratic approxi\appro\APPROX\ndcg10_100k','ndcg10');



 
 