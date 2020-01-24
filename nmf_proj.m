function [W,H,rel_err,iter]=nmf_proj(V,W0,H0,epsilon,maxtime,maxiter,alph,bet)

W=W0;   H=H0;  
time0=cputime;
gradW=(W*(H*H')-V*H'+2*bet*W*(W'*W)-2*bet*W);
gradH=((W'*W)*H-W'*V+2*alph*(H*H')*H-2*alph*H);
grad0=norm([gradW;gradH'],'fro');
fprintf('Initial gradient norm %f\n',grad0);

epsW=min(epsilon*grad0,1e-5);  epsH=epsW;
maxit_subprob=20;

for iter=1:maxiter
    %stoping condition
    projnorm = norm([gradW(gradW<0 | W>0);gradH(gradH<0 | H>0)])
    if projnorm < epsilon*grad0 || 1*(cputime-time0) > maxtime
        break;
    end
        
    [W,gradW,iterW]=nlssubprob_JP(V',H',W',epsW,maxit_subprob,bet,alph); W = W'; gradW = gradW';
    if iterW==1
        epsW=0.5*epsW;
        W=W+1e-8*rand(size(W));
    end
    
    [H,gradH,iterH]=nlssubprob_JP(V,W,H,epsH,maxit_subprob,alph,bet); 
    if iterH==1
        epsH=0.5*epsH;
        H=H+1e-8*rand(size(H));
    end
    
        
    if rem(iter,40)==0
        maxit_subprob=maxit_subprob+1;
    end
end
fprintf('\n Iter=%d Final Proj-grad norm %f\n', iter,projnorm);
rel_err=norm(V-W*H,'fro')/(1+norm(V,'fro'));
fprintf('\n relative error=%f\n', rel_err);