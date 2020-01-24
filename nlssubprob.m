function [H,grad,iter]=nlssubprob(V,W,H0,epsilon,maxiter,alph,bet)
r=size(W,2);
H=H0;   WtW=W'*W;  WtV=W'*V;
beta=0.75;  
max_in_it=20;

f_H=0.5*norm(V-W*H,'fro')^2/(1+norm(V,'fro'))^2+0.5*alph*norm(H*H'-eye(r),'fro')^2/(1+r)^2+0.5*bet*norm(W'*W-eye(r),'fro')^2/(1+r)^2;
for iter=1:maxiter
    grad=(WtW*H-WtV)/(1+norm(V,'fro'))^2+(alph*(H*H')*H-alph*H)/(1+r)^2;%
    row_col=(grad<0 | H>0);
    projgrad=norm(grad(row_col));
    grad_P=grad/norm(grad,'fro');
    if projgrad < epsilon
        break;
    end

    %search step size
    alpha=1;  
    for inner_iter=1:max_in_it
        Hn=max(H-alpha*grad_P,0); d=Hn-H;
        f_Hn=0.5*norm(V-W*Hn,'fro')^2/(1+norm(V,'fro'))^2+0.5*alph*norm(Hn*Hn'-eye(r),'fro')^2/(1+r)^2+0.5*bet*norm(W'*W-eye(r),'fro')^2/(1+r)^2;
        gradd=sum(sum(grad.*d));  
        suff_decr=(f_Hn-f_H)-0.001*gradd < 0;
        
        if inner_iter==1
            decr_alpha=~suff_decr;   Hp=H;  %Hp is last iterate before Hn (i.e. H_n-1)
            f_Hp=f_H;
        end
        if decr_alpha
            if suff_decr
                H=Hn;
                break;
            else
                alpha=alpha*beta;
            end
        else  % we increase alpha
            if ~suff_decr | (f_Hn>=f_Hp) | norm(Hp-Hn,'fro')<1e-8  % if not suff. decrease we stop
                H=Hp;
                break;
            else
                alpha=alpha/beta;   
                Hp=Hn;
                f_Hp=f_Hn;
            end
        end
    end

    f_Hnew=0.5*norm(V-W*H,'fro')^2/(1+norm(V,'fro'))^2+0.5*alph*norm(H*H'-eye(r),'fro')^2/(1+r)^2+0.5*bet*norm(W'*W-eye(r),'fro')^2/(1+r)^2;
    if abs(f_H-f_Hnew)<epsilon
        % we did not find a solution with sufficient decrease so H has not
        % changed at all or has changed very little 
        % so we quit
        break;
    else
        f_H=f_Hnew;    
    end
end