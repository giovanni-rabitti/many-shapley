function [Shap,Val,ST,more]=squaredcohortshapmobMEDICAL(x,x1,m)
% Higher SHAPLEY Shapley values via Möbius transform.
%%
k=size(x,2);
n=size(x,1);
%medium=mean(x,1);
%m = media dell'output
n0=n;
%m=mean(MedicalNeuralNetworkFunction(x(1:n0,:)));
%sqrtx = (var(x)).^0.5;

try % only windows
u=memory;
if(k>log2(u.MaxPossibleArrayBytes)-3), warning('Out of memory likely.'); end
end
if(k>=log2(flintmax)), warning('Precision (and patience) may be lost.'); end

if(k>11), pp=0;hh=waitbar(pp,'Progress'); else hh=0;pp=1; end
l=2^k-1;H=zeros(1,l);
sz=zeros(1,l);
for i=1:l
            ppnew=floor(100*i/l)/200;
            if(ppnew>pp)  
                if(hh~=0),waitbar(ppnew,hh); end
                pp=ppnew;
            end    
% selection of input subset: bitmanipulation
 g=bitand(i,2.^(0:k-1))~=0; % lsb codes first index

 m1=sum(g); % subset size
 sz(i)=m1;
 % normal ranks
 gc=~g; % g complement
 
 ybar=zeros(n0,1);
 comp = zeros(n0,1);
 for ii=1:n0
 newpoint = x(ii,:);
 newpoint(g) = x1(g);
 %kk = dsearchn(x,newpoint);
% resu=(x(kk,:)-x1)<0.1.*range(x);
 %if sum(resu)==k
  ybar(ii)=MedicalNeuralNetworkFunction(newpoint);
 %comp(ii)=1;
 %else 
 %    comp(ii)=0;
 end
 

 H(i)=(mean(ybar)-m).^2;
 % H(i)=(mean(ybar(comp==1)-m)).^2;

 %newpoint=medium;
 %newpoint(g) = x1(g);
 %H(i)=(MedicalNeuralNetworkFunction(newpoint)-y0).^2;
 
 % cohort of i
 %H(i)=mean((ybar).^p)-mean(y)^p; %Owen
 %H(i)=mean((ybar-mean(y)).^p); 

end

%H(isnan(H))=0;

%% Shapley values via Möbius Trafo: 
% poset inclusion matrix is pascal triangle mod 2
mob=zeros(size(H));
sel=false(1,l);
for i=1:l
%    ii=find(sel);
    sel(1:i)=xor(sel(1:i),[true,sel(1:i-1)]);
    ii=find(sel(1:i));
    mob(:,i)=(H(:,ii)*(-1).^(sz(i)+sz(ii)'))/sz(i);
    ppnew=.5+floor(100*i/l)/200;
    if(ppnew>pp)
       if(hh~=0),waitbar(ppnew,hh); end
       pp=ppnew;
    end
end
%% Owen/Grabisch formula (weights are already included)
Shap=ones(size(H,1),k);
for i=1:k
    Shap(:,i)=sum(mob(:,logical(bitand(1:l,2^(i-1)))),2);
end
%% variance
Val=H(:,end);
%% First and total effects
ST(2,:)=H(2.^(0:k-1));
ST(1,:)=Val-H(2^k-1-2.^(0:k-1));
if(1) % test for Shapley-Owen second order effects
ShapOwen=zeros(k,k);
%ShapOwen2=zeros(k,k);
 for i=1:(k-1)
  for j=(i+1):k
   ii=bitor(2^(i-1),2^(j-1));
   ShapOwen(j,i)=mob(ii)*sz(ii);
  % ShapOwen2(j,i)=V-H(2^k-1-ii); % val(k)-val(~a)
   jj=logical(bitand(1:l,ii));
   ShapOwen(i,i)=sum(mob(:,jj).*sz(jj)./(sz(jj)-1),2); % different weights
  end
 end
more.ShapOwen=ShapOwen;
%more.ShapOwen2=ShapOwen2; 
end
more.MoebiusInv=mob;
more.ValueFun=H;
more.Sizes=sz;
if(hh~=0),delete(hh);end
end