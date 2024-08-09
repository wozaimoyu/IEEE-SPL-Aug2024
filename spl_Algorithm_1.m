clc,clear,close all

P=32;
NRF=32;

NS=51;
NR=51;
fc=7e9;
lambda=3e8/fc;
delta=lambda/2;
k=2*pi/lambda;
xR=rand()*10; %x0
xS=0;
yR=0;
yS=5+95*rand(); %y0

a=0;
for simulationtimes = 1 %set simulation times
RMSE_jiaodu=[];
theta=rand()*pi/3;

for SNR = -60:10:10

liespread=round(NS/30); %Equivalent Delta k
increment=2*pi/180; %Increment

H=zeros(NR,NS);
for u = 1:NR
    for v = 1:NS
        r=sqrt((u*delta-v*delta*cos(theta)+xR-xS)^2+(yR-yS-v*delta*sin(theta))^2);
        H(u,v)=exp(-j*k*r)/r;
    end
end

WR=DFT(NR)';
WS=DFT(NS);
Ha=WR'*H*WS';

[Y,RF,X] = shengchengxinhao(H,NRF,P);
Y=AWGN(Y,SNR);

A=RF*WR;
B=WS*X;

zuixiaocancha=+Inf;
varphi=0;

for jiaodu = 0:increment:89*pi/180

%Calculate S(varphi')
[xindex,yindex] = xiabiaoji(NS,jiaodu,delta,lambda);
geshu=length(xindex);

rxiabiao=[];
cxiabiao=[];
for u = 1:NS
    for v = -liespread:liespread
        if yindex(u) + v <= NS && yindex(u) + v >= 1
            rxiabiao=[rxiabiao,u];
            cxiabiao=[cxiabiao,yindex(u)+v];
        end
    end
end

%Minimize the residue with S(varphi')
G = LS_2D_yizhixiabiao(Y,A,B,rxiabiao,cxiabiao);
Res=Y-A*G*B;
if norm(Res,'fro')<zuixiaocancha
    zuixiaocancha=norm(Res,'fro');
    varphi=jiaodu
end

end

RMSE_jiaodu=[RMSE_jiaodu,abs(varphi-theta)*180/pi];

end
a=a+RMSE_jiaodu;
end
a=a/simulationtimes;

function [Y,RF,X] = shengchengxinhao(H,NRF,P)
    [NR,NS]=size(H);
    X=randn(NS,P);
    X=X/norm(X,'F');
    RF=randn(NRF,NR);
    RF=RF/norm(RF,'F');
    Y=RF*H*X;
end

function W = DFT(N)
    W=zeros(N);
    if mod(N,2) == 0
        error("N需要为奇数");
    end
    kmax = (N-1)/2;
    for k = -kmax:kmax
        for u = -kmax:kmax
            W(u+kmax+1,k+kmax+1)=exp(j*k*u*2*pi/N)/sqrt(N);
        end
    end
end

function [xindex,yindex] = xiabiaoji(N,jiajiao,delta,lambda)
    if mod(N,2) == 0
        error("N需要为奇数");
    end
    L=N*delta;
    k=2*pi/lambda;
    zhongjian=round((N+1)/2);
    ksmax=round((N-1)/2);
    for u = 1:ksmax
        xindex(zhongjian+u)=zhongjian+u;
        ksx=2*pi*u/L;
        ksy=sqrt(k*k-ksx*ksx);
        krx=cos(jiajiao)*ksx-sin(jiajiao)*ksy;
        rindex=round(krx*L/2/pi);
        yindex(zhongjian+u)=zhongjian+rindex;
    end
    krx=-sin(jiajiao)*k;
    rindex=round(krx*L/2/pi);
    yindex(zhongjian)=zhongjian+rindex;
    for u = -ksmax:-1
        xindex(zhongjian+u)=zhongjian+u;
        ksx=abs(2*pi*u/L);
        ksy=sqrt(k*k-ksx*ksx);
        krx=-cos(jiajiao)*ksx+sin(jiajiao)*ksy;
        rindex=round(krx*L/2/pi);
        yindex(zhongjian+u)=zhongjian+rindex;
    end
end

function sparsity = sparsityanalysis(A)
    [r,c]=size(A);
    B=abs(A);
    n=r*c;
    vec=zeros(n,1);
    for j = 1:c
        for i = 1:r
            vec(n+1-(j-1)*r-i)=B(i,j);
        end
    end
    vec=sort(vec);
    totalenergy=sum(vec.*vec);
    he=0;
    for t = n:-1:1
        he=he+vec(t)*vec(t);
        if he > 0.95*totalenergy
            sparsity = n+1-t;
            break;
        end
    end
end

function [rindex,cindex] = AdezuidaKgeyuansudexiabiao(A,K)
    [r,c]=size(A);
    B=abs(A);
    n=r*c;
    B=A(:);
    [M,I]=maxk(B,K);
    for u = 1:K
        rindex(u)=mod(I(u)-1,r)+1;
        cindex(u)=floor(I(u)/r)+1;
    end
end

function G = LS_2D_yizhixiabiao(Y,A,B,rxiabiao,cxiabiao)
    [ly rg]=size(A);
    [cg lx]=size(B);
    G=zeros(rg,cg);
    K=length(rxiabiao);
    f=zeros(K,1);
    F=zeros(K);
    for i = 1:K
        ui=A(:,rxiabiao(i));
        vi=B(cxiabiao(i),:);
        f(i)=trace(ui*vi*Y');
        for j = 1:K
            uj=A(:,rxiabiao(j));
            vj=B(cxiabiao(j),:);
            F(i,j)=trace(ui*vi*vj'*uj');
        end
    end
    w=conj(pinv(F)*f);
    G=zeros(rg,cg);
    for i = 1:K
        G(rxiabiao(i),cxiabiao(i))=w(i);
    end
end

function x_noise=AWGN(x,SNR)
    [m,n]=size(x);
    Eb=norm(x,2)^2/m/n;
    sigma=Eb/exp(SNR/10)/2;
    [size_x size_y]=size(x);
    x_noise=x+sqrt(sigma)*randn(size_x,size_y);
    x_noise=x_noise+j*sqrt(sigma)*randn(size_x,size_y);
end