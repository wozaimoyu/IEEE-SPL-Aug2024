clc,clear,close all

P=32;
NRF=32;
SNR=30;

a=0;
b=0;
c=0;
d=0;
e=0;
f=0;
g=0;

NMSE_GCCS=[];
NMSE_LS=[];
NMSE_OMP=[];
NMSE_GCCS_restricted=[];
NMSE_LS_restricted=[];
NMSE_OMP_restricted=[];
NMSE_lspro=[];

NS=51;
NR=51;
N=NR*NS;
fc=7e9;
lambda=3e8/fc;
delta=lambda/2;
k=2*pi/lambda;
wavenumber_leakage=k/10; %Delta k
x_0=rand()*10; %x_0
y_0=rand()*95+5; %y_0
varphi=rand()*pi/3; %varphi
increment=2*pi/180; %increment

H=zeros(NR,NS);
for u = 1:NR
    for v = 1:NS
        r=sqrt((u*delta-v*delta*cos(varphi)+x_0)^2+(-y_0-v*delta*sin(varphi))^2);
        H(u,v)=exp(-j*k*r)/r;
    end
end

WR=DFT(NR)';
WS=DFT(NS);
Ha=WR'*H*WS';

%Calculate the sparsness of H_a
K = sparsityanalysis(Ha);
[nonzerorindex,nonzerocindex] = indices_of_the_maximum_K_entries_in_A(Ha,K);

%Eliminating the power leakage effect detailed in \cite{gxfICC}
Ha2=zeros(size(Ha));
Ha2(nonzerorindex,nonzerocindex)=Ha(nonzerorindex,nonzerocindex);
Ha=Ha2;

liespread=round(NR/30); %Equivalent Delta k

%Generate Y C X
[Y,RF,X] = Generatepilot(H,NRF,P);
Y=AWGN(Y,SNR);

%For ease of expression
A=RF*WR;
B=WS*X;

%Algorithm 1
zuixiaocancha=+Inf;
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
    hatvarphi=jiaodu
end

end

%Algorithm 2

%calculate Eq. (13)
[xindex,yindex] = calculate_eq_13(NS,hatvarphi,delta,lambda);
geshu=length(xindex);

%Calculate S(varphi)
column_spread=round(NS/2*wavenumber_leakage/k);
rxiabiao=[];
cxiabiao=[];
for u = 1:NS
    for v = -column_spread:column_spread
        if yindex(u) + v <= NS && yindex(u) + v >= 1
            rxiabiao=[rxiabiao,u];
            cxiabiao=[cxiabiao,yindex(u)+v];
        end
    end
end

% G0 = GCCS_2D(Y,A,B,K);
% NMSE_GCCS=[NMSE_GCCS,norm(G0-Ha,'F')^2/norm(Ha,'F')^2];

G1 = OMP_2D_restricted_support(Y,A,B,min(K,geshu),rxiabiao,cxiabiao);
NMSE_OMP_restricted=[NMSE_OMP_restricted,norm(G1-Ha,'F')^2/norm(Ha,'F')^2];

% G2 = LS_2D_restricted_support(Y,A,B,nonzerorindex,nonzerocindex);
% NMSE_lspro=[NMSE_lspro,norm(G2-Ha,'F')^2/norm(Ha,'F')^2];

G3 = LS_2D_restricted_support(Y,A,B,rxiabiao,cxiabiao);
NMSE_LS_restricted=[NMSE_LS_restricted,norm(G3-Ha,'F')^2/norm(Ha,'F')^2];

G4 = pinv(A)*Y*pinv(B);
NMSE_LS=[NMSE_LS,norm(G4-Ha,'F')^2/norm(Ha,'F')^2];

G5 = OMP_2D(Y,A,B,min(K,geshu));
NMSE_OMP=[NMSE_OMP,norm(G5-Ha,'F')^2/norm(Ha,'F')^2];

% G6 = GCCS_restriced(Y,A,B,min(K,geshu),rxiabiao,cxiabiao);
% NMSE_GCCS_restricted=[NMSE_GCCS_restricted,norm(G6-Ha,'F')^2/norm(Ha,'F')^2];

a=a+NMSE_GCCS;
b=b+b+NMSE_LS;
c=c+NMSE_OMP;
d=d+NMSE_GCCS_restricted;
e=e+NMSE_LS_restricted;
f=f+NMSE_OMP_restricted;
g=g+NMSE_lspro;

function [Y,RF,X] = Generatepilot(H,NRF,P)
    [NR,NS]=size(H);
    X=randn(NS,P);
    X=X/norm(X,'F');
    RF=randn(NRF,NR);
    RF=RF/norm(RF,'F');
    Y=RF*H*X;
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

function W = DFT(N)
    W=zeros(N);
    if mod(N,2) == 0
        error("N must be odd");
    end
    kmax = (N-1)/2;
    for k = -kmax:kmax
        for u = -kmax:kmax
            W(u+kmax+1,k+kmax+1)=exp(j*k*u*2*pi/N)/sqrt(N);
        end
    end
end

function [xindex,yindex] = calculate_eq_13(N,jiajiao,delta,lambda)
    if mod(N,2) == 0
        error("N must be odd");
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

function [rindex,cindex] = indices_of_the_maximum_K_entries_in_A(A,K)
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

function G = LS_2D_restricted_support(Y,A,B,rindices,cindices)
    [ly rg]=size(A);
    [cg lx]=size(B);
    G=zeros(rg,cg);
    K=length(rindices);
    f=zeros(K,1);
    F=zeros(K);
    for i = 1:K
        ui=A(:,rindices(i));
        vi=B(cindices(i),:);
        f(i)=trace(ui*vi*Y');
        for j = 1:K
            uj=A(:,rindices(j));
            vj=B(cindices(j),:);
            F(i,j)=trace(ui*vi*vj'*uj');
        end
    end
    w=conj(pinv(F)*f);
    G=zeros(rg,cg);
    for i = 1:K
        G(rindices(i),cindices(i))=w(i);
    end
end

function G = OMP_2D_restricted_support(Y,A,B,K,rindices,cindices)
    Yres=Y;
    [ly rg]=size(A);
    [cg lx]=size(B);
    G=zeros(rg,cg);
    yibuhuoxiabiaor=[];
    yibuhuoxiabiaoc=[];
    xiabiaoshumu=length(rindices);
    for t = 1:K
        maxinnerproduct=0;
        u=0;
        v=0;
        for l=1:xiabiaoshumu
            i=rindices(l);
            j=cindices(l);
            if G(i,j)==0
                Z=A(:,i)*B(j,:);
                innerproduct=sum(sum(Yres.*conj(Z)))/sqrt(sum(sum(Z.*conj(Z))))/sqrt(sum(sum(Yres.*conj(Yres))));
                if abs(innerproduct)>abs(maxinnerproduct)
                    maxinnerproduct=innerproduct;
                    u=i;
                    v=j;
                end
            end
        end
        G(u,v)=maxinnerproduct;
        yibuhuoxiabiaor=[yibuhuoxiabiaor u];
        yibuhuoxiabiaoc=[yibuhuoxiabiaoc v];
        f=zeros(t,1);
        F=zeros(t);
        for i = 1:t
        ui=A(:,yibuhuoxiabiaor(i));
        vi=B(yibuhuoxiabiaoc(i),:);
        f(i)=trace(ui*vi*Y');
            for j = 1:t
                uj=A(:,yibuhuoxiabiaor(j));
                vj=B(yibuhuoxiabiaoc(j),:);
                F(i,j)=trace(ui*vi*vj'*uj');
            end
        end
        w=conj(pinv(F)*f);
        Yres=Y;
        for gg = 1:t
            Yres=Yres-w(gg)*A(:,yibuhuoxiabiaor(gg))*B(yibuhuoxiabiaoc(gg),:);
        end
        fprintf('t=%d, residue is %f\n',t,norm(Yres,2));
    end
    f=zeros(K,1);
    F=zeros(K);
    for i = 1:K
        ui=A(:,yibuhuoxiabiaor(i));
        vi=B(yibuhuoxiabiaoc(i),:);
        f(i)=trace(ui*vi*Y');
        for j = 1:K
            uj=A(:,yibuhuoxiabiaor(j));
            vj=B(yibuhuoxiabiaoc(j),:);
            F(i,j)=trace(ui*vi*vj'*uj');
        end
    end
    w=conj(pinv(F)*f);
    G=zeros(rg,cg);
    for i = 1:K
        G(yibuhuoxiabiaor(i),yibuhuoxiabiaoc(i))=w(i);
    end
end

function G = OMP_2D(Y,A,B,K)
    Yres=Y;
    [ly rg]=size(A);
    [cg lx]=size(B);
    G=zeros(rg,cg);
    yibuhuoxiabiaor=[];
    yibuhuoxiabiaoc=[];
    for t = 1:K
        maxinnerproduct=0;
        u=0;
        v=0;
        for i = 1:rg
            for j = 1:cg
                if G(i,j)==0
                    Z=A(:,i)*B(j,:);
                    innerproduct=sum(sum(Yres.*conj(Z)))/sqrt(sum(sum(Z.*conj(Z))))/sqrt(sum(sum(Yres.*conj(Yres))));
                    if abs(innerproduct)>abs(maxinnerproduct)
                        maxinnerproduct=innerproduct;
                        u=i;
                        v=j;
                    end
                end
            end
        end
        G(u,v)=maxinnerproduct;
        yibuhuoxiabiaor=[yibuhuoxiabiaor u];
        yibuhuoxiabiaoc=[yibuhuoxiabiaoc v];
        f=zeros(t,1);
        F=zeros(t);
        for i = 1:t
        ui=A(:,yibuhuoxiabiaor(i));
        vi=B(yibuhuoxiabiaoc(i),:);
        f(i)=trace(ui*vi*Y');
            for j = 1:t
                uj=A(:,yibuhuoxiabiaor(j));
                vj=B(yibuhuoxiabiaoc(j),:);
                F(i,j)=trace(ui*vi*vj'*uj');
            end
        end
        w=conj(pinv(F)*f);
        Yres=Y;
        for gg = 1:t
            Yres=Yres-w(gg)*A(:,yibuhuoxiabiaor(gg))*B(yibuhuoxiabiaoc(gg),:);
        end
        fprintf('t=%d, residue is %f\n',t,norm(Yres,2));
    end
    f=zeros(K,1);
    F=zeros(K);
    for i = 1:K
        ui=A(:,yibuhuoxiabiaor(i));
        vi=B(yibuhuoxiabiaoc(i),:);
        f(i)=trace(ui*vi*Y');
        for j = 1:K
            uj=A(:,yibuhuoxiabiaor(j));
            vj=B(yibuhuoxiabiaoc(j),:);
            F(i,j)=trace(ui*vi*vj'*uj');
        end
    end
    w=conj(pinv(F)*f);
    G=zeros(rg,cg);
    for i = 1:K
        G(yibuhuoxiabiaor(i),yibuhuoxiabiaoc(i))=w(i);
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

function G = GCCS_2D(Y,A,B,K)
    eta=1; %when eta is larger, the clustering is greater
    y=Y(:);
    [NRF,NR]=size(A);
    [NS,P]=size(B);
    N=NS*NR;
    geshu=NRF*P;
    yibuhuor=[];
    yibuhuoc=[];
    fprintf("Begin build graph\n");
    n_node=NS*NS;
    edge_head=[];
    edge_tail=[];
    bianhao=0;
    for v = 1:NS
        for u = 1:NR
            bianhao=bianhao+1;
            if v >= 2
                edge_head=[edge_head,bianhao];
                edge_tail=[edge_tail,bianhao-NR];
            end
            if v <= NS-1
                edge_head=[edge_head,bianhao];
                edge_tail=[edge_tail,bianhao+NR];
            end
            if u >= 2
                edge_head=[edge_head,bianhao];
                edge_tail=[edge_tail,bianhao-1];
            end
            if u <= NR-1
                edge_head=[edge_head,bianhao];
                edge_tail=[edge_tail,bianhao+1];
            end
        end
    end
    bianshu=length(edge_head);
    fprintf("Graph built\n");
    fprintf("Begin Kronecker Product\n");
    zidianjuzhen=zeros(geshu,N);
    for v = 1:NS
        for u = 1:NR
            t=A(:,u)*B(v,:);
            zidianjuzhen(:,v*NR-NR+u)=t(:);
        end
    end
    r=Y(:);
    h=pinv(zidianjuzhen)*r;
    fprintf("Kronecker Product Complete\n");
    while norm(r,'fro')>0.5*norm(y,'fro')
        cap_source=abs(h).^2;
        cap_source=cap_source/norm(cap_source,2)*sqrt(N);
        cap_sink=1./cap_source;
        [set_index_source,set_index_sink]=maxflow_mincut(cap_source,cap_sink,edge_head,edge_tail,eta);
        set_index_source
        indices_num=length(set_index_source);
        rindex=zeros(1,indices_num);
        cindex=zeros(1,indices_num);
        for u = 1:indices_num
            rindex(u)=mod(set_index_source(u)-1,NR)+1;
            cindex(u)=floor((set_index_source(u)-1)/NR)+1;
        end
        G=OMP_2D_restricted_support(Y,A,B,min(K,indices_num),rindex,cindex);
        return;
    end
end

function [set_index_source,set_index_sink] = maxflow_mincut(cap_source,cap_sink,edge_head,edge_tail,eta)
    fprintf("Begin maxflow-mincut\n");
    n_edge=2*length(cap_sink)+length(edge_head);
    l=length(edge_head);
    n_node=length(cap_sink);
    edgeset=zeros(n_edge,3);
    for u = 1:n_node
        edgeset(u,1)=n_node+1; %source
        edgeset(u,2)=u;
        edgeset(u,3)=cap_source(u);
        edgeset(u+n_node,1)=u;
        edgeset(u+n_node,2)=n_node+2; %sink
        edgeset(u+n_node,3)=cap_sink(u);
    end
    for u=1:l
        edgeset(u+2*n_node,1)=edge_head(u);
        edgeset(u+2*n_node,2)=edge_tail(u);
        edgeset(u+2*n_node,3)=eta;
    end
    yizhaodaozuidaliu=0;
    %Begin maxflow
    m=n_node+2; 
    A=zeros(m); 
    for u = 1:length(edgeset(:,1))
        A(edgeset(u,1),edgeset(u,2))=single(edgeset(u,3));
    end
    maxflow=zeros(m);
    liushu=0;
    while 1        
        flag=[];            %closed set
        flag=[flag n_node+1];   
        head=n_node+1;
        tail=n_node+1;
        queue=[];           %open set
        queue(n_node+1)=n_node+1;
        head=1;
        previous=zeros(1,m);     
        previous(n_node+1)=n_node+1;            %source is the previous of itselt
        while tail~=head 
            u=queue(tail);
            for v=1:m
                if A(u,v)>0 && isempty(find(flag==v,1))
                    queue(head)=v;
                    if head<n_node
                        head=head+1;
                    else
                        head=n_node+2;
                    end
                    flag=[flag v];
                    previous(v)=u;
                end
            end
            if tail==n_node+1
                tail=1;
            else
                if tail==n_node
                    tail=n_node+2;
                else
                    tail=tail+1;
                end
            end
        end
        if previous(m)==0
            break;
        end
        path=[];
        u=m;             
        k=0;              
        while u ~= n_node+1        
            path=[path;previous(u) u A(previous(u),u)];   
            u=previous(u);            %Dijsktra slacken
            k=k+1;             
        end
        Mi=min(path(:,3));    
        for u=1:k  
            A(path(u,1),path(u,2))=A(path(u,1),path(u,2))-Mi; 
            maxflow(path(u,1),path(u,2))=maxflow(path(u,1),path(u,2))+Mi;
        end                   
        liushu=liushu+1;
        fprintf("Find the %d-th available flow\n",liushu);
    end    
    set_index_source=[];
    set_index_sink=[];
    for u = 1:n_node
        if isempty(find(flag==u)) 
            set_index_sink=[set_index_sink,u];
        else 
            set_index_source=[set_index_source,u];
        end
    end
end

function G = GCCS_restriced(Y,A,B,K,rindex,cindex)
    eta=0.1;
    y=Y(:);
    [NRF,NR]=size(A);
    [NS,P]=size(B);
    geshu=NRF*P;
    n_node=length(rindex);
    fprintf("Begin build graph\n");
    edge_tou=[];
    edge_wei=[];
    for u = 1:n_node
        for v = 1:n_node
            if abs(rindex(u)-rindex(v))+abs(cindex(u)-cindex(v)) == 1
                edge_tou=[edge_tou,u,v];
                edge_wei=[edge_wei,v,u];
            end
        end
    end
    fprintf("Graph built\n");

    fprintf("Begin Kronecker Product\n");
    zidianjuzhen=zeros(geshu,n_node);
    for u = 1:n_node
        t=A(:,rindex(u))*B(cindex(u),:);
        zidianjuzhen(:,u)=t(:);
    end
    r=Y(:);
    h=pinv(zidianjuzhen)*r;
    fprintf("Kronecker product complete\n");
    while norm(r,'fro')>0.5*norm(y,'fro')
        cap_source=sqrt(abs(h));
        plot(cap_source)
        cap_source=cap_source/norm(cap_source,2)*sqrt(n_node);
        cap_sink=1./cap_source;
        [set_index_source,set_index_sink]=maxflow_mincut(cap_source,cap_sink,edge_tou,edge_wei,eta);
        set_index_source
        indices_num=length(set_index_source);
        rindexobtained=zeros(1,indices_num);
        cindexobtained=zeros(1,indices_num);
        for u = 1:indices_num
            rindexobtained(u)=rindex(set_index_source(u));
            cindexobtained(u)=cindex(set_index_source(u));
        end
        G=OMP_2D_restricted_support(Y,A,B,min(K,indices_num),rindexobtained,cindexobtained);
        return;
    end

end