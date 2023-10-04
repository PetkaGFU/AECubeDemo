function hF1=plotCube(typ,Csig,varargin)
%
% plot data in form of  cube
% size of plot is taken from size of Csig varialble
%

hF1=figure; hold on


%Csig=zeros(3,3,50,1)+0.5*(rand(3,3,50,1)-0.5);


[n1,n2,n3,n4]=size(Csig);


Psig=zeros(n1,n2,n3,n4)+NaN;
if nargin > 2
    Psig=varargin{1};
end


rada=ones(n3,1);
Z=(1:n3)/20;
Z=(1:n3)/1;
for ii=1:n1
    for jj=1:n2

        %      Csig(ii,jj,2*(ii+jj),1) =1.5;
        if typ=='P'
            sig = squeeze(Csig(ii,jj,:,1:3));
            sig = sig(:,2) + (1 - sig(:,1) - sig(:,3));
        else
            sig=squeeze(Csig(ii,jj,:,2));
        end
        sig=sig/(1.5*max(abs(sig)));

        plot3(sig+ii*rada,jj*rada,-Z,'b');
        plot3(ii,jj,Z(1),'xb');
        plot3(ii,jj,-Z(end),'ob');

        sig=squeeze(Psig(ii,jj,:,2));
        sig=sig/(1.5*max(abs(sig)));
        plot3(sig+ii*rada,jj*rada,-Z,'m');
    end
end
zlabel('samples');
set(gca,'xlim',[0.05 4.95],'ylim',[0.5 4.5] );
     legend('signal','start','end','target');

a=1;
view(3)
rotate3d on

figure; hold on
n=1;
for ii=1:n1
    for jj=1:n2
        if n >14, break; end
        sig = squeeze(Csig(ii,jj,:,1:3));
        sig=sig/(1.5*max(max(abs(sig))));
        if typ=='P'
            plot(n+sig(:,1),'r:');
            plot(n+sig(:,2),'b:');
            plot(n+sig(:,3),'g:');
            sig = sig(:,2) + (1 - sig(:,1) - sig(:,3));           
            plot(n+sig,'m');

        else
            ;
        end

        sig=squeeze(Psig(ii,jj,:,2));
        sig=sig/(1.5*max(abs(sig)));
        plot(n+sig,'k');


        n=n+1;
    end
end

end
%=========================eof==================================
