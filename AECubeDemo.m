function predictedProbab = AECubeDemo
%
%
%
% "Accoustic Emission Cube Demo " 
%  Accoustic Emission multiple evetnst detection with use of CubeNet/U-Net CNN
%
%
%
% the code is suplement to the article:
% Petr Kolář and Matěj Petružálek:
% Discrimination of doubled Acoustic Emission events using Neural Networks
%
% submited to Ultrasonic
%
% input: data file indetification is presribed below
% output: predicted probabilities from testing data
% note: the interpretation of these probabilities is not included in this
% demo code and should be done by user
%
% required files:
%           * code  (function AECubeDemo)
%           * (testing) data (examples quotted in the article)
%           * trained RNN for onset(s) detecion
%           * trained RNN for OT prediction
% (all these files are part of the package)
%

%


%
% created by P. Kolar   kolar@ig.cas.cz
%
%
% compatibility: created under MATLAB R2022b
% required: MATLAB core
%           Statistics and Machine Learning Toolbox
%           Signal Processing Toolbox
%
% version 1.0 / 04/10/2023
%


% how the Net was created:
%
% l=unet3dLayers([4 4 100 1],3,'EncoderDepth',2,'NumFirstEncoderFilters',16)
%

close all

close(findall(0,'tag','NNET_CNN_TRAININGPLOT_FIGURE'));

netName='C2NN400_50-50_L123f.mat';  % net trainde on large data

train=0;     % no training - pre-trained Net is used
%train=1;    % training on available data (seimograms) is performed
if train
    netName='trainedNet';   % Net trained on actual data
end

% plot outputs
plot1=0;
plot1=1;

rng('default')   % for reprodacibility

% input data files

files=what('d:\Data2count\dataAE_SyntDoubleAg_4mw');    % directory with input files


len1=400;       % sub-seismogram length
dt=1/10e6;      % sampling

%%{
                   
nFiles=length(files.mat);
iFile=randperm(nFiles);

% data division
nT=floor(0.7*nFiles);   %   0.0 - 0.7  for training
nV=floor(0.8*nFiles);   %   0.7 - 0.8  for validation


% position of channes in Cube
posTab=[2 1; 2 2; 3 3; 3 4;...
    1 1; 1 2; 2 3; 2 4;...
    3 1; 3 2; 4 3; 4 4;...
    4 2; 4 1; 1 3; 1 4;];




signal=[];
probab=[];
info=[];
for i=1:nT
    jj=iFile(i);
    str=[files.path,'\',files.mat{jj}];

    out=isfile(str);
    if out ~= 1
        aa=1;
    end
    [signal1,probab1,info1]=getSignal3D(str,posTab,len1);
    lenS=length(signal1);
    signal=[signal signal1];
    probab=[probab probab1];
    info=[info info1];
    aa=1;
end
signalTrain=signal';
probabTrain=probab';
infoTrain=info';


signal=[];
probab=[];
info=[];
for i=nT+1:nV
    jj=iFile(i);
    str=[files.path,'/',files.mat{jj}];

    [signal1,probab1,info1]=getSignal3D(str,posTab,len1);
    lenS=length(signal1);
    signal=[signal signal1];
    probab=[probab probab1];
    info=[info info1];
    aa=1;
end
signalValid=signal';
probabValid=probab';
infoValid=info';


signal=[];
probab=[];
info=[];
iY=1;
for i=nV+1:nFiles

    jj=iFile(i);
    str=[files.path,'/',files.mat{jj}];

    [signal1,probab1,info1,signal0,info0,probab0]=getSignal3D(str,posTab,len1);
    lenS=length(signal1);
    signal=[signal signal1];
    probab=[probab probab1];
    info=[info info1];

    Ysignal{iY}=signal0;
    Yprobab{iY}=probab0;
    Yinfo{iY}=info0;
    iY=iY+1;

    aa=1;
end
signalTest=signal';
probabTest=probab';
infoTest=info';




disp("DATA was red");
%%{

% an example of input data is ploted
% the seismogram number may be modified, if required
len=length(signalTrain);
np=min([2000 round(0.7*len)-1]);
if nV > 0
    hF1=plotCube('',signalTrain{np},probabTrain{np});
end

%............................................................

aa=1;

%
% Cube-net definition
%

lgraph = layerGraph();

% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
tempLayers = [
    sequenceInputLayer([4 4 400 3],"Name","sequence3Dinput")
    sequenceFoldingLayer("Name","seqfold")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],16,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-1")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],32,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-2")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2 2])
    convolution3dLayer([3 3 3],32,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-1")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],64,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-2")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2 2])
    convolution3dLayer([3 3 3],64,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-1")
    reluLayer("Name","Bridge-ReLU-1")
    convolution3dLayer([3 3 3],128,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-2")
    reluLayer("Name","Bridge-ReLU-2")
    transposedConv3dLayer([2 2 2],128,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","Decoder-Stage-1-Concatenation")
    convolution3dLayer([3 3 3],64,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-1")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],64,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-2")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv3dLayer([2 2 2],64,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","Decoder-Stage-2-Concatenation")
    convolution3dLayer([3 3 3],32,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-1")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],32,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-2")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    convolution3dLayer([1 1 1],3,"Name","Final-ConvolutionLayer","Padding","same","WeightsInitializer","he")
    softmaxLayer("Name","Softmax-Layer")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    sequenceUnfoldingLayer("Name","sequnfold")
    RegressionLayerTdiff2f('L123f')];    % private Loos function

lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"seqfold/out","Encoder-Stage-1-Conv-1");
lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Decoder-Stage-2-Concatenation/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Decoder-Stage-1-Concatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpConv","Decoder-Stage-1-Concatenation/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpConv","Decoder-Stage-2-Concatenation/in1");
lgraph = connectLayers(lgraph,"Softmax-Layer","sequnfold/in");


% Cube-net definition is finished and plotted
figure; hold on
plot(lgraph);

layers=lgraph;

%..........................................................


% training opitons

% this option set is enough for test (with limited data set)
maxEpochs = 60;
maxEpochs = 5;
miniBatchSize = 10;

% %  possible full ver. of options
% maxEpochs = 30;
% maxEpochs = 40;
% maxEpochs = 120;
% miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.0045, ...
    'GradientThreshold',0.5, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto',...
    'ValidationData',{signalValid,probabValid}, ...
    'Verbose',0);

tic
%%{
if train
    netCNN_3dUnetL2 = trainNetwork(signalTrain,probabTrain,layers,options);
    save(netName,'netCNN_3dUnetL2');
    disp(' trained Cube-net was saved')
end
toc
load(netName);

disp(['net ',netName,' - net was loaded']);



Yplus=1:14;
Yplus=repmat(Yplus',[1 1024]);

nEvT=nFiles-nV;   % nummber of predicted events

predictedProbab=zeros(nEvT,14,1024);

hFp=[];
for i=1:nEvT
    YpredP=zeros(14,1024);

    Ysig1=Ysignal{i};
    Yprobab1=Yprobab{i};
    if plot1
        if plot1==2, close(hFp); end
        hFp=figure('Tag','Onpred'); hold on
    end

    for ii=1:8  % shift loop
        ii1=(ii-1)*78+1;
        ii2=ii1+len1-1;
        Ysig1s=Ysig1(:,:,ii1:ii2,:);
        [Ypred1_3d] = predict(netCNN_3dUnetL2,Ysig1s);

        % Cube data decomposition into maxtrix
        for jj=1:14
            pos1=posTab(jj,1);
            pos2=posTab(jj,2);
            tmp=squeeze(Ypred1_3d(pos1,pos2,:,1:3));
            tmp=smoothdata(tmp,1);
            tmp=tmp(:,2) + (1 -tmp(:,1) - tmp(:,3));
            tmp(tmp<0)=0;

            tmp0=squeeze(YpredP(jj,ii1:ii2));
            YpredP(jj,ii1:ii2)= max([tmp0; tmp']);
            if ii == 1
                tmp=squeeze(Yprobab1(pos1,pos2,:,1:3));
                tmp=tmp(:,2) + (1 -tmp(:,1) - tmp(:,3));
                tmp(tmp<0)=0;
                YprobabP(jj,:)= tmp;
            end
            aa=1;
        end
        if plot1
            plot((YpredP+Yplus)');
            plot((0.45*YprobabP+Yplus)',':b');
            strTit=(['evNo.: ',num2str(i),', 2nd ev.shift: ',num2str(Yinfo{i}.shift)]);
            title({' Predicted probab. (full) and Targed (dotted)';strTit});
            xlabel('samples');
            ylabel('chanels');
        end
        aa=1;
    end

    on0=Yinfo{i}.on0;
    on1=Yinfo{i}.on1;

    Yinfo1=Yinfo{i};

    if plot1 
        disp(['evNo./2nd ev.shift ',num2str([i Yinfo1.shift])]);
    end
    predictedProbab(i,1:14,:) = YpredP;
end

return
end

%-------------------------------------------------
function [signal3,probab3,info3,signal0,info0,probab0]=getSignal3D(str,posTab,len1)
%
% read data
%
% input: file_name in 'str'
% output:   xx3  - subwindowed signals/probab/infos  for training/validataion
%           xx0  - the whole signals/probab/infos for interpratation (prediction)
%

tmp = load(str);
cNNdatS2 = tmp.cNNdatS2;


sigma=4;
sigma2=800;

x=1:1024;

tmp=cNNdatS2;
tmp.sig=[];
info0=tmp;
info0.pP1=zeros(1,14);
info0.pP2=zeros(1,14);



nSig = 1;
shiftSig1 = 64;
on0=double(cNNdatS2.on0);
on1=double(cNNdatS2.on1);
NN=12;   % numbr of shifts to get sub-seismograms

signal3=num2cell(zeros(1,NN));
probab3=num2cell(zeros(1,NN));
info3=num2cell(zeros(1,NN));

signal0=zeros(4,4,1024,3);
probab0=zeros(4,4,1024,3);

for j=1:NN
    shiftSig0 = round(1+rand(1)*shiftSig1/2);
    %    shiftSig0 = 1;

    sig2=zeros(14,len1+1,3);
    probab2=zeros(14,len1+1,3);
    probab14=zeros(14,1024,3);
    for i=1:14
        sig1=cNNdatS2.sig(i,:);

        % for basic signal
        p0=exp(-(x-on0(i)).^2/(2*sigma));
        p2=exp(-(x-on0(i)).^2/(2*sigma2));

        pi=-p0+1;
        p2i=-p2+1;

        n0(1:1024)=0;
        n0(1:on0(i)-2*sigma-1)=1;
        %     disp([i j]);

        fin1=1024-(on0(i)+2*sigma);
        n0(on0(i)-2*sigma : on0(i))=pi(on0(i)-2*sigma:on0(i));
        n0(on0(i)+2*sigma:end)=p2i(on0(i):on0(i)+fin1);

        e0(1:1024)=1;
        e0(1:on0(i)-1)=0;
        e0(on0(i):on0(i)+2*sigma)=pi(on0(i) : on0(i)+2*sigma);
        e0(on0(i)+2*sigma:end)=p2(on0(i) : on0(i)+fin1);

        % for shifted signal
        p1=exp(-(x-on1(i)).^2/(2*sigma));
        p2=exp(-(x-on1(i)).^2/(2*sigma2));

        pi=-p1+1;
        p2i=-p2+1;

        n1(1:1024)=0;
        n1(1:on1(i)-2*sigma-1)=1;
        %     disp([i j]);

        fin1=1024-(on1(i)+2*sigma);
        n1(on1(i)-2*sigma : on1(i))=pi(on1(i)-2*sigma:on1(i));
        n1(on1(i)+2*sigma:end)=p2i(on1(i):on1(i)+fin1);

        e1(1:1024)=1;
        e1(1:on1(i)-1)=0;
        e1(on1(i):on1(i)+2*sigma)=pi(on1(i) : on1(i)+2*sigma);
        e1(on1(i)+2*sigma:end)=p2(on1(i) : on1(i)+fin1);



        n=min([n0;n1]);
        p=max([p0;p1]);
        e=max([e0;e1]);

        prb1=[n; p; e];

        % % %        plot of  probabilities (if desired)
        % % %
        % % aa=1;
        % % if info0.nEv0 ==955  && info0.nEv1==1121
        % %     if i==3 && info0.shift == -120
        % %
        % %         figure; hold on
        % %         plot(prb1');
        % %         s=sum(prb1,1);
        % %         pTot= 1 +p -n -e;
        % %         plot(pTot,':','LineWidth',1.5);
        % % %        plot(s,':');
        % % sig1=sig1/max(abs(sig1));
        % % plot(sig1-1.1)
        % %         legend(' pN',' pP',' pC',' P',' signal');
        % %         set(gca,'ylim',[-2.2 2.1]);
        % %         xlabel('samples');
        % %         aa=1;
        % %     end
        % % end


        sigEnd =  (j-1)*shiftSig1+1+shiftSig0 + len1;
        if sigEnd > 1024, break, end

        for k=1:3
            sig2(i,:,k)=sig1((j-1)*shiftSig1+1+shiftSig0:sigEnd);
            probab2(i,:,k)=prb1(k,(j-1)*shiftSig1+1+shiftSig0:sigEnd);
            probab14(i,:,1:3)=prb1';
        end


    end

    if sigEnd > 1024, break, end

    signal1=zeros(4,4,len1,3);
    probab1=zeros(4,4,len1,3);
    for i=1:16
        pos=posTab(i,:);
        if i <=14
            signal1(pos(1),pos(2),1:len1,1:3)=sig2(i,1:len1,:);
            probab1(pos(1),pos(2),1:len1,1:3)=probab2(i,1:len1,:);
        end
        if i==13
            ii=15;
            pos=posTab(ii,:);
            signal1(pos(1),pos(2),1:len1,1:3)=sig2(i,1:len1,:);
            probab1(pos(1),pos(2),1:len1,1:3)=probab2(i,1:len1,:);
        elseif i==14
            ii=16;
            pos=posTab(ii,:);
            signal1(pos(1),pos(2),1:len1,1:3)=sig2(i,1:len1,:);
            probab1(pos(1),pos(2),1:len1,1:3)=probab2(i,1:len1,:);
        end

    end
    signal3{nSig}=signal1;
    probab3{nSig}=probab1;


    info1=info0;
    info1.on0=info0.on0-(j-1)*shiftSig1+1+shiftSig0;
    info1.on1=info0.on1-(j-1)*shiftSig1+1+shiftSig0;
    info1.TO0o=info0.TO0o-(j-1)*shiftSig1+1+shiftSig0;
    info1.TO1o=info0.TO1o-(j-1)*shiftSig1+1+shiftSig0;
    info3{nSig}=info1;

    %
    % the whole signal
    %
    if j==1
        for i=1:16
            pos=posTab(i,:);
            if i <=14
                signal0(pos(1),pos(2),:,1)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,2)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,3)=cNNdatS2.sig(i,:);
                probab0(pos(1),pos(2),:,1:3)=probab14(i,:,1:3);
            end
            if i==13
                ii=15;
                pos=posTab(ii,:);
                signal0(pos(1),pos(2),:,1)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,2)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,3)=cNNdatS2.sig(i,:);
                probab0(pos(1),pos(2),:,1:3)=probab14(i,:,1:3);
            elseif i==14
                ii=16;
                pos=posTab(ii,:);
                signal0(pos(1),pos(2),:,1)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,2)=cNNdatS2.sig(i,:);
                signal0(pos(1),pos(2),:,3)=cNNdatS2.sig(i,:);
                probab0(pos(1),pos(2),:,1:3)=probab14(i,:,1:3);
            end

        end



    end



    % % %   possible plot for prove
    % % %
    % % %     if info0.nEv0==955 && info0.nEv1==1121 && info0.shift== -120
    % % %         plotCube('',signal3{nSig},probab3{nSig});
    % % %         aa=1;
    % % %     end

    nSig=nSig+1;
    aa=1;

end

signal3(nSig:NN)=[];
probab3(nSig:NN)=[];
info3(nSig:NN)=[];
aa=1;

if info0.TO1o <  info0.TO0o
    info0a=info0;

    info0.evid0=info0a.evId1;
    info0.evid1=info0a.evId0;
    info0.nEv0=info0a.nEv1;
    info0.nEv1=info0a.nEv0;
    info0.on0=info0a.on1;
    info0.on1=info0a.on0;
    info0.Mw0=info0a.Mw1;
    info0.Mw1=info0a.Mw0;
    info0.TO0o=info0a.TO1o;
    info0.TO1o=info0a.TO0o;
    info0.vX=[info0a.vX(2,:);info0a.vX(1,:)];
end

end

%===============eof=============================================
