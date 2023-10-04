classdef RegressionLayerTdiff2f < nnet.layer.RegressionLayer
    % vlatni loss pro rozdil casu predikovanych a vzorovych

    % tady se pouzitje korelace signalu
    % a jeste minimalizace odchylky maxim


    methods
        function layer = RegressionLayerTdiff2f(name)
            % layer = sseClassificationLayer(name) creates a sum of squares
            % error classification layer and specifies the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'T diff 2e';
        end

        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the Tdiff loss between
            % the predictions Y and the training targets T.

            lenSig=600;
            lenSig=400;

            % Calculate

            N1=size(Y,1);
            N2=size(Y,2);

            loss2=0;
            loss3=0;

            vOne=ones(1,lenSig);

            x=1:lenSig;
            sigma=5;

            for ii=1:N1
                for jj=1:N2

                    nStat=(ii-1)*N1 + jj;
%%%                    if nStat > 14, continue, end

                    diff=0;

                    T1=T(ii,jj,:,2);
                    T1=(squeeze(T1))';

                    Y1=Y(ii,jj,:,1);
                    Y1=(squeeze(Y1))';
                    Y2=Y(ii,jj,:,2);
                    Y2=(squeeze(Y2))';
                    Y3=Y(ii,jj,:,3);
                    Y3=(squeeze(Y3))';

%                      Y1=movmean(Y1,5);
%                      Y2=movmean(Y2,5);
%                      Y3=movmean(Y3,5);

                    Ys = Y2 + (1 - Y1 - Y3);

                    % pozice cile
                    [~,posPt]=findpeaks(T1,'MinPeakProminence',0.9);
                    [valPy,posPy]=findpeaks(Ys,'MinPeakProminence',0.05,'Sortstr','descend',MinPeakDistance=10);

                    lenT=length(posPt);
                    lenT=min([lenT 2]);

                    lenY=length(posPy);

                    nPos = min([lenY 5]);
                    YsG = zeros(1,lenSig);

                    for k=1:nPos
                        p0 = valPy(k) * exp(-(x-posPy(k)).^2/(2*sigma));
                        YsG = YsG + p0;
                    end



                    switch lenT
                        case 0
                            if lenY == 0
                                continue
                            else
                                diff = dot(vOne,YsG) / (norm(vOne)*norm(YsG));
                            end

                        otherwise
                            if lenY == 0
                                diff = dot(T1,vOne) / (norm(T1)*norm(vOne));
                            else
                                diff = dot(T1,YsG) / (norm(T1)*norm(YsG)) ;
                            end

                    end


                    %                diff = (Tpred-Ttarget) ./ lenSig05;
                    diff = -diff + 1;
                    %                   diff = diff ./lenSig ;
                    %                loss2 = loss2 + log10(1+norm(diff));
                    loss2 = loss2 + diff;


                    % tady se spocte ta odchylka  (relativni)
                    if length(posPt) == 1, posPt(2)=posPt(1); end
                    if length(posPy) == 1, posPy(2)=posPy(1); end

                    if isempty(posPy) && isempty(posPt)
                        d12(1:2)=0;
                        d21(1:2)=0;
                    elseif isempty(posPt)
                        d12(1)=min([posPy(1) posPy(1)-lenSig]);
                        d12(2)=min([posPy(2) posPy(2)-lenSig]);

                        d12=d12./lenSig;
                        d21(1:2)=1;

                    elseif isempty(posPy)
                        d12(1)=min([posPt(1) posPt(1)-lenSig]);
                        d12(2)=min([posPt(2) posPt(2)-lenSig]);

                        d12=d12./lenSig;
                        d21(1:2)=1;
                    else
                        d12(1)=(posPt(1)-posPy(1))/lenSig;
                        d12(2)=(posPt(2)-posPy(2))/lenSig;
                        d21(1)=(posPt(1)-posPy(2))/lenSig;
                        d21(2)=(posPt(2)-posPy(1))/lenSig;


                    end
                    dd12=sum(d12.^2);
                    dd21=sum(d21.^2);


                    difMax=dd12;
                    if dd21 < dd12
                        difMax=dd21;
                    end



                    loss3 = loss3 + difMax;
                end
            end

            % Calculate sum of squares.
            sumSquares = sum((Y-T).^2);

            % Take mean over mini-batch.
            N = size(Y,4);
            %%%loss0 = sum(sumSquares)/N;

            squares = 0.5*(Y-T).^2;
            %            numElements = prod( size(Y, [this.ObservationDim this.ObservationDim+1]) );
            dim=ndims(Y);
            numElements = prod( size(Y, [dim dim+1]) );
            loss0 = sum( squares (:) ) / numElements;

%           loss3=0;

            loss = single(loss0 /(4*N) + loss2/N + loss3/(2*N));
        end



        function dX = backwardLoss(layer, Y, T)

            numObservations = size( Y, 3);
            dX = (Y - T)./numObservations;
        end

    end
end




%{
classdef sseClassificationLayer < nnet.layer.ClassificationLayer
    % Example custom classification layer with sum of squares error loss.
    
    methods
function layer = sseClassificationLayer(name)
            % layer = sseClassificationLayer(name) creates a sum of squares
            % error classification layer and specifies the layer name.
    
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Sum of squares error';
        end
        
function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SSE loss between
            % the predictions Y and the training targets T.

            % Calculate sum of squares.
            sumSquares = sum((Y-T).^2);
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sum(sumSquares)/N;
        end
    end
end
%}