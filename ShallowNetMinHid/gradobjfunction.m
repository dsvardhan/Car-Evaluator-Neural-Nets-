function gradientvector = gradobjfunction(thetavector,constants,thedata)
% USAGE: gradientvector = gradobjfunction(thetavector,constants,thedata)

% "gradientvector" is a column vector defined such that the first
% set of H+1 elements are defined as the derivative of the objective
% function with respect to the row vector "vmatrix" and whose second
% set of D elements are defined as the derivative of the objective
% function with respect to the connection weights from the input
% units to the first hidden unit and whose third set of elements
% are defined as the derivative of the objective function with respect
% to the second hidden unit and so on...
% (H is the number of hidden units and D is the number of input units)

%--------------------------------------------------------------------------------------------------
%------------------------------BEGIN function "getpredictedresponses"------------------------------
%--------------------------------------------------------------------------------------------------
    function [predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata)
    % USAGE: [predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata)

    % Get Predicted Responses using "getpredictedresponses.m"
    % The ith column of "hidmx" is the activation pattern over the
    % H hidden units for the ith input pattern which is the
    % ith row of "inputvectors" (see definition of "inputvectors" below)
    %
    % The ith column of the row vector "predictedresponses" is the activity
    % of the ith output unit given the ith input pattern which is the ith
    % row of input vectors (see definition of "inputvectors" below)

    % Unpack Constants
    nrhidden = constants.nrhidden;

    % Unpack Event History (same as "gradobjfunction.m")
    eventhistory = thedata.eventhistory;
    nrtargets = thedata.nrtargets;
    [nrstim,nrvars] = size(eventhistory);

    % Get "desiredresponses". The ith row of "desired response"
    % for the ith row of "inputvectors". The matrices "desiredresponse"
    % and "inputvectors" comprise the "training data set"
    desiredresponse = eventhistory(:,(1:nrtargets));
    inputvectors = eventhistory(:,(nrtargets+1):nrvars);
    [nrstim,inputvectordim] = size(inputvectors);

    % Unpack parameter values
    % Let H be number of hidden units. Let each input vector have
    % dimension D. 
    % wmatrix: matrix with H rows and D columns as defined in problem
    % vmatrix: matrix with 1 rows and H+1 columns as defined in problem
    [wmatrix,vmatrix] = unpackparameters(thetavector,constants,thedata);

    % STUDENT: Now Compute Responses to Hidden Units 
    hidmx = logisticsigmoid(inputvectors * wmatrix');
    biased_hidmx = [hidmx, ones(size(hidmx, 1), 1)];

    % STUDENT: Now compute output unit responses 
     predictedresponses = logisticsigmoid(biased_hidmx * vmatrix');
    predictedresponses = predictedresponses'; 

    hidmx = hidmx';
    end  % END OF function "getpredictedresponses"
%------------------------------------------- END FUNCTION "getpredictedresponses"--------------


%--------------------------------BEGIN MAIN FUNCTION ------------------------------------------

% MODEL CONSTANTS
% Unpack Constants
lambda = constants.lambda;
nrhidden = constants.nrhidden;

% Unpack Event History
eventhistory = thedata.eventhistory;
nrtargets = thedata.nrtargets;
[nrstim,nrvars] = size(eventhistory);

% Get Desired Responses and Input Vectors
% "inputvectors" is a matrix whose ith row is an input vector
% "desiredresponse" is a matrix whose ith row is the desired response
%     to the ith input vector (ith row of "inputvectors")
desiredresponse = eventhistory(:,(1:nrtargets));
inputvectors = eventhistory(:,(nrtargets+1):nrvars);
[nrstim,inputvectordim] = size(inputvectors);
inputvectorsT = inputvectors';

% Unpack Parameter Vector
% Let H be number of hidden units. Let each input vector have
% dimension D. 
% wmatrix: matrix with H rows and D columns as defined in problem
% vmatrix: matrix with 1 rows and H+1 columns as defined in problem
[wmatrix,vmatrix] = unpackparameters(thetavector,constants,thedata);

% Now Compute Responses to Hidden Units 
% Get Predicted Responses using "getpredictedresponses.m"
% The ith column of "hidmx" is the activation pattern over the
% H hidden units for the ith input pattern which is the
% ith row of "inputvectors"
%
% The ith column of the row vector "predictedresponses" is the activity
% of the ith output unit given the ith input pattern which is the ith
% row of input vectors
[predictedresponses,hidmx] = getpredictedresponses(thetavector,constants,thedata);


% STUDENT: Compute Derivative of Objective Function with
% with respect to the learning machine's parameters
% This is a vector called "gradientvector" whose first
% set of H+1 elements are defined as the derivative of the objective
% function with respect to the row vector "vmatrix" and whose second
% set of D elements are defined as the derivative of the objective
% function with respect to the connection weights from the input
% units to the first hidden unit and whose third set of elements
% are defined as the derivative of the objective function with respect
% to the second hidden unit and so on...
gradientvector = zeros(nrhidden + 1 + nrhidden*inputvectordim, 1);
dl_dv = zeros(1, nrhidden+1);
for i = 1:nrstim
    dl_dv(1,:) = dl_dv(1,:) + (predictedresponses(1,i) - desiredresponse(i)) * [hidmx(:,i)', 1];
end
dl_dv(1,:) = dl_dv(1,:)/nrstim;

dl_dw = zeros(nrhidden, inputvectordim);
for j = 1:nrhidden
    for k = 1:inputvectordim
        for i = 1:nrstim
            constant = (predictedresponses(1,i) - desiredresponse(i));
            dl_dw(j,k) = dl_dw(j,k) + constant * vmatrix(1,j) * hidmx(j,i)*(1-hidmx(j,i))*inputvectors(i,k);
        end
        dl_dw(j,k) = dl_dw(j,k)/nrstim;  
    end
    [row, col] = size(dl_dw(j,:));
    value = dl_dw(j,:);
    disp(['row ',num2str(row), ' col ', num2str(col), ' value ', num2str(value)]);
end

for i = 1:nrhidden+1
    gradientvector(i,1) = dl_dv(1,i);
end

count = 1;
for j = 1:nrhidden
    for k = 1:inputvectordim
        gradientvector(nrhidden + 1 + count,1) = dl_dw(j,k);
        count = count + 1;
    end
end



end
