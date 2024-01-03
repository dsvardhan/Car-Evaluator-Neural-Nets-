function [trainingerror,testingerror,traincorrect,testcorrect] = simulationrun();
lambdavalues = [0,0.0001,0.001,0.01,0.1,1.0]
replications = 3;
nrlambdavalues = length(lambdavalues);
nrhidden = 30; nrhiddenindex = 1;
for j = 1:replications,
    for lambdaindex = 1:nrlambdavalues,
        lambda = lambdavalues(lambdaindex);
        disp('----------------------------------------------------------------------------------')
        disp(['Simulation: Replication = ',num2str(j),', Lambda = ',num2str(lambda)]);
        disp('----------------------------------------------------------------------------------');
        pause(2);
        [trainingerror,testingerror,traincorrect,testcorrect] = runperceptron(lambda,nrhidden);
        trainingerrorlist(j,lambdaindex) = trainingerror;
        testingerrorlist(j,lambdaindex) = testingerror;
        traincorrectlist(j,lambdaindex) = traincorrect;
        testcorrectlist(j,lambdaindex) = testcorrect;
    end;
end;


% Now Compute Statistics
meantrainerror = mean(trainingerrorlist);
meantesterror = mean(testingerrorlist);
meantraincorrect = 100*mean(traincorrectlist);
meantestcorrect = 100*mean(testcorrectlist);

% Plot Simulation Results
% Now Compute Statistics
meantrainerror = mean(trainingerrorlist);
meantesterror = mean(testingerrorlist);
meantraincorrect = 100*mean(traincorrectlist);
meantestcorrect = 100*mean(testcorrectlist);

% Plot Simulation Results
subplot(1,2,1);
semilogx(lambdavalues,meantrainerror,'--gx',lambdavalues,meantesterror,'-ro','LineWidth',2,'MarkerSize',4);
legend('Train','Test');
ylabel('Prediction Error');
xlabel('Lambda');
subplot(1,2,2);
semilogx(lambdavalues,meantraincorrect,'--gx',lambdavalues,meantestcorrect,'-ro','LineWidth',2,'MarkerSize',4);
legend('Train','Test');
ylabel('% Correct');
xlabel('Lambda');

% Print out Simulation Results
disp('Copy these vectors into your answer in MATLAB GRADER!')
meantrainerror
meantesterror
meantraincorrect
meantestcorrect

% Save results in local file
save results; % Save simulation results in file "results.mat" 
              % Load these results by typing: load results.mat
