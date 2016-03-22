%% Preliminary data analysis
% Get number of rows in file
% N = numel(textread('data/intersected_final_chr1_cutoff_20_train.bed','%1c%*[^\n]'));
% 
% % Load all training data into a matrix
% fileID = fopen('data/intersected_final_chr1_cutoff_20_train.bed','r');
% trainData = zeros(N,37);
% iter = 1;
% while ~feof(fileID)
%     fprintf('Reading training data: %d/%d\n', iter, N);
%     tline = fgets(fileID);
%     C = textscan(tline,'chr1 %f %f %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
%     if strcmp(C{3},'+')
%         C{3} = 1;
%     else
%         C{3} = 0;
%     end
%     trainData(iter,:) = cell2mat(C);
%     iter = iter + 1;
% end
% fclose(fileID);
% 
% % figure(1); hold on;
% % plot(trainData(:,2),trainData(:,9));
% % set(gca,'FontName','Helvetica','fontsize',16); 
% % hxlabel = ylabel('\beta','fontweight','bold','fontsize',20);
% % hylabel = xlabel('CpG Site (Position)','fontweight','bold','fontsize',16);
% % set(gca,'FontName','Helvetica'); 
% % hold off;
% 
% figure(2); hold on;
% for i=4:37
%     plot(trainData(100000:100100,2),trainData(100000:100100,i));
% end
% set(gca,'FontName','Helvetica','fontsize',16); 
% hxlabel = ylabel('\beta','fontweight','bold','fontsize',20);
% hylabel = xlabel('CpG Site (Position)','fontweight','bold','fontsize',16);
% set(gca,'FontName','Helvetica'); 
% hold off;

%% Step 1 - fill holes in training data
% % Fill holes with mean values
% for i=1:size(trainData,1)
%     fprintf('Filling holes in training data: %d/%d\n', i, N);
%     currRow = trainData(i,4:37);
%     nanInd = find(isnan(currRow));
%     currRow(nanInd) = mean(currRow(setdiff(1:34,nanInd)));
%     trainData(i,4:37) = currRow;
% end
% 
% %% Step 2 - do prediction of testing data
% % Load all testing data into a matrix
% fileID = fopen('data/intersected_final_chr1_cutoff_20_sample_partial.bed','r');
% testData = zeros(N,5);
% iter = 1;
% while ~feof(fileID)
%     fprintf('Reading testing data: %d/%d\n', iter, N);
%     tline = fgets(fileID);
%     C = textscan(tline,'chr1 %f %f %s %f %f');
%     if strcmp(C{3},'+')
%         C{3} = 1;
%     else
%         C{3} = 0;
%     end
%     testData(iter,:) = cell2mat(C);
%     iter = iter + 1;
% end
% fclose(fileID);
% 
% % Load all ground truth data into a matrix
% fileID = fopen('data/intersected_final_chr1_cutoff_20_sample_full.bed','r');
% trueData = zeros(N,5);
% iter = 1;
% while ~feof(fileID)
%     fprintf('Reading ground truth data: %d/%d\n', iter, N);
%     tline = fgets(fileID);
%     C = textscan(tline,'chr1 %f %f %s %f %f');
%     if strcmp(C{3},'+')
%         C{3} = 1;
%     else
%         C{3} = 0;
%     end
%     trueData(iter,:) = cell2mat(C);
%     iter = iter + 1;
% end
% fclose(fileID);

predData = testData;
predInd = find(isnan(predData(:,4)));
labelInd = find(~isnan(predData(:,4)));

% Do sample weighting
sampleWeights = 1 - mean(abs(repmat(predData(labelInd,4),1,33)-trainData(labelInd, 4:36)));

labelWeights = (1 - abs(repmat(predData(labelInd,4),1,33)-trainData(labelInd, 4:36)));
labelWeights = sqrt(sum(((repmat(sampleWeights,size(labelInd,1),1)-labelWeights).^2),2));
labelWeights = (max(labelWeights) - labelWeights)/max(labelWeights);

% % Predict using mean
% predData(predInd,4) = mean(trainData(predInd, 4:36), 2);

% % Predict using linear regression model
% mdl = fitlm(trainData(labelInd, 4:36),predData(labelInd,4),'RobustOpts','on','Weights',labelWeights);
% predData(predInd,4) = predict(mdl,trainData(predInd, 4:36));

% Predict using decision forests
mdl = TreeBagger(1,trainData(labelInd, 4:36),predData(labelInd,4),'NumPrint',1);
predData(predInd,4) = predict(mdl,trainData(predInd, 4:36));

% % Compute RMSE
% validInd = setdiff(1:N, find(isnan(trueData(:,4))));
% RMSE = sqrt(mean((predData(validInd,4) - trueData(validInd,4)).^2))




