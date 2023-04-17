function [time] = extract_hctsa(datapath, savepath, dataset_name, n_features, split)

cd ~
data = importdata(fullfile(datapath, dataset_name, ...
    strcat(dataset_name, '_', upper(split), '.txt')));

shape = size(data);
X = data(:,2:shape(2));
y = data(:,1);

cd(fullfile(savepath, 'datasets_mat'))

timeSeriesData = transpose(num2cell(X,2));
keywords = transpose(num2cell(num2str(y),2));
labels = transpose(num2cell(num2str(transpose(1:shape(1))),2));
mat_file = strcat(dataset_name, '_', lower(split),'.mat');

save(mat_file, 'timeSeriesData', 'keywords', 'labels')

%---------------------------------------------------------------
%Begin extraction
%---------------------------------------------------------------

TS_Init(mat_file,'INP_mops.txt','INP_ops.txt',[false,false,false]);
tic;
TS_Compute(true,1:shape(1),1:n_features,'missing','raw',false);
time = toc;

load('HCTSA.mat', 'TS_DataMat');
matrix = rmmissing(TS_DataMat, 2);

delete HCTSA.mat

cd ~
if ~exist(fullfile(savepath, dataset_name), 'dir')
    mkdir(fullfile(savepath, dataset_name))
end

path = fullfile(savepath, dataset_name);
cd(path);

filename = strcat('hctsa_', lower(split), '.csv');
writematrix(matrix, filename);

end 