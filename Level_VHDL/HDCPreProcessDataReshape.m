
%load preprocessed
load('C:\Users\seanl\Documents\Research\bins_sets.mat');

%convert to 2d matrix
%bins_sets = permute(bins_sets,[1 3 2]);
%% store every bin value in one dimension so should be 1500 x 30 after
bins_sets = reshape(bins_sets,[size(bins_sets,2)*size(bins_sets,1),size(bins_sets,3)]);
bins_vector = bins_sets(:);

savdir = 'C:\Users\seanl\Documents\Research\';
%featureMX = featureMX';
save(fullfile(savdir, 'bins_vector'), 'bins_vector');