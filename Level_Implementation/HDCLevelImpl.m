%%Sean Lane
%Hyperdimensional Computing Speech Recognition
%Thesis Research
%Villanova University
%Fall 2021 - Spring 2022


%% Training Phase Execution
%tic
bins = 75;
dim = 2880;

%use if need to generate HVs
[digitHV, posHV] = GetBaseHVs(bins);
num_epochs = 20;

%load preprocessed
load('C:\Users\seanl\Documents\Research\bins_sets.mat');

% Set global variables

classes = [{'down'},{'eight'},{'five'},{'four'}];
% classes = [{'bed'},{'bird'},{'cat'},{'dog'},{'down'},{'eight'}...
%     {'five'},{'four'},{'go'},{'happy'},{'house'},{'left'},{'marvel'}...
%     {'nine'},{'no'},{'off'},{'on'},{'one'},{'right'},{'seven'}...
%     {'sheila'},{'six'},{'stop'},{'three'},{'tree'},{'two'},{'up'}...
%     {'wow'},{'yes'},{'zero'}];                         %Different classes

num_classes = length(classes);
total_records = 100;                                     %Number of audio samples for each class
kFold = 5;
test_records = total_records/kFold;                      %To use 20% as testing samples
sample_records = total_records - test_records;

indices = crossvalind('Kfold',total_records,kFold);      %5-fold crossvalind


maxFileLength = 16384;                                   %the files of this dataset are 16000, use 16384 for mult of 2 (fft)

for foldIDX = 1:kFold
fprintf('Fold: %d \n',foldIDX);
fprintf('\n');
    
Test_idx = (indices == foldIDX);                          %Generating random indexes to choose test samples
totalCounts = zeros(length(classes),1);                   %Store total samples executed
sampleCounts = zeros(length(classes),1);                  %Store training samples executed
testCounts = zeros(length(classes),1);                    %Store test samples executed
bins_train_sets = zeros(sample_records,bins,num_classes); %Bins for the training samples
bins_test_sets = zeros(test_records,bins,num_classes);    %Bins or the testing samples


%For loop to split samples into testing and training based on random split
%80/20
    for i = 1:num_classes
        for k = 1:100
                totalCounts(i) = totalCounts(i) + 1;
                if(Test_idx(totalCounts(i),1) == 1)
                    testCounts(i) = testCounts(i) + 1;
                    bins_test_sets(testCounts(i),:,i) = bins_sets(k,:,i);
                else
                    sampleCounts(i) = sampleCounts(i) + 1;
                    bins_train_sets(sampleCounts(i),:,i) = bins_sets(k,:,i);
                end
        end
    end

%Stores encoded hypervectors for sample and test samples
enc_HVs = zeros(sample_records,dim,num_classes);
enc_test_HVs = zeros(test_records,dim,num_classes);

tic
%Sending bins for training and testing to be audio encoded then stored in
%respected matrices
        for i = 1:length(classes)
                for j = 1:sample_records
                    enc_HVs(j,:,i) = AudioEncoder(bins_train_sets(j,:,i), digitHV, posHV);
                end
                for k = 1:test_records
                    enc_test_HVs(k,:,i) = AudioEncoder(bins_test_sets(k,:,i), digitHV, posHV);
                end
        end 


% if even, subtract 1; if odd, leave be
%last_hv = ((sample_records - 1) + mod(sample_records,2));
%Hardcoded for 80  sample records
last_hv = 79;

%Matrix to store Class Hypervectors
class_HVs = Train(enc_HVs(1:last_hv,:,:),dim);
%Second matrix to store next class HV values between retraining
next_epoch_class_HVs = class_HVs;
toc
%% Sample inference test
test_HVs = enc_test_HVs;
%total = 0;
%correct = 0;

%Number of passes through dataset - retraining done in each epoch
for l=1:num_epochs

%counts for total and corret predictions
correct = 0; 
total = 0;

class_HVs = next_epoch_class_HVs;

check = zeros(1,num_classes);
difference_array = zeros(2880,1);
classHV_Manhattan_distance = zeros(length(classes),test_records,length(classes));
for j = 1:length(classes)
    for i = 1:test_records
        for k = 1:length(classes)
            %Get manhattan distancefrom each class HV to the current sample
            %test HV
            classHV_Manhattan_distance(k,i,j) = sum(abs(class_HVs(k,:) - test_HVs(i,:,j)));
            check(k) = classHV_Manhattan_distance(k,i,j);
            if(k == length(classes))
                %Check which class had lowest Manhattan distance
                total = total + 1;
                [M,I] = min(check);
                if ( I == j)
                    correct = correct + 1;
                else
                    %difference array stores the difference between each
                    %class and the test HV, this is / by 80 (for number of
                    %samples) and then added to incorrectly predicted class
                    %HV,and subtracted from the correct class HV
                    difference_array = (class_HVs(I,:) - test_HVs(i,:,j))/sample_records;
                    difference_array = difference_array;
                    %difference_array = floor(difference_array);
                    %subtract the hypervector from the correct guess
                    %class HV, add it to the incorrect HV. Correct is j,
                    %incorrect is I
                    next_epoch_class_HVs(j,:) = next_epoch_class_HVs(j,:) - difference_array;
                    next_epoch_class_HVs(I,:) = next_epoch_class_HVs(I,:) + difference_array;
                end
                
            end
        end
    end
    %fprintf("%i\n\n", sum(xor(class_HVs(i,:),class_HVs((mod(i,(classes/2))+1),:)))); 
end
    fprintf('Number of Bins: %d Number Correct: %d / %d Epoch Number: %d \n',bins, correct,total, l);
    fprintf('\n');

end 
end
%toc


function [level_HVs, pos_HVs] = GetBaseHVs(num_bins)
%% Initial Hypervector Generation
% Gives 2 hypervector sets: the 10 digit vectors and a number of position
%       vectors dependent on the number of audio bins.
%
% Takes num_bins, the number of bins of the audio samples.
%
% Returns level_HVs, the set of 10 hypervectors corresponding to the
%       numbers 0 through 9.
% Returns pos_HVs, the set of hypervectors corresponding to the
%       positions of the bins of the audio sample.

    D = 2880;
    %M Is used for the randi function to decide number of bits to flip in
    %HV
    M = 10;
   
    lev_HVs(1,:) = randi([0,1],1,D);

    %High accuracy, but uncorrelated levels
    %Not for VHDL implementation because uses randi

    for d = 2:117
        lev_HVs(d,:) = lev_HVs(d-1,:);
        rand = randi(D,(D/M),1);
        for i = 1:length(rand)
            lev_HVs(d,rand(i)) = ~lev_HVs(d,rand(i));
        end
    end

%Correlated HVs, not as accurate and the randi level HVs but easier to
%implement in VHDL and proven in previous HDC implementations to be
%effective
%     for d = 2:117
%         lev_HVs(d,:) = lev_HVs(d-1,:);
%         for i = 1:24
%             lev_HVs(d,((d-2)*24)+i) = ~lev_HVs(d,((d-2)*24)+i); %24
%         end
%     end
       
    
    ID_HVs = zeros(num_bins,D); 
    for p = 1:num_bins
        ID_HVs(p,:) = randi([0,1],1,D);
    end
    
    level_HVs = lev_HVs;
    pos_HVs = ID_HVs;

end


function [audio_HV] = AudioEncoder(bin_values, level_HVs, pos_HVs)
%% Audio-Hypervector Encoding
% Encodes the audio sample into one hypervector using the audio bins and
%       digit and bin position hypervectors.
%256
% Takes bin_values, the set of decimal numbers representing the average
%       frequency values of one audio sample.
% Takes level_HVs, the set of 10 hypervectors, each corresponding to a
%       digit (the number 0 through 9).
% Takes pos_HVs, the set of hypervectors corresponding to the positions of
%       the bins of the audio sample.
%
% Returns audio_HV, the hypervector representation of the audio sample

    audlevel_HVs = zeros(75,2880);  %75
    for i = 1:length(bin_values)
        %audlevel_HVs(i,:) = BinEncoder(bin_values(i), level_HVs);
        audlevel_HVs(i,:) = level_HVs((bin_values(i)+1), :);
        %audlevel_HVs(i,:) = level_HVs(bin_values(i), :);
    end
    
    audio_HV = audlevel_HVs(1,:) .* pos_HVs(1,:);
    for i = 2:size(audlevel_HVs,1)
        audio_HV = audio_HV + (audlevel_HVs(i,:) .* pos_HVs(i,:));
    end
end


function [class_HVs] = Train(HV_sets,dim)
%% Training Function
% Takes HV_sets, a three-dimensional array that represents the full set of
%       all of the hypervectors, where each is a one-dimensional array, and
%       they are organized into two-dimensional arrays based on the class.
%   
% Returns class_HVs, a two-dimensional array of the trained class
%       hypervectors that are represented as one-dimensional arrays
class_HVs = zeros(size(HV_sets,3),dim);
    for i = 1:size(HV_sets,3)
        class_set = HV_sets(:,:,i);
        div_num = (size(class_set,1) + 1);
        
        for j = 1:size(class_set,2)
            col_sum = sum(class_set(:,j));
            class_HVs(i,j) = fix(abs(col_sum/div_num));
        end
    end

end
