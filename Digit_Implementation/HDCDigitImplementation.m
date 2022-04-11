%%Sean Lane
%Hyperdimensional Computing Speech Recognition
%Thesis Research
%Villanova University
%Fall 2021 - Spring 2022


%% Training Phase Execution
tic
bins = 150;
dim = 2880;

%use if need to generate HVs
[digitHV, posHV] = GetBaseHVs(bins);
num_epochs = 5;

%load preprocessed
load('C:\Users\seanl\Documents\Research\digitbins_sets.mat');

% Set global variables

classes = [{'down'},{'eight'},{'five'},{'four'}];
%classes = [{'bed'},{'bird'},{'cat'},{'dog'},{'down'},{'eight'}...
%    {'five'},{'four'},{'go'},{'happy'},{'house'},{'left'},{'marvel'}...
%    {'nine'},{'no'},{'off'},{'on'},{'one'},{'right'},{'seven'}...
%    {'sheila'},{'six'},{'stop'},{'three'},{'tree'},{'two'},{'up'}...
%    {'wow'},{'yes'},{'zero'}];                         %Different classes
num_classes = length(classes);
total_records = 100;  %Number of audio samples for each class
kFold = 5;
test_records = total_records/kFold;                  %To use 20% as testing samples
sample_records = total_records - test_records;

indices = crossvalind('Kfold',total_records,kFold);     %5-fold crossvalind


maxFileLength = 16384; %the files of this dataset are 16000, use 16384 for mult of 2 (fft)
% audio_sets = zeros(16384,sample_records,length(classes));
% audio_test_sets = zeros(16384,test_records,length(classes));
% totalCounts = zeros(length(classes),1);
% sampleCounts = zeros(length(classes),1);
% testCounts = zeros(length(classes),1);
% 
% 
% %loading each .wav file, converting to .mat, storing in mx, running 
% MyDir = 'C:\Users\seanl\Documents\Research\SingleWordsSmall\';
%     for i = 1:length(classes)
%         name = strcat(MyDir,char(classes(i)), '\');
%         files1 = dir(name);
%         for k = 3:length(files1)                         % looping through records of same motions
%             path_name = strcat(name,files1(k).name); 
%                 totalCounts(i) = totalCounts(i) + 1;
%                 [sig,fs] = audioread(path_name);
%                 if(length(sig) < maxFileLength)
%                     sig(end:maxFileLength)=0;
%                 end
%                 if(Test_idx(totalCounts(i),1) == 1)
%                     testCounts(i) = testCounts(i) + 1;
%                     audio_test_sets(:,testCounts(i),i) = sig;
%                 else
%                     sampleCounts(i) = sampleCounts(i) + 1;
%                     audio_sets(:,sampleCounts(i),i) = sig;
%                 end
%         end
%     end
%     
%     
% bins_sets = zeros(sample_records,bins,num_classes);
% bins_test_sets = zeros(test_records,bins,num_classes);
%         for i = 1:length(classes)
%             bins_sets(:,:,i) = GetFreqBins(audio_sets(:,:,i), bins);
%             bins_test_sets(:,:,i) = GetFreqBins(audio_test_sets(:,:,i), bins);
%           
%         end 




for foldIDX = 1:kFold
totalCounts = zeros(length(classes),1);
sampleCounts = zeros(length(classes),1);
testCounts = zeros(length(classes),1);
 
Test_idx = (indices == foldIDX);%Generating random indexes to choose test samples
%Train_idx = ~Test_idx;   
bins_train_sets = zeros(sample_records,bins,num_classes);
bins_test_sets = zeros(test_records,bins,num_classes);

    for i = 1:num_classes
        for k = 1:100
                totalCounts(i) = totalCounts(i) + 1;
                if(Test_idx(totalCounts(i),1) == 1)
                    testCounts(i) = testCounts(i) + 1;
                    bins_test_sets(testCounts(i),:,i) = digitbins_sets(k,:,i);
                else
                    sampleCounts(i) = sampleCounts(i) + 1;
                    bins_train_sets(sampleCounts(i),:,i) = digitbins_sets(k,:,i);
                end
        end
    end

% for loading files into a 3-D matrix
% matrix declaration
% maxFileLength = 16384; %the files of this dataset are 16000, use 16384 for mult of 2 (fft)
% audio_sets = zeros(16384,sample_records,length(classes));
% audio_test_sets = zeros(16384,test_records,length(classes));
% totalCounts = zeros(length(classes),1);
% sampleCounts = zeros(length(classes),1);
% testCounts = zeros(length(classes),1);
% bins_sets = zeros(sample_records,bins,num_classes);
% bins_test_sets = zeros(test_records,bins,num_classes);
enc_HVs = zeros(sample_records,dim,num_classes);
enc_test_HVs = zeros(test_records,dim,num_classes);
% %loading each .wav file, converting to .mat, storing in mx, running 
% MyDir = 'C:\Users\seanl\Documents\Research\SingleWordsSmall\';
%     for i = 1:length(classes)
%         name = strcat(MyDir,char(classes(i)), '\');
%         files1 = dir(name);
%         for k = 3:length(files1)                         % looping through records of same motions
%             path_name = strcat(name,files1(k).name); 
%                 totalCounts(i) = totalCounts(i) + 1;
%                 [sig,fs] = audioread(path_name);
%                 if(length(sig) < maxFileLength)
%                     sig(end:maxFileLength)=0;
%                 end
%                 if(Test_idx(totalCounts(i),1) == 1)
%                     testCounts(i) = testCounts(i) + 1;
%                     audio_test_sets(:,testCounts(i),i) = sig;
%                 else
%                     sampleCounts(i) = sampleCounts(i) + 1;
%                     audio_sets(:,sampleCounts(i),i) = sig;
%                 end
%         end
%     end
%     
    
        for i = 1:length(classes)
%             bins_sets(:,:,i) = GetFreqBins(audio_sets(:,:,i), bins);
%             bins_test_sets(:,:,i) = GetFreqBins(audio_test_sets(:,:,i), bins);
            
            %up to here should be preprocessed
                for j = 1:sample_records
                    enc_HVs(j,:,i) = AudioEncoder(bins_train_sets(j,:,i), digitHV, posHV);
                end
                for k = 1:test_records
                    enc_test_HVs(k,:,i) = AudioEncoder(bins_test_sets(k,:,i), digitHV, posHV);
                end
        end 


% if even, subtract 1; if odd, leave be
last_hv = ((sample_records - 1) + mod(sample_records,2));


% everything before this would be in the testbench?
class_HVs = Train(enc_HVs(1:last_hv,:,:),dim);
next_epoch_class_HVs = class_HVs;

%% Sample inference test
test_HVs = enc_test_HVs;
%total = 0;
%correct = 0;


for l=1:num_epochs

correct = 0; 
total = 0;

class_HVs = next_epoch_class_HVs;

check = zeros(1,num_classes);
difference_array = zeros(10000,1);
classHV_Manhattan_distance = zeros(length(classes),test_records,length(classes));
for j = 1:length(classes)
    for i = 1:test_records
        for k = 1:length(classes)
            classHV_Manhattan_distance(k,i,j) = sum(abs(class_HVs(k,:) - test_HVs(i,:,j)));
            check(k) = classHV_Manhattan_distance(k,i,j);
            if(k == length(classes))
                total = total + 1;
                [M,I] = min(check);
                if ( I == j)
                    correct = correct + 1;
                else
                    difference_array = (class_HVs(I,:) - test_HVs(i,:,j))/sample_records;
                    %subtract the hypervector from the correct guess
                    %class HV, add it to the incorrect HV. Correct is j,
                    %incorrect is I
                    %next_epoch_class_HVs(I,:) = next_epoch_class_HVs(I,:) + (test_HVs(i,:,j)/sample_records);
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
toc
function [bin_values] = GetFreqBins(audioData, num_bins)
%% Frequency Bin Generation
% Turns audio samples into sets of frequency bins.
%
% Takes audioData, the audio samples.
% Takes num_bins, the number of bins of the audio samples.
%
% Returns bin_values, the set of bin values split into sections.

    num_takes = size(audioData,2);

    %n = length(audioData);
    %x = 2; %looking for closes power of 2 for fft
    %n = n + x/2;
    %n = n - mod(n,x);
  
    %numberzeros = n - length(audioData); %number of zeros to pad
    
    %for i = 1:num_takes
        
    %end
    n=0;
    W=[1 + 0i, 0 - 1i, -1 - 0i,  -0 + 1i]; %result of exp(-1i*2*pi*(0:4-1)/4);
    X = zeros(16384,num_takes);
    for i=1:num_takes
        %myFFt replacement fft attempt
        %X(:,i) = myFFT(audioData(:,i));
        
        %Radix 4 FFT (from video) attempt
        %X(:,i) = bitrevorder(radix4FFThdl(audioData(:,i),n,W));
        
        %original fft, cannot be converted
        X(:,i) = fft(audioData(:,i));
    end
    
    %X(:,1:num_takes) = fft(audioData(:,1:num_takes));
    
    Z = zeros(16000,num_takes);
    for s = 1:num_takes
        Z(:,s) = abs(X(1:16000,s));
    end
    
    total = length(1:length(Z(:,1)));
    
    binsize = total/num_bins;
    
    average = zeros(num_takes,num_bins);
    for s = 1:num_takes
        for i = 1:num_bins
            if ( i == 1)
                average(s,i) = mean(0:Z(round((i*binsize)+1),s));
            else
                average(s,i) = mean(Z(round(((i-1)*binsize)+1):round((i)*binsize),s));
            end
        end
    end
    
    bin_values = average;
end

function [X] = myFFT(x)

% N = length(x);        % The fundamental period of the DFT
% n = 0:1:N-1;           % row vector for n
% k = 0:1:N-1;           % row vecor for k
% WN = cordiccexp(real(-1i*2*pi/N));  % Wn factor 
% nk = n'*k;             % creates a N by N matrix of nk values
% WNnk = WN .^ nk;       % DFT matrix
% Xk = x .* WNnk;        % row vector for DFT coefficients
% X = abs(Xk);


%only works if N = 2^k
N = numel(x);
xp = x(1:2:end);
xpp = x(2:2:end);
if N>=8
    Xp = myFFT(xp);
    Xpp = myFFT(xpp);
    X = zeros(N,1);
    n = length(Xpp);
    
    %exp - original, cannot be converted
    %Wn = exp(-1i*2*pi*((0:N/2-1)')/N);
    
    %MacLaurin Series replacement of exp attemp
    Wn = MaclaurinSer((-1i*2*pi*((0:N/2-1)')/N), n);
    %Wn = Wn.';

    %Custom Fixed point attempt
    %Wn = my_fcn_fixpt(real(-1i*2*pi*((0:N/2-1)')/N));

    %cordic exp attemp
    %Wn = cordiccexp(real(-1i*2*pi*((0:N/2-1)')/N),3);
   
    tmp = Wn .* Xpp;
    X = [(Xp + tmp);(Xp -tmp)];
else
    switch N
        case 2
            X = [1 1;1 -1]*x;
        case 4
            X = [1 0 1 0; 0 1 0 -1i; 1 0 -1 0;0 1 0 1i]*[1 0 1 0;1 0 -1 0;0 1 0 1;0 1 0 -1]*x;
        otherwise
            error('N not correct.');
    end
end


    
%     s = 1;
%     Xex = (-1i*2*pi*((0:N/2-1)')/N);
%     a = 1;
%     y = 1;
%     x1 = 1;
% 
%     for k=1:200
%         x1 = Xex./200;
%         a = a/k;
%         y = y.*Xex;
%         s = s + a*y;
%     end
% 
%     s = s.^200;
%     Wn = s;
 end

function [digit_HVs, pos_HVs] = GetBaseHVs(num_bins)
%% Initial Hypervector Generation
% Gives 2 hypervector sets: the 10 digit vectors and a number of position
%       vectors dependent on the number of audio bins.
%
% Takes num_bins, the number of bins of the audio samples.
%
% Returns digit_HVs, the set of 10 hypervectors corresponding to the
%       numbers 0 through 9.
% Returns pos_HVs, the set of hypervectors corresponding to the
%       positions of the bins of the audio sample.

    D = 2880;
    M = 10;
    
    num_HVs(1,:) = randi([0,1],1,D);
    for d = 2:10
        num_HVs(d,:) = num_HVs(d-1,:);
        rand = randi(D,(D/M),1);
        for i = 1:length(rand)
            num_HVs(d,rand(i)) = ~num_HVs(d,rand(i));
        end
    end
    
    for d = 1:10
        for i = 1:D%length(rand)
            if (num_HVs(d,i) == 0)
                num_HVs(d,i) = -1;
            end
        end
    end
    
    
%     for p = 1:M
%         num_HVs(p,:) = randi([0,1],1,D);
%     end
%     for d = 1:M
%         for i = 1:D
%             if (num_HVs(d,i) == 0)
%                 num_HVs(d,i) = -1;
%             end
%         end
%     end
    ID_HVs = zeros(num_bins,D); 
    for p = 1:num_bins
        ID_HVs(p,:) = randi([0,1],1,D);
    end
    
    digit_HVs = num_HVs;
    pos_HVs = ID_HVs;

end


function [audio_HV] = AudioEncoder(bin_values, digit_HVs, pos_HVs)
%% Audio-Hypervector Encoding
% Encodes the audio sample into one hypervector using the audio bins and
%       digit and bin position hypervectors.
%
% Takes bin_values, the set of decimal numbers representing the average
%       frequency values of one audio sample.
% Takes digit_HVs, the set of 10 hypervectors, each corresponding to a
%       digit (the number 0 through 9).
% Takes pos_HVs, the set of hypervectors corresponding to the positions of
%       the bins of the audio sample.
%
% Returns audio_HV, the hypervector representation of the audio sample
    scaled_bins = zeros(150,1);
    level_HVs = zeros(150,2880);

%     twoNorm = 0;
%     for i = 1:length(bin_values)
%         twoNorm = twoNorm + bin_values(i)^2;
%     end
%     twoNorm = sqrt(twoNorm);
%     bin_values = bin_values/twoNorm;
    %bin_values = bin_values/sum(abs(bin_values));
    bin_values = bin_values/norm(bin_values);
    for i = 1:length(bin_values)
        scaled_bins(i) = round(bin_values(i),6);
        %scaled_bins(i) = round(bin_values(i),5);
    end
    
    for i = 1:length(scaled_bins)
        level_HVs(i,:) = BinEncoder(scaled_bins(i), digit_HVs);
    end
    
    audio_HV = level_HVs(1,:) .* pos_HVs(1,:);
    for i = 2:size(level_HVs,1)
        audio_HV = audio_HV + (level_HVs(i,:) .* pos_HVs(i,:));
    end
end


function [binHV] = BinEncoder(bin_value, digit_HVs)
%% Bin-Hypervector Encoding
% Encodes one bin into one hypervector using the digit hypervectors.
%
% Takes bin_value, the decimal number representing the average
%       frequency value of one part of the audio sample.
% Takes digit_HVs, the set of 10 hypervectors, each corresponding to a
%       digit (the number 0 through 9).
%
% Returns audio_HV, the hypervector representation of that bin.
    numdigits = 5;
    %bin_digi = bin_value * 1000000;
    bin_digi = bin_value * 10^(numdigits-1);
    
    digit_rev = zeros(numdigits,1);
    for i=1:numdigits
        digit_rev(i) = fix(mod(bin_digi,10));
        bin_digi = floor(bin_digi/10);
    end
    
    product = digit_HVs(digit_rev(numdigits)+1,:);
    for i=(numdigits-1):-1:1
        product = product .* circshift(digit_HVs(fix(floor(digit_rev(i)+1)),:),numdigits-i);
    end
   
    binHV = zeros(length(product),1);
    for i = 1:length(product)
        if (product(i) == -1)
            binHV(i) = 0;
        else
            binHV(i) = product(i);
        end
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