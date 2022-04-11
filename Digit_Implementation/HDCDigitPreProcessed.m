%%Sean Lane
%Hyperdimensional Computing Speech Recognition
%Thesis Research
%Villanova University
%Fall 2021 - Spring 2022

bins = 150;

%use if need to generate HVs
%[digitHV, posHV] = GetBaseHVs(bins);

% Set global variables

classes = [{'down'},{'eight'},{'five'},{'four'}];
%classes = [{'bed'},{'bird'},{'cat'},{'dog'},{'down'},{'eight'}...
%     {'five'},{'four'},{'go'},{'happy'},{'house'},{'left'},{'marvel'}...
%     {'nine'},{'no'},{'off'},{'on'},{'one'},{'right'},{'seven'}...
%     {'sheila'},{'six'},{'stop'},{'three'},{'tree'},{'two'},{'up'}...
%     {'wow'},{'yes'},{'zero'}];                         %Different classes
num_classes = length(classes);


% for loading files into a 3-D matrix
% matrix declaration
maxFileLength = 16384; %the files of this dataset are 16000, use 16384 for mult of 2 (fft)
audio_sets = zeros(16384,100,length(classes));
totalCounts = zeros(length(classes),1);
sampleCounts = zeros(length(classes),1);
digitbins_sets = zeros(100,bins,num_classes);


%loading each .wav file, converting to .mat, storing in mx, running 
MyDir = 'C:\Users\seanl\Documents\Research\SingleWordsSmall\';
    for i = 1:length(classes)
        name = strcat(MyDir,char(classes(i)), '\');
        files1 = dir(name);
        for k = 3:length(files1)                         % looping through records of same motions
            path_name = strcat(name,files1(k).name); 
                totalCounts(i) = totalCounts(i) + 1;
                [sig,fs] = audioread(path_name);
                if(length(sig) < maxFileLength)
                    sig(end:maxFileLength)=0;
                end
                sampleCounts(i) = sampleCounts(i) + 1;
                audio_sets(:,sampleCounts(i),i) = sig;
        end
    end
        
        for i = 1:length(classes)
            digitbins_sets(:,:,i) = GetFreqBins(audio_sets(:,:,i), bins);
        end
        


%% Save Features to savDir directory 
savdir = 'C:\Users\seanl\Documents\Research\';



%digitbins_sets = round(digitbins_sets);
save(fullfile(savdir, 'digitbins_sets'), 'digitbins_sets');
        
        
function [bin_values] = GetFreqBins(audioData, num_bins)
%% Frequency Bin Generation
% Turns audio samples into sets of frequency bins.
%
% Takes audioData, the audio samples.
% Takes num_bins, the number of bins of the audio samples.
%
% Returns bin_values, the set of bin values split into sections.

    num_takes = size(audioData,2);

    X = zeros(16384,num_takes);
    for i=1:num_takes
        X(:,i) = fft(audioData(:,i));
    end
     
    Z = zeros(16000,num_takes);
    for s = 1:num_takes
        Z(:,s) = abs(X(1:16000,s));
    end
    
    total = length(1:length(Z(:,1)));
    
    binsize = total/num_bins;
    
    average = zeros(num_takes,num_bins);
    tiers = zeros(num_bins,1);


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