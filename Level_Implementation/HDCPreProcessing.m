%%Sean Lane
%Hyperdimensional Computing Speech Recognition
%Thesis Research
%Villanova University
%Fall 2021 - Spring 2022

bins = 75;

%use if need to generate HVs
%[digitHV, posHV] = GetBaseHVs(bins);

% Set global variables

classes = [{'down'},{'eight'},{'five'},{'four'}];
% classes = [{'bed'},{'bird'},{'cat'},{'dog'},{'down'},{'eight'}...
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
bins_sets = zeros(100,bins,num_classes);


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
            bins_sets(:,:,i) = GetFreqBins(audio_sets(:,:,i), bins);
        end
        


%% Save Features to savDir directory 
savdir = 'C:\Users\seanl\Documents\Research\';



%bins_sets = rescale(bins_sets,0,116);
bins_sets = round(bins_sets);
save(fullfile(savdir, 'bins_sets'), 'bins_sets');
        
        
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
    %X = zeros(num_bins,num_takes);
    for i=1:num_takes
        X(:,i) = fft(audioData(:,i));
        %X(:,i) = fft(audioData(:,i),num_bins);
    end
     
    Z = zeros(16000,num_takes);
    %Z = zeros(num_bins,num_takes);
    for s = 1:num_takes
        Z(:,s) = abs(X(1:16000,s));
        %Z(:,s) = abs(X(1:num_bins,s));
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

 average = average * 10;

%     increaserate = 1.13;
% 
%     %tiers(1,1) = 0.001;
%     tiers(1,1) = 0.01;
%     for s = 2:num_bins
%         tiers(s,1) = tiers(s-1,1) * increaserate;
%     end
% 
%      for s = 1:num_takes
%         for i = 1:num_bins
%             k = 1;
%             while((average(s,i) > tiers(k,1)) && (k < num_bins))
%                 k = k + 1;
%             end
%             if(average(s,i) < tiers(k,1))
%                 average(s,i) = tiers(k,1);
%             else 
%                 average(s,i) = tiers(num_bins,1);
%             end
%         end
%     end

%      for s = 1:num_takes
%         for i = 1:num_bins
%             k = 1;
%             while((Z(i,s) > tiers(k,1)) && (k < num_bins))
%                 k = k + 1;
%             end
%             if(Z(i,s) < tiers(k,1))
%                 Z(i,s) = k;
%             else 
%                 Z(i,s) = num_bins;
%             end
%         end
%     end
       
     bin_values = average;
      %Z=Z';
      %bin_values = Z;
% 
%     for s = 1:num_takes
%         for i = 1:num_bins
%             if (Z(i,s) < 0.025)
%                 bin_values(s,i) = 1;
%             elseif (Z(i,s) < 0.05)
%                 bin_values(s,i) = 2;
%             elseif (Z(i,s) < 0.075)
%                 bin_values(s,i) = 3;
%             elseif (Z(i,s) < 0.1)
%                 bin_values(s,i) = 4;
%             elseif (Z(i,s) < 0.125)
%                 bin_values(s,i) = 5;
%             elseif (Z(i,s) < 0.15)
%                 bin_values(s,i) = 6;
%             elseif (Z(i,s) < 0.175)
%                 bin_values(s,i) = 7;
%             elseif (Z(i,s) < 0.2)
%                 bin_values(s,i) = 8;
%             elseif (Z(i,s) < 0.25)
%                 bin_values(s,i) = 9;
%             elseif (Z(i,s) < 0.3)
%                 bin_values(s,i) = 10;
%             elseif (Z(i,s) < 0.35)
%                 bin_values(s,i) = 11;
%             elseif (Z(i,s) < 0.4)
%                 bin_values(s,i) = 12;
%             elseif (Z(i,s) < 0.5)
%                 bin_values(s,i) = 13;
%             elseif (Z(i,s) < 0.6)
%                 bin_values(s,i) = 14;
%             elseif (Z(i,s) < 0.7)
%                 bin_values(s,i) = 15;
%             elseif (Z(i,s) < 0.8)
%                 bin_values(s,i) = 16;
%             elseif (Z(i,s) < 0.9)
%                 bin_values(s,i) = 17;
%             elseif (Z(i,s) < 1)
%                 bin_values(s,i) = 18;
%            elseif (Z(i,s) < 1.25)
%                 bin_values(s,i) = 19;
%            elseif (Z(i,s) < 1.5)
%                 bin_values(s,i) = 20;
%              elseif (Z(i,s) < 2)
%                 bin_values(s,i) = 21;
%              elseif (Z(i,s) < 2.5)
%                 bin_values(s,i) = 22;
%              elseif (Z(i,s) < 3)
%                 bin_values(s,i) = 23;
%             elseif (Z(i,s) < 5)
%                 bin_values(s,i) = 24;
%             else
%                 bin_values(s,i) = 25;
%             end
%         end
%    end
end
