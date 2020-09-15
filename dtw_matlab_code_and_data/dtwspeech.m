clear;
clc;
close all;
files = dir('*.wav');
i = 1;
for file = files'
    [data{i},sr] = audioread(file.name);
    data{i} = data{i} / max(data{i});
    names{i} = file.name;
    i = i+1; 
end
for k = 1:size(data,2)-1
[mfca , adelta, adeltadelta] = mfcc(data{k}, sr,'LogEnergy','Ignore');
disp(k);
[mfcb,  bdelta, bdeltadelta] = mfcc(data{k+1}, sr,'LogEnergy','Ignore');
mfcca = [mfca, adelta, adeltadelta]';
mfccb = [mfcb, bdelta, bdeltadelta]';
figure();
subplot(2,1,1);
dtw(data{k}, data{k+1});
[D, path] = dtw(mfcca, mfccb);
distances(k) = D;
subplot(2,1,2);
plot(path);
n = strcat(names{k}(11),{' '},'vs',{' '},names{k+1}(11),{' '}, '=',{' '}, num2str(D));
title(n);
end

    
