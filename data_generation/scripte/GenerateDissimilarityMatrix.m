function [D,midiCell,vMidis,cStr,vStr,L,mStr] = GenerateDissimilarityMatrix
vMidis=dir('../midi/classic/**/*.mid*');
N        = numel(vMidis);
D        = zeros(N);
midiCell = [];
mStr = [];
for(k=1:N)
    [midiCell{k},mStr{k}] = readmidi(vMidis(k).name);
end
% for(k=1:N)
%     midiCell{k} = trim(midiCell{k});
% end
for(k=1:N)
    for(l=1:N)
        D(k,l) = meldistance(midiCell{k},midiCell{l},'pcdist1','taxi'); %durdist1
    end
end
cStr=arrayfun(@(k)vMidis(k).name,1:numel(vMidis),'UniformOutput',false)';
for(k=1:N),cTemp=regexp(cStr{k},'[.]*[-,_,\.]','split');L{k}=cTemp{1};end;
vStr = lower(L);
[~,L] = ismember(vStr',unique(vStr)');
end 

% D = GenerateDissimilarityMatrix()
% csvwrite("midi_d.csv",D)