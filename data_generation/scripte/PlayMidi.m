function [] = PlayMidi(hFigure,currentPoint,Y,midiCell,mStr)
C = get (gca, 'CurrentPoint');
x = C(1,1);
y = C(1,2);
N = numel(midiCell);
[x,y]
min(sqrt(sum((Y-[x,y]).^2,2)))
idx = find(sqrt(sum((Y-[x,y]).^2,2))<0.5);
idx
Fs = 44100;%22050;%8192;44100
synthtype='fm';
global player
if(~isempty(idx))
    if(~isempty(player))
        stop(player);
    end
    vSnd = nmat2snd(midiCell{idx(1)},synthtype,Fs);
    player = audioplayer(vSnd, Fs);
    play(player);
else
    player = [];
end
