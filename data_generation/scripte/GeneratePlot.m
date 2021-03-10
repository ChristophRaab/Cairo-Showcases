function [h,hf] = GeneratePlot(D,vStr,midiCell,mStr)
close all
hf=figure;
K=DoubleCentering(D.^2);
[e,v]  = eig(K);
bar(diag(v))
Y=tsne(K);
h = gscatter(Y(:,1),Y(:,2),vStr',[],[],50);%,'rollover');
set (hf, 'WindowButtonMotionFcn', @(hF,cP)PlayMidi(hF,cP,Y,midiCell,mStr));
%pointerBehavior.enterFcn = [];
%@(hfig, cpp)set(findobj(hfig, 'rollover'),'color', 'red');
%pointerBehavior.traverseFcn = @(hF,cP)PlayMidi(hF,cP,Y,midiCell);
pointerBehavior.exitFcn = @StopMidi;
%@(hfig, cpp)set(findobj(hfig, 'tag', 'rollover'),'color', 'black');
%iptSetPointerBehavior(h, pointerBehavior);
iptPointerManager(hf, 'enable');