function [] = StopMidi
    global player
    if(~isempty(player))
        stop(player);
    end