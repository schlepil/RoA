function runSwingUp()
%% runs trajectory optimization and animates open-loop playback

p = AcrobotPlant;
v = AcrobotVisualizer(p);
[utraj,xtraj] = swingUpTrajectory(p);
%      sys = cascade(utraj,p);
%      xtraj=simulate(sys,utraj.tspan,zeros(4,1));

Teval = 0:0.001:6;
X = ppval(xtraj.pp,Teval);
U = ppval(utraj.pp,Teval);

dlmwrite('/home/elfuius/ownCloud/thesis/RoA/input/acroT.txt', Teval,',');
dlmwrite('/home/elfuius/ownCloud/thesis/RoA/input/acroX.txt', X,',');
dlmwrite('/home/elfuius/ownCloud/thesis/RoA/input/acroU.txt', U,',');

v.playback(xtraj);

end
