% Load the dataset
clear all
load('dataset/female_1_norm.mat');
load('dataset/female_1.mat');

% VMD parameters
alpha = 4000;        % Moderate bandwidth constraint
tau = 0;             % Noise-tolerance (no strict fidelity enforcement)
K = 36;              % Number of modes
DC = 0;              % No DC part imposed
init = 1;            % Initialize omegas uniformly
tol = 1e-7;          % Tolerance for convergence

% Apply VMD to non-normalized signal (cyl_ch1)
[u, u_hat, omega] = VMD(cyl_ch1(10,:), alpha, tau, K, DC, init, tol);

% Apply VMD to normalized signal (cyl_ch1_norm)
[u_norm, u_hat_norm, omega_norm] = VMD(cyl_ch1_norm(10,:), alpha, tau, K, DC, init, tol);

% Time vector (assuming you have it defined already)
sampling_rate = 500;
N = size(cyl_ch1, 2);  % Adjust as needed based on the signal length
f = (0:N-1)*(sampling_rate/N);     % Frequency axis (0 to Fs)
time = 0:6/(length(cyl_ch1)-1):6;

% Create figure with 24 subplots (3 rows, 8 columns)
%figure;
% Plot the 12 modes of the non-normalized signal
%for i = 1:K
%    subplot(3,8,floor((i-1)/4)*4+i);  % First 12 subplots (non-normalized)
%    u_freq = fft(u(i,:));
%    plot(f, abs(u_freq),'Color',[1 0 0]);
%    title(['Mode ', num2str(i)]);
%    xlabel('Time (seconds)');
%    ylabel(['u_{', num2str(i), '}']);
%end
% Plot the 12 modes of the normalized signal
%for i = 1:K
%    subplot(3,8,floor((i-1)/4)*4+i+4); 
%    u_freq = fft(u_norm(i,:));
%    plot(f,abs(u_freq),"Color",[0 0 0]);
%    title(['Mode ', num2str(i)]);
%    xlabel('Time (seconds)');
%
% end
%
%sgtitle('Cylinder grasp - Flexor Capri Ulnaris - Female 1 ')

reconstructed_signal = sum(u_norm,1);
figure;
subplot(1,1,1);
plot(time,cyl_ch1_norm(10,:)-reconstructed_signal,"Color",[0 0 0]);
ylim([-0.5 0.5]);
sgtitle('Difference between real and reconstructed, K = 36')
