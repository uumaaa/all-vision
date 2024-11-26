% Load the dataset
clear all
load('dataset2/male_day_3.mat');

% VMD parameters
alpha = 4000;        % Moderate bandwidth constraint
tau = 0;             % Noise-tolerance (no strict fidelity enforcement)
K = 12;              % Number of modes
DC = 0;              % No DC part imposed
init = 1;            % Initialize omegas uniformly
tol = 1e-7;          % Tolerance for convergence


matrix_names_norm = {'spher_ch1', 'spher_ch2', 'tip_ch1', 'tip_ch2', ...
                'palm_ch1', 'palm_ch2', 'lat_ch1', 'lat_ch2', ...
                'cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2'};

for i = 1:length(matrix_names_norm)
    m_norm_name = matrix_names_norm{i};
    m_norm = eval(m_norm_name);
    data_norm = zeros([100 K 2500]);
    for j=1:100
        [u_norm, u_hat_norm, omega_norm] = VMD(m_norm(j,:), alpha, tau, K, DC, init, tol);
        data_norm(j,:,:) = u_norm;
    end
    vmd_data.(m_norm_name) = data_norm;
end
save('dataset2/day_3.mat', '-struct','vmd_data');


