% Load the dataset
clear all
load('dataset2/day_3.mat');
%load('dataset/female_2.mat');

matrix_names_norm = {'spher_ch1', 'spher_ch2', 'tip_ch1', 'tip_ch2', ...
                'palm_ch1', 'palm_ch2', 'lat_ch1', 'lat_ch2', ...
                'cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2'};

for i = 1:length(matrix_names_norm)
    m_norm_name = matrix_names_norm{i};
    m_norm = eval(m_norm_name);
    data_norm = zeros([100 12]);
    for j=1:100
        for k=1:12
        res = PE(m_norm(j,k,:),1,3,512);
        res = res(end);
        data_norm(j,k) = res;
        end
    end
    vmd_data.(m_norm_name) = data_norm;
end
save('dataset2/pe_day_3.mat', '-struct','vmd_data');