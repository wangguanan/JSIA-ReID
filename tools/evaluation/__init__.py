from .classification import accuracy
from .sysu_mm01_python import evaluate_sysymm01

if __name__ == '__main__':

	matlab = '/mnt/xfs1/home/IVA/Software/MATLAB/R2014b/bin/matlab'
	root_path = 'sysu_mm01_matlab/'
	mode = 'all_search'
	number_shot = 1

	python_api.run_matlab_evaluate_file(matlab, root_path, mode, number_shot)
	python_api.read_m_file_results('sysu_mm01_matlab/2/result_feature_euclidean_all_search_1shot.mat')
