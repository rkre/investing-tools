import scipy.io

file_path = f'C:\\Users\\Kabajisan\Documents\\NSF Training\\mg_samples.mat'
mat = scipy.io.loadmat(file_path)

print(mat)