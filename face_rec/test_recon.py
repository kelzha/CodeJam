from recognize import FishRecon,EigenfaceRecon,EigenRecon
from utils import make_array, get_IDs
import glob
import timeit
import itertools

def fish_main():
	all_files = glob.glob('training_dataset_cropped2/*.png')
	# all_files_ID = [int(x.split('\\')[1][:2]) for x in all_files]
	fish = FishRecon('bin/fishfaces/')
	false = []
	for a_file in all_files:
		a_picture = make_array(a_file)
		predicted_lda,fit = fish.recognize(a_picture)
		print (fit,a_file)
	    # real = int(a_ID)
	    # print real
	#     predicted_pca = recon.recognize(a_picture)
	    # if predicted_lda != real:
	    #     false.append((predicted_lda,real))
	#         print predicted_lda
	#         print trained_names[-1]
	# print false
	# return len(all_files)

def eigen_main():
	all_files = glob.glob('fullDataSetB_cropped2/*.png')
	all_files_ID = get_IDs(all_files)
	recon = EigenfaceRecon('bin/eigenfaces/')
	false = []
	for a_file,a_ID in itertools.izip(all_files,all_files_ID):
		a_picture = make_array(a_file)
		predicted,fit = recon.recognize(a_picture)
		real = int(a_ID)
		if predicted != real:
			false.append((predicted,real))

	return false

def new_eigen_main():
	all_files = glob.glob('fullDataSetB_cropped2/*.png')
	all_files_ID = get_IDs(all_files)
	# recon = EigenRecon('bin/new_eigenfaces/')
	recon = EigenRecon('bin/new_eigenfaces_25/')
	false = []
	for a_file,a_ID in itertools.izip(all_files[:1],all_files_ID[:1]):
		a_picture = make_array(a_file)
		predicted,fit = recon.recognize(a_picture)
		real = int(a_ID)
		if predicted != real:
			false.append((predicted,real))

	return false

if __name__ == '__main__':
	ret = new_eigen_main()
	print ret
	# print 50*120,"pictures"
	# print timeit.timeit(main, number=50)
        

