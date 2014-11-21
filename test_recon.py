def main():
	from face_rec.recognize import FishRecon
	import glob
	from face_rec.utils import make_array

	all_files = glob.glob('fullDataSetB_cropped2/*.png')
	all_files_ID = [int(x.split('\\')[1][:2]) for x in all_files]

	false = []
	fish = FishRecon()
	for a_file,a_ID in zip(all_files,all_files_ID):
	    a_picture = make_array(a_file)
	    predicted_lda,fit = fish.recognize(a_picture)
	    real = int(a_ID)
	    if predicted_lda != real:
	        false.append((predicted_lda,real))

	print false
#         print predicted_lda
#         print trained_names[-1]

if __name__ == '__main__':
	main()

