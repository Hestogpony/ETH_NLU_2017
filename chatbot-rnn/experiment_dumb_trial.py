import os


print("start experiment")

try:
	os.system('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=1')
	os.system('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=2')
	os.system('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=3')
	os.system('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=4')
	os.system('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=5')
except KeyboardInterrupt:
	sys.exit()
except:
	#report error
	print("There is an error")
