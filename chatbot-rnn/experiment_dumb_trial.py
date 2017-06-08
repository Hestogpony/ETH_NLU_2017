from bash import bash


print("start experiment")
bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=1')
bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=2')
bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=3')
bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=4')
bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=5')



#bash('python chatbot.py --test=data/our_data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=1')