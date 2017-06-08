import os
import sys

print("start experiment")

try:
    #Relevance
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --relevance=-1')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --relevance=0.1')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --relevance=0.2')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --relevance=0.3')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --relevance=0.4')

    #Beam width
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=1')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=2')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=3')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=4')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --beam_width=5')

    # Temperature
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --temperature=0.6')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --temperature=0.8')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --temperature=1.0')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --temperature=1.2')
    os.system('python chatbot.py --test=data/Validation_Shuffled_Dataset.txt  --save_dir=models/combined_model --temperature=1.4')

except KeyboardInterrupt:
    sys.exit()
