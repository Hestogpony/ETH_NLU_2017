import re


def read_data(path, output_path):

    with open(output_path, 'a', encoding="ascii") as output_f:
        with open(path, 'r') as input_f:
            print("reading data from %s..." % path)
            input_data_lines = input_f.readlines()

            for data_line in input_data_lines:
                input_sentences = data_line.split('\t')
                assert (len(input_sentences) == 3)

                first_sentence = re.sub(r'\s([?.,!%\/\\[\]$@])', r'\1', ("> " + input_sentences[0]))
                second_sentence = re.sub(r'\s([?.,!%\/\\[\]$@])', r'\1', ("> " + input_sentences[1]))
                third_sentence = re.sub(r'\s([?.,!%\/\\[\]$@])', r'\1', ("> " + input_sentences[2]))

                first_sentence = re.sub(r'\s([\'{``}])\s', r'\1', first_sentence)
                second_sentence = re.sub(r'\s([\'{``}])\s', r'\1', second_sentence)
                third_sentence = re.sub(r'\s([\'{``}])\s', r'\1', third_sentence)

                first_sentence = re.sub(r'[^\x00-\x7F]', ' ', first_sentence)
                second_sentence = re.sub(r'[^\x00-\x7F]', ' ', second_sentence)
                third_sentence = re.sub(r'[^\x00-\x7F]', ' ', third_sentence)

                output_f.write(first_sentence + "\n")
                output_f.write(second_sentence + "\n")
                output_f.write(third_sentence)

        input_f.close()
        output_f.close()

read_data("Validation_Shuffled_Dataset.txt", "Validation_Shuffled_Dataset_ascii.txt")