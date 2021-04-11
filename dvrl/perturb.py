import random
import argparse
import csv


def file_len(fname):
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
    return line_count - 1 #ignoring the header.

def filter(input_file, output_file, noise_file, noise_ratio):
	
    #extract vocabulary.
    vocab = set()
    with open(input_file, mode="r") as ip_csv:
        csv_reader = csv.reader(ip_csv, delimiter=',')
        header = True
        for row in csv_reader:
            if header:
                header = False
            else:
                text = row[4]
                words = text.split()
                for word in words: 
                    vocab.add(word)

    vocab=list(vocab)
    
    num_utterances = file_len(input_file)
    clean_utterance_count = int((1-noise_ratio) * num_utterances)
    clean_utterances = random.sample(range(num_utterances), clean_utterance_count)
	
    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]
    
    line_count = 0
    noise_ids = []
    with open(input_file, mode="r") as ip_csv:
        csv_reader = csv.reader(ip_csv, delimiter=',')
        header = True
        for row in csv_reader:
            if header:
                header = False
            else:
                original_text = row[4]

                if line_count in clean_utterances:
                    text = original_text
                else:
                    noise_ids.append(line_count)
                    #transform the utterance and add noise.
                    new_words = []
                    words = original_text.strip().split()
                    for word in words:
                        #0.5 likelihood we do not change a word
                        #0.16 likelihood we insert new word or delete word or substitute a word
                        how_to_change = random.sample([0,1,2,3,4,5], 1)[0]
                        if how_to_change == 0:
                            #insertion of a new word
                            select_word = random.choice(vocab)
                            new_words.append(select_word)
                            new_words.append(word)
                        elif how_to_change==1:
                            #subsitution of the existing word
                            select_word=random.choice(vocab)
                            while select_word == word:
                                select_word = random.choice(vocab)
                            new_words.append(select_word)
                        elif how_to_change == 2:
                            #deletion of the word.
                            None
                        else:
                            #keep the word as is.
                            new_words.append(word)
                    
                    #additional check to see if perturbation yielded empty sentence.
                    if len(new_words) == 0:
                        text = original_text
                    
                    text = " ".join(new_words)
                
                csv_line = [row[0], row[1], row[2], row[3], text]
                #  Appending current file to the csv_lines list
                csv_lines.append(csv_line)
                line_count += 1

    # Writing the csv_lines
    with open(output_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    #Writing the noise_ids
    with open(noise_file, mode="w") as noise_f:
        for noise_id in noise_ids:
            noise_f.write(str(noise_id) + "\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='perturbation of csv files')
    parser.add_argument('--input_file', type=str,help='input csv file', required=True)
    parser.add_argument('--output_file', type=str,help='output csv file', required=True)
    parser.add_argument('--noise_file', type=str,help='output noise idx', required=True)
    parser.add_argument('--noise', type=float, help='noise ratio [0.1, 0.2, 0.3, 0.4, 0.5]', required=True)
    
    args = parser.parse_args()
    
    filter(args.input_file, args.output_file, args.noise_file, args.noise)
