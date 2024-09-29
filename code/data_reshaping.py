import pandas as pd
import re
import string
from camel_tools.disambig.mle import MLEDisambiguator

class DataReshaping:
    def __init__(self, data_path, pos_path, freq_path, index_path):
        # Load the datasets
        self.data = pd.read_excel(data_path)
        self.pos = pd.read_excel(pos_path)
        self.freq = pd.read_excel(freq_path)
        self.index = pd.read_excel(index_path)
        
        # Combine data
        self._combine_data()
        
        # Create the senses dictionary
        self.senses_dictionary = self.data.groupby('Word')['Meaning'].unique().apply(list).to_dict()
        
        # Initialize the disambiguator
        self.mle = MLEDisambiguator.pretrained()

    def _combine_data(self):
        self.data['POS'] = self.pos['POS']
        self.data['Term Frequency'] = self.freq['Frequency']
        self.data['Start Index'] = self.index['Target_index_Start']
        self.data['End Index'] = self.index['Target_index_End']

    @staticmethod
    def remove_punctuation_arabic(sentence):
        # Define the punctuation characters to replace
        punctuation_to_replace = string.punctuation.replace('.', '').replace(':', '').replace(',', '').replace('،', '')
        # Create a translation table to replace punctuations with spaces
        translator = str.maketrans(punctuation_to_replace, ' ' * len(punctuation_to_replace))
        # Apply the translation table
        sentence = sentence.translate(translator)
        
        sentence = re.sub(r'\s\.\.', '.', sentence)
        
        punctuation = string.punctuation + "ـ"
        sentence = ''.join([char if char not in punctuation or prev_char in punctuation else '.' if char == "ـ" else ' ' + char 
                            for prev_char, char in zip(' ' + sentence, sentence)])
        
        # Define the punctuation characters to remove
        punctuation_to_remove = ['.', ':', ',', '،']
        # Remove spaces for the specified punctuation characters
        for char in punctuation_to_remove:
            sentence = sentence.replace(' ' + char, char)
        
        # Preserve the structure of numbers separated by '.', ':', ',', and '،'
        sent = re.sub(r'(\d)([.:،,])(\d)', r'\1\2\3', sentence)
        # Clear space between punctuation and words
        final_sent = re.sub(r'\s([\?!":,،؛](?:\s|$))', r'\1', sent)
        cleaned_sentence = re.sub(r'\s\.', '.', final_sent)

        return cleaned_sentence

    def process_data(self):
        # Initialize lists for the transformed data
        target_words = []
        sentences = []
        labels = []
        gloss = []
        target_ids = []
        part_of_speechs = []
        term_freqs = []
        start_indexs = []
        end_indexs = []
        
        # Iterate over each row in the DataFrame
        for index, row in self.data.iterrows():
            target_word = row['Word']
            sense = row['Meaning']
            sentence = row['Sentence']
            target_id = row['TargetID']
            part_of_speech = row['POS']
            term_freq = row['Term Frequency']
            start_index = row['Start Index']
            end_index = row['End Index']

            sentence = self.remove_punctuation_arabic(sentence)
            
            # Get the senses for the target word from the dictionary
            senses = self.senses_dictionary.get(target_word, [])
            
            # Assign a label of 1 if the sense matches, otherwise 0
            label = 1 if sense in senses else 0
            
            # Repeat the target word, sentence, and label for each sense
            for sen in senses:
                target_ids.append(target_id)
                target_words.append(target_word)
                sentences.append(sentence)
                gloss.append(sen)
                part_of_speechs.append(part_of_speech)
                term_freqs.append(term_freq)
                start_indexs.append(start_index)
                end_indexs.append(end_index)

                labels.append(1 if sense == sen else 0)

        # Create the new DataFrame
        self.token_pos_freq = pd.DataFrame({
            'Target_ID': target_ids, 
            'Target_Word': target_words, 
            'Label': labels, 
            'Sentence': sentences, 
            'Gloss': gloss,
            'POS': part_of_speechs,
            'Freq': term_freqs,
            'start_index': start_indexs,
            'end_index': end_indexs
        })

    def modify_ws_sentences(self, ws_pos_freq):
        modified_sentences = []
        for i, (text, int_index) in enumerate(zip(ws_pos_freq['Sentence'], ws_pos_freq['start_index'])):
            text = self.remove_punctuation_arabic(text)
            words = text.split()
            
            # Check if the target index is within the bounds of the words list
            if 0 <= int(int_index) < len(words):
                # Add quotation marks around the target word at the specified index
                words[int(int_index)] = f'"{words[int(int_index)]}"'
                
            modified_sentences.append(' '.join(words))

        # Update the DataFrame with modified sentences
        ws_pos_freq['Sentence'] = modified_sentences
        return ws_pos_freq

    def create_ws_pos_freq(self):
        ws_pos_freq = self.token_pos_freq.copy()
        ws_pos_freq["Gloss_Pair"] = ws_pos_freq.apply(lambda row: f"{row['Target_Word']}: {row['Gloss']}", axis=1)
        ws_pos_freq.drop(columns=['Gloss'], axis=1, inplace=True)
        return ws_pos_freq

    def get_result(self):
        # Process the data to create token_pos_freq
        self.process_data()
        # Create ws_pos_freq from token_pos_freq
        ws_pos_freq = self.create_ws_pos_freq()
        # Modify the sentences only in ws_pos_freq
        ws_pos_freq = self.modify_ws_sentences(ws_pos_freq)
        # Return both token_pos_freq and ws_pos_freq
        return self.token_pos_freq, ws_pos_freq