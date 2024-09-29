from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor
from torch.utils.data import Dataset
import torch
import string
import re
import pyarabic.araby as araby

tokenizer = AutoTokenizer.from_pretrained('enhancedBERTmodel')

class preprocess_ws_freq(Dataset):
    def __init__(self, text, textb, freq, model_name, max_len=256):
      super(preprocess_ws_freq).__init__()
      """
      Args:
      text (List[str]): List of the training text
      target (List[str]): List of the training labels
      tokenizer_name (str): The tokenizer name (same as model_name).
      max_len (int): Maximum sentence length
      label_map (Dict[str,int]): A dictionary that maps the class labels to integer
      """
      self.text = text
      self.textb = textb
      self.freq = freq
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.max_len = max_len


    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())

      text = araby.strip_diacritics(text)

      # Normalize specific Arabic characters to "ا"
      text = re.sub("[إأآا]", "ا", text)

      textb = str(self.textb[item])

      arabic_prep = ArabertPreprocessor('aubmindlab/bert-base-arabertv02')
      text = arabic_prep.preprocess(text)
      textb = arabic_prep.preprocess(textb)

      w_freq = self.freq[item]

      tokens = []
      label_ids = []
      token_type_ids = []

      features = []
      bert_tokens = []
      bert_tokens.append("[CLS]")

      target_to_tok_map_start = []
      tokens = self.tokenizer.tokenize(text)
      # print(tokens[start_in[0]])
      # print('_________________')

      # bert_tokens.append("[SEP]")
      bert_tokens = self.tokenizer.tokenize(text)

      special_tokens_count = self.tokenizer.num_special_tokens_to_add()

      tokens_b = None
      if textb:
        tokens_b = self.tokenizer.tokenize(textb)
        while (len(bert_tokens) + len(tokens_b)) > self.max_len - special_tokens_count:
          if (len(bert_tokens) + len(tokens_b)):
            break
          tokens_b.pop()

      else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(bert_tokens) > self.max_len - 2:
          bert_tokens = bert_tokens[:(self.max_len - 2)]

      bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
      segment_ids = [0] * len(bert_tokens)

      bert_tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)
      zeros = len(segment_ids)

      bert_tokens += ['a'] + ["[SEP]"]
      segment_ids += [1] * 2

      input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

      input_ids[-2] = int(w_freq)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # The mask has 1 for real target
      target_mask = [0] * zeros
      target_mask += [input_ids[len(target_mask)]]
      target_mask += [3]
      # print(target_mask)

      # Zero-pad up to the sequence length.
      padding = [0] * (self.max_len - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding
      target_mask += padding

      assert len(input_ids) == self.max_len
      assert len(input_mask) == self.max_len
      assert len(segment_ids) == self.max_len
      assert len(target_mask) == self.max_len

      features.append((input_ids,input_mask,
                      segment_ids, target_mask))
      return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([input_mask], dtype=torch.long),
        'token_type_ids': torch.tensor([segment_ids], dtype=torch.long),
        'target_mask': torch.tensor([target_mask], dtype=torch.long)
    }

# Preprocessing function for model2
class preprocess_ws_pos(Dataset):
    def __init__(self, text, textb, POS, start_in, end_in, model_name, max_len=256):
      super(preprocess_ws_pos).__init__()
      """
      Args:
      text (List[str]): List of the training text
      target (List[str]): List of the training labels
      tokenizer_name (str): The tokenizer name (same as model_name).
      max_len (int): Maximum sentence length
      label_map (Dict[str,int]): A dictionary that maps the class labels to integer
      """
      self.text = text
      self.textb = textb
      self.POS = POS
      self.start_in = start_in
      self.end_in = end_in
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.max_len = max_len


    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())

      textb = str(self.textb[item])

      start_in = self.start_in[item]

      end_in = str(self.end_in[item])

      words = text.split()

      arabic_prep = ArabertPreprocessor('aubmindlab/bert-base-arabertv02')

      if start_in  < len(words):
          target_word = words[start_in]
          if target_word[0] != '"' and start_in < len(words)-1:
              target_word = words[start_in + 1]
          elif target_word[0] != '"' and start_in == len(words)-1:
              target_word = words[start_in]
      elif start_in == len(words):
          target_word = words[start_in-1]
          # if target_word[0] != '"':
          #     target_word = words[start_in]
      else:
          target_word = None  # Add None if the target index is out of range

      after_filter = araby.strip_diacritics(target_word)
      targets = re.sub("[إأآا]", "ا", after_filter)

      text = araby.strip_diacritics(text)

      # Normalize specific Arabic characters to "ا"
      text = re.sub("[إأآا]", "ا", text)

      # coding=utf-8
      arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
      english_punctuations = string.punctuation
      punctuations_list = arabic_punctuations + english_punctuations

      translator = str.maketrans('', '', punctuations_list)
      targets = targets.translate(translator)

      preprocessed_targets = arabic_prep.preprocess(targets)

      token = self.tokenizer.tokenize(preprocessed_targets)

      start_index_train = []
      sent_token = self.tokenizer.tokenize(text)
      common_tokens = set(token) & set(sent_token)
      for word in common_tokens:
          start_index_train.extend([j for j, val in enumerate(sent_token) if val == word])

      # print(start_index_train, start_in, common_tokens[start_in])
      # print(start_index_train)
      end_index_train = start_index_train[-1] + 1

      pos = self.POS[item]

      text = arabic_prep.preprocess(text)
      textb = arabic_prep.preprocess(textb)

      tokens = []
      label_ids = []
      token_type_ids = []

      features = []
      bert_tokens = []
      bert_tokens.append("[CLS]")

      target_to_tok_map_start = []
      tokens = self.tokenizer.tokenize(text)

      for length in range(len(tokens)):
        k = tokens[length]
        if length == start_index_train[0]:
          target_to_tok_map_start.append(len(bert_tokens))
          if len(start_index_train) > 1:
            target_to_tok_map_start.append(len(bert_tokens) + 1)
        if length == end_index_train:
          target_to_tok_map_end = len(bert_tokens)
          break
        bert_tokens.extend([k])

      if end_in == len(tokens):
        target_to_tok_map_end = len(bert_tokens)

      # bert_tokens.append("[SEP]")
      bert_tokens = self.tokenizer.tokenize(text)

      special_tokens_count = self.tokenizer.num_special_tokens_to_add()

      tokens_b = None
      if textb:
        tokens_b = self.tokenizer.tokenize(textb)
        while (len(bert_tokens) + len(tokens_b)) > self.max_len - special_tokens_count:
          if (len(bert_tokens) + len(tokens_b)):
            break
          tokens_b.pop()

      else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(bert_tokens) > self.max_len - 2:
          bert_tokens = bert_tokens[:(self.max_len - 2)]

      bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
      segment_ids = [0] * len(bert_tokens)

      bert_tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

      input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (self.max_len - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == self.max_len
      assert len(input_mask) == self.max_len
      assert len(segment_ids) == self.max_len
      # print(target_to_tok_map_start)

      if len(target_to_tok_map_start)>1:
          target_to_tok_map_end = target_to_tok_map_start[-1] + 1
      else:
         target_to_tok_map_end = target_to_tok_map_start[0] + 1 

      # The mask has 1 for real target
      target_mask = [0] * self.max_len
      for i in range(target_to_tok_map_start[0], target_to_tok_map_end):
        # print(pos)
        target_mask[i] = int(pos)

      features.append((input_ids, input_mask,
                      segment_ids, target_mask))
      return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([input_mask], dtype=torch.long),
        'token_type_ids': torch.tensor([segment_ids], dtype=torch.long),
        'target_mask': torch.tensor([target_mask], dtype=torch.long)
    }

# Preprocessing function for model3
class preprocess_ws(Dataset):
    def __init__(self, text, textb, model_name, max_len=256):
      super(preprocess_ws).__init__()
      """
      Args:
      text (List[str]): List of the training text
      target (List[str]): List of the training labels
      tokenizer_name (str): The tokenizer name (same as model_name).
      max_len (int): Maximum sentence length
      label_map (Dict[str,int]): A dictionary that maps the class labels to integer
      """
      self.text = text
      self.textb = textb
      self.tokenizer_name = model_name
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.max_len = max_len


    def __len__(self):
      return len(self.text)

    def __getitem__(self,item):
      text = str(self.text[item])
      text = " ".join(text.split())

      textb = str(self.textb[item])

      text = araby.strip_diacritics(text)

      # Normalize specific Arabic characters to "ا"
      text = re.sub("[إأآا]", "ا", text)

      arabic_prep = ArabertPreprocessor('aubmindlab/bert-base-arabertv02')
      text = arabic_prep.preprocess(text)
      textb = arabic_prep.preprocess(textb)

      tokens = []
      label_ids = []
      token_type_ids = []


      features = []
      tokens_a = self.tokenizer.tokenize(text)
      tokens_b = self.tokenizer.tokenize(textb)

      special_tokens_count = self.tokenizer.num_special_tokens_to_add()

      while (len(tokens_a) + len(tokens_b)) > self.max_len - special_tokens_count:
          if (len(tokens_a) + len(tokens_b)):
            break
          tokens_b.pop()

      tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
      segment_ids = [0] * len(tokens)
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

      input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding = [0] * (self.max_len - len(input_ids))
      input_ids += padding
      input_mask += padding
      segment_ids += padding

      assert len(input_ids) == self.max_len
      assert len(input_mask) == self.max_len
      assert len(segment_ids) == self.max_len

      features.append((input_ids, input_mask, segment_ids))


      return {
        'input_ids': torch.tensor([input_ids], dtype=torch.long),
        'attention_mask': torch.tensor([input_mask], dtype=torch.long),
        'token_type_ids': torch.tensor([segment_ids], dtype=torch.long)
    }