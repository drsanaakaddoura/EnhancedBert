import torch.nn as nn


class EnsembleBERT(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5, weights=[0.25, 0.125, 0.15, 0.3, 0.4]):
        super(EnsembleBERT, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.weights = weights

    def forward(self, input1, input2, input3, input4, input5):
        logits1 = self.model1(input1['input_ids'], attention_mask=input1['attention_mask'], token_type_ids=input1['token_type_ids'], position_ids=input1['target_mask']).logits
        logits2 = self.model2(input2['input_ids'], attention_mask=input2['attention_mask'], token_type_ids=input1['token_type_ids'], position_ids=input2['target_mask']).logits
        logits3 = self.model3(input3['input_ids'], attention_mask=input3['attention_mask'], token_type_ids=input1['token_type_ids'], position_ids=input3['target_mask']).logits
        logits4 = self.model4(input4['input_ids'], attention_mask=input4['attention_mask'], token_type_ids=input4['token_type_ids']).logits
        logits5 = self.model5(input5['input_ids'], attention_mask=input5['attention_mask'], token_type_ids=input5['token_type_ids'], position_ids=input2['target_mask']).logits

        logits = self.weights[0] * logits1 + self.weights[1] * logits2 + self.weights[2] * logits3 + self.weights[3] * logits4 + self.weights[4] * logits5
        return logits