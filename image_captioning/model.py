"""
reference: https://www.youtube.com/watch?v=y2BaTt1fxJU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=21
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super().__init__()

        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights='DEFAULT', aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN

    def forward(self, images):
        features = self.inception(images)
        if type(features) != torch.Tensor:
            features = features[0]
        return self.dropout(self.relu(features))



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat([features.unsqueeze(0), embeddings], dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs



class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()

        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens)
                predicted = output.argmax(-1).squeeze()

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0).unsqueeze(0)  # add the batch size 1 and timestep 1 dim

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        
        return [vocabulary.itos[idx] for idx in result_caption]