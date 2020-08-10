import torch.nn as nn

class genQLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes,input_dropout_p=0,variable_lengths=False):
        super(genQLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dropout_p = input_dropout_p
        self.variable_lengths = variable_lengths
        self.input_dropout = nn.Dropout(p=input_dropout_p)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2classes = nn.Linear(hidden_dim, num_classes)

    def forward(self, sentence, input_lengths=None):

        embeds = self.word_embeddings(sentence)
        embeds = self.input_dropout(embeds)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                    embeds, input_lengths, batch_first=True)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        classes_space = self.hidden2classes(lstm_out.view(len(sentence), -1))
        classes_scores = F.log_softmax(classes_space, dim=1)
        return classes_scores

