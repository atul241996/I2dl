import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################

        self.hidden_size = hidden_size
        self.i_layer = nn.Linear(input_size, hidden_size)
        self.h_layer = nn.Linear(hidden_size, hidden_size)
        if activation == 'tanh':
            self.act_func = nn.Tanh()
        else:
            self.act_func = nn.ReLU()

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        seq_len, batch_size, _ = x.shape
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            h = self.act_func(self.h_layer(h) + self.i_layer(x[i]))
            h_seq.append(h)
        h_seq = torch.stack(h_seq, dim=0)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################

        self.hidden_size = hidden_size
        self.i_gates = nn.Linear(input_size, 4*hidden_size)
        self.h_gates = nn.Linear(hidden_size, 4*hidden_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []


        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        seq_len, batch_size, _ = x.shape
        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size)
        if c is None:
            c = torch.zeros(1, batch_size, self.hidden_size)
        assert h.shape[0] == 1 and c.shape[0] == 1, 'this lstm module only works with 1 layer'

        h, c = h[0], c[0]
        x_scores = self.i_gates(x)
        for i in range(seq_len):
            gates = self.h_gates(h) + x_scores[i]
            forgetgate, ingate, cellgate, outgate = gates.chunk(chunks=4, dim=-1)
            forgetgate = torch.sigmoid(forgetgate)
            ingate = torch.sigmoid(ingate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)
            c = forgetgate * c + ingate * cellgate
            h = outgate * torch.tanh(c)
            h_seq.append(h)
        h_seq = torch.stack(h_seq, dim=0)
        h.unsqueeze(dim=0)
        c.unsqueeze(dim=0)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq , (h, c)

