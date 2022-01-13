import numpy as np
from time import time, sleep
import pickle

text = open('input.txt', 'rb').read()
text = np.frombuffer(text, dtype=np.uint8)
chars = np.unique(text)
data_size, vocab_size = len(text), len(chars)
print("data has {} characters, {} unique.".format(data_size, vocab_size))

hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LSTMCell():
    def __init__(self, input_size, hidden_size, output_size):
        self.i = input_size
        self.h = hidden_size
        self.o = output_size
        self.Whx = np.random.randn(4*hidden_size, input_size) / np.sqrt(4*hidden_size)
        self.Whh = np.random.randn(4*hidden_size, hidden_size) / np.sqrt(4*hidden_size)
        self.Wyh = np.random.randn(output_size, hidden_size) / np.sqrt(output_size)
        self.bh = np.ones((4*hidden_size, 1))*0.1
        self.by = np.ones((output_size, 1))*0.1
        self.mWhx = np.zeros((4*hidden_size, input_size))*0.01
        self.mWhh = np.zeros((4*hidden_size, hidden_size))*0.01
        self.mWyh = np.zeros((output_size, hidden_size))*0.01
        self.mbh = np.zeros((4*hidden_size, 1))
        self.mby = np.zeros((output_size, 1))

    def forward(self, Gv, Hv, Ov, Bv, t):
        h = self.h
        i = self.i
        gates = Gv[:, [t]]
        xv = Ov[:i, [t]]
        hvprev = Hv[:h, [t-1]]
        cvprev = Hv[h:, [t-1]]
        Gv[:, [t]] = self.Whx @ xv + self.Whh @ hvprev
        Gv[:h, [t]] = np.tanh(gates[:h])
        Gv[h:, [t]] = sigmoid(gates[h:])
        Hv[h:, [t]] = gates[2*h:3*h] * cvprev + gates[:h] * gates[h:2*h]
        Bv[:h, [t]] = np.tanh(Hv[h:, [t]])
        Hv[:h,[t]] = Bv[:h, [t]] * gates[3*h:]
        Ov[i:, [t]] = self.Wyh @ Hv[:h,[t]] + self.by

    def backprop(self, t, delta, Gv, Hv, Bv, Ov, dWhh, dWhx, dWyh, dbh, dby, dhnext, dcnext, dGv):
        i = self.i
        h = self.h
        av = Gv[:h, [t]]
        iv = Gv[h:2*h, [t]]
        fv = Gv[2*h:3*h, [t]]
        ov = Gv[3*h:, [t]]
        dWyh += delta @ Hv[:h, [t]].T
        dby += delta
        dh = self.Wyh.T @ delta + dhnext
        dc = dh * ov * (1 - Bv[:h, [t]]**2) + dcnext
        dcnext = dc * fv
        dGv[:h] = dc * iv * (1 - av**2)
        dGv[h:2*h] = dc * av * iv * (1 - iv)
        dGv[2*h:3*h] = dc * Hv[h:, [t-1]] * fv * (1 - fv)
        dGv[3*h:] = dh * Bv[:h, [t]] * ov * (1 - ov)
        dhnext = self.Whh.T @ dGv
        dWhx += dGv @ Ov[:i, [t]].T
        dWhh += dGv @ Hv[:h, [t-1]].T
        dbh += dGv
        return self.Whx.T @ dGv

    def update(self, lr, dWhh, dWhx, dWyh, dbh, dby):
        for dparam in [dWhh, dWhx, dWyh, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

        for param, dparam, m in zip(
                [self.Whh, self.Whx, self.Wyh, self.bh, self.by],
                [dWhh, dWhx, dWyh, dbh, dby],
                [self.mWhh, self.mWhx, self.mWyh, self.mbh, self.mby]):
            m = 0.99 * m - lr * dparam
            param += m




class LSTM_NN():
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.lstms = [  LSTMCell(input_size, hidden_size, hidden_size),
                        LSTMCell(hidden_size, hidden_size, hidden_size),
                        LSTMCell(hidden_size, hidden_size, output_size)]
        self.h =  hidden_size
        self.o =  output_size
        self.i =  input_size

    def train_sequence(self, inputs, targets, Hvprev, lr=0.01):
        # declarationh
        h = self.h
        i = self.i
        o = self.o
        n = inputs.shape[0]
        Hv = np.zeros((3, 2*h, n))
        Gv = np.zeros((3, 4*h, n))
        Ov1, Ov2, Ov3 = np.zeros((i+h, n)), np.zeros((h+h, n)), np.zeros((h+o, n))
        Bv = np.zeros((3, h, n))
        dWhh = np.zeros((3, 4*h, h))
        dWhx1, dWhx2, dWhx3 = np.zeros((4*h, i)), np.zeros((4*h, h)), np.zeros((4*h, h))
        dWyh1, dWyh2, dWyh3 = np.zeros((h, h)), np.zeros((h, h)), np.zeros((o, h))
        dbh = np.zeros((3, 4*h, 1))
        dby1, dby2, dby3 = np.zeros((h, 1)), np.zeros((h, 1)), np.zeros((o, 1))
        probs = np.zeros((o, n))
        dhnext = np.zeros((3, h, 1))
        dcnext = np.zeros((3, h, 1))
        dGv = np.zeros((4*h, 1))

        # initialisation
        Ov1[inputs, range(n)] = 1
        Hv[:, :, [-1]] = Hvprev

        # forward
        loss = 0
        for t in range(n):
            self.lstms[0].forward(Gv[0], Hv[0], Ov1, Bv[0], t)
            Ov2[:h, [t]] = Ov1[i:, [t]]
            self.lstms[1].forward(Gv[1], Hv[1], Ov2, Bv[1], t)
            Ov3[:h, [t]] = Ov2[h:, [t]]
            self.lstms[2].forward(Gv[2], Hv[2], Ov3, Bv[2], t)
            exps = np.exp(Ov3[h:, [t]])
            probs[:, [t]] = exps/np.sum(exps, axis=0)
            loss += -np.log(probs[targets[t], t])

        for t in reversed(range(n)):
            dy = np.copy(probs[:, [t]])
            dy[targets[t]] -= 1
            dy2 = self.lstms[2].backprop(t, dy, Gv[2], Hv[2], Bv[2], Ov3, dWhh[2], dWhx3, dWyh3, dbh[2], dby3, dhnext[2], dcnext[2], dGv)
            dy1 = self.lstms[1].backprop(t, dy2, Gv[1], Hv[1], Bv[1], Ov2, dWhh[1], dWhx2, dWyh2, dbh[1], dby2, dhnext[1], dcnext[1], dGv)
            self.lstms[0].backprop(t, dy1, Gv[0], Hv[0], Bv[0], Ov1, dWhh[0], dWhx1, dWyh1, dbh[0], dby1, dhnext[0], dcnext[0], dGv)


        self.lstms[2].update(lr, dWhh[2], dWhx3, dWyh3, dbh[2], dby3)
        self.lstms[1].update(lr, dWhh[1], dWhx2, dWyh2, dbh[1], dby2)
        self.lstms[0].update(lr, dWhh[0], dWhx1, dWyh1, dbh[0], dby1)
        return loss, Hv[:, :, [n-1]]


    def check_grad(self, c, dWhh, dWhx, dWyh, dbh, dby, inputs, targets, Hvprev):
        num_checks, delta = 50, 1e-5
        for param, dparam, name in zip(
                [self.lstms[c].Whh, self.lstms[c].Whx, self.lstms[c].Wyh, self.lstms[c].bh, self.lstms[c].by],
                [dWhh, dWhx, dWyh, dbh, dby],
                ['Whh', 'Whx', 'Wyh', 'bh', 'by']):
            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1, 'Error dims dont match: {s0} and {s1}.'
            print(f'debug {name} ---------------------')
            for _ in range(num_checks):
                ri = int(np.random.uniform(0, param.size))
                flat_param = param.flat
                old_val = flat_param[ri]
                flat_param[ri] = old_val + delta
                lossright = self.train_sequence(inputs, targets, Hvprev, debug=True)
                flat_param[ri] = old_val - delta
                lossleft = self.train_sequence(inputs, targets, Hvprev, debug=True)
                flat_param[ri] = old_val # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                grad_numerical = (lossright - lossleft) / ( 2 * delta )
                if grad_analytic == 0 and grad_numerical == 0:
                    rel_error = 0
                else:
                    rel_error = abs(grad_analytic - grad_numerical) / (abs(grad_numerical) + abs(grad_analytic))
                print(f'{grad_numerical}, {grad_analytic} => {rel_error} ')

    def train(self, data, epoch=10, lr=0.1):
        n, p = 0, 0
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
        Hvprev = np.ones((3, hidden_size*2, 1)) # reset RNN memory
        nmbr_batch = int(np.ceil(len(data)/seq_length))
        print(f'number of batches: {nmbr_batch}')
        sleep(5)
        mu = 10**(-1/(nmbr_batch*epoch))
        for epoch_idx in range(epoch):
            print(f'epoch {epoch_idx+1}: -------------------------------------')
            for i in range(nmbr_batch):
                lr = max(lr*mu, 1e-5)
                if (p+seq_length+1 >= len(data)):
                    Hvprev = np.zeros((3, hidden_size*2, 1)) # reset RNN memory
                    p = 0 # go from start of data
                inputs = np.array([int(*np.where(chars==ch)[0]) for ch in data[p:p+seq_length]])
                targets = np.array([int(np.where(chars==ch)[0]) for ch in data[p+1:p+seq_length+1]])
                loss, Hvprev = self.train_sequence(inputs, targets, Hvprev, lr)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                if i & 1023 == 0 and i != 0:
                    self.save()
                if i & 63 == 0 and i != 0:
                    print(f'iter {n}, loss: {smooth_loss:9.6f}, lr: {lr:9.8f}')
                p += seq_length # move data pointer
                n += 1 # iteration counter

    def save(self, filename=f'sequence_{int(time())}.bin'):
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)
        print('checkpoint done ...')

def load_lstm(filename):
    with open(filename, 'rb') as fd:
        return pickle.load(fd)


lstm = LSTM_NN(vocab_size, hidden_size, vocab_size)
# lstm = load_lstm('./sequence_1641903932.bin')
lstm.train(text, epoch=1000, lr=0.5)

