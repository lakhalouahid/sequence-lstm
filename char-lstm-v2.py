import cupy as np
import pickle

data = open('input', 'rb').read()
data = np.frombuffer(data, dtype=np.uint8)
chars = np.unique(data)
data_size, vocab_size = len(data), len(chars)
print("data has {} characters, {} unique.".format(data_size, vocab_size))

hidden_size = 1000
seq_length = 20
learning_rate = 1e-2

class LSTMCell():
    def __init__(self, in_size, hi_size, ou_size):
        self.i = in_size
        self.h = hi_size
        self.o = ou_size
        self.Whx = np.random.randn(4*self.h, self.i) / np.sqrt(self.i)
        self.Whh = np.random.randn(4*self.h, self.h) / np.sqrt(self.h)
        self.Wyh = np.random.randn(self.o, self.h) / np.sqrt(self.h)
        self.bh = np.zeros((4*self.h, 1))
        self.by = np.zeros((self.o, 1))

    def forward(self, t, xv, yv, hv, cv, tv, gv):
        h = self.h
        gv[: , [t]] = self.Whx @ xv[:, [t]] + self.Whh @ hv[:, [t-1]] + self.bh
        gv[:h, [t]] = np.tanh(gv[:h, [t]])
        gv[h:, [t]] = sigmoid(gv[h:, [t]])
        cv[:, [t]] = gv[:h, [t]] * gv[h:2*h, [t]] + gv[2*h:3*h, [t]] * cv[:, [t-1]]
        tv[:, [t]] = np.tanh(cv[:, [t]])
        hv[:, [t]] = tv[:, [t]] * gv[3*h:, [t]]
        yv[:, [t]] = self.Wyh @ hv[:, [t]] + self.by
        return None



    def backprop(self, t, xv, hv, cv, tv, gv, dy, dWhh, dWhx, dWyh, dbh, dby, dhnext, dcnext):
        h = self.h
        dWyh += np.dot(dy, hv[:, [t]].T)
        dby += dy
        dh = np.dot(self.Wyh.T, dy) + dhnext # backprop into h
        dc = dh * gv[3*h:, [t]] * (1 - tv[:, [t]]**2) + dcnext
        dcnext = dc * gv[2*h:3*h, [t]]
        da = dc * gv[h:2*h, [t]] * (1 - gv[:h, [t]]**2)
        di = dc * gv[:h, [t]] * gv[h:2*h, [t]] * (1 - gv[h:2*h, [t]])
        df = dc * cv[:, [t-1]] * gv[h*2:3*h, [t]] * (1 - gv[h*2:3*h, [t]])
        do = dh * tv[:, [t]] * gv[3*h:, [t]] * (1 - gv[3*h:, [t]])
        dg = np.vstack((da, di, df, do))
        dWhx += np.dot(dg, xv[:, [t]].T)
        dWhh += np.dot(dg, hv[:, [t-1]].T)
        dbh += dg
        dhnext = np.dot(self.Whh.T, dg)
        return self.Whx.T @ dg

    def train_sequence(self, inputs, targets, hprev, cprev, xv, yv, pv, hv, cv, tv, gv, dWhh, dWhx, dWyh, dbh, dby, dhnext, dcnext):
        n = len(inputs)
        loss = 0

        # initialisation
        xv[inputs, range(n)] = 1
        hv[:, [-1]] = hprev
        cv[:, [-1]] = cprev
        for t in range(n):
            self.forward(t, xv, yv, hv, cv, tv, gv)
            exp_yv = np.exp(yv[:, [t]])
            pv[:, [t]] = exp_yv / np.sum(exp_yv, axis=0)
            loss += -np.log(pv[targets[t], t])

        for t in reversed(range(n)):
            dy = np.copy(pv[:, [t]])
            dy[targets[t]] -= 1
            self.backprop(t, xv, hv, cv, tv, gv, dy, dWhh, dWhx, dWyh, dbh, dby, dhnext, dcnext)
        return loss, hv[:, [-1]], cv[:, [-1]]

    def train(self, data, chars, seq_length, lr=0.1):
        n, p, l = seq_length, 0, 0
        i, h, o = self.i, self.h, self.o
        mWhx, mWhh, mWyh = np.zeros_like(self.Whx), np.zeros_like(self.Whh), np.zeros_like(self.Wyh)
        dWhh, dWhx, dWyh = np.zeros_like(self.Whh), np.zeros_like(self.Whx), np.zeros_like(self.Wyh)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
        xv, yv, pv = np.zeros((i, n)), np.zeros((o, n)), np.zeros((o, n))
        hv, cv, tv, gv = np.zeros((h, n)), np.zeros((h, n)), np.zeros((h, n)), np.zeros((4*h, n))
        dhnext, dcnext = np.zeros((h, 1)), np.zeros((h, 1))
        hprev, cprev = np.zeros((h,1)), np.ones((h,1))
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
        while True:
            if p+seq_length+1 >= data_size:
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                cprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data
            inputs = np.array([int(*np.where(chars==ch)[0]) for ch in data[p:p+seq_length]])
            targets = np.array([int(np.where(chars==ch)[0]) for ch in data[p+1:p+seq_length+1]])

            # forward seq_length characters through the net and fetch gradient
            dWhh[:, :], dWhx[:, :], dWyh[:, :] = 0, 0, 0
            dby[:], dbh[:] = 0, 0
            dhnext[:], dcnext[:] = 0, 0
            xv[:, :] = 0
            loss, hprev, cprev = self.train_sequence(inputs, targets, hprev, cprev, xv, yv, pv, hv, cv, tv, gv, dWhh, dWhx, dWyh, dbh, dby, dhnext, dcnext)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if l & 127 == 0:
                print(f'iter {l}, loss: {smooth_loss}')
            if l & 1023 == 0 and l != 0:
                self.save(smooth_loss)

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Whx, self.Whh, self.Wyh, self.bh, self.by],
                                        [dWhx, dWhh, dWyh, dbh, dby],
                                        [mWhx, mWhh, mWyh, mbh, mby]):
                mem = 0.99 * mem - lr * dparam
                param += mem # adagrad update

            p += seq_length # move data pointer
            l += 1 # iteration counter

    def save(self, loss):
        with open(f'char-lstm-2_{loss:.2f}.bin', 'wb') as fd:
            pickle.dump(self, fd)


    def sample(self, chars, hv, cv, seed_ix, n):
      h, v = self.h, self.i
      x = np.zeros((self.i, 1))
      x[seed_ix] = 1
      ixes = [seed_ix]
      for t in range(n):
        wab = self.Whx @ x + self.Whh @ hv + self.bh
        a = np.tanh(wab[:h])
        i, f, o = sigmoid(wab[h:2*h]), sigmoid(wab[2*h:3*h]), sigmoid(wab[3*h:])
        cv = a * i + f * cv
        tv = np.tanh(cv)
        hv = tv * o
        yv = self.Wyh @ hv + self.by
        p = np.exp(yv) / np.sum(np.exp(yv))
        ix = np.random.choice(range(v), p=p.ravel())
        x = np.zeros((v, 1))
        x[ix] = 1
        ixes.append(ix)
      return ''.join(chr(int(chars[ix])) for ix in ixes)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def load(filename):
    with open(filename, 'rb') as fd:
         return pickle.load(fd)

lstm = LSTMCell(vocab_size, hidden_size,  vocab_size)
