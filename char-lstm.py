import numpy as np
from random import uniform
import pickle



# data I/O
data = open('input', 'rb').read()
data = np.frombuffer(data, dtype=np.uint8)
chars = np.unique(data)
data_size, vocab_size = len(data), len(chars)

hidden_size = 1000
seq_length = 20
learning_rate = 1e-2
debug=False
if debug:
    print("data has {} characters, {} unique.".format(data_size, vocab_size))

# model parameters
Whx = np.random.randn(4*hidden_size, vocab_size)*0.1 # input to hidden
Whh = np.random.randn(4*hidden_size, hidden_size)*0.1 # hidden to hidden
Wyh = np.random.randn(vocab_size, hidden_size)*0.1 # hidden to output
bh = np.ones((4*hidden_size, 1))*0.1 # hidden bias
by = np.zeros((vocab_size, 1))*0.1 # output bias

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def lossFun(inputs, targets, hprev, cprev):
    xs, hs, ys, ps, gates, cs, ts = {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)
    h, loss = hidden_size, 0
# forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        gates[t] = (np.dot(Whx, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
        gates[t][:h] = np.tanh(gates[t][:h])
        gates[t][h:] = sigmoid(gates[t][h:])
        cs[t] = gates[t][:h] * gates[t][h:2*h] + gates[t][2*h:3*h] * cs[t-1]
        ts[t] = np.tanh(cs[t])
        hs[t] = gates[t][3*h:] * ts[t]
        ys[t] = np.dot(Wyh, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Whx), np.zeros_like(Whh), np.zeros_like(Wyh)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Wyh.T, dy) + dhnext # backprop into h
        dc = dh * gates[t][3*h:] * (1 -ts[t]**2) + dcnext
        dcnext = dc * gates[t][2*h:3*h]
        da = dc * gates[t][h:2*h] * (1 - gates[t][:h]**2)
        di = dc * gates[t][:h] * gates[t][h:2*h] * (1 - gates[t][h:2*h])
        df = dc * cs[t-1] * gates[t][h*2:3*h] * (1 - gates[t][h*2:3*h])
        do = dh * ts[t] * gates[t][3*h:] * (1 - gates[t][3*h:])
        dg = np.vstack((da, di, df, do))
        dWxh += np.dot(dg, xs[t].T)
        dWhh += np.dot(dg, hs[t-1].T)
        dbh += dg
        dhnext = np.dot(Whh.T, dg)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], cs[len(inputs)-1]


def sample(hv, cv, seed_ix, n):
    h, v = hidden_size, vocab_size
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        wab = Whx @ x + Whh @ hv + bh
        a = np.tanh(wab[:h])
        i, f, o = sigmoid(wab[h:2*h]), sigmoid(wab[2*h:3*h]), sigmoid(wab[3*h:])
        cv = a * i + f * cv
        tv = np.tanh(cv)
        hv = tv * o
        yv = Wyh @ hv + by
        p = np.exp(yv) / np.sum(np.exp(yv))
        ix = np.random.choice(range(v), 1, p=p.ravel())
        x = np.zeros((v, 1))
        x[ix] = 1
        ixes.append(ix)
    return ''.join([chr(int(chars[ix])) for ix in ixes])

# gradient checking
def gradCheck(inputs, targets, hprev, cprev):
    global Whx, Whh, Wyh, bh, by
    num_checks, delta = 10, 1e-5
    _, dWxh, dWhh, dWhy, dbh, dby, _, _ = lossFun(inputs, targets, hprev, cprev)
    for param,dparam,name in zip([Whx, Whh, Wyh, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        print(f"----------------- {name} ----------------")
        for i in range(num_checks):
            ri = int(uniform(0,param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev, cprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _, _ = lossFun(inputs, targets, hprev, cprev)
            param.flat[ri] = old_val # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / ( 2 * delta )
            rel_error = 0 if cg0 == 0 and cg1 == 0 else abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print(f'{grad_numerical}, {grad_analytic} => {rel_error}')
            # rel_error should be on order of 1e-7 or less

def save():
    global Whh, Whx, Wyh, bh, by
    with open(f'char-lstm.bin', 'wb') as fd:
        pickle.dump([np.asnumpy(Whh), np.asnumpy(Whx), np.asnumpy(Wyh), np.asnumpy(bh), np.asnumpy(by)], fd)

def load(filename):
    global Whh, Whx, Wyh, bh, by
    with open(filename, 'rb') as fd:
        weights = pickle.load(fd)
    Whh, Whx, Wyh, bh, by = weights


def train():
    global Whh, Whx, Wyh, bh, by
    n, p = 0, 0
    mWhx, mWhh, mWyh = np.zeros_like(Whx), np.zeros_like(Whh), np.zeros_like(Wyh)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    hprev, cprev = np.zeros((hidden_size,1)), np.ones((hidden_size,1))
    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    while True:
        if p+seq_length+1 >= data_size:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            cprev = np.zeros((hidden_size,1)) # reset RNN memory
            p = 0 # go from start of data
        inputs = np.array([int(*np.where(chars==ch)[0]) for ch in data[p:p+seq_length]])
        targets = np.array([int(np.where(chars==ch)[0]) for ch in data[p+1:p+seq_length+1]])

        # forward seq_length characters through the net and fetch gradient
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n & 127 == 0:
            print(f'iter {n}, loss: {smooth_loss}')
        if n & 1023 == 0 and n != 0:
            save(smooth_loss)

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Whx, Whh, Wyh, bh, by],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWhx, mWhh, mWyh, mbh, mby]):
            mem = 0.99 * mem - learning_rate * dparam
            param += mem # adagrad update

        p += seq_length # move data pointer
        n += 1 # iteration counter


def sample_sequence():
    global Whh, Whx, Wyh, bh, by
    load('char-lstm.bin')
    h = np.zeros((hidden_size, 1))
    seed = np.random.randint(65, 90)
    n = 1000
    text = sample(h, h, seed, n)
    print(text)

sample_sequence()
