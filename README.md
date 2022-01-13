# Sequence Generator/Mimicking

## Topic

The sequence generator is a recurrent neural network, more specifically a [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)] cell.
the cell is trained with a text (could be anything), one sequence one at a time.

## Algorithm

We use the backpropagation algo to update the weight, by minimising the cost. and the
is just the difference between the predict sequence and the input sequence. then
we compute the gradient of the loss with respect to the weights. And this process is
repeated, until the cost is satisfying.

## Notes

1. the file `char-lstm.py` and `char-lstm-v2.py` contains the code for training the lstm cell.
2. the file `char-lstm.bin` is pickled file for the weights used in `char-lstm.py`
3. i wanted to implement a n-cell LSTM which are stacked. but i didn't get it right, because of errors in backprop implementation. but i am still working on that.
