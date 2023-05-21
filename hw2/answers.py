r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.3
    lr = 0.02
    reg = 0.06
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        wstd = 0.2
        lr = 0.004
        reg = 0.005
    if opt_name == 'momentum':
        wstd = 0.2
        lr = 0.003
        reg = 0.005
    if opt_name == 'rmsprop':
        wstd = 0.1
        lr = 0.0001
        reg = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.15
    lr = 0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Overall, the graphs showing the results match what we expected to see.
For dropout = 0 (no dropout), we get overfitting, as when the training loss keeps decreasing, the test loss starts to go up at some point, and the test accuracy cease to improve.
This is to be expected, as the dataset is small and we don't get good generalization. 
Using dropout allowed us to achieve better generalization, as we can see that the there is no overfitting - the test loss is lower and doesn't increase by noticeable amount, and the accuracy increases as we progress through the epochs. 

2. Using dropout = 0.4 yielded better test results compared to dropout = 0.8 (about the same test loss and higher test accuracy). 
With dropout = 0.8 we are dropping too many nodes, which significantly hinders our abillity to fit well, and that is why it leads to worse test results.
"""

part2_q2 = r"""
**Your answer:**
Indeed, it is possible for the test loss to increase for a few epochs while the test accuracy also increases.
The loss is measured through as continuous loss function, while the accuracy is computed by the percentage of how many labels were predicted correctly.
We can encounter a situation where a few steps were taken in the direction that caused the total test loss to increase, yet at the same time it just so happens that the number of labels which were predicted correctly increased abit.
In general, with the steps we hope to eventually converge into the lowest minimum of the loss function, but along the way we can encounter positive slopes that we need to go through in order to eventually reach it.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Our configuration was to train for 50 epochs but most of them were finished before because of the early stopping feature we implemented.
Our main focus for the hyper parameters was to find the best value for the amount of batches and the hidden dimension layers.
with the initial values we saw the test loss decreasing fast for the first 10 epochs and then goes up,
this behavior suggests that the model might be overfitting so we decided to increase the amount of batches to 200.
also we wanted to add some more depth to the network so we defined 2 hidden layers [128, 56].
We used pool_every = 2 except for L = 16 we had to increase it to 8 to preserve the image height and width as expected.

The depth that produces the best results was L=4 both when K=32 and K=64.
When L=2 the network missing data and when increasing it to L=8/16 there were too many layers which resulted the network to overfit and deliver bad results
due to the vanishing and exploding gradients, for L=8/16 the network didn't learn anything the accuracy and lost didn't change since the beginning.

We can try to solve this issue above for L=8/16 by adding residual blocks to the network, we can try to change out activation function or use batch normalization.

"""

part3_q2 = r"""
We ran it with the same configuration as the previous section.
Similarly to experiment 1.1, for L=4 we got the best results, one thing to notice that in both experiment 1.1 and 1.2 the best result was for the highest K,
meaning we can probably get better accuracy by increasing K.

for L=2 in terms of the test loss and accuracy K=32 has the best results and in terms of the train loss and accuracy K=258 has the best results.

for L=4 we can see that K=258 has the best results for all parameters, the train accuracy higher the 80% and the test log loss was under 1.

for L=8 similarly to section 1.1 we were overfitting, the loss was stuck at 2.3 and the accuracy didn't go above 10% (like guessing the result), the cause of this is similar to what was explained in 1.1

"""

part3_q3 = r"""
Because the now the network has higher complexity we can reduce it from the other side by defining only 1 hidden dimension of 128 and reducing to 100 batches.
We ran it with the same configuration as the previous section except for pool_every.

for L=1 we set pool_every = 1.

for L=2 we set pool_every = 2.

for L=3 we set pool_every = 2.

for L=4 we set pool_every = 3.

we can see that for L=3,4 the network wasn't able to train, the accuracy was 10% and wasn't changed since the beginning.

for L=1 we can see good results. we got test loss ~ 1, the test accuracy was ~ 70% and train accuracy ~ 80%
for L=2 we got slightly worse result than L=1 but they were close.
In this experiment the network was much deeper than the previous ones, hence we were overfitting very fast, that's why we got the best results for L=1.

"""

part3_q4 = r"""

We saw in the previous section vanishing and exploding gradients problems. to deal with this problem we implemented 2 technics:
1. batch normalization
2. residual blocks (skip connection)

after each convolutional layer we added batch normalization and skip connection after each 2 blocks except maybe for the last layer (if P was odd)

we used hidden_dims=[128], and increased the bs_train to 150.

we can see now that for L=3,4 unlike experiment 1 the network was able to train and got results even better then the best of experiment 1, 
it means that technics really helped to deal with the gradients, we used pool_every 3.

in terms of the train loss:

for L=1 we can got accuracy of ~ 90% and loss of ~ 0.3. (pool every 1)

for L=2 we can see we got out best result of ~ 94% accuracy and loss of ~ 0.1. (pool every 2)

in terms of the test loss we can see that both L=1 and L=2 got similar results.

"""
