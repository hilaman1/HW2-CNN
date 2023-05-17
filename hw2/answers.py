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
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
