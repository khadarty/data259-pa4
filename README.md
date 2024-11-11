# Programming Assignment 4

**DATA 259**

**Fall 2024** 

## Introduction

For questions on this and other assignments, you may need to write Python code to do data analysis, provide only an explanation in prose, or do both. In this course, we will be using Quarto documents to author assignment submissions and reports to develop our skills in reproducible research tools. 

Use a code chunk for any code you write in solving problems, and use Markdown for your explanation of what you did. We will not count your answer as correct if we cannot see it/it is written in a code comment. We recommend one code chunk for each question, but feel free to add more as you see fit. 

In general, within the Markdown, you should explain the approach you took in your code at a high level in addition to explicitly answering any questions that were asked. It is up to you to decide what code-based analyses, if any, are appropriate for a particular problem.

When you are finished, please render your document to a PDF and upload your assignment in Gradescope. Make sure to select the areas of the page corresponding to the questions on the assignment outline. It is much easier for the graders to give you feedback this way, and you will therefore get your homework assignments back faster. If there is a lot of excess output, either revisit your code to make sure you are not printing excessively, or delete the pages with excess output from the PDF before submitting.

## $k$-anonymity

[k-anonymity](https://en.wikipedia.org/wiki/K-anonymity) is a privacy concept - a dataset satisfies $k$-anonymity if every individual appearing in the dataset cannot be distinguished from at least $k-1$ other individuals in the dataset. $k$-anonymization can prevent re-identification attack since the attackers cannot identify a single individual out of $k$ records that all share the same [quasi-identifiers](https://en.wikipedia.org/wiki/Quasi-identifier) attributes. However, one of the biggest caveat of $k$-anonymization is that the privacy protection it offers can degrade drastically when dealing with *multiple releases*.

Consider the following two datasets containing patients records from two different hospitals. In both datasets, zipcode, age, nationality are considered non-sensitive quasi-identifiers in this dataset, and condition is a sensitive attribute that should be excluded. The records from hospital A is 4-anonymous and the records from hospital B is 6-anonymous (double check this is the case).

```python
import pandas as pd

df1 = pd.read_csv("hospital_A.csv")
df2 = pd.read_csv("hospital_B.csv")
```

1. If Alice visited both hospitals, and she is 28, can you deduce Alice’s medical condition from the combination of the two datasets?

2. Based on your answer to the previous question, does the combined dataset still satisfy $k$-anonymity?

## Randomized Response

Differential privacy is a mathematical criterion for quantifying and preserving privacy during data analysis. In this assignment you will be asked to build up an implementation of certain analyses that achieve (epsilon, delta) differential privacy.

3. The origin of differential privacy is in \textbf{randomized response}. In a randomized response protocol, the person taking a survey is asked to privately flip a coin. If the coin lands on heads, then the person should answer the yes or no question truthfully. If the coin lands on tails, then the person should privately flip the coin again and respond yes if heads and no if tails.

4. Write a program to simulate this process with a population of 1000 people. Run the simulation 100 times where the percentage of people for whom the true answer to the survey question is yes is 1\%, 10\%, 25\% and 50\%. For example, if the percentage is 1\%, then there should be 10 people whose true answer is yes. You can use methods such as \href{https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html}{\texttt{numpy.random.choice}} to generate the actual answers based on the coin flips (for this question consider the coin to be fair with probability of 0.5 of flipping heads or tails). We provide you with some skeleton code to help you get started.

```python
def simulate(percentage, n_pop=1000, coin_prob=0.5): # assign default values
    # fill in your code here
    # returns the proportion of yes responses out of n_pop 
    # (ie. num of yes responses / 1000)
    return

# a list containing the simulation result for each percentage
simulations = []

# iterate over different percentages
for percentage in [0.01, 0.1, 0.25, 0.5]:
    # simulate result for the current percentage
    simulation = []
    # simulate 100 times
    for _ in range(100):
        # run the simulate function and add the result to list
        proportion = simulate(percentage)
        simulation.append(proportion)
    # add the simulation result to the simulations list
    simulations.append(simulation)
```

5. For each percentage, plot a histogram with the $x$-axis being the proportion of yes responses and $y$-axis being the number of runs. Also compute the expected probability of answering yes for each percentage. What do you notice about the distribution in relation to the probability you just computed? Write a few sentences describing what you see.

6. In the randomized response, the coin flip was parameterized with probability $\frac{1}{2}$ of answering truthfully. Try biasing the coin so that the probability of deciding to not answer truthfully is 0, $\frac{1}{8}$, $\frac{1}{4}$ and $\frac{1}{2}$. Run the simulation again this time choosing the percentage of true yes answers equal to $\frac{1}{4}$, and make histograms similar to the ones made in the previous question. Note that the probability of responding yes given a decision to produce a fake answer (when the first coin lands on tail) should still be $\frac{1}{2}$. Note: tou should reuse the \texttt{simulate} function you wrote earlier (notice the function parameters include everything you need to vary!)

7. Do you notice any pattern in the histograms? Specifically, as the coin probability decreases from $\frac{1}{2}$ to 0, how does the distribution change?

8. Calculate the standard deviation of each simulation's distribution (of the proportion of yes responses). How does the standard deviation change?

## Laplace Mechanism

Differential privacy is a definition, rather than a technique. The randomized response you saw in Question 1 is an example of a technique that satisfies ($ln(3)$, 0) differential privacy. Another common method for achieving differential privacy is the Laplace mechanism.

The Laplace mechanism is a method for achieving differential privacy. While randomized response is distributed, in that the data aggregator finds out about only the noisy data, the Laplace mechanism is centralized. The Laplace mechanism is meant for a scenario in which a data aggregator already has the "true" data, but wants to release the results of queries without violating the privacy of whose data it controls. It does this by adding random noise to the output of these queries. The noise in this case is sampled from the [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution), hence the name.

The key to the Laplace mechanism is the scale of the distribution the noise is sampled from. The scale of the distribution is the "spread"--how large the standard deviation is and how likely you are to see significant outliers. A scale that is larger means that on average there will be more noise (and more privacy), whereas a scale that is closer to 0 will be closer to the original value (and therefore less private). With the Laplace mechanism, the scale is set as the sensitivity of the query over epsilon. Recall that smaller epsilon implies more privacy, and a larger epsilon implies less privacy.

What is the sensitivity of a query? For our purposes, we can just think of a query as a function that takes data and produces a numerical answer. The sensitivity of the query is the maximum amount that the query can differ if you give it data that is modified in a single entry. For example, let's say we want to count the number of entries in a vector that are greater than 12.6.

```python
def query(array): # our query
    return sum(array > 12.6)
```

Then think about two hypothetical databases (which we can think of right now as just vectors). These databases (vectors) are *neighbors* meaning that they differ in exactly one entry. `x` and `y` in the cell below are examples of one such set of neighboring databases.

```python
import numpy as np

x = np.array([0, 15, 26, 35, -10, 12])
y = np.array([12.7, 15, 26, 35, -10, 12])
```

The difference in answers between running the query on `x` and `y` is 1. A hypothetical attacker could use information like this (or more realistically a series of slightly different queries) to recover information about people in the database.

```python
abs(query(x) - query(y))
```

Crucially, the most that the output of this query could be altered given any database `z` is by taking an entry of `z` that is above 12.6 to be below 12.6, or taking an entry below 12.6 to be above 12.6. In either case, this would only change the output of the query by at most 1, so the sensitivity of this query is 1. Therefore, the Laplace mechanism would add to the output of this query a value drawn from a Laplace distribution with scale `1/eps`.

9. Now, suppose we have a counting query that counts the number of people answering yes to the survey as in the previous question (but without the random response, so it's simply the number of truthful "yes" answers). What is the sensitivity of this query? Implement the Laplace mechanism for this query with your specified sensitivity, by adding noise drawn from the Laplace distribution to the output count. Like you did previously, construct a population with percentage of true yes answers equal to $\frac{1}{4}$, and run the simulation (100 runs each) for epsilon values 10, 1, 0.1 and 0.01. You can generate Laplacian noise through the \texttt{laplace.rvs} function.

10. Plot the distribution of laplace count for each of the four epsilon values using a boxplot.

```python
def laplace_count(arr, epsilon):
    # fill in your code here
    # Given a list of survey responses and an epsilon value, 
    # return the noisy count of yes responses
    return

# construct the arr list, 
# run simulations by calling laplace_count 100 times for each epsilon 

# plot
```

11. Examine the graph you just made. What do you notice about the distribution of answers as epsilon decreases? What happens when epsilon is quite small (e.g. less than 0.01?) Does this pose any problems?

12. Compute the average laplace count (over 100 runs) for each epsilon. How much do they differ from the true count of the population? What is the standard deviation for each epsilon? What does this suggest about repeated queries under the differential privacy framework? How can a data curator/aggregator defend against repeated queries?

13. Counting is useful, but sometimes we also need answers to other questions like mean and median. Implement the Laplace mechanism for computing the mean of a \textit{real valued array}. For this you will need to derive the sensitivity of the mean. That is, how much a query may vary given a difference of one entry. To do this, you will need to add as a parameter the allowable range of the list -- the maximum and minimum values that are allowable.

14. You should think carefully about how to handle data that is out of the specified range. Specifically, if you drop data outside of the range, how would that affect the sensitivity analysis? Could this inadvertently reveal information meant to remain private? Is there a better way to handle data that avoids these problems?

```python
def laplace_mean(arr, epsilon, min_val, max_val):
    # fill in your code here
    # arr is a list of real values
    return
```

15. Using the \texttt{age} column in the income data set (``income.csv"), plot the average difference between the true mean age and the differentially private mean age over 100 runs for feasible points in the grid formed by minimum and maximum age, using epsilon = 0.1. 

```python
import pandas as pd
df = pd.read_csv("income.csv")
age = df["age"]
true_mean_age = np.mean(age) # true mean age

ages = [17, 20, 30, 40, 50, 60, 70, 80, 90]
grid = [(x, y) for x in ages for y in ages]
feas_grid = [pt for pt in grid if pt[0] < pt[1]]

indexer = {ages[i] : i for i in range(len(ages))}

data = np.full((len(ages), len(ages)), None, dtype = float)

for pt in feas_grid:
    #  You should pass minimum age and manixum age 
    #       when calling laplace_mean function
    min_age, max_age = pt

    i = indexer[min_age]
    j = indexer[max_age]

    # fill in the code here and assign data[i, j]
    # data[i, j] should be the average difference between 
    # the true mean age and the differentially private 
    # mean age over 100 runs

    data[i, j] =
```

16. Look at magnitude and direction of the error in the plot. What does this suggest about choosing reasonable cutoffs when handling differentially private data? How might one go about minimizing this sort of error? To interpret the graph it may be useful to also plot a histogram of ages in the dataset.
