Download link :https://programming.engineering/product/cs440-ece448-mp08-repeated-games/

# CS440-ECE448-MP08-Repeated-Games
CS440/ECE448 MP08: Repeated Games
The first thing you need to do is to download this file: mp08.zip. It has the following

content:

submitted.py



Your homework. Edit, and then submit to Gradescope.

mp08_notebook.ipynb



This is a Jupyter notebook to help you debug. You can

completely ignore it if you want, although you might find that it gives you useful

instructions.

grade.py



Once your homework seems to be working, you can test it by typing

python grade.py tests/tests_visible.py

, which will run the tests in .

tests/test_visible.py



This file contains about half of the unit tests that

Gradescope will run in order to grade your homework. If you can get a perfect score on these tests, then you should also get a perfect score on the additional hidden tests that Gradescope uses.

solution.json



This file contains the solutions for the visible test cases, in JSON

format. If the instructions are confusing you, please look at this file, to see if it can help to clear up your confusion.

requirements.txt



This tells you which python packages you need to have

grade.py

installed, in order to run . You can install all of those packages by typing

pip install -r requirements.txt pip3 install -r


or

requirements.txt

.

mp08_notebook.ipynb

This file ( ) will walk you through the whole MP, giving you

instructions and debugging tips as you go.

Table of Contents

Episodic Games: Gradient Ascent

Episodic Games: Corrected Ascent

Extra Credit: Sequential Games

Grade Your Homework

Episodic Games: Gradient Ascent

It is possible to learn an optimal strategy for a two-player game using machine learning methods. A simple gradient descent, however, doesn’t work very well: gradient descent

3/31/24, 5:21 PM mp08_notebook

simultaneously on two different criteria can converge to an orbit, rather than converging

to a stable point. The first part of the MP explores this outcome.


In [1]: import submitted, importlib

import numpy as np

import matplotlib.pyplot as plt

importlib.reload(submitted)

help(submitted.episodic_game_gradient_ascent)

Help on function episodic_game_gradient_ascent in module submitted:

episodic_game_gradient_ascent(init, rewards, nsteps, learningrate)

nsteps of a 2-player, 2-action episodic game, strategies adapted using g radient ascent.

@param:

init (2) – intial logits for the two players

rewards (2,2,2) – player i receives rewards[i,a,b] if player 0 plays a a nd player 1 plays b

nsteps (scalar) – number of steps of gradient descent to perform learningrate (scalar) – learning rate

@return:

logits (nsteps,2) – logits of two players in each iteration of gradient descent

utilities (nsteps,2) – utilities[t,i] is utility to player i of logits [t,:]

Initialize: logits[0,:] = init.

Iterate: In iteration t, player 0’s actions have probabilities sig2(logi ts[t,0]),

and player 1’s actions have probabilities sig2(logits[t,1]).

The utility (expected reward) for player i is sig2(logits[t,0])@rewards [i,:,:]@sig2(logits[t,1]),

and the next logits are logits[t+1,i] = logits[t,i] + learningrate * uti lity_partials(rewards, logits[t,:]).

nsteps

As you can see, the intended behavior of this function is to perform steps of gradient descent on a simple 2-player, 2-action game. Let’s explore this.

utility_partials

First, this function seems to depend on the function . Let’s look at that.



3/31/24, 5:21 PM mp08_notebook

Help on function utility_partials in module submitted:

utility_partials(R, x)

Calculate vector of partial derivatives of utilities with respect to log

its.

If u[i] = sig2(x[0])@R[i,:,:]@sig2(x[1]),

then partial[i] is the derivative of u[i] with respect to x[i].

@param:

R (2,2,2) – R[i,a,b] is reward to player i if player 0 plays a, player 1 plays b

x (2) – player i plays move j with probability softmax([0,x[i]])[j]

@return:

partial (2) – partial[i] is the derivative of u[i] with respect to x[i].

HINT: You may find the functions sig2 and dsig2 to be useful.

Following up on the hint:


In [3]: importlib.reload(submitted)

help(submitted.sig2)

help(submitted.dsig2)

Help on function sig2 in module submitted:

sig2(x)

Calculate the vector p = [1-sigmoid(x), sigmoid(x)] for scalar x

Help on function dsig2 in module submitted:

dsig2(p)

Assume p=sig2(x). Calculate the vector v such that v[i] is the derivati ve of p[i] with respect to x.

Let’s start with a rather difficult game to optimize, in which player 0 wins more if their

move is the opposite of player 1’s move, whereas player 1 wins more if their move is the

same as that of player 0:


In [4]: rewards = np.array([[[1,2],[2,1]],[[2,1],[1,2]]])

print(rewards)

[[[1

2]

[2

1]]

[[2

1]

[1

2]]]

utility_partials

Once you have written the function , you can experiment with the

partial derivatives of the utility with respect to the players’ logits:

3/31/24, 5:21 PM mp08_notebook

fig, ax = plt.subplots(1, figsize=(4,4))


ax.scatter(1/(1+np.exp(–logits[:,0])), 1/(1+np.exp(–logits[:,1])), c=np.aran ax.set_xlabel(‘Probability that player 0 cooperates’) ax.set_ylabel(‘Probability that player 1 cooperates’)

Out[6]: Text(0, 0.5, ‘Probability that player 1 cooperates’)


However, if we start from any position other than the equilibrium, then gradient descent

fails to converge.


3/31/24, 5:21 PM mp08_notebook


As you can see, gradient descent does not converge to the Nash equilibrium; instead,

the players orbit around the Nash equilibrium:



When player 1 is usually defecting, then player 0 increases their cooperation

probability



When player 0 is usually cooperating, then player 1 increases their cooperation

probability



When player 1 is usually cooperating, then player 0 decreases their cooperation

probability



When player 0 is usually defecting, then player 1 decreases their cooperation

probability

Episodic Games: Corrected Ascent

The paper “The mechanics of n-player games” proposed a solution to the problem shown above, by borrowing an idea from orbital mechanics. They suggested that we should impose some friction, so the orbit will decay toward a stable equilibrium. They called this friction term the symplectic correction:


3/31/24, 5:21 PM mp08_notebook

Help on function symplectic_correction in module submitted:

symplectic_correction(partials, hessian)

Calculate the symplectic correction matrix from Balduzzi et al., “The Me chanics of n-player Games,” 2018.

Apparently we need to calculate the Hessian:


In [9]: importlib.reload(submitted)

help(submitted.utility_hessian)

Help on function utility_hessian in module submitted:

utility_hessian(R, x)

Calculate matrix of partial second derivatives of utilities with respect to logits.

Define u[i] = sig2(x[0])@R[i,:,:]@sig2(x[1]),

then hessian[i,j] is the second derivative of u[j] with respect to x[i] and x[j].

@param:

R (2,2,2) – R[i,a,b] is reward to player i if player 0 plays a, player 1 plays b

x (2) – player i plays move j with probability softmax([0,x[i]])[j]

@return:

hessian (2) – hessian[i,j] is the second derivative of u[i] with respect to x[i] and x[j].

HINT: You may find the functions sig2, dsig2, and Hsig2 to be useful.


In [10]: importlib.reload(submitted)

help(submitted.Hsig2)

Help on function Hsig2 in module submitted:

Hsig2(p)

Assume p=sig2(x). Calculate the vector v such that v[i] is the second d erivative of p[i] with respect to x.

utility_hessian

Once you have written the function , you can test it.

sig2(logits[t,0]) sig2(logits[t,1])

Notice that, if and is any mixed

equilibrium, then:



The main diagonals of the Hessian will both be zero. Each player cannot change

their own utility by changing the probabilities with which they choose actions.



The off-diagonal elements of the Hessian might not be zero. In this case, player 0

would prefer that player 1 change their strategy (because then the two players

would sometimes make different moves, which would benefit player 0), whereas

3/31/24, 5:21 PM mp08_notebook

player 1 would prefer that player 0 NOT change their strategy (because then the two players would sometimes make different moves, which harms player 1).


In [11]: importlib.reload(submitted)

= submitted.utility_hessian(rewards, [0,0]) print(H)

[[ 0. -0.125]

[ 0.125 0. ]]

The symmetric part of the Hessian is S = 0.5(H + HT ). Notice that, because of

symmetry, dT Hd = dT Sd for any vector d. The symmetric part therefore has a unique relationship to stability of the Nash equilibrium:



Positive definite symmetric part: changing the Nash equilibrium by any small

vector, d, causes the sum of the utilities for all players to increase, i.e., dT Hd > 0.

All of the eigenvalues of a positive-definite symmetric matrix are positive real

numbers. Notice that, if the Hessian’s symmetric part is positive definite, the Nash

equilibrium is unstable: no player has an incentive to change from the Nash

equilibrium, but as soon as one player changes, even slightly, then all of the other

players can increase their utility by making the change bigger.



Negative definite symmetric part: changing the Nash equilibrium by any small

vector, d, causes the sum of the utilities for all players to decrease, i.e., dT Hd < 0.

All of the eigenvalues of a negative-definite symmetric matrix are negative. Notice

that a negative definite Hessian means that the Nash equilibrium is stable: If any player moves away from the equilibrium by even a small amount, the utilities of the players go down or stay the same, so all players want to move back toward the equilibrium.



Zero-valued symmetric part: In the example we’ve been working with so far, the

Hessian is anti-symmetric, so its symmetric part is identically zero. In this case,

changing the Nash equilibrium by a small amount does not change the average loss

at all (dT Hd = 0, so, although the players have no particular incentive to move

back to the equilibrium, on average they don’t mind moving back to the equilibrium, so we can say that the game is marginally stable.



Neither negative nor positive definite: Some eigenvalues have positive real parts, some have negative. If one player moves away from the equilibrium in a direction d,

the other players might want to move further away (dT Hd might be positive) or they might want to move back to the equilibrium (dT Hd might be negative).

Notice that the game we’ve been using so far has a perfectly anti-symmetric Hessian, so its symmetric part is exactly zero:


3/31/24, 5:21 PM mp08_notebook

[[0. 0.]

[0. 0.]]

The eigenvalues are [0. 0.]

So this game is marginally stable.

In the prisoner’s dilemma, in the Nash equilibrium, both players always defect. Each player would prefer that the other player cooperate, but that preference is wiped out because sigmoid(x)=0 corresponds to x=-np.inf, which is so small that any measurable change in the other player’s logit has no effect, hence the entire Hessian is zero.


3/31/24, 5:21 PM mp08_notebook

Since the game of Chicken is neither stable nor unstable, when one player moves away

from equilibrium, the other player might want to move back. If you think about it, you can

see that the player who decides to defect (e.g., by crashing their car) prefers to move

away from the mixed equilibrium toward a pure equilibrium, but the other player is

unhappy with this choice, because they are thereby forced to cooperate (by chickening

out). The outcome cannot be defined based on mathematics alone; it depends on the

psychology of the players.

The symplectic correction is used in the function

episodic_games_corrected_ascent

:


In [19]: importlib.reload(submitted)

init = [–1,–1] # Initial logits [-1,-1] means that initial probabilities ar nsteps, learningrate = 5000, 0.1

logits, utilities = submitted.episodic_game_corrected_ascent([1,1], rewards,

fig, ax = plt.subplots(1, figsize=(4,4))

ax.scatter(1/(1+np.exp(–logits[:,0])),1/(1+np.exp(–logits[:,1])), c=np.arang ax.set_xlabel(‘Probability that player 0 cooperates’) ax.set_ylabel(‘Probability that player 1 cooperates’)

print(‘The logits have converged to:’,logits[–1,:])

The logits have converged to: [ 0.00086998 -0.00093214]


As you can see, the symplectic correction has added a kind of friction to the orbit,

causing it to decay toward the nearest stable Nash equilibrium, with logits nearly zero, and the probability of cooperation approximately 0.5 for both players.

3/31/24, 5:21 PM mp08_notebook

Extra Credit: Sequential Games

For extra credit, you can try to propose a strategy that will accumulate positive rewards, in a series of 16 sequential games of Prisoner’s dilemma, against the 16 possible pure-strategy opponents.

Rather than creating a function for this, all you need to do is create a matrix to specify your strategy. The autograder will then play your strategy against 100 random opponents.

submitted.sequential_strategy

The strategy is the variable .

sequential_strategy[a,b]

is the probability that your player will perform action 1

on the next round of play if, during the previous round of play, the other player performed action a, and your player performed action b.

Examples:



If you want to always act uniformly at random, return [[0.5,0.5],[0.5,0.5]] If you want to always perform action 1, return [[1,1],[1,1]].



If you want to return the other player’s action (tit-for-tat), return [[0,0],[1,1]].



If you want to repeat your own previous move, return [[0,1],[0,1]].



If you want to repeat your last move with probability 0.8, and the other player’s last move

with probability 0.2, return [[0.0, 0.8],[0.2, 1.0]].

You will be scored by testing your strategy in 100 sequential games against each of the

[[0,0],[0,0] [[0,0],[0,1]

sixteen pure strategy opponents, i.e., and and … and

[[1,1],[1,1]]

If your average score, averaged across all sixteen opponents, is

above 0.2, then you pass.


In [24]: importlib.reload(submitted)

print(submitted.sequential_strategy)

[[0.5 0.5]

[0.5 0.5]]

As you can see, the default strategy is to cooperate with 50% probability, always,

regardless of what your player or your opponent did in the last round of game play.

The reward matrix is like Prisoner’s dilemma, except that each player earns a positive

score if the other player cooperates, and a negative score if the other player defects.

The rewards matrix looks like this, where the number before ∥ is the reward for player A,

the number after ∥ is the reward for player B:

3/31/24, 5:21 PM mp08_notebook

R = a = 0

b = 0

b = 1

−1∥−1

2∥−2

a = 1

−2∥2

1∥1

We can see how well this strategy does against random opponents by running

:

grade.py


In [25]: !python grade.py

You played 1600 games, against all 16 possible fixed-strategy opponents and you won an average of 0.101875 points per game

F….

====================================================================== FAIL: test_extra (test_extra.TestStep)

———————————————————————-

Traceback (most recent call last):

File “/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/mp0 8/src/tests/test_extra.py”, line 39, in test_extra

self.assertGreater(score/played,0.2,msg=’That score is not enough to get extra credit!’)

AssertionError: 0.101875 not greater than 0.2 : That score is not enough to get extra credit!

———————————————————————-

Ran 5 tests in 0.017s

FAILED (failures=1)

In order to get the extra credit, you just need to change

submitted.sequential_strategy

to a sequential strategy that will induce your

opponent to cooperate, on average, at least 20% more often than they defect. This is

grade.py

how will look if you succeed:


In [26]: !python grade.py

You played 1600 games, against all 16 possible fixed-strategy opponents and you won an average of 0.390625 points per game Congratulations! That score is enough for extra credit!

…..

———————————————————————-

Ran 5 tests in 0.014s

OK

Grade your homework

If you’ve reached this point, and all of the above sections work, then you’re ready to try

grading your homework! Before you submit it to Gradescope, try grading it on your own

3/31/24, 5:21 PM mp08_notebook

machine. This will run some visible test cases (which you can read in

tests/test_visible.py

), and compare the results to the solutions (which you can

solution.json

read in ).

The exclamation point (!) tells python to run the following as a shell command. Obviously you don’t need to run the code this way — this usage is here just to remind you that you can also, if you wish, run this command in a terminal window.

-j grade.py


The option tells to print out a complete JSON description, which sometimes has a little more information than the default printout.


3/31/24, 5:21 PM mp08_notebook

{

“tests”: [

{

“name”: “test_extra (test_extra.TestStep)”,

“score”: 10,

“max_score”: 10,

“status”: “passed”,

“output”: “You played 1600 games, against all 16 possible fixed-strategy opponents\nand you won an average of 0.390625 points per game\nCong ratulations! That score is enough for extra credit!\n”

},

{

“name”: “test_corrected_ascent (test_hidden.TestStep)”,

“score”: 25,

“max_score”: 25,

“status”: “passed”

},

{

“name”: “test_gradient_ascent (test_hidden.TestStep)”,

“score”: 25,

“max_score”: 25,

“status”: “passed”

},

{

“name”: “test_corrected_ascent (test_visible.TestStep)”,

“score”: 25,

“max_score”: 25,

“status”: “passed”

},

{

“name”: “test_gradient_ascent (test_visible.TestStep)”,

“score”: 25,

“max_score”: 25,

“status”: “passed”

}

],

“leaderboard”: [],

“visibility”: “visible”,

“execution_time”: “0.02”,

“score”: 110

}

grade.py

If your outputs look like the above, then go ahead and:

Submit to

MP08

on Gradescope to get credit for the main assignment

on Gradescope to get the extra credit

Submit to

MP08

Extra Credit



In [ ]:

https://courses.grainger.illinois.edu/ece448/sp2024/mp/mp08_notebook.html 14/14
