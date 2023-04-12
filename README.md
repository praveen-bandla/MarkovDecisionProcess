# An implementation of a Markov Decision Process


### Objective

The goal of this project is to generalize an implementation of the Markov Decision Process. This has involved a lot of reading and certainly has many many steps. The full desired scope of this project may not be completed at time of publishing. In the notebook uploaded, I will walk through my progress and results of some of the things built, along with a reflection

---

### Project Description

Markov Chains form the crux of one of the most significant components of Machine Learning - Reinforcement Learning. For my implementation of the Markov Decision Process, I decided to work towards generalizing an implementation of Markov Chains and see how far I could take it in terms of running simulations. A majority of the work here will be code building, as a lot of the analysis is contingent on me picking up from where I will leave off.

---

### Contents

1. `mdp_notebook`: a notebook with a walkthrough of the progression of my work
2. `mdp_implementation`: a python script containing all the necessary classes and functions

---

### Flow of things

Guided by: https://builtin.com/machine-learning/markov-decision-process

NB: Each includes section is not exhaustive, only covers some of the things

1. Markov Chain
    - Goal: To create a Markov Chain
    - Includes:
        - Transition Matrix processing
        - Validating formats
        - Graph capabilities

<br>

2. Markov Process
    - Goal: To simulate Markov Chain process over n time period
    - Includes:
        - For a given Markov Chain and time n, simulates Markov Chain
        - Analyzes the results of running the simulation

<br>


3. Markov Reward
    - Goal: To generate reward values for each state based on monetary gain
    - Includes:
        - A reward function (concave based on Expected Utility Theory)

<br>

4. Markov Reward Process (To be worked on)
    - Goal: For a given policy, generates expected total reward (utility)
    - Includes:
        - Integrates agency into the overall process
        - Calculates reward at each stage, and sum to total using Bellman's equation

<br>

5. Markov Decision Process (To be Worked on):
    - Goal: To identify the optimal policy using policy iteration that maximizes total reward
    - Includes:
        - Using reinforcement learning principles (specifically policy iteration) to optimize for the best policy

<br>


---

### Challenges

1. **Research and time**: There is so much to learn! A lot of my time was spent doing background reading on the matter before I could get to any implementation. I definitely enjoyed the time I expended studying the material and Math behind MDPs, though I did realize there was a tradeoff with the progress I made. Given that the intention of doing this project is to learn, this was not particularly bothering! 

<br>

2. **Graph implementation** : A large part of the study of Markov Chains is founded in graph theory. The transition graph, as it is called is thus a cruical component of the implementation. However, I found that `networkx` (which I found to the most useful of the ones I looked at) was more suited to visualizations rather than graphical operations. In a further expanded scope of this project, I hope to further investigate libraries that may exist, and make changes to the source code as tailored to my needs.
