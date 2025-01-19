# Physics of Skill Learning!

This is the github repo for the paper "physics of skill learning" (TBA). 

Research question: Do skills learn in series or in parallel? It is probably a mixture of both, but how much do they learn in series vs in parallel?

![](https://github.com/user-attachments/assets/2d93381e-7fc4-4d52-9397-6182c1b16eaa)

Language models are demonstrating impressive skills in, e.g., coding and mathematics. Many tasks,
including language modeling, are complex composite tasks that can be decomposed into many
atomic skills. The learning dynamics of skills appear to be complex and intriguing:
Throughout training, a skill can be completely learned, partially learned, or not learned at all. Even for
learned skills, they could display quite diverse learning curves, including sudden jumps (grokking),
gradual improvements, or non-monotonic oscillations. Despite the diverse phenomenology observed
in real-world experiments, our intuitive understanding of them is quite limited. Intuitive understanding, or physics-style understanding, has the potential to bridge between theory (mathematics-like
understanding) and experiments (engineering-like understanding).

To gain some intuition about skill learning, we take physicists’ approach of abstraction and simplification (see illustration below): when trying to understand a cow in the wild, physicists would
make assumptions to simplify the subject matter. It is science but also art to determine the appropriate
level of abstraction and simplification. As Einstein famously put it, “Everything should be made as
simple as possible, but not simpler.” In the same philosophy, we will propose three models trading
off between reality and simplicity – the Geometry model, the Resource model and the Domino model.
Each of these models is able to capture some realistic aspects of rich skill dynamics.

<img width="1141" alt="Screenshot 2025-01-19 at 18 23 21" src="https://github.com/user-attachments/assets/198aa13e-2d28-4e47-b5c2-16268ae964aa" />

As a motivation, we start by making an observation called the Domino effect, which shows that
skills tend to learn sequentially, and notably, some skills start to learn right after other skills finish
learning. For example, when we train two independent sparse parity tasks (with frequencies $p_1 = 1$
and $p_2 = 0.1$) on a two-layer MLP using the Adam optimizer, the second task
starts to progress rapidly only after the first task finishes. Quantitatively, learning task 2 only takes roughly
two more times (instead of $p_1/p_2 = 10$ times that one would reasonably expect since the gradient signals differ by 10 times). In a more complicated setup, compositional
task dependency can also lead to the Domino effect. It is thus very intriguing to understand the
mechanisms underneath the Domino effect. Although the Domino effect serves as a good starting point, our ambitious goal is to understand skill
dynamics in general.

<img width="923" alt="Screenshot 2025-01-19 at 18 27 58" src="https://github.com/user-attachments/assets/2f018782-40fb-4339-b282-52be7e94f842" />

Good physics-like theories are inspired by experimental observations, and should be able to make predictions testable by new experiments. We stick to this philosophy by applying the toy models to many topics in deep learning,
including neural scaling laws, optimization, task dependency and modularity. Although these toy models are extremely simple, they are able to characterize key aspects of real-world learning dynamics.

<img width="1107" alt="Screenshot 2025-01-19 at 18 28 23" src="https://github.com/user-attachments/assets/6107730d-c560-449d-98bc-ecd48449da41" />

# Get started
We aim to make examples minimal and self-contained, so examples are coded in jupyter notebooks.
* Geometry model: [geometry_model.ipynb](https://github.com/KindXiaoming/physics_of_skill_learning/blob/master/geometry_model.ipynb)
* Resource model: [resource_model.ipynb](https://github.com/KindXiaoming/physics_of_skill_learning/blob/master/resource_model.ipynb)
* Domino model: [domino_model.ipynb](https://github.com/KindXiaoming/physics_of_skill_learning/blob/master/domino_model.ipynb)

To reproduce figures in the paper, see the folder [`./scripts`](https://github.com/KindXiaoming/physics_of_skill_learning/tree/master/scripts). File names start with "Figx_", indicating correspondence to which figure.

