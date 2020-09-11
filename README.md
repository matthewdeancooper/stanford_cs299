# CS229 Fall 2018 Problem Sets


## Explanation

1. In the spirit of the Stanford honour code, full solutions are not provided in
   this repository. Rather, *solutions* to written problems contain important
   observations to sketch out the right direction - the rest is *just* algebra.
   This repository is designed to serve as a blueprint cf. a walk-through.
   Remember:
   
   > "What I cannot create, I do not understand." - Feynman

![Feynman's blackboard](https://sylviecerise.files.wordpress.com/2010/02/feynman-blackboard.jpg)

2. Working examples of each coding problem and their respective outputs are
   included. However, some coding solution make use of modified algorithms. An
   example of this can be seen in file `stanford_cs299/ps1/src/p01b_logreg.py`
   below, where the problem set requested option 2).

        class LogisticRegression(LinearModel):
            """Logistic regression with Newton's Method as the solver.

            There are multiple ways to formulate this solution:
            1) Maximise the log likelihood by solving l'(theta) = 0 via Newtons method
            2) Minimise the cost J(theta) by solving J'(theta) = 0 via Newtons method
            3) Use gradient ascent to maximise the log likelihood l(theta)
            4) Use gradient descent to minimise J(theta)

            Here, we implement 1) and calculate the Hessian and gradients accordingly.
            ...

            """

3. Where GitHub size limits permit, datasets have been included in the
   repository. However, this is not the case for the missing
   `stanford_cs229/ps4/data` directory, which is readily available
   [here](https://github.com/s-ai-kia/CS229_ML/blob/master/PSET/2018/ps4_v5_release.zip)
   in a zipped format.
4. The official setup requires Miniconda to be installed to initialise the
   `environment.yml` virtual environment. However, if you're not officially
   submitting code then I recommend keeping it simple (for Linux) with pip,
   venv, and pyenv.


        
            
