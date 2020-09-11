# CS229 Fall 2018 Problem Sets


## Explanation

1. In the spirit of the Stanford honour code, full solutions are not provided in
   this repository. Rather, written problem *solutions* will contain important
   observations to sketch out the right direction - the rest is *just* algebra.
2. Working examples of each coding problem and their respective outputs will be included. However, some coding solution make use of modified algorithms. An example of this can be seen in `stanford_cs299/ps1/src/p01b_logreg.py`

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

        
            
