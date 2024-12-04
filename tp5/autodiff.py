import sympy as sp


# Exercice 1: Dérivée finie.
def finite_derivative(f, a, h=0.0001):
    return (f(a + h) - f(a - h)) / (2 * h)


def f_square(x):
    return x ** 2


def backward_derivative(f, a, h=0.0001):
    return (f(a) - f(a - h)) / h


def forward_derivative(f, a, h=0.0001):
    return (f(a + h) - f(a)) / h


def ex1_test(a, a_square, threshold=0.001):
    print("Exercice 1: Dérivée finie.")
    assert abs(finite_derivative(f_square, a) - a_square) < threshold
    print(finite_derivative(f_square, a))
    assert abs(backward_derivative(f_square, a) - a_square) < threshold
    print(backward_derivative(f_square, a))
    assert abs(forward_derivative(f_square, a) - a_square) < threshold
    print(backward_derivative(f_square, a))
    print("Exercice 1: Success\n")


def ex1_supp():
    print("Question supplémentaire 1:")
    print("-> Un h plus petit donne des résultats plus précis, jusqu'à un certain point où les limites du processeur prennent le dessus et les résulstats deviennent moins précis.")

    print("Question supplémentaire 2:")
    print("-> La méthode des différences finies est préférable lorsque la fonction est plus complexe et que la dérivée n'est pas facile à calculer.")
    print("-> La méthode des différences finies est plus précise que la méthode de la dérivée finie, mais elle est plus coûteuse en termes de calculs.")
    print()


# Exercice 2: Gradient numérique.
def numerical_gradient(a, b, h=1e-5):
    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    return finite_derivative(lambda x: rosenbrock(x, b), a, h), finite_derivative(lambda y: rosenbrock(a, y), b, h)


def optimize_rosebrock_numerical(a_init, b_init, learning_rate=0.00001, num_iterations=1000):
    a = a_init
    b = b_init
    for i in range(num_iterations):
        grad_a, grad_b = numerical_gradient(a, b)
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
    return a, b


def ex2_test(a, b):
    print("Exercice 2: Gradient numérique.")
    print(numerical_gradient(a, b))
    print(optimize_rosebrock_numerical(a, b))
    print("Exercice 2: Success\n")


def ex2_supp():
    print("Question supplémentaire 3:")
    print("-> Un taux d'apprentissage plus grand accélère la convergence, mais peut aussi causer des oscillations (bonds trop grands) et des divergences.")

    print("Question supplémentaire 4:")
    print("-> Les gradients numériques sont moins précis que les gradients symboliques, ce qui peut entraîner des problèmes de convergence et de stabilité.")
    print()


# Exercice 3: Dérivée symbolique.
def symbolic_derivative(expr, var):
    return sp.diff(expr, var)


def ex3_test():
    print("Exercice 3: Dérivée symbolique.")
    x = sp.symbols('x')
    f = x ** 2
    print(symbolic_derivative(f, x))
    print("Exercice 3: Success\n")


def ex3_supp():
    print("Question supplémentaire 5:")
    print("-> Les avantages de la dérivation symbolique sont qu'elle est plus précise et plus rapide que la dérivation numérique.")
    print("-> Les inconvénients de la dérivation symbolique sont qu'elle est plus complexe et nécessite plus de ressources que la dérivation numérique.")

    print("Question supplémentaire 6:")
    print("-> Sympy gère les expressions plus complexes en les simplifiant et en les factorisant pour les rendre plus faciles à manipuler.")
    x = sp.symbols('x')
    print(f"Exemple: x**2 + 2*x + 1 => {sp.factor(x ** 2 + 2 * x + 1)}")
    print()


# Exercice 4: Gradient symbolique et optimisation.
def rosenbrock_symbolic_gradient():
    x, y = sp.symbols('x y')
    f = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return sp.diff(f, x), sp.diff(f, y)


def optimize_rosebrock_symbolic(a_init, b_init, learning_rate=0.00001, num_iterations=1000):
    a = a_init
    b = b_init
    grad_a, grad_b = rosenbrock_symbolic_gradient()
    for i in range(num_iterations):
        a -= learning_rate * grad_a.subs('x', a).subs('y', b)
        b -= learning_rate * grad_b.subs('x', a).subs('y', b)
    return a, b


def ex4_test(a, b):
    print("Exercice 4: Gradient symbolique et optimisation.")
    print(rosenbrock_symbolic_gradient())
    print(optimize_rosebrock_symbolic(a, b))
    print("Exercice 4: Success\n")


def ex4_supp():
    print("Question supplémentaire 7:")
    print("-> Les gradients symboliques sont plus efficaces que les gradients numériques, car ils sont plus précis.")

    print("Question supplémentaire 8:")
    print("-> La précision des gradients symboliques affecte la convergence et la stabilité de l'optimisation.")
    print("-> Des gradients symboliques plus précis permettent une convergence plus rapide et une meilleure stabilité.")
    print()


# Exercice 5: Comparaison entre dérivée finie et symbolique.
def compare_derivative(f, f_prime, a, h=0.0001):
    return abs(finite_derivative(f, a, h) - f_prime(a))


def ex5_test(a):
    print("Exercice 5: Comparaison entre dérivée finie et symbolique.")
    f = lambda x: x ** 3
    f_prime = lambda x: 3 * x ** 2
    print(compare_derivative(f, f_prime, a))
    print("Exercice 5: Success\n")


def ex5_supp():
    print("Question supplémentaire 9:")
    print("-> Les limites des dérivées finies en termes de précision sont que les résultats deviennent moins précis lorsque h est trop petit ou trop grand.")
    print("-> Les limites des dérivées finies en termes de performance sont que les calculs deviennent plus coûteux lorsque h est plus petit.")

    print("Question supplémentaire 10:")
    print("-> Il est préférable d'utiliser des dérivées symboliques lorsque la fonction est simple et que la dérivée est facile à calculer.")
    print("-> On la priorise quand la précision et la performance sont des facteurs importants.")
    print()


if __name__ == "__main__":
    ex1_test(2, 4)
    ex1_supp()

    ex2_test(2, 2)
    ex2_supp()

    ex3_test()
    ex3_supp()

    ex4_test(2, 2)
    ex4_supp()

    ex5_test(2)
    ex5_supp()
