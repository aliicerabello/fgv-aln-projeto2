#-----------------LETRA A)-------------------

import numpy as np
import random


def gerar_sistema_diag_dominante(n):
    # gera x
    x = [random.uniform(1, 5) for _ in range(n)] 
    
    # monta matriz densa
    A = [[0.0]*n for _ in range(n)]
    for i in range(n): # acessa as linhas de A
        for j in range(n): # acessa as cols de A
            if i != j:
                A[i][j] = random.uniform(1, 3)  # fora da diagonal
        
        # diag dominante
        soma_linha = sum(A[i][j] for j in range(n) if j != i)
        A[i][i] = soma_linha + random.uniform(1, 10) 

    # calcula b = Ax
    b = [0.0]*n 
    for i in range(n):
        for j in range(n):
            b[i] += A[i][j]*x[j]
    
    return A, b


def jacobi(n: int, e: float, max_iter: int) -> np.ndarray: 
    """
    Calcula o algoritmo de Jacobi por meio da sua lei de formação.
    Paramêtros:
    n = núm. de linhas e colunas de A;
    e = erro usado para determinar convergência;
    max_iter = máximo de iterações que o algoritmo vai realizar
    """
    A, b = gerar_sistema_diag_dominante(n)
    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    for i in range(max_iter):
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        for j in range(n):
            soma = sum(A[j][k] * x0[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - soma)*(1/A[j][j])

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {i+1} iterações")
            print(erros)
            break
        
    else:
        print("Não convergiu.")
        
jacobi(2, 1e-8, 20)


def jacobi_matricial(n:int, e:float, max_iter:int):
    A, b = gerar_sistema_diag_dominante(n)

    b = np.array(b)
    D = np.diag(A)
    L = np.tril(A, k = -1) # posso usar?
    U = np.triu(A, k = 1) # muda complexidade?
    D_inv_vals = 1/D #ambas sao 1 dimensional
    D_inv = np.diag(D_inv_vals)

    #equação = D @ x_new = b - R @ x0

    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    R = L + U

    for i in range(max_iter):
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        soma = b - (R @ x0)
        x_new = D_inv @ soma

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {i+1} iterações")
            print(erros)
            break

    else:
        print("Não convergiu.")


jacobi_matricial(2, 1e-8, 20)


def gauss_seidel(n: int, e: float, max_iter: int):
    """
    Calcula o algoritmo de Jacobi por meio da sua lei de formação.
    Paramêtros:
    n = núm. de linhas e colunas de A;
    e = erro usado para determinar convergência;
    max_iter = máximo de iterações que o algoritmo vai realizar
    """
    A, b = gerar_sistema_diag_dominante(n)
    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    for i in range(max_iter):
        x0 = x_new.copy() #não alterar o array de x_new
        
        for j in range(n):
            # atualiza enqt altera todo o x_new (principal diff pra gauss seidel)
            soma = sum(A[j][k] * x_new[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - soma)*(1/A[j][j])

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {i+1} iterações")
            print(erros)
            break

    else:
        print("Não convergiu.")
    
gauss_seidel(2, 1e-8, 100)


def gauss_seidel_matricial(n:int, e:float, max_iter:int):
    A, b = gerar_sistema_diag_dominante(n)

    b = np.array(b)
    D = np.diag(A)
    L = np.tril(A, k = -1) # posso usar?
    U = np.triu(A, k = 1) # muda complexidade?
    DL = np.diag(D) + L
    # equação = (D + L) @ x_new = (b - U @ x0)

    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)

    for i in range(max_iter):
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        x_new = np.linalg.solve(DL, b - U @ x0) 
        #explicar a diferenca desse pro linalg.inv (complexidade)

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {i+1} iterações")
            print(erros)
            break

    else:
        print("Não convergiu.")



gauss_seidel_matricial(2, 1e-8, 100)

