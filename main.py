#-----------------LETRA A)-------------------

import numpy as np
import random


def jacobi(A, b, e: float) -> np.ndarray: 
    """
    Calcula o algoritmo de Jacobi por meio da sua lei de formação.
    Paramêtros:
    n = núm. de linhas e colunas de A;
    e = erro usado para determinar convergência;
    max_iter = máximo de iterações que o algoritmo vai realizar
    """
    A = np.array(A)
    b = np.array(b)
    n = len(A)

    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    cont_iter = 0

    while True:
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        for j in range(n):
            soma = sum(A[j][k] * x0[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - soma)*(1/A[j][j])

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {cont_iter} iterações")
            break
        
        if erros[(cont_iter)-1] - erros[(cont_iter)] < 0 and cont_iter > 0:
            break
        
        cont_iter += 1

    return erros
        


def jacobi_matricial(A, b, e:float):
    
    A = np.array(A)
    b = np.array(b)
    n = len(A)

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
    cont_iter = 0

    while True:
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        soma = b - (R @ x0)
        x_new = D_inv @ soma

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)


        if erro < e:
            print(f"Convergiu em {cont_iter} iterações")
            break

        if erros[(cont_iter)-1] - erros[(cont_iter)] < 0 and cont_iter > 0:
            break
        
        cont_iter += 1

    return erros



def gauss_seidel(A, b, e: float):
    """
    Calcula o algoritmo de Jacobi por meio da sua lei de formação.
    Paramêtros:
    n = núm. de linhas e colunas de A;
    e = erro usado para determinar convergência;
    max_iter = máximo de iterações que o algoritmo vai realizar
    """
    A = np.array(A)
    b = np.array(b)
    n = len(A)

    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    cont_iter = 0

    while True:
        x0 = x_new.copy() #não alterar o array de x_new
        
        for j in range(n):
            # atualiza enqt altera todo o x_new (principal diff pra gauss seidel)
            soma = sum(A[j][k] * x_new[k] for k in range(n) if k != j)
            x_new[j] = (b[j] - soma)*(1/A[j][j])

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {cont_iter} iterações")
            break

        if erros[(cont_iter)-1] - erros[(cont_iter)] < 0 and cont_iter > 0:
            print("Divergiu!")
            break
        
        cont_iter += 1

    return erros
    


def gauss_seidel_matricial(A, b, e:float):

    A = np.array(A)
    b = np.array(b)
    n = len(A)

    D = np.diag(A)
    L = np.tril(A, k = -1) # posso usar?
    U = np.triu(A, k = 1) # muda complexidade?
    DL = np.diag(D) + L
    # equação = (D + L) @ x_new = (b - U @ x0)

    erros = []
    x0 = np.zeros(n)
    x_new = np.zeros(n)
    cont_iter = 0

    while True:
        # atualiza depois de alterar todo o x_new (principal diff pra gauss seidel)
        x0 = x_new.copy() #não alterar o array de x_new

        x_new = np.linalg.solve(DL, b - U @ x0) 
        #explicar a diferenca desse pro linalg.inv (complexidade)

        erro = np.linalg.norm(x_new - x0)
        erros.append(erro)

        if erro < e:
            print(f"Convergiu em {cont_iter} iterações")
            break

        if erros[(cont_iter)-1] - erros[(cont_iter)] < 0 and cont_iter > 0:
            print("Divergiu!")
            break
        
        cont_iter += 1



    return erros



#-------------------------LETRA B)-------------------------

import matplotlib.pyplot as plt

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



def testes_diag_dominante(tamanhos) -> None:

    for n in tamanhos:
        A, b = gerar_sistema_diag_dominante(n)
        A_np = np.array(A)
        B_np = np.array(b)

        # gerar funçoẽs
        erros_j = jacobi(A, b, 1e-8)
        erros_j_mat = jacobi_matricial(A, b, 1e-8)
        erros_gs = gauss_seidel(A, b, 1e-8)
        erros_gs_mat = gauss_seidel_matricial(A, b, 1e-8)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Comparação de métodos — n={n}")

        # Jacobi vs Jacobi matricial
        axs[0, 0].plot(erros_j, label="Jacobi")
        axs[0, 0].plot(erros_j_mat, label="Jacobi Matricial")
        axs[0, 0].set_title("Jacobi vs Jacobi Matricial")
        axs[0, 0].set_xlabel("Iterações")
        axs[0, 0].set_ylabel("Erro")
        axs[0, 0].legend()

        # Gauss-Seidel vs Gauss-Seidel matricial
        axs[0, 1].plot(erros_gs, label="Gauss-Seidel")
        axs[0, 1].plot(erros_gs_mat, label="Gauss-Seidel Matricial")
        axs[0, 1].set_title("GS vs GS Matricial")
        axs[0, 1].set_xlabel("Iterações")
        axs[0, 1].set_ylabel("Erro")
        axs[0, 1].legend()

        # Jacobi vs Gauss-Seidel
        axs[1, 0].plot(erros_j, label="Jacobi")
        axs[1, 0].plot(erros_gs, label="Gauss-Seidel")
        axs[1, 0].set_title("Jacobi vs Gauss-Seidel")
        axs[1, 0].set_xlabel("Iterações")
        axs[1, 0].set_ylabel("Erro")
        axs[1, 0].legend()

        # Jacobi matricial vs Gauss-Seidel matricial
        axs[1, 1].plot(erros_j_mat, label="Jacobi Matricial")
        axs[1, 1].plot(erros_gs_mat, label="GS Matricial")
        axs[1, 1].set_title("Jacobi Mat. vs GS Mat.")
        axs[1, 1].set_xlabel("Iterações")
        axs[1, 1].set_ylabel("Erro")
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()



tamanhos = [2, 3, 10, 50]

testes_diag_dominante(tamanhos)


def gerar_sistema(n):
    # gera x
    x = [random.uniform(1, 5) for _ in range(n)]

    # monta matriz tridiagonal
    A = np.random.rand(n,n)

    # calcula b = Ax
    b = [0]*n
    for i in range(n):
        for j in range(n):
            b[i] += A[i][j]*x[j]

    return A, b



def testes_normal(tamanhos) -> None:

    for n in tamanhos:
        A, b = gerar_sistema(n)
        A_np = np.array(A)
        B_np = np.array(b)

        # gerar funçoẽs
        erros_j = jacobi(A, b, 1e-8)
        erros_j_mat = jacobi_matricial(A, b, 1e-8)
        erros_gs = gauss_seidel(A, b, 1e-8)
        erros_gs_mat = gauss_seidel_matricial(A, b, 1e-8)


testes_normal(tamanhos)