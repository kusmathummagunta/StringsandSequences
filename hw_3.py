import time, random
import tracemalloc 
import pandas as pd
import matplotlib.pyplot as plt 

def globalAlignment(X, Y):
    m = len(X)
    n = len(Y)
    gap_score = -3
    
    # Initialize the matrix
    S = [[0 for j in range(n+1)] for i in range(m+1)]

    # Initialize the first row and column
    for i in range(m+1):
        S[i][0] = gap_score * i
    
    for j in range(n+1):
        S[0][j] = gap_score * j
    
    # Fill the matrix
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = S[i-1][j-1] + cost(X[i-1], Y[j-1])
            delete = S[i-1][j] + gap_score
            insert = S[i][j-1] + gap_score
            S[i][j] = max(match, delete, insert)
        
    return S


def cost(a, b):
    if a == b:
        return 2
    else:
        return -3

def generateString(n):
    alphabet = ['A', 'C', 'G', 'T']
    return ''.join(random.choice(alphabet) for i in range(n))


def main(N):
    start = time.time()
    tracemalloc.start()

    X = generateString(N)
    Y = generateString(N)
    Y_r = Y[::-1]

    n = len(X)
    m = len(Y)

    X_half = X[:n//2 + 1]
    X_remaining = X[n//2 + 1:]

    X_half_r = X_half[::-1]
    X_remaining_r = X_remaining[::-1]

    M = globalAlignment(X_half, Y)
    M_r = globalAlignment(X_remaining_r, Y_r)

    # Add all values of M[-1] and M_r[-1] and store it in M_sum
    M_sum = []
    for a, b in zip(M[-1], M_r[-1][::-1]):
        M_sum.append(a + b)

    max_value = max(M_sum)
    k_star = M_sum.index(max_value) # k* is max value's index in M_sum

    # We cut string Y at index k_star
    Y_cut = Y[:k_star+1]
    Y_remaining = Y[k_star+1:]

    M1 = globalAlignment(X_half, Y_cut)
    M2 = globalAlignment(X_remaining, Y_remaining)

    current , peak =  tracemalloc.get_traced_memory()   
    #print("Time taken:", time.time() - start)
    #print("Space taken:", peak )
    
    tracemalloc.stop()  
    
    return time.time() - start, peak


N_values = [100, 200, 400, 800, 1000]
iterations = 10

temptime = list()
tempspace = list()
avgtimelist = list()
avgspacelist = list()

for N in N_values:
    timeTaken = 0
    spaceTaken = 0
    print(f"N = {N}, iterations = {iterations}")
    temp1 = list()
    temp2 = list()
    
    temp1.append(N)
    temp2.append(N)
    for i in range(iterations):
        new1, new2 = main(N)
        timeTaken += new1
        spaceTaken += new2
        temp1.append(new1)
        temp2.append(new2)
        
    temptime.append(temp1)
    tempspace.append(temp2)
    
    avgtime = timeTaken/ iterations
    avgspace = spaceTaken / iterations
    
    avgtimelist.append(avgtime)
    avgspacelist.append(avgspace)
    
    

    print("\nAverage time taken for" , N ,":", timeTaken/ iterations)
    print("Average space taken for" ,N ,":",spaceTaken / iterations,"\n")
 

df=pd.DataFrame(temptime ,columns=["N","1","2","3","4","5","6","7","8","9","10"])
df1 = pd.DataFrame(tempspace,columns=["N","1","2","3","4","5","6","7","8","9","10"])
df.plot(x="N", y=["1","2","3","4","5","6","7","8","9","10"], kind="bar",title ="Timegraph",ylabel="Time")
df1.plot(x="N", y=["1","2","3","4","5","6","7","8","9","10"], kind="bar",title ="Spacegraph",ylabel="Space")
plt.plot(N_values,avgtimelist,'o-k', color='black')
plt.plot(N_values,avgspacelist,'o-k', color='red',label="space")
plt.show()
    

