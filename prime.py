def is_prime(num):
    if num<2: #check for negative numbers
        return False
    elif not isinstance(num, int): # check for floating numbers
        return False
    elif num==2: # check for smallest prime
        return True
    elif num%2==0: #check for even numbers
        return False
    else:
        for i in range(2,int(num**0.5)+1):
            if num%i==0:
                return False
        return True
    
def repknapsackdp(W,weight,value): #knapsack problem with repetition
    n=len(weight)
    maxval=np.zeros(W+1)
    for a in range(1,W+1): # outer for loop for weight of the knapsack
        amaxval=0
        for i in range(0,n): # inner for loop for items
            if weight[i]<=a:
                ival=value[i]+maxval[a-weight[i]]
                if ival>amaxval:
                    amaxval=ival
        maxval[a]=amaxval
    return int(maxval[W])
