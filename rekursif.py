# fibbonacci 1 1 2 3 5 8...
def fib(n):
    if n <= 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)



def fibdp(n, arr = None):

    arr = [-1] *(n+1) if arr is None else arr

    if arr[n] == -1:
        if n <= 1:
            arr[n] =  1
        else:
            arr[n] = fibdp(n-1, arr) + fibdp(n-2, arr)

    return arr[n]



if __name__ == '__main__':
    # O(2^n)
    print(fib(5))
    # O(n)
    print(fibdp(5))


