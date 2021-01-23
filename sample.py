
def test(param):
    return param

def bizbuz(value):

    if value % 2:
        return True
    else:
        return False

if __name__ == '__main__':
    print('hello world')
    print(test('from function'))
    print(test(2))

    print('ganjil' if bizbuz(4) else 'genap')

    temp_list = [1,2,3,4,5,6,7]
    temp_list.map(x -> x*2)
    lambda(x : x*2)
    print([x*2 for x in temp_list])
