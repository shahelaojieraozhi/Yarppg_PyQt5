def test_args(first, *args):
    print('Required argument: ', first)
    print(type(args))
    for v in args:
        print('Optional argument: ', v)


test_args(1, 2, 3, 4)
