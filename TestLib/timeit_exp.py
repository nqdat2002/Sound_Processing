import timeit

def test():
    s = 'Hello world'
    s.endswith('d')


if __name__ == '__main__':
    t = timeit.Timer('test()', setup='from __main__ import test')
    num_of_repeat = 1000
    runs = t.repeat(repeat=num_of_repeat, number=1)
    print(t)
    # print('Fastest run of {3} repeats: {0}ms  Slowest: {1}ms  Average: {2}ms'.format(
    #     min(runs) * 1000, max(runs) * 1000, (sum(runs) / float(len(runs))) * 1000, num_of_repeat))