import json
import timeit
from pytson import decodeTSON
import cProfile, pstats, io

json_file = "pytson/tests/test_data.json"
tson_file = "pytson/tests/test_data.tson"


def profile(fnc):
    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


j = open(json_file, "r")


def load_json():
    j.seek(0)
    json.load(j)


t = open(tson_file, "rb")


def load_tson():
    t.seek(0)
    decodeTSON(t)


n10_json = timeit.timeit(load_json, number=10)
n10_tson = timeit.timeit(load_tson, number=10)

print(f"Timing for load_json: {n10_json}")
print(f"Timing for load_tson: {n10_tson}")
