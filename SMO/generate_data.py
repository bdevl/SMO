import sys

sys.path.insert(0, "..")
from SMO.factories import CaseFactory
import time


def main(name, reference=False):

    factory = CaseFactory.FromIdentifier(name, dict())
    print("Starting data generation ... ")
    t1 = time.time()

    if reference:
        raise DeprecationWarning
    else:
        factory.data(ForceRecompute=True)

    t2 = time.time()

    print("========================================")
    print("TOTAL RUNTIME: {} SECONDS".format(t2 - t1))
    print("========================================")


if __name__ == "__main__":

    # precompute dataset for various factory configurations, e.g.
    main("ChannelizedFlow64")
