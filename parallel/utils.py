import warnings


class DummyFuture:
    def __init__(self, catch_exceptions, f, args, kwargs):

        self._catch_exceptions = catch_exceptions
        self._f = f
        self._args = args
        self._kwargs = kwargs
        self._results = None
        self._exception = None

    def compute(self):

        if self._results is None:
            try:
                self._results = self._f(*self._args, **self._kwargs)
            except Exception as e:
                self._exception = e

        if not self._catch_exceptions and self._exception is not None:
            raise self._exception

    def result(self):

        if self._results is None and self._exception is None:
            self.compute()

        if self._exception is not None:
            raise self._exception

        return self._results

    def done(self):
        return True


class DummyProcessPool(object):
    def __init__(self, MAXWORKERS=None, catch_exceptions=True):

        if MAXWORKERS is not None:
            warnings.warn(
                "MAXWORKERS argument supplied to Dummy Process Pool has no impact"
            )
        self._catch_exceptions = True

    def activate_exceptions(self):
        self._catch_exceptions = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def submit(self, f, *args, **kwargs):

        return DummyFuture(self._catch_exceptions, f, args, kwargs)

    def shutdown(self, *args, **kwargs):
        pass
