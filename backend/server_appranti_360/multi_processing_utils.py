from multiprocessing import Process, Value, Lock, Array, Queue
import time
import copy


class AtomicString(object):
    def __init__(self, max_length=1024):
        self.val = Array('c', max_length)
        self.lock = Lock()
        self.max_length = max_length

    def peek(self):
        with self.lock:
            return str(copy.deepcopy(self.val.value))[2:-1]

    def set_and_get(self, r):
        last_value, actual_value = None, bytes(r, 'utf-8')
        assert len(actual_value) <= self.max_length, f'Longueur maximale ({self.max_length}) dépassée: {len(r)}'
        with self.lock:
            last_value = copy.deepcopy(self.val.value)
            self.val.value = copy.deepcopy(actual_value)
        return str(last_value)[2:-1], str(actual_value)[2:-1]


class AtomicValue(object):
    def __init__(self, init_val=0, type_='i'):
        self.val = Value(type_, init_val)
        self.lock = Lock()  # Pourrais utiliser self.val.get_lock()

    def decrement_and_get(self, r=1):
        last_value, actual_value = None, None
        with self.lock:
            last_value = copy.deepcopy(self.val.value)
            self.val.value -= r
            actual_value = copy.deepcopy(self.val.value)
        return last_value, actual_value

    def increment_and_get(self, r=1):
        last_value, actual_value = None, None
        with self.lock:
            last_value = copy.deepcopy(self.val.value)
            self.val.value += r
            actual_value = copy.deepcopy(self.val.value)
        return last_value, actual_value

    def peek(self):
        # Peak for queue --> https://stackoverflow.com/questions/43088633/peek-of-multiprocessing-queue
        with self.lock:
            return copy.deepcopy(self.val.value)

    def set_and_get(self, r):
        last_value, actual_value = None, r
        with self.lock:
            last_value = copy.deepcopy(self.val.value)
            self.val.value = copy.deepcopy(r)
        return last_value, actual_value


class AtomicInteger(AtomicValue):
    def __init__(self, init_val=0):
        AtomicValue.__init__(self, init_val=init_val, type_='i')


class AtomicFloat(AtomicValue):
    def __init__(self, init_val=0.0):
        AtomicValue.__init__(self, init_val=init_val, type_='f')


class AtomicBoolean(AtomicValue):
    def __init__(self, init_val=False):
        AtomicValue.__init__(self, init_val=init_val, type_='b')


class Queue_DoNothing:

    def __init__(self):
        None

    def task_done(self):
        None

    def join(self):
        None

    def qsize(self):
        return 0

    def empty(self):
        None

    def full(self):
        None

    def put(self, item, block=True, timeout=None):
        None

    def get(self, block=True, timeout=None):
        None

    def put_nowait(self, item):
        None

    def get_nowait(self):
        return 0
