# Basic introduction about multithreading programming

import threading


# The function we want to run in multiple threads.
def object_function(thread_number):
    print("Thread " + repr(thread_number) + ": start to run...")


# Run a single thread
def run_a_thread():
    # Start a separate thread
    t = threading.Thread(target=object_function, args=(1,))
    # Run the thread
    t.start()


# Run multiple threads
def run_threads():
    threads = list()
    # Start and run multiple threads
    for index in range(5):
        t = threading.Thread(target=object_function, args=(index,))
        threads.append(t)
        t.start()

    # Wait until all threads terminate
    for index, thread in enumerate(threads):
        thread.join()