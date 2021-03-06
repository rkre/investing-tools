# Socket programming example: client side config.
import socket


# From the client side, we create the socket to sent the request
def socket_example_client():
    host = '127.0.0.1'
    port = 65525

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connect with the destination port and host
        s.connect((host, port))

        # Send out request data
        s.sendall(b'Hello')

        # Capture the reply information from the server
        data = s.recv(1024)

    print('Received the data!', repr(data))
