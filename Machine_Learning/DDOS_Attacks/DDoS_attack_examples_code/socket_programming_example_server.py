# Socket programming example: server side config.
import socket


# From the server side, we create the socket to listening the port.
def socket_example_server():
    host = '127.0.0.1'
    port = 65525

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Associate the host, port with the socket
        s.bind((host, port))

        # Listening and accept the connection request from the port
        s.listen()
        conn, addr = s.accept()

        # Process the request and reply
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall(b"The server has received your data!")
