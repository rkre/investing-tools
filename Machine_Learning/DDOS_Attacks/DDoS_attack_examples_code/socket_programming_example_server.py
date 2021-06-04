# Socket programming example: server side config.
import socket
import sys



def connect_to_ip(hostname):
    print(f"\nConnect to {hostname}")
    try: 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        print ("Socket successfully created")
    except socket.error as err: 
        print ("socket creation failed with error %s" %(err))
    
    # default port for socket 
    port = 80
    
    try: 
        host_ip = socket.gethostbyname('www.google.com') 
    except socket.gaierror: 
    
        # this means could not resolve the host 
        print ("there was an error resolving the host")
        sys.exit() 
    
    # connecting to the server 
    s.connect((host_ip, port)) 
    
    print ("the socket has successfully connected to google") 

# From the server side, we create the socket to listening the port.
def socket_example_server():
    print("Start Server Socket")
    host = '127.0.0.1'
    port = 65525

    # AF_NET = ipv4
    # SOCK_STREAM is TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Associate the host, port with the socket
        print("Associate the host, port with the socket")
        s.bind((host, port))
        print(host,port,s.bind)
        # print("Set NonBlocking..")
        # socket.setblocking(0)

        # Listening and accept the connection request from the port
        print("Listening on the connection request from the port")
        s.listen(5)
        print("Socket is listening...")
        print("Accept the connection request from the port")
        conn, addr = s.accept()
        print ('Got connection from', addr )
  
        # # send a thank you message to the client. 
        # s.send(b"Thank you for connecting") 
  
        # # Close the connection with the client 
        # s.close() 
        print(s.listen, conn, addr)

        # Process the request and reply
        print("Process the request and reply")
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                conn.sendall("The server has received your data!")
    
    print("Server Socket Done!")
    return conn


    
