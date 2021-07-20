import socket
import pickle

class TransferInst:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def send_data(cs, data):
        byteStream = pickle.dumps(data)
        length = len(byteStream)
        byteStream = bytes(f"{length:<16}", 'utf-8')+byteStream
        cs.sendall(byteStream)
    
    @staticmethod
    def recv_data(cs):
        msg = cs.recv(1024)
        length = int(msg[:16])
        full_msg = b''
        full_msg += msg[16:]
        nowsize = len(full_msg)
        while nowsize < length:
            more = cs.recv(length - nowsize)
            full_msg = full_msg + more
            nowsize += len(more)
        return pickle.loads(full_msg)


class TransferInstGuest(TransferInst):
    def __init__(self, port: int=12345, conn_num: int=1, max_conn_num: int=5) -> None:
        super().__init__()

        self.conns = []

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('', port))
        self.server_socket.listen(max_conn_num)

        for _ in range(conn_num):
            client_socket, _ = self.server_socket.accept()
            self.conns.append(client_socket)
    
    def send_data_to_hosts(self, data, index: int):
        if index == -1:  # 转发列表中有所有的连接
            conns = self.conns  
        elif index < len(self.conns) and index >= 0:  # 转发列表中只有一个连接
            conns = [self.conns[index]]
        else:
            raise ValueError('the index is out of range, there are {} conns, while the index given is {}'.format(len(self.conns), index))
        
        for conn in conns:
            self.send_data(conn, data)
    
    def recv_data_from_hosts(self, index):

        data = None

        if index == -1:  # 接收所有host发来的消息
            data = []
            for conn in self.conns:
                data.append(self.recv_data(conn))
        elif index < len(self.conns) and index >= 0:  # 接收某个host发来的消息
            data = self.recv_data(self.conns[index])
        else:
            raise ValueError('the index is out of range, there are {} conns, while the index given is {}'.format(len(self.conns), index))

        return data


class TransferInstHost(TransferInst):
    def __init__(self,ip: int, port: int) -> None:
        super().__init__()

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip, port))
    
    def send_data_to_guest(self, data):
        self.send_data(self.client_socket, data)
    
    def recv_data_from_guest(self):
        data = self.recv_data(self.client_socket)
        return data