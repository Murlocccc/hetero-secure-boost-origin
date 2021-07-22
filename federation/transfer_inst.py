import socket
import pickle
from ml.utils.logger import LOGGER

class TransferInst:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def send_data(cs, data):
        byteStream = pickle.dumps(data)
        length = len(byteStream)
        # LOGGER.debug('len of send_msg is {}'.format(length))

        # byteStream = bytes(f"{length:<16}", 'utf-8')+byteStream

        # cs.sendall(byteStream)

        cs.sendall(bytes(f"{length:<16}", 'utf-8'))

        cs.sendall(byteStream)
    
    @staticmethod
    def recv_data(cs):
        # msg = cs.recv(1024)
        # length = int(msg[:16])
        # # LOGGER.debug('len of recv_msg is {}'.format(length))
        # full_msg = b''
        # full_msg += msg[16:]
        # nowsize = len(full_msg)
        # while nowsize < length:
        #     more = cs.recv(length - nowsize)
        #     full_msg = full_msg + more
        #     nowsize += len(more)

        msg = cs.recv(16)
        length = int(msg[:16])
        # LOGGER.debug('len of recv_msg is {}'.format(length))
        full_msg = b''
        nowsize = len(full_msg)
        while nowsize < length:
            more = cs.recv(length - nowsize)
            full_msg = full_msg + more
            nowsize += len(more)

        # LOGGER.debug('true len of recv_msg is {}'.format(length))
        return pickle.loads(full_msg)


class TransferInstGuest(TransferInst):
    def __init__(self, port: int=12345, conn_num: int=1) -> None:
        super().__init__()

        self.conns = []

        max_conn_num = conn_num + 1

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('', port))
        self.server_socket.listen(max_conn_num)

        for _ in range(conn_num):
            client_socket, _ = self.server_socket.accept()
            self.conns.append(client_socket)
    
    def send_data_to_hosts(self, data, index: int):
        """
        send data to host according to the index

        Parameters
        ----------
        data
            object to send
        index: int
            the index of the host, -1 means the data will be send to all hosts

        """

        if index == -1:  # 转发列表中有所有的连接
            conns = self.conns  
        elif index < len(self.conns) and index >= 0:  # 转发列表中只有一个连接
            conns = [self.conns[index]]
        else:
            raise ValueError('the index is out of range, there are {} conns, while the index given is {}'.format(len(self.conns), index))
        
        for conn in conns:
            self.send_data(conn, data)
    
    def recv_data_from_hosts(self, index):
        """
        receive data from host according to the index

        Parameters
        ----------
        index: int
            the index of the host, -1 means the data will be received from all hosts

        Returns
        -------
        data
            a list of object received if index == -1 else a object

        """

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
    def __init__(self,ip: str='127.0.0.1', port: int=12345) -> None:
        super().__init__()

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip, port))
    
    def send_data_to_guest(self, data):
        """
        send data to guest

        Parameters
        ----------
        data
            the object to send

        """
        self.send_data(self.client_socket, data)
    
    def recv_data_from_guest(self):
        """
        receive data from host according to the index

        Returns
        -------
        data
            the object received

        """

        data = self.recv_data(self.client_socket)
        return data