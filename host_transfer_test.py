from federation.transfer_inst import TransferInstHost

transfer_inst = TransferInstHost('127.0.0.1', 10086)

info = transfer_inst.recv_data_from_guest()

print(info)