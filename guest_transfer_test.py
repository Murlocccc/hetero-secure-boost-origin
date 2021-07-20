from federation.transfer_inst import TransferInstGuest

transfer_inst = TransferInstGuest(10086,2)

info = {'name': 'bupt'}
asd = {'name': 'beijing'}
ans = {
    1: info,
    2: asd,
}

# transfer_inst.send_data_to_hosts(ans, -1)

datas = transfer_inst.recv_data_from_hosts(-1)

print(datas)

