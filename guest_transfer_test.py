from federation.transfer_inst import TransferInstGuest
from computing.d_table import DTable

transfer_inst = TransferInstGuest(10086)

ans = DTable(False, [1,2,3,4,5])

transfer_inst.send_data_to_hosts(ans, -1)


