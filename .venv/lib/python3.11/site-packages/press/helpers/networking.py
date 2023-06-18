from press.helpers.sysfs_info import NetDeviceInfo


def cidr2mask(prefix):
    return [(0xffffffff << (32 - prefix) >> i) & 0xff for i in [24, 16, 8, 0]]


def mask2cidr(mask):
    mask = list(map(int, mask.split('.')))
    count = 32
    for octet in reversed(mask):
        if octet:
            bstr = bin(octet)[2:].zfill(8)
            count -= (8 - len(bstr.strip('0')))
            break
        count -= 8
    return count


def get_network(cidr_ip):
    addr, cidr = cidr_ip.split('/')
    addr = addr.split('.')
    cidr = int(cidr)

    mask = cidr2mask(cidr)

    net = []
    for i in range(4):
        net.append(int(addr[i]) & mask[i])

    net = '.'.join(map(str, net))
    mask = '.'.join(map(str, mask))

    return net, mask


def get_device_by_dev(dev):
    if dev not in NetDeviceInfo.list_interfaces():
        return
    return NetDeviceInfo(dev)


def get_device_by_mac(mac):
    mac = mac.replace('-', ':')
    mac = mac.lower()
    for interface in NetDeviceInfo.list_interfaces():
        ndi = NetDeviceInfo(interface)
        if ndi.address and ndi.address.lower() == mac:
            return ndi


# get_device_by_pci
# get_device_by_smbios_name
# get_device_by_systemd_name
# get_device_by_etc....
