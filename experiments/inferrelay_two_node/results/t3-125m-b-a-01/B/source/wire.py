"""Bounded framed control messages and direct contiguous tensor byte transfers."""
import json
import struct

MAX_HEADER = 16384
MAX_PAYLOAD = 16 * 1024**2


def receive_into(sock, view):
    view = memoryview(view).cast('B')
    offset = 0
    while offset < len(view):
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise EOFError('Peer closed connection')
        offset += n


def send_json(sock, data):
    raw = json.dumps(data, separators=(',', ':')).encode()
    if len(raw) > MAX_HEADER:
        raise ValueError('Oversized header')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def recv_json(sock):
    size = bytearray(4)
    receive_into(sock, size)
    n, = struct.unpack('!I', size)
    if not 0 < n <= MAX_HEADER:
        raise ValueError('Invalid header size')
    raw = bytearray(n)
    receive_into(sock, raw)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected object')
    return value


def validate(got, expected):
    for key, value in expected.items():
        if got.get(key) != value:
            raise ValueError(f'Protocol mismatch for {key}: {got.get(key)!r} != {value!r}')
    if 'shape' in got:
        shape = got['shape']
        if (got.get('dtype') != 'float16' or len(shape) != 3
                or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError('Invalid tensor descriptor')
        size = 2
        for n in shape:
            size *= n
        if size != got.get('bytes') or size > MAX_PAYLOAD:
            raise ValueError('Invalid tensor size')
