import socket
import struct
import threading
import unittest
from .wire import receive_into,send_json,recv_json,validate,MAX_HEADER


class ProtocolTests(unittest.TestCase):
    def test_fragmented_payload(self):
        a,b=socket.socketpair();a.settimeout(2);b.settimeout(2)
        try:
            def send():
                for part in [b'ab',b'c',b'def']: a.sendall(part)
            t=threading.Thread(target=send);t.start()
            dst=bytearray(6);receive_into(b,dst);t.join()
            self.assertEqual(dst,b'abcdef')
        finally: a.close();b.close()

    def test_eof_is_failure(self):
        a,b=socket.socketpair();a.settimeout(2);b.settimeout(2);a.sendall(b'a');a.close()
        try:
            with self.assertRaises(EOFError): receive_into(b,bytearray(2))
        finally: b.close()

    def test_bound_and_identity(self):
        a,b=socket.socketpair();a.settimeout(2);b.settimeout(2)
        try:
            a.sendall(struct.pack('!I',MAX_HEADER+1))
            with self.assertRaises(ValueError): recv_json(b)
            send_json(a,{'step_id':2})
            with self.assertRaises(ValueError): validate(recv_json(b),{'step_id':1})
            with self.assertRaises(ValueError): validate({'shape':[1,2,3],'dtype':'float16','bytes':11},{})
        finally: a.close();b.close()


if __name__=='__main__': unittest.main()
