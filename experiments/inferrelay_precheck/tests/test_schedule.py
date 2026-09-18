import unittest

from experiments.inferrelay_precheck.e3 import schedule


class ScheduleTests(unittest.TestCase):
    def test_serial_and_double_buffer_known_solution(self):
        one=schedule([(2,3,10)]*3,[0]*3,buffers=1)
        two=schedule([(2,3,10)]*3,[0]*3,buffers=2)
        self.assertEqual(one['makespan_ms'],15)
        self.assertEqual(two['makespan_ms'],11)

    def test_resources_and_slot_lifetimes(self):
        for buffers in [1,2]:
            result=schedule([(2,3,10)]*4,[0,1,0,1],requests=3,cycles=2,buffers=buffers,
                            activation_bytes=4096,return_bytes=4)
            resource_events={}
            slot_last_read={}
            for e in result['events']:
                resource_events.setdefault(e['resource'],[]).append(e)
                if e['kind']=='weight_h2d':
                    node=e['resource'][-1]
                    key=(node,e['slot'])
                    self.assertGreaterEqual(e['start_ms'],slot_last_read.get(key,0))
                elif e['kind']=='block_group':
                    node=e['resource'][-1]
                    slot_last_read[(node,e['slot'])]=e['end_ms']
            for events in resource_events.values():
                ordered=sorted(events,key=lambda e:e['start_ms'])
                for a,b in zip(ordered,ordered[1:]):
                    self.assertGreaterEqual(b['start_ms'],a['end_ms'])
            self.assertEqual(result['wire_messages'],3*2*4)  # 3 crossings + token return

    def test_no_network_for_local_and_return_for_remote(self):
        self.assertEqual(schedule([(2,3,10)]*2,[0,0])['wire_messages'],0)
        self.assertEqual(schedule([(2,3,10)]*2,[0,1])['wire_messages'],2)

    def test_contiguous_remote_can_prefetch_second_buffer_before_input(self):
        result=schedule([(2,20,10),(2,3,10),(2,3,10)],[0,1,1],buffers=2)
        second_copy=next(e for e in result['events'] if e['kind']=='weight_h2d' and e['group']==2)
        first_compute=next(e for e in result['events'] if e['kind']=='block_group' and e['group']==1)
        self.assertLessEqual(second_copy['end_ms'],first_compute['start_ms'])

    def test_network_penalty_is_not_hidden_by_addition(self):
        fast=schedule([(2,3,10)]*4,[0,1,0,1],latency_us=0,activation_bytes=1_000_000,bandwidth_gbps=100)
        slow=schedule([(2,3,10)]*4,[0,1,0,1],latency_us=1000,activation_bytes=1_000_000,bandwidth_gbps=1)
        self.assertGreater(slow['makespan_ms'],fast['makespan_ms'])


if __name__=='__main__':
    unittest.main()
