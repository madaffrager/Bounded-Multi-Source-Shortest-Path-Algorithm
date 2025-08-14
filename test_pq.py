# file: test_pq.py

import unittest
from specialized_priority_queue import SpecializedPriorityQueue

class TestSpecializedPriorityQueue(unittest.TestCase):

    def setUp(self):
        self.pq = SpecializedPriorityQueue()

    def test_initialization(self):
        self.pq.INITIALIZE(M=None, B=100)
        self.assertEqual(len(self.pq.buckets), 7)
        self.assertIn(0, self.pq.buckets)
        self.assertIn(6, self.pq.buckets)
        print("\n✓ test_initialization passed")

    def test_bucketing_logic(self):
        self.assertEqual(self.pq._get_bucket_idx(1), 0)
        self.assertEqual(self.pq._get_bucket_idx(7), 2)
        self.assertEqual(self.pq._get_bucket_idx(8), 3)
        self.assertEqual(self.pq._get_bucket_idx(15), 3)
        self.assertEqual(self.pq._get_bucket_idx(16), 4)
        print("✓ test_bucketing_logic passed")

    def test_batch_and_lazy_updates(self):
        self.pq.INITIALIZE(M=None, B=50)
        updates = [('A', 10), ('B', 20)]
        self.pq.BATCHDECREASEKEY(updates)
        self.assertEqual(len(self.pq.pending_updates), 2)
        self.assertTrue(all(not bucket for bucket in self.pq.buckets.values()))
        self.assertFalse(self.pq.is_empty())
        self.assertEqual(len(self.pq.pending_updates), 0)
        self.assertIn((10, 'A'), self.pq.buckets[3])
        self.assertIn((20, 'B'), self.pq.buckets[4])
        print("✓ test_batch_and_lazy_updates passed")

    def test_extract_min_sequence(self):
        self.pq.INITIALIZE(M=None, B=200)
        self.pq.BATCHDECREASEKEY([('A', 90), ('B', 5), ('C', 35), ('D', 2)])

        _, dist1, v1 = self.pq.EXTRACT_MIN()
        self.assertEqual(v1, 'D')
        
        _, dist2, v2 = self.pq.EXTRACT_MIN()
        self.assertEqual(v2, 'B')
        
        self.pq.BATCHDECREASEKEY([('E', 1), ('F', 100)])
        
        _, dist3, v3 = self.pq.EXTRACT_MIN()
        self.assertEqual(v3, 'E')
        
        _, dist4, v4 = self.pq.EXTRACT_MIN()
        self.assertEqual(v4, 'C')
        
        _, dist5, v5 = self.pq.EXTRACT_MIN()
        self.assertEqual(v5, 'A')
        
        _, dist6, v6 = self.pq.EXTRACT_MIN()
        self.assertEqual(v6, 'F')
        
        self.assertTrue(self.pq.is_empty())
        print("✓ test_extract_min_sequence passed")


if __name__ == '__main__':
    unittest.main()