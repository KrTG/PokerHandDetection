import unittest
import data
import data_new

class TestCombinations(unittest.TestCase):
    def test(self):   
        c1 = data.get_combination([1,1,2,2,3])        
        self.assertEqual(c1, 'Flush, 5 high')
        c2 = data.get_combination([0,13,26,26,5])
        self.assertEqual(c2, 'Four of a kind, 2 high')
        c3 = data.get_combination([2,6,9,46,47])
        self.assertEqual(c3, 'High card B')
        c4 = data.get_combination([4,5,19,20,21])
        self.assertEqual(c4, 'Straight, 10 high')
        c5 = data.get_combination([12,11,10,9,8])
        self.assertEqual(c5, 'Royal flush')
        c6 = data.get_combination([5,5,18,19,20])
        self.assertEqual(c6, 'Three of a kind, 7 high')
        c7 = data.get_combination([4, 17, 32, 32, 1])
        self.assertEqual(c7, 'Two pairs: 8s and 6s')
        c8 = data.get_combination([8, 8, 21, 23, 23])
        self.assertEqual(c8, 'Full house, 10 high')
        c9 = data.get_combination([1,2,38,18,51])
        self.assertEqual(c9, 'Pair of As')
        c10 = data.get_combination([50,48,49,46,47])
        self.assertEqual(c10, 'Straight flush, K high')

class TestLeafs(unittest.TestCase):
    def test(self):
        paths = ["1\\", "1\\2", "1\\2\\3", "1\\2\\4", "1\\3", "1\\4", "1\\4\\2\\3"]
        true_leafs = ["1\\4\\2\\3", "1\\2\\3", "1\\2\\4", "1\\3"]

        given_leafs = data_new.get_leafs(paths)
        self.assertEqual(set(given_leafs), set(true_leafs))
            
if __name__ == "__main__":
    unittest.main()