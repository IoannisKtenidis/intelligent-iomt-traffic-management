import os
import sys
import subprocess
import unittest

class TestIoMTSimulation(unittest.TestCase):
    def setUp(self):
        self.src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "core")
        
    def run_mini_sim(self, script_name):
        script_path = os.path.join(self.src_dir, script_name)
        self.assertTrue(os.path.exists(script_path), f"Script '{script_name}' not found at '{script_path}'.")
        
        # Run a 5-second mini-simulation with 2 nodes
        cmd = [
            sys.executable, script_path,
            "2",          # nrNodes
            "1000000",    # avgSendTime
            "4",          # experiment ID
            "5000",       # simtime (5 seconds)
            "0",          # full_collision flag (False)
            "1",          # scenario ID
            "3"           # k_intervals
        ]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res

    def test_lora_dir_aloha_execution(self):
        res = self.run_mini_sim("lora_dir_aloha.py")
        self.assertEqual(res.returncode, 0, f"lora_dir_aloha.py exited with error code {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("nrCollisions", res.stdout)
        self.assertIn("energy (in J)", res.stdout)

    def test_lora_dir_plain_lbt_execution(self):
        res = self.run_mini_sim("lora_dir_plain_lbt.py")
        self.assertEqual(res.returncode, 0, f"lora_dir_plain_lbt.py exited with error code {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("nrCollisions", res.stdout)

    def test_lora_dir_classifier_lbt_execution(self):
        res = self.run_mini_sim("lora_dir_classifier_lbt.py")
        self.assertEqual(res.returncode, 0, f"lora_dir_classifier_lbt.py exited with error code {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("nrCollisions", res.stdout)

    def test_lora_dir_hospital_scenario_execution(self):
        res = self.run_mini_sim("lora_dir_hospital_scenario.py")
        self.assertEqual(res.returncode, 0, f"lora_dir_hospital_scenario.py exited with error code {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("nrCollisions", res.stdout)


if __name__ == "__main__":
    unittest.main()
