import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout, redirect_stderr

from scripts import prepare_bc250, train_bc250
from replica import train


class TrainingLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("scripts", "data/style-v1-1024", "models/Qwen3.5-2B-Base", "runtime/transformers-5.19.0-cp312"):
            (self.root/name).mkdir(parents=True)
        (self.root/"scripts/train_lora.py").write_text("")
        (self.root/"requirements-bc250.txt").write_text("transformers==5.19.0\n")
        digest = hashlib.sha256((self.root/"requirements-bc250.txt").read_bytes()).hexdigest()
        (self.root/"runtime/transformers-5.19.0-cp312/overlay.json").write_text(json.dumps({"requirements_sha256": digest}))
        self.env = self.root/"training.env"
        self.env.write_text("PYTHONPATH=/reviewed/site-packages\n")
        self.profile = self.root/"profile.json"
        self.profile.write_text(json.dumps({"max_seq_length":1024,"epochs":3,"rank":16,"alpha":32,"schedule":"cosine","eval_subset":128}))

    def args(self, **kwargs):
        defaults = dict(root=self.root, env_file=self.env, profile=self.profile, model=None, dataset=None, output=None,
                        resume=None, preflight=False, smoke=False, dry_run=False, attention_backend=None)
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_command_uses_python_script_not_python_executable(self):
        command = train_bc250.make_command(self.args())
        self.assertEqual(command[0],"bc250-train")
        self.assertIn(str(self.root/"scripts/train_lora.py"),command)
        self.assertNotIn("python3",command)
        self.assertIn("--schedule",command)
        self.assertIn("cosine",command)
        self.assertIn("PYTHONPATH="+str(self.root/"runtime/transformers-5.19.0-cp312")+":/reviewed/site-packages",command)

    def test_attention_backend_override_and_math_default(self):
        self.assertNotIn("--attention-backend", train_bc250.make_command(self.args()))
        self.profile.write_text(json.dumps({"attention_backend": "tiled"}))
        for override, expected in ((None, "tiled"), ("math", "math"), ("tiled", "tiled")):
            command = train_bc250.make_command(self.args(attention_backend=override))
            self.assertEqual(command[command.index("--attention-backend") + 1], expected)

    def test_smoke_is_small_and_separate(self):
        command = train_bc250.make_command(self.args(smoke=True))
        self.assertEqual(command[command.index("--max-steps")+1],"2")
        self.assertEqual(command[command.index("--accumulation-steps")+1],"1")
        self.assertEqual(command[command.index("--eval-subset")+1],"2")
        self.assertIn("--longest-first",command)
        self.assertIn(str(self.root/"checkpoints/qwen35-style-smoke"),command)

    def test_preflight_and_resume_forward(self):
        resume = self.root/"checkpoint-00000050"
        command = train_bc250.make_command(self.args(preflight=True,resume=resume))
        self.assertIn("--preflight",command)
        self.assertEqual(command[command.index("--resume")+1],str(resume))

    def test_invalid_overlay_and_unknown_profile_fail(self):
        (self.root/"requirements-bc250.txt").write_text("different")
        with self.assertRaisesRegex(ValueError,"does not match"):
            train_bc250.make_command(self.args())
        (self.root/"requirements-bc250.txt").write_text("transformers==5.19.0\n")
        self.profile.write_text('{"unknown":1}')
        with self.assertRaisesRegex(ValueError,"Unknown profile"):
            train_bc250.make_command(self.args())

    def test_dry_run_does_not_invoke_launcher(self):
        with patch.object(train_bc250.subprocess,"Popen") as popen, redirect_stdout(io.StringIO()):
            code = train_bc250.main(["--root",str(self.root),"--env-file",str(self.env),"--profile",str(self.profile),"--dry-run"])
        self.assertEqual(code,0)
        popen.assert_not_called()

    def test_noninteractive_launch_refused(self):
        with patch.object(train_bc250.sys.stdin,"isatty",return_value=False), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_bc250.main(["--root",str(self.root),"--env-file",str(self.env),"--profile",str(self.profile)])

    def test_longest_first_training_order(self):
        rows = [{"input_ids":[1]*length} for length in (3,10,1,7)]
        order = train.training_order(rows,longest_first=True)
        self.assertEqual([len(rows[i]["input_ids"]) for i in order],[10,7,3,1])
        self.assertEqual(sorted(train.training_order(rows)),[0,1,2,3])

    def test_overlay_staged_publish_no_torch(self):
        target = self.root/"new-overlay"
        def install(command, **kwargs):
            stage = Path(command[command.index("--target")+1])
            (stage/"transformers/models/qwen3_5").mkdir(parents=True)
            (stage/"transformers/models/qwen3_5/modeling_qwen3_5.py").write_text("")
        distributions = [SimpleNamespace(metadata={"Name":"transformers"},version="5.19.0")]
        with patch.object(prepare_bc250,"ROOT",self.root), patch.object(prepare_bc250.subprocess,"run",side_effect=install) as run, patch.object(prepare_bc250.importlib.metadata,"distributions",return_value=distributions), redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_bc250.main(["--target",str(target)]),0)
            self.assertEqual(prepare_bc250.main(["--target",str(target)]),0)
        self.assertEqual(run.call_count,1)
        command = run.call_args.args[0]
        self.assertIn("--only-binary=:all:",command)
        self.assertIn("3.12",command)
        self.assertTrue((target/"overlay.json").exists())

    def test_overlay_rejects_torch_dependency(self):
        target = self.root/"bad-overlay"
        distributions = [SimpleNamespace(metadata={"Name":"torch"},version="stock")]
        with patch.object(prepare_bc250,"ROOT",self.root), patch.object(prepare_bc250.subprocess,"run"), patch.object(prepare_bc250.importlib.metadata,"distributions",return_value=distributions), self.assertRaisesRegex(RuntimeError,"Torch"):
            prepare_bc250.main(["--target",str(target)])
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".overlay-*")))


if __name__ == "__main__":
    unittest.main()
