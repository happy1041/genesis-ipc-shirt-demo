import contextlib
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.run_scene527_reproduction import inside, main, read_checkpoint_archive


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, name='checkpoint.pkl', wrong_hash=False):
        blob=b'checkpoint test data'
        archive=self.root/'checkpoint.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            info=tarfile.TarInfo(name)
            info.size=len(blob)
            tar.addfile(info,io.BytesIO(blob))
        return dict(checkpoint_archive=archive.name,checkpoint_members={name:dict(
            bytes=len(blob),sha256='bad' if wrong_hash else hashlib.sha256(blob).hexdigest())})

    def test_verified_checkpoint_extraction(self):
        stage=self.archive()
        result=read_checkpoint_archive(self.root,stage,self.root/'input')
        self.assertEqual(result.read_bytes(),b'checkpoint test data')

    def test_checkpoint_hash_failure_prevents_extraction(self):
        stage=self.archive(wrong_hash=True)
        with self.assertRaises(ValueError):
            read_checkpoint_archive(self.root,stage,self.root/'input')
        self.assertFalse((self.root/'input').exists())

    def test_unsafe_archive_and_asset_paths_rejected(self):
        stage=self.archive('../checkpoint.pkl')
        with self.assertRaises(ValueError):
            read_checkpoint_archive(self.root,stage,self.root/'input')
        with self.assertRaises(ValueError):
            inside(self.root,'../outside')

    def test_restore_command_preserves_global_frame_numbering(self):
        profile=dict(cloth='shirt.obj',robot_urdf='robot.urdf',trajectory='commands.npz',
            preset='preset.args',stages=[
                dict(id='01_first',name='first',frame_range=[0,739],previous_checkpoint=None,args=['--frames','740']),
                dict(id='02_second',name='second',frame_range=[740,1135],previous_checkpoint='01_first',args=['--frames','1136'])])
        (self.root/'bundle.json').write_text(json.dumps(profile))
        (self.root/'preset.args').write_text('--frames\n4373\n')
        output=io.StringIO()
        args=['run','--bundle',str(self.root),'--mode','reference-boundaries',
              '--stage','second','--smoke-frames','2','--physics-only','--dry-run']
        with patch('sys.argv',args),patch('subprocess.check_output',return_value='/tmp/no_python_libs'),\
             patch('subprocess.run') as run,contextlib.redirect_stdout(output):
            main()
        run.assert_not_called()
        command=shlex.split(output.getvalue().strip())
        occurrences=[command[i+1] for i,x in enumerate(command) if x=='--frames']
        self.assertEqual(occurrences[-1],'742')
        self.assertIn('--allow-checkpoint-path-relocation',command)
        self.assertIn('--viewer',command)
        self.assertIn('--load-third-fold-checkpoint',command)


if __name__=='__main__':
    unittest.main()
