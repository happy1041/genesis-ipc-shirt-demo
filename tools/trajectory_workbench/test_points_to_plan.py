import copy
import unittest
from tools.points_to_plan import convert


class PointConversionTest(unittest.TestCase):
    def setUp(self):
        self.plan=dict(start_frame=1939,actions=[dict(id=n,right=dict(pos=[.4,.1,z],opening=.022,quat=[1,0,0,0]),duration_frames=20) for n,z in [('approach',.9),('grasp',.8),('drop',.91)]])
        self.mapping=dict(bindings=[dict(hand='right',kind='grasp',anchor_action='grasp',tcp_offset_world_mm=[0,0,-10],translate_actions=['approach','grasp']),dict(hand='right',kind='placement',anchor_action='drop',tcp_offset_world_mm=[0,0,0],translate_actions=['drop'])])
        self.points=dict(schema='workbench-material-regrasp-v1',source_frame=1939,points=dict(right=dict(material_world_m=[.41,.12,.82])))

    def test_surface_offset_and_partial_selection(self):
        before=copy.deepcopy(self.plan);out=convert(self.plan,self.points,self.mapping)
        self.assertAlmostEqual(out['actions'][1]['right']['pos'][2],.81)
        self.assertAlmostEqual(out['actions'][0]['right']['pos'][2],.91)
        self.assertEqual(out['actions'][2],before['actions'][2])
        self.assertEqual(out['actions'][1]['right']['opening'],.022)
        self.assertEqual(self.plan,before)

    def test_wrong_frame_rejected(self):
        self.points['source_frame']=1940
        with self.assertRaises(ValueError):convert(self.plan,self.points,self.mapping)

    def test_ambiguous_double_translation_rejected(self):
        self.mapping['bindings'].append(copy.deepcopy(self.mapping['bindings'][0]))
        with self.assertRaises(ValueError):convert(self.plan,self.points,self.mapping)

    def test_drop_only_keeps_grasp(self):
        self.points['points']={};self.points['placement_targets_world_m']={'right':[.45,-.1,.95]}
        out=convert(self.plan,self.points,self.mapping)
        self.assertEqual(out['actions'][0],self.plan['actions'][0])
        self.assertEqual(out['actions'][2]['right']['pos'],[.45,-.1,.95])
