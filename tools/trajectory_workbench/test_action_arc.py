import unittest
import numpy as np
from tools.trajectory_workbench.action_plan import compile_action_plan

class ArcTests(unittest.TestCase):
    def test_world_x_tilt_with_independent_tool_roll(self):
        from scipy.spatial.transform import Rotation
        a=Rotation.from_euler('y',90,degrees=True)*Rotation.from_euler('x',90,degrees=True)
        b=Rotation.from_euler('x',45,degrees=True)*a*Rotation.from_euler('x',-90,degrees=True)
        left={'pos':[.3,.2,.85],'quat':[1,0,0,0],'opening':0.}
        right={'pos':[.4,-.2,.85],'quat':a.as_quat()[[3,0,1,2]].tolist(),'opening':0.}
        result=compile_action_plan({'start':{'left':left,'right':right},'actions':[{'id':'tilt','duration_frames':120,'left':left,'right':{**right,'quat':b.as_quat()[[3,0,1,2]].tolist()},'rotation_profile':{'right':{'world_x_deg':45,'tool_x_deg':-90}}}]})
        q=np.asarray(result['right_quat'][60]);axis=Rotation.from_quat(q[[1,2,3,0]]).apply([1,0,0])
        self.assertTrue(np.allclose(axis,[0,np.sin(np.pi/8),-np.cos(np.pi/8)],atol=1e-10))
    def test_empty_events_preserve_small_opening(self):
        state={'pos':[.4,0,.85],'quat':[1,0,0,0],'opening':0.}
        result=compile_action_plan({'start':{'left':state,'right':state},'gripper_events':[],
            'actions':[{'id':'small_release','duration_frames':120,'left':state,'right':{**state,'opening':.005}}]})
        self.assertEqual(result['right_opening'][-1],.005)
        self.assertEqual(result['left_opening'][-1],0.)
    def test_one_hand_bow_and_exact_endpoints(self):
        state={'pos':[.4,0,.85],'quat':[1,0,0,0],'opening':.02}
        request={'start':{'left':state,'right':state},'actions':[{'id':'arc','duration_frames':120,'left':state,'right':{**state,'pos':[.4,.2,.85]},'arc_height_m':{'right':.06}}]}
        result=compile_action_plan(request)
        self.assertTrue(result['accepted'])
        self.assertTrue(np.allclose(result['left_pos'],state['pos']))
        self.assertAlmostEqual(result['right_pos'][60][2],.91)
        self.assertEqual(result['right_pos'][-1],[.4,.2,.85])
        request['actions'][0]['arc_height_m']=-.01
        with self.assertRaises(ValueError):compile_action_plan(request)

if __name__=='__main__':unittest.main()
