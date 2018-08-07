from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from rllab.misc.overrides import overrides


class SawyerPickPlaceMILEnv( SawyerXYZEnv):
    def __init__(
            self,
            xml_file=None,
            obj_low=None,
            obj_high=None,
            obj_init_pos=(0, 0.6, 0.02),
            goal_low=None,
            goal_high=None,
            hand_init_pos = (0, 0.4, 0.05),
            #hand_init_pos = (0, 0.5, 0.35) ,
            blockSize = 0.02,

            **kwargs
    ):
        if xml_file is None:
            xml_file = get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')
        self.quick_init(locals())
        
        SawyerXYZEnv.__init__(
            self,
            model_name=xml_file,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.max_path_length = 150
        self.obj_init_pos = np.array(obj_init_pos)
        self.hand_init_pos = np.array(hand_init_pos)
        # use 0.02 for now
        self.blockSize = blockSize

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )
        
        self.goal_space = Box(goal_low, goal_high)

        self.observation_space = Dict([
           
            ('state_observation', self.hand_and_obj_space),
            ('desired_goal', self.goal_space),
        ])
        
        self.random_hand_init_pos = kwargs.get('random_hand_init_pos', True)
        self.hand_pos_is_init = kwargs.get('hand_pos_is_init', True)

    # @property
    # def model_name(self):
    #     return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
        # pass
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.3#0
        self.viewer.cam.lookat[1] = 0.7#1.0
        self.viewer.cam.lookat[2] = 0.3#0.5
        self.viewer.cam.distance = 0.7#0.6
        self.viewer.cam.elevation = -35#-45
        self.viewer.cam.azimuth = 180#270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):

        #debug mode:

        # action[:3] = [0,0,-1]
        # self.do_simulation([1,-1])
     

        # print(action[-1])

        # if self.pickCompleted:


        #     self.set_xyz_action(action[:3], action_scale = 2/100)

        # else:
        self.set_xyz_action(action[:3])


        # print(action[-1])
        self.do_simulation([action[-1], -action[-1]])
        
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
       

        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_rewards(action, ob)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'reward' : reward, 'placingDist': placingDist}



   
    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
      
        flat_obs = np.concatenate((e, b))

        return dict(
            
            desired_goal=self._state_goal,
            
            state_observation=flat_obs,
            
        )

    def _get_info(self):
        pass
    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()
        # return self.model.body_pos[self.model.body_name2id('obj')].copy()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:16] = 0
        self.set_state(qpos, qvel)


    def reset_model(self):
        self._state_goal = self.model.body_pos[self.model.body_name2id('goal')].copy()
        self._state_goal[-1] = 0.18 # stay above the box
        self._state_goal = np.concatenate((self._state_goal, [self.model.body_pos[self.model.body_name2id('dragonball1')].copy()[-1] + 0.05]))
        self._reset_hand()

        obj_pos = np.array([np.random.uniform(low=-0.2, high=0.2), 
                            np.random.uniform(low=0.5, high=0.7), 
                            self.get_obj_pos()[-1]])

        self._set_obj_xyz(obj_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        init_obj = obj_pos

        heightTarget , placingGoal = self._state_goal[3], self._state_goal[:3]
        self.maxPlacingDist = np.linalg.norm(np.array([init_obj[0], init_obj[1], heightTarget]) - np.array(placingGoal)) + heightTarget
        #Can try changing this

        return self._get_obs()

    def _reset_hand(self):
        if self.random_hand_init_pos:
            if np.random.random() > 0.5:
                hand_pos = self.hand_init_pos
            else:
                hand_pos = self._state_goal[:3] + 0.02*(np.random.random(3) - 0.5)
        else:
            if self.hand_pos_is_init:
                hand_pos = self.hand_init_pos
            else:
                hand_pos = self._state_goal[:3] + 0.02*(np.random.random(3) - 0.5)
        self.real_hand_init_pos = hand_pos
        for _ in range(10):
            self.data.set_mocap_pos('mocap', hand_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            #self.do_simulation([-1,1], self.frame_skip)
            self.do_simulation(None, self.frame_skip)

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()





    def compute_rewards(self, actions, obs):
           
        state_obs = obs['state_observation']

        endEffPos , objPos = state_obs[0:3], state_obs[3:6]

        
       
        heightTarget = self._state_goal[3]
        placingGoal = self._state_goal[:3]

        
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        objPos = self.get_body_com("obj")
        fingerCOM = (rightFinger + leftFinger)/2


        graspDist = np.linalg.norm(objPos - fingerCOM)
        # graspRew = -graspDist

        placingDist = np.linalg.norm(objPos - placingGoal)
        
        def reachReward():
            graspDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.real_hand_init_pos[-1])
            if graspDistxy < 0.1:
                graspRew = -graspDist
            else:
                graspRew =  -graspDistxy - zRew
            #incentive to close fingers when graspDist is small
            if graspDist < 0.02:
                graspRew = -graspDist + max(actions[-1],0)/50
            return graspRew , graspDist

        def pickCompletionCriteria():

            tolerance = 0.01

            if objPos[2] >= (heightTarget - tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

       


        def objDropped():

            return (objPos[2] < (self.blockSize + 0.005)) and (placingDist >0.02) and (graspDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits


        def pickReward():
            

            if self.pickCompleted and not(objDropped()):
                return 100*heightTarget
       
            elif (objPos[2]> (self.blockSize + 0.005)) and (graspDist < 0.1):
                
                return 100* min(heightTarget, objPos[2])
         
            else:
                return 0

        def placeReward():

          
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if self.pickCompleted and (graspDist < 0.1) and not(objDropped()):


                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))

               
                placeRew = max(placeRew,0)
           

                return [placeRew , placingDist]

            else:
                return [0 , placingDist]


      
        reachRew, reachDist = reachReward()
        pickRew = pickReward()
        placeRew , placingDist = placeReward()

      
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew

        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist] 
     

   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics
        
    def get_param_values(self):
        return None
        
    @overrides
    def log_diagnostics(self, paths):
        pass
   
