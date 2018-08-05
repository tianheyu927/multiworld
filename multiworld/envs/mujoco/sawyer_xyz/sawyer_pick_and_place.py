from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


<<<<<<< HEAD
class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
=======
class SawyerPickPlaceEnv( SawyerXYZEnv):
>>>>>>> russell
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            
            

            obj_init_pos=(0, 0.6, 0.02),

<<<<<<< HEAD
            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
=======
            goals = [[0, 0.7, 0.02, 0.1]],

            
>>>>>>> russell
            goal_low=None,
            goal_high=None,

            hand_init_pos = (0, 0.4, 0.05),
            #hand_init_pos = (0, 0.5, 0.35) ,
            blockSize = 0.02,

            **kwargs
    ):
        self.quick_init(locals())
        
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
<<<<<<< HEAD
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.obj_init_pos = np.array(obj_init_pos)

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
=======
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

       


        self.max_path_length = 150

        self.goals = goals
        self.num_goals = len(goals)

        
>>>>>>> russell

        self.obj_init_pos = np.array(obj_init_pos)
        self.hand_init_pos = np.array(hand_init_pos)

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
<<<<<<< HEAD
            ('observation', self.hand_and_obj_space),
            ('desired_goal', self.hand_and_obj_space),
            ('achieved_goal', self.hand_and_obj_space),
=======
           
>>>>>>> russell
            ('state_observation', self.hand_and_obj_space),

            ('desired_goal', self.goal_space)
        ])

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
<<<<<<< HEAD
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info
=======
        pass
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1

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
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
       

        reward , reachRew, pickRew, placeRew , placingDist = self.compute_rewards(action, ob)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, { 'reachRew':reachRew,  'pickRew':pickRew, 'placeRew': placeRew, 'reward' : reward, 'placingDist': placingDist}


>>>>>>> russell

   
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

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
<<<<<<< HEAD
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (
                -1000
            )
=======
       
       
>>>>>>> russell

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = pos.copy()
        qvel[8:15] = 0
        self.set_state(qpos, qvel)


    def sample_goal(self):


        goal_idx = np.random.randint(0, self.num_goals)
    
        return self.goals[goal_idx]

    def reset_model(self):
        self._reset_hand()
        
        self._state_goal = self.sample_goal()


        self._set_goal_marker(self._state_goal)

        self._set_obj_xyz(self.obj_init_pos)

        self.curr_path_length = 0
        self.pickCompleted = False

        init_obj = self.obj_init_pos

        heightTarget , placingGoal = self._state_goal[3], self._state_goal[:3]


       

        self.maxPlacingDist = np.linalg.norm(np.array([init_obj[0], init_obj[1], heightTarget]) - np.array(placingGoal)) + heightTarget
        #Can try changing this

        return self._get_obs()

    def _reset_hand(self):


        
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
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
        graspRew = -graspDist

        placingDist = np.linalg.norm(objPos - placingGoal)
       
        


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
                return 10*heightTarget
       
            elif (objPos[2]> (self.blockSize + 0.005)) and (graspDist < 0.1):
                
                return 10* min(heightTarget, objPos[2])
         
            else:
                return 0

        def placeReward():

          
            c1 = 100 ; c2 = 0.01 ; c3 = 0.001
            if self.pickCompleted and (graspDist < 0.1) and not(objDropped()):


                placeRew = 100*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))

               
                placeRew = max(placeRew,0)
           

                return [placeRew , placingDist]

            else:
                return [0 , placingDist]


      

        pickRew = pickReward()
        placeRew , placingDist = placeReward()

      
        assert ((placeRew >=0) and (pickRew>=0))
        reward = graspRew + pickRew + placeRew

        return [reward, graspRew, pickRew, placeRew, placingDist] 
     

   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

   
