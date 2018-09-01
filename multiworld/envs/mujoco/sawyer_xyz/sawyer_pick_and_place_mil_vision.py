from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from rllab.misc.overrides import overrides
from scipy.spatial.distance import cdist
from PIL import Image


class SawyerPickPlaceMILVisionEnv( SawyerXYZEnv):
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
            include_distractors=False,
            distr_init_pos=None,
            random_reset=True,

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

        self.max_path_length = 200
        self.obj_init_pos = np.array(obj_init_pos)
        self.hand_init_pos = np.array(hand_init_pos)
        # use 0.02 for now
        self.blockSize = blockSize

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.include_goal = kwargs.get('include_goal', False)
        if self.include_goal:
            self.hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            self.hand_and_obj_space = Box(
                np.hstack((self.hand_low, obj_low)),
                np.hstack((self.hand_high, obj_high)),
            )
        
        self.goal_space = Box(goal_low, goal_high)

        self.observation_space = Dict([
           
            ('state_observation', self.hand_and_obj_space),
            ('desired_goal', self.goal_space),
        ])
        
        self.random_reset = random_reset
        self.random_hand_init_pos = kwargs.get('random_hand_init_pos', True)
        self.hand_pos_is_init = kwargs.get('hand_pos_is_init', True)
        self.include_distractors = include_distractors
        self.distr_init_pos = distr_init_pos
        if include_distractors:
            self.n_distractors = kwargs.get('n_distractors', 4)

    # @property
    # def model_name(self):
    #     return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')6
    def viewer_setup(self):
        # pass
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.2
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0.3#0.3
        # self.viewer.cam.lookat[1] = 0.7#0.7
        # self.viewer.cam.lookat[2] = 0.4
        # self.viewer.cam.distance = 0.6#0.6
        # self.viewer.cam.elevation = -30#-30
        # self.viewer.cam.azimuth = 180
        # self.viewer.cam.trackbodyid = -1
        self.viewer.opengl_context.set_buffer_size(500, 500)

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
        
        # update _state_goal in case it moved
        self._state_goal = self.data.get_body_xpos('goal').copy()
        self._state_goal[-1] = 0.15# stay above the box
        self._state_goal = np.concatenate((self._state_goal, [0.15]))


        reward , reachRew, reachDist, pickRew, placeRew , placingDist, handGoalDist, actRew, distrRew = self.compute_rewards(action, ob)
        self.curr_path_length +=1

       
        #info = self._get_info()

        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        objPos = self.get_body_com("obj")
        fingerCOM = (rightFinger + leftFinger)/2
        dropDistx = np.abs(objPos[0] - self._state_goal[0])
        dropDisty = np.abs(objPos[1] - self._state_goal[1])
        # print("obj pos is", objPos)
        # print("goal pos is", self.data.get_body_xpos('goal').copy())
        # print("hand init pos is", self.real_hand_init_pos)
        # print("right finger pos is", rightFinger)
        # print("left finger pos is", leftFinger)
        # print("ee pos is", fingerCOM)
        # print('placing dist is', placingDist)
        # print('placing dist xy is', np.linalg.norm(objPos[:-1] - self._state_goal[:2]))
        # print('is picking completed?', self.pickCompleted)
        # print('grasp dist is', reachDist)
        # print('grasp z dist is', np.linalg.norm(fingerCOM[-1] - self.real_hand_init_pos[-1]))
        # print('grasp xy dist is', np.linalg.norm(fingerCOM[:-1] - objPos[:-1]))
        # print('placing reward is', placeRew)
        # print('pick reward is', pickRew)
        # print('act reward is', actRew)
        # print('total reward is', reward)
        return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'reward' : reward, 'placingDist': placingDist, 'handGoalDist': handGoalDist, 'dropDistx': dropDistx, 'dropDisty': dropDisty}#, 'actRew': actRew, 'distrRew': distrRew}



   
    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        g = self._state_goal[:3]
      
        flat_obs = np.concatenate((e, b))
        if self.include_goal:
            flat_obs = np.concatenate((flat_obs, g))

        return dict(
            
            desired_goal=self._state_goal[:3],
            
            state_observation=flat_obs,
            
        )
    
    def get_current_image_obs(self):
        e = self.get_endeff_pos()
        img = self.render(mode='rgb_array')
        # img = img[2400:-50, :2100, :]
        # img = img[150:, :, :]
        # img = img[200:, :, :]
        img = img[100:, :, :]
        pil_image = Image.fromarray(img, 'RGB')
        # img = np.array(pil_image.resize((160,140), Image.ANTIALIAS))
        # img = np.array(pil_image.resize((180,110), Image.ANTIALIAS))
        img = np.array(pil_image.resize((160,128), Image.ANTIALIAS))
        return img, dict(
            
            desired_goal=self._state_goal[:3],
            
            state_observation=e,
            
        )

    def _get_info(self):
        pass
    
    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()
        # return self.model.body_pos[self.model.body_name2id('obj')].copy()
    
    def get_distr_pos(self):
        assert self.include_distractors
        return [self.data.get_body_xpos('distractor_%d' % i).copy() for i in range(self.n_distractors)]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:16] = 0
        self.set_state(qpos, qvel)
    
    def _set_distr_xyz(self, poses):
        assert self.include_distractors
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        for i in range(self.n_distractors):
            qpos[16+7*i:19+7*i] = poses[i].copy()
            qvel[16+7*i:19+7*i] = 0
        self.set_state(qpos, qvel)


    def reset_model(self):
        self._state_goal = self.model.body_pos[self.model.body_name2id('goal')].copy()
        self._state_goal[-1] = 0.15 #0.18# stay above the box
        self._state_goal = np.concatenate((self._state_goal, [0.15])) #0.2
        self._reset_hand()
        
        if not self.random_reset:
            obj_pos = self.obj_init_pos
        else:
            obj_pos = np.array([np.random.uniform(low=-0.2, high=0.2), 
                                np.random.uniform(low=0.45, high=0.8), 
                                self.get_obj_pos()[-1]])
        
        # obj_pos = np.array([0, 0.9, 0.02])

        self._set_obj_xyz(obj_pos)
        if self.include_distractors:
            if not self.random_reset and self.distr_init_pos is not None:
                distr_poses = self.distr_init_pos
            else:
                distr_poses = np.array([obj_pos] + [np.array([np.random.uniform(low=-0.2, high=0.2), 
                                            np.random.uniform(low=0.45, high=0.8), 
                                            self.get_obj_pos()[-1]]) for _ in range(self.n_distractors)])
                while min(cdist(distr_poses, distr_poses)[cdist(distr_poses, distr_poses) > 0]) <= 0.12:
                    distr_poses = np.array([obj_pos] + [np.array([np.random.uniform(low=-0.2, high=0.2), 
                                            np.random.uniform(low=0.45, high=0.8), 
                                            self.get_obj_pos()[-1]]) for _ in range(self.n_distractors)])
                distr_poses = list(distr_poses)
                distr_poses.pop(0)
            self._set_distr_xyz(distr_poses)

        self.curr_path_length = 0
        self.pickCompleted = False
        self.placeCompleted = False

        init_obj = obj_pos

        heightTarget , placingGoal = self._state_goal[3], self._state_goal[:3]
        self.maxPlacingDist = np.linalg.norm(np.array([init_obj[0], init_obj[1], heightTarget]) - np.array(placingGoal)) + heightTarget
        self.maxPlacingDistxy = np.linalg.norm(np.array([init_obj[0], init_obj[1]]) - np.array(placingGoal[:-1]))
        #Can try changing this

        return self._get_obs()

    def _reset_hand(self):
        if self.random_hand_init_pos:
            if np.random.random() > 0.5:
                hand_pos = self.hand_init_pos
            else:
                hand_pos = self._state_goal[:3] + 0.05*(np.random.random(3) - 0.5)
                hand_pos[-1] += 0.05
        else:
            if self.hand_pos_is_init:
                hand_pos = self.hand_init_pos
            else:
                hand_pos = self._state_goal[:3] + 0.05*(np.random.random(3) - 0.5)
                hand_pos[-1] += 0.05
        # print('hand pos is', hand_pos)
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
        # print('left finger is', leftFinger)
        # print('right finger is', rightFinger)


        graspDist = np.linalg.norm(objPos - fingerCOM)
        # graspRew = -graspDist
    
        placingDist = np.linalg.norm(objPos - placingGoal)
        placingDistxy = np.linalg.norm(objPos[:-1] - placingGoal[:-1])
        handGoalDist = np.linalg.norm(fingerCOM - placingGoal)
        actRew = -np.linalg.norm(actions[:-1])
        distrRew = 0.0
        if self.include_distractors:
            distr_poses = self.get_distr_pos()
            for i in range(self.n_distractors):
                distrRew += 0.001*np.linalg.norm(fingerCOM[:-1] - distr_poses[i][:-1])/4
        def reachReward():
            graspDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.real_hand_init_pos[-1])
            if graspDistxy < 0.02: #0.02
                graspRew = -graspDist
            else:
                graspRew =  -graspDistxy - zRew
            #incentive to close fingers when graspDist is small
            if graspDist < 0.05:
                graspRew = -graspDist + max(actions[-1],0)/50 #50
            # graspRew += 0.1 * actRew
            return graspRew , graspDist

        def pickCompletionCriteria():

            tolerance = 0.01

            if objPos[2] >= (heightTarget - tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def placeCompletionCriteria():
           tolerance = 0.02
           if objPos[2] <= self.blockSize + tolerance and placingDistxy < 0.05:
               return True
           else:
               return False
        
        if placeCompletionCriteria():
            self.placeCompleted = True
        


        def objDropped():

            return (objPos[2] < (self.blockSize + 0.005)) and (placingDist >0.02) and (graspDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits
        def pickReward():
            # if objPos[2] > heightTarget + 0.05: #0.01
            #     return -1000*(objPos[2] - heightTarget)# + actRew
            if self.placeCompleted or (self.pickCompleted and not(objDropped())):
                # return 100*heightTarget
                # return 1000*heightTarget + actRew
                return 100*heightTarget# - 200*max(objPos[2] - heightTarget - 0.03, 0)# + 0.5*actRew
       
            elif (objPos[2]> (self.blockSize + 0.005)) and (graspDist < 0.1):
                
                # return 100* min(heightTarget, objPos[2])
                # return 100* min(heightTarget, objPos[2]) + actRew
                return 100*min(heightTarget, objPos[2])# + 0.5*actRew
         
            else:
                return 0

        def placeReward():

          
            c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            placeRewxy = 1000*(self.maxPlacingDistxy - placingDistxy) + c1*(np.exp(-(placingDistxy**2)/c2) + np.exp(-(placingDistxy**2)/c3))
            placeRew = max(placeRew,0)
            placeRewxy = max(placeRewxy, 0)
            # drop the object after reaching the target
            # if self.placeCompleted:
            #     return [1000 - 1000*min(1, actions[-1]), placingDist]
            # elif self.pickCompleted and (graspDist < 0.1) and not(objDropped()):
            #     if placingDistxy < 0.05:
            #         return [1000 - 1000*min(1, actions[-1]) + placeRewxy, placingDist]
            if self.placeCompleted:
                return [-200*actions[-1] + placeRew, placingDist]
            elif self.pickCompleted and (graspDist < 0.1) and not(objDropped()):
                if placingDistxy < 0.05: #0.05
                    return [-200*actions[-1] + placeRew, placingDist]
                else:
                    return [placeRew, placingDist]
            else:
                return [0 , placingDist]


      
        reachRew, reachDist = reachReward()
        pickRew = pickReward()
        placeRew , placingDist = placeReward()

        # assert (placeRew >= 0 and pickRew>=0)
        # assert (placeRew >= 0)
        reward = reachRew + pickRew + placeRew + distrRew

        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, handGoalDist, actRew, distrRew] 
     

   

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics
        
    def get_param_values(self):
        return None
        
    @overrides
    def log_diagnostics(self, paths):
        pass
    
    #required by rllab parallel sampler
    def terminate(self):
        """
        Clean up operation,
        """
        pass
   
