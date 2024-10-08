from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core_brevitas as core
from spinup.utils.logx import EpochLogger

import wandb


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device='cpu'):
        self.device = device
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=self.device)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=self.device)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

global_t = 0
def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        episodes_per_epoch=10, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_episode=100, 
        update_after=1, gradient_steps=50, num_test_episodes=1, ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        episodes_per_epoch (int): Number of episodes 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_episode (int): Number of episodes for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of episodes to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        gradient_steps (int): Number gradient updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1, so this should be always equal to ep_len

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        ep_len (int): Length of a single trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    wandb.init(
        project="test_simple",
        config={
            "seed": seed,
            "episodes_per_epoch": episodes_per_epoch,
            "epochs": epochs,
            "replay_size": replay_size,
            "gamma": gamma,
            "polyak": polyak,
            "lr": lr,
            "alpha": alpha,
            "batch_size": batch_size,
            "gradient_steps": gradient_steps,
            "save_freq": save_freq
        }
    )

    # Initialize the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, device, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)

    # Initialize alpha as a trainable parameter
    log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    target_entropy = torch.tensor(-np.prod(env.action_space.shape))

    # Optimizer for alpha
    alpha_optimizer = Adam([log_alpha], lr=lr/10)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data, alpha):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data, alpha):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info, logp_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Set up function for computing alpha loss
    def compute_loss_alpha(logp_pi):
        return -(log_alpha * (logp_pi + target_entropy).detach()).mean()

    def update(data, alpha):
        global global_t
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data, alpha)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)
        wandb.log({"LossQ": loss_q.item()}, step=global_t)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info, log_pi = compute_loss_pi(data, alpha)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        wandb.log({"LossPi": loss_pi.item()}, step=global_t)
        wandb.log({"LogPi": log_pi.cpu().detach().numpy()}, step=global_t)

        # Update alpha
        alpha_optimizer.zero_grad()
        loss_alpha = compute_loss_alpha(log_pi)
        loss_alpha.backward()
        alpha_optimizer.step()
    
        # Update alpha value
        alpha = log_alpha.exp().item()
    
        # Record alpha loss and value
        logger.store(LossAlpha=loss_alpha.item(), Alpha=alpha)
        wandb.log({"LossAlpha": loss_alpha.item(), "Alpha": alpha}, step=global_t)        
            
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
                
        return alpha

    def get_action(o, deterministic=False):
        ac.pi.eval()
        a = ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), 
                      deterministic)
        ac.pi.train()
        return a

    def test_agent():
        global global_t  
        for j in range(num_test_episodes):
            ep_ret = 0
            o, d = test_env.reset(), False
            o = torch.as_tensor(o, dtype=torch.float32, device=device).view(1, -1)
            for step in range(ep_len):
                global_t += 1
                # Take deterministic actions at test time 
                a = get_action(o, True)
                o2, r, d, _ = test_env.step(a.view(-1).cpu().numpy())
                ep_ret += r
                
    
                # Convert o2, r, d to tensors and move them to the appropriate device
                o2 = torch.as_tensor(o2, dtype=torch.float32, device=device).view(1, -1)
                r = torch.as_tensor(r, dtype=torch.float32, device=device)
                d = torch.as_tensor(d, dtype=torch.float32, device=device)
    
                o = o2
        
                # Log action
                wandb.log({"TestActions": a.cpu().numpy()}, step=global_t)
                
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            wandb.log({"TestEpRet": ep_ret, "EpLen": ep_len}, step=global_t)  
        
    def process_episode(o, ep_len): 
        ep_ret = 0
        acts, obs, rews = [], [], []
        
        obs.append(o)
        
        for step in range(ep_len):
            global global_t
            global_t += 1

            # Until the start_episode is reached, we randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if global_t > start_episode * ep_len:
                a = get_action(o)
            else:
                a = torch.as_tensor(env.action_space.sample(), dtype=torch.float32, device=device)

            # Step the env
            o2, r, d, _ = env.step(a.view(-1).cpu().numpy())
            ep_ret += r
            
            # Convert o2, r, d to tensors and move them to the appropriate device
            o2 = torch.as_tensor(o2, dtype=torch.float32, device=device).view(1, -1)
            r = torch.as_tensor(r, dtype=torch.float32, device=device)
            d = torch.as_tensor(d, dtype=torch.float32, device=device)

            # Log action
            wandb.log({"actions": a.cpu().numpy()}, step=global_t)
            
            # Populate the lists
            obs.append(o2)
            acts.append(a)
            rews.append(r)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

        # Log epoch return
        wandb.log({"EpRet": ep_ret, "EpLen": ep_len}, step=global_t)
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        # restart
        o = env.reset()
        o = torch.as_tensor(o, dtype=torch.float32, device=device).view(1, -1)        
        return obs, acts, rews

    # Prepare for interaction with environment
    total_epidodes = episodes_per_epoch * epochs
    start_time = time.time()
    o = env.reset()
    # Convert o to tensors and move to the appropriate device
    o = torch.as_tensor(o, dtype=torch.float32, device=device).view(1, -1)

    global global_t

    # Main loop: collect experience in env and update/log each epoch
    for episode_num in range(total_epidodes):
        
        # Process one episode (it returns the initial o)
        obs, acts, rews = process_episode(o, ep_len)
                
        # Store experience to replay buffer
        for step in range(ep_len-1):
            
            o = obs[step]
            o2 = obs[step+1]
            r = rews[step]
            a = acts[step]
     
            # We ignore the "done" signal as it comes from hitting the time
            # horizon (that is, it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = torch.tensor(False, dtype=torch.float32, device=device)
            
            replay_buffer.store(o, a, r, o2, d)


        # Update handling
        if episode_num >= update_after:
            for j in range(gradient_steps):
                batch = replay_buffer.sample_batch(batch_size)
                alpha = update(data=batch, alpha=alpha)

        # End of epoch handling
        if (episode_num+1) % episodes_per_epoch == 0:
            epoch = (episode_num+1) // episodes_per_epoch

#             # Save model
#             if (epoch % save_freq == 0) or (epoch == epochs):
#                 logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', global_t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
