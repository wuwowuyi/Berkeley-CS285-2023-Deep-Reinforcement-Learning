from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: torch.Tensor) -> np.ndarray:
        """
        Compute the action for a single given observation. not for batched observations.
        """
        with torch.no_grad():
            #observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "vanilla", we can just leave the Q-values as-is (we only have one critic).
         - for double-Q, swap the critics' predictions (so each uses the other as the target).
         - for clip-Q, clip to the minimum of the two critics' predictions.

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size). 
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        # Implement the different backup strategies.
        if num_critic_networks > 1:
            if self.target_critic_backup_type == "doubleq":
                next_qs = torch.stack([next_qs[i] for i in range(num_critic_networks, -1, -1)])
            elif self.target_critic_backup_type == "min":
                next_qs = torch.min(next_qs, dim=0, keepdim=True)[0]
            else:
                # Default, we don't need to do anything.
                pass

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs.expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        # Compute target values
        # Important: we don't need gradients for target values!
        with torch.no_grad():
            # Sample from the actor
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)  # batch_shape=batch_size
            next_action: torch.Tensor = next_action_distribution.sample()  # shape=(batch_size, act_dim)

            # Compute the next Q-values for the sampled actions
            next_qs: torch.Tensor = self.target_critic(next_obs, next_action)  # shape=(num_critics, batch_size)

            # Handle Q-values from multiple different target critic networks (if necessary)
            # (For double-Q, clip-Q, etc.)
            next_qs: torch.Tensor = self.q_backup_strategy(next_qs)

            # next_qs = self.q_backup_strategy(next_qs)  # why do we need to run this again?

            assert next_qs.shape == (
                self.num_critic_networks,
                batch_size,
            ), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                # Add entropy bonus to the target values for SAC
                next_action_entropy = self.temperature * self.entropy(next_action_distribution)
                next_qs += next_action_entropy.expand(self.num_critic_networks, batch_size)

            # Compute the target Q-value, whose shape must be (num_critics, batch_size)
            # expand shape to (num_critics, batch_size)
            r = reward.expand((self.num_critic_networks, batch_size))
            d = (1 - done).expand((self.num_critic_networks, batch_size))
            target_values: torch.Tensor = r + self.discount * d * next_qs

        # Update the critic
        # Predict Q-values
        q_values = self.critic(obs, action)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape
        # Compute loss
        self.critic_optimizer.zero_grad()
        loss: torch.Tensor = self.critic_loss(q_values, target_values)
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution) -> torch.Tensor:
        """
        Compute the (approximate) entropy of the action distribution for each batch element.
        """
        # Compute the entropy of the action distribution.
        # Note: Think about whether to use .rsample() or .sample() here...
        num_samples = 100  # arbitrary number
        sampled_act = action_distribution.rsample((num_samples, ))  # shape=(num_samples, batch_size, act_dim)
        logp = action_distribution.log_prob(sampled_act)  # shape=(num_samples, batch_size)
        return torch.mean(-logp, dim=0)  # return's shape=(batch_size,)

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        # Generate an action distribution
        action_distribution: torch.distributions.Distribution = self.actor(obs)  # batch_shape=(batch_size,)

        with torch.no_grad():
            # draw num_actor_samples samples from the action distribution for each batch element
            action = action_distribution.sample((self.num_actor_samples,))
            assert action.shape == (
                self.num_actor_samples,
                batch_size,
                self.action_dim,
            ), action.shape

            # Compute Q-values for the current state-action pair
            q_values = self.critic(obs.expand((self.num_actor_samples, *obs)), action)
            assert q_values.shape == (
                self.num_critic_networks,
                self.num_actor_samples,
                batch_size,
            ), q_values.shape

            # Our best guess of the Q-values is the mean of the ensemble
            q_values = torch.mean(q_values, dim=0)  # shape becomes (num_actor_samples, batch_size)
            advantage = q_values

        # Do REINFORCE: calculate log-probs and use the Q-values
        log_probs = action_distribution.log_prob(action)  # shape is (num_actor_samples, batch_size)
        loss = -(log_probs * advantage).mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        # Sample from the actor
        action_distribution: torch.distributions.Distribution = self.actor(obs)

        # Sample actions
        # Note: Think about whether to use .rsample() or .sample() here...
        action: torch.Tensor = action_distribution.rsample()

        # Compute Q-values for the sampled state-action pair
        q_values = self.critic(obs, action)

        # Compute the actor loss
        loss = -q_values.mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        # Update the critic for num_critic_upates steps, and add the output stats to critic_infos
        critic_infos = [self.update_critic(observations, actions, rewards, next_observations, dones)
                        for _ in range(self.num_critic_updates)]

        # Update the actor
        actor_info = self.update_actor(observations)

        # Perform either hard or soft target updates.
        # Relevant variables:
        #  - step
        #  - self.target_update_period (None when using soft updates)
        #  - self.soft_target_update_rate (None when using hard updates)
        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif self.target_update_period is not None and step % self.target_update_period == 0:
            self.update_target_critic()

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }
