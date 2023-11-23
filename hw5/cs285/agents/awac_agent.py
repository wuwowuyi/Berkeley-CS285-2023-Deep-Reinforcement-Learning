from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn

from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # compute the actor distribution, then use it to compute E[Q(s, a)]
            # num_samples = 10  # 10 is arbitrary, to make an average.
            dist = self.actor(next_observations)
            # policy_actions = dist.sample((num_samples,))  # (num_samples, batch_size, act_dim)
            policy_actions = dist.sample()
            next_qa_values = self.target_critic(next_observations)

            # Use the actor to compute a critic backup
            # next_qs = torch.gather(torch.tile(next_qa_values, (num_samples, 1)), -1, policy_actions)
            # next_qs = next_qs.mean(dim=0).squeeze()
            next_qs = torch.gather(next_qa_values, -1, torch.unsqueeze(policy_actions.long(), 1)).squeeze()

            # Compute the TD target
            target_values = rewards + self.discount * (1 - dones.int()) * next_qs

        # Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)
        q_values = torch.gather(qa_values, -1, torch.unsqueeze(actions.long(), 1)).squeeze()
        assert q_values.shape == target_values.shape

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # compute the advantage of the actions compared to E[Q(s, a)]
        with torch.no_grad():
            qa_values = self.target_critic(observations)

        q_values = torch.gather(qa_values, -1, torch.unsqueeze(actions.long(), 1)).squeeze()

        # num_samples = 10  # arbitrary number to make an average.
        # policy_actions = action_dist.sample((num_samples,))  # shape=(num_samples, batch_size)
        # qa_values = torch.tile(qa_values, (num_samples, 1, 1))  # shape=(num_samples, batch_size)
        # values = torch.gather(qa_values, -1, torch.unsqueeze(policy_actions.long(), 1))
        # values = values.mean(dim=0).squeeze()

        policy_actions = action_dist.sample()
        values = torch.gather(qa_values, -1, torch.unsqueeze(policy_actions.long(), 1)).squeeze()

        advantages = q_values - values
        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # update the actor using AWAC
        dist = self.actor(observations)
        adv = self.compute_advantage(observations, actions, dist)
        logp = dist.log_prob(actions)
        weight = torch.exp(torch.clip(adv / self.temperature, min=-5, max=5))
        loss = -(logp * weight).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()

        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
               next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
